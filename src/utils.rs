#![allow(clippy::double_must_use)]

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{UniformRand, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::cfg_iter;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::error::Error;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[inline]
#[must_use]
pub fn batch_into_projective<A: AffineRepr>(points: &[A]) -> Vec<A::Group> {
    cfg_iter!(points)
        .map(|point| (*point).into_group())
        .collect()
}

#[inline]
#[must_use]
pub fn batch_into_affine<P: CurveGroup>(points: &[P]) -> Vec<P::Affine> {
    P::normalize_batch(points)
}

#[inline]
pub fn batch_mul_fixed_scalar<A: AffineRepr>(points: &mut [A], scalar: A::ScalarField) {
    // Perform scalar multiplication in projective coordinates
    let projective: Vec<A::Group> = cfg_iter!(points)
        .map(|point| *point * scalar)
        .collect();

    // Batch normalize to affine (more efficient than individual into_affine calls)
    let affine = A::Group::normalize_batch(&projective);

    // Copy results back
    points.copy_from_slice(&affine);
}

#[inline]
#[must_use]
pub fn same_ratio<E: Pairing>(
    lhs: (E::G1Affine, E::G2Affine),
    rhs: (E::G1Affine, E::G2Affine),
) -> bool {
    E::pairing(lhs.0, lhs.1) == E::pairing(rhs.0, rhs.1)
}

#[inline]
#[must_use]
pub fn same_ratio_swap<E: Pairing>(
    lhs: (E::G1Affine, E::G1Affine),
    rhs: (E::G2Affine, E::G2Affine),
) -> bool {
    E::pairing(lhs.0, rhs.1) == E::pairing(lhs.1, rhs.0)
}

#[must_use]
pub fn seeded_rng(bytes: &[u8]) -> ChaCha20Rng {
    let seed = blake3::hash(bytes);
    ChaCha20Rng::from_seed(*seed.as_bytes())
}

#[must_use]
pub fn serialize<T: CanonicalSerialize>(value: &T) -> Result<Vec<u8>, Error> {
    let mut output = vec![];
    value
        .serialize_compressed(&mut output)
        .map_err(|e| Error::Custom(e.to_string()))?;
    Ok(output)
}

#[must_use]
pub fn serialize_uncompressed<T: CanonicalSerialize>(value: &T) -> Result<Vec<u8>, Error> {
    let mut output = vec![];
    value
        .serialize_uncompressed(&mut output)
        .map_err(|e| Error::Custom(e.to_string()))?;
    Ok(output)
}

/// Generate a seed for random linear combination from the input vectors.
/// This creates a deterministic but unpredictable seed based on the input data.
#[inline]
fn merge_seed<A: AffineRepr>(lhs: &[A], rhs: &[A]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();

    // Hash a sample of elements instead of all elements for speed
    // We sample elements at different positions to ensure coverage
    let len = lhs.len();
    let sample_indices = if len <= 16 {
        (0..len).collect::<Vec<_>>()
    } else {
        // Sample 16 elements spread across the vectors
        (0..16).map(|i| i * len / 16).collect::<Vec<_>>()
    };

    for &idx in &sample_indices {
        let mut buf = Vec::new();
        lhs[idx].serialize_compressed(&mut buf).ok();
        hasher.update(&buf);
        buf.clear();
        rhs[idx].serialize_compressed(&mut buf).ok();
        hasher.update(&buf);
    }

    // Also include the length to differentiate same-prefix vectors
    hasher.update(&(len as u64).to_le_bytes());

    *hasher.finalize().as_bytes()
}

#[must_use]
#[inline]
pub fn merge_ratio_affine_vec<A: AffineRepr>(lhs: &[A], rhs: &[A]) -> (A, A) {
    assert_eq!(lhs.len(), rhs.len(), "lhs.len() != rhs.len()");

    // Generate seed from input data for reproducible but unpredictable randomness
    let seed = merge_seed(lhs, rhs);

    #[cfg(not(feature = "parallel"))]
    let result = {
        let mut rng = ChaCha20Rng::from_seed(seed);
        let (mut l, mut r) = (A::Group::zero(), A::Group::zero());
        lhs.iter()
            .zip(rhs)
            .for_each(|(l1, r1)| {
                let s = A::ScalarField::rand(&mut rng);
                l += *l1 * s;
                r += *r1 * s;
            });
        (l, r)
    };

    #[cfg(feature = "parallel")]
    let result = {
        // Use thread-local RNGs seeded from the main seed + thread index
        // This maintains determinism while avoiding OsRng overhead
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = AtomicU64::new(0);

        lhs.into_par_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| {
                // Create thread-local RNG with unique seed
                let idx = counter.fetch_add(1, Ordering::Relaxed);
                let mut thread_seed = seed;
                // Mix in the counter to get unique seed per element
                let idx_bytes = idx.to_le_bytes();
                for i in 0..8 {
                    thread_seed[i] ^= idx_bytes[i];
                }
                let mut rng = ChaCha20Rng::from_seed(thread_seed);
                let s = A::ScalarField::rand(&mut rng);
                (*lhs * s, *rhs * s)
            })
            .reduce(
                || (A::Group::zero(), A::Group::zero()),
                |(l, r), (l1, r1)| (l + l1, r + r1),
            )
    };

    (result.0.into_affine(), result.1.into_affine())
}
