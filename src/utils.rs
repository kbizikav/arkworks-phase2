#![allow(clippy::double_must_use)]

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{UniformRand, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::{cfg_iter, cfg_iter_mut};
use rand::rngs::OsRng;
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
    cfg_iter_mut!(points).for_each(|point| *point = (*point * scalar).into_affine())
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

#[must_use]
#[inline]
pub fn merge_ratio_affine_vec<A: AffineRepr>(lhs: &[A], rhs: &[A]) -> (A, A) {
    assert_eq!(lhs.len(), rhs.len(), "lhs.len() != rhs.len()");
    #[cfg(not(feature = "parallel"))]
    let result = {
        let (mut l, mut r) = (A::Group::zero(), A::Group::zero());
        (0..lhs.len())
            .map(|_| A::ScalarField::rand(&mut OsRng))
            .zip(lhs)
            .zip(rhs)
            .for_each(|((s, l1), r1)| {
                l += *l1 * s;
                r += *r1 * s;
            });
        (l, r)
    };

    #[cfg(feature = "parallel")]
    let result = {
        lhs.into_par_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| {
                let s = A::ScalarField::rand(&mut OsRng);
                (*lhs * s, *rhs * s)
            })
            .reduce(
                || (A::Group::zero(), A::Group::zero()),
                |(l, r), (l1, r1)| (l + l1, r + r1),
            )
    };

    (result.0.into_affine(), result.1.into_affine())
}
