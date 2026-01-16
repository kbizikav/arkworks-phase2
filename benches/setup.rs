use ark_bn254::{Bn254, Fr};
use ark_crypto_primitives::sponge::{
    constraints::CryptographicSpongeVar,
    poseidon::{constraints::PoseidonSpongeVar, find_poseidon_ark_and_mds, PoseidonConfig},
};
use ark_ff::PrimeField;
use ark_r1cs_std::{alloc::AllocVar, fields::fp::FpVar};
use ark_relations::gr1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;

use arkworks_phase2::{accumulator::Accumulator, transcript::Transcript};

const NUM_HASHES: usize = 50;

fn poseidon_config<F: PrimeField>() -> PoseidonConfig<F> {
    let full_rounds = 8usize;
    let partial_rounds = 56usize;
    let alpha = 5u64;
    let rate = 2usize;
    let capacity = 1usize;

    let (ark, mds) = find_poseidon_ark_and_mds::<F>(
        F::MODULUS_BIT_SIZE as u64,
        rate,
        full_rounds as u64,
        partial_rounds as u64,
        0,
    );
    PoseidonConfig::new(full_rounds, partial_rounds, alpha, mds, ark, rate, capacity)
}

#[derive(Clone)]
struct PoseidonHashCircuit<F: PrimeField> {
    num_hashes: usize,
    config: PoseidonConfig<F>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for PoseidonHashCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let mut current = FpVar::new_witness(cs.clone(), || Ok(F::from(0u64)))?;

        for _ in 0..self.num_hashes {
            let mut sponge = PoseidonSpongeVar::new(cs.clone(), &self.config);
            sponge.absorb(&current)?;
            let output = sponge.squeeze_field_elements(1)?;
            current = output.into_iter().next().unwrap();
        }

        Ok(())
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let config = poseidon_config::<Fr>();
    let circuit = PoseidonHashCircuit {
        num_hashes: NUM_HASHES,
        config: config.clone(),
    };

    // First, create a dummy run to get the constraint count
    let cs = ark_relations::gr1cs::ConstraintSystem::<Fr>::new_ref();
    circuit.clone().generate_constraints(cs.clone()).unwrap();
    let num_constraints = cs.num_constraints();
    drop(cs);

    let mut accum = Accumulator::<Bn254>::empty_from_max_constraints(num_constraints + 1).unwrap();
    accum.contribute(&mut rng);

    let bench_label = format!("setup_{}_poseidon_hashes", NUM_HASHES);
    let start = Instant::now();
    let _ = Transcript::<Bn254>::new_from_accumulator(&accum, circuit).unwrap();
    let elapsed = start.elapsed();

    println!(
        "{}: {:?} (num_constraints={})",
        bench_label, elapsed, num_constraints
    );
}
