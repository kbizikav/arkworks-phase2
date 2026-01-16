use ark_bn254::{Bn254, Fr};
use ark_relations::gr1cs::{
    ConstraintSynthesizer, ConstraintSystemRef, LinearCombination, SynthesisError, Variable,
    R1CS_PREDICATE_LABEL,
};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;

use arkworks_phase2::{accumulator::Accumulator, transcript::Transcript};

const LOG_TOTAL_CONSTRAINTS: usize = 17;
const TARGET_TOTAL_CONSTRAINTS: usize = 1 << LOG_TOTAL_CONSTRAINTS;
const NUM_CONSTRAINTS: usize = TARGET_TOTAL_CONSTRAINTS - 1;

#[derive(Clone)]
struct LinearCircuit {
    num_constraints: usize,
}

impl ConstraintSynthesizer<Fr> for LinearCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        let a = cs.new_witness_variable(|| Ok(Fr::from(3u64)))?;
        let a_lc = LinearCombination::from(a);
        let one_lc = LinearCombination::from(Variable::One);

        for _ in 0..self.num_constraints {
            cs.enforce_constraint_arity_3(
                R1CS_PREDICATE_LABEL,
                || a_lc.clone(),
                || one_lc.clone(),
                || a_lc.clone(),
            )?;
        }

        Ok(())
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut accum =
        Accumulator::<Bn254>::empty_from_max_constraints(TARGET_TOTAL_CONSTRAINTS).unwrap();
    accum.contribute(&mut rng);

    let bench_label = format!("setup_2^{}_constraints", LOG_TOTAL_CONSTRAINTS);
    let circuit = LinearCircuit {
        num_constraints: NUM_CONSTRAINTS,
    };
    let start = Instant::now();
    let _ = Transcript::<Bn254>::new_from_accumulator(&accum, circuit).unwrap();
    let elapsed = start.elapsed();

    println!(
        "{}: {:?} (total_constraints={})",
        bench_label, elapsed, TARGET_TOTAL_CONSTRAINTS
    );
}
