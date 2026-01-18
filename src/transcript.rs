use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{Field, UniformRand};
use ark_groth16::{ProvingKey, VerifyingKey};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_relations::gr1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError,
    R1CS_PREDICATE_LABEL,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_into_iter, end_timer, start_timer};
use rand::Rng;

use crate::{
    accumulator::{Accumulator, PreparedAccumulator},
    error::Error,
    key::{FullKey, PartialKey},
    keypair::PublicKey,
    ratio::RatioProof,
    utils::{
        batch_into_affine, batch_mul_fixed_scalar, merge_ratio_affine_vec, same_ratio_swap,
        seeded_rng,
    },
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type SparseRow<F> = Vec<(F, usize)>;
type SparseMatrix<F> = Vec<SparseRow<F>>;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, PartialEq)]
pub struct Transcript<E: Pairing> {
    pub key: FullKey<E>,
    pub initial_key: PartialKey<E>,
    pub contributions: Vec<PublicKey<E>>,
}

impl<E: Pairing> Transcript<E> {
    fn r1cs_constraint_count(cs: &ConstraintSystemRef<E::ScalarField>) -> Result<usize, Error> {
        let r1cs_constraints = cs
            .get_predicates_num_constraints(R1CS_PREDICATE_LABEL)
            .ok_or_else(|| Error::Custom("missing R1CS predicate".to_string()))?;

        for (label, count) in cs.get_all_predicates_num_constraints() {
            if label != R1CS_PREDICATE_LABEL {
                return Err(Error::Custom(format!(
                    "non-R1CS predicate '{}' ({} constraints) is not supported",
                    label, count
                )));
            }
        }

        Ok(r1cs_constraints)
    }

    /// Transpose a sparse matrix from constraint-indexed to witness-indexed representation.
    /// Returns a vector where each element contains (constraint_index, coefficient) pairs
    /// for a given witness variable.
    fn transpose_matrix<F: Field>(
        matrix: &SparseMatrix<F>,
        num_witnesses: usize,
    ) -> Vec<Vec<(usize, F)>> {
        let mut transposed = vec![Vec::new(); num_witnesses];
        for (constraint_idx, row) in matrix.iter().enumerate() {
            for (coeff, witness_idx) in row {
                transposed[*witness_idx].push((constraint_idx, *coeff));
            }
        }
        transposed
    }

    fn new_from_prepared_accumulator_finalized_cs(
        accum: &PreparedAccumulator<E>,
        cs: ConstraintSystemRef<E::ScalarField>,
    ) -> Result<Self, Error> {
        use std::time::Instant;

        let timer = start_timer!(|| "Generating transcript from prepared accumulator");
        eprintln!("[new_from_prepared_accumulator_finalized_cs] Starting...");

        let num_constraints = Self::r1cs_constraint_count(&cs)?;
        let num_instance_variables = cs.num_instance_variables();
        let total_constraints = num_constraints + num_instance_variables;

        let t_matrices = Instant::now();
        let constraint_matrices = cs.to_matrices()?;
        eprintln!("[new_from_prepared_accumulator_finalized_cs] cs.to_matrices: {:?}", t_matrices.elapsed());

        let r1cs_matrices = constraint_matrices
            .get(R1CS_PREDICATE_LABEL)
            .ok_or(Error::MissingCSMatrices)?;
        let (a_matrix, b_matrix, c_matrix) = match r1cs_matrices.as_slice() {
            [a, b, c] => (a, b, c),
            _ => return Err(Error::InvariantViolated("unexpected R1CS matrix arity")),
        };

        let (valid, len) = accum.check_pow_len();
        valid.then_some(()).ok_or(Error::InvalidPOTSize)?;

        let domain = Radix2EvaluationDomain::<E::ScalarField>::new(total_constraints)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let size = domain.size();

        (len == size)
            .then_some(())
            .ok_or(Error::InvalidPOTDegree(ark_std::log2(size)))?;

        let num_instance_variables = cs.num_instance_variables();
        let num_witness_variables = cs.num_witness_variables();
        let num_witnesses = num_instance_variables + num_witness_variables;
        eprintln!("[new_from_prepared_accumulator_finalized_cs] num_witnesses={}", num_witnesses);

        // Transpose matrices: from constraint-indexed to witness-indexed
        let transpose_timer = start_timer!(|| "Transposing constraint matrices");
        let t_transpose = Instant::now();
        let a_by_witness = Self::transpose_matrix(a_matrix, num_witnesses);
        let b_by_witness = Self::transpose_matrix(b_matrix, num_witnesses);
        let c_by_witness = Self::transpose_matrix(c_matrix, num_witnesses);
        end_timer!(transpose_timer);
        eprintln!("[new_from_prepared_accumulator_finalized_cs] transpose_matrix: {:?}", t_transpose.elapsed());

        // Convert projective to affine for MSM
        let convert_timer = start_timer!(|| "Converting to affine for MSM");
        let t_convert = Instant::now();
        let tau_lagrange_g1_affine = batch_into_affine(&accum.tau_lagrange_g1);
        let tau_lagrange_g2_affine = batch_into_affine(&accum.tau_lagrange_g2);
        let alpha_lagrange_g1_affine = batch_into_affine(&accum.alpha_lagrange_g1);
        let beta_lagrange_g1_affine = batch_into_affine(&accum.beta_lagrange_g1);
        end_timer!(convert_timer);
        eprintln!("[new_from_prepared_accumulator_finalized_cs] batch_into_affine: {:?}", t_convert.elapsed());

        let specialize_constraints_timer =
            start_timer!(|| "Specializing constraints into phase 2 key (MSM)");

        // Process each witness variable using batched scalar multiplication
        // a_g1[i] = sum over constraints j: tau_g1[j] * a_matrix[j][i]
        // b_g1[i] = sum over constraints j: tau_g1[j] * b_matrix[j][i]
        // b_g2[i] = sum over constraints j: tau_g2[j] * b_matrix[j][i]
        // ext[i]  = sum over constraints j: (beta_tau[j] * a_matrix[j][i]
        //                                  + alpha_tau[j] * b_matrix[j][i]
        //                                  + tau_g1[j] * c_matrix[j][i])

        // Batched computation for G1 points: parallel over witnesses, using scalar mul + sum
        // This avoids nested thread pools that cause resource exhaustion while maintaining
        // good parallelism across witnesses
        fn batch_compute_g1<E: Pairing>(
            entries_by_witness: &[Vec<(usize, E::ScalarField)>],
            bases: &[E::G1Affine],
        ) -> Vec<E::G1Affine> {
            cfg_into_iter!(entries_by_witness)
                .map(|entries| {
                    if entries.is_empty() {
                        return E::G1Affine::zero();
                    }
                    // Sequential scalar mul and sum within each witness
                    // Parallelism is at the witness level, not nested
                    entries
                        .iter()
                        .map(|&(idx, scalar)| bases[idx] * scalar)
                        .sum::<E::G1>()
                        .into_affine()
                })
                .collect()
        }

        fn batch_compute_g2<E: Pairing>(
            entries_by_witness: &[Vec<(usize, E::ScalarField)>],
            bases: &[E::G2Affine],
        ) -> Vec<E::G2Affine> {
            cfg_into_iter!(entries_by_witness)
                .map(|entries| {
                    if entries.is_empty() {
                        return E::G2Affine::zero();
                    }
                    entries
                        .iter()
                        .map(|&(idx, scalar)| bases[idx] * scalar)
                        .sum::<E::G2>()
                        .into_affine()
                })
                .collect()
        }

        // Compute all query types in sequence (each internally parallel over witnesses)
        let a_g1_timer = start_timer!(|| "Computing a_g1 query");
        let t_a_g1 = Instant::now();
        let a_query_results = batch_compute_g1::<E>(&a_by_witness, &tau_lagrange_g1_affine);
        end_timer!(a_g1_timer);
        eprintln!("[new_from_prepared_accumulator_finalized_cs] a_g1 query: {:?}", t_a_g1.elapsed());

        let b_g1_timer = start_timer!(|| "Computing b_g1 query");
        let t_b_g1 = Instant::now();
        let b_g1_query_results = batch_compute_g1::<E>(&b_by_witness, &tau_lagrange_g1_affine);
        end_timer!(b_g1_timer);
        eprintln!("[new_from_prepared_accumulator_finalized_cs] b_g1 query: {:?}", t_b_g1.elapsed());

        let b_g2_timer = start_timer!(|| "Computing b_g2 query");
        let t_b_g2 = Instant::now();
        let b_g2_query_results = batch_compute_g2::<E>(&b_by_witness, &tau_lagrange_g2_affine);
        end_timer!(b_g2_timer);
        eprintln!("[new_from_prepared_accumulator_finalized_cs] b_g2 query: {:?}", t_b_g2.elapsed());

        let ext_timer = start_timer!(|| "Computing ext query");
        let t_ext = Instant::now();
        let ext_a = batch_compute_g1::<E>(&a_by_witness, &beta_lagrange_g1_affine);
        let ext_b = batch_compute_g1::<E>(&b_by_witness, &alpha_lagrange_g1_affine);
        let ext_c = batch_compute_g1::<E>(&c_by_witness, &tau_lagrange_g1_affine);
        end_timer!(ext_timer);
        eprintln!("[new_from_prepared_accumulator_finalized_cs] ext query: {:?}", t_ext.elapsed());

        // Combine ext results and build final result tuples
        let results: Vec<_> = (0..num_witnesses)
            .map(|i| {
                let ext_i = (ext_a[i].into_group() + ext_b[i] + ext_c[i]).into_affine();
                (a_query_results[i], b_g1_query_results[i], b_g2_query_results[i], ext_i)
            })
            .collect();

        // Unzip results
        let mut a_query: Vec<E::G1Affine> = Vec::with_capacity(num_witnesses);
        let mut b_g1_query: Vec<E::G1Affine> = Vec::with_capacity(num_witnesses);
        let mut b_g2_query: Vec<E::G2Affine> = Vec::with_capacity(num_witnesses);
        let mut ext: Vec<E::G1Affine> = Vec::with_capacity(num_witnesses);

        for (a, b1, b2, e) in results {
            a_query.push(a);
            b_g1_query.push(b1);
            b_g2_query.push(b2);
            ext.push(e);
        }

        // Add dummy constraints for instance variables
        let add_dummy_constraints_timer = start_timer!(|| "Adding dummy constraints");
        for i in 0..num_instance_variables {
            a_query[i] = (a_query[i] + tau_lagrange_g1_affine[num_constraints + i]).into_affine();
            ext[i] = (ext[i] + beta_lagrange_g1_affine[num_constraints + i]).into_affine();
        }
        end_timer!(add_dummy_constraints_timer);

        end_timer!(specialize_constraints_timer);

        let public_cross_terms = ext[..num_instance_variables].to_vec();
        let private_cross_terms = ext[num_instance_variables..].to_vec();

        // Check for unconstrained variables (warning only for compatibility with sonobe's DeciderEthCircuit)
        let unconstrained_count = private_cross_terms.iter().filter(|l| l.is_zero()).count();
        if unconstrained_count > 0 {
            eprintln!(
                "[WARNING] Found {} unconstrained variables (out of {} private witnesses). \
                This may affect Groth16 security guarantees.",
                unconstrained_count,
                private_cross_terms.len()
            );
        }

        let key = ProvingKey::<E> {
            vk: VerifyingKey {
                alpha_g1: accum.alpha,
                beta_g2: accum.beta_g2,
                gamma_g2: E::G2Affine::generator(),
                delta_g2: E::G2Affine::generator(),
                gamma_abc_g1: public_cross_terms,
            },
            beta_g1: accum.beta,
            delta_g1: E::G1Affine::generator(),
            a_query,
            b_g1_query,
            b_g2_query,
            h_query: accum.h_query.to_vec(),
            l_query: private_cross_terms,
        };

        end_timer!(timer);

        Ok(Self {
            initial_key: (&key).into(),
            key: FullKey { key },
            contributions: vec![],
        })
    }

    pub fn new_from_prepared_accumulator<C: ConstraintSynthesizer<E::ScalarField>>(
        accum: &PreparedAccumulator<E>,
        circuit: C,
    ) -> Result<Self, Error> {
        let cs = ConstraintSystem::new_ref();
        circuit.generate_constraints(cs.clone())?;
        cs.finalize();

        let num_constraints = Self::r1cs_constraint_count(&cs)?;
        let num_instance_variables = cs.num_instance_variables();
        let total_constraints = num_constraints + num_instance_variables;

        let (valid, len) = accum.check_pow_len();
        valid.then_some(()).ok_or(Error::InvalidPOTSize)?;

        let needed_degree = ark_std::log2(total_constraints);
        let degree = ark_std::log2(len);

        (degree == needed_degree)
            .then_some(())
            .ok_or(Error::InvalidPOTDegree(needed_degree))?;

        Self::new_from_prepared_accumulator_finalized_cs(accum, cs)
    }

    pub fn new_from_accumulator<C: ConstraintSynthesizer<E::ScalarField>>(
        accum: &Accumulator<E>,
        circuit: C,
    ) -> Result<Self, Error> {
        use std::time::Instant;

        eprintln!("[new_from_accumulator] Starting...");

        let t0 = Instant::now();
        let cs = ConstraintSystem::new_ref();
        circuit.generate_constraints(cs.clone())?;
        cs.finalize();
        eprintln!("[new_from_accumulator] ConstraintSystem generate_constraints: {:?}", t0.elapsed());

        let num_constraints = Self::r1cs_constraint_count(&cs)?;
        let num_instance_variables = cs.num_instance_variables();
        let total_constraints = num_constraints + num_instance_variables;
        eprintln!("[new_from_accumulator] num_constraints={}, num_instance_variables={}, total_constraints={}",
                  num_constraints, num_instance_variables, total_constraints);

        let (valid, g1_len, g2_len) = accum.check_pow_len();
        valid.then_some(()).ok_or(Error::InvalidPOTSize)?;

        let domain = Radix2EvaluationDomain::<E::ScalarField>::new(total_constraints)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let size = domain.size();
        eprintln!("[new_from_accumulator] domain size={}", size);

        (g2_len >= total_constraints && g1_len >= size)
            .then_some(())
            .ok_or(Error::NotEnoughPOTDegree(ark_std::log2(total_constraints)))?;

        let t1 = Instant::now();
        let prepared = accum.prepare_with_size(size)?;
        eprintln!("[new_from_accumulator] prepare_with_size: {:?}", t1.elapsed());

        let t2 = Instant::now();
        let result = Self::new_from_prepared_accumulator_finalized_cs(&prepared, cs);
        eprintln!("[new_from_accumulator] new_from_prepared_accumulator_finalized_cs: {:?}", t2.elapsed());

        result
    }

    pub fn contribute_seed(&mut self, seed: &[u8]) -> Result<(), Error> {
        self.contribute_rng(&mut seeded_rng(seed))
    }

    pub fn contribute_rng<R: Rng>(&mut self, rng: &mut R) -> Result<(), Error> {
        let timer = start_timer!(|| "Contributing to transcript");

        let delta = E::ScalarField::rand(rng);
        let delta_inverse = delta.inverse().expect("delta is not invertible");

        let proof = RatioProof::<E>::generate(delta, &self.key.challenge()?)?;

        let l_timer = start_timer!(|| "Updating l_query");
        batch_mul_fixed_scalar(&mut self.key.key.l_query, delta_inverse);
        end_timer!(l_timer);

        let h_timer = start_timer!(|| "Updating h_query");
        batch_mul_fixed_scalar(&mut self.key.key.h_query, delta_inverse);
        end_timer!(h_timer);

        self.key.key.delta_g1 = (self.key.key.delta_g1 * delta).into_affine();
        self.key.key.vk.delta_g2 = (self.key.key.vk.delta_g2 * delta).into_affine();
        self.contributions.push(PublicKey {
            delta_g2: self.key.key.vk.delta_g2,
            proof,
        });

        end_timer!(timer);

        Ok(())
    }

    #[inline]
    pub fn verify(&self) -> Result<(), Error> {
        let mut challenge: (E::G2Affine, Vec<u8>) =
            (self.initial_key.delta_g2, self.initial_key.challenge()?);
        for contribution in self.contributions.iter() {
            contribution
                .proof
                .verify(&challenge.1)
                .map_err(|_| Error::InvalidRatioProof)?;

            same_ratio_swap::<E>(
                contribution.proof.get_g1(),
                (challenge.0, contribution.delta_g2),
            )
            .then_some(())
            .ok_or(Error::InvalidRatioProof)?;
            challenge = (contribution.delta_g2, contribution.challenge()?);
        }

        same_ratio_swap::<E>(
            (self.initial_key.delta_g1, self.key.key.delta_g1),
            (self.initial_key.delta_g2, challenge.0),
        )
        .then_some(())
        .ok_or(Error::InvalidRatioProof)?;

        same_ratio_swap::<E>(
            merge_ratio_affine_vec(&self.key.key.h_query, &self.initial_key.h_query),
            (self.initial_key.delta_g2, self.key.key.vk.delta_g2),
        )
        .then_some(())
        .ok_or(Error::InconsistentHChange)?;

        same_ratio_swap::<E>(
            merge_ratio_affine_vec(&self.key.key.l_query, &self.initial_key.l_query),
            (self.initial_key.delta_g2, self.key.key.vk.delta_g2),
        )
        .then_some(())
        .ok_or(Error::InconsistentLChange)?;
        Ok(())
    }

    #[inline]
    pub fn verify_key_transform(
        prev: &PartialKey<E>,
        next: &PartialKey<E>,
        proof: &RatioProof<E>,
    ) -> Result<(), Error> {
        proof
            .verify(&prev.challenge()?)
            .map_err(|_| Error::InvalidRatioProof)?;

        (same_ratio_swap::<E>(proof.get_g1(), (prev.delta_g2, next.delta_g2))
            && same_ratio_swap::<E>(
                (prev.delta_g1, next.delta_g1),
                (prev.delta_g2, next.delta_g2),
            ))
        .then_some(())
        .ok_or(Error::InvalidRatioProof)?;

        same_ratio_swap::<E>(
            merge_ratio_affine_vec(&next.h_query, &prev.h_query),
            (prev.delta_g2, next.delta_g2),
        )
        .then_some(())
        .ok_or(Error::InconsistentHChange)?;

        same_ratio_swap::<E>(
            merge_ratio_affine_vec(&next.l_query, &prev.l_query),
            (prev.delta_g2, next.delta_g2),
        )
        .then_some(())
        .ok_or(Error::InconsistentLChange)?;

        Ok(())
    }

    #[inline]
    pub fn verify_from_accumulator<C: ConstraintSynthesizer<E::ScalarField>>(
        &self,
        accum: &Accumulator<E>,
        circuit: C,
    ) -> Result<(), Error> {
        let initial_transcript = Transcript::new_from_accumulator(accum, circuit)?;
        self.verify_from_initial_transcript(&initial_transcript)
    }

    /// Verify transcript against a pre-computed initial transcript.
    /// This is much faster than verify_from_accumulator as it skips the expensive
    /// IFFT and MSM computations needed to regenerate the initial transcript.
    #[inline]
    pub fn verify_from_initial_transcript(
        &self,
        initial_transcript: &Transcript<E>,
    ) -> Result<(), Error> {
        (initial_transcript.initial_key == self.initial_key)
            .then_some(())
            .ok_or(Error::InvalidKey("initial_key"))?;
        (initial_transcript.key.key.beta_g1 == self.key.key.beta_g1)
            .then_some(())
            .ok_or(Error::InvalidKey("pk: beta_g1"))?;
        (initial_transcript.key.key.a_query == self.key.key.a_query)
            .then_some(())
            .ok_or(Error::InvalidKey("pk: a_query"))?;
        (initial_transcript.key.key.b_g1_query == self.key.key.b_g1_query)
            .then_some(())
            .ok_or(Error::InvalidKey("pk: b_g1"))?;
        (initial_transcript.key.key.b_g2_query == self.key.key.b_g2_query)
            .then_some(())
            .ok_or(Error::InvalidKey("pk: b_g2"))?;
        (initial_transcript.key.key.vk.alpha_g1 == self.key.key.vk.alpha_g1)
            .then_some(())
            .ok_or(Error::InvalidKey("vk: alpha_g1"))?;
        (initial_transcript.key.key.vk.beta_g2 == self.key.key.vk.beta_g2)
            .then_some(())
            .ok_or(Error::InvalidKey("vk: beta_g2"))?;
        (initial_transcript.key.key.vk.gamma_g2 == self.key.key.vk.gamma_g2)
            .then_some(())
            .ok_or(Error::InvalidKey("vk: gamma_g2"))?;
        (initial_transcript.key.key.vk.gamma_abc_g1 == self.key.key.vk.gamma_abc_g1)
            .then_some(())
            .ok_or(Error::InvalidKey("vk: gamma_abc_g1"))?;

        self.verify()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Transcript;
    use crate::{accumulator::Accumulator, error::Error};
    use ark_bn254::{Bn254, Fr};
    use ark_relations::gr1cs::predicate::polynomial_constraint::SR1CS_PREDICATE_LABEL;
    use ark_relations::gr1cs::predicate::PredicateConstraintSystem;
    use ark_relations::gr1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
    use ark_relations::lc;

    struct Sr1csCircuit;

    impl ConstraintSynthesizer<Fr> for Sr1csCircuit {
        fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
            cs.register_predicate(
                SR1CS_PREDICATE_LABEL,
                PredicateConstraintSystem::new_sr1cs_predicate()?,
            )?;

            let x = cs.new_witness_variable(|| Ok(Fr::from(3u64)))?;
            let y = cs.new_witness_variable(|| Ok(Fr::from(9u64)))?;
            cs.enforce_sr1cs_constraint(|| lc![x], || lc![y])?;
            Ok(())
        }
    }

    #[test]
    fn rejects_non_r1cs_predicate() {
        let accum = Accumulator::<Bn254>::empty_from_degree(1).expect("accumulator");
        let prepared = accum.prepare().expect("prepare");

        let err = Transcript::new_from_prepared_accumulator(&prepared, Sr1csCircuit)
            .expect_err("expected non-R1CS predicate to be rejected");

        match err {
            Error::Custom(message) => assert!(message.contains("non-R1CS predicate")),
            _ => panic!("unexpected error: {err:?}"),
        }
    }
}
