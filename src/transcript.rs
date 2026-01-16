use std::sync::{Arc, Mutex};

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{Field, UniformRand, Zero};
use ark_groth16::{ProvingKey, VerifyingKey};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_relations::gr1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError,
    R1CS_PREDICATE_LABEL,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter, end_timer, start_timer};
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

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, PartialEq)]
pub struct Transcript<E: Pairing> {
    pub key: FullKey<E>,
    pub initial_key: PartialKey<E>,
    pub contributions: Vec<PublicKey<E>>,
}

impl<E: Pairing> Transcript<E> {
    fn r1cs_constraint_count(
        cs: &ConstraintSystemRef<E::ScalarField>,
    ) -> Result<usize, Error> {
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

    fn new_from_prepared_accumulator_finalized_cs(
        accum: &PreparedAccumulator<E>,
        cs: ConstraintSystemRef<E::ScalarField>,
    ) -> Result<Self, Error> {
        let timer = start_timer!(|| "Generating transcript from prepared accumulator");

        let num_constraints = Self::r1cs_constraint_count(&cs)?;
        let num_instance_variables = cs.num_instance_variables();
        let total_constraints = num_constraints + num_instance_variables;
        let constraint_matrices = cs.to_matrices()?;
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
        let a_g1 = Arc::new(Mutex::new(vec![E::G1::zero(); num_witnesses]));
        let b_g1 = Arc::new(Mutex::new(vec![E::G1::zero(); num_witnesses]));
        let b_g2 = Arc::new(Mutex::new(vec![E::G2::zero(); num_witnesses]));
        let ext = Arc::new(Mutex::new(vec![E::G1::zero(); num_witnesses]));

        let add_dummy_constraints_timer = start_timer!(|| "Adding dummy constraints");
        a_g1.lock()?[0..num_instance_variables]
            .clone_from_slice(&accum.tau_lagrange_g1[num_constraints..total_constraints]);
        ext.lock()?[0..num_instance_variables]
            .clone_from_slice(&accum.beta_lagrange_g1[num_constraints..total_constraints]);
        end_timer!(add_dummy_constraints_timer);

        let specialize_constraints_timer =
            start_timer!(|| "Specializing constraints into phase 2 key");
        cfg_iter!(a_matrix)
            .zip(cfg_iter!(b_matrix))
            .zip(cfg_iter!(c_matrix))
            .zip(cfg_iter!(accum.tau_lagrange_g1))
            .zip(cfg_iter!(accum.tau_lagrange_g2))
            .zip(cfg_iter!(accum.alpha_lagrange_g1))
            .zip(cfg_iter!(accum.beta_lagrange_g1))
            .for_each(
                |((((((a_poly, b_poly), c_poly), tau_g1), tau_g2), alpha_tau), beta_tau)| {
                    cfg_iter!(a_poly).for_each(|(coeff, index)| {
                        a_g1.lock().unwrap()[*index] += *tau_g1 * *coeff;
                        ext.lock().unwrap()[*index] += *beta_tau * *coeff;
                    });
                    cfg_iter!(b_poly).for_each(|(coeff, index)| {
                        b_g1.lock().unwrap()[*index] += *tau_g1 * *coeff;
                        b_g2.lock().unwrap()[*index] += *tau_g2 * *coeff;
                        ext.lock().unwrap()[*index] += *alpha_tau * *coeff;
                    });
                    cfg_iter!(c_poly).for_each(|(coeff, index)| {
                        ext.lock().unwrap()[*index] += *tau_g1 * *coeff;
                    });
                },
            );
        end_timer!(specialize_constraints_timer);

        let a_query = batch_into_affine(&a_g1.lock()?);
        let b_g1_query = batch_into_affine(&b_g1.lock()?);
        let b_g2_query = batch_into_affine(&b_g2.lock()?);
        let ext = batch_into_affine(&ext.lock()?);

        let public_cross_terms = ext[..num_instance_variables].to_vec();
        let private_cross_terms = ext[num_instance_variables..].to_vec();

        for l in &private_cross_terms {
            if l.is_zero() {
                return Err(Error::InvariantViolated("unconstrained variable"));
            }
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
        let cs = ConstraintSystem::new_ref();
        circuit.generate_constraints(cs.clone())?;
        cs.finalize();

        let num_constraints = Self::r1cs_constraint_count(&cs)?;
        let num_instance_variables = cs.num_instance_variables();
        let total_constraints = num_constraints + num_instance_variables;

        let (valid, g1_len, g2_len) = accum.check_pow_len();
        valid.then_some(()).ok_or(Error::InvalidPOTSize)?;

        let domain = Radix2EvaluationDomain::<E::ScalarField>::new(total_constraints)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let size = domain.size();

        (g2_len >= total_constraints && g1_len >= size)
            .then_some(())
            .ok_or(Error::NotEnoughPOTDegree(ark_std::log2(total_constraints)))?;

        Self::new_from_prepared_accumulator_finalized_cs(&accum.prepare_with_size(size)?, cs)
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
    use ark_relations::gr1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
    use ark_relations::gr1cs::predicate::PredicateConstraintSystem;
    use ark_relations::gr1cs::predicate::polynomial_constraint::SR1CS_PREDICATE_LABEL;
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
