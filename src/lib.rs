pub mod accumulator;
pub mod error;
pub mod hasher;
pub mod key;
pub mod keypair;
pub mod ratio;
pub mod reader;
pub mod transcript;
pub mod utils;

#[cfg(test)]
mod tests {
    use std::error::Error;

    use ark_bn254::{Bn254, Fr};
    use ark_ff::Zero;
    use ark_groth16::Groth16;
    use ark_r1cs_std::{
        fields::fp::FpVar,
        prelude::{AllocVar, EqGadget},
    };
    use ark_relations::gr1cs::ConstraintSynthesizer;
    use ark_serialize::CanonicalDeserialize;
    use ark_snark::SNARK;
    use rand::rngs::OsRng;

    use crate::{
        accumulator::Accumulator,
        transcript::{ContributionContext, Transcript},
        utils::serialize_uncompressed,
    };

    const NUM_CONSTRAINTS: usize = 50;

    struct DummyCircuit {
        pub a: Fr,
        pub b: Fr,
        pub c: Fr,
    }

    impl ConstraintSynthesizer<Fr> for DummyCircuit {
        fn generate_constraints(
            self,
            cs: ark_relations::gr1cs::ConstraintSystemRef<Fr>,
        ) -> ark_relations::gr1cs::Result<()> {
            let a = FpVar::new_witness(cs.clone(), || Ok(self.a))?;
            let b = FpVar::new_witness(cs.clone(), || Ok(self.b))?;
            let c = FpVar::new_input(cs, || Ok(self.c))?;
            let d = &a + &b;

            for _ in 0..(NUM_CONSTRAINTS - 5) {
                c.enforce_equal(&d)?;
            }

            Ok(())
        }
    }

    // Test helper: create a mock signature (all zeros is not a valid signature, but we can use it for basic tests)
    fn mock_sign(_message: &[u8; 32]) -> Result<[u8; 65], crate::error::Error> {
        // This is a mock signature - not valid for real verification
        // In production, this would use a real Ethereum signing key
        Ok([0u8; 65])
    }

    // Test helper: create a test address
    fn test_address() -> [u8; 20] {
        [
            0xde, 0xad, 0xbe, 0xef, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        ]
    }

    #[test]
    fn groth16_domain() -> Result<(), Box<dyn Error>> {
        let rng = &mut OsRng;
        let mut accum = Accumulator::<Bn254>::empty_from_max_constraints(NUM_CONSTRAINTS)?;

        accum.contribute(rng);

        let mut transcript = Transcript::new_from_accumulator(
            &accum,
            DummyCircuit {
                a: Fr::zero(),
                b: Fr::zero(),
                c: Fr::zero(),
            },
        )?;

        let context1 = ContributionContext::new("test-ceremony", 1, "test-circuit");
        transcript.contribute_seed(b"ofekwoo", test_address(), mock_sign, &context1)?;
        transcript.verify()?;

        let context2 = ContributionContext::new("test-ceremony", 2, "test-circuit");
        transcript.contribute_seed(b"aewrog", test_address(), mock_sign, &context2)?;
        transcript.verify()?;

        let pk = transcript.key.key;
        let proof = Groth16::<Bn254>::prove(
            &pk,
            DummyCircuit {
                a: Fr::from(1),
                b: Fr::from(2),
                c: Fr::from(3),
            },
            rng,
        )?;

        let valid = Groth16::<Bn254>::verify(&pk.vk, &[Fr::from(3)], &proof)?;
        assert!(valid, "Proof must be valid");

        let valid = Groth16::<Bn254>::verify(&pk.vk, &[Fr::from(4)], &proof)?;
        assert!(!valid, "Proof must be not valid");

        Ok(())
    }

    #[test]
    fn correct_serialization() -> Result<(), Box<dyn Error>> {
        let rng = &mut OsRng;
        let mut accum = Accumulator::<Bn254>::empty_from_max_constraints(NUM_CONSTRAINTS)?;

        accum.contribute(rng);

        let transcript = Transcript::new_from_accumulator(
            &accum,
            DummyCircuit {
                a: Fr::zero(),
                b: Fr::zero(),
                c: Fr::zero(),
            },
        )?;

        let bytes = serialize_uncompressed(&transcript)?;
        let transcript_deserialized = Transcript::<Bn254>::deserialize_uncompressed(&bytes[..])?;

        assert_eq!(transcript, transcript_deserialized);

        Ok(())
    }
}
