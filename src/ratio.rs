use crate::{
    error::Error,
    hasher::HashToCurve,
    utils::{same_ratio, seeded_rng, serialize},
};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Copy, PartialEq)]
pub struct RatioProof<E: Pairing> {
    pub point: E::G1Affine,
    pub matching_point: E::G2Affine,
}

impl<E: Pairing> RatioProof<E> {
    pub fn get_g1(&self) -> (E::G1Affine, E::G1Affine) {
        (E::G1Affine::generator(), self.point)
    }

    pub fn generate(delta: E::ScalarField, challenge: &[u8]) -> Result<Self, Error> {
        let generator: E::G1Affine = E::G1Affine::generator();
        let point_scaled: E::G1Affine = (generator * delta).into();
        let matching_point_scaled = (HashToCurve::<E>::hash_g2(&mut seeded_rng(
            &challenge
                .iter()
                .cloned()
                .chain(serialize(&generator)?.into_iter())
                .chain(serialize(&point_scaled)?.into_iter())
                .collect::<Vec<u8>>(),
        )) * delta)
            .into();
        Ok(Self {
            matching_point: matching_point_scaled,
            point: point_scaled,
        })
    }

    pub fn verify(&self, challenge: &[u8]) -> Result<bool, Error> {
        let generator = E::G1Affine::generator();
        let challenge_point = HashToCurve::<E>::hash_g2(&mut seeded_rng(
            &challenge
                .iter()
                .cloned()
                .chain(serialize(&generator)?.into_iter())
                .chain(serialize(&self.point)?.into_iter())
                .collect::<Vec<u8>>(),
        ));
        Ok(same_ratio::<E>(
            (generator, self.matching_point),
            (self.point, challenge_point),
        ))
    }
}
