use ark_ec::pairing::Pairing;

use crate::{error::Error, ratio::RatioProof, utils::serialize};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub type PrivateKey<E> = <E as Pairing>::ScalarField;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Copy, PartialEq)]
pub struct PublicKey<E: Pairing> {
    pub delta_g2: E::G2Affine,
    pub proof: RatioProof<E>,
}

impl<E: Pairing> PublicKey<E> {
    pub fn challenge(&self) -> Result<Vec<u8>, Error> {
        serialize(&self.delta_g2)
    }
}
