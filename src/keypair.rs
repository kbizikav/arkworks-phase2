use ark_ec::pairing::Pairing;

use crate::{error::Error, ratio::RatioProof, utils::serialize};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub type PrivateKey<E> = <E as Pairing>::ScalarField;

/// Public key for a contribution, including Ethereum signature for attribution.
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, PartialEq)]
pub struct PublicKey<E: Pairing> {
    pub delta_g1: E::G1Affine,
    pub delta_g2: E::G2Affine,
    pub proof: RatioProof<E>,
    /// Ethereum address of the contributor (20 bytes)
    pub eth_address: [u8; 20],
    /// ECDSA signature over the contribution context (r: 32, s: 32, v: 1)
    pub eth_signature: [u8; 65],
}

impl<E: Pairing> PublicKey<E> {
    pub fn challenge(&self) -> Result<Vec<u8>, Error> {
        Ok(serialize(&self.delta_g1)?
            .into_iter()
            .chain(serialize(&self.delta_g2)?)
            .collect())
    }
}
