use std::marker::PhantomData;

use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use rand::Rng;

pub struct HashToCurve<E: Pairing>(PhantomData<E>);

impl<E: Pairing> HashToCurve<E> {
    pub fn hash_g1<R: Rng>(rng: &mut R) -> E::G1Affine {
        <E::G1 as UniformRand>::rand(rng).into()
    }
    pub fn hash_g2<R: Rng>(rng: &mut R) -> E::G2Affine {
        <E::G2 as UniformRand>::rand(rng).into()
    }
}
