use std::{io::Read, ops::Div};

use ark_ec::{
    bls12::{Bls12, Bls12Config},
    bn::{Bn, BnConfig},
};
use ark_ec::pairing::Pairing;
use ark_ff::{BigInteger, PrimeField, QuadExtField};

use crate::error::Error;

fn read_bigint_le<F: PrimeField>(bytes: &[u8]) -> Result<F::BigInt, Error> {
    let mut repr = F::BigInt::default();
    let limbs = repr.as_mut();
    if bytes.len() != limbs.len() * 8 {
        return Err(Error::Read(format!(
            "Unexpected bigint size: {}",
            bytes.len()
        )));
    }
    for (i, limb) in limbs.iter_mut().enumerate() {
        let start = i * 8;
        let mut chunk = [0u8; 8];
        chunk.copy_from_slice(&bytes[start..start + 8]);
        *limb = u64::from_le_bytes(chunk);
    }
    Ok(repr)
}

fn montgomery_r<F: PrimeField>() -> F {
    let exp = 64u64 * <F::BigInt as BigInteger>::NUM_LIMBS as u64;
    F::from(2u64).pow([exp])
}

pub trait PairingReader: Pairing {
    fn read_ptau_g1<R: Read>(reader: &mut R) -> Result<Self::G1Affine, Error>;
    fn read_ptau_g2<R: Read>(reader: &mut R) -> Result<Self::G2Affine, Error>;

    fn read_radix_g1<R: Read>(reader: &mut R) -> Result<Self::G1Affine, Error>;
    fn read_radix_g2<R: Read>(reader: &mut R) -> Result<Self::G2Affine, Error>;
}

impl<T: BnConfig> PairingReader for Bn<T> {
    fn read_ptau_g1<R: Read>(reader: &mut R) -> Result<Self::G1Affine, Error> {
        let r = montgomery_r::<Self::BaseField>();
        let len = <Self::BaseField as PrimeField>::BigInt::NUM_LIMBS * 8;
        let mut b = vec![0; len * 2];
        reader.read_exact(&mut b)?;

        let x = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..len])?)
            .ok_or(Error::Read(String::from("Unable to read Fq x")))?
            .div(r);
        let y = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[len..])?)
            .ok_or(Error::Read(String::from("Unable to read Fq y")))?
            .div(r);

        let g1 = Self::G1Affine::new_unchecked(x, y);

        assert!(g1.is_on_curve(), "G1 must be on curve");

        Ok(g1)
    }

    fn read_ptau_g2<R: Read>(reader: &mut R) -> Result<Self::G2Affine, Error> {
        let r = montgomery_r::<Self::BaseField>();
        let len = <Self::BaseField as PrimeField>::BigInt::NUM_LIMBS * 8;
        let mut b = vec![0; len * 4];
        reader.read_exact(&mut b)?;

        let x0 = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..len])?)
            .ok_or(Error::Read(String::from("Unable to read Fq x0")))?
            .div(r);
        let x1 = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(
            &b[len..(len * 2)],
        )?)
            .ok_or(Error::Read(String::from("Unable to read Fq x1")))?
            .div(r);
        let x = QuadExtField::new(x0, x1);
        let y0 = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(
            &b[(len * 2)..(len * 3)],
        )?)
        .ok_or(Error::Read(String::from("Unable to read Fq y0")))?
        .div(r);
        let y1 = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[(len * 3)..])?)
            .ok_or(Error::Read(String::from("Unable to read Fq y1")))?
            .div(r);
        let y = QuadExtField::new(y0, y1);
        let g2 = Self::G2Affine::new_unchecked(x, y);

        assert!(g2.is_on_curve(), "G2 must be on curve");

        Ok(g2)
    }

    fn read_radix_g1<R: Read>(reader: &mut R) -> Result<Self::G1Affine, Error> {
        let len = <Self::BaseField as PrimeField>::BigInt::NUM_LIMBS * 8;
        let x = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b[0] &= 0x3f;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq x")))
        }?;
        let y = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq y")))
        }?;
        let g1 = Self::G1Affine::new_unchecked(x, y);

        assert!(g1.is_on_curve(), "G1 must be on curve");

        Ok(g1)
    }

    fn read_radix_g2<R: Read>(reader: &mut R) -> Result<Self::G2Affine, Error> {
        let len = <Self::BaseField as PrimeField>::BigInt::NUM_LIMBS * 8;
        let x1 = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq x1")))
        }?;
        let x0 = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b[0] &= 0x3f;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq x0")))
        }?;
        let x = QuadExtField::new(x0, x1);
        let y1 = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq y1")))
        }?;
        let y0 = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq y0")))
        }?;
        let y = QuadExtField::new(y0, y1);
        let g2 = Self::G2Affine::new_unchecked(x, y);

        assert!(g2.is_on_curve(), "G2 must be on curve");

        Ok(g2)
    }
}

impl<T: Bls12Config> PairingReader for Bls12<T> {
    fn read_ptau_g1<R: Read>(reader: &mut R) -> Result<Self::G1Affine, Error> {
        let r = montgomery_r::<Self::BaseField>();
        let len = <Self::BaseField as PrimeField>::BigInt::NUM_LIMBS * 8;
        let mut b = vec![0; len * 2];
        reader.read_exact(&mut b)?;

        let x = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..len])?)
            .ok_or(Error::Read(String::from("Unable to read Fq x")))?
            .div(r);
        let y = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[len..])?)
            .ok_or(Error::Read(String::from("Unable to read Fq y")))?
            .div(r);

        let g1 = Self::G1Affine::new_unchecked(x, y);

        assert!(g1.is_on_curve(), "G1 must be on curve");

        Ok(g1)
    }

    fn read_ptau_g2<R: Read>(reader: &mut R) -> Result<Self::G2Affine, Error> {
        let r = montgomery_r::<Self::BaseField>();
        let len = <Self::BaseField as PrimeField>::BigInt::NUM_LIMBS * 8;
        let mut b = vec![0; len * 4];
        reader.read_exact(&mut b)?;

        let x0 = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..len])?)
            .ok_or(Error::Read(String::from("Unable to read Fq x0")))?
            .div(r);
        let x1 = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(
            &b[len..(len * 2)],
        )?)
            .ok_or(Error::Read(String::from("Unable to read Fq x1")))?
            .div(r);
        let x = QuadExtField::new(x0, x1);
        let y0 = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(
            &b[(len * 2)..(len * 3)],
        )?)
        .ok_or(Error::Read(String::from("Unable to read Fq y0")))?
        .div(r);
        let y1 = Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[(len * 3)..])?)
            .ok_or(Error::Read(String::from("Unable to read Fq y1")))?
            .div(r);
        let y = QuadExtField::new(y0, y1);
        let g2 = Self::G2Affine::new_unchecked(x, y);

        assert!(g2.is_on_curve(), "G2 must be on curve");

        Ok(g2)
    }

    fn read_radix_g1<R: Read>(reader: &mut R) -> Result<Self::G1Affine, Error> {
        let len = <Self::BaseField as PrimeField>::BigInt::NUM_LIMBS * 8;
        let x = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b[0] &= 0x3f;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq x")))
        }?;
        let y = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq y")))
        }?;
        let g1 = Self::G1Affine::new_unchecked(x, y);

        assert!(g1.is_on_curve(), "G1 must be on curve");

        Ok(g1)
    }

    fn read_radix_g2<R: Read>(reader: &mut R) -> Result<Self::G2Affine, Error> {
        let len = <Self::BaseField as PrimeField>::BigInt::NUM_LIMBS * 8;
        let x1 = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b[0] &= 0x3f;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq x1")))
        }?;
        let x0 = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq x0")))
        }?;
        let x = QuadExtField::new(x0, x1);
        let y1 = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq y1")))
        }?;
        let y0 = {
            let mut b = vec![0; len];
            reader.read_exact(&mut b)?;
            b.reverse();
            Self::BaseField::from_bigint(read_bigint_le::<Self::BaseField>(&b[..])?)
                .ok_or(Error::Read(String::from("Unable to read Fq y0")))
        }?;
        let y = QuadExtField::new(y0, y1);
        let g2 = Self::G2Affine::new_unchecked(x, y);

        assert!(g2.is_on_curve(), "G2 must be on curve");

        Ok(g2)
    }
}
