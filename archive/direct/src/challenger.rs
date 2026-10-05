//! A byte-serialising Fiat-Shamir challenger for 64-bit prime fields, forked from
//! `p3_challenger::SerializingChallenger64` (0.4.3-succinct) with one fix.
//!
//! FINDING (this lane, 2026-09-22).  The upstream `sample` computes `pow_of_two_bound = (1 << log_size) - 1`
//! with `log_size = log2_ceil(p) = 64` for Goldilocks.  In a release build `1 << 64` wraps to `1`, the mask is
//! `0`, and EVERY sampled challenge is zero: alpha = 0 (only the last constraint is folded into the quotient),
//! zeta = 0, all FRI betas = 0 and every query index is 0.  Proofs of traces violating any constraint other than
//! the last one verified.  Detected by `tests/negative.rs::mutated_traces_do_not_verify`.  This fork masks with
//! `u64::MAX` when `log_size >= 64`.

use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger, HashChallenger};
use p3_field::{ExtensionField, PrimeField64};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, Hash};
use p3_util::log2_ceil_u64;

#[derive(Clone, Debug)]
pub struct FixedSerializingChallenger64<F, Inner> {
    inner: Inner,
    _marker: PhantomData<F>,
}

impl<F: PrimeField64, Inner: CanObserve<u8>> FixedSerializingChallenger64<F, Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self { inner, _marker: PhantomData }
    }
}

impl<F, H> FixedSerializingChallenger64<F, HashChallenger<u8, H, 32>>
where
    F: PrimeField64,
    H: CryptographicHasher<u8, [u8; 32]>,
{
    pub fn from_hasher(initial_state: Vec<u8>, hasher: H) -> Self {
        Self::new(HashChallenger::new(initial_state, hasher))
    }
}

impl<F: PrimeField64, Inner: CanObserve<u8>> CanObserve<F> for FixedSerializingChallenger64<F, Inner> {
    fn observe(&mut self, value: F) {
        self.inner.observe_slice(&value.as_canonical_u64().to_le_bytes());
    }
}

impl<F: PrimeField64, const N: usize, Inner: CanObserve<u8>> CanObserve<Hash<F, u8, N>>
    for FixedSerializingChallenger64<F, Inner>
{
    fn observe(&mut self, values: Hash<F, u8, N>) {
        for value in values {
            self.inner.observe(value);
        }
    }
}

impl<F, EF, Inner> CanSample<EF> for FixedSerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    Inner: CanSample<u8>,
{
    fn sample(&mut self) -> EF {
        let modulus = F::ORDER_U64;
        let log_size = log2_ceil_u64(F::ORDER_U64);
        let pow_of_two_bound: u64 = if log_size >= 64 { u64::MAX } else { (1u64 << log_size) - 1 };
        let sample_base = |inner: &mut Inner| loop {
            let value = u64::from_le_bytes(inner.sample_array::<8>());
            let value = value & pow_of_two_bound;
            if value < modulus {
                return F::from_canonical_u64(value);
            }
        };
        EF::from_base_fn(|_| sample_base(&mut self.inner))
    }
}

impl<F, Inner> CanSampleBits<usize> for FixedSerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    Inner: CanSample<u8>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        assert!(bits < (usize::BITS as usize));
        let rand_usize = u64::from_le_bytes(self.inner.sample_array::<8>()) as usize;
        rand_usize & ((1 << bits) - 1)
    }
}

impl<F, Inner> GrindingChallenger for FixedSerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    Inner: CanSample<u8> + CanObserve<u8> + Clone + Send + Sync,
{
    type Witness = F;

    fn grind(&mut self, bits: usize) -> Self::Witness {
        let witness = (0..F::ORDER_U64)
            .into_par_iter()
            .map(|i| F::from_canonical_u64(i))
            .find_any(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find witness");
        assert!(self.check_witness(bits, witness));
        witness
    }
}

impl<F, Inner> FieldChallenger<F> for FixedSerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    Inner: CanSample<u8> + CanObserve<u8> + Clone + Send + Sync,
{
}
