//! Plonky3 uni-stark with Keccak-256 Merkle trees and FRI (Fiat-Shamir, non-ZK): `NON_ZK_PROOF_DIAGNOSTIC`.
//! Not Candidate B's PCS (Ligero-style, interactive; `PROTOCOL.md`).  Two backends: Goldilocks with the
//! quadratic extension (the wave-1 configuration) and BabyBear with the degree-4 extension.

use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::PrimeField64;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32, SerializingHasher64};
use p3_uni_stark::{get_log_quotient_degree, prove, verify, Proof, ProverConstraintFolder, StarkConfig, StarkGenericConfig, SymbolicAirBuilder, Val, VerificationError, VerifierConstraintFolder};
use p3_util::log2_ceil_usize;

use crate::challenger::FixedSerializingChallenger64;
use crate::ctx::FieldKind;
use crate::rtt::RttChallenger;

type ByteHash = Keccak256Hash;
type MyCompress = CompressionFunctionFromHasher<u8, ByteHash, 2, 32>;
type Dft = Radix2DitParallel;

// --- Goldilocks -------------------------------------------------------------------------------------------------
pub type Val64 = Goldilocks;
pub type Challenge = BinomialExtensionField<Val64, 2>;
type FieldHash64 = SerializingHasher64<ByteHash>;
type ValMmcs64 = FieldMerkleTreeMmcs<Val64, u8, FieldHash64, MyCompress, 32>;
type ChallengeMmcs64 = ExtensionMmcs<Val64, Challenge, ValMmcs64>;
pub type Challenger = RttChallenger<FixedSerializingChallenger64<Val64, HashChallenger<u8, ByteHash, 32>>>;
pub type Pcs = TwoAdicFriPcs<Val64, Dft, ValMmcs64, ChallengeMmcs64>;
pub type Config = StarkConfig<Pcs, Challenge, Challenger>;
pub type UnitProof = Proof<Config>;

// --- BabyBear ---------------------------------------------------------------------------------------------------
pub type Val32 = BabyBear;
pub type ChallengeBb = BinomialExtensionField<Val32, 4>;
type FieldHash32 = SerializingHasher32<ByteHash>;
type ValMmcs32 = FieldMerkleTreeMmcs<Val32, u8, FieldHash32, MyCompress, 32>;
type ChallengeMmcs32 = ExtensionMmcs<Val32, ChallengeBb, ValMmcs32>;
pub type ChallengerBb = RttChallenger<SerializingChallenger32<Val32, HashChallenger<u8, ByteHash, 32>>>;
pub type PcsBb = TwoAdicFriPcs<Val32, Dft, ValMmcs32, ChallengeMmcs32>;
pub type ConfigBb = StarkConfig<PcsBb, ChallengeBb, ChallengerBb>;

// --- BabyBear with the degree-5 extension (section 14.2: the lookup identity term needs a larger challenge field) --
pub type ChallengeBb5 = BinomialExtensionField<Val32, 5>;
type ChallengeMmcs32x5 = ExtensionMmcs<Val32, ChallengeBb5, ValMmcs32>;
pub type PcsBb5 = TwoAdicFriPcs<Val32, Dft, ValMmcs32, ChallengeMmcs32x5>;
pub type ConfigBb5 = StarkConfig<PcsBb5, ChallengeBb5, ChallengerBb>;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct FriParams {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
}

/// log_blowup = 2 admits constraint degree 5 (quotient degree 4); 100 queries at rate 1/4 give, under the
/// proven unique-decoding bound, -log2((1 + 1/4) / 2) = 0.678 bits per query = 67.8 bits, plus 16 PoW bits.
pub const DEFAULT_FRI: FriParams = FriParams { log_blowup: 2, num_queries: 100, proof_of_work_bits: 16 };

impl FriParams {
    /// Proven FRI soundness (unique decoding radius): `-log2((1 + rate) / 2)` per query plus the PoW bits; field
    /// terms (DEEP/quotient batching over the challenge field) omitted.
    pub fn bits_proven(&self) -> f64 {
        let rate = 1.0 / (1u64 << self.log_blowup) as f64;
        self.num_queries as f64 * -((1.0 + rate) / 2.0).log2() + self.proof_of_work_bits as f64
    }
    /// Conjectured (list-decoding up to capacity): `log_blowup` bits per query plus PoW.
    pub fn bits_conjectured(&self) -> f64 {
        self.num_queries as f64 * (self.log_blowup as f64) + self.proof_of_work_bits as f64
    }
    /// Queries that reach `bits` under the proven bound at this blowup and PoW.
    pub fn queries_for_proven(&self, bits: f64) -> usize {
        let rate = 1.0 / (1u64 << self.log_blowup) as f64;
        let per = -((1.0 + rate) / 2.0).log2();
        ((bits - self.proof_of_work_bits as f64) / per).ceil() as usize
    }
}

/// A backend: the trace field and the Plonky3 configuration over it.
pub trait Backend
where
    Val<Self::SC>: PrimeField64 + Send + Sync,
{
    type SC: StarkGenericConfig;
    const KIND: FieldKind;
    fn config(log_n: usize, fri: FriParams) -> Self::SC;
    fn challenger() -> <Self::SC as StarkGenericConfig>::Challenger;
    fn challenge_bits() -> usize;
    fn challenge_name() -> &'static str;
}

pub struct GoldilocksBackend;
pub struct BabyBearBackend;

impl Backend for GoldilocksBackend {
    type SC = Config;
    const KIND: FieldKind = FieldKind::Goldilocks;
    fn config(log_n: usize, fri: FriParams) -> Config {
        let byte_hash = ByteHash {};
        let field_hash = FieldHash64::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs64::new(field_hash, compress);
        let challenge_mmcs = ChallengeMmcs64::new(val_mmcs.clone());
        let fri_config = FriConfig { log_blowup: fri.log_blowup, num_queries: fri.num_queries, proof_of_work_bits: fri.proof_of_work_bits, mmcs: challenge_mmcs };
        Config::new(Pcs::new(log_n, Dft {}, val_mmcs, fri_config))
    }
    fn challenger() -> Challenger {
        RttChallenger::new(FixedSerializingChallenger64::from_hasher(vec![], ByteHash {}))
    }
    fn challenge_bits() -> usize {
        128
    }
    fn challenge_name() -> &'static str {
        "goldilocks^2"
    }
}

impl Backend for BabyBearBackend {
    type SC = ConfigBb;
    const KIND: FieldKind = FieldKind::BabyBear;
    fn config(log_n: usize, fri: FriParams) -> ConfigBb {
        let byte_hash = ByteHash {};
        let field_hash = FieldHash32::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs32::new(field_hash, compress);
        let challenge_mmcs = ChallengeMmcs32::new(val_mmcs.clone());
        let fri_config = FriConfig { log_blowup: fri.log_blowup, num_queries: fri.num_queries, proof_of_work_bits: fri.proof_of_work_bits, mmcs: challenge_mmcs };
        ConfigBb::new(PcsBb::new(log_n, Dft {}, val_mmcs, fri_config))
    }
    fn challenger() -> ChallengerBb {
        RttChallenger::new(SerializingChallenger32::from_hasher(vec![], ByteHash {}))
    }
    fn challenge_bits() -> usize {
        124
    }
    fn challenge_name() -> &'static str {
        "babybear^4"
    }
}

pub struct BabyBear5Backend;

impl Backend for BabyBear5Backend {
    type SC = ConfigBb5;
    const KIND: FieldKind = FieldKind::BabyBear;
    fn config(log_n: usize, fri: FriParams) -> ConfigBb5 {
        let byte_hash = ByteHash {};
        let field_hash = FieldHash32::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs32::new(field_hash, compress);
        let challenge_mmcs = ChallengeMmcs32x5::new(val_mmcs.clone());
        let fri_config = FriConfig { log_blowup: fri.log_blowup, num_queries: fri.num_queries, proof_of_work_bits: fri.proof_of_work_bits, mmcs: challenge_mmcs };
        ConfigBb5::new(PcsBb5::new(log_n, Dft {}, val_mmcs, fri_config))
    }
    fn challenger() -> ChallengerBb {
        RttChallenger::new(SerializingChallenger32::from_hasher(vec![], ByteHash {}))
    }
    fn challenge_bits() -> usize {
        155
    }
    fn challenge_name() -> &'static str {
        "babybear^5"
    }
}

pub fn log_quotient_degree_of<G: PrimeField64, A: Air<SymbolicAirBuilder<G>>>(air: &A) -> usize {
    get_log_quotient_degree::<G, A>(air, 0, 0)
}

#[cfg(debug_assertions)]
pub fn prove_with<B: Backend, A>(air: &A, trace: RowMajorMatrix<Val<B::SC>>, fri: FriParams) -> Proof<B::SC>
where
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<ProverConstraintFolder<'a, B::SC>> + for<'a> Air<p3_uni_stark::DebugConstraintBuilder<'a, Val<B::SC>>>,
{
    let log_n = log2_ceil_usize(trace.height());
    let config = B::config(log_n, fri);
    let mut challenger = B::challenger();
    prove(&config, air, &mut challenger, trace, &vec![])
}

#[cfg(not(debug_assertions))]
pub fn prove_with<B: Backend, A>(air: &A, trace: RowMajorMatrix<Val<B::SC>>, fri: FriParams) -> Proof<B::SC>
where
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<ProverConstraintFolder<'a, B::SC>>,
{
    let log_n = log2_ceil_usize(trace.height());
    let config = B::config(log_n, fri);
    let mut challenger = B::challenger();
    prove(&config, air, &mut challenger, trace, &vec![])
}

pub fn verify_with<B: Backend, A>(air: &A, proof: &Proof<B::SC>, log_n: usize, fri: FriParams) -> Result<(), VerificationError>
where
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<VerifierConstraintFolder<'a, B::SC>>,
{
    let config = B::config(log_n, fri);
    let mut challenger = B::challenger();
    verify(&config, air, &mut challenger, proof, &vec![])
}

pub fn proof_bytes_of<SC: StarkGenericConfig>(proof: &Proof<SC>) -> usize
where
    Proof<SC>: serde::Serialize,
{
    bincode::serialize(proof).expect("serialise proof").len()
}

// --- the wave-1 Goldilocks API (tests, crosscheck) ---------------------------------------------------------------

pub fn log_quotient_degree<A: Air<SymbolicAirBuilder<Val64>>>(air: &A) -> usize {
    log_quotient_degree_of::<Val64, A>(air)
}

#[cfg(debug_assertions)]
pub fn prove_trace<A>(air: &A, trace: RowMajorMatrix<Val64>, fri: FriParams) -> UnitProof
where
    A: Air<SymbolicAirBuilder<Val64>> + for<'a> Air<ProverConstraintFolder<'a, Config>> + for<'a> Air<p3_uni_stark::DebugConstraintBuilder<'a, Val64>>,
{
    prove_with::<GoldilocksBackend, A>(air, trace, fri)
}

#[cfg(not(debug_assertions))]
pub fn prove_trace<A>(air: &A, trace: RowMajorMatrix<Val64>, fri: FriParams) -> UnitProof
where
    A: Air<SymbolicAirBuilder<Val64>> + for<'a> Air<ProverConstraintFolder<'a, Config>>,
{
    prove_with::<GoldilocksBackend, A>(air, trace, fri)
}

pub fn verify_proof<A>(air: &A, proof: &UnitProof, log_n: usize, fri: FriParams) -> Result<(), VerificationError>
where
    A: Air<SymbolicAirBuilder<Val64>> + for<'a> Air<VerifierConstraintFolder<'a, Config>>,
{
    verify_with::<GoldilocksBackend, A>(air, proof, log_n, fri)
}

pub fn proof_bytes(proof: &UnitProof) -> usize {
    proof_bytes_of(proof)
}

