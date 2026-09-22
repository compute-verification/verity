//! Plonky3 uni-stark over Goldilocks with Keccak-256 Merkle trees and FRI (Fiat-Shamir, non-ZK):
//! `NON_ZK_PROOF_DIAGNOSTIC`.  Not Candidate B's PCS (Ligero-style, interactive; `PROTOCOL.md`).

use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher64};
use p3_uni_stark::{get_log_quotient_degree, prove, verify, Proof, StarkConfig, VerificationError};
use p3_util::log2_ceil_usize;

use crate::air::UnitAir;

pub type Val = Goldilocks;
pub type Challenge = BinomialExtensionField<Val, 2>;
type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher64<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<u8, ByteHash, 2, 32>;
type ValMmcs = FieldMerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel;
pub type Challenger = SerializingChallenger64<Val, HashChallenger<u8, ByteHash, 32>>;
pub type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
pub type Config = StarkConfig<Pcs, Challenge, Challenger>;
pub type UnitProof = Proof<Config>;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct FriParams {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
}

/// log_blowup = 2 admits constraint degree 5 (quotient degree 4); 100 queries at rate 1/4 give, under the
/// proven unique-decoding bound, -log2((1 + 1/4) / 2) = 0.678 bits per query = 67.8 bits, plus 16 PoW bits.
pub const DEFAULT_FRI: FriParams = FriParams { log_blowup: 2, num_queries: 100, proof_of_work_bits: 16 };

pub fn config(log_n: usize, fri: FriParams) -> (Config, ByteHash) {
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);
    let compress = MyCompress::new(byte_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft {};
    let fri_config = FriConfig {
        log_blowup: fri.log_blowup,
        num_queries: fri.num_queries,
        proof_of_work_bits: fri.proof_of_work_bits,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(log_n, dft, val_mmcs, fri_config);
    (Config::new(pcs), byte_hash)
}

pub fn log_quotient_degree(air: &UnitAir) -> usize {
    get_log_quotient_degree::<Val, UnitAir>(air, 0, 0)
}

pub fn prove_trace(air: &UnitAir, trace: RowMajorMatrix<Val>, fri: FriParams) -> UnitProof {
    let log_n = log2_ceil_usize(trace.height());
    let (config, byte_hash) = config(log_n, fri);
    let mut challenger = Challenger::from_hasher(vec![], byte_hash);
    prove(&config, air, &mut challenger, trace, &vec![])
}

pub fn verify_proof(air: &UnitAir, proof: &UnitProof, log_n: usize, fri: FriParams) -> Result<(), VerificationError> {
    let (config, byte_hash) = config(log_n, fri);
    let mut challenger = Challenger::from_hasher(vec![], byte_hash);
    verify(&config, air, &mut challenger, proof, &vec![])
}

pub fn proof_bytes(proof: &UnitProof) -> usize {
    bincode::serialize(proof).expect("serialise proof").len()
}
