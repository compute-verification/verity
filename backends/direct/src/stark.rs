//! Plonky3 uni-stark with Keccak-256 Merkle trees and FRI (Fiat-Shamir, non-ZK): `NON_ZK_PROOF_DIAGNOSTIC`.
//! Not Candidate B's PCS (Ligero-style, interactive; `PROTOCOL.md`).  Two backends: Goldilocks with the
//! quadratic extension (the wave-1 configuration) and BabyBear with the degree-4 extension.

use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, CanSample, FieldChallenger, HashChallenger, SerializingChallenger32};
use p3_commit::{ExtensionMmcs, Pcs as PcsTrait, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractExtensionField, AbstractField, Field, PrimeField64, TwoAdicField};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;
use p3_matrix::Matrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32, SerializingHasher64};
use p3_uni_stark::{get_log_quotient_degree, prove, verify, Proof, ProverConstraintFolder, StarkConfig, StarkGenericConfig, SymbolicAirBuilder, Val, VerificationError, VerifierConstraintFolder};
use p3_util::log2_ceil_usize;

use crate::air::Layout;
use crate::challenger::FixedSerializingChallenger64;
use crate::ctx::FieldKind;
use crate::rtt::RttChallenger;
use crate::statement::{self, Statement};

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
/// BabyBear^6 (`vendor/p3-baby-bear`: X^6 - 22), 186-bit challenges: the extension the 2^-128 lookup label needs.
pub type ChallengeBb6 = BinomialExtensionField<Val32, 6>;
type ChallengeMmcs32x6 = ExtensionMmcs<Val32, ChallengeBb6, ValMmcs32>;
pub type PcsBb6 = TwoAdicFriPcs<Val32, Dft, ValMmcs32, ChallengeMmcs32x6>;
pub type ConfigBb6 = StarkConfig<PcsBb6, ChallengeBb6, ChallengerBb>;

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

pub struct BabyBear6Backend;

impl Backend for BabyBear6Backend {
    type SC = ConfigBb6;
    const KIND: FieldKind = FieldKind::BabyBear;
    fn config(log_n: usize, fri: FriParams) -> ConfigBb6 {
        let byte_hash = ByteHash {};
        let field_hash = FieldHash32::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs32::new(field_hash, compress);
        let challenge_mmcs = ChallengeMmcs32x6::new(val_mmcs.clone());
        let fri_config = FriConfig { log_blowup: fri.log_blowup, num_queries: fri.num_queries, proof_of_work_bits: fri.proof_of_work_bits, mmcs: challenge_mmcs };
        ConfigBb6::new(PcsBb6::new(log_n, Dft {}, val_mmcs, fri_config))
    }
    fn challenger() -> ChallengerBb {
        RttChallenger::new(SerializingChallenger32::from_hasher(vec![], ByteHash {}))
    }
    fn challenge_bits() -> usize {
        186
    }
    fn challenge_name() -> &'static str {
        "babybear^6"
    }
}

pub fn log_quotient_degree_of<G: PrimeField64, A: Air<SymbolicAirBuilder<G>>>(air: &A) -> usize {
    get_log_quotient_degree::<G, A>(air, 0, 0)
}

// --- unbound (statement-free) prove/verify: the wave-1 UnitAir diagnostic only --------------------------------------
//
// SOUNDNESS (red team F1/F2, 2026-09-22): a proof produced here proves "some satisfying trace of the height written
// in the proof exists" and nothing about which statement.  The chain AIR must go through `prove_bound` /
// `verify_bound` below; these two stay for the transition-unit diagnostic (`bench`, `tests/negative.rs`), whose
// rows carry no statement.

#[cfg(debug_assertions)]
pub fn prove_unbound<B: Backend, A>(air: &A, trace: RowMajorMatrix<Val<B::SC>>, fri: FriParams) -> Proof<B::SC>
where
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<ProverConstraintFolder<'a, B::SC>> + for<'a> Air<p3_uni_stark::DebugConstraintBuilder<'a, Val<B::SC>>>,
{
    let log_n = log2_ceil_usize(trace.height());
    let config = B::config(log_n, fri);
    let mut challenger = B::challenger();
    prove(&config, air, &mut challenger, trace, &vec![])
}

#[cfg(not(debug_assertions))]
pub fn prove_unbound<B: Backend, A>(air: &A, trace: RowMajorMatrix<Val<B::SC>>, fri: FriParams) -> Proof<B::SC>
where
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<ProverConstraintFolder<'a, B::SC>>,
{
    let log_n = log2_ceil_usize(trace.height());
    let config = B::config(log_n, fri);
    let mut challenger = B::challenger();
    prove(&config, air, &mut challenger, trace, &vec![])
}

pub fn verify_unbound<B: Backend, A>(air: &A, proof: &Proof<B::SC>, log_n: usize, fri: FriParams) -> Result<(), VerificationError>
where
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<VerifierConstraintFolder<'a, B::SC>>,
{
    let config = B::config(log_n, fri);
    let mut challenger = B::challenger();
    verify(&config, air, &mut challenger, proof, &vec![])
}

// --- statement-bound prove/verify for the chain AIR (PROTOCOL.md "Statement binding") ----------------------------

/// Why a bound verification failed.  `DegreeBits` and `Statement*` are the checks the red team's forgeries
/// F2 and F1 now hit; `Inner` is uni-stark's own verdict.
#[derive(Debug)]
pub enum BoundError {
    /// The proof's `degree_bits` differs from the one the verifier derives from its statement (`B`).
    DegreeBits { expected: usize, got: usize },
    /// The proof's opened rows do not have the AIR's width, or the bytes do not decode.
    ProofShape,
    /// A public column's committed polynomial does not evaluate at ζ to the statement's polynomial.
    StatementColumn { column: usize, name: String },
    Inner(VerificationError),
}

impl std::fmt::Display for BoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundError::DegreeBits { expected, got } => write!(f, "DegreeBitsMismatch(expected {expected}, proof says {got})"),
            BoundError::ProofShape => write!(f, "InvalidProofShape"),
            BoundError::StatementColumn { column, name } => write!(f, "StatementColumnMismatch({name} = column {column} at zeta)"),
            BoundError::Inner(e) => write!(f, "{e:?}"),
        }
    }
}

fn absorb_header<B: Backend>(challenger: &mut <B::SC as StarkGenericConfig>::Challenger, kind: FieldKind, lay: &Layout, fri: FriParams, stmt: &Statement)
where
    Val<B::SC>: PrimeField64 + Send + Sync,
{
    let id = statement::air_id(kind, lay);
    let ext = <<B::SC as StarkGenericConfig>::Challenge as AbstractExtensionField<Val<B::SC>>>::D;
    for v in statement::header::<Val<B::SC>>(kind, &id, lay.width, ext, fri, stmt) {
        challenger.observe(v);
    }
}

fn check_trace_against_statement<G: PrimeField64>(trace: &RowMajorMatrix<G>, lay: &Layout, stmt: &Statement) {
    assert_eq!(trace.height(), stmt.rows(), "trace height {} != statement rows {} (B = {})", trace.height(), stmt.rows(), stmt.b());
    assert_eq!(trace.width(), lay.width, "trace width != layout width");
    for (col, name, want) in stmt.public_columns::<G>(lay) {
        for (r, w) in want.iter().enumerate() {
            assert_eq!(trace.values[r * lay.width + col], *w, "prover: column {name} row {r} differs from the statement");
        }
    }
}

/// Prove the chain trace *of the statement* `stmt`: the header (AIR id, field, B, K, degree_bits, FRI parameters,
/// every claimed word) is absorbed before the first challenge; the trace must be the honest trace for `stmt`
/// (the prover checks its own public columns against the statement and panics otherwise -- a dishonest prover
/// skips that check and is caught by `verify_bound`).
#[cfg(debug_assertions)]
pub fn prove_bound<B: Backend, A>(air: &A, lay: &Layout, trace: RowMajorMatrix<Val<B::SC>>, fri: FriParams, stmt: &Statement) -> Proof<B::SC>
where
    Val<B::SC>: PrimeField64 + Send + Sync,
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<ProverConstraintFolder<'a, B::SC>> + for<'a> Air<p3_uni_stark::DebugConstraintBuilder<'a, Val<B::SC>>>,
{
    check_trace_against_statement(&trace, lay, stmt);
    let config = B::config(stmt.degree_bits(), fri);
    let mut challenger = B::challenger();
    absorb_header::<B>(&mut challenger, B::KIND, lay, fri, stmt);
    prove(&config, air, &mut challenger, trace, &vec![])
}

#[cfg(not(debug_assertions))]
pub fn prove_bound<B: Backend, A>(air: &A, lay: &Layout, trace: RowMajorMatrix<Val<B::SC>>, fri: FriParams, stmt: &Statement) -> Proof<B::SC>
where
    Val<B::SC>: PrimeField64 + Send + Sync,
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<ProverConstraintFolder<'a, B::SC>>,
{
    check_trace_against_statement(&trace, lay, stmt);
    let config = B::config(stmt.degree_bits(), fri);
    let mut challenger = B::challenger();
    absorb_header::<B>(&mut challenger, B::KIND, lay, fri, stmt);
    prove(&config, air, &mut challenger, trace, &vec![])
}

/// The red team's battery needs the prover to sign a trace for a statement it does *not* match (that is the
/// forgery attempt).  Same as `prove_bound` without the prover-side consistency check.
#[cfg(not(debug_assertions))]
pub fn prove_bound_unchecked<B: Backend, A>(air: &A, lay: &Layout, trace: RowMajorMatrix<Val<B::SC>>, fri: FriParams, stmt: &Statement) -> Proof<B::SC>
where
    Val<B::SC>: PrimeField64 + Send + Sync,
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<ProverConstraintFolder<'a, B::SC>>,
{
    let config = B::config(log2_ceil_usize(trace.height()).max(stmt.degree_bits()), fri);
    let mut challenger = B::challenger();
    absorb_header::<B>(&mut challenger, B::KIND, lay, fri, stmt);
    prove(&config, air, &mut challenger, trace, &vec![])
}

#[cfg(debug_assertions)]
pub fn prove_bound_unchecked<B: Backend, A>(air: &A, lay: &Layout, trace: RowMajorMatrix<Val<B::SC>>, fri: FriParams, stmt: &Statement) -> Proof<B::SC>
where
    Val<B::SC>: PrimeField64 + Send + Sync,
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<ProverConstraintFolder<'a, B::SC>> + for<'a> Air<p3_uni_stark::DebugConstraintBuilder<'a, Val<B::SC>>>,
{
    let config = B::config(log2_ceil_usize(trace.height()).max(stmt.degree_bits()), fri);
    let mut challenger = B::challenger();
    absorb_header::<B>(&mut challenger, B::KIND, lay, fri, stmt);
    prove(&config, air, &mut challenger, trace, &vec![])
}

type Com<SC> = <<SC as StarkGenericConfig>::Pcs as PcsTrait<<SC as StarkGenericConfig>::Challenge, <SC as StarkGenericConfig>::Challenger>>::Commitment;
type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as PcsTrait<<SC as StarkGenericConfig>::Challenge, <SC as StarkGenericConfig>::Challenger>>::Proof;

/// `p3_uni_stark::Proof` as the verifier receives it (its fields are crate-private upstream; the wire format is
/// bincode of exactly this shape, pinned by `=0.4.3-succinct`).
#[derive(serde::Deserialize)]
#[serde(bound = "")]
struct WireProof<SC: StarkGenericConfig> {
    commitments: WireCommitments<Com<SC>>,
    opened_values: WireOpened<SC::Challenge>,
    opening_proof: PcsProof<SC>,
    degree_bits: usize,
}
#[derive(serde::Deserialize)]
struct WireCommitments<C> {
    trace: C,
    quotient_chunks: C,
}
#[derive(serde::Deserialize)]
struct WireOpened<EF> {
    trace_local: Vec<EF>,
    trace_next: Vec<EF>,
    quotient_chunks: Vec<Vec<EF>>,
}

/// Verify a proof of `stmt` from its wire bytes.  Order: (1) `degree_bits` is fixed by the statement, the proof's
/// must equal it (F2); (2) the header is absorbed (F1/F3: nothing challenge-relevant is read from the proof);
/// (3) uni-stark's transcript up to ζ; (4) every public column's opening at ζ and ζ·g equals the verifier's own
/// evaluation of the statement polynomial (F1); (5) the PCS opening argument and the OOD quotient identity.
pub fn verify_bound_bytes<B: Backend, A>(air: &A, lay: &Layout, proof_bytes: &[u8], fri: FriParams, stmt: &Statement) -> Result<(), BoundError>
where
    Val<B::SC>: PrimeField64 + TwoAdicField + Send + Sync,
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<VerifierConstraintFolder<'a, B::SC>>,
{
    type SCChallenge<B> = <<B as Backend>::SC as StarkGenericConfig>::Challenge;
    let proof: WireProof<B::SC> = bincode::deserialize(proof_bytes).map_err(|_| BoundError::ProofShape)?;
    let expected_bits = stmt.degree_bits();
    if proof.degree_bits != expected_bits {
        return Err(BoundError::DegreeBits { expected: expected_bits, got: proof.degree_bits });
    }
    let degree_bits = expected_bits;
    let degree = 1usize << degree_bits;
    let config = B::config(degree_bits, fri);
    let mut challenger = B::challenger();
    absorb_header::<B>(&mut challenger, B::KIND, lay, fri, stmt);

    let log_quotient_degree = get_log_quotient_degree::<Val<B::SC>, A>(air, 0, 0);
    let quotient_degree = 1 << log_quotient_degree;
    let pcs = config.pcs();
    let trace_domain = pcs.natural_domain_for_degree(degree);
    let quotient_domain = trace_domain.create_disjoint_domain(1 << (degree_bits + log_quotient_degree));
    let quotient_chunks_domains = quotient_domain.split_domains(quotient_degree);

    let air_width = <A as BaseAir<Val<B::SC>>>::width(air);
    let ext_d = <SCChallenge<B> as AbstractExtensionField<Val<B::SC>>>::D;
    let opened = &proof.opened_values;
    let valid_shape = air_width == lay.width
        && opened.trace_local.len() == air_width
        && opened.trace_next.len() == air_width
        && opened.quotient_chunks.len() == quotient_degree
        && opened.quotient_chunks.iter().all(|qc| qc.len() == ext_d);
    if !valid_shape {
        return Err(BoundError::ProofShape);
    }

    challenger.observe(proof.commitments.trace.clone());
    let alpha: SCChallenge<B> = challenger.sample_ext_element();
    challenger.observe(proof.commitments.quotient_chunks.clone());
    let zeta: SCChallenge<B> = challenger.sample();
    let zeta_next = trace_domain.next_point(zeta).unwrap();

    // the statement's public columns at ζ and ζ·g
    for (col, name, want) in stmt.public_columns::<Val<B::SC>>(lay) {
        let (w0, w1) = statement::eval_pair::<Val<B::SC>, SCChallenge<B>>(&want, zeta);
        if opened.trace_local[col] != w0 || opened.trace_next[col] != w1 {
            return Err(BoundError::StatementColumn { column: col, name });
        }
    }

    pcs.verify(
        vec![
            (proof.commitments.trace.clone(), vec![(trace_domain, vec![(zeta, opened.trace_local.clone()), (zeta_next, opened.trace_next.clone())])]),
            (
                proof.commitments.quotient_chunks.clone(),
                quotient_chunks_domains.iter().zip(&opened.quotient_chunks).map(|(domain, values)| (*domain, vec![(zeta, values.clone())])).collect(),
            ),
        ],
        &proof.opening_proof,
        &mut challenger,
    )
    .map_err(|_| BoundError::Inner(VerificationError::InvalidOpeningArgument))?;

    let zps: Vec<SCChallenge<B>> = quotient_chunks_domains
        .iter()
        .enumerate()
        .map(|(i, domain)| {
            quotient_chunks_domains
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, other)| other.zp_at_point(zeta) * other.zp_at_point(domain.first_point()).inverse())
                .product::<SCChallenge<B>>()
        })
        .collect();
    let quotient = opened
        .quotient_chunks
        .iter()
        .enumerate()
        .map(|(ch_i, ch)| ch.iter().enumerate().map(|(e_i, &c)| zps[ch_i] * <SCChallenge<B> as AbstractExtensionField<Val<B::SC>>>::monomial(e_i) * c).sum::<SCChallenge<B>>())
        .sum::<SCChallenge<B>>();

    let sels = trace_domain.selectors_at_point(zeta);
    let main = VerticalPair::new(RowMajorMatrixView::new_row(&opened.trace_local), RowMajorMatrixView::new_row(&opened.trace_next));
    let public_values: Vec<Val<B::SC>> = vec![];
    let mut folder = VerifierConstraintFolder::<B::SC> {
        main,
        public_values: &public_values,
        is_first_row: sels.is_first_row,
        is_last_row: sels.is_last_row,
        is_transition: sels.is_transition,
        alpha,
        accumulator: SCChallenge::<B>::zero(),
    };
    air.eval(&mut folder);
    if folder.accumulator * sels.inv_zeroifier != quotient {
        return Err(BoundError::Inner(VerificationError::OodEvaluationMismatch));
    }
    Ok(())
}

/// `verify_bound_bytes` on an in-memory proof (serialised first: the verifier only ever sees bytes).
pub fn verify_bound<B: Backend, A>(air: &A, lay: &Layout, proof: &Proof<B::SC>, fri: FriParams, stmt: &Statement) -> Result<(), BoundError>
where
    Val<B::SC>: PrimeField64 + TwoAdicField + Send + Sync,
    A: Air<SymbolicAirBuilder<Val<B::SC>>> + for<'a> Air<VerifierConstraintFolder<'a, B::SC>>,
    Proof<B::SC>: serde::Serialize,
{
    let bytes = bincode::serialize(proof).expect("serialise proof");
    verify_bound_bytes::<B, A>(air, lay, &bytes, fri, stmt)
}

pub fn proof_bytes_of<SC: StarkGenericConfig>(proof: &Proof<SC>) -> usize
where
    Proof<SC>: serde::Serialize,
{
    bincode::serialize(proof).expect("serialise proof").len()
}

// --- the wave-1 Goldilocks API (UnitAir tests, crosscheck): unbound, transition-unit diagnostic only -------------

pub fn log_quotient_degree<A: Air<SymbolicAirBuilder<Val64>>>(air: &A) -> usize {
    log_quotient_degree_of::<Val64, A>(air)
}

#[cfg(debug_assertions)]
pub fn prove_trace<A>(air: &A, trace: RowMajorMatrix<Val64>, fri: FriParams) -> UnitProof
where
    A: Air<SymbolicAirBuilder<Val64>> + for<'a> Air<ProverConstraintFolder<'a, Config>> + for<'a> Air<p3_uni_stark::DebugConstraintBuilder<'a, Val64>>,
{
    prove_unbound::<GoldilocksBackend, A>(air, trace, fri)
}

#[cfg(not(debug_assertions))]
pub fn prove_trace<A>(air: &A, trace: RowMajorMatrix<Val64>, fri: FriParams) -> UnitProof
where
    A: Air<SymbolicAirBuilder<Val64>> + for<'a> Air<ProverConstraintFolder<'a, Config>>,
{
    prove_unbound::<GoldilocksBackend, A>(air, trace, fri)
}

pub fn verify_proof<A>(air: &A, proof: &UnitProof, log_n: usize, fri: FriParams) -> Result<(), VerificationError>
where
    A: Air<SymbolicAirBuilder<Val64>> + for<'a> Air<VerifierConstraintFolder<'a, Config>>,
{
    verify_unbound::<GoldilocksBackend, A>(air, proof, log_n, fri)
}

pub fn proof_bytes(proof: &UnitProof) -> usize {
    proof_bytes_of(proof)
}

