//! The VU chain end to end on two real VUs (the bases of `vu-k1536-neg` 0 and 1 with their correct words, taken
//! from the committed negative arrays), over Goldilocks and BabyBear: honest trace proves and verifies; a wrong
//! claimed word, a broken accumulator link, a non-zero `c_0`, or a wrong step counter is rejected by `verify`.

use std::path::{Path, PathBuf};

use p3_field::{PrimeField64, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_uni_stark::Val;
use verity_direct::air::{self, ChainAir, VuIo};
use verity_direct::ctx::FieldKind;
use verity_direct::program::{COL_CLAIM, COL_STEP, STEPS};
use verity_direct::stark::{self, BabyBearBackend, Backend, BoundError, FriParams, GoldilocksBackend};
use verity_direct::statement::Statement;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/bench-instances/v1")
}

fn read_u16(p: &Path) -> Vec<u64> {
    std::fs::read(p).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u64).collect()
}

/// Two real VUs with correct words (negatives 0 and 1 are `ulp-plus`: claimed = correct + 1).
fn two_vus() -> Vec<VuIo> {
    let k = 16 * STEPS as usize;
    let x = read_u16(&root().join("vu-k1536-neg.x.u16"));
    let w = read_u16(&root().join("vu-k1536-neg.w.u16"));
    let y = read_u16(&root().join("vu-k1536-neg.y.u16"));
    (0..2).map(|i| VuIo { x: x[i * k..(i + 1) * k].to_vec(), w: w[i * k..(i + 1) * k].to_vec(), y16: y[i] - 1 }).collect()
}

const FRI: FriParams = FriParams { log_blowup: 2, num_queries: 8, proof_of_work_bits: 1 };

/// Prove the (possibly mutated) two-VU trace *for the statement of the two real words* and verify it against
/// that statement.
fn roundtrip<B: Backend>(mutate: impl Fn(&ChainAir, &air::Layout, &mut RowMajorMatrix<Val<B::SC>>)) -> bool
where
    Val<B::SC>: PrimeField64 + TwoAdicField + Send + Sync,
{
    let kind = B::KIND;
    let (air_, lay) = ChainAir::new(kind);
    let vus = two_vus();
    let stmt = Statement::new(vus.iter().map(|v| v.y16).collect());
    let (mut trace, accs): (RowMajorMatrix<Val<B::SC>>, _) = air::chain_trace(kind, lay.width, &air_.idx, &vus, 2);
    assert_eq!(accs.len(), 2);
    assert_eq!(trace.height(), 256);
    assert_eq!(stmt.rows(), 256);
    mutate(&air_, &lay, &mut trace);
    let proof = stark::prove_bound_unchecked::<B, _>(&air_, &lay, trace, FRI, &stmt);
    stark::verify_bound::<B, _>(&air_, &lay, &proof, FRI, &stmt).is_ok()
}

/// The red team's F1: a satisfying trace of *other* VUs (here the two zero VUs) offered as a proof of the two real
/// words; and a proof of the real words offered for a statement with one word changed.  Both must hit the
/// statement-column check.  F2: a 2-VU proof offered to the verifier of a 4096-VU statement hits the degree check.
fn statement_suite<B: Backend>()
where
    Val<B::SC>: PrimeField64 + TwoAdicField + Send + Sync,
{
    let kind = B::KIND;
    let (air_, lay) = ChainAir::new(kind);
    let vus = two_vus();
    let real = Statement::new(vus.iter().map(|v| v.y16).collect());
    // F1a: zero batch signed as the real statement
    let zeros = vec![air::zero_vu(), air::zero_vu()];
    let (ztrace, _): (RowMajorMatrix<Val<B::SC>>, _) = air::chain_trace(kind, lay.width, &air_.idx, &zeros, 2);
    let zproof = stark::prove_bound_unchecked::<B, _>(&air_, &lay, ztrace, FRI, &real);
    let r = stark::verify_bound::<B, _>(&air_, &lay, &zproof, FRI, &real);
    assert!(matches!(r, Err(BoundError::StatementColumn { .. })), "zero batch accepted as the real statement: {r:?}");
    // the same zero trace is a fine proof of the zero statement
    let zero_stmt = Statement::new(vec![0, 0]);
    let (ztrace, _): (RowMajorMatrix<Val<B::SC>>, _) = air::chain_trace(kind, lay.width, &air_.idx, &zeros, 2);
    let zproof = stark::prove_bound::<B, _>(&air_, &lay, ztrace, FRI, &zero_stmt);
    assert!(stark::verify_bound::<B, _>(&air_, &lay, &zproof, FRI, &zero_stmt).is_ok());
    // F1b: honest proof of the real words, verified for a statement with word 0 changed / words swapped
    let (trace, _): (RowMajorMatrix<Val<B::SC>>, _) = air::chain_trace(kind, lay.width, &air_.idx, &vus, 2);
    let proof = stark::prove_bound::<B, _>(&air_, &lay, trace, FRI, &real);
    assert!(stark::verify_bound::<B, _>(&air_, &lay, &proof, FRI, &real).is_ok(), "honest proof rejected");
    let mut other = real.clone();
    other.words[0] ^= 1;
    assert!(matches!(stark::verify_bound::<B, _>(&air_, &lay, &proof, FRI, &other), Err(BoundError::StatementColumn { .. })));
    let swapped = Statement::new(vec![real.words[1], real.words[0]]);
    if swapped.words != real.words {
        assert!(matches!(stark::verify_bound::<B, _>(&air_, &lay, &proof, FRI, &swapped), Err(BoundError::StatementColumn { .. })));
    }
    // F1c: a header difference alone (other FRI parameters) changes every challenge -> the opening argument fails
    let mut fri2 = FRI;
    fri2.num_queries += 1;
    assert!(stark::verify_bound::<B, _>(&air_, &lay, &proof, fri2, &real).is_err());
    // F2: a 2-VU proof for the 4096-VU verifier
    let mut big = real.clone();
    big.words.resize(4096, 0);
    assert_eq!(big.degree_bits(), 19);
    let r = stark::verify_bound::<B, _>(&air_, &lay, &proof, FRI, &big);
    assert!(matches!(r, Err(BoundError::DegreeBits { expected: 19, got: 8 })), "2-VU proof accepted for B = 4096: {r:?}");
    // and a 3-VU statement (same degree_bits 9? no: 3*96 = 288 -> 512) also differs in degree
    let three = Statement::new(vec![real.words[0], real.words[1], 0]);
    assert!(matches!(stark::verify_bound::<B, _>(&air_, &lay, &proof, FRI, &three), Err(BoundError::DegreeBits { .. })));
}

#[test]
fn statement_binding_goldilocks() {
    if !root().join("vu-k1536-neg.x.u16").is_file() {
        eprintln!("fixture missing; skipped");
        return;
    }
    statement_suite::<GoldilocksBackend>();
}

#[test]
fn statement_binding_babybear() {
    if !root().join("vu-k1536-neg.x.u16").is_file() {
        eprintln!("fixture missing; skipped");
        return;
    }
    statement_suite::<BabyBearBackend>();
}

/// With `--bind-operands` the operand columns are public too: a proof of the real VUs is not a proof of the same
/// words with one operand changed.
#[test]
fn operand_binding_goldilocks() {
    if !root().join("vu-k1536-neg.x.u16").is_file() {
        eprintln!("fixture missing; skipped");
        return;
    }
    type B = GoldilocksBackend;
    let (air_, lay) = ChainAir::new(FieldKind::Goldilocks);
    let vus = two_vus();
    let x: Vec<u64> = vus.iter().flat_map(|v| v.x.iter().copied()).collect();
    let w: Vec<u64> = vus.iter().flat_map(|v| v.w.iter().copied()).collect();
    let stmt = Statement::with_operands(vus.iter().map(|v| v.y16).collect(), x.clone(), w.clone());
    let (trace, _): (RowMajorMatrix<Val<<B as Backend>::SC>>, _) = air::chain_trace(FieldKind::Goldilocks, lay.width, &air_.idx, &vus, 2);
    let proof = stark::prove_bound::<B, _>(&air_, &lay, trace, FRI, &stmt);
    assert!(stark::verify_bound::<B, _>(&air_, &lay, &proof, FRI, &stmt).is_ok());
    let mut x2 = x;
    x2[1536 + 7] ^= 1;
    let other = Statement::with_operands(stmt.words.clone(), x2, w);
    let r = stark::verify_bound::<B, _>(&air_, &lay, &proof, FRI, &other);
    assert!(matches!(r, Err(BoundError::StatementColumn { .. })), "{r:?}");
}

fn set<G: PrimeField64>(t: &mut RowMajorMatrix<G>, row: usize, col: usize, v: u64) {
    let w = t.width();
    t.values[row * w + col] = G::from_canonical_u64(v);
}

fn suite<B: Backend>()
where
    Val<B::SC>: PrimeField64 + TwoAdicField + Send + Sync,
{
    let steps = STEPS as usize;
    assert!(roundtrip::<B>(|_, _, _| {}), "honest chain rejected");
    // wrong claimed word on the last row of VU 0
    assert!(!roundtrip::<B>(|_, lay, t| {
        let c = lay.col(COL_CLAIM);
        let r = steps - 1;
        let w = t.width();
        let v = t.values[r * w + c].as_canonical_u64();
        set(t, r, c, v ^ 1);
    }));
    // broken accumulator link: the input word of row 1 no longer equals the output word of row 0
    assert!(!roundtrip::<B>(|air_, _, t| {
        let c = air_.idx.c[air_.idx.c.len() - 1];
        let w = t.width();
        let v = t.values[w + c].as_canonical_u64();
        set(t, 1, c, v ^ 1);
    }));
    // c_0 != 0 at the VU boundary (row 96): flip the low bit of the input word
    assert!(!roundtrip::<B>(|air_, _, t| {
        let c = air_.idx.c[air_.idx.c.len() - 1];
        set(t, steps, c, 1);
    }));
    // step counter skips a row
    assert!(!roundtrip::<B>(|_, lay, t| {
        let s = lay.col(COL_STEP);
        set(t, 5, s, 6);
    }));
}

#[test]
fn chain_goldilocks() {
    if !root().join("vu-k1536-neg.x.u16").is_file() {
        eprintln!("fixture missing; skipped");
        return;
    }
    suite::<GoldilocksBackend>();
}

#[test]
fn chain_babybear() {
    if !root().join("vu-k1536-neg.x.u16").is_file() {
        eprintln!("fixture missing; skipped");
        return;
    }
    suite::<BabyBearBackend>();
}

#[test]
fn babybear_words_are_split() {
    let (_, lay) = ChainAir::new(FieldKind::BabyBear);
    assert!(lay.has("x.c_hi") && lay.has("x.c_lo") && lay.has("pack.y_hi") && lay.has("pack.y_lo"));
    let (_, lay_g) = ChainAir::new(FieldKind::Goldilocks);
    assert!(lay_g.has("x.c") && lay_g.has("pack.y"));
}
