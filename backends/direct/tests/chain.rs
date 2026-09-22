//! The VU chain end to end on two real VUs (the bases of `vu-k1536-neg` 0 and 1 with their correct words, taken
//! from the committed negative arrays), over Goldilocks and BabyBear: honest trace proves and verifies; a wrong
//! claimed word, a broken accumulator link, a non-zero `c_0`, or a wrong step counter is rejected by `verify`.

use std::path::{Path, PathBuf};

use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_uni_stark::Val;
use verity_direct::air::{self, ChainAir, VuIo};
use verity_direct::ctx::FieldKind;
use verity_direct::program::{COL_CLAIM, COL_STEP, STEPS};
use verity_direct::stark::{self, BabyBearBackend, Backend, FriParams, GoldilocksBackend};

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

fn roundtrip<B: Backend>(mutate: impl Fn(&ChainAir, &air::Layout, &mut RowMajorMatrix<Val<B::SC>>)) -> bool
where
    Val<B::SC>: PrimeField64 + Send + Sync,
{
    let kind = B::KIND;
    let (air_, lay) = ChainAir::new(kind);
    let vus = two_vus();
    let (mut trace, accs): (RowMajorMatrix<Val<B::SC>>, _) = air::chain_trace(kind, lay.width, &air_.idx, &vus, 2);
    assert_eq!(accs.len(), 2);
    assert_eq!(trace.height(), 256);
    mutate(&air_, &lay, &mut trace);
    let log_n = 8;
    let proof = stark::prove_with::<B, _>(&air_, trace, FRI);
    stark::verify_with::<B, _>(&air_, &proof, log_n, FRI).is_ok()
}

fn set<G: PrimeField64>(t: &mut RowMajorMatrix<G>, row: usize, col: usize, v: u64) {
    let w = t.width();
    t.values[row * w + col] = G::from_canonical_u64(v);
}

fn suite<B: Backend>()
where
    Val<B::SC>: PrimeField64 + Send + Sync,
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
