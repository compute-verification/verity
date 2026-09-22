//! The LogUp realisation (PROTOCOL.md section 14): layout consistency, the evaluator on honest and mutated rows,
//! and (pod-sized, `--ignored` locally) prove / verify round trips with the chain mutations.

use std::path::{Path, PathBuf};

use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractExtensionField, AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use verity_direct::air::{self, LookupChainAir, VuIo};
use verity_direct::ctx::FieldKind;
use verity_direct::lookup::{aux_matrix, group_queries, lookup_soundness_bits, GROUP_DEG, INVALID, PRE_WIDTH};
use verity_direct::lookup_stark::{prove_lookup, verify_lookup};
use verity_direct::program::{COL_CLAIM, COL_STEP, STEPS};
use verity_direct::stark::{BabyBearBackend, Backend, FriParams, Val32};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/bench-instances/v1")
}

fn read_u16(p: &Path) -> Vec<u64> {
    std::fs::read(p).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u64).collect()
}

fn two_vus() -> Vec<VuIo> {
    let k = 16 * STEPS as usize;
    let x = read_u16(&root().join("vu-k1536-neg.x.u16"));
    let w = read_u16(&root().join("vu-k1536-neg.w.u16"));
    let y = read_u16(&root().join("vu-k1536-neg.y.u16"));
    (0..2).map(|i| VuIo { x: x[i * k..(i + 1) * k].to_vec(), w: w[i * k..(i + 1) * k].to_vec(), y16: y[i] - 1 }).collect()
}

type EF4 = BinomialExtensionField<Val32, 4>;

#[test]
fn lookup_layout_consistent() {
    let (air_, lay) = LookupChainAir::new(FieldKind::BabyBear, 4);
    let ll = &air_.lay;
    // every query in exactly one group, groups within the degree budget
    let mut seen = vec![0usize; ll.n_queries()];
    for g in &ll.groups {
        assert!(g.iter().map(|&i| ll.query_degs[i]).sum::<usize>() <= GROUP_DEG);
        for &i in g {
            seen[i] += 1;
        }
    }
    assert!(seen.iter().all(|&c| c == 1));
    assert_eq!(group_queries(&ll.query_degs), ll.groups);
    assert_eq!(lay.width, lay.col_names.len());
    assert_eq!(lay.col_names.last().unwrap(), "lookup.m");
    assert_eq!(ll.aux_width(4), 4 * (ll.n_helpers() + 2));
    // no limb / selector columns remain
    assert!(!lay.col_names.iter().any(|n| n.contains(".limb[") || (n.contains(".sel[") && !n.contains(".max.sel["))));
    // tables: the 16-bit table and the four fixed tables are present
    let names: Vec<String> = ll.tables.describe().into_iter().map(|(n, _)| n).collect();
    for t in ["T16", "POW", "ALIGN", "LEAD", "NORM"] {
        assert!(names.iter().any(|n| n == t), "missing table {t}");
    }
    // the committed width is below a third of the limb realisation's
    let (_, limb_lay) = air::ChainAir::new(FieldKind::BabyBear);
    let committed = lay.width + ll.aux_width(4) + PRE_WIDTH;
    assert!(committed * 3 < limb_lay.width, "committed {committed} vs limbs {}", limb_lay.width);
    let bits = lookup_soundness_bits(1 << 19, ll.n_queries(), ll.tables.total_rows, 124.0);
    assert!(bits > 90.0 && bits < 100.0, "{bits}");
}

/// The evaluator on one honest VU (row-local + helper constraints): nothing fails; an out-of-table hint fails
/// its group's helper; a wrong table output fails likewise.
#[test]
fn lookup_rows_honest_and_mutated() {
    if !root().join("vu-k1536-neg.x.u16").is_file() {
        eprintln!("fixture missing; skipped");
        return;
    }
    let kind = FieldKind::BabyBear;
    let (air_, lay) = LookupChainAir::new(kind, 4);
    let vu = &two_vus()[0];
    let (rows, qs, _, viol) = air::chain_rows_lookup(kind, air_.width, &air_.idx, &air_.lay.tables, vu);
    assert!(viol.is_empty(), "{viol:?}");
    let fails = air::lookup_failing_rows::<Val32, EF4>(&air_, &lay, &rows, &qs, true);
    assert!(fails.is_empty(), "{:?}", &fails[..fails.len().min(5)]);
    // mutate: an out-of-table query index on row 3 (as an adversarial witness that puts a value outside T16 in
    // a range-checked column would produce)
    let mut qs2 = qs.clone();
    qs2[3][0] = INVALID;
    let fails = air::lookup_failing_rows::<Val32, EF4>(&air_, &lay, &rows, &qs2, true);
    assert!(fails.iter().any(|(r, n)| *r == 3 && n == "lookup.helper[0]"), "{fails:?}");
    // a genuinely mutated witness: the leading-bit window hint of row 7 set out of range via IntCtx overrides
    let mut ctx = verity_direct::ctx::IntCtx::new_kind(kind).with_lookup(Some(air_.lay.tables.clone()));
    ctx.record_names = false;
    let step = 7usize;
    let io = verity_direct::program::ChainIo { c: 0, a: vu.x[16 * step..16 * step + 16].to_vec(), b: vu.w[16 * step..16 * step + 16].to_vec(), step: step as u64, y16: vu.y16 };
    let name = "g0.lead.l".to_string();
    ctx.overrides.insert(name.clone(), 1 << 20);
    verity_direct::program::chain_program(&mut ctx, &io);
    assert!(ctx.violations.iter().any(|v| v.contains("not in") || v.contains("differ")), "{:?}", ctx.violations);
    let fails = air::lookup_failing_rows::<Val32, EF4>(&air_, &lay, &[ctx.cols.clone()], &[ctx.queries.clone()], false);
    assert!(!fails.is_empty());
}

const FRI: FriParams = FriParams { log_blowup: 2, num_queries: 8, proof_of_work_bits: 1 };

fn roundtrip<B: Backend>(mutate: impl Fn(&LookupChainAir, &air::Layout, &mut RowMajorMatrix<Val<B::SC>>)) -> bool
where
    Val<B::SC>: PrimeField64 + Send + Sync,
    <<B::SC as StarkGenericConfig>::Pcs as p3_commit::Pcs<<B::SC as StarkGenericConfig>::Challenge, <B::SC as StarkGenericConfig>::Challenger>>::Commitment: PartialEq,
{
    let kind = B::KIND;
    let d = <<B::SC as StarkGenericConfig>::Challenge as p3_field::AbstractExtensionField<Val<B::SC>>>::D;
    let (air_, lay) = LookupChainAir::new(kind, d);
    let vus = two_vus();
    let (mut main, queries, accs): (RowMajorMatrix<Val<B::SC>>, Vec<u32>, _) = air::chain_trace_lookup(&air_, &vus, 2);
    assert_eq!(accs.len(), 2);
    let rows = main.height();
    assert!(rows >= air_.lay.tables.total_rows);
    let pre: RowMajorMatrix<Val<B::SC>> = air_.lay.tables.preprocessed(rows);
    mutate(&air_, &lay, &mut main);
    let width = main.width();
    let mult: Vec<Val<B::SC>> = (0..rows).map(|r| main.values[r * width + width - 1]).collect();
    let log_n = rows.trailing_zeros() as usize;
    let config = B::config(log_n, FRI);
    let llay = air_.lay.clone();
    let mut ch = B::challenger();
    let (proof, _) = prove_lookup(&config, &air_, &mut ch, pre.clone(), main, |z, beta| aux_matrix(&llay, &queries, &mult, rows, z, beta, true));
    let mut ch = B::challenger();
    verify_lookup(&config, &air_, &mut ch, &proof, pre).is_ok()
}

fn set<G: PrimeField64>(t: &mut RowMajorMatrix<G>, row: usize, col: usize, v: u64) {
    let w = t.width();
    t.values[row * w + col] = G::from_canonical_u64(v);
}

/// Pod-sized (2^17 rows for the tables): the honest round trip, the exhaustive evaluator over the committed
/// trace, and the chain mutations of `tests/chain.rs` plus lookup-specific ones.
#[test]
#[ignore]
fn lookup_babybear_roundtrip_and_mutations() {
    if !root().join("vu-k1536-neg.x.u16").is_file() {
        eprintln!("fixture missing; skipped");
        return;
    }
    type B = BabyBearBackend;
    let steps = STEPS as usize;
    // exhaustive evaluator on the honest committed trace
    {
        let (air_, lay) = LookupChainAir::new(FieldKind::BabyBear, 4);
        let vus = two_vus();
        let (main, queries, _): (RowMajorMatrix<Val32>, Vec<u32>, _) = air::chain_trace_lookup(&air_, &vus, 2);
        let rows = main.height();
        let pre: RowMajorMatrix<Val32> = air_.lay.tables.preprocessed(rows);
        let width = main.width();
        let mult: Vec<Val32> = (0..rows).map(|r| main.values[r * width + width - 1]).collect();
        let z = EF4::from_canonical_u32(123456789) + <EF4 as AbstractExtensionField<Val32>>::monomial(2) * Val32::from_canonical_u32(987);
        let beta = EF4::from_canonical_u32(55555) + <EF4 as AbstractExtensionField<Val32>>::monomial(1) * Val32::from_canonical_u32(31337);
        let aux = aux_matrix(&air_.lay, &queries, &mult, rows, z, beta, true);
        let fails = air::lookup_failing_trace(&air_, &lay, &main, &pre, &aux, z, beta);
        assert!(fails.is_empty(), "{:?}", &fails[..fails.len().min(5)]);
    }
    assert!(roundtrip::<B>(|_, _, _| {}), "honest chain rejected");
    // wrong claimed word on the last row of VU 0
    assert!(!roundtrip::<B>(|_, lay, t| {
        let c = lay.col(COL_CLAIM);
        let r = steps - 1;
        let w = t.width();
        let v = t.values[r * w + c].as_canonical_u64();
        set(t, r, c, v ^ 1);
    }));
    // broken accumulator link
    assert!(!roundtrip::<B>(|air_, _, t| {
        let c = air_.idx.c[air_.idx.c.len() - 1];
        let w = t.width();
        let v = t.values[w + c].as_canonical_u64();
        set(t, 1, c, v ^ 1);
    }));
    // c_0 != 0 at the VU boundary
    assert!(!roundtrip::<B>(|air_, _, t| {
        let c = air_.idx.c[air_.idx.c.len() - 1];
        set(t, steps, c, 1);
    }));
    // step counter skips a row
    assert!(!roundtrip::<B>(|_, lay, t| {
        let s = lay.col(COL_STEP);
        set(t, 5, s, 6);
    }));
    // lookup-specific: a range-checked hint pushed out of its table (the helper has no term for the new value)
    assert!(!roundtrip::<B>(|_, lay, t| {
        let c = lay.col("x.c_lo");
        set(t, 2, c, 70000);
    }));
    // lookup-specific: the multiplicity column tampered (the table side no longer balances)
    assert!(!roundtrip::<B>(|_, lay, t| {
        let c = lay.col("lookup.m");
        let w = t.width();
        let v = t.values[c].as_canonical_u64();
        set(t, 0, c, v + 1);
        let _ = w;
    }));
}
