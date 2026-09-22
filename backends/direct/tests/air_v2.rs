//! The checker-v2 AIR (PROTOCOL.md section 15, V2_AIR.md): layout consistency, the exhaustive evaluator (every
//! constraint incl. the global LogUp identities) on the honest committed trace and on mutations, and (pod-sized,
//! `--ignored` locally) the statement-bound prove / verify round trip.

use std::path::{Path, PathBuf};

use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractExtensionField, AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use verity_direct::air::VuIo;
use verity_direct::air_v2::{self, V2ChainAir};
use verity_direct::ctx::FieldKind;
use verity_direct::lookup::aux_matrix;
use verity_direct::program::{COL_CLAIM, STEPS};
use verity_direct::stark::{BabyBearBackend, Val32, DEFAULT_FRI};
use verity_direct::statement::Statement;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/bench-instances/v1")
}

fn read_u16(p: &Path) -> Vec<u64> {
    std::fs::read(p).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u64).collect()
}

/// Two honest VUs: the first two `vu-k1536-neg` instances (mistake `ulp-plus`) with their correct words.
fn two_vus() -> Vec<VuIo> {
    let k = 16 * STEPS as usize;
    let x = read_u16(&root().join("vu-k1536-neg.x.u16"));
    let w = read_u16(&root().join("vu-k1536-neg.w.u16"));
    let y = read_u16(&root().join("vu-k1536-neg.y.u16"));
    (0..2).map(|i| VuIo { x: x[i * k..(i + 1) * k].to_vec(), w: w[i * k..(i + 1) * k].to_vec(), y16: y[i] - 1 }).collect()
}

type EF4 = BinomialExtensionField<Val32, 4>;

fn have_fixture() -> bool {
    root().join("vu-k1536-neg.x.u16").is_file()
}

#[test]
fn v2_layout_consistent() {
    let (air_, lay) = V2ChainAir::new(FieldKind::BabyBear, 4);
    assert_eq!(lay.col_names.len(), lay.width);
    assert_eq!(air_.width, lay.width);
    assert!(lay.width + air_.lay.aux_width(4) + air_.lay.tables.pre_width() < 863, "committed width {} not below v1+LogUp", lay.width + air_.lay.aux_width(4) + air_.lay.tables.pre_width());
    // the same layout for BabyBear^5 (only the aux width changes)
    let (air5, lay5) = V2ChainAir::new(FieldKind::BabyBear, 5);
    assert_eq!(lay5.col_names, lay.col_names);
    assert_eq!(lay5.constraint_names, lay.constraint_names);
    assert_eq!(air5.lay.aux_width(5) - air_.lay.aux_width(4), air_.lay.n_helpers() + 2);
    assert_eq!(air_.lay.query_names.len(), air_.lay.n_queries());
    assert!(air_.lay.tables.total_rows <= 1 << 19, "table set {} rows", air_.lay.tables.total_rows);
}

fn set(t: &mut RowMajorMatrix<Val32>, row: usize, col: usize, v: u64) {
    let w = t.width();
    t.values[row * w + col] = Val32::from_canonical_u64(v);
}

/// Every constraint of the AIR (row program, tail, LogUp helpers, table side, running sum) on the committed trace.
fn exhaustive(air_: &V2ChainAir, lay: &verity_direct::air::Layout, main: &RowMajorMatrix<Val32>, queries: &[u32]) -> Vec<(usize, String)> {
    let rows = main.height();
    let width = main.width();
    let pre: RowMajorMatrix<Val32> = air_.lay.tables.preprocessed(rows);
    let mult: Vec<Val32> = (0..rows).map(|r| main.values[r * width + width - 1]).collect();
    let z = EF4::from_canonical_u32(123456789) + <EF4 as AbstractExtensionField<Val32>>::monomial(2) * Val32::from_canonical_u32(987);
    let beta = EF4::from_canonical_u32(55555) + <EF4 as AbstractExtensionField<Val32>>::monomial(1) * Val32::from_canonical_u32(31337);
    let aux = aux_matrix(&air_.lay, queries, &mult, rows, z, beta, false);
    air_v2::failing_trace(air_, lay, main, &pre, &aux, z, beta)
}

#[test]
fn v2_honest_trace_satisfies_every_constraint() {
    if !have_fixture() {
        eprintln!("fixture missing; skipped");
        return;
    }
    let (air_, lay) = V2ChainAir::new(FieldKind::BabyBear, 4);
    let vus = two_vus();
    let (main, queries, accs): (RowMajorMatrix<Val32>, Vec<u32>, _) = air_v2::trace(&air_, &vus, 2);
    assert_eq!(accs.len(), 2);
    assert!(main.height() >= air_.lay.tables.total_rows);
    let fails = exhaustive(&air_, &lay, &main, &queries);
    assert!(fails.is_empty(), "{:?}", &fails[..fails.len().min(5)]);
    // the claim column carries the word on the last row of each VU
    let c = lay.col(COL_CLAIM);
    let steps = STEPS as usize;
    for (i, vu) in vus.iter().enumerate() {
        assert_eq!(main.values[(i * steps + steps - 1) * main.width() + c].as_canonical_u64(), vu.y16);
    }
}

#[test]
fn v2_mutations_fail_a_constraint() {
    if !have_fixture() {
        eprintln!("fixture missing; skipped");
        return;
    }
    let (air_, lay) = V2ChainAir::new(FieldKind::BabyBear, 4);
    let vus = two_vus();
    let (honest, queries, _): (RowMajorMatrix<Val32>, Vec<u32>, _) = air_v2::trace(&air_, &vus, 2);
    let steps = STEPS as usize;
    let cases: Vec<(&str, Box<dyn Fn(&mut RowMajorMatrix<Val32>)>)> = vec![
        ("wrong claimed word", Box::new(|t| {
            let c = lay.col(COL_CLAIM);
            let v = t.values[(steps - 1) * t.width() + c].as_canonical_u64();
            set(t, steps - 1, c, v ^ 1);
        })),
        ("broken state link (M)", Box::new(|t| {
            let c = air_.idx.st[2];
            let v = t.values[5 * t.width() + c].as_canonical_u64();
            set(t, 5, c, (v + 1) % (1 << 24));
        })),
        ("operand outside T_OP (NaN)", Box::new(|t| {
            let c = lay.col("x.a[3]");
            set(t, 7, c, 0x7FC0);
        })),
        ("multiplicity off by one", Box::new(|t| {
            let m = t.width() - 1;
            let v = t.values[m].as_canonical_u64();
            set(t, 0, m, v + 1);
        })),
        ("state exponent out of range", Box::new(|t| {
            let c = air_.idx.st[1];
            set(t, 9, c, 200);
        })),
    ];
    for (name, mutate) in cases {
        let mut t = honest.clone();
        mutate(&mut t);
        let fails = exhaustive(&air_, &lay, &t, &queries);
        assert!(!fails.is_empty(), "{name}: no constraint fails");
        eprintln!("{name}: {} failing, first {:?}", fails.len(), fails[0]);
    }
}

/// The statement-bound round trip on two VUs (2^19 rows by the table set: pod-sized).
#[test]
#[ignore]
fn v2_statement_bound_roundtrip() {
    let (air_, lay) = V2ChainAir::new(FieldKind::BabyBear, 4);
    let vus = two_vus();
    let (main, queries, _): (RowMajorMatrix<Val32>, Vec<u32>, _) = air_v2::trace(&air_, &vus, 2);
    let stmt = Statement::new(vus.iter().map(|v| v.y16).collect());
    let (proof, _) = air_v2::prove_bound::<BabyBearBackend>(&air_, &lay, main, &queries, DEFAULT_FRI, &stmt);
    assert!(air_v2::verify_bound::<BabyBearBackend>(&air_, &lay, &proof, DEFAULT_FRI, &stmt).is_ok());
    let mut other = stmt.clone();
    other.words[1] ^= 1;
    assert!(air_v2::verify_bound::<BabyBearBackend>(&air_, &lay, &proof, DEFAULT_FRI, &other).is_err(), "bound to the wrong statement");
    let three = Statement::new(vec![stmt.words[0], stmt.words[1], stmt.words[0]]);
    assert!(air_v2::verify_bound::<BabyBearBackend>(&air_, &lay, &proof, DEFAULT_FRI, &three).is_err(), "bound to a statement of another size");
}

/// The vendored BabyBear^6 (`vendor/p3-baby-bear`, X^6 - 22): field axioms on random elements, the 2-adic
/// generator's order, Frobenius x^(p^6) = x, and the extension's relation to the base field.
#[test]
fn babybear6_field() {
    use p3_field::{Field, TwoAdicField};
    use rand::{Rng, SeedableRng};
    type EF6 = BinomialExtensionField<Val32, 6>;
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let rnd = |rng: &mut rand::rngs::StdRng| EF6::from_base_slice(&(0..6).map(|_| Val32::from_canonical_u32(rng.gen_range(0..2013265921))).collect::<Vec<_>>());
    let p: u64 = 2013265921;
    for _ in 0..50 {
        let (a, b, c) = (rnd(&mut rng), rnd(&mut rng), rnd(&mut rng));
        assert_eq!((a * b) * c, a * (b * c));
        assert_eq!(a * (b + c), a * b + a * c);
        assert_eq!(a * b, b * a);
        if a != EF6::zero() {
            assert_eq!(a * a.inverse(), EF6::one());
        }
        // Frobenius: x^p is an automorphism; x^(p^6) = x, and x^(p^k) != x generically for k < 6
        let mut x = a;
        for k in 1..=6 {
            x = x.exp_u64(p);
            if k < 6 {
                assert_ne!(x, a, "x^(p^{k}) = x: the extension would be smaller than degree 6");
            }
        }
        assert_eq!(x, a);
    }
    // 2-adicity 28: g has order exactly 2^28
    let g = EF6::two_adic_generator(28);
    assert_eq!(g.exp_power_of_2(28), EF6::one());
    assert_ne!(g.exp_power_of_2(27), EF6::one());
    for bits in [0usize, 1, 5, 27] {
        let h = EF6::two_adic_generator(bits);
        assert_eq!(h.exp_power_of_2(bits), EF6::one());
        if bits > 0 {
            assert_ne!(h.exp_power_of_2(bits - 1), EF6::one());
        }
    }
    // X^6 = 22
    let x = EF6::from_base_slice(&[Val32::zero(), Val32::one(), Val32::zero(), Val32::zero(), Val32::zero(), Val32::zero()]);
    assert_eq!(x.exp_u64(6), EF6::from_canonical_u32(22));
    // the multiplicative generator has order p^6 - 1: not 1 at (p^6-1)/q for the two small prime factors we can reach
    let gen = EF6::generator();
    let n_over_2: u128 = ((p as u128).pow(6) - 1) / 2;
    let mut acc = EF6::one();
    let mut base = gen;
    let mut e = n_over_2;
    while e > 0 {
        if e & 1 == 1 {
            acc *= base;
        }
        base = base.square();
        e >>= 1;
    }
    assert_ne!(acc, EF6::one());
}
