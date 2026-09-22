//! Soundness by mutation, mirroring `backends/numerical/tests/checker/test_negative.py` case for case: each
//! mutation overrides named hints and regenerates everything downstream honestly (limbs, selectors, copies),
//! exactly as the Python checker recomputes its expressions from mutated hints; the AIR must then have a failing
//! constraint whose name contains the Python predicate's name.  A sample of the mutations is also driven through
//! the real prover: the proof of a mutated trace must be rejected by `verify`.
//!
//! Not mirrored (outside the transition unit): the epilogue tests (VU only) and the misshapen-witness tests (a
//! fixed layout has no shape to break).

use verity_direct::air::{self, failing_constraints, gen_row, layout, to_f, Layout, UnitAir};
use verity_direct::params::{FLOOR, LAST};
use verity_direct::program::UnitIo;
use verity_direct::stark::{self, DEFAULT_FRI};

const fn bf16(s: u64, e: u64, f: u64) -> u64 {
    (s << 15) | (e << 7) | f
}
fn f32w(s: u64, e: u64, f: u64) -> u64 {
    (s << 31) | (e << 23) | f
}
const ONE: u64 = bf16(0, 127, 0);

fn case(c: u64, pairs: &[(u64, u64)]) -> UnitIo {
    let mut a: Vec<u64> = pairs.iter().map(|p| p.0).collect();
    let mut b: Vec<u64> = pairs.iter().map(|p| p.1).collect();
    a.resize(16, 0);
    b.resize(16, 0);
    let y = verity_direct::step_output(c, &a, &b);
    UnitIo { c, a, b, y }
}

struct Fixture {
    lay: Layout,
}

impl Fixture {
    fn new() -> Self {
        Self { lay: layout() }
    }
    fn honest(&self, io: &UnitIo) -> Vec<i128> {
        let mut ctx = verity_direct::ctx::IntCtx::new();
        verity_direct::program::unit_program(&mut ctx, io);
        assert!(ctx.violations.is_empty(), "{:?}", ctx.violations);
        ctx.cols
    }
    fn value(&self, io: &UnitIo, name: &str) -> i128 {
        self.honest(io)[self.lay.col(name)]
    }
    /// `deltas`: (hint name, added value); `sets`: (hint name, new value)
    fn rejects(&self, io: &UnitIo, deltas: &[(&str, i128)], sets: &[(&str, i128)], predicate: &str) {
        let honest_row: Vec<_> = self.honest(io).iter().map(|&v| to_f(v)).collect();
        assert!(failing_constraints(&self.lay, &honest_row).is_empty(), "honest witness rejected");
        let mut overrides: Vec<(&str, i128)> = Vec::new();
        for (name, d) in deltas {
            overrides.push((name, self.value(io, name) + d));
        }
        for (name, v) in sets {
            overrides.push((name, *v));
        }
        let (row, _int_violations) = gen_row(io, &overrides);
        let failing = failing_constraints(&self.lay, &row);
        assert!(
            failing.iter().any(|f| f.contains(predicate)),
            "mutation {:?} on predicate {predicate}: failing = {:?}",
            overrides,
            failing
        );
    }
}

fn nonzero() -> UnitIo {
    case(f32w(0, 130, 0x123456), &[(bf16(0, 128, 3), bf16(1, 126, 5)); 16])
}
fn with_zero_product() -> UnitIo {
    let mut pairs = vec![(ONE, ONE), (0, ONE)];
    pairs.extend(std::iter::repeat((ONE, ONE)).take(14));
    case(f32w(0, 130, 0), &pairs)
}
fn subnormal_state() -> UnitIo {
    case(f32w(0, 0, 0x2A), &[(ONE, bf16(0, 20, 0))])
}
fn cancel() -> UnitIo {
    let pairs: Vec<_> = (0..8).flat_map(|_| [(ONE, ONE), (ONE, bf16(1, 127, 0))]).collect();
    case(0, &pairs)
}
fn shifted() -> UnitIo {
    case(f32w(0, 140, 0), &[(bf16(0, 121, 0x55), bf16(0, 121, 0x33)); 16])
}
fn wide() -> UnitIo {
    case(f32w(0, 130, 0x7FFFFF), &[(bf16(0, 128, 0x7F), bf16(0, 129, 0x7F)); 16])
}
fn flush_state() -> UnitIo {
    case(f32w(0, 10, 0x7), &[(bf16(0, 200, 1), bf16(0, 200, 1))])
}
fn overflow() -> UnitIo {
    case(0, &[(bf16(0, 254, 0x7F), bf16(0, 254, 0x7F)); 8])
}

#[test]
fn params_match_python_real() {
    verity_direct::params::check_params();
}

#[test]
fn honest_cases_are_accepted() {
    let fx = Fixture::new();
    for io in [nonzero(), with_zero_product(), subnormal_state(), cancel(), shifted(), wide(), flush_state(), overflow()] {
        let row: Vec<_> = fx.honest(&io).iter().map(|&v| to_f(v)).collect();
        assert!(failing_constraints(&fx.lay, &row).is_empty());
    }
    assert_eq!(overflow().y, 0x7F80_0000);
}

// ---- bf16 decode ----------------------------------------------------------------------------------------------------

#[test]
fn bf16_decode_wrapped_reconstruction_is_rejected() {
    Fixture::new().rejects(&nonzero(), &[("a[0].f", 128), ("a[0].t", -1)], &[], "a[0].f");
}

#[test]
fn bf16_decode_sign_flip_is_rejected() {
    let fx = Fixture::new();
    let io = nonzero();
    let s = fx.value(&io, "b[0].s");
    fx.rejects(&io, &[], &[("b[0].s", s ^ 1)], "b[0].word");
}

#[test]
fn bf16_decode_false_nonzero_flag_is_rejected_both_ways() {
    let fx = Fixture::new();
    fx.rejects(&nonzero(), &[], &[("a[0].nz", 0)], "a[0].nz0");
    fx.rejects(&with_zero_product(), &[], &[("a[1].nz", 1)], "a[1].nz1");
}

#[test]
fn bf16_decode_rejects_non_finite_operands() {
    let fx = Fixture::new();
    let io = nonzero();
    for word in [0x7F80u64, 0xFF80, 0x7FC0, 0x7F81] {
        // the word changes, the decode hints are regenerated for it (t = 255), y stays: as gen_step_unchecked
        fx.rejects(&io, &[], &[("x.a[0]", word as i128)], "a[0].t");
    }
}

// ---- fp32 decode ----------------------------------------------------------------------------------------------------

#[test]
fn fp32_decode_wrapped_reconstruction_and_flags_are_rejected() {
    let fx = Fixture::new();
    fx.rejects(&nonzero(), &[("acc.f", 1 << 23), ("acc.t", -1)], &[], "acc.f");
    let s = fx.value(&nonzero(), "acc.s");
    fx.rejects(&nonzero(), &[], &[("acc.s", s ^ 1)], "acc.word");
    fx.rejects(&nonzero(), &[], &[("acc.z", 1)], "acc.z0");
    fx.rejects(&cancel(), &[], &[("acc.z", 0)], "acc.z1");
    fx.rejects(&subnormal_state(), &[], &[("acc.nz", 1)], "acc.nz1");
}

#[test]
fn fp32_decode_rejects_an_infinite_accumulator() {
    Fixture::new().rejects(&nonzero(), &[], &[("x.c", 0x7F80_0000)], "acc.t");
}

// ---- product --------------------------------------------------------------------------------------------------------

#[test]
fn product_zero_flag_is_pinned() {
    let fx = Fixture::new();
    fx.rejects(&nonzero(), &[], &[("prod[0].z", 1)], "prod[0].z0");
    fx.rejects(&with_zero_product(), &[], &[("prod[1].z", 0)], "prod[1].z1");
}

// ---- group maximum --------------------------------------------------------------------------------------------------

#[test]
fn group_max_excessive_exponent_is_rejected() {
    let fx = Fixture::new();
    fx.rejects(&nonzero(), &[("g0.max.E", 1)], &[], "g0.max.attained");
    let mut sets: Vec<(&str, i128)> = (0..9).map(|i| (Box::leak(format!("g1.max.sel[{i}]").into_boxed_str()) as &str, 0)).collect();
    sets.push(("g1.max.sel[9]", 1));
    fx.rejects(&nonzero(), &[("g1.max.E", 3)], &sets, "g1.max.attained");
}

#[test]
fn group_max_false_selector_is_rejected() {
    let fx = Fixture::new();
    let io = shifted();
    assert_eq!(fx.value(&io, "g0.max.sel[0]"), 1);
    let mut sets: Vec<(&str, i128)> = (0..10).map(|i| (Box::leak(format!("g0.max.sel[{i}]").into_boxed_str()) as &str, 0)).collect();
    sets[1].1 = 1;
    fx.rejects(&io, &[("g0.max.E", -25)], &sets, "g0.max.d[0]");
    fx.rejects(&io, &[], &[("g0.max.sel[1]", 1)], "g0.max.onehot");
    fx.rejects(&io, &[], &[("g0.max.E", FLOOR - 1)], "g0.max.E");
}

// ---- alignment ------------------------------------------------------------------------------------------------------

#[test]
fn align_state_wrong_remainder_is_rejected() {
    let fx = Fixture::new();
    let io = shifted();
    assert_eq!(fx.value(&io, "g0.align_state.r"), 0);
    fx.rejects(&io, &[("g0.align_state.r", 1)], &[], "g0.align_state.div");
    fx.rejects(&io, &[("g0.align_state.q", -1), ("g0.align_state.r", 1)], &[], "g0.align_state.rem");
}

#[test]
fn align_state_flushed_state_cannot_claim_a_quotient() {
    let fx = Fixture::new();
    let io = flush_state();
    let d = fx.value(&io, "g0.max.E") - (10 - 127);
    assert!(d >= 25);
    assert_eq!(fx.value(&io, "g0.align_state.q"), 0);
    fx.rejects(&io, &[("g0.align_state.r", -(1 << 25))], &[("g0.align_state.q", 1)], "g0.align_state.r");
    fx.rejects(&io, &[], &[("g0.align_state.q", 1)], "g0.align_state.div");
}

#[test]
fn align_product_wrong_remainder_and_wrapped_quotient_are_rejected() {
    let fx = Fixture::new();
    let io = shifted();
    let d = fx.value(&io, "g0.max.E") - 2 * (121 - 127);
    assert!(10 < d && d < 26);
    let sr = 1i128 << (d - 10);
    assert!(fx.value(&io, "g0.align_prod[0].r") < sr);
    fx.rejects(&io, &[("g0.align_prod[0].r", 1)], &[], "g0.align_prod[0].div");
    fx.rejects(&io, &[("g0.align_prod[0].q", 1), ("g0.align_prod[0].r", -sr)], &[], "g0.align_prod[0].r");
    fx.rejects(&io, &[("g0.align_prod[0].q", -1), ("g0.align_prod[0].r", sr)], &[], "g0.align_prod[0].rem");
}

#[test]
fn align_product_unshifted_product_has_no_remainder() {
    let fx = Fixture::new();
    let io = nonzero();
    assert_eq!(fx.value(&io, "g0.align_prod[0].r"), 0);
    fx.rejects(&io, &[("g0.align_prod[0].q", -1), ("g0.align_prod[0].r", 1)], &[], "g0.align_prod[0].rem");
}

// ---- signed sum -----------------------------------------------------------------------------------------------------

#[test]
fn signed_sum_sign_flip_and_zero_flag_are_rejected() {
    let fx = Fixture::new();
    let s = fx.value(&nonzero(), "g0.sum.sgn");
    fx.rejects(&nonzero(), &[], &[("g0.sum.sgn", s ^ 1)], "g0.sum.v");
    fx.rejects(&nonzero(), &[], &[("g0.sum.z", 1)], "g0.sum.z0");
    fx.rejects(&cancel(), &[], &[("g0.sum.z", 0)], "g0.sum.z1");
    fx.rejects(&cancel(), &[], &[("g0.sum.sgn", 1)], "g0.sum.zero_sign");
}

// ---- leading bit ----------------------------------------------------------------------------------------------------

#[test]
fn leading_bit_off_by_one_is_rejected() {
    let fx = Fixture::new();
    for delta in [-1i128, 1] {
        fx.rejects(&nonzero(), &[("g0.lead.l", delta)], &[], "g0.lead.window");
    }
    fx.rejects(&cancel(), &[], &[("g0.lead.l", 1)], "g0.lead.window");
}

// ---- normalisation --------------------------------------------------------------------------------------------------

#[test]
fn normalise_clamp_flag_and_quotient_are_pinned() {
    let fx = Fixture::new();
    for io in [nonzero(), subnormal_state()] {
        let c = fx.value(&io, "g0.norm.clamp");
        fx.rejects(&io, &[], &[("g0.norm.clamp", c ^ 1)], "g0.norm.clamp_proof");
    }
    let io = wide();
    let (e, l) = (fx.value(&io, "g0.max.E"), fx.value(&io, "g0.lead.l"));
    assert_eq!(l, 30);
    let k = e - (e + l - 25).max(-126) + 24 - 25;
    assert!(k < 0);
    let div = 1i128 << -k;
    fx.rejects(&io, &[("g0.norm.M", 1)], &[], "g0.norm.div");
    fx.rejects(&nonzero(), &[("g0.norm.M", 1)], &[], "g0.norm.div");
    fx.rejects(&io, &[("g0.norm.M", -1), ("g0.norm.r", div)], &[], "g0.norm.rem");
    fx.rejects(&nonzero(), &[("g0.norm.M", -1), ("g0.norm.r", 1)], &[], "g0.norm.rem");
}

// ---- state out ------------------------------------------------------------------------------------------------------

#[test]
fn state_out_flags_are_pinned() {
    let fx = Fixture::new();
    fx.rejects(&nonzero(), &[], &[("g1.out.z", 1)], "g1.out.z0");
    fx.rejects(&cancel(), &[], &[("g1.out.z", 0)], "g1.out.z1");
    fx.rejects(&nonzero(), &[], &[("g1.out.ovf", 1)], "g1.out.ovf_proof");
    fx.rejects(&overflow(), &[], &[("g0.out.ovf", 0)], "g0.out.ovf_proof");
}

// ---- pack and the output --------------------------------------------------------------------------------------------

#[test]
fn pack_wrapped_reconstruction_and_output_sign_flip_are_rejected() {
    let fx = Fixture::new();
    let io = nonzero();
    let hidden = fx.value(&io, "pack.hidden");
    let low_delta = if hidden ^ 1 == 0 { 1i128 << 23 } else { -(1i128 << 23) };
    fx.rejects(&io, &[("pack.low", low_delta)], &[("pack.hidden", hidden ^ 1)], "pack.low");
    fx.rejects(&io, &[], &[("pack.hidden", hidden ^ 1)], "pack.M");
    fx.rejects(&io, &[], &[("y", (io.y ^ (1 << 31)) as i128)], "y");
    fx.rejects(&io, &[], &[("y", io.y as i128 + 1)], "y");
}

#[test]
fn overflow_output_is_the_infinity_and_nothing_else() {
    let fx = Fixture::new();
    let io = overflow();
    assert_eq!(io.y, 0x7F80_0000);
    for other in [0x7F7F_FFFFi128, 0xFF80_0000, 0] {
        fx.rejects(&io, &[], &[("y", other)], "y");
    }
}

// ---- every hint is range-checked against the contract ------------------------------------------------------------

#[test]
fn every_hint_is_range_checked_against_the_contract() {
    let fx = Fixture::new();
    let io = nonzero();
    let bit = (0i128, 2i128);
    let places: Vec<(&str, (i128, i128))> = vec![
        ("acc.s", bit), ("acc.t", (0, 255)), ("acc.f", (0, 1 << 23)), ("acc.nz", bit), ("acc.z", bit),
        ("a[3].s", bit), ("a[3].t", (0, 255)), ("a[3].f", (0, 1 << 7)), ("a[3].nz", bit),
        ("prod[3].z", bit),
        ("g1.align_state.q", (0, 1 << 25)), ("g1.align_state.r", (0, 1 << 25)),
        ("g1.align_prod[2].q", (0, 1 << 16)), ("g1.align_prod[2].r", (0, 1 << 16)),
        ("g1.sum.sgn", bit), ("g1.sum.z", bit),
        ("g1.lead.l", (0, 31)),
        ("g1.norm.clamp", bit), ("g1.norm.M", (0, 1 << 24)), ("g1.norm.r", (0, 1 << 7)),
        ("g1.out.z", bit), ("g1.out.ovf", bit),
        ("pack.hidden", bit), ("pack.low", (0, 1 << 23)),
        ("g0.max.E", (FLOOR, LAST + 1)),
    ];
    let honest: Vec<_> = fx.honest(&io);
    for (name, (lo, hi)) in &places {
        for value in [lo - 1, *hi] {
            let (row, _) = gen_row(&io, &[(name, value)]);
            let failing = failing_constraints(&fx.lay, &row);
            assert!(!failing.is_empty(), "{name} = {value} accepted");
        }
    }
    for i in 0..10 {
        let name = format!("g0.max.sel[{i}]");
        for value in [-1i128, 2] {
            let (row, _) = gen_row(&io, &[(&name, value)]);
            assert!(!failing_constraints(&fx.lay, &row).is_empty(), "{name} = {value} accepted");
        }
    }
    let _ = honest;
}

// ---- through the real prover ----------------------------------------------------------------------------------------

/// A sample of the mutations, driven through `prove` + `verify`: the proof of a mutated trace is rejected.
#[test]
fn mutated_traces_do_not_verify() {
    let fx = Fixture::new();
    let air_ = UnitAir { width: fx.lay.width };
    let honest = nonzero();
    let honest_trace = air::trace(&[honest.clone()], 2);
    let proof = stark::prove_trace(&air_, honest_trace, DEFAULT_FRI);
    assert!(stark::verify_proof(&air_, &proof, 1, DEFAULT_FRI).is_ok());

    let mutations: Vec<(UnitIo, Vec<(&str, i128)>)> = vec![
        (nonzero(), vec![("g0.max.E", fx.value(&nonzero(), "g0.max.E") + 1)]),
        (nonzero(), vec![("y", (nonzero().y ^ (1 << 31)) as i128)]),
        (shifted(), vec![("g0.align_state.q", fx.value(&shifted(), "g0.align_state.q") - 1), ("g0.align_state.r", 1)]),
        (cancel(), vec![("g0.sum.sgn", 1)]),
        (nonzero(), vec![("g0.norm.M", fx.value(&nonzero(), "g0.norm.M") + 1)]),
    ];
    for (io, overrides) in mutations {
        let (row, _) = gen_row(&io, &overrides);
        assert!(!failing_constraints(&fx.lay, &row).is_empty());
        let mut values = row.clone();
        values.extend_from_slice(&row);
        let trace = p3_matrix::dense::RowMajorMatrix::new(values, fx.lay.width);
        let proof = stark::prove_trace(&air_, trace, DEFAULT_FRI);
        let res = stark::verify_proof(&air_, &proof, 1, DEFAULT_FRI);
        assert!(res.is_err(), "mutation {:?} verified", overrides);
    }
}

/// Reproducer for the finding that motivated MAX_DEG = 2: with MAX_DEG = 5 and 2-bit limbs, the rows mutated
/// below had a violated degree-3 (`attained`) or degree-4 (limb) constraint and still produced verifying proofs,
/// while degree-<=2 violations were rejected.  Run with `--ignored --nocapture` after setting MAX_DEG = 5 and
/// LIMB_BITS = 2 to reproduce; at MAX_DEG = 2 every line must print a verification error.
#[test]
#[ignore]
fn reproducer_degree_ge3_constraints_unenforced() {
    let fx = Fixture::new();
    let air_ = UnitAir { width: fx.lay.width };
    let u = nonzero();
    let muts: Vec<(&str, Vec<(&str, i128)>)> = vec![
        ("E+1", vec![("g0.max.E", fx.value(&u, "g0.max.E") + 1)]),
        ("y+1", vec![("y", u.y as i128 + 1)]),
        ("acc.s flip", vec![("acc.s", fx.value(&u, "acc.s") ^ 1)]),
        ("a0.f+128", vec![("a[0].f", fx.value(&u, "a[0].f") + 128)]),
        ("sum.sgn flip", vec![("g0.sum.sgn", fx.value(&u, "g0.sum.sgn") ^ 1)]),
        ("limb4 only", vec![("acc.f.limb[0]", fx.value(&u, "acc.f.limb[0]") + 4), ("acc.f.limb[1]", fx.value(&u, "acc.f.limb[1]") - 1)]),
        ("sel swap (deg3 only)", vec![("g0.max.sel[0]", 0), ("g0.max.sel[1]", 1)]),
    ];
    for (label, ov) in muts {
        let (row, _) = gen_row(&u, &ov);
        let failing = failing_constraints(&fx.lay, &row);
        let mut values = row.clone();
        values.extend_from_slice(&row);
        let trace = p3_matrix::dense::RowMajorMatrix::new(values, fx.lay.width);
        let proof = stark::prove_trace(&air_, trace, DEFAULT_FRI);
        let res = stark::verify_proof(&air_, &proof, 1, DEFAULT_FRI);
        println!("EXPERIMENT {label}: failing={:?} verify={:?}", failing, res.err());
        let g = |n: &str| verity_direct::air::f_to_u64(row[fx.lay.col(n)]) as i128;
        let sels: Vec<i128> = (0..10).map(|i| g(&format!("g0.max.sel[{i}]"))).collect();
        println!("  E={} sels={:?} acc.t={} acc.nz={} acc.z={} prod0: a.t={} a.nz={} b.t={} b.nz={} z={}", g("g0.max.E"), sels, g("acc.t"), g("acc.nz"), g("acc.z"), g("a[0].t"), g("a[0].nz"), g("b[0].t"), g("b[0].nz"), g("prod[0].z"));
    }
    println!("EXPERIMENT max_constraint_degree={} log_quotient_degree={}", p3_uni_stark::get_max_constraint_degree::<verity_direct::air::F, UnitAir>(&air_, 0, 0), stark::log_quotient_degree(&air_));
}
