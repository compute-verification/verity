//! Checker v2 (`verity_numerical.checker.v2`, `checker_min.md`) as a straight-line program over `Ctx`, in the
//! BabyBear LogUp realisation of `V2_AIR.md` / PROTOCOL.md 15: tuple lookups, chunked shift tables, three
//! 10-bit chunks of the sum magnitude, the product form of `attained`, the FP32 word as two 16-bit limbs.  The
//! relation is the frozen one; every difference from the Python gadgets is an embedding (V2_AIR.md 4) and is
//! checked by `redteam/z3_air_v2.py` (`b_air_v2_z3.json`: 7/7 UNSAT over Z and BabyBear).
//!
//! One row = one transition unit on the chained state `(s, e_hat, M)`; `chain_program_v2` returns the incoming
//! state columns and the outgoing state expressions, which `air_v2::V2ChainAir` links row to row.

use std::sync::Arc;

use crate::lookup::TupleTable;
use crate::params::*;
use crate::program::{epilogue, Ctx, Word32, COL_CLAIM, COL_FIRST, COL_INV, COL_LAST, COL_STEP, STEPS};

// --- derived widths (`checker/v2/params.py::V2`, at REAL) -------------------------------------------------------

/// Exponent of a zero operand in T_OP (never attains a group maximum).
pub const E_ZERO: i128 = FLOOR - E_MAX - 1;
/// Smallest product exponent (two zero operands).
pub const CAND_LO: i128 = 2 * E_ZERO;
/// `d = E - cand in [0, SHIFT_ROWS_V2)`.
pub const SHIFT_ROWS_V2: i128 = LAST - CAND_LO + 1;
/// `d~ = min(d, DT)`: every shifted term is 0 once `d >= DT`.
pub const DT: i128 = if PROD_BITS > ACC_BITS { PROD_BITS } else { ACC_BITS };
/// `dsp = 2 d~ + sp in [0, DSP_ROWS)`.
pub const DSP_ROWS: i128 = 2 * (DT + 1);
/// `koff = k - K_LO in [0, K_ROWS)`.
pub const K_ROWS: i128 = K_HI - K_LO + 1;
/// Chunk width of the sum magnitude and of the LEAD10 / TNORM10 tables.
pub const CHUNK: i128 = 10;
/// Byte chunks of the shift tables.
pub const BYTE: i128 = 8;
/// `A1 = 2^(CHUNK + k)` is clamped to 0 above this `k` (those rows have `l <= CHUNK`, so the high chunks are 0).
pub const A1_K_MAX: i128 = 13;

pub fn check_params_v2() {
    assert_eq!(E_ZERO, -260);
    assert_eq!(CAND_LO, -520);
    assert_eq!(SHIFT_ROWS_V2, 780);
    assert_eq!(DT, 26);
    assert_eq!(DSP_ROWS, 54);
    assert_eq!(K_ROWS, 32);
    assert!(3 * CHUNK >= S_BITS);
    // A1 fits the field: 2^(10 + 13) = 2^23; and k > 13 => acc_sig - l <= k means l <= 10: v2 = v1 = 0
    assert!(ACC_SIG - A1_K_MAX <= CHUNK + 1);
}

// --- tables -------------------------------------------------------------------------------------------------------

fn tt(name: &'static str, keys: Vec<(i128, i128)>, n_out: usize, f: impl Fn(&[i128]) -> Option<Vec<i128>> + Send + Sync + 'static) -> TupleTable {
    TupleTable { name, keys, n_out, f: Arc::new(f) }
}

/// `T_OP: (w; s, m, e)`; non-finite words are padding rows.
pub fn t_op() -> TupleTable {
    tt("T_OP", vec![(0, 1 << WORD_BITS)], 3, |k| {
        let w = k[0];
        let (s, t, f) = (w >> (WORD_BITS - 1), (w >> FRAC) & ((1 << EXP) - 1), w & ((1 << FRAC) - 1));
        if t == T_FINITE {
            return None;
        }
        let m = f + if t != 0 { 1 << FRAC } else { 0 };
        let e = if m == 0 { E_ZERO } else { t.max(1) - BIAS };
        Some(vec![s, m, e])
    })
}

/// `ALIGN4: (d, s4; dsp)`, `dsp = 2 min(d, DT) + [s4 in {1, 2}]`.
pub fn align4() -> TupleTable {
    tt("ALIGN4", vec![(0, SHIFT_ROWS_V2), (0, 4)], 1, |k| Some(vec![2 * k[0].min(DT) + (k[1] == 1 || k[1] == 2) as i128]))
}

fn shift_term(x: i128, scale: i128, dsp: i128) -> i128 {
    let (dt, s) = (dsp / 2, dsp % 2);
    let v = (x << scale) >> dt;
    if s == 1 { -v } else { v }
}

/// `(x_c, dsp; +-floor(x_c 2^scale / 2^dt))` on a byte chunk.
pub fn shift_table(name: &'static str, scale: i128) -> TupleTable {
    tt(name, vec![(0, 1 << BYTE), (0, DSP_ROWS)], 1, move |k| Some(vec![shift_term(k[0], scale, k[1])]))
}

pub fn shift_hi() -> TupleTable {
    shift_table("SHIFT_HI", PROD_SHIFT + BYTE)
}
pub fn shift_lo() -> TupleTable {
    shift_table("SHIFT_LO", PROD_SHIFT)
}
pub fn sshift_2() -> TupleTable {
    shift_table("SSHIFT_2", ACC_SHIFT + 2 * BYTE)
}
pub fn sshift_1() -> TupleTable {
    shift_table("SSHIFT_1", ACC_SHIFT + BYTE)
}
pub fn sshift_0() -> TupleTable {
    shift_table("SSHIFT_0", ACC_SHIFT)
}

/// `LEAD10: (x; nz, l)`, `x < 2^10`.
pub fn lead10() -> TupleTable {
    tt("LEAD10", vec![(0, 1 << CHUNK)], 2, |k| Some(vec![(k[0] != 0) as i128, 128 - (k[0] as u128).leading_zeros() as i128]))
}

/// `LEADNORM: (l, E; koff, A1, e_hat, z, ovf)`.
pub fn leadnorm() -> TupleTable {
    tt("LEADNORM", vec![(0, L_HI + 1), (FLOOR, LAST + 1)], 5, |k| {
        let (l, e) = (k[0], k[1]);
        let cl = ACC_SIG - WIDTH;
        let kk = (ACC_SIG - l).min(e - E_MIN + cl);
        let e1 = e - kk + cl;
        let z = l == 0 || l + kk <= 0;
        let a1 = if kk <= A1_K_MAX { 1i128 << (CHUNK + kk) } else { 0 };
        Some(vec![kk - K_LO, a1, if z { FLOOR } else { e1 }, z as i128, (!z && e1 > E_MAX) as i128])
    })
}

/// `TNORM10: (v0, koff; floor(v0 2^k))`, clamped to 0 where the value would reach `2^ACC_SIG`.
pub fn tnorm10() -> TupleTable {
    tt("TNORM10", vec![(0, 1 << CHUNK), (0, K_ROWS)], 1, |k| {
        let kk = k[1] + K_LO;
        let t = if kk >= 0 { k[0] << kk } else { k[0] >> -kk };
        Some(vec![if t < 1 << ACC_SIG { t } else { 0 }])
    })
}

/// `T_HDR: (hdr; s, nz, e)` for the word-in unit's `state_decode` (gadget battery only).
pub fn t_hdr() -> TupleTable {
    tt("T_HDR", vec![(0, 1 << (EXP + 1))], 3, |k| {
        let (s, t) = (k[0] >> EXP, k[0] & ((1 << EXP) - 1));
        if t == T_FINITE {
            return None;
        }
        Some(vec![s, (t != 0) as i128, t.max(1) - BIAS])
    })
}

// --- gadgets --------------------------------------------------------------------------------------------------------

/// `+-floor(u 2^PROD_SHIFT / 2^dt)` on the bytes of `u = m_a m_b < 2^16`.
fn product_shift<C: Ctx>(ctx: &mut C, u: &C::V, dsp: &C::V, tag: &str) -> C::V {
    let hi = ctx.hint(&format!("{tag}.u_hi"), &[u], &|v| v[0] >> BYTE);
    let lo = ctx.hint(&format!("{tag}.u_lo"), &[u], &|v| v[0] & ((1 << BYTE) - 1));
    let hs = ctx.scale(&hi, 1 << BYTE);
    let rec = ctx.add(&hs, &lo);
    ctx.eq(&format!("{tag}.split"), &rec, u);
    let t_hi = ctx.lookup_tuple(&format!("{tag}.hi"), &shift_hi(), &[hi, dsp.clone()]);
    let t_lo = ctx.lookup_tuple(&format!("{tag}.lo"), &shift_lo(), &[lo, dsp.clone()]);
    ctx.add(&t_hi[0], &t_lo[0])
}

/// `+-floor(M 2^ACC_SHIFT / 2^dt)` on the three bytes of `M < 2^24`.
fn state_shift<C: Ctx>(ctx: &mut C, m: &C::V, dsp: &C::V, tag: &str) -> C::V {
    let m2 = ctx.hint(&format!("{tag}.M2"), &[m], &|v| v[0] >> (2 * BYTE));
    let m1 = ctx.hint(&format!("{tag}.M1"), &[m], &|v| (v[0] >> BYTE) & ((1 << BYTE) - 1));
    let m0 = ctx.hint(&format!("{tag}.M0"), &[m], &|v| v[0] & ((1 << BYTE) - 1));
    let a = ctx.scale(&m2, 1 << (2 * BYTE));
    let b = ctx.scale(&m1, 1 << BYTE);
    let rec = ctx.sum(&[a, b, m0.clone()]);
    ctx.eq(&format!("{tag}.split"), &rec, m);
    let t2 = ctx.lookup_tuple(&format!("{tag}.t2"), &sshift_2(), &[m2, dsp.clone()]);
    let t1 = ctx.lookup_tuple(&format!("{tag}.t1"), &sshift_1(), &[m1, dsp.clone()]);
    let t0 = ctx.lookup_tuple(&format!("{tag}.t0"), &sshift_0(), &[m0, dsp.clone()]);
    ctx.sum(&[t2[0].clone(), t1[0].clone(), t0[0].clone()])
}

/// The magnitude chunks of a value already known to be `< 2^30` (`|S| <= S_MAX`): hints `v2, v1, v0`, the split
/// identity, the three LEAD10 lookups; returns `(v2, v1, v0, l)` with `l` the materialised bit length.
fn chunks_lead<C: Ctx>(ctx: &mut C, v: &C::V, tag: &str) -> (C::V, C::V, C::V, C::V) {
    let mask = (1i128 << CHUNK) - 1;
    let v2 = ctx.hint(&format!("{tag}.v2"), &[v], &move |x| x[0] >> (2 * CHUNK));
    let v1 = ctx.hint(&format!("{tag}.v1"), &[v], &move |x| (x[0] >> CHUNK) & mask);
    let v0 = ctx.hint(&format!("{tag}.v0"), &[v], &move |x| x[0] & mask);
    let a = ctx.scale(&v2, 1 << (2 * CHUNK));
    let b = ctx.scale(&v1, 1 << CHUNK);
    let rec = ctx.sum(&[a, b, v0.clone()]);
    ctx.eq(&format!("{tag}.split"), &rec, v);
    let l2 = ctx.lookup_tuple(&format!("{tag}.lead2"), &lead10(), &[v2.clone()]);
    let l1 = ctx.lookup_tuple(&format!("{tag}.lead1"), &lead10(), &[v1.clone()]);
    let l0 = ctx.lookup_tuple(&format!("{tag}.lead0"), &lead10(), &[v0.clone()]);
    let (nz2, lb2, nz1, lb1, lb0) = (&l2[0], &l2[1], &l1[0], &l1[1], &l0[1]);
    // l = nz2 (20 + l2) + (1 - nz2) (nz1 (10 + l1) + (1 - nz1) l0)
    let one_m_nz1 = ctx.ksub(1, nz1);
    let t0 = ctx.mul(&one_m_nz1, lb0);
    let l1p = ctx.addk(lb1, CHUNK);
    let t1 = ctx.mul(nz1, &l1p);
    let inner = ctx.add(&t1, &t0);
    let one_m_nz2 = ctx.ksub(1, nz2);
    let outer = ctx.mul(&one_m_nz2, &inner);
    let l2p = ctx.addk(lb2, 2 * CHUNK);
    let top = ctx.mul(nz2, &l2p);
    let l = ctx.add(&top, &outer);
    let l = ctx.mat(&format!("{tag}.l"), &l);
    (v2, v1, v0, l)
}

/// Signed sum of the terms: `sgn`, the chunks of `|S|`, the bit length.
fn signed_sum<C: Ctx>(ctx: &mut C, terms: &[C::V], tag: &str) -> (C::V, C::V, C::V, C::V, C::V) {
    let s = ctx.sum(terms);
    let sgn = ctx.hint(&format!("{tag}.sgn"), &[&s], &|v| (v[0] < 0) as i128);
    ctx.boolean(&format!("{tag}.sgn"), &sgn);
    let ss = ctx.mul(&sgn, &s);
    let ss2 = ctx.scale(&ss, 2);
    let v = ctx.sub(&s, &ss2);
    let (v2, v1, v0, l) = chunks_lead(ctx, &v, tag);
    (sgn, v2, v1, v0, l)
}

/// LEADNORM + TNORM10: `(e_hat, M', z, ovf)`.
fn normalise<C: Ctx>(ctx: &mut C, e: &C::V, l: &C::V, v2: &C::V, v1: &C::V, v0: &C::V, tag: &str) -> (C::V, C::V, C::V, C::V) {
    let ln = ctx.lookup_tuple(tag, &leadnorm(), &[l.clone(), e.clone()]);
    let (koff, a1, e_hat, z, ovf) = (&ln[0], &ln[1], &ln[2], &ln[3], &ln[4]);
    let t = ctx.lookup_tuple(&format!("{tag}.t"), &tnorm10(), &[v0.clone(), koff.clone()]);
    let hi = ctx.scale(v2, 1 << CHUNK);
    let hi = ctx.add(&hi, v1);
    let m = ctx.mul(&hi, a1);
    let m = ctx.add(&m, &t[0]);
    (e_hat.clone(), m, z.clone(), ovf.clone())
}

/// The group maximum: the `E` hint, one ALIGN4 tuple per candidate, the product form of `attained`.  The range
/// of `E` is bound by the group's LEADNORM lookup (`E in [FLOOR, LAST]`).
fn group_max<C: Ctx>(ctx: &mut C, cands: &[C::V], s4s: &[C::V], tag: &str) -> (C::V, Vec<C::V>) {
    let refs: Vec<&C::V> = cands.iter().collect();
    let e = ctx.hint(&format!("{tag}.E"), &refs, &|v| v.iter().copied().fold(FLOOR, i128::max));
    let mut dsps = Vec::with_capacity(cands.len());
    let mut prod = ctx.subk(&e, FLOOR);
    for (i, (c, s4)) in cands.iter().zip(s4s).enumerate() {
        let d = ctx.sub(&e, c);
        let out = ctx.lookup_tuple(&format!("{tag}[{i}]"), &align4(), &[d.clone(), s4.clone()]);
        dsps.push(out[0].clone());
        prod = ctx.mul(&prod, &d);
    }
    let zero = ctx.konst(0);
    ctx.eq(&format!("{tag}.attained"), &prod, &zero);
    (e, dsps)
}

/// One group on the state `(s, e, M)` with the decoded products; returns the new state `(s', e', M')` (`s'` and
/// `M'` are degree-2 expressions), `z`, `ovf` and the sum sign.
fn group<C: Ctx>(ctx: &mut C, state: &(C::V, C::V, C::V), prods: &[(C::V, C::V, C::V)], tag: &str) -> ((C::V, C::V, C::V), C::V, C::V, C::V) {
    let (st_s, st_e, st_m) = state;
    let mut cands = vec![st_e.clone()];
    let mut s4s = vec![st_s.clone()];
    for (sp, _, ep) in prods {
        cands.push(ep.clone());
        s4s.push(sp.clone());
    }
    let (e, dsps) = group_max(ctx, &cands, &s4s, &format!("{tag}.max"));
    let mut terms = vec![state_shift(ctx, st_m, &dsps[0], &format!("{tag}.sshift"))];
    for (i, (_, u, _)) in prods.iter().enumerate() {
        terms.push(product_shift(ctx, u, &dsps[i + 1], &format!("{tag}.pshift[{i}]")));
    }
    let (sgn, v2, v1, v0, l) = signed_sum(ctx, &terms, &format!("{tag}.sum"));
    let (e_hat, m, z, ovf) = normalise(ctx, &e, &l, &v2, &v1, &v0, &format!("{tag}.norm"));
    let one_m_z = ctx.ksub(1, &z);
    let s_out = ctx.mul(&sgn, &one_m_z);
    ((s_out, e_hat, m), z, ovf, sgn)
}

/// Pack on limbs: `(y_hi, y_lo)` of the final state, the infinity of the first saturating group.
fn pack<C: Ctx>(ctx: &mut C, state: (&C::V, &C::V, &C::V, &C::V), ovfs: &[(C::V, C::V)], tag: &str) -> (C::V, C::V) {
    let (s, e_hat, m, z) = state;
    let m_hi = ctx.hint(&format!("{tag}.M_hi8"), &[m], &|v| v[0] >> HALF16);
    let m_lo = ctx.hint(&format!("{tag}.M_lo16"), &[m], &|v| v[0] & ((1 << HALF16) - 1));
    ctx.rng(&format!("{tag}.M_hi8"), &m_hi, 0, 1 << (ACC_SIG - HALF16));
    ctx.rng(&format!("{tag}.M_lo16"), &m_lo, 0, 1 << HALF16);
    let hs = ctx.scale(&m_hi, 1 << HALF16);
    let rec = ctx.add(&hs, &m_lo);
    ctx.eq(&format!("{tag}.M"), &rec, m);
    // fld = e_hat + (BIAS - 1) + z (E_MIN - FLOOR): 0 for the zero state, <= 254 on a finite state
    let fld = ctx.addk(e_hat, BIAS - 1);
    let zt = ctx.scale(z, E_MIN - FLOOR);
    let fld = ctx.add(&fld, &zt);
    let s_sh = ctx.scale(s, 1 << (ACC_WORD_BITS - 1 - HALF16));
    let f_sh = ctx.scale(&fld, 1 << (ACC_FRAC - HALF16));
    let fin_hi = ctx.sum(&[s_sh, f_sh, m_hi]);
    let mut any_ovf = ctx.konst(0);
    let mut inf_sign = ctx.konst(0);
    let mut none_yet = ctx.konst(1);
    for (ovf_g, sgn_g) in ovfs {
        let first = ctx.mul(&none_yet, ovf_g);
        any_ovf = ctx.add(&any_ovf, &first);
        let fs = ctx.mul(&first, sgn_g);
        inf_sign = ctx.add(&inf_sign, &fs);
        none_yet = ctx.sub(&none_yet, &first);
    }
    let any_ovf = ctx.mat(&format!("{tag}.any_ovf"), &any_ovf);
    let inf_sign = ctx.mat(&format!("{tag}.inf_sign"), &inf_sign);
    let inf_a = ctx.scale(&inf_sign, 1 << (ACC_WORD_BITS - 1 - HALF16));
    let inf_hi = ctx.addk(&inf_a, T_FINITE << (ACC_FRAC - HALF16));
    let d_hi = ctx.sub(&inf_hi, &fin_hi);
    let t = ctx.mul(&any_ovf, &d_hi);
    let y_hi = ctx.add(&fin_hi, &t);
    let t = ctx.mul(&any_ovf, &m_lo);
    let y_lo = ctx.sub(&m_lo, &t);
    (ctx.mat(&format!("{tag}.y_hi"), &y_hi), ctx.mat(&format!("{tag}.y_lo"), &y_lo))
}

const HALF16: i128 = 16;

// --- the unit and the chain ------------------------------------------------------------------------------------------

/// One row's data: the incoming state (the zero state on the first row of a VU; the previous row's outgoing
/// state otherwise -- the transition constraint enforces it), the operands, the position, the VU's claim.
#[derive(Clone, Debug, Default)]
pub struct V2Io {
    /// `(s, e_hat, M)`
    pub st: (i128, i128, i128),
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub step: u64,
    pub y16: u64,
}

impl V2Io {
    pub const ZERO_STATE: (i128, i128, i128) = (0, FLOOR, 0);
}

/// What the row program hands to the AIR: the incoming state columns and the outgoing state expressions.
pub struct V2Row<V> {
    pub st_in: [V; 3],
    pub st_out: [V; 3],
    /// The output word limbs (materialised columns `pack.y_hi`, `pack.y_lo`).
    pub word: Word32<V>,
}

pub const COL_ST_S: &str = "st.s";
pub const COL_ST_E: &str = "st.e";
pub const COL_ST_M: &str = "st.M";

/// The row-local part of the v2 chain AIR.  Row-to-row constraints (`next.first = last`, step increment, state
/// linkage) are emitted by `air_v2::V2ChainAir` and named by `air_v2::tail_names`.
pub fn chain_program_v2<C: Ctx>(ctx: &mut C, io: &V2Io) -> V2Row<C::V> {
    let p = ctx.prime();
    // position
    let step = ctx.hint(COL_STEP, &[], &|_| io.step as i128);
    let first = ctx.hint(COL_FIRST, &[], &|_| (io.step == 0) as i128);
    let last = ctx.hint(COL_LAST, &[], &|_| (io.step == STEPS - 1) as i128);
    let inv = ctx.hint(COL_INV, &[&step], &move |v| crate::ctx::inv_modp(v[0] - (STEPS as i128 - 1), p));
    let claim = ctx.hint(COL_CLAIM, &[], &|_| io.y16 as i128);
    ctx.boolean(COL_FIRST, &first);
    ctx.boolean(COL_LAST, &last);
    let zero = ctx.konst(0);
    let d = ctx.subk(&step, STEPS as i128 - 1);
    let di = ctx.mul(&d, &inv);
    let one_m_last = ctx.ksub(1, &last);
    ctx.eq("vu.last_def", &di, &one_m_last);
    let ld = ctx.mul(&last, &d);
    ctx.eq("vu.last_key", &ld, &zero);
    let fs = ctx.mul(&first, &step);
    ctx.eq("vu.first_step", &fs, &zero);
    // the incoming state
    let st_s = ctx.hint(COL_ST_S, &[], &|_| io.st.0);
    let st_e = ctx.hint(COL_ST_E, &[], &|_| io.st.1);
    let st_m = ctx.hint(COL_ST_M, &[], &|_| io.st.2);
    let fs = ctx.mul(&first, &st_s);
    ctx.eq("vu.first_s", &fs, &zero);
    let ef = ctx.subk(&st_e, FLOOR);
    let fe = ctx.mul(&first, &ef);
    ctx.eq("vu.first_e", &fe, &zero);
    let fm = ctx.mul(&first, &st_m);
    ctx.eq("vu.first_M", &fm, &zero);
    // operands through T_OP
    let a: Vec<C::V> = (0..K).map(|i| ctx.hint(&format!("x.a[{i}]"), &[], &|_| io.a[i] as i128)).collect();
    let b: Vec<C::V> = (0..K).map(|i| ctx.hint(&format!("x.b[{i}]"), &[], &|_| io.b[i] as i128)).collect();
    let top = t_op();
    let dec = |ctx: &mut C, w: &C::V, tag: &str| -> (C::V, C::V, C::V) {
        let o = ctx.lookup_tuple(tag, &top, &[w.clone()]);
        (o[0].clone(), o[1].clone(), o[2].clone())
    };
    let dec_a: Vec<_> = (0..K).map(|i| dec(ctx, &a[i], &format!("a[{i}]"))).collect();
    let dec_b: Vec<_> = (0..K).map(|i| dec(ctx, &b[i], &format!("b[{i}]"))).collect();
    // products: (s_p as s4 = s_a + 2 s_b, u = m_a m_b, e_p = e_a + e_b)
    let prods: Vec<(C::V, C::V, C::V)> = (0..K)
        .map(|i| {
            let (sa, ma, ea) = &dec_a[i];
            let (sb, mb, eb) = &dec_b[i];
            let sb2 = ctx.scale(sb, 2);
            let s4 = ctx.add(sa, &sb2);
            let u = ctx.mul(ma, mb);
            let e = ctx.add(ea, eb);
            (s4, u, e)
        })
        .collect();
    let mut state = (st_s.clone(), st_e.clone(), st_m.clone());
    let mut ovfs = Vec::new();
    let mut z_last = ctx.konst(0);
    let mut start = 0;
    for (g, &size) in GROUPS.iter().enumerate() {
        let tag = format!("g{g}");
        let (st, z, ovf, sgn) = group(ctx, &state, &prods[start..start + size], &tag);
        start += size;
        // a non-final unit's state stays finite (checker v2 `u{i}.g{g}.finite`)
        let nf = ctx.mul(&one_m_last, &ovf);
        ctx.eq(&format!("{tag}.finite"), &nf, &zero);
        ovfs.push((ovf, sgn));
        state = st;
        z_last = z;
    }
    let (y_hi, y_lo) = pack(ctx, (&state.0, &state.1, &state.2, &z_last), &ovfs, "pack");
    let word = Word32::Limbs { hi: y_hi, lo: y_lo };
    let y16 = epilogue(ctx, &word, "cast");
    let diff = ctx.sub(&y16, &claim);
    let out = ctx.mul(&last, &diff);
    ctx.eq("vu.out", &out, &zero);
    V2Row { st_in: [st_s, st_e, st_m], st_out: [state.0, state.1, state.2], word }
}

// --- gadgets in isolation (the battery's gadget-level cases, `tools/battery_adapter.py --v2`) ------------------------

fn gadget_input<C: Ctx>(ctx: &mut C, case: &crate::program::GadgetCase, n: &str, lo: i128, hi: i128) -> C::V {
    let v = *case.inputs.get(n).unwrap_or_else(|| panic!("gadget {}: missing input {n}", case.name));
    let x = ctx.hint(&format!("in.{n}"), &[], &move |_| v);
    ctx.rng(&format!("in.{n}"), &x, lo, hi);
    x
}

/// The word-in unit's decode on limbs (V2_AIR.md 4.5; absent from the state-chained VU): `(s, e_hat, M, z)`.
fn state_decode_limbs<C: Ctx>(ctx: &mut C, c_hi: &C::V, c_lo: &C::V, tag: &str) -> (C::V, C::V, C::V, C::V) {
    let f_bits = ACC_FRAC - HALF16; // 7
    let hdr = ctx.hint(&format!("{tag}.hdr"), &[c_hi], &move |v| v[0] >> f_bits);
    let f_hi = ctx.hint(&format!("{tag}.f_hi"), &[c_hi], &move |v| v[0] & ((1 << f_bits) - 1));
    ctx.rng(&format!("{tag}.f_hi"), &f_hi, 0, 1 << f_bits);
    let hs = ctx.scale(&hdr, 1 << f_bits);
    let rec = ctx.add(&hs, &f_hi);
    ctx.eq(&format!("{tag}.word_hi"), &rec, c_hi);
    let h = ctx.lookup_tuple(&format!("{tag}.hdr"), &t_hdr(), &[hdr]);
    let (s, nz, e) = (&h[0], &h[1], &h[2]);
    let nzs = ctx.scale(nz, 1 << ACC_FRAC);
    let fhs = ctx.scale(&f_hi, 1 << HALF16);
    let m = ctx.sum(&[nzs, fhs, c_lo.clone()]);
    let z = ctx.hint(&format!("{tag}.z"), &[&m], &|v| (v[0] == 0) as i128);
    ctx.boolean(&format!("{tag}.z"), &z);
    let zm = ctx.mul(&z, &m);
    let zero = ctx.konst(0);
    ctx.eq(&format!("{tag}.z0"), &zm, &zero);
    // M + z - 1 in [0, 2^24) as a 16 + 8 split
    let mz = ctx.add(&m, &z);
    let mz = ctx.subk(&mz, 1);
    let mz_hi = ctx.hint(&format!("{tag}.mz_hi"), &[&mz], &|v| v[0] >> HALF16);
    ctx.rng(&format!("{tag}.mz_hi"), &mz_hi, 0, 1 << (ACC_SIG - HALF16));
    let hs = ctx.scale(&mz_hi, 1 << HALF16);
    let mz_lo = ctx.sub(&mz, &hs);
    ctx.rng(&format!("{tag}.mz_lo"), &mz_lo, 0, 1 << HALF16);
    let fl = ctx.ksub(FLOOR, e);
    let zf = ctx.mul(&z, &fl);
    let e_hat = ctx.add(e, &zf);
    (s.clone(), e_hat, m, z)
}

/// A v1-battery gadget case on the v2 gadgets: inputs pinned, hints of the same name imposed by the caller's
/// overrides, outputs bound to the claimed values.  Output conventions follow the v1 gadget contracts where the
/// v2 gadget computes the same function (`align_state` / `align_product`: the unsigned shifted term at
/// `dsp = 2 min(d, DT)`; `leading_bit`: `l`; `epilogue`: `y16`; `fp32_decode`: `(s, e_hat, M)`); `normalise` and
/// `lead_norm` output v2's `(e_hat, M')` (`e_hat = FLOOR` on a zero result, where v1 reported the clamped `e1`).
pub fn gadget_program_v2<C: Ctx>(ctx: &mut C, case: &crate::program::GadgetCase) {
    let outs: Vec<C::V> = match case.name.as_str() {
        "fp32_decode" => {
            let lo = gadget_input(ctx, case, "c_lo", 0, 1 << HALF16);
            let hi = gadget_input(ctx, case, "c_hi", 0, 1 << HALF16);
            let (s, e_hat, m, _z) = state_decode_limbs(ctx, &hi, &lo, "acc");
            vec![s, e_hat, m]
        }
        "epilogue" => {
            let lo = gadget_input(ctx, case, "y32_lo", 0, 1 << HALF16);
            let hi = gadget_input(ctx, case, "y32_hi", 0, 1 << HALF16);
            vec![epilogue(ctx, &Word32::Limbs { hi, lo }, "cast")]
        }
        "align_state" => {
            let m = gadget_input(ctx, case, "M", 0, 1 << ACC_SIG);
            let d = gadget_input(ctx, case, "d", 0, SHIFT_ROWS_V2);
            let zero = ctx.konst(0);
            let dsp = ctx.lookup_tuple("align_state.dsp", &align4(), &[d, zero]);
            vec![state_shift(ctx, &m, &dsp[0], "align_state")]
        }
        "align_product" => {
            let u = gadget_input(ctx, case, "u", 0, 1 << (2 * SIG));
            let d = gadget_input(ctx, case, "d", 0, SHIFT_ROWS_V2);
            let zero = ctx.konst(0);
            let dsp = ctx.lookup_tuple("align_prod.dsp", &align4(), &[d, zero]);
            vec![product_shift(ctx, &u, &dsp[0], "align_prod")]
        }
        "leading_bit" => {
            let v = gadget_input(ctx, case, "v", 0, S_MAX + 1);
            let (_, _, _, l) = chunks_lead(ctx, &v, "lead");
            vec![l]
        }
        "normalise" => {
            let e = gadget_input(ctx, case, "E", FLOOR, LAST + 1);
            let l = gadget_input(ctx, case, "l", 0, L_HI + 1);
            let v = gadget_input(ctx, case, "v", 0, S_MAX + 1);
            let (v2, v1, v0, _) = chunks_lead(ctx, &v, "norm");
            let (e_hat, m, _, _) = normalise(ctx, &e, &l, &v2, &v1, &v0, "norm");
            vec![e_hat, m]
        }
        "lead_norm" => {
            let e = gadget_input(ctx, case, "E", FLOOR, LAST + 1);
            let v = gadget_input(ctx, case, "v", 0, S_MAX + 1);
            let (v2, v1, v0, l) = chunks_lead(ctx, &v, "lead");
            let (e_hat, m, _, _) = normalise(ctx, &e, &l, &v2, &v1, &v0, "norm");
            vec![e_hat, m]
        }
        other => panic!("gadget_program_v2: unknown gadget {other}"),
    };
    assert_eq!(outs.len(), case.claimed.len(), "gadget {}: {} outputs, {} claimed", case.name, outs.len(), case.claimed.len());
    for (i, (o, &c)) in outs.iter().zip(case.claimed.iter()).enumerate() {
        let k = ctx.konst(c);
        ctx.eq(&format!("out[{i}]"), o, &k);
    }
}
