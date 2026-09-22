//! The transition unit as a straight-line program over an abstract context, mirroring
//! `verity_numerical.checker.step.check_step_expr` gadget for gadget (same hints, same range bounds, same
//! identities, same predicate names).  The program runs twice on identical control flow:
//!
//! * on the witness context (`witness::IntCtx`): values are integers, every `hint` is computed by its closure
//!   (the gadget's `gen`) and appended to the row, every range check appends its 2-bit limbs, every lookup its
//!   one-hot selectors;
//! * on the AIR context (`air::AirCtx`): values are symbolic expressions, every `hint`/limb/selector reads the
//!   next column of the row, every predicate emits a constraint.
//!
//! Column allocation is by call order, so the two runs agree on the layout by construction.
//!
//! Predicates (the checker's five, section "Predicates" of the checker docstring):
//!   rng(x, lo, hi)     lo <= x < hi        realised as 2-bit limbs of x - lo (plus an exclusion product or a
//!                                          second decomposition when hi - lo is not a power of two)
//!   eq(a, b)           a == b              one polynomial identity
//!   boolean(b)         b (b - 1) = 0
//!   lookup(T, key)     one-hot selectors over the table's *distinct* rows; outputs are linear in the selectors
//!   lt_pow(x, ipow, B) rng(x * ipow, 0, 2^B) -- the reference realisation, sound over Goldilocks (p > 2^60, F1)

use crate::params::*;

/// Maximum constraint degree the AIR admits (log_blowup = 2 lets the quotient have degree 4).
pub const MAX_DEG: usize = 5;

pub trait Ctx {
    type V: Clone;
    fn konst(&mut self, k: i128) -> Self::V;
    fn add(&mut self, a: &Self::V, b: &Self::V) -> Self::V;
    fn sub(&mut self, a: &Self::V, b: &Self::V) -> Self::V;
    /// Product of two non-constant values; materialises an operand when the degree budget is exceeded.
    fn mul(&mut self, a: &Self::V, b: &Self::V) -> Self::V;
    fn scale(&mut self, a: &Self::V, k: i128) -> Self::V;
    fn addk(&mut self, a: &Self::V, k: i128) -> Self::V;
    /// A witness column the checker cannot compute by ring arithmetic: `f` is the gadget's `gen`, evaluated on
    /// the integer values of `deps` by the witness context only.
    fn hint(&mut self, name: &str, deps: &[&Self::V], f: &dyn Fn(&[i128]) -> i128) -> Self::V;
    /// Copy a value into a fresh column bound by an identity (degree reset to 1).
    fn mat(&mut self, name: &str, a: &Self::V) -> Self::V;
    fn eq(&mut self, name: &str, a: &Self::V, b: &Self::V);
    fn boolean(&mut self, name: &str, b: &Self::V);
    fn rng(&mut self, name: &str, x: &Self::V, lo: i128, hi: i128);
    fn lookup(&mut self, name: &str, table: &Table, key: &Self::V) -> Vec<Self::V>;

    fn lt_pow(&mut self, name: &str, x: &Self::V, ipow: &Self::V, bits: i128) {
        let p = self.mul(x, ipow);
        self.rng(name, &p, 0, 1 << bits);
    }
    fn subk(&mut self, a: &Self::V, k: i128) -> Self::V {
        self.addk(a, -k)
    }
    /// k - a
    fn ksub(&mut self, k: i128, a: &Self::V) -> Self::V {
        let n = self.scale(a, -1);
        self.addk(&n, k)
    }
    fn sum(&mut self, vs: &[Self::V]) -> Self::V {
        let mut acc = self.konst(0);
        for v in vs {
            acc = self.add(&acc, v);
        }
        acc
    }
}

/// A fixed table realised by one-hot selection over its distinct rows.  `rows[j]` matches `key == keys[j]`;
/// when `saturate` is set, the last row instead matches `key >= keys[last]` (proved by a range-checked slack
/// below `slack_hi`).  Outputs are linear combinations of the selectors.
pub struct Table {
    pub name: &'static str,
    pub keys: Vec<i128>,
    pub rows: Vec<Vec<i128>>,
    pub saturate: bool,
    pub slack_hi: i128,
}

pub fn pow_table() -> Table {
    // POW: d -> (2^min(d, W), 2^(W - min(d, W))); rows d >= W all equal (2^W, 1): one saturating row.
    let w = ACC_BITS;
    let keys: Vec<i128> = (0..=w).collect();
    let rows = keys.iter().map(|&d| vec![1i128 << d, 1i128 << (w - d)]).collect();
    Table { name: "POW", keys, rows, saturate: true, slack_hi: SHIFT_ROWS }
}

pub fn align_table() -> Table {
    // ALIGN: d -> (sl, sr, isr); rows d >= prod_bits all equal (1, 2^U, 1): one saturating row.
    let u = 2 * SIG;
    let keys: Vec<i128> = (0..=PROD_BITS).collect();
    let rows = keys
        .iter()
        .map(|&d| {
            if d <= PROD_SHIFT {
                vec![1i128 << (PROD_SHIFT - d), 1, 1i128 << u]
            } else if d < PROD_BITS {
                vec![1, 1i128 << (d - PROD_SHIFT), 1i128 << (PROD_BITS - d)]
            } else {
                vec![1, 1i128 << u, 1]
            }
        })
        .collect();
    Table { name: "ALIGN", keys, rows, saturate: true, slack_hi: SHIFT_ROWS }
}

pub fn lead_table() -> Table {
    // LEAD: l -> (2^(l-1), 2^(B-l+1)); row 0 is (0, 2^B).  Exact rows, l in [0, l_hi].
    let b = S_BITS;
    let keys: Vec<i128> = (0..=L_HI).collect();
    let rows = keys
        .iter()
        .map(|&l| if l == 0 { vec![0, 1i128 << b] } else { vec![1i128 << (l - 1), 1i128 << (b - (l - 1))] })
        .collect();
    Table { name: "LEAD", keys, rows, saturate: false, slack_hi: 0 }
}

pub fn norm_table() -> Table {
    // NORM: k -> (mul, div, idiv), k in [k_lo, k_hi].  Exact rows.
    let r = NORM_REM_BITS;
    let keys: Vec<i128> = (K_LO..=K_HI).collect();
    let rows = keys
        .iter()
        .map(|&k| {
            let neg = if k < 0 { -k } else { 0 };
            let pos = if k > 0 { k } else { 0 };
            vec![1i128 << pos, 1i128 << neg, 1i128 << (r - neg)]
        })
        .collect();
    Table { name: "NORM", keys, rows, saturate: false, slack_hi: 0 }
}

/// One unit's public words (`x`, `y`).  The AIR context receives zeros; it never reads them.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct UnitIo {
    pub c: u64,
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub y: u64,
}

fn bits(n: i128) -> (i128, i128) {
    (0, 1 << n)
}

// ---------------------------------------------------------------------------------------------------------------
// Gadgets (each mirrors the Python class of the same name)
// ---------------------------------------------------------------------------------------------------------------

/// Bf16Decode: (s, m, e)
fn bf16_decode<C: Ctx>(ctx: &mut C, w: &C::V, tag: &str) -> (C::V, C::V, C::V) {
    let s = ctx.hint(&format!("{tag}.s"), &[w], &|v| v[0] >> (WORD_BITS - 1));
    let t = ctx.hint(&format!("{tag}.t"), &[w], &|v| (v[0] >> FRAC) & ((1 << EXP) - 1));
    let f = ctx.hint(&format!("{tag}.f"), &[w], &|v| v[0] & ((1 << FRAC) - 1));
    let nz = ctx.hint(&format!("{tag}.nz"), &[&t], &|v| (v[0] != 0) as i128);
    ctx.boolean(&format!("{tag}.s"), &s);
    ctx.rng(&format!("{tag}.t"), &t, 0, T_FINITE);
    let (lo, hi) = bits(FRAC);
    ctx.rng(&format!("{tag}.f"), &f, lo, hi);
    ctx.boolean(&format!("{tag}.nz"), &nz);
    let s_sh = ctx.scale(&s, 1 << (WORD_BITS - 1));
    let t_sh = ctx.scale(&t, 1 << FRAC);
    let word = ctx.sum(&[s_sh, t_sh, f.clone()]);
    ctx.eq(&format!("{tag}.word"), &word, w);
    let one_m_nz = ctx.ksub(1, &nz);
    let nz0 = ctx.mul(&one_m_nz, &t);
    let zero = ctx.konst(0);
    ctx.eq(&format!("{tag}.nz0"), &nz0, &zero);
    let t_m_nz = ctx.sub(&t, &nz);
    ctx.rng(&format!("{tag}.nz1"), &t_m_nz, 0, T_FINITE - 1);
    let nz_sh = ctx.scale(&nz, 1 << FRAC);
    let m = ctx.add(&f, &nz_sh);
    let e = ctx.subk(&t_m_nz, BIAS - 1);
    (s, m, e)
}

/// Fp32Decode: (s, e_hat, M)
fn fp32_decode<C: Ctx>(ctx: &mut C, c: &C::V, tag: &str) -> (C::V, C::V, C::V) {
    let s = ctx.hint(&format!("{tag}.s"), &[c], &|v| v[0] >> (ACC_WORD_BITS - 1));
    let t = ctx.hint(&format!("{tag}.t"), &[c], &|v| (v[0] >> ACC_FRAC) & ((1 << EXP) - 1));
    let f = ctx.hint(&format!("{tag}.f"), &[c], &|v| v[0] & ((1 << ACC_FRAC) - 1));
    let nz = ctx.hint(&format!("{tag}.nz"), &[&t], &|v| (v[0] != 0) as i128);
    let z = ctx.hint(&format!("{tag}.z"), &[&t, &f], &|v| (v[0] == 0 && v[1] == 0) as i128);
    ctx.boolean(&format!("{tag}.s"), &s);
    ctx.rng(&format!("{tag}.t"), &t, 0, T_FINITE);
    let (lo, hi) = bits(ACC_FRAC);
    ctx.rng(&format!("{tag}.f"), &f, lo, hi);
    ctx.boolean(&format!("{tag}.nz"), &nz);
    ctx.boolean(&format!("{tag}.z"), &z);
    let s_sh = ctx.scale(&s, 1 << (ACC_WORD_BITS - 1));
    let t_sh = ctx.scale(&t, 1 << ACC_FRAC);
    let word = ctx.sum(&[s_sh, t_sh, f.clone()]);
    ctx.eq(&format!("{tag}.word"), &word, c);
    let one_m_nz = ctx.ksub(1, &nz);
    let nz0 = ctx.mul(&one_m_nz, &t);
    let zero = ctx.konst(0);
    ctx.eq(&format!("{tag}.nz0"), &nz0, &zero);
    let t_m_nz = ctx.sub(&t, &nz);
    ctx.rng(&format!("{tag}.nz1"), &t_m_nz, 0, T_FINITE - 1);
    let nz_sh = ctx.scale(&nz, 1 << ACC_FRAC);
    let m = ctx.add(&f, &nz_sh);
    let z0 = ctx.mul(&z, &m);
    ctx.eq(&format!("{tag}.z0"), &z0, &zero);
    let mz = ctx.add(&m, &z);
    let mz1 = ctx.subk(&mz, 1);
    let (lo, hi) = bits(ACC_SIG);
    ctx.rng(&format!("{tag}.z1"), &mz1, lo, hi);
    let e = ctx.subk(&t_m_nz, BIAS - 1);
    let fl_m_e = ctx.ksub(FLOOR, &e);
    let ze = ctx.mul(&z, &fl_m_e);
    let e_hat = ctx.add(&e, &ze);
    (s, e_hat, m)
}

/// Product: (u, s_p, e_hat)
fn product<C: Ctx>(ctx: &mut C, a: &(C::V, C::V, C::V), b: &(C::V, C::V, C::V), tag: &str) -> (C::V, C::V, C::V) {
    let (s_a, m_a, e_a) = a;
    let (s_b, m_b, e_b) = b;
    let u = ctx.mul(m_a, m_b);
    let z = ctx.hint(&format!("{tag}.z"), &[&u], &|v| (v[0] == 0) as i128);
    ctx.boolean(&format!("{tag}.z"), &z);
    let z0 = ctx.mul(&z, &u);
    let zero = ctx.konst(0);
    ctx.eq(&format!("{tag}.z0"), &z0, &zero);
    let uz = ctx.add(&u, &z);
    let uz1 = ctx.subk(&uz, 1);
    let (lo, hi) = bits(2 * SIG);
    ctx.rng(&format!("{tag}.z1"), &uz1, lo, hi);
    let e_p = ctx.add(e_a, e_b);
    let sab = ctx.mul(s_a, s_b);
    let sab2 = ctx.scale(&sab, -2);
    let s_sum = ctx.add(s_a, s_b);
    let s_p = ctx.add(&s_sum, &sab2);
    let fl_m_ep = ctx.ksub(FLOOR, &e_p);
    let ze = ctx.mul(&z, &fl_m_ep);
    let e_hat = ctx.add(&e_p, &ze);
    (u, s_p, e_hat)
}

/// GroupMax: (E, ds)
fn group_max<C: Ctx>(ctx: &mut C, cands: &[C::V], tag: &str) -> (C::V, Vec<C::V>) {
    let m = cands.len();
    let refs: Vec<&C::V> = cands.iter().collect();
    let e = ctx.hint(&format!("{tag}.E"), &refs, &|v| v.iter().copied().fold(FLOOR, i128::max));
    let mut sel = Vec::with_capacity(m + 1);
    for i in 0..=m {
        // sel[i] = [i == first index attaining E] over (cands..., floor)
        // computed from the candidates alone (as Python's gen), so a mutated E leaves the selectors honest
        let s = ctx.hint(&format!("{tag}.sel[{i}]"), &refs, &move |v| {
            let opts: Vec<i128> = v[..m].iter().copied().chain(std::iter::once(FLOOR)).collect();
            let e = opts.iter().copied().fold(FLOOR, i128::max);
            let first = opts.iter().position(|&c| c == e).unwrap();
            (first == i) as i128
        });
        sel.push(s);
    }
    ctx.rng(&format!("{tag}.E"), &e, FLOOR, LAST + 1);
    for (i, b) in sel.iter().enumerate() {
        ctx.boolean(&format!("{tag}.sel[{i}]"), b);
    }
    let onehot = ctx.sum(&sel);
    let one = ctx.konst(1);
    ctx.eq(&format!("{tag}.onehot"), &onehot, &one);
    let mut terms = Vec::with_capacity(m + 1);
    for i in 0..m {
        terms.push(ctx.mul(&sel[i], &cands[i]));
    }
    terms.push(ctx.scale(&sel[m], FLOOR));
    let attained = ctx.sum(&terms);
    ctx.eq(&format!("{tag}.attained"), &attained, &e);
    let mut ds = Vec::with_capacity(m);
    for (i, c) in cands.iter().enumerate() {
        let d = ctx.sub(&e, c);
        ctx.rng(&format!("{tag}.d[{i}]"), &d, 0, SHIFT_ROWS);
        ds.push(d);
    }
    let d_floor = ctx.subk(&e, FLOOR);
    ctx.rng(&format!("{tag}.d_floor"), &d_floor, 0, SHIFT_ROWS);
    (e, ds)
}

/// AlignState: q
fn align_state<C: Ctx>(ctx: &mut C, m: &C::V, d: &C::V, tag: &str) -> C::V {
    let w = ACC_BITS;
    let q = ctx.hint(&format!("{tag}.q"), &[m, d], &|v| {
        let p = v[0] << ACC_SHIFT;
        let dd = v[1].min(w);
        p >> dd
    });
    let r = ctx.hint(&format!("{tag}.r"), &[m, d], &|v| {
        let p = v[0] << ACC_SHIFT;
        let dd = v[1].min(w);
        p - ((p >> dd) << dd)
    });
    ctx.rng(&format!("{tag}.q"), &q, 0, 1 << w);
    ctx.rng(&format!("{tag}.r"), &r, 0, 1 << w);
    let row = ctx.lookup(&format!("{tag}.pow"), &pow_table(), d);
    let (pow, ipow) = (&row[0], &row[1]);
    let qp = ctx.mul(&q, pow);
    let lhs = ctx.add(&qp, &r);
    let rhs = ctx.scale(m, 1 << ACC_SHIFT);
    ctx.eq(&format!("{tag}.div"), &lhs, &rhs);
    ctx.lt_pow(&format!("{tag}.rem"), &r, ipow, w);
    ctx.lt_pow(&format!("{tag}.quot"), &q, pow, w);
    q
}

/// AlignProduct: aligned = q * sl
fn align_product<C: Ctx>(ctx: &mut C, u: &C::V, d: &C::V, tag: &str) -> C::V {
    let ubits = 2 * SIG;
    let q = ctx.hint(&format!("{tag}.q"), &[u, d], &|v| {
        let (u, d) = (v[0], v[1]);
        if d <= PROD_SHIFT { u } else if d < PROD_BITS { u >> (d - PROD_SHIFT) } else { 0 }
    });
    let r = ctx.hint(&format!("{tag}.r"), &[u, d], &|v| {
        let (u, d) = (v[0], v[1]);
        if d <= PROD_SHIFT {
            0
        } else if d < PROD_BITS {
            let q = u >> (d - PROD_SHIFT);
            u - (q << (d - PROD_SHIFT))
        } else {
            u
        }
    });
    ctx.rng(&format!("{tag}.q"), &q, 0, 1 << ubits);
    ctx.rng(&format!("{tag}.r"), &r, 0, 1 << ubits);
    let row = ctx.lookup(&format!("{tag}.align"), &align_table(), d);
    let (sl, sr, isr) = (&row[0], &row[1], &row[2]);
    let qs = ctx.mul(&q, sr);
    let lhs = ctx.add(&qs, &r);
    ctx.eq(&format!("{tag}.div"), &lhs, u);
    ctx.lt_pow(&format!("{tag}.rem"), &r, isr, ubits);
    ctx.lt_pow(&format!("{tag}.quot"), &q, sr, ubits);
    ctx.mul(&q, sl)
}

/// SignedSum: (sgn, v) with v materialised.
fn signed_sum<C: Ctx>(ctx: &mut C, terms: &[(C::V, C::V)], tag: &str) -> (C::V, C::V) {
    // premise: (2^acc_bits - 1) + n (2^prod_bits - 1) < 2^s_bits
    assert!((1i128 << ACC_BITS) - 1 + (terms.len() as i128 - 1) * ((1i128 << PROD_BITS) - 1) < (1i128 << S_BITS));
    let mut parts = Vec::with_capacity(terms.len());
    for (s, q) in terms {
        let sq = ctx.mul(s, q);
        let sq2 = ctx.scale(&sq, -2);
        parts.push(ctx.add(q, &sq2));
    }
    let s_expr = ctx.sum(&parts);
    let s_val = ctx.mat(&format!("{tag}.S"), &s_expr);
    let sgn = ctx.hint(&format!("{tag}.sgn"), &[&s_val], &|v| (v[0] < 0) as i128);
    let z = ctx.hint(&format!("{tag}.z"), &[&s_val], &|v| (v[0] == 0) as i128);
    ctx.boolean(&format!("{tag}.sgn"), &sgn);
    ctx.boolean(&format!("{tag}.z"), &z);
    let ss = ctx.mul(&sgn, &s_val);
    let ss2 = ctx.scale(&ss, -2);
    let v_expr = ctx.add(&s_val, &ss2);
    let v = ctx.mat(&format!("{tag}.v"), &v_expr);
    ctx.rng(&format!("{tag}.v"), &v, 0, 1 << S_BITS);
    let zv = ctx.mul(&z, &v);
    let zero = ctx.konst(0);
    ctx.eq(&format!("{tag}.z0"), &zv, &zero);
    let vz = ctx.add(&v, &z);
    let vz1 = ctx.subk(&vz, 1);
    ctx.rng(&format!("{tag}.z1"), &vz1, 0, 1 << S_BITS);
    let sz = ctx.mul(&sgn, &z);
    ctx.eq(&format!("{tag}.zero_sign"), &sz, &zero);
    (sgn, v)
}

/// LeadingBit: l
fn leading_bit<C: Ctx>(ctx: &mut C, v: &C::V, tag: &str) -> C::V {
    let l = ctx.hint(&format!("{tag}.l"), &[v], &|v| 128 - (v[0] as u128).leading_zeros() as i128);
    let row = ctx.lookup(&format!("{tag}.lead"), &lead_table(), &l);
    let (lo, ispan) = (&row[0], &row[1]);
    let vlo = ctx.sub(v, lo);
    ctx.lt_pow(&format!("{tag}.window"), &vlo, ispan, S_BITS);
    l
}

/// Normalise: (e1, M)
fn normalise<C: Ctx>(ctx: &mut C, e: &C::V, l: &C::V, v: &C::V, tag: &str) -> (C::V, C::V) {
    let clamp = ctx.hint(&format!("{tag}.clamp"), &[e, l], &|x| ((x[0] + x[1] - WIDTH) < E_MIN) as i128);
    let m = ctx.hint(&format!("{tag}.M"), &[e, l, v], &|x| {
        let (e, l, v) = (x[0], x[1], x[2]);
        let e1 = (e + l - WIDTH).max(E_MIN);
        let k = e - e1 + ACC_SIG - WIDTH;
        if k >= 0 { v << k } else { v >> -k }
    });
    let r = ctx.hint(&format!("{tag}.r"), &[e, l, v], &|x| {
        let (e, l, v) = (x[0], x[1], x[2]);
        let e1 = (e + l - WIDTH).max(E_MIN);
        let k = e - e1 + ACC_SIG - WIDTH;
        if k >= 0 { 0 } else { v - ((v >> -k) << -k) }
    });
    ctx.boolean(&format!("{tag}.clamp"), &clamp);
    ctx.rng(&format!("{tag}.M"), &m, 0, 1 << ACC_SIG);
    ctx.rng(&format!("{tag}.r"), &r, 0, 1 << NORM_REM_BITS);
    let el = ctx.add(e, l);
    let e_raw = ctx.subk(&el, WIDTH);
    // clamp_proof = clamp (e_min - 1 - e_raw) + (1 - clamp) (e_raw - e_min)
    let a = ctx.ksub(E_MIN - 1, &e_raw);
    let ca = ctx.mul(&clamp, &a);
    let one_m_clamp = ctx.ksub(1, &clamp);
    let b = ctx.subk(&e_raw, E_MIN);
    let cb = ctx.mul(&one_m_clamp, &b);
    let proof = ctx.add(&ca, &cb);
    ctx.rng(&format!("{tag}.clamp_proof"), &proof, 0, CMP_ROWS);
    let emin_m_eraw = ctx.ksub(E_MIN, &e_raw);
    let ce = ctx.mul(&clamp, &emin_m_eraw);
    let e1_expr = ctx.add(&e_raw, &ce);
    let e1 = ctx.mat(&format!("{tag}.e1"), &e1_expr);
    let e_m_e1 = ctx.sub(e, &e1);
    let k = ctx.addk(&e_m_e1, ACC_SIG - WIDTH);
    let row = ctx.lookup(&format!("{tag}.norm"), &norm_table(), &k);
    let (mul, div, idiv) = (&row[0], &row[1], &row[2]);
    let dm = ctx.mul(div, &m);
    let lhs = ctx.add(&dm, &r);
    let rhs = ctx.mul(v, mul);
    ctx.eq(&format!("{tag}.div"), &lhs, &rhs);
    ctx.lt_pow(&format!("{tag}.rem"), &r, idiv, NORM_REM_BITS);
    (e1, m)
}

/// StateOut: (s, e_hat, M, ovf_eff, z)
fn state_out<C: Ctx>(ctx: &mut C, sgn: &C::V, e1: &C::V, m: &C::V, tag: &str) -> (C::V, C::V, C::V, C::V, C::V) {
    let z = ctx.hint(&format!("{tag}.z"), &[m], &|v| (v[0] == 0) as i128);
    let ovf = ctx.hint(&format!("{tag}.ovf"), &[e1], &|v| (v[0] > E_MAX) as i128);
    ctx.boolean(&format!("{tag}.z"), &z);
    ctx.boolean(&format!("{tag}.ovf"), &ovf);
    let zm = ctx.mul(&z, m);
    let zero = ctx.konst(0);
    ctx.eq(&format!("{tag}.z0"), &zm, &zero);
    let mz = ctx.add(m, &z);
    let mz1 = ctx.subk(&mz, 1);
    ctx.rng(&format!("{tag}.z1"), &mz1, 0, 1 << ACC_SIG);
    // ovf_proof = ovf (e1 - e_max - 1) + (1 - ovf) (e_max - e1)
    let a = ctx.subk(e1, E_MAX + 1);
    let oa = ctx.mul(&ovf, &a);
    let one_m_ovf = ctx.ksub(1, &ovf);
    let b = ctx.ksub(E_MAX, e1);
    let ob = ctx.mul(&one_m_ovf, &b);
    let proof = ctx.add(&oa, &ob);
    ctx.rng(&format!("{tag}.ovf_proof"), &proof, 0, CMP_ROWS);
    let oz = ctx.mul(&ovf, &z);
    let ovf_eff = ctx.sub(&ovf, &oz);
    let sz = ctx.mul(sgn, &z);
    let s = ctx.sub(sgn, &sz);
    let fl_m_e1 = ctx.ksub(FLOOR, e1);
    let ze = ctx.mul(&z, &fl_m_e1);
    let e_hat = ctx.add(e1, &ze);
    (s, e_hat, m.clone(), ovf_eff, z)
}

/// Pack: the output word expression.
fn pack<C: Ctx>(ctx: &mut C, state: (&C::V, &C::V, &C::V, &C::V), ovfs: &[(C::V, C::V)], tag: &str) -> C::V {
    let (s, e1, m, z) = state;
    let hidden = ctx.hint(&format!("{tag}.hidden"), &[m], &|v| v[0] >> ACC_FRAC);
    let low = ctx.hint(&format!("{tag}.low"), &[m], &|v| v[0] & ((1 << ACC_FRAC) - 1));
    ctx.boolean(&format!("{tag}.hidden"), &hidden);
    ctx.rng(&format!("{tag}.low"), &low, 0, 1 << ACC_FRAC);
    let h_sh = ctx.scale(&hidden, 1 << ACC_FRAC);
    let hm = ctx.add(&h_sh, &low);
    ctx.eq(&format!("{tag}.M"), &hm, m);
    let fld_a = ctx.addk(e1, BIAS - 1);
    let fld = ctx.add(&fld_a, &hidden);
    let s_sh = ctx.scale(s, 1 << (ACC_WORD_BITS - 1));
    let fld_sh = ctx.scale(&fld, 1 << ACC_FRAC);
    let fin = ctx.sum(&[s_sh, fld_sh, low.clone()]);
    let mut any_ovf = ctx.konst(0);
    let mut inf_sign = ctx.konst(0);
    let mut none_yet = ctx.konst(1);
    for (ovf_g, sgn_g) in ovfs {
        let first = ctx.mul(&none_yet, ovf_g);
        any_ovf = ctx.add(&any_ovf, &first);
        let fs = ctx.mul(&first, sgn_g);
        inf_sign = ctx.add(&inf_sign, &fs);
        let ny = ctx.mul(&none_yet, ovf_g);
        none_yet = ctx.sub(&none_yet, &ny);
    }
    let any_ovf = ctx.mat(&format!("{tag}.any_ovf"), &any_ovf);
    let inf_sign = ctx.mat(&format!("{tag}.inf_sign"), &inf_sign);
    let inf_a = ctx.scale(&inf_sign, 1 << (ACC_WORD_BITS - 1));
    let inf = ctx.addk(&inf_a, T_FINITE << ACC_FRAC);
    let t1 = ctx.mul(&any_ovf, &inf);
    let one_m_any = ctx.ksub(1, &any_ovf);
    let one_m_z = ctx.ksub(1, z);
    let t2a = ctx.mul(&one_m_any, &one_m_z);
    let t2 = ctx.mul(&t2a, &fin);
    ctx.add(&t1, &t2)
}

// ---------------------------------------------------------------------------------------------------------------
// The unit
// ---------------------------------------------------------------------------------------------------------------

/// `check_step_expr` with the public words as the first 1 + 2k + 1 columns and `y = word` as the final identity.
pub fn unit_program<C: Ctx>(ctx: &mut C, io: &UnitIo) {
    let c = ctx.hint("x.c", &[], &|_| io.c as i128);
    let a: Vec<C::V> = (0..K).map(|i| ctx.hint(&format!("x.a[{i}]"), &[], &|_| io.a[i] as i128)).collect();
    let b: Vec<C::V> = (0..K).map(|i| ctx.hint(&format!("x.b[{i}]"), &[], &|_| io.b[i] as i128)).collect();
    let y = ctx.hint("y", &[], &|_| io.y as i128);
    ctx.rng("x.c", &c, 0, 1 << ACC_WORD_BITS);
    for i in 0..K {
        ctx.rng(&format!("x.a[{i}]"), &a[i], 0, 1 << WORD_BITS);
        ctx.rng(&format!("x.b[{i}]"), &b[i], 0, 1 << WORD_BITS);
    }
    let mut state = fp32_decode(ctx, &c, "acc");
    let dec_a: Vec<_> = (0..K).map(|i| bf16_decode(ctx, &a[i], &format!("a[{i}]"))).collect();
    let dec_b: Vec<_> = (0..K).map(|i| bf16_decode(ctx, &b[i], &format!("b[{i}]"))).collect();
    let prods: Vec<_> = (0..K).map(|i| product(ctx, &dec_a[i], &dec_b[i], &format!("prod[{i}]"))).collect();
    let mut ovfs = Vec::new();
    let mut start = 0;
    let mut z_last = ctx.konst(0);
    for (g, &size) in GROUPS.iter().enumerate() {
        let group = &prods[start..start + size];
        start += size;
        let tag = format!("g{g}");
        let mut cands = vec![state.1.clone()];
        cands.extend(group.iter().map(|(_, _, e)| e.clone()));
        let (e, ds) = group_max(ctx, &cands, &format!("{tag}.max"));
        let q_acc = align_state(ctx, &state.2, &ds[0], &format!("{tag}.align_state"));
        let q_prods: Vec<C::V> = group
            .iter()
            .zip(ds[1..].iter())
            .enumerate()
            .map(|(i, ((u, _, _), d))| align_product(ctx, u, d, &format!("{tag}.align_prod[{i}]")))
            .collect();
        let mut terms = vec![(state.0.clone(), q_acc)];
        terms.extend(group.iter().zip(q_prods).map(|((_, s, _), q)| (s.clone(), q)));
        let (sgn, v) = signed_sum(ctx, &terms, &format!("{tag}.sum"));
        let l = leading_bit(ctx, &v, &format!("{tag}.lead"));
        let (e1, m) = normalise(ctx, &e, &l, &v, &format!("{tag}.norm"));
        let (s_out, e_out, m_out, ovf_eff, z) = state_out(ctx, &sgn, &e1, &m, &format!("{tag}.out"));
        ovfs.push((ovf_eff, sgn));
        // the next group's state; e_out and s_out are masked (degree 2) -- materialise so their uses stay cheap
        let e_out = ctx.mat(&format!("{tag}.out.e_hat"), &e_out);
        let s_out = ctx.mat(&format!("{tag}.out.s"), &s_out);
        state = (s_out, e_out, m_out);
        z_last = z;
    }
    // Pack receives (s, e_hat, M, z) exactly as step.py: `Pack.check(ctx, p, (*state, z), ovfs, h)`.
    let word = pack(ctx, (&state.0, &state.1, &state.2, &z_last), &ovfs, "pack");
    let word = ctx.mat("pack.y", &word);
    ctx.eq("y", &word, &y);
}
