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

/// Maximum constraint degree the AIR admits (log_blowup = 2 lets the quotient have degree 4).  An earlier
/// observation that degree >= 3 constraints were "not enforced" was the challenger bug documented in
/// `challenger.rs` (every challenge sampled as zero); with the fixed challenger every mutation of the negative
/// suite is rejected at degree 5 (`tests/negative.rs`).
pub const MAX_DEG: usize = 5;

pub trait Ctx {
    type V: Clone;
    /// 31-bit trace field: the wrap-free realisations below are selected.
    fn small_field(&self) -> bool;
    /// The trace field's prime (witness-side inverses).
    fn prime(&self) -> u128;
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

    /// `x * ipow < 2^bits`, i.e. `x < bound` with `bound = 2^bits / ipow` a table column.  Over Goldilocks the
    /// reference realisation (the product is below 2^60 < p).  Over a 31-bit field the product wraps (checker
    /// finding F1), so the predicate is realised directly as `bound - 1 - x in [0, 2^bits)`: `x` is already
    /// range-checked to `[0, 2^bits)` by its gadget and `bound <= 2^bits`, so a violating `x` gives a negative
    /// difference whose representative is `>= p - 2^bits + 1 >= 2^bits` (true for every `bits <= 30` at
    /// `p = 2^31 - 2^27 + 1`... for `bits = 30` this needs `p - 2^30 + 1 >= 2^30`, which BabyBear FAILS; the one
    /// 30-bit use, `LeadingBit.window`, is therefore sound only under the signed-sum premise bound `v <= S_MAX`,
    /// which the `sum.premise` check enforces explicitly).
    fn lt_pow(&mut self, name: &str, x: &Self::V, ipow: &Self::V, bound: &Self::V, bits: i128) {
        if self.small_field() {
            let d = self.sub(bound, x);
            let d = self.subk(&d, 1);
            self.rng(name, &d, 0, 1 << bits);
        } else {
            let p = self.mul(x, ipow);
            self.rng(name, &p, 0, 1 << bits);
        }
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
#[derive(Clone, Debug)]
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
    // LEAD: l -> (2^(l-1), 2^(B-l+1), span = 2^B / ispan); row 0 is (0, 2^B, 1).  Exact rows, l in [0, l_hi].
    // The third column is the window width the direct `lt_pow` realisation compares against (free: outputs are
    // linear in the selectors).
    let b = S_BITS;
    let keys: Vec<i128> = (0..=L_HI).collect();
    let rows = keys
        .iter()
        .map(|&l| {
            if l == 0 {
                vec![0, 1i128 << b, 1]
            } else {
                vec![1i128 << (l - 1), 1i128 << (b - (l - 1)), 1i128 << (l - 1)]
            }
        })
        .collect();
    Table { name: "LEAD", keys, rows, saturate: false, slack_hi: 0 }
}

pub fn norm_table() -> Table {
    // NORM: k -> (mul, div, idiv, [k == k_lo]), k in [k_lo, k_hi].  Exact rows.  The fourth column selects the
    // one row whose division identity can reach 2^31 - 1 (checker finding F3); the 31-bit realisation bounds M
    // there.
    let r = NORM_REM_BITS;
    let keys: Vec<i128> = (K_LO..=K_HI).collect();
    let rows = keys
        .iter()
        .map(|&k| {
            let neg = if k < 0 { -k } else { 0 };
            let pos = if k > 0 { k } else { 0 };
            vec![1i128 << pos, 1i128 << neg, 1i128 << (r - neg), (k == K_LO) as i128]
        })
        .collect();
    Table { name: "NORM", keys, rows, saturate: false, slack_hi: 0 }
}

/// A 32-bit word: one column over Goldilocks, two 16-bit limbs `hi * 2^16 + lo` over a 31-bit field (checker
/// finding F4: the word does not fit the field).
#[derive(Clone, Debug)]
pub enum Word32<V> {
    Full(V),
    Limbs { hi: V, lo: V },
}

impl<V: Clone> Word32<V> {
    pub fn cols(&self) -> Vec<V> {
        match self {
            Word32::Full(v) => vec![v.clone()],
            Word32::Limbs { hi, lo } => vec![hi.clone(), lo.clone()],
        }
    }
    fn refs(&self) -> Vec<&V> {
        match self {
            Word32::Full(v) => vec![v],
            Word32::Limbs { hi, lo } => vec![hi, lo],
        }
    }
    fn is_full(&self) -> bool {
        matches!(self, Word32::Full(_))
    }
}

/// The integer value of a word from its dependency values (`Full`: `v[0]`; `Limbs`: `v[0] << 16 | v[1]`).
fn word_of(full: bool, v: &[i128]) -> i128 {
    if full {
        v[0]
    } else {
        (v[0] << HALF) + v[1]
    }
}

/// Half a 32-bit word.
pub const HALF: i128 = 16;

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

/// Fp32Decode: (s, e_hat, M).  Over a 31-bit field the word is two 16-bit limbs and the fraction is split at the
/// limb boundary (`f = f_hi 2^16 + f_lo`, `f_hi < 2^7`): the reconstruction identity becomes two 16-bit
/// identities (`c_lo = f_lo`, `c_hi = s 2^15 + t 2^7 + f_hi`), each side below 2^16.
fn fp32_decode<C: Ctx>(ctx: &mut C, c: &Word32<C::V>, tag: &str) -> (C::V, C::V, C::V) {
    let full = c.is_full();
    let deps = c.refs();
    let s = ctx.hint(&format!("{tag}.s"), &deps, &move |v| word_of(full, v) >> (ACC_WORD_BITS - 1));
    let t = ctx.hint(&format!("{tag}.t"), &deps, &move |v| (word_of(full, v) >> ACC_FRAC) & ((1 << EXP) - 1));
    let (f, parts) = if full {
        let f = ctx.hint(&format!("{tag}.f"), &deps, &move |v| word_of(full, v) & ((1 << ACC_FRAC) - 1));
        let (lo, hi) = bits(ACC_FRAC);
        ctx.rng(&format!("{tag}.f"), &f, lo, hi);
        (f, None)
    } else {
        let f_lo = ctx.hint(&format!("{tag}.f_lo"), &deps, &move |v| word_of(full, v) & ((1 << HALF) - 1));
        let f_hi = ctx.hint(&format!("{tag}.f_hi"), &deps, &move |v| (word_of(full, v) >> HALF) & ((1 << (ACC_FRAC - HALF)) - 1));
        ctx.rng(&format!("{tag}.f_lo"), &f_lo, 0, 1 << HALF);
        ctx.rng(&format!("{tag}.f_hi"), &f_hi, 0, 1 << (ACC_FRAC - HALF));
        let f_hi_sh = ctx.scale(&f_hi, 1 << HALF);
        (ctx.add(&f_hi_sh, &f_lo), Some((f_lo, f_hi)))
    };
    let nz = ctx.hint(&format!("{tag}.nz"), &[&t], &|v| (v[0] != 0) as i128);
    let z = ctx.hint(&format!("{tag}.z"), &[&t, &f], &|v| (v[0] == 0 && v[1] == 0) as i128);
    ctx.boolean(&format!("{tag}.s"), &s);
    ctx.rng(&format!("{tag}.t"), &t, 0, T_FINITE);
    ctx.boolean(&format!("{tag}.nz"), &nz);
    ctx.boolean(&format!("{tag}.z"), &z);
    match (c, parts) {
        (Word32::Full(c), _) => {
            let s_sh = ctx.scale(&s, 1 << (ACC_WORD_BITS - 1));
            let t_sh = ctx.scale(&t, 1 << ACC_FRAC);
            let word = ctx.sum(&[s_sh, t_sh, f.clone()]);
            ctx.eq(&format!("{tag}.word"), &word, c);
        }
        (Word32::Limbs { hi: c_hi, lo: c_lo }, Some((f_lo, f_hi))) => {
            // c_lo = f_lo and c_hi = s 2^15 + t 2^7 + f_hi: every side below 2^16.
            ctx.eq(&format!("{tag}.word_lo"), &f_lo, c_lo);
            let s_sh = ctx.scale(&s, 1 << (ACC_WORD_BITS - 1 - HALF));
            let t_sh = ctx.scale(&t, 1 << (ACC_FRAC - HALF));
            let word_hi = ctx.sum(&[s_sh, t_sh, f_hi]);
            ctx.eq(&format!("{tag}.word_hi"), &word_hi, c_hi);
        }
        _ => unreachable!(),
    }
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
        shr(p, v[1].min(w))
    });
    let r = ctx.hint(&format!("{tag}.r"), &[m, d], &|v| {
        let p = v[0] << ACC_SHIFT;
        let dd = v[1].min(w);
        p - shl(shr(p, dd), dd)
    });
    ctx.rng(&format!("{tag}.q"), &q, 0, 1 << w);
    ctx.rng(&format!("{tag}.r"), &r, 0, 1 << w);
    let row = ctx.lookup(&format!("{tag}.pow"), &pow_table(), d);
    let (pow, ipow) = (&row[0], &row[1]);
    let qp = ctx.mul(&q, pow);
    let lhs = ctx.add(&qp, &r);
    let rhs = ctx.scale(m, 1 << ACC_SHIFT);
    ctx.eq(&format!("{tag}.div"), &lhs, &rhs);
    // r < pow (= 2^dd) and q < ipow (= 2^(W - dd)); with both, q * pow + r < 2^(W + 1) and the identity is exact
    ctx.lt_pow(&format!("{tag}.rem"), &r, ipow, pow, w);
    ctx.lt_pow(&format!("{tag}.quot"), &q, pow, ipow, w);
    q
}

/// AlignProduct: aligned = q * sl
fn align_product<C: Ctx>(ctx: &mut C, u: &C::V, d: &C::V, tag: &str) -> C::V {
    let ubits = 2 * SIG;
    let q = ctx.hint(&format!("{tag}.q"), &[u, d], &|v| {
        let (u, d) = (v[0], v[1]);
        if d <= PROD_SHIFT { u } else if d < PROD_BITS { shr(u, d - PROD_SHIFT) } else { 0 }
    });
    let r = ctx.hint(&format!("{tag}.r"), &[u, d], &|v| {
        let (u, d) = (v[0], v[1]);
        if d <= PROD_SHIFT {
            0
        } else if d < PROD_BITS {
            let q = shr(u, d - PROD_SHIFT);
            u - shl(q, d - PROD_SHIFT)
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
    ctx.lt_pow(&format!("{tag}.rem"), &r, isr, sr, ubits);
    ctx.lt_pow(&format!("{tag}.quot"), &q, sr, isr, ubits);
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
    if ctx.small_field() {
        // BabyBear: p - 2^30 < 2^30, so a 30-bit range check alone does not exclude a negative representative;
        // tighten |S| to the premise bound (fields.py BABYBEAR_EXTRA).  Sound: S_MAX - v for v in [0, 2^30) is
        // negative only for v > S_MAX, and then its representative p + S_MAX - v > p - 2^30 + S_MAX > 2^30 iff
        // v < S_MAX + p - 2^30 = S_MAX + 0.94e9, which every v < 2^30 satisfies.
        let slack = ctx.ksub(S_MAX, &v);
        ctx.rng(&format!("{tag}.premise"), &slack, 0, 1 << S_BITS);
    }
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
    let (lo, ispan, span) = (&row[0], &row[1], &row[2]);
    let vlo = ctx.sub(v, lo);
    if ctx.small_field() {
        // LT_DIRECT, two-range form (babybear_rules.md 2, 6): the direct bound `span - 1 - (v - lo) in [0, 2^30)`
        // alone only bounds `v - lo` from ABOVE.  `v - lo` is not a range-checked hint, and for `v < lo` the
        // difference `span - 1 - (v - lo) = 2^l - 1 - v` is a small positive integer that PASSES the direct check:
        // `v = 16, l = 29` was accepted by the port as shipped (M = v >> 5 = 0, the group's output forced to zero).
        // The lower bound is a second range on the linear wire: `v - lo in [0, 2^30)`; with `v <= S_MAX` (the
        // `sum.premise` check) a negative `v - lo >= -2^29` has representative `>= p - 2^29 > 2^30`, so it is
        // excluded.  SOUNDNESS FIX (b-lookup, 2026-09-22).
        ctx.rng(&format!("{tag}.window_lo"), &vlo, 0, 1 << S_BITS);
    }
    ctx.lt_pow(&format!("{tag}.window"), &vlo, ispan, span, S_BITS);
    l
}

/// Shifts that do not panic on adversarial (overridden) hint inputs in debug builds: the hint closures run on
/// whatever the row supplies (`IntCtx::overrides`, the battery); the constraints, not these closures, are the check.
fn shl(v: i128, k: i128) -> i128 { if (0..127).contains(&k) { v << k } else { 0 } }
fn shr(v: i128, k: i128) -> i128 { if (0..127).contains(&k) { v >> k } else if v < 0 { -1 } else { 0 } }

/// Normalise: (e1, M)
fn normalise<C: Ctx>(ctx: &mut C, e: &C::V, l: &C::V, v: &C::V, tag: &str) -> (C::V, C::V) {
    let clamp = ctx.hint(&format!("{tag}.clamp"), &[e, l], &|x| ((x[0] + x[1] - WIDTH) < E_MIN) as i128);
    let m = ctx.hint(&format!("{tag}.M"), &[e, l, v], &|x| {
        let (e, l, v) = (x[0], x[1], x[2]);
        let e1 = (e + l - WIDTH).max(E_MIN);
        let k = e - e1 + ACC_SIG - WIDTH;
        if k >= 0 { shl(v, k) } else { shr(v, -k) }
    });
    let r = ctx.hint(&format!("{tag}.r"), &[e, l, v], &|x| {
        let (e, l, v) = (x[0], x[1], x[2]);
        let e1 = (e + l - WIDTH).max(E_MIN);
        let k = e - e1 + ACC_SIG - WIDTH;
        if k >= 0 { 0 } else { v - shl(shr(v, -k), -k) }
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
    let (mul, div, idiv, k_lo) = (&row[0], &row[1], &row[2], &row[3]);
    if ctx.small_field() {
        // F3: div M + r reaches 2^31 - 1 > p only in the k = K_LO row (div = 2^7, M up to 2^24), where the honest
        // M = v >> 7 < 2^23 because v <= S_MAX < 2^30.  Bounding M below 2^23 there keeps every side of the
        // division identity below 2^30 < p, so the field identity is the integer one.  In the k >= 0 rows
        // (div = 1, r = 0) M is the unique residue of v 2^k and the honest one is below 2^24.
        let km = ctx.mul(k_lo, &m);
        ctx.rng(&format!("{tag}.k_lo_bound"), &km, 0, 1 << (ACC_SIG - 1));
    }
    let dm = ctx.mul(div, &m);
    let lhs = ctx.add(&dm, &r);
    let rhs = ctx.mul(v, mul);
    ctx.eq(&format!("{tag}.div"), &lhs, &rhs);
    ctx.lt_pow(&format!("{tag}.rem"), &r, idiv, div, NORM_REM_BITS);
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

/// Pack: the output word expression (`Full`), or its two 16-bit limbs over a 31-bit field (`low` split at the
/// limb boundary, the infinity constant and the word selection written per limb).
fn pack<C: Ctx>(ctx: &mut C, state: (&C::V, &C::V, &C::V, &C::V), ovfs: &[(C::V, C::V)], tag: &str) -> Word32<C::V> {
    let (s, e1, m, z) = state;
    let hidden = ctx.hint(&format!("{tag}.hidden"), &[m], &|v| v[0] >> ACC_FRAC);
    ctx.boolean(&format!("{tag}.hidden"), &hidden);
    let small = ctx.small_field();
    // fin = s 2^31 + fld 2^23 + low, as one expression or as (fin_hi, fin_lo)
    let fld_a = ctx.addk(e1, BIAS - 1);
    let fld = ctx.add(&fld_a, &hidden);
    let fin: Word32<C::V> = if !small {
        let low = ctx.hint(&format!("{tag}.low"), &[m], &|v| v[0] & ((1 << ACC_FRAC) - 1));
        ctx.rng(&format!("{tag}.low"), &low, 0, 1 << ACC_FRAC);
        let h_sh = ctx.scale(&hidden, 1 << ACC_FRAC);
        let hm = ctx.add(&h_sh, &low);
        ctx.eq(&format!("{tag}.M"), &hm, m);
        let s_sh = ctx.scale(s, 1 << (ACC_WORD_BITS - 1));
        let fld_sh = ctx.scale(&fld, 1 << ACC_FRAC);
        Word32::Full(ctx.sum(&[s_sh, fld_sh, low]))
    } else {
        let low_lo = ctx.hint(&format!("{tag}.low_lo"), &[m], &|v| v[0] & ((1 << HALF) - 1));
        let low_hi = ctx.hint(&format!("{tag}.low_hi"), &[m], &|v| (v[0] >> HALF) & ((1 << (ACC_FRAC - HALF)) - 1));
        ctx.rng(&format!("{tag}.low_lo"), &low_lo, 0, 1 << HALF);
        ctx.rng(&format!("{tag}.low_hi"), &low_hi, 0, 1 << (ACC_FRAC - HALF));
        let h_sh = ctx.scale(&hidden, 1 << ACC_FRAC);
        let lh_sh = ctx.scale(&low_hi, 1 << HALF);
        let hm = ctx.sum(&[h_sh, lh_sh, low_lo.clone()]);
        ctx.eq(&format!("{tag}.M"), &hm, m);
        let s_sh = ctx.scale(s, 1 << (ACC_WORD_BITS - 1 - HALF));
        let fld_sh = ctx.scale(&fld, 1 << (ACC_FRAC - HALF));
        Word32::Limbs { hi: ctx.sum(&[s_sh, fld_sh, low_hi]), lo: low_lo }
    };
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
    let one_m_any = ctx.ksub(1, &any_ovf);
    let one_m_z = ctx.ksub(1, z);
    let t2a = ctx.mul(&one_m_any, &one_m_z);
    match fin {
        Word32::Full(fin) => {
            let inf_a = ctx.scale(&inf_sign, 1 << (ACC_WORD_BITS - 1));
            let inf = ctx.addk(&inf_a, T_FINITE << ACC_FRAC);
            let t1 = ctx.mul(&any_ovf, &inf);
            let t2 = ctx.mul(&t2a, &fin);
            Word32::Full(ctx.add(&t1, &t2))
        }
        Word32::Limbs { hi, lo } => {
            // inf = (inf_sign 2^15 + T_FINITE 2^7) 2^16 + 0
            let inf_a = ctx.scale(&inf_sign, 1 << (ACC_WORD_BITS - 1 - HALF));
            let inf_hi = ctx.addk(&inf_a, T_FINITE << (ACC_FRAC - HALF));
            let t1 = ctx.mul(&any_ovf, &inf_hi);
            let t2 = ctx.mul(&t2a, &hi);
            let y_hi = ctx.add(&t1, &t2);
            let y_lo = ctx.mul(&t2a, &lo);
            Word32::Limbs { hi: y_hi, lo: y_lo }
        }
    }
}

/// Epilogue (`checker/gadgets.py::Epilogue`): `f32_to_bf16` round-to-nearest-even on the final word.
/// `y32 = hi 2^L + lo` (L = 16), `hi = 2 hi2 + lsb`, `lo + lsb + (2^(L-1) - 1) = carry 2^L + rem`,
/// `y16 = hi + carry < 2^16`.  Over a 31-bit field the word's limbs ARE `hi` and `lo` (no split identity); they
/// are range-checked here as the contract requires (`hints: hi, lo: bits(16)`).
pub fn epilogue<C: Ctx>(ctx: &mut C, y32: &Word32<C::V>, tag: &str) -> C::V {
    let l = ACC_FRAC - FRAC;
    assert_eq!(l, HALF);
    let (hi, lo) = match y32 {
        Word32::Full(y) => {
            let hi = ctx.hint(&format!("{tag}.hi"), &[y], &|v| v[0] >> l);
            let lo = ctx.hint(&format!("{tag}.lo"), &[y], &|v| v[0] & ((1 << l) - 1));
            (hi, lo)
        }
        Word32::Limbs { hi, lo } => (hi.clone(), lo.clone()),
    };
    ctx.rng(&format!("{tag}.hi"), &hi, 0, 1 << WORD_BITS);
    ctx.rng(&format!("{tag}.lo"), &lo, 0, 1 << l);
    if let Word32::Full(y) = y32 {
        let hi_sh = ctx.scale(&hi, 1 << l);
        let sum = ctx.add(&hi_sh, &lo);
        ctx.eq(&format!("{tag}.split"), &sum, y);
    }
    let hi2 = ctx.hint(&format!("{tag}.hi2"), &[&hi], &|v| v[0] >> 1);
    let lsb = ctx.hint(&format!("{tag}.lsb"), &[&hi], &|v| v[0] & 1);
    let carry = ctx.hint(&format!("{tag}.carry"), &[&lo, &lsb], &|v| (v[0] + v[1] + ((1 << (l - 1)) - 1)) >> l);
    let rem = ctx.hint(&format!("{tag}.rem"), &[&lo, &lsb], &|v| (v[0] + v[1] + ((1 << (l - 1)) - 1)) & ((1 << l) - 1));
    ctx.rng(&format!("{tag}.hi2"), &hi2, 0, 1 << (WORD_BITS - 1));
    ctx.boolean(&format!("{tag}.lsb"), &lsb);
    ctx.boolean(&format!("{tag}.carry"), &carry);
    ctx.rng(&format!("{tag}.rem"), &rem, 0, 1 << l);
    let hi2_2 = ctx.scale(&hi2, 2);
    let lhs = ctx.add(&hi2_2, &lsb);
    ctx.eq(&format!("{tag}.lsb_split"), &lhs, &hi);
    let c_sh = ctx.scale(&carry, 1 << l);
    let lhs = ctx.add(&c_sh, &rem);
    let rhs = ctx.add(&lo, &lsb);
    let rhs = ctx.addk(&rhs, (1 << (l - 1)) - 1);
    ctx.eq(&format!("{tag}.round"), &lhs, &rhs);
    let y16 = ctx.add(&hi, &carry);
    ctx.rng(&format!("{tag}.y16"), &y16, 0, 1 << WORD_BITS);
    y16
}

// ---------------------------------------------------------------------------------------------------------------
// The unit
// ---------------------------------------------------------------------------------------------------------------

/// `check_step_expr` with the public words as the first 1 + 2k + 1 columns and `y = word` as the final identity
/// (the transition-unit AIR, Goldilocks).
pub fn unit_program<C: Ctx>(ctx: &mut C, io: &UnitIo) {
    unit_program_with(ctx, io, true);
}

/// The unit with (`bind_y`) or without the public `y` column and its identity; returns the incoming accumulator
/// column(s) and the materialised output word (`pack.y`, or `pack.y_hi` / `pack.y_lo` over a 31-bit field),
/// which the chain links to the next row.
pub fn unit_program_with<C: Ctx>(ctx: &mut C, io: &UnitIo, bind_y: bool) -> (Word32<C::V>, Word32<C::V>) {
    let c = if ctx.small_field() {
        let hi = ctx.hint("x.c_hi", &[], &|_| (io.c >> HALF) as i128);
        let lo = ctx.hint("x.c_lo", &[], &|_| (io.c & ((1 << HALF) - 1)) as i128);
        Word32::Limbs { hi, lo }
    } else {
        Word32::Full(ctx.hint("x.c", &[], &|_| io.c as i128))
    };
    let a: Vec<C::V> = (0..K).map(|i| ctx.hint(&format!("x.a[{i}]"), &[], &|_| io.a[i] as i128)).collect();
    let b: Vec<C::V> = (0..K).map(|i| ctx.hint(&format!("x.b[{i}]"), &[], &|_| io.b[i] as i128)).collect();
    let y = if bind_y { Some(ctx.hint("y", &[], &|_| io.y as i128)) } else { None };
    match &c {
        Word32::Full(c) => ctx.rng("x.c", c, 0, 1 << ACC_WORD_BITS),
        Word32::Limbs { hi, lo } => {
            ctx.rng("x.c_hi", hi, 0, 1 << HALF);
            ctx.rng("x.c_lo", lo, 0, 1 << HALF);
        }
    }
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
    let word = match word {
        Word32::Full(w) => Word32::Full(ctx.mat("pack.y", &w)),
        Word32::Limbs { hi, lo } => Word32::Limbs { hi: ctx.mat("pack.y_hi", &hi), lo: ctx.mat("pack.y_lo", &lo) },
    };
    if let Some(y) = y {
        match &word {
            Word32::Full(w) => ctx.eq("y", w, &y),
            Word32::Limbs { hi, lo } => {
                // the unit AIR is Goldilocks-only today; a 31-bit unit AIR would carry two public y limbs
                let hi_sh = ctx.scale(hi, 1 << HALF);
                let w = ctx.add(&hi_sh, lo);
                ctx.eq("y", &w, &y);
            }
        }
    }
    (c, word)
}

// ---------------------------------------------------------------------------------------------------------------
// One gadget in isolation (the red team's gadget-level battery cases)
// ---------------------------------------------------------------------------------------------------------------

/// A gadget-level case (`battery_babybear`, level `gadget`: the z3 models): the gadget's inputs as free columns
/// pinned to the contract's input ranges (the premises the composition guarantees -- `V_PREMISE` for `v`), its
/// hints as columns the case overrides (names carry the tag the unit uses: `acc.*`, `cast.*`, `align_state.*`,
/// `align_prod.*`, `lead.*`, `norm.*`), and every output bound to the claimed value.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct GadgetCase {
    pub name: String,
    pub inputs: std::collections::BTreeMap<String, i128>,
    pub claimed: Vec<i128>,
}

fn gadget_input<C: Ctx>(ctx: &mut C, case: &GadgetCase, n: &str, lo: i128, hi: i128) -> C::V {
    let v = *case.inputs.get(n).unwrap_or_else(|| panic!("gadget {}: missing input {n}", case.name));
    let x = ctx.hint(&format!("in.{n}"), &[], &move |_| v);
    ctx.rng(&format!("in.{n}"), &x, lo, hi);
    x
}

pub fn gadget_program<C: Ctx>(ctx: &mut C, case: &GadgetCase) {
    let outs: Vec<C::V> = match case.name.as_str() {
        "fp32_decode" => {
            let lo = gadget_input(ctx, case, "c_lo", 0, 1 << HALF);
            let hi = gadget_input(ctx, case, "c_hi", 0, 1 << HALF);
            let c = if ctx.small_field() {
                Word32::Limbs { hi, lo }
            } else {
                let hs = ctx.scale(&hi, 1 << HALF);
                Word32::Full(ctx.add(&hs, &lo))
            };
            let (s, e_hat, m) = fp32_decode(ctx, &c, "acc");
            vec![s, e_hat, m]
        }
        "epilogue" => {
            let lo = gadget_input(ctx, case, "y32_lo", 0, 1 << HALF);
            let hi = gadget_input(ctx, case, "y32_hi", 0, 1 << HALF);
            let y = if ctx.small_field() {
                Word32::Limbs { hi, lo }
            } else {
                let hs = ctx.scale(&hi, 1 << HALF);
                Word32::Full(ctx.add(&hs, &lo))
            };
            vec![epilogue(ctx, &y, "cast")]
        }
        "align_state" => {
            let m = gadget_input(ctx, case, "M", 0, 1 << ACC_SIG);
            let d = gadget_input(ctx, case, "d", 0, SHIFT_ROWS);
            vec![align_state(ctx, &m, &d, "align_state")]
        }
        "align_product" => {
            let u = gadget_input(ctx, case, "u", 0, 1 << (2 * SIG));
            let d = gadget_input(ctx, case, "d", 0, SHIFT_ROWS);
            vec![align_product(ctx, &u, &d, "align_prod")]
        }
        "leading_bit" => {
            let v = gadget_input(ctx, case, "v", 0, S_MAX + 1);
            vec![leading_bit(ctx, &v, "lead")]
        }
        "normalise" => {
            let e = gadget_input(ctx, case, "E", FLOOR, LAST + 1);
            let l = gadget_input(ctx, case, "l", 0, L_HI + 1);
            let v = gadget_input(ctx, case, "v", 0, S_MAX + 1);
            let (e1, m) = normalise(ctx, &e, &l, &v, "norm");
            vec![e1, m]
        }
        "lead_norm" => {
            let e = gadget_input(ctx, case, "E", FLOOR, LAST + 1);
            let v = gadget_input(ctx, case, "v", 0, S_MAX + 1);
            let l = leading_bit(ctx, &v, "lead");
            let (e1, m) = normalise(ctx, &e, &l, &v, "norm");
            vec![e1, m]
        }
        other => panic!("gadget_program: unknown gadget {other}"),
    };
    assert_eq!(outs.len(), case.claimed.len(), "gadget {}: {} outputs, {} claimed", case.name, outs.len(), case.claimed.len());
    for (i, (o, &c)) in outs.iter().zip(case.claimed.iter()).enumerate() {
        let k = ctx.konst(c);
        ctx.eq(&format!("out[{i}]"), o, &k);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The VU chain: one row per transition unit, the accumulator carried row to row
// ---------------------------------------------------------------------------------------------------------------

/// Units per K=1536 VU.
pub const STEPS: u64 = 96;

/// One row of the chain: the unit's words, its position in the VU, and the VU's claimed output word (read on
/// the last row only).  `c` is the incoming accumulator (0 on the first row; the previous row's output word
/// otherwise -- the transition constraint enforces it, witness generation supplies it).
#[derive(Clone, Debug, Default)]
pub struct ChainIo {
    pub c: u64,
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub step: u64,
    pub y16: u64,
}

/// Names of the chain's own columns (`Layout::col` finds them; `ChainAir` reads them on the next row too).
pub const COL_STEP: &str = "vu.step";
pub const COL_FIRST: &str = "vu.first";
pub const COL_LAST: &str = "vu.last";
pub const COL_INV: &str = "vu.inv";
pub const COL_CLAIM: &str = "vu.y16";

/// The row-local part of the chain AIR: the unit, the VU-position selectors, the epilogue and the output
/// binding.  Row-to-row constraints (`next.first = last`, step increment, accumulator linkage) are emitted by
/// `air::ChainAir::eval` and named by `air::chain_tail_names`.
///
/// Position columns: `step in [0, 96)`, `first = [step == 0]`, `last = [step == 95]`.  `last` is pinned by an
/// inverse (`(step - 95) inv = 1 - last`, `last (step - 95) = 0`), so with the increment rule of the transition
/// constraint `step` is an exact counter and the epilogue identity fires on exactly the 96th row of every VU.
/// `first c = 0` is the chain start (`c_0 = 0`), `last (y16 - claim) = 0` the chain end.
pub fn chain_program<C: Ctx>(ctx: &mut C, io: &ChainIo) -> Word32<C::V> {
    let unit = UnitIo { c: io.c, a: io.a.clone(), b: io.b.clone(), y: 0 };
    let (c_in, word) = unit_program_with(ctx, &unit, false);
    let c_cols = c_in.cols();
    let p = ctx.prime();
    let step = ctx.hint(COL_STEP, &[], &|_| io.step as i128);
    let first = ctx.hint(COL_FIRST, &[], &|_| (io.step == 0) as i128);
    let last = ctx.hint(COL_LAST, &[], &|_| (io.step == STEPS - 1) as i128);
    let inv = ctx.hint(COL_INV, &[&step], &move |v| crate::ctx::inv_modp(v[0] - (STEPS as i128 - 1), p));
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
    for (i, c) in c_cols.iter().enumerate() {
        let fc = ctx.mul(&first, c);
        ctx.eq(&format!("vu.first_c[{i}]"), &fc, &zero);
    }
    let y16 = epilogue(ctx, &word, "cast");
    let claim = ctx.hint(COL_CLAIM, &[], &|_| io.y16 as i128);
    let diff = ctx.sub(&y16, &claim);
    let out = ctx.mul(&last, &diff);
    ctx.eq("vu.out", &out, &zero);
    word
}
