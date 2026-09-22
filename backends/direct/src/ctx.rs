//! The two interpreters of `program::Ctx`: integer witness generation (`IntCtx`) and symbolic constraint
//! emission on any Plonky3 `AirBuilder` (`AirCtx`), plus the provided predicates both share structurally
//! (range checks as 2-bit limbs, lookups as one-hot selectors) so the column layout agrees by construction.

use std::collections::HashMap;

use p3_air::AirBuilder;
use p3_field::AbstractField;

use crate::lookup::{Needs, Query, TableSet, GROUP_DEG, INVALID, SPLIT_BITS};
use crate::program::{Ctx, Table, MAX_DEG};

/// How range checks and lookups are realised (PROTOCOL.md sections 12 and 14).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Realisation {
    /// 2-bit limbs and one-hot selectors (section 12).
    Limbs,
    /// LogUp queries against preprocessed tables (section 14).
    LogUp,
}

pub const GOLDILOCKS_P: u128 = 0xFFFF_FFFF_0000_0001;
pub const BABYBEAR_P: u128 = 0x7800_0001; // 15 * 2^27 + 1 = 2013265921

/// The trace field.  `small` selects the 31-bit realisation of the program (`program.rs`: F1 direct `lt_pow`,
/// F3 top-bit exclusion in `Normalise`, F4 16-bit word limbs, the signed-sum premise bound).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FieldKind {
    Goldilocks,
    BabyBear,
}

impl FieldKind {
    pub fn prime(self) -> u128 {
        match self {
            FieldKind::Goldilocks => GOLDILOCKS_P,
            FieldKind::BabyBear => BABYBEAR_P,
        }
    }
    pub fn small(self) -> bool {
        matches!(self, FieldKind::BabyBear)
    }
    pub fn name(self) -> &'static str {
        match self {
            FieldKind::Goldilocks => "goldilocks",
            FieldKind::BabyBear => "babybear",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "goldilocks" | "gl" => Some(FieldKind::Goldilocks),
            "babybear" | "bb" => Some(FieldKind::BabyBear),
            _ => None,
        }
    }
}

pub fn modp(v: i128, p: u128) -> u64 {
    let p = p as i128;
    (((v % p) + p) % p) as u64
}

fn mulmod(a: u64, b: u64, p: u128) -> u64 {
    ((a as u128 * b as u128) % p) as u64
}

pub fn inv_modp(v: i128, p: u128) -> i128 {
    let mut base = modp(v, p);
    if base == 0 {
        return 0; // no inverse: the constraint prod * inv = 1 then fails, which is the intent
    }
    let mut e = p as u64 - 2;
    let mut acc = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            acc = mulmod(acc, base, p);
        }
        base = mulmod(base, base, p);
        e >>= 1;
    }
    acc as i128
}

// ---------------------------------------------------------------------------------------------------------------
// Shared predicate realisations (structure identical in both contexts)
// ---------------------------------------------------------------------------------------------------------------

/// The primitive operations a context provides; `Ctx` is implemented once on top of them (below).
pub trait Prim {
    type V: Clone;
    /// True for a 31-bit trace field (the program then uses its wrap-free realisations).
    fn small_field(&self) -> bool;
    /// The trace field's prime (for the exclusion inverses of `rng`).
    fn prime(&self) -> u128;
    fn konst(&mut self, k: i128) -> Self::V;
    fn deg(&self, v: &Self::V) -> usize;
    fn add(&mut self, a: &Self::V, b: &Self::V) -> Self::V;
    fn sub(&mut self, a: &Self::V, b: &Self::V) -> Self::V;
    fn mul_raw(&mut self, a: &Self::V, b: &Self::V) -> Self::V;
    fn scale(&mut self, a: &Self::V, k: i128) -> Self::V;
    fn addk(&mut self, a: &Self::V, k: i128) -> Self::V;
    fn hint(&mut self, name: &str, deps: &[&Self::V], f: &dyn Fn(&[i128]) -> i128) -> Self::V;
    fn mat(&mut self, name: &str, a: &Self::V) -> Self::V;
    fn eq_raw(&mut self, name: &str, a: &Self::V, b: &Self::V);
    /// b (b - 1) = 0
    fn boolean_raw(&mut self, name: &str, b: &Self::V);
    /// x (x - 1) (x - 2) (x - 3) = 0
    fn limb4_raw(&mut self, name: &str, x: &Self::V);
    /// LogUp realisation of the range checks and lookups (section 14) instead of limbs / one-hot selectors.
    fn lookup_mode(&self) -> bool;
    /// LogUp query `y in T_w = [0, 2^w)`.
    fn range_query(&mut self, name: &str, y: &Self::V, w: u32);
    /// LogUp query `(key, outs..) in table`.
    fn table_query(&mut self, name: &str, table: &Table, key: &Self::V, outs: &[Self::V]);
}

/// Widest range query in one piece; wider ranges are split into a 16-bit low part and a high part.
fn range_query_split<P: Prim + ?Sized>(ctx: &mut P, name: &str, y: &P::V, w: u32) {
    if w <= SPLIT_BITS {
        ctx.range_query(name, y, w);
        return;
    }
    let mask = (1i128 << SPLIT_BITS) - 1;
    let lo = ctx.hint(&format!("{name}.lo16"), &[y], &move |v| v[0] & mask);
    let hi = ctx.hint(&format!("{name}.hi16"), &[y], &move |v| v[0] >> SPLIT_BITS);
    let hs = ctx.scale(&hi, 1i128 << SPLIT_BITS);
    let rec = ctx.add(&lo, &hs);
    ctx.eq_raw(&format!("{name}.split"), &rec, y);
    ctx.range_query(&format!("{name}.lo16"), &lo, SPLIT_BITS);
    ctx.range_query(&format!("{name}.hi16"), &hi, w - SPLIT_BITS);
}

impl<P: Prim> Ctx for P {
    type V = P::V;
    fn small_field(&self) -> bool {
        Prim::small_field(self)
    }
    fn prime(&self) -> u128 {
        Prim::prime(self)
    }
    fn konst(&mut self, k: i128) -> Self::V {
        Prim::konst(self, k)
    }
    fn add(&mut self, a: &Self::V, b: &Self::V) -> Self::V {
        Prim::add(self, a, b)
    }
    fn sub(&mut self, a: &Self::V, b: &Self::V) -> Self::V {
        Prim::sub(self, a, b)
    }
    fn scale(&mut self, a: &Self::V, k: i128) -> Self::V {
        Prim::scale(self, a, k)
    }
    fn addk(&mut self, a: &Self::V, k: i128) -> Self::V {
        Prim::addk(self, a, k)
    }
    fn hint(&mut self, name: &str, deps: &[&Self::V], f: &dyn Fn(&[i128]) -> i128) -> Self::V {
        Prim::hint(self, name, deps, f)
    }
    fn mat(&mut self, name: &str, a: &Self::V) -> Self::V {
        Prim::mat(self, name, a)
    }
    fn mul(&mut self, a: &Self::V, b: &Self::V) -> Self::V {
        let (mut a, mut b) = (a.clone(), b.clone());
        while self.deg(&a) + self.deg(&b) > MAX_DEG {
            if self.deg(&a) >= self.deg(&b) {
                a = self.mat("_mat", &a);
            } else {
                b = self.mat("_mat", &b);
            }
        }
        self.mul_raw(&a, &b)
    }
    fn eq(&mut self, name: &str, a: &Self::V, b: &Self::V) {
        let a = if self.deg(a) > MAX_DEG { self.mat("_mat", a) } else { a.clone() };
        let b = if self.deg(b) > MAX_DEG { self.mat("_mat", b) } else { b.clone() };
        self.eq_raw(name, &a, &b)
    }
    fn boolean(&mut self, name: &str, b: &Self::V) {
        let b = if 2 * self.deg(b) > MAX_DEG { self.mat("_mat", b) } else { b.clone() };
        self.boolean_raw(name, &b)
    }
    fn rng(&mut self, name: &str, x: &Self::V, lo: i128, hi: i128) {
        let n = hi - lo;
        assert!(n > 0, "{name}: empty range");
        let y = if lo == 0 { x.clone() } else { self.addk(x, -lo) };
        let y = if self.deg(&y) > MAX_DEG { self.mat(name, &y) } else { y };
        if n == 1 {
            let zero = Prim::konst(self, 0);
            self.eq_raw(name, &y, &zero);
            return;
        }
        let w = 128 - ((n - 1) as u128).leading_zeros() as i128;
        let full = 1i128 << w;
        let k = full - n;
        if self.lookup_mode() {
            // Section 14: `y in T_w`, and for a non-power-of-two range the second query `n - 1 - y in T_w`
            // (their intersection is exactly `[0, n)` as `n <= 2^w`).  Boolean ranges stay a polynomial.
            if n == 2 {
                self.boolean(name, &y);
                return;
            }
            let y = if self.deg(&y) > GROUP_DEG { self.mat(name, &y) } else { y };
            range_query_split(self, name, &y, w as u32);
            if k != 0 {
                let y2 = self.scale(&y, -1);
                let y2 = self.addk(&y2, n - 1);
                range_query_split(self, &format!("{name}.hi"), &y2, w as u32);
            }
            return;
        }
        limbs(self, name, &y, w);
        if k == 0 {
            // exact power of two
        } else if k <= 2 {
            // exclude the top k values by an inverse: prod_j (y - (n + j)) * inv = 1
            let mut prod = self.addk(&y, -n);
            for j in 1..k {
                let f = self.addk(&y, -(n + j));
                prod = self.mul(&prod, &f);
            }
            let p = self.prime();
            let inv = self.hint(&format!("{name}.inv"), &[&prod], &move |v| inv_modp(v[0], p));
            let pi = self.mul(&prod, &inv);
            let one = Prim::konst(self, 1);
            self.eq_raw(&format!("{name}.excl"), &pi, &one);
        } else {
            // second decomposition of (n - 1 - y)
            let y2 = self.scale(&y, -1);
            let y2 = self.addk(&y2, n - 1);
            limbs(self, &format!("{name}.hi"), &y2, w);
        }
    }
    fn lookup(&mut self, name: &str, table: &Table, key: &Self::V) -> Vec<Self::V> {
        let nrows = table.keys.len();
        let keys = table.keys.clone();
        let saturate = table.saturate;
        if self.lookup_mode() {
            // Section 14: the outputs are hints, the tuple `(key, outs)` is one query on the table (saturating
            // rows are expanded up to the slack bound, so saturation needs no selector or slack range).
            let key = if self.deg(key) > GROUP_DEG { self.mat(name, key) } else { key.clone() };
            let ncols = table.rows[0].len();
            let mut outs = Vec::with_capacity(ncols);
            for c in 0..ncols {
                let keys = keys.clone();
                let rows = table.rows.clone();
                let o = self.hint(&format!("{name}.out[{c}]"), &[&key], &move |v| {
                    let k = v[0];
                    let idx = if saturate && k >= keys[nrows - 1] { Some(nrows - 1) } else { keys.iter().position(|&kk| kk == k) };
                    idx.map(|i| rows[i][c]).unwrap_or(0)
                });
                outs.push(o);
            }
            self.table_query(name, table, &key, &outs);
            return outs;
        }
        let mut sel = Vec::with_capacity(nrows);
        for j in 0..nrows {
            let keys = keys.clone();
            let s = self.hint(&format!("{name}.sel[{j}]"), &[key], &move |v| {
                let k = v[0];
                let idx = if saturate && k >= keys[nrows - 1] {
                    Some(nrows - 1)
                } else {
                    keys.iter().position(|&kk| kk == k)
                };
                (idx == Some(j)) as i128
            });
            sel.push(s);
        }
        for (j, s) in sel.iter().enumerate() {
            self.boolean_raw(&format!("{name}.sel[{j}]"), s);
        }
        let onehot = self.sum(&sel);
        let one = Prim::konst(self, 1);
        self.eq_raw(&format!("{name}.onehot"), &onehot, &one);
        let exact = if saturate { nrows - 1 } else { nrows };
        let mut terms = Vec::with_capacity(exact);
        for j in 0..exact {
            let diff = self.addk(key, -keys[j]);
            terms.push(self.mul(&sel[j], &diff));
        }
        let matched = self.sum(&terms);
        let zero = Prim::konst(self, 0);
        self.eq_raw(&format!("{name}.key"), &matched, &zero);
        if saturate {
            let last = keys[nrows - 1];
            let slack = self.hint(&format!("{name}.slack"), &[key], &move |v| (v[0] - last).max(0));
            self.rng(&format!("{name}.slack"), &slack, 0, table.slack_hi);
            let d = self.addk(key, -last);
            let d = self.sub(&d, &slack);
            let sd = self.mul(&sel[nrows - 1], &d);
            self.eq_raw(&format!("{name}.sat"), &sd, &zero);
        }
        let ncols = table.rows[0].len();
        (0..ncols)
            .map(|c| {
                let parts: Vec<Self::V> = (0..nrows).map(|j| self.scale(&sel[j], table.rows[j][c])).collect();
                self.sum(&parts)
            })
            .collect()
    }
}

/// Range limbs: LIMB_BITS = 2 (degree-4 limb constraint x(x-1)(x-2)(x-3)); LIMB_BITS = 1 gives booleans.
pub const LIMB_BITS: i128 = 2;

fn limbs<P: Prim + ?Sized>(ctx: &mut P, name: &str, y: &P::V, w: i128) {
    let nl = ((w + LIMB_BITS - 1) / LIMB_BITS) as usize;
    let mut acc = ctx.konst(0);
    for i in 0..nl {
        let shift = (LIMB_BITS as u32) * i as u32;
        let mask = (1i128 << w) - 1;
        let lmask = (1i128 << LIMB_BITS) - 1;
        let limb = ctx.hint(&format!("{name}.limb[{i}]"), &[y], &move |v| ((v[0] & mask) >> shift) & lmask);
        let top_bits = w - shift as i128;
        if LIMB_BITS == 1 || (i == nl - 1 && top_bits == 1) {
            ctx.boolean_raw(&format!("{name}.limb[{i}]"), &limb);
        } else {
            ctx.limb4_raw(&format!("{name}.limb[{i}]"), &limb);
        }
        let t = ctx.scale(&limb, 1i128 << shift);
        acc = ctx.add(&acc, &t);
    }
    ctx.eq_raw(&format!("{name}.limbs"), &acc, y);
}

// ---------------------------------------------------------------------------------------------------------------
// Integer witness context
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct IVal {
    pub v: i128,
    pub deg: usize,
}

/// Generates one row.  `overrides` replaces named hints (the negative tests' mutations: everything downstream
/// -- limbs, selectors, materialised copies -- is recomputed honestly from the mutated value, exactly as the
/// Python checker recomputes its expressions from mutated hints).  Integer predicate violations are recorded in
/// `violations` (the analogue of `Reject`); the field evaluator in `air.rs` is the authoritative check.
pub struct IntCtx {
    pub kind: FieldKind,
    pub cols: Vec<i128>,
    /// Column names in allocation order; not recorded when `record_names` is false (trace generation).
    pub names: Vec<String>,
    pub record_names: bool,
    pub overrides: HashMap<String, i128>,
    pub violations: Vec<String>,
    pub n_hints: usize,
    pub n_mat: usize,
    pub n_constraints: usize,
    pub hint_values: Vec<(String, i128)>,
    pub record_hints: bool,
    pub mode: Realisation,
    /// LogUp: the tables (`None` = collecting run: `needs` is filled, indices are not computed).
    pub tables: Option<std::sync::Arc<TableSet>>,
    pub needs: Needs,
    /// LogUp: this row's query indices into the preprocessed table (`INVALID` for an out-of-table value).
    pub queries: Vec<u32>,
    pub query_names: Vec<String>,
    pub query_degs: Vec<usize>,
}

impl Default for IntCtx {
    fn default() -> Self {
        Self::new_kind(FieldKind::Goldilocks)
    }
}

impl IntCtx {
    /// Goldilocks, names recorded (the layout / test constructor).
    pub fn new() -> Self {
        Self::default()
    }
    pub fn new_kind(kind: FieldKind) -> Self {
        Self {
            kind,
            cols: Vec::with_capacity(3072),
            names: vec![],
            record_names: true,
            overrides: HashMap::new(),
            violations: vec![],
            n_hints: 0,
            n_mat: 0,
            n_constraints: 0,
            hint_values: vec![],
            record_hints: false,
            mode: Realisation::Limbs,
            tables: None,
            needs: Needs::default(),
            queries: vec![],
            query_names: vec![],
            query_degs: vec![],
        }
    }
    /// The fast constructor for trace generation: no names, no hint log.
    pub fn for_trace(kind: FieldKind) -> Self {
        let mut c = Self::new_kind(kind);
        c.record_names = false;
        c
    }
    /// LogUp realisation with the given tables (`None`: a collecting run that discovers the tables needed).
    pub fn with_lookup(mut self, tables: Option<std::sync::Arc<TableSet>>) -> Self {
        self.mode = Realisation::LogUp;
        self.tables = tables;
        self
    }
    fn push_query(&mut self, name: &str, idx: u32, deg: usize) {
        self.queries.push(idx);
        self.query_degs.push(deg);
        if self.record_names {
            self.query_names.push(name.to_string());
        }
    }
    fn push(&mut self, name: &str, v: i128) -> IVal {
        self.cols.push(v);
        if self.record_names {
            self.names.push(name.to_string());
        }
        IVal { v, deg: 1 }
    }
}

impl Prim for IntCtx {
    type V = IVal;
    fn small_field(&self) -> bool {
        self.kind.small()
    }
    fn prime(&self) -> u128 {
        self.kind.prime()
    }
    fn konst(&mut self, k: i128) -> IVal {
        IVal { v: k, deg: 0 }
    }
    fn deg(&self, v: &IVal) -> usize {
        v.deg
    }
    fn add(&mut self, a: &IVal, b: &IVal) -> IVal {
        IVal { v: a.v + b.v, deg: a.deg.max(b.deg) }
    }
    fn sub(&mut self, a: &IVal, b: &IVal) -> IVal {
        IVal { v: a.v - b.v, deg: a.deg.max(b.deg) }
    }
    fn mul_raw(&mut self, a: &IVal, b: &IVal) -> IVal {
        IVal { v: a.v.wrapping_mul(b.v), deg: a.deg + b.deg }
    }
    fn scale(&mut self, a: &IVal, k: i128) -> IVal {
        IVal { v: a.v.wrapping_mul(k), deg: a.deg }
    }
    fn addk(&mut self, a: &IVal, k: i128) -> IVal {
        IVal { v: a.v + k, deg: a.deg }
    }
    fn hint(&mut self, name: &str, deps: &[&IVal], f: &dyn Fn(&[i128]) -> i128) -> IVal {
        let vals: Vec<i128> = deps.iter().map(|d| d.v).collect();
        let v = if self.overrides.is_empty() {
            f(&vals)
        } else {
            match self.overrides.get(name) {
                Some(&o) => o,
                None => f(&vals),
            }
        };
        self.n_hints += 1;
        if self.record_hints {
            let realisation = name.contains(".limb[")
                || (name.contains(".sel[") && !name.contains(".max.sel["))
                || name.ends_with(".inv")
                || name.ends_with(".slack")
                || name.contains(".out[")
                || name.ends_with(".lo16")
                || name.ends_with(".hi16");
            if !realisation {
                self.hint_values.push((name.to_string(), v));
            }
        }
        self.push(name, v)
    }
    fn mat(&mut self, name: &str, a: &IVal) -> IVal {
        self.n_mat += 1;
        self.n_constraints += 1;
        self.push(name, a.v)
    }
    fn eq_raw(&mut self, name: &str, a: &IVal, b: &IVal) {
        self.n_constraints += 1;
        // field semantics (the exclusion inverses are field elements); every other honest value is small
        let p = self.kind.prime();
        if modp(a.v, p) != modp(b.v, p) {
            self.violations.push(format!("{name}: {} != {}", a.v, b.v));
        }
    }
    fn boolean_raw(&mut self, name: &str, b: &IVal) {
        self.n_constraints += 1;
        if b.v != 0 && b.v != 1 {
            self.violations.push(format!("{name}: {} is not a bit", b.v));
        }
    }
    fn limb4_raw(&mut self, name: &str, x: &IVal) {
        self.n_constraints += 1;
        if !(0..4).contains(&x.v) {
            self.violations.push(format!("{name}: {} is not a 2-bit limb", x.v));
        }
    }
    fn lookup_mode(&self) -> bool {
        self.mode == Realisation::LogUp
    }
    fn range_query(&mut self, name: &str, y: &IVal, w: u32) {
        // field semantics: the query is on the canonical representative (an adversarial negative value is a
        // large representative, out of the table)
        let v = modp(y.v, self.kind.prime()) as i128;
        let idx = match &self.tables {
            None => {
                self.needs.ranges.insert(w);
                0
            }
            Some(t) => t.range(w).index(v),
        };
        if idx == INVALID {
            self.violations.push(format!("{name}: {} not in T{w}", y.v));
        }
        self.push_query(name, idx, y.deg.max(1));
    }
    fn table_query(&mut self, name: &str, table: &Table, key: &IVal, outs: &[IVal]) {
        let v = modp(key.v, self.kind.prime()) as i128;
        // keys are small signed integers: fold the representative back
        let p = self.kind.prime() as i128;
        let v = if v > p / 2 { v - p } else { v };
        let idx = match &self.tables {
            None => {
                self.needs.fixed.entry(table.name.to_string()).or_insert_with(|| table.clone());
                0
            }
            Some(t) => {
                let spec = t.fixed(table.name);
                let idx = spec.index(v);
                if idx != INVALID {
                    // the tuple must match the table row exactly (the outputs are hints)
                    let ok = outs.iter().enumerate().all(|(c, o)| modp(o.v, p as u128) == modp(spec.out(v, c), p as u128));
                    if !ok {
                        self.violations.push(format!("{name}: outputs differ from the table row"));
                        INVALID
                    } else {
                        idx
                    }
                } else {
                    idx
                }
            }
        };
        if idx == INVALID {
            self.violations.push(format!("{name}: key {} not in {}", key.v, table.name));
        }
        let deg = outs.iter().map(|o| o.deg).fold(key.deg.max(1), usize::max);
        self.push_query(name, idx, deg);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Symbolic AIR context
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct SVal<E> {
    pub e: E,
    pub deg: usize,
}

pub struct AirCtx<'a, AB: AirBuilder> {
    pub builder: &'a mut AB,
    pub row: &'a [AB::Var],
    pub cursor: usize,
    pub kind: FieldKind,
    pub constraint_names: Option<Vec<String>>,
    pub mode: Realisation,
    pub tables: Option<std::sync::Arc<TableSet>>,
    /// LogUp: this row's queries in program order (consumed by `lookup::eval_lookup`).
    pub queries: Vec<Query<AB::Expr>>,
}

impl<'a, AB: AirBuilder> AirCtx<'a, AB> {
    /// Goldilocks realisation.
    pub fn new(builder: &'a mut AB, row: &'a [AB::Var]) -> Self {
        Self::with_kind(builder, row, FieldKind::Goldilocks)
    }
    pub fn with_kind(builder: &'a mut AB, row: &'a [AB::Var], kind: FieldKind) -> Self {
        Self { builder, row, cursor: 0, kind, constraint_names: None, mode: Realisation::Limbs, tables: None, queries: vec![] }
    }
    pub fn with_lookup(mut self, tables: std::sync::Arc<TableSet>) -> Self {
        self.mode = Realisation::LogUp;
        self.tables = Some(tables);
        self
    }
    fn next(&mut self) -> AB::Expr {
        let v = self.row[self.cursor];
        self.cursor += 1;
        v.into()
    }
    /// A constant: every constant of the program is below the trace prime by construction (the 31-bit realisation
    /// splits the 32-bit words), so the canonical conversion is exact.
    pub fn k(k: i128) -> AB::Expr {
        if k >= 0 {
            AB::Expr::from_canonical_u64(k as u64)
        } else {
            -AB::Expr::from_canonical_u64((-k) as u64)
        }
    }
    pub fn note(&mut self, name: &str) {
        if let Some(n) = self.constraint_names.as_mut() {
            n.push(name.to_string());
        }
    }
}

impl<'a, AB: AirBuilder> Prim for AirCtx<'a, AB> {
    type V = SVal<AB::Expr>;
    fn small_field(&self) -> bool {
        self.kind.small()
    }
    fn prime(&self) -> u128 {
        self.kind.prime()
    }
    fn konst(&mut self, k: i128) -> Self::V {
        SVal { e: Self::k(k), deg: 0 }
    }
    fn deg(&self, v: &Self::V) -> usize {
        v.deg
    }
    fn add(&mut self, a: &Self::V, b: &Self::V) -> Self::V {
        SVal { e: a.e.clone() + b.e.clone(), deg: a.deg.max(b.deg) }
    }
    fn sub(&mut self, a: &Self::V, b: &Self::V) -> Self::V {
        SVal { e: a.e.clone() - b.e.clone(), deg: a.deg.max(b.deg) }
    }
    fn mul_raw(&mut self, a: &Self::V, b: &Self::V) -> Self::V {
        SVal { e: a.e.clone() * b.e.clone(), deg: a.deg + b.deg }
    }
    fn scale(&mut self, a: &Self::V, k: i128) -> Self::V {
        SVal { e: a.e.clone() * Self::k(k), deg: a.deg }
    }
    fn addk(&mut self, a: &Self::V, k: i128) -> Self::V {
        SVal { e: a.e.clone() + Self::k(k), deg: a.deg }
    }
    fn hint(&mut self, _name: &str, _deps: &[&Self::V], _f: &dyn Fn(&[i128]) -> i128) -> Self::V {
        SVal { e: self.next(), deg: 1 }
    }
    fn mat(&mut self, name: &str, a: &Self::V) -> Self::V {
        let col = self.next();
        self.note(&format!("{name}.mat"));
        self.builder.assert_eq(col.clone(), a.e.clone());
        SVal { e: col, deg: 1 }
    }
    fn eq_raw(&mut self, name: &str, a: &Self::V, b: &Self::V) {
        self.note(name);
        self.builder.assert_eq(a.e.clone(), b.e.clone());
    }
    fn boolean_raw(&mut self, name: &str, b: &Self::V) {
        self.note(name);
        self.builder.assert_bool(b.e.clone());
    }
    fn limb4_raw(&mut self, name: &str, x: &Self::V) {
        self.note(name);
        let e = x.e.clone();
        let p = e.clone() * (e.clone() - AB::Expr::one()) * (e.clone() - AB::Expr::two()) * (e - Self::k(3));
        self.builder.assert_zero(p);
    }
    fn lookup_mode(&self) -> bool {
        self.mode == Realisation::LogUp
    }
    fn range_query(&mut self, name: &str, y: &Self::V, w: u32) {
        let tag = self.tables.as_ref().expect("AirCtx lookup tables").range(w).tag;
        self.queries.push(Query { name: name.to_string(), tag, vals: vec![y.e.clone()], deg: y.deg.max(1) });
    }
    fn table_query(&mut self, name: &str, table: &Table, key: &Self::V, outs: &[Self::V]) {
        let tag = self.tables.as_ref().expect("AirCtx lookup tables").fixed(table.name).tag;
        let mut vals = vec![key.e.clone()];
        vals.extend(outs.iter().map(|o| o.e.clone()));
        let deg = outs.iter().map(|o| o.deg).fold(key.deg.max(1), usize::max);
        self.queries.push(Query { name: name.to_string(), tag, vals, deg });
    }
}
