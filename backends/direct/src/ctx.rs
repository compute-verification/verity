//! The two interpreters of `program::Ctx`: integer witness generation (`IntCtx`) and symbolic constraint
//! emission on any Plonky3 `AirBuilder` (`AirCtx`), plus the provided predicates both share structurally
//! (range checks as 2-bit limbs, lookups as one-hot selectors) so the column layout agrees by construction.

use std::collections::HashMap;

use p3_air::AirBuilder;
use p3_field::AbstractField;

use crate::program::{Ctx, Table, MAX_DEG};

pub const GOLDILOCKS_P: u128 = 0xFFFF_FFFF_0000_0001;

fn modp(v: i128) -> u64 {
    let p = GOLDILOCKS_P as i128;
    (((v % p) + p) % p) as u64
}

fn mulmod(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % GOLDILOCKS_P) as u64
}

fn inv_modp(v: i128) -> i128 {
    let mut base = modp(v);
    if base == 0 {
        return 0; // no inverse: the constraint prod * inv = 1 then fails, which is the intent
    }
    let mut e = GOLDILOCKS_P as u64 - 2;
    let mut acc = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            acc = mulmod(acc, base);
        }
        base = mulmod(base, base);
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
}

impl<P: Prim> Ctx for P {
    type V = P::V;
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
        limbs(self, name, &y, w);
        if k == 0 {
            // exact power of two
        } else if k <= 2 {
            // exclude the top k values by an inverse: prod_j (y - (n + j)) * inv = 1
            let mut prod = self.addk(&y, -n);
            for j in 1..k {
                let f = self.addk(&y, -(n + j));
                prod = self.mul_raw(&prod, &f);
            }
            let inv = self.hint(&format!("{name}.inv"), &[&prod], &|v| inv_modp(v[0]));
            let pi = self.mul_raw(&prod, &inv);
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

fn limbs<P: Prim + ?Sized>(ctx: &mut P, name: &str, y: &P::V, w: i128) {
    let nl = ((w + 1) / 2) as usize;
    let mut acc = ctx.konst(0);
    for i in 0..nl {
        let shift = 2 * i as u32;
        let mask = (1i128 << w) - 1;
        let limb = ctx.hint(&format!("{name}.limb[{i}]"), &[y], &move |v| ((v[0] & mask) >> shift) & 3);
        if i == nl - 1 && w % 2 == 1 {
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
#[derive(Default)]
pub struct IntCtx {
    pub cols: Vec<i128>,
    pub names: Vec<String>,
    pub overrides: HashMap<String, i128>,
    pub violations: Vec<String>,
    pub n_hints: usize,
    pub n_mat: usize,
    pub n_constraints: usize,
    pub hint_values: Vec<(String, i128)>,
    pub record_hints: bool,
}

impl IntCtx {
    pub fn new() -> Self {
        Self::default()
    }
    fn push(&mut self, name: &str, v: i128) -> IVal {
        self.cols.push(v);
        self.names.push(name.to_string());
        IVal { v, deg: 1 }
    }
}

impl Prim for IntCtx {
    type V = IVal;
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
        let v = match self.overrides.get(name) {
            Some(&o) => o,
            None => f(&vals),
        };
        self.n_hints += 1;
        let realisation = name.contains(".limb[")
            || (name.contains(".sel[") && !name.contains(".max.sel["))
            || name.ends_with(".inv")
            || name.ends_with(".slack");
        if self.record_hints && !realisation {
            self.hint_values.push((name.to_string(), v));
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
        if modp(a.v) != modp(b.v) {
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
    pub constraint_names: Option<Vec<String>>,
}

impl<'a, AB: AirBuilder> AirCtx<'a, AB> {
    pub fn new(builder: &'a mut AB, row: &'a [AB::Var]) -> Self {
        Self { builder, row, cursor: 0, constraint_names: None }
    }
    fn next(&mut self) -> AB::Expr {
        let v = self.row[self.cursor];
        self.cursor += 1;
        v.into()
    }
    fn k(k: i128) -> AB::Expr {
        if k >= 0 {
            AB::Expr::from_canonical_u64(k as u64)
        } else {
            -AB::Expr::from_canonical_u64((-k) as u64)
        }
    }
    fn note(&mut self, name: &str) {
        if let Some(n) = self.constraint_names.as_mut() {
            n.push(name.to_string());
        }
    }
}

impl<'a, AB: AirBuilder> Prim for AirCtx<'a, AB> {
    type V = SVal<AB::Expr>;
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
}
