//! `verity/flock-ir-block/v1`: identical units of an IR subcircuit template (`flock-ir-unit/v1`
//! netlists, `verity_flock.ir_lower`) as ONE block R1CS, so the CPU prover and Flock-CUDA's host-witness mode prove it.
//!
//! Block `b` holds units `U b .. U b + U - 1` (instance-major over the instance file), unit `u` at the 2^unit_log-bit slot
//! `u`, so `k_log = unit_log + log2 U`. `A = I ⊗ (I_U ⊗ unit) + Δ` (and B), where Δ makes every unit's constant column a
//! copy of the one pinned column (slot 0's). Units past the file's last (the block count is a power of two, at least 8)
//! are units of all-zero inputs.
//!
//! Public regions: every input word and every output word of every unit (the 128-bit word column, unused bits zero),
//! each opened at two verifier points sampled after the commitment (both reps). The verifier computes their values from its
//! own instance file, so the statement is the relation on public inputs and outputs (no commitment scheme binds them).

use std::sync::OnceLock;

use flock_core::{
    lincheck::{CscCircuit, LincheckCircuit, SparseMatrixCircuit, pack_z_lincheck_from_packed},
    proof::eq_skip_weights,
    r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
    verifier::ExtraClaim,
};
use flock_field::F128;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

pub const TAG: &[u8] = b"verity/flock-ir-block/v1";
/// Coordinates per verifier point: the regions' free in-block bits come from `r[..16]`, block bits from `r[16..]`.
pub const PT_LOCAL: usize = 16;
/// The Flock-CUDA host-witness mode takes blocks of 2^20..2^22 bits and at most 256 unit slots.
pub const MAX_UNITS: usize = 256;
pub const MIN_K_LOG: usize = 20;
const WORD: usize = 128;

/// A port group of a unit: its first 128-bit word column, its (power-of-two, aligned) word count and its ports' widths.
#[derive(Clone, Debug)]
pub struct Group {
    pub col: usize,
    pub words: usize,
    pub bits: Vec<usize>,
}

/// A net's rows by index, as the builders read them: row `i` lists the columns its A (or B) reads, in order, repeats kept.
/// `[Vec<usize>]` and [`Rows`] read alike.
pub trait RowRead {
    fn n_rows(&self) -> usize;
    fn row_len(&self, i: usize) -> usize;
    fn row_cols(&self, i: usize) -> impl ExactSizeIterator<Item = usize> + '_;
    fn nnz(&self) -> usize {
        (0..self.n_rows()).map(|i| self.row_len(i)).sum()
    }
    /// Whether row `i` is exactly `cols`, in order.
    fn row_is(&self, i: usize, cols: &[usize]) -> bool {
        self.row_len(i) == cols.len() && self.row_cols(i).eq(cols.iter().copied())
    }
    /// Row `i`'s columns, and none past the last row.
    fn cols_or_empty(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        (i < self.n_rows()).then(|| self.row_cols(i)).into_iter().flatten()
    }
}

impl RowRead for [Vec<usize>] {
    fn n_rows(&self) -> usize {
        self.len()
    }
    fn row_len(&self, i: usize) -> usize {
        self[i].len()
    }
    fn row_cols(&self, i: usize) -> impl ExactSizeIterator<Item = usize> + '_ {
        self[i].iter().copied()
    }
}

impl RowRead for Vec<Vec<usize>> {
    fn n_rows(&self) -> usize {
        self.len()
    }
    fn row_len(&self, i: usize) -> usize {
        self[i].len()
    }
    fn row_cols(&self, i: usize) -> impl ExactSizeIterator<Item = usize> + '_ {
        self[i].iter().copied()
    }
}

/// A net's rows as a builder writes them: one row at a time, in row order. `Vec<Vec<usize>>` and [`Rows`] write alike.
pub trait RowWrite: RowRead + Sized {
    fn with_rows(rows: usize) -> Self;
    fn push_row(&mut self, row: &[usize]);
    /// These rows with row `i` replaced by `row`, and empty rows before it if it is past the last.
    fn with_row(&self, i: usize, row: &[usize]) -> Self {
        let n = self.n_rows().max(i + 1);
        let mut out = Self::with_rows(n);
        let mut cols = Vec::new();
        for r in 0..n {
            cols.clear();
            cols.extend(self.cols_or_empty(r));
            out.push_row(if r == i { row } else { &cols });
        }
        out
    }
}

impl RowWrite for Vec<Vec<usize>> {
    fn with_rows(rows: usize) -> Self {
        Vec::with_capacity(rows)
    }
    fn push_row(&mut self, row: &[usize]) {
        self.push(row.to_vec());
    }
}

/// A net's rows held once, in one allocation per side: row `i` is `cols[off[i]..off[i + 1]]`. The offsets are u64 because a
/// unit's rows can hold 2^32 entries or more; a column is a u32, and one past `u32::MAX` is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rows {
    off: Vec<u64>,
    cols: Vec<u32>,
}

impl Default for Rows {
    fn default() -> Self {
        Rows::with_capacity(0, 0)
    }
}

impl Rows {
    pub fn with_capacity(rows: usize, nnz: usize) -> Self {
        let mut off = Vec::with_capacity(rows + 1);
        off.push(0);
        Rows { off, cols: Vec::with_capacity(nnz) }
    }

    pub fn push(&mut self, row: &[usize]) {
        self.cols.extend(row.iter().map(|&c| u32::try_from(c).unwrap_or_else(|_| panic!("column {c} is past u32::MAX"))));
        self.off.push(self.cols.len() as u64);
    }

    pub fn from_vecs(rows: &[Vec<usize>]) -> Self {
        let mut r = Rows::with_capacity(rows.len(), rows.iter().map(Vec::len).sum());
        rows.iter().for_each(|row| r.push(row));
        r
    }

    pub fn to_vecs(&self) -> Vec<Vec<usize>> {
        self.iter().map(|row| row.iter().map(|&c| c as usize).collect()).collect()
    }

    pub fn row(&self, i: usize) -> &[u32] {
        &self.cols[self.off[i] as usize..self.off[i + 1] as usize]
    }

    pub fn len(&self) -> usize {
        self.off.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn nnz(&self) -> usize {
        self.cols.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[u32]> + '_ {
        (0..self.len()).map(|i| self.row(i))
    }
}

impl RowRead for Rows {
    fn n_rows(&self) -> usize {
        self.len()
    }
    fn row_len(&self, i: usize) -> usize {
        (self.off[i + 1] - self.off[i]) as usize
    }
    fn row_cols(&self, i: usize) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.row(i).iter().map(|&c| c as usize)
    }
    fn nnz(&self) -> usize {
        Rows::nnz(self)
    }
}

impl RowWrite for Rows {
    fn with_rows(rows: usize) -> Self {
        Rows::with_capacity(rows, 0)
    }
    fn push_row(&mut self, row: &[usize]) {
        self.push(row);
    }
}

/// A `flock-ir-unit/v2` netlist.
#[derive(Clone)]
pub struct IrUnitNet {
    pub name: String,
    pub useful: usize,
    pub const_pos: usize,
    pub in_groups: Vec<Group>,
    pub out_groups: Vec<Group>,
    pub in_words: usize,
    pub out_cols: Vec<usize>,
    pub a: Rows,
    pub b: Rows,
    pub unit_log: usize,
    pub sha256: [u8; 32],
    /// The units' leaf maps (the `LEAVES <json>` line: per-unit `in` / `out` instance leaves, `in_ports` / `out_ports` as
    /// `[leaves, bits]`), covered by `sha256` and so by the pin.
    pub leaves: Option<serde_json::Value>,
    /// A cut lowering's cut structure (its final `CUT <json>` line: `words`, `tail`, per-unit `in` / `out` ports), covered by
    /// `sha256` and so by the pin.
    pub cut: Option<serde_json::Value>,
    /// A net committed at a position map (the Lean `Net.ofRowsAt`, a templated unit's `Tmpl.posMap`; its final
    /// `ORDER <json>` line): logical row `x` is row `order[x]`, so the rows are evaluated in that order, not by column.
    pub order: Option<Vec<usize>>,
    /// Whether [`IrUnitNet::eval64_flat`] takes one pass, decided on its first call: `a` and `b` must not change after it.
    pub pass: PassCache,
}

/// A clone starts undecided, so a copy whose rows are then edited decides anew.
#[derive(Default)]
pub struct PassCache(OnceLock<bool>);

impl Clone for PassCache {
    fn clone(&self) -> Self {
        PassCache::default()
    }
}

/// Whether row `i` of `rows` is `[i]`.
fn own_row<R: RowRead + ?Sized>(rows: &R, i: usize) -> bool {
    rows.row_len(i) == 1 && rows.row_cols(i).next() == Some(i)
}

/// Row `i`'s XOR over `z`, whose entry `i` is still 0, and all ones where it reads column `i` an odd number of times.
fn fold_own(row: &[u32], i: usize, z: &[u64]) -> (u64, u64) {
    row.iter().fold((0, 0), |(x, o), &c| (x ^ z[c as usize], o ^ 0u64.wrapping_sub(u64::from(c as usize == i))))
}

/// A netlist's `useful` rows, each line `na a.. nb b..`, written in order.
fn read_rows<'a, R: RowWrite>(lines: &mut impl Iterator<Item = &'a str>, useful: usize) -> (R, R) {
    let (mut a, mut b) = (R::with_rows(useful), R::with_rows(useful));
    let mut v: Vec<usize> = Vec::new();
    for _ in 0..useful {
        v.clear();
        v.extend(lines.next().expect("row").split_whitespace().map(|t| t.parse::<usize>().unwrap()));
        let na = v[0];
        let nb = v[1 + na];
        a.push_row(&v[1..1 + na]);
        b.push_row(&v[2 + na..2 + na + nb]);
    }
    (a, b)
}

impl IrUnitNet {
    pub fn parse(text: &str) -> Self {
        let sha256: [u8; 32] = Sha256::digest(text.as_bytes()).into();
        let mut lines = text.lines();
        let h: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
        assert_eq!(h[0], "flock-ir-unit/v2", "not a flock-ir-unit/v2 netlist");
        let n = |i: usize| -> usize { h[i].parse().unwrap() };
        let (useful, const_pos) = (n(2), n(3));
        let mut at = 4;
        let groups = |at: &mut usize| -> Vec<Group> {
            let count = n(*at);
            *at += 1;
            (0..count)
                .map(|_| {
                    let (col, words, np) = (n(*at), n(*at + 1), n(*at + 2));
                    let bits = (0..np).map(|i| n(*at + 3 + i)).collect();
                    *at += 3 + np;
                    assert!(words.is_power_of_two() && col % words == 0, "a port group must be an aligned power of two of words");
                    Group { col, words, bits }
                })
                .collect()
        };
        let in_groups = groups(&mut at);
        let out_groups = groups(&mut at);
        let in_words = in_groups.last().map(|g| g.col + g.words).unwrap_or(0);
        let (o0, o1) = (out_groups[0].col, out_groups.last().map(|g| g.col + g.words).unwrap());
        let out_cols: Vec<usize> = (o0..o1).collect();
        assert!(const_pos == useful - 1);
        let (a, b): (Rows, Rows) = read_rows(&mut lines, useful);
        for i in 0..in_words * WORD {
            assert!((a.row_is(i, &[i]) && b.row_is(i, &[i])) || (a.row_len(i) == 0 && b.row_len(i) == 0), "unit row {i} is not an input row");
        }
        assert!(a.row_is(const_pos, &[const_pos]) && b.row_is(const_pos, &[const_pos]));
        let mut rest = lines.peekable();
        let mut tagged = |tag: &str| -> Option<serde_json::Value> {
            let j = rest.next_if(|l| l.starts_with(tag))?;
            Some(serde_json::from_str(&j[tag.len()..]).unwrap_or_else(|e| panic!("the {tag}line is not JSON: {e}")))
        };
        let leaves = tagged("LEAVES ");
        let cut = tagged("CUT ");
        let order: Option<Vec<usize>> = tagged("ORDER ").map(|v| serde_json::from_value(v).expect("the ORDER line is not a list of columns"));
        assert!(rest.next().is_none(), "netlist text after the rows that is not its LEAVES, CUT and ORDER lines, in that order");
        if let Some(o) = &order {
            let first = in_words * WORD;
            let mut seen = vec![false; useful];
            assert!(o.len() > first && o.last() == Some(&const_pos), "ORDER: the constant is not the last logical row");
            for (x, &p) in o.iter().enumerate() {
                assert!(p < useful && !std::mem::replace(&mut seen[p], true), "ORDER: column {p} is placed twice or past the rows");
                assert!(x >= first || p == x, "ORDER: input row {x} moves");
            }
            assert!(seen.iter().enumerate().all(|(p, &s)| s || (a.row_len(p) == 0 && b.row_len(p) == 0)), "ORDER: an unplaced row is not empty");
        }
        let unit_log = (useful.next_power_of_two().trailing_zeros() as usize).max(7);
        // D11: an output group lies inside its unit's slot, so a region at its column stays in the slot
        if let Some(g) = out_groups.iter().find(|g| g.col + g.words > (1 << unit_log) / WORD) {
            panic!("flock-ir-unit/v2: output group at word {} ({} words) overruns its unit's {} words (D11)", g.col, g.words, (1usize << unit_log) / WORD);
        }
        IrUnitNet { name: h[1].to_string(), useful, const_pos, in_groups, out_groups, in_words, out_cols, a, b, unit_log, sha256,
                    leaves, cut, order, pass: PassCache::default() }
    }

    /// Whether column `i` is an input column (its row is `A = B = [i]`).
    pub fn is_input(&self, i: usize) -> bool {
        i < self.in_words * WORD && self.a.row_len(i) == 1
    }

    pub fn load(path: &str) -> Self {
        Self::parse(&std::fs::read_to_string(path).expect("netlist"))
    }

    /// The unit's slot type as the lincheck reads it: its rows over its `2^unit_log` columns, read in place.
    pub fn csc(&self) -> CscCircuit {
        let (n, k) = (self.a.n_rows(), 1usize << self.unit_log);
        assert!(self.b.n_rows() == n && n <= k, "a unit net of {n} A rows and {} B rows in a {k}-row slot", self.b.n_rows());
        CscCircuit::from_rows(n, k, |r| self.a.row_cols(r), |r| self.b.row_cols(r))
    }

    fn matrices(&self) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        self.padded(1 << self.unit_log)
    }

    /// A and B as `k`-row matrices: the rows, resized to `k` with empty rows.
    pub fn padded(&self, k: usize) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let pad = |rows: &Rows| {
            let mut r = rows.to_vecs();
            r.resize(k, Vec::new());
            SparseBinaryMatrix { num_rows: k, num_cols: k, rows: r.into() }
        };
        (pad(&self.a), pad(&self.b))
    }

    /// Up to 64 units at once (lane l = unit l): `input(col)` is the lanes' bit at input column `col`; returns (z, a, b) per
    /// row, bit-sliced.
    pub fn eval64(&self, input: impl Fn(usize) -> u64) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        self.eval64_of(&self.a, &self.b, input)
    }

    /// [`IrUnitNet::eval64`] of this net's layout with its rows `a` and `b` read through [`RowRead`].
    fn eval64_of<R: RowRead + ?Sized>(&self, a: &R, b: &R, input: impl Fn(usize) -> u64) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let u = self.useful;
        let first = self.in_words * WORD;
        let mut z = vec![0u64; u];
        for (i, zi) in z.iter_mut().enumerate().take(first) {
            if a.row_len(i) == 1 {
                *zi = input(i);
            }
        }
        z[self.const_pos] = !0;
        let rows: Box<dyn Iterator<Item = usize> + '_> = match &self.order {
            Some(o) => Box::new(o[first..].iter().copied()),
            None => Box::new(first..u),
        };
        for i in rows {
            if i == self.const_pos {
                continue;
            }
            let x = a.row_cols(i).fold(0u64, |s, k| s ^ z[k]);
            let y = b.row_cols(i).fold(0u64, |s, k| s ^ z[k]);
            z[i] = x & y;
        }
        let av: Vec<u64> = (0..u).map(|i| a.row_cols(i).fold(0u64, |s, k| s ^ z[k])).collect();
        let bv: Vec<u64> = (0..u).map(|i| b.row_cols(i).fold(0u64, |s, k| s ^ z[k])).collect();
        (z, av, bv)
    }

    /// [`IrUnitNet::eval64`], equal to it bit for bit, in one pass over `a` and `b` where [`IrUnitNet::one_pass`] holds (the
    /// two-pass evaluation where it doesn't).
    pub fn eval64_flat(&self, input: impl Fn(usize) -> u64) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        if !self.one_pass() {
            return self.eval64(input);
        }
        let (u, cp, first) = (self.useful, self.const_pos, self.in_words * WORD);
        let (mut z, mut a, mut b) = (vec![0u64; u], vec![0u64; u], vec![0u64; u]);
        for i in 0..first {
            if own_row(&self.a, i) {
                z[i] = input(i);
                a[i] = z[i];
            }
            if own_row(&self.b, i) {
                b[i] = z[i];
            }
        }
        z[cp] = !0;
        (a[cp], b[cp]) = (!0, !0);
        for i in first..cp {
            let (x, oa) = fold_own(self.a.row(i), i, &z);
            let (y, ob) = fold_own(self.b.row(i), i, &z);
            let zi = x & y;
            z[i] = zi;
            a[i] = x ^ (zi & oa);
            b[i] = y ^ (zi & ob);
        }
        (z, a, b)
    }

    /// Whether every row's z, a and b are final after one pass in row order, decided once: no position map, the constant
    /// last with rows `[cp]`, every input row `[i]` or empty, and every other row reading only earlier columns, its own and
    /// the constant's.
    pub fn one_pass(&self) -> bool {
        *self.pass.0.get_or_init(|| {
            let (u, cp, first) = (self.useful, self.const_pos, self.in_words * WORD);
            let input = |rows: &Rows, i: usize| rows.row_len(i) == 0 || own_row(rows, i);
            let earlier = |rows: &Rows, i: usize| rows.row(i).iter().all(|&c| c as usize <= i || c as usize == cp);
            u > 0 && cp == u - 1 && first <= cp && self.order.is_none() && own_row(&self.a, cp) && own_row(&self.b, cp)
                && (0..first).all(|i| input(&self.a, i) && input(&self.b, i))
                && (first..cp).into_par_iter().all(|i| earlier(&self.a, i) && earlier(&self.b, i))
        })
    }

    /// Every output word of a single unit from its input words.
    pub fn outputs(&self, inputs: &[F128]) -> Vec<F128> {
        let (z, _, _) = self.eval64(|col| bit(&inputs[col / WORD], col % WORD) as u64);
        self.out_cols.iter().map(|&c| (0..WORD).fold(F128::ZERO, |w, t| if z[c * WORD + t] & 1 == 1 { set(w, t) } else { w })).collect()
    }
}

fn bit(w: &F128, t: usize) -> bool {
    (if t < 64 { w.lo >> t } else { w.hi >> (t - 64) }) & 1 == 1
}

fn set(mut w: F128, t: usize) -> F128 {
    if t < 64 { w.lo |= 1 << t } else { w.hi |= 1 << (t - 64) }
    w
}

fn words(b: &[u8]) -> Vec<F128> {
    b.chunks(16).map(|c| F128::new(u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..].try_into().unwrap()))).collect()
}

/// A `flock-ir-instances/v1` file (`verity_flock.ir_lower.write_instances`).
#[derive(Clone)]
pub struct IrInstances {
    pub header: serde_json::Value,
    pub unit: String,
    pub units: usize,
    pub in_words: usize,
    pub out_words: usize,
    pub inputs: Vec<F128>,
    pub outputs: Vec<F128>,
    pub sha256: [u8; 32],
}

impl IrInstances {
    pub fn load(path: &str) -> Self {
        let b = std::fs::read(path).expect("instances");
        let sha256: [u8; 32] = Sha256::digest(&b).into();
        let nl = b.iter().position(|&c| c == b'\n').unwrap();
        let header: serde_json::Value = serde_json::from_slice(&b[..nl]).unwrap();
        assert_eq!(header["format"], "flock-ir-instances/v1");
        let g = |k: &str| header[k].as_u64().unwrap() as usize;
        let (units, in_words, out_words) = (g("units"), g("in_words"), g("out_words"));
        let ni = units * in_words * 16;
        let no = units * out_words * 16;
        assert_eq!(b.len(), nl + 1 + ni + no, "instance file length");
        let inputs = words(&b[nl + 1..nl + 1 + ni]);
        let outputs = words(&b[nl + 1 + ni..]);
        IrInstances { unit: header["unit"].as_str().unwrap().to_string(), header, units, in_words, out_words, inputs, outputs, sha256 }
    }

    pub fn input(&self, g: usize) -> &[F128] {
        &self.inputs[g * self.in_words..(g + 1) * self.in_words]
    }

    pub fn output(&self, g: usize) -> &[F128] {
        &self.outputs[g * self.out_words..(g + 1) * self.out_words]
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Region {
    /// input words [col, col + words) of every unit
    In(usize, usize),
    /// output words [col, col + words) of every unit
    Out(usize, usize),
}

/// One public region: the in-block bits it fixes (`fixed` over `mask`) and its free in-block bits, LSB first.
pub struct RegionShape {
    pub region: Region,
    pub free: Vec<usize>,
    pub mask: u32,
    pub fixed: u32,
}

pub struct IrCircuit {
    pub pin: usize,
    pub k_log: usize,
    pub unit_log: usize,
    pub units: usize,
    pub unit: SparseMatrixCircuit<'static>,
    pub da: Vec<(u32, u32)>,
    pub db: Vec<(u32, u32)>,
}

impl LincheckCircuit for IrCircuit {
    fn n_cols(&self) -> usize {
        1 << self.k_log
    }

    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        let us = 1usize << self.unit_log;
        assert_eq!(eq.len(), 1 << self.k_log);
        let c0 = eq[..us].iter().position(|e| *e != F128::ZERO).expect("eq tensor vanishes on the first unit slot");
        let inv = eq[c0].inv();
        let base = self.unit.fold_alpha_batched(alpha, &eq[..us]);
        let mut out = vec![F128::ZERO; 1 << self.k_log];
        out.par_chunks_mut(us).enumerate().for_each(|(p, o)| {
            if p < self.units {
                let s = eq[(p << self.unit_log) | c0] * inv;
                for (x, y) in o.iter_mut().zip(&base) {
                    *x = s * *y;
                }
            }
        });
        for &(i, c) in &self.da {
            out[c as usize] += alpha * eq[i as usize];
        }
        for &(i, c) in &self.db {
            out[c as usize] += eq[i as usize];
        }
        out
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(self.pin)
    }
}

pub struct IrStmt {
    pub net: IrUnitNet,
    pub units_real: usize,
    pub units: usize,
    pub unit_log: usize,
    pub k_log: usize,
    pub nbl: usize,
    pub m: usize,
    pub m_pts: usize,
    pub pin: usize,
    pub regions: Vec<RegionShape>,
    pub circuit: IrCircuit,
    pub r1cs: BlockR1cs,
    pub digest: [u8; 32],
    /// A padding unit's output words (all-zero inputs).
    pub dummy_out: Vec<F128>,
}

fn log2ceil(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

impl IrStmt {
    pub fn new(net: IrUnitNet, units_real: usize) -> Self {
        let unit_log = net.unit_log;
        let k_log = (unit_log + log2ceil(MAX_UNITS)).min(MIN_K_LOG.max(unit_log));
        let units = 1usize << (k_log - unit_log);
        assert!(units <= MAX_UNITS && k_log >= MIN_K_LOG, "unit_log {unit_log}: a block of {units} units is {k_log} bits");
        let nbl = log2ceil(units_real.div_ceil(units)).max(3);
        let m = k_log + nbl;
        let pin = net.const_pos;
        // every unit's constant row copies the pinned column (Δ adds to the base: the own column cancels)
        let (mut da, mut db) = (Vec::new(), Vec::new());
        for p in 1..units {
            let i = (p << unit_log) + net.const_pos;
            for d in [&mut da, &mut db] {
                d.push((i as u32, i as u32));
                d.push((i as u32, pin as u32));
            }
        }
        // one region for all input words, one for all output words: each an aligned power-of-two subcube of every unit
        let unit_bits: Vec<usize> = (unit_log..k_log).collect();
        let shape = |region: Region, words: usize, col: usize| {
            assert!(words.is_power_of_two() && col % words == 0, "unit IO words must be an aligned power of two");
            let wb = words.trailing_zeros() as usize;
            let free: Vec<usize> = (0..7 + wb).chain(unit_bits.iter().copied()).collect();
            let fixed = (col as u32) << 7;
            let mask = ((1u32 << k_log) - 1) & !free.iter().fold(0u32, |m, &b| m | (1 << b));
            assert_eq!(fixed & !mask, 0);
            assert!(free.len() <= PT_LOCAL, "a region of {} free bits", free.len());
            RegionShape { region, free, mask, fixed }
        };
        let mut regions: Vec<RegionShape> = net.in_groups.iter().map(|g| shape(Region::In(g.col, g.words), g.words, g.col)).collect();
        regions.extend(net.out_groups.iter().map(|g| shape(Region::Out(g.col, g.words), g.words, g.col)));
        let dummy_out = net.outputs(&vec![F128::ZERO; net.in_words]);
        let mut h = Sha256::new();
        h.update(TAG);
        h.update(net.sha256);
        for x in [units_real, units, unit_log, k_log, nbl, m, net.in_words, net.out_cols.len()] {
            h.update((x as u64).to_le_bytes());
        }
        for d in [&da, &db] {
            h.update((d.len() as u64).to_le_bytes());
            for &(i, c) in d.iter() {
                h.update(i.to_le_bytes());
                h.update(c.to_le_bytes());
            }
        }
        let digest: [u8; 32] = h.finalize().into();
        let kk = 1usize << k_log;
        let empty = || SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: vec![Vec::new(); kk].into() };
        let r1cs = BlockR1cs {
            m,
            k_log,
            k_skip: 6,
            useful_bits: kk,
            a_0: empty(),
            b_0: empty(),
            c_0: SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: (0..kk).map(|i| vec![i]).collect::<Vec<_>>().into() },
            layout: WitnessLayout::RowMajor,
            const_pin: Some(pin),
            digest_cache: OnceLock::from(digest),
            csc_cache: OnceLock::new(),
        };
        let (ua, ub) = net.matrices();
        let (ua, ub): (&'static SparseBinaryMatrix, &'static SparseBinaryMatrix) = (Box::leak(Box::new(ua)), Box::leak(Box::new(ub)));
        let circuit = IrCircuit { pin, k_log, unit_log, units, unit: SparseMatrixCircuit::new(ua, ub), da, db };
        IrStmt { net, units_real, units, unit_log, k_log, nbl, m, m_pts: PT_LOCAL + nbl, pin, regions, circuit, r1cs, digest, dummy_out }
    }

    pub fn blocks(&self) -> usize {
        1 << self.nbl
    }

    /// `(z, a, b, z_lincheck)` of the block witness; `inputs(g)` gives unit g's input words (any, possibly dishonest,
    /// witness; units past the real ones get all-zero inputs from the caller).
    pub fn witness(&self, inputs: &(dyn Fn(usize) -> Vec<F128> + Sync)) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let n = 1usize << (self.m - 7);
        let uw = 1usize << (self.unit_log - 7);
        let total = self.blocks() * self.units;
        let (mut z, mut a, mut b) = (vec![F128::ZERO; n], vec![F128::ZERO; n], vec![F128::ZERO; n]);
        let groups: Vec<Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)>> = (0..total.div_ceil(64))
            .into_par_iter()
            .map(|gi| {
                let ids: Vec<usize> = (64 * gi..(64 * gi + 64).min(total)).collect();
                let ins: Vec<Vec<F128>> = ids.iter().map(|&g| inputs(g)).collect();
                let (zz, aa, bb) = self.net.eval64(|col| {
                    let mut x = 0u64;
                    for (l, r) in ins.iter().enumerate() {
                        x |= (bit(&r[col / WORD], col % WORD) as u64) << l;
                    }
                    x
                });
                ids.iter()
                    .enumerate()
                    .map(|(l, &g)| {
                        let pack = |v: &Vec<u64>| -> Vec<F128> {
                            let mut o = vec![F128::ZERO; uw];
                            for (r, &x) in v.iter().enumerate() {
                                if (x >> l) & 1 == 1 {
                                    o[r / WORD] = set(o[r / WORD], r % WORD);
                                }
                            }
                            o
                        };
                        (g, pack(&zz), pack(&aa), pack(&bb))
                    })
                    .collect()
            })
            .collect();
        for (g, uz, ua, ub) in groups.into_iter().flatten() {
            let at = g * uw;
            z[at..at + uw].copy_from_slice(&uz);
            a[at..at + uw].copy_from_slice(&ua);
            b[at..at + uw].copy_from_slice(&ub);
        }
        let zl = pack_z_lincheck_from_packed(&z, self.m, self.k_log);
        (z, a, b, zl)
    }

    /// The honest unit inputs: the instance file's, all-zero past its last unit.
    pub fn honest_inputs<'a>(&'a self, inst: &'a IrInstances) -> impl Fn(usize) -> Vec<F128> + Sync + 'a {
        move |g| if g < self.units_real { inst.input(g).to_vec() } else { vec![F128::ZERO; self.net.in_words] }
    }

    fn words_of<'a>(&'a self, reg: Region, inst: &'a IrInstances, g: usize, zero: &'a [F128]) -> &'a [F128] {
        match (reg, g < self.units_real) {
            (Region::In(c, w), true) => &inst.input(g)[c..c + w],
            (Region::In(_, w), false) => &zero[..w],
            (Region::Out(c, w), true) => { let o = c - self.net.out_cols[0]; &inst.output(g)[o..o + w] }
            (Region::Out(c, w), false) => { let o = c - self.net.out_cols[0]; &self.dummy_out[o..o + w] }
        }
    }

    /// The region's claim point from verifier point `r` (m_pts coordinates).
    pub fn region_point(&self, reg: &RegionShape, r: &[F128]) -> Vec<F128> {
        let mut p = vec![F128::ZERO; self.m];
        for j in 0..self.k_log {
            if (reg.mask >> j) & 1 == 1 && (reg.fixed >> j) & 1 == 1 {
                p[j] = F128::ONE;
            }
        }
        for (t, &j) in reg.free.iter().enumerate() {
            p[j] = r[t];
        }
        p[self.k_log..].copy_from_slice(&r[PT_LOCAL..PT_LOCAL + self.nbl]);
        p
    }

    /// The multilinear value of the region at `r`: sum over blocks of eq(block) times the region's bits of that block, the
    /// free assignment `f = t + 128 (w + W u)` being bit t of word w of unit u (W words per unit).
    pub fn region_value(&self, reg: &RegionShape, r: &[F128], inst: &IrInstances) -> F128 {
        let nf = reg.free.len();
        let wt = eq_skip_weights(&r[..nf]);
        let eb = eq_skip_weights(&r[PT_LOCAL..PT_LOCAL + self.nbl]);
        let nbytes = (1usize << nf) / 8;
        let mut t = vec![[F128::ZERO; 256]; nbytes];
        for (i, tb) in t.iter_mut().enumerate() {
            for x in 1..256usize {
                let rest = x & (x - 1);
                let bitpos = 8 * i + (x ^ rest).trailing_zeros() as usize;
                tb[x] = tb[rest] + wt[bitpos];
            }
        }
        let zero = vec![F128::ZERO; self.net.in_words];
        (0..self.blocks())
            .into_par_iter()
            .map(|blk| {
                let mut acc = F128::ZERO;
                let mut k = 0;
                for u in 0..self.units {
                    for w in self.words_of(reg.region, inst, blk * self.units + u, &zero) {
                        for x in [w.lo.to_le_bytes(), w.hi.to_le_bytes()].concat() {
                            acc += t[k][x as usize];
                            k += 1;
                        }
                    }
                }
                eb[blk] * acc
            })
            .reduce(|| F128::ZERO, |a, b| a + b)
    }

    pub fn extra_points(&self, pts: &[Vec<F128>]) -> Vec<Vec<F128>> {
        self.regions.iter().flat_map(|reg| pts.iter().map(move |r| self.region_point(reg, r))).collect()
    }

    pub fn extra_claims(&self, pts: &[Vec<F128>], inst: &IrInstances) -> Vec<ExtraClaim> {
        let values: Vec<F128> = self.regions.iter().flat_map(|reg| pts.iter().map(move |r| self.region_value(reg, r, inst))).collect();
        self.extra_points(pts)
            .into_iter()
            .zip(values)
            .map(|(pt, value)| ExtraClaim { value, skip_weights: eq_skip_weights(&pt[..6]), x_full: pt[6..].to_vec() })
            .collect()
    }
}

/// The cut words of a cut template's instance file: per instance-local unit, the (column, cut word) of each cut input port
/// and each cut output port, and the native tail. Red-team-flock-2 IR1: the verifier accepts no cut word it does not
/// account for itself.
#[derive(Clone)]
pub struct CutMap {
    pub words: usize,
    pub tail: Vec<crate::ir_tail::Op>,
    pub unit_in: Vec<Vec<(usize, usize)>>,
    pub unit_out: Vec<Vec<(usize, usize)>>,
    /// the bit width of each cut input / output port (16 or 32; the same for every unit)
    pub in_bits: Vec<usize>,
    pub out_bits: Vec<usize>,
    /// `(cut word, input leaf)`: cut words that are input leaves the units read publicly (the verifier checks them against
    /// their port's committed row)
    pub public: Vec<(usize, usize)>,
    /// per output leaf, the tail slot that computes it, when the units return none
    pub outputs: Vec<usize>,
}

fn pairs(v: &serde_json::Value) -> Result<Vec<Vec<(usize, usize)>>, String> {
    v.as_array()
        .ok_or("cut ports are not a list")?
        .iter()
        .map(|u| {
            u.as_array()
                .ok_or("a unit's cut ports are not a list")?
                .iter()
                .map(|p| match p.as_array().map(|x| (x.first().and_then(|a| a.as_u64()), x.get(1).and_then(|a| a.as_u64()))) {
                    Some((Some(c), Some(k))) => Ok((c as usize, k as usize)),
                    _ => Err("a cut port is not [column, cut word]".to_string()),
                })
                .collect()
        })
        .collect()
}

/// The cut structure is the pinned netlist's (IR4), never the instance header's: a header carrying its own is refused, and
/// its cut word and per-instance unit counts must be the netlist's. Checked: units = instances x units per instance; no cut
/// word (and no second port group) without a tail that computes it; the cut ports exactly the netlist's cut groups (32-bit
/// ports, in order); every cut word computed exactly once, by one unit of the instance or by the tail.
pub fn cut_map(inst: &IrInstances, net: &IrUnitNet) -> Result<CutMap, String> {
    cut_map_of(net, &inst.header, inst.units)
}

/// [`cut_map`] for any instance file: its header (`instances`, `units_per_instance`, `cut_words`, no `cut`) and unit count.
pub fn cut_map_of(net: &IrUnitNet, h: &serde_json::Value, units: usize) -> Result<CutMap, String> {
    let g = |k: &str| h[k].as_u64().map(|x| x as usize);
    let (n, upi) = (g("instances").ok_or("no instances count")?, g("units_per_instance").ok_or("no units_per_instance")?);
    if units != n * upi {
        return Err(format!("{units} units, but {n} instances x {upi} units each"));
    }
    if h.get("cut").is_some() {
        return Err("the instance header carries a cut structure; the cut is the pinned netlist's".into());
    }
    let null = serde_json::Value::Null;
    let cut = net.cut.as_ref().unwrap_or(&null);
    let words = cut["words"].as_u64().unwrap_or(0) as usize;
    if g("cut_words").unwrap_or(0) != words {
        return Err(format!("the header claims {} cut words, the pinned netlist {words}", g("cut_words").unwrap_or(0)));
    }
    if words == 0 {
        if net.cut.is_some() || net.in_groups.len() != 1 || net.out_groups.len() != 1 {
            return Err("a netlist with cut groups or a CUT line but no cut words".into());
        }
        return Ok(CutMap { words: 0, tail: vec![], unit_in: vec![vec![]; upi], unit_out: vec![vec![]; upi], in_bits: vec![], out_bits: vec![],
                           public: vec![], outputs: vec![] });
    }
    let outputs: Vec<usize> = cut["outputs"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as usize)).collect()).unwrap_or_default();
    let public: Vec<(usize, usize)> = pairs(&serde_json::json!([cut["public"].clone()]))
        .ok().and_then(|v| v.into_iter().next()).unwrap_or_default();
    // units that return output leaves have a returned-output group before their cut group; tail-output units only a cut group
    let want_out = if outputs.is_empty() { 2 } else { 1 };
    if net.in_groups.len() != 2 || net.out_groups.len() != want_out {
        return Err(format!("{words} cut words, but the netlist's port groups are not a leaf and a cut input group and {want_out} output group(s)"));
    }
    let (unit_in, unit_out) = (pairs(&cut["in"])?, pairs(&cut["out"])?);
    if unit_in.len() != upi || unit_out.len() != upi {
        return Err(format!("{upi} units per instance, but the netlist's cut ports list {}", unit_in.len()));
    }
    let group_cols = |g: &Group, base: usize| -> Result<Vec<usize>, String> {
        if g.bits.iter().any(|&b| b != 16 && b != 32) {
            return Err("a cut port is not 16 or 32 bits".into());
        }
        Ok(g.bits.iter().scan(g.col * WORD - base, |a, &b| { let c = *a; *a += b; Some(c) }).collect())
    };
    let cg_out = net.out_groups.last().unwrap();
    let (cin, cout) = (group_cols(&net.in_groups[1], 0)?, group_cols(cg_out, net.out_cols[0] * WORD)?);
    let (in_bits, out_bits) = (net.in_groups[1].bits.clone(), cg_out.bits.clone());
    let mut produced = std::collections::BTreeSet::new();
    for &(k, _) in &public {
        if k >= words || !produced.insert(k) {
            return Err(format!("public cut word {k} out of range or listed twice"));
        }
    }
    for u in 0..upi {
        if unit_in[u].iter().map(|p| p.0).collect::<Vec<_>>() != cin || unit_out[u].iter().map(|p| p.0).collect::<Vec<_>>() != cout {
            return Err(format!("unit {u}'s cut ports are not the netlist's cut groups"));
        }
        for &(_, k) in unit_in[u].iter().chain(&unit_out[u]) {
            if k >= words {
                return Err(format!("unit {u} names cut word {k} of {words}"));
            }
        }
        for &(_, k) in &unit_out[u] {
            if !produced.insert(k) {
                return Err(format!("cut word {k} is computed by two units"));
            }
        }
    }
    let tail = crate::ir_tail::parse(&cut["tail"], words, &produced)?;
    if tail.iter().any(|o| o.out < words && produced.contains(&o.out)) {
        return Err("a cut word is computed by a unit, by the tail or by the file twice".into());
    }
    if outputs.iter().any(|&s| !tail.iter().any(|o| o.out == s)) {
        return Err("an output slot the tail does not compute".into());
    }
    Ok(CutMap { words, tail, unit_in, unit_out, in_bits, out_bits, public, outputs })
}

fn word32(words: &[F128], col: usize) -> u32 {
    let w = &words[col / WORD];
    let t = col % WORD;
    (if t < 64 { w.lo >> t } else { w.hi >> (t - 64) }) as u32
}

/// Per instance: the units' cut output words, the tail evaluated on them, and every unit's cut input words equal to the
/// words so computed.
pub fn check_cuts(inst: &IrInstances, cm: &CutMap, t: &crate::ir_tail::Tables) -> Result<(), String> {
    if cm.words == 0 {
        return Ok(());
    }
    let upi = cm.unit_in.len();
    for i in 0..inst.units / upi {
        let mut slots: Vec<Option<u32>> = vec![None; cm.words];
        for u in 0..upi {
            for &(c, k) in &cm.unit_out[u] {
                slots[k] = Some(word32(inst.output(i * upi + u), c));
            }
        }
        crate::ir_tail::eval(&cm.tail, &mut slots, t);
        for u in 0..upi {
            for &(c, k) in &cm.unit_in[u] {
                if Some(word32(inst.input(i * upi + u), c)) != slots[k] {
                    return Err(format!("instance {i}: unit {u}'s cut word {k} is not what the tail computes from the units' words"));
                }
            }
        }
    }
    Ok(())
}

/// Topological order: every AND row reads only earlier columns or the constant, and no row past the inputs is free.
pub fn check_order(net: &IrUnitNet) -> Result<(), String> {
    for i in net.in_words * WORD..net.useful {
        if i == net.const_pos {
            continue;
        }
        let assertion = net.b.row_is(i, &[net.const_pos]) && net.a.row_cols(i).any(|c| c == i);
        for c in net.a.row_cols(i).chain(net.b.row_cols(i)) {
            if c >= i && c != net.const_pos && !(assertion && c == i) {
                return Err(format!("unit row {i} reads column {c}, not an earlier one"));
            }
        }
        if net.a.row_is(i, &[i]) && net.b.row_is(i, &[i]) {
            return Err(format!("unit row {i} is a free row"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(s: &mut u64, m: usize) -> usize {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s % m as u64) as usize
    }

    /// Random rows of every kind `eval64` reads: input rows and empty ones, the constant, repeated columns, and assertion
    /// rows (A reads the row's own column, B the constant).
    fn random_net(seed: u64, in_words: usize, useful: usize) -> IrUnitNet {
        let (mut s, first, cp) = (seed, in_words * WORD, useful - 1);
        let (mut a, mut b) = (Rows::with_rows(useful), Rows::with_rows(useful));
        for i in 0..useful {
            let (ra, rb) = if i == cp {
                (vec![cp], vec![cp])
            } else if i < first {
                if next(&mut s, 5) == 0 { (vec![], vec![]) } else { (vec![i], vec![i]) }
            } else {
                let (na, nb, kind) = (1 + next(&mut s, 7), 1 + next(&mut s, 7), next(&mut s, 5));
                let mut pick = |n: usize| -> Vec<usize> { (0..n).map(|_| if next(&mut s, 9) == 0 { cp } else { next(&mut s, i) }).collect() };
                let (mut ra, mut rb) = (pick(na), pick(nb));
                match kind {
                    0 => { ra.push(i); rb = vec![cp]; }
                    1 => { ra.push(ra[0]); rb.push(rb[0]); }
                    2 => { ra.extend([i, i]); }
                    _ => {}
                }
                (ra, rb)
            };
            a.push_row(&ra);
            b.push_row(&rb);
        }
        IrUnitNet { name: "t".into(), useful, const_pos: cp, in_groups: vec![], out_groups: vec![], in_words, out_cols: vec![], a, b,
                    unit_log: useful.next_power_of_two().trailing_zeros().max(7) as usize, sha256: [0; 32], leaves: None, cut: None,
                    order: None, pass: PassCache::default() }
    }

    /// `rows` with column `c` appended to row `i`.
    fn appended(rows: &Rows, i: usize, c: usize) -> Rows {
        let row: Vec<usize> = rows.row_cols(i).chain([c]).collect();
        rows.with_row(i, &row)
    }

    /// `net` committed at a position map (`Net.ofRowsAt`): the computed rows scattered over twice their span, the constant
    /// last, every reference read through the map.
    fn placed(net: &IrUnitNet) -> IrUnitNet {
        let (first, cp) = (net.in_words * WORD, net.const_pos);
        let span = 2 * (cp - first) + 1;
        let pos: Vec<usize> = (0..=cp).map(|x| if x < first { x } else if x == cp { first + span } else { first + (x - first) * 7919 % span }).collect();
        let size = first + span + 1;
        let mut at = vec![None; size];
        (0..=cp).for_each(|x| at[pos[x]] = Some(x));
        let (mut a, mut b) = (Rows::with_rows(size), Rows::with_rows(size));
        for x in at {
            let moved = |rows: &Rows| x.map_or(Vec::new(), |x| rows.row_cols(x).map(|c| pos[c]).collect::<Vec<_>>());
            a.push_row(&moved(&net.a));
            b.push_row(&moved(&net.b));
        }
        IrUnitNet { useful: size, const_pos: size - 1, a, b, unit_log: size.next_power_of_two().trailing_zeros().max(7) as usize,
                    order: Some(pos), ..net.clone() }
    }

    /// `net` as a `flock-ir-unit/v2` netlist: its rows under one input group of its input words and one output group of the
    /// word after them.
    fn text_of(net: &IrUnitNet) -> String {
        let (w, cp) = (net.in_words, net.const_pos);
        let mut t = format!("flock-ir-unit/v2 t {} {cp} 1 0 {w} 1 {} 1 {w} 1 1 1\n", net.useful, w * WORD);
        for (ra, rb) in net.a.iter().zip(net.b.iter()) {
            let row = |r: &[u32]| std::iter::once(r.len()).chain(r.iter().map(|&c| c as usize)).map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
            t += &format!("{} {}\n", row(ra), row(rb));
        }
        t
    }

    #[test]
    fn a_placed_net_evaluates_as_its_logical_rows_moved() {
        for (seed, in_words, useful) in [(5u64, 1usize, 300usize), (0x51ed, 2, 3000)] {
            let net = random_net(seed, in_words, useful);
            let at = placed(&net);
            let pos = at.order.clone().unwrap();
            let input = |col: usize| (col as u64 + 3).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let (z, a, b) = net.eval64(input);
            let (pz, pa, pb) = at.eval64_flat(input);
            assert!(at.pass.0.get() == Some(&false), "a placed net took the one-pass path");
            for x in 0..net.useful {
                assert_eq!((pz[pos[x]], pa[pos[x]], pb[pos[x]]), (z[x], a[x], b[x]), "net {seed}: logical row {x}");
            }
            let mut gap = vec![true; at.useful];
            pos.iter().for_each(|&p| gap[p] = false);
            assert!((0..at.useful).filter(|&p| gap[p]).all(|p| (pz[p], pa[p], pb[p]) == (0, 0, 0)), "net {seed}: a gap row is not zero");
        }
    }

    #[test]
    fn eval64_flat_is_eval64() {
        for (seed, in_words, useful) in [(1u64, 1usize, 200usize), (0x9e37_79b9, 2, 5000), (77, 3, 40_000)] {
            let net = random_net(seed, in_words, useful);
            for k in 0..3u64 {
                let input = |col: usize| (col as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15 ^ k).rotate_left(k as u32 * 7);
                assert!(net.eval64_flat(input) == net.eval64(input), "net {seed}, inputs {k}");
            }
            assert!(net.pass.0.get() == Some(&true), "net {seed} took the two-pass path");
            assert!(net.clone().pass.0.get().is_none(), "a clone keeps the decision");
            // a row reading a later column: the two-pass evaluation, still equal
            let mut fwd = net.clone();
            let i = in_words * WORD + 3;
            fwd.a = appended(&fwd.a, i, i + 5);
            let input = |col: usize| col as u64 * 0x2545_f491_4f6c_dd1d;
            assert!(fwd.eval64_flat(input) == fwd.eval64(input));
            assert!(fwd.pass.0.get() == Some(&false), "a later column took the one-pass path");
        }
    }

    #[test]
    fn csc_reads_the_rows_in_place() {
        for (seed, in_words, useful) in [(1u64, 1usize, 200usize), (0x9e37_79b9, 2, 5000)] {
            let net = random_net(seed, in_words, useful);
            for n in [&net, &placed(&net)] {
                let (a, b) = n.matrices();
                assert!(n.csc() == CscCircuit::from_matrices(&a, &b), "net {seed}: the slot type's CSC");
                let (va, vb) = (n.a.to_vecs(), n.b.to_vecs());
                let got = CscCircuit::from_rows(va.len(), 1 << n.unit_log, |r| va.row_cols(r), |r| vb.row_cols(r));
                assert!(got == n.csc(), "net {seed}: the CSC from vectors");
            }
        }
    }

    #[test]
    fn row_shapes_read_alike() {
        let net = random_net(0x51ed, 2, 3000);
        let (rows, v) = (&net.a, net.a.to_vecs());
        let (n, cp) = (v.len(), net.const_pos);
        for i in 0..n + 3 {
            let want = v.get(i).cloned().unwrap_or_default();
            assert_eq!(rows.cols_or_empty(i).collect::<Vec<_>>(), want, "row {i} from Rows");
            assert_eq!(v[..].cols_or_empty(i).collect::<Vec<_>>(), want, "row {i} from a slice");
            assert_eq!(v.cols_or_empty(i).collect::<Vec<_>>(), want, "row {i} from a Vec");
            if i >= n {
                continue;
            }
            let (mut short, mut rev) = (want.clone(), want.clone());
            short.pop();
            rev.reverse();
            for probe in [want.clone(), short, rev, vec![i], vec![cp], vec![], vec![i, i]] {
                let is = want == probe;
                assert_eq!((rows.row_is(i, &probe), v[..].row_is(i, &probe), v.row_is(i, &probe)), (is, is, is), "row {i} is {probe:?}");
            }
        }
    }

    /// Both evaluators over `Rows` are the two-pass evaluation over the same rows as vectors, on nets one pass orders and on
    /// nets it refuses, and the refusals are the ones the one-pass evaluation needs.
    #[test]
    fn evaluators_read_rows_alike() {
        for (seed, in_words, useful) in [(1u64, 1usize, 200usize), (0x9e37_79b9, 2, 5000)] {
            let net = random_net(seed, in_words, useful);
            // a later column, an input row other than `[i]` or empty, and a placed net: each refused
            let mut fwd = net.clone();
            fwd.a = appended(&fwd.a, in_words * WORD + 3, in_words * WORD + 8);
            let mut odd = net.clone();
            odd.b = odd.b.with_row(2, &[2, 5]);
            for (n, one_pass) in [(&net, true), (&placed(&net), false), (&fwd, false), (&odd, false)] {
                let (va, vb) = (n.a.to_vecs(), n.b.to_vecs());
                for k in 0..3u64 {
                    let input = |col: usize| (col as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15 ^ k).rotate_left(k as u32 * 7);
                    let want = n.eval64_of(&va, &vb, input);
                    assert!(n.eval64(input) == want, "net {seed}, inputs {k}: eval64 from Rows");
                    assert!(n.eval64_flat(input) == want, "net {seed}, inputs {k}: eval64_flat from Rows");
                }
                assert_eq!(n.one_pass(), one_pass, "net {seed}: the one-pass refusal");
            }
        }
    }

    /// The parser writes a netlist's rows into `Rows` exactly as into vectors, and those are the rows the text holds.
    #[test]
    fn the_parser_writes_rows_alike() {
        for (seed, in_words, useful) in [(1u64, 1usize, 200usize), (0x9e37_79b9, 2, 5000)] {
            let net = random_net(seed, in_words, useful);
            let text = text_of(&net);
            let parsed = IrUnitNet::parse(&text);
            assert!(parsed.a == net.a && parsed.b == net.b, "net {seed}: the rows parsed");
            let (va, vb): (Vec<Vec<usize>>, Vec<Vec<usize>>) = read_rows(&mut text.lines().skip(1), useful);
            assert!(va == net.a.to_vecs() && vb == net.b.to_vecs(), "net {seed}: the rows parsed into vectors");
        }
    }

    #[test]
    fn rows_are_edited_alike() {
        let net = random_net(0x51ed, 1, 400);
        let (rows, v) = (&net.a, net.a.to_vecs());
        for (i, row) in [(0, vec![]), (200, vec![3, 3, 399]), (399, vec![399]), (450, vec![7])] {
            let mut want = v.clone();
            want.resize(want.len().max(i + 1), Vec::new());
            want[i] = row.clone();
            assert_eq!(v.with_row(i, &row), want, "row {i} of vectors");
            assert_eq!(rows.with_row(i, &row), Rows::from_vecs(&want), "row {i} of Rows");
        }
    }

    fn reads<R: RowRead + ?Sized>(r: &R) -> (Vec<Vec<usize>>, Vec<usize>, usize) {
        ((0..r.n_rows()).map(|i| r.row_cols(i).collect()).collect(), (0..r.n_rows()).map(|i| r.row_len(i)).collect(), r.nnz())
    }

    #[test]
    fn rows_read_as_the_vectors_they_hold() {
        let nets = [random_net(3, 1, 300), random_net(0x51ed, 2, 3000)];
        let gaps = placed(&nets[0]);
        for (k, net) in nets.iter().chain([&gaps]).enumerate() {
            for held in [&net.a, &net.b] {
                let side = &held.to_vecs();
                let rows = Rows::from_vecs(side);
                assert_eq!(rows, *held, "net {k}: the vectors' rows");
                assert_eq!(rows.to_vecs(), *side, "net {k}: the round trip");
                let mut pushed = Rows::default();
                side.iter().for_each(|r| pushed.push(r));
                assert_eq!(pushed, rows, "net {k}: built row by row");
                let want = (side.clone(), side.iter().map(Vec::len).collect::<Vec<_>>(), side.iter().map(Vec::len).sum::<usize>());
                assert_eq!(reads(&rows), want, "net {k}: Rows");
                assert_eq!(reads(side.as_slice()), want, "net {k}: a slice of rows");
                assert_eq!(reads(side), want, "net {k}: a vector of rows");
                assert!(rows.iter().zip(side.iter()).all(|(r, s)| r.iter().map(|&c| c as usize).eq(s.iter().copied())));
                assert_eq!((rows.len(), rows.nnz()), (side.len(), want.2));
            }
        }
        let none = Rows::default();
        assert!(none.is_empty() && none.nnz() == 0 && none.to_vecs().is_empty() && none == Rows::from_vecs(&[]));
        assert_eq!(Rows::from_vecs(&[vec![], vec![]]).to_vecs(), vec![Vec::<usize>::new(); 2]);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    #[should_panic(expected = "past u32::MAX")]
    fn a_column_past_u32_is_refused() {
        Rows::default().push(&[1, u32::MAX as usize + 1]);
    }
}
