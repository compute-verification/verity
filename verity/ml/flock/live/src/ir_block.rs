//! `verity/flock-ir-block/v1`: identical units of an IR subcircuit template (lane flock-ir-lowering's `flock-ir-unit/v1`
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
    lincheck::{LincheckCircuit, SparseMatrixCircuit, pack_z_lincheck_from_packed},
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
    pub a: Vec<Vec<usize>>,
    pub b: Vec<Vec<usize>>,
    pub unit_log: usize,
    pub sha256: [u8; 32],
    /// The units' leaf maps (the `LEAVES <json>` line: per-unit `in` / `out` instance leaves, `in_ports` / `out_ports` as
    /// `[leaves, bits]`), covered by `sha256` and so by the pin (red-team-flock-2 IR6).
    pub leaves: Option<serde_json::Value>,
    /// A cut lowering's cut structure (its final `CUT <json>` line: `words`, `tail`, per-unit `in` / `out` ports), covered by
    /// `sha256` and so by the pin (red-team-flock-2 IR4).
    pub cut: Option<serde_json::Value>,
    /// A net committed at a position map (the Lean `Net.ofRowsAt`, a templated unit's `Tmpl.posMap`; its final
    /// `ORDER <json>` line): logical row `x` is row `order[x]`, so the rows are evaluated in that order, not by column.
    pub order: Option<Vec<usize>>,
    /// `a` and `b` as [`IrUnitNet::eval64_flat`] reads them, built on its first call: `a` and `b` must not change after it.
    pub flat: FlatCache,
}

/// A net's rows in one flat pass: row i's A columns other than i are `cols[off[2i]..off[2i + 1]]`, its B columns other than i
/// `cols[off[2i + 1]..off[2i + 2]]`, and bit 0 (1) of `own[i]` whether A (B) reads column i an odd number of times. Every
/// column a row reads is an earlier one or the constant's, so one pass leaves each row's z, a and b final.
struct Flat {
    off: Vec<u32>,
    cols: Vec<u32>,
    own: Vec<u8>,
}

/// None: a row [`Flat`] cannot express (a later column, an input row that is not `[i]` or empty, more than 2^32 entries).
/// A clone starts empty, so a copy whose rows are then edited builds its own.
#[derive(Default)]
pub struct FlatCache(OnceLock<Option<Flat>>);

impl Clone for FlatCache {
    fn clone(&self) -> Self {
        FlatCache::default()
    }
}

impl Flat {
    fn of(net: &IrUnitNet) -> Option<Flat> {
        let (u, cp, first) = (net.useful, net.const_pos, net.in_words * WORD);
        if u == 0 || cp != u - 1 || first > cp || net.a[cp] != [cp] || net.b[cp] != [cp] || net.order.is_some() {
            return None;
        }
        const CHUNK: usize = 1 << 14;
        let parts: Vec<Option<(Vec<u32>, Vec<u32>, Vec<u8>)>> = (0..cp.div_ceil(CHUNK)).into_par_iter().map(|p| {
            let rows = p * CHUNK..((p + 1) * CHUNK).min(cp);
            let (mut off, mut cols, mut own) = (Vec::with_capacity(2 * rows.len()), Vec::new(), Vec::with_capacity(rows.len()));
            for i in rows {
                let mut o = 0u8;
                for (side, row) in [&net.a[i], &net.b[i]].into_iter().enumerate() {
                    off.push(cols.len() as u32);
                    if i < first {
                        match row.as_slice() {
                            [] => {}
                            [c] if *c == i => o |= 1 << side,
                            _ => return None,
                        }
                        continue;
                    }
                    for &c in row {
                        if c == i {
                            o ^= 1 << side;
                        } else if c < i || c == cp {
                            cols.push(c as u32);
                        } else {
                            return None;
                        }
                    }
                }
                own.push(o);
            }
            Some((off, cols, own))
        }).collect();
        let total: usize = parts.iter().map(|p| p.as_ref().map_or(0, |x| x.1.len())).sum();
        if total >= u32::MAX as usize {
            return None;
        }
        let (mut off, mut cols, mut own) = (Vec::with_capacity(2 * cp + 1), Vec::with_capacity(total), Vec::with_capacity(cp));
        for p in parts {
            let (o, c, w) = p?;
            let base = cols.len() as u32;
            off.extend(o.iter().map(|x| x + base));
            cols.extend_from_slice(&c);
            own.extend_from_slice(&w);
        }
        off.push(cols.len() as u32);
        Some(Flat { off, cols, own })
    }

    /// [`IrUnitNet::eval64`] in one pass.
    fn eval64(&self, net: &IrUnitNet, input: impl Fn(usize) -> u64) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let (u, cp, first) = (net.useful, net.const_pos, net.in_words * WORD);
        let (mut z, mut a, mut b) = (vec![0u64; u], vec![0u64; u], vec![0u64; u]);
        let mask = |o: u8, side: u8| 0u64.wrapping_sub(((o >> side) & 1) as u64);
        for i in 0..first {
            let o = self.own[i];
            if o & 1 == 1 {
                z[i] = input(i);
            }
            a[i] = z[i] & mask(o, 0);
            b[i] = z[i] & mask(o, 1);
        }
        z[cp] = !0;
        (a[cp], b[cp]) = (!0, !0);
        let (off, cols) = (&self.off[..], &self.cols[..]);
        for i in first..cp {
            let (s, m, e) = (off[2 * i] as usize, off[2 * i + 1] as usize, off[2 * i + 2] as usize);
            let x = cols[s..m].iter().fold(0u64, |x, &k| x ^ z[k as usize]);
            let y = cols[m..e].iter().fold(0u64, |y, &k| y ^ z[k as usize]);
            let zi = x & y;
            let o = self.own[i];
            z[i] = zi;
            a[i] = x ^ (zi & mask(o, 0));
            b[i] = y ^ (zi & mask(o, 1));
        }
        (z, a, b)
    }
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
        let (mut a, mut b) = (Vec::with_capacity(useful), Vec::with_capacity(useful));
        for _ in 0..useful {
            let v: Vec<usize> = lines.next().expect("row").split_whitespace().map(|t| t.parse().unwrap()).collect();
            let na = v[0];
            let nb = v[1 + na];
            a.push(v[1..1 + na].to_vec());
            b.push(v[2 + na..2 + na + nb].to_vec());
        }
        for i in 0..in_words * WORD {
            assert!((a[i] == [i] && b[i] == [i]) || (a[i].is_empty() && b[i].is_empty()), "unit row {i} is not an input row");
        }
        assert!(a[const_pos] == [const_pos] && b[const_pos] == [const_pos]);
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
            assert!(seen.iter().enumerate().all(|(p, &s)| s || (a[p].is_empty() && b[p].is_empty())), "ORDER: an unplaced row is not empty");
        }
        let unit_log = (useful.next_power_of_two().trailing_zeros() as usize).max(7);
        // D11: an output group lies inside its unit's slot, so a region at its column stays in the slot
        if let Some(g) = out_groups.iter().find(|g| g.col + g.words > (1 << unit_log) / WORD) {
            panic!("flock-ir-unit/v2: output group at word {} ({} words) overruns its unit's {} words (D11)", g.col, g.words, (1usize << unit_log) / WORD);
        }
        IrUnitNet { name: h[1].to_string(), useful, const_pos, in_groups, out_groups, in_words, out_cols, a, b, unit_log, sha256,
                    leaves, cut, order, flat: FlatCache::default() }
    }

    /// Whether column `i` is an input column (its row is `A = B = [i]`).
    pub fn is_input(&self, i: usize) -> bool {
        i < self.in_words * WORD && self.a[i].len() == 1
    }

    pub fn load(path: &str) -> Self {
        Self::parse(&std::fs::read_to_string(path).expect("netlist"))
    }

    fn matrices(&self) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let k = 1usize << self.unit_log;
        let pad = |rows: &Vec<Vec<usize>>| {
            let mut r = rows.clone();
            r.resize(k, Vec::new());
            SparseBinaryMatrix { num_rows: k, num_cols: k, rows: r.into() }
        };
        (pad(&self.a), pad(&self.b))
    }

    /// Up to 64 units at once (lane l = unit l): `input(col)` is the lanes' bit at input column `col`; returns (z, a, b) per
    /// row, bit-sliced.
    pub fn eval64(&self, input: impl Fn(usize) -> u64) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let u = self.useful;
        let first = self.in_words * WORD;
        let mut z = vec![0u64; u];
        for (i, zi) in z.iter_mut().enumerate().take(first) {
            if self.a[i].len() == 1 {
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
            let x = self.a[i].iter().fold(0u64, |s, &k| s ^ z[k]);
            let y = self.b[i].iter().fold(0u64, |s, &k| s ^ z[k]);
            z[i] = x & y;
        }
        let a: Vec<u64> = (0..u).map(|i| self.a[i].iter().fold(0u64, |s, &k| s ^ z[k])).collect();
        let b: Vec<u64> = (0..u).map(|i| self.b[i].iter().fold(0u64, |s, &k| s ^ z[k])).collect();
        (z, a, b)
    }

    /// [`IrUnitNet::eval64`], equal to it bit for bit, in one pass over flat u32 forms of `a` and `b` built on the first call
    /// (the two-pass evaluation where a row reads a later column).
    pub fn eval64_flat(&self, input: impl Fn(usize) -> u64) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        match self.flat.0.get_or_init(|| Flat::of(self)) {
            Some(f) => f.eval64(self, input),
            None => self.eval64(input),
        }
    }

    /// Builds the flat forms [`IrUnitNet::eval64_flat`] reads, if not yet built: whether it evaluates in one pass.
    pub fn one_pass(&self) -> bool {
        self.flat.0.get_or_init(|| Flat::of(self)).is_some()
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
        let assertion = net.b[i] == [net.const_pos] && net.a[i].contains(&i);
        for &c in net.a[i].iter().chain(&net.b[i]) {
            if c >= i && c != net.const_pos && !(assertion && c == i) {
                return Err(format!("unit row {i} reads column {c}, not an earlier one"));
            }
        }
        if net.a[i] == [i] && net.b[i] == [i] {
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
        let (mut a, mut b) = (Vec::new(), Vec::new());
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
            a.push(ra);
            b.push(rb);
        }
        IrUnitNet { name: "t".into(), useful, const_pos: cp, in_groups: vec![], out_groups: vec![], in_words, out_cols: vec![], a, b,
                    unit_log: useful.next_power_of_two().trailing_zeros().max(7) as usize, sha256: [0; 32], leaves: None, cut: None,
                    order: None, flat: FlatCache::default() }
    }

    /// `net` committed at a position map (`Net.ofRowsAt`): the computed rows scattered over twice their span, the constant
    /// last, every reference read through the map.
    fn placed(net: &IrUnitNet) -> IrUnitNet {
        let (first, cp) = (net.in_words * WORD, net.const_pos);
        let span = 2 * (cp - first) + 1;
        let pos: Vec<usize> = (0..=cp).map(|x| if x < first { x } else if x == cp { first + span } else { first + (x - first) * 7919 % span }).collect();
        let size = first + span + 1;
        let (mut a, mut b) = (vec![Vec::new(); size], vec![Vec::new(); size]);
        for x in 0..=cp {
            a[pos[x]] = net.a[x].iter().map(|&c| pos[c]).collect();
            b[pos[x]] = net.b[x].iter().map(|&c| pos[c]).collect();
        }
        IrUnitNet { useful: size, const_pos: size - 1, a, b, unit_log: size.next_power_of_two().trailing_zeros().max(7) as usize,
                    order: Some(pos), ..net.clone() }
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
            assert!(at.flat.0.get().is_some_and(|f| f.is_none()), "a placed net took the one-pass path");
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
            assert!(net.flat.0.get().is_some_and(|f| f.is_some()), "net {seed} took the two-pass path");
            assert!(net.clone().flat.0.get().is_none(), "a clone keeps the flat forms");
            // a row reading a later column: the two-pass evaluation, still equal
            let mut fwd = net.clone();
            let i = in_words * WORD + 3;
            fwd.a[i].push(i + 5);
            let input = |col: usize| col as u64 * 0x2545_f491_4f6c_dd1d;
            assert!(fwd.eval64_flat(input) == fwd.eval64(input));
            assert!(fwd.flat.0.get().is_some_and(|f| f.is_none()), "a later column took the one-pass path");
        }
    }
}
