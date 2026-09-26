//! `verity/flock-ir-block/v1`: identical units of an IR subcircuit template (lane flock-ir-lowering's `flock-ir-unit/v1`
//! netlists, `verity_flock.ir_lower`) as ONE block R1CS, so the CPU prover and Flock-CUDA's host-witness mode prove it.
//!
//! Block `b` holds units `U b .. U b + U - 1` (instance-major over the instance file), unit `u` at the 2^unit_log-bit slot
//! `u`, so `k_log = unit_log + log2 U`. `A = I ⊗ (I_U ⊗ unit) + Δ` (and B), where Δ makes every unit's constant column a
//! copy of the one pinned column (slot 0's). Units past the file's last (the block count is a power of two, at least 8)
//! are units of all-zero inputs.
//!
//! Public regions: every input word and every output word of every unit (the 128-bit word column, unused bits zero),
//! each opened at two verifier points drawn after the commitment (both reps). The verifier computes their values from its
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

/// A `flock-ir-unit/v1` netlist.
pub struct IrUnitNet {
    pub name: String,
    pub useful: usize,
    pub const_pos: usize,
    pub in_words: usize,
    pub in_bits: Vec<usize>,
    pub out_cols: Vec<usize>,
    pub out_bits: Vec<usize>,
    pub a: Vec<Vec<usize>>,
    pub b: Vec<Vec<usize>>,
    pub unit_log: usize,
    pub sha256: [u8; 32],
}

impl IrUnitNet {
    pub fn parse(text: &str) -> Self {
        let sha256: [u8; 32] = Sha256::digest(text.as_bytes()).into();
        let mut lines = text.lines();
        let h: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
        assert_eq!(h[0], "flock-ir-unit/v1", "not a flock-ir-unit/v1 netlist");
        let n = |i: usize| -> usize { h[i].parse().unwrap() };
        let (useful, const_pos, in_words, n_in) = (n(2), n(3), n(4), n(5));
        let in_bits: Vec<usize> = (0..n_in).map(|i| n(6 + i)).collect();
        let (out_col, out_words, n_out) = (n(6 + n_in), n(7 + n_in), n(8 + n_in));
        let out_bits: Vec<usize> = (0..n_out).map(|i| n(9 + n_in + i)).collect();
        let out_cols: Vec<usize> = (out_col..out_col + out_words).collect();
        assert!(out_bits.iter().sum::<usize>() <= out_words * WORD);
        assert!(const_pos == useful - 1 && in_bits.iter().sum::<usize>() <= in_words * WORD);
        let (mut a, mut b) = (Vec::with_capacity(useful), Vec::with_capacity(useful));
        for _ in 0..useful {
            let v: Vec<usize> = lines.next().expect("row").split_whitespace().map(|t| t.parse().unwrap()).collect();
            let na = v[0];
            let nb = v[1 + na];
            a.push(v[1..1 + na].to_vec());
            b.push(v[2 + na..2 + na + nb].to_vec());
        }
        let used: usize = in_bits.iter().sum();
        for i in 0..in_words * WORD {
            let want: Vec<usize> = if i < used { vec![i] } else { vec![] };
            assert!(a[i] == want && b[i] == want, "unit row {i} is not an input row");
        }
        assert!(a[const_pos] == [const_pos] && b[const_pos] == [const_pos]);
        let unit_log = useful.next_power_of_two().trailing_zeros() as usize;
        IrUnitNet { name: h[1].to_string(), useful, const_pos, in_words, in_bits, out_cols, out_bits, a, b, unit_log: unit_log.max(7), sha256 }
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
        let used: usize = self.in_bits.iter().sum();
        for (i, zi) in z.iter_mut().enumerate().take(used) {
            *zi = input(i);
        }
        z[self.const_pos] = !0;
        for i in first..u {
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
    In,
    Out,
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
        let regions = vec![shape(Region::In, net.in_words, 0), shape(Region::Out, net.out_cols.len(), net.out_cols[0])];
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
            (Region::In, true) => inst.input(g),
            (Region::In, false) => zero,
            (Region::Out, true) => inst.output(g),
            (Region::Out, false) => &self.dummy_out,
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
