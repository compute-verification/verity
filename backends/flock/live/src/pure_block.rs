//! `verity/flock-pure-block/v1`: the Flock backend's committed relation as ONE block R1CS, so Flock-CUDA can prove it
//! (no union prover, no wiring argument). Lane flock-gpu-link; the backend and the CPU union statement are flock-backend's.
//!
//! Block `b = 3 v + c` is chunk `c` of VU `v` (a 16-bit row is 3 BLAKE3 chunks, each read by 32 units). `k_log` 20:
//! - slots 0..16 (2^14 bits each): the x row's chunk `c`, 16 compressions (Flock's circuit);
//! - slots 16..32: the W column's chunk `c`;
//! - 2^13-bit positions 64..96: the 32 units `j = 32 c + u` (the pinned `flock-unit-io/v1` lowering);
//! - the rest is empty (forced zero).
//!
//! `A = I ⊗ circuit + Δ` (and B), where Δ makes, inside the block:
//! - each chunk a chain (chaining input of block j ≥ 1 = `out_lo` of block j − 1), with block_len 64 and flags
//!   (KEYED_HASH, CHUNK_START, CHUNK_END) constant;
//! - unit u's x and W input bits copies of the message bits they hash (`A = B = [message bit]`, as the unit's input rows
//!   are `A = B = [i]`), and its accumulator input a copy of unit u − 1's `c_out` (u ≥ 1);
//! - every sub-circuit's constant a copy of the one pinned column (slot 0's).
//!
//! Public regions, each opened at two verifier points drawn after the commitment (both reps):
//! the key (slot 0 and 16 chaining inputs), the counter (all 32 compressions: `c`), the chunk chaining values (slot 15
//! and 31 `out_lo`, sent in `Commit` and checked natively against the leaf digests), unit 0's accumulator input and
//! unit 31's `c_out` (the instance file's recorded accumulators: 0, then `accs[32c + 31]`), and unit 31's `y16` (the
//! output word at c = 2, prover-committed at c < 2). Dummy blocks (the block count is a power of two) are chunks of zero
//! messages under a zero key and zero-input units.

use std::{array::from_fn, sync::OnceLock};

use flock_core::{
    lincheck::{LincheckCircuit, SparseMatrixCircuit, pack_z_lincheck_from_packed},
    proof::eq_skip_weights,
    r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
    verifier::ExtraClaim,
};
use flock_field::F128;
use flock_hash::blake3_compress;
use flock_prover::r1cs_hashes::blake3 as b3;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use crate::chunk::{flags_of, key_words, row_key, comp, log2ceil};

pub const K_LOG: usize = 20;
pub const UNIT_LOG: usize = 13;
pub const COMP_SLOTS: usize = 32;
pub const UNITS: usize = 32;
pub const UNIT_POS0: usize = 64;
pub const PIN: usize = b3::Z_CONST_POS;
pub const TAG: &[u8] = b"verity/flock-pure-block/v1";
/// Coordinates per verifier point: the regions' free in-block bits come from `r[..16]`, block bits from `r[16..]`.
pub const PT_LOCAL: usize = 16;

/// The `flock-unit-io/v1` lowered unit (flock-backend's `verity_flock.lowering`), read-only.
pub struct UnitNet {
    pub relation: String,
    pub useful: usize,
    pub const_pos: usize,
    pub out_cols: Vec<usize>,
    pub a: Vec<Vec<usize>>,
    pub b: Vec<Vec<usize>>,
    pub sha256: [u8; 32],
}

impl UnitNet {
    pub fn parse(text: &str) -> Self {
        let sha256: [u8; 32] = Sha256::digest(text.as_bytes()).into();
        let mut lines = text.lines();
        let h: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
        assert_eq!(h[0], "flock-unit-io/v1", "not a flock-unit-io/v1 netlist");
        let n = |i: usize| -> usize { h[i].parse().unwrap() };
        let (useful, const_pos, n_in, ni) = (n(2), n(3), n(4), n(5));
        assert_eq!((0..ni).map(|i| n(6 + i)).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
        let no = n(6 + ni);
        let out_cols: Vec<usize> = (0..no).map(|i| n(7 + ni + i)).collect();
        assert!(useful <= 1 << UNIT_LOG && const_pos == useful - 1 && n_in == 544);
        let (mut a, mut b) = (Vec::with_capacity(useful), Vec::with_capacity(useful));
        for _ in 0..useful {
            let v: Vec<usize> = lines.next().expect("row").split_whitespace().map(|t| t.parse().unwrap()).collect();
            let na = v[0];
            let nb = v[1 + na];
            a.push(v[1..1 + na].to_vec());
            b.push(v[2 + na..2 + na + nb].to_vec());
        }
        for i in 0..544 {
            assert!(a[i] == [i] && b[i] == [i], "unit row {i} is not an input row");
        }
        assert!(a[const_pos] == [const_pos] && b[const_pos] == [const_pos]);
        UnitNet { relation: h[1].to_string(), useful, const_pos, out_cols, a, b, sha256 }
    }

    pub fn load(path: &str) -> Self {
        Self::parse(&std::fs::read_to_string(path).expect("netlist"))
    }

    pub fn epilogue(&self) -> bool {
        self.out_cols.len() > 1
    }

    fn matrices(&self) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let k = 1usize << UNIT_LOG;
        let pad = |rows: &Vec<Vec<usize>>| {
            let mut r = rows.clone();
            r.resize(k, Vec::new());
            SparseBinaryMatrix { num_rows: k, num_cols: k, rows: r.into() }
        };
        (pad(&self.a), pad(&self.b))
    }

    /// Up to 64 units at once (lane l = unit l): inputs are 544 bits per unit; returns (z, a, b) per row, bit-sliced.
    pub fn eval64(&self, input: impl Fn(usize) -> u64) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let u = self.useful;
        let mut z = vec![0u64; u];
        for (i, zi) in z.iter_mut().enumerate().take(544) {
            *zi = input(i);
        }
        z[self.const_pos] = !0;
        for i in 544..u {
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
}

/// The instance file (`flock-pure-instances/v1`, flock-backend's `verity_flock.instances`), read-only.
#[derive(Clone)]
pub struct Instances {
    pub header: serde_json::Value,
    pub relation: String,
    pub vus: usize,
    pub units: usize,
    pub row_bytes: usize,
    pub x: Vec<u8>,
    pub w: Vec<u8>,
    pub accs: Vec<u32>,
    pub out: Vec<u32>,
    pub y: Vec<u32>,
}

fn le_words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

impl Instances {
    pub fn load(path: &str) -> Self {
        let b = std::fs::read(path).expect("instances");
        let nl = b.iter().position(|&c| c == b'\n').unwrap();
        let header: serde_json::Value = serde_json::from_slice(&b[..nl]).unwrap();
        assert_eq!(header["format"], "flock-pure-instances/v1");
        let g = |k: &str| header[k].as_u64().unwrap() as usize;
        let (vus, units, row_bytes) = (g("vus"), g("units"), g("row_bytes"));
        let mut at = nl + 1;
        let mut take = |n: usize| {
            let s = b[at..at + n].to_vec();
            at += n;
            s
        };
        let x = take(vus * row_bytes);
        let w = take(vus * row_bytes);
        let accs = le_words(&take(vus * units * 4));
        let out = le_words(&take(vus * 4));
        let y = le_words(&take(vus * 4));
        assert_eq!(at, b.len(), "instance file length");
        Instances { relation: header["relation"].as_str().unwrap().to_string(), header, vus, units, row_bytes, x, w, accs, out, y }
    }

    pub fn row(&self, role: usize, v: usize) -> &[u8] {
        let src = if role == 0 { &self.x } else { &self.w };
        &src[v * self.row_bytes..(v + 1) * self.row_bytes]
    }

    pub fn acc(&self, v: usize, j: usize) -> u32 {
        self.accs[v * self.units + j]
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Region {
    Key,
    Counter,
    Cv,
    AccIn,
    AccOut,
    Y,
}

/// One public region: the in-block bits it fixes (`fixed` over `mask`) and its free in-block bits, LSB first.
pub struct RegionShape {
    pub region: Region,
    pub free: Vec<usize>,
    pub mask: u32,
    pub fixed: u32,
}

fn shape(region: Region, net: &UnitNet) -> RegionShape {
    let bits = |lo: usize, n: usize| (lo..lo + n).collect::<Vec<_>>();
    // (free bits, fixed value) in the 20 in-block bits
    let (free, fixed): (Vec<usize>, u32) = match region {
        Region::Key => ([bits(0, 8), vec![18]].concat(), 0),
        Region::Counter => ([bits(0, 6), bits(14, 5)].concat(), 1 << 10),
        Region::Cv => ([bits(0, 8), vec![18]].concat(), (1 << 8) | (0xF << 14)),
        Region::AccIn => (bits(0, 5), (1 << 9) | ((UNIT_POS0 as u32) << UNIT_LOG)),
        Region::AccOut => (bits(0, 5), ((net.out_cols[0] as u32) << 7) | (((UNIT_POS0 + UNITS - 1) as u32) << UNIT_LOG)),
        Region::Y => (bits(0, 4), ((net.out_cols[1] as u32) << 7) | (((UNIT_POS0 + UNITS - 1) as u32) << UNIT_LOG)),
    };
    let mask = ((1u32 << K_LOG) - 1) & !free.iter().fold(0u32, |m, &b| m | (1 << b));
    assert_eq!(fixed & !mask, 0);
    RegionShape { region, free, mask, fixed }
}

/// The prover's per-block publics: each real block's x and W chunk chaining values and its unit 31 `y16` word.
#[derive(Clone)]
pub struct Publics {
    pub cv: Vec<[[u32; 8]; 2]>,
    pub y16: Vec<u32>,
}

impl Publics {
    pub fn bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.cv.len() * 68);
        for (c, y) in self.cv.iter().zip(&self.y16) {
            for w in c.iter().flatten() {
                v.extend_from_slice(&w.to_le_bytes());
            }
            v.extend_from_slice(&y.to_le_bytes());
        }
        v
    }

    pub fn parse(b: &[u8], blocks: usize) -> Result<Self, String> {
        if b.len() != 68 * blocks {
            return Err(format!("C4: {} bytes of publics, want {}", b.len(), 68 * blocks));
        }
        let w = le_words(b);
        let cv = (0..blocks).map(|i| from_fn(|r| from_fn(|k| w[17 * i + 8 * r + k]))).collect();
        let y16 = (0..blocks).map(|i| w[17 * i + 16]).collect();
        Ok(Publics { cv, y16 })
    }
}

pub struct PureCircuit {
    pub unit: SparseMatrixCircuit<'static>,
    pub da: Vec<(u32, u32)>,
    pub db: Vec<(u32, u32)>,
}

impl LincheckCircuit for PureCircuit {
    fn n_cols(&self) -> usize {
        1 << K_LOG
    }

    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        let (cs, us) = (1usize << b3::K_LOG, 1usize << UNIT_LOG);
        assert_eq!(eq.len(), 1 << K_LOG);
        let c0 = eq[..us].iter().position(|e| *e != F128::ZERO).expect("eq tensor vanishes on the first unit slot");
        let inv = eq[c0].inv();
        let base_c = comp().csc_lincheck_circuit().fold_alpha_batched(alpha, &eq[..cs]);
        let base_u = self.unit.fold_alpha_batched(alpha, &eq[..us]);
        let mut out = vec![F128::ZERO; 1 << K_LOG];
        out.par_chunks_mut(us).enumerate().for_each(|(p, o)| {
            if p < 2 * COMP_SLOTS {
                let s = eq[((p >> 1) << b3::K_LOG) | c0] * inv;
                let half = (p & 1) * us;
                for (x, y) in o.iter_mut().zip(&base_c[half..half + us]) {
                    *x = s * *y;
                }
            } else if (UNIT_POS0..UNIT_POS0 + UNITS).contains(&p) {
                let s = eq[(p << UNIT_LOG) | c0] * inv;
                for (x, y) in o.iter_mut().zip(&base_u) {
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
        Some(PIN)
    }
}

fn delta(net: &UnitNet) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let c = comp();
    let (a, b) = (&c.a_0.rows, &c.b_0.rows);
    let z = b3::Z_CONST_POS;
    let input = |r: usize| assert!(a[r] == [r] && b[r] == [z], "compression row {r} is not an input row");
    let (mut da, mut db) = (Vec::new(), Vec::new());
    let both = |da: &mut Vec<(u32, u32)>, db: &mut Vec<(u32, u32)>, i: usize, old: usize, new: usize| {
        for d in [da, db] {
            d.push((i as u32, old as u32));
            d.push((i as u32, new as u32));
        }
    };
    for s in 0..COMP_SLOTS {
        let (j, g) = (s % 16, |col: usize| (s << b3::K_LOG) + col);
        if s > 0 {
            both(&mut da, &mut db, g(z), g(z), PIN);
        }
        if j > 0 {
            for r in 0..256 {
                input(b3::CV_BASE + r);
                da.push((g(b3::CV_BASE + r) as u32, g(b3::CV_BASE + r) as u32));
                da.push((g(b3::CV_BASE + r) as u32, (((s - 1) << b3::K_LOG) + b3::OUT_LO_BASE + r) as u32));
            }
        }
        for (base, val) in [(b3::BLEN_BASE, 64u32), (b3::FLAGS_BASE, flags_of(j))] {
            for bit in 0..32 {
                let r = base + bit;
                input(r);
                da.push((g(r) as u32, g(r) as u32));
                if (val >> bit) & 1 == 1 {
                    da.push((g(r) as u32, g(z) as u32));
                }
            }
        }
    }
    for u in 0..UNITS {
        let base = (UNIT_POS0 + u) << UNIT_LOG;
        both(&mut da, &mut db, base + net.const_pos, base + net.const_pos, PIN);
        for role in 0..2 {
            let slot = 16 * role + u / 2;
            for t in 0..256 {
                let msg = (slot << b3::K_LOG) + b3::M_BASE + (u % 2) * 256 + t;
                both(&mut da, &mut db, base + 256 * role + t, base + 256 * role + t, msg);
            }
        }
        if u > 0 {
            let prev = ((UNIT_POS0 + u - 1) << UNIT_LOG) + net.out_cols[0] * 128;
            for t in 0..32 {
                both(&mut da, &mut db, base + 512 + t, base + 512 + t, prev + t);
            }
        }
    }
    (da, db)
}

pub struct PureStmt {
    pub net: UnitNet,
    pub vus: usize,
    pub blocks_real: usize,
    pub nbl: usize,
    pub m: usize,
    pub m_pts: usize,
    pub regions: Vec<RegionShape>,
    pub circuit: PureCircuit,
    pub r1cs: BlockR1cs,
    pub digest: [u8; 32],
    pub keys: [[u32; 8]; 2],
    pub dummy_cv: [u32; 8],
    pub dummy_unit: (u32, u32),
}

impl PureStmt {
    pub fn new(net: UnitNet, vus: usize) -> Self {
        let blocks_real = 3 * vus;
        let nbl = log2ceil(blocks_real).max(3);
        let m = K_LOG + nbl;
        let (da, db) = delta(&net);
        let mut regions: Vec<RegionShape> = [Region::Key, Region::Counter, Region::Cv, Region::AccIn, Region::AccOut]
            .into_iter()
            .map(|r| shape(r, &net))
            .collect();
        if net.epilogue() {
            regions.push(shape(Region::Y, &net));
        }
        let mut cv = [0u32; 8];
        for j in 0..16 {
            cv = blake3_compress(&cv, &[0u32; 16], 0, 64, flags_of(j))[..8].try_into().unwrap();
        }
        let (z, _, _) = net.eval64(|_| 0);
        let word = |col: usize, n: usize| (0..n).fold(0u32, |w, t| w | (((z[col * 128 + t] & 1) as u32) << t));
        let dummy_unit = (word(net.out_cols[0], 32), if net.epilogue() { word(net.out_cols[1], 16) } else { 0 });
        let mut h = Sha256::new();
        h.update(TAG);
        h.update(comp().statement_digest());
        h.update(net.sha256);
        for x in [vus, blocks_real, nbl, m, K_LOG, UNIT_LOG, COMP_SLOTS, UNITS, UNIT_POS0] {
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
        let kk = 1usize << K_LOG;
        let empty = || SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: vec![Vec::new(); kk].into() };
        let r1cs = BlockR1cs {
            m,
            k_log: K_LOG,
            k_skip: b3::K_SKIP,
            useful_bits: kk,
            a_0: empty(),
            b_0: empty(),
            c_0: SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: (0..kk).map(|i| vec![i]).collect::<Vec<_>>().into() },
            layout: WitnessLayout::RowMajor,
            const_pin: Some(PIN),
            digest_cache: OnceLock::from(digest),
            csc_cache: OnceLock::new(),
        };
        let (ua, ub) = net.matrices();
        let (ua, ub): (&'static SparseBinaryMatrix, &'static SparseBinaryMatrix) = (Box::leak(Box::new(ua)), Box::leak(Box::new(ub)));
        let unit = SparseMatrixCircuit::new(ua, ub);
        let keys = [key_words(&row_key(b"x")), key_words(&row_key(b"w"))];
        PureStmt { net, vus, blocks_real, nbl, m, m_pts: PT_LOCAL + nbl, regions, circuit: PureCircuit { unit, da, db }, r1cs, digest, keys,
                   dummy_cv: cv, dummy_unit }
    }

    pub fn blocks(&self) -> usize {
        1 << self.nbl
    }

    /// Every compression, block-major (block b's 32: x chunk then W chunk), and the chunk chaining values per block.
    pub fn compressions(&self, inst: &Instances) -> (Vec<b3::Compression>, Vec<[[u32; 8]; 2]>) {
        let per: Vec<(Vec<b3::Compression>, [[u32; 8]; 2])> = (0..self.blocks())
            .into_par_iter()
            .map(|b| {
                let real = b < self.blocks_real;
                let (v, c) = (b / 3, b % 3);
                let mut out = Vec::with_capacity(COMP_SLOTS);
                let mut cvs = [[0u32; 8]; 2];
                for role in 0..2 {
                    let mut cv = if real { self.keys[role] } else { [0; 8] };
                    let ctr = if real { c as u64 } else { 0 };
                    for j in 0..16 {
                        let msg: [u32; 16] = if real {
                            let at = c * 1024 + 64 * j;
                            let row = inst.row(role, v);
                            from_fn(|i| u32::from_le_bytes(row[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
                        } else {
                            [0; 16]
                        };
                        out.push((cv, msg, ctr, 64, flags_of(j)));
                        cv = blake3_compress(&cv, &msg, ctr, 64, flags_of(j))[..8].try_into().unwrap();
                    }
                    cvs[role] = cv;
                }
                (out, cvs)
            })
            .collect();
        let cvs = per.iter().map(|p| p.1).take(self.blocks_real).collect();
        (per.into_iter().flat_map(|p| p.0).collect(), cvs)
    }

    /// Unit (block b, u)'s 544 input bits: x bytes, W bytes (as the compressions hash them), accumulator in.
    pub fn unit_inputs(&self, comps: &[b3::Compression], b: usize, u: usize, c_in: u32) -> [F128; 5] {
        let words = |slot: usize| -> [F128; 2] {
            let m = &comps[b * COMP_SLOTS + slot].1;
            let h = (u % 2) * 8;
            [crate::chunk::pack4(&m[h..h + 4]), crate::chunk::pack4(&m[h + 4..h + 8])]
        };
        let (x, w) = (words(u / 2), words(16 + u / 2));
        [x[0], x[1], w[0], w[1], F128::new(c_in as u64, 0)]
    }

    /// The honest accumulator input of every unit (the recorded accumulators; zero for dummy blocks).
    pub fn c_in(&self, inst: &Instances, b: usize, u: usize) -> u32 {
        if b >= self.blocks_real {
            return 0;
        }
        let (v, j) = (b / 3, 32 * (b % 3) + u);
        if j == 0 { 0 } else { inst.acc(v, j - 1) }
    }

    /// `(z, a, b, z_lincheck)` of the block witness: compressions, then units evaluated from `inputs(b, u)`.
    pub fn witness(&self, comps: &[b3::Compression], inputs: &(dyn Fn(usize, usize) -> [F128; 5] + Sync)) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let n = 1usize << (self.m - 7);
        let (cz, ca, cb) = b3::generate_witness_with_ab_packed(comps, self.nbl + 5);
        let (mut z, mut a, mut b) = (vec![F128::ZERO; n], vec![F128::ZERO; n], vec![F128::ZERO; n]);
        let wpb = 1usize << (K_LOG - 7);
        let cw = 1usize << (b3::K_LOG - 7);
        for (dst, src) in [(&mut z, &cz), (&mut a, &ca), (&mut b, &cb)] {
            dst.par_chunks_mut(wpb).enumerate().for_each(|(blk, d)| {
                for s in 0..COMP_SLOTS {
                    let ci = blk * COMP_SLOTS + s;
                    d[s * cw..(s + 1) * cw].copy_from_slice(&src[ci * cw..(ci + 1) * cw]);
                }
            });
        }
        drop((cz, ca, cb));
        let uw = 1usize << (UNIT_LOG - 7);
        let total = self.blocks() * UNITS;
        let groups: Vec<Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)>> = (0..total.div_ceil(64))
            .into_par_iter()
            .map(|g| {
                let ids: Vec<usize> = (64 * g..(64 * g + 64).min(total)).collect();
                let ins: Vec<[F128; 5]> = ids.iter().map(|&i| inputs(i / UNITS, i % UNITS)).collect();
                let (zz, aa, bb) = self.net.eval64(|bit| {
                    let mut x = 0u64;
                    for (l, r) in ins.iter().enumerate() {
                        let w = r[bit >> 7];
                        let j = bit & 127;
                        x |= ((if j < 64 { w.lo >> j } else { w.hi >> (j - 64) }) & 1) << l;
                    }
                    x
                });
                ids.iter()
                    .enumerate()
                    .map(|(l, &i)| {
                        let pack = |v: &Vec<u64>| -> Vec<F128> {
                            let mut o = vec![F128::ZERO; uw];
                            for (r, &x) in v.iter().enumerate() {
                                if (x >> l) & 1 == 1 {
                                    if r % 128 < 64 { o[r / 128].lo |= 1 << (r % 128) } else { o[r / 128].hi |= 1 << (r % 128 - 64) }
                                }
                            }
                            o
                        };
                        (i, pack(&zz), pack(&aa), pack(&bb))
                    })
                    .collect()
            })
            .collect();
        for (i, uz, ua, ub) in groups.into_iter().flatten() {
            let (blk, u) = (i / UNITS, i % UNITS);
            let at = blk * wpb + (UNIT_POS0 + u) * uw;
            z[at..at + uw].copy_from_slice(&uz);
            a[at..at + uw].copy_from_slice(&ua);
            b[at..at + uw].copy_from_slice(&ub);
        }
        let zl = pack_z_lincheck_from_packed(&z, self.m, K_LOG);
        (z, a, b, zl)
    }

    /// The value of every public region at every block: bit `f` of the region's free assignment.
    fn region_bits(&self, reg: &RegionShape, inst: &Instances, p: &Publics, b: usize) -> Vec<u8> {
        let real = b < self.blocks_real;
        let (v, c) = (b / 3, b % 3);
        let le = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        match reg.region {
            Region::Key => {
                let k = |r: usize| if real { self.keys[r] } else { [0; 8] };
                [le(&k(0)), le(&k(1))].concat()
            }
            Region::Counter => (if real { c as u64 } else { 0 }).to_le_bytes().repeat(COMP_SLOTS),
            Region::Cv => {
                let cv = |r: usize| if real { p.cv[b][r] } else { self.dummy_cv };
                [le(&cv(0)), le(&cv(1))].concat()
            }
            Region::AccIn => self.c_in(inst, b, 0).to_le_bytes().to_vec(),
            Region::AccOut => (if real { inst.acc(v, 32 * c + 31) } else { self.dummy_unit.0 }).to_le_bytes().to_vec(),
            Region::Y => {
                let y = if !real { self.dummy_unit.1 } else if c == 2 { inst.out[v] } else { p.y16[b] };
                (y as u16).to_le_bytes().to_vec()
            }
        }
    }

    /// The region's claim point from verifier point `r` (m_pts coordinates).
    pub fn region_point(&self, reg: &RegionShape, r: &[F128]) -> Vec<F128> {
        let mut p = vec![F128::ZERO; self.m];
        for j in 0..K_LOG {
            if (reg.mask >> j) & 1 == 1 && (reg.fixed >> j) & 1 == 1 {
                p[j] = F128::ONE;
            }
        }
        for (t, &j) in reg.free.iter().enumerate() {
            p[j] = r[t];
        }
        p[K_LOG..].copy_from_slice(&r[PT_LOCAL..PT_LOCAL + self.nbl]);
        p
    }

    pub fn region_value(&self, reg: &RegionShape, r: &[F128], inst: &Instances, p: &Publics) -> F128 {
        let nf = reg.free.len();
        let wt = eq_skip_weights(&r[..nf]);
        let eb = eq_skip_weights(&r[PT_LOCAL..PT_LOCAL + self.nbl]);
        let nbytes = (1usize << nf).div_ceil(8);
        let mut t = vec![[F128::ZERO; 256]; nbytes];
        for (i, tb) in t.iter_mut().enumerate() {
            for x in 1..256usize {
                let rest = x & (x - 1);
                let bit = 8 * i + (x ^ rest).trailing_zeros() as usize;
                tb[x] = tb[rest] + if bit < wt.len() { wt[bit] } else { F128::ZERO };
            }
        }
        (0..self.blocks())
            .into_par_iter()
            .map(|b| {
                let bits = self.region_bits(reg, inst, p, b);
                eb[b] * bits.iter().enumerate().fold(F128::ZERO, |a, (i, &x)| a + t[i][x as usize])
            })
            .reduce(|| F128::ZERO, |a, b| a + b)
    }

    pub fn extra_points(&self, pts: &[Vec<F128>]) -> Vec<Vec<F128>> {
        self.regions.iter().flat_map(|reg| pts.iter().map(move |r| self.region_point(reg, r))).collect()
    }

    pub fn extra_claims(&self, pts: &[Vec<F128>], inst: &Instances, p: &Publics) -> Vec<ExtraClaim> {
        let values: Vec<F128> = self.regions.iter().flat_map(|reg| pts.iter().map(move |r| self.region_value(reg, r, inst, p))).collect();
        self.extra_points(pts)
            .into_iter()
            .zip(values)
            .map(|(pt, value)| ExtraClaim { value, skip_weights: eq_skip_weights(&pt[..6]), x_full: pt[6..].to_vec() })
            .collect()
    }
}
