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
//! unit 31's `c_out` (a prover-committed public word per block: block (v, c)'s AccOut and block (v, c + 1)'s AccIn open
//! the same word; AccIn of chunk 0 is +0; the verifier never sees the recorded accumulators), and unit 31's `y16` (the
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
use flock_prover::r1cs_hashes::{blake3 as b3, sha2 as sha2c};
use rayon::prelude::*;
use ::sha2::{Digest, Sha256};

use crate::chunk::{key_words, row_key, comp, log2ceil};

pub const K_LOG: usize = 20;
pub const UNIT_LOG: usize = 13;
pub const COMP_SLOTS: usize = 32;
pub const UNITS: usize = 32;
pub const UNIT_POS0: usize = 64;
pub const PIN: usize = b3::Z_CONST_POS;
pub const TAG: &[u8] = b"verity/flock-pure-block/v2";
/// Coordinates per verifier point: the regions' free in-block bits come from `r[..16]`, block bits from `r[16..]`.
pub const PT_LOCAL: usize = 16;

/// The `flock-unit-io/v1` lowered unit (flock-backend's `verity_flock.lowering`), read-only.
/// The block shape of one relation family.
/// - `Bf16` (16-bit rows, 3 chunks, 96 units): block (v, c) is chunk c of VU v; `k_log` 20; 16 + 16 compressions and 32 units.
/// - `Fp8` (8-bit rows, 1.5 chunks, 48 units): block v is the whole VU; `k_log` 21; x chunk 0 (16) and chunk 1 (8), then W's,
///   48 compressions and the 48 units at 2^13 positions 96..144. Key, counter, flags and block_len are all per-slot
///   constants, the accumulator chain is inside the block (c_in of unit 0 forced to +0), and only the four chunk chaining
///   values and the output word are public.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Layout {
    Bf16,
    Fp8,
    /// `sha256/row/v1` leaves: block v is the whole VU; each role is one SHA-256 chain (the verifier's midstate after the
    /// 64-byte prefix block, the row's 64-byte blocks, one constant padding block) of 2^15-bit compressions, then the
    /// units at 2^13 positions after them. 16-bit rows: 49 + 49 compressions, 96 units, k_log 22.
    ShaBf16,
    /// 8-bit rows: 25 + 25 compressions, 48 units, k_log 21.
    ShaFp8,
}

impl Layout {
    pub fn of(row_bytes: usize, units: usize, sha: bool) -> Self {
        match (row_bytes, units, sha) {
            (3072, 96, false) => Layout::Bf16,
            (1536, 48, false) => Layout::Fp8,
            (3072, 96, true) => Layout::ShaBf16,
            (1536, 48, true) => Layout::ShaFp8,
            _ => panic!("flock-pure-block covers 16-bit rows (96 units) and 8-bit rows (48 units)"),
        }
    }
    pub fn is_sha(self) -> bool { matches!(self, Layout::ShaBf16 | Layout::ShaFp8) }
    pub fn row_bytes(self) -> usize { match self { Layout::Bf16 | Layout::ShaBf16 => 3072, _ => 1536 } }
    /// log2 of a compression sub-block: BLAKE3 14, SHA-256 15.
    pub fn comp_log(self) -> usize { if self.is_sha() { 15 } else { 14 } }
    pub fn k_log(self) -> usize { match self { Layout::Bf16 => 20, Layout::Fp8 => 21, Layout::ShaBf16 => 22, Layout::ShaFp8 => 21 } }
    pub fn comp_slots(self) -> usize { match self { Layout::Bf16 => 32, Layout::Fp8 => 48, Layout::ShaBf16 => 98, Layout::ShaFp8 => 50 } }
    pub fn per_role(self) -> usize { self.comp_slots() / 2 }
    pub fn units(self) -> usize { match self { Layout::Bf16 => 32, Layout::Fp8 | Layout::ShaFp8 => 48, Layout::ShaBf16 => 96 } }
    /// The first unit's 2^13 position: right after the compressions.
    pub fn unit_pos0(self) -> usize { self.comp_slots() << (self.comp_log() - 13) }
    pub fn blocks_per_vu(self) -> usize { match self { Layout::Bf16 => 3, _ => 1 } }
    /// Compressions per chunk of a block, in order (SHA: one chain per role, its padding block included).
    pub fn chunks(self) -> &'static [usize] { match self { Layout::Bf16 => &[16], Layout::Fp8 => &[16, 8], Layout::ShaBf16 => &[49], Layout::ShaFp8 => &[25] } }
    /// Chaining values per block the prover publishes (x chunks, then W chunks); SHA publishes none (the digests are the
    /// verifier's).
    pub fn n_cv(self) -> usize { if self.is_sha() { 0 } else { 2 * self.chunks().len() } }
    /// (role, chunk in block, block in chunk) of compression slot s.
    pub fn slot(self, s: usize) -> (usize, usize, usize) {
        let (role, mut j) = (s / self.per_role(), s % self.per_role());
        for (c, &nb) in self.chunks().iter().enumerate() {
            if j < nb {
                return (role, c, j);
            }
            j -= nb;
        }
        unreachable!()
    }
    /// The chunk's index in its row (the BLAKE3 counter) for block b, chunk c of the block.
    pub fn chunk_index(self, b: usize, c: usize) -> usize { (b % self.blocks_per_vu()) * self.chunks().len() + c }
    /// The one pinned constant column: slot 0's constant.
    pub fn pin(self) -> usize { if self.is_sha() { sha2c::Z_CONST_POS } else { b3::Z_CONST_POS } }
    /// The compression circuit of the layout.
    pub fn comp(self) -> &'static BlockR1cs {
        static S: OnceLock<BlockR1cs> = OnceLock::new();
        if self.is_sha() { S.get_or_init(|| sha2c::build_block_r1cs(3)) } else { comp() }
    }
}

/// `sha256/row/v1`: the 64-byte prefix block (`verity.commitments.rowleaf.sha256_row_prefix`).
pub fn sha_prefix(role: usize, word_bits: usize, n_words: usize) -> [u8; 64] {
    let mut p = [0u8; 64];
    let tag = b"verity/sha256-row/v1\0";
    p[..tag.len()].copy_from_slice(tag);
    p[tag.len()] = role as u8 + 1;
    p[tag.len() + 1] = word_bits as u8;
    p[tag.len() + 2..tag.len() + 6].copy_from_slice(&(n_words as u32).to_be_bytes());
    p
}

pub fn be_words16(b: &[u8]) -> [u32; 16] {
    from_fn(|i| u32::from_be_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}

/// The SHA-256 state after the role's prefix block (the chain's constant input).
pub fn sha_midstate(role: usize, row_bytes: usize) -> [u32; 8] {
    let word_bits = if row_bytes == 3072 { 16 } else { 8 };
    sha2c::sha256_compress(&sha2c::SHA256_IV, &be_words16(&sha_prefix(role, word_bits, row_bytes * 8 / word_bits)))
}

/// The constant padding block of a `64 + row_bytes`-byte message.
pub fn sha_pad_block(row_bytes: usize) -> [u32; 16] {
    let mut b = [0u8; 64];
    b[0] = 0x80;
    b[56..].copy_from_slice(&(((64 + row_bytes) * 8) as u64).to_be_bytes());
    be_words16(&b)
}

pub fn chunk_flags(j: usize, nb: usize) -> u32 {
    crate::chunk::KEYED_HASH | if j == 0 { crate::chunk::CHUNK_START } else { 0 } | if j + 1 == nb { crate::chunk::CHUNK_END } else { 0 }
}

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
    /// fp8: the chaining value of compression slot k (a chunk's last block).
    CvSlot(usize),
    /// sha: the final state (the row digest, big-endian words) of role r's chain: the verifier's own digest.
    Digest(usize),
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

fn shape(region: Region, net: &UnitNet, lay: Layout) -> RegionShape {
    let bits = |lo: usize, n: usize| (lo..lo + n).collect::<Vec<_>>();
    // (free bits, fixed value) in the 20 in-block bits
    let (free, fixed): (Vec<usize>, u32) = match region {
        Region::CvSlot(k) => (bits(0, 8), (1 << 8) | ((k as u32) << b3::K_LOG)),
        Region::Digest(r) => (bits(0, 8), (1 << 8) | (((lay.per_role() * r + lay.per_role() - 1) as u32) << lay.comp_log())),
        Region::Key => ([bits(0, 8), vec![18]].concat(), 0),
        Region::Counter => ([bits(0, 6), bits(14, 5)].concat(), 1 << 10),
        Region::Cv => ([bits(0, 8), vec![18]].concat(), (1 << 8) | (0xF << 14)),
        Region::AccIn => (bits(0, 5), (1 << 9) | ((lay.unit_pos0() as u32) << UNIT_LOG)),
        Region::AccOut => (bits(0, 5), ((net.out_cols[0] as u32) << 7) | (((lay.unit_pos0() + lay.units() - 1) as u32) << UNIT_LOG)),
        Region::Y => (bits(0, 4), ((net.out_cols[1] as u32) << 7) | (((lay.unit_pos0() + lay.units() - 1) as u32) << UNIT_LOG)),
    };
    let mask = ((1u32 << lay.k_log()) - 1) & !free.iter().fold(0u32, |m, &b| m | (1 << b));
    assert_eq!(fixed & !mask, 0);
    RegionShape { region, free, mask, fixed }
}

/// The prover's per-block publics: each real block's x and W chunk chaining values, its unit 31 `y16` word and its unit 31
/// accumulator (`c_out`). 72 bytes per block: CV_x (8 u32 LE), CV_w (8 u32), y16 (u32), acc_out (u32).
#[derive(Clone)]
pub struct Publics {
    /// Per block: the chaining value of every chunk (x chunks, then W chunks).
    pub cv: Vec<Vec<[u32; 8]>>,
    pub y16: Vec<u32>,
    pub acc: Vec<u32>,
}

impl Publics {
    pub fn bytes(&self) -> Vec<u8> {
        let mut v = Vec::new();
        for ((c, y), a) in self.cv.iter().zip(&self.y16).zip(&self.acc) {
            for w in c.iter().flatten() {
                v.extend_from_slice(&w.to_le_bytes());
            }
            v.extend_from_slice(&y.to_le_bytes());
            v.extend_from_slice(&a.to_le_bytes());
        }
        v
    }

    pub fn parse(b: &[u8], blocks: usize, n_cv: usize) -> Result<Self, String> {
        let per = 8 * n_cv + 2;
        if b.len() != 4 * per * blocks {
            return Err(format!("C4: {} bytes of publics, want {}", b.len(), 4 * per * blocks));
        }
        let w = le_words(b);
        let cv = (0..blocks).map(|i| (0..n_cv).map(|r| from_fn(|k| w[per * i + 8 * r + k])).collect()).collect();
        let y16 = (0..blocks).map(|i| w[per * i + 8 * n_cv]).collect();
        let acc = (0..blocks).map(|i| w[per * i + 8 * n_cv + 1]).collect();
        Ok(Publics { cv, y16, acc })
    }
}

pub struct PureCircuit {
    pub lay: Layout,
    pub unit: SparseMatrixCircuit<'static>,
    pub da: Vec<(u32, u32)>,
    pub db: Vec<(u32, u32)>,
}

impl LincheckCircuit for PureCircuit {
    fn n_cols(&self) -> usize {
        1 << self.lay.k_log()
    }

    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        let (cl, us) = (self.lay.comp_log(), 1usize << UNIT_LOG);
        let cs = 1usize << cl;
        let per = 1usize << (cl - UNIT_LOG);
        let (kl, slots, units, p0) = (self.lay.k_log(), self.lay.comp_slots(), self.lay.units(), self.lay.unit_pos0());
        assert_eq!(eq.len(), 1 << kl);
        let c0 = eq[..us].iter().position(|e| *e != F128::ZERO).expect("eq tensor vanishes on the first unit slot");
        let inv = eq[c0].inv();
        let base_c = self.lay.comp().csc_lincheck_circuit().fold_alpha_batched(alpha, &eq[..cs]);
        let base_u = self.unit.fold_alpha_batched(alpha, &eq[..us]);
        let mut out = vec![F128::ZERO; 1 << kl];
        out.par_chunks_mut(us).enumerate().for_each(|(p, o)| {
            if p < per * slots {
                let s = eq[((p / per) << cl) | c0] * inv;
                let half = (p % per) * us;
                for (x, y) in o.iter_mut().zip(&base_c[half..half + us]) {
                    *x = s * *y;
                }
            } else if (p0..p0 + units).contains(&p) {
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
        Some(self.lay.pin())
    }
}

/// Δ of the SHA-256 layouts: chain copies, the midstate at each role's first block and the padding block's message as
/// constants, the unit operands as copies of the big-endian message bits, the accumulator chain (c_in(0) = +0), and every
/// sub-circuit's constant a copy of the pinned one.
fn delta_sha(net: &UnitNet, lay: Layout, row_bytes: usize) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let c = lay.comp();
    let (a, b) = (&c.a_0.rows, &c.b_0.rows);
    let (z, pin, cl) = (sha2c::Z_CONST_POS, lay.pin(), lay.comp_log());
    let input = |r: usize| assert!(a[r] == [r] && b[r] == [z], "SHA-256 row {r} is not an input row");
    let (mut da, mut db) = (Vec::new(), Vec::new());
    let both = |da: &mut Vec<(u32, u32)>, db: &mut Vec<(u32, u32)>, i: usize, old: usize, new: usize| {
        for d in [da, db] {
            d.push((i as u32, old as u32));
            d.push((i as u32, new as u32));
        }
    };
    let constant = |da: &mut Vec<(u32, u32)>, g: &dyn Fn(usize) -> usize, base: usize, val: u32| {
        for bit in 0..32 {
            let r = base + bit;
            input(r);
            da.push((g(r) as u32, g(r) as u32));
            if (val >> bit) & 1 == 1 {
                da.push((g(r) as u32, g(z) as u32));
            }
        }
    };
    let pad = sha_pad_block(row_bytes);
    for s in 0..lay.comp_slots() {
        let (role, j) = (s / lay.per_role(), s % lay.per_role());
        let g = |col: usize| (s << cl) + col;
        if s > 0 {
            both(&mut da, &mut db, g(z), g(z), pin);
        }
        if j > 0 {
            for r in 0..256 {
                input(sha2c::H_BASE + r);
                da.push((g(sha2c::H_BASE + r) as u32, g(sha2c::H_BASE + r) as u32));
                da.push((g(sha2c::H_BASE + r) as u32, (((s - 1) << cl) + sha2c::H_OUT_BASE + r) as u32));
            }
        } else {
            for (w, &hw) in sha_midstate(role, row_bytes).iter().enumerate() {
                constant(&mut da, &g, sha2c::H_BASE + 32 * w, hw);
            }
        }
        if j + 1 == lay.per_role() {
            for (w, &mw) in pad.iter().enumerate() {
                constant(&mut da, &g, sha2c::M_BASE + 32 * w, mw);
            }
        }
    }
    let p0 = lay.unit_pos0();
    for u in 0..lay.units() {
        let base = (p0 + u) << UNIT_LOG;
        both(&mut da, &mut db, base + net.const_pos, base + net.const_pos, pin);
        for role in 0..2 {
            let slot = role * lay.per_role() + u / 2;
            for t in 0..256 {
                let bk = 32 * (u % 2) + t / 8;
                let msg = (slot << cl) + sha2c::M_BASE + 32 * (bk / 4) + 8 * (3 - bk % 4) + t % 8;
                both(&mut da, &mut db, base + 256 * role + t, base + 256 * role + t, msg);
            }
        }
        if u > 0 {
            let prev = ((p0 + u - 1) << UNIT_LOG) + net.out_cols[0] * 128;
            for t in 0..32 {
                both(&mut da, &mut db, base + 512 + t, base + 512 + t, prev + t);
            }
        } else {
            for t in 0..32 {
                da.push(((base + 512 + t) as u32, (base + 512 + t) as u32));
                db.push(((base + 512 + t) as u32, (base + 512 + t) as u32));
            }
        }
    }
    (da, db)
}

fn delta(net: &UnitNet, lay: Layout, keys: &[[u32; 8]; 2]) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
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
    // a constant word: rows [base, base + 32) become A = [bit · Z_s]
    let constant = |da: &mut Vec<(u32, u32)>, g: &dyn Fn(usize) -> usize, base: usize, val: u32| {
        for bit in 0..32 {
            let r = base + bit;
            input(r);
            da.push((g(r) as u32, g(r) as u32));
            if (val >> bit) & 1 == 1 {
                da.push((g(r) as u32, g(z) as u32));
            }
        }
    };
    for s in 0..lay.comp_slots() {
        let (role, ch, j) = lay.slot(s);
        let nb = lay.chunks()[ch];
        let g = |col: usize| (s << b3::K_LOG) + col;
        if s > 0 {
            both(&mut da, &mut db, g(z), g(z), lay.pin());
        }
        if j > 0 {
            for r in 0..256 {
                input(b3::CV_BASE + r);
                da.push((g(b3::CV_BASE + r) as u32, g(b3::CV_BASE + r) as u32));
                da.push((g(b3::CV_BASE + r) as u32, (((s - 1) << b3::K_LOG) + b3::OUT_LO_BASE + r) as u32));
            }
        } else if lay == Layout::Fp8 {
            for (w, &kw) in keys[role].iter().enumerate() {
                constant(&mut da, &g, b3::CV_BASE + 32 * w, kw);
            }
        }
        constant(&mut da, &g, b3::BLEN_BASE, 64);
        constant(&mut da, &g, b3::FLAGS_BASE, chunk_flags(j, nb));
        if lay == Layout::Fp8 {
            constant(&mut da, &g, b3::T_LO_BASE, ch as u32);
            constant(&mut da, &g, b3::T_HI_BASE, 0);
        }
    }
    let p0 = lay.unit_pos0();
    for u in 0..lay.units() {
        let base = (p0 + u) << UNIT_LOG;
        both(&mut da, &mut db, base + net.const_pos, base + net.const_pos, lay.pin());
        for role in 0..2 {
            let slot = role * lay.per_role() + u / 2;
            for t in 0..256 {
                let msg = (slot << b3::K_LOG) + b3::M_BASE + (u % 2) * 256 + t;
                both(&mut da, &mut db, base + 256 * role + t, base + 256 * role + t, msg);
            }
        }
        if u > 0 {
            let prev = ((p0 + u - 1) << UNIT_LOG) + net.out_cols[0] * 128;
            for t in 0..32 {
                both(&mut da, &mut db, base + 512 + t, base + 512 + t, prev + t);
            }
        } else if lay == Layout::Fp8 {
            for t in 0..32 {
                da.push(((base + 512 + t) as u32, (base + 512 + t) as u32));
                db.push(((base + 512 + t) as u32, (base + 512 + t) as u32));
            }
        }
    }
    (da, db)
}

/// The compressions of a pure-block witness (BLAKE3 for the keyed-BLAKE3 layouts, SHA-256 for `sha256/row/v1`).
#[derive(Clone)]
pub enum Comps {
    B3(Vec<b3::Compression>),
    Sha(Vec<([u32; 8], [u32; 16])>),
}

pub struct PureStmt {
    pub lay: Layout,
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
    /// A dummy block's chunk chaining values (x chunks, then W chunks).
    pub dummy_cv: Vec<[u32; 8]>,
    pub dummy_unit: (u32, u32),
}

impl PureStmt {
    pub fn new(net: UnitNet, vus: usize, lay: Layout) -> Self {
        let blocks_real = lay.blocks_per_vu() * vus;
        let nbl = log2ceil(blocks_real).max(3);
        let (kl, slots, units, p0) = (lay.k_log(), lay.comp_slots(), lay.units(), lay.unit_pos0());
        let m = kl + nbl;
        let keys = [key_words(&row_key(b"x")), key_words(&row_key(b"w"))];
        let (da, db) = if lay.is_sha() { delta_sha(&net, lay, lay.row_bytes()) } else { delta(&net, lay, &keys) };
        let list: Vec<Region> = match lay {
            Layout::ShaBf16 | Layout::ShaFp8 => vec![Region::Digest(0), Region::Digest(1), if net.epilogue() { Region::Y } else { Region::AccOut }],
            Layout::Bf16 => {
                let mut v = vec![Region::Key, Region::Counter, Region::Cv, Region::AccIn, Region::AccOut];
                if net.epilogue() {
                    v.push(Region::Y);
                }
                v
            }
            Layout::Fp8 => {
                let mut v: Vec<Region> = (0..slots).filter(|&k| { let (_, c, j) = lay.slot(k); j + 1 == lay.chunks()[c] }).map(Region::CvSlot).collect();
                v.push(Region::AccOut);
                v
            }
        };
        let regions: Vec<RegionShape> = list.into_iter().map(|r| shape(r, &net, lay)).collect();
        // dummy blocks: zero messages; bf16 under a zero key and counter 0, fp8 under the per-slot constants
        let dummy_cv: Vec<[u32; 8]> = if lay.is_sha() {
            (0..2)
                .map(|role| {
                    let mut h = sha_midstate(role, lay.row_bytes());
                    for _ in 0..lay.per_role() - 1 {
                        h = sha2c::sha256_compress(&h, &[0u32; 16]);
                    }
                    sha2c::sha256_compress(&h, &sha_pad_block(lay.row_bytes()))
                })
                .collect()
        } else { (0..2)
            .flat_map(|role| {
                let keys = keys;
                lay.chunks().iter().enumerate().map(move |(c, &nb)| {
                    let (mut cv, ctr) = match lay { Layout::Bf16 => ([0u32; 8], 0u64), _ => (keys[role], c as u64) };
                    for j in 0..nb {
                        cv = blake3_compress(&cv, &[0u32; 16], ctr, 64, chunk_flags(j, nb))[..8].try_into().unwrap();
                    }
                    cv
                })
            })
            .collect() };
        let (z, _, _) = net.eval64(|_| 0);
        let word = |col: usize, n: usize| (0..n).fold(0u32, |w, t| w | (((z[col * 128 + t] & 1) as u32) << t));
        let dummy_unit = (word(net.out_cols[0], 32), if net.epilogue() { word(net.out_cols[1], 16) } else { 0 });
        let mut h = Sha256::new();
        h.update(TAG);
        h.update(comp().statement_digest());
        h.update(net.sha256);
        for x in [vus, blocks_real, nbl, m, kl, UNIT_LOG, slots, units, p0] {
            h.update((x as u64).to_le_bytes());
        }
        if lay == Layout::Fp8 {
            h.update(b"layout fp8: block = VU, chunks 16+8, per-slot key/counter constants, c_in(0) = 0");
        }
        if lay.is_sha() {
            h.update(b"layout sha256/row/v1: block = VU, one SHA-256 chain per role from the prefix midstate, constant padding block, c_in(0) = 0");
            h.update(lay.comp().statement_digest());
        }
        for d in [&da, &db] {
            h.update((d.len() as u64).to_le_bytes());
            for &(i, c) in d.iter() {
                h.update(i.to_le_bytes());
                h.update(c.to_le_bytes());
            }
        }
        let digest: [u8; 32] = h.finalize().into();
        let kk = 1usize << kl;
        let empty = || SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: vec![Vec::new(); kk].into() };
        let r1cs = BlockR1cs {
            m,
            k_log: kl,
            k_skip: b3::K_SKIP,
            useful_bits: kk,
            a_0: empty(),
            b_0: empty(),
            c_0: SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: (0..kk).map(|i| vec![i]).collect::<Vec<_>>().into() },
            layout: WitnessLayout::RowMajor,
            const_pin: Some(lay.pin()),
            digest_cache: OnceLock::from(digest),
            csc_cache: OnceLock::new(),
        };
        let (ua, ub) = net.matrices();
        let (ua, ub): (&'static SparseBinaryMatrix, &'static SparseBinaryMatrix) = (Box::leak(Box::new(ua)), Box::leak(Box::new(ub)));
        let unit = SparseMatrixCircuit::new(ua, ub);
        PureStmt { lay, net, vus, blocks_real, nbl, m, m_pts: PT_LOCAL + nbl, regions, circuit: PureCircuit { lay, unit, da, db }, r1cs, digest,
                   keys, dummy_cv, dummy_unit }
    }

    pub fn blocks(&self) -> usize {
        1 << self.nbl
    }

    /// VU and chunk-in-row of block b's chunk c (the row's bytes it hashes start at 1024 · chunk).
    fn vu_chunk(&self, b: usize, c: usize) -> (usize, usize) {
        (b / self.lay.blocks_per_vu(), self.lay.chunk_index(b, c))
    }

    /// Every compression, block-major (each block's `comp_slots`: x chunks then W chunks), and each real block's chunk
    /// chaining values (x chunks, then W chunks).
    pub fn compressions(&self, inst: &Instances) -> (Vec<b3::Compression>, Vec<Vec<[u32; 8]>>) {
        let lay = self.lay;
        let per: Vec<(Vec<b3::Compression>, Vec<[u32; 8]>)> = (0..self.blocks())
            .into_par_iter()
            .map(|b| {
                let real = b < self.blocks_real;
                let mut out = Vec::with_capacity(lay.comp_slots());
                let mut cvs = Vec::with_capacity(lay.n_cv());
                for role in 0..2 {
                    for (c, &nb) in lay.chunks().iter().enumerate() {
                        let (v, ci) = self.vu_chunk(b, c);
                        let (mut cv, ctr) = match (lay, real) {
                            (_, true) => (self.keys[role], ci as u64),
                            (Layout::Bf16, false) => ([0; 8], 0),
                            (_, false) => (self.keys[role], c as u64),
                        };
                        for j in 0..nb {
                            let msg: [u32; 16] = if real {
                                let at = ci * 1024 + 64 * j;
                                let row = inst.row(role, v);
                                from_fn(|i| u32::from_le_bytes(row[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
                            } else {
                                [0; 16]
                            };
                            let f = chunk_flags(j, nb);
                            out.push((cv, msg, ctr, 64, f));
                            cv = blake3_compress(&cv, &msg, ctr, 64, f)[..8].try_into().unwrap();
                        }
                        cvs.push(cv);
                    }
                }
                (out, cvs)
            })
            .collect();
        let cvs = per.iter().map(|p| p.1.clone()).take(self.blocks_real).collect();
        (per.into_iter().flat_map(|p| p.0).collect(), cvs)
    }

    /// The SHA-256 chain's final state for row (role, v): the `sha256/row/v1` digest as big-endian words.
    pub fn sha_state(&self, inst: &Instances, role: usize, v: usize) -> [u32; 8] {
        let row = inst.row(role, v);
        let mut h = sha_midstate(role, row.len());
        for blk in row.chunks(64) {
            h = sha2c::sha256_compress(&h, &be_words16(blk));
        }
        sha2c::sha256_compress(&h, &sha_pad_block(row.len()))
    }

    /// Every SHA-256 compression `(h_in, m)`, block-major (each VU: the x chain, then the W chain; dummy blocks hash zero
    /// rows).
    pub fn sha_compressions(&self, inst: &Instances) -> Vec<([u32; 8], [u32; 16])> {
        let lay = self.lay;
        (0..self.blocks())
            .into_par_iter()
            .flat_map_iter(|b| {
                let real = b < self.blocks_real;
                let mut out = Vec::with_capacity(lay.comp_slots());
                for role in 0..2 {
                    let mut h = sha_midstate(role, lay.row_bytes());
                    for j in 0..lay.per_role() - 1 {
                        let m = if real { be_words16(&inst.row(role, b)[64 * j..64 * j + 64]) } else { [0; 16] };
                        out.push((h, m));
                        h = sha2c::sha256_compress(&h, &m);
                    }
                    out.push((h, sha_pad_block(lay.row_bytes())));
                }
                out
            })
            .collect()
    }

    /// Unit (block b, u)'s 544 input bits straight from the rows (the SHA layouts; zero rows for dummy blocks).
    pub fn unit_inputs_rows(&self, inst: &Instances, b: usize, u: usize, c_in: u32) -> [F128; 5] {
        let words = |role: usize| -> [F128; 2] {
            if b >= self.blocks_real {
                return [F128::ZERO; 2];
            }
            let r = &inst.row(role, b)[32 * u..32 * u + 32];
            let w: Vec<u32> = r.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            [crate::chunk::pack4(&w[..4]), crate::chunk::pack4(&w[4..])]
        };
        let (x, w) = (words(0), words(1));
        [x[0], x[1], w[0], w[1], F128::new(c_in as u64, 0)]
    }

    /// Unit (block b, u)'s 544 input bits: x bytes, W bytes (as the compressions hash them), accumulator in.
    pub fn unit_inputs(&self, comps: &[b3::Compression], b: usize, u: usize, c_in: u32) -> [F128; 5] {
        let base = b * self.lay.comp_slots();
        let words = |slot: usize| -> [F128; 2] {
            let m = &comps[base + slot].1;
            let h = (u % 2) * 8;
            [crate::chunk::pack4(&m[h..h + 4]), crate::chunk::pack4(&m[h + 4..h + 8])]
        };
        let (x, w) = (words(u / 2), words(self.lay.per_role() + u / 2));
        [x[0], x[1], w[0], w[1], F128::new(c_in as u64, 0)]
    }

    /// The honest accumulator input of every unit (the recorded accumulators; zero for dummy blocks).
    pub fn c_in(&self, inst: &Instances, b: usize, u: usize) -> u32 {
        if b >= self.blocks_real {
            return 0;
        }
        let bpv = self.lay.blocks_per_vu();
        let (v, j) = (b / bpv, (b % bpv) * self.lay.units() + u);
        if j == 0 { 0 } else { inst.acc(v, j - 1) }
    }

    /// `(z, a, b, z_lincheck)` of the block witness: compressions, then units evaluated from `inputs(b, u)`.
    pub fn witness(&self, comps: &Comps, inputs: &(dyn Fn(usize, usize) -> [F128; 5] + Sync)) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let (kl, slots, units, p0) = (self.lay.k_log(), self.lay.comp_slots(), self.lay.units(), self.lay.unit_pos0());
        let n = 1usize << (self.m - 7);
        let (cz, ca, cb) = match comps {
            Comps::B3(c) => b3::generate_witness_with_ab_packed(c, log2ceil(c.len())),
            Comps::Sha(c) => {
                let (z, a, b, _) = sha2c::generate_witness_with_ab_packed_and_lincheck(c, log2ceil(c.len()).max(3));
                (z, a, b)
            }
        };
        let (mut z, mut a, mut b) = (vec![F128::ZERO; n], vec![F128::ZERO; n], vec![F128::ZERO; n]);
        let wpb = 1usize << (kl - 7);
        let cw = 1usize << (self.lay.comp_log() - 7);
        for (dst, src) in [(&mut z, &cz), (&mut a, &ca), (&mut b, &cb)] {
            dst.par_chunks_mut(wpb).enumerate().for_each(|(blk, d)| {
                for s in 0..slots {
                    let ci = blk * slots + s;
                    d[s * cw..(s + 1) * cw].copy_from_slice(&src[ci * cw..(ci + 1) * cw]);
                }
            });
        }
        drop((cz, ca, cb));
        let uw = 1usize << (UNIT_LOG - 7);
        let total = self.blocks() * units;
        let groups: Vec<Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)>> = (0..total.div_ceil(64))
            .into_par_iter()
            .map(|g| {
                let ids: Vec<usize> = (64 * g..(64 * g + 64).min(total)).collect();
                let ins: Vec<[F128; 5]> = ids.iter().map(|&i| inputs(i / units, i % units)).collect();
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
            let (blk, u) = (i / units, i % units);
            let at = blk * wpb + (p0 + u) * uw;
            z[at..at + uw].copy_from_slice(&uz);
            a[at..at + uw].copy_from_slice(&ua);
            b[at..at + uw].copy_from_slice(&ub);
        }
        let zl = pack_z_lincheck_from_packed(&z, self.m, kl);
        (z, a, b, zl)
    }

    /// The value of every public region at every block: bit `f` of the region's free assignment.
    fn region_bits(&self, reg: &RegionShape, inst: &Instances, p: &Publics, b: usize) -> Vec<u8> {
        let real = b < self.blocks_real;
        let (v, c) = (b / self.lay.blocks_per_vu(), b % self.lay.blocks_per_vu());
        let le = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        match reg.region {
            Region::Digest(r) => le(&if real { self.sha_state(inst, r, v) } else { self.dummy_cv[r] }),
            Region::CvSlot(k) => {
                let (role, ch, _) = self.lay.slot(k);
                let i = role * self.lay.chunks().len() + ch;
                le(&if real { p.cv[b][i] } else { self.dummy_cv[i] })
            }
            Region::Key => {
                let k = |r: usize| if real { self.keys[r] } else { [0; 8] };
                [le(&k(0)), le(&k(1))].concat()
            }
            Region::Counter => (if real { c as u64 } else { 0 }).to_le_bytes().repeat(COMP_SLOTS),
            Region::Cv => {
                let cv = |r: usize| if real { p.cv[b][r] } else { self.dummy_cv[r] };
                [le(&cv(0)), le(&cv(1))].concat()
            }
            Region::AccIn => (if real && c > 0 { p.acc[b - 1] } else { 0 }).to_le_bytes().to_vec(),
            Region::AccOut => (match (self.lay, real) {
                (_, false) => self.dummy_unit.0,
                (Layout::Bf16, true) => p.acc[b],
                (_, true) => inst.out[v],
            })
            .to_le_bytes()
            .to_vec(),
            Region::Y => {
                let y = if !real { self.dummy_unit.1 } else if self.lay.blocks_per_vu() == 1 || c == 2 { inst.out[v] } else { p.y16[b] };
                (y as u16).to_le_bytes().to_vec()
            }
        }
    }

    /// The region's claim point from verifier point `r` (m_pts coordinates).
    pub fn region_point(&self, reg: &RegionShape, r: &[F128]) -> Vec<F128> {
        let kl = self.lay.k_log();
        let mut p = vec![F128::ZERO; self.m];
        for j in 0..kl {
            if (reg.mask >> j) & 1 == 1 && (reg.fixed >> j) & 1 == 1 {
                p[j] = F128::ONE;
            }
        }
        for (t, &j) in reg.free.iter().enumerate() {
            p[j] = r[t];
        }
        p[kl..].copy_from_slice(&r[PT_LOCAL..PT_LOCAL + self.nbl]);
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
