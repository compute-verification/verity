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
//! Public regions, each opened at two verifier points sampled after the commitment (both reps):
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
/// The largest unit slot: a netlist of more than 2^13 rows (a total unit) takes 2^14-row slots (`UnitNet::unit_log`).
pub const UNIT_LOG_MAX: usize = 14;
pub const COMP_SLOTS: usize = 32;
pub const UNITS: usize = 32;
pub const UNIT_POS0: usize = 64;
pub const PIN: usize = b3::Z_CONST_POS;
pub const TAG: &[u8] = b"verity/flock-pure-block/v2";
/// Coordinates per verifier point: the regions' free in-block bits come from `r[..16]`, block bits from `r[16..]`.
pub const PT_LOCAL: usize = 16;

/// The `flock-unit-io/v1` lowered unit (flock-backend's `verity_flock.lowering`), read-only.
/// The block shape of one relation family.
/// - `Chunk(n)` (rows of n whole BLAKE3 chunks, 32 units per chunk: 16-bit K = 1536 is `Chunk(3)`, K = 2048 `Chunk(4)`,
///   K = 8192 `Chunk(16)`; 8-bit K = 2048 `Chunk(2)`, K = 8192 `Chunk(8)`): block (v, c) is chunk c of VU v; `k_log` 20;
///   16 + 16 compressions and 32 units; the cross-chunk accumulators are prover-committed public words.
/// - `Fp8` (8-bit rows, 1.5 chunks, 48 units): block v is the whole VU; `k_log` 21; x chunk 0 (16) and chunk 1 (8), then W's,
///   48 compressions and the 48 units at 2^13 positions 96..144. Key, counter, flags and block_len are all per-slot
///   constants, the accumulator chain is inside the block (c_in of unit 0 forced to +0), and only the four chunk chaining
///   values and the output word are public.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Layout {
    Chunk(u8),
    /// `ChunkTail(n)`: rows of n whole chunks and a 512-byte partial chunk (16-bit K = 1024 n + 256: K = 2304 is
    /// `ChunkTail(4)`, K = 8960 `ChunkTail(17)`). Blocks are `Chunk`'s; block (v, n) holds the partial chunk: its 8
    /// BLAKE3 blocks at slots 0..8 (CHUNK_START on 0, CHUNK_END on 7, counter n; not the ROOT, the parent tree is) and its
    /// 16 units at 0..16. Its slots 8..16 and units 16..32 are an unread continuation of the chains. Δ is block-uniform,
    /// so what differs in that block is public: every slot's flags (a Flags region, the verifier's per block), slot 7 /
    /// 23's out_lo (CvMid: the partial chunk's chaining value in the verifier's tree) and unit 15's output (AccMid, or
    /// YMid with the epilogue: the verifier's output word at block n).
    ChunkTail(u8),
    Fp8,
    /// `sha256/row/v1` leaves: block v is the whole VU; each role is one SHA-256 chain (the verifier's midstate after the
    /// 64-byte prefix block, the row's 64-byte blocks, one constant padding block) of 2^15-bit compressions, then the
    /// units at 2^13 positions after them. 16-bit rows: 49 + 49 compressions, 96 units, k_log 22.
    ShaBf16,
    /// 8-bit rows: 25 + 25 compressions, 48 units, k_log 21.
    ShaFp8,
    /// fp4-nvf4 (`blake3-keyed/row-nvfp4/v1`, 864-byte rows: 768 code bytes, then 96 scale bytes): block v is the whole
    /// VU; each role is one BLAKE3 chunk of 13 full blocks and a 32-byte final block (CHUNK_END | ROOT, message words
    /// 8..16 zero), all per-slot constants; the 24 units at 2^13 positions 56..80 read 32 code bytes and one scale word
    /// each; `k_log` 20. The two row digests are the root outputs: verifier-known regions like the SHA-256 digests.
    Fp4,
    /// fp4-nvf4 (`sha256/row-nvfp4/v1`): each role one SHA-256 chain from the nvfp4 prefix midstate, 13 row blocks and a
    /// final block of 8 row words and 8 constant padding words; 24 units at 2^13 positions 112..136; `k_log` 21.
    ShaFp4,
}

impl Layout {
    pub fn of(row_bytes: usize, units: usize, sha: bool) -> Self {
        match (row_bytes, units, sha) {
            (1536, 48, false) => Layout::Fp8,
            (rb, u, false) if rb % 1024 == 0 && u == 32 * (rb / 1024) && rb / 1024 <= 64 => Layout::Chunk((rb / 1024) as u8),
            (rb, u, false) if rb % 1024 == 512 && u == 32 * (rb / 1024) + 16 && rb / 1024 < 64 => Layout::ChunkTail((rb / 1024) as u8),
            (3072, 96, true) => Layout::ShaBf16,
            (1536, 48, true) => Layout::ShaFp8,
            (864, 24, false) => Layout::Fp4,
            (864, 24, true) => Layout::ShaFp4,
            _ => panic!("flock-pure-block covers rows of whole BLAKE3 chunks (32 units each) and the K = 1536 8-bit and SHA-256 rows"),
        }
    }
    pub fn is_sha(self) -> bool { matches!(self, Layout::ShaBf16 | Layout::ShaFp8 | Layout::ShaFp4) }
    pub fn is_fp4(self) -> bool { matches!(self, Layout::Fp4 | Layout::ShaFp4) }
    /// The row-leaf schema of the a / b trees (a file naming another is refused: NV3).
    pub fn row_schema(self) -> &'static str {
        match self {
            Layout::Fp4 => "blake3-keyed/row-nvfp4/v1",
            Layout::ShaFp4 => "sha256/row-nvfp4/v1",
            Layout::ShaBf16 | Layout::ShaFp8 => "sha256/row/v1",
            _ => "blake3-keyed/row/v2",
        }
    }
    /// Per-slot key / counter constants (the whole VU in one block).
    pub fn per_slot_keys(self) -> bool { matches!(self, Layout::Fp8 | Layout::Fp4) }
    /// One block per (VU, chunk): `Chunk(n)` and `ChunkTail(n)`.
    pub fn is_chunked(self) -> bool { matches!(self, Layout::Chunk(_) | Layout::ChunkTail(_)) }
    /// `ChunkTail(n)`'s n: the partial chunk's block within a VU.
    pub fn tail(self) -> Option<usize> { if let Layout::ChunkTail(n) = self { Some(n as usize) } else { None } }
    /// BLAKE3 flags of slot j (within a role) of block c of a VU (the Flags region of `ChunkTail`; dummy blocks hash
    /// whole zero chunks): the partial chunk's 8 blocks, then KEYED_HASH alone on the unread continuation.
    pub fn slot_flags(self, c: usize, real: bool, j: usize) -> u32 {
        match self.tail() {
            Some(n) if real && c == n => if j < 8 { chunk_flags(j, 8) } else { 16 },
            _ => chunk_flags(j, 16),
        }
    }
    pub fn row_bytes(self) -> usize { match self { Layout::Chunk(n) => 1024 * n as usize, Layout::ChunkTail(n) => 1024 * n as usize + 512, Layout::ShaBf16 => 3072, Layout::Fp4 | Layout::ShaFp4 => 864, _ => 1536 } }
    /// log2 of a compression sub-block: BLAKE3 14, SHA-256 15.
    pub fn comp_log(self) -> usize { if self.is_sha() { 15 } else { 14 } }
    pub fn k_log(self) -> usize { match self { Layout::Chunk(_) | Layout::ChunkTail(_) => 20, Layout::Fp8 => 21, Layout::ShaBf16 => 22, Layout::ShaFp8 => 21, Layout::Fp4 => 20, Layout::ShaFp4 => 21 } }
    pub fn comp_slots(self) -> usize { match self { Layout::Chunk(_) | Layout::ChunkTail(_) => 32, Layout::Fp8 => 48, Layout::ShaBf16 => 98, Layout::ShaFp8 => 50, Layout::Fp4 | Layout::ShaFp4 => 28 } }
    pub fn per_role(self) -> usize { self.comp_slots() / 2 }
    pub fn units(self) -> usize { match self { Layout::Chunk(_) | Layout::ChunkTail(_) => 32, Layout::Fp8 | Layout::ShaFp8 => 48, Layout::ShaBf16 => 96, Layout::Fp4 | Layout::ShaFp4 => 24 } }
    /// The first unit's 2^13 position: right after the compressions.
    pub fn unit_pos0(self, unit_log: usize) -> usize { self.comp_slots() << (self.comp_log() - unit_log) }
    /// Whether `units` slots of 2^unit_log bits fit after the compressions in the 2^k_log block.
    pub fn fits(self, unit_log: usize) -> bool { self.unit_pos0(unit_log) + self.units() <= 1 << (self.k_log() - unit_log) }
    pub fn blocks_per_vu(self) -> usize { match self { Layout::Chunk(n) => n as usize, Layout::ChunkTail(n) => n as usize + 1, _ => 1 } }
    /// Compressions per chunk of a block, in order (SHA: one chain per role, its padding block included).
    pub fn chunks(self) -> &'static [usize] { match self { Layout::Chunk(_) | Layout::ChunkTail(_) => &[16], Layout::Fp8 => &[16, 8], Layout::ShaBf16 => &[49], Layout::ShaFp8 => &[25], Layout::Fp4 | Layout::ShaFp4 => &[14] } }
    /// Chaining values per block the prover publishes (x chunks, then W chunks); SHA publishes none (the digests are the
    /// verifier's).
    pub fn n_cv(self) -> usize {
        if self.is_sha() || self.is_fp4() { 0 } else if self.tail().is_some() { 4 } else { 2 * self.chunks().len() }
    }
    /// BLAKE3 block_len of block j of a chunk of nb blocks (the fp4 row's last block holds 32 bytes).
    pub fn blen(self, j: usize, nb: usize) -> u32 { if self.is_fp4() && j + 1 == nb { 32 } else { 64 } }
    /// BLAKE3 flags of block j of a chunk of nb blocks (the fp4 row is a single chunk: its last block is the ROOT).
    pub fn flags(self, j: usize, nb: usize) -> u32 { chunk_flags(j, nb) | if self.is_fp4() && j + 1 == nb { 8 } else { 0 } }
    /// Message words of a role's last compression that are row data (the rest are constants).
    pub fn last_data_words(self) -> usize { if self.is_fp4() { 8 } else if self.is_sha() { 0 } else { 16 } }
    /// The first compression slot of a role holding the fp4 scale bytes (row byte 768 = block 12).
    pub const FP4_SCALE_BLOCK: usize = 12;
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

/// `sha256/row/v1`: the 64-byte prefix block (`verity.ml.commitments.rowleaf.sha256_row_prefix`).
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

/// `sha256/row-nvfp4/v1`: the 64-byte prefix block (`verity.ml.commitments.rowleaf.sha256_row_nvfp4_prefix`).
pub fn sha_prefix_nvfp4(role: usize, k: usize) -> [u8; 64] {
    let mut p = [0u8; 64];
    let tag = b"verity/sha256-row-nvfp4/v1\0";
    p[..tag.len()].copy_from_slice(tag);
    p[tag.len()] = role as u8 + 1;
    p[tag.len() + 1] = 4;
    p[tag.len() + 2..tag.len() + 6].copy_from_slice(&(k as u32).to_be_bytes());
    p[tag.len() + 6..tag.len() + 10].copy_from_slice(&((k / 16) as u32).to_be_bytes());
    p
}

/// The prefix block of a row of `row_bytes` (864: the k = 1536 NVFP4 row; else 8- or 16-bit words, k = 1536).
pub fn sha_row_prefix(role: usize, row_bytes: usize) -> [u8; 64] {
    match row_bytes {
        864 => sha_prefix_nvfp4(role, 1536),
        3072 => sha_prefix(role, 16, 1536),
        _ => sha_prefix(role, 8, row_bytes),
    }
}

/// The SHA-256 state after the role's prefix block (the chain's constant input).
pub fn sha_midstate(role: usize, row_bytes: usize) -> [u32; 8] {
    sha2c::sha256_compress(&sha2c::SHA256_IV, &be_words16(&sha_row_prefix(role, row_bytes)))
}

/// The final block of a `64 + row_bytes`-byte message with its row bytes zero: the `row_bytes % 64` data bytes, then
/// 0x80 and the bit length (the whole block is constant when the row is whole blocks).
pub fn sha_pad_block(row_bytes: usize) -> [u32; 16] {
    let mut b = [0u8; 64];
    b[row_bytes % 64] = 0x80;
    b[56..].copy_from_slice(&(((64 + row_bytes) * 8) as u64).to_be_bytes());
    be_words16(&b)
}

pub fn chunk_flags(j: usize, nb: usize) -> u32 {
    crate::chunk::KEYED_HASH | if j == 0 { crate::chunk::CHUNK_START } else { 0 } | if j + 1 == nb { crate::chunk::CHUNK_END } else { 0 }
}

#[derive(Clone)]
pub struct UnitNet {
    pub relation: String,
    pub useful: usize,
    pub const_pos: usize,
    /// Input bits: 544 (x, W, c_in), 608 with the fp4-nvf4 scale bytes.
    pub n_in: usize,
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
        assert!(useful <= 1 << UNIT_LOG_MAX && const_pos == useful - 1 && (n_in == 544 || n_in == 608));
        let (mut a, mut b) = (Vec::with_capacity(useful), Vec::with_capacity(useful));
        for _ in 0..useful {
            let v: Vec<usize> = lines.next().expect("row").split_whitespace().map(|t| t.parse().unwrap()).collect();
            let na = v[0];
            let nb = v[1 + na];
            a.push(v[1..1 + na].to_vec());
            b.push(v[2 + na..2 + na + nb].to_vec());
        }
        for i in 0..n_in {
            assert!(a[i] == [i] && b[i] == [i], "unit row {i} is not an input row");
        }
        assert!(a[const_pos] == [const_pos] && b[const_pos] == [const_pos]);
        UnitNet { relation: h[1].to_string(), useful, const_pos, n_in, out_cols, a, b, sha256 }
    }

    pub fn load(path: &str) -> Self {
        Self::parse(&std::fs::read_to_string(path).expect("netlist"))
    }

    /// log2 of the unit's slot: 13, or 14 for a netlist past 2^13 rows. Statements of 2^13-row units are unchanged.
    pub fn unit_log(&self) -> usize {
        if self.useful <= 1 << UNIT_LOG { UNIT_LOG } else { UNIT_LOG_MAX }
    }

    pub fn epilogue(&self) -> bool {
        self.out_cols.len() > 1
    }

    fn matrices(&self) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let k = 1usize << self.unit_log();
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
        for (i, zi) in z.iter_mut().enumerate().take(self.n_in) {
            *zi = input(i);
        }
        z[self.const_pos] = !0;
        for i in self.n_in..u {
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

    /// The output word VU v's committed y word opens (the word the output regions check), `!0` when y opens none.
    pub fn out_word(&self, v: usize) -> u32 {
        out_of_y(&self.relation, self.y[v]).unwrap_or(!0)
    }
}

/// The circuit output word a committed y word stands for: the fp8 relations commit the FP32 accumulator >> 10 (an
/// E4M3 accumulator's low 10 bits are zero: B-Ligero's `fp8.relation.pack_public`), every other relation the word itself.
pub fn out_of_y(relation: &str, y: u32) -> Option<u32> {
    if relation.starts_with("fp8-") {
        (y < 1 << 22).then_some(y << 10)
    } else {
        Some(y)
    }
}

/// The y leaf a relation's statement pins: the BF16 word (`u16`, 2 bytes) for the epilogue relations, the 32-bit word
/// (`u32`, 4 bytes: fp8's packed accumulator, fp4's FP32 word) otherwise.
pub fn y_leaf(net: &UnitNet) -> (&'static str, usize) {
    if net.epilogue() { ("u16", 2) } else { ("u32", 4) }
}

/// The most region claims the CUDA prover takes (`max_extra` in `cuda/prove_chunk.cuh`): two per public region, up to the
/// verifier's proved range (`circuit::IN_RANGE_REGIONS`).
pub const MAX_EXTRA_CLAIMS: usize = 2 * crate::circuit::IN_RANGE_REGIONS;

/// The largest proof: m ≤ 35 (the Fast100 schedule's reach; Chunk(n): n × VUs ≤ 32,768 blocks).
pub const MAX_M: usize = 35;

/// What both parties check of an (instances, netlist, layout) triple before any coin:
/// - CN3: no `Chunk(1)` (a one-chunk row's digest is its ROOT output, which Chunk(n) does not compute);
/// - CN2: the proof fits m ≤ MAX_M;
/// - NV2: a netlist with the fp4-nvf4 scale inputs (608 input bits) pairs only with the fp4 layouts, and the reverse;
/// - NV3: the a / b trees' row schema is the layout's pinned one;
/// - NV5: the y leaf's schema and width are the relation's (`y_leaf`), and every y word fits it;
/// - NV1: every VU's circuit output word is the one its committed y word opens (the y root is built from y).
pub fn admit(lay: Layout, net: &UnitNet, inst: &Instances) -> Result<(), String> {
    if lay == Layout::Chunk(1) || lay == Layout::ChunkTail(0) {
        return Err(format!("CN3: a {}-byte row is one BLAKE3 chunk, whose last block is the ROOT, and {lay:?} never sets ROOT", lay.row_bytes()));
    }
    if !lay.fits(net.unit_log()) {
        return Err(format!("UL1: {} units of 2^{} bits (a {}-row netlist) do not fit layout {lay:?}", lay.units(), net.unit_log(), net.useful));
    }
    let blocks = lay.blocks_per_vu() * inst.vus;
    if lay.k_log() + log2ceil(blocks).max(3) > MAX_M {
        return Err(format!("CN2: {} VUs × {} blocks per VU need m = {} > {MAX_M} (Chunk(n): n × VUs ≤ 32,768)", inst.vus,
                           lay.blocks_per_vu(), lay.k_log() + log2ceil(blocks).max(3)));
    }
    if (net.n_in == 608) != lay.is_fp4() {
        return Err(format!("NV2: a netlist with {}-bit unit inputs does not pair with layout {lay:?}", net.n_in));
    }
    for t in ["a", "b"] {
        let s = inst.header["schemas"][t].as_str().unwrap_or("");
        if s != lay.row_schema() {
            return Err(format!("NV3: tree {t} names row schema {s:?}; layout {lay:?} pins {:?}", lay.row_schema()));
        }
    }
    let (ys, yb) = y_leaf(net);
    let (hs, hb) = (inst.header["schemas"]["y"].as_str().unwrap_or(""), inst.header["y_bytes"].as_u64().unwrap_or(0));
    if hs != ys || hb != yb as u64 {
        return Err(format!("NV5: the y leaf is {hs:?} over {hb} bytes; relation {} pins {ys:?} over {yb}", inst.relation));
    }
    if let Some(v) = (0..inst.vus).find(|&v| yb < 4 && inst.y[v] >= 1 << (8 * yb)) {
        return Err(format!("NV5: VU {v}'s y word {:#x} does not fit the {yb}-byte y leaf", inst.y[v]));
    }
    if let Some(v) = (0..inst.vus).find(|&v| out_of_y(&inst.relation, inst.y[v]) != Some(inst.out[v])) {
        return Err(format!("NV1: VU {v}'s output word {:#010x} is not the one its committed y word {:#x} opens", inst.out[v], inst.y[v]));
    }
    Ok(())
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
    /// `ChunkTail`: every slot's flags word (the verifier's per block).
    Flags,
    /// `ChunkTail`: slot 7 / 23's out_lo (the partial chunk's chaining value at block n; prover-committed).
    CvMid,
    /// `ChunkTail`: unit 15's c_out (the verifier's output word at block n; prover-committed elsewhere).
    AccMid,
    /// `ChunkTail`, epilogue: unit 15's y16 (likewise).
    YMid,
}

/// One public region: the in-block bits it fixes (`fixed` over `mask`) and its free in-block bits, LSB first.
pub struct RegionShape {
    pub region: Region,
    pub free: Vec<usize>,
    pub mask: u32,
    pub fixed: u32,
}

fn shape(region: Region, net: &UnitNet, lay: Layout) -> RegionShape {
    let ul = net.unit_log();
    let bits = |lo: usize, n: usize| (lo..lo + n).collect::<Vec<_>>();
    // (free bits, fixed value) in the 20 in-block bits
    let (free, fixed): (Vec<usize>, u32) = match region {
        Region::CvSlot(k) => (bits(0, 8), (1 << 8) | ((k as u32) << b3::K_LOG)),
        Region::Digest(r) => (bits(0, 8), (1 << 8) | (((lay.per_role() * r + lay.per_role() - 1) as u32) << lay.comp_log())),
        Region::Key => ([bits(0, 8), vec![18]].concat(), 0),
        Region::Counter => ([bits(0, 6), bits(14, 5)].concat(), 1 << 10),
        Region::Cv => ([bits(0, 8), vec![18]].concat(), (1 << 8) | (0xF << 14)),
        Region::AccIn => (bits(0, 5), (1 << 9) | ((lay.unit_pos0(ul) as u32) << ul)),
        Region::AccOut => (bits(0, 5), ((net.out_cols[0] as u32) << 7) | (((lay.unit_pos0(ul) + lay.units() - 1) as u32) << ul)),
        Region::Y => (bits(0, 4), ((net.out_cols[1] as u32) << 7) | (((lay.unit_pos0(ul) + lay.units() - 1) as u32) << ul)),
        Region::Flags => ([bits(0, 5), bits(14, 5)].concat(), b3::FLAGS_BASE as u32),
        Region::CvMid => ([bits(0, 8), vec![18]].concat(), (1 << 8) | (0x7 << 14)),
        Region::AccMid => (bits(0, 5), ((net.out_cols[0] as u32) << 7) | (((lay.unit_pos0(ul) + 15) as u32) << ul)),
        Region::YMid => (bits(0, 4), ((net.out_cols[1] as u32) << 7) | (((lay.unit_pos0(ul) + 15) as u32) << ul)),
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
    /// `ChunkTail` only (else empty): unit 15's (y16, c_out) per block, 8 more bytes per block.
    pub mid: Vec<(u32, u32)>,
}

impl Publics {
    pub fn bytes(&self) -> Vec<u8> {
        let mut v = Vec::new();
        for (i, ((c, y), a)) in self.cv.iter().zip(&self.y16).zip(&self.acc).enumerate() {
            for w in c.iter().flatten() {
                v.extend_from_slice(&w.to_le_bytes());
            }
            v.extend_from_slice(&y.to_le_bytes());
            v.extend_from_slice(&a.to_le_bytes());
            if let Some(&(my, ma)) = self.mid.get(i) {
                v.extend_from_slice(&my.to_le_bytes());
                v.extend_from_slice(&ma.to_le_bytes());
            }
        }
        v
    }

    pub fn parse(b: &[u8], blocks: usize, n_cv: usize, mid: bool) -> Result<Self, String> {
        let per = 8 * n_cv + 2 + 2 * mid as usize;
        if b.len() != 4 * per * blocks {
            return Err(format!("C4: {} bytes of publics, want {}", b.len(), 4 * per * blocks));
        }
        let w = le_words(b);
        let cv = (0..blocks).map(|i| (0..n_cv).map(|r| from_fn(|k| w[per * i + 8 * r + k])).collect()).collect();
        let y16 = (0..blocks).map(|i| w[per * i + 8 * n_cv]).collect();
        let acc = (0..blocks).map(|i| w[per * i + 8 * n_cv + 1]).collect();
        let mid = if mid { (0..blocks).map(|i| (w[per * i + 8 * n_cv + 2], w[per * i + 8 * n_cv + 3])).collect() } else { Vec::new() };
        Ok(Publics { cv, y16, acc, mid })
    }
}

pub struct PureCircuit {
    pub lay: Layout,
    /// log2 of the unit slot (`UnitNet::unit_log`).
    pub ul: usize,
    pub unit: SparseMatrixCircuit<'static>,
    pub da: Vec<(u32, u32)>,
    pub db: Vec<(u32, u32)>,
}

impl LincheckCircuit for PureCircuit {
    fn n_cols(&self) -> usize {
        1 << self.lay.k_log()
    }

    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        let (cl, ul) = (self.lay.comp_log(), self.ul);
        let us = 1usize << ul;
        let cs = 1usize << cl;
        let per = 1usize << (cl - ul);
        let (kl, slots, units, p0) = (self.lay.k_log(), self.lay.comp_slots(), self.lay.units(), self.lay.unit_pos0(self.ul));
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
                let s = eq[(p << ul) | c0] * inv;
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
    let ul = net.unit_log();
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
            for (w, &mw) in pad.iter().enumerate().skip(lay.last_data_words()) {
                constant(&mut da, &g, sha2c::M_BASE + 32 * w, mw);
            }
        }
    }
    let p0 = lay.unit_pos0(ul);
    for u in 0..lay.units() {
        let base = (p0 + u) << ul;
        both(&mut da, &mut db, base + net.const_pos, base + net.const_pos, pin);
        for role in 0..2 {
            let slot = role * lay.per_role() + u / 2;
            for t in 0..256 {
                let bk = 32 * (u % 2) + t / 8;
                let msg = (slot << cl) + sha2c::M_BASE + 32 * (bk / 4) + 8 * (3 - bk % 4) + t % 8;
                both(&mut da, &mut db, base + 256 * role + t, base + 256 * role + t, msg);
            }
            if lay.is_fp4() {
                let slot = role * lay.per_role() + Layout::FP4_SCALE_BLOCK + u / 16;
                for t in 0..32 {
                    let bk = 4 * (u % 16) + t / 8;
                    let msg = (slot << cl) + sha2c::M_BASE + 32 * (bk / 4) + 8 * (3 - bk % 4) + t % 8;
                    both(&mut da, &mut db, base + 544 + 32 * role + t, base + 544 + 32 * role + t, msg);
                }
            }
        }
        if u > 0 {
            let prev = ((p0 + u - 1) << ul) + net.out_cols[0] * 128;
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
    let ul = net.unit_log();
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
        } else if lay.per_slot_keys() {
            for (w, &kw) in keys[role].iter().enumerate() {
                constant(&mut da, &g, b3::CV_BASE + 32 * w, kw);
            }
        }
        constant(&mut da, &g, b3::BLEN_BASE, lay.blen(j, nb));
        if lay.tail().is_none() {
            constant(&mut da, &g, b3::FLAGS_BASE, lay.flags(j, nb));
        }
        if j + 1 == nb {
            for w in lay.last_data_words()..16 {
                constant(&mut da, &g, b3::M_BASE + 32 * w, 0);
            }
        }
        if lay.per_slot_keys() {
            constant(&mut da, &g, b3::T_LO_BASE, ch as u32);
            constant(&mut da, &g, b3::T_HI_BASE, 0);
        }
    }
    let p0 = lay.unit_pos0(ul);
    for u in 0..lay.units() {
        let base = (p0 + u) << ul;
        both(&mut da, &mut db, base + net.const_pos, base + net.const_pos, lay.pin());
        for role in 0..2 {
            let slot = role * lay.per_role() + u / 2;
            for t in 0..256 {
                let msg = (slot << b3::K_LOG) + b3::M_BASE + (u % 2) * 256 + t;
                both(&mut da, &mut db, base + 256 * role + t, base + 256 * role + t, msg);
            }
            if lay.is_fp4() {
                let slot = role * lay.per_role() + Layout::FP4_SCALE_BLOCK + u / 16;
                for t in 0..32 {
                    let msg = (slot << b3::K_LOG) + b3::M_BASE + 32 * (u % 16) + t;
                    both(&mut da, &mut db, base + 544 + 32 * role + t, base + 544 + 32 * role + t, msg);
                }
            }
        }
        if u > 0 {
            let prev = ((p0 + u - 1) << ul) + net.out_cols[0] * 128;
            for t in 0..32 {
                both(&mut da, &mut db, base + 512 + t, base + 512 + t, prev + t);
            }
        } else if lay.per_slot_keys() {
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
    /// log2 of the unit slot (`UnitNet::unit_log`).
    pub ul: usize,
    /// `ChunkTail`: per real VU, the accumulator into units 16..32 of its partial-chunk block (the continuation's units
    /// read zero operands; each one's input is the previous one's evaluated output).
    tail_cin: OnceLock<Vec<[u32; 16]>>,
}

impl PureStmt {
    pub fn new(net: UnitNet, vus: usize, lay: Layout) -> Self {
        assert_eq!(net.n_in == 608, lay.is_fp4(), "NV2: {}-bit unit inputs under layout {lay:?}", net.n_in);
        let blocks_real = lay.blocks_per_vu() * vus;
        let nbl = log2ceil(blocks_real).max(3);
        let ul = net.unit_log();
        assert!(lay.fits(ul), "UL1: {} units of 2^{ul} bits do not fit layout {lay:?}", lay.units());
        let (kl, slots, units, p0) = (lay.k_log(), lay.comp_slots(), lay.units(), lay.unit_pos0(ul));
        let m = kl + nbl;
        let keys = [key_words(&row_key(b"x")), key_words(&row_key(b"w"))];
        let (da, db) = if lay.is_sha() { delta_sha(&net, lay, lay.row_bytes()) } else { delta(&net, lay, &keys) };
        let list: Vec<Region> = match lay {
            Layout::ShaBf16 | Layout::ShaFp8 | Layout::ShaFp4 | Layout::Fp4 => {
                vec![Region::Digest(0), Region::Digest(1), if net.epilogue() { Region::Y } else { Region::AccOut }]
            }
            Layout::Chunk(_) | Layout::ChunkTail(_) => {
                let mut v = vec![Region::Key, Region::Counter, Region::Cv, Region::AccIn, Region::AccOut];
                if net.epilogue() {
                    v.push(Region::Y);
                }
                if lay.tail().is_some() {
                    v.extend([Region::Flags, Region::CvMid, if net.epilogue() { Region::YMid } else { Region::AccMid }]);
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
        assert!(2 * regions.len() <= MAX_EXTRA_CLAIMS, "{lay:?}: {} region claims, the GPU prover takes {MAX_EXTRA_CLAIMS}", 2 * regions.len());
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
                    let (mut cv, ctr) = if lay.is_chunked() { ([0u32; 8], 0u64) } else { (keys[role], c as u64) };
                    for j in 0..nb {
                        cv = blake3_compress(&cv, &[0u32; 16], ctr, lay.blen(j, nb), lay.flags(j, nb))[..8].try_into().unwrap();
                    }
                    cv
                })
            })
            .collect() };
        // ChunkTail: a dummy block's slot 7 / 23 chaining values follow
        let dummy_cv = if lay.tail().is_some() {
            let mut cv = [0u32; 8];
            for j in 0..8 {
                cv = blake3_compress(&cv, &[0u32; 16], 0, 64, chunk_flags(j, 16))[..8].try_into().unwrap();
            }
            [dummy_cv.clone(), vec![cv, cv]].concat()
        } else {
            dummy_cv
        };
        let (z, _, _) = net.eval64(|_| 0);
        let word = |col: usize, n: usize| (0..n).fold(0u32, |w, t| w | (((z[col * 128 + t] & 1) as u32) << t));
        let dummy_unit = (word(net.out_cols[0], 32), if net.epilogue() { word(net.out_cols[1], 16) } else { 0 });
        let mut h = Sha256::new();
        h.update(TAG);
        h.update(comp().statement_digest());
        h.update(net.sha256);
        for x in [vus, blocks_real, nbl, m, kl, ul, slots, units, p0] {
            h.update((x as u64).to_le_bytes());
        }
        if let Layout::ChunkTail(n) = lay {
            h.update(format!("layout chunk-tail: block = (VU, chunk), {n} whole chunks and a 512-byte partial chunk per VU, committed cross-chunk accumulators, per-block flags, partial chunk's chaining value at slot 7 and output at unit 15").as_bytes());
        }
        if let Layout::Chunk(n) = lay {
            if n != 3 {
                h.update(format!("layout chunk: block = (VU, chunk), {n} chunks per VU, committed cross-chunk accumulators").as_bytes());
            }
        }
        if lay == Layout::Fp8 {
            h.update(b"layout fp8: block = VU, chunks 16+8, per-slot key/counter constants, c_in(0) = 0");
        }
        if lay.is_fp4() {
            h.update(lay.row_schema().as_bytes());
            h.update(b"layout fp4-nvf4: block = VU, 864-byte nvfp4 rows (768 code bytes, 96 scale bytes), one chunk / chain per role, 24 units reading 32 code bytes and one scale word, c_in(0) = 0");
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
        PureStmt { lay, net, vus, blocks_real, nbl, m, m_pts: PT_LOCAL + nbl, regions, circuit: PureCircuit { lay, ul, unit, da, db }, r1cs, digest,
                   keys, dummy_cv, dummy_unit, ul, tail_cin: OnceLock::new() }
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
                let mut mids = Vec::new();
                for role in 0..2 {
                    for (c, &nb) in lay.chunks().iter().enumerate() {
                        let (v, ci) = self.vu_chunk(b, c);
                        let (mut cv, ctr) = match (lay, real) {
                            (_, true) => (self.keys[role], ci as u64),
                            (l, false) if l.is_chunked() => ([0; 8], 0),
                            (_, false) => (self.keys[role], c as u64),
                        };
                        for j in 0..nb {
                            let msg: [u32; 16] = if real {
                                let at = ci * 1024 + 64 * j;
                                let row = inst.row(role, v);
                                from_fn(|i| if at + 4 * i < row.len() { u32::from_le_bytes(row[at + 4 * i..at + 4 * i + 4].try_into().unwrap()) } else { 0 })
                            } else {
                                [0; 16]
                            };
                            let f = if lay.tail().is_some() { lay.slot_flags(b % lay.blocks_per_vu(), real, j) } else { lay.flags(j, nb) };
                            let bl = lay.blen(j, nb);
                            out.push((cv, msg, ctr, bl, f));
                            cv = blake3_compress(&cv, &msg, ctr, bl, f)[..8].try_into().unwrap();
                            if lay.tail().is_some() && j == 7 {
                                mids.push(cv);
                            }
                        }
                        if lay.n_cv() > 0 {
                            cvs.push(cv);
                        }
                    }
                }
                cvs.extend(mids);
                (out, cvs)
            })
            .collect();
        let cvs = per.iter().map(|p| p.1.clone()).take(self.blocks_real).collect();
        (per.into_iter().flat_map(|p| p.0).collect(), cvs)
    }

    /// The SHA-256 chain's final state for row (role, v): the `sha256/row/v1` digest as big-endian words.
    pub fn sha_state(&self, inst: &Instances, role: usize, v: usize) -> [u32; 8] {
        let row = inst.row(role, v);
        let (mut h, full) = (sha_midstate(role, row.len()), row.len() - row.len() % 64);
        for blk in row[..full].chunks(64) {
            h = sha2c::sha256_compress(&h, &be_words16(blk));
        }
        let mut last = sha_pad_block(row.len());
        for (i, w) in row[full..].chunks(4).enumerate() {
            last[i] = u32::from_be_bytes(w.try_into().unwrap());
        }
        sha2c::sha256_compress(&h, &last)
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
                    let mut last = sha_pad_block(lay.row_bytes());
                    if real {
                        let tail = &inst.row(role, b)[64 * (lay.per_role() - 1)..];
                        for (i, w) in tail.chunks(4).enumerate() {
                            last[i] = u32::from_be_bytes(w.try_into().unwrap());
                        }
                    }
                    out.push((h, last));
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
        let scale = |role: usize| -> u64 {
            if !self.lay.is_fp4() || b >= self.blocks_real {
                return 0;
            }
            let r = &inst.row(role, b)[768 + 4 * u..768 + 4 * u + 4];
            u32::from_le_bytes(r.try_into().unwrap()) as u64
        };
        [x[0], x[1], w[0], w[1], F128::new(c_in as u64 | (scale(0) << 32), scale(1))]
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
        let scale = |role: usize| -> u64 {
            if !self.lay.is_fp4() {
                return 0;
            }
            comps[base + role * self.lay.per_role() + Layout::FP4_SCALE_BLOCK + u / 16].1[u % 16] as u64
        };
        [x[0], x[1], w[0], w[1], F128::new(c_in as u64 | (scale(0) << 32), scale(1))]
    }

    /// The honest accumulator input of every unit (the recorded accumulators; zero for dummy blocks).
    pub fn c_in(&self, inst: &Instances, b: usize, u: usize) -> u32 {
        if b >= self.blocks_real {
            return 0;
        }
        let bpv = self.lay.blocks_per_vu();
        let (v, j) = (b / bpv, (b % bpv) * self.lay.units() + u);
        if j == 0 {
            0
        } else if j - 1 < inst.units {
            inst.acc(v, j - 1)
        } else {
            self.tail_chain(inst)[v][u - 16]
        }
    }

    /// Everything a prover session precomputes from the instance set before any parallel witness work: `ChunkTail`'s
    /// continuation accumulators.
    pub fn prepare(&self, inst: &Instances) {
        if self.lay.tail().is_some() {
            self.tail_chain(inst);
        }
    }

    /// `ChunkTail`: the accumulator into units 16..32 of each real VU's partial-chunk block, evaluated (zero operands).
    /// Sequential: `c_in` reaches it from rayon workers, and a parallel initialiser would wait on the workers blocked on
    /// this lock (a deadlock); `prepare` computes it up front.
    fn tail_chain(&self, inst: &Instances) -> &Vec<[u32; 16]> {
        self.tail_cin.get_or_init(|| {
            let col = self.net.out_cols[0];
            (0..inst.vus.div_ceil(64))
                .flat_map(|g| {
                    let vs: Vec<usize> = (64 * g..(64 * g + 64).min(inst.vus)).collect();
                    let mut t = vec![[0u32; 16]; vs.len()];
                    for (l, &v) in vs.iter().enumerate() {
                        t[l][0] = inst.acc(v, inst.units - 1);
                    }
                    for k in 1..16 {
                        let (z, _, _) = self.net.eval64(|bit| {
                            if (512..544).contains(&bit) {
                                t.iter().enumerate().fold(0u64, |x, (l, c)| x | ((((c[k - 1] >> (bit - 512)) & 1) as u64) << l))
                            } else {
                                0
                            }
                        });
                        for (l, c) in t.iter_mut().enumerate() {
                            c[k] = (0..32).fold(0u32, |w, s| w | ((((z[col * 128 + s] >> l) & 1) as u32) << s));
                        }
                    }
                    t
                })
                .collect()
        })
    }

    /// `(z, a, b, z_lincheck)` of the block witness: compressions, then units evaluated from `inputs(b, u)`.
    pub fn witness(&self, comps: &Comps, inputs: &(dyn Fn(usize, usize) -> [F128; 5] + Sync)) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let (kl, slots, units, p0) = (self.lay.k_log(), self.lay.comp_slots(), self.lay.units(), self.lay.unit_pos0(self.ul));
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
        let uw = 1usize << (self.ul - 7);
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
            Region::Digest(r) => le(&if !real {
                self.dummy_cv[r]
            } else if self.lay.is_sha() {
                self.sha_state(inst, r, v)
            } else {
                let key = crate::chunk::row_key(if r == 0 { b"x" } else { b"w" });
                from_fn(|i| u32::from_le_bytes(blake3::keyed_hash(&key, inst.row(r, v)).as_bytes()[4 * i..4 * i + 4].try_into().unwrap()))
            }),
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
                (l, true) if l.is_chunked() => p.acc[b],
                (_, true) => inst.out_word(v),
            })
            .to_le_bytes()
            .to_vec(),
            Region::Y => {
                let y = if !real {
                    self.dummy_unit.1
                } else if c + 1 == self.lay.blocks_per_vu() && self.lay.tail().is_none() {
                    inst.out_word(v)
                } else {
                    p.y16[b]
                };
                (y as u16).to_le_bytes().to_vec()
            }
            Region::Flags => le(&(0..COMP_SLOTS).map(|s| self.lay.slot_flags(c, real, s % 16)).collect::<Vec<u32>>()),
            Region::CvMid => {
                let cv = |r: usize| if real { p.cv[b][2 + r] } else { self.dummy_cv[2 + r] };
                [le(&cv(0)), le(&cv(1))].concat()
            }
            Region::AccMid | Region::YMid => {
                let at_tail = real && Some(c) == self.lay.tail();
                let (dummy, mine) = if reg.region == Region::AccMid { (self.dummy_unit.0, p.mid.get(b).map_or(0, |m| m.1)) } else { (self.dummy_unit.1, p.mid.get(b).map_or(0, |m| m.0)) };
                let w = if !real { dummy } else if at_tail { inst.out_word(v) } else { mine };
                if reg.region == Region::AccMid { w.to_le_bytes().to_vec() } else { (w as u16).to_le_bytes().to_vec() }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A lowered unit's shape without its logic: input rows, one output word (two with the epilogue), the constant.
    fn shape_net(n_in: usize, epilogue: bool) -> UnitNet {
        let useful = 1024;
        let (mut a, mut b): (Vec<Vec<usize>>, Vec<Vec<usize>>) = (0..useful).map(|i| (if i < n_in { vec![i] } else { vec![] }, if i < n_in { vec![i] } else { vec![] })).unzip();
        a[useful - 1] = vec![useful - 1];
        b[useful - 1] = vec![useful - 1];
        UnitNet { relation: "shape".into(), useful, const_pos: useful - 1, n_in, out_cols: if epilogue { vec![5, 6] } else { vec![5] }, a, b, sha256: [0; 32] }
    }

    /// Every layout's region claims fit the GPU prover (ChunkTail with the epilogue once had 18 against a cap of 16).
    #[test]
    fn region_claims_fit_the_gpu_prover() {
        let layouts = [Layout::Chunk(3), Layout::Chunk(16), Layout::ChunkTail(4), Layout::ChunkTail(17), Layout::Fp8, Layout::ShaBf16, Layout::ShaFp8];
        for lay in layouts {
            for epilogue in [false, true] {
                let st = PureStmt::new(shape_net(544, epilogue), 8, lay);
                assert!(2 * st.regions.len() <= MAX_EXTRA_CLAIMS, "{lay:?} epilogue {epilogue}");
            }
        }
        for lay in [Layout::Fp4, Layout::ShaFp4] {
            let st = PureStmt::new(shape_net(608, false), 8, lay);
            assert!(2 * st.regions.len() <= MAX_EXTRA_CLAIMS, "{lay:?}");
        }
        assert_eq!(PureStmt::new(shape_net(544, true), 8, Layout::ChunkTail(4)).regions.len(), 9);
    }

    /// A unit past 2^13 rows takes 2^14-row slots: the Chunk / ChunkTail / Fp8 / Fp4 / ShaFp4 blocks keep their k_log
    /// (their spare room holds the doubled units), the ShaBf16 / ShaFp8 blocks refuse it (UL1).
    #[test]
    fn total_unit_slots() {
        let mut big = shape_net(544, true);
        let useful = 8449;
        big.a.resize(useful, vec![]);
        big.b.resize(useful, vec![]);
        big.a[1023] = vec![];
        big.b[1023] = vec![];
        big.a[useful - 1] = vec![useful - 1];
        big.b[useful - 1] = vec![useful - 1];
        big.useful = useful;
        big.const_pos = useful - 1;
        assert_eq!(big.unit_log(), 14);
        for lay in [Layout::Chunk(4), Layout::Chunk(16), Layout::ChunkTail(17), Layout::Fp8] {
            assert!(lay.fits(14), "{lay:?}");
            let st = PureStmt::new(big.clone(), 8, lay);
            assert_eq!((st.ul, st.lay.k_log()), (14, lay.k_log()));
        }
        assert_eq!(Layout::Chunk(4).unit_pos0(14), 32);
        for lay in [Layout::ShaBf16, Layout::ShaFp8] {
            assert!(!lay.fits(14), "{lay:?}");
        }
        assert!(Layout::Fp4.fits(14) && Layout::ShaFp4.fits(14));
    }

    /// ChunkTail's continuation accumulators, first reached from rayon workers (as the witness builders reach them) and
    /// without `prepare`, complete: a parallel initialiser under the OnceLock deadlocked the GPU prover.
    #[test]
    fn tail_chain_from_rayon_workers_completes() {
        let (vus, units) = (2048, 144);
        let st = std::sync::Arc::new(PureStmt::new(shape_net(544, true), vus, Layout::ChunkTail(4)));
        let inst = std::sync::Arc::new(Instances {
            header: serde_json::json!({}), relation: "shape".into(), vus, units, row_bytes: 4608, x: vec![0; vus * 4608],
            w: vec![0; vus * 4608], accs: (0..vus * units).map(|i| i as u32).collect(), out: vec![0; vus], y: vec![0; vus],
        });
        let (tx, rx) = std::sync::mpsc::channel();
        let (st2, inst2) = (st.clone(), inst.clone());
        // 128 workers: enough that a worker waiting inside the initialiser steals an outer c_in task (the old code hung here)
        let pool = rayon::ThreadPoolBuilder::new().num_threads(128).build().unwrap();
        std::thread::spawn(move || {
            let n: u64 = pool.install(|| (0..st2.blocks_real * 32).into_par_iter().map(|i| st2.c_in(&inst2, i / 32, i % 32) as u64).sum());
            tx.send(n).unwrap();
        });
        assert!(rx.recv_timeout(std::time::Duration::from_secs(120)).is_ok(), "c_in from rayon workers did not complete");
    }
}
