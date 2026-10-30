//! `verity/flock-vllm-block/v1`: the Flock backend's relation as ONE block R1CS whose operands are committed under
//! `vllm-v1` (core spec `verity/ml/commitments/vllm_v1/PROTOCOL.md`), lane flock-vllm-v1. The shape is flock-pure-block's
//! (`pure_block`, lane flock-gpu-link), with the keyed-BLAKE3 row leaves replaced by `vllm-v1` position leaves.
//!
//! Block `v` is VU `v` (`k_log` = log2ceil(2 · per_role · 2^15 + units · 2^13): 21 for 8-bit rows):
//! - SHA-256 slots `0..per_role` (2^15 bits each, Flock's `sha2` circuit): the x row's position leaf
//!   `SHA-256("verity/pos-leaf/v0" || u64be(row_bytes) || row)`, one compression per 64-byte message block
//!   (per_role = ⌈(35 + row_bytes) / 64⌉: 25 for 1536-byte rows); slots `per_role..2 per_role`: the W column's;
//! - 2^13-bit positions `unit_pos0..unit_pos0 + units` (unit_pos0 = 4 · 2 per_role): the transition units (the pinned
//!   `flock-unit-io/v1` lowering), unit u reading the row bytes `[32 u, 32 u + 32)` of x and of W;
//! - the rest is empty (forced zero).
//!
//! `A = I ⊗ circuit + Δ` (and B), where Δ makes, inside the block:
//! - each role a chain: `H_in` of slot 0 the SHA-256 IV, `H_in` of slot j ≥ 1 a copy of `H_out` of slot j − 1;
//! - the message bytes that are not row bytes constants: the 26-byte prefix at message bytes `[0, 26)` (so the value
//!   length is the statement's, PROTOCOL.md §9.4) and FIPS 180-4 padding after byte `26 + row_bytes`; row byte t sits at
//!   message byte `26 + t` (offset absorption, §8.1) as the big-endian SHA-256 word bit it is;
//! - unit u's x and W input bits copies of the message bits that carry those row bytes, its accumulator input a copy of
//!   unit u − 1's `c_out` (u ≥ 1), unit 0's forced to +0;
//! - every sub-circuit's constant a copy of the one pinned column (slot 0's).
//!
//! Public regions, each opened at two verifier points sampled after the commitment (both reps): each role's leaf digest
//! (`H_out` of its last slot, 256 bits, sent in `Commit`) and the VU's output word (unit `units − 1`'s `c_out`, or its
//! `y16` for an epilogue relation). The verifier folds the digests it was sent, VU v's as leaf v, into the `vllm-v1`
//! tree of its port (pairs `node`, an odd tail `lift`, no padding), binds the tree root under the port's verifier-held
//! `StepDomain` (§4, §5) and compares that step root with the root it holds; the output words' tree it computes itself.
//! Dummy blocks (the block count is a power of two) are position leaves of zero rows and zero-input unit chains.

use std::{array::from_fn, sync::OnceLock};

use flock_core::{
    lincheck::{LincheckCircuit, SparseMatrixCircuit, pack_z_lincheck_from_packed},
    proof::eq_skip_weights,
    r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
    verifier::ExtraClaim,
};
use flock_field::F128;
use flock_prover::r1cs_hashes::sha2 as sh;
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    chunk::{log2ceil, pack4},
    pure_block::{Instances, UNIT_LOG, UnitNet},
};

pub const TAG: &[u8] = b"verity/flock-vllm-block/v1";
pub const SUB_LOG: usize = sh::K_LOG;
pub const PIN: usize = sh::Z_CONST_POS;
/// Coordinates per verifier point: the regions' free in-block bits come from `r[..8]`, block bits from `r[8..]`.
pub const PT_LOCAL: usize = 8;
pub const PREFIX_LEN: usize = 26;
const POS_LEAF_TAG: &[u8] = b"verity/pos-leaf/v0";

/// Flock's SHA-256 compression circuit (one 2^15 block), the sub-circuit of every hash slot.
pub fn sha_comp() -> &'static BlockR1cs {
    static C: OnceLock<BlockR1cs> = OnceLock::new();
    C.get_or_init(|| sh::build_block_r1cs(3))
}

/// The message a position leaf of `row` hashes, padded (FIPS 180-4): prefix, row, 0x80, zeros, u64be(bit length).
pub fn pos_leaf_message(row: &[u8]) -> Vec<u8> {
    let n = PREFIX_LEN + row.len();
    let blocks = (n + 9).div_ceil(64);
    let mut m = Vec::with_capacity(64 * blocks);
    m.extend_from_slice(POS_LEAF_TAG);
    m.extend_from_slice(&(row.len() as u64).to_be_bytes());
    m.extend_from_slice(row);
    m.push(0x80);
    m.resize(64 * blocks - 8, 0);
    m.extend_from_slice(&(8 * n as u64).to_be_bytes());
    m
}

fn be_words(b: &[u8]) -> [u32; 16] {
    from_fn(|i| u32::from_be_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}

pub fn digest_bytes(h: &[u32; 8]) -> [u8; 32] {
    let mut d = [0u8; 32];
    for (i, w) in h.iter().enumerate() {
        d[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
    }
    d
}

/// The block shape for a row width and unit count.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Layout {
    pub row_bytes: usize,
    pub units: usize,
}

impl Layout {
    pub fn of(row_bytes: usize, units: usize) -> Self {
        assert!(row_bytes == 32 * units, "flock-vllm-block: every unit reads 32 bytes of each operand row");
        Layout { row_bytes, units }
    }
    pub fn per_role(self) -> usize { (PREFIX_LEN + self.row_bytes + 9).div_ceil(64) }
    pub fn comp_slots(self) -> usize { 2 * self.per_role() }
    pub fn unit_pos0(self) -> usize { self.comp_slots() << (SUB_LOG - UNIT_LOG) }
    pub fn k_log(self) -> usize { log2ceil((self.unit_pos0() + self.units) << UNIT_LOG) }
    /// The message byte (slot of the role, byte in the block) and its bit in the big-endian SHA-256 word, for row byte t.
    pub fn row_byte(self, t: usize) -> (usize, usize) {
        let at = PREFIX_LEN + t;
        (at / 64, at % 64)
    }
}

/// The SHA-256 message column of byte `o` (0..64), bit `bit` (LSB first) of a slot: word `o / 4` is big-endian.
fn msg_col(o: usize, bit: usize) -> usize {
    sh::m_bit(o / 4, 8 * (3 - o % 4) + bit)
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Region {
    /// Role r's leaf digest: `H_out` of its last slot.
    Digest(usize),
    /// The VU's output word: unit `units − 1`'s c_out (or y16 with an epilogue).
    Out,
}

pub struct RegionShape {
    pub region: Region,
    pub free: Vec<usize>,
    pub mask: u32,
    pub fixed: u32,
}

/// The prover's per-block publics: each real block's x and W leaf digests (SHA-256 words, as the circuit holds them).
#[derive(Clone)]
pub struct Publics {
    pub digest: Vec<[[u32; 8]; 2]>,
}

impl Publics {
    pub fn bytes(&self) -> Vec<u8> {
        self.digest.iter().flatten().flatten().flat_map(|w| w.to_le_bytes()).collect()
    }

    pub fn parse(b: &[u8], blocks: usize) -> Result<Self, String> {
        if b.len() != 64 * blocks {
            return Err(format!("C4: {} bytes of publics, want {}", b.len(), 64 * blocks));
        }
        let w: Vec<u32> = b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(Publics { digest: (0..blocks).map(|i| from_fn(|r| from_fn(|k| w[16 * i + 8 * r + k]))).collect() })
    }
}

pub struct VllmCircuit {
    pub lay: Layout,
    pub unit: SparseMatrixCircuit<'static>,
    pub da: Vec<(u32, u32)>,
    pub db: Vec<(u32, u32)>,
}

impl LincheckCircuit for VllmCircuit {
    fn n_cols(&self) -> usize {
        1 << self.lay.k_log()
    }

    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        let (cs, us) = (1usize << SUB_LOG, 1usize << UNIT_LOG);
        let q = SUB_LOG - UNIT_LOG;
        let (kl, slots, units, p0) = (self.lay.k_log(), self.lay.comp_slots(), self.lay.units, self.lay.unit_pos0());
        assert_eq!(eq.len(), 1 << kl);
        let c0 = eq[..us].iter().position(|e| *e != F128::ZERO).expect("eq tensor vanishes on the first unit slot");
        let inv = eq[c0].inv();
        let base_c = sha_comp().csc_lincheck_circuit().fold_alpha_batched(alpha, &eq[..cs]);
        let base_u = self.unit.fold_alpha_batched(alpha, &eq[..us]);
        let mut out = vec![F128::ZERO; 1 << kl];
        out.par_chunks_mut(us).enumerate().for_each(|(p, o)| {
            if p >> q < slots {
                let s = eq[((p >> q) << SUB_LOG) | c0] * inv;
                let part = (p & ((1 << q) - 1)) * us;
                for (x, y) in o.iter_mut().zip(&base_c[part..part + us]) {
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
        Some(PIN)
    }
}

/// Δ as XOR toggles of the circuit's rows: `(row, col)` pairs of `A` and of `B`, in block coordinates.
fn delta(net: &UnitNet, lay: Layout) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let c = sha_comp();
    let (a, b) = (&c.a_0.rows, &c.b_0.rows);
    let z = sh::Z_CONST_POS;
    let input = |r: usize| assert!(a[r] == [r] && b[r] == [z], "SHA-256 row {r} is not an input row");
    let (mut da, mut db) = (Vec::new(), Vec::new());
    let both = |da: &mut Vec<(u32, u32)>, db: &mut Vec<(u32, u32)>, i: usize, old: usize, new: usize| {
        for d in [da, db] {
            d.push((i as u32, old as u32));
            d.push((i as u32, new as u32));
        }
    };
    // an input row r of slot s forced to the constant `bit`: A = [bit · Z_s], B = [Z_s]
    let constant = |da: &mut Vec<(u32, u32)>, s: usize, r: usize, bit: bool| {
        input(r);
        let g = |col: usize| ((s << SUB_LOG) + col) as u32;
        da.push((g(r), g(r)));
        if bit {
            da.push((g(r), g(z)));
        }
    };
    let (per, n) = (lay.per_role(), PREFIX_LEN + lay.row_bytes);
    let zeros = vec![0u8; lay.row_bytes];
    let msg = pos_leaf_message(&zeros);
    for s in 0..lay.comp_slots() {
        let j = s % per;
        let g = |col: usize| (s << SUB_LOG) + col;
        if s > 0 {
            both(&mut da, &mut db, g(z), g(z), PIN);
        }
        if j == 0 {
            for (w, &iv) in sh::SHA256_IV.iter().enumerate() {
                for bit in 0..32 {
                    constant(&mut da, s, sh::h_bit(w, bit), (iv >> bit) & 1 == 1);
                }
            }
        } else {
            for r in 0..256 {
                input(sh::H_BASE + r);
                da.push((g(sh::H_BASE + r) as u32, g(sh::H_BASE + r) as u32));
                da.push((g(sh::H_BASE + r) as u32, (((s - 1) << SUB_LOG) + sh::H_OUT_BASE + r) as u32));
            }
        }
        for o in 0..64 {
            let at = 64 * j + o;
            if at < PREFIX_LEN || at >= n {
                for bit in 0..8 {
                    constant(&mut da, s, msg_col(o, bit), (msg[at] >> bit) & 1 == 1);
                }
            }
        }
    }
    let p0 = lay.unit_pos0();
    for u in 0..lay.units {
        let base = (p0 + u) << UNIT_LOG;
        both(&mut da, &mut db, base + net.const_pos, base + net.const_pos, PIN);
        for role in 0..2 {
            for t in 0..256 {
                let (j, o) = lay.row_byte(32 * u + t / 8);
                let col = ((role * per + j) << SUB_LOG) + msg_col(o, t % 8);
                both(&mut da, &mut db, base + 256 * role + t, base + 256 * role + t, col);
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

pub struct VllmStmt {
    pub lay: Layout,
    pub net: UnitNet,
    pub vus: usize,
    pub nbl: usize,
    pub m: usize,
    pub m_pts: usize,
    pub regions: Vec<RegionShape>,
    pub circuit: VllmCircuit,
    pub r1cs: BlockR1cs,
    pub digest: [u8; 32],
    /// A dummy block's leaf digest (a zero row's position leaf) and its unit chain's output word.
    pub dummy_digest: [u32; 8],
    pub dummy_out: u32,
    /// A dummy block's accumulator input of every unit (the zero-input chain from +0).
    pub dummy_cin: Vec<u32>,
}

impl VllmStmt {
    pub fn new(net: UnitNet, vus: usize, lay: Layout) -> Self {
        // UL2: this statement's units are 2^13-row slots; refuse a larger netlist rather than truncate it
        assert_eq!(net.unit_log(), UNIT_LOG, "UL2: a {}-row unit does not fit flock-vllm-block's 2^13-row unit slot", net.useful);
        let nbl = log2ceil(vus).max(3);
        let (kl, slots, units, p0) = (lay.k_log(), lay.comp_slots(), lay.units, lay.unit_pos0());
        assert!(p0 + units <= 1 << (kl - UNIT_LOG));
        let m = kl + nbl;
        let (da, db) = delta(&net, lay);
        let out_col = |net: &UnitNet| if net.epilogue() { (net.out_cols[1], 16) } else { (net.out_cols[0], 32) };
        let (oc, ob) = out_col(&net);
        let shape = |region: Region| -> RegionShape {
            let bits = |n: usize| (0..n).collect::<Vec<_>>();
            let (free, fixed): (Vec<usize>, u32) = match region {
                Region::Digest(r) => (bits(8), (1 << 8) | ((((r + 1) * lay.per_role() - 1) as u32) << SUB_LOG)),
                Region::Out => (bits(log2ceil(ob)), ((oc as u32) << 7) | (((p0 + units - 1) as u32) << UNIT_LOG)),
            };
            let mask = ((1u32 << kl) - 1) & !free.iter().fold(0u32, |m, &b| m | (1 << b));
            assert_eq!(fixed & !mask, 0);
            RegionShape { region, free, mask, fixed }
        };
        let regions = vec![shape(Region::Digest(0)), shape(Region::Digest(1)), shape(Region::Out)];
        let (mut h, zero_msg) = (sh::SHA256_IV, pos_leaf_message(&vec![0u8; lay.row_bytes]));
        for blk in zero_msg.chunks(64) {
            h = sh::sha256_compress(&h, &be_words(blk));
        }
        let (mut dummy_cin, mut c) = (Vec::with_capacity(units), 0u32);
        let mut dummy_out = 0;
        for _ in 0..units {
            dummy_cin.push(c);
            let (z, _, _) = net.eval64(|bit| if (512..544).contains(&bit) { if (c >> (bit - 512)) & 1 == 1 { !0 } else { 0 } } else { 0 });
            let word = |col: usize, n: usize| (0..n).fold(0u32, |w, t| w | (((z[col * 128 + t] & 1) as u32) << t));
            c = word(net.out_cols[0], 32);
            dummy_out = word(oc, ob);
        }
        let mut hd = Sha256::new();
        hd.update(TAG);
        hd.update(sha_comp().statement_digest());
        hd.update(net.sha256);
        for x in [vus, nbl, m, kl, SUB_LOG, UNIT_LOG, slots, units, p0, lay.row_bytes, PREFIX_LEN] {
            hd.update((x as u64).to_le_bytes());
        }
        hd.update(b"vllm-v1 position leaves: SHA-256(verity/pos-leaf/v0 || u64be(row_bytes) || row), IV and FIPS 180-4 padding constant");
        for d in [&da, &db] {
            hd.update((d.len() as u64).to_le_bytes());
            for &(i, c) in d.iter() {
                hd.update(i.to_le_bytes());
                hd.update(c.to_le_bytes());
            }
        }
        let digest: [u8; 32] = hd.finalize().into();
        let kk = 1usize << kl;
        let empty = || SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: vec![Vec::new(); kk].into() };
        let r1cs = BlockR1cs {
            m,
            k_log: kl,
            k_skip: sh::K_SKIP,
            useful_bits: kk,
            a_0: empty(),
            b_0: empty(),
            c_0: SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: (0..kk).map(|i| vec![i]).collect::<Vec<_>>().into() },
            layout: WitnessLayout::RowMajor,
            const_pin: Some(PIN),
            digest_cache: OnceLock::from(digest),
            csc_cache: OnceLock::new(),
        };
        let k = 1usize << UNIT_LOG;
        let pad = |rows: &Vec<Vec<usize>>| {
            let mut r = rows.clone();
            r.resize(k, Vec::new());
            SparseBinaryMatrix { num_rows: k, num_cols: k, rows: r.into() }
        };
        let (ua, ub): (&'static SparseBinaryMatrix, &'static SparseBinaryMatrix) = (Box::leak(Box::new(pad(&net.a))), Box::leak(Box::new(pad(&net.b))));
        let unit = SparseMatrixCircuit::new(ua, ub);
        VllmStmt { lay, net, vus, nbl, m, m_pts: PT_LOCAL + nbl, regions, circuit: VllmCircuit { lay, unit, da, db }, r1cs, digest,
                   dummy_digest: h, dummy_out, dummy_cin }
    }

    pub fn blocks(&self) -> usize {
        1 << self.nbl
    }

    /// Every compression, block-major (each block's x slots, then its W slots), and each real block's leaf digests.
    pub fn compressions(&self, inst: &Instances) -> (Vec<sh::Compression>, Vec<[[u32; 8]; 2]>) {
        let zeros = vec![0u8; self.lay.row_bytes];
        let per: Vec<(Vec<sh::Compression>, [[u32; 8]; 2])> = (0..self.blocks())
            .into_par_iter()
            .map(|b| {
                let mut out = Vec::with_capacity(self.lay.comp_slots());
                let d: [[u32; 8]; 2] = from_fn(|role| {
                    let row = if b < self.vus { inst.row(role, b) } else { &zeros };
                    let mut h = sh::SHA256_IV;
                    for blk in pos_leaf_message(row).chunks(64) {
                        let m = be_words(blk);
                        out.push((h, m));
                        h = sh::sha256_compress(&h, &m);
                    }
                    h
                });
                (out, d)
            })
            .collect();
        let digests = per.iter().take(self.vus).map(|p| p.1).collect();
        (per.into_iter().flat_map(|p| p.0).collect(), digests)
    }

    /// Unit (block b, u)'s 544 input bits: the x and W row bytes it reads (from the compressions' messages), c_in.
    pub fn unit_inputs(&self, comps: &[sh::Compression], b: usize, u: usize, c_in: u32) -> [F128; 5] {
        let base = b * self.lay.comp_slots();
        let per = self.lay.per_role();
        let bytes = |role: usize| -> [u8; 32] {
            from_fn(|i| {
                let (j, o) = self.lay.row_byte(32 * u + i);
                let w = comps[base + role * per + j].1[o / 4];
                (w >> (8 * (3 - o % 4))) as u8
            })
        };
        let words = |role: usize| -> [F128; 2] {
            let b = bytes(role);
            let w: [u32; 8] = from_fn(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()));
            [pack4(&w[..4]), pack4(&w[4..])]
        };
        let (x, w) = (words(0), words(1));
        [x[0], x[1], w[0], w[1], F128::new(c_in as u64, 0)]
    }

    /// The honest accumulator input of every unit (the recorded accumulators; the zero chain for dummy blocks).
    pub fn c_in(&self, inst: &Instances, b: usize, u: usize) -> u32 {
        if b >= self.vus {
            return self.dummy_cin[u];
        }
        if u == 0 { 0 } else { inst.acc(b, u - 1) }
    }

    /// `(z, a, b, z_lincheck)` of the block witness: SHA-256 compressions, then units evaluated from `inputs(b, u)`.
    pub fn witness(&self, comps: &[sh::Compression], inputs: &(dyn Fn(usize, usize) -> [F128; 5] + Sync)) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let (kl, slots, units, p0) = (self.lay.k_log(), self.lay.comp_slots(), self.lay.units, self.lay.unit_pos0());
        let n = 1usize << (self.m - 7);
        assert_eq!(comps.len(), self.blocks() * slots);
        let trace = std::env::var_os("FLOCK_VLLM_TRACE").is_some();
        let t0 = std::time::Instant::now();
        let (cz, ca, cb, _) = sh::generate_witness_with_ab_packed_and_lincheck(comps, log2ceil(comps.len()));
        let (mut z, mut a, mut b) = (vec![F128::ZERO; n], vec![F128::ZERO; n], vec![F128::ZERO; n]);
        let wpb = 1usize << (kl - 7);
        let cw = 1usize << (SUB_LOG - 7);
        for (dst, src) in [(&mut z, &cz), (&mut a, &ca), (&mut b, &cb)] {
            dst.par_chunks_mut(wpb).enumerate().for_each(|(blk, d)| {
                let ci = blk * slots;
                d[..slots * cw].copy_from_slice(&src[ci * cw..(ci + slots) * cw]);
            });
        }
        drop((cz, ca, cb));
        let t1 = t0.elapsed().as_secs_f64();
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
        let t2 = t0.elapsed().as_secs_f64();
        let zl = pack_z_lincheck_from_packed(&z, self.m, kl);
        if trace {
            eprintln!("WITNESS\tsha256+scatter {t1:.3}s units {:.3}s zlin {:.3}s", t2 - t1, t0.elapsed().as_secs_f64() - t2);
        }
        (z, a, b, zl)
    }

    /// The value of a public region at block b, as bytes (bit f of the region's free assignment is bit f % 8 of byte f / 8).
    fn region_bits(&self, reg: &RegionShape, inst: &Instances, p: &Publics, b: usize) -> Vec<u8> {
        let real = b < self.vus;
        match reg.region {
            Region::Digest(r) => (if real { p.digest[b][r] } else { self.dummy_digest }).iter().flat_map(|w| w.to_le_bytes()).collect(),
            Region::Out => {
                let y = if real { inst.out[b] } else { self.dummy_out };
                let n = (1usize << reg.free.len()) / 8;
                y.to_le_bytes()[..n].to_vec()
            }
        }
    }

    pub fn region_point(&self, reg: &RegionShape, r: &[F128]) -> Vec<F128> {
        let kl = self.lay.k_log();
        let mut p = vec![F128::ZERO; self.m];
        for (j, x) in p.iter_mut().enumerate().take(kl) {
            if (reg.mask >> j) & 1 == 1 && (reg.fixed >> j) & 1 == 1 {
                *x = F128::ONE;
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

/// `vllm-v1` natively (PROTOCOL.md §1-§5, §9), with the port domains of `archive/direct/ligero/vllm_tree.py` (the
/// same derivation as B-Ligero's `+vllm-v1` statements, so a Flock cell and a B-Ligero cell on one line bind the same
/// three roots).
pub mod v1 {
    use super::*;

    pub const DOMAIN_TAG: &str = "verity/vllm-v1/domain/v1";
    pub const PORT_TAG: &str = "verity/ligero-b/vllm-v1";
    const STEP_ROOT_TAG: &[u8] = b"verity/fa2c/root/v0";
    const NODE_TAG: &[u8] = b"verity-vllm/node/v1";
    const LIFT_TAG: &[u8] = b"verity-vllm/lift/v1";
    const EMPTY_TAG: &[u8] = b"verity-vllm/empty/v1";
    pub const TREES: [&str; 3] = ["a", "b", "y"];
    const VALUES: [&str; 3] = ["x-row", "w-column", "y-word-le"];

    fn framed(tag: &[u8], parts: &[&[u8]]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update((tag.len() as u32).to_be_bytes());
        h.update(tag);
        for p in parts {
            h.update((p.len() as u64).to_be_bytes());
            h.update(p);
        }
        h.finalize().into()
    }

    pub fn node(l: &[u8; 32], r: &[u8; 32]) -> [u8; 32] {
        framed(NODE_TAG, &[l, r])
    }

    pub fn lift(c: &[u8; 32]) -> [u8; 32] {
        framed(LIFT_TAG, &[c])
    }

    pub fn pos_leaf(value: &[u8]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(POS_LEAF_TAG);
        h.update((value.len() as u64).to_be_bytes());
        h.update(value);
        h.finalize().into()
    }

    /// The tree root over `leaves` in order (§3): pairs `node`, an odd trailing node `lift`, no padding.
    pub fn fold(leaves: &[[u8; 32]]) -> [u8; 32] {
        if leaves.is_empty() {
            return framed(EMPTY_TAG, &[]);
        }
        let mut level = leaves.to_vec();
        while level.len() > 1 {
            level = level.par_chunks(2).map(|p| if p.len() == 2 { node(&p[0], &p[1]) } else { lift(&p[0]) }).collect();
        }
        level[0]
    }

    /// `identity_digest(tag, doc)` (`verity.ml.commitments.identity`): tagged SHA-256 of the compact sorted-key JSON.
    pub fn identity(tag: &str, doc: &Value) -> [u8; 32] {
        let json = serde_json::to_string(doc).unwrap();
        let mut h = Sha256::new();
        h.update(b"veritor/tagged-sha256/v1\0");
        h.update((tag.len() as u32).to_be_bytes());
        h.update(tag.as_bytes());
        h.update((json.len() as u64).to_be_bytes());
        h.update(json.as_bytes());
        h.finalize().into()
    }

    pub fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// What a port's digests derive from: the relation's shape, never the prover's.
    #[derive(Clone, Copy, Debug)]
    pub struct Shape {
        pub k: u64,
        pub word_bits: u64,
        pub y_bytes: u64,
    }

    /// One port's verifier-held `StepDomain` (§5): program, ctx, geo, layout and N.
    #[derive(Clone, Debug)]
    pub struct Domain {
        pub program: [u8; 32],
        pub ctx: [u8; 32],
        pub geo: [u8; 32],
        pub layout: [u8; 32],
        pub n: u64,
    }

    impl Domain {
        /// `vllm_tree.domain(tree, ctx_digest(dataset, tier, manifest, lo, hi, tree), hi - lo, K, word_bits, y_bytes)`.
        pub fn of(tree: usize, s: &Shape, instances: &Value) -> Self {
            let t = |k: &str| instances[k].as_str().unwrap_or_else(|| panic!("instances.{k}")).to_string();
            let (lo, hi) = (instances["range"][0].as_u64().unwrap(), instances["range"][1].as_u64().unwrap());
            let ctx = identity(&format!("{PORT_TAG}/ctx/v1"), &json!({"dataset": t("dataset"), "tier": t("tier"),
                "manifest_sha256": t("manifest_sha256"), "lo": lo, "hi": hi, "tree": TREES[tree]}));
            let program = identity(&format!("{PORT_TAG}/program/v1"), &json!({"scheme": "vllm-v1", "subcircuit": "gemm-coordinate", "K": s.k}));
            let geo = identity(&format!("{PORT_TAG}/geo/v1"), &json!({"K": s.k, "word_bits": s.word_bits, "row_bytes": s.k * s.word_bits / 8, "y_bytes": s.y_bytes}));
            let value_len = if tree == 2 { s.y_bytes } else { s.k * s.word_bits / 8 };
            let layout = identity(&format!("{PORT_TAG}/layout/v1"), &json!({"tree": TREES[tree], "leaf": "pos-leaf", "value": VALUES[tree],
                "value_len": value_len, "word_bits": if tree == 2 { 8 * s.y_bytes } else { s.word_bits }}));
            Domain { program, ctx, geo, layout, n: hi - lo }
        }

        /// `domain_digest(StepDomain)`: what the statement absorbs (§9.1).
        pub fn digest(&self) -> [u8; 32] {
            identity(DOMAIN_TAG, &json!({"kind": "step", "leaf_count": self.n, "program_digest": hex(&self.program), "ctx_digest": hex(&self.ctx),
                "geo_digest": hex(&self.geo), "layout_digest": hex(&self.layout), "chunk": Value::Null}))
        }

        /// The step-root binding of a tree root (§4): the root a statement names, never the bare tree root.
        pub fn bind(&self, tree_root: &[u8; 32]) -> [u8; 32] {
            let mut h = Sha256::new();
            for p in [STEP_ROOT_TAG, &self.program[..], &self.ctx[..], &self.geo[..], &self.layout[..], &self.n.to_be_bytes()[..], &tree_root[..]] {
                h.update(p);
            }
            h.finalize().into()
        }
    }

    /// The output words' leaves: `pos_leaf(y word, y_bytes little-endian)`.
    pub fn y_leaves(y: &[u32], y_bytes: usize) -> Vec<[u8; 32]> {
        y.par_iter().map(|w| pos_leaf(&w.to_le_bytes()[..y_bytes])).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{v1::*, *};

    fn h(s: &str) -> [u8; 32] {
        (0..32).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect::<Vec<_>>().try_into().unwrap()
    }

    fn vectors() -> Option<Value> {
        let p = std::env::var("VLLM_V1_VECTORS").ok()?;
        Some(serde_json::from_str(&std::fs::read_to_string(p).ok()?).unwrap())
    }

    /// UL2: a unit past 2^13 rows (a total unit) is refused, not truncated to the 2^13-row slot.
    #[test]
    #[should_panic(expected = "UL2")]
    fn a_unit_past_2_13_rows_is_refused() {
        let useful = 8449;
        let (mut a, mut b): (Vec<Vec<usize>>, Vec<Vec<usize>>) = (0..useful).map(|i| if i < 544 { (vec![i], vec![i]) } else { (vec![], vec![]) }).unzip();
        a[useful - 1] = vec![useful - 1];
        b[useful - 1] = vec![useful - 1];
        let net = UnitNet { relation: "shape".into(), useful, const_pos: useful - 1, n_in: 544, out_cols: vec![5, 6], a, b, sha256: [0; 32] };
        VllmStmt::new(net, 8, Layout::of(1536, 48));
    }

    #[test]
    fn pos_leaf_by_compressions_is_sha256() {
        for n in [0usize, 1, 37, 38, 102, 1536, 3072] {
            let row: Vec<u8> = (0..n).map(|i| (i * 131 + 7) as u8).collect();
            let mut hh = sh::SHA256_IV;
            for blk in pos_leaf_message(&row).chunks(64) {
                hh = sh::sha256_compress(&hh, &be_words(blk));
            }
            assert_eq!(digest_bytes(&hh), pos_leaf(&row), "n = {n}");
        }
        assert_eq!(Layout::of(1536, 48).per_role(), 25);
        assert_eq!(Layout::of(1536, 48).k_log(), 21);
        assert_eq!(Layout::of(3072, 96).k_log(), 22);
    }

    #[test]
    fn row_bytes_land_in_the_message_bits_the_units_copy() {
        let lay = Layout::of(1536, 48);
        let row: Vec<u8> = (0..1536).map(|i| (i * 37 + 11) as u8).collect();
        let msg = pos_leaf_message(&row);
        for t in [0usize, 1, 37, 38, 101, 102, 1509, 1510, 1535] {
            let (j, o) = lay.row_byte(t);
            let w = be_words(&msg[64 * j..64 * j + 64]);
            for bit in 0..8 {
                let col = msg_col(o, bit) - sh::M_BASE;
                assert_eq!((w[col / 32] >> (col % 32)) & 1, ((row[t] >> bit) & 1) as u32, "row byte {t} bit {bit}");
            }
        }
    }

    #[test]
    fn core_vectors_five_leaf_tree_and_step_domain() {
        // vllm_v1/vectors.json trees[leaf_count = 5] and openings[kind = step, chunk = null, accept]
        let leaves = [
            h("c2a6241fc150fed734162ce4d9275170d893511c8e6d62660f10760717aff704"),
            h("c1fc4595c06884a5eb8ce2bf58ef713a58f02bb2ce18947b13d3eea01a83f613"),
            h("ee97d21d8d747649c4c9d0bad4bf5b363f006d167bd3d0fd8e73abc1773d0af5"),
            h("bf7dc118dd2f36187938c69d2f3211e67deb0054ebad3fed172fb3152418dee1"),
            h("09b89a0fa7e7288b94db34366273332f32f41f2e3ce1a6b2ced095923646cbc6"),
        ];
        assert_eq!(fold(&leaves), h("a044fbedcadda5234607f8168dd7d5efe3e29163f8dd25e6de43644ee3a08b7f"));
        let d = Domain { program: h("506511e550d9a329b253c9d4303a1d67fd8216953ef94417729e31e18325a0a3"),
                         ctx: h("145178591562508490adcbafbee329e07596c8448d3d86d2970894f0774cade3"),
                         geo: h("09a5f931b7afff1fd468818b013c0adec8b53f27c4ceb090214dfeac5888946f"),
                         layout: h("d6d6be6eb71f5db00ffd5b285a3bf7b4ce9b45019bb848bfb022947e7a5a41f6"), n: 6 };
        assert_eq!(d.bind(&h("81fe9933d665eb834710df6658186ea76d2016d96808b441d6c053a209498520")),
                   h("18f5e9dc6da12fb1458a91992f3ba09db8307f16cb3bad43abc49c04c467dbe5"));
        assert_eq!(d.digest(), h("f127172f9fa97c7217ea9484cd3c0c1e212435b9a6c5022686d909e44f499bbd"));
    }

    #[test]
    fn core_vectors_every_tree() {
        let Some(v) = vectors() else { return };
        for t in v["trees"].as_array().unwrap() {
            let leaves: Vec<[u8; 32]> = t["leaves"].as_array().unwrap().iter().map(|x| h(x.as_str().unwrap())).collect();
            assert_eq!(hex(&fold(&leaves)), t["root"].as_str().unwrap(), "N = {}", leaves.len());
        }
    }

    /// `FLOCK_SHA_DUMP=path`: write 64 compressions' inputs and Flock's packed (z, a, b) for them, for the host harness
    /// that runs `pure_sha256_witness` (prove_chunk.cuh) as C++ and compares word for word.
    #[test]
    fn dump_sha256_witness_for_the_device_kernel_check() {
        let Some(path) = std::env::var_os("FLOCK_SHA_DUMP") else { return };
        let row: Vec<u8> = (0..1536).map(|i| (i * 131 + 7) as u8).collect();
        let msg = pos_leaf_message(&row);
        let mut comps: Vec<sh::Compression> = Vec::new();
        let mut hh = sh::SHA256_IV;
        for k in 0..64 {
            let blk = &msg[64 * (k % 25)..64 * (k % 25) + 64];
            let m = be_words(blk);
            comps.push((hh, m));
            hh = if k % 25 == 24 { sh::SHA256_IV } else { sh::sha256_compress(&hh, &m) };
        }
        comps[63] = ([0xFFFF_FFFF; 8], [0xFFFF_FFFF; 16]);
        let (z, a, b, _) = sh::generate_witness_with_ab_packed_and_lincheck(&comps, 6);
        let mut out = (comps.len() as u64).to_le_bytes().to_vec();
        out.extend(comps.iter().flat_map(|c| c.0).flat_map(|w| w.to_le_bytes()));
        out.extend(comps.iter().flat_map(|c| c.1).flat_map(|w| w.to_le_bytes()));
        for v in [&z, &a, &b] {
            out.extend(v.iter().flat_map(|x| x.lo.to_le_bytes().into_iter().chain(x.hi.to_le_bytes())));
        }
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn port_domains_match_b_ligero() {
        // archive/ligero-verify vllm_v1::tests::port_domains_match_python: fp8-ada, 4096 VUs, manifest "ab" * 32
        let s = Shape { k: 1536, word_bits: 8, y_bytes: 4 };
        let inst = json!({"dataset": "bench-instances-fp8-ada/v1", "tier": "vu-k1536-fp8-ada", "manifest_sha256": "ab".repeat(32), "range": [0, 4096]});
        let want = [("8c9e2fe698825dbd417132d1be07b802befdfb0be53a47e73c47b2b908cfcd88", "917265b14cb3fff7074a53be5887479176a1a9b0e17ac24a9a720d036811cddc")];
        let d = Domain::of(0, &s, &inst);
        assert_eq!(hex(&d.ctx), "5bdb6d775233d25d78c1c81c93450c50f398bb9955f69921ae59ddfb5e35b638");
        assert_eq!(hex(&d.program), "bb1839cf3e05e8ea421cf6f80a3da20029fb3f64038be21919df46d9d80369d6");
        assert_eq!(hex(&d.geo), "fd30fd95a7cf373c0980df7e48905c4471599e62149967812e6853254f7c64a3");
        assert_eq!(hex(&d.digest()), want[0].0);
        assert_eq!(hex(&d.bind(&[0x11u8; 32])), want[0].1);
    }
}
