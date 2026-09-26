//! The BLAKE3 chunk chain as ONE block R1CS (verity flock-gpu-link, route (a) on Flock-CUDA).
//!
//! Every R1CS block is one BLAKE3 chunk: 16 sub-blocks of Flock's compression circuit (`k_log` 14 each, so the block
//! is `k_log` 18), and block `b` is the `b`-th chunk of the operand rows (x rows then W columns, 3 chunks per 1536-word
//! row). The block matrix is `I_16 ⊗ A_comp + Δ` (and the same for B), where Δ replaces:
//! - the chaining input of sub-block `j ≥ 1` with a copy of `out_lo` of sub-block `j − 1`;
//! - `block_len` and `flags` of every sub-block with constants (64; KEYED_HASH, CHUNK_START at 0, CHUNK_END at 15);
//! - the constant slot of sub-block `j ≥ 1` with a copy of sub-block 0's, the one column the lincheck pins.
//!
//! For an honest witness `A·z` and `B·z` equal the independent-compression circuit's, so Flock's witness generators
//! (CPU and the CUDA kernel) are reused unchanged over `16 · 2^nbl` compressions. The lincheck fold stays one
//! compression's: `eq_inner` over 18 bits is a tensor of 4 sub-block bits and 14 in-block bits, so the fold is the
//! compression circuit's fold, scaled per sub-block, plus the sparse Δ.
//!
//! Endpoints are public-value claims, each opened at two points drawn after the commitment: the key (the first
//! sub-block's chaining input), the counter (every sub-block's 64 counter bits) and the chunk's output chaining value
//! (the last sub-block's `out_lo`). The verifier knows the key and counter of every chunk and checks the chaining
//! values, which the prover sends in `Commit`, natively against the leaf digests. Messages are linked: the linked
//! position `chunk · 2^13 + j · 512 + u` is Flock bit `chunk · 2^18 + j · 2^14 + 512 + u`, a coordinate map onto the
//! prime side's `((port·vus + v)·K + word)·16 + i` because a row is exactly 3 · 2^13 positions.
//!
//! Chunks past the last real one (the block count is a power of two) are dummy chunks of zero messages under a zero
//! key and counter 0; the link forces their messages to zero (positions ≥ N are zero on the prime side).

use std::{array::from_fn, sync::OnceLock};

use flock_core::{
    lincheck::{LincheckCircuit, pack_z_lincheck_from_packed},
    proof::eq_skip_weights,
    r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
    verifier::ExtraClaim,
};
use flock_field::F128;
use flock_hash::blake3_compress;
use flock_prover::r1cs_hashes::blake3 as b3;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

pub const SUB_LOG: usize = b3::K_LOG;
pub const SUBS: usize = 16;
pub const K_LOG: usize = SUB_LOG + 4;
/// Sub-block 0's constant slot: the column the lincheck pins to 1.
pub const PIN: usize = b3::Z_CONST_POS;
pub const CHUNK_START: u32 = 1;
pub const CHUNK_END: u32 = 2;
pub const PARENT: u32 = 4;
pub const ROOT: u32 = 8;
pub const KEYED_HASH: u32 = 16;
/// Linked bits per chunk (1024 message bytes).
pub const CHUNK_LINK_LOG: usize = 13;
pub const KEY_PREFIX: &[u8] = b"verity/blake3-leaf/v1/";
pub const DIGEST_TAG: &[u8] = b"verity/flock-chunk-block/v1";

pub fn row_key(role: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[..KEY_PREFIX.len()].copy_from_slice(KEY_PREFIX);
    k[KEY_PREFIX.len()..KEY_PREFIX.len() + role.len()].copy_from_slice(role);
    k
}

pub fn key_words(k: &[u8; 32]) -> [u32; 8] {
    from_fn(|i| u32::from_le_bytes(k[4 * i..4 * i + 4].try_into().unwrap()))
}

pub fn log2ceil(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

pub fn flags_of(j: usize) -> u32 {
    KEYED_HASH | if j == 0 { CHUNK_START } else { 0 } | if j == SUBS - 1 { CHUNK_END } else { 0 }
}

/// Flock's BLAKE3 compression circuit (one 2^14 block), the sub-block of every chunk block.
pub fn comp() -> &'static BlockR1cs {
    static C: OnceLock<BlockR1cs> = OnceLock::new();
    C.get_or_init(|| b3::build_block_r1cs(3))
}

/// Δ as the symmetric difference of the old and new rows: `(row, col)` pairs of `A` and of `B`, in block coordinates.
pub fn delta() -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let c = comp();
    let (a, b) = (&c.a_0.rows, &c.b_0.rows);
    let z = b3::Z_CONST_POS;
    let input = |r: usize| {
        assert_eq!(a[r].as_slice(), [r], "compression row {r} is not an input row");
        assert_eq!(b[r].as_slice(), [z], "compression row {r} is not an input row");
    };
    assert_eq!(a[z].as_slice(), [z]);
    assert_eq!(b[z].as_slice(), [z]);
    let (mut da, mut db) = (Vec::new(), Vec::new());
    for j in 0..SUBS {
        let g = |col: usize| ((j << SUB_LOG) + col) as u32;
        if j > 0 {
            for d in [&mut da, &mut db] {
                d.push((g(z), g(z)));
                d.push((g(z), PIN as u32));
            }
            for r in 0..256 {
                input(b3::CV_BASE + r);
                da.push((g(b3::CV_BASE + r), g(b3::CV_BASE + r)));
                da.push((g(b3::CV_BASE + r), (((j - 1) << SUB_LOG) + b3::OUT_LO_BASE + r) as u32));
            }
        }
        for (base, val) in [(b3::BLEN_BASE, 64u32), (b3::FLAGS_BASE, flags_of(j))] {
            for bit in 0..32 {
                let r = base + bit;
                input(r);
                da.push((g(r), g(r)));
                if (val >> bit) & 1 == 1 {
                    da.push((g(r), g(z)));
                }
            }
        }
    }
    (da, db)
}

/// The chunk block's lincheck circuit: the compression circuit's fold per sub-block, plus Δ.
pub struct ChunkCircuit {
    pub da: Vec<(u32, u32)>,
    pub db: Vec<(u32, u32)>,
}

impl LincheckCircuit for ChunkCircuit {
    fn n_cols(&self) -> usize {
        1 << K_LOG
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        let sub = 1usize << SUB_LOG;
        assert_eq!(eq_inner.len(), 1 << K_LOG);
        // eq_inner[j·2^14 + c] = e_hi(j)·e_lo(c); any nonzero entry fixes the per-sub-block ratios.
        let at = eq_inner.iter().position(|e| *e != F128::ZERO).expect("eq tensor is zero");
        let (j0, c0) = (at >> SUB_LOG, at & (sub - 1));
        let base = comp().csc_lincheck_circuit().fold_alpha_batched(alpha, &eq_inner[j0 << SUB_LOG..(j0 + 1) << SUB_LOG]);
        let inv = eq_inner[at].inv();
        let mut out = vec![F128::ZERO; 1 << K_LOG];
        out.par_chunks_mut(sub).enumerate().for_each(|(j, o)| {
            let s = eq_inner[(j << SUB_LOG) | c0] * inv;
            for (x, y) in o.iter_mut().zip(&base) {
                *x = s * *y;
            }
        });
        for &(i, c) in &self.da {
            out[c as usize] += alpha * eq_inner[i as usize];
        }
        for &(i, c) in &self.db {
            out[c as usize] += eq_inner[i as usize];
        }
        out
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(PIN)
    }
}

pub fn pack4(w: &[u32]) -> F128 {
    F128::new(w[0] as u64 | ((w[1] as u64) << 32), w[2] as u64 | ((w[3] as u64) << 32))
}

/// A public region of every block: which block columns it covers (a subcube: `free` low column bits over `base`,
/// optionally every sub-block) and the known value bits of each block.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Region {
    /// The key: sub-block 0's chaining input, cols [0, 256).
    Key,
    /// The 64-bit counter of every sub-block, cols [1024, 1088).
    Counter,
    /// The chunk's output chaining value: sub-block 15's out_lo, cols [256, 512).
    Cv,
}

pub const REGIONS: [Region; 3] = [Region::Key, Region::Counter, Region::Cv];

impl Region {
    fn free_bits(self) -> usize {
        match self {
            Region::Counter => 6,
            _ => 8,
        }
    }
}

pub struct ChunkStmt {
    pub vus: usize,
    pub k: usize,
    pub leaves: usize,
    pub cpl: usize,
    pub chunks: usize,
    pub nbl: usize,
    pub m: usize,
    pub m_link: usize,
    pub keys: [[u32; 8]; 2],
    pub dummy_cv: [u32; 8],
    pub circuit: ChunkCircuit,
    pub r1cs: BlockR1cs,
    pub digest: [u8; 32],
}

impl ChunkStmt {
    pub fn new(vus: usize, k: usize) -> Self {
        let leaves = 2 * vus;
        assert!((2 * k) % 1024 == 0 && 2 * k >= 2048, "a leaf is at least two whole BLAKE3 chunks");
        let cpl = 2 * k / 1024;
        let chunks = leaves * cpl;
        let nbl = log2ceil(chunks).max(3);
        let m = K_LOG + nbl;
        let keys = [key_words(&row_key(b"x")), key_words(&row_key(b"w"))];
        let mut cv = [0u32; 8];
        for j in 0..SUBS {
            cv = blake3_compress(&cv, &[0u32; 16], 0, 64, flags_of(j))[..8].try_into().unwrap();
        }
        let (da, db) = delta();
        let mut h = Sha256::new();
        h.update(DIGEST_TAG);
        h.update(comp().statement_digest());
        for x in [vus, k, leaves, cpl, chunks, nbl, m, K_LOG, SUBS] {
            h.update((x as u64).to_le_bytes());
        }
        for kw in keys.iter().flatten() {
            h.update(kw.to_le_bytes());
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
        ChunkStmt { vus, k, leaves, cpl, chunks, nbl, m, m_link: CHUNK_LINK_LOG + nbl, keys, dummy_cv: cv, circuit: ChunkCircuit { da, db }, r1cs, digest }
    }

    pub fn blocks(&self) -> usize {
        1 << self.nbl
    }

    /// Chunk `b`'s key and counter (dummy chunks: zero key, counter 0).
    pub fn key_counter(&self, b: usize) -> ([u32; 8], u64) {
        if b < self.chunks {
            (self.keys[usize::from(b / self.cpl >= self.vus)], (b % self.cpl) as u64)
        } else {
            ([0; 8], 0)
        }
    }

    /// The honest compressions of `bytes` (every chunk of every row, then the dummy chunks), and each real chunk's
    /// output chaining value.
    pub fn compressions(&self, bytes: &[u8]) -> (Vec<b3::Compression>, Vec<[u32; 8]>) {
        assert_eq!(bytes.len(), self.chunks * 1024);
        let per: Vec<(Vec<b3::Compression>, [u32; 8])> = (0..self.blocks())
            .into_par_iter()
            .map(|b| {
                let (key, ctr) = self.key_counter(b);
                let mut cv = key;
                let mut out = Vec::with_capacity(SUBS);
                for j in 0..SUBS {
                    let m: [u32; 16] = if b < self.chunks {
                        let at = b * 1024 + 64 * j;
                        from_fn(|i| u32::from_le_bytes(bytes[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
                    } else {
                        [0; 16]
                    };
                    out.push((cv, m, ctr, 64, flags_of(j)));
                    cv = blake3_compress(&cv, &m, ctr, 64, flags_of(j))[..8].try_into().unwrap();
                }
                (out, cv)
            })
            .collect();
        let cvs = per[..self.chunks].iter().map(|p| p.1).collect();
        (per.into_iter().flat_map(|p| p.0).collect(), cvs)
    }

    /// `(z, a, b, z_lincheck)` for the given compressions (the prover's; possibly dishonest).
    pub fn witness(&self, comps: &[b3::Compression]) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        assert_eq!(comps.len(), SUBS * self.blocks());
        let (z, a, b) = b3::generate_witness_with_ab_packed(comps, self.nbl + 4);
        let zl = pack_z_lincheck_from_packed(&z, self.m, K_LOG);
        (z, a, b, zl)
    }

    /// A link point (m_link coordinates over the linked positions) as a point of the block witness.
    pub fn embed(&self, r: &[F128]) -> Vec<F128> {
        assert_eq!(r.len(), self.m_link);
        let mut p = r[..9].to_vec();
        p.extend_from_slice(&[F128::ONE, F128::ZERO, F128::ZERO, F128::ZERO, F128::ZERO]);
        p.extend_from_slice(&r[9..]);
        assert_eq!(p.len(), self.m);
        p
    }

    /// The public claim point of `region` derived from link point `r` (a uniform point of the region's subcube:
    /// its in-word bits from `r[..8]`, its sub-block bits from `r[9..13]`, its block bits from `r[13..]`).
    pub fn region_point(&self, region: Region, r: &[F128]) -> Vec<F128> {
        let (o, z) = (F128::ONE, F128::ZERO);
        let blk = &r[CHUNK_LINK_LOG..];
        let mut p: Vec<F128> = match region {
            Region::Key => [&r[..8], &[z; 6][..], &[z; 4][..]].concat(),
            Region::Counter => [&r[..6], &[z, z, z, z, o, z, z, z][..], &r[9..13]].concat(),
            Region::Cv => [&r[..8], &[o, z, z, z, z, z][..], &[o; 4][..]].concat(),
        };
        p.extend_from_slice(blk);
        assert_eq!(p.len(), self.m);
        p
    }

    /// The region's value at its point: `Σ_b eq(p_blk, b) · Σ_t eq(p_bits, t) · bit_t(value of block b)`. The
    /// counter region spans every sub-block, whose eq weights sum to 1.
    pub fn region_value(&self, region: Region, r: &[F128], cvs: &[[u32; 8]]) -> F128 {
        let nb = region.free_bits();
        let wt = eq_skip_weights(&r[..nb]);
        let eb = eq_skip_weights(&r[CHUNK_LINK_LOG..]);
        // byte tables: t[i][x] = Σ over set bits j of x of wt[8 i + j]
        let nbytes = (1usize << nb) / 8;
        let mut t = vec![[F128::ZERO; 256]; nbytes];
        for (i, tb) in t.iter_mut().enumerate() {
            for x in 1..256usize {
                let rest = x & (x - 1);
                tb[x] = tb[rest] + wt[8 * i + (x ^ rest).trailing_zeros() as usize];
            }
        }
        let val = |bytes: &[u8]| -> F128 { bytes.iter().enumerate().fold(F128::ZERO, |a, (i, &x)| a + t[i][x as usize]) };
        let words8 = |w: &[u32; 8]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
        (0..self.blocks())
            .into_par_iter()
            .map(|b| {
                let (key, ctr) = self.key_counter(b);
                let v = match region {
                    Region::Key => val(&words8(&key)),
                    Region::Counter => val(&ctr.to_le_bytes()),
                    Region::Cv => val(&words8(if b < self.chunks { &cvs[b] } else { &self.dummy_cv })),
                };
                eb[b] * v
            })
            .reduce(|| F128::ZERO, |a, b| a + b)
    }

    /// The prover's extra claims' coordinates past the skip block, in the order the verifier opens them: the link
    /// points, then each region at each point.
    pub fn extra_points(&self, pts: &[Vec<F128>]) -> Vec<Vec<F128>> {
        let mut v: Vec<Vec<F128>> = pts.iter().map(|r| self.embed(r)).collect();
        for reg in REGIONS {
            v.extend(pts.iter().map(|r| self.region_point(reg, r)));
        }
        v
    }

    /// The verifier's extra claims: `ẑ(embed(r_k)) = y_k`, then every region's public value at both points.
    pub fn extra_claims(&self, pts: &[Vec<F128>], y: &[F128], cvs: &[[u32; 8]]) -> Vec<ExtraClaim> {
        let mut values: Vec<F128> = y.to_vec();
        for reg in REGIONS {
            values.extend(pts.iter().map(|r| self.region_value(reg, r, cvs)));
        }
        self.extra_points(pts)
            .into_iter()
            .zip(values)
            .map(|(p, value)| ExtraClaim { value, skip_weights: eq_skip_weights(&p[..6]), x_full: p[6..].to_vec() })
            .collect()
    }
}

pub fn parent_cv(key: &[u32; 8], l: &[u32; 8], r: &[u32; 8], root: bool) -> [u32; 8] {
    let mut m = [0u32; 16];
    m[..8].copy_from_slice(l);
    m[8..].copy_from_slice(r);
    blake3_compress(key, &m, 0, 64, PARENT | KEYED_HASH | if root { ROOT } else { 0 })[..8].try_into().unwrap()
}

/// BLAKE3's keyed tree over at least two chunk chaining values.
pub fn tree(key: &[u32; 8], cvs: &[[u32; 8]], root: bool) -> [u32; 8] {
    if cvs.len() == 1 {
        return cvs[0];
    }
    let left = 1usize << (usize::BITS - 1 - (cvs.len() - 1).leading_zeros());
    parent_cv(key, &tree(key, &cvs[..left], false), &tree(key, &cvs[left..], false), root)
}

pub fn digest_of(key: &[u32; 8], cvs: &[[u32; 8]]) -> [u8; 32] {
    let mut d = [0u8; 32];
    for (i, w) in tree(key, cvs, true).iter().enumerate() {
        d[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    d
}
