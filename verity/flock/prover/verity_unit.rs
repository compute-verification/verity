//! flock-bench (verity), copied by flock-link with an explicit-profile prove/params: a netlist-defined boolean table on the union prover — one
//! tensor-core transition unit (the binary census's `unit.py`, exported by
//! `export_unit.py`) per 2^13-bit block, `C = I`, constant wire pinned at the
//! last useful column. Mirrors `sha2::Sha256HybridSetup` (single-slot union,
//! batch-major partial witness, CSC lincheck circuit).
//!
//! Instances are indices into the netlist's validated input vectors; the
//! witness is evaluated from the netlist rows (inputs, then rows in order).

use std::array::from_fn;

use flock_core::{
    lincheck::LincheckCircuit,
    pcs::{
        Commitment, PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k_or_default},
    },
    proof::{R1csClaim, R1csProofMergedLigerito},
    r1cs::{BlockR1cs, SparseBinaryMatrix},
    union::UnionInstance,
    verifier::{FlockVerifyError, verify_ligerito_union},
};
use flock_field::F128;
use flock_transcript::challenger::Challenger;

use crate::{
    prover::{UnionSlotProverInput, prove_fast_ligerito_union},
    r1cs_hashes::common::{BM_V, BmRow, build_block_r1cs_with_matrices, drive_witness_batch_major_partial},
    schedule::Registry,
};

pub const K_LOG: usize = 13;
pub const K_SKIP: usize = 6;

#[derive(Debug)]
pub struct Netlist {
    pub useful: usize,
    pub const_pos: usize,
    pub n_in: usize,
    pub a: Vec<Vec<usize>>,
    pub b: Vec<Vec<usize>>,
    a_off: Vec<u32>,
    a_col: Vec<u32>,
    b_off: Vec<u32>,
    b_col: Vec<u32>,
    /// (input bits LSB-first, expected z bits LSB-first), validated by the exporter.
    pub vectors: Vec<(Vec<u8>, Vec<u8>)>,
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn bit(v: &[u8], i: usize) -> u8 {
    (v[i >> 3] >> (i & 7)) & 1
}

impl Netlist {
    pub fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path).expect("netlist");
        let mut lines = text.lines();
        let h: Vec<usize> = lines.next().unwrap().split_whitespace().map(|t| t.parse().unwrap()).collect();
        let (useful, const_pos, n_in) = (h[0], h[1], h[2]);
        let mut a = Vec::with_capacity(useful);
        let mut b = Vec::with_capacity(useful);
        for _ in 0..useful {
            let v: Vec<usize> = lines.next().unwrap().split_whitespace().map(|t| t.parse().unwrap()).collect();
            let na = v[0];
            let nb = v[1 + na];
            a.push(v[1..1 + na].to_vec());
            b.push(v[2 + na..2 + na + nb].to_vec());
        }
        let nv: usize = lines.next().unwrap().trim().parse().unwrap();
        let vectors = (0..nv)
            .map(|_| {
                let mut it = lines.next().unwrap().split_whitespace();
                (hex(it.next().unwrap()), hex(it.next().unwrap()))
            })
            .collect();
        let csr = |rows: &Vec<Vec<usize>>| {
            let mut off = vec![0u32];
            let mut col = Vec::new();
            for r in rows {
                col.extend(r.iter().map(|&c| c as u32));
                off.push(col.len() as u32);
            }
            (off, col)
        };
        let (a_off, a_col) = csr(&a);
        let (b_off, b_col) = csr(&b);
        Self { useful, const_pos, n_in, a, b, a_off, a_col, b_off, b_col, vectors }
    }

    pub fn nnz(&self) -> usize {
        self.a_col.len() + self.b_col.len()
    }

    /// Evaluate 8 instances bit-sliced (bit j of each byte = instance j): (z, a, b) per row.
    pub fn eval8(&self, ids: [usize; BM_V], z: &mut [u8], av: &mut [u8], bv: &mut [u8]) {
        for i in 0..self.n_in {
            let mut x = 0u8;
            for (j, &id) in ids.iter().enumerate() {
                x |= bit(&self.vectors[id].0, i) << j;
            }
            z[i] = x;
            av[i] = x;
            bv[i] = x;
        }
        z[self.n_in..self.useful].fill(0);
        let c = self.const_pos;
        z[c] = 0xFF;
        av[c] = 0xFF;
        bv[c] = 0xFF;
        for i in self.n_in..self.useful {
            if i == c {
                continue;
            }
            let mut x = 0u8;
            for &k in &self.a_col[self.a_off[i] as usize..self.a_off[i + 1] as usize] {
                x ^= z[k as usize];
            }
            let mut y = 0u8;
            for &k in &self.b_col[self.b_off[i] as usize..self.b_off[i + 1] as usize] {
                y ^= z[k as usize];
            }
            av[i] = x;
            bv[i] = y;
            z[i] = x & y;
        }
    }

    /// Recompute every exported vector's z and re-check (a & b) = z; returns mismatching vectors.
    pub fn self_check(&self) -> usize {
        let u = self.useful;
        let nv = self.vectors.len();
        let (mut z, mut a, mut b) = (vec![0u8; u], vec![0u8; u], vec![0u8; u]);
        let mut bad = 0;
        for g in (0..nv).step_by(BM_V) {
            let ids: [usize; BM_V] = from_fn(|j| (g + j) % nv);
            self.eval8(ids, &mut z, &mut a, &mut b);
            for j in 0..BM_V.min(nv - g) {
                let want = &self.vectors[g + j].1;
                let ok = (0..u).all(|i| ((z[i] >> j) & 1) == bit(want, i) && (((a[i] & b[i]) >> j) & 1) == bit(want, i));
                if !ok {
                    bad += 1;
                }
            }
        }
        bad
    }

    fn build_group(&self, ids: [&u32; BM_V], rz: &mut [BmRow], ra: &mut [BmRow], rb: &mut [BmRow]) {
        thread_local! {
            static BUF: std::cell::RefCell<(Vec<u8>, Vec<u8>, Vec<u8>)> =
                const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
        }
        BUF.with(|buf| {
            let (z, a, b) = &mut *buf.borrow_mut();
            z.resize(self.useful, 0);
            a.resize(self.useful, 0);
            b.resize(self.useful, 0);
            let nv = self.vectors.len();
            self.eval8(from_fn(|j| *ids[j] as usize % nv), z, a, b);
            for (src, dst) in [(&*z, &mut *rz), (&*a, &mut *ra), (&*b, &mut *rb)] {
                for (i, &v) in src.iter().enumerate() {
                    if v == 0 {
                        continue;
                    }
                    let (w, s) = (i >> 6, i & 63);
                    for j in 0..BM_V {
                        dst[w][j] |= (((v >> j) & 1) as u64) << s;
                    }
                }
            }
        });
    }
}

impl Netlist {
    /// The unit as a plain (non-union, RowMajor) block R1CS, for `prove_ligerito` and Flock-CUDA.
    pub fn block_r1cs(&self, n_blocks_log: usize) -> BlockR1cs {
        let k = 1usize << K_LOG;
        let pad = |rows: &Vec<Vec<usize>>| {
            let mut r = rows.clone();
            r.resize(k, Vec::new());
            SparseBinaryMatrix::new(k, k, r)
        };
        build_block_r1cs_with_matrices(n_blocks_log, K_LOG, K_SKIP, self.useful, pad(&self.a), pad(&self.b), Some(self.const_pos))
    }

    /// RowMajor packed (z, a, b, z_lincheck) with every one of the 2^n_blocks_log blocks a real
    /// instance (vector id = block index mod #vectors), from per-vector blocks evaluated once.
    pub fn witness_row_major(&self, n_blocks_log: usize) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let u = self.useful;
        let nv = self.vectors.len();
        let words = (1usize << K_LOG) / 64;
        let mut pre: Vec<[Vec<u64>; 3]> = Vec::with_capacity(nv);
        let (mut z, mut a, mut b) = (vec![0u8; u], vec![0u8; u], vec![0u8; u]);
        for g in (0..nv).step_by(BM_V) {
            self.eval8(from_fn(|j| (g + j) % nv), &mut z, &mut a, &mut b);
            for j in 0..BM_V.min(nv - g) {
                let blk: [Vec<u64>; 3] = from_fn(|t| {
                    let src = [&z, &a, &b][t];
                    let mut w = vec![0u64; words];
                    for i in 0..u {
                        w[i >> 6] |= (((src[i] >> j) & 1) as u64) << (i & 63);
                    }
                    w
                });
                pre.push(blk);
            }
        }
        let ids: Vec<usize> = (0..1usize << n_blocks_log).map(|i| i % nv).collect();
        crate::r1cs_hashes::common::drive_witness_packed_and_lincheck(&ids, None, n_blocks_log, K_LOG, |&id, zw, aw, bw| {
            zw.copy_from_slice(&pre[id][0]);
            aw.copy_from_slice(&pre[id][1]);
            bw.copy_from_slice(&pre[id][2]);
        })
    }
}

/// The BLAKE3 row-leaf table (slot 0, k = 2^14) and the unit table (slot 1, k = 2^13) in ONE union
/// proof (one commitment, one opening), as `mixed::MixedSetup` does for SHA-256 + BLAKE3. The
/// glue between them (chains, relation-leaf region equality, public endpoints) is not modelled.
#[derive(Debug)]
pub struct CombinedSetup {
    pub nu: usize,
    pub net: Netlist,
    pub blake3_r1cs: BlockR1cs,
    pub unit_r1cs: BlockR1cs,
    pub registry: Registry,
}

impl CombinedSetup {
    pub fn new(net: Netlist, max_count: usize) -> Self {
        let nu = min_n_blocks_log(max_count);
        let blake3_r1cs = crate::r1cs_hashes::blake3::build_block_r1cs(nu);
        let unit_r1cs = net.block_r1cs(nu);
        blake3_r1cs.csc_lincheck_circuit();
        unit_r1cs.csc_lincheck_circuit();
        let registry = Registry::new(
            vec![
                crate::schedule::TableType::from_block_r1cs(&blake3_r1cs),
                crate::schedule::TableType::from_block_r1cs(&unit_r1cs),
            ],
            nu,
        );
        let _ = registry.digest();
        Self { nu, net, blake3_r1cs, unit_r1cs, registry }
    }

    pub fn pcs_params(&self, counts: [usize; 2]) -> PcsParams {
        self.pcs_params_with(counts, LigeritoProfile::Fast)
    }

    /// [`Self::pcs_params`] at an explicit profile (flock-link F2 pins Fast100).
    pub fn pcs_params_with(&self, counts: [usize; 2], profile: LigeritoProfile) -> PcsParams {
        let union = UnionInstance::new(&self.registry, counts.to_vec());
        let lb = embedded_initial_k_or_default(union.dense_m(), profile);
        PcsParams {
            m: union.dense_m(),
            log_inv_rate: profile.log_inv_rate(),
            log_batch_size: lb,
            profile,
            num_lanes: union.commit_lanes(lb),
            merkle_hash: Default::default(),
        }
    }

    pub fn prove<Ch: Challenger>(
        &self,
        blake3_inputs: &[crate::r1cs_hashes::blake3::Compression],
        ids: &[u32],
        challenger: &mut Ch,
    ) -> (R1csProofMergedLigerito, Commitment, R1csClaim) {
        self.prove_with(blake3_inputs, ids, LigeritoProfile::Fast, challenger)
    }

    pub fn prove_with<Ch: Challenger>(
        &self,
        blake3_inputs: &[crate::r1cs_hashes::blake3::Compression],
        ids: &[u32],
        profile: LigeritoProfile,
        challenger: &mut Ch,
    ) -> (R1csProofMergedLigerito, Commitment, R1csClaim) {
        let counts = [blake3_inputs.len(), ids.len()];
        let union = UnionInstance::new(&self.registry, counts.to_vec());
        let pcs_params = self.pcs_params_with(counts, profile);
        let (nu, net) = (self.nu, &self.net);
        let slots = vec![
            UnionSlotProverInput::in_place(
                move |dst| crate::r1cs_hashes::blake3::generate_witness_batch_major_partial_into(blake3_inputs, nu, dst),
                self.blake3_r1cs.csc_lincheck_circuit(),
            ),
            UnionSlotProverInput::in_place(
                move |dst| {
                    crate::r1cs_hashes::common::drive_witness_batch_major_partial_into(
                        ids,
                        nu,
                        K_LOG,
                        net.useful,
                        dst,
                        |g, rz, ra, rb| net.build_group(g, rz, ra, rb),
                    )
                },
                self.unit_r1cs.csc_lincheck_circuit(),
            ),
        ];
        prove_fast_ligerito_union(&union, &pcs_params, slots, challenger)
    }

    pub fn verify<Ch: Challenger>(
        &self,
        counts: [usize; 2],
        commitment: &Commitment,
        proof: &R1csProofMergedLigerito,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, FlockVerifyError> {
        let union = UnionInstance::new(&self.registry, counts.to_vec());
        let circuits: [&dyn LincheckCircuit; 2] =
            [self.blake3_r1cs.csc_lincheck_circuit(), self.unit_r1cs.csc_lincheck_circuit()];
        verify_ligerito_union(&union, &circuits, commitment, proof, &self.pcs_params(counts), challenger)
    }
}

pub fn min_n_blocks_log(n: usize) -> usize {
    n.max(8).next_power_of_two().trailing_zeros() as usize
}

#[derive(Debug)]
pub struct VerityUnitSetup {
    pub n_units: usize,
    pub net: Netlist,
    pub r1cs: BlockR1cs,
    pub registry: Registry,
    pub pcs_params: PcsParams,
}

impl VerityUnitSetup {
    pub fn new(net: Netlist, n_units: usize) -> Self {
        assert!(n_units >= 1);
        let profile = LigeritoProfile::Fast;
        let n_log = min_n_blocks_log(n_units);
        let k = 1usize << K_LOG;
        let pad = |rows: &Vec<Vec<usize>>| {
            let mut r = rows.clone();
            r.resize(k, Vec::new());
            SparseBinaryMatrix::new(k, k, r)
        };
        let r1cs = build_block_r1cs_with_matrices(
            n_log,
            K_LOG,
            K_SKIP,
            net.useful,
            pad(&net.a),
            pad(&net.b),
            Some(net.const_pos),
        );
        r1cs.csc_lincheck_circuit();
        flock_core::scratch::prewarm_prover(r1cs.m);
        let registry = Registry::new(vec![crate::schedule::TableType::from_block_r1cs(&r1cs)], n_log);
        let _ = registry.digest();
        let pcs_params = {
            let union = UnionInstance::new(&registry, vec![n_units]);
            let m = union.dense_m();
            let batch = embedded_initial_k_or_default(m, profile);
            PcsParams {
                m,
                log_inv_rate: profile.log_inv_rate(),
                log_batch_size: batch,
                profile,
                num_lanes: union.commit_lanes(batch),
                merkle_hash: Default::default(),
            }
        };
        Self { n_units, net, r1cs, registry, pcs_params }
    }

    pub fn m(&self) -> usize {
        self.r1cs.m
    }
    pub fn n_blocks_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }

    pub fn witness(&self, ids: &[u32]) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let net = &self.net;
        drive_witness_batch_major_partial(ids, self.n_blocks_log(), K_LOG, net.useful, |g, rz, ra, rb| {
            net.build_group(g, rz, ra, rb)
        })
    }

    pub fn prove_fast<Ch: Challenger>(
        &self,
        ids: &[u32],
        challenger: &mut Ch,
    ) -> (R1csProofMergedLigerito, Commitment, R1csClaim) {
        assert_eq!(ids.len(), self.n_units);
        let union = UnionInstance::new(&self.registry, vec![self.n_units]);
        let slot = UnionSlotProverInput::new(self.witness(ids), self.r1cs.csc_lincheck_circuit());
        prove_fast_ligerito_union(&union, &self.pcs_params, vec![slot], challenger)
    }

    pub fn verify<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &R1csProofMergedLigerito,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, FlockVerifyError> {
        let union = UnionInstance::new(&self.registry, vec![self.n_units]);
        let circuit = self.r1cs.csc_lincheck_circuit();
        let circs: [&dyn LincheckCircuit; 1] = [circuit];
        verify_ligerito_union(&union, &circs, commitment, proof, &self.pcs_params, challenger)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// flock-backend: the unit as a wireable circuit gate (`flock-unit-io/v1`, verity_flock.lowering)
// ---------------------------------------------------------------------------------------------------------------

/// One unit's input words: x bytes [0, 256), W bytes [256, 512) as four 128-bit words, then the accumulator word.
pub type UnitRow = [F128; 5];

/// A `flock-unit-io/v1` netlist: the lowered unit with word-aligned IO (see `verity_flock/lowering.py`).
#[derive(Debug)]
pub struct UnitNet {
    pub relation: String,
    pub useful: usize,
    pub const_pos: usize,
    pub n_in: usize,
    pub in_cols: Vec<usize>,
    pub out_cols: Vec<usize>,
    pub a: Vec<Vec<usize>>,
    pub b: Vec<Vec<usize>>,
    a_off: Vec<u32>,
    a_col: Vec<u32>,
    b_off: Vec<u32>,
    b_col: Vec<u32>,
    pub sha256: [u8; 32],
}

impl UnitNet {
    pub fn parse(text: &str) -> Self {
        use sha2::Digest;
        let sha256: [u8; 32] = sha2::Sha256::digest(text.as_bytes()).into();
        let mut lines = text.lines();
        let h: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
        assert_eq!(h[0], "flock-unit-io/v1", "not a flock-unit-io/v1 netlist");
        let n = |i: usize| -> usize { h[i].parse().unwrap() };
        let (useful, const_pos, n_in) = (n(2), n(3), n(4));
        let ni = n(5);
        let in_cols: Vec<usize> = (0..ni).map(|i| n(6 + i)).collect();
        let no = n(6 + ni);
        let out_cols: Vec<usize> = (0..no).map(|i| n(7 + ni + i)).collect();
        assert_eq!(in_cols, vec![0, 1, 2, 3, 4]);
        assert!(useful <= 1 << K_LOG && const_pos == useful - 1 && n_in == 544);
        let mut a = Vec::with_capacity(useful);
        let mut b = Vec::with_capacity(useful);
        for _ in 0..useful {
            let v: Vec<usize> = lines.next().expect("row").split_whitespace().map(|t| t.parse().unwrap()).collect();
            let na = v[0];
            let nb = v[1 + na];
            a.push(v[1..1 + na].to_vec());
            b.push(v[2 + na..2 + na + nb].to_vec());
        }
        let csr = |rows: &Vec<Vec<usize>>| {
            let mut off = vec![0u32];
            let mut col = Vec::new();
            for r in rows {
                col.extend(r.iter().map(|&c| c as u32));
                off.push(col.len() as u32);
            }
            (off, col)
        };
        let (a_off, a_col) = csr(&a);
        let (b_off, b_col) = csr(&b);
        Self { relation: h[1].to_string(), useful, const_pos, n_in, in_cols, out_cols, a, b, a_off, a_col, b_off, b_col, sha256 }
    }

    pub fn load(path: &str) -> Self {
        Self::parse(&std::fs::read_to_string(path).expect("netlist"))
    }

    pub fn block_r1cs(&self, n_blocks_log: usize) -> BlockR1cs {
        let k = 1usize << K_LOG;
        let pad = |rows: &Vec<Vec<usize>>| {
            let mut r = rows.clone();
            r.resize(k, Vec::new());
            SparseBinaryMatrix::new(k, k, r)
        };
        build_block_r1cs_with_matrices(n_blocks_log, K_LOG, K_SKIP, self.useful, pad(&self.a), pad(&self.b), Some(self.const_pos))
    }

    pub fn io_schema(&self) -> Vec<flock_core::schedule::IoWord> {
        use flock_core::schedule::IoWord;
        self.in_cols.iter().map(|&c| IoWord::input(c)).chain(self.out_cols.iter().map(|&c| IoWord::output(c))).collect()
    }

    /// Evaluate `L` instances bit-sliced (bit j of lane word = instance j): (z, a, b) per row.
    fn eval_lanes<T>(&self, input_bit: impl Fn(usize) -> T, z: &mut [T], av: &mut [T], bv: &mut [T], ones: T)
    where
        T: Copy + Default + core::ops::BitXor<Output = T> + core::ops::BitAnd<Output = T>,
    {
        for i in 0..self.n_in {
            let x = input_bit(i);
            z[i] = x;
            av[i] = x;
            bv[i] = x;
        }
        let c = self.const_pos;
        z[c] = ones;
        av[c] = ones;
        bv[c] = ones;
        for i in self.n_in..self.useful {
            if i == c {
                continue;
            }
            let mut x = T::default();
            for &k in &self.a_col[self.a_off[i] as usize..self.a_off[i + 1] as usize] {
                x = x ^ z[k as usize];
            }
            let mut y = T::default();
            for &k in &self.b_col[self.b_off[i] as usize..self.b_off[i + 1] as usize] {
                y = y ^ z[k as usize];
            }
            av[i] = x;
            bv[i] = y;
            z[i] = x & y;
        }
    }

    /// The output words of one unit, from the rows (slow: the reference view of the gate).
    pub fn eval_one(&self, row: &UnitRow) -> (Vec<F128>, bool) {
        let u = self.useful;
        let (mut z, mut a, mut b) = (vec![0u8; u], vec![0u8; u], vec![0u8; u]);
        self.eval_lanes(|i| row_bit(row, i), &mut z, &mut a, &mut b, 1u8);
        let ok = (0..u).all(|i| (a[i] & b[i]) == z[i]);
        let outs = self.out_cols.iter().map(|&c| word_of(&z, c)).collect();
        (outs, ok)
    }

    fn build_group_rows(&self, rows: [&UnitRow; BM_V], rz: &mut [BmRow], ra: &mut [BmRow], rb: &mut [BmRow]) {
        thread_local! {
            static BUF: std::cell::RefCell<(Vec<u8>, Vec<u8>, Vec<u8>)> =
                const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
        }
        BUF.with(|buf| {
            let (z, a, b) = &mut *buf.borrow_mut();
            z.resize(self.useful, 0);
            a.resize(self.useful, 0);
            b.resize(self.useful, 0);
            self.eval_lanes(
                |i| {
                    let mut x = 0u8;
                    for (j, r) in rows.iter().enumerate() {
                        x |= row_bit(r, i) << j;
                    }
                    x
                },
                z,
                a,
                b,
                0xFF,
            );
            for (src, dst) in [(&*z, &mut *rz), (&*a, &mut *ra), (&*b, &mut *rb)] {
                for (i, &v) in src.iter().enumerate() {
                    if v == 0 {
                        continue;
                    }
                    let (w, s) = (i >> 6, i & 63);
                    for j in 0..BM_V {
                        dst[w][j] |= (((v >> j) & 1) as u64) << s;
                    }
                }
            }
        });
    }

    /// The unit slot's union witness from its rows (instantiation order).
    pub fn witness_rows_into(&self, rows: &[UnitRow], n_blocks_log: usize, dst: flock_core::union::SlotWitnessDest<'_>) -> Vec<u8> {
        crate::r1cs_hashes::common::drive_witness_batch_major_partial_into(rows, n_blocks_log, K_LOG, self.useful, dst, |g, rz, ra, rb| {
            self.build_group_rows(g, rz, ra, rb)
        })
    }
}

fn row_bit(row: &UnitRow, i: usize) -> u8 {
    let w = row[i >> 7];
    let j = i & 127;
    (if j < 64 { w.lo >> j } else { w.hi >> (j - 64) } & 1) as u8
}

fn word_of(z: &[u8], col: usize) -> F128 {
    let (mut lo, mut hi) = (0u64, 0u64);
    for j in 0..64 {
        lo |= ((z[col * 128 + j] & 1) as u64) << j;
        hi |= ((z[col * 128 + 64 + j] & 1) as u64) << j;
    }
    F128::new(lo, hi)
}

/// The unit as a circuit gate: inputs x0, x1, w0, w1, c_in; outputs c_out (and y16 for an epilogue relation).
pub struct UnitGate {
    pub net: std::sync::Arc<UnitNet>,
    pub nu: usize,
}

impl flock_core::circuit::builder::GateType for UnitGate {
    type Row = UnitRow;
    type Hint = ();
    fn table(&self) -> crate::schedule::TableType {
        crate::schedule::TableType::from_block_r1cs(&self.net.block_r1cs(self.nu)).with_io_schema(self.net.io_schema())
    }
    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> UnitRow {
        let row: UnitRow = from_fn(|i| inputs[i]);
        let (outs, _) = self.net.eval_one(&row);
        outputs.extend_from_slice(&outs);
        row
    }
    fn witness(&self, _rows: &[UnitRow], _nu: usize) -> flock_core::circuit::builder::SlotWitness {
        flock_core::circuit::builder::SlotWitness::DeferredToRows
    }
}
