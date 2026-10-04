//! rec-holo's kill experiment for route P (a hidden wiring permutation over a public table of gates) on upstream Flock
//! b684b12. 78-rec-holo-kill.sh copies this file to crates/flock-prover/examples and appends rec_holo_gkr_stream.rs to
//! flock-core's product_gkr.rs. Each mode is its own process and prints one JSON line per measurement:
//! - `build DIR NETS full|sampled`: compose a circuit in which no unit repeats from the netlists NETS/*.rhn
//!   (rec_holo_netlist.py), compile it as route P's gate table and wiring σ and as C-Flock rows, check that the two
//!   compute the same outputs and that w is constant on σ's cycles, and write DIR/*.bin and DIR/meta.json.
//! - `gate DIR`: C-Flock's proof of route P's gate table (A = I ⊗ A_gate), with the opening's two claims (verified) and
//!   with the three more that the wiring argument's ŵ(ρ) adds.
//! - `gkr DIR dense|stream UNSTORED DOUBLE`: the wiring product GKR on (w, σ): upstream's `prove_batched`, or the
//!   streaming prover; proof digest, verified.
//! - `sigma DIR [UNIT_LOG]`: commit σ as a binary polynomial (registration) and open it at one point (each audit); with
//!   UNIT_LOG, only unit 0's σ_u.
//! - `base DIR COPIES`: C-Flock's proof of the same circuit's rows as one block A_0, tiled COPIES ≥ 8 times (upstream's
//!   lincheck needs 8 blocks), verified.
//! - `selftest`: the streaming GKR against `prove_batched` (same proof bytes) at small sizes.
//!
//! env: RUNS=3 (timed runs after a first, cold run that also gives peak memory), FILL=0.9 BASE_KLOG=23 P_FRESH=0.25
//! SEED=1 (build, full), UNITS=8 UNIT_LOG=22 (build, sampled), GATE_KLOG=16.
use std::{
    collections::HashMap,
    env::{args, var},
    fs,
    hash::{BuildHasherDefault, Hasher},
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Instant,
};

use flock_core::{
    challenger::FsChallenger,
    field::F128,
    lincheck::{QuirkyPoint, pack_z_lincheck_from_packed},
    pcs::{
        self, PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k_or_default},
        open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding,
    },
    product_gkr::{prove_batched, rec_holo_prove_batched_stream, verify_batched},
    r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
    test_rng::Rng,
    verifier::verify_ligerito,
    zerocheck::PaddingSpec,
};
use flock_prover::{
    init_perf_thread_pool,
    prover::{prove_fast_core, prove_fast_ligerito_from_witness},
};
use rayon::prelude::*;

const NONE: u32 = u32::MAX;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn status_mb(key: &str) -> f64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map_or(0.0, |kb| kb / 1024.0)
}

/// Resets VmHWM to the current RSS; returns the RSS (MB) it was reset to, or a negative value if the kernel refused.
fn reset_peak() -> f64 {
    match fs::write("/proc/self/clear_refs", "5") {
        Ok(()) => status_mb("VmRSS:"),
        Err(_) => -1.0,
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn runs() -> usize {
    env_or("RUNS", 3)
}

fn params(m: usize) -> PcsParams {
    let profile = LigeritoProfile::Fast;
    PcsParams {
        m,
        log_inv_rate: profile.log_inv_rate(),
        log_batch_size: embedded_initial_k_or_default(m, profile),
        profile,
        num_lanes: None,
        merkle_hash: Default::default(),
    }
}

fn block_r1cs(m: usize, k_log: usize, a_0: SparseBinaryMatrix, b_0: SparseBinaryMatrix) -> BlockR1cs {
    let k = 1usize << k_log;
    BlockR1cs {
        m,
        k_log,
        k_skip: 6,
        useful_bits: k,
        a_0,
        b_0,
        c_0: SparseBinaryMatrix::new(k, k, (0..k).map(|i| vec![i]).collect()),
        layout: WitnessLayout::RowMajor,
        const_pin: Some(0),
        digest_cache: OnceLock::new(),
        csc_cache: OnceLock::new(),
    }
}

fn digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex()[..16].to_string()
}

fn bits_to_packed(z: &[u64]) -> Vec<F128> {
    z.par_chunks(2).map(|c| F128::new(c[0], c[1])).collect()
}

fn read_u64s(p: &Path) -> Vec<u64> {
    fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn read_u32s(p: &Path) -> Vec<u32> {
    fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn write_u64s(p: &Path, v: &[u64]) {
    fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

fn write_u32s(p: &Path, v: &[u32]) {
    fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

/// A tiny multiplicative hasher for the per-instance (operand, operand) → row map.
#[derive(Default)]
struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, b: &[u8]) {
        for &x in b {
            self.write_u32(x as u32);
        }
    }
    fn write_u32(&mut self, x: u32) {
        self.0 = (self.0.rotate_left(5) ^ x as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

// ---------------------------------------------------------------------------
// Netlists (rec_holo_netlist.py's .rhn: all u32 LE)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Form {
    c: bool,
    start: u32,
    len: u32,
}

struct Unit {
    name: String,
    cat: usize,
    n_in: usize,
    gates: Vec<[u32; 3]>,
    outputs: Vec<u32>,
    rows: Vec<(Form, Form)>,
    out_forms: Vec<Form>,
    ids: Vec<u32>,
    rp_rows: usize,
}

const CATS: [(&str, f64); 4] = [("sha", 0.30), ("tc", 0.25), ("fp", 0.30), ("mufu", 0.15)];

fn category(name: &str) -> usize {
    if name.starts_with("sha") {
        0
    } else if name.starts_with("tc_") {
        1
    } else if name.starts_with("mufu") {
        3
    } else {
        2
    }
}

fn load_unit(p: &Path) -> Unit {
    let u = read_u32s(p);
    assert_eq!(u[0], 0x524E_4C31, "{}: not a .rhn file", p.display());
    let (n_in, n_gates, n_out, n_and) = (u[1] as usize, u[2] as usize, u[3] as usize, u[4] as usize);
    let mut at = 5;
    let gates: Vec<[u32; 3]> = (0..n_gates).map(|j| [u[at + 3 * j], u[at + 3 * j + 1], u[at + 3 * j + 2]]).collect();
    at += 3 * n_gates;
    let outputs = u[at..at + n_out].to_vec();
    at += n_out;
    let mut ids = Vec::new();
    let mut form = |at: &mut usize| {
        let h = u[*at];
        let len = h >> 1;
        let f = Form { c: h & 1 == 1, start: ids.len() as u32, len };
        ids.extend_from_slice(&u[*at + 1..*at + 1 + len as usize]);
        *at += 1 + len as usize;
        f
    };
    let rows: Vec<(Form, Form)> = (0..n_and).map(|_| (form(&mut at), form(&mut at))).collect();
    let out_forms: Vec<Form> = (0..n_out).map(|_| form(&mut at)).collect();
    assert_eq!(at, u.len(), "{}: trailing data", p.display());
    let name = p.file_stem().unwrap().to_string_lossy().into_owned();
    Unit { cat: category(&name), name, n_in, gates, outputs, rows, out_forms, ids, rp_rows: 0 }
}

// ---------------------------------------------------------------------------
// Route P: gate rows (l, r, a = l·r, x = l + r), cells 4g..4g+3, z = (l, r, a, 0), w = (l, r, a, x)
// ---------------------------------------------------------------------------

struct Rp {
    sigma: Vec<u32>,
    last: Vec<u32>,
    z: Vec<u64>,
    w: Vec<u64>,
    rows: usize,
    rpb: usize,
    c0: u32,
    c1: u32,
    io_rows: usize,
    const_rows: usize,
}

#[derive(Clone, Copy)]
enum Src {
    Fresh(bool),
    Out(usize),
}

struct Wire {
    net: u32,
    val: bool,
    form: Form,
}

impl Rp {
    fn new(gate_klog: usize) -> Self {
        Rp {
            sigma: Vec::new(),
            last: Vec::new(),
            z: Vec::new(),
            w: Vec::new(),
            rows: 0,
            rpb: 1 << (gate_klog - 2),
            c0: NONE,
            c1: NONE,
            io_rows: 0,
            const_rows: 0,
        }
    }

    fn set(v: &mut Vec<u64>, i: usize, b: bool) {
        if v.len() <= i >> 6 {
            v.resize((i >> 6) + 1, 0);
        }
        if b {
            v[i >> 6] |= 1 << (i & 63);
        }
    }

    /// Appends a row with input values (l, r) and returns its index; a block's row 0 is the constant row (l pinned to 1).
    fn raw_row(&mut self, l: bool, r: bool) -> usize {
        let g = self.rows;
        self.rows += 1;
        for t in 0..4 {
            self.sigma.push((4 * g + t) as u32);
            self.last.push(NONE);
        }
        let c = 4 * g;
        for (t, b) in [l, r, l & r].into_iter().enumerate() {
            Self::set(&mut self.z, c + t, b);
            Self::set(&mut self.w, c + t, b);
        }
        Self::set(&mut self.z, c + 3, false);
        Self::set(&mut self.w, c + 3, l ^ r);
        g
    }

    fn row(&mut self, l: bool, r: bool) -> usize {
        if self.rows % self.rpb == 0 {
            self.raw_row(true, true);
            self.const_rows += 1;
        }
        self.raw_row(l, r)
    }

    fn head(&mut self, cell: usize) -> u32 {
        self.last[cell] = cell as u32;
        cell as u32
    }

    fn join(&mut self, net: u32, cell: usize) {
        let l = self.last[net as usize] as usize;
        self.sigma[l] = cell as u32;
        self.last[net as usize] = cell as u32;
    }

    /// Pads to a multiple of `align` rows, then opens a unit: its first block's constant row carries the unit's
    /// constant-1 net (l, pinned, and r) and constant-0 net (x).
    fn start_unit(&mut self, align: usize) {
        self.pad_to(self.rows.next_multiple_of(align));
        let g = self.raw_row(true, true);
        self.const_rows += 1;
        self.c1 = self.head(4 * g);
        self.join(self.c1, 4 * g + 1);
        self.c0 = self.head(4 * g + 3);
    }

    fn pad_to(&mut self, rows: usize) {
        while self.rows < rows {
            if self.rows % self.rpb == 0 {
                self.raw_row(true, true);
                self.const_rows += 1;
            } else {
                self.raw_row(false, false);
            }
        }
    }

    /// The output cell of the row computing `which` (2: a, 3: x) of nets (p, q), reusing a row of this instance on the
    /// same pair.
    fn gate(&mut self, seen: &mut FxMap<(u32, u32), (usize, u8)>, p: (u32, bool), q: (u32, bool), which: usize) -> u32 {
        let (p, q) = if p.0 <= q.0 { (p, q) } else { (q, p) };
        let bit = 1u8 << which;
        if let Some(&(g, flags)) = seen.get(&(p.0, q.0)) {
            if flags & bit == 0 {
                seen.insert((p.0, q.0), (g, flags | bit));
                return self.head(4 * g + which);
            }
            return (4 * g + which) as u32;
        }
        let g = self.row(p.1, q.1);
        self.join(p.0, 4 * g);
        self.join(q.0, 4 * g + 1);
        seen.insert((p.0, q.0), (g, bit));
        self.head(4 * g + which)
    }

    /// Compiles one instance; `ins` are the input nets and values. Returns the output nets and values.
    fn instance(&mut self, t: &Unit, ins: &[(u32, bool)]) -> Vec<(u32, bool)> {
        let (c0, c1) = (self.c0, self.c1);
        let mut net = Vec::with_capacity(2 + t.n_in + t.gates.len());
        net.push((c0, false));
        net.push((c1, true));
        net.extend_from_slice(ins);
        let mut seen: FxMap<(u32, u32), (usize, u8)> = FxMap::default();
        seen.reserve(t.gates.len());
        for g in &t.gates {
            let p = net[g[1] as usize];
            let out = match g[0] {
                0 => {
                    let q = net[g[2] as usize];
                    let v = p.1 & q.1;
                    if p.0 == c0 || q.0 == c0 {
                        (c0, false)
                    } else if p.0 == c1 {
                        q
                    } else if q.0 == c1 || p.0 == q.0 {
                        p
                    } else {
                        (self.gate(&mut seen, p, q, 2), v)
                    }
                }
                1 => {
                    let q = net[g[2] as usize];
                    let v = p.1 ^ q.1;
                    if p.0 == c0 {
                        q
                    } else if q.0 == c0 {
                        p
                    } else if p.0 == q.0 {
                        (c0, false)
                    } else {
                        (self.gate(&mut seen, p, q, 3), v)
                    }
                }
                _ => {
                    if p.0 == c0 {
                        (c1, true)
                    } else if p.0 == c1 {
                        (c0, false)
                    } else {
                        (self.gate(&mut seen, p, (c1, true), 3), !p.1)
                    }
                }
            };
            net.push(out);
        }
        t.outputs.iter().map(|&o| net[o as usize]).collect()
    }

    /// Fresh input bits as cells of I/O rows (two a row), each the head of its own net.
    fn fresh(&mut self, vals: &[bool]) -> Vec<(u32, bool)> {
        let mut out = Vec::with_capacity(vals.len());
        for pair in vals.chunks(2) {
            let (l, r) = (pair[0], pair.get(1).copied().unwrap_or(false));
            let g = self.row(l, r);
            self.io_rows += 1;
            out.push((self.head(4 * g), l));
            if pair.len() == 2 {
                out.push((self.head(4 * g + 1), r));
            }
        }
        out
    }

    fn close(&mut self) {
        for c in 0..self.last.len() {
            let l = self.last[c];
            if l != NONE {
                self.sigma[l as usize] = c as u32;
            }
        }
    }

    fn check(&self) {
        let n = self.sigma.len();
        let bit = |v: &[u64], i: usize| (v[i >> 6] >> (i & 63)) & 1;
        let mut hit = vec![0u64; n.div_ceil(64)];
        for i in 0..n {
            let s = self.sigma[i] as usize;
            assert!(hit[s >> 6] >> (s & 63) & 1 == 0, "σ is not a permutation (cell {s} hit twice)");
            hit[s >> 6] |= 1 << (s & 63);
            assert_eq!(bit(&self.w, i), bit(&self.w, s), "w is not constant on σ's cycle through cell {i}");
        }
        for g in 0..n / 4 {
            let c = 4 * g;
            assert_eq!(bit(&self.z, c + 2), bit(&self.z, c) & bit(&self.z, c + 1));
            assert_eq!(bit(&self.w, c + 3), bit(&self.z, c) ^ bit(&self.z, c + 1));
            assert_eq!(bit(&self.z, c + 3), 0);
            if g % self.rpb == 0 {
                assert_eq!(bit(&self.z, c), 1, "block row 0's l is the pinned constant");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// C-Flock rows: z = (1, unit inputs, ANDs, …) as one block; row i's (A_i, B_i) gives z_i = A_i·z ∘ B_i·z
// ---------------------------------------------------------------------------

struct Base {
    a_ptr: Vec<u32>,
    a_idx: Vec<u32>,
    b_ptr: Vec<u32>,
    b_idx: Vec<u32>,
    z: Vec<u64>,
    ids: Vec<u32>,
    copy_rows: usize,
    fresh_rows: usize,
}

impl Base {
    fn new() -> Self {
        let mut b = Base {
            a_ptr: vec![0],
            a_idx: Vec::new(),
            b_ptr: vec![0],
            b_idx: Vec::new(),
            z: Vec::new(),
            ids: Vec::new(),
            copy_rows: 0,
            fresh_rows: 0,
        };
        b.push(&[0], &[0], true);
        b
    }

    fn rows(&self) -> usize {
        self.a_ptr.len() - 1
    }

    fn bit(&self, i: u32) -> bool {
        (self.z[i as usize >> 6] >> (i & 63)) & 1 == 1
    }

    fn push(&mut self, a: &[u32], b: &[u32], v: bool) {
        let i = self.rows();
        self.a_idx.extend_from_slice(a);
        self.a_ptr.push(self.a_idx.len() as u32);
        self.b_idx.extend_from_slice(b);
        self.b_ptr.push(self.b_idx.len() as u32);
        Rp::set(&mut self.z, i, v);
    }

    /// A form of `t` over this instance's columns (inputs at `base..`, then the ANDs), with the constant as column 0.
    fn map(&self, t: &Unit, f: Form, base: u32, out: &mut Vec<u32>) -> bool {
        out.clear();
        let mut v = f.c;
        if f.c {
            out.push(0);
        }
        for &id in &t.ids[f.start as usize..(f.start + f.len) as usize] {
            out.push(base + id);
            v ^= self.bit(base + id);
        }
        v
    }

    /// Compiles one instance; `ins` are its inputs (fresh values, or earlier outputs as forms over global columns).
    fn instance(&mut self, t: &Unit, ins: &[Result<bool, (Form, bool)>]) -> Vec<(Form, bool)> {
        let base = self.rows() as u32;
        let mut scratch = Vec::new();
        for (i, s) in ins.iter().enumerate() {
            let col = base + i as u32;
            match *s {
                Ok(v) => {
                    self.push(&[col], &[col], v);
                    self.fresh_rows += 1;
                }
                Err((f, v)) => {
                    scratch.clear();
                    if f.c {
                        scratch.push(0);
                    }
                    scratch.extend_from_slice(&self.ids[f.start as usize..(f.start + f.len) as usize]);
                    let a = std::mem::take(&mut scratch);
                    self.push(&a, &[0], v);
                    scratch = a;
                    self.copy_rows += 1;
                }
            }
        }
        let (mut fa, mut fb) = (Vec::new(), Vec::new());
        for &(a, b) in &t.rows {
            let va = self.map(t, a, base, &mut fa);
            let vb = self.map(t, b, base, &mut fb);
            self.push(&fa, &fb, va & vb);
        }
        t.out_forms
            .iter()
            .map(|&f| {
                let v = self.map(t, f, base, &mut fa);
                let start = self.ids.len() as u32;
                let (c, cols) = (fa.first() == Some(&0), if fa.first() == Some(&0) { &fa[1..] } else { &fa[..] });
                self.ids.extend_from_slice(cols);
                (Form { c, start, len: cols.len() as u32 }, v)
            })
            .collect()
    }

    fn write(&self, dir: &Path, k_log: usize) {
        let k = 1usize << k_log;
        assert!(self.rows() <= k);
        let mut z = self.z.clone();
        z.resize(k / 64, 0);
        write_u64s(&dir.join("base_z.bin"), &z);
        for (name, ptr, idx) in [("base_a.bin", &self.a_ptr, &self.a_idx), ("base_b.bin", &self.b_ptr, &self.b_idx)] {
            let mut v = vec![self.rows() as u32, idx.len() as u32];
            v.extend_from_slice(ptr);
            v.extend_from_slice(idx);
            write_u32s(&dir.join(name), &v);
        }
    }
}

fn load_matrix(p: &Path, k: usize) -> (SparseBinaryMatrix, usize) {
    let v = read_u32s(p);
    let (n, nnz) = (v[0] as usize, v[1] as usize);
    let ptr = &v[2..3 + n];
    let idx = &v[3 + n..3 + n + nnz];
    let mut rows: Vec<Vec<usize>> = (0..n).into_par_iter().map(|i| idx[ptr[i] as usize..ptr[i + 1] as usize].iter().map(|&c| c as usize).collect()).collect();
    rows.resize(k, Vec::new());
    (SparseBinaryMatrix::new(k, k, rows), nnz)
}

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------

struct Composer {
    units: Vec<Unit>,
    rng: Rng,
    rp: Rp,
    base: Base,
    outs: Vec<Wire>,
    p_fresh: f64,
    cat_rows: [usize; 4],
    count: Vec<usize>,
    fresh_bits: usize,
}

impl Composer {
    fn pick(&mut self, fits: impl Fn(&Unit) -> bool) -> Option<usize> {
        let total: usize = self.cat_rows.iter().sum();
        let mut order: Vec<usize> = (0..4).filter(|&c| self.units.iter().any(|u| u.cat == c)).collect();
        order.sort_by(|&a, &b| {
            let d = |c: usize| CATS[c].1 * (total as f64 + 1.0) - self.cat_rows[c] as f64;
            d(b).partial_cmp(&d(a)).unwrap()
        });
        for c in order {
            let of: Vec<usize> = (0..self.units.len()).filter(|&i| self.units[i].cat == c && fits(&self.units[i])).collect();
            if !of.is_empty() {
                return Some(of[self.rng.below(of.len())]);
            }
        }
        None
    }

    /// Adds one instance of unit type `i`, its inputs drawn from outputs at index ≥ `from` or fresh.
    fn add(&mut self, i: usize, from: usize) {
        let t = &self.units[i];
        let rows0 = self.rp.rows;
        let mut src = Vec::with_capacity(t.n_in);
        for _ in 0..t.n_in {
            let fresh = from == self.outs.len() || (self.rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 <= self.p_fresh;
            src.push(if fresh { Src::Fresh(self.rng.next_u64() & 1 == 1) } else { Src::Out(from + self.rng.below(self.outs.len() - from)) });
        }
        let fresh_vals: Vec<bool> = src.iter().filter_map(|s| if let Src::Fresh(v) = s { Some(*v) } else { None }).collect();
        self.fresh_bits += fresh_vals.len();
        let mut fresh = self.rp.fresh(&fresh_vals).into_iter();
        let rp_in: Vec<(u32, bool)> = src
            .iter()
            .map(|s| match *s {
                Src::Fresh(_) => fresh.next().unwrap(),
                Src::Out(o) => (self.outs[o].net, self.outs[o].val),
            })
            .collect();
        let base_in: Vec<Result<bool, (Form, bool)>> = src
            .iter()
            .map(|s| match *s {
                Src::Fresh(v) => Ok(v),
                Src::Out(o) => Err((self.outs[o].form, self.outs[o].val)),
            })
            .collect();
        let rp_out = self.rp.instance(&self.units[i], &rp_in);
        let base_out = self.base.instance(&self.units[i], &base_in);
        for ((net, v), (form, vb)) in rp_out.into_iter().zip(base_out) {
            assert_eq!(v, vb, "{}: route P and C-Flock rows disagree on an output", self.units[i].name);
            self.outs.push(Wire { net, val: v, form });
        }
        self.cat_rows[self.units[i].cat] += self.rp.rows - rows0;
        self.count[i] += 1;
    }
}

fn build(dir: &Path, nets: &Path, mode: &str) {
    let t0 = Instant::now();
    fs::create_dir_all(dir).unwrap();
    let gate_klog: usize = env_or("GATE_KLOG", 16);
    let mut paths: Vec<PathBuf> = fs::read_dir(nets).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "rhn")).collect();
    paths.sort();
    let mut units: Vec<Unit> = paths.iter().map(|p| load_unit(p)).collect();
    for u in units.iter_mut() {
        let mut rp = Rp::new(gate_klog);
        rp.start_unit(1);
        let r0 = rp.rows;
        let ins = rp.fresh(&vec![false; u.n_in]);
        rp.instance(u, &ins);
        u.rp_rows = rp.rows - r0;
    }
    let mut c = Composer {
        rng: Rng::new(env_or("SEED", 1u64)),
        rp: Rp::new(gate_klog),
        base: Base::new(),
        outs: Vec::new(),
        p_fresh: env_or("P_FRESH", 0.25),
        cat_rows: [0; 4],
        count: vec![0; units.len()],
        fresh_bits: 0,
        units,
    };
    let rpb = c.rp.rpb;
    let margin = |u: &Unit| u.rp_rows + u.rp_rows / rpb + 2;
    let mut unit_rows = Vec::new();
    let mut unit_fresh = Vec::new();
    let (mu, base_klog, units_n, unit_log) = if mode == "full" {
        let base_klog: usize = env_or("BASE_KLOG", 23);
        let cap = (env_or("FILL", 0.9) * (1usize << base_klog) as f64) as usize;
        c.rp.start_unit(1);
        loop {
            let room = cap.saturating_sub(c.base.rows());
            let Some(i) = c.pick(|u| u.n_in + u.rows.len() <= room) else { break };
            c.add(i, 0);
        }
        unit_rows.push(c.rp.rows);
        unit_fresh.push(c.fresh_bits);
        let mu = (4 * c.rp.rows).next_power_of_two().trailing_zeros() as usize;
        (mu.max(gate_klog + 3), base_klog, 1, mu)
    } else {
        let n: usize = env_or("UNITS", 8);
        let unit_log: usize = env_or("UNIT_LOG", 22);
        let g = 1usize << (unit_log - 2);
        for _ in 0..n {
            c.rp.start_unit(g);
            let (r0, f0, from) = (c.rp.rows, c.fresh_bits, c.outs.len());
            loop {
                let room = g - (c.rp.rows - r0);
                let Some(i) = c.pick(|u| margin(u) <= room) else { break };
                c.add(i, from);
            }
            unit_rows.push(c.rp.rows - r0);
            unit_fresh.push(c.fresh_bits - f0);
        }
        let mu = unit_log + n.next_power_of_two().trailing_zeros() as usize;
        let base_klog = (c.base.rows()).next_power_of_two().trailing_zeros() as usize;
        (mu, base_klog, n, unit_log)
    };
    let used_rows = c.rp.rows;
    c.rp.pad_to(1 << (mu - 2));
    c.rp.close();
    c.rp.check();
    write_u64s(&dir.join("rp_z.bin"), &c.rp.z);
    write_u64s(&dir.join("rp_w.bin"), &c.rp.w);
    write_u32s(&dir.join("rp_sigma.bin"), &c.rp.sigma);
    c.base.write(dir, base_klog);
    let names: Vec<String> = c.units.iter().zip(&c.count).map(|(u, n)| format!(r#""{}":{n}"#, u.name)).collect();
    let per_type: Vec<String> = c.units.iter().map(|u| format!(r#""{}":[{},{}]"#, u.name, u.rp_rows, u.n_in + u.rows.len())).collect();
    let fixed = c.rp.sigma.iter().enumerate().filter(|&(i, &s)| i as u32 == s).count();
    let meta = format!(
        r#"{{"kind":"build","mode":"{mode}","mu":{mu},"gate_klog":{gate_klog},"gate_rows_used":{used_rows},"cells":{},"cells_fixed":{fixed},"io_rows":{},"const_rows":{},"base_klog":{base_klog},"base_rows":{},"base_nnz_a":{},"base_nnz_b":{},"base_copy_rows":{},"base_fresh_rows":{},"fresh_bits":{},"units":{units_n},"unit_log":{unit_log},"unit_rows":{:?},"unit_fresh_bits":{:?},"cat_gate_rows":{:?},"instances":{{{}}},"rows_per_instance_rp_base":{{{}}},"build_s":{:.1}}}"#,
        4 * used_rows,
        c.rp.io_rows,
        c.rp.const_rows,
        c.base.rows(),
        c.base.a_idx.len(),
        c.base.b_idx.len(),
        c.base.copy_rows,
        c.base.fresh_rows,
        c.fresh_bits,
        unit_rows,
        unit_fresh,
        c.cat_rows,
        names.join(","),
        per_type.join(","),
        t0.elapsed().as_secs_f64()
    );
    fs::write(dir.join("meta.json"), &meta).unwrap();
    println!("{meta}");
}

fn meta(dir: &Path, key: &str) -> usize {
    let s = fs::read_to_string(dir.join("meta.json")).unwrap();
    let at = s.find(&format!(r#""{key}":"#)).unwrap_or_else(|| panic!("meta.json has no {key}")) + key.len() + 3;
    s[at..].split([',', '}']).next().unwrap().parse().unwrap()
}

// ---------------------------------------------------------------------------
// gate: C-Flock's proof of route P's gate table
// ---------------------------------------------------------------------------

fn gate_matrices(k_log: usize) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let k = 1usize << k_log;
    let (mut a, mut b) = (vec![Vec::new(); k], vec![Vec::new(); k]);
    for g in 0..k / 4 {
        let c = 4 * g;
        a[c] = vec![c];
        b[c] = vec![c];
        a[c + 1] = vec![c + 1];
        b[c + 1] = vec![c + 1];
        a[c + 2] = vec![c];
        b[c + 2] = vec![c + 1];
    }
    (SparseBinaryMatrix::new(k, k, a), SparseBinaryMatrix::new(k, k, b))
}

fn x_full(p: &QuirkyPoint) -> Vec<F128> {
    let mut v = p.x_inner_rest.clone();
    v.extend_from_slice(&p.x_outer);
    v
}

/// One R1CS proof (witness phase: A·z, B·z, the lincheck packing; then upstream's fast prover) with `extra` more random
/// ring-switched claims in the opening. Returns (witness s, prove s, peak MB, proof bytes, verified); with `extra` > 0 there
/// is no verifier, and verified is true.
fn r1cs_prove(r1cs: &BlockR1cs, z: &[F128], extra: usize, verify: bool, rng: &mut Rng) -> (f64, f64, f64, usize, bool) {
    let pp = params(r1cs.m);
    let t = Instant::now();
    let a = r1cs.apply_a_packed(z);
    let b = r1cs.apply_b_packed(z);
    let zl = pack_z_lincheck_from_packed(z, r1cs.m, r1cs.k_log);
    let tw = t.elapsed().as_secs_f64();
    let mut ch = FsChallenger::new(b"rec-holo-kill");
    let csc = r1cs.csc_lincheck_circuit();
    let zc = z.to_vec();
    let t = Instant::now();
    if extra == 0 {
        let (proof, commitment, _) = prove_fast_ligerito_from_witness(r1cs, &pp, zc, a, b, zl, csc, None, &mut ch);
        let tp = t.elapsed().as_secs_f64();
        let peak = status_mb("VmHWM:");
        let bytes = bincode::serialize(&proof).unwrap().len();
        let ok = !verify || {
            let mut chv = FsChallenger::new(b"rec-holo-kill");
            verify_ligerito(r1cs, &commitment, &proof, csc, &pp, &mut chv).is_ok()
        };
        return (tw, tp, peak, bytes, ok);
    }
    let core = prove_fast_core(r1cs, &pp, zc, a, b, zl, csc, &mut ch);
    let mut xs = vec![x_full(&core.ab.point), x_full(&core.c.point)];
    xs.extend((0..extra).map(|_| rng.f128_vec(r1cs.m - 6)));
    let refs: Vec<&[F128]> = xs.iter().map(|v| v.as_slice()).collect();
    let mut pre: Vec<Option<&[F128]>> = vec![core.s_hat_v_ab.as_deref(), Some(core.s_hat_v_c.as_slice())];
    pre.extend((0..extra).map(|_| None));
    let lig = pp.ligerito_prover_config().unwrap();
    let open = open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
        core.z_packed,
        &core.prover_data,
        &core.commitment,
        &refs,
        &pre,
        &[],
        &r1cs.padding_spec(),
        &lig,
        pp.opening_grinding(),
        &mut ch,
    );
    let tp = t.elapsed().as_secs_f64();
    let peak = status_mb("VmHWM:");
    let bytes = bincode::serialize(&open).unwrap().len()
        + bincode::serialize(&core.zc_proof).unwrap().len()
        + bincode::serialize(&core.lc_proof).unwrap().len();
    (tw, tp, peak, bytes, true)
}

fn r1cs_runs(label: &str, r1cs: &BlockR1cs, z: &[F128], extra: usize, json: &str) {
    let mut rng = Rng::new(0xE47A);
    let rss = reset_peak();
    let (tw0, tp0, peak, bytes, ok) = r1cs_prove(r1cs, z, extra, true, &mut rng);
    assert!(ok, "{label}: the proof does not verify");
    let (mut tw, mut tp) = (Vec::new(), Vec::new());
    for _ in 0..runs() {
        let (a, b, _, _, _) = r1cs_prove(r1cs, z, extra, false, &mut rng);
        tw.push(a);
        tp.push(b);
    }
    let (w, p) = (median(tw), median(tp));
    println!(
        r#"{{"kind":"{label}","m":{},"k_log":{},"extra_claims":{extra},{json}"threads":{},"witness_s":{w:.6},"prove_s":{p:.6},"total_s":{:.6},"cold_total_s":{:.6},"rss_at_reset_mb":{rss:.1},"peak_mb":{peak:.1},"proof_bytes":{bytes},"verified":{}}}"#,
        r1cs.m,
        r1cs.k_log,
        rayon::current_num_threads(),
        w + p,
        tw0 + tp0,
        if extra == 0 { "true" } else { "null" }
    );
}

fn gate(dir: &Path) {
    let (mu, k_log) = (meta(dir, "mu"), meta(dir, "gate_klog"));
    let z = bits_to_packed(&read_u64s(&dir.join("rp_z.bin")));
    let (a_0, b_0) = gate_matrices(k_log);
    let r1cs = block_r1cs(mu, k_log, a_0, b_0);
    assert!(r1cs.satisfies_packed(&z), "route P's gate table does not satisfy its R1CS");
    let t = Instant::now();
    r1cs.statement_digest();
    r1cs.csc_lincheck_circuit();
    let setup = t.elapsed().as_secs_f64();
    let json = format!(r#""setup_s":{setup:.3},"#);
    r1cs_runs("gate", &r1cs, &z, 0, &json);
    r1cs_runs("gate", &r1cs, &z, 3, &json);
}

// ---------------------------------------------------------------------------
// base: C-Flock's proof of the same circuit's rows
// ---------------------------------------------------------------------------

fn base(dir: &Path, copies: usize) {
    assert!(copies.is_power_of_two() && copies >= 8);
    let k_log = meta(dir, "base_klog");
    let k = 1usize << k_log;
    let m = k_log + copies.trailing_zeros() as usize;
    let block = read_u64s(&dir.join("base_z.bin"));
    let z: Vec<F128> = bits_to_packed(&block.iter().cycle().take(block.len() * copies).copied().collect::<Vec<u64>>());
    let t = Instant::now();
    let (a_0, nnz_a) = load_matrix(&dir.join("base_a.bin"), k);
    let (b_0, nnz_b) = load_matrix(&dir.join("base_b.bin"), k);
    let load = t.elapsed().as_secs_f64();
    let r1cs = block_r1cs(m, k_log, a_0, b_0);
    assert!(r1cs.satisfies_packed(&z), "the circuit's C-Flock rows are not satisfied");
    let t = Instant::now();
    r1cs.statement_digest();
    r1cs.csc_lincheck_circuit();
    let setup = t.elapsed().as_secs_f64();
    let json = format!(r#""copies":{copies},"nnz_a":{nnz_a},"nnz_b":{nnz_b},"load_s":{load:.3},"setup_s":{setup:.3},"#);
    r1cs_runs("base", &r1cs, &z, 0, &json);
}

// ---------------------------------------------------------------------------
// gkr: the wiring argument
// ---------------------------------------------------------------------------

fn gkr(dir: &Path, how: &str, unstored: usize, double: bool) {
    let mu = meta(dir, "mu");
    let w = read_u64s(&dir.join("rp_w.bin"));
    let sigma = read_u32s(&dir.join("rp_sigma.bin"));
    assert_eq!(sigma.len(), 1 << mu);
    let prove = || -> (flock_core::product_gkr::ProductGkrBatchedProof, f64) {
        let mut ch = FsChallenger::new(b"rec-holo-gkr");
        if how == "dense" {
            let f: Vec<F128> = (0..sigma.len()).into_par_iter().map(|x| if (w[x >> 6] >> (x & 63)) & 1 == 1 { F128::ONE } else { F128::ZERO }).collect();
            let g = f.clone();
            let s: Vec<usize> = sigma.par_iter().map(|&v| v as usize).collect();
            let t = Instant::now();
            let (p, _) = prove_batched(&f, &g, &s, None, &mut ch);
            (p, t.elapsed().as_secs_f64())
        } else {
            let t = Instant::now();
            let (p, _) = rec_holo_prove_batched_stream(&w, &sigma, unstored, double, &mut ch);
            (p, t.elapsed().as_secs_f64())
        }
    };
    let rss = reset_peak();
    let (proof, cold) = prove();
    let peak = status_mb("VmHWM:");
    let bytes = bincode::serialize(&proof).unwrap();
    let mut chv = FsChallenger::new(b"rec-holo-gkr");
    let t = Instant::now();
    let ok = verify_batched(mu, &proof, None, &mut chv).is_ok();
    let tv = t.elapsed().as_secs_f64();
    assert!(ok, "gkr {how}: the proof does not verify");
    let ts: Vec<f64> = (0..runs()).map(|_| prove().1).collect();
    let p = median(ts);
    println!(
        r#"{{"kind":"gkr","how":"{how}","unstored":{unstored},"double":{double},"mu":{mu},"threads":{},"prove_s":{p:.6},"cold_s":{cold:.6},"ns_per_cell":{:.3},"verify_s":{tv:.6},"proof_bytes":{},"proof_digest":"{}","rss_at_reset_mb":{rss:.1},"peak_mb":{peak:.1},"peak_b_per_cell":{:.2}}}"#,
        rayon::current_num_threads(),
        p * 1e9 / sigma.len() as f64,
        bytes.len(),
        digest(&bytes),
        peak * (1u64 << 20) as f64 / sigma.len() as f64
    );
}

// ---------------------------------------------------------------------------
// sigma: the registered wiring commitment
// ---------------------------------------------------------------------------

fn sigma(dir: &Path, unit_log: Option<usize>) {
    let mut s = read_u32s(&dir.join("rp_sigma.bin"));
    if let Some(u) = unit_log {
        s.truncate(1 << u);
        assert!(s.iter().all(|&v| (v as usize) < s.len()), "unit 0's σ_u leaves the unit");
    }
    let mu = s.len().trailing_zeros() as usize;
    let m = mu + 5;
    let packed: Vec<F128> = s.par_chunks(4).map(|c| F128::new(c[0] as u64 | (c[1] as u64) << 32, c[2] as u64 | (c[3] as u64) << 32)).collect();
    drop(s);
    let pp = params(m);
    let rss0 = reset_peak();
    let t = Instant::now();
    let (commitment, prover_data) = pcs::commit(&packed, &pp);
    let tc = t.elapsed().as_secs_f64();
    let peak_c = status_mb("VmHWM:");
    let lig = pp.ligerito_prover_config().unwrap();
    let mut rng = Rng::new(0x51_6A);
    let open = |rng: &mut Rng| {
        let x = rng.f128_vec(m - 6);
        let mut ch = FsChallenger::new(b"rec-holo-sigma");
        let t = Instant::now();
        let proof = open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
            packed.clone(),
            &prover_data,
            &commitment,
            &[x.as_slice()],
            &[None],
            &[],
            &PaddingSpec::dense(m),
            &lig,
            pp.opening_grinding(),
            &mut ch,
        );
        (t.elapsed().as_secs_f64(), bincode::serialize(&proof).unwrap().len())
    };
    let rss1 = reset_peak();
    let (cold, bytes) = open(&mut rng);
    let peak_o = status_mb("VmHWM:");
    let ts: Vec<f64> = (0..runs()).map(|_| open(&mut rng).0).collect();
    let p = median(ts);
    println!(
        r#"{{"kind":"sigma","cells_log":{mu},"m":{m},"unit":{},"threads":{},"commit_s":{tc:.6},"open_s":{p:.6},"open_cold_s":{cold:.6},"ns_per_cell_open":{:.3},"proof_bytes":{bytes},"rss_at_commit_mb":{rss0:.1},"peak_commit_mb":{peak_c:.1},"rss_at_open_mb":{rss1:.1},"peak_open_mb":{peak_o:.1}}}"#,
        unit_log.is_some(),
        rayon::current_num_threads(),
        p * 1e9 / (1u64 << mu) as f64
    );
}

// ---------------------------------------------------------------------------
// selftest: the streaming prover against upstream's
// ---------------------------------------------------------------------------

fn selftest() {
    for mu in [6usize, 8, 11, 14, 17] {
        let n = 1usize << mu;
        let mut rng = Rng::new(0x7E57 ^ mu as u64);
        let perm = rng.permutation(n);
        let mut w = vec![0u64; n / 64];
        let mut seen = vec![false; n];
        let mut long = None;
        for s in 0..n {
            if seen[s] {
                continue;
            }
            let b = rng.next_u64() & 1 == 1;
            let (mut x, mut len) = (s, 0);
            while !seen[x] {
                seen[x] = true;
                if b {
                    w[x >> 6] |= 1 << (x & 63);
                }
                x = perm[x];
                len += 1;
            }
            if len >= 2 {
                long = Some(s);
            }
        }
        let sigma: Vec<u32> = perm.iter().map(|&v| v as u32).collect();
        let f: Vec<F128> = (0..n).map(|x| if (w[x >> 6] >> (x & 63)) & 1 == 1 { F128::ONE } else { F128::ZERO }).collect();
        let mut ch = FsChallenger::new(b"rec-holo-selftest");
        let (dense, _) = prove_batched(&f, &f, &perm, None, &mut ch);
        let want = bincode::serialize(&dense).unwrap();
        for (unstored, double) in [(1, false), (2, false), (1, true), (2, true), (3, true)] {
            let mut ch = FsChallenger::new(b"rec-holo-selftest");
            let (p, _) = rec_holo_prove_batched_stream(&w, &sigma, unstored, double, &mut ch);
            let same = bincode::serialize(&p).unwrap() == want;
            let mut chv = FsChallenger::new(b"rec-holo-selftest");
            let ok = verify_batched(mu, &p, None, &mut chv).is_ok();
            let mut bad = w.clone();
            let x = long.unwrap();
            bad[x >> 6] ^= 1 << (x & 63);
            let mut chb = FsChallenger::new(b"rec-holo-selftest");
            let (pb, _) = rec_holo_prove_batched_stream(&bad, &sigma, unstored, double, &mut chb);
            let mut chv = FsChallenger::new(b"rec-holo-selftest");
            let rejects = verify_batched(mu, &pb, None, &mut chv).is_err();
            println!(r#"{{"kind":"selftest","mu":{mu},"unstored":{unstored},"double":{double},"same_bytes":{same},"verifies":{ok},"rejects_broken_wire":{rejects}}}"#);
            assert!(same && ok && rejects, "streaming GKR selftest failed at μ {mu}");
        }
    }
}

fn main() {
    let _ = init_perf_thread_pool();
    let a: Vec<String> = args().collect();
    let dir = || PathBuf::from(&a[2]);
    match a.get(1).map(String::as_str) {
        Some("build") => build(&dir(), Path::new(&a[3]), &a[4]),
        Some("gate") => gate(&dir()),
        Some("base") => base(&dir(), a[3].parse().unwrap()),
        Some("gkr") => gkr(&dir(), &a[3], a.get(4).map_or(1, |s| s.parse().unwrap()), a.get(5).is_some_and(|s| s == "1")),
        Some("sigma") => sigma(&dir(), a.get(3).map(|s| s.parse().unwrap())),
        Some("selftest") => selftest(),
        _ => panic!("usage: rec_holo_kill build|gate|base|gkr|sigma|selftest …"),
    }
}
