//! Flock-CUDA for `verity/flock-circuit` (feature `gpu`): `flock_cuda_prove_circuit` (backends/flock/cuda/prove_circuit.cuh,
//! installed by backends/flock/cuda_circuit_patch.py). The honest witness is built on the device from the compressions'
//! inputs and every unit's input words (both computed here from the rows, a few MB), a deep unit's slots here (`deep_unit`);
//! a tampered witness is uploaded whole. Rep 1 proves from rep 0's witness and level-0 commitment (`Reuse`).

use std::{ffi::c_void, ptr::{null, null_mut}, sync::OnceLock};

use flock_field::F128;
use flock_prover::{
    pcs::{Commitment, PcsParams},
    proof::R1csProofLigerito,
    r1cs::SparseBinaryMatrix,
};
use rayon::prelude::*;

use crate::{
    LiveChallenger,
    chunk::comp,
    gpu::{ChunkParams, ProveParams, Rounds, csc_from_rows, decode_with, link_cb, prove_params, sched, statics_for},
    ir_block::IrUnitNet,
    circuit::{Kind, Stmt},
};

const FC_MAX_TYPES: usize = 12;
const FC_MAX_RANGES: usize = 32;

#[repr(C)]
struct CircuitParams {
    c: ChunkParams,
    n_types: i32,
    type_log: [i32; FC_MAX_TYPES],
    t_a_col_ptr: [*const u32; FC_MAX_TYPES],
    t_a_rows: [*const u32; FC_MAX_TYPES],
    t_a_nnz: [u32; FC_MAX_TYPES],
    t_b_col_ptr: [*const u32; FC_MAX_TYPES],
    t_b_rows: [*const u32; FC_MAX_TYPES],
    t_b_nnz: [u32; FC_MAX_TYPES],
    n_ranges: i32,
    r_type: [i32; FC_MAX_RANGES],
    r_pos0: [i32; FC_MAX_RANGES],
    r_count: [i32; FC_MAX_RANGES],
    /// the compression slots (sha512x3), evaluated on the device like the units
    h_range: i32,
    h_useful: i32,
    h_const: i32,
    h_in_words: i32,
    h_a_off: *const u32,
    h_a_col: *const u32,
    h_b_off: *const u32,
    h_b_col: *const u32,
    h_levels: i32,
    h_level_off: *const u32,
    h_order: *const u32,
    h_inputs: *const u64,
    /// host slots given up front (the hm96 slots): absolute F128 word offsets, word counts, z ‖ a ‖ b words
    n_host0: i64,
    host0_at: *const i64,
    host0_words: *const i32,
    host0_zab: *const u64,
    u_range: i32,
    u_useful: i32,
    u_const: i32,
    u_in_words: i32,
    u_a_off: *const u32,
    u_a_col: *const u32,
    u_b_off: *const u32,
    u_b_col: *const u32,
    u_levels: i32,
    u_level_off: *const u32,
    u_order: *const u32,
    u_inputs: *const u64,
    u_warps: i32,
    witness_s: *mut f64,
    u_levels0: i32,
    u_n_late: i32,
    u_late_rows: *const u32,
    u_n_cut: i32,
    u_cut_rows: *const u32,
    mid_cb: Option<MidCb>,
    mid_ctx: *mut c_void,
    /// hm96-sha512 Merkle leaves (the SHA-512 build): the device expands each tree's salts with ChaCha20 under `hm96_key` (32
    /// bytes, `zk_hooks::LeafSalts`) with the nonce `hm96_tree` returns for this proof's k-th tree; and the scheme's constants
    /// (salt prefix ‖ leaf prefix, 256 bytes; the key's byte tables, 192 × 256 × 8 u64). Null: unsalted.
    hm96_tree: Option<Hm96TreeCb>,
    hm96_ctx: *mut c_void,
    hm96_key: *const u8,
    hm96_prefixes: *const u8,
    hm96_tables: *const u64,
    hm96_level0_leaves: i64,
    /// M1 (`zk_on`): Ligerito level 0's hiding (backends/flock/cuda/prove_circuit.cuh): the rep whose pair the lane phase folds,
    /// the padding per lane, the negatives' tamper code, every lane's padding (lane-major, t_pad words) and the 4 extra lanes
    /// (L words each)
    zk_on: i32,
    zk_rep: i32,
    zk_t_pad: i32,
    zk_tamper: i32,
    zk_pads: *const u64,
    zk_extra: *const u64,
    /// the units' slots are among the host slots: the device's unit kernel does not run
    u_host: i32,
    /// `Reuse`: 0 none, 1 keep for `reuse_tag`, 2 take `reuse_tag`'s
    rep_reuse: i32,
    reuse_tag: u64,
}

#[cfg(feature = "sha512")]
type Level0 = flock_prover::pcs::ligerito::zk::Level0Zk;
#[cfg(not(feature = "sha512"))]
type Level0 = ();

/// Where a rep's live rounds go.
pub enum Live<'a> {
    /// M0: straight to the rep's stream
    Stream(&'a mut LiveChallenger),
    /// M1: through `rounds` (the masking challenger in front of the rep's stream, `crate::replay_round`), with level 0's hiding
    /// (`level0`: the session's padding and extra lanes, this rep's pair)
    #[cfg(feature = "sha512")]
    Zk { level0: &'a Level0, rounds: &'a mut dyn FnMut(&[u8], usize) -> Vec<F128> },
}

type Hm96TreeCb = unsafe extern "C" fn(*mut c_void, i32) -> i64;

/// The salt-stream nonce of this proof's k-th tree: 0 for level 0 (both reps share it), a fresh id for every later tree.
#[cfg(feature = "sha512")]
unsafe extern "C" fn hm96_tree(ctx: *mut c_void, tree: i32) -> i64 {
    if tree == 0 {
        return 0;
    }
    let salts = unsafe { &*(ctx as *const std::sync::Mutex<crate::zk_hooks::LeafSalts>) };
    salts.lock().unwrap_or_else(std::sync::PoisonError::into_inner).next_id() as i64
}

type MidCb = unsafe extern "C" fn(*mut c_void, *const u32, i64, *mut u32, *mut i64, *mut *const i64, *mut *const i32, *mut *const u64) -> i32;

/// What the device's first pass hands the host, and what the host hands back (kept alive until the prover returns).
pub type Mid<'a> = &'a mut dyn FnMut(&[u32]) -> (Vec<u32>, Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)>);

struct MidCtx<'a> {
    mid: Mid<'a>,
    n_late: usize,
    n_cut: usize,
    at: Vec<i64>,
    words: Vec<i32>,
    zab: Vec<u64>,
}

unsafe extern "C" fn mid_cb(ctx: *mut c_void, cut: *const u32, groups: i64, late: *mut u32, n_host: *mut i64, host_at: *mut *const i64,
                            host_words: *mut *const i32, host_zab: *mut *const u64) -> i32 {
    let m = unsafe { &mut *(ctx as *mut MidCtx<'_>) };
    let cut_all = unsafe { std::slice::from_raw_parts(cut, groups as usize * m.n_cut) };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (m.mid)(cut_all))) {
        Ok((lt, host)) => {
            if lt.len() != groups as usize * m.n_late {
                return -1;
            }
            unsafe { std::ptr::copy_nonoverlapping(lt.as_ptr(), late, lt.len()) };
            m.at.clear();
            m.words.clear();
            m.zab.clear();
            for (at, z, a, b) in host {
                m.at.push(at as i64);
                m.words.push(z.len() as i32);
                for v in [z, a, b] {
                    m.zab.extend(v.iter().flat_map(|w| [w.lo, w.hi]));
                }
            }
            unsafe {
                *n_host = m.at.len() as i64;
                *host_at = m.at.as_ptr();
                *host_words = m.words.as_ptr();
                *host_zab = m.zab.as_ptr();
            }
            0
        }
        Err(_) => -1,
    }
}


unsafe extern "C" {
    fn flock_cuda_prove_circuit(p: *const ProveParams, n: *const CircuitParams, out: *mut *mut u8, out_len: *mut usize) -> i32;
    fn flock_cuda_free(p: *mut u8);
    #[cfg(feature = "sha512")]
    fn flock_cuda_hm96_salts(key: *const u8, nonce: u64, idx: *const i64, n: i32, out: *mut u8) -> i32;
    fn flock_cuda_circuit_kept(tag: u64) -> i32;
}

/// Rep 1 proves from rep 0's witness and level-0 commitment, which both reps bind (one level-0 root): rep 0 keeps them on the
/// device for the session's tag, and rep 1 takes them. Where the device has no room to keep them, rep 1 proves in full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reuse {
    None,
    Keep(u64),
    Take(u64),
}

/// Whether the device holds a rep 0's witness and level-0 commitment for `tag`.
pub fn kept(tag: u64) -> bool {
    unsafe { flock_cuda_circuit_kept(tag) != 0 }
}

/// The device's unit kernel syncs once per AND-level for 32 units per CTA, so a deep unit leaves the device idle (a flattened
/// GEMM coordinate has 305 levels per tensor-core step), while the host's `eval64` walks its rows once per 64 units. A unit at
/// least this deep is evaluated on the host; `FC_HOST_UNITS=0|1` overrides.
pub const HOST_UNIT_LEVELS: usize = 2048;

/// Whether the statement's units are evaluated on the host (`HOST_UNIT_LEVELS`).
pub fn deep_unit(st: &Stmt) -> bool {
    match std::env::var("FC_HOST_UNITS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => types(st).sched.level_off.len() - 1 >= HOST_UNIT_LEVELS,
    }
}

/// The salts of leaves `idx` of tree `nonce` under `key` as the device expands them (192 bytes each): a test vector against
/// `zk_hooks::LeafSalts::salt`.
#[cfg(feature = "sha512")]
pub fn device_salts(key: &[u8; 32], nonce: u64, idx: &[i64]) -> Vec<u8> {
    let mut out = vec![0u8; 192 * idx.len()];
    let rc = unsafe { flock_cuda_hm96_salts(key.as_ptr(), nonce, idx.as_ptr(), idx.len() as i32, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "device salt expansion failed ({rc})");
    out
}

/// A slot whose witness the host computed: its absolute F128 word offset and its z, a, b words.
pub type HostSlot = (usize, Vec<F128>, Vec<F128>, Vec<F128>);

/// Where a rep's witness comes from.
pub enum CircuitSrc<'a> {
    /// The whole packed `(z, a, b)` from the host (any witness, a tampered one included).
    Host { z: &'a [F128], a: &'a [F128], b: &'a [F128] },
    /// A cut template: as `Device`, the units in two passes around `mid`.
    DeviceCut { rows: &'a [F128], host: &'a [HostSlot], inputs: &'a [F128], mid: Mid<'a> },
    /// Every compression slot's input words (`[blocks][slots per block][in_words]`, `Stmt::row_inputs`), the hm96 slots' witness
    /// (host slots), and every unit's input words (`[blocks][units per block][in_words]`): the honest witness, built on the device.
    /// With `inputs` empty, every unit slot is among the host slots (`deep_unit`).
    Device { rows: &'a [F128], host: &'a [HostSlot], inputs: &'a [F128] },
}

/// The unit's expanded circuit's rows as CSR and its level schedule: level 0 the input rows and the constant, a row one past its
/// deepest operand (itself, in an assertion row, excluded).
pub struct UnitSched {
    /// levels [0, levels0) read no late input row; the rest do
    pub levels0: usize,
    pub late_rows: Vec<u32>,
    pub cut_rows: Vec<u32>,
    pub a_off: Vec<u32>,
    pub a_col: Vec<u32>,
    pub b_off: Vec<u32>,
    pub b_col: Vec<u32>,
    pub level_off: Vec<u32>,
    pub order: Vec<u32>,
}

/// The unit's schedule; a cut unit's (a second input group: its late rows, the cut words the tail computes; a second output
/// group: its cut output rows) in two phases: the rows that read no late row, then the rest.
pub fn schedule(net: &IrUnitNet) -> UnitSched {
    let (u, first, cpos) = (net.useful, net.in_words * 128, net.const_pos);
    let rows_of = |g: &crate::ir_block::Group| -> Vec<u32> { (0..g.bits.iter().sum::<usize>()).map(|j| (g.col * 128 + j) as u32).collect() };
    let mut late_rows: Vec<u32> = net.in_groups.get(1).map(rows_of).unwrap_or_default();
    let mut cut_rows: Vec<u32> = if late_rows.is_empty() { vec![] } else { net.out_groups.last().map(rows_of).unwrap_or_default() };
    let mut late = vec![false; u];
    for &r in &late_rows {
        late[r as usize] = true;
    }
    for i in first..u {
        if i != cpos {
            late[i] = net.a[i].iter().chain(&net.b[i]).any(|&c| c != i && c != cpos && late[c]);
        }
    }
    if cut_rows.iter().any(|&r| late[r as usize]) {
        // not two-phase (a cut output reads a late input: attention's and GEMM's tensor-core steps): one phase; such a unit's
        // witness is the host's (flock-circuit keeps it off the device)
        late_rows.clear();
        cut_rows.clear();
        late = vec![false; u];
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
    let (a_off, a_col) = csr(&net.a);
    let (b_off, b_col) = csr(&net.b);
    let mut level_off = vec![0u32];
    let mut order = Vec::new();
    let mut levels0 = 0;
    for phase in [false, true] {
        let mut level = vec![0u32; u];
        let mut max = 0;
        for i in first..u {
            if i == cpos || late[i] != phase {
                continue;
            }
            let l = 1 + net.a[i].iter().chain(&net.b[i]).filter(|&&c| c != i && c != cpos && late[c] == phase && c >= first).map(|&c| level[c]).max().unwrap_or(0);
            level[i] = l;
            max = max.max(l);
        }
        let mut by: Vec<Vec<u32>> = vec![Vec::new(); max as usize];
        for i in first..u {
            if i != cpos && late[i] == phase {
                by[level[i] as usize - 1].push(i as u32);
            }
        }
        for l in by {
            order.extend(l);
            level_off.push(order.len() as u32);
        }
        if !phase {
            levels0 = level_off.len() - 1;
        }
    }
    UnitSched { levels0, late_rows, cut_rows, a_off, a_col, b_off, b_col, level_off, order }
}

struct Types {
    logs: Vec<i32>,
    csc: Vec<Option<((Vec<u32>, Vec<u32>), (Vec<u32>, Vec<u32>))>>,
    ranges: Vec<(i32, i32, i32)>,
    u_range: usize,
    sched: UnitSched,
    h_range: usize,
    hsched: UnitSched,
}

fn types(st: &Stmt) -> &'static Types {
    static T: OnceLock<Types> = OnceLock::new();
    T.get_or_init(|| {
        let c = &st.c;
        let mut logs = Vec::new();
        let mut csc = Vec::new();
        let mut key: Vec<String> = Vec::new();
        let mut ranges = Vec::new();
        for r in &c.ranges {
            let k = format!("{:?}", r.kind);
            let t = match key.iter().position(|x| *x == k) {
                Some(t) => t,
                None => {
                    key.push(k);
                    logs.push(r.slot_log as i32);
                    csc.push(match r.kind {
                        Kind::Net(i) => {
                            let net = &c.nets[i].1;
                            let n = 1usize << net.unit_log;
                            let pad = |rows: &Vec<Vec<usize>>| { let mut x = rows.clone(); x.resize(n, Vec::new()); SparseBinaryMatrix { num_rows: n, num_cols: n, rows: x.into() } };
                            Some((csc_from_rows(&pad(&net.a)), csc_from_rows(&pad(&net.b))))
                        }
                        Kind::Mask => {
                            let n = 1usize << r.slot_log;
                            let id = csc_from_rows(&SparseBinaryMatrix { num_rows: n, num_cols: n, rows: (0..n).map(|i| vec![i]).collect::<Vec<_>>().into() });
                            Some((id.clone(), id))
                        }
                    });
                    key.len() - 1
                }
            };
            ranges.push((t as i32, r.pos0 as i32, r.count as i32));
        }
        assert!(logs.len() <= FC_MAX_TYPES && ranges.len() <= FC_MAX_RANGES, "too many slot types or ranges for the FFI");
        let u_range = c.ranges.iter().position(|r| r.kind == Kind::Net(c.unit_net)).unwrap();
        let h_range = c.ranges.iter().position(|r| r.kind == Kind::Net(c.sha_net)).unwrap();
        Types { logs, csc, ranges, u_range, sched: schedule(c.unit()), h_range, hsched: schedule(&c.nets[c.sha_net].1) }
    })
}

/// The device witness's host inputs for an honest prover: every compression slot's input words, the hm96 slots' witness, and
/// every unit's input words.
pub fn device_inputs(st: &Stmt, rows: &[u8], salts: &[u8]) -> (Vec<F128>, Vec<HostSlot>, Vec<F128>) {
    let ri: Vec<_> = (0..st.blocks()).into_par_iter().map(|b| st.row_inputs(Some(rows), Some(salts), b, Default::default())).collect();
    let sha: Vec<F128> = ri.iter().flat_map(|x| x.0.iter().flatten().copied()).collect();
    let hm: Vec<Vec<F128>> = ri.into_iter().flat_map(|x| x.1).collect();
    let host = host_slots(st, st.c.hm_net, &hm);
    let ur = st.c.unit_range();
    let inputs: Vec<F128> = (0..st.blocks() * ur.count).into_par_iter().flat_map_iter(|x| st.unit_inputs(Some(rows), x / ur.count, x % ur.count)).collect();
    (sha, host, inputs)
}

/// Circuit `ni`'s slots evaluated on the host from their input words (`[blocks * count]`), placed at their absolute word offsets.
pub fn host_slots(st: &Stmt, ni: usize, ins: &[Vec<F128>]) -> Vec<HostSlot> {
    let r = st.c.range_of(ni);
    let wpb = 1usize << (st.c.k_log - 7);
    st.slot_zab(ni, &st.c.nets[ni].1, ins).into_iter().map(|(x, z, a, b)| ((x / r.count) * wpb + r.slot(x % r.count) / 128, z, a, b)).collect()
}

/// One rep on the GPU: proof, commitment, and the phase split (witness, commit, link, zerocheck, lincheck, ring switch,
/// ligerito, total; seconds), plus the device witness's own split (compressions, units). Under M1 (`Live::Zk`) the proof is the
/// unmasked one the CPU prover leaves before masking (with level 0's hiding messages and record, and lincheck's folded comb),
/// and a round the handler refused (the masking's degenerate coins, a refused coin) is raised once the device returns. With
/// `Reuse::Take` of a kept tag, `src` is not read: the witness and level-0 commitment are rep 0's (never under M1).
#[allow(clippy::too_many_arguments)]
pub fn prove_circuit(
    st: &Stmt,
    src: CircuitSrc<'_>,
    pcs: &PcsParams,
    domain: &[u8],
    mut hook: &mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>,
    live: Live<'_>,
    salts: Option<&std::sync::Mutex<crate::zk_hooks::LeafSalts>>,
    reuse: Reuse,
) -> (R1csProofLigerito, Commitment, [f64; 8], [f64; 3]) {
    let s = statics_for(comp(), &st.circuit.da, &st.circuit.db);
    let (sc, cfg) = sched(pcs);
    let m = pcs.m;
    // the device reads the schedule through pointers into `sc`: it lives until the prover returns
    #[cfg(feature = "sha512")]
    let sc = match &live {
        Live::Zk { level0, .. } => sc.with_queries0(level0.queries0),
        Live::Stream(_) => sc,
    };
    let (params, mut rounds, level0): (ProveParams, Option<Rounds<'_>>, Option<&Level0>) = match live {
        Live::Stream(ch) => (prove_params(s, &sc, &cfg, pcs, &st.flock_digest, domain, st.c.k_log, st.pin, Some(ch)), None, None),
        #[cfg(feature = "sha512")]
        Live::Zk { level0, rounds } => (prove_params(s, &sc, &cfg, pcs, &st.flock_digest, domain, st.c.k_log, st.pin, None),
                                        Some(Rounds { handler: rounds, failed: None, rounds: 0 }), Some(level0)),
    };
    let params = match rounds.as_mut() {
        Some(r) => params.with_rounds(r),
        None => params,
    };
    // M1's level-0 commitment and lane phase are its own (fc_commit_l0_zk, ligerito_zk.cuh): both reps prove in full under it
    let reuse = if level0.is_some() { Reuse::None } else { reuse };
    let t = types(st);
    let unit = st.c.unit();
    let n_claims = 2 + 2 * st.regions.len();
    let og = pcs.opening_grinding();
    let mut phase = [0f64; 8];
    let mut wphase = [0f64; 3];
    let (mut hz, mut ha, mut hb): (*const u64, *const u64, *const u64) = (null(), null(), null());
    let (mut uin, mut hin): (*const u64, *const u64) = (null(), null());
    let (mut h_at, mut h_words, mut h_zab): (Vec<i64>, Vec<i32>, Vec<u64>) = (Vec::new(), Vec::new(), Vec::new());
    let mut src = src;
    let mut mctx: Option<MidCtx<'_>> = None;
    if let CircuitSrc::DeviceCut { rows, host, inputs, mid } = src {
        mctx = Some(MidCtx { mid, n_late: t.sched.late_rows.len(), n_cut: t.sched.cut_rows.len(), at: Vec::new(), words: Vec::new(), zab: Vec::new() });
        src = CircuitSrc::Device { rows, host, inputs };
    }
    let taken = matches!(reuse, Reuse::Take(tag) if kept(tag));
    let mut u_host = 0;
    match &src {
        _ if taken => {}
        CircuitSrc::Host { z, a, b } => {
            assert_eq!(z.len(), 1 << (m - 7));
            (hz, ha, hb) = (z.as_ptr() as *const u64, a.as_ptr() as *const u64, b.as_ptr() as *const u64);
        }
        CircuitSrc::DeviceCut { .. } => unreachable!(),
        CircuitSrc::Device { rows, host, inputs } => {
            if inputs.is_empty() {
                assert!(mctx.is_none(), "a cut template's units are the device's");
                u_host = 1;
            } else {
                assert_eq!(inputs.len(), st.blocks() * st.c.unit_range().count * unit.in_words);
                uin = inputs.as_ptr() as *const u64;
            }
            assert_eq!(rows.len(), st.blocks() * st.c.range_of(st.c.sha_net).count * st.c.nets[st.c.sha_net].1.in_words);
            hin = rows.as_ptr() as *const u64;
            for (at, z, a, b) in host.iter() {
                h_at.push(*at as i64);
                h_words.push(z.len() as i32);
                for v in [z, a, b] {
                    h_zab.extend(v.iter().flat_map(|x| [x.lo, x.hi]));
                }
            }
        }
    }
    let c = ChunkParams {
        delta_a: s.da.as_ptr(),
        n_delta_a: (s.da.len() / 2) as i32,
        delta_b: s.db.as_ptr(),
        n_delta_b: (s.db.len() / 2) as i32,
        sub_log: 14,
        link_cb: Some(link_cb),
        link_ctx: &mut hook as *mut &mut dyn FnMut(&[u8]) -> Vec<Vec<F128>> as *mut c_void,
        gamma_bits: if n_claims > 1 { og.claim_batch_bits as i32 } else { 0 },
        phase_s: phase.as_mut_ptr(),
        mode: 1,
        host_z: hz,
        host_a: ha,
        host_b: hb,
        ..ChunkParams::none()
    };
    let mut p = CircuitParams {
        c,
        n_types: t.logs.len() as i32,
        type_log: [0; FC_MAX_TYPES],
        t_a_col_ptr: [null(); FC_MAX_TYPES],
        t_a_rows: [null(); FC_MAX_TYPES],
        t_a_nnz: [0; FC_MAX_TYPES],
        t_b_col_ptr: [null(); FC_MAX_TYPES],
        t_b_rows: [null(); FC_MAX_TYPES],
        t_b_nnz: [0; FC_MAX_TYPES],
        n_ranges: t.ranges.len() as i32,
        r_type: [0; FC_MAX_RANGES],
        r_pos0: [0; FC_MAX_RANGES],
        r_count: [0; FC_MAX_RANGES],
        h_range: t.h_range as i32,
        h_useful: st.c.nets[st.c.sha_net].1.useful as i32,
        h_const: st.c.nets[st.c.sha_net].1.const_pos as i32,
        h_in_words: st.c.nets[st.c.sha_net].1.in_words as i32,
        h_a_off: t.hsched.a_off.as_ptr(),
        h_a_col: t.hsched.a_col.as_ptr(),
        h_b_off: t.hsched.b_off.as_ptr(),
        h_b_col: t.hsched.b_col.as_ptr(),
        h_levels: (t.hsched.level_off.len() - 1) as i32,
        h_level_off: t.hsched.level_off.as_ptr(),
        h_order: t.hsched.order.as_ptr(),
        h_inputs: hin,
        n_host0: h_at.len() as i64,
        host0_at: h_at.as_ptr(),
        host0_words: h_words.as_ptr(),
        host0_zab: h_zab.as_ptr(),
        u_range: t.u_range as i32,
        u_useful: unit.useful as i32,
        u_const: unit.const_pos as i32,
        u_in_words: unit.in_words as i32,
        u_a_off: t.sched.a_off.as_ptr(),
        u_a_col: t.sched.a_col.as_ptr(),
        u_b_off: t.sched.b_off.as_ptr(),
        u_b_col: t.sched.b_col.as_ptr(),
        u_levels: (t.sched.level_off.len() - 1) as i32,
        u_level_off: t.sched.level_off.as_ptr(),
        u_order: t.sched.order.as_ptr(),
        u_inputs: uin,
        u_warps: std::env::var("FC_WARPS").ok().and_then(|x| x.parse().ok()).unwrap_or(if unit.useful > 65536 { 8 } else { 1 }),
        witness_s: wphase.as_mut_ptr(),
        u_levels0: if mctx.is_some() { t.sched.levels0 as i32 } else { (t.sched.level_off.len() - 1) as i32 },
        u_n_late: if mctx.is_some() { t.sched.late_rows.len() as i32 } else { 0 },
        u_late_rows: t.sched.late_rows.as_ptr(),
        u_n_cut: t.sched.cut_rows.len() as i32,
        u_cut_rows: t.sched.cut_rows.as_ptr(),
        mid_cb: mctx.is_some().then_some(mid_cb as MidCb),
        mid_ctx: mctx.as_mut().map_or(null_mut(), |m| m as *mut MidCtx<'_> as *mut c_void),
        hm96_tree: None,
        hm96_ctx: null_mut(),
        hm96_key: std::ptr::null(),
        hm96_prefixes: std::ptr::null(),
        hm96_tables: std::ptr::null(),
        hm96_level0_leaves: 0,
        zk_on: 0,
        zk_rep: 0,
        zk_t_pad: 0,
        zk_tamper: 0,
        zk_pads: null(),
        zk_extra: null(),
        u_host,
        rep_reuse: match reuse { Reuse::None => 0, Reuse::Keep(_) => 1, Reuse::Take(_) => 2 },
        reuse_tag: match reuse { Reuse::None => 0, Reuse::Keep(t) | Reuse::Take(t) => t },
    };
    #[cfg(feature = "sha512")]
    let zk_words: Option<(Vec<F128>, Vec<F128>)> = level0.map(|z| {
        let (w, l) = (1usize << cfg.initial_k, 1usize << (m - 7 - cfg.initial_k));
        assert!(z.pads.len() == w + flock_prover::pcs::ligerito::zk::EXTRA_LANES && z.pads.iter().all(|p| p.len() == z.t_pad));
        assert!(z.extra.iter().all(|x| x.len() == l), "zk: an extra lane of L words");
        (z.pads.concat(), z.extra.concat())
    });
    #[cfg(feature = "sha512")]
    if let (Some(z), Some((pads, extra))) = (level0, &zk_words) {
        (p.zk_on, p.zk_rep, p.zk_t_pad, p.zk_tamper) = (1, z.rep as i32, z.t_pad as i32, z.tamper as i32);
        (p.zk_pads, p.zk_extra) = (pads.as_ptr() as *const u64, extra.as_ptr() as *const u64);
    }
    #[cfg(not(feature = "sha512"))]
    let _ = level0;
    #[cfg(feature = "sha512")]
    let hm96_prefixes: Vec<u8> = {
        let c = flock_merkle::hm96::consts();
        [c.salt_prefix, c.leaf_prefix].concat()
    };
    #[cfg(feature = "sha512")]
    let key_guard = salts.map(|s| crate::zk_hooks::ZeroOnDrop(s.lock().unwrap_or_else(std::sync::PoisonError::into_inner).key().to_vec()));
    #[cfg(feature = "sha512")]
    if let (Some(s), Some(k)) = (salts, &key_guard) {
        p.hm96_tree = Some(hm96_tree as Hm96TreeCb);
        p.hm96_ctx = s as *const _ as *mut c_void;
        p.hm96_key = k.0.as_ptr();
        p.hm96_level0_leaves = pcs.n_leaves() as i64;
        p.hm96_prefixes = hm96_prefixes.as_ptr();
        p.hm96_tables = flock_merkle::hm96::consts().tables.as_ptr() as *const u64;
    }
    #[cfg(not(feature = "sha512"))]
    let _ = salts;
    for (i, l) in t.logs.iter().enumerate() {
        p.type_log[i] = *l;
        if let Some(((ac, ar), (bc, br))) = &t.csc[i] {
            p.t_a_col_ptr[i] = ac.as_ptr();
            p.t_a_rows[i] = ar.as_ptr();
            p.t_a_nnz[i] = ar.len() as u32;
            p.t_b_col_ptr[i] = bc.as_ptr();
            p.t_b_rows[i] = br.as_ptr();
            p.t_b_nnz[i] = br.len() as u32;
        }
    }
    for (j, &(ty, pos0, count)) in t.ranges.iter().enumerate() {
        (p.r_type[j], p.r_pos0[j], p.r_count[j]) = (ty, pos0, count);
    }
    let mut out: *mut u8 = null_mut();
    let mut out_len: usize = 0;
    let rc = unsafe { flock_cuda_prove_circuit(&params, &p, &mut out, &mut out_len) };
    drop(mctx);
    #[cfg(feature = "sha512")]
    drop(key_guard);
    if let Some(why) = rounds.and_then(|r| r.failed) {
        if rc == 0 {
            unsafe { flock_cuda_free(out) };
        }
        panic!("{why}");
    }
    assert!(rc != 240, "PROVER-STOPPED: degenerate coins: rho_0, rho_1 dependent over F128");
    assert_eq!(rc, 0, "CUDA circuit prover returned error {rc} at m={m}");
    let bytes = unsafe { std::slice::from_raw_parts(out, out_len) }.to_vec();
    unsafe { flock_cuda_free(out) };
    let (proof, commitment) = decode_with(&bytes, pcs, level0.is_some());
    (proof, commitment, phase, wphase)
}
