//! Flock-CUDA for `verity/flock-circuit` (feature `gpu`): `flock_cuda_prove_circuit` (backends/flock/cuda/prove_circuit.cuh,
//! installed by backends/flock/cuda_circuit_patch.py). The honest witness is built on the device from the compressions'
//! inputs and every unit's input words (both computed here from the rows, a few MB); a tampered witness is uploaded whole.

use std::{ffi::c_void, ptr::{null, null_mut}, sync::OnceLock};

use flock_field::F128;
use flock_prover::{
    pcs::{Commitment, PcsParams},
    proof::R1csProofLigerito,
    r1cs::SparseBinaryMatrix,
    r1cs_hashes::blake3::Compression,
};
use rayon::prelude::*;

use crate::{
    LiveChallenger,
    chunk::comp,
    gpu::{ChunkParams, ProveParams, csc_from_rows, decode, link_cb, prove_params, sched, statics_for},
    ir_block::IrUnitNet,
    circuit::{Kind, Stmt},
};

const FC_MAX_TYPES: usize = 8;
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
    n_comp: i64,
    comp_per_block: i32,
    comp_slot: *const u32,
    cv: *const u32,
    msg: *const u32,
    ctr: *const u64,
    blen: *const u32,
    flags: *const u32,
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
}

/// Where a rep's witness comes from.
pub enum CircuitSrc<'a> {
    /// The whole packed `(z, a, b)` from the host (any witness, a tampered one included).
    Host { z: &'a [F128], a: &'a [F128], b: &'a [F128] },
    /// A cut template: the compressions and the units' leaf words as `Device`, the units in two passes around `mid`.
    DeviceCut { comps: &'a [Compression], inputs: &'a [F128], mid: Mid<'a> },
    /// Every block's compressions (block-major, each block's in slot order) and every unit's input words
    /// (`[blocks][units per block][in_words]`): the honest witness, built on the device.
    Device { comps: &'a [Compression], inputs: &'a [F128] },
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
    let late_rows: Vec<u32> = net.in_groups.get(1).map(rows_of).unwrap_or_default();
    let cut_rows: Vec<u32> = if late_rows.is_empty() { vec![] } else { net.out_groups.last().map(rows_of).unwrap_or_default() };
    let mut late = vec![false; u];
    for &r in &late_rows {
        late[r as usize] = true;
    }
    for i in first..u {
        if i != cpos {
            late[i] = net.a[i].iter().chain(&net.b[i]).any(|&c| c != i && c != cpos && late[c]);
        }
    }
    assert!(cut_rows.iter().all(|&r| !late[r as usize]), "a cut output row reads a late input: the unit is not two-phase");
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
    comp_slot: Vec<u32>,
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
                        Kind::Comp => None,
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
        let comp_slot: Vec<u32> = c.ranges.iter().filter(|r| r.kind == Kind::Comp).flat_map(|r| (0..r.count).map(|s| (r.slot(s) >> 14) as u32))
            .collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        Types { logs, csc, ranges, u_range, sched: schedule(c.unit()), comp_slot }
    })
}

/// The device witness's host inputs: every block's compressions and every unit's input words, from the prover's rows.
pub fn device_inputs(st: &Stmt, rows: &[u8]) -> (Vec<Compression>, Vec<F128>) {
    let comps: Vec<Compression> = (0..st.blocks()).into_par_iter().flat_map_iter(|b| st.compressions(Some(rows), b)).collect();
    let ur = st.c.unit_range();
    let inputs: Vec<F128> = (0..st.blocks() * ur.count).into_par_iter().flat_map_iter(|x| st.unit_inputs(Some(rows), x / ur.count, x % ur.count)).collect();
    (comps, inputs)
}

/// One rep on the GPU: proof, commitment, and the phase split (witness, commit, link, zerocheck, lincheck, ring switch,
/// ligerito, total; seconds), plus the device witness's own split (compressions, units).
#[allow(clippy::too_many_arguments)]
pub fn prove_circuit(
    st: &Stmt,
    src: CircuitSrc<'_>,
    pcs: &PcsParams,
    domain: &[u8],
    mut hook: &mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>,
    ch: &mut LiveChallenger,
) -> (R1csProofLigerito, Commitment, [f64; 8], [f64; 3]) {
    let s = statics_for(comp(), &st.circuit.da, &st.circuit.db);
    let (sc, cfg) = sched(pcs);
    let m = pcs.m;
    let params = prove_params(s, &sc, &cfg, pcs, &st.digest, domain, st.c.k_log, st.pin, ch);
    let t = types(st);
    let unit = st.c.unit();
    let n_claims = 2 + 2 * st.regions.len();
    let og = pcs.opening_grinding();
    let mut phase = [0f64; 8];
    let mut wphase = [0f64; 3];
    let (mut hz, mut ha, mut hb): (*const u64, *const u64, *const u64) = (null(), null(), null());
    let (mut cv, mut msg, mut ctr, mut blen, mut flags, mut n_comp, mut uin) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0usize, null());
    let mut src = src;
    let mut mctx: Option<MidCtx<'_>> = None;
    if let CircuitSrc::DeviceCut { comps, inputs, mid } = src {
        mctx = Some(MidCtx { mid, n_late: t.sched.late_rows.len(), n_cut: t.sched.cut_rows.len(), at: Vec::new(), words: Vec::new(), zab: Vec::new() });
        src = CircuitSrc::Device { comps, inputs };
    }
    match &src {
        CircuitSrc::Host { z, a, b } => {
            assert_eq!(z.len(), 1 << (m - 7));
            (hz, ha, hb) = (z.as_ptr() as *const u64, a.as_ptr() as *const u64, b.as_ptr() as *const u64);
        }
        CircuitSrc::DeviceCut { .. } => unreachable!(),
        CircuitSrc::Device { comps, inputs } => {
            assert_eq!(comps.len(), st.blocks() * t.comp_slot.len());
            assert_eq!(inputs.len(), st.blocks() * st.c.unit_range().count * unit.in_words);
            cv = comps.iter().flat_map(|c| c.0).collect();
            msg = comps.iter().flat_map(|c| c.1).collect();
            ctr = comps.iter().map(|c| c.2).collect();
            blen = comps.iter().map(|c| c.3).collect();
            flags = comps.iter().map(|c| c.4).collect();
            n_comp = comps.len();
            uin = inputs.as_ptr() as *const u64;
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
        n_comp: n_comp as i64,
        comp_per_block: t.comp_slot.len() as i32,
        comp_slot: t.comp_slot.as_ptr(),
        cv: cv.as_ptr(),
        msg: msg.as_ptr(),
        ctr: ctr.as_ptr(),
        blen: blen.as_ptr(),
        flags: flags.as_ptr(),
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
    };
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
    assert_eq!(rc, 0, "CUDA circuit prover returned error {rc} at m={m}");
    let bytes = unsafe { std::slice::from_raw_parts(out, out_len) }.to_vec();
    unsafe { flock_cuda_free(out) };
    let (proof, commitment) = decode(&bytes, pcs);
    (proof, commitment, phase, wphase)
}
