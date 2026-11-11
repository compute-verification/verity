//! Flock-CUDA for `verity/flock-circuit` (feature `gpu`): `flock_cuda_prove_circuit` (verity/protocols/flock/prover/cuda/prove_circuit.cuh,
//! installed by verity/protocols/flock/prover/cuda_circuit_patch.py). The honest witness is built on the device from the compressions' and
//! hm96 slots' inputs and every unit's input words (all computed here from the rows, a few MB), a deep unit's slots here (`deep_unit`);
//! a tampered witness is uploaded whole. Rep 1 proves from rep 0's level-0 commitment (`Reuse`).

use std::{ffi::c_void, ptr::{null, null_mut}, sync::{atomic::{AtomicU64, Ordering}, Mutex, OnceLock}, time::Instant};

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
    gpu::{ChunkParams, ProveParams, Rounds, csc_from_rows, csc_of, csr, decode_with, link_cb, prove_params, sched, statics_for},
    ir_block::{IrUnitNet, RowRead},
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
    /// hm96-sha512 Merkle leaves (the SHA-512 build): `hm96_tree` hands the device this proof's k-th tree's salts, read on the
    /// host from the OS (`zk_hooks::LeafSalts`, 192 bytes per leaf, held until the prover returns); and the scheme's constants
    /// (salt prefix ‖ leaf prefix, 256 bytes; the key's byte tables, 192 × 256 × 8 u64). Null: unsalted.
    hm96_tree: Option<Hm96TreeCb>,
    hm96_ctx: *mut c_void,
    hm96_prefixes: *const u8,
    hm96_tables: *const u64,
    hm96_level0_leaves: i64,
    /// M1 (`zk_on`): Ligerito level 0's hiding (verity/protocols/flock/prover/cuda/prove_circuit.cuh): the rep whose pair the lane phase folds,
    /// the padding per lane, the negatives' tamper code, every lane's padding (lane-major, t_pad words) and the 4 extra lanes
    /// (L words each)
    zk_on: i32,
    zk_rep: i32,
    zk_t_pad: i32,
    zk_tamper: i32,
    zk_pads: *const u64,
    zk_extra: *const u64,
    /// where set, `zk_pads` and `zk_extra` once the host has them, which the device asks for when level 0's commitment first
    /// reads them (`ZkWords`)
    zk_words: Option<ZkWordsCb>,
    zk_words_ctx: *mut c_void,
    /// the units' slots are among the host slots: the device's unit kernel does not run
    u_host: i32,
    /// `Reuse`: 0 none, 1 keep for `reuse_tag`, 2 take `reuse_tag`'s
    rep_reuse: i32,
    reuse_tag: u64,
    /// the compression slots' native witness: per sha512x3 row its (a, b) sources, and the tape's words per slot; null: level by level
    h_plan: *const u32,
    h_tape_words: i32,
    /// `host0_zab`'s layout: 0 each slot's z, a, b words; 1 its a, b words, z = a & b (mapped host memory only)
    host0_ab: i32,
    /// the hm96 slots' native witness beside `h_plan`'s: their range, the hm96 circuit's plan and useful rows, the tape's words
    /// per slot, every hm96 slot's input words, the salt prefix's midstate and the key's byte tables; null `m_plan`: the hm96
    /// slots are among the host slots
    m_range: i32,
    m_useful: i32,
    m_in_words: i32,
    m_tape_words: i32,
    m_plan: *const u32,
    m_inputs: *const u64,
    m_mid: *const u64,
    m_tables: *const u64,
    /// [`Types::types_gen`]: the key the device holds the slot types' CSC and the unit's CSR under; the row arrays are null where it
    /// holds a copy of them
    types_gen: u64,
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

type Hm96TreeCb = unsafe extern "C" fn(*mut c_void, i32, i64) -> *const u8;

type ZkWordsCb = unsafe extern "C" fn(*mut c_void, *mut *const u64, *mut *const u64) -> i32;

/// Level 0's padding and extra lanes in page-locked memory ([`Words`]), copied on a thread of their own while the device builds
/// the witness: [`zk_words`] waits for it.
#[cfg(feature = "sha512")]
struct ZkWords<'s> {
    pending: Option<std::thread::ScopedJoinHandle<'s, (Words, Words)>>,
    words: Option<(Words, Words)>,
}

#[cfg(feature = "sha512")]
unsafe extern "C" fn zk_words(ctx: *mut c_void, pads: *mut *const u64, extra: *mut *const u64) -> i32 {
    // SAFETY: `ctx` is the `ZkWords` that `prove_circuit` keeps until the prover returns
    let w = unsafe { &mut *(ctx as *mut ZkWords<'_>) };
    if let Some(h) = w.pending.take() {
        w.words = h.join().ok();
    }
    let Some((p, e)) = &w.words else { return 1 };
    // SAFETY: the device passes two writable pointers
    unsafe { (*pads, *extra) = (p.as_ptr(), e.as_ptr()) };
    0
}

/// The salts the device's trees take (`hm96_tree`): the table's, and every tree's handed to the device so far, which its
/// openings read until the prover returns.
#[cfg(feature = "sha512")]
struct Hm96Trees<'a> {
    salts: &'a std::sync::Mutex<crate::zk_hooks::LeafSalts>,
    given: std::sync::Mutex<Vec<std::sync::Arc<Vec<u8>>>>,
}

/// This proof's k-th tree's salts, `leaves` × 192 bytes: level 0's for tree 0 (both reps share them), a fresh tree's for every
/// later one. Null when they cannot be read (the device refuses the tree).
#[cfg(feature = "sha512")]
unsafe extern "C" fn hm96_tree(ctx: *mut c_void, tree: i32, leaves: i64) -> *const u8 {
    let h = unsafe { &*(ctx as *const Hm96Trees<'_>) };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut s = h.salts.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let b = match tree {
            0 => s.level0(leaves as usize),
            _ => {
                let id = s.next_id();
                s.take(id, leaves as usize)
            }
        };
        let p = b.as_ptr();
        h.given.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(b);
        p
    }))
    .unwrap_or(std::ptr::null())
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
    fn flock_cuda_hm96_leaves(x: *const u8, salts: *const u8, n: i64, prefixes: *const u8, tables: *const u64, out: *mut u8) -> i32;
    #[cfg(feature = "sha512")]
    fn flock_cuda_hm96_slots(inputs: *const u64, in_words: i32, n: i64, plan: *const u32, useful: i32, tape_words: i32, mid: *const u64,
                             tables: *const u64, slot_log: i32, z: *mut u64, a: *mut u64, b: *mut u64) -> i32;
    fn flock_cuda_circuit_kept(tag: u64) -> i32;
    fn flock_cuda_host_alloc(bytes: usize) -> *mut c_void;
    fn flock_cuda_host_register(p: *mut c_void, bytes: usize) -> i32;
    fn flock_cuda_host_unregister(p: *mut c_void);
    fn flock_cuda_dev_alloc(bytes: usize) -> *mut c_void;
    fn flock_cuda_copy(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    fn flock_cuda_dev_zero(p: *mut c_void, bytes: usize) -> i32;
    fn flock_cuda_types_upload(types_gen: u64, set: i32, h: *const *const u32, len: *const usize, n: usize, host: u64, key: *const u8) -> i32;
    fn flock_cuda_types_held(types_gen: u64, set: i32) -> i32;
}

/// Rep 1 proves from rep 0's level-0 commitment, which both reps bind (one level-0 root): rep 0 keeps it on the device for the
/// session's tag, and rep 1 takes it and builds its witness again. Where nothing was kept, rep 1 proves in full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reuse {
    None,
    Keep(u64),
    Take(u64),
}

/// Whether the device holds a rep 0's level-0 commitment for `tag`.
pub fn kept(tag: u64) -> bool {
    unsafe { flock_cuda_circuit_kept(tag) != 0 }
}

/// Salts page-locked for the device's copies, unlocked when dropped; holding them, so they are never freed while locked.
pub struct Pinned(std::sync::Arc<Vec<u8>>);

impl Pinned {
    /// `b` page-locked; None where the runtime refuses (its copies then stage through pageable memory)
    pub fn new(b: &std::sync::Arc<Vec<u8>>) -> Option<Self> {
        (!b.is_empty() && unsafe { flock_cuda_host_register(b.as_ptr() as *mut c_void, b.len()) } != 0).then(|| Pinned(b.clone()))
    }

    pub fn bytes(&self) -> usize {
        self.0.len()
    }

    /// Its `n` bytes rewritten in place by `fill` and shared again, still page-locked; None, and `fill` not called, while
    /// anything else holds them or at another size.
    pub fn refill(&mut self, n: usize, fill: &mut dyn FnMut(&mut [u8])) -> Option<std::sync::Arc<Vec<u8>>> {
        let b = std::sync::Arc::get_mut(&mut self.0).filter(|b| b.len() == n)?;
        fill(b);
        Some(self.0.clone())
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe { flock_cuda_host_unregister(self.0.as_ptr() as *mut c_void) }
    }
}

/// The device's unit kernel syncs once per AND-level for 32 units per CTA, so a deep unit leaves the device idle (a flattened
/// GEMM coordinate has 305 levels per tensor-core step), while the host's `eval64` walks its rows once per 64 units. A unit at
/// least this deep is evaluated on the host; `FC_HOST_UNITS=0|1` overrides.
pub const HOST_UNIT_LEVELS: usize = 2048;

/// A slot circuit's native witness plan (`sha512_native`): its rows' `(a, b)` sources as `u32` pairs, and the tape's words per
/// slot. Derived and checked against `eval64` once per process (the slot circuit is pinned); `None` if the plan cannot be derived.
#[cfg(feature = "sha512")]
fn plan_of(cell: &'static OnceLock<Option<(Vec<u32>, i32)>>, net: &IrUnitNet) -> Option<&'static (Vec<u32>, i32)> {
    cell.get_or_init(|| match crate::sha512_native::NativeSha::derive(net) {
        Ok(p) => Some((p.rows.iter().flat_map(|&(a, b)| [a, b]).collect(), p.roles.len() as i32)),
        Err(e) => {
            eprintln!("the native witness of {} is off: {e}", net.name);
            None
        }
    })
    .as_ref()
}

/// The compression slots' (sha512x3) native witness plan.
#[cfg(feature = "sha512")]
fn native_plan(st: &Stmt) -> Option<&'static (Vec<u32>, i32)> {
    static PLAN: OnceLock<Option<(Vec<u32>, i32)>> = OnceLock::new();
    plan_of(&PLAN, &st.c.nets[st.c.sha_net].1)
}

/// The hm96 slots' native witness plan.
#[cfg(feature = "sha512")]
fn hm96_plan(st: &Stmt) -> Option<&'static (Vec<u32>, i32)> {
    static PLAN: OnceLock<Option<(Vec<u32>, i32)>> = OnceLock::new();
    plan_of(&PLAN, &st.c.nets[st.c.hm_net].1)
}

/// Whether the device builds the hm96 slots natively beside the compressions (both plans derived), so the host hands it their
/// input words ([`HostSlots::hm96_rows`]) rather than their witness; not with `FC_HM96_HOST=1`.
pub fn hm96_native(st: &Stmt) -> bool {
    #[cfg(feature = "sha512")]
    return std::env::var("FC_HM96_HOST").as_deref() != Ok("1") && native_plan(st).is_some() && hm96_plan(st).is_some();
    #[cfg(not(feature = "sha512"))]
    {
        let _ = st;
        false
    }
}

/// SHA-512's chaining value after hm96-sha512's salt prefix block: where every hm96 slot's salt compressions start.
#[cfg(feature = "sha512")]
fn hm96_midstate() -> &'static [u64; 8] {
    static MID: OnceLock<[u64; 8]> = OnceLock::new();
    MID.get_or_init(|| {
        let mut mid = crate::circuit::SHA512_IV;
        crate::circuit::compress(&mut mid, &flock_merkle::hm96::consts().salt_prefix);
        mid
    })
}

/// The hm96 slots' words as the device builds them natively (`flock_cuda_hm96_slots`, the kernels `fc_witness` runs), from
/// their input words (`Stmt::row_inputs`' hm96 slots): a test vector against the host's (`Stmt::slot_zab`).
#[cfg(feature = "sha512")]
pub fn device_hm96_slots(st: &Stmt, ins: &[Vec<F128>]) -> Result<Vec<(Vec<F128>, Vec<F128>, Vec<F128>)>, String> {
    let net = &st.c.nets[st.c.hm_net].1;
    let (plan, words) = hm96_plan(st).ok_or("the hm96 slots have no native plan")?;
    let uw = 1usize << (net.unit_log - 7);
    let flat: Vec<u64> = ins.iter().flat_map(|w| {
        assert_eq!(w.len(), net.in_words, "an hm96 slot's input words");
        w.iter().flat_map(|x| [x.lo, x.hi])
    }).collect();
    let (mut z, mut a, mut b) = (vec![0u64; 2 * uw * ins.len()], vec![0u64; 2 * uw * ins.len()], vec![0u64; 2 * uw * ins.len()]);
    let rc = unsafe {
        flock_cuda_hm96_slots(flat.as_ptr(), net.in_words as i32, ins.len() as i64, plan.as_ptr(), net.useful as i32, *words,
                              hm96_midstate().as_ptr(), flock_merkle::hm96::consts().tables.as_ptr() as *const u64, net.unit_log as i32,
                              z.as_mut_ptr(), a.as_mut_ptr(), b.as_mut_ptr())
    };
    if rc != 0 {
        return Err(format!("the device's hm96 slots failed ({rc})"));
    }
    let slot = |v: &[u64], s: usize| v[2 * uw * s..2 * uw * (s + 1)].chunks_exact(2).map(|p| F128::new(p[0], p[1])).collect::<Vec<F128>>();
    Ok((0..ins.len()).map(|s| (slot(&z, s), slot(&a, s), slot(&b, s))).collect())
}

/// Whether the statement's units are evaluated on the host (`HOST_UNIT_LEVELS`).
pub fn deep_unit(st: &Stmt) -> bool {
    // the device kernel evaluates rows by column; a unit committed at a position map is evaluated in its order, on the host
    if st.c.unit().order.is_some() {
        return true;
    }
    match std::env::var("FC_HOST_UNITS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => types(st).sched.level_off.len() - 1 >= HOST_UNIT_LEVELS,
    }
}

/// The hm96 leaves of the digests `x` (64 bytes each) under the host's `salts` (192 bytes each) as the device finishes a tree's
/// leaves: a test vector against `flock_merkle::hm96::leaf`.
#[cfg(feature = "sha512")]
pub fn device_leaves(x: &[u8], salts: &[u8]) -> Vec<u8> {
    let n = x.len() / 64;
    assert!(x.len() == 64 * n && salts.len() == 192 * n, "a 64-byte digest and a 192-byte salt per leaf");
    let pre = { let c = flock_merkle::hm96::consts(); [c.salt_prefix, c.leaf_prefix].concat() };
    let mut out = vec![0u8; 64 * n];
    let rc = unsafe {
        flock_cuda_hm96_leaves(x.as_ptr(), salts.as_ptr(), n as i64, pre.as_ptr(), flock_merkle::hm96::consts().tables.as_ptr() as *const u64,
                               out.as_mut_ptr())
    };
    assert_eq!(rc, 0, "the device's hm96 leaves failed ({rc})");
    out
}

/// A slot whose witness the host computed: its absolute F128 word offset and its z, a, b words.
pub type HostSlot = (usize, Vec<F128>, Vec<F128>, Vec<F128>);

/// u64 words the host packs for the device: page-locked and mapped (the device reads them directly, beside the compression
/// witness), or a heap `Vec` (three pageable copies per slot). Mapped buffers go back to a pool when dropped: pinning a
/// witness's host slots costs more than a rep's upload saves, so a process pins its largest few once.
pub enum HostBuf {
    Mapped { ptr: std::ptr::NonNull<u64>, len: usize, cap: usize },
    Heap(Vec<u64>),
}

// SAFETY: a mapped buffer is owned memory like a Vec's; the device reads it only inside prove_circuit, while it is borrowed
unsafe impl Send for HostBuf {}
unsafe impl Sync for HostBuf {}

/// Mapped buffers dropped by earlier witnesses: (address, capacity in u64)
static HOST_POOL: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

impl HostBuf {
    /// `len` words of mapped host memory, from the pool or newly pinned; None where the runtime refuses or `FC_HOST_PINNED=0`
    pub fn mapped(len: usize) -> Option<Self> {
        if len == 0 || std::env::var("FC_HOST_PINNED").as_deref() == Ok("0") {
            return None;
        }
        let mut pool = HOST_POOL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let best = pool.iter().enumerate().filter(|(_, e)| e.1 >= len).min_by_key(|(_, e)| e.1).map(|(i, _)| i);
        let (addr, cap) = match best {
            Some(i) => pool.swap_remove(i),
            None => {
                drop(pool);
                let p = unsafe { flock_cuda_host_alloc(len * 8) } as usize;
                (p, len)
            }
        };
        std::ptr::NonNull::new(addr as *mut u64).map(|ptr| HostBuf::Mapped { ptr, len, cap })
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self, HostBuf::Mapped { .. })
    }

    /// `count` buffers of `len` words pinned into the pool (or taken from it and put back), for witnesses built ahead
    pub fn prepin(len: usize, count: usize) {
        let held: Vec<HostBuf> = (0..count).map_while(|_| HostBuf::mapped(len)).collect();
        drop(held);
    }

    /// `parts` back to back in mapped memory from the pool ([`crate::concat_into`]), which the device copies from at full
    /// speed rather than through pageable staging; None where the runtime refuses or with `FC_COPY_PINNED=0`
    pub fn concat<P: AsRef<[F128]>>(parts: &[P]) -> Option<Self> {
        if std::env::var("FC_COPY_PINNED").as_deref() == Ok("0") {
            return None;
        }
        let n: usize = parts.iter().map(|p| p.as_ref().len()).sum();
        let mut b = HostBuf::mapped(2 * n)?;
        // SAFETY: an F128 is two u64 words (lo, hi), and mapped memory is page-aligned
        let out = unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr() as *mut std::mem::MaybeUninit<F128>, n) };
        crate::concat_into(parts, out);
        Some(b)
    }
}

/// Words the device copies from the host once a rep: in mapped memory from the pool ([`HostBuf::concat`]), else on the heap
#[cfg(feature = "sha512")]
enum Words {
    Pinned(HostBuf),
    Heap(Vec<F128>),
}

#[cfg(feature = "sha512")]
impl Words {
    fn of(parts: &[Vec<F128>]) -> Self {
        HostBuf::concat(parts).map_or_else(|| Words::Heap(crate::concat(parts)), Words::Pinned)
    }

    fn as_ptr(&self) -> *const u64 {
        match self {
            Words::Pinned(b) => b.as_ptr(),
            Words::Heap(v) => v.as_ptr() as *const u64,
        }
    }
}

impl std::ops::Deref for HostBuf {
    type Target = [u64];
    fn deref(&self) -> &[u64] {
        match self {
            HostBuf::Mapped { ptr, len, .. } => unsafe { std::slice::from_raw_parts(ptr.as_ptr(), *len) },
            HostBuf::Heap(v) => v,
        }
    }
}

impl std::ops::DerefMut for HostBuf {
    fn deref_mut(&mut self) -> &mut [u64] {
        match self {
            HostBuf::Mapped { ptr, len, .. } => unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), *len) },
            HostBuf::Heap(v) => v,
        }
    }
}

impl Drop for HostBuf {
    fn drop(&mut self) {
        if let HostBuf::Mapped { ptr, cap, .. } = self {
            HOST_POOL.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((ptr.as_ptr() as usize, *cap));
        }
    }
}

impl Clone for HostBuf {
    fn clone(&self) -> Self {
        match self {
            HostBuf::Mapped { len, .. } => {
                let mut c = HostBuf::mapped(*len).unwrap_or_else(|| HostBuf::Heap(vec![0; *len]));
                c.copy_from_slice(self);
                c
            }
            HostBuf::Heap(v) => HostBuf::Heap(v.clone()),
        }
    }
}

impl Default for HostBuf {
    fn default() -> Self {
        HostBuf::Heap(Vec::new())
    }
}

/// Device memory holding a copy of host words ([`HostSlots::upload`]). Dropped buffers are zeroed and go back to a pool, not to
/// `cudaFree`, which would synchronize the device under the session proving there; a process allocates at most
/// `FC_DEV_PREFETCH_GB` (default 16) of them.
pub struct DevBuf {
    ptr: usize,
    len: usize,
    cap: usize,
}

/// Device buffers dropped by earlier witnesses (address, capacity in u64), and the bytes allocated
static DEV_POOL: std::sync::Mutex<(Vec<(usize, usize)>, usize)> = std::sync::Mutex::new((Vec::new(), 0));

impl DevBuf {
    /// A device copy of `words`, from the pool or newly allocated; None where the runtime refuses or the bound is reached
    pub fn copy_of(words: &[u64]) -> Option<Self> {
        let len = words.len();
        if len == 0 {
            return None;
        }
        let bound = std::env::var("FC_DEV_PREFETCH_GB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(16) << 30;
        let mut pool = DEV_POOL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let best = pool.0.iter().enumerate().filter(|(_, e)| e.1 >= len).min_by_key(|(_, e)| e.1).map(|(i, _)| i);
        let (ptr, cap) = match best {
            Some(i) => pool.0.swap_remove(i),
            None if pool.1 + 8 * len <= bound => {
                let p = unsafe { flock_cuda_dev_alloc(8 * len) } as usize;
                if p == 0 {
                    return None;
                }
                pool.1 += 8 * len;
                (p, len)
            }
            None => return None,
        };
        drop(pool);
        let d = DevBuf { ptr, len, cap };
        (unsafe { flock_cuda_copy(ptr as *mut c_void, words.as_ptr() as *const c_void, 8 * len) } == 0).then_some(d)
    }

    pub fn as_ptr(&self) -> *const u64 {
        self.ptr as *const u64
    }

    /// its length in u64 words
    pub fn words(&self) -> usize {
        self.len
    }

    /// Whether the device holds `words`
    pub fn holds(&self, words: &[u64]) -> bool {
        let mut v = vec![0u64; self.len];
        words.len() == self.len
            && unsafe { flock_cuda_copy(v.as_mut_ptr() as *mut c_void, self.ptr as *const c_void, 8 * self.len) } == 0
            && v == words
    }
}

impl Drop for DevBuf {
    /// A buffer the device can't zero never goes back to the pool: its words stay out of any later witness's copy
    fn drop(&mut self) {
        if unsafe { flock_cuda_dev_zero(self.ptr as *mut c_void, 8 * self.cap) } != 0 {
            return;
        }
        DEV_POOL.lock().unwrap_or_else(std::sync::PoisonError::into_inner).0.push((self.ptr, self.cap));
    }
}

/// Host slots as the device takes them (`host0`): each slot's absolute F128 word offset and word count, and every slot's words
/// back to back. In mapped memory a slot is its a, b words when its z is a & b word for word (an honest witness's, since each
/// row's output is its own bit), and the device forms z; otherwise, and in heap memory, its z, a, b words. Packed once per
/// witness, so a rep hands the device pointers rather than a copy.
#[derive(Default)]
pub struct HostSlots {
    pub at: Vec<i64>,
    pub words: Vec<i32>,
    pub zab: HostBuf,
    /// `zab` holds a, b per slot (z = a & b)
    pub ab: bool,
    /// device copies of `zab` and of the compressions' input rows ([`HostSlots::upload`]), which the device reads instead
    pub dev_zab: Option<DevBuf>,
    pub dev_rows: Option<DevBuf>,
    /// a page-locked copy of the compressions' input rows ([`HostSlots::pin_rows`]), which the device copies at full speed
    pub pinned_rows: Option<HostBuf>,
    /// the hm96 slots' input words (`[blocks][hm96 slots per block][in_words]`, `Stmt::row_inputs`), from which the device builds
    /// them natively ([`hm96_native`]); empty: the hm96 slots are among the host slots
    pub hm96_rows: Vec<F128>,
}

impl Clone for HostSlots {
    /// The host words only: a clone may be changed, so it takes no device copy
    fn clone(&self) -> Self {
        HostSlots { at: self.at.clone(), words: self.words.clone(), zab: self.zab.clone(), ab: self.ab, dev_zab: None, dev_rows: None,
            pinned_rows: None, hm96_rows: self.hm96_rows.clone() }
    }
}

impl HostSlots {
    fn of(at: Vec<i64>, words: Vec<i32>, zab: HostBuf, ab: bool) -> Self {
        HostSlots { at, words, zab, ab, dev_zab: None, dev_rows: None, pinned_rows: None, hm96_rows: Vec::new() }
    }

    /// Copy `rows` (the compressions' input rows) into page-locked memory ([`HostBuf::concat`]), which the device copies from
    /// at full speed rather than through pageable staging; for mapped host slots only, and not with `FC_ROWS_PINNED=0`
    pub fn pin_rows(&mut self, rows: &[F128]) {
        if !self.zab.is_mapped() || std::env::var("FC_ROWS_PINNED").as_deref() == Ok("0") {
            return;
        }
        self.pinned_rows = HostBuf::concat(&[rows]);
    }

    /// Copy the a, b host slots and `rows` (the compressions' input rows the device takes beside them) to device memory, which
    /// the device then reads in place of the mapped host memory and the pageable rows. The rows go through mapped memory from
    /// the pool ([`HostBuf::concat`]), back in the pool once copied. For a witness built ahead of its session
    /// (`FC_DEV_PREFETCH=1`), the copies run beside the previous session's device work. Whatever the device has no room for
    /// stays on the host.
    pub fn upload(&mut self, rows: &[F128]) {
        if !(self.zab.is_mapped() && self.ab) {
            return;
        }
        let t = Instant::now();
        let staged = HostBuf::concat(&[rows]);
        // SAFETY: an F128 is two u64 words (lo, hi), as the device takes the rows
        let r: &[u64] = match &staged {
            Some(b) => &b[..],
            None => unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u64, 2 * rows.len()) },
        };
        let rows_bytes = 8 * r.len();
        self.dev_zab = DevBuf::copy_of(&self.zab);
        self.dev_rows = DevBuf::copy_of(r);
        let rows_pinned = staged.is_some();
        if self.dev_rows.is_none() {
            self.pinned_rows = staged;
        }
        if crate::circuit::unit_profile() {
            eprintln!("UNITPROF\t{}", serde_json::json!({"part": "upload", "zab_bytes": 8 * self.zab.len(), "rows_bytes": rows_bytes,
                "zab": self.dev_zab.is_some(), "rows": self.dev_rows.is_some(), "rows_pinned": rows_pinned,
                "wall_s": t.elapsed().as_secs_f64()}));
        }
    }

    pub fn pack(slots: &[HostSlot]) -> Self {
        Self::pack_in(slots, HostBuf::mapped)
    }

    /// z, a, b per slot in heap memory: the device copies each slot (the path before mapped host slots)
    pub fn pack_pageable(slots: &[HostSlot]) -> Self {
        Self::pack_in(slots, |_| None)
    }

    /// [`HostSlots::pack`] of `head` and then circuit `ni`'s slots from their input words (`host_slots(st, ni, ins)`), the
    /// latter's a, b words written into mapped memory straight from their lane groups (`unit_ab_into`) rather than first as
    /// z, a, b vectors: the same slots, layout and words. `FC_UNIT_FUSED=0` packs `host_slots` instead; `FC_UNIT_CHECK=1`
    /// packs both and asserts they are equal.
    pub fn pack_units(st: &Stmt, head: &[HostSlot], ni: usize, ins: &[Vec<F128>]) -> Self {
        let unfused = || {
            let mut all = head.to_vec();
            all.extend(host_slots(st, ni, ins));
            Self::pack(&all)
        };
        if std::env::var("FC_UNIT_FUSED").as_deref() == Ok("0") || st.c.lookups.contains_key(&ni) {
            return unfused();
        }
        let t = Instant::now();
        let (r, wpb) = (st.c.range_of(ni), 1usize << (st.c.k_log - 7));
        let uw = 1usize << (st.c.nets[ni].1.unit_log - 7);
        let mut words: Vec<i32> = head.iter().map(|s| s.1.len() as i32).collect();
        let hw: usize = words.iter().map(|&w| w as usize).sum();
        words.extend(std::iter::repeat_n(uw as i32, ins.len()));
        let at: Vec<i64> = head.iter().map(|s| s.0 as i64)
            .chain((0..ins.len()).map(|x| ((x / r.count) * wpb + r.slot(x % r.count) / 128) as i64)).collect();
        let Some(mut zab) = HostBuf::mapped(4 * (hw + uw * ins.len())) else { return unfused() };
        let (h, u) = zab.split_at_mut(4 * hw);
        if !(Self::fill(h, &words[..head.len()], head, true) && crate::circuit::unit_ab_into(&st.c.nets[ni].1, ni, uw, ins, u)) {
            drop(zab);
            return unfused();
        }
        let got = HostSlots::of(at, words, zab, true);
        if crate::circuit::unit_profile() {
            eprintln!("UNITPROF\t{}", serde_json::json!({"part": "pack_units", "slots": got.words.len(), "layout": got.layout(), "wall_s": t.elapsed().as_secs_f64()}));
        }
        if std::env::var("FC_UNIT_CHECK").as_deref() == Ok("1") {
            let want = unfused();
            assert!(want.at == got.at && want.words == got.words && want.ab == got.ab && want.zab[..] == got.zab[..],
                    "circuit {ni}: the fused host slots are not host_slots' packed");
        }
        got
    }

    /// With `mapped` (a device-readable buffer of n words, or None), a, b per slot when every z is a & b; else z, a, b in heap memory
    fn pack_in(slots: &[HostSlot], mapped: impl Fn(usize) -> Option<HostBuf>) -> Self {
        let t = Instant::now();
        let h = Self::pack_slots(slots, mapped);
        if crate::circuit::unit_profile() {
            eprintln!("UNITPROF\t{}", serde_json::json!({"part": "pack", "slots": slots.len(), "layout": h.layout(), "wall_s": t.elapsed().as_secs_f64()}));
        }
        h
    }

    fn pack_slots(slots: &[HostSlot], mapped: impl Fn(usize) -> Option<HostBuf>) -> Self {
        let words: Vec<i32> = slots.iter().map(|s| s.1.len() as i32).collect();
        let at = slots.iter().map(|s| s.0 as i64).collect();
        let total: usize = words.iter().map(|&w| w as usize).sum();
        if let Some(mut zab) = mapped(4 * total) {
            if Self::fill(&mut zab, &words, slots, true) {
                return HostSlots::of(at, words, zab, true);
            }
            if let Some(mut zab) = mapped(6 * total) {
                Self::fill(&mut zab, &words, slots, false);
                return HostSlots::of(at, words, zab, false);
            }
        }
        let mut zab = HostBuf::Heap(vec![0u64; 6 * total]);
        Self::fill(&mut zab, &words, slots, false);
        HostSlots::of(at, words, zab, false)
    }

    /// Every slot's words into `out` (a, b with `ab`, else z, a, b); with `ab`, whether every z word is a & b
    fn fill(out: &mut [u64], words: &[i32], slots: &[HostSlot], ab: bool) -> bool {
        let per = if ab { 4 } else { 6 };
        let mut rest: &mut [u64] = out;
        let mut dst = Vec::with_capacity(slots.len());
        for &w in words {
            let (a, b) = std::mem::take(&mut rest).split_at_mut(per * w as usize);
            dst.push(a);
            rest = b;
        }
        dst.into_par_iter().zip(slots.par_iter()).all(|(d, (_, z, a, b))| {
            let w = z.len();
            assert!(a.len() == w && b.len() == w, "a host slot's z, a and b are not one length");
            let vs: &[&Vec<F128>] = if ab { &[a, b] } else { &[z, a, b] };
            for (k, v) in vs.iter().enumerate() {
                for (o, x) in d[2 * k * w..2 * (k + 1) * w].chunks_exact_mut(2).zip(v.iter()) {
                    (o[0], o[1]) = (x.lo, x.hi);
                }
            }
            !ab || z.iter().zip(a).zip(b).all(|((z, a), b)| z.lo == a.lo & b.lo && z.hi == a.hi & b.hi)
        })
    }

    /// How the device takes them: `device-ab` (from its copy, z formed there), `mapped-ab` (read directly, z formed there),
    /// `mapped` (read directly), `pageable` (copied)
    pub fn layout(&self) -> &'static str {
        match (self.zab.is_mapped(), self.ab) {
            _ if self.dev_zab.is_some() => "device-ab",
            (true, true) => "mapped-ab",
            (true, false) => "mapped",
            _ => "pageable",
        }
    }
}

/// Where a rep's witness comes from.
pub enum CircuitSrc<'a> {
    /// The whole packed `(z, a, b)` from the host (any witness, a tampered one included).
    Host { z: &'a [F128], a: &'a [F128], b: &'a [F128] },
    /// A cut template: as `Device`, the units in two passes around `mid`.
    DeviceCut { rows: &'a [F128], host: &'a HostSlots, inputs: &'a [F128], mid: Mid<'a> },
    /// Every compression slot's input words (`[blocks][slots per block][in_words]`, `Stmt::row_inputs`), the hm96 slots' witness
    /// (host slots) or input words (`HostSlots::hm96_rows`), and every unit's input words (`[blocks][units per block][in_words]`):
    /// the honest witness, built on the device.
    /// With `inputs` empty, every unit slot is among the host slots (`deep_unit`).
    Device { rows: &'a [F128], host: &'a HostSlots, inputs: &'a [F128] },
}

/// The unit's expanded circuit's level schedule: level 0 the input rows and the constant, a row one past its deepest operand
/// (itself, in an assertion row, excluded). The device reads its rows beside it as CSR ([`csr`]).
pub struct UnitSched {
    /// levels [0, levels0) read no late input row; the rest do
    pub levels0: usize,
    pub late_rows: Vec<u32>,
    pub cut_rows: Vec<u32>,
    pub level_off: Vec<u32>,
    pub order: Vec<u32>,
}

/// The unit's schedule; a cut unit's (a second input group: its late rows, the cut words the tail computes; a second output
/// group: its cut output rows) in two phases: the rows that read no late row, then the rest.
pub fn schedule(net: &IrUnitNet) -> UnitSched {
    schedule_of(net, &net.a, &net.b)
}

/// [`schedule`] of `net`'s layout with its rows `a` and `b` read through [`RowRead`].
fn schedule_of<R: RowRead + ?Sized>(net: &IrUnitNet, a: &R, b: &R) -> UnitSched {
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
            late[i] = a.row_cols(i).chain(b.row_cols(i)).any(|c| c != i && c != cpos && late[c]);
        }
    }
    if cut_rows.iter().any(|&r| late[r as usize]) {
        // not two-phase (a cut output reads a late input: attention's and GEMM's tensor-core steps): one phase; such a unit's
        // witness is the host's (flock-circuit keeps it off the device)
        late_rows.clear();
        cut_rows.clear();
        late = vec![false; u];
    }
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
            let l = 1 + a.row_cols(i).chain(b.row_cols(i)).filter(|&c| c != i && c != cpos && late[c] == phase && c >= first).map(|c| level[c]).max().unwrap_or(0);
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
    UnitSched { levels0, late_rows, cut_rows, level_off, order }
}

/// A slot type's circuit as the device reads it, CSC over its slot's columns: A's and B's column pointers and nonzero counts,
/// and their row indices, the bulk ([`Types::types_gen`]'s set 0). The device holds one copy of them per GPU, which every prover
/// there with the same slot types maps ([`arrays_key`]), and they're freed here. With `FC_CSC_HOST_MIB=M`, rows of at least M
/// MiB stay here instead, page-locked and mapped, where the fold reads them over the bus each rep.
struct TypeCsc {
    col_ptr: [Vec<u32>; 2],
    nnz: [u32; 2],
    /// A's and B's row indices: `None` once the device holds a copy of them
    rows: Option<[Vec<u32>; 2]>,
}

struct Types {
    /// The key the device holds this build's slot types' CSC (set 0) and its unit's CSR (set 1) under (`FcTypeUploads`,
    /// prove_circuit.cuh): drawn once from a counter that only goes up, so no two builds share one, and a set is read from the
    /// host only while the device lacks it. A freed host array can then never be taken for another's, as an address could.
    types_gen: u64,
    logs: Vec<i32>,
    csc: Vec<TypeCsc>,
    ranges: Vec<(i32, i32, i32)>,
    u_range: usize,
    sched: UnitSched,
    h_range: usize,
    hsched: UnitSched,
    /// the compression circuit's rows as CSR, A's then B's: small, and kept, since the device keys its copy by their addresses
    /// (`FcUploads`)
    hcsr: [(Vec<u32>, Vec<u32>); 2],
}

/// A digest of a set's arrays: the key under which every prover on a GPU shares one device copy of them (`fc_share_get`,
/// prove_circuit.cuh's csc_share.cuh), so a copy another prover made holds exactly these arrays.
fn arrays_key(arrays: &[&Vec<u32>]) -> [u8; 64] {
    use sha2::{Digest, Sha512};
    let mut h = Sha512::new();
    h.update(b"verity flock-circuit: the slot types' CSC");
    h.update((arrays.len() as u64).to_le_bytes());
    for a in arrays {
        let bytes = unsafe { std::slice::from_raw_parts(a.as_ptr().cast::<u8>(), 4 * a.len()) };
        let pieces: Vec<[u8; 64]> = bytes.par_chunks(1 << 26).map(|p| Sha512::digest(p).into()).collect();
        h.update((a.len() as u64).to_le_bytes());
        pieces.iter().for_each(|p| h.update(p));
    }
    h.finalize().into()
}

/// The circuit's slot types, built once per circuit and kept for the process. Their CSC goes to the device as it is built, one
/// copy per GPU ([`TypeCsc`]); the unit's rows as CSR are built only for a rep whose units the device computes, until it holds
/// them ([`prove_circuit`]).
fn types(st: &Stmt) -> &'static Types {
    static T: Mutex<Vec<([u8; 64], &'static Types)>> = Mutex::new(Vec::new());
    static GEN: AtomicU64 = AtomicU64::new(1);
    let mut held = T.lock().unwrap();
    if let Some(&(_, t)) = held.iter().find(|(k, _)| *k == st.c.sha512) {
        return t;
    }
    let t: &'static Types = Box::leak(Box::new({
        let c = &st.c;
        let types_gen = GEN.fetch_add(1, Ordering::Relaxed);
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
                    let ((ac, ar), (bc, br)) = match r.kind {
                        Kind::Net(i) => {
                            let net = &c.nets[i].1;
                            let n = 1usize << net.unit_log;
                            (csc_of(&net.a, n), csc_of(&net.b, n))
                        }
                        Kind::Mask => {
                            let n = 1usize << r.slot_log;
                            let id = csc_from_rows(&SparseBinaryMatrix { num_rows: n, num_cols: n, rows: (0..n).map(|i| vec![i]).collect::<Vec<_>>().into() });
                            (id.clone(), id)
                        }
                    };
                    // the device's fold reads 2^slot_log + 1 column pointers of each
                    assert!(ac.len() == (1 << r.slot_log) + 1 && bc.len() == ac.len(), "slot type {:?}: its CSC is not over its slot's columns", r.kind);
                    csc.push(TypeCsc { nnz: [ar.len() as u32, br.len() as u32], col_ptr: [ac, bc], rows: Some([ar, br]) });
                    key.len() - 1
                }
            };
            ranges.push((t as i32, r.pos0 as i32, r.count as i32));
        }
        assert!(logs.len() <= FC_MAX_TYPES && ranges.len() <= FC_MAX_RANGES, "too many slot types or ranges for the FFI");
        let u_range = c.ranges.iter().position(|r| r.kind == Kind::Net(c.unit_net)).unwrap();
        let h_range = c.ranges.iter().position(|r| r.kind == Kind::Net(c.sha_net)).unwrap();
        // set 0 in the fold's order: per type A's column pointers and rows, then B's
        let arrays: Vec<&Vec<u32>> =
            csc.iter().flat_map(|t: &TypeCsc| { let r = t.rows.as_ref().unwrap(); [&t.col_ptr[0], &r[0], &t.col_ptr[1], &r[1]] }).collect();
        let (h, len): (Vec<*const u32>, Vec<usize>) = arrays.iter().map(|v| (v.as_ptr(), v.len())).unzip();
        // FC_CSC_HOST_MIB=M: a type's rows (its odd arrays) of at least M MiB stay here, which the leaked `Types` keeps, unmoved,
        // for the process; unset, the device holds every type's rows
        let min = std::env::var("FC_CSC_HOST_MIB").ok().and_then(|v| v.parse::<usize>().ok()).map(|m| m.saturating_mul(1 << 20));
        let mapped: Vec<bool> = csc.iter().map(|t| min.is_some_and(|m| 4 * (t.nnz[0] as usize + t.nnz[1] as usize) >= m)).collect();
        let host = mapped.iter().enumerate().filter(|(_, m)| **m).fold(0u64, |b, (t, _)| b | 0b1010 << (4 * t));
        if mapped.contains(&true) {
            let words: usize = csc.iter().zip(&mapped).filter(|(_, m)| **m).map(|(t, _)| t.nnz[0] as usize + t.nnz[1] as usize).sum();
            eprintln!("the slot types' rows read from the host, mapped: {} MiB ({} of {} types)", (4 * words) >> 20,
                      mapped.iter().filter(|m| **m).count(), csc.len());
        }
        // the device's copy is shared by every prover on its GPU whose slot types' CSC is this one (FC_CSC_SHARE=0: a copy of
        // this process's own)
        let key = (host == 0 && std::env::var("FC_CSC_SHARE").map_or(true, |v| v != "0")).then(|| arrays_key(&arrays));
        let key_ptr = key.as_ref().map_or(null(), |k| k.as_ptr());
        // a device that can't take them now keeps them on the host, and the first rep's fold copies them under the same key
        if unsafe { flock_cuda_types_upload(types_gen, 0, h.as_ptr(), len.as_ptr(), h.len(), host, key_ptr) } == 0 {
            csc.iter_mut().zip(&mapped).filter(|(_, m)| !**m).for_each(|(t, _)| t.rows = None);
        }
        let sha = &c.nets[c.sha_net].1;
        Types { types_gen, logs, csc, ranges, u_range, sched: schedule(c.unit()), h_range, hsched: schedule(sha), hcsr: [csr(&sha.a), csr(&sha.b)] }
    }));
    held.push((st.c.sha512, t));
    t
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
    let t = Instant::now();
    let h: Vec<HostSlot> = st.slot_zab(ni, &st.c.nets[ni].1, ins).into_iter().map(|(x, z, a, b)| ((x / r.count) * wpb + r.slot(x % r.count) / 128, z, a, b)).collect();
    if crate::circuit::unit_profile() {
        eprintln!("UNITPROF\t{}", serde_json::json!({"part": "host_slots", "circuit": ni, "slots": ins.len(), "wall_s": t.elapsed().as_secs_f64()}));
    }
    h
}

/// One rep on the GPU: proof, commitment, and the phase split (witness, commit, link, zerocheck, lincheck, ring switch,
/// ligerito, total; seconds), the device witness's own split (compressions, units), and the host's around the device (its
/// setup: parameters and schedule; staging the witness's inputs and host slots; decoding the proof). Under M1 (`Live::Zk`) the proof is the
/// unmasked one the CPU prover leaves before masking (with level 0's hiding messages and record, and lincheck's folded comb),
/// and a round the handler refused (the masking's degenerate coins, a refused coin) is raised once the device returns. With
/// `Reuse::Take` of a kept tag, the level-0 commitment is rep 0's (under M1 with level 0's pads and extra lanes, the session's,
/// so `level0`'s are not read), and the witness is built again from `src`.
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
    native_sha: bool,
) -> Result<(R1csProofLigerito, Commitment, [f64; 8], [f64; 3], [f64; 3]), String> {
    st.c.gpu_nnz_ok()?;
    let t_setup = Instant::now();
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
    let t = types(st);
    let unit = st.c.unit();
    let n_claims = 2 + 2 * st.regions.len();
    let og = pcs.opening_grinding();
    let mut phase = [0f64; 8];
    let mut wphase = [0f64; 3];
    let (mut hz, mut ha, mut hb): (*const u64, *const u64, *const u64) = (null(), null(), null());
    let (mut uin, mut hin, mut hzab): (*const u64, *const u64, *const u64) = (null(), null(), null());
    let mut host0: Option<&HostSlots> = None;
    let mut src = src;
    let mut mctx: Option<MidCtx<'_>> = None;
    if let CircuitSrc::DeviceCut { rows, host, inputs, mid } = src {
        mctx = Some(MidCtx { mid, n_late: t.sched.late_rows.len(), n_cut: t.sched.cut_rows.len(), at: Vec::new(), words: Vec::new(), zab: Vec::new() });
        src = CircuitSrc::Device { rows, host, inputs };
    }
    let (h_setup, t_stage) = (t_setup.elapsed().as_secs_f64(), Instant::now());
    let taken = matches!(reuse, Reuse::Take(tag) if kept(tag));
    let mut u_host = 0;
    match &src {
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
            hzab = host.zab.as_ptr();
            // SAFETY: an F128 is two u64 words (lo, hi)
            let rw: &[u64] = unsafe { std::slice::from_raw_parts(hin, 2 * rows.len()) };
            let check = std::env::var("FC_UNIT_CHECK").as_deref() == Ok("1");
            if let Some(p) = host.pinned_rows.as_ref().filter(|p| p.len() == rw.len()) {
                assert!(!check || p[..] == *rw, "the pinned copy of the compression rows is not the rows");
                hin = p.as_ptr();
            }
            if let Some(d) = host.dev_rows.as_ref().filter(|d| d.words() == rw.len()) {
                assert!(!check || d.holds(rw), "the device's copy of the compression rows is not the host's");
                hin = d.as_ptr();
            }
            if let Some(d) = host.dev_zab.as_ref().filter(|d| d.words() == host.zab.len()) {
                assert!(!check || d.holds(&host.zab), "the device's copy of the host slots is not the host's");
                hzab = d.as_ptr();
            }
            host0 = Some(*host);
        }
    }
    // the unit's rows as CSR for a device that computes the units and lacks them, dropped once the prover returns
    let ucsr = (matches!(src, CircuitSrc::Device { .. }) && u_host == 0 && unsafe { flock_cuda_types_held(t.types_gen, 1) } == 0)
        .then(|| [csr(&unit.a), csr(&unit.b)]);
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
        h_a_off: t.hcsr[0].0.as_ptr(),
        h_a_col: t.hcsr[0].1.as_ptr(),
        h_b_off: t.hcsr[1].0.as_ptr(),
        h_b_col: t.hcsr[1].1.as_ptr(),
        h_levels: (t.hsched.level_off.len() - 1) as i32,
        h_level_off: t.hsched.level_off.as_ptr(),
        h_order: t.hsched.order.as_ptr(),
        h_inputs: hin,
        n_host0: host0.map_or(0, |h| h.at.len() as i64),
        host0_at: host0.map_or(null(), |h| h.at.as_ptr()),
        host0_words: host0.map_or(null(), |h| h.words.as_ptr()),
        host0_zab: hzab,
        host0_ab: host0.map_or(0, |h| h.ab as i32),
        u_range: t.u_range as i32,
        u_useful: unit.useful as i32,
        u_const: unit.const_pos as i32,
        u_in_words: unit.in_words as i32,
        u_a_off: ucsr.as_ref().map_or(null(), |u| u[0].0.as_ptr()),
        u_a_col: ucsr.as_ref().map_or(null(), |u| u[0].1.as_ptr()),
        u_b_off: ucsr.as_ref().map_or(null(), |u| u[1].0.as_ptr()),
        u_b_col: ucsr.as_ref().map_or(null(), |u| u[1].1.as_ptr()),
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
        hm96_prefixes: std::ptr::null(),
        hm96_tables: std::ptr::null(),
        hm96_level0_leaves: 0,
        zk_on: 0,
        zk_rep: 0,
        zk_t_pad: 0,
        zk_tamper: 0,
        zk_pads: null(),
        zk_extra: null(),
        zk_words: None,
        zk_words_ctx: null_mut(),
        u_host,
        rep_reuse: match reuse { Reuse::None => 0, Reuse::Keep(_) => 1, Reuse::Take(_) => 2 },
        reuse_tag: match reuse { Reuse::None => 0, Reuse::Keep(t) | Reuse::Take(t) => t },
        h_plan: null(),
        h_tape_words: 0,
        m_range: 0,
        m_useful: 0,
        m_in_words: 0,
        m_tape_words: 0,
        m_plan: null(),
        m_inputs: null(),
        m_mid: null(),
        m_tables: null(),
        types_gen: t.types_gen,
    };
    #[cfg(feature = "sha512")]
    if let Some((plan, words)) = native_sha.then(|| native_plan(st)).flatten() {
        (p.h_plan, p.h_tape_words) = (plan.as_ptr(), *words);
    }
    #[cfg(not(feature = "sha512"))]
    let _ = native_sha;
    if let Some(rows) = host0.map(|h| &h.hm96_rows).filter(|r| !r.is_empty()) {
        #[cfg(feature = "sha512")]
        {
            let net = &st.c.nets[st.c.hm_net].1;
            let (plan, words) = hm96_plan(st).ok_or("the hm96 slots' input words without their native plan")?;
            if p.h_plan.is_null() {
                return Err("the hm96 slots' native witness needs the compressions' native witness".into());
            }
            assert_eq!(rows.len(), st.blocks() * st.c.range_of(st.c.hm_net).count * net.in_words, "the hm96 slots' input words");
            let r = st.c.ranges.iter().position(|r| r.kind == Kind::Net(st.c.hm_net)).unwrap();
            (p.m_range, p.m_useful, p.m_in_words, p.m_tape_words) = (r as i32, net.useful as i32, net.in_words as i32, *words);
            (p.m_plan, p.m_inputs, p.m_mid) = (plan.as_ptr(), rows.as_ptr() as *const u64, hm96_midstate().as_ptr());
            p.m_tables = flock_merkle::hm96::consts().tables.as_ptr() as *const u64;
        }
        #[cfg(not(feature = "sha512"))]
        {
            let _ = rows;
            return Err("the hm96 slots' native witness needs the SHA-512 build".into());
        }
    }
    #[cfg(feature = "sha512")]
    let zk_src: Option<&Level0> = level0.filter(|_| !taken).map(|z| {
        let (w, l) = (1usize << cfg.initial_k, 1usize << (m - 7 - cfg.initial_k));
        assert!(z.pads.len() == w + flock_prover::pcs::ligerito::zk::EXTRA_LANES && z.pads.iter().all(|p| p.len() == z.t_pad));
        assert!(z.extra.iter().all(|x| x.len() == l), "zk: an extra lane of L words");
        z
    });
    #[cfg(feature = "sha512")]
    if let Some(z) = level0 {
        (p.zk_on, p.zk_rep, p.zk_t_pad, p.zk_tamper) = (1, z.rep as i32, z.t_pad as i32, z.tamper as i32);
    }
    #[cfg(not(feature = "sha512"))]
    let _ = level0;
    #[cfg(feature = "sha512")]
    let hm96_prefixes: Vec<u8> = {
        let c = flock_merkle::hm96::consts();
        [c.salt_prefix, c.leaf_prefix].concat()
    };
    #[cfg(feature = "sha512")]
    let trees = salts.map(|salts| Hm96Trees { salts, given: Default::default() });
    #[cfg(feature = "sha512")]
    if let Some(h) = &trees {
        p.hm96_tree = Some(hm96_tree as Hm96TreeCb);
        p.hm96_ctx = h as *const Hm96Trees<'_> as *mut c_void;
        p.hm96_level0_leaves = pcs.n_leaves() as i64;
        p.hm96_prefixes = hm96_prefixes.as_ptr();
        p.hm96_tables = flock_merkle::hm96::consts().tables.as_ptr() as *const u64;
    }
    #[cfg(not(feature = "sha512"))]
    let _ = salts;
    for (i, l) in t.logs.iter().enumerate() {
        p.type_log[i] = *l;
        let tc = &t.csc[i];
        (p.t_a_col_ptr[i], p.t_a_nnz[i], p.t_b_col_ptr[i], p.t_b_nnz[i]) = (tc.col_ptr[0].as_ptr(), tc.nnz[0], tc.col_ptr[1].as_ptr(), tc.nnz[1]);
        if let Some([ar, br]) = &tc.rows {
            (p.t_a_rows[i], p.t_b_rows[i]) = (ar.as_ptr(), br.as_ptr());
        }
    }
    for (j, &(ty, pos0, count)) in t.ranges.iter().enumerate() {
        (p.r_type[j], p.r_pos0[j], p.r_count[j]) = (ty, pos0, count);
    }
    let mut out: *mut u8 = null_mut();
    let mut out_len: usize = 0;
    let h_stage = t_stage.elapsed().as_secs_f64();
    let rc = std::thread::scope(|sc| {
        #[cfg(feature = "sha512")]
        let mut words = ZkWords { pending: zk_src.map(|z| sc.spawn(move || (Words::of(&z.pads), Words::of(&z.extra)))), words: None };
        #[cfg(feature = "sha512")]
        if words.pending.is_some() {
            (p.zk_words, p.zk_words_ctx) = (Some(zk_words as ZkWordsCb), &mut words as *mut ZkWords<'_> as *mut c_void);
        }
        #[cfg(not(feature = "sha512"))]
        let _ = sc;
        // the device's copies from the words are in stream order before it returns, so they may go once it has
        unsafe { flock_cuda_prove_circuit(&params, &p, &mut out, &mut out_len) }
    });
    let t_decode = Instant::now();
    drop(mctx);
    #[cfg(feature = "sha512")]
    drop(trees);
    if let Some(why) = rounds.and_then(|r| r.failed) {
        if rc == 0 {
            unsafe { flock_cuda_free(out) };
        }
        return Err(why);
    }
    match rc {
        0 => {}
        240 => return Err("PROVER-STOPPED: degenerate coins: rho_0, rho_1 dependent over F128".into()),
        _ => return Err(format!("CUDA circuit prover returned error {rc} at m={m}")),
    }
    let bytes = unsafe { std::slice::from_raw_parts(out, out_len) }.to_vec();
    unsafe { flock_cuda_free(out) };
    let (proof, commitment) = decode_with(&bytes, pcs, level0.is_some(), salts.is_some());
    Ok((proof, commitment, phase, wphase, [h_setup, h_stage, t_decode.elapsed().as_secs_f64()]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(slots: &[HostSlot], ab: bool) -> Vec<u64> {
        let mut want = Vec::new();
        for (_, z, a, b) in slots {
            for v in if ab { vec![a, b] } else { vec![z, a, b] } {
                want.extend(v.iter().flat_map(|x| [x.lo, x.hi]));
            }
        }
        want
    }

    #[test]
    fn host_slots_pack_as_the_device_reads_them() {
        let f = |k: u64, n: usize| (0..n as u64).map(|i| F128::new(k * 1000 + 2 * i, k * 1000 + 2 * i + 1)).collect::<Vec<F128>>();
        let and = |a: &[F128], b: &[F128]| a.iter().zip(b).map(|(a, b)| F128::new(a.lo & b.lo, a.hi & b.hi)).collect::<Vec<F128>>();
        let heap = |n: usize| Some(HostBuf::Heap(vec![0; n]));
        // pageable: z, a, b
        let slots: Vec<HostSlot> = vec![(7, f(1, 3), f(2, 3), f(3, 3)), (40, f(4, 1), f(5, 1), f(6, 1)), (9, vec![], vec![], vec![])];
        let h = HostSlots::pack_in(&slots, |_| None);
        assert_eq!((h.at.clone(), h.words.clone(), h.zab.to_vec(), h.ab), (vec![7, 40, 9], vec![3, 1, 0], packed(&slots, false), false));
        // device-readable, every z = a & b: a, b
        let honest: Vec<HostSlot> = slots.iter().map(|(x, _, a, b)| (*x, and(a, b), a.clone(), b.clone())).collect();
        let h = HostSlots::pack_in(&honest, heap);
        assert_eq!((h.at.clone(), h.words.clone(), h.zab.to_vec(), h.ab), (vec![7, 40, 9], vec![3, 1, 0], packed(&honest, true), true));
        // device-readable, one z word not a & b (a tampered witness): z, a, b
        let mut odd = honest.clone();
        odd[1].1[0].hi ^= 1 << 5;
        let h = HostSlots::pack_in(&odd, heap);
        assert_eq!((h.zab.to_vec(), h.ab), (packed(&odd, false), false));
    }

    #[test]
    fn csr_holds_each_row_in_order() {
        let rows: Vec<Vec<usize>> = vec![vec![3, 1], vec![], vec![0, 5, 2]];
        let (off, col) = csr(&rows);
        assert_eq!((off.clone(), col.clone()), (vec![0, 2, 2, 5], vec![3, 1, 0, 5, 2]));
        let back: Vec<Vec<usize>> = off.windows(2).map(|w| col[w[0] as usize..w[1] as usize].iter().map(|&c| c as usize).collect()).collect();
        assert_eq!(back, rows);
    }

    fn next(s: &mut u64, m: usize) -> usize {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s % m as u64) as usize
    }

    /// One side's rows of a unit: input rows `[i]` or empty, the constant last, and every other row a few columns before it or
    /// the constant, repeats and empty rows among them, some reading their own column (an assertion row). A row before `until`
    /// reads no column in `avoid`.
    fn random_side(s: &mut u64, first: usize, useful: usize, avoid: std::ops::Range<usize>, until: usize) -> Vec<Vec<usize>> {
        let cp = useful - 1;
        (0..useful)
            .map(|i| {
                if i == cp {
                    return vec![cp];
                }
                if i < first {
                    return if next(s, 5) == 0 { vec![] } else { vec![i] };
                }
                let n = next(s, 6);
                let mut r: Vec<usize> = (0..n)
                    .map(|_| match next(s, 9) {
                        0 => cp,
                        _ => match next(s, i) {
                            c if i < until && avoid.contains(&c) => c - avoid.start,
                            c => c,
                        },
                    })
                    .collect();
                if n > 1 && next(s, 3) == 0 {
                    r.push(r[0]);
                }
                if next(s, 7) == 0 {
                    r.push(i);
                }
                r
            })
            .collect()
    }

    #[test]
    fn device_builders_read_rows_alike() {
        use crate::ir_block::{Group, Rows};
        let g = |col: usize| Group { col, words: 1, bits: vec![128] };
        // a one-phase unit, and a cut unit (late input rows 128..256, cut output rows 256..384, which read no late row)
        for (seed, useful, ins, outs) in [(3u64, 600usize, vec![g(0)], vec![g(4)]), (0x51ed, 5000, vec![g(0), g(1)], vec![g(30), g(2)])] {
            let (mut s, first, k) = (seed, ins.len() * 128, useful.next_power_of_two().max(128));
            let a = random_side(&mut s, first, useful, 128..256, 1000);
            let b = random_side(&mut s, first, useful, 128..256, 1000);
            let (ra, rb) = (Rows::from_vecs(&a), Rows::from_vecs(&b));
            let pad = |rows: &Vec<Vec<usize>>| {
                let mut r = rows.clone();
                r.resize(k, Vec::new());
                SparseBinaryMatrix { num_rows: k, num_cols: k, rows: r.into() }
            };
            for (v, r) in [(&a, &ra), (&b, &rb)] {
                assert_eq!(csc_of(r, k), csc_of(v, k), "net {seed}: the CSC from Rows");
                assert_eq!(csc_of(v, k), csc_from_rows(&pad(v)), "net {seed}: the CSC of the rows padded to the slot");
                assert_eq!(csr(r), csr(v), "net {seed}: the CSR from Rows");
            }
            let in_words = ins.len();
            let net = IrUnitNet { name: "t".into(), useful, const_pos: useful - 1, in_groups: ins, out_groups: outs, in_words,
                                  out_cols: vec![], a: ra, b: rb, unit_log: k.trailing_zeros() as usize, sha256: [0; 32], leaves: None,
                                  cut: None, order: None, pass: Default::default() };
            let (x, y) = (schedule(&net), schedule_of(&net, &a, &b));
            assert_eq!((x.levels0, &x.late_rows, &x.cut_rows, &x.level_off, &x.order), (y.levels0, &y.late_rows, &y.cut_rows, &y.level_off, &y.order),
                       "net {seed}: the schedule from vectors");
            assert!(x.level_off.len() > 3, "net {seed}: a schedule of under three levels");
            assert_eq!(x.cut_rows.is_empty(), net.in_groups.len() == 1, "net {seed}: the cut unit's schedule is not two-phase");
        }
    }
}
