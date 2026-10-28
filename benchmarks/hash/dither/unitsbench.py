"""Dither-a's units launch on one GPU, as served (`-hbr -aomn`): each call's noise (`kh_one_noise`, h1s_noise into its
workspace), then its units (`kh_one_units` of a receipts-only build: h1k where the call's live units fit, h1u past
H1K_MAXU), at Llama-3.1-8B's four linears, a window of T rows with M live (decode B32 at T = 64: one live row tile of
two). Each build in `--lib NAME=PATH` (the first is the base, today's) is gated, then timed against the others:

    python unitsbench.py --lib base=h1u_kr.so --lib NAME=h1u_x.so ... --out OUT [--t 64] [--m 32] [--reps 7]
                         [--linears qkv,o,gate_up,down] [--profile]

Gates before any timing, on every build and linear (a failed gate exits 4 and its build is not timed):
  * 0xA5 over the receipts and the workspace, then seeds, noise and units: the receipts of sampled live units (first,
    last and random) against served_h1/receipts.py's call_receipts, every dead unit's receipt zero, no poison left, the
    seeds against seeds.py;
  * a variant's receipts equal the base build's on every unit and word (the units compute what they computed);
  * the no-write control (the units launched over zero rows, under the same poison) refused;
  * the fault: one bit of unit 0's first noise chunk flipped in the workspace moves unit 0's receipt and no other.
Timing (CUDA graphs, builds interleaved rep by rep): each call's units alone (`units`) and its noise (`noise`, the base
build's), and the step's units stream: 32 layers x the four calls' units in order on two alternating weight copies
(`step`, 128 launches, what the served units stream runs apart from the z tables). --profile skips the timing and runs
each gated build's four calls once between torch.cuda.profiler.start and stop (for ncu --profile-from-start off),
and writes its gates to units-prof.json; with --multi, each multi-gated build's one layer in one kh_units_multi launch
after them.

--j10r (with --multi) also gates and times each build's J10-R units (kh_units_multi_j10r, where a build has it) over the
same calls: every live unit's receipt against j10r_units.py's (J10R's h1tile.j10r_reference receipt), every dead unit's
zero, no 0xA5 left in the receipts (the row workspaces poisoned too), nothing written past any workspace (each the
build's own size, j10r_ws.py's, with a sentinel tail; under H1U_JW its last bytes written), the call seeds against
seeds.call_seed and receipt_seed, the no-write control refused, one bit of call 0's row workspace moving exactly call
0's units of row tile 0 and one bit of unit 0's seed_c moving unit 0 alone; then the step in one launch pair
(step_multi_j10r: the rows once a call, then the units) and its rows alone (step_rows_j10r), beside the build's
step_multi. --j10r-sm (with --j10r) gates
kh_units_multi_j10r_sm on every SM (on the SMs whose id % D < Q alone under --smq Q/D, as $DITHER_SO_UNITS_SMQ), the
served launch whose warps take units from a counter (its receipts, seeds and control; its row and seed faults through
kh_units_multi_j10r's rows and units alone, the same kernels), and times step_multi_j10r_sm and sm_over_j10r.
--profile-multi skips the
timing and runs each gated build's step_multi and step_multi_j10r once between torch.cuda.profiler.start and stop.

A build with kh_units_multi_j10rc (h1u.cu's H1U_JRC: each call's column tiles hashed once a call too, into its workspace
after its rows) is gated the same way under --j10r, its workspaces' column tiles poisoned too. Its units read no
seed_c, so its seed fault flips unit 0's sigma_R (unit 0 alone moves), and one bit of call 0's first column tile moves
exactly call 0's units of column tile 0. It is timed as step_multi_j10rc and step_blocks_j10rc (its rows and columns).
With --j10r-sm, a build with kh_units_multi_j10rc_sm too is gated the same way under it, on the SMs --smq names (its
faults through kh_units_multi_j10rc's blocks and units alone), and timed as step_multi_j10rc_sm. --probe NAME (with
--j10r): build NAME is a timing probe whose J10-R receipts are wrong by construction (h1u.cu's H1U_JPROBE): its J10-R
gates are recorded, its J10-R arms timed anyway and marked probe, never a row.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "pouw" / "served_h1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import j10r_units as J10R
import j10r_ws as JWS
import receipts as RC
from h1bench import auto_j
from h1ubench import call_data, digest
from hb_kbench import LINEARS, POISON, Clock, capture, inputs, one_call_ms, w_seed, x_seed

from verity.core.protocols.pouw.circuit import seeds as SD

ORDER = ("qkv", "o", "gate_up", "down")


class Lib:
    """One receipts-only build of h1u.cu (H1U_KSPLIT, H1K_RONLY): kh_seeds, kh_one_noise, kh_one_units."""

    def __init__(self, path: str):
        h = self.h = ctypes.CDLL(path)
        vp, u32 = ctypes.c_void_p, ctypes.c_uint
        one = [vp, vp, u32, u32, u32, vp, vp, vp, vp, u32, vp, vp, vp, vp, vp, vp, vp, u32, vp]
        h.kh_seeds.argtypes = [vp, ctypes.c_ulonglong, u32, vp, vp]
        h.kh_one_noise.argtypes = one
        h.kh_one_units.argtypes = one
        h.kh_words.restype = u32
        h.kh_words.argtypes = [u32]
        h.kh_noise_blocks.restype = u32
        h.kh_noise_blocks.argtypes = [u32]
        h.kh_ws_bytes.restype = ctypes.c_ulonglong
        h.kh_ws_bytes.argtypes = [u32, u32, u32]
        self.multi = hasattr(h, "kh_units_multi")
        if self.multi:
            h.kh_units_multi.argtypes = [vp, u32, u32, u32, vp, vp]
            h.kh_ucall_bytes.restype = u32
            h.kh_digits.argtypes = [u32]
        self.j10r = self.multi and hasattr(h, "kh_units_multi_j10r")
        if self.j10r:
            h.kh_seeds_j10r.argtypes = [vp, ctypes.c_ulonglong, ctypes.c_ulonglong, u32, vp, vp]
            h.kh_units_multi_j10r.argtypes = [vp, u32, u32, u32, vp, ctypes.c_int, vp]
            h.kh_ucallr_bytes.restype = u32
            # H1U_JW: the column pass alone reads seed_c (the call's, at its first position), and no unit reads its own
            self.jw = bool(JWS.flags(h) & JWS.JW)
            if hasattr(h, "kh_j10r_row_fault_word"):
                h.kh_j10r_row_fault_word.restype = ctypes.c_ulonglong
                h.kh_j10r_row_fault_word.argtypes = [u32, u32, u32, ctypes.c_int]
        self.j10r_sm = self.j10r and hasattr(h, "kh_units_multi_j10r_sm") and hasattr(h, "kh_smc_words")
        if self.j10r_sm:
            h.kh_units_multi_j10r_sm.argtypes = [vp, u32, u32, u32, vp, u32, u32, vp, vp]
            h.kh_smc_words.restype = u32
        self.j10rc = self.j10r and hasattr(h, "kh_units_multi_j10rc")
        if self.j10rc:
            h.kh_units_multi_j10rc.argtypes = [vp, u32, u32, u32, vp, ctypes.c_int, vp]
        self.j10rc_sm = self.j10rc and self.j10r_sm and hasattr(h, "kh_units_multi_j10rc_sm")
        if self.j10rc_sm:
            h.kh_units_multi_j10rc_sm.argtypes = [vp, u32, u32, u32, vp, u32, u32, vp, vp]


UCALL = np.dtype([("A", "<u8"), ("B", "<u8"), ("seeds", "<u8"), ("out", "<u8"), ("N", "<u4"), ("K", "<u4"), ("u0", "<u4"),
                  ("md", "<u4")])


class Multi:
    """kh_units_multi over (call, B, out) entries on one build, the three-digit calls first (as dither_so's
    $DITHER_SO_FUSED tables): every unit of every entry in one launch, each entry's seeds and live row tiles its call's."""

    def __init__(self, lib: Lib, entries: list):
        if int(lib.h.kh_ucall_bytes()) != UCALL.itemsize:
            raise RuntimeError(f"kh_units_multi's UCall is {int(lib.h.kh_ucall_bytes())} bytes, not {UCALL.itemsize}")
        self.lib = lib
        md = [int(lib.h.kh_digits(c.a.shape[1])) for c, _, _ in entries]
        order = sorted(range(len(entries)), key=lambda i: -md[i])
        t = np.zeros(len(entries) + 1, dtype=UCALL)
        u0 = 0
        for j, i in enumerate(order):
            c, b, out = entries[i]
            t[j] = (c.a.data_ptr(), b.data_ptr(), c.seeds.data_ptr(), out.data_ptr(), b.shape[0], c.a.shape[1], u0, md[i])
            u0 += c.units
        t[len(entries)]["u0"] = u0
        self.n, self.units, self.rows = len(entries), u0, entries[0][0].a.shape[0]
        self.live = entries[0][0].live
        self.tab = torch.from_numpy(t.view(np.uint8).copy()).to("cuda")

    def run(self, units: int | None = None) -> None:
        if rc := self.lib.h.kh_units_multi(self.tab.data_ptr(), self.n, self.units if units is None else units, self.rows,
                                           self.live.data_ptr(), torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(f"kh_units_multi rc {rc}")


def multi_gate(lib: Lib, cs: list, base_outs: list) -> dict:
    """kh_units_multi over one layer's calls against each call's own receipts (its two launches, every word), under
    0xA5; the no-write control (zero units) refused; one bit of unit 0's seed of the first call flipped moves that
    call's unit 0 alone."""
    outs = [torch.empty_like(c.out) for c in cs]
    mu = Multi(lib, [(c, c.b, o) for c, o in zip(cs, outs)])
    for o in outs:
        o.fill_(POISON)
    mu.run()
    torch.cuda.synchronize()
    bad = [int((o != b).sum()) for o, b in zip(outs, base_outs)]
    poison = sum(int((o == POISON).sum()) for o in outs)
    got = [o.clone() for o in outs]
    for o in outs:
        o.fill_(POISON)
    mu.run(units=0)
    torch.cuda.synchronize()
    k_bad = sum(int((o != b).sum()) for o, b in zip(outs, base_outs))
    s = cs[0].seeds.view(torch.int32)
    s[0, 0] ^= 1
    mu.run()
    torch.cuda.synchronize()
    s[0, 0] ^= 1
    moved = [(i, u) for i, (o, g) in enumerate(zip(outs, got)) for u in (o != g).any(dim=1).nonzero().flatten().tolist()]
    r = {"calls": len(cs), "units": mu.units, "bad_words": bad, "poison_left": poison, "control_bad_words": k_bad,
         "control_refused": k_bad > 0, "fault_moved": moved[:8], "fault_conforming": moved == [(0, 0)]}
    r["ok"] = sum(bad) == 0 and poison == 0 and k_bad > 0 and r["fault_conforming"]
    return r


UCALLR = np.dtype([("A", "<u8"), ("B", "<u8"), ("seeds", "<u8"), ("out", "<u8"), ("rws", "<u8"), ("N", "<u4"),
                   ("K", "<u4"), ("u0", "<u4"), ("md", "<u4")])


#: the bytes past each MultiJ10R workspace, WS_SENTINEL each, that no launch may write (MultiJ10R.tail_written)
WS_TAIL, WS_SENTINEL = 1 << 20, 0x5C


class MultiJ10R:
    """kh_units_multi_j10r over (call, B, out) entries on one build, the three-digit calls first as Multi: each entry its
    call's J10-R seeds (Call.j10r_seed) and a workspace of its own (rws[i], entry i's), the build's size for the call
    (j10r_ws.workspace_bytes: an H1U_JW build's column tiles and W' after its rows), with WS_TAIL bytes of WS_SENTINEL
    past it (tails[i]).  sm: a whole run (mode 0) is the served launch pair, kh_units_multi_j10r_sm on every SM (its
    warps taking units from a counter); its rows alone and units alone (modes 1, 2) stay kh_units_multi_j10r's, the same
    rows kernel and unit.  rc: kh_units_multi_j10rc, each workspace its call's rows and then its columns
    (kh_j10r_ws_bytes); with sm, a whole run is kh_units_multi_j10rc_sm's."""

    def __init__(self, lib: Lib, entries: list, sm: bool = False, rc: bool = False, smq: tuple = (1, 1)):
        if int(lib.h.kh_ucallr_bytes()) != UCALLR.itemsize:
            raise RuntimeError(f"kh_units_multi_j10r's UCallR is {int(lib.h.kh_ucallr_bytes())} bytes, not {UCALLR.itemsize}")
        self.lib, self.sm, self.rc, self.smq = lib, sm, rc, smq
        self.next = torch.zeros(int(lib.h.kh_smc_words()), dtype=torch.int32, device="cuda") if sm else None
        md = [int(lib.h.kh_digits(c.a.shape[1])) for c, _, _ in entries]
        order = sorted(range(len(entries)), key=lambda i: -md[i])
        t = np.zeros(len(entries) + 1, dtype=UCALLR)
        self.rws: list = [None] * len(entries)
        self.tails: list = [None] * len(entries)
        u0 = 0
        for j, i in enumerate(order):
            c, b, out = entries[i]
            nbytes = JWS.workspace_bytes(lib.h, c.a.shape[0], b.shape[0], c.a.shape[1], rc=rc)
            full = torch.full((nbytes + WS_TAIL,), WS_SENTINEL, dtype=torch.uint8, device="cuda")
            rw = self.rws[i] = full[:nbytes]
            self.tails[i] = full[nbytes:]
            t[j] = (c.a.data_ptr(), b.data_ptr(), c.jseeds.data_ptr(), out.data_ptr(), rw.data_ptr(), b.shape[0],
                    c.a.shape[1], u0, md[i])
            u0 += c.units
        t[len(entries)]["u0"] = u0
        self.n, self.units, self.rows = len(entries), u0, entries[0][0].a.shape[0]
        self.live = entries[0][0].live
        self.tab = torch.from_numpy(t.view(np.uint8).copy()).to("cuda")

    def run(self, units: int | None = None, mode: int = 0) -> None:
        n = self.units if units is None else units
        st = torch.cuda.current_stream().cuda_stream
        if self.sm and mode == 0:
            f = self.lib.h.kh_units_multi_j10rc_sm if self.rc else self.lib.h.kh_units_multi_j10r_sm
            if rc := f(self.tab.data_ptr(), self.n, n, self.rows, self.live.data_ptr(), *self.smq, self.next.data_ptr(),
                       st):
                raise RuntimeError(f"{f.__name__} rc {rc}")
            return
        f = self.lib.h.kh_units_multi_j10rc if self.rc else self.lib.h.kh_units_multi_j10r
        if rc := f(self.tab.data_ptr(), self.n, n, self.rows, self.live.data_ptr(), mode, st):
            raise RuntimeError(f"{f.__name__} rc {rc}")

    def tail_written(self) -> int:
        """The bytes past the workspaces (tails) the launches so far wrote: 0, or a launch wrote past its buffer."""
        torch.cuda.synchronize()
        return sum(int((t != WS_SENTINEL).sum()) for t in self.tails)

    def end_written(self, last: int = 4096) -> list:
        """Each workspace's bytes of its last `last` that aren't 0xA5 (the gates' poison): an H1U_JW build's W' fills
        its workspace to the end, so a launch over a poisoned one leaves some, and its size is the one its units use."""
        torch.cuda.synchronize()
        return [int((w[-last:] != 0xA5).sum()) for w in self.rws]


def j10r_gate(lib: Lib, cs: list, refs: list, sm: bool = False, rc: bool = False, smq: tuple = (1, 1)) -> dict:
    """kh_units_multi_j10r over one layer's calls: each call's live units against refs (j10r_units.call_receipts, every
    word), its dead units zero, under 0xA5 over the receipts and the row workspaces; seeds against seeds.call_seed and
    receipt_seed; the no-write control (zero units) refused; one bit of call 0's row workspace (row 0's first block, or
    the build's kh_j10r_row_fault_word: under H1U_JAP A'[0][0], which its units read for the block) moving exactly call
    0's units of row tile 0; one bit of unit 0's seed_c, flipped after the rows, moving unit 0 alone.  sm:
    the receipts, seeds and control under kh_units_multi_j10r_sm (MultiJ10R's sm), the faults under its modes 1 and 2.
    rc (kh_units_multi_j10rc): its units read no seed_c, so the seed fault flips unit 0's sigma_R instead, and one bit of
    call 0's first column tile (column 0's, after the rows) moves exactly call 0's live units of column tile 0.  An
    H1U_JW build's units read no seed_c either (its column pass, in the units' launch, reads the call's): the seed fault
    flips unit 0's sigma_R, and one bit of call 0's seed_c moves every live unit of call 0.  Every workspace is the
    build's size (j10r_ws), and no launch of the gate writes the WS_TAIL bytes past it (ws_tail_written 0); under H1U_JW
    the first launch writes into each workspace's last 4 KiB (ws_end_written), so that size is the one its units use,
    and ws_past_rows_written counts what it wrote past kh_j10r_rows_bytes, the size a rows-only caller allocated."""
    for c in cs:
        c.j10r_seed()
    outs = [torch.empty_like(c.out) for c in cs]
    mu = MultiJ10R(lib, [(c, c.b, o) for c, o in zip(cs, outs)], sm=sm, rc=rc, smq=smq)

    def poison() -> None:
        for o in outs:
            o.fill_(POISON)
        for w in mu.rws:
            w.fill_(0xA5)

    poison()
    mu.run()
    ws_end = mu.end_written()
    ws_past_rows = [int((w[JWS.rows_bytes(lib.h, *c.a.shape):] != 0xA5).sum()) for w, c in zip(mu.rws, cs)]
    bad = [int((o[:c.live_units] != r).sum()) for o, c, r in zip(outs, cs, refs)]
    dead = sum(int((o[c.live_units:] != 0).sum()) for o, c in zip(outs, cs))
    left = sum(int((o == POISON).sum()) for o in outs)
    seed_bad = 0
    for c in cs:
        sd = c.jseeds.cpu().numpy()
        for u in sorted({0, c.live_units - 1, c.units - 1}):
            seed_bad += int(digest(sd[u, :8]) != J10R.call_seed(c.salt, c.base, *c.roots))
            seed_bad += int(digest(sd[u, 8:]) != SD.receipt_seed(c.salt, c.base + u, *c.roots))
    got = [o.clone() for o in outs]
    poison()
    mu.run(units=0)
    torch.cuda.synchronize()
    k_bad = sum(int((o != g).sum()) for o, g in zip(outs, got))

    def moved() -> list:
        return [(i, u) for i, (o, g) in enumerate(zip(outs, got)) for u in (o != g).any(dim=1).nonzero().flatten().tolist()]

    poison()
    mu.run(mode=1)
    t0, k0 = cs[0].a.shape
    rw = (int(lib.h.kh_j10r_row_fault_word(t0, cs[0].b.shape[0], k0, int(rc)))
          if hasattr(lib.h, "kh_j10r_row_fault_word") else 0)
    mu.rws[0].view(torch.int32)[rw] ^= 1
    mu.run(mode=2)
    torch.cuda.synchronize()
    row_moved = moved()
    row_want = [(0, u) for u in range(cs[0].b.shape[0] // 32)]
    def seed_fault(sw: int) -> list:
        poison()
        mu.run(mode=1)
        s = cs[0].jseeds.view(torch.int32)
        s[0, sw] ^= 1
        mu.run(mode=2)
        torch.cuda.synchronize()
        s[0, sw] ^= 1
        return moved()

    jw = lib.jw and not rc
    seed_moved = seed_fault(16 if rc or jw else 0)
    r = {"calls": len(cs), "units": mu.units, "sm": sm, **({"smq": "/".join(map(str, smq))} if sm else {}),
         "bad_words": bad, "dead_bad_words": dead, "poison_left": left, "seed_bad": seed_bad, "control_bad_words": k_bad,
         "control_refused": k_bad > 0,
         "fault_rows_moved": len(row_moved), "fault_rows_want": len(row_want), "fault_rows_conforming": row_moved == row_want,
         "fault_seed": "sigma_R" if rc or jw else "seed_c", "fault_seed_moved": seed_moved[:8],
         "fault_seed_conforming": seed_moved == [(0, 0)]}
    r["ok"] = (sum(bad) == 0 and dead == 0 and left == 0 and seed_bad == 0 and k_bad > 0 and r["fault_rows_conforming"]
               and r["fault_seed_conforming"])
    r["fault_rows_word"] = rw
    if jw:
        call_moved = seed_fault(0)
        call_want = [(0, u) for u in range(cs[0].live_units)]
        r.update({"fault_call_seed_moved": len(call_moved), "fault_call_seed_want": len(call_want),
                  "fault_call_seed_conforming": call_moved == call_want})
        r["ok"] = r["ok"] and r["fault_call_seed_conforming"]
    if rc:
        poison()
        mu.run(mode=1)
        t, k = cs[0].a.shape
        mu.rws[0].view(torch.int32)[16 * t * (k // 512 + 1)] ^= 1
        mu.run(mode=2)
        torch.cuda.synchronize()
        col_moved = moved()
        ncb = cs[0].b.shape[0] // 32
        col_want = [(0, ri * ncb) for ri in range(cs[0].live_rb)]
        r.update({"fault_cols_moved": len(col_moved), "fault_cols_want": len(col_want),
                  "fault_cols_conforming": col_moved == col_want})
        r["ok"] = r["ok"] and r["fault_cols_conforming"]
    r.update({"ws_bytes": [int(w.numel()) for w in mu.rws], "ws_past_rows_written": ws_past_rows,
              "ws_end_written": ws_end, "ws_tail_bytes": WS_TAIL, "ws_tail_written": mu.tail_written()})
    r["ok"] = r["ok"] and r["ws_tail_written"] == 0 and (not jw or all(ws_end))
    return r


class Call:
    """One call's buffers on one build: the A slot [T, K] (rows >= M zero), B [N, K], live row tiles, seeds, the
    workspace, receipts, and the receipts-only units' z/y scratch and flagless step (dither_a's s_unit)."""

    def __init__(self, lib: Lib, a: torch.Tensor, b: torch.Tensor, m: int, salt: bytes, roots: tuple, base: int, sms: int):
        self.lib, self.a, self.b, self.m, self.salt, self.roots, self.base = lib, a, b, m, salt, roots, base
        t, k = a.shape
        n = b.shape[0]
        self.units = (t // 32) * (n // 32)
        self.live_rb = (m + 31) // 32
        self.live_units = self.live_rb * (n // 32)
        self.words = int(lib.h.kh_words(k))
        self.nb = int(lib.h.kh_noise_blocks(k))
        self.j = auto_j(self.live_rb * (n // 32), self.nb, sms)
        self.live = torch.tensor([1 if ri < self.live_rb else 0 for ri in range(t // 32)], dtype=torch.uint8,
                                 device="cuda")
        self.data = call_data(salt, *roots)
        self.seeds = torch.empty((self.units, 16), dtype=torch.int64, device="cuda")
        self.ws = torch.empty((int(lib.h.kh_ws_bytes(t, n, k)),), dtype=torch.uint8, device="cuda")
        self.out = torch.empty((self.units, self.words), dtype=torch.int32, device="cuda")
        self.zs = torch.empty((t, n), dtype=torch.int32, device="cuda")
        self.ys = torch.empty((t, n), dtype=torch.bfloat16, device="cuda")
        self.sa = torch.zeros((t,), dtype=torch.float32, device="cuda")
        self.sw = torch.zeros((n,), dtype=torch.float32, device="cuda")
        self.step = torch.zeros((4,), dtype=torch.int32, device="cuda")

    def twin(self, lib: Lib, sms: int) -> Call:
        return Call(lib, self.a, self.b, self.m, self.salt, self.roots, self.base, sms)

    def _args(self, rows: int, b: torch.Tensor | None = None):
        t, k = self.a.shape
        n = self.b.shape[0]
        return (self.a.data_ptr(), (self.b if b is None else b).data_ptr(), rows, n, k, self.seeds.data_ptr(),
                self.live.data_ptr(), self.out.data_ptr(), self.ws.data_ptr(), self.j, self.zs.data_ptr(), None,
                self.ys.data_ptr(), None, self.sa.data_ptr(), self.sw.data_ptr(), self.step.data_ptr(), min(self.m, t),
                torch.cuda.current_stream().cuda_stream)

    def seed(self) -> None:
        if self.lib.h.kh_seeds(self.data.data_ptr(), self.base, self.units, self.seeds.data_ptr(),
                               torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("kh_seeds")

    def j10r_seed(self) -> None:
        """J10-R's seeds (kh_seeds_j10r): each unit's [seed_c, σ_R], seed_c at the call's first position, base."""
        if not hasattr(self, "jseeds"):
            w = np.array(J10R.seed_words(self.salt, *self.roots), dtype=np.uint64).view(np.int64)
            self.jdata = torch.tensor(w, device="cuda")
            self.jseeds = torch.empty((self.units, 16), dtype=torch.int64, device="cuda")
        if self.lib.h.kh_seeds_j10r(self.jdata.data_ptr(), self.base, self.base, self.units, self.jseeds.data_ptr(),
                                    torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("kh_seeds_j10r")

    def noise(self) -> None:
        if rc := self.lib.h.kh_one_noise(*self._args(self.a.shape[0])):
            raise RuntimeError(f"kh_one_noise rc {rc}")

    def run_units(self, rows: int | None = None, b: torch.Tensor | None = None) -> None:
        if rc := self.lib.h.kh_one_units(*self._args(self.a.shape[0] if rows is None else rows, b)):
            raise RuntimeError(f"kh_one_units rc {rc}")

    def poison(self) -> None:
        self.out.fill_(POISON)
        self.ws.fill_(0xA5)

    def full(self) -> torch.Tensor:
        """Poisoned, then seeds, noise and units once: the receipts."""
        self.poison()
        self.seed()
        self.noise()
        self.run_units()
        torch.cuda.synchronize()
        return self.out.clone()


def gate(c: Call, picks: list[int], ref_t: torch.Tensor, base_out: torch.Tensor | None) -> dict:
    pk = torch.tensor(picks, device="cuda")
    out = c.full()
    bad = int((out[pk] != ref_t).sum())
    dead_bad = int((out[c.live_units:] != 0).sum())
    poison = int((out == POISON).sum())
    sd = c.seeds.cpu().numpy()
    seed_bad = sum(int(digest(sd[u, :8]) != SD.seed(c.salt, c.base + u, *c.roots))
                   + int(digest(sd[u, 8:]) != SD.receipt_seed(c.salt, c.base + u, *c.roots)) for u in picks)
    vs_base = None if base_out is None else int((out != base_out).sum())
    # the no-write control: the units over zero rows under the same poison
    c.out.fill_(POISON)
    c.run_units(rows=0)
    torch.cuda.synchronize()
    k_bad = int((c.out[pk] != ref_t).sum())
    # the fault: bit 0 of unit 0's first noise chunk in the workspace, the units again
    c.out.fill_(POISON)
    c.ws.view(torch.int32)[0] ^= 1
    c.run_units()
    torch.cuda.synchronize()
    moved = (c.out != out).any(dim=1).nonzero().flatten().tolist()
    c.ws.view(torch.int32)[0] ^= 1
    fault_ok = moved == [0]
    r = {"units": c.units, "live_units": c.live_units, "j": c.j, "bad_words": bad, "dead_bad_words": dead_bad,
         "poison_left": poison, "seed_bad": seed_bad, "vs_base_bad_words": vs_base, "control_bad_words": k_bad,
         "control_refused": k_bad > 0, "fault_moved_units": moved[:8], "fault_conforming": fault_ok}
    r["ok"] = (bad == 0 and dead_bad == 0 and poison == 0 and seed_bad == 0 and (vs_base in (None, 0)) and k_bad > 0
               and fault_ok)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", type=int, default=64)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--item-ms", type=float, default=4.0)
    ap.add_argument("--picks", type=int, default=6)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--linears", default=",".join(ORDER))
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--multi", action="store_true",
                    help="also gate kh_units_multi (where a build has it) over one layer's calls against their own "
                         "receipts, and time the step in one launch (step_multi) against the base's noise and units "
                         "call by call (step_noise_units)")
    ap.add_argument("--j10r", action="store_true",
                    help="with --multi: also gate kh_units_multi_j10r (where a build has it) against j10r_units.py's "
                         "receipts on every live unit, and time step_multi_j10r and step_rows_j10r")
    ap.add_argument("--j10r-sm", action="store_true",
                    help="with --j10r: also gate kh_units_multi_j10r_sm on every SM (the served launch, its warps taking "
                         "units from a counter) and time step_multi_j10r_sm")
    ap.add_argument("--smq", default="1/1", metavar="Q/D",
                    help="with --j10r-sm: kh_units_multi_j10r_sm (and kh_units_multi_j10rc_sm) on the SMs whose id "
                         "%% D < Q alone (the served $DITHER_SO_UNITS_SMQ), gated and timed there")
    ap.add_argument("--probe", action="append", default=[], metavar="NAME",
                    help="with --j10r: build NAME (not the base) is a timing probe (h1u.cu's H1U_JPROBE), never a row: "
                         "its J10-R gates are recorded and its J10-R arms timed whatever they say, its timing marked "
                         "probe")
    ap.add_argument("--profile-multi", action="store_true",
                    help="with --multi: skip the timing and run each gated build's step_multi (and step_multi_j10r) "
                         "once between torch.cuda.profiler.start and stop, for ncu --profile-from-start off")
    ap.add_argument("--sets", default="multi,j10r,j10rc",
                    help="with --multi: the whole-step sets timed for each gated build (multi, j10r, j10rc), each "
                         "with its own 32 layers of receipts, so a screen of many builds times only the set it compares")
    args = ap.parse_args()
    sets = set(args.sets.split(","))
    if (args.j10r or args.profile_multi) and not args.multi:
        ap.error("--j10r and --profile-multi run beside step_multi: give --multi too")
    if args.j10r_sm and not args.j10r:
        ap.error("--j10r-sm runs beside step_multi_j10r: give --j10r too")
    q, _, den = args.smq.partition("/")
    if not (q.isdigit() and den.isdigit() and 0 < int(q) <= int(den) <= 256):
        ap.error(f"--smq {args.smq}: give Q/D with 0 < Q <= D <= 256 (h1u.cu's SMQ_DEN_MAX)")
    smq = (int(q), int(den))
    if args.probe and not args.j10r:
        ap.error("--probe times J10-R arms: give --j10r too")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc_path = out / ("units-prof.json" if args.profile else "units.json")
    paths = dict(v.partition("=")[::2] for v in args.lib)
    libs = {name: Lib(p) for name, p in paths.items()}
    names = list(libs)
    if names[0] in args.probe or set(args.probe) - set(names):
        ap.error(f"--probe {args.probe}: builds of --lib other than the base {names[0]}")
    clock = Clock()
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    rng = random.Random(0x55A17)
    doc = {"schema": "dither-a/units/v1", "device": clock.name, "sms": sms, "t": args.t, "m": args.m, "libs": paths,
           **({"smq": args.smq} if args.j10r_sm else {}),
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reference": "served_h1/receipts.py",
           "probes": args.probe, "linears": {}}
    T, M = args.t, args.m
    linears = [x for x in ORDER if x in args.linears.split(",")]
    calls: dict = {}
    bcopy: dict = {}
    base_outs: dict = {}
    failed: set = set()
    for name in linears:
        n, k = LINEARS[name]
        x = torch.from_numpy(inputs(x_seed(k), M, k)).to("cuda")
        a = torch.cat([x, x.new_zeros(T - M, k)]).contiguous() if T > M else x.contiguous()
        b = torch.from_numpy(inputs(w_seed(name), n, k)).to("cuda")
        bcopy[name] = torch.from_numpy(inputs(w_seed(name) + 77, n, k)).to("cuda")
        first = Call(libs[names[0]], a, b, M, rng.randbytes(64), (rng.randbytes(64), rng.randbytes(64)),
                     rng.getrandbits(40), sms)
        cs = {nm: first if i == 0 else first.twin(libs[nm], sms) for i, nm in enumerate(names)}
        L = first.live_units
        picks = sorted({0, L - 1, *rng.sample(range(L), min(L, args.picks))})
        t0 = time.time()
        ref = RC.call_receipts(RC.Call(a, b), first.salt, first.base, *first.roots, only=picks)
        ref_t = torch.from_numpy(ref.view(np.int32)).to("cuda")
        d = doc["linears"][name] = {"n": n, "k": k, "units": first.units, "live_units": L, "words": first.words,
                                    "noise_blocks": first.nb, "picks": picks, "ref_s": round(time.time() - t0, 2),
                                    "gates": {}}
        base_out = None
        for nm, c in cs.items():
            g = gate(c, picks, ref_t, base_out)
            if base_out is None:
                base_out = c.full()
            d["gates"][nm] = g
            print(json.dumps({"linear": name, "lib": nm, "gate": g}), flush=True)
            if not g["ok"]:
                failed.add(nm)
        calls[name] = cs
        base_outs[name] = base_out
        for c in cs.values():
            c.full()
    multi_ok: set = set()
    if args.multi:
        doc["multi_gates"] = {}
        for nm in names:
            if nm in failed or not libs[nm].multi:
                continue
            g = multi_gate(libs[nm], [calls[x][nm] for x in linears], [base_outs[x] for x in linears])
            doc["multi_gates"][nm] = g
            print(json.dumps({"multi": nm, "gate": g}), flush=True)
            if g["ok"]:
                multi_ok.add(nm)
            else:
                failed.add(f"{nm}:multi")
    j10r_ok: set = set()
    j10rc_ok: set = set()
    if args.j10r:
        doc["j10r_reference"] = ("h1tile.j10r_reference" if J10R.DEFINITION else
                                 "j10r_units' restatement (its anchor checks it against J10R's h1tile.j10r_reference)")
        doc["j10r_gates"], doc["j10r_ref_s"] = {}, {}
        refs = []
        for x in linears:
            c0 = calls[x][names[0]]
            t0 = time.time()
            ref = J10R.call_receipts(c0.a, c0.b, c0.salt, c0.base, *c0.roots, only=list(range(c0.live_units)))
            doc["j10r_ref_s"][x] = round(time.time() - t0, 2)
            refs.append(torch.from_numpy(ref.view(np.int32)).to("cuda"))
        for nm in names:
            if nm in failed or not libs[nm].j10r:
                continue
            g = j10r_gate(libs[nm], [calls[x][nm] for x in linears], refs)
            doc["j10r_gates"][nm] = g
            print(json.dumps({"j10r": nm, "gate": g}), flush=True)
            if g["ok"] or nm in args.probe:
                j10r_ok.add(nm)
            else:
                failed.add(f"{nm}:j10r")
        for nm in names:
            if nm in failed or not libs[nm].j10rc:
                continue
            g = j10r_gate(libs[nm], [calls[x][nm] for x in linears], refs, rc=True)
            doc.setdefault("j10rc_gates", {})[nm] = g
            print(json.dumps({"j10rc": nm, "gate": g}), flush=True)
            if g["ok"]:
                j10rc_ok.add(nm)
            else:
                failed.add(f"{nm}:j10rc")
    sm_ok: set = set()
    rcsm_ok: set = set()
    if args.j10r_sm:
        doc["j10r_sm_gates"] = {}
        for nm in names:
            if nm not in j10r_ok or not libs[nm].j10r_sm:
                continue
            g = j10r_gate(libs[nm], [calls[x][nm] for x in linears], refs, sm=True, smq=smq)
            doc["j10r_sm_gates"][nm] = g
            print(json.dumps({"j10r_sm": nm, "gate": g}), flush=True)
            if g["ok"] or nm in args.probe:
                sm_ok.add(nm)
            else:
                failed.add(f"{nm}:j10r_sm")
        for nm in names:
            if nm not in j10rc_ok or not libs[nm].j10rc_sm:
                continue
            g = j10r_gate(libs[nm], [calls[x][nm] for x in linears], refs, sm=True, rc=True, smq=smq)
            doc.setdefault("j10rc_sm_gates", {})[nm] = g
            print(json.dumps({"j10rc_sm": nm, "gate": g}), flush=True)
            if g["ok"]:
                rcsm_ok.add(nm)
            else:
                failed.add(f"{nm}:j10rc_sm")
    doc["gate_failed"] = sorted(failed)
    doc_path.write_text(json.dumps(doc, indent=1))
    if names[0] in failed:
        print("unitsbench: the base build failed its gate; nothing timed", file=sys.stderr)
        return 4
    timed = [nm for nm in names if nm not in failed]
    if args.profile:
        for nm in timed:
            mu = Multi(libs[nm], [(calls[x][nm], calls[x][nm].b, calls[x][nm].out) for x in linears]) \
                if nm in multi_ok else None
            for name in linears:
                calls[name][nm].run_units()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(nm)
            torch.cuda.profiler.start()
            for name in linears:
                torch.cuda.nvtx.range_push(f"{nm}/{name}")
                calls[name][nm].run_units()
                torch.cuda.nvtx.range_pop()
            if mu is not None:
                torch.cuda.nvtx.range_push(f"{nm}/multi")
                mu.run()
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            torch.cuda.nvtx.range_pop()
        doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        doc_path.write_text(json.dumps(doc, indent=1))
        return 4 if failed else 0

    arms: dict = {}
    for name in linears:
        for nm in timed:
            arms[(nm, f"units:{name}")] = (lambda c: lambda i: c.run_units(b=None))(calls[name][nm])
        arms[(names[0], f"noise:{name}")] = (lambda c: lambda i: c.noise())(calls[name][names[0]])
    if len(linears) == len(ORDER):
        for nm in timed:
            def step(i, nm=nm):
                for layer in range(args.layers):
                    for name in ORDER:
                        c = calls[name][nm]
                        c.run_units(b=c.b if layer % 2 == 0 else bcopy[name])
            arms[(nm, "step")] = step
        keep = []
        for nm in timed:
            if nm not in multi_ok or "multi" not in sets:
                continue
            outs = [torch.empty_like(calls[name][nm].out) for _ in range(args.layers) for name in ORDER]
            mu = Multi(libs[nm], [(calls[name][nm], calls[name][nm].b if layer % 2 == 0 else bcopy[name],
                                   outs[ORDER.index(name) + len(ORDER) * layer])
                                  for layer in range(args.layers) for name in ORDER])
            keep.append((mu, outs))
            arms[(nm, "step_multi")] = (lambda mu: lambda i: mu.run())(mu)
        for nm in timed:
            if nm not in j10r_ok or "j10r" not in sets:
                continue
            outs = [torch.empty_like(calls[name][nm].out) for _ in range(args.layers) for name in ORDER]
            mu = MultiJ10R(libs[nm], [(calls[name][nm], calls[name][nm].b if layer % 2 == 0 else bcopy[name],
                                       outs[ORDER.index(name) + len(ORDER) * layer])
                                      for layer in range(args.layers) for name in ORDER])
            keep.append((mu, outs))
            arms[(nm, "step_multi_j10r")] = (lambda mu: lambda i: mu.run())(mu)
            arms[(nm, "step_rows_j10r")] = (lambda mu: lambda i: mu.run(mode=1))(mu)
        for nm in timed:
            if nm not in sm_ok:
                continue
            outs = [torch.empty_like(calls[name][nm].out) for _ in range(args.layers) for name in ORDER]
            mu = MultiJ10R(libs[nm], [(calls[name][nm], calls[name][nm].b if layer % 2 == 0 else bcopy[name],
                                       outs[ORDER.index(name) + len(ORDER) * layer])
                                      for layer in range(args.layers) for name in ORDER], sm=True, smq=smq)
            keep.append((mu, outs))
            arms[(nm, "step_multi_j10r_sm")] = (lambda mu: lambda i: mu.run())(mu)
        for nm in timed:
            if nm not in j10rc_ok or "j10rc" not in sets:
                continue
            outs = [torch.empty_like(calls[name][nm].out) for _ in range(args.layers) for name in ORDER]
            mu = MultiJ10R(libs[nm], [(calls[name][nm], calls[name][nm].b if layer % 2 == 0 else bcopy[name],
                                       outs[ORDER.index(name) + len(ORDER) * layer])
                                      for layer in range(args.layers) for name in ORDER], rc=True)
            keep.append((mu, outs))
            arms[(nm, "step_multi_j10rc")] = (lambda mu: lambda i: mu.run())(mu)
            arms[(nm, "step_blocks_j10rc")] = (lambda mu: lambda i: mu.run(mode=1))(mu)
        for nm in timed:
            if nm not in rcsm_ok:
                continue
            outs = [torch.empty_like(calls[name][nm].out) for _ in range(args.layers) for name in ORDER]
            mu = MultiJ10R(libs[nm], [(calls[name][nm], calls[name][nm].b if layer % 2 == 0 else bcopy[name],
                                       outs[ORDER.index(name) + len(ORDER) * layer])
                                      for layer in range(args.layers) for name in ORDER], sm=True, rc=True,
                                     smq=smq)
            keep.append((mu, outs))
            arms[(nm, "step_multi_j10rc_sm")] = (lambda mu: lambda i: mu.run())(mu)
        if args.multi:
            def step_nu(i, nm=names[0]):
                for layer in range(args.layers):
                    for name in ORDER:
                        c = calls[name][nm]
                        c.noise()
                        c.run_units(b=c.b if layer % 2 == 0 else bcopy[name])
            arms[(names[0], "step_noise_units")] = step_nu
    if args.profile_multi:
        for (nm, arm), fn in arms.items():
            if arm not in ("step_multi", "step_multi_j10r", "step_multi_j10rc"):
                continue
            fn(0)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"{nm}/{arm}")
            torch.cuda.profiler.start()
            fn(0)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            torch.cuda.nvtx.range_pop()
        doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        doc_path.write_text(json.dumps(doc, indent=1))
        return 4 if failed else 0
    graphs, per_graph = {}, {}
    for key, fn in arms.items():
        t = one_call_ms(fn)
        G = max(1, min(400, round(args.item_ms / max(t, 1e-3))))
        graphs[key], per_graph[key] = capture(fn, G), G
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    us = {key: [] for key in graphs}
    mhz = {key: [] for key in graphs}
    order = list(graphs)
    for rep in range(args.reps):
        for key in order[rep % len(order):] + order[:rep % len(order)]:
            e0.record()
            graphs[key].replay()
            e1.record()
            e1.synchronize()
            us[key].append(e0.elapsed_time(e1) * 1e3 / per_graph[key])
            mhz[key].append(clock.sm())
    res: dict = {}
    for (nm, arm), v in us.items():
        res.setdefault(nm, {})[arm] = {"median_us": round(statistics.median(v), 3), "us": [round(x, 3) for x in v],
                                       "per_graph": per_graph[(nm, arm)], "sm_mhz": mhz[(nm, arm)]}
    for nm, r in res.items():
        lay = [r[f"units:{x}"]["median_us"] for x in ORDER if f"units:{x}" in r]
        if len(lay) == len(ORDER):
            r["layer_units_us"] = round(sum(lay), 3)
            r["step_units_ms_from_calls"] = round(sum(lay) * args.layers / 1e3, 3)
        for arm in ("step", "step_multi", "step_noise_units", "step_multi_j10r", "step_rows_j10r", "step_multi_j10r_sm",
                    "step_multi_j10rc", "step_blocks_j10rc", "step_multi_j10rc_sm"):
            if arm in r:
                r[f"{arm}_ms"] = round(r[arm]["median_us"] / 1e3, 3)
        if "step_multi_j10r" in r and "step_multi" in r:
            r["j10r_over_multi"] = round(r["step_multi_j10r"]["median_us"] / r["step_multi"]["median_us"], 4)
        if "step_multi_j10r_sm" in r and "step_multi_j10r" in r:
            r["sm_over_j10r"] = round(r["step_multi_j10r_sm"]["median_us"] / r["step_multi_j10r"]["median_us"], 4)
        if "step_multi_j10rc" in r and "step_multi_j10r" in r:
            r["j10rc_over_j10r"] = round(r["step_multi_j10rc"]["median_us"] / r["step_multi_j10r"]["median_us"], 4)
        if nm in args.probe:
            r["probe"] = True
        if "step_multi_j10rc_sm" in r and "step_multi_j10r_sm" in r:
            r["rcsm_over_j10r_sm"] = round(r["step_multi_j10rc_sm"]["median_us"] / r["step_multi_j10r_sm"]["median_us"], 4)
    b0 = res[names[0]]
    for nm in timed[1:]:
        r = res[nm]
        r["over_base"] = {arm: round(v["median_us"] / b0[arm]["median_us"], 4) for arm, v in r.items()
                          if isinstance(v, dict) and arm in b0 and "median_us" in v}
    doc["timing"] = res
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc_path.write_text(json.dumps(doc, indent=1))
    print(json.dumps({nm: {k: (v["median_us"] if isinstance(v, dict) and "median_us" in v else v)
                           for k, v in r.items() if k != "noise"} for nm, r in res.items()}, indent=1), flush=True)
    return 4 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
