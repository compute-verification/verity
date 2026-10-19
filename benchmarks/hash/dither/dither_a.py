"""The `dither-a` arm: ASK-71's option A (CA's card), the provisional credit form since Daniel's 21:59Z ruling (19 Oct),
labelled `form=ASK-71 A (provisional)` on every run.  Under it the noised work is credited on the operands A, B of a
served product that is value-checked (on every drawn unit z_u = A_u B_u^T, proved and committed before the draw, y from
the committed z), so the served product may come from a plain GEMM beside the receipted one.

Serving stream, per int8-exact linear (`dithera::mm`, one op opaque to Inductor): exactlin's quantizer, which also stores
the call's live rows and scales (rows < m, unless the step's flags say no-write) into the window's A slot
(`exactlin_quant_rows_dithera`); then exactlin's GEMM at exactlin's config for the call (`exactlin.pick`), whose
epilogue stores z = X W^T into the window's z slot (rows < m; rows m to T zero, from extra row tiles past M) beside
y = dequant(z), exact-fused's y bit for bit (`dithera_mm`, `dithera_mm_tma`).

Side stream, forked after the quantizer (it reads only window-owned buffers): the call's A table (`h1c_a_call`, the
window's form: DITHER_A_FORM, 3 unless set), its seeds (h1r version 0, from the window's salt and root(A_j), root(B_j))
and its receipts of the noised product T = (A + N_A)(B + N_B)^T only, no peel and no z: h1r's block or split form per
call (`unit="h1r"`: versions 3 and 4, wide where the call's column blocks outnumber the SMs, as dither-1s-mix's units)
or h1u.cu's kh_unit_ws (`unit="kh"`), into a staging buffer.  At the forward's last call `dithera_commit` copies the
staging receipts into the plan's, which the receipt trees commit, unless the step's flags say no-write (F_NW_UNITS),
and under F_FAULT flips bit FAULT_SHIFT of word 0 of unit 0's receipt of every call.  The side stream joins the serving
stream at the forward's last call, and under piecewise graphs at every qkv call too (each piece's capture must join its
forks, and a piece ends at an attention).  Between steps dither_one's Window builds the admit and z tables, the step
trees, the round and the receipt trees, as dither-1s's: the A table before the call's units, z before the draw, the
step and receipt trees between steps, and nothing waits on the side stream but those.

    import dither_a as DA
    DA.patch(t=T, h1c="libh1c.so", h1o="libh1o.so", unit="h1r")   # before LLM()
    llm = LLM(model=int8_exact_checkpoint, ...)
    win = DO.hook(llm); win.captures = DA.CAPTURES

`check` on the gate, control and fault captures (each step's buffers 0xA5 before its kernels wrote them): every call's
A, admit and z roots and the round against commit.py; z = A B^T (int_mm, the A the A table committed) on every call and
zero past m; the served y (dumped) = dequant(z); every receipt = h1r v1's on the saved slots under the window's salt and
the CPU's roots (another unit implementation); three units' receipts by CA's definition on the CPU (`h1v3_unit`) and by
H1V3's own code.  The control (nothing written: the A slot, z, receipts) must be refused on z and on the receipts; the
fault must be refused on the receipts at exactly the flipped words, z and y untouched.
"""
from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

import dither_fused as DF
import dither_one as DO
from dither_fused import CM, CR, H1R, Call
from h1c import Dom, Tab
from hb_side import auto_j

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exactlin"))
import exactlin as EL

FORM = "ASK-71 A (provisional)"
CAPTURES = ("gate", "control", "fault")
LIB_ENV = DO.LIB_ENV
LIB = "dithera"
UNITS = ("h1r", "kh")
# z's commitment: "row" each slot's N int32 a hiding leaf (dither_one's z table); "unit" each 32 x 32 unit's z a hiding
# leaf, z stored unit-major by the GEMM's epilogue (ZU_KIND, provisional until INT-TIE names the leaf's fields); "none"
# no z stored and no z table, y only: the leaves' cost floor, never credited
ZMODES = {"none": 0, "row": 1, "unit": 2}
ZU_KIND = "z-unit-table"
H1R_BLOCK, H1R_SPLIT = 3, 4     # h1r_launch's versions: a unit a block, a unit six warps
M32 = DO.M32


@triton.jit
def exactlin_quant_rows_dithera(x_ptr, q_ptr, s_ptr, slot_ptr, scale_ptr, S_ptr, K, stride_x, stride_q,
                                BLOCK_K: tl.constexpr):
    # exactlin_quant_rows' body, unchanged
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.max(tl.abs(x), axis=0)
    s = tl.div_rn(tl.maximum(a, 1e-30), 127.0)
    sb = s.to(tl.int32, bitcast=True)
    s1 = ((sb & 0x7FFFFF) | 0x3F800000).to(tl.float32, bitcast=True)
    p = ((254 - (sb >> 23)) << 23).to(tl.float32, bitcast=True)
    xs = x * p
    small = tl.abs(xs) < 0.25
    q = EL._rni_s32(tl.div_rn(tl.where(small, 1.0, xs), tl.broadcast_to(s1, (BLOCK_K,))))
    q = tl.where(small, 0, q)
    q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    tl.store(q_ptr + row * stride_q + offs, q, mask=mask)
    tl.store(s_ptr + row, s)
    # the window's A slot: S = (parity, m, pad, flags), F_NW_TILES = 2
    m = tl.load(S_ptr + 1)
    flags = tl.load(S_ptr + 3)
    if (row < m) & ((flags & 2) == 0):
        tl.store(slot_ptr + row * K + offs, q, mask=mask)
        tl.store(scale_ptr + row, s)


@triton.jit
def _dithera_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, d_ptr, offs_m, offs_n, M, N, T, m, flags, stride_cm,
                      ZL: tl.constexpr):
    mask_m = offs_m < M
    mask_n = offs_n < N
    EL._epilogue(acc, sa_ptr, sb_ptr, c_ptr, offs_m, offs_n, mask_m, mask_n, stride_cm)
    if (flags & 1) != 0:    # F_DUMP: the served y into the window's dump too
        EL._epilogue(acc, sa_ptr, sb_ptr, d_ptr, offs_m, offs_n, mask_m, mask_n, N)
    if ZL != 0:
        if (flags & 2) == 0:    # F_NW_TILES: no z
            zv = tl.where((offs_m < m)[:, None], acc, 0)
            if ZL == 1:     # row-major, a slot's N int32 a leaf
                za = offs_m[:, None] * N + offs_n[None, :]
            else:           # unit-major: unit (i // 32, j // 32)'s 32 x 32 int32 row-major at 1024 (i // 32 (N / 32) + j // 32)
                za = (((offs_m[:, None] >> 5) * (N >> 5) + (offs_n[None, :] >> 5)) * 1024 + (offs_m[:, None] & 31) * 32
                      + (offs_n[None, :] & 31))
            tl.store(z_ptr + za, zv, mask=(offs_m < T)[:, None] & mask_n[None, :])


@triton.jit
def dithera_mm(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, z_ptr, d_ptr, S_ptr, ws_ptr, cnt_ptr, M, N, K, T, stride_am,
               stride_bn, stride_cm, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               SPLIT_K: tl.constexpr, ZL: tl.constexpr):
    # exactlin_scaled_mm over T rows: a row tile at or past M multiplies nothing and stores only its zero z rows
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    m = tl.load(S_ptr + 1)
    flags = tl.load(S_ptr + 3)
    live = pid_m * BLOCK_M < M
    nk = tl.cdiv(K, BLOCK_K)
    per = tl.cdiv(nk, SPLIT_K)
    k_lo = tl.program_id(1) * per
    k_hi = tl.where(live, tl.minimum(k_lo + per, nk), k_lo)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_ptrs = a_ptr + stride_am * offs_m[:, None] + (k_lo * BLOCK_K + offs_k)[None, :]
    b_ptrs = b_ptr + (k_lo * BLOCK_K + offs_k)[:, None] + stride_bn * offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(k_lo, k_hi):
        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    if SPLIT_K == 1:
        _dithera_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, d_ptr, offs_m, offs_n, M, N, T, m, flags, stride_cm, ZL)
    else:
        if live:
            # exactlin_scaled_mm's split-K: the workspace and the tile's counter zero again after the last split
            mask = mask_m[:, None] & mask_n[None, :]
            ws_ptrs = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.atomic_add(ws_ptrs, acc, mask=mask, sem="relaxed")
            EL._membar_gl(pid)
            tl.debug_barrier()
            done = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
            if done == SPLIT_K - 1:
                EL._membar_gl(pid)
                tot = tl.atomic_xchg(ws_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32), mask=mask, sem="relaxed")
                tl.atomic_xchg(cnt_ptr + pid, 0, sem="relaxed")
                _dithera_epilogue(tot, sa_ptr, sb_ptr, c_ptr, z_ptr, d_ptr, offs_m, offs_n, M, N, T, m, flags, stride_cm, ZL)
        elif tl.program_id(1) == 0:
            _dithera_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, d_ptr, offs_m, offs_n, M, N, T, m, flags, stride_cm, ZL)


@triton.jit
def dithera_mm_tma(a_desc, b_desc, sa_ptr, sb_ptr, c_ptr, z_ptr, d_ptr, S_ptr, M, N, K, T, stride_cm,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, WARP_SPECIALIZE: tl.constexpr,
                   ZL: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    off_m = (pid // num_pid_n) * BLOCK_M
    off_n = (pid % num_pid_n) * BLOCK_N
    m = tl.load(S_ptr + 1)
    flags = tl.load(S_ptr + 3)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    nk = tl.where(off_m < M, tl.cdiv(K, BLOCK_K), 0)
    for k in tl.range(0, nk, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([off_m, k * BLOCK_K])
        b = b_desc.load([off_n, k * BLOCK_K])
        acc = tl.dot(a, b.T, acc, out_dtype=tl.int32)
    offs_m = off_m + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = off_n + tl.arange(0, BLOCK_N).to(tl.int64)
    _dithera_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, d_ptr, offs_m, offs_n, M, N, T, m, flags, stride_cm, ZL)


@triton.jit
def dithera_commit(src_ptr, dst_ptr, S_ptr, n, BLOCK: tl.constexpr):
    # F_NW_UNITS = 4: the step's receipts stay unwritten
    flags = tl.load(S_ptr + 3)
    if (flags & 4) == 0:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=offs < n), mask=offs < n)


@triton.jit
def dithera_fault(dst_ptr, S_ptr, at_ptr, L, bit, BLOCK: tl.constexpr):
    # F_FAULT = 8: word `at[w]` of the committed receipts flipped at `bit`, every call w
    flags = tl.load(S_ptr + 3)
    if (flags & 8) != 0:
        i = tl.arange(0, BLOCK)
        at = tl.load(at_ptr + i, mask=i < L, other=0)
        v = tl.load(dst_ptr + at, mask=i < L)
        tl.store(dst_ptr + at, v ^ bit, mask=i < L)


def _piecewise() -> bool:
    """Whether vLLM is capturing or running piecewise graphs (its forward context's mode); unknown counts as yes."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if not is_forward_context_available():
            return False
        return getattr(getattr(get_forward_context(), "cudagraph_runtime_mode", None), "name", "PIECEWISE") == "PIECEWISE"
    except Exception:  # noqa: BLE001
        return True


class Side:
    """The serving stream's GEMM state and the side stream's units over a dither_one Window."""

    def __init__(self, win: DO.Window, unit: str, h1u: str | None, z: str = "row"):
        if unit not in UNITS:
            raise ValueError(f"dither-a's unit: one of {UNITS}, not {unit!r}")
        self.win, self.unit, self.z, self.zl, dev = win, unit, z, ZMODES[z], win.dev
        self.side = torch.cuda.Stream(device=dev)
        self.fork, self.join = torch.cuda.Event(), torch.cuda.Event()
        self.work = EL.Workspace(dev)
        self.arch = EL.arch(dev.index)
        self.stage = torch.empty_like(win.plan.out)
        self.at = torch.tensor([win.plan.offs[w][0] for w in range(win.L)], dtype=torch.int64, device=dev)
        self.shift = int(win.o.h1o_fault_shift())
        self.ks1 = [(C.c_int * 1)(k) for _, k in win.nk]
        self.sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.versions = [H1R_BLOCK if n // 32 > self.sms else H1R_SPLIT for n, _ in win.nk]
        self.joins = self.calls = 0
        self.cfgs: dict = {}
        if z == "unit":
            win.pre["z"] = win.c.set([Tab(Dom(ZU_KIND, zu_fields(win, w, 0), zu_count(win, w), {"step": (1, 0)}),
                                          zu_count(win, w), data=win.z[w].view(torch.uint8), stride=4096, body=4096,
                                          length=4096, lab=("z/", f"/{w}"), out=win.pre["mid"].leaves_addr(2) + 64 * w)
                                      for w in range(win.L)])
        elif z == "none":
            zset, build = win.pre["z"], win.c.build
            win.c.build = lambda st, step, domains=True: None if st is zset else build(st, step, domains)
        if unit == "kh":
            if not h1u:
                raise RuntimeError("dither-a's kh unit needs h1u.so (HB_H1U)")
            h = self.h = C.CDLL(h1u)
            vp, u32 = C.c_void_p, C.c_uint
            h.kh_unit_ws.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp, u32, vp]
            h.kh_ws_bytes.restype = C.c_ulonglong
            h.kh_ws_bytes.argtypes = [u32, u32, u32]
            h.kh_noise_blocks.restype = u32
            h.kh_noise_blocks.argtypes = [u32]
            h.kh_words.restype = u32
            h.kh_words.argtypes = [u32]
            self.seed0, self.nb, need, s0 = [], [], 0, 0
            for w, (n, k) in enumerate(win.nk):
                _, units, nw = win.plan.offs[w]
                if units != (win.t // 32) * (n // 32) or int(h.kh_words(k)) != nw:
                    raise RuntimeError(f"linear {w}: the plan's {units} units of {nw} words, kh_unit_ws' "
                                       f"{(win.t // 32) * (n // 32)} of {int(h.kh_words(k))}")
                self.seed0.append(s0)
                self.nb.append(int(h.kh_noise_blocks(k)))
                need = max(need, int(h.kh_ws_bytes(win.t, n, k)))
                s0 += units
            self.ws = torch.empty((max(need, 64),), dtype=torch.uint8, device=dev)

    def gemm(self, w: int, xq: torch.Tensor, s_a: torch.Tensor, wt: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor):
        win = self.win
        M, k = xq.shape
        n = wt.shape[0]
        cfg = EL.pick(M, n, k, self.arch)
        self.cfgs[(n, k, M)] = str(cfg)
        bm, bn, bk, warps, stages, split, tma = cfg
        z = win.z[w]
        d = win.ydump[w] if win.ydump is not None else y
        tiles = triton.cdiv(win.t, bm) * triton.cdiv(n, bn)
        if tma:
            from triton.tools.tensor_descriptor import TensorDescriptor
            dithera_mm_tma[(tiles,)](TensorDescriptor.from_tensor(xq, [bm, bk]), TensorDescriptor.from_tensor(wt, [bn, bk]),
                                     s_a, s_w, y, z, d, win.S, M, n, k, win.t, y.stride(0), BLOCK_M=bm, BLOCK_N=bn,
                                     BLOCK_K=bk, WARP_SPECIALIZE=tma == 2, ZL=self.zl, num_warps=warps, num_stages=stages,
                                     **EL.OPTS)
            return
        if split > 1:
            assert M * n <= self.work.ws.numel() and tiles <= self.work.cnt.numel(), (M, n, tiles)
            ws, cnt = self.work.ws, self.work.cnt
        else:
            ws, cnt = y, y
        dithera_mm[(tiles, split)](xq, wt, s_a, s_w, y, z, d, win.S, ws, cnt, M, n, k, win.t, xq.stride(0), wt.stride(0),
                                   y.stride(0), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, ZL=self.zl,
                                   num_warps=warps, num_stages=stages, **EL.OPTS)

    def units(self, w: int, M: int, st: int) -> int:
        win = self.win
        n, k = win.nk[w]
        tab = win.plan.table.data_ptr() + win.call_bytes * w
        if self.unit == "h1r":
            return win.r.lib.h1r_launch(tab, 1, win.units[w], self.ks1[w], win.salt.data_ptr(), win.step_base.data_ptr(),
                                        win.plan.seeds.data_ptr(), self.stage.data_ptr(), 0, self.versions[w], st)
        rc = win.r.lib.h1r_launch(tab, 1, win.units[w], self.ks1[w], win.salt.data_ptr(), win.step_base.data_ptr(),
                                  win.plan.seeds.data_ptr(), self.stage.data_ptr(), 0, 0, st)
        live = -(-min(M, win.t) // 32)
        return rc or self.h.kh_unit_ws(win.slot[w].data_ptr(), win.wt[w].data_ptr(), win.t, n, k,
                                       win.plan.seeds.data_ptr() + 8 * 16 * self.seed0[w], win.live.data_ptr(),
                                       self.stage.data_ptr() + 4 * win.plan.offs[w][0], self.ws.data_ptr(),
                                       auto_j(live * (n // 32), self.nb[w], self.sms), st)

    def call(self, w: int, x: torch.Tensor, wt: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor) -> None:
        win = self.win
        M, k = x.shape
        main = torch.cuda.current_stream()
        xq = torch.empty((M, k), dtype=torch.int8, device=x.device)
        s_a = torch.empty(M, dtype=torch.float32, device=x.device)
        exactlin_quant_rows_dithera[(M,)](x, xq, s_a, win.slot[w], win.scale[w], win.S, k, x.stride(0), xq.stride(0),
                                          BLOCK_K=triton.next_power_of_2(k), num_warps=EL.quant_warps(k), **EL.OPTS)
        self.fork.record(main)
        self.side.wait_event(self.fork)
        self.gemm(w, xq, s_a, wt, s_w, y)
        st = self.side.cuda_stream
        rc = win.a_call(w, win.slot[w], win.scale[w], st) or self.units(w, M, st)
        if rc:
            raise RuntimeError(f"dither-a call {w} (M={M}): a side launch returned {rc}")
        last = w == win.L - 1
        if last:
            with torch.cuda.stream(self.side):
                n = self.stage.numel()
                dithera_commit[(triton.cdiv(n, 4096),)](self.stage, win.plan.out, win.S, n, BLOCK=4096)
                dithera_fault[(1,)](win.plan.out, win.S, self.at, win.L, 1 << self.shift,
                                    BLOCK=triton.next_power_of_2(win.L))
        if last or (w % 4 == 0 and _piecewise()):
            self.join.record(self.side)
            main.wait_event(self.join)
            self.joins += 1
        self.calls += 1

    def info(self) -> dict:
        return {"form": FORM, "unit": self.unit, "z": self.z, "a_form": self.win.c.a_form, "calls": self.calls, "joins": self.joins,
                "h1r_versions": {DF.SITES[w % 4]: self.versions[w] for w in range(min(4, self.win.L))}
                if self.unit == "h1r" else None, "gemm_cfgs": {f"{n}x{k}@M{m}": c for (n, k, m), c in self.cfgs.items()}}


def zu_fields_of(window: str, s: int, w: int, module: str, t: int, n: int) -> dict:
    return {"window": window, "step": s, "linear": w, "module": module, "slots": t, "n": n,
            "unit": "32x32 int32 LE row-major, units row block then column block", "schema": CM.SCHEMA}


def zu_fields(win: DO.Window, w: int, s: int) -> dict:
    return zu_fields_of(win.window, s, w, win.names[w], win.t, win.nk[w][0])


def zu_count(win: DO.Window, w: int) -> int:
    n = win.nk[w][0]
    if win.t % 32 or n % 32:
        raise ValueError(f"linear {w}: T = {win.t}, N = {n}, not 32 x 32 units")
    return (win.t // 32) * (n // 32)


def zrows(win: DO.Window, z: torch.Tensor, w: int) -> torch.Tensor:
    """A z slot, row-major (T, N), from its layout under the arm's z mode."""
    if not SIDE or SIDE[0].z != "unit":
        return z
    n = win.nk[w][0]
    return z.reshape(win.t // 32, n // 32, 32, 32).permute(0, 2, 1, 3).reshape(win.t, n)


SIDE: list[Side] = []
CONF: dict = {}


def side() -> Side:
    if not SIDE:
        win = DO._window(torch.device("cuda"))
        SIDE.append(Side(win, CONF["unit"], CONF["h1u"], CONF["z"]))
        before = DO.Window.before

        def before_a(self: DO.Window, m: int) -> None:
            before(self, m)
            if self.cap:
                SIDE[0].stage.view(torch.uint8).fill_(DF.POISON)
                torch.cuda.synchronize()
        DO.Window.before = before_a
    return SIDE[0]


_LIB = torch.library.Library(LIB, "DEF")
_LIB.define("mm(Tensor x, Tensor w, Tensor s_w, int idx) -> Tensor")


def _mm_cuda(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, idx: int) -> torch.Tensor:
    y = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    if x.shape[0]:
        side().call(idx, x if x.stride(1) == 1 else x.contiguous(), w, s_w.reshape(-1), y)
    return y


_LIB.impl("mm", _mm_cuda, "CUDA")


@torch.library.register_fake(f"{LIB}::mm")
def _mm_fake(x, w, s_w, idx):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16)


def patch(t: int, h1c: str, h1o: str, unit: str = "h1r", h1u: str | None = None, key: bytes | None = None,
          z: str = "row") -> None:
    """dither_one's patch (its weights' registry, the window's libraries), then every int8-exact linear's `apply`
    through ``dithera::mm`` (before the engine is built, in its process)."""
    from verity_vllm.engine import int8_vllm as IV
    if unit not in UNITS or z not in ZMODES:
        raise ValueError(f"dither-a's unit: one of {UNITS}, not {unit!r}; its z: one of {tuple(ZMODES)}, not {z!r}")
    CONF.update(unit=unit, h1u=h1u, z=z)
    os.environ.setdefault("DITHER_A_FORM", "3")
    DO.patch(t=t, h1c=h1c, h1o=h1o, key=key, gate=True)

    def apply(self, layer, x, bias=None):
        if bias is not None or x.dtype != torch.bfloat16:
            raise ValueError(f"{self.prefix}: dither-a takes bf16 activations and no bias")
        lead, k = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, k)
        n = layer.weight.shape[0]
        if x2.shape[0] == 0:
            return x.new_empty(*lead, n)
        return torch.ops.dithera.mm(x2, layer.weight, layer.weight_scale, self.dither_idx).reshape(*lead, n)

    IV.Int8PlainLinearMethod.apply = apply


def stats(win: DO.Window):
    def doc() -> dict:
        return {**win.stats(), "dither_a": SIDE[0].info() if SIDE else None}
    return doc


# -- the gate ----------------------------------------------------------------------------------------------------------------

def check(win: DO.Window, lib_h1o: str, which: str = "gate", n_cpu_units: int = 3) -> dict:
    """`which`'s capture against references other than the served kernels (the module docstring's list)."""
    from verity_vllm.engine import int8_exact as E
    g = win.caps.get(which, {})
    if "receipts" not in g or "a" not in g:
        return {"ok": False, "complete": False, "why": f"{which} step incomplete: {sorted(g)}"}
    m, s, t, L = g["m"], g["s"], win.t, win.L
    doc: dict = {"which": which, "form": FORM, "complete": True, "s": s, "m": m, "t": t, "poison": f"{DF.POISON:#04x}",
                 "epsilon": "seeded", "unit": SIDE[0].unit if SIDE else None}
    key, W = win.key, win.window
    bits = np.arange(t) < m
    wrows = CR.word_rows(CR.words(bits))
    a_cpu, s_cpu, z_cpu = {}, {}, {}
    for w in range(L):
        n, k = win.nk[w]
        q, sc = g["a"][w].cpu().numpy(), g["sa"][w].cpu().numpy()
        fa = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "k": k, "schema": CM.SCHEMA}
        a_cpu[w] = CM.s8_table(CM.domain(fa, t, "a-table"), key, f"a/{s}/{w}", q, sc).root
        fs = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "schema": CM.SCHEMA}
        sd = CM.domain(fs, len(wrows), "admit")
        s_cpu[w] = CM.levels(sd, [CM.leaf(sd, key, f"s/{s}/{w}", i, x)[0] for i, x in enumerate(wrows)])[-1][0]
        if SIDE and SIDE[0].z == "unit":
            zd = CM.domain(zu_fields(win, w, s), zu_count(win, w), ZU_KIND)
            zb = g["z"][w].cpu().numpy().astype("<i4").view(np.uint8).reshape(-1, 4096)
        else:
            fz = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "n": n, "schema": CM.SCHEMA}
            zd = CM.domain(fz, t, "z-table")
            zb = g["z"][w].cpu().numpy().astype("<i4").view(np.uint8).reshape(t, -1)
        z_cpu[w] = CM.levels(zd, [CM.leaf(zd, key, f"z/{s}/{w}", i, bytes(zb[i]))[0] for i in range(len(zb))])[-1][0]
    doc["a_roots_bad"] = [w for w in range(L) if a_cpu[w] != g["a_roots"][w]]
    doc["a_bad_diag"] = DO.a_bad_diag(g, doc["a_roots_bad"])
    doc["admit_roots_bad"] = [w for w in range(L) if s_cpu[w] != g["admit_roots"][w]]
    doc["z_roots_bad"] = [w for w in range(L) if z_cpu[w] != g["z_roots"][w]]
    f_step = {"window": W, "step": s, "linears": L}
    a_step = CM.step_root(CM.domain(f_step, L, "a-step"), [a_cpu[w] for w in range(L)])
    admit_step = CM.step_root(CM.domain(f_step, L, "admit-step"), [s_cpu[w] for w in range(L)])
    z_step = CM.step_root(CM.domain(f_step, L, "z-step"), [z_cpu[w] for w in range(L)])
    rnd = CM.step_root(CM.domain({"window": W, "step": s, "leaves": "a-step root, admit-step root, z-step root"}, 3, "round"),
                       [a_step, admit_step, z_step])
    doc["round"] = {"a_step": a_step == g["leaves"][0], "admit_step": admit_step == g["leaves"][1],
                    "z_step": z_step == g["leaves"][2], "root": rnd == g["root"]}
    # z = A B^T by int_mm on the committed A, zero past m; the served y = dequant(z)
    z_bad, y_bad = [], []
    zr = [zrows(win, g["z"][w], w) for w in range(L)]
    for w in range(L):
        zt = E.int_mm(g["a"][w][:m], win.wt[w])
        if not (torch.equal(zr[w][:m], zt) and not bool(zr[w][m:].any())):
            z_bad.append(w)
        if g["y"] is not None and not torch.equal(
                E.dequantize(zr[w][:m], g["sa"][w][:m], win.sw[w]).view(torch.int16), g["y"][w].view(torch.int16)):
            y_bad.append(w)
    doc["z_vs_int_mm_bad"], doc["y_vs_dequant_z_bad"] = z_bad, y_bad if g["y"] is not None else None
    # every receipt against h1r v1 on the saved slots, the window's salt and the CPU's roots
    r = H1R(lib_h1o)
    live = torch.from_numpy((np.arange(t // 32) * 32 < m).astype(np.uint8)).to(win.dev)
    roots = [torch.from_numpy(np.concatenate([np.frombuffer(a_cpu[w], ">u8"), np.frombuffer(g["b_roots"][w], ">u8")])
                              .astype("<u8").view("<i8").copy()).to(win.dev) for w in range(L)]
    plan = r.plan([Call(g["a"][w], win.wt[w], s * win.u_step + win.offs[w], roots[w], live) for w in range(L)])
    r.launch(plan, win.salt, version=1)
    torch.cuda.synchronize()
    got = g["receipts"]
    shift = SIDE[0].shift if SIDE else 24

    def got_of(w: int) -> torch.Tensor:
        o_, units, nw = win.plan.offs[w]
        return got[o_:o_ + units * nw].view(units, nw)

    rc_bad, fault_bad = [], []
    for w in range(L):
        ref, mine = plan.receipts(w), got_of(w)
        if which == "fault":
            x = (ref ^ mine)
            cells = x.nonzero().tolist()
            if cells != [[0, 0]] or (int(x[0, 0]) & M32) != (1 << shift):
                fault_bad.append({"call": w, "words_off": cells[:4], "xor00": int(x[0, 0]) & M32})
        if (ref != mine).any():
            rc_bad.append(w)
    doc["receipts_vs_h1r_bad"] = rc_bad
    doc["live_units"] = int(sum((min(m, t) + 31) // 32 * (n // 32) for n, _ in win.nk))
    commits_ok = (not doc["a_roots_bad"] and not doc["admit_roots_bad"] and not doc["z_roots_bad"]
                  and all(doc["round"].values()))
    if which == "fault":
        doc["fault_shift"], doc["fault_bad"] = shift, fault_bad
        doc["conforming"] = not fault_bad and not z_bad and not y_bad and commits_ok
        doc["ok"] = not rc_bad and doc["conforming"]
        return doc
    # three units' receipts by CA's definition on the CPU and by H1V3's own code; its z (peeled: A B^T) against ours
    units_doc, h1v3_in = [], []
    if which == "gate":
        rng = np.random.default_rng(s)
        for w in [0, 3, 2][:n_cpu_units]:
            n, k = win.nk[w]
            u = int(rng.integers(n // 32))
            a32 = g["a"][w][0:32].cpu().numpy()
            b32 = win.wt[w][32 * u:32 * u + 32].cpu().numpy()
            pos = s * win.u_step + win.offs[w] + u
            ref = DO.h1v3_unit(a32, b32, win.salt_bytes, pos, (a_cpu[w], g["b_roots"][w]))
            zk = zr[w][0:32, 32 * u:32 * u + 32].cpu().numpy().astype(np.int64)
            rk = (got_of(w)[u].cpu().numpy().astype(np.int64) & M32).tolist()
            units_doc.append({"call": w, "unit": u, "k": k, "receipt_equal": rk == ref["receipt"],
                              "z_equal_today": bool((zk == ref["today"]).all()), "_zk": zk, "_rk": rk})
            h1v3_in.append({"a": (a32.astype(np.int64) & 0xFF).tolist(), "b": (b32.astype(np.int64) & 0xFF).tolist(),
                            "salt": win.salt_bytes.hex(), "position": int(pos), "roots": [a_cpu[w].hex(), g["b_roots"][w].hex()]})
        hv = DO.h1v3_itself(h1v3_in)
        if "units" in hv:
            for x, hu in zip(units_doc, hv["units"]):
                x["h1v3_receipt_equal"] = x["_rk"] == hu["receipt"]
                if hv.get("peeled"):
                    x["h1v3_z_equal"] = bool((x["_zk"] == np.array(hu["z"], dtype=np.int64)).all())
        for x in units_doc:
            del x["_zk"], x["_rk"]
        doc["h1v3"] = {k_: v for k_, v in hv.items() if k_ != "units"}
    doc["vs_h1v3_cpu"] = units_doc
    h1v3_ok = which != "gate" or doc["h1v3"].get("tree") is None or (
        "units" in hv and all(x["h1v3_receipt_equal"] and x.get("h1v3_z_equal", True) for x in units_doc))
    doc["ok"] = (not rc_bad and not z_bad and (g["y"] is None or not y_bad) and commits_ok and h1v3_ok
                 and all(x["receipt_equal"] and x["z_equal_today"] for x in units_doc))
    return doc


def gate_verdict(gate: dict, control: dict, fault: dict) -> dict:
    """The poisoned gate passes; its no-write control (complete) is refused on z and on the receipts; the fault step is
    refused on the receipts at exactly the flipped words, with z, y and every commitment but the receipts' kept."""
    refused = bool(control.get("complete") and not control.get("ok") and control.get("z_vs_int_mm_bad")
                   and control.get("receipts_vs_h1r_bad"))
    fault_refused = bool(fault.get("complete") and not fault.get("ok") and fault.get("conforming"))
    return {"ok": bool(gate.get("ok")) and refused and fault_refused, "form": FORM, "gate_ok": bool(gate.get("ok")),
            "control_refused": refused, "fault_refused": fault_refused,
            "control_bad": {k: len(control.get(k) or []) for k in ("z_vs_int_mm_bad", "y_vs_dequant_z_bad",
                                                                    "receipts_vs_h1r_bad", "a_roots_bad", "z_roots_bad")},
            "fault_receipt_calls_off": len(fault.get("receipts_vs_h1r_bad") or [])}
