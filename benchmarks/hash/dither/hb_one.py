"""The `dither-1s-hb` arm: the served linear in one stream (`dither_one`, H1V3 with the exact wrap correction) with
the one-stream unit h1u.cu's `kh_one` (the h1u walk, its receipts, and z = O' - sum_d 128^d D_d
+ 256 X (wb+ - wb-)^T and y = dequant(z) from the same MMAs), in place of h1o.cu's `h1o_units`.

`patch(path)` replaces `dither_one.DitherState.launch` before the engine is built (vLLM captures its graphs inside `LLM()`, so
an install after it would be too late). A call keeps dither-1s's A slot and A table: on dither-rtx's 007a40aac1 and later
(the DitherState has `apre`) one `h1c_a_call` (the slot rows and the table, with `h1c_a_step` once a step in the DitherState's own
step hook), before it h1o_slot and h1c_build. Then h1r's seeds (version 0) and `kh_one` on the A slot (xq's rows < m and
zero rows above) and the layer's B, into the window's z slot, the plan's seeds and receipts at the call's offsets, and
the op's z and y. The step (m, flags) is read from the window's step on the device, as h1o reads it, and the flags mean
h1o's (F_DUMP, F_NW_TILES, F_NW_UNITS, F_FAULT, F_FAULT_WB): so `dither_one.check`, with h1o.so's own references
(h1o_ref, h1r v1, the CPU H1V3 units, H1V3's h1tile.reference), gates this kernel unchanged. One difference: under the
gate's no-write control (F_NW_TILES) the slot may go unwritten, so that one step's y comes from the poisoned slot (h1o
reads xq); dither-1s arms' tokens are reported, not gated.

The workspace (`kh_unit_ws`' noise, the largest linear's `kh_ws_bytes` at T rows) is allocated at the window's first
call, vLLM's eager profile run; each call's J is set for the row tiles its M can hold live (M is fixed per captured
graph). The library is h1u.cu built by job.sh (`HB_H1U`), in the run's inputs.

`patch(path, peel=True)` is the `dither-1s-hbp` arm: `kh_one_peel` (the peel) in kh_one's place, the same
receipts and z = A B^T, so y is today's; its gate is hb_peel.py's.
"""
from __future__ import annotations

import ctypes as C

import torch

import dither_one as DO
from hb_side import auto_j


class One:
    """`kh_one` over the window's calls: per linear its seed and receipt offsets in the window's plan."""

    def __init__(self, path: str, win: DO.DitherState, peel: bool = False):
        h = self.h = C.CDLL(path)
        vp, u32 = C.c_void_p, C.c_uint
        h.kh_one.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp, u32, vp, vp, vp, vp, vp, vp, vp, u32, vp]
        if peel and not hasattr(h, "kh_one_peel"):
            raise RuntimeError(f"{path} has no kh_one_peel")
        if peel:
            h.kh_one_peel.argtypes = h.kh_one.argtypes
        self.fn = h.kh_one_peel if peel else h.kh_one
        h.kh_ws_bytes.restype = C.c_ulonglong
        h.kh_ws_bytes.argtypes = [u32, u32, u32]
        h.kh_noise_blocks.restype = u32
        h.kh_noise_blocks.argtypes = [u32]
        h.kh_words.restype = u32
        h.kh_words.argtypes = [u32]
        if h.kh_fault_shift() != win.o.h1o_fault_shift():
            raise RuntimeError(f"kh_one's fault shift {h.kh_fault_shift()} is not h1o's {win.o.h1o_fault_shift()}")
        self.sms = torch.cuda.get_device_properties(win.dev).multi_processor_count
        self.seed0, self.nb, need, s0 = [], [], 0, 0
        for w, (n, k) in enumerate(win.nk):
            o, units, nw = win.plan.offs[w]
            if units != (win.t // 32) * (n // 32) or int(h.kh_words(k)) != nw:
                raise RuntimeError(f"linear {w}: the plan's {units} units of {nw} words, kh_one's "
                                   f"{(win.t // 32) * (n // 32)} of {int(h.kh_words(k))}")
            if win.wt[w].stride(0) != k or win.slot[w].stride(0) != k:
                raise RuntimeError(f"linear {w}: kh_one takes row stride K")
            self.seed0.append(s0)
            self.nb.append(int(h.kh_noise_blocks(k)))
            need = max(need, int(h.kh_ws_bytes(win.t, n, k)))
            s0 += units
        self.ws = torch.empty((max(need, 64),), dtype=torch.uint8, device=win.dev)
        self.ks1 = [(C.c_int * 1)(k) for _, k in win.nk]
        self.calls = 0

    def units(self, win: DO.DitherState, w: int, xq: torch.Tensor, s_a: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor,
              z: torch.Tensor, st: int) -> int:
        n, k = win.nk[w]
        M = int(xq.shape[0])
        j = auto_j(-(-min(M, win.t) // 32) * (n // 32), self.nb[w], self.sms)
        o = win.plan.offs[w][0]
        self.calls += 1
        return self.fn(win.slot[w].data_ptr(), win.wt[w].data_ptr(), win.t, n, k,
                       win.plan.seeds.data_ptr() + 8 * 16 * self.seed0[w], win.live.data_ptr(),
                       win.plan.out.data_ptr() + 4 * o, self.ws.data_ptr(), j, win.z[w].data_ptr(), z.data_ptr(),
                       y.data_ptr(), win.ydump[w].data_ptr() if win.ydump else None, s_a.data_ptr(),
                       s_w.data_ptr(), win.S.data_ptr(), M, st)


def launch(self: DO.DitherState, w: int, xq: torch.Tensor, s_a: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor,
           z: torch.Tensor) -> None:
    one = getattr(self, "hb_one", None)
    if one is None:
        one = self.hb_one = One(PATH[0], self, PEEL[0])
    st = torch.cuda.current_stream().cuda_stream
    a, tb = self.pre["a"], self.tab_bytes * w
    if hasattr(self, "apre"):
        rc = self.c.lib.h1c_a_call(C.addressof(a.host) + tb, a.dev.data_ptr() + tb,
                                   self.apre.data_ptr() + 8 * self.awords * w, self.S.data_ptr(), xq.data_ptr(),
                                   s_a.data_ptr(), self.slot[w].data_ptr(), self.scale[w].data_ptr(), self.nk[w][1], st)
    else:
        fw = self.F.data_ptr() + self.fsize * w
        rc = self.o.h1o_slot_launch(fw, self.S.data_ptr(), self.t, xq.data_ptr(), s_a.data_ptr(), st)
        rc = rc or self.c.lib.h1c_build(C.addressof(a.host) + tb, a.dev.data_ptr() + tb, 1, self.stepv.data_ptr(),
                                        self.c.btab.data_ptr(), 1, st)
    rc = rc or self.r.lib.h1r_launch(self.plan.table.data_ptr() + self.call_bytes * w, 1, self.units[w], one.ks1[w],
                                     self.salt.data_ptr(), self.step_base.data_ptr(), self.plan.seeds.data_ptr(),
                                     self.plan.out.data_ptr(), 0, 0, st)
    rc = rc or one.units(self, w, xq, s_a, s_w, y, z, st)
    if rc:
        raise RuntimeError(f"dither1-hb call {w} (M={xq.shape[0]}): a launch returned {rc}")
    self.calls += 1


PATH: list[str] = []
PEEL: list[bool] = [False]


def patch(path: str, peel: bool = False) -> None:
    PATH[:] = [path]
    PEEL[:] = [peel]
    DO.DitherState.launch = launch
