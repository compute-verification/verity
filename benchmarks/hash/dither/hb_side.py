"""The `dither-hb` arm's receipts: `dither-side`'s side receipts (`dither_fused`, rc="side": step s's receipts on a
second stream after its coin, beside step s + 1) with h1u.cu's two-launch unit, `kh_unit_ws` (the noise
grid-wide into a workspace, then the multiply and fold from it), in place of h1r's single-launch version 2.

`install(win, path)` replaces the window's h1r object with `SideH1R`, which `DitherState._side_receipts` launches through:
version 2 becomes h1r's own seeds (version 0, the same h1r_seeds kernel) and then `kh_unit_ws` over the plan's calls, one at
a time on the side stream, into the plan's seeds and receipts buffers; a no-write launch (the gate's control) runs the
seeds and none of the units. Every other version and every other attribute is h1r's, so the window's plans, tables, trees,
coins, poisoning and `dither_fused.check` (every receipt against h1r v1, three units against h1tile.reference) are
unchanged. What is committed, before which coin, and every receipt are the same as `dither-side`'s.

The library is h1u.cu built by job.sh (a variant's .so, `HB_H1U`), in the run's inputs.
"""
from __future__ import annotations

import ctypes

import torch


def auto_j(units: int, nb: int, sms: int) -> int:
    """h1bench.py's: the largest J (noise chunks a lane) that still leaves about 64 warps an SM."""
    return max(1, min(32, (units * nb) // (32 * sms * 64)))


class Units:
    """`kh_unit_ws` over a plan's calls: each call's A [T, K] and B [N, K] s8 (row stride K), its live row tiles, its
    seeds at the plan's seed offset (16 words a unit) and its receipts at the plan's word offset, in h1r's unit order."""

    def __init__(self, path: str, plans: list, sms: int):
        h = self.h = ctypes.CDLL(path)
        vp, u32 = ctypes.c_void_p, ctypes.c_uint
        h.kh_unit_ws.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp, u32, vp]
        h.kh_ws_bytes.restype = ctypes.c_ulonglong
        h.kh_ws_bytes.argtypes = [u32, u32, u32]
        h.kh_noise_blocks.restype = u32
        h.kh_noise_blocks.argtypes = [u32]
        h.kh_words.restype = u32
        h.kh_words.argtypes = [u32]
        self.sms, self.calls, need = sms, {}, 0
        for plan in plans:
            rows, s0 = [], 0
            for i, (c, (o, units, w)) in enumerate(zip(plan.calls, plan.offs)):
                t, k = c.a.shape
                n = c.b.shape[0]
                if c.b.shape[1] != k or c.a.stride(0) != k or c.b.stride(0) != k:
                    raise ValueError(f"call {i}: A {tuple(c.a.shape)} stride {c.a.stride(0)}, B {tuple(c.b.shape)} stride "
                                     f"{c.b.stride(0)}: kh_unit_ws takes row stride K")
                if units != (t // 32) * (n // 32) or int(h.kh_words(k)) != w:
                    raise ValueError(f"call {i}: the plan's {units} units of {w} words, kh_unit_ws's "
                                     f"{(t // 32) * (n // 32)} of {int(h.kh_words(k))}")
                rows.append((c.a.data_ptr(), c.b.data_ptr(), t, n, k, plan.seeds.data_ptr() + 8 * 16 * s0,
                             c.live.data_ptr(), plan.out.data_ptr() + 4 * o, int(h.kh_noise_blocks(k))))
                need = max(need, int(h.kh_ws_bytes(t, n, k)))
                s0 += units
            self.calls[id(plan)] = rows
        self.ws = torch.empty((max(need, 64),), dtype=torch.uint8, device=plans[0].out.device)

    def launch(self, plan, m: int) -> None:
        st, live = torch.cuda.current_stream().cuda_stream, -(-m // 32)
        for a, b, t, n, k, seeds, lv, out, nb in self.calls[id(plan)]:
            rc = self.h.kh_unit_ws(a, b, t, n, k, seeds, lv, out, self.ws.data_ptr(), auto_j(live * (n // 32), nb, self.sms),
                                   st)
            if rc:
                raise RuntimeError(f"kh_unit_ws rc {rc} (A [{t}, {k}], B [{n}, {k}])")


class SideH1R:
    """The window's h1r object for the dither-hb arm: version 2 is h1r's seeds, then `Units` over the plan."""

    def __init__(self, r, units: Units, win):
        self.r, self.units, self.win = r, units, win

    def __getattr__(self, name: str):
        return getattr(self.r, name)

    def launch(self, plan, salt: torch.Tensor, nowrite: bool = False, version: int = 2,
               step_base: torch.Tensor | None = None) -> None:
        if version != 2:
            return self.r.launch(plan, salt, nowrite=nowrite, version=version, step_base=step_base)
        self.r.launch(plan, salt, version=0, step_base=step_base)
        if not nowrite:
            self.units.launch(plan, self.win.m)
        return None


def install(win, path: str) -> SideH1R:
    if win.rc_mode != "side":
        raise ValueError(f"dither-hb replaces the side receipts; the window's receipts mode is {win.rc_mode!r}")
    units = Units(path, win.plans, torch.cuda.get_device_properties(win.dev).multi_processor_count)
    win.r = SideH1R(win.r, units, win)
    return win.r
