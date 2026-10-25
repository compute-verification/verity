"""dithera_mm's GEMM configs at Llama-3.1-8B's decode shapes, its z/y records epilogue included (dither_a's -zy form: M
live rows of T slots, one record a 32 x 32 unit), each gated against the served config's outputs before it is timed.

Per shape (n, k), with kgate's activations quantized by int8_exact and `kbench.copies_for` weight copies (each with its
own y and records, so the weights stream from DRAM), timed kbench's way (a CUDA graph an arm, interleaved items):

    coarse   `coarse_grid`: BLOCK_N x BLOCK_K x SPLIT_K at BLOCK_M 32, 4 warps, 3 stages, and the TMA kernel's
    refine   the best `top` across warps x stages
    final    the best `top`, the served config (exactlin.pick, as dither_a's gemm takes it) and exactlin's scaled_mm at
             the served config (the same product without the z/y epilogue), `reps` items each

A config's gate: y and the records (z and y halves) equal the served config's into 0xA5, twice (a split K's workspace
and counters zero again), and the last copy's after the timing's replays; the served config's y is int8_exact's
(`dequantize(int_mm(...))`) bit for bit.  Writes sweep.jsonl and best.json ({"n,k": [m, best, us, served, us,
scaled_mm us]}, and the step's sum over 32 layers each way: what $DITHER_MM_CFG would take).  --block Q/D times every
item with the SMs whose %smid % D < Q held (`Blocker`: the SMs $DITHER_SO_UNITS_SMQ=Q/D gives a step's units, which
hold them through most of the next forward), so the configs are ranked on the SMs a served forward gets.  --grid
prefill starts from `prefill_grid` (prefill's M = T = 2048: 64- and 128-row tiles), and --groups G,.. adds a phase after
refine: the best `top` at each tile group G (dither_a's ":gG", `_tile`), so a config's text may end in ":gG".

    python dmmsweep.py --out OUT [--m 32] [--t 64] [--top 3] [--reps 9] [--block Q/D] [--grid prefill] [--groups 1,8]
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes as C
import json
import sys
from pathlib import Path

import torch
import triton

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "exactlin")]
import dither_a as DA  # noqa: E402
import exactlin as X  # noqa: E402
import kbench  # noqa: E402
import kgate  # noqa: E402

SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
LAYERS = 32
UR = UC = 32
WARPS = (4, 8)
STAGES = (2, 3, 4, 5)
HOLD_SRC = r"""
extern "C" __global__ void hold_sms(volatile int *stop, int *arrived, int *held, int q, int d) {
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    if ((int)(sm % d) < q) atomicAdd(held, 1);
    __threadfence();
    atomicAdd(arrived, 1);
    if ((int)(sm % d) >= q) return;
    while (*stop == 0) __nanosleep(2000);
}
extern "C" __global__ void set_stop(int *stop) { *stop = 1; __threadfence(); }
"""


class Blocker:
    """The SMs whose %smid % d < q held while a timed item runs (`item`): one one-thread CTA an SM, each taking the
    device's largest dynamic shared memory so no GEMM CTA shares its SM, parked until the item's kernels are done (a
    `set_stop` after them on the serving stream); the others exit at once."""

    def __init__(self, q: int, d: int):
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc
        self.cu, self.q, self.d = cu, q, d
        self.flags = torch.zeros(3, dtype=torch.int32, device="cuda")     # stop, arrived, held
        major, minor = torch.cuda.get_device_capability()
        prog = self._ok(nvrtc.nvrtcCreateProgram(HOLD_SRC.encode(), b"hold.cu", 0, [], []))
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            n = self._ok(nvrtc.nvrtcGetProgramLogSize(prog))
            buf = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, buf)
            raise RuntimeError(f"dmmsweep Blocker: nvrtc {res[0]!r}: {buf.decode(errors='replace')}")
        cubin = b" " * self._ok(nvrtc.nvrtcGetCUBINSize(prog))
        self._ok(nvrtc.nvrtcGetCUBIN(prog, cubin))
        self.mod = self._ok(cu.cuModuleLoadData(cubin))
        self.hold = self._ok(cu.cuModuleGetFunction(self.mod, b"hold_sms"))
        self.stop = self._ok(cu.cuModuleGetFunction(self.mod, b"set_stop"))
        dev = self._ok(cu.cuDeviceGet(torch.cuda.current_device()))
        self.smem = self._ok(cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev))
        self._ok(cu.cuFuncSetAttribute(self.hold, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                       self.smem))
        self.sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.side = torch.cuda.Stream()
        self.held: set[int] = set()     # each item's count of held SMs

    @staticmethod
    def _ok(res):
        err, *out = res if isinstance(res, tuple) else (res,)
        if int(err) != 0:
            raise RuntimeError(f"dmmsweep Blocker: {err!r}")
        return out[0] if len(out) == 1 else out

    def _launch(self, fn, grid: int, smem: int, stream, args: tuple, types: tuple) -> None:
        self._ok(self.cu.cuLaunchKernel(fn, grid, 1, 1, 1, 1, 1, smem, self.cu.CUstream(stream.cuda_stream),
                                        (args, types), 0))

    @contextlib.contextmanager
    def item(self):
        main = torch.cuda.current_stream()
        stop = self.flags.data_ptr()
        self.flags.zero_()
        self.side.wait_stream(main)
        self._launch(self.hold, self.sms, self.smem, self.side, (stop, stop + 4, stop + 8, self.q, self.d),
                     (C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int))
        for _ in range(100000):
            arrived, held = self.flags[1:].tolist()
            if arrived >= self.sms:
                break
        else:
            self._launch(self.stop, 1, 0, main, (stop,), (C.c_void_p,))
            raise RuntimeError(f"dmmsweep Blocker: {arrived} of {self.sms} hold CTAs started")
        self.held.add(held)
        try:
            yield
        finally:
            self._launch(self.stop, 1, 0, main, (stop,), (C.c_void_p,))


def coarse_grid() -> list[X.Cfg]:
    grid = [X.Cfg(32, bn, bk, 4, 3, sp) for bn in (16, 32, 64, 128) for bk in (128, 256, 512) for sp in (1, 2, 4)
            if not (bn == 128 and sp > 1)]
    return grid + [X.Cfg(32, bn, bk, 4, 3, 1, t) for bn in (32, 64, 128) for bk in (128, 256) for t in (1, 2)]


def prefill_grid() -> list[X.Cfg]:
    """64- and 128-row tiles at 3 stages whose A and B stages fit 99 KiB of shared memory, 8 warps past 128 x 128."""
    return [X.Cfg(bm, bn, bk, 8 if bm * bn > 128 * 128 else 4, 3, 1, t) for bm in (64, 128) for bn in (64, 128, 256)
            for bk in (64, 128, 256) for t in (0, 1, 2) if 3 * (bm + bn) * bk <= 99 * 1024]


class GCfg(X.Cfg):
    """An exactlin Cfg and its tile group g (dither_a's ":gG"); equal tuples of other groups are other configs, so
    lists of them are deduplicated by text (`uniq`)."""

    def __new__(cls, *a, g: int = 1):
        self = super().__new__(cls, *a)
        self.g = g
        return self

    @classmethod
    def parse(cls, text: str) -> GCfg:
        return cls(*X.Cfg.parse(text), g=DA.mm_group(text))

    def __str__(self) -> str:
        return super().__str__() + (f":g{self.g}" if self.g > 1 else "")


def uniq(cfgs) -> list:
    return list({str(c): c for c in cfgs}.values())


def gemm(cfg: X.Cfg, xq, sa, w, sw, y, zy, S, T: int, work) -> None:
    """dither_a's `gemm` for one call at `cfg` (ZL 2, no dump: flags 0)."""
    M, k = xq.shape
    n = w.shape[0]
    bm, bn, bk, warps, stages, split, tma = cfg
    group = getattr(cfg, "g", 1)
    z, yl = zy.view(torch.int32), zy.view(torch.int16)
    tiles = triton.cdiv(T, bm) * triton.cdiv(n, bn)
    if tma:
        from triton.tools.tensor_descriptor import TensorDescriptor
        a_desc, b_desc = TensorDescriptor.from_tensor(xq, [bm, bk]), TensorDescriptor.from_tensor(w, [bn, bk])
        DA.dithera_mm_tma[(tiles,)](a_desc, b_desc, sa, sw, y, z, yl, y, S, M, n, k, T, y.stride(0), BLOCK_M=bm,
                                    BLOCK_N=bn, BLOCK_K=bk, WARP_SPECIALIZE=tma == 2, ZL=2, UR=UR, UC=UC,
                                    GROUP_M=group, num_warps=warps, num_stages=stages, **X.OPTS)
        return
    if split > 1:
        assert M * n <= work.ws.numel() and tiles <= work.cnt.numel(), (M, n, tiles)
        ws, cnt = work.ws, work.cnt
    else:
        ws, cnt = y, y
    DA.dithera_mm[(tiles, split)](xq, w, sa, sw, y, z, yl, y, S, ws, cnt, M, n, k, T, xq.stride(0), w.stride(0),
                                  y.stride(0), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, ZL=2, UR=UR, UC=UC,
                                  GROUP_M=group, num_warps=warps, num_stages=stages, **X.OPTS)


class Shape:
    def __init__(self, n: int, k: int, m: int, t: int):
        e = kgate.E()
        self.n, self.k, self.m, self.t = n, k, m, t
        self.copies = kbench.copies_for(n, k)
        self.xq, self.sa = e.quantize_rows(kgate.activations(m, k, 7 + m))
        self.ws = [kgate.weights(n, k, 100 + i) for i in range(self.copies)]
        self.ys = [torch.empty((m, n), dtype=torch.bfloat16, device="cuda") for _ in range(self.copies)]
        recs = (t // UR) * (n // UC) * (UR * UC * 3 // 2)
        self.zys = [torch.empty(recs, dtype=torch.int32, device="cuda") for _ in range(self.copies)]
        self.S = torch.tensor([0, m, 0, 0], dtype=torch.int32, device="cuda")
        self.served = X.pick(m, n, k)
        self.ref = []
        for i in (0, self.copies - 1):
            w, sw = self.ws[i]
            kgate.poison(self.ys[i], self.zys[i])
            gemm(self.served, self.xq, self.sa, w, sw, self.ys[i], self.zys[i], self.S, t, kgate.work())
            torch.cuda.synchronize()
            self.ref.append((self.ys[i].clone(), self.zys[i].clone()))
        w, sw = self.ws[0]
        self.served_y_diff = kgate.diff(self.ref[0][0], e.dequantize(e.int_mm(self.xq, w), self.sa, sw))

    def fn(self, cfg: X.Cfg):
        work = kgate.work()

        def f():
            for (w, sw), y, zy in zip(self.ws, self.ys, self.zys):
                gemm(cfg, self.xq, self.sa, w, sw, y, zy, self.S, self.t, work)
        return f

    def scaled_mm(self):
        work, cfg = kgate.work(), self.served

        def f():
            for (w, sw), y in zip(self.ws, self.ys):
                X.scaled_mm(self.xq, self.sa, w, sw, y, cfg, work)
        return f

    def check(self, i: int) -> int:
        j = 0 if i == 0 else self.copies - 1
        return kgate.diff(self.ys[j], self.ref[i][0]) + kgate.diff(self.zys[j], self.ref[i][1])

    def gate(self, cfg: X.Cfg) -> dict:
        work = kgate.work()
        w, sw = self.ws[0]
        rec: dict = {}
        for i in range(2):
            kgate.poison(self.ys[0], self.zys[0])
            gemm(cfg, self.xq, self.sa, w, sw, self.ys[0], self.zys[0], self.S, self.t, work)
            torch.cuda.synchronize()
            rec[f"diff_{i}"] = self.check(0)
        rec["workspace_nonzero"] = int((work.ws != 0).sum()) + int((work.cnt != 0).sum())
        rec["ok"] = rec["diff_0"] == 0 and rec["diff_1"] == 0 and rec["workspace_nonzero"] == 0
        return rec


def log(out: Path, rec: dict) -> None:
    with open(out / "sweep.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def stage(out: Path, shape: Shape, cfgs: list[X.Cfg], reps: int, clock, phase: str, extra: dict | None = None,
          around=None) -> dict:
    fns, recs = {}, {}
    for cfg in cfgs:
        rec = {"phase": phase, "n": shape.n, "k": shape.k, "m": shape.m, "t": shape.t, "cfg": str(cfg)}
        try:
            rec.update(shape.gate(cfg))
        except Exception as ex:  # noqa: BLE001 - a config that doesn't compile or launch is a record, not a crash
            rec.update({"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:200]}"})
            torch.cuda.synchronize()
        recs[str(cfg)] = rec
        if rec["ok"]:
            fns[str(cfg)] = shape.fn(cfg)
    fns.update(extra or {})
    med = {}
    if fns:
        med, _, _ = kbench.timed(fns, reps, clock, shape.copies, around)
        for name, us in med.items():
            if name in recs:
                kgate.poison(shape.ys[-1], shape.zys[-1])
                fns[name]()
                torch.cuda.synchronize()
                recs[name]["diff_after_timing"] = shape.check(1)
                recs[name]["ok"] = recs[name]["ok"] and recs[name]["diff_after_timing"] == 0
                recs[name]["us"] = us
    for rec in recs.values():
        log(out, rec)
    for name in extra or {}:
        log(out, {"phase": phase, "n": shape.n, "k": shape.k, "m": shape.m, "cfg": name, "us": med.get(name)})
    return {name: us for name, us in med.items() if name not in recs or recs[name]["ok"]}


def mmax_of(n: int, k: int, m: int) -> int | None:
    """The CONFIGS row (m_max) dither_a's $DITHER_MM_CFG key names for a call of m rows."""
    return next((mm for mm, _ in X.CONFIGS.get(X.arch(), {}).get((n, k), ()) if m <= mm), None)


def best(med: dict, top: int, skip=()) -> list[X.Cfg]:
    ranked = sorted(((c, u) for c, u in med.items() if c not in skip), key=lambda cu: cu[1])
    return [GCfg.parse(c) for c, _ in ranked[:top]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--t", type=int, default=64)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--block", default="", help="Q/D: time with the SMs whose %%smid %% D < Q held (`Blocker`)")
    ap.add_argument("--grid", choices=("decode", "prefill"), default="decode")
    ap.add_argument("--groups", default="1", help="G,..: the tile groups the best `top` are timed at after refine")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clock = kbench.nvml()
    blocker = None
    if a.block:
        q, d = (int(x) for x in a.block.split("/"))
        if not 0 < q < d:
            ap.error(f"--block {a.block}: Q/D with 0 < Q < D")
        blocker = Blocker(q, d)
    around = blocker.item if blocker else None
    groups = [int(g) for g in a.groups.split(",")]
    table, step = {}, {"served_us": 0.0, "best_us": 0.0, "scaled_mm_us": 0.0}
    for name in a.shapes.split(","):
        n, k = SHAPES[name]
        shape = Shape(n, k, a.m, a.t)
        if shape.served_y_diff:
            raise SystemExit(f"{name}: the served config's y differs from int8_exact's in {shape.served_y_diff} places")
        med = stage(out, shape, coarse_grid() if a.grid == "decode" else prefill_grid(), 3, clock, "coarse", around=around)
        refine = [X.Cfg(*c[:3], w, st, *c[5:]) for c in best(med, a.top) for w in WARPS for st in STAGES]
        med = stage(out, shape, uniq(refine), 3, clock, "refine", around=around)
        if groups != [1]:
            grouped = [GCfg(*c, g=g) for c in best(med, a.top) for g in groups]
            med = stage(out, shape, uniq(grouped), 3, clock, "group", around=around)
        final = uniq(best(med, a.top) + [shape.served])
        med = stage(out, shape, final, a.reps, clock, "final", {"scaled_mm": shape.scaled_mm()}, around)
        win = best(med, 1, skip=("scaled_mm",))[0]
        row = [a.m, str(win), med[str(win)], str(shape.served), med.get(str(shape.served)), med.get("scaled_mm")]
        table[f"{n},{k}"] = row
        step["best_us"] += LAYERS * row[2]
        step["served_us"] += LAYERS * (row[4] or 0.0)
        step["scaled_mm_us"] += LAYERS * (row[5] or 0.0)
        print(json.dumps({"layer": name, "n": n, "k": k, "row": row}), flush=True)
        del shape
        torch.cuda.empty_cache()
    mm_cfg = "+".join(f"{nk},{mmax_of(*map(int, nk.split(',')), a.m)}={row[1]}" for nk, row in table.items()
                      if row[1] != row[3])
    doc = {"device": torch.cuda.get_device_name(), "arch": X.arch(), "m": a.m, "t": a.t,
           "block": {"q_d": a.block, "held_sms": sorted(blocker.held), "smem": blocker.smem} if blocker else None,
           "table": table,
           "step_ms": {key: round(v / 1e3, 4) for key, v in step.items()}, "dither_mm_cfg": mm_cfg}
    (out / "best.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["step_ms"]), doc["dither_mm_cfg"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
