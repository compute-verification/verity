"""The autotuned native arms (`native-bf16-tuned`, `native-fp8-tuned`): at each point, the fastest correct GEMM of the arm's
precision that the PoUW harness's baselines find (`benchmarks/pouw/harness/baselines.py`, through its `libpouw_harness.so`).

The candidates are every cuBLASLt heuristic candidate from each cuBLASLt the library loads (`native.LtLib`) and every
CUTLASS sm_120 kernel of the family that implements the shape, in each tile order its scheduler takes where the library
has them.  Each is gated by the harness's gate (bit-exact on ternary operands at the point's shape, its second call
checked, D's sentinel gone), and the gate's known-bad variants (an output byte flipped, cuBLASLt over half of k) must be
rejected, or the arm is refused at the point.  Each gated candidate is then timed as the chain's L GEMMs alone, on the
chain's own operands in one CUDA graph (as the arm's ``-gemm`` item is); the `FINALISTS` fastest and each library's
fastest are timed again in turn, `FINAL_ROUNDS` items each, and the fastest median is the pick (`Pick`), which
`chain.Tuned` runs as the chain.

The library shares this process's CUDA context (its runtime and torch's both take the device's primary context), and its
launches go on torch's current stream, so a CUDA graph captures them with the chain's activations between.  The FP8
chain's weights carry a per-tensor scale, which cuBLASLt reads through its scale pointers; the registry's plain FP8
kernels (CUTLASS's and nvf4_plain.cu's, kind 1) fix alpha at 1 and read none, so they are gated and timed but are not
the FP8 chain's pick (`ineligible`), and the record keeps the fastest of them.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HARNESS = Path(__file__).resolve().parents[1] / "pouw" / "harness"
#: each tuned arm's harness family (`baselines.FAMILIES`) and the torch arm it tunes
FAMILY = {"native-bf16-tuned": "bf16", "native-fp8-tuned": "fp8-e4m3"}
SCREEN_ITEMS, FINALISTS, FINAL_ROUNDS = 3, 3, 5
#: an item's seconds at least (its chains of L GEMMs replayed back to back)
ITEM_S = 0.01
SEED = 20261012


def harness() -> tuple[Any, Any, Any]:
    """The PoUW harness's `arm`, `baselines` and `native`, by its directory, after this one's (it has a chain.py too)."""
    if str(HARNESS) not in sys.path:
        sys.path.append(str(HARNESS))
    import arm
    import baselines
    import native
    return arm, baselines, native


def library(c: Any) -> str:
    """The library a candidate comes from: each cuBLASLt loaded on its own (``lt``, ``lt13``), else its source."""
    return c.config.get("library", c.source) if c.source == "cublaslt" else c.source


def ineligible(family: str, c: Any) -> str | None:
    """Why a gated candidate can't be the chain's GEMM, or None."""
    if family == "fp8-e4m3" and c.cutlass is not None:
        return "a kind-1 registry kernel fixes alpha at 1 and reads no scale; the FP8 chain's weights have one each"
    return None


def finalists(timed: list[dict], n: int = FINALISTS) -> list[dict]:
    """The `n` fastest eligible screened candidates and each library's fastest eligible one, fastest first."""
    ok = sorted((t for t in timed if t["ineligible"] is None and t.get("ms") is not None), key=lambda t: (t["ms"], t["name"]))
    best: dict[str, dict] = {}
    for t in ok:
        best.setdefault(t["library"], t)
    return [t for i, t in enumerate(ok) if i < n or best[t["library"]] is t]


@dataclass
class Pick:
    """A point's tuned GEMM: the candidate (its cuBLASLt plan kept alive while the pick is) at the harness's shape, and how
    it was chosen."""

    arm: str
    shape: Any
    cand: Any
    record: dict = field(default_factory=dict)


def buffer(t: Any) -> Any:
    A, _, _ = harness()
    return A.Buffer(t.data_ptr(), t.numel() * t.element_size())


class Tuner:
    """The harness's device (`native.Device`: device 0 of the process, its primary context) and every point's pick."""

    def __init__(self, lib: str | Path | None = None, *, uuid: str | None = None, want: int | None = None,
                 sched: bool = True, seed: int = SEED) -> None:
        if lib:
            os.environ["POUW_HARNESS_LIB"] = str(lib)
        self.A, self.B, self.N = harness()
        self.dev = self.N.Device()
        if uuid and self.dev.info["uuid"] != uuid:
            raise RuntimeError(f"the harness's device 0 is {self.dev.info['uuid']}, not the lease's {uuid}")
        self.sched = sched and self.dev.has_cutlass_sched
        self.kernels = self.B.cutlass_kernels(self.dev)
        self.lt_ws = self.dev.alloc(self.B.LT_WORKSPACE)
        self.want, self.seed = want or self.B.LT_CANDIDATES, seed
        self.picks: dict[tuple, Pick] = {}
        self.record = {"lib": str(self.N.lib_path()), "versions": self.N.versions(self.dev), "device": self.dev.info,
                       "cutlass_kernels": len(self.kernels), "cutlass_sched": self.sched, "lt_candidates": self.want,
                       "screen_items": SCREEN_ITEMS, "finalists": FINALISTS, "final_rounds": FINAL_ROUNDS}

    # ---- the gate -------------------------------------------------------------------------------------------------

    def gated(self, fam: Any, shape: Any) -> tuple[list, list, dict]:
        """Every candidate of `fam` at `shape`, each gated on the gate set, and the known-bad variants, which must fail.
        Returns the gated candidates, every cuBLASLt candidate made (whose plans `release` frees) and the record."""
        B, dev = self.B, self.dev
        cuts, unimpl = B.cutlass_candidates(dev, fam, shape, self.kernels, self.sched)
        sf = {fam.name: (max((c.sf_bytes[0] for c in cuts), default=0), max((c.sf_bytes[1] for c in cuts), default=0))}
        so = B.allocate(dev, shape, [fam], 0, self.seed, sf)
        lts: list = []
        rec: dict = {"cutlass_unimplementable": len(unimpl), "cublaslt_refused": []}
        try:
            gops = so.ops(fam, -1)
            for lt in dev.lts:
                lts += B.lt_candidates(dev, fam, shape, gops, self.want, refused=rec["cublaslt_refused"], lt=lt)
            passed, failed = [], []
            for c in lts + cuts:
                try:
                    call = B.BaselineCall(dev, c, shape, gops, self.lt_ws)
                except Exception as ex:  # noqa: BLE001 - recorded; not a candidate
                    failed.append({"name": c.name, "why": f"prepare: {type(ex).__name__}: {ex}"})
                    continue
                try:
                    r = B.gate(dev, fam, shape, call, B.hash_seed(f"{self.seed}/{shape.label}/{fam.name}/{c.name}"))
                except Exception as ex:  # noqa: BLE001
                    r = {"ok": False, "error": f"{type(ex).__name__}: {ex}"}
                finally:
                    call.free()
                if r["ok"]:
                    passed.append(c)
                else:
                    failed.append({"name": c.name, **{k: v for k, v in r.items() if k != "ok"}})
            rec.update(tried=len(lts) + len(cuts), gated=len(passed), failed=failed[:8], failed_count=len(failed),
                       controls=self.controls(fam, shape, gops, passed[0]) if passed else None)
        except Exception:
            self.release(lts, None)
            raise
        finally:
            so.free(dev)
            dev.sync()
        return passed, lts, rec

    def controls(self, fam: Any, shape: Any, gops: Any, c0: Any) -> dict:
        """The gate's known-bad variants: `c0` with an output byte flipped, and cuBLASLt over half of k.  Either accepted
        refuses the arm."""
        B, dev = self.B, self.dev
        call = B.BaselineCall(dev, c0, shape, gops, self.lt_ws)
        try:
            flipped = B.gate(dev, fam, shape, call, 1, corrupt=True)
        finally:
            call.free()
        out = {"flipped": {"candidate": c0.name, "rejected": not flipped["ok"]}, "half_k": None}
        half = B.half_k_call(dev, fam, shape, gops, self.lt_ws)
        if half is not None:
            try:
                out["half_k"] = {"candidate": half.cand.name, "rejected": not B.gate(dev, fam, shape, half, 2)["ok"]}
            finally:
                half.free()
                B.release_plans(dev, [half.cand])
        bad = [k for k, v in out.items() if v is not None and not v["rejected"]]
        if bad:
            raise RuntimeError(f"the harness's gate accepted its known-bad variants {bad} at {shape.label}: {out}")
        return out

    # ---- the time -------------------------------------------------------------------------------------------------

    def calls(self, pick_or_cand: Any, shape: Any, weights: list, scales: list | None, one: Any, xs: list, ys: list) -> list:
        """One prepared call per weight: x_i · W_iᵀ into y_i (with the per-tensor scales where the family has them)."""
        B = self.B
        cand = getattr(pick_or_cand, "cand", pick_or_cand)
        out: list = []
        try:
            for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
                ops = B.Operands(buffer(x), buffer(weights[i]), buffer(one) if scales else None,
                                 buffer(scales[i]) if scales else None, buffer(y))
                out.append(B.BaselineCall(self.dev, cand, shape, ops, self.lt_ws))
        except Exception:
            free(out)
            raise
        self.dev.sync()
        return out

    def graph(self, calls: list) -> Any:
        """A CUDA graph of the calls on torch's current stream, after one eager pass on a side stream."""
        import torch

        def body() -> None:
            s = torch.cuda.current_stream().cuda_stream
            for c in calls:
                c.launch(s)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            body()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        return g

    @staticmethod
    def item(g: Any, reps: int) -> float:
        """Seconds a replay: `reps` replays between two events on the current stream."""
        import torch
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) / 1e3 / reps

    def pick(self, arm: str, weights: list, scales: list | None, one: Any, x0: Any, ys: list) -> Pick:
        """The point's pick (once per arm and shape): `weights` (n × k, the arm's format), `scales` and `one` (the FP8
        arm's per-tensor scales), `x0` (m × k) and `ys` (each m × n BF16) the chain's own operands."""
        m, k = x0.shape
        n, length = weights[0].shape[0], len(weights)
        key = (arm, m, n, k, length)
        if key in self.picks:
            return self.picks[key]
        B = self.B
        fam = B.FAMILIES[FAMILY[arm]]
        shape = self.A.Shape(f"n{n}-k{k}-b{m}", m, k, n)
        t0 = time.monotonic()
        passed, lts, rec = self.gated(fam, shape)
        rec["gate_s"] = time.monotonic() - t0
        timed, kept = [], None
        try:
            for c in passed:
                t = {"name": c.name, "source": c.source, "library": library(c), "config": c.config,
                     "ineligible": ineligible(fam.name, c)}
                timed.append(t)
                calls = None
                try:
                    calls = self.calls(c, shape, weights, scales, one, [x0] * length, ys)
                    g = self.graph(calls)
                    reps = max(1, round(ITEM_S / max(self.item(g, 1), 1e-9)))
                    t["ms"] = statistics.median(self.item(g, reps) for _ in range(SCREEN_ITEMS)) * 1e3
                    t["reps"] = reps
                    del g
                except Exception as ex:  # noqa: BLE001 - recorded; not timed
                    t["error"] = f"{type(ex).__name__}: {ex}"
                finally:
                    free(calls or [])
            final = finalists(timed)
            if not final:
                raise RuntimeError(f"no eligible candidate of {fam.name} at {shape.label} passed its gate and timed")
            byname = {c.name: c for c in passed}
            graphs = {}
            held = []
            for t in final:
                calls = self.calls(byname[t["name"]], shape, weights, scales, one, [x0] * length, ys)
                held.append(calls)
                graphs[t["name"]] = (self.graph(calls), t["reps"])
            rounds: dict[str, list[float]] = {name: [] for name in graphs}
            names = list(graphs)
            for r in range(FINAL_ROUNDS):
                for name in names[r % len(names):] + names[:r % len(names)]:
                    g, reps = graphs[name]
                    rounds[name].append(self.item(g, reps) * 1e3)
            del graphs
            for calls in held:
                free(calls)
            medians = {name: statistics.median(v) for name, v in rounds.items()}
            best = min(medians, key=lambda x: (medians[x], x))
            kept = byname[best]
            fastest_out = min((t for t in timed if t["ineligible"] and t.get("ms") is not None), key=lambda t: t["ms"],
                              default=None)
            rec.update(screened=sorted(timed, key=lambda t: t.get("ms") or float("inf")),
                       finals={name: {"ms": v, "median_ms": medians[name]} for name, v in rounds.items()},
                       pick={"name": kept.name, "source": kept.source, "library": library(kept), "config": kept.config,
                             "ms": medians[best]},
                       fastest_ineligible=None if fastest_out is None else {k: fastest_out[k] for k in
                                                                            ("name", "library", "ms", "ineligible")},
                       tune_s=time.monotonic() - t0)
        finally:
            self.release(lts, kept)
        p = self.picks[key] = Pick(arm, shape, kept, rec)
        return p

    def release(self, lts: list, kept: Any) -> None:
        """Every cuBLASLt plan these candidates hold but the pick's."""
        plan = kept.lt[0].value if kept is not None and kept.lt is not None else None
        self.B.release_plans(self.dev, [c for c in lts if c.lt is not None and c.lt[0].value != plan])


def free(calls: list) -> None:
    for c in calls:
        c.free()
