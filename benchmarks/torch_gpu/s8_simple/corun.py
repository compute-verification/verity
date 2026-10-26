"""On one CUDA device: a decoder layer's W8A8 linears (`s8_simple.linear`, the fused epilogue) beside that layer's
serve-time commit (HASH-20's `sha512_rows`: every row's digest and hm96 leaf) on two CUDA streams, at the frozen H200
shape, as JSON to ``--out``/corun.json and a summary to ``--out``/summary.txt.

compute-accounting's step model adds the commit to the GEMM serially. The GEMM runs on the tensor cores and the row
kernel on the integer ALU, so the two might overlap. At each of ``--ms`` (token rows, and the commit's slots) this times,
with CUDA events, each phase's reps queued back to back (a rep starts when every stream of the last one has ended):

* ``linears``: q, kv, o, up and down (`check.SHAPES`, `check.LAYER`) on one stream, one x [M, 16384] for the first four
  and one [M, 65536] for down;
* ``commit``: `row_digests`, then `leaves`, over one layer's 25 rows (`bench.h200_rows`, 663,824 bytes a slot) on another;
* ``corun``: both from one start event at equal stream priority, then ``corun_commit_high`` and ``corun_linears_high``
  with that stream at the highest priority;
* ``linears`` and ``commit`` again (``*_again``), so a drift in the clock shows.

The overlap won is linears + commit - corun, each the median of its reps. Every co-run's words (each linear's bf16
output and each leaf) must be the alone run's, or the exit is 1. An NVML thread samples the leased GPU (by its UUID)
through every phase: the SM clock, power (the driver's average over the last second, and its instant reading where it
has one), the temperature and the clock's throttle reasons; a phase's statistics skip its first second. The codes are
uniform s8 and the commit's bytes uniform, so power reads high for served values.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROWS = HERE.parent / "sha512_rows"
sys.path[:0] = [str(HERE), str(ROWS)]

import bench as B  # noqa: E402
import engine as E  # noqa: E402
import s8_simple as S  # noqa: E402

from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: s8_simple's check (sha512_rows has a check.py too)
C = _module("s8_simple_check", HERE / "check.py")
MS = (16384, 4096)
PHASES = ("linears", "commit", "corun", "corun_commit_high", "corun_linears_high", "linears_again", "commit_again")
#: NVML's clock throttle reasons, by bit
THROTTLE = {0x1: "gpu_idle", 0x2: "applications_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
            0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks"}


class Telemetry:
    """NVML readings of one GPU every ``period`` seconds on a thread, each stamped with `time.monotonic`."""

    def __init__(self, period: float = 0.02) -> None:
        self.period, self.samples, self.error, self.instant = period, [], None, None
        self._stop = threading.Event()
        try:
            import pynvml as nv
            nv.nvmlInit()
            self.nv, self.h, self.how = nv, *self._handle(nv)
            self.instant = getattr(nv, "NVML_FI_DEV_POWER_INSTANT", None)
            self.reasons = (getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None)
                            or getattr(nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None))
            self.read()
        except Exception as e:  # noqa: BLE001
            self.nv, self.error = None, repr(e)
        self._thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _handle(nv: Any) -> tuple[Any, str]:
        uuid = os.environ.get("GPU_LEASE_UUID")
        for u in (uuid, uuid.encode()) if uuid else ():
            try:
                return nv.nvmlDeviceGetHandleByUUID(u), f"GPU_LEASE_UUID {uuid}"
            except Exception:  # noqa: BLE001, S112
                continue
        want = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            got = nv.nvmlDeviceGetUUID(h)
            if (got.decode() if isinstance(got, bytes) else got) == want:
                return h, f"torch's device {want}"
        return nv.nvmlDeviceGetHandleByIndex(0), "NVML index 0 (no UUID matched)"

    def read(self) -> dict[str, Any]:
        nv, h = self.nv, self.h
        r = {"t": time.monotonic(), "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
             "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1e3, "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)}
        if self.reasons is not None:
            r["throttle"] = int(self.reasons(h))
        if self.instant is not None:
            try:
                v = nv.nvmlDeviceGetFieldValues(h, [self.instant])[0]
            except Exception:  # noqa: BLE001
                self.instant = None
            else:
                if v.nvmlReturn == 0:
                    r["power_instant_w"] = v.value.uiVal / 1e3
        return r

    def _loop(self) -> None:
        while not self._stop.wait(self.period):
            try:
                self.samples.append(self.read())
            except Exception as e:  # noqa: BLE001
                self.error = repr(e)
                return

    def start(self) -> None:
        if self.nv is not None:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def stats(self, t0: float, t1: float, skip: float = 1.0) -> dict[str, Any]:
        """The samples in [t0 + skip, t1] (the later half if the phase is shorter than 2 skip)."""
        if self.nv is None:
            return {"missing": self.error}
        lo = t0 + min(skip, (t1 - t0) / 2)
        xs = [s for s in self.samples if lo <= s["t"] <= t1]
        if not xs:
            return {"samples": 0}
        out: dict[str, Any] = {"samples": len(xs), "seconds": round(t1 - lo, 3)}
        for k in ("sm_mhz", "power_w", "power_instant_w", "temp_c"):
            v = [s[k] for s in xs if k in s]
            if v:
                out[k] = {"median": statistics.median(v), "min": min(v), "max": max(v)}
        bits = 0
        for s in xs:
            bits |= s.get("throttle", 0)
        out["throttle"] = ([name for bit, name in THROTTLE.items() if bits & bit] or ["none"]) if self.reasons else ["unread"]
        return out


def linears(m: int, seed: int) -> list[tuple[str, tuple]]:
    """Each of a layer's linears with its operands: uniform s8 codes, typical scales (|N(0, 1)| / 127 a token, 0.02 of
    that a channel), x shared by the linears of one K."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def codes(*shape: int) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.int8, device="cuda").random_(-128, 128, generator=g)

    def scale(n: int, s: float) -> torch.Tensor:
        return torch.randn(n, device="cuda", generator=g).abs_().mul_(s / 127)

    xs: dict[int, tuple] = {}
    out = []
    for name in C.LAYER:
        n, k = C.SHAPES[name]
        if k not in xs:
            xs[k] = (codes(m, k), scale(m, 1.0))
        out.append((name, (*xs[k], codes(n, k), scale(n, 0.02))))
    return out


def commit(engine: E.Engine, slots: int, variant: str | None, minb: int):
    """One layer's commit as a call (its tensors and buffers kept alive by the closure), and its rows' sizes."""
    tensors, rows = B.layer(slots, 1)
    table = engine.rows(rows, tensors)
    G = len(rows)
    dids = [identity_digest_sha512("verity/sha512-rows/bench/unit/v1", {"unit": g}) for g in range(G)]
    lt = engine.leaf_table(dids, list(range(G)), [0] * G, None, list(range(G)))
    x = torch.empty((G, slots, 8), dtype=torch.int64, device="cuda")
    leaves = torch.empty_like(x)
    salts = E.SaltKey(B.KEY)

    def call(marks: list) -> list[torch.Tensor]:
        engine.row_digests(table, variant, x)
        marks.append(("rows", _mark()))
        engine.leaves(lt, x, salts, leaves, minb)
        marks.append(("leaves", _mark()))
        return [leaves]

    info = {"rows": G, "bytes_a_slot": sum(table.nbytes), "bytes": slots * sum(table.nbytes),
            "compressions": slots * sum(B.blocks(n) for n in table.nbytes), "leaves": G * slots}
    return call, info


def _mark() -> torch.cuda.Event:
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def layer_call(ops: list[tuple[str, tuple]]):
    def call(marks: list) -> list[torch.Tensor]:
        out = []
        for name, args in ops:
            out.append(S.linear(*args))
            marks.append((name, _mark()))
        return out
    return call


def phase(work: list[tuple[str, torch.cuda.Stream, Any]], reps: int, telemetry: Telemetry, sleep_cycles: int) -> dict[str, Any]:
    """``reps`` reps of ``work`` (name, stream, call) queued back to back; each rep's streams wait on one start event, and
    the next rep's start waits on every stream's end. A sleep on the current stream first holds the GPU while the host
    queues ahead. Returns each rep's ms, each stream's end and marks (ms after the start), the telemetry and the last
    rep's words."""
    main = torch.cuda.current_stream()
    torch.cuda.synchronize()
    t0 = time.monotonic()
    torch.cuda._sleep(sleep_cycles)
    reps_ev = []
    words: dict[str, list[torch.Tensor]] = {}
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        start.record(main)
        ends, marks = {}, {}
        for name, stream, call in work:
            stream.wait_event(start)
            with torch.cuda.stream(stream):
                marks[name] = []
                words[name] = call(marks[name])
                ends[name] = _mark()
            main.wait_event(ends[name])
        reps_ev.append((start, ends, marks))
    torch.cuda.synchronize()
    t1 = time.monotonic()
    rep_ms, stream_ms, mark_ms = [], {n: [] for n, _, _ in work}, {}
    for start, ends, marks in reps_ev:
        e = {n: start.elapsed_time(v) for n, v in ends.items()}
        rep_ms.append(max(e.values()))
        for n, v in e.items():
            stream_ms[n].append(v)
        for n, ms in marks.items():
            for label, ev in ms:
                mark_ms.setdefault(f"{n}.{label}", []).append(start.elapsed_time(ev))
    med = statistics.median
    return {"reps": reps, "ms": med(rep_ms), "ms_min": min(rep_ms), "ms_max": max(rep_ms),
            "streams_ms": {n: med(v) for n, v in stream_ms.items()}, "marks_ms": {k: med(v) for k, v in mark_ms.items()},
            "priorities": {n: s.priority for n, s, _ in work}, "seconds": round(t1 - t0, 3),
            "telemetry": telemetry.stats(t0, t1), "_words": words}


def _equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b))


def at(m: int, engine: E.Engine, a: argparse.Namespace, telemetry: Telemetry) -> dict[str, Any]:
    ops = linears(m, a.seed)
    lin = layer_call(ops)
    com, info = commit(engine, m, a.variant, a.leaf_minb)
    hi = -(1 << 20)  # out of range: CUDA maps it to the highest priority
    plain = {"linears": torch.cuda.Stream(), "commit": torch.cuda.Stream()}
    high = {"linears": torch.cuda.Stream(priority=hi), "commit": torch.cuda.Stream(priority=hi)}
    configs = {"linears": [("linears", plain["linears"], lin)], "commit": [("commit", plain["commit"], com)],
               "corun": [("commit", plain["commit"], com), ("linears", plain["linears"], lin)],
               "corun_commit_high": [("commit", high["commit"], com), ("linears", plain["linears"], lin)],
               "corun_linears_high": [("commit", plain["commit"], com), ("linears", high["linears"], lin)]}
    configs["linears_again"], configs["commit_again"] = configs["linears"], configs["commit"]
    est = {}
    for name in ("linears", "commit", "corun", "corun_commit_high", "corun_linears_high"):
        for _ in range(max(1, a.warm)):
            w = phase(configs[name], 1, telemetry, a.sleep_cycles)
        est[name] = w["ms"]
    res: dict[str, Any] = {"m": m, "commit": info, "linears": {n: list(C.SHAPES[n]) for n in C.LAYER},
                           "variant": a.variant or engine.default, "leaf_minb": a.leaf_minb, "phases": {}}
    ref: dict[str, list[torch.Tensor]] = {}
    for name in PHASES:
        reps = max(a.min_reps, min(a.max_reps, math.ceil(a.phase_s * 1e3 / est[name.removesuffix("_again")])))
        r = phase(configs[name], reps, telemetry, a.sleep_cycles)
        words = r.pop("_words")
        if name in ("linears", "commit"):
            ref[name] = [w.clone() for w in words[name]]  # the commit's leaves are one buffer every rep writes
        else:
            r["words_equal"] = {n: _equal(words[n], ref[n]) for n in words}
        res["phases"][name] = r
        print(json.dumps({"m": m, "phase": name, **{k: r[k] for k in ("reps", "ms", "streams_ms", "seconds")},
                          "telemetry": r["telemetry"]}), flush=True)
    p = res["phases"]
    alone_l, alone_c = p["linears"]["ms"], p["commit"]["ms"]
    res["alone_ms"] = {"linears": alone_l, "commit": alone_c}
    res["sum_ms"] = alone_l + alone_c
    res["drift"] = {"linears": p["linears_again"]["ms"] / alone_l - 1, "commit": p["commit_again"]["ms"] / alone_c - 1}
    res["corun"] = {}
    for name in ("corun", "corun_commit_high", "corun_linears_high"):
        r = p[name]
        won = res["sum_ms"] - r["ms"]
        res["corun"][name] = {"ms": r["ms"], "overlap_won_ms": won, "commit_hidden": won / alone_c,
                              "linears_end_ms": r["streams_ms"]["linears"], "commit_end_ms": r["streams_ms"]["commit"],
                              "linears_slowdown": r["streams_ms"]["linears"] / alone_l - 1,
                              "commit_slowdown": r["streams_ms"]["commit"] / alone_c - 1,
                              "words_equal": all(r["words_equal"].values())}
    return res


def _t(t: dict[str, Any]) -> str:
    if "sm_mhz" not in t:
        return f"telemetry {t}"
    s = f"SM {t['sm_mhz']['median']:.0f} MHz (min {t['sm_mhz']['min']:.0f}), {t['power_w']['median']:.0f} W"
    if "power_instant_w" in t:
        s += f" (instant {t['power_instant_w']['median']:.0f}, max {t['power_instant_w']['max']:.0f})"
    return s + f", {t['temp_c']['max']:.0f} C, throttle {'+'.join(t['throttle'])}"


def summary(doc: dict[str, Any]) -> str:
    lines = [f"{doc['env']['name']} lease {doc['env']['lease_uuid']} commit variant {doc['runs'][0]['variant'] if doc['runs'] else '-'}"]
    for r in doc["runs"]:
        p = r["phases"]
        lines.append(f"M = {r['m']}: linears alone {r['alone_ms']['linears']:.3f} ms, commit alone {r['alone_ms']['commit']:.3f} "
                     f"ms ({r['commit']['bytes'] / 1e9:.2f} GB), sum {r['sum_ms']:.3f} ms; drift linears "
                     f"{100 * r['drift']['linears']:+.2f}%, commit {100 * r['drift']['commit']:+.2f}%")
        for name in ("linears", "commit"):
            lines.append(f"  {name:20s} {p[name]['ms']:9.3f} ms x{p[name]['reps']:<4d} {_t(p[name]['telemetry'])}")
        for name, c in r["corun"].items():
            lines.append(f"  {name:20s} {c['ms']:9.3f} ms x{p[name]['reps']:<4d} won {c['overlap_won_ms']:7.3f} ms "
                         f"({100 * c['commit_hidden']:.1f}% of the commit); linears end {c['linears_end_ms']:.3f} "
                         f"({100 * c['linears_slowdown']:+.1f}%), commit end {c['commit_end_ms']:.3f} "
                         f"({100 * c['commit_slowdown']:+.1f}%); words equal {c['words_equal']}; {_t(p[name]['telemetry'])}")
    lines.append(f"words equal everywhere: {doc['words_equal']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ms", type=int, nargs="*", default=list(MS))
    ap.add_argument("--phase-s", type=float, default=3.0, help="each phase's seconds of reps (at least --min-reps)")
    ap.add_argument("--min-reps", type=int, default=10)
    ap.add_argument("--max-reps", type=int, default=1000)
    ap.add_argument("--warm", type=int, default=2, help="untimed reps of each configuration first")
    ap.add_argument("--sleep-cycles", type=int, default=40_000_000, help="the sleep that holds the GPU while a phase queues")
    ap.add_argument("--variant", default=None, help="the row kernel's variant (default: the build's, 0000)")
    ap.add_argument("--leaf-minb", type=int, choices=(0, 1, 2), default=0)
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    import triton
    props = torch.cuda.get_device_properties(0)
    engine = E.Engine()
    telemetry = Telemetry()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,power.limit", "--format=csv,noheader",
                          "-i", os.environ.get("GPU_LEASE_UUID", "0")], capture_output=True, text=True, check=False).stdout.strip()
    doc: dict[str, Any] = {"env": {"name": props.name, "capability": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
                                   "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
                                   "lease_uuid": os.environ.get("GPU_LEASE_UUID"), "lease_who": os.environ.get("GPU_LEASE_WHO"),
                                   "nvidia_smi": smi, "telemetry": telemetry.error or telemetry.how, "engine_lib": str(engine.lib.path),
                                   "priority_range": getattr(torch.cuda.Stream, "priority_range", lambda: None)()},
                           "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "runs": []}
    telemetry.start()
    t0 = time.monotonic()
    try:
        for m in a.ms:
            doc["runs"].append(at(m, engine, a, telemetry))
            torch.cuda.empty_cache()
    finally:
        telemetry.stop()
        doc["seconds"] = round(time.monotonic() - t0, 3)
        doc["telemetry_error"] = telemetry.error
        doc["words_equal"] = bool(doc["runs"]) and all(c["words_equal"] for r in doc["runs"] for c in r["corun"].values())
        (a.out / "corun.json").write_text(json.dumps(doc, indent=1) + "\n")
        if doc["runs"]:
            (a.out / "summary.txt").write_text(summary(doc))
    return 0 if doc["words_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
