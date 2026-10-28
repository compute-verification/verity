"""Where the H1 unit's flat lowering holds its memory, at one shape: each phase's peak RSS and wall time, a census of the
live objects by type, and (with ``--frames``) the allocation sites holding the bytes, at the two points the peak can be.

    python mem_profile.py --k K --tile R,C --scratch DIR --out DIR [--frames N]

It runs `pouw_h1_circuits._lower`'s build in this process, `fast_stage` installed as `run.py`'s child installs it, with
`PHASES` wrapped. A phase's peak is the process's VmHWM over the phase, reset at each phase's start (``/proc/self/clear_refs``),
so a phase's peak includes its sub-phases'. The census and the allocation sites are taken at the exit of `CENSUS`'s phases:
``ir_lower.lower_gates`` (the unit circuit and every IR gate's output word, ``wires``, alive) and
``fast_stage.ands_layout`` (the circuit and its layout alive, the words freed). `tracemalloc` (``--frames``) slows the build
and adds its own bytes to RSS, so run the phase table without it. Writes OUT/profile.json and OUT/rss.tsv (RSS each second,
with the innermost phase), and prints the tables as Markdown.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import threading
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from run import BIND

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VLLM = ROOT / "integrations" / "vllm"
H1_PROVE = ROOT / "benchmarks" / "pouw" / "h1_prove"
#: (phase, module attribute it wraps): what `UnitStager` runs for a flat unit, outermost first
PHASES = (("unit_lowering", "CS.unit_lowering"), ("ir_lower.lower", "IL.lower"), ("ir_lower.standalone", "IL.standalone"),
          ("ir_lower.lower_gates", "IL.lower_gates"), ("ir_lower.layout", "IL.layout"),
          ("fast_stage.ands_layout", "fast_stage.ands_layout"), ("Program", "CS.Program"), ("program_digest", "CS.program_digest"),
          ("circuit.compose", "FC.compose"), ("compose.netlist", "fast_stage._unit"), ("compose.dummy", "FC._dummy"),
          ("circuit.meta_of", "FC.meta_of"))
CENSUS = ("ir_lower.lower_gates", "fast_stage.ands_layout")
GIB = 1 << 30


def status(field: str) -> int:
    """``/proc/self/status``'s ``field`` in bytes."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) << 10
    raise KeyError(field)


def reset_hwm() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


class Phases:
    """A stack of the wrapped phases running now; each finished phase's RSS at entry and exit, peak and seconds."""

    def __init__(self, at_exit):
        self.stack, self.rows, self.at_exit, self.resets, self.entered = [], [], at_exit, True, 0

    def wrap(self, name: str, fn):
        def wrapped(*a, **k):
            self.enter(name)
            try:
                return fn(*a, **k)
            finally:
                self.exit()
        return wrapped

    def enter(self, name: str) -> None:
        if self.stack:
            self.stack[-1]["peak"] = max(self.stack[-1]["peak"], status("VmHWM"))
        self.resets &= reset_hwm()
        self.entered += 1
        self.stack.append({"phase": name, "order": self.entered, "depth": len(self.stack), "rss_in": status("VmRSS"), "peak": 0,
                           "t": time.monotonic()})

    def exit(self) -> None:
        f = self.stack[-1]
        f["peak"] = max(f["peak"], status("VmHWM"))
        f["rss_out"], f["seconds"] = status("VmRSS"), round(time.monotonic() - f.pop("t"), 2)
        if f["phase"] in self.at_exit:
            f["census"] = self.at_exit[f["phase"]](f["phase"])
        self.stack.pop()
        if self.stack:
            self.stack[-1]["peak"] = max(self.stack[-1]["peak"], f["peak"])
        self.rows.append(f)

    def now(self) -> str:
        return self.stack[-1]["phase"] if self.stack else "-"


def census(top: int = 20) -> list[dict]:
    """The live objects the collector tracks (containers and slotted objects, not ints, floats or strings), by type: count and
    `sys.getsizeof` bytes (each object's own, not what it refers to)."""
    n, size = Counter(), Counter()
    objs = gc.get_objects()
    for o in objs:
        t = type(o).__qualname__
        n[t] += 1
        size[t] += sys.getsizeof(o)
    del objs
    return [{"type": t, "count": n[t], "bytes": b} for t, b in size.most_common(top)]


def sites(snapshot, key: str, top: int) -> list[dict]:
    return [{"bytes": s.size, "count": s.count, "where": [f"{f.filename.replace(str(ROOT) + '/', '')}:{f.lineno}" for f in s.traceback]}
            for s in snapshot.statistics(key)[:top]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--tile", type=lambda v: tuple(int(x) for x in v.split(",")), required=True, metavar="R,C")
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=0, help="tracemalloc's frames a trace (0: no tracemalloc)")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    a.scratch.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(VLLM), str(H1_PROVE), str(H1_PROVE.parent / "hidden_zk")]
    from tests.protocols import pouw_h1_circuits as PC
    import fast_stage
    from verity_flock import circuit as FC
    from verity_flock import class_statement as CS
    from verity_flock import ir_lower as IL

    out: dict = {"k": a.k, "tile": list(a.tile), "frames": a.frames, "sites": {}}

    def at_exit(phase: str) -> dict:
        t = time.monotonic()
        doc = {"types": census(a.top)}
        if a.frames:
            snap = tracemalloc.take_snapshot().filter_traces([tracemalloc.Filter(False, tracemalloc.__file__)])
            out["sites"][phase] = {"traced": tracemalloc.get_traced_memory()[0], "traceback": sites(snap, "traceback", a.top),
                                   "lineno": sites(snap, "lineno", a.top)}
            del snap
        doc["seconds"] = round(time.monotonic() - t, 2)
        return doc

    phases = Phases({p: at_exit for p in CENSUS})
    fast_stage.install(a.scratch / ".scratch")
    mods = {"CS": CS, "IL": IL, "FC": FC, "fast_stage": fast_stage}
    for name, attr in PHASES:
        mod, fn = attr.split(".")
        setattr(mods[mod], fn, phases.wrap(name, getattr(mods[mod], fn)))
    write_text = Path.write_text
    big = phases.wrap("write_text", write_text)
    Path.write_text = lambda self, data, *x, **k: (big if isinstance(data, str) and len(data) > 1 << 20 else write_text)(
        self, data, *x, **k)

    stop = threading.Event()
    rss = open(a.out / "rss.tsv", "w")
    rss.write("seconds\trss_gib\tphase\n")
    start = time.monotonic()

    def sample():
        while not stop.wait(1.0):
            rss.write(f"{time.monotonic() - start:.0f}\t{status('VmRSS') / GIB:.3f}\t{phases.now()}\n")

    threading.Thread(target=sample, daemon=True).start()
    if a.frames:
        tracemalloc.start(a.frames)
    import h1unit as U
    path = a.scratch / f"k{a.k}.circuit.txt"
    st = phases.wrap("stager", lambda: U.Shape(a.k, *a.tile).stager(PC.NAME, circuit=path, bind=BIND))()
    stop.set()
    rss.close()
    counts = getattr(st.low.layout, "counts", {})
    out.update(ands=counts.get("ands"), rows=counts.get("rows"), nnz_a=counts.get("nnz_a"), nnz_b=counts.get("nnz_b"),
               bytes=fast_stage.size(fast_stage.composed()), peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss << 10,
               hwm_resets=phases.resets, phases=sorted(phases.rows, key=lambda f: f["order"]))
    path.unlink(missing_ok=True)
    (a.out / "profile.json").write_text(json.dumps(out, indent=1) + "\n")
    ands = out["ands"] or 1
    print(f"## K={a.k} tile={a.tile[0]}x{a.tile[1]}: {out['ands']:,} unit ANDs, peak {out['peak'] / GIB:.2f} GiB "
          f"({out['peak'] / ands:.0f} B an AND), file {out['bytes']:,} B\n")
    print("| phase | depth | peak GiB | B/AND at peak | RSS in -> out GiB | seconds |\n| --- | --- | --- | --- | --- | --- |")
    for f in out["phases"]:
        if f["seconds"] >= 0.5 or f["peak"] > (1 << 28) + f["rss_in"]:
            print(f"| {'  ' * f['depth']}{f['phase']} | {f['depth']} | {f['peak'] / GIB:.2f} | {f['peak'] / ands:.0f} | "
                  f"{f['rss_in'] / GIB:.2f} -> {f['rss_out'] / GIB:.2f} | {f['seconds']} |")
    for f in out["phases"]:
        if "census" in f:
            print(f"\n### live objects at {f['phase']}'s exit (census {f['census']['seconds']} s)\n\n| type | count | own bytes | "
                  "B/AND |\n| --- | --- | --- | --- |")
            for t in f["census"]["types"]:
                print(f"| {t['type']} | {t['count']:,} | {t['bytes']:,} | {t['bytes'] / ands:.0f} |")
    for phase, s in out["sites"].items():
        print(f"\n### allocation sites at {phase}'s exit (traced {s['traced'] / GIB:.2f} GiB)\n\n| bytes | blocks | B/AND | where |"
              "\n| --- | --- | --- | --- |")
        for x in s["lineno"]:
            print(f"| {x['bytes']:,} | {x['count']:,} | {x['bytes'] / ands:.0f} | {x['where'][0]} |")
        print("\n| bytes | blocks | traceback (innermost first) |\n| --- | --- | --- |")
        for x in s["traceback"]:
            print(f"| {x['bytes']:,} | {x['count']:,} | {' <- '.join(reversed(x['where']))} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
