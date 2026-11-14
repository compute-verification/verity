"""What the unread gate (`tools/verity/lean/unread_gate.py`, drive's landing gate) costs on one change, and its guard.

    research run --on auto --store-read --cwd source -- \
      bash -c 'python3 benchmarks/lean_unread_gate/run.py MAIN HEAD --out "$OUT" [--against REV]'

It judges ``HEAD`` against ``MAIN`` as a ready mark's preflight runs the gate (`research.queue.run_gate`), in this process,
with the time of each stage (`STAGES`, each inclusive of the stages it calls) and the count of git calls by subcommand. Each
gate starts from an empty cache (its `cache_dir` a temporary directory), as on an agent VM that never ran it. With
``--against REV`` it judges the same change again with the gate as it is at ``REV`` and compares: the guard holds when both
exit the same and print the same `--json`, or, when neither can judge, say the same why. It writes `unread-gate.json`
under ``--out`` and exits 1 when the guard fails.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = "tools/verity/lean/unread_gate.py"
#: the gate's functions timed, each by the name it is reported under
STAGES = {"removed": "seeds", "declared": "declared", "plausible": "plausible", "unnamed": "unnamed", "judge": "judge",
          "facts_of": "facts", "stored": "store", "ancestor_facts": "ancestors", "archive": "archive"}


class Meter:
    """Inclusive seconds and calls by stage, and git calls by subcommand, of one gate's run."""

    def __init__(self):
        self.seconds, self.calls, self.git = defaultdict(float), Counter(), Counter()

    def timed(self, name: str, f):
        def wrapped(*a, **k):
            t = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                self.seconds[name] += time.perf_counter() - t
                self.calls[name] += 1
        return wrapped

    def count(self, argv) -> None:
        if isinstance(argv, (list, tuple)) and argv and Path(str(argv[0])).name == "git":
            args = [str(a) for a in argv[1:]]
            while args and args[0].startswith("-"):
                args = args[2:] if args[0] in ("-C", "-c") else args[1:]
            self.git[args[0] if args else "?"] += 1


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def instrument(gate, meter: Meter, cache: Path) -> None:
    """``gate`` with its stages timed, and its cache and that of the Lean check it reads kept passes from (`common.cache_dir`)
    under ``cache``."""
    for fn, name in STAGES.items():
        if hasattr(gate, fn):
            setattr(gate, fn, meter.timed(name, getattr(gate, fn)))
    loader = gate.load

    def load_timed(path, name):
        m = loader(path, name)
        for common in (m, getattr(m, "COMMON", None)):
            if hasattr(common, "cache_dir"):
                common.cache_dir = lambda: cache
        if hasattr(m, "analyse"):
            analyse, n = m.analyse, [0]

            def each(*a, **k):
                n[0] += 1
                return meter.timed(("ratchet LO", "ratchet HEAD")[min(n[0], 2) - 1], analyse)(*a, **k)
            m.analyse = each
        return m
    gate.load = load_timed


def judge(script: Path, main: str, head: str, label: str) -> dict:
    """``script``'s judgement of ``head`` against ``main``: its exit, its `--json` (or what it printed when it couldn't
    judge), and what it cost."""
    meter = Meter()
    run, popen = subprocess.run, subprocess.Popen

    def run_counted(argv, *a, **k):
        meter.count(argv)
        return run(argv, *a, **k)

    class Popen(popen):
        def __init__(self, argv, *a, **k):
            meter.count(argv)
            super().__init__(argv, *a, **k)

    gate = load(script, f"unread_gate_{label}")
    out, err = io.StringIO(), io.StringIO()
    with tempfile.TemporaryDirectory(prefix="unread-gate-bench-") as cache:
        instrument(gate, meter, Path(cache))
        subprocess.run, subprocess.Popen = run_counted, Popen
        t = time.perf_counter()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = gate.main([main, head, "--json"])
        finally:
            took = time.perf_counter() - t
            subprocess.run, subprocess.Popen = run, popen
    doc = json.loads(out.getvalue()) if rc in (0, 1) else None
    return {"script": str(script), "exit": rc, "doc": doc, "said": err.getvalue().strip()[-2000:],
            "seconds": round(took, 3), "stages": {k: {"seconds": round(v, 3), "calls": meter.calls[k]}
                                                  for k, v in sorted(meter.seconds.items())},
            "git_calls": sum(meter.git.values()), "git": dict(meter.git.most_common())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("main")
    ap.add_argument("head")
    ap.add_argument("--against", metavar="REV", help="also judge with the gate as it is at REV, and compare (the guard)")
    ap.add_argument("--gate", type=Path, default=ROOT / GATE, help="the gate to time (default: this tree's)")
    ap.add_argument("--out", type=Path, required=True, help="where it writes unread-gate.json (`$OUT` under `research run`)")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    rows = {"gate": judge(a.gate, a.main, a.head, "here")}
    if a.against:
        with tempfile.TemporaryDirectory(prefix="unread-gate-against-") as d:
            old = Path(d) / Path(GATE).name
            old.write_bytes(subprocess.run(["git", "show", f"{a.against}:{GATE}"], check=True, capture_output=True).stdout)
            rows["against"] = judge(old, a.main, a.head, "against") | {"rev": a.against}
    result = {"main": a.main, "head": a.head, **rows}
    if a.against:
        result["same"] = all(rows["gate"][k] == rows["against"][k] for k in ("exit", "doc", "said"))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "unread-gate.json").write_text(json.dumps(result, indent=1) + "\n")
    for k, r in rows.items():
        stages = ", ".join(f"{s} {v['seconds']:.1f} s" for s, v in r["stages"].items())
        print(f"{k}: exit {r['exit']}, {r['seconds']:.1f} s, {r['git_calls']} git calls ({stages})")
    if a.against:
        print("guard: " + ("the same verdict and --json" if result["same"] else "DIFFERENT verdicts"))
    return 0 if result.get("same", True) else 1


if __name__ == "__main__":
    sys.exit(main())
