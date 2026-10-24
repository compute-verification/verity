"""REDTEAM-ADVICE phase-18 target 3 (#3878): "at serve time" in the network-e2e report.

The lead's question: `at serve time = registered before the window's draw AND no later than the next window's close,
lag printed. Does the report enforce both?`

Driven through the real `chain.prove_path` with the suite's own endpoint fixture (reusing `test_chain`'s helpers).

CONFIRMED (probe A): a window is at_serve_time=True only when BOTH hold -- every rank's before_draw receipt is True AND
the registration time is no later than the next window's close; it is False when either fails with a definite signal;
and the lag after the window's own close is printed every time.

FINDING (probe B, LOW): the per-rank before_draw aggregation in `served_commit`,
    None if not order or None in order else all(x is True for x in order)
lets a MISSING receipt (None) override a DEFINITE after-the-draw receipt (False). A window one rank reports was
registered after ITS draw, while another rank's receipt is missing (order = [False, None]), is reported
at_serve_time=None ("unknown: no draw receipt orders it") instead of False ("after serving ... after its draw"). A
definite False is dispositive: a rank that drew before receiving the advice means the advice wasn't before that draw.
One-line fix in `served_commit`: `False if any(x is False for x in order) else (None if not order or None in order
else all(x is True for x in order))`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parent / "benchmarks" / "network_e2e" / "tests"


def _load_tc():
    spec = importlib.util.spec_from_file_location("test_chain_rt", TESTS / "test_chain.py")
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tc = _load_tc()
chain = tc.chain


def _scene(tmp: Path):
    ep = tc._endpoint(tmp)
    io = json.loads((ep / "io.json").read_text())["requests"]
    recs = chain._jsonl(ep / "advice_trace.jsonl")
    first, closed = chain._first_steps(recs), chain.closes(recs)
    info, b = chain.served_advice(ep, ep / "advice", {"origin_unix": first[0] + 1.2, "windows": 2, "window_s": 6.0},
                                  io, 60, [128001])
    bits = {x["window"]: x["v0_bits"] for x in b["commit_windows"]}
    return ep, info, b, bits, closed


def _registered(tmp, ep, info, b, bits, closed, name, w, at, *receipts):
    """A served arm registered at time `at` with per-rank before_draw `receipts` (rank 0 first). Returns
    (at_serve_time, after_close_s, rendered label)."""
    os.utime(ep / f"advice/w{w}/registration.json", (at,) * 2)
    tc._serve_arm(tmp / name, "served", w, bits[w], before_draw=receipts[0])
    s = json.loads((tmp / name / "served" / "summary.json").read_text())
    for i, r in enumerate(receipts[1:], 1):
        s["ranks"][str(i)] = s["ranks"]["0"] | {"advice_before_draw": {"ok": r}}
    (tmp / name / "served" / "summary.json").write_text(json.dumps(s))
    p = chain.prove_path([tmp / name], info, b["bound"], b, ep, {str(w): {"by": "vllm"}})
    g = p["arms"][f"w{w} served"]["commit"]["registered"]
    return g["at_serve_time"], g["after_close_s"], chain._registered(g)


def probe_A(tmp) -> None:
    ep, info, b, bits, closed = _scene(tmp / "A")
    assert round(closed[2] - closed[1], 1) == 0.7
    both = _registered(tmp / "A", ep, info, b, bits, closed, "in-time", 1, closed[1] + 0.5, True, True)
    after_next = _registered(tmp / "A", ep, info, b, bits, closed, "after-next", 1, closed[2] + 3.0, True)
    after_draw = _registered(tmp / "A", ep, info, b, bits, closed, "after-draw", 1, closed[1] + 0.5, True, False)
    print("A. both conditions required for True, lag always printed:")
    for tag, (ast, lag, label) in (("in-time (before draw & before next close)", both),
                                   ("after next close (2nd fails)", after_next),
                                   ("after the draw (1st fails)", after_draw)):
        print(f"   {tag:42} -> at_serve_time={ast!s:5} lag={lag:+} s | {label}")
    assert both[0] is True and after_next[0] is False and after_draw[0] is False
    assert both[1] == 0.5 and after_next[1] == 3.7 and after_draw[1] == 0.5  # lag printed in every case


def probe_B(tmp) -> None:
    ep, info, b, bits, closed = _scene(tmp / "B")
    masked = _registered(tmp / "B", ep, info, b, bits, closed, "false-none", 1, closed[1] + 0.5, False, None)
    both_false = _registered(tmp / "B", ep, info, b, bits, closed, "true-false", 1, closed[1] + 0.5, True, False)
    print("B. a definite after-the-draw receipt masked by a missing one:")
    print("   ranks [False, None] (one rank drew first, one receipt missing):")
    print(f"      at_serve_time={masked[0]!s:5} | {masked[2]}")
    print("   ranks [True, False] (one rank drew first, the other confirmed):")
    print(f"      at_serve_time={both_false[0]!s:5} | {both_false[2]}")
    print("   -> a definite False is dispositive only when no receipt is missing; [False, None] should be False too")
    assert masked[0] is None and both_false[0] is False


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "A").mkdir()
        (tmp / "B").mkdir()
        probe_A(tmp)
        probe_B(tmp)
    print("\nALL CHECKS PASSED (A: both enforced + lag printed; B: [False, None] masks a definite after-the-draw)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
