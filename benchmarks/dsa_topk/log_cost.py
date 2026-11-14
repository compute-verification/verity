"""What keeping the counters (`dsa_topk.log_to`) costs a call of the indexer's top-k on the host: a wrapper's time a
call with the record kept and without, over stand-in ops that do nothing and calls that launch nothing. No device, no
torch and no vLLM, so no capture is on and no device is read; a call's time on a device is `stress.py time`'s.

    python benchmarks/dsa_topk/log_cost.py [--calls N] [--repeats R] [--dir DIR]
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections.abc import Callable
from types import SimpleNamespace

import dsa_topk


def arm(kept: str | None) -> Callable[[], None]:
    """A call of a wrapped op that does nothing, its record kept in ``kept`` or nowhere."""
    ns = SimpleNamespace(**{name: lambda *args: None for name in dsa_topk.OPS})
    canon = dsa_topk.install(ns, dsa_topk.Canon())
    if kept:
        dsa_topk.log_to(canon, kept)
    logits = SimpleNamespace(shape=(1, 16), stride=lambda i: (16, 1)[i])
    return lambda: ns.persistent_topk(logits, None, None, None, 16, 16)


def ns_a_call(call: Callable[[], None], calls: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(calls):
        call()
    return (time.perf_counter_ns() - start) / calls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calls", type=int, default=100_000, help="calls a repeat")
    ap.add_argument("--repeats", type=int, default=9, help="repeats of each arm, the two taking turns")
    ap.add_argument("--dir", help="the directory the record's own is made in (default: the system's temporary one)")
    args = ap.parse_args()
    dsa_topk._capturing = lambda: False
    with tempfile.TemporaryDirectory(dir=args.dir) as d:
        arms = {"plain": arm(None), "kept": arm(d)}
        times: dict[str, list[float]] = {name: [] for name in arms}
        for _ in range(args.repeats):
            for name, call in arms.items():
                times[name].append(ns_a_call(call, args.calls))
    ns = {name: {"least": round(min(t)), "median": round(statistics.median(t))} for name, t in times.items()}
    print(json.dumps({"calls": args.calls, "repeats": args.repeats, "ns_a_call": ns,
                      "write_ns_a_call": {k: ns["kept"][k] - ns["plain"][k] for k in ("least", "median")}}))


if __name__ == "__main__":
    main()
