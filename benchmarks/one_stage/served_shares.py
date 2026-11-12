"""A served run's two shares of the card's capacity, from its serve dir: the declared utilization, the work the served
model's steps declare at θ over the wall the engine took to serve them, and the accounted share, the work PoUW's period
credit certified at θ over that same wall.

    python benchmarks/one_stage/served_shares.py SERVE_DIR [--credit ANSWER.json] [--floor 9/64] [--model NAME]
        prints one JSON object

The wall is the trace's loads' busy seconds (`served_overhead.load_metrics`): each load's span, its first arrival to its
last step's output, plus the on-loop part of the closes of the Commit windows it opened. Each executed step of the trace
(``advice_trace.jsonl``) declares, at the served model's shapes (`MODELS`, its linears `served_verdicts.DECLARED`'s):

- the linears: every scheduled token through every layer's declared linears, at θ;
- attention: each row's queries against its causal keys (a row of n tokens from position p is n·p + n(n+1)/2
  query-key pairs a head and layer, QKᵀ and PV head_dim MACs each), at θ/2, the card's bf16 rate;
- the LM head: hidden × vocab MACs a token the step emitted, at θ/2.

θ is the H200's dense int8 rate in MACs at each reading of its clock (`RHO`, the tensor clock's and the boost
clock's), side by side. The declared utilization is the steps' ideal seconds over the wall.

The accounted share reads the period's credit (``--credit``: `pouw_piece.py period`'s ``answer.json``, whose ``public`` is
the Lean's ``Credit.PeriodCredit``, or that credit itself): its credited served work, m_s·n·K a credited call summed over
its windows, times the floor ``--floor`` (`FLOOR`), at θ, over the same wall. A rejected answer accounts 0, and so does a
run with no ``--credit``. The credited share (the floor at 1) is reported beside it. Neither is part of the declared
share, and the two are never added.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import served_overhead as O  # noqa: E402
import served_verdicts as SV  # noqa: E402

#: the H200's clock readings, the tensor clock's (the data sheet's dense int8 rate) and the boost clock's, and θ at
#: each: 132 SMs × 4,096 int8 MACs a clock
CLOCKS_MHZ = {"tensor": 1830, "boost": 1980}
RHO = {name: 4096 * 132 * mhz * 1e6 for name, mhz in CLOCKS_MHZ.items()}
#: the served models' attention and LM head shapes; their linears are `served_verdicts.DECLARED`'s
MODELS = {"verity-int8/Llama-3.2-1B-INT8": {"heads": 32, "head_dim": 64, "hidden": 2048, "vocab": 128256}}
#: J10-R's floor on the H200: the share of a credited call's work its b1 path forces
FLOOR = Fraction(9, 64)
SCHEMA = "verity-bench/served-shares/v0"


def step_macs(step: dict[str, Any], model: str) -> dict[str, int]:
    """One step record's declared MACs, ``{linears, attention, lm_head}`` (module docstring)."""
    layers, linears = SV.DECLARED[model]
    g = MODELS[model]
    if "rows" not in step:
        raise ValueError(f"step {step.get('step')} has no rows: a trace older than verity-vllm/step-trace/v1's rows")
    tokens = pairs = 0
    for rid, p, n in step["rows"]:
        if p is None:
            raise ValueError(f"step {step.get('step')}: row {rid} has no start position, so its attention is unknown")
        tokens += n
        pairs += n * p + n * (n + 1) // 2
    emitted = sum(int(v) for v in (step.get("emitted") or {}).values())
    return {"linears": tokens * layers * sum(n * k for _, n, k in linears),
            "attention": pairs * layers * g["heads"] * 2 * g["head_dim"], "lm_head": emitted * g["hidden"] * g["vocab"]}


def ideal_s(macs: dict[str, int], rho: float) -> float:
    return macs["linears"] / rho + (macs["attention"] + macs["lm_head"]) / (rho / 2)


def credited_work(doc: dict[str, Any] | None) -> tuple[int, str | None]:
    """The credit's served work summed over its windows, and the answer's verdict (None for a bare credit)."""
    if doc is None:
        return 0, None
    verdict = doc.get("verdict")
    if verdict is not None and verdict != "accept":
        return 0, verdict
    credit = doc["public"] if "verdict" in doc else doc
    work = (credit or {}).get("work")
    if not (isinstance(work, list) and all(type(w) is int and w >= 0 for w in work)):
        raise ValueError(f"the credit's work isn't a list of credited works: {work!r}")
    return sum(work), verdict


def model_of(serve_dir: Path) -> str:
    s = json.loads((serve_dir / O.SERVE).read_text())
    return ((s.get("engine") or {}).get("pin") or {}).get("model")


def shares(serve_dir: Path, credit: dict[str, Any] | None = None, floor: Fraction = FLOOR, model: str | None = None
           ) -> dict[str, Any]:
    """The serve dir's two shares at each reading of the clock (module docstring)."""
    model = model or model_of(serve_dir)
    if model not in MODELS or model not in SV.DECLARED:
        raise ValueError(f"no declared shapes for {model!r}: add them to MODELS and served_verdicts.DECLARED "
                         f"({sorted(MODELS)})")
    trace = O._jsonl(serve_dir / O.TRACE)
    if not trace or trace[0].get("kind") != "header":
        raise ValueError(f"{serve_dir}: no {O.TRACE} with a header")
    windows = {int(w["window"]): w for w in O._jsonl(serve_dir / O.WINDOWS)}
    loads, total, wall = [], {"linears": 0, "attention": 0, "lm_head": 0}, 0.0
    for ld in O.loads(trace):
        m = O.load_metrics(ld, windows)
        macs = {"linears": 0, "attention": 0, "lm_head": 0}
        for s in ld["steps"]:
            for k, v in step_macs(s, model).items():
                macs[k] += v
        busy = m["span_s"] + m["windows"]["close_on_loop_s"]
        loads.append({"steps": m["steps"], "tokens": m["tokens"], "wall_s": busy, "macs": macs,
                      "declared": {r: ideal_s(macs, rho) / busy for r, rho in RHO.items()}})
        total = {k: total[k] + macs[k] for k in total}
        wall += busy
    if not loads:
        raise ValueError(f"{serve_dir}: no load in its trace")
    work, verdict = credited_work(credit)
    readings = {}
    for r, rho in RHO.items():
        ideal = ideal_s(total, rho)
        readings[r] = {"clock_mhz": CLOCKS_MHZ[r], "theta_macs_s": rho, "ideal_s": ideal,
                       "declared": ideal / wall, "credited": work / rho / wall,
                       "accounted": float(floor * work) / rho / wall}
    return {"schema": SCHEMA, "dir": str(serve_dir), "model": model, "wall_s": wall, "macs": total, "loads": loads,
            "credit": {"verdict": verdict, "work": work, "floor": str(floor)}, "readings": readings}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("serve_dir", type=Path)
    ap.add_argument("--credit", type=Path, help="pouw_piece.py period's answer.json, or the period's credit")
    ap.add_argument("--floor", type=Fraction, default=FLOOR, help=f"the share of the credited work certified ({FLOOR})")
    ap.add_argument("--model", help="the served model (default: serve.json's engine pin)")
    a = ap.parse_args(argv)
    try:
        got = shares(a.serve_dir, json.loads(a.credit.read_text()) if a.credit else None, a.floor, a.model)
    except (OSError, ValueError) as e:
        print(f"served_shares.py: {e}", file=sys.stderr)
        return 2
    print(json.dumps(got))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
