"""advice-charge: honest serving's advice charge, measured on a test run whose inputs are public (``advice-charge.json``,
`FORMAT`): each Commit period's advice length |a_j| as `verity-vllm advice` writes it, unpadded, beside the prompt and
output lengths and finishes of the requests it admits, so the mix it was measured at is on the record. A Commit period
is WINDOW_STEPS engine steps, one in the run of record, so an execution phase of T buckets of τ holds several, and
ℓ_{a,i}, the width the network certifier's bound charges execution phase i, is set against their sum, which this record
doesn't hold. A served run's advice length is private (a value), so only a test run with public inputs is measured, and
its record says so (``label``): a figure of that test run, never of a served run.

    python benchmarks/one_stage/advice_charge.py ADVICE --run RUN [--tau S] --to FILE

ADVICE is `verity-vllm advice`'s output over every closed Commit period of a serve-openai run (its ``windows.json`` and
each ``w<k>/``). Each of ``commit_periods`` is one Commit period's: ``bits``, its registration's binding's length, and
``by_egress_window``, the bits I1 charges each egress window of ``T`` buckets of τ (``--tau``, the advice's) that holds
any, counted from the Commit period's first step, with the requests it admits and carries in and its engine steps.
``bits`` summarizes the Commit periods, and ``prompt_len``, ``output_len`` and ``finishes`` (``eos``: a stop token
ended it; ``length``: its cap did; ``abort``) the requests, each once, in the Commit period that admits it. ``serving`` is the engine's configuration the step trace
records (`verity-vllm advice`'s report), the settings it was measured at. ``refused`` and ``open`` are the Commit
periods the advice refused or left open, which no figure counts. `served_verdicts.sh advice-charge` serves such a run
and writes it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

FORMAT = "verity/advice-charge/v1"
INDEX = "verity-vllm/advice-windows-index/v1"
LABEL = ("measured on test run {run}: honest serving of public inputs on the plain path, its advice unpadded; a figure "
         "of that test run, never of a served run")
FINISHES = ("eos", "length", "abort")


def stats(xs: Sequence[int]) -> dict[str, Any]:
    return {"count": len(xs), "min": min(xs), "median": statistics.median(xs), "mean": statistics.fmean(xs),
            "max": max(xs), "total": sum(xs)}


def charge(advice: Path, run: str, tau: float) -> dict[str, Any]:
    """The test run's record (module docstring); ValueError where ADVICE holds no closed Commit period's advice."""
    index = json.loads((advice / "windows.json").read_text())
    if index.get("schema") != INDEX:
        raise ValueError(f"{advice / 'windows.json'} is {index.get('schema')!r}, not {INDEX} (`verity-vllm advice` over "
                         "a serve-openai run)")
    periods: list[dict[str, Any]] = []
    prompts: list[int] = []
    outputs: list[int] = []
    finishes: Counter[str] = Counter({f: 0 for f in FINISHES})
    compilers = set()
    serving = None
    for k, w in sorted(index["windows"].items(), key=lambda kw: int(kw[0])):
        reg = json.loads((advice / w["registration"]).read_text())
        adv = json.loads((advice / f"w{k}" / "private" / "advice.json").read_text())
        report = advice / f"w{k}" / "private" / "report.json"
        if serving is None and report.exists():
            serving = json.loads(report.read_text()).get("config")
        bits = int(reg["binding"]["advice"]["bits"])
        if bits != int(w["bits"]):
            raise ValueError(f"Commit period {k}'s registration binds {bits} bits, and windows.json {w['bits']}")
        compilers.add(json.dumps(reg["binding"]["compiler"], sort_keys=True))
        admitted = adv["requests"]
        periods.append({"period": int(k), "bits": bits, "by_egress_window": reg["bits_by_window"],
                       "admitted": len(admitted), "carried_in": len(adv.get("carried") or ()),
                       "engine_steps": len(adv["steps"])})
        prompts += [int(q["prompt_len"]) for q in admitted]
        outputs += [int(q["output_len"]) for q in admitted]
        finishes.update(str(q["finish"]) for q in admitted)
    if not periods:
        raise ValueError(f"{advice}: no closed Commit period's advice (refused {index.get('refused')}, open "
                         f"{index.get('open')}): nothing measured")
    if len(compilers) != 1:
        raise ValueError(f"{advice}: its Commit periods' advice is bound to {len(compilers)} compilers, not one")
    return {"format": FORMAT, "label": LABEL.format(run=run), "run": run, "T": int(index["T"]), "tau": tau,
            "compiler": json.loads(compilers.pop()), "serving": serving, "commit_periods": periods,
            "bits": stats([p["bits"] for p in periods]), "prompt_len": stats(prompts), "output_len": stats(outputs),
            "finishes": dict(finishes), "refused": index.get("refused") or {}, "open": index.get("open") or []}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("advice", type=Path, help="`verity-vllm advice`'s --out over a test run's every closed Commit period")
    ap.add_argument("--run", required=True, help="the test run's id, which the label names")
    ap.add_argument("--tau", type=float, default=0.1, help="the egress bucket's seconds the advice was written at")
    ap.add_argument("--to", type=Path, required=True, help="the record's path")
    a = ap.parse_args(argv)
    try:
        rec = charge(a.advice, a.run, a.tau)
    except (OSError, ValueError, KeyError) as e:
        print(f"ADVICE-CHARGE: not written: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    a.to.parent.mkdir(parents=True, exist_ok=True)
    a.to.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print("ADVICE-CHARGE " + json.dumps({k: rec[k] for k in ("run", "bits", "finishes", "prompt_len", "output_len")},
                                        sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
