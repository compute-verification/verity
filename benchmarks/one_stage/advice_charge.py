"""advice-charge: honest serving's advice charge, measured on a test run whose inputs are public (``advice-charge.json``,
`FORMAT`): each Commit period's advice length |a_j|, unpadded, beside the prompt and output lengths and finishes of the
requests it admits, so the mix it was measured at is on the record. A Commit period is WINDOW_STEPS engine steps, one in
the run of record, so an execution phase of T buckets of τ holds several, and their sum over each execution phase
(``by_phase``) is the advice an execution phase carries, which the onsite record sets beside the network certifier's
bound (`onsite_record.py`). A served run's advice length is
private (a value), so only a test run with public inputs is measured, and its record says so (``label``): a figure of
that test run, never of a served run.

    python benchmarks/one_stage/advice_charge.py ADVICE --run RUN [--tau S] [--T BUCKETS] --to FILE

ADVICE is a serve-openai run's every closed Commit period's advice, as `verity-vllm advice` writes it (its
``windows.json`` and each ``w<k>/``), or as the server committed it while serving (``--advice``, the same layout). Each
of ``commit_periods`` is one Commit period's: ``bits``, |a_j|, its advice's encoding (`window_advice.total_bits`), and
``registered_bits``, what its registration binds (ℓ_a once the advice is padded before it is committed; |a_j| before),
with the requests it admits and carries in and its engine steps. ``bits`` summarizes the Commit periods, and
``prompt_len``, ``output_len`` and ``finishes`` (``eos``: a stop token ended it; ``length``: its cap did; ``abort``) the
requests, each once, in the Commit period that admits it: a Commit period's advice lists the requests it carries in
first, then those it admits. Where each Commit period's private report records its first step's engine clock reading
(``t0``), ``start_s`` is it, counted from the first Commit period's, and ``by_phase`` sums the bits of the Commit
periods that start in each execution phase of T buckets of τ (``--T``, ``--tau``) on that clock (the test run's
phases, not the network certifier's phase boundaries: a Commit period near a boundary can count in the next), with
``phase_bits`` summarizing those phases; otherwise ``by_phase`` says why it is missing. ``serving`` is the engine's configuration the
step trace records (the report's), the settings it was measured at. A report that places outputs at another τ is
refused. ``refused`` and ``open`` are the Commit periods the advice refused or left open, which no figure counts.
`served_verdicts.sh advice-charge` serves such a run and writes it.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verity_vllm.pipeline import window_advice as WA

FORMAT = "verity/advice-charge/v2"
INDEX = "verity-vllm/advice-windows-index/v1"
LABEL = ("measured on test run {run}: honest serving of public inputs on the plain path, its advice unpadded; a figure "
         "of that test run, never of a served run")
FINISHES = ("eos", "length", "abort")
NO_CLOCK = ("a Commit period's private report records no first-step clock reading (t0), so no Commit period is placed "
            "in an execution phase")


def stats(xs: Sequence[int]) -> dict[str, Any]:
    return {"count": len(xs), "min": min(xs), "median": statistics.median(xs), "mean": statistics.fmean(xs),
            "max": max(xs), "total": sum(xs)}


def charge(advice: Path, run: str, tau: float, T: int | None = None) -> dict[str, Any]:
    """The test run's record (module docstring); ValueError where ADVICE holds no closed Commit period's advice."""
    index = json.loads((advice / "windows.json").read_text())
    if index.get("schema") != INDEX:
        raise ValueError(f"{advice / 'windows.json'} is {index.get('schema')!r}, not {INDEX} (`verity-vllm advice` over "
                         "a serve-openai run, or its server's own)")
    T = int(index["T"]) if T is None else T
    periods: list[dict[str, Any]] = []
    starts: list[float | None] = []
    prompts: list[int] = []
    outputs: list[int] = []
    finishes: Counter[str] = Counter({f: 0 for f in FINISHES})
    compilers = set()
    serving = None
    for k, w in sorted(index["windows"].items(), key=lambda kw: int(kw[0])):
        reg = json.loads((advice / w["registration"]).read_text())
        adv = WA.from_json(json.loads((advice / f"w{k}" / "private" / "advice.json").read_text()))
        path = advice / f"w{k}" / "private" / "report.json"
        report = json.loads(path.read_text()) if path.exists() else {}
        registered = int(reg["binding"]["advice"]["bits"])
        if registered != int(w["bits"]):
            raise ValueError(f"Commit period {k}'s registration binds {registered} bits, and windows.json {w['bits']}")
        bits = WA.total_bits(adv)
        if bits > registered:
            raise ValueError(f"Commit period {k}'s advice is {bits} bits, over the {registered} its registration binds")
        if report and report["registration"]["binding"] != reg["binding"]:
            raise ValueError(f"Commit period {k}'s private report is of another registration than {w['registration']}")
        release = report.get("release") or {}
        if release.get("rule") == "engine clock" and float(release["tau_s"]) != tau:
            raise ValueError(f"Commit period {k}'s advice places its outputs at τ = {release['tau_s']} s, not {tau}: "
                             f"commit it at {tau} (serve-openai --advice-tau, or `verity-vllm advice --tau`)")
        if serving is None:
            serving = report.get("config")
        compilers.add(json.dumps(reg["binding"]["compiler"], sort_keys=True))
        carried = len(adv.carried)
        admitted = adv.requests[carried:]
        periods.append({"period": int(k), "bits": bits, "registered_bits": registered, "admitted": len(admitted),
                        "carried_in": carried, "engine_steps": len(adv.steps)})
        starts.append(release.get("t0"))
        prompts += [q.prompt_len for q in admitted]
        outputs += [q.output_len for q in admitted]
        finishes.update(q.finish for q in admitted)
    if not periods:
        raise ValueError(f"{advice}: no closed Commit period's advice (refused {index.get('refused')}, open "
                         f"{index.get('open')}): nothing measured")
    if len(compilers) != 1:
        raise ValueError(f"{advice}: its Commit periods' advice is bound to {len(compilers)} compilers, not one")
    by_phase: list[dict[str, Any]] | dict[str, str] = {"missing": NO_CLOCK}
    phase_bits = None
    if None not in starts:
        t00 = min(float(t) for t in starts)
        sums: dict[int, list[int]] = {}
        for p, t in zip(periods, starts):
            p["start_s"] = float(t) - t00
            sums.setdefault(math.floor(p["start_s"] / (T * tau)), []).append(p["bits"])
        by_phase = [{"phase": j, "bits": sum(b), "commit_periods": len(b)} for j, b in sorted(sums.items())]
        phase_bits = stats([x["bits"] for x in by_phase])
    return {"format": FORMAT, "label": LABEL.format(run=run), "run": run, "T": T, "tau": tau,
            "compiler": json.loads(compilers.pop()), "serving": serving, "commit_periods": periods,
            "bits": stats([p["bits"] for p in periods]), "by_phase": by_phase, "phase_bits": phase_bits,
            "prompt_len": stats(prompts), "output_len": stats(outputs), "finishes": dict(finishes),
            "refused": index.get("refused") or {}, "open": index.get("open") or []}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("advice", type=Path, help="a test run's every closed Commit period's advice (`verity-vllm advice`'s "
                                              "--out, or the server's own advice/)")
    ap.add_argument("--run", required=True, help="the test run's id, which the label names")
    ap.add_argument("--tau", type=float, default=0.1, help="the egress bucket's seconds the advice was written at")
    ap.add_argument("--T", type=int, default=None, help="an execution phase's buckets (default: the advice index's T)")
    ap.add_argument("--to", type=Path, required=True, help="the record's path")
    a = ap.parse_args(argv)
    try:
        rec = charge(a.advice, a.run, a.tau, a.T)
    except (OSError, ValueError, KeyError) as e:
        print(f"ADVICE-CHARGE: not written: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    a.to.parent.mkdir(parents=True, exist_ok=True)
    a.to.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print("ADVICE-CHARGE " + json.dumps({k: rec[k] for k in ("run", "bits", "phase_bits", "finishes", "prompt_len",
                                                              "output_len")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
