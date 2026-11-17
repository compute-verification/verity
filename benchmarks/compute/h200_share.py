"""`benchmarks/compute/h200_share.py`: the number of the goal `accounting/h200-97-percent` ("An H200's capacity
credited at the data sheet's ceilings"), the share of one H200's capacity an honest Pearl+ prover is credited, at each
data-sheet reading of the card's tensor clock. Modeled: it multiplies two recorded runs' figures and the card file's
ceilings, and nothing ran the product.

    uv run python benchmarks/compute/h200_share.py --from RUN --duty RUN

The number, at each reading r of `READINGS` (the card's `clock.tensor` and `clock.sm`, the reading credit defaults to,
each the card file's value):

    value(r) = rate / ceiling(r) * share
    rate       an honest H200's e4m3 MACs a second: `benchmarks/tensor_duty`'s `--arms fp32-beside` arm A (the e4m3
               wgmma alone) on uniform operands at the driver's default clocks, its MACs over the host's interval
               around the kernel, read from that run's `out/beside.jsonl` (`--duty RUN`)
    ceiling(r) the card's peak e4m3 MACs a second at reading r (`verity.device.rate` on `hardware/cards/` `CARD`)
    share      `compute-accounting/verified-compute-share`: the MACs the Pearl+ verifier credits over the MACs the
               served model computes, on a workload of record (`run.py --from RUN`, which refuses an answer that
               isn't an accept)

Credit counts each MAC at its pipe's data-sheet ceiling, so a span of T seconds holds ceiling(r) * T MACs of capacity,
and a prover that computes every MAC of the served model at `rate` is credited share * rate * T of them. So the goal's
target is the honest rate's share of each reading's ceiling times the verified share, one value a reading, the higher
at the lower clock. What the model leaves out makes the number an upper bound on the served
share: Pearl+'s own work beside its matmuls (its commitments' hashing, the overhead `benchmarks/overhead/run.py table`
measures on the H200) takes time and earns nothing; attention, the norms and the rest run below the dense loop's rate;
and a served matmul's operands are not uniform bytes. Uniform operands are the goal's stated case: of the duty run's
operand classes they draw the most power, and the tensor pipe held its lowest clock on them.

It prints the number at each reading as text and one line of JSON; exits 0 with it, 1 with none (the record's answer is
not an accept, or the Lean verifier's verdict is not the same) and 2 when a run can't be read or holds no such point.
It prints and writes through the harness kit (`kit.report`): under `research run`, `h200_share.json`, the printed
document, in the run's `out/`, and no entry of `result.json`, as its figures are other runs' (`from_run`).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import run as compute
from verity.device import load_card, rate as peak
from verity_bench import kit

CARD_FILES = Path(__file__).resolve().parents[2] / "hardware/cards"

__all__ = ["BESIDE", "CARD", "METRIC", "READINGS", "UNIT", "document", "honest_rate", "main", "measured"]

#: the number's id in metrics.toml beside this file, the name result.json records it under
METRIC = "compute-accounting/h200-capacity-credited-share"
UNIT = "share of one H200's capacity, at the reading's data-sheet ceiling, credited to an honest Pearl+ prover"
CARD = "h200-sxm-141gb"
#: the readings of the card's tensor clock, the default last
READINGS = ("tensor", "sm")
#: the duty run's points, one JSON object a line
BESIDE = "out/beside.jsonl"
#: the honest point: arm A, the e4m3 wgmma alone, on uniform operands at the driver's default clocks
ARM, CLASS, CLOCK = "A", "uniform", "default"
FORMAT = compute.FORMAT


def honest_rate(points: list[dict]) -> Fraction:
    """The honest point's MACs a second over the host's interval around its kernel, the most of such points (as
    `tensor_duty/summarize.py` takes arm A's rate). `compute.Unreadable` where the run holds none."""
    honest = [p for p in points if (p.get("arm"), p.get("class"), p.get("clock", CLOCK)) == (ARM, CLASS, CLOCK)]
    if not honest:
        raise compute.Unreadable(f"the duty run holds no arm {ARM} point on {CLASS} operands at the {CLOCK} clocks: "
                                 f"run benchmarks/tensor_duty/run.sh --arms fp32-beside")
    rates = []
    for p in honest:
        span = Fraction(str(p["host_t1"])) - Fraction(str(p["host_t0"]))
        if span <= 0:
            raise compute.Unreadable(f"an arm {ARM} point's host interval is {float(span)} s")
        rates.append(Fraction(str(p["macs"])) / span)
    return max(rates)


def duty_points(store: compute.EvidenceStore, run: str) -> list[dict]:
    tree, _ = store.files(run, [BESIDE])
    if tree is None:
        raise compute.Unreadable(f"{run} holds no {BESIDE}: it is not a tensor_duty --arms fp32-beside run")
    return [json.loads(line) for line in (tree / BESIDE).read_text().splitlines() if line.startswith("{")]


def document(share_doc: dict, rate: Fraction, duty_run: str) -> dict:
    """The printed document: the number at each reading, from `run.py`'s document `share_doc` and the honest `rate`."""
    card = load_card(CARD, directory=CARD_FILES)
    share = Fraction(share_doc["credited_macs"], share_doc["model_macs"])
    readings = {}
    for r in READINGS:
        mhz = card["clock"][r]["value"]
        ceiling = peak(card, fmt=FORMAT, mhz=mhz)
        readings[f"clock.{r}"] = {"mhz": mhz, "ceiling_macs_per_s": float(ceiling), "utilization": float(rate / ceiling),
                                  "value": float(rate / ceiling * share)}
    return {"value": readings[f"clock.{READINGS[-1]}"]["value"], "unit": UNIT, "commit": share_doc["commit"],
            "run": share_doc["run"], "duty_run": duty_run, "card": CARD, "readings": readings,
            "honest_rate_macs_per_s": float(rate), "verified_compute_share": float(share), "mark": "modeled"}


def text(doc: dict) -> str:
    lines = [f"h200 share: an honest Pearl+ prover is credited {100 * doc['value']:.2f}% of an H200's capacity at "
             f"clock.{READINGS[-1]}'s ceiling (modeled)"]
    for name, r in doc["readings"].items():
        lines.append(f"  at {name} ({r['mhz']} MHz, {r['ceiling_macs_per_s'] * 1e-12:.1f} T e4m3 MACs/s): "
                     f"{100 * r['value']:.2f}%, the honest rate being {100 * r['utilization']:.2f}% of the ceiling")
    lines.append(f"  honest rate {doc['honest_rate_macs_per_s'] * 1e-12:.2f} T e4m3 MACs/s, arm {ARM} on {CLASS} operands "
                 f"at the {CLOCK} clocks, run {doc['duty_run']}")
    lines.append(f"  verified compute share {100 * doc['verified_compute_share']:.2f}% ({compute.METRIC}), run {doc['run']}, "
                 f"commit {doc['commit'] or 'unknown'}")
    lines.append("  an upper bound on the served share: Pearl+'s work beside its matmuls, the non-matmul work and served "
                 "operands are not in it")
    return "\n".join(lines)


def measured(doc: dict) -> list[dict]:
    """The result's entries, each the keywords of a `Writer.measurement`, which `kit.report` writes."""
    both = [doc["duty_run"], doc["run"]]
    return [*({"name": METRIC, "value": r["value"], "unit": UNIT, "inputs": both, "mark": "modeled", "member": name}
              for name, r in doc["readings"].items()),
            {"name": "honest rate", "value": doc["honest_rate_macs_per_s"], "unit": "e4m3 MAC/s",
             "inputs": [doc["duty_run"]], "mark": "measured"},
            {"name": "verified compute share", "value": doc["verified_compute_share"], "unit": compute.UNIT,
             "inputs": [doc["run"]], "mark": "measured"}]


def main(argv: list[str] | None = None, *,
         store_of: Callable[[], compute.EvidenceStore] = compute.opened) -> int:
    ap = argparse.ArgumentParser(prog="h200_share.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="run", metavar="RUN", required=True,
                    help="a workload of record: its verified compute share read from its record (run.py --from)")
    ap.add_argument("--duty", metavar="RUN", required=True,
                    help="a benchmarks/tensor_duty --arms fp32-beside run on an H200: its honest arm A rate")
    a = ap.parse_args(argv)
    try:
        store = store_of()
        share_doc = compute.from_run(store, a.run)
        doc = document(share_doc, honest_rate(duty_points(store, a.duty)), a.duty)
    except compute.NoNumber as e:
        print(f"h200 share: no number: {e}", file=sys.stderr)
        return 1
    except compute.Unreadable as e:
        print(f"h200 share: {e}", file=sys.stderr)
        return 2
    return kit.report("h200_share", text(doc), doc, measured(doc), from_run=a.run)


if __name__ == "__main__":
    sys.exit(main())
