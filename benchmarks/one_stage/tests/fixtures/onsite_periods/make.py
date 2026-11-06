"""Writes this directory's stand-in onsite records from the onsite record's writer: its record of its own tests'
accepted run (``test_onsite_record.accepted``, toy inputs), each cut to the fields `onsite_periods.py` reads, their
values as the writer wrote them. The suite reads the JSON and never runs this; it runs by hand, against a tree that
holds the writer (``benchmarks/one_stage/onsite_record.py`` and its test), with that tree's packages on the path:

    PYTHONPATH=WRITER_TREE:... python benchmarks/one_stage/tests/fixtures/onsite_periods/make.py WRITER_TREE

``m1.json``: one execution phase, a short run, so every node cell is scaled. ``m2.json``: two phases, so K/m is
scaled. ``m1-week.json``: one phase whose run covers a week, so the week's node cell is proved.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELL = ("bits", "share", "delta_net", "vacuous", "how", "rule", "statement", "seconds", "phases")
KEEP = {"format": True, "run": True, "of_record": ("holds", "missing", "not_accepting"),
        "verdicts": {"the network certifier": ("verdict",)},
        "bound": {"K": True, "phases": True, "period_s": True, "notes": True, "statement": True,
                  "of_record": {"K": True, "open": True, "advice": ("bits_per_phase",)}},
        "advice_charge": {"cap": ("ell_a",), "charged": ("bits_per_phase", "share")}}


def cut(x, keep):
    if keep is True:
        return x
    if isinstance(keep, tuple):
        return {k: x[k] for k in keep if k in x}
    return {k: cut(x[k], v) for k, v in keep.items() if k in x}


def cell(c):
    return {k: c[k] for k in CELL if k in c} | ({"per_phase": cell(c["per_phase"])} if "per_phase" in c else {})


def grid(g):
    out = {"phase_s": g["phase_s"], "rules": g["rules"], "run": cell(g["run"])}
    out |= {p: {"seconds": g[p]["seconds"], "phases": g[p]["phases"],
                **{c: cell(g[p][c]) for c in ("node", "datacenter", "world")}} for p in ("week", "month", "year")}
    return out | ({"conditional_on": g["conditional_on"]} if "conditional_on" in g else {})


def stand_in(rec):
    out = {k: cut(rec[k], v) for k, v in KEEP.items()}
    return out | {k: grid(rec[k]) for k in ("periods", "periods_tight") if k in rec}


def main(tree: Path) -> None:
    sys.path[:0] = [str(tree / "benchmarks" / "one_stage" / "tests"), str(tree / "benchmarks" / "one_stage")]
    import test_onsite_record as T
    two = dict(T.PUBLIC, K=T.PUBLIC["K"] + 9088 + 63 + 1, eps="1/524288", windows=2,
               scans=[T.SCAN, dict(T.SCAN, faulty_sets=1)])
    cases = {"m1": {}, "m2": {"public": two, "windows": 2},
             "m1-week": {"schedule": dict(T.SCHEDULE, bucket_ms=10_000_000)}}
    for name, kw in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            rec = T.OR.record(T.accepted(Path(tmp), **kw))
        (HERE / f"{name}.json").write_text(json.dumps(stand_in(rec), indent=1, sort_keys=True, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
