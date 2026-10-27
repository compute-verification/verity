"""Results per monitoring period, over all compute (`ARCHITECTURE.md`'s Glossary, "Monitoring period"), from what
onsite's own executable evaluates (``onsite bound``, `Core.Onsite.Bound` and `Core.Onsite.Pooled`, the
composition README's "The joint bound") over the run's blocks, and over a week, a month and a year of every machine of a
census deployment.

A monitoring period of a deployment (`benchmarks/census/deployments.json`: its machines and the length of each machine's
server-time block, ``window_s``, stated parameters) is ``B = machines × ⌊period / window_s⌋`` server-time blocks, each
one onsite decides.
For each period ``onsite bound`` gives:

* the joint bound, the right-hand side of ``Accounting.Guarantees.JointHorizonBlocks``: each accepted block's
  subprotocols' errors summed, ``B`` times the run's largest. It is a union over the blocks, so it is vacuous past about
  ``1/δ`` of them, and says so;
* at each δ of `DELTAS`, each resource's pooled figure, from the statement that pays δ once over the period: integrity's
  faulty proof blocks (``IntegrityHorizon``) with the per-block threshold at δ/B beside them (``IntegrityPeriod``),
  PoUS's certified sampling periods (``pooledShort_log``) and the network certifier's leak in bits
  (``HorizonLeakAccepted``, conditional on its per-block inputs);
* what no guarantee states (``open``): the memory term's ``δ_PoUS``, bandwidth pooled over a period, and the others it
  names.

This module computes no bound: it counts each period's blocks, and hands onsite's accepted answers, those counts and
the δs to ``onsite bound``. What is proved is each block's verdict, by onsite's Lean, and each figure's evaluation, by
the same executable; what is scaled is the number of blocks (the deployment's machines and block length) and that every
block is audited as the run's worst is.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verity.ml.circuits.codec import canonical_json

ROOT = Path(__file__).resolve().parents[3]
CENSUS = ROOT / "benchmarks" / "census" / "deployments.json"
REFERENCE = "dc-12500-machines-600s"
PERIODS: tuple[tuple[str, int], ...] = (("week", 7 * 86400), ("month", 30 * 86400), ("year", 365 * 86400))
DELTAS: tuple[str, ...] = ("1/100", "1/1000000")
FORMAT = "verity/all-protocols-monitoring-period/v2"


def deployment(ident: str = REFERENCE, path: Path = CENSUS) -> dict[str, Any]:
    """The census deployment ``ident``."""
    for d in json.loads(path.read_text()):
        if d["id"] == ident:
            return d
    raise KeyError(f"{path} has no deployment {ident!r}: add it there (its machines and window_s are stated parameters)")


def periods(dep: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Each monitoring period's length and its blocks: ``machines × ⌊seconds / window_s⌋``."""
    out = {}
    for name, seconds in PERIODS:
        per_machine = int(seconds // dep["window_s"])
        out[name] = {"seconds": seconds, "blocks": dep["machines"] * per_machine, "blocks_per_machine": per_machine}
    return out


def accepted(reports: Sequence[Mapping[str, Any]]) -> list[tuple[Any, dict]]:
    """Each server-time block onsite accepted (its window report's ``window``), with onsite's answer (`compose.composed`'s ``answer``)."""
    return [(r.get("window"), r["composed"]["answer"]) for r in reports
            if ((r.get("composed") or {}).get("answer") or {}).get("verdict") == "accept"]


def bound(exe: str, answers: Sequence[Mapping[str, Any]], counts: Sequence[int],
          deltas: Sequence[str] = DELTAS) -> dict[str, Any]:
    """``onsite bound`` on onsite's accepted ``answers`` over monitoring periods of each of ``counts`` blocks, with the pooled
    figures at each of ``deltas``: its JSON line."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "answers.jsonl"
        f.write_bytes(b"".join(canonical_json(a) + b"\n" for a in answers))
        argv = [exe, "bound", str(f), *(x for d in deltas for x in ("--delta", d)), *map(str, counts)]
        p = subprocess.run(argv, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"onsite bound exited {p.returncode}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def monitoring(reports: Sequence[Mapping[str, Any]], dep: Mapping[str, Any], exe: str | None = None) -> dict[str, Any]:
    """The monitoring-period report over the run's window reports (module docstring): the deployment, each period's
    blocks and the bound's error over them, ``onsite bound``'s output over the run's accepted blocks, each protocol's
    verdicts over the windows, and the composed verdict (any window's reject rejects)."""
    good = [r for r in reports if "protocols" in r]
    names = list(good[0]["protocols"]) if good else []
    verdicts = {n: sorted({r["protocols"][n]["verdict"] for r in good if n in r["protocols"]}) for n in names}
    composed = [r["composed"]["verdict"] for r in good]
    worst = "reject" if "reject" in composed else "accept" if good and set(composed) == {"accept"} else \
        (sorted(set(composed) - {"accept"}) or ["none"])[0]
    blocks = accepted(good)
    ps = periods(dep)
    out: dict[str, Any] = {"format": FORMAT, "deployment": dict(dep), "periods": ps, "accepted_windows": [w for w, _ in blocks]}
    if exe is None:
        out.update(bound=None, why="no onsite executable given")
    elif not blocks:
        out.update(bound=None, why="no block accepted by onsite, so the bound has no terms")
    else:
        b = bound(exe, [a for _, a in blocks], [p["blocks"] for p in ps.values()])
        for p, h in zip(ps.values(), b["horizons"]):
            assert h["blocks"] == p["blocks"], (h, p)
            p.update(error=h["error"], vacuous=h["vacuous"], pooled=h["pooled"])
        out["bound"] = b
    return {**out, "protocols": {n: {"verdicts": v, "owner": good[0]["protocols"][n].get("owner")}
                                 for n, v in verdicts.items()},
            "composed": {"verdict": worst, "windows": len(composed), "errors": len(reports) - len(good)},
            "windows": [{"window": r.get("window"), "run": r.get("run"), "node": r.get("node"),
                         "composed": (r.get("composed") or {}).get("verdict"), "error": r.get("error")} for r in reports]}


def lead(m: Mapping[str, Any]) -> dict[str, Any]:
    """The monitoring-period figures a summary line leads with: per period its blocks, the joint bound's error over them
    and at each δ each resource's pooled figure (a resource with no inputs gives its why), the run's own blocks beside
    it, what is open, and the composed verdict."""
    b = m.get("bound")
    if b is None:
        return {"why": m["why"], "composed": m["composed"]["verdict"]}
    return {"run": b["run"], **{p: {**{k: row[k] for k in ("blocks", "error", "vacuous")},
                                    **{d: {r: _headline(f) for r, f in fs.items()} for d, fs in row["pooled"].items()}}
                                for p, row in m["periods"].items()},
            "open": [o["figure"] for o in b["open"]], "composed": m["composed"]["verdict"]}


#: each pooled figure's headline field
HEADLINE = ("faulty_blocks_at_most", "certified_at_least", "K_bits")


def _headline(f: Mapping[str, Any]) -> Any:
    return next((f[k] for k in HEADLINE if k in f), f.get("why"))


def main(argv: list[str] | None = None) -> int:
    """``python -m verity_all_protocols.monitoring REPORTS.jsonl... --onsite EXE [--deployment ID] [--out FILE]``: the
    monitoring-period report over every window report in the files (several runs' windows, on several machines, pooled)."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--onsite", help="onsite's executable, whose `bound` evaluates the joint bound")
    ap.add_argument("--deployment", default=REFERENCE)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    reports = [json.loads(x) for f in a.reports for x in Path(f).read_text().splitlines() if x.strip()]
    m = monitoring(reports, deployment(a.deployment), a.onsite)
    text = json.dumps(m, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print("MONITORING " + json.dumps(lead(m), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
