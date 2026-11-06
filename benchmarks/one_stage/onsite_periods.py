"""The onsite demo's record (``verity/onsite-network-record/v1``) restated per monitoring period: a table for a week, a
month and a year, columns the node, a datacenter and the world, each cell its leak bound K over the period, its share of
the weights, δ_net and K over one execution phase beside it, marked **proved** or **scaled** with its rule and sources.
It reads the record as JSON and nothing else (the lock aside), imports no module of its writer, and computes nothing the
record doesn't state.

    python benchmarks/one_stage/onsite_periods.py OUT/record/onsite.json [--lock Security/lean-audit.json] [--out T.md]

Before rendering, it holds each cell's label to the record's rules (`expected`) and refuses the record, exit 1 with each
reason, where one differs. The run's own cell is proved; its K/m over one execution phase is proved only at m = 1
(``bound.phases``), else scaled by ``per-phase``; a period's node cell is proved only where the run's own period
(``bound.period_s``) covers it, else scaled by ``run-to-period``; the datacenter's and the world's, and their per-phase
figures, are scaled by ``datacenter`` and ``world``. A proved cell carries no rule and its statement is in the lock; a
scaled cell's rule is in the record's ``rules`` with its sources. ``periods_tight`` (the tight K, where the figure of
record charges ℓ_a + 1 for each tied audit while a check or a premise is open) is rendered as conditional on what is
open, never as proved.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

FORMAT = "verity/onsite-network-record/v1"
PERIODS = ("week", "month", "year")
COLUMNS = {"node": "the node", "datacenter": "a datacenter", "world": "the world"}
LOCK = Path(__file__).resolve().parents[2] / "Security" / "lean-audit.json"


def signature(entry: Mapping[str, Any]) -> str:
    """A lock entry's signature on one line: the lock wraps a long one over several, often right after its colon."""
    return " ".join(s.strip() for s in entry.get("signature", []))


def locked(lock: Mapping[str, Any], name: str) -> bool:
    """Whether the lock holds ``name``: as a guarantee's key, or as the statement a proof constant's signature proves."""
    g = lock["guarantees"]
    return name in g or any(signature(v).endswith(f" : {name}") for v in g.values())


def expected(grid: Mapping[str, Any], m: int, period_s: Fraction) -> dict[str, tuple[str, str | None]]:
    """Each cell of a ``periods`` grid by its path, with the label the record's rules give it: (how, rule)."""
    one = ("proved", None) if m == 1 else ("scaled", "per-phase")
    out = {"run": ("proved", None), "run.per_phase": one}
    for p in PERIODS:
        node = ("proved", None) if period_s >= grid[p]["seconds"] else ("scaled", "run-to-period")
        out |= {f"{p}.node": node, f"{p}.node.per_phase": one}
        out |= {f"{p}.{c}{x}": ("scaled", c) for c in ("datacenter", "world") for x in ("", ".per_phase")}
    return out


def cell_at(grid: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    c: Any = grid
    for k in path.split("."):
        c = c[k]
    return c


def check(rec: Mapping[str, Any], lock: Mapping[str, Any] | None = None) -> list[str]:
    """Every reason the record's cells don't carry the labels its rules give them (`expected`); empty where they do."""
    if rec.get("format") != FORMAT:
        return [f"the record is {rec.get('format')!r}, not {FORMAT}"]
    b, why = rec.get("bound") or {}, []
    for k in ("missing", "error"):
        if k in b or k in (rec.get("periods") or {}):
            return [f"no bound: {b.get(k) or rec['periods'].get(k)}"]
    m, period_s, fig = int(b["phases"]), Fraction(b["period_s"]), b["of_record"]
    if not b.get("notes"):
        why.append("bound.notes: the overcount note is missing")
    open_ = list(fig["open"])
    if fig["K"] != b["K"] + (sum(fig["advice"]["bits_per_phase"]) if open_ else 0):
        why.append(f"of_record.K {fig['K']} isn't K {b['K']} plus ℓ_a + 1 for each tied audit while {open_} is open")
    net = (rec.get("verdicts") or {}).get("the network certifier", {}).get("verdict")
    grids = {"periods": (rec["periods"], fig["K"])}
    if "periods_tight" in rec:
        grids["periods_tight"] = (rec["periods_tight"], b["K"])
        if not open_ or rec["periods_tight"]["conditional_on"] != open_:
            why.append(f"periods_tight is conditional on {rec['periods_tight']['conditional_on']}, and what is open "
                       f"is {open_}")
    elif fig["K"] != b["K"]:
        why.append("K of record isn't the tight K, and the record has no periods_tight")
    for name, (grid, K) in grids.items():
        run = grid["run"]
        if run["phases"] != m or Fraction(run["seconds"]) != period_s or run["bits"] != K:
            why.append(f"{name}.run is {run['bits']} bits over {run['phases']} phases of {run['seconds']} s, and the "
                       f"bound {K} over m = {m} of {period_s} s")
        for path, (how, rule) in expected(grid, m, period_s).items():
            c, at = cell_at(grid, path), f"{name}.{path}"
            if (c.get("how"), c.get("rule")) != (how, rule):
                why.append(f"{at} is {c.get('how')} by {c.get('rule')}, and its rule makes it {how} by {rule}")
            elif how == "proved" and net != "accept":
                why.append(f"{at} is proved, and the network certifier's verdict is {net}")
            elif how == "proved" and lock is not None and not locked(lock, c["statement"]):
                why.append(f"{at} is proved by {c['statement']}, which the lock doesn't hold")
            elif how == "scaled" and not (grid["rules"].get(rule) or {}).get("sources"):
                why.append(f"{at} is scaled by {rule}, which the record's rules don't give with its sources")
    return why


def bits(n: int) -> str:
    return f"{n:,}" if n < 10**12 else f"{n:.3e}"


def delta(terms: list[Mapping[str, Any]]) -> str:
    return " + ".join(f"{t['symbol']} {t['value']}" if "value" in t else f"{t['times']:,} {t['symbol']}" for t in terms)


def label(c: Mapping[str, Any], conditional: bool) -> str:
    if c["how"] == "scaled":
        return f"**scaled** ({c['rule']})"
    return "**conditional**" if conditional else f"**proved** (`{c['statement']}`)"


def weights(share: float) -> str:
    return f"{share:.3g} of the weights" if share < 1 else f"{share:.3g} × the weights"


def entry(c: Mapping[str, Any], conditional: bool) -> str:
    one = c["per_phase"]
    return (f"{label(c, conditional)}: K {bits(c['bits'])} bits, {weights(c['share'])}; δ_net "
            f"{delta(c['delta_net'])}{' (vacuous)' if c['vacuous'] else ''}<br>a phase: {label(one, conditional)} "
            f"K {bits(one['bits'])} bits, {weights(one['share'])}")


def grid_md(grid: Mapping[str, Any], conditional: bool = False) -> list[str]:
    run = grid["run"]
    L = [f"The run: {label(run, conditional)}: K {bits(run['bits'])} bits over its {run['phases']} execution phases "
         f"({run['seconds']} s), δ_net {delta(run['delta_net'])}; over one phase {label(run['per_phase'], conditional)} "
         f"K {bits(run['per_phase']['bits'])} bits.", "",
         "| period | " + " | ".join(COLUMNS.values()) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    L += [f"| {p} (the node's phases: {grid[p]['phases']:,}) | " + " | ".join(entry(grid[p][c], conditional)
                                                                            for c in COLUMNS) + " |" for p in PERIODS]
    return L


def markdown(rec: Mapping[str, Any]) -> str:
    b, a = rec["bound"], rec["advice_charge"]
    fig, p = b["of_record"], rec["periods"]
    L = [f"# The onsite record of `{rec['run']}` per monitoring period", "",
         f"The run is {'of record' if rec['of_record']['holds'] else 'not of record'} (every verdict accepts: "
         f"{rec['of_record']['holds']}; missing {rec['of_record']['missing']}, not accepting "
         f"{rec['of_record']['not_accepting']}). The network certifier: {rec['verdicts']['the network certifier']['verdict']}. "
         f"m = {b['phases']} execution phases of {p['phase_s']} s, the run's period {b['period_s']} s. "
         f"K of record {bits(fig['K'])} bits, the tight K {bits(b['K'])} (`{b['statement']}`); open: "
         f"{fig['open'] or 'nothing'}.", "",
         f"The advice: ℓ_a = {[bits(x) for x in a['cap']['ell_a']]} bits, the registered cap, is the charge's upper "
         f"bound; K of record charges ℓ_a + 1 for each tied audit, {[bits(x) for x in fig['advice']['bits_per_phase']]} "
         f"a phase, while anything is open. The tight K's ℓ_{{a,j}}: "
         f"{[bits(x) for x in a['charged']['bits_per_phase']]} a phase, {a['charged']['share']:.3g} of the tight K.", ""]
    L += [f"Note: {n}" for n in b["notes"]] + ["", "## K of record", ""] + grid_md(p)
    if "periods_tight" in rec:
        cond = ", ".join(rec["periods_tight"]["conditional_on"])
        L += ["", "## The tight K, conditional", "", f"A cell marked conditional is proved only once each of {cond} holds "
              "(the figure of record's open checks and premises).", ""] + grid_md(rec["periods_tight"], True)
    L += ["", "## Rules", ""]
    for name, r in p["rules"].items():
        src = "; ".join(s if isinstance(s, str) else f"{s['file']} `{s['id']}` {json.dumps(s['read'])}" for s in r["sources"])
        L.append(f"- **{name}**: {r['rule']}. Sources: {src}.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("record", type=Path, help="the onsite record (its writer's OUT/record/onsite.json)")
    ap.add_argument("--lock", type=Path, default=LOCK, help="the lock a proved cell's statement is held to")
    ap.add_argument("--out", type=Path, help="where the table goes (default stdout)")
    a = ap.parse_args(argv)
    rec = json.loads(a.record.read_text())
    if why := check(rec, json.loads(a.lock.read_text())):
        print("\n".join(f"REFUSED {x}" for x in why), file=sys.stderr)
        return 1
    if a.out:
        a.out.write_text(markdown(rec))
    else:
        sys.stdout.write(markdown(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
