"""The switch digest (``kb/TABLES.md`` "The switch"): the 6 PM PT digest that moves the published tables from the frozen
:mod:`verity_numerical.bench.tables` to the approved spec's :mod:`verity_numerical.bench.views`, labelled as a change of method.

    python -m verity_numerical.bench.switch --root <store root> [--format md|json]

Every number the frozen Table 2 publishes is shown beside its replacement -- the new Table 2 cell of the same family on the same
subcircuit and hardware class, per commitment scheme -- and, when the old artifact is not itself a new cell, beside the reason it
left (its first reject reason under the spec's rules, ``views.REASON_LETTERS``).  New cells with no frozen predecessor are listed
after, and the parity gate (``views.parity``) heads the digest.  The frozen ``+ in-proof hash`` columns fold into their family.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from verity_numerical.bench import tables as T
from verity_numerical.bench import views as V

__all__ = ["build", "main", "render_markdown"]

LABEL = ("CHANGE OF METHOD: from this digest the published tables are the approved spec's (kb/TABLES.md): Table 2 is end-to-end "
         "N ÷ P over the full committed relation on a core-defined commitment scheme, per subcircuit, hardware class and scheme, one "
         "column per backend family (A-GKR, B-Ligero, SP1, Flock). The frozen 2026-09-22 tables stay available as a drill-down.")


def _family(column: str) -> str:
    for suffix in T.AUTH_COLUMN_SUFFIX.values():
        if suffix and column.endswith(suffix):
            return column[: -len(suffix)]
    return column


def build(store: Any, rules: V.Rules = V.SPEC) -> dict[str, Any]:
    """The digest's rows: one per frozen Table 2 cell, then the new cells no frozen cell leads to."""
    frozen = T.build(store)
    views = V.build(store, rules)
    new_by: dict[tuple[str, str], list[V.ViewCell]] = {}
    for c in views.cells:
        new_by.setdefault((c.line, c.column), []).append(c)
    rejected = {r["artifact"]: r for r in views.rejected}
    cell_of = {c.artifact: c for c in views.cells}

    def cell(c: V.ViewCell) -> dict[str, Any]:
        return {"line": c.line, "scheme": c.scheme, "column": c.column, "artifact": c.artifact, "configuration": c.configuration,
                "overhead": c.overhead, "printed": T.fmt_overhead(c.overhead) + (" (prov.)" if c.provisional else "") + (" (alg.)" if c.algebraic else "")}

    rows, used = [], set()
    for o in frozen.cells:
        column = o.candidate + T.AUTH_COLUMN_SUFFIX.get(o.authentication, "")
        fam = _family(column)
        repl = [cell(c) for c in new_by.get((o.target, fam), [])]
        used |= {r["artifact"] for r in repl}
        row: dict[str, Any] = {"line": o.target, "column": column, "family": fam, "artifact": o.artifact, "overhead": o.overhead,
                               "printed": T.fmt_overhead(o.overhead), "replacements": repl}
        if o.artifact in cell_of:
            row["left"] = None
        elif o.artifact in rejected:
            k, why = rejected[o.artifact]["codes"][0]
            row["left"] = {"code": k, "reason": f"{V.REASON_LETTERS.get(k, k)}: {why}", "codes": [x for x, _ in rejected[o.artifact]["codes"]]}
        else:
            row["left"] = {"code": None, "reason": "admissible, but not its family's best on the line"}
        if not repl:
            empties = [e for ln in views.lines if ln["line"] == o.target for e in [ln["empty"].get(fam)] if e]
            row["empty"] = sorted({e["text"] for e in empties})
        rows.append(row)
    added = [cell(c) for c in views.cells if c.artifact not in used]
    return {"schema": "verity/tables-switch-digest/v1", "label": LABEL, "rules": rules.name, "parity": V.parity(store, frozen),
            "old_cells": len(frozen.cells), "new_cells": len(views.cells), "rows": rows, "added": added}


def _empty_text(texts: Sequence[str]) -> str:
    if len(texts) == 1:
        return texts[0]
    return "no cell on any scheme line; the published Table 2's footnotes give each line's reasons" if texts else "no result"


def render_markdown(d: dict[str, Any]) -> str:
    kept = sum(1 for r in d["rows"] if r["left"] is None)
    out = [f"> **{d['label']}**", "",
           f"Parity (the switch's gate 2: the new views under today's parameters select exactly the frozen cells): "
           + ("ok." if not d["parity"] else f"DIFFERENT ({len(d['parity'])}): " + "; ".join(d["parity"])), "",
           f"{d['old_cells']} published number(s) before the switch, {d['new_cells']} Table 2 cell(s) after it (rules `{d['rules']}`); "
           f"{kept} old cell(s) are new cells as they are, {d['old_cells'] - kept} left Table 2.", "",
           "## Each old number beside its replacement", ""]
    rows = []
    for r in d["rows"]:
        repl = "; ".join(f"{c['printed']} ({c['scheme'] or 'no scheme'}, {c['configuration'] or c['column']}, {c['artifact'][:12]}…)"
                         for c in r["replacements"]) or f"{T.EMPTY} ({_empty_text(r.get('empty') or [])})"
        left = "stays (it is the new cell)" if r["left"] is None else (f"{r['left']['code']} — {r['left']['reason']}" if r["left"]["code"]
                                                                        else r["left"]["reason"])
        rows.append([r["line"], r["column"], f"{r['printed']} ({r['artifact'][:12]}…)", left, repl])
    out += [T._md_table(("Subcircuit · hardware", "Old column", "Old number (artifact)", "Why it left Table 2", "Replacement: the family's new cell"), rows)
            if rows else f"(no published number before the switch {T.EMPTY})", ""]
    if d["added"]:
        out += ["## New cells with no old number", "",
                T._md_table(("Subcircuit · hardware", "Scheme", "Family", "Cell (artifact)"),
                            [[c["line"], c["scheme"] or T.EMPTY, c["column"], f"{c['printed']} ({c['artifact'][:12]}…)"] for c in d["added"]]), ""]
    out += ["Reason letters: " + ", ".join(f"{k} {v}" for k, v in V.REASON_LETTERS.items() if k != "B") + ".", ""]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, metavar="ROOT", help="store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--format", choices=("md", "json"), default="md")
    ns = ap.parse_args(argv)
    from research.store.local import LocalStore  # noqa: PLC0415  (only the CLI needs the store package)

    d = build(LocalStore(ns.root))
    print(json.dumps(d, indent=2, default=str) if ns.format == "json" else render_markdown(d), end="\n" if ns.format == "json" else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
