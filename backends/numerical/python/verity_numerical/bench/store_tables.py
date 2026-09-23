"""The like-for-like decision table rebuilt from the evidence store (``research.store``) instead of the ledger files.

    python -m verity_numerical.bench.store_tables [--store ROOT] [--proof-class-by PATTERN]        # print the table
    python -m verity_numerical.bench.store_tables --diff-ledger [--proof-class-by PATTERN]          # compare with decision_tables

``decision_tables.LIKE_FOR_LIKE`` stays the skeleton: which rows, their design names, and the constants transcribed from
reports (verifier seconds, proof bytes, depth, who verified).  Every value the ledger used to supply -- hardware, B, prover
seconds per VU, overhead, proof class, soundness, lane, run id -- is read from the store with ``Index.sql`` over the
``labels`` / ``attempts`` / ``artifact_meta`` tables (``ledger_to_store`` put it there): the row's target is the attempt whose
id is the run id, else the ``ledger-row/v1`` artifact of that row; the producing lane's labels grouped by ``ref`` (one ledger
row = one ref) give the row whose ``label`` starts with the key's prefix.

Labels are assertions, policy picks.  Without ``--proof-class-by`` a row shows the producing lane's own ``proof_class``.  With
``--proof-class-by 'red-team-*'`` the latest audit event by a matching asserter on the same target (one event = one ``by`` +
``ref`` + ``ts``) caps it: the row shows ``min(lane's claim, the highest class that event granted)`` -- downgrades stand,
nothing is upgraded, an EARNED re-audit leaves the lane's claim in place.  ``--diff-ledger`` compares cell by cell with the
ledger-built table; in the ``label`` column only the class is compared (``decision_tables`` appends hand-written override prose).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from verity_numerical.bench import contract, decision_tables as DT, ledger, tables
from verity_numerical.bench.ledger_to_store import LEDGER_ROW_KIND

__all__ = ["ATTEMPT_SQL", "LEDGER_ROW_SQL", "ROW_LABELS_SQL", "diff_tables", "main", "render_like_for_like", "row_values"]

#: Is the run id an attempt here?
ATTEMPT_SQL = "SELECT id FROM attempts WHERE id = ?"
#: The ledger-row artifact(s) of (lane, run id text, label prefix); run id None is the flattened text 'null'.
LEDGER_ROW_SQL = """
SELECT a.id FROM artifacts a
  JOIN artifact_meta l ON l.id = a.id AND l.key = 'lane'   AND l.value = ?
  JOIN artifact_meta r ON r.id = a.id AND r.key = 'run_id' AND r.value = ?
  JOIN artifact_meta b ON b.id = a.id AND b.key = 'label'  AND substr(b.value, 1, length(?)) = ?
 WHERE a.kind = ?
 ORDER BY a.id"""
#: Every assertion on a target: the lane's rows (grouped by ref) and everyone else's verdicts.
ROW_LABELS_SQL = "SELECT key, value, by, ts, ref, path FROM labels WHERE target = ? ORDER BY ts, path"

RANK = {c: i for i, c in enumerate(("ARITHMETIC_DIAGNOSTIC", "COMPONENT", "NO_PROOF", "NON_ZK_PROOF_DIAGNOSTIC", "NON_ZK_PROOF",
                                     "COMPLETE_HVZK_BACKEND", "COMPLETE_ZK_BACKEND"))}


def _rank(c: str | None) -> int:
    return RANK.get(c or "", -1)


def _value(text: str | None) -> Any:
    """Undo ``index.scalar_text``: strings are stored raw, everything else as JSON text."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text


def _label_order(path: str) -> tuple[str, int]:
    name = path.rsplit("/", 1)[-1]
    stem = name[:-5] if name.endswith(".json") else name
    base, sep, n = stem.rpartition("~")
    return (base, int(n)) if sep and n.isdigit() else (stem, 1)


@dataclass(frozen=True)
class Assertion:
    key: str
    value: Any
    by: str
    ts: str
    ref: str | None
    path: str


def _assertions(index: Any, target: str) -> list[Assertion]:
    rows = [Assertion(k, _value(v), by, ts, ref, path) for k, v, by, ts, ref, path in index.sql(ROW_LABELS_SQL, (target,))]
    return sorted(rows, key=lambda a: (a.ts, _label_order(a.path)))


def resolve_target(index: Any, lane: str, run_id: str | None, prefix: str) -> str:
    """The attempt (run id) when the store has it, else the row's ledger-row artifact; exactly one or LookupError."""
    if run_id and index.sql(ATTEMPT_SQL, (run_id,)):
        return run_id
    hits = [r[0] for r in index.sql(LEDGER_ROW_SQL, (lane, run_id if run_id is not None else "null", prefix, prefix, LEDGER_ROW_KIND))]
    if len(hits) != 1:
        raise LookupError(f"store key ({lane!r}, {run_id!r}, {prefix!r}): no attempt {run_id!r} and {len(hits)} {LEDGER_ROW_KIND} artifact(s), need exactly 1")
    return hits[0]


def lane_row(assertions: Sequence[Assertion], lane: str, prefix: str) -> dict[str, Any]:
    """The producing lane's claims of the one row whose ``label`` starts with ``prefix``: labels grouped by ref."""
    groups: dict[str | None, dict[str, Any]] = {}
    for a in assertions:
        if a.by == lane:
            key, value = tables.canonical_label(a.key, a.value)     # old spellings (`batch`, `notes`, `candidate=A`) read as the canonical
            groups.setdefault(a.ref, {})[key] = value
    hits = [g for g in groups.values() if isinstance(g.get("label"), str) and g["label"].startswith(prefix)]
    if len(hits) != 1:
        raise LookupError(f"lane {lane!r} asserted {len(hits)} row(s) whose label starts with {prefix!r}; need exactly 1")
    return hits[0]


def policy_proof_class(assertions: Sequence[Assertion], lane: str, claim: str, by_glob: str | None) -> tuple[str, Assertion | None]:
    """Downgrades stand, nothing is upgraded: cap the lane's claim by the latest matching audit event on the target."""
    if not by_glob:
        return claim, None
    verdicts = [a for a in assertions if a.key == "proof_class" and a.by != lane and fnmatch.fnmatchcase(a.by, by_glob)]
    if not verdicts:
        return claim, None
    latest = verdicts[-1]
    event = [a for a in verdicts if (a.by, a.ref, a.ts) == (latest.by, latest.ref, latest.ts)]
    ceiling = max(event, key=lambda a: _rank(a.value))
    return (ceiling.value, ceiling) if _rank(ceiling.value) < _rank(claim) else (claim, ceiling)


def row_values(index: Any, r: DT.Row, by_glob: str | None) -> dict[str, Any]:
    """The dict ``decision_tables._row_values`` builds from a ledger entry, built from store assertions instead."""
    if r.key is None:
        return DT._row_values(r, [])
    lane, run_id, prefix = r.key
    target = resolve_target(index, lane, run_id, prefix)
    asserted = _assertions(index, target)
    row = lane_row(asserted, lane, prefix)
    B = row.get("B") or 1
    spv = r.prove_s / B if r.prove_s is not None else row.get("seconds_per_vu")
    prove = r.prove_s if r.prove_s is not None else (spv * B if spv else None)
    shown, verdict = policy_proof_class(asserted, lane, row.get("proof_class"), by_glob)
    return {"prove_s": prove, "seconds_per_vu": spv,
            "overhead": row.get("overhead") if r.prove_s is None else (spv / (DT.NATIVE_PS * 1e-12) if spv else None),
            "proof_class": shown, "claimed_proof_class": row.get("proof_class"), "verdict": verdict, "hardware": row.get("hardware"),
            "security": row.get("soundness"), "run_id": target if run_id and target == run_id else run_id, "lane": lane, "B": row.get("B"),
            "target": target}


def _label_cell(v: dict[str, Any]) -> str:
    if v.get("verdict") is not None and v["proof_class"] != v.get("claimed_proof_class"):
        return f"{v['proof_class']} ({v['verdict'].by} {v['verdict'].ref}; lane says {v['claimed_proof_class']})"
    return v["proof_class"]


def render_like_for_like(index: Any, by_glob: str | None = None) -> str:
    header = ["design", "hardware", "B", "prover s", "ms/VU", "overhead x native", "verifier s", "proof", "depth",
              *[f"e2e @{int(r)} ms" for r in DT.RTTS_MS], "soundness (as stated)", "label", "verified by", "lane / run id", "flags"]
    rows = []
    for r in DT.LIKE_FOR_LIKE:
        v = row_values(index, r, by_glob)
        d = DT._design(r.design, v, r)
        e2e = [DT._f(DT.latency.e2e(d, rtt, DT.LINK_GBPS)["total"]) if d else "--" for rtt in DT.RTTS_MS]
        spv = v.get("seconds_per_vu")
        name = ("**PENDING** (" + r.pending + ") " if r.pending else "") + r.design
        depth = ("--" if r.depth is None else f"{r.depth}") + (f" ({r.depth_note})" if r.depth_note else "")
        verify = DT._f(r.verify_s) + (f" ({r.verify_note})" if r.verify_note else "")
        rows.append([name, v["hardware"], f"{v['B']}", DT._f(v.get("prove_s")), DT._f(spv * 1e3, 3) if spv else "--", DT._sci(v.get("overhead")),
                     verify, DT._mb(r.proof_bytes), depth, *e2e, v["security"], _label_cell(v), r.verified or "--", f"{v['lane']} / {v['run_id']}", r.flags or ""])
    note = (f"\nEnd-to-end = prove + verify + depth x RTT + proof bytes / {DT.LINK_GBPS:.0f} Gb/s (`bench.latency.e2e`; the a-verifier "
            "lane's convention, `latency.md` used 25 Gb/s). Overhead = prover s/VU over 9.85 ps. Sub-batched rows: the soundness "
            "is the union bound over the sub-batches and the prover time is the sum.")
    return DT._table(header, rows) + note


# --- diff against the ledger-built table -----------------------------------------------------------------------------------


def _cells(table: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln for ln in table.splitlines() if ln.startswith("|")]
    split = lambda ln: [c.strip() for c in ln.strip().strip("|").split(" | ")]
    return split(lines[0]), [split(ln) for ln in lines[2:]]


def diff_tables(store_table: str, ledger_table: str) -> tuple[int, int, list[str]]:
    """(matching rows, rows, differences).  The `label` column is compared by its leading class token."""
    header, ours = _cells(store_table)
    header2, theirs = _cells(ledger_table)
    diffs: list[str] = []
    if header != header2:
        diffs.append(f"header differs: {header} vs {theirs}")
    label_col = header.index("label")
    n = max(len(ours), len(theirs))
    ok = 0
    for i in range(n):
        if i >= len(ours) or i >= len(theirs):
            diffs.append(f"row {i + 1}: present in {'ledger' if i >= len(ours) else 'store'} table only")
            continue
        a, b = ours[i], theirs[i]
        row_ok = True
        for j, col in enumerate(header):
            x, y = a[j] if j < len(a) else "", b[j] if j < len(b) else ""
            if j == label_col:
                x, y = x.split(" ")[0], y.split(" ")[0]
            if x != y:
                row_ok = False
                diffs.append(f"row {i + 1} ({a[0][:60]}) [{col}]: store={x!r} ledger={y!r}")
        ok += row_ok
    return ok, n, diffs


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None, metavar="ROOT", help="store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--proof-class-by", default=None, metavar="PATTERN", help="policy: cap each row's class by the latest matching asserter, e.g. 'red-team-*'")
    ap.add_argument("--diff-ledger", action="store_true", help="compare with decision_tables' ledger-built like-for-like table")
    ap.add_argument("--ledger-dir", default=ledger.DIR, help="ledger directory for --diff-ledger")
    ns = ap.parse_args(argv)
    from research.store.local import LocalStore

    st = LocalStore(ns.store)
    table = render_like_for_like(st.index, ns.proof_class_by)
    if not ns.diff_ledger:
        print(table)
        return 0
    ok, n, diffs = diff_tables(table, DT.render_like_for_like(DT._entries(ns.ledger_dir)))
    print(f"like-for-like from the store vs from the ledger: {ok}/{n} rows match ({100 * ok / n:.1f}%)"
          + (f"; policy --proof-class-by {ns.proof_class_by!r}" if ns.proof_class_by else "; policy: the producing lane's own claim"))
    for d in diffs:
        print("  " + d)
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
