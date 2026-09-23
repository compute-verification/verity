"""Import the overhead ledger (``backends/numerical/reports/ledger/*.jsonl``) into the evidence store as labels.

    python -m verity_numerical.bench.ledger_to_store --store ROOT [--ledger-dir DIR] [--dry-run] [--json]

The ledger is frozen history: one line per claim a lane made, written when it was made.  The store keeps every claim as an
assertion with provenance (``research.store`` README §8) and never picks a winner; ``store_tables`` does.  For every row:

* the target is the row's run when that run is an attempt in the store (attempt id == run id); otherwise -- rows with no run
  id, or a run that never reached ``~/.research/runs`` -- the row itself becomes a ``ledger-row/v1`` artifact (``meta`` = the
  row as written) and that artifact is the target.  Nothing about the row is lost either way.
* every field of the row is one label by the lane (``by`` = lane): ``label`` (the headline), ``proof_class``, ``hardware``,
  ``soundness`` (the ledger's ``security_target``), ``authentication``, ``assumptions`` (JSON list), ``breakthrough``, ``B``,
  ``K``, ``seconds_per_vu``, ``overhead``, ``scope``, ``track``, ``omitted``, ``note`` (+ ``verified_by`` where the row has it; a
  row's ``notes`` is written as ``note``), plus ``candidate`` derived from the lane name (``CANDIDATES``, Table 1 spelling: ``A-GKR``,
  ``B-Ligero``, ``SP1``...).  ``ts`` is the row's own timestamp; ``ref``
  is ``<ledger file>:<line>`` -- the frozen source, and the key that groups one row's labels when several rows share a run.
* the red-team verdicts that the campaign carried as table overrides are separate assertions by the red-team, with the
  red-team's run as ``ref`` (``REDTEAM``): the downgrade of the pre-fix B-Ligero ZK rows and the EARNED re-audit of b-zk-fix.
  ``research data labels <target> --history`` then shows the whole dispute; a table's policy decides what stands.

Idempotent: a label whose (key, value, by, ref) already exists on the target is skipped; artifacts are content-addressed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from verity_numerical.bench import ledger

__all__ = ["CANDIDATES", "KEY_ALIASES", "REDTEAM", "ROW_STRUCTURAL", "VALUE_ALIASES", "candidate", "import_ledger", "ledger_ref", "main", "row_labels", "ts_z"]

LEDGER_ROW_KIND = "ledger-row/v1"
#: Ledger keys that are the row's identity or provenance, not claims: they become the target / ``by`` / ``ts`` / ``ref``.
ROW_STRUCTURAL = frozenset({"schema", "ts", "lane", "run_id"})
#: ledger field -> label key.  ``notes`` (a handful of red-team rows) is the same claim as ``note`` (store vocabulary, 2026-09-23 hygiene).
RENAMED = {"security_target": "soundness", "notes": "note"}
#: Old label spellings that mean the same claim, so that a re-import after the 2026-09-23 hygiene stays idempotent: an existing
#: ``candidate=A`` IS the ``A-GKR`` claim, an existing ``notes`` IS the ``note`` claim.  Mirrors ``research.store.vocab`` (which this
#: package does not import) and is cross-checked by the tests.
KEY_ALIASES = {"notes": "note", "verify_note": "note", "batch": "B", "checker": "verifier"}
VALUE_ALIASES = {("candidate", "A"): "A-GKR"}

#: lane glob -> candidate (the Table 1 spelling: ``bench.tables.CANDIDATES`` ids), first match wins (``shared`` otherwise).
CANDIDATES: tuple[tuple[str, str], ...] = (
    ("a-*", "A-GKR"),
    ("b-air", "B-AIR"), ("b-air-v2", "B-AIR"), ("b-lookup", "B-AIR"), ("b-chain", "B-AIR"), ("b-bind", "B-AIR"), ("soundness-fixes", "B-AIR"),
    ("b-ligero*", "B-Ligero"), ("b-zk*", "B-Ligero"), ("b-e2e-v2", "B-Ligero"), ("b-v2", "B-Ligero"), ("b-lintest", "B-Ligero"),
    ("b-commit", "B-Ligero"), ("b-encode", "B-Ligero"),
    ("c-*", "C"),
    ("sp1*", "SP1"),
)


def candidate(lane: str) -> str:
    for pat, cand in CANDIDATES:
        if fnmatch.fnmatchcase(lane, pat):
            return cand
    return "shared"


#: The red-team assertions the campaign applied on top of the ledger (``decision_tables.Row.label_override`` and
#: ``ZK_STATUS``).  ``rows`` selects affected ledger rows as (lane, run id, predicate on the row); ``labels`` are (key, value)
#: pairs asserted on every selected row's target; ``ts_from`` names the red-team's own ledger row whose timestamp dates the assertion.
REDTEAM: tuple[dict[str, Any], ...] = (
    {"by": "red-team-zk", "ref": "r20260922-100305-2b13", "ts_from": ("red-team-zk", "rt-zk VERDICT"),
     "rows": (("b-zk", "r20260922-092404-2a1b", None),
              ("b-ligero2", None, lambda r: r["proof_class"] in ("COMPLETE_HVZK_BACKEND", "COMPLETE_ZK_BACKEND")),
              ("gate4-h100", "r20260922-092841-94ac", lambda r: r["label"].startswith("B e2e HVZK"))),
     "labels": (("proof_class", "NON_ZK_PROOF_DIAGNOSTIC"),
                ("finding", "DOWNGRADE (red-team-zk F1: q unmasked on H, accumulators recovered; F2: simulator PROM-only; F3: FS run priced with "
                            "the interactive bound): COMPLETE_ZK/HVZK_BACKEND -> NON_ZK_PROOF_DIAGNOSTIC"))},
    {"by": "red-team-zk-2", "ref": "r20260922-112029-2db3", "ts_from": ("red-team-zk2", "EARNED: COMPLETE_ZK_BACKEND interactive"),
     "rows": (("b-zk-fix", "r20260922-105905-2021", None),),
     "labels": (("proof_class", "COMPLETE_HVZK_BACKEND"), ("proof_class", "COMPLETE_ZK_BACKEND"),
                ("finding", "EARNED (red-team-zk2 re-audit of b-zk-fix): COMPLETE_ZK_BACKEND interactive (mask rank 197/197, 0/16150 "
                            "accumulators, coins-alone soundness holds); COMPLETE_HVZK_BACKEND fiat-shamir (rank 288/288, FS accounting "
                            "reproduces 2^-128.04, t=288 minimal)"))},
)


def ts_z(ts: str) -> str:
    """Ledger ``2026-09-22T09:20:19+00:00`` -> label ``2026-09-22T09:20:19Z`` (UTC, seconds)."""
    return _dt.datetime.fromisoformat(ts).astimezone(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ledger_ref(path: Path, line: int) -> str:
    """``backends/numerical/reports/ledger/<lane>.jsonl:<line>`` -- repo-relative when the file is under the repo."""
    p = Path(path).resolve()
    for anc in p.parents:
        if (anc / "backends" / "numerical" / "reports" / "ledger").is_dir() and p.is_relative_to(anc):
            return f"{p.relative_to(anc).as_posix()}:{line}"
    return f"{p.as_posix()}:{line}"


def run_id_of(row: dict[str, Any]) -> str | None:
    """The row's run id: the ``run_id`` field (the ledger's own key; a run id mentioned in ``note`` names other evidence)."""
    rid = row.get("run_id")
    return rid if isinstance(rid, str) and rid else None


def row_labels(row: dict[str, Any]) -> dict[str, Any]:
    """Every claim in the row, keyed for the store.  ``None`` values are not asserted."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in ROW_STRUCTURAL or v is None:
            continue
        out[RENAMED.get(k, k)] = v
    out["candidate"] = candidate(row["lane"])
    return out


def _rows(directory: Path) -> list[tuple[Path, int, dict[str, Any]]]:
    """(file, 1-based line, row) for every ledger row, validated by ``ledger.load``."""
    out = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        rows = ledger.load(path)
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        numbered = [i for i, ln in enumerate(path.read_text().splitlines(), 1) if ln.strip()]
        assert len(rows) == len(lines) == len(numbered)
        out += [(path, n, r) for n, r in zip(numbered, rows)]
    return out


def _canon_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class _Target:
    """The store side of one target: the labels it already carries, as (key, canonical value, by, ref)."""

    def __init__(self, store: Any, target: str):
        self.store, self.target = store, target
        self.have: set[tuple[str, str, str, str | None]] = {self._sig(lab.key, lab.value, lab.by, lab.ref) for lab in store.labels(target)}

    @staticmethod
    def _sig(key: str, value: Any, by: str, ref: str | None) -> tuple[str, str, str, str | None]:
        """The claim's identity, old spellings read as the canonical one (so the store is never asked to repeat a claim it holds)."""
        key = KEY_ALIASES.get(key, key)
        if isinstance(value, str):
            value = VALUE_ALIASES.get((key, value), value)
        return key, _canon_text(value), by, ref

    def add(self, key: str, value: Any, by: str, ref: str, ts: str, *, dry_run: bool) -> bool:
        sig = self._sig(key, value, by, ref)
        if sig in self.have:
            return False
        if not dry_run:
            self.store.add_label(self.target, key, value, by, ref=ref, ts=ts)
        self.have.add(sig)
        return True


def _attempt_exists(store: Any, run_id: str) -> bool:
    """Local presence only: a remote that is configured but unusable here must not turn the import into a network question."""
    p = store.attempt_path(run_id) if hasattr(store, "attempt_path") else None
    if p is not None:
        return Path(p).exists()
    return store.get_attempt(run_id) is not None


def _artifact_id(store: Any, row: dict[str, Any], *, dry_run: bool) -> str:
    if dry_run:
        from research.store.model import ArtifactManifest

        return ArtifactManifest(kind=LEDGER_ROW_KIND, meta=row, refs={}, payload=None).validate().id
    return store.put_artifact(LEDGER_ROW_KIND, row)


def import_ledger(store: Any, directory: Path = ledger.DIR, *, dry_run: bool = False, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Apply every row and every red-team assertion; return counts and the per-row targets."""
    say = log or (lambda s: None)
    rows = _rows(directory)
    stats: dict[str, Any] = {"rows": len(rows), "on_attempts": 0, "on_artifacts": 0, "artifacts_new": 0, "labels_added": 0, "labels_skipped": 0,
                             "redteam_added": 0, "redteam_skipped": 0, "targets": {}, "dry_run": dry_run}
    targets: dict[str, _Target] = {}
    existing_art = set(store.artifact_ids()) if hasattr(store, "artifact_ids") else set()

    def target_of(t: str) -> _Target:
        if t not in targets:
            targets[t] = _Target(store, t)
        return targets[t]

    placed: list[tuple[Path, int, dict[str, Any], str]] = []
    for path, line, row in rows:
        ref = ledger_ref(path, line)
        rid = run_id_of(row)
        if rid and _attempt_exists(store, rid):
            target = rid
            stats["on_attempts"] += 1
        else:
            target = _artifact_id(store, row, dry_run=dry_run)
            stats["on_artifacts"] += 1
            if target not in existing_art:
                stats["artifacts_new"] += 1
                existing_art.add(target)
        placed.append((path, line, row, target))
        stats["targets"][ref] = target
        t = target_of(target)
        ts = ts_z(row["ts"])
        added = 0
        for key, value in row_labels(row).items():
            if t.add(key, value, row["lane"], ref, ts, dry_run=dry_run):
                added += 1
            else:
                stats["labels_skipped"] += 1
        stats["labels_added"] += added
        say(f"{ref:<58} -> {target[:44]:<44} {added:>3} label(s) added" + ("" if added else " (all present)"))

    by_lane: dict[str, list[dict[str, Any]]] = {}
    for _, _, row, _ in placed:
        by_lane.setdefault(row["lane"], []).append(row)
    for rt in REDTEAM:
        lane, prefix = rt["ts_from"]
        src = [r for r in by_lane.get(lane, []) if r["label"].startswith(prefix)]
        if len(src) != 1:
            raise LookupError(f"red-team ledger row ({lane!r}, {prefix!r}) matched {len(src)} rows, need exactly 1")
        ts = ts_z(src[0]["ts"])
        hit_targets: list[str] = []
        for sel_lane, sel_run, pred in rt["rows"]:
            for _, _, row, target in placed:
                if row["lane"] == sel_lane and (sel_run is None or row["run_id"] == sel_run) and (pred is None or pred(row)) and target not in hit_targets:
                    hit_targets.append(target)
        if not hit_targets:
            raise LookupError(f"red-team assertion by {rt['by']} selected no ledger rows")
        for target in hit_targets:
            t = target_of(target)
            for key, value in rt["labels"]:
                if t.add(key, value, rt["by"], rt["ref"], ts, dry_run=dry_run):
                    stats["redteam_added"] += 1
                else:
                    stats["redteam_skipped"] += 1
            say(f"red-team {rt['by']:<14} ref {rt['ref']} -> {target[:44]}")
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None, metavar="ROOT", help="store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--ledger-dir", type=Path, default=ledger.DIR)
    ap.add_argument("--dry-run", action="store_true", help="resolve targets and count what would be added; write nothing")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true", help="counts only")
    ns = ap.parse_args(argv)
    from research.store.local import LocalStore

    st = LocalStore(ns.store)
    stats = import_ledger(st, ns.ledger_dir, dry_run=ns.dry_run, log=None if (ns.quiet or ns.json) else print)
    if ns.json:
        print(json.dumps(stats, indent=2, sort_keys=True))
        return 0
    print(f"{'DRY RUN: ' if ns.dry_run else ''}{stats['rows']} ledger rows -> {stats['on_attempts']} on attempts, {stats['on_artifacts']} on "
          f"{LEDGER_ROW_KIND} artifacts ({stats['artifacts_new']} new); labels {stats['labels_added']} added, {stats['labels_skipped']} already present; "
          f"red-team assertions {stats['redteam_added']} added, {stats['redteam_skipped']} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
