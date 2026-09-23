"""(f) `commit_summary.json` re-derives: `verity_capture.sweep.commit_summary.summarize(row_dir, rc)` with the CURRENT code over the
row's `commit/summary.json`, `commit/runs.jsonl` and `commit.log` gives the recorded summary's stable fields (run counts, tokens_equal,
committer, collector variant / retention, engine settings, engine steps, generated tokens, `pass`).

NOTE (inventory 2026-09-22): `openings` and `coverage` are `None` on every reference row (stale keys: runs.jsonl no longer carries them),
so `pass` degenerates to `rc == 0 and tokens_equal` here.  The Commit VERDICT of record is `commit/verdict.json` (`commit_pass`,
`required_classes_source`) + `commit/sampled_replay_p*.json`, lifted beside the summary in `commit_verdict`; a v2 must state the same.
STRONG on the lifted verdict fields; the summary itself is a derived convenience.  Wall-clock fields (`delta_wall*`, `*_wall`) are
excluded: they are measurements, not reproducible facts.
"""
from __future__ import annotations

import json
import os

from .. import resolver

NAME = "commit_summary"
TIER = "T0"
STRENGTH = "strong"
INPUTS = ("commit_summary.json", "commit/summary.json", "commit/runs.jsonl", "commit.log", "commit/verdict.json")
STABLE = ("rc", "n_runs", "n_control", "n_treatment", "tokens_equal", "committer", "collector_variant", "retention", "engine_settings",
          "engine_steps", "generated_tokens", "pass", "openings", "coverage")
VERDICT = ("commit_pass", "runs_instrumented", "runs_failed", "first_fail_reason", "required_classes_source", "required_manifest")


def applies(row: resolver.Row) -> str | None:
    if not row.has("commit_summary.json"):
        return "no commit_summary.json on this row (Commit never closed)"
    return None


def _lift(cs: dict, verdict: dict | None) -> dict:
    out = {k: cs.get(k) for k in STABLE}
    coll = cs.get("collector") or {}
    out["collector"] = {k: coll.get(k) for k in ("variant", "retention", "window_mb", "window_slots", "layout")} if coll else None
    if verdict is not None:
        out["commit_verdict"] = {k: verdict.get(k) for k in VERDICT if k in verdict}
    return out


def expected(row: resolver.Row) -> dict:
    cs = row.json("commit_summary.json")
    v = row.json("commit/verdict.json") if row.has("commit/verdict.json") else None
    return _lift(cs, v)


def actual(row: resolver.Row) -> dict:
    croot = row.candidate_root()
    if croot is not None and (croot / "commit_summary.json").is_file():
        v = json.load(open(croot / "commit" / "verdict.json")) if (croot / "commit" / "verdict.json").is_file() else None
        return dict(_lift(json.load(open(croot / "commit_summary.json")), v), source="candidate")
    from verity_capture.sweep.commit_summary import summarize
    rc = int(row.json("commit_summary.json").get("rc", 0))
    row_dir = row.path("commit_summary.json").parent
    if not (row_dir / "commit" / "summary.json").is_file():
        raise resolver.NotResolvable(f"{row.key}: commit/summary.json not beside commit_summary.json at {row_dir}")
    if not (row_dir / "commit.log").is_file() and row.has("commit.log"):
        # the store's per-row `records` tree omitted commit.log (engine_settings' source); it lives in the shared `commit_logs` tree
        # (fixtures.toml [artifacts], resolved + sha-checked by row.path) -- summarize() reads one directory, so re-join the two here
        view = resolver.scratch(row) / "commit_view"
        view.mkdir(parents=True, exist_ok=True)
        for name, src in (("commit", row_dir / "commit"), ("commit_summary.json", row_dir / "commit_summary.json"), ("commit.log", row.path("commit.log"))):
            dst = view / name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        row_dir = view
    cs = summarize(str(row_dir), rc)
    v = row.json("commit/verdict.json") if row.has("commit/verdict.json") else None
    return dict(_lift(cs, v), source="recomputed")
