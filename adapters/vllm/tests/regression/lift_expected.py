"""Freeze the reference: `files.toml` (the index of every reference file: sha256 / bytes / role), `fixtures.toml` (per row, where
the bytes live) and `expected/<row>.json` (every check's contract, lifted from the row's OWN documents).  Nothing here is typed by
hand.  files.toml is a registered fixture (`fixtures/artifacts.json`): fetch it before a run (`research data fetch-fixtures`), and
re-register it after one (`research data refresh-fixtures --preserve`), or the next fetch puts the registered bytes back.

    python -m tests.regression.lift_expected --index /path/to/INDEX.json --oracle /path/to/oracle-mirror [--only-expected]

`INDEX.json` is the vault extraction's inventory (`verity/r19-oracle-index/v1`: rows -> files -> {bytes, sha256, extracted, source});
`--oracle` a directory of `<row>/<rel>` files (the vault dir itself, or a mirror of its small records).  Roles: `record` (a small
document), `manifest` (manifest.json, 39-222 MB), `program` (`build_request*/instances.json.gz`), `fold` (match/program.json,
match/instances.jsonl, binding maps: multi-GB, never staged).  Artifact ids (`[rows.<row>.artifacts]`) are filled by `stage_store.py`
after the trees are put; until then the store path is empty and the resolver uses the oracle / live dirs.

Re-running this over the same inventory is idempotent (sorted keys, canonical JSON); a changed expected/ file in the diff means the
reference itself moved -- a finding, not a refresh.  The hand-written `[[rows.<row>.decisions]]` blocks (labelled decisions, see
test_regression.py LABELLED DECISIONS) are the one thing not lifted from the records: they are read from the existing fixtures.toml
and written back verbatim.

expected/ is lifted only for the rows the index names.  A row recorded outside the vault extraction (workload 101, the stochastic canary)
has its own INDEX-shaped inventory: `--append` adds the rows it names to the fixtures.toml and files.toml on disk (the existing text is
kept byte for byte; a key already there is refused), and a later regeneration from the vault's INDEX.json carries every row that index does not name
over as it is (spec, files, artifacts).  An index entry may state its `selection` (default: `green` for GREEN, else `by-name`).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from . import checks, resolver

GREEN = {11, 23, 39, 60, 67, 68, 73, 74}
REFERENCE = {"tag": "v1-reference", "release": "r19-int-21", "commit": "89af272a8c1d37a44b3e8f3253378d7a152fc281", "branch": "lane/vllm-integration"}
SNAPSHOT_NAME = "r19-reference-v1"
PROGRAM_RE = re.compile(r"(^|/)build(_request[^/]*|/rank\d+/build_request[^/]*)/instances\.json\.gz$")
FOLD_RE = re.compile(r"^match/(program\.json|instances\.jsonl|accesses\.jsonl\.gz)$|^commit/binding_map_p\d+\.json$")
LIVE_ROOT = "/workspace/cp/sweep"


def _toml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def dumps_expected(doc: dict) -> str:
    """The serialisation of `expected/<row>.json`: sorted keys, one top-level check per line, the check's document COMPACT (a
    `replay_partition.strata` contract is ~5 MB of numbers; `tests/test_repository.py` BLOB_LIMIT is 5 MiB -- indentation is not content)."""
    lines = ["{"]
    keys = sorted(doc)
    for i, k in enumerate(keys):
        lines.append(f" {json.dumps(k)}: {json.dumps(doc[k], sort_keys=True, separators=(',', ':'), default=str)}" + ("," if i < len(keys) - 1 else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def _toml_val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return _toml_str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_val(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_val(x)}" for k, x in v.items()) + " }"
    raise TypeError(type(v))


DECISION_KEYS = ("check", "since", "since_commit", "finding", "status", "reason", "expected", "pins")


def existing_decisions() -> dict[str, list[dict]]:
    """The `[[rows.<row>.decisions]]` blocks of the fixtures.toml on disk (hand-written; carried over on regeneration), by row key."""
    if not resolver.FIXTURES.is_file():
        return {}
    fx = resolver.load_fixtures()
    return {k: v["decisions"] for k, v in (fx.get("rows") or {}).items() if v.get("decisions")}


def existing_retired() -> dict[str, dict[str, dict]]:
    """The `[rows.<row>.retired.<check>]` tables on disk (hand-written; carried over on regeneration), by row key."""
    if not resolver.FIXTURES.is_file():
        return {}
    return {k: v["retired"] for k, v in (resolver.load_fixtures().get("rows") or {}).items() if v.get("retired")}


def existing_rows(*, with_files: bool = False) -> dict[str, dict]:
    """The `[rows.<row>]` tables of the fixtures.toml on disk (carried over when the index does not name them), by row key; with
    their files.toml entries when `with_files`."""
    if not resolver.FIXTURES.is_file():
        return {}
    rows = dict(resolver.load_fixtures().get("rows") or {})
    files = resolver.load_files() if with_files and rows else {}
    return {k: dict(v, files=files.get(k) or {}) for k, v in rows.items()} if with_files else rows


def existing_shared_artifacts() -> dict[str, str]:
    """The top-level `[artifacts]` table on disk (shared `<row_id>/<rel>` trees such as `commit_logs`; carried over on regeneration)."""
    if not resolver.FIXTURES.is_file():
        return {}
    return dict(resolver.load_fixtures().get("artifacts") or {})


def role_of(rel: str) -> str:
    if rel == "manifest.json":
        return "manifest"
    if PROGRAM_RE.search(rel):
        return "program"
    if FOLD_RE.search(rel):
        return "fold"
    return "record"


def build_fixtures(index: dict, oracle: Path, keep_roles: set[str], decisions: dict[str, list[dict]] | None = None,
                   shared_artifacts: dict[str, str] | None = None, carried: dict[str, dict] | None = None,
                   retired: dict[str, dict[str, dict]] | None = None) -> tuple[str, str, dict]:
    """fixtures.toml's text, files.toml's, and the rows."""
    decisions = decisions or {}
    retired = retired or {}
    rows_out: dict[str, dict] = {}
    for key, v in sorted(index["rows"].items(), key=lambda kv: kv[1]["row"]):
        n = int(v["row"])
        m_tp = re.search(r"__tp(\d+)__", key)
        m_b = re.search(r"__b(\d+)__", key)
        spec: dict = {"row": n, "class": v.get("class"), "selection": v.get("selection") or ("green" if n in GREEN else "by-name"),
                      "tp": int(m_tp.group(1)) if m_tp else 1, "batch": int(m_b.group(1)) if m_b else None,
                      "oracle": v.get("dir") or f"/vault/tmp/oracle/{key}", "local": {}, "artifacts": {"records": "", "programs": ""},
                      "programs_state": "pending_upload", "files": {}}
        for s in v.get("sources") or []:
            if s.get("pod") == "tp2" and s.get("prefix", "").startswith("workspace/cp/sweep/") and s.get("members_for_row", 0) > 0:
                spec["local"]["tp2"] = "/" + s["prefix"].rstrip("/")
        for rel, f in sorted(v["files"].items()):
            role = role_of(rel)
            if role not in keep_roles:
                continue
            if role == "record" and not f.get("extracted"):
                continue
            ent = {"bytes": int(f["bytes"]), "role": role}
            if f.get("sha256"):
                ent["sha256"] = f["sha256"]
            if role in ("manifest", "program", "fold"):
                ent["extracted"] = bool(f.get("extracted"))
            spec["files"][rel] = ent
        rows_out[key] = spec
    for key, spec in (carried or {}).items():
        if key not in rows_out:
            rows_out[key] = {"local": {}, "artifacts": {}, "files": {}, **spec}
    rows_out = dict(sorted(rows_out.items(), key=lambda kv: int(kv[1]["row"])))
    lines = ["# Reference fixtures index -- GENERATED by tests/regression/lift_expected.py from the vault extraction's INDEX.json.",
             "# Per row: where the bytes live (oracle dir, live pod dirs, store artifacts); the sha256 / bytes / role of each reference file",
             "# are files.toml's, a registered fixture (fixtures/artifacts.json).",
             "# Nothing large is in Git: this is an index.  `artifacts.*` are filled by stage_store.py once the trees are put.",
             "# `[[rows.<row>.decisions]]` blocks are hand-written labelled decisions (sanctioned differences from v1; test_regression.py",
             "# LABELLED DECISIONS) and are carried over verbatim when this file is regenerated; so are the hand-written",
             "# `[rows.<row>.retired.<check>]` tables (a check that cannot run on the row's record, test_regression.py HOW TO RUN).", "",
             "schema = \"verity-vllm/r19-reference-fixtures/v1\"", f"snapshot = {_toml_str(SNAPSHOT_NAME)}", "",
             "[reference]"] + [f"{k} = {_toml_str(v)}" for k, v in REFERENCE.items()] + [""]
    flines = ["# Reference files -- GENERATED by tests/regression/lift_expected.py from the vault extraction's INDEX.json: the",
              "# sha256 / bytes / role of every reference file, which resolver.Row.path checks a resolved file against.  A registered",
              "# fixture (fixtures/artifacts.json); after a change, `research data refresh-fixtures --preserve` re-registers it.", ""]
    if shared_artifacts:
        lines += ["# shared fixture/v1 trees laid out `<row_id>/<rel>` (resolver.Row.path, step 3b): files the per-row `records` trees omitted",
                  "[artifacts]"] + [f"{k} = {_toml_str(v)}" for k, v in sorted(shared_artifacts.items())] + [""]
    for key, spec in rows_out.items():
        q = _toml_str(key)
        lines.append(f"[rows.{q}]")
        for k in ("row", "class", "selection", "tp", "batch", "oracle", "programs_state", "manifest_flags"):
            if spec.get(k) is not None:
                lines.append(f"{k} = {_toml_val(spec[k])}")
        lines.append("")
        for d in decisions.get(key) or []:
            lines.append("# labelled decision (Draft 3 §7): a sanctioned difference from v1, consulted by the named check when its result differs from v1 "
                         "(test_regression.py, LABELLED DECISIONS)")
            lines.append(f"[[rows.{q}.decisions]]")
            for dk in list(DECISION_KEYS) + sorted(set(d) - set(DECISION_KEYS)):
                if dk in d:
                    lines.append(f"{dk} = {_toml_val(d[dk])}")
            lines.append("")
        for check, r in sorted((retired.get(key) or {}).items()):
            lines.append(f"[rows.{q}.retired.{check}]")
            lines += [f"{rk} = {_toml_val(r[rk])}" for rk in list(resolver.RETIREMENT_FIELDS) + sorted(set(r) - set(resolver.RETIREMENT_FIELDS)) if rk in r]
            lines.append("")
        lines.append(f"[rows.{q}.local]")
        for mk, mv in sorted(spec["local"].items()):
            lines.append(f"{mk} = {_toml_str(mv)}")
        lines.append("")
        lines.append(f"[rows.{q}.artifacts]")
        for ak, av in spec["artifacts"].items():
            lines.append(f"{ak} = {_toml_str(av)}")
        lines.append("")
        for rel, ent in spec["files"].items():
            flines.append(f"[rows.{q}.files.{_toml_str(rel)}]")
            flines += [f"{ek} = {_toml_val(ent[ek])}" for ek in ("role", "bytes", "sha256", "extracted") if ek in ent]
            flines.append("")
    return "\n".join(lines), "\n".join(flines), rows_out


def lift_expected(oracle: Path, keys: set[str] | None = None) -> dict[str, dict]:
    os.environ[resolver.ROWS_ROOT_ENV] = str(oracle)
    out: dict[str, dict] = {}
    for row in resolver.rows():
        if keys is not None and row.key not in keys:
            continue
        doc: dict = {"row": row.number, "class": row.cls, "reference": REFERENCE, "checks": {}}
        for chk in checks.all_checks():
            why = chk.applies(row)
            if why:
                doc["checks"][chk.NAME] = {"applies": False, "why": why}
                continue
            try:
                doc[chk.NAME] = chk.expected(row)
                doc["checks"][chk.NAME] = {"applies": True, "tier": chk.TIER, "strength": chk.STRENGTH}
            except resolver.NotResolvable as e:
                doc["checks"][chk.NAME] = {"applies": True, "lifted": False, "why": str(e)[:300]}
            except Exception as e:  # noqa: BLE001
                doc["checks"][chk.NAME] = {"applies": True, "lifted": False, "error": f"{type(e).__name__}: {e}"[:300]}
                print(f"[lift] {row.key} {chk.NAME}: {type(e).__name__}: {e}", file=sys.stderr)
        out[row.key] = doc
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--only-expected", action="store_true", help="keep fixtures.toml and files.toml, refresh expected/ only")
    ap.add_argument("--append", action="store_true", help="add the index's rows to fixtures.toml and files.toml (existing text untouched), lift expected/ for them")
    ap.add_argument("--roles", default="record,manifest,program", help="roles to index (default: record,manifest,program; fold adds the multi-GB pins)")
    a = ap.parse_args(argv)
    index = json.load(open(a.index))
    oracle = Path(a.oracle)
    if a.append:
        dup = sorted(set(index["rows"]) & set(existing_rows()))
        if dup:
            print(f"[lift] --append: already in {resolver.FIXTURES}: {dup}", file=sys.stderr)
            return 2
        text, ftext, _ = build_fixtures(index, oracle, set(a.roles.split(",")), decisions=existing_decisions())
        old_files = resolver.FILES.read_text() if resolver.FILES.is_file() else ""
        if not old_files:
            print(f"[lift] --append: {resolver.FILES} is not here; `research data fetch-fixtures` fetches it", file=sys.stderr)
            return 2
        resolver.FIXTURES.write_text(resolver.FIXTURES.read_text().rstrip("\n") + "\n\n" + text[text.index("\n[rows.") + 1:])
        if "\n[rows." in ftext:
            resolver.FILES.write_text(old_files.rstrip("\n") + "\n\n" + ftext[ftext.index("\n[rows.") + 1:])
        print(f"[lift] appended {sorted(index['rows'])} to {resolver.FIXTURES} and {resolver.FILES}")
    elif not a.only_expected:
        text, ftext, _ = build_fixtures(index, oracle, set(a.roles.split(",")), decisions=existing_decisions(),
                                        shared_artifacts=existing_shared_artifacts(), carried=existing_rows(with_files=True),
                                        retired=existing_retired())
        resolver.FIXTURES.write_text(text)
        resolver.FILES.write_text(ftext)
        print(f"[lift] wrote {resolver.FIXTURES} ({len(text) / 1024:.0f} KB) and {resolver.FILES} ({len(ftext) / 1024:.0f} KB)")
    resolver.EXPECTED_DIR.mkdir(exist_ok=True)
    docs = lift_expected(oracle, keys=set(index["rows"]))
    for key, doc in docs.items():
        p = resolver.EXPECTED_DIR / f"{key}.json"
        p.write_text(dumps_expected(doc))
        lifted = [k for k, v in doc["checks"].items() if v.get("applies") and v.get("lifted", True)]
        print(f"[lift] #{doc['row']:>2} {key[:48]:<48} lifted {len(lifted)}/{len(doc['checks'])}: {', '.join(lifted)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
