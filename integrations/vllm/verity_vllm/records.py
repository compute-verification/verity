"""The circuit database: per served row, the exact circuit it served (the Build's Programs, per rank), the evidence that it is
faithful (the Match on live data, the Commit's gates) and every config that determines it (the row file, `config.RowConfig`), in the
evidence store by row id.  Committed values are never kept.

A row's entry is one preserved `vllm-config-record/v1` artifact labelled `row=<id>`.  Its payload is the row's config record
(`config_record.json`, `verity-vllm/config-record/v1`, `pipeline.config_record`) and its refs are the evidence the record names under
`evidence`: `build` (the `vllm-build/v1` holding every rank's Program descriptors, `research_outputs`), `manifest`, `match`,
`commit`, and one ref per conformance capture, named by its property.  `manifests/records.json` indexes the database: row id -> the
record's art: id.  The store is reached through the research CLI (`research data ...`, `pipeline.research.research_argv`), so
`RESEARCH_STORE` / `RESEARCH_STORE_CONFIG` select it.

    from verity_vllm import records
    from verity.ml.circuits.evaluation.reference import evaluate
    rec = records.record(row)                         # the config record
    progs = records.programs(row, ranks=[0])          # {rank: {part: DecodedProgram}}, each digest checked against the record
    p = progs[0]["workload"]                          # the Program the Commit binds; "build_step", "build_request_LP*_T*" beside it
    p.root, p.defs                                    # its root and its Definitions, decoded through the registry
    evaluate(p.defs[fid], *synthetic_inputs)          # any Definition on synthetic inputs

    verity-vllm records publish <row dir> <row> --build art:… --tree REF@SHA --by LANE [--manifest art:…] [--match art:…]
                                [--match-run R] [--commit art:…] [--commit-run R] [--summary FILE] [--conformance PROPERTY=art:…]
    verity-vllm records show [<row>]                  # the index, or a row's record
    verity-vllm records programs <row> [--rank R]     # fetch, decode and check a row's Programs, one rank at a time
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.config import MANIFESTS, UsageError, option
from verity_vllm.pipeline import config_record as CR

INDEX = MANIFESTS / "records.json"
ART = re.compile(r"art:[0-9a-f]{64}")
RUN = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
TREE = re.compile(r"\S+@[0-9a-f]{7,40}")
REFS = ("build", "manifest", "match", "commit")


def data(*args: str) -> str:
    """`research data ARGS`'s stdout; a refusal raises with what the CLI said."""
    from verity_vllm.pipeline.research import abs_store_env, research_argv
    argv, env = research_argv(None, sys.executable, abs_store_env(dict(os.environ)))
    p = subprocess.run([*argv, "data", *args], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"research data {' '.join(args[:2])}: rc={p.returncode}: {p.stderr.strip()[-1500:]}")
    return p.stdout


def fetch(art: str, to: Path | None = None, paths: list[str] = ()) -> Path:
    """Materialise `art`'s payload (only `paths` of a tree, when given) and return its path."""
    return Path(data("fetch", art, *(["--to", str(to)] if to else []), *(f"--path={p}" for p in paths)).strip().splitlines()[-1])


def index() -> dict[str, str]:
    """Row id -> its record's art: id."""
    return json.loads(INDEX.read_text()) if INDEX.is_file() else {}


def art_of(row: str) -> str:
    try:
        return index()[row]
    except KeyError:
        raise UsageError(f"{row}: not in the circuit database ({INDEX.name}; `verity-vllm records publish` adds a row)") from None


def record(row: str) -> dict[str, Any]:
    return json.loads(fetch(art_of(row)).read_text())


def part_dir(part: str) -> str:
    """The Build dir of a record's Program part: `workload` -> build_workload; a TP1 summary's `step` / `request` -> build_step /
    build_request; a TP rank's parts are their dirs."""
    return "build_workload" if part == "workload" else part if part.startswith("build_") else f"build_{part}"


def parts(rec: dict[str, Any]) -> dict[int, dict[str, str]]:
    """{rank: {part: program digest}} of every Program the record names (a TP1 row's are rank 0), the workload Program included."""
    b = rec["build"]
    per = b.get("per_rank") or {"0": {"programs": b.get("programs") or {}, "workload_digest": b.get("workload_digest")}}
    out = {}
    for r, p in per.items():
        d = {k: v for k, v in (p.get("programs") or {}).items() if v}
        if p.get("workload_digest"):
            d["workload"] = p["workload_digest"]
        out[int(r)] = d
    return out


def descriptor(rec: dict[str, Any], rank: int, part: str) -> str:
    """A part's descriptor, relative to the root of the record's Build tree."""
    return f"{f'build/rank{rank}/' if rec['build'].get('per_rank') else ''}{part_dir(part)}/descriptor.json.gz"


def programs(row: str, ranks: list[int] | None = None, *, to: Path | None = None) -> dict[int, dict[str, Any]]:
    """The row's Programs, {rank: {part: DecodedProgram}}, fetched from its Build (`evidence.build`, only the descriptors asked for)
    and decoded through the registry (`program.boolean.load_program`).  A descriptor whose digest is not the record's is refused."""
    from verity_vllm.program.boolean import load_program
    rec = record(row)
    named = parts(rec)
    if missing := sorted(set(ranks or ()) - set(named)):
        raise UsageError(f"{row}: no rank {missing} (the record names ranks {sorted(named)})")
    want = {r: ps for r, ps in named.items() if ranks is None or r in ranks}
    root = fetch(rec["evidence"]["build"], to, [descriptor(rec, r, p) for r, ps in want.items() for p in ps])
    out: dict[int, dict[str, Any]] = {}
    for r, ps in sorted(want.items()):
        out[r] = {}
        for part, digest in sorted(ps.items()):
            prog, got = load_program(root / descriptor(rec, r, part))
            if got != digest:
                raise ValueError(f"{row} rank {r} {part}: the descriptor's digest {got[:16]}… is not the record's {digest[:16]}…")
            out[r][part] = prog
    return out


def evidence_of(build: str, *, manifest: str | None = None, match: str | None = None, match_run: str | None = None,
                commit: str | None = None, commit_run: str | None = None, conformance: dict[str, str] | None = None) -> dict[str, Any]:
    """The record's `evidence` block, each id checked for its form."""
    ev = {"build": build, "manifest": manifest, "match": match, "match_run": match_run, "commit": commit, "commit_run": commit_run,
          "conformance": dict(sorted((conformance or {}).items()))}
    bad = [f"{k}={v!r}" for k, v in ev.items() if k in REFS and v is not None and not ART.fullmatch(v)]
    bad += [f"{k}={v!r}" for k, v in ev.items() if k.endswith("_run") and v is not None and not RUN.fullmatch(v)]
    bad += [f"conformance {k}={v!r}" for k, v in ev["conformance"].items() if not ART.fullmatch(v) or k in ev]
    if bad:
        raise UsageError(f"evidence: not an art: id / run id: {', '.join(bad)}")
    return ev


def publish(d: str | Path, row: str, *, evidence: dict[str, Any], tree: str, by: str, summary: dict[str, Any] | None = None) -> str:
    """Write the row's config record from its row dir `d` (the stages' files, or a copy of them) and its Commit of record's
    `summary`, refuse it unless it is PASS and its Build artifact holds every Program it names, then label it `row=<id>`, preserve
    it with its label and index it.  Returns its art: id."""
    if not TREE.fullmatch(tree):
        raise UsageError(f"--tree {tree!r}: REF@SHA, the code the row was built and served from")
    rec = CR.record(d, row, tree=tree, summary=summary, evidence=evidence)
    if rec["outcome"] != "PASS":
        raise UsageError(f"{row}: the record is {rec['outcome']} ({rec['why']}): the database keeps rows whose circuit is faithful")
    listed = {f["path"] for f in json.loads(data("fetch", evidence["build"], "--list", "--json"))}
    if missing := [p for r, ps in parts(rec).items() for p in (descriptor(rec, r, q) for q in ps) if p not in listed]:
        raise UsageError(f"{row}: the Build {evidence['build']} lacks {len(missing)} Program descriptor(s): {', '.join(missing[:4])}")
    refs = {**{k: evidence[k] for k in REFS if evidence.get(k)}, **(evidence.get("conformance") or {})}
    meta = {"row": row, "outcome": rec["outcome"], "world": rec["config"].get("tp"), "tree": tree}
    with tempfile.TemporaryDirectory(prefix="config-record-") as tmp:
        p = Path(tmp) / CR.NAME
        p.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
        art = ART.findall(data("put", "--kind", CR.KIND, "--meta", json.dumps(meta, sort_keys=True), "--file", str(p),
                               *(f"--ref={k}={v}" for k, v in refs.items())))[-1]
    data("label", art, "row", row, "--by", by)
    data("push", art)
    INDEX.write_text(json.dumps(dict(sorted({**index(), row: art}.items())), indent=1) + "\n")
    return art


@dataclass(frozen=True, kw_only=True)
class PublishOptions:
    """Write a row's config record, preserve it labelled row=<id> and add it to manifests/records.json."""

    row_dir: str = option("row_dir", help="the row dir, or a copy of the stages' files the record reads")
    row: str = option("row")
    build: str = option("--build", help="the vllm-build/v1 holding every rank's Program descriptors")
    tree: str = option("--tree", help="the code the row was built and served from, REF@SHA")
    by: str = option("--by", help="the label's author (a lane)")
    manifest: str | None = option("--manifest", default=None)
    match: str | None = option("--match", default=None, help="the Match's record")
    match_run: str | None = option("--match-run", default=None)
    commit: str | None = option("--commit", default=None, help="the Commit of record")
    commit_run: str | None = option("--commit-run", default=None)
    summary: str | None = option("--summary", default=None, help="the Commit of record's summary.json (else the row dir's commit/)")
    conformance: list[str] = option("--conformance", action="append", default=[], metavar="PROPERTY=art:...")
    cmd: str = field(default="publish", init=False)


@dataclass(frozen=True, kw_only=True)
class ShowOptions:
    """Print the index, or a row's config record."""

    row: str | None = option("row", nargs="?", default=None)
    cmd: str = field(default="show", init=False)


@dataclass(frozen=True, kw_only=True)
class ProgramsOptions:
    """Fetch and decode a row's Programs, checking each digest against its record; print each part's digest, gates and Definitions."""

    row: str = option("row")
    rank: list[int] = option("--rank", type=int, action="append", default=[], help="these ranks only (default every rank)")
    cmd: str = field(default="programs", init=False)


COMMANDS = {"publish": PublishOptions, "show": ShowOptions, "programs": ProgramsOptions}


def main(a: PublishOptions | ShowOptions | ProgramsOptions) -> int:
    if a.cmd == "publish":
        conf = dict(c.split("=", 1) for c in a.conformance if "=" in c)
        if len(conf) != len(a.conformance):
            raise UsageError(f"--conformance {a.conformance}: PROPERTY=art:...")
        ev = evidence_of(a.build, manifest=a.manifest, match=a.match, match_run=a.match_run, commit=a.commit, commit_run=a.commit_run,
                         conformance=conf)
        summary = json.loads(Path(a.summary).read_text()) if a.summary else None
        print(publish(a.row_dir, a.row, evidence=ev, tree=a.tree, by=a.by, summary=summary))
        return 0
    if a.cmd == "show":
        print(json.dumps(record(a.row) if a.row else index(), indent=1, sort_keys=True))
        return 0
    named = parts(record(a.row))
    for r in a.rank or sorted(named):
        for part, p in programs(a.row, [r])[r].items():
            print(f"rank {r}  {part:28s} {named[r][part][:16]}  gates {p.gates}  definitions {len(p.defs)}", flush=True)
    return 0
