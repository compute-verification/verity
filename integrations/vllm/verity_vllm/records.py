"""The circuit database: per served row, the exact circuit it served (the Build's Programs, per rank), the evidence that it is
faithful (the Match on live data, the Commit's gates, the conformance captures) and every config that determines it (the row file,
`config.RowConfig`), in the evidence store by row id.  Committed values are never kept.

A row's entry is one preserved `vllm-config-record/v1` artifact labelled `row=<id>`.  Its payload is the row's config record
(`config_record.json`, `verity-vllm/config-record/v1`, `pipeline.config_record`) and its refs are the evidence the record names under
`evidence`: `build` (the `vllm-build/v1` holding every rank's Program descriptors, `research_outputs`), `manifest`, `match`,
`commit`, and one ref per conformance capture, named by its property.  `manifests/records.json` indexes the database: row id -> the
record's art: id.  The store is reached through the research CLI (`python -m research data ...`), so `RESEARCH_STORE` /
`RESEARCH_STORE_CONFIG` select it.

Conformance: the properties a row's Programs need are those its capture map (`commit.capture_map.of_checkpoint`, the Build's
checkpoint and world) names for the Programs the Build uses (`captures` of each map Program one of whose Definitions a construction
manifest's `spec_histogram` holds), the record's `conformance.due`.  The record cites each one's capture of record under
`evidence.conformance` (a `property-record/v1`: one run's `<property>.json`, validated and holding, referring to the run's record;
`capture` preserves them) or names it a gap with its reason under `evidence.gaps`; `publish` refuses a record that leaves one silent.

A `research run` job has no store (its custody key stays with the runner): the launcher writes the row's record and the descriptors
the job reads into one file (`bundle`), ships it (`research run --send FILE`), and the job decodes them from it (`programs(row,
bundle=...)`), each digest checked against the record as from the store.

    from verity_vllm import records
    from verity.ml.circuits.evaluation.reference import evaluate
    rec = records.record(row)                         # the config record
    progs = records.programs(row, ranks=[0], part=["workload"])   # {rank: {part: DecodedProgram}}, each digest checked
    p = progs[0]["workload"]                          # the Program the Commit binds; "build_step", "build_request_LP*_T*" beside it
    p.root, p.defs                                    # its root and its Definitions, decoded through the registry
    evaluate(p.defs[fid], *synthetic_inputs)          # any Definition on synthetic inputs
    records.programs(row, [0], part=["workload"], bundle=Path(os.environ["RESEARCH_RUN_DIR"]) / "inputs" / "glm53.tar")  # in a job

    verity-vllm records publish <row dir> <row> --build art:… --tree REF@SHA --by LANE [--manifest art:…] [--match art:…]
                                [--match-run R] [--commit art:…] [--commit-run R] [--summary FILE] [--conformance PROPERTY=art:…]
                                [--gap PROPERTY=REASON]
    verity-vllm records capture <run> [--property P]  # preserve a capture run's property records; prints PROPERTY=art:…
    verity-vllm records show [<row>]                  # the index, or a row's record
    verity-vllm records programs <row> [--rank R] [--part P] [--bundle FILE]   # fetch, decode and check a row's Programs
    verity-vllm records fetch <row> --to FILE [--rank R] [--part P]   # a job's bundle: research run --send FILE
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tarfile
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
PROPERTY_KIND = "property-record/v1"
CONSTRUCTION = "construction_manifest.json"


def data(*args: str) -> str:
    """`research data ARGS`'s stdout (this interpreter's research, in this process's environment); a refusal raises with what the CLI
    said."""
    p = subprocess.run([sys.executable, "-m", "research", "data", *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"research data {' '.join(args[:2])}: rc={p.returncode}: {p.stderr.strip()[-1500:]}")
    return p.stdout


def fetch(art: str, to: Path | None = None, paths: list[str] = ()) -> Path:
    """Materialise `art`'s payload (only `paths` of a tree, when given) and return its path.  A store with no remote (a `research
    run` job's) that lacks it names the bundle that serves a job instead."""
    try:
        out = data("fetch", art, *(["--to", str(to)] if to else []), *(f"--path={p}" for p in paths))
    except RuntimeError as e:
        if "no remote configured" not in str(e):
            raise
        raise RuntimeError(f"{e}: a research run job has no store; the launcher writes the row's bundle (`verity-vllm records fetch "
                           f"ROW --to FILE`), sends it (`research run --send FILE`) and the job reads it (`records programs ROW --bundle "
                           f"$RESEARCH_RUN_DIR/inputs/FILE`, `programs(row, bundle=...)`)") from None
    return Path(out.strip().splitlines()[-1])


def listing(art: str) -> list[str]:
    """The paths of a tree artifact's files."""
    return [f["path"] for f in json.loads(data("fetch", art, "--list", "--json"))]


def index() -> dict[str, str]:
    """Row id -> its record's art: id."""
    return json.loads(INDEX.read_text()) if INDEX.is_file() else {}


def art_of(row: str) -> str:
    try:
        return index()[row]
    except KeyError:
        raise UsageError(f"{row}: not in the circuit database ({INDEX.name}; `verity-vllm records publish` adds a row)") from None


def record(row: str, bundle: Path | None = None) -> dict[str, Any]:
    """The row's config record, from the store or from a `bundle` (`records fetch`)."""
    if bundle is None:
        return json.loads(fetch(art_of(row)).read_text())
    with tarfile.open(bundle) as tf:
        rec = json.load(tf.extractfile(CR.NAME))
    if rec["config"].get("row") != row:
        raise UsageError(f"{bundle}: the record of {rec['config'].get('row')}, not {row}")
    return rec


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


def selected(rec: dict[str, Any], ranks: list[int] | None = None, part: list[str] | None = None) -> dict[int, dict[str, str]]:
    """`parts` of the record, only `ranks` and the parts `part` names when given; one the record does not name is refused."""
    row, named = rec["config"].get("row"), parts(rec)
    if missing := sorted(set(ranks or ()) - set(named)):
        raise UsageError(f"{row}: no rank {missing} (the record names ranks {sorted(named)})")
    if unknown := sorted(set(part or ()) - {p for ps in named.values() for p in ps}):
        raise UsageError(f"{row}: no part {unknown} (the record names {sorted({p for ps in named.values() for p in ps})})")
    return {r: {p: d for p, d in ps.items() if part is None or p in part} for r, ps in named.items() if ranks is None or r in ranks}


def _unpack(bundle: Path, to: Path, paths: list[str]) -> Path:
    with tarfile.open(bundle) as tf:
        if missing := sorted(set(paths) - set(tf.getnames())):
            raise UsageError(f"{bundle}: lacks {len(missing)} descriptor(s), e.g. {missing[0]}: `verity-vllm records fetch` it with "
                             f"those ranks and parts")
        for p in paths:
            tf.extract(p, to, filter="data")
    return to


def programs(row: str, ranks: list[int] | None = None, *, part: list[str] | None = None, to: Path | None = None,
             bundle: Path | None = None) -> dict[int, dict[str, Any]]:
    """The row's Programs, {rank: {part: DecodedProgram}} (only `ranks` and the parts `part` names, when given), fetched from its Build
    (`evidence.build`, only the descriptors asked for) or read from a `bundle` (`records fetch`), and decoded through the registry
    (`program.boolean.load_program`).  A descriptor whose digest is not the record's is refused."""
    from verity_vllm.program.boolean import load_program
    rec = record(row, bundle)
    want = selected(rec, ranks, part)
    paths = [descriptor(rec, r, p) for r, ps in want.items() for p in ps]
    out: dict[int, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="records-") as tmp:
        root = fetch(rec["evidence"]["build"], to, paths) if bundle is None else _unpack(bundle, Path(to or tmp), paths)
        for r, ps in sorted(want.items()):
            out[r] = {}
            for p, digest in sorted(ps.items()):
                prog, got = load_program(root / descriptor(rec, r, p))
                if got != digest:
                    raise ValueError(f"{row} rank {r} {p}: the descriptor's digest {got[:16]}… is not the record's {digest[:16]}…")
                out[r][p] = prog
    return out


def bundle(row: str, to: Path, ranks: list[int] | None = None, *, part: list[str] | None = None) -> Path:
    """One file holding the row's record and the descriptors `ranks` / `part` select, at their Build paths: what a `research run` job
    reads the row's Programs from (`programs(row, bundle=...)`) after the launcher sends it (`--send`)."""
    rec = record(row)
    paths = [descriptor(rec, r, p) for r, ps in selected(rec, ranks, part).items() for p in ps]
    with tempfile.TemporaryDirectory(prefix="records-") as tmp:
        root = fetch(rec["evidence"]["build"], Path(tmp), paths)
        body = (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode()
        with tarfile.open(to, "w") as tf:
            ti = tarfile.TarInfo(CR.NAME)
            ti.size = len(body)
            tf.addfile(ti, io.BytesIO(body))
            for p in paths:
                tf.add(root / p, arcname=p)
    return to


def capture(run: str, properties: list[str] | None = None) -> dict[str, str]:
    """Preserve the property records run `run` wrote (each property's shallowest `out/**/<property>.json` in its run record; only
    `properties`, when given) as `property-record/v1` artifacts referring to the run's record, each refused unless its schema is
    the registered property's, its digest recomputes and it holds.  Returns {property: art: id}."""
    from verity_vllm.properties.record import REGISTRY, PropertyRecord
    att = json.loads(data("show", run, "--json"))
    rr = (att.get("outputs") or {}).get("run_record")
    if not rr:
        raise UsageError(f"{run}: no run record (outputs.run_record; `research run --on` keeps one)")
    found: dict[str, str] = {}
    for p in sorted(listing(rr), key=lambda p: (p.count("/"), p)):
        name = Path(p).stem
        if p.startswith("out/") and p.endswith(".json") and name in REGISTRY and (properties is None or name in properties):
            found.setdefault(name, p)
    missing = sorted(set(properties or ()) - set(found))
    if missing or not found:
        raise UsageError(f"{run}: its record {rr} holds no out/**/<property>.json{' for ' + ', '.join(missing) if missing else ''}")
    out = {}
    with tempfile.TemporaryDirectory(prefix="records-") as tmp:
        root = fetch(rr, Path(tmp), list(found.values()))
        for name, p in sorted(found.items()):
            pr = PropertyRecord.load(root / p)
            problems = pr.problems() + ([] if pr.name in (None, name) else [f"the record is {pr.name}'s"])
            if problems or not pr.ok:
                raise UsageError(f"{run} {p}: {'; '.join(problems) or 'the property does not hold'}: not a capture of record")
            meta = {"property": name, "schema": pr.doc["schema"], "digest": pr.digest, "ok": True, "path": p, "run_id": run,
                    "commit": (att.get("source") or {}).get("commit")}
            art = ART.findall(data("put", "--kind", PROPERTY_KIND, "--meta", json.dumps(meta, sort_keys=True), "--file", str(root / p),
                                   f"--ref=run={rr}"))[-1]
            data("push", art)
            out[name] = art
    return out


def check_capture(prop: str, art: str) -> None:
    """Refuse `art` as `prop`'s capture of record unless it is a holding `property-record/v1` of `prop` at its registered schema."""
    from verity_vllm.properties.record import REGISTRY
    m = json.loads(data("show", art, "--json"))["manifest"]
    meta, want = m.get("meta") or {}, REGISTRY.get(prop)
    if m.get("kind") != PROPERTY_KIND or meta.get("property") != prop or meta.get("ok") is not True or want is None \
            or meta.get("schema") != want.schema:
        raise UsageError(f"--conformance {prop}={art}: a {m.get('kind')} of {meta.get('property')} at {meta.get('schema')} "
                         f"(ok {meta.get('ok')}), not a holding {PROPERTY_KIND} of {prop} at {want.schema if want else '<unregistered>'} "
                         f"(`verity-vllm records capture RUN` preserves one)")


def due(build: str, listed: list[str]) -> dict[str, Any]:
    """The record's `conformance`: the capture map (`map`, its id) of the Build's checkpoint and world (its construction manifests'
    `model.pin.model` and `tp.world`) and `due`, each property its `captures` name for a map Program the Build uses (one of whose
    Definitions a construction manifest's `spec_histogram` holds) -> those Programs.  A Build with no construction manifest is
    refused: what its Programs use is unknown."""
    from verity_vllm.commit import capture_map as CM
    from verity_vllm.query.program_view import spec_base
    cms = [p for p in listed if p.rsplit("/", 1)[-1] == CONSTRUCTION]
    if not cms:
        raise UsageError(f"the Build {build} holds no {CONSTRUCTION}: the Definitions its Programs use, so the conformance they need, "
                         f"are unknown")
    with tempfile.TemporaryDirectory(prefix="records-") as tmp:
        root = fetch(build, Path(tmp), cms)
        docs = [json.loads((root / p).read_text()) for p in cms]
    pins = {(((m.get("model") or {}).get("pin") or {}).get("model"), int((m.get("tp") or {}).get("world") or 1)) for m in docs}
    if len(pins) != 1:
        raise UsageError(f"the Build {build}'s construction manifests name {len(pins)} checkpoints and worlds: {sorted(map(str, pins))}")
    cmap = CM.of_checkpoint(*pins.pop())
    if cmap is None:
        return {"map": None, "due": {}}
    used = {spec_base(s) for m in docs for s in m.get("spec_histogram") or {}}
    progs = {prop: [p for p in pids if used & set(cmap["programs"][p]["definitions"])] for prop, pids in CM.captures(cmap).items()}
    return {"map": cmap["id"], "due": {prop: ps for prop, ps in progs.items() if ps}}


def evidence_of(build: str, *, manifest: str | None = None, match: str | None = None, match_run: str | None = None,
                commit: str | None = None, commit_run: str | None = None, conformance: dict[str, str] | None = None,
                gaps: dict[str, str] | None = None) -> dict[str, Any]:
    """The record's `evidence` block, each id checked for its form, each gap for its reason."""
    ev = {"build": build, "manifest": manifest, "match": match, "match_run": match_run, "commit": commit, "commit_run": commit_run,
          "conformance": dict(sorted((conformance or {}).items())), "gaps": dict(sorted((gaps or {}).items()))}
    bad = [f"{k}={v!r}" for k, v in ev.items() if k in REFS and v is not None and not ART.fullmatch(v)]
    bad += [f"{k}={v!r}" for k, v in ev.items() if k.endswith("_run") and v is not None and not RUN.fullmatch(v)]
    bad += [f"conformance {k}={v!r}" for k, v in ev["conformance"].items() if not ART.fullmatch(v) or k in ev]
    bad += [f"gap {k}={v!r}" for k, v in ev["gaps"].items() if not str(v).strip() or k in ev or k in ev["conformance"]]
    if bad:
        raise UsageError(f"evidence: not an art: id / run id / a gap's reason (a property is cited or a gap, not both): {', '.join(bad)}")
    return ev


def publish(d: str | Path, row: str, *, evidence: dict[str, Any], tree: str, by: str, summary: dict[str, Any] | None = None) -> str:
    """Write the row's config record from its row dir `d` (the stages' files, or a copy of them) and its Commit of record's
    `summary`, refuse it unless it is PASS, the Commit committed against the Build's manifest and Programs, its Build artifact
    holds every Program it names, and every conformance property its Programs need (`due`) is cited by a holding capture of
    record or named a gap, then label it `row=<id>`, preserve it with its label and index it.  Returns its art: id."""
    if not TREE.fullmatch(tree):
        raise UsageError(f"--tree {tree!r}: REF@SHA, the code the row was built and served from")
    rec = CR.record(d, row, tree=tree, summary=summary, evidence=evidence)
    if rec["outcome"] != "PASS":
        raise UsageError(f"{row}: the record is {rec['outcome']} ({rec['why']}): the database keeps rows whose circuit is faithful")
    cm = (rec["commit"].get("manifest") or {}) if summary is not None else {}
    workloads = {str(r): ps.get("workload") for r, ps in parts(rec).items()}
    if cm.get("manifest_digest") not in (None, rec["build"].get("manifest_digest")) or any(
            workloads.get(str(r)) != v for r, v in (cm.get("program_digest_per_rank") or {}).items()):
        raise UsageError(f"{row}: the Commit committed against manifest {str(cm.get('manifest_digest'))[:16]}… and its Programs, not the "
                         f"Build's ({str(rec['build'].get('manifest_digest'))[:16]}…): the record's Build is not the one served")
    listed = listing(evidence["build"])
    if missing := [p for r, ps in parts(rec).items() for p in (descriptor(rec, r, q) for q in ps) if p not in listed]:
        raise UsageError(f"{row}: the Build {evidence['build']} lacks {len(missing)} Program descriptor(s): {', '.join(missing[:4])}")
    rec["conformance"] = due(evidence["build"], listed)
    cited, gaps = evidence.get("conformance") or {}, evidence.get("gaps") or {}
    if silent := sorted(set(rec["conformance"]["due"]) - set(cited) - set(gaps)):
        raise UsageError(f"{row}: its Programs need {', '.join(silent)} ({rec['conformance']['map']}), which the record neither cites nor "
                         f"names a gap: --conformance PROPERTY=art:… (`verity-vllm records capture RUN`) or --gap PROPERTY=REASON")
    for prop, art in cited.items():
        check_capture(prop, art)
    refs = {**{k: evidence[k] for k in REFS if evidence.get(k)}, **cited}
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
    conformance: list[str] = option("--conformance", action="append", default=[], metavar="PROPERTY=art:...",
                                    help="a property's capture of record (`records capture`)")
    gap: list[str] = option("--gap", action="append", default=[], metavar="PROPERTY=REASON",
                            help="a property the Programs need with no capture of record for the tree, and why")
    cmd: str = field(default="publish", init=False)


@dataclass(frozen=True, kw_only=True)
class CaptureOptions:
    """Preserve a capture run's property records as property-record/v1 artifacts; print PROPERTY=art:… for --conformance."""

    run: str = option("run", help="the capture run (a run id with a run record)")
    property: list[str] = option("--property", action="append", default=[], help="these properties only (default every one it wrote)")
    cmd: str = field(default="capture", init=False)


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
    part: list[str] = option("--part", action="append", default=[], help="these parts only, e.g. workload (default every part)")
    bundle: str | None = option("--bundle", default=None, help="read the record and descriptors from this file (`records fetch`)")
    cmd: str = field(default="programs", init=False)


@dataclass(frozen=True, kw_only=True)
class FetchOptions:
    """Write a row's record and the descriptors of the ranks and parts asked for into one file, for a research run job (--send it)."""

    row: str = option("row")
    to: str = option("--to", help="the file to write (a tar)")
    rank: list[int] = option("--rank", type=int, action="append", default=[], help="these ranks only (default every rank)")
    part: list[str] = option("--part", action="append", default=[], help="these parts only, e.g. workload (default every part)")
    cmd: str = field(default="fetch", init=False)


COMMANDS = {"publish": PublishOptions, "capture": CaptureOptions, "show": ShowOptions, "programs": ProgramsOptions, "fetch": FetchOptions}


def _pairs(flag: str, items: list[str], form: str) -> dict[str, str]:
    out = dict(c.split("=", 1) for c in items if "=" in c)
    if len(out) != len(items):
        raise UsageError(f"{flag} {items}: {form}, each property once")
    return out


def main(a: PublishOptions | CaptureOptions | ShowOptions | ProgramsOptions | FetchOptions) -> int:
    if a.cmd == "publish":
        ev = evidence_of(a.build, manifest=a.manifest, match=a.match, match_run=a.match_run, commit=a.commit, commit_run=a.commit_run,
                         conformance=_pairs("--conformance", a.conformance, "PROPERTY=art:..."),
                         gaps=_pairs("--gap", a.gap, "PROPERTY=REASON"))
        summary = json.loads(Path(a.summary).read_text()) if a.summary else None
        print(publish(a.row_dir, a.row, evidence=ev, tree=a.tree, by=a.by, summary=summary))
        return 0
    if a.cmd == "capture":
        for prop, art in capture(a.run, a.property or None).items():
            print(f"{prop}={art}")
        return 0
    if a.cmd == "show":
        print(json.dumps(record(a.row) if a.row else index(), indent=1, sort_keys=True))
        return 0
    if a.cmd == "fetch":
        print(bundle(a.row, Path(a.to), a.rank or None, part=a.part or None))
        return 0
    src = Path(a.bundle) if a.bundle else None
    named = parts(record(a.row, src))
    for r in a.rank or sorted(named):
        for part, p in programs(a.row, [r], part=a.part or None, bundle=src)[r].items():
            print(f"rank {r}  {part:28s} {named[r][part][:16]}  gates {p.gates}  definitions {len(p.defs)}", flush=True)
    return 0
