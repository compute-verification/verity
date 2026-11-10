"""The circuit database: per served row, the exact circuit it served (the Build's Programs, per rank), the evidence that it is
faithful (the Match on live data, the Commit's gates, the conformance captures) and every config that determines it (the row file,
`config.RowConfig`), in the evidence store by row id.  Committed values are never kept.

A row's entry is one preserved `vllm-config-record/v1` artifact labelled `row=<id>`.  Its payload is the row's config record
(`config_record.json`, `verity-vllm/config-record/v1`, `pipeline.config_record`) and its refs are the evidence the record names under
`evidence`: `build` (the `vllm-build/v1` holding every rank's Program descriptors, `research_outputs`), `manifest`, `match`,
`commit`, and one ref per conformance capture, named by its property.  `manifests/records.json` indexes the database
(`config_record.load_index`): per row id the record's art: id and its circuit's pins, every rank's Program digests as the Build made
them, which `publish` writes and `verity-vllm neutrality` rebuilds and compares.  The store is reached through the research CLI
(`python -m research data ...`), so `RESEARCH_STORE` / `RESEARCH_STORE_CONFIG` select it.

Conformance: the properties a row's Programs need are those its capture map (`commit.capture_map.of_checkpoint`, the Build's
checkpoint and world) names for the Programs the Build uses (`captures` of each map Program one of whose Definitions a construction
manifest's `spec_histogram` holds), the record's `conformance.due`.  The record cites each one's capture of record under
`evidence.conformance` (a `property-record/v1`: one run's `<property>.json`, validated and holding, referring to the run's record;
`capture` preserves them) or names it a gap with its reason under `evidence.gaps`; `publish` refuses a record that leaves one silent.
With `newest`, each one the caller leaves out cites its newest holding capture in the store (`newest_capture`): what the row pipeline
publishes with (`pipeline.research.chain`, once a row's Build, Match and Commit pass).  Where the store has no index (`unindexed`: a
`research run` job's), one its catalog lacks is refused, not read as no capture.  A record published from another tree (a
pod's job tree) reaches this tree's index with `pull`: the entry is rebuilt from the record alone (`entry`).

A checkpoint's staging is the label `staged` on `ckpt:<repo>@<revision>`: when a host finished staging its weights and checking every
shard's sha256, its ref the evidence (`staged`).  The step that stages weights writes it (`ops/pod_bootstrap.sh`, through its run's
custody key); `glue` reads it as the start of a model's hours to record.

A `research run` job has no store (its custody key stays with the runner): the launcher writes the row's record and the descriptors
the job reads into one file (`bundle`), ships it (`research run --send FILE`), and the job decodes them from it (`programs(row,
bundle=...)`), each digest checked against the record as from the store.

    from verity_vllm.pipeline import circuit_db
    from verity.ml.circuits.evaluation.reference import evaluate
    rec = circuit_db.record(row)                      # the config record
    progs = circuit_db.programs(row, ranks=[0], part=["workload"])   # {rank: {part: DecodedProgram}}, each digest checked
    p = progs[0]["workload"]                          # the Program the Commit binds; "build_step", "build_request_LP*_T*" beside it
    p.root, p.defs                                    # its root and its Definitions, decoded through the registry
    evaluate(p.defs[fid], *synthetic_inputs)          # any Definition on synthetic inputs
    circuit_db.programs(row, [0], part=["workload"], bundle=Path(os.environ["RESEARCH_RUN_DIR"]) / "inputs" / "glm53.tar")  # in a job

    verity-vllm records publish <row dir> <row> --build art:… --tree REF@SHA --by LANE [--tier fast|heavy] [--manifest art:…]
                                [--match art:…] [--match-run R] [--commit art:…] [--commit-run R] [--summary FILE]
                                [--conformance PROPERTY=art:…] [--gap PROPERTY=REASON] [--newest-captures]
    verity-vllm records pull <art:…> [--tier fast|heavy]   # index a record published elsewhere (a pod's `row chain`) in records.json
    verity-vllm records capture <run> [--property P]  # preserve a capture run's property records; prints PROPERTY=art:…
    verity-vllm records show [<row>]                  # the rows and their records, or a row's record
    verity-vllm records programs <row> [--rank R] [--part P] [--bundle FILE]   # fetch, decode and check a row's Programs
    verity-vllm records fetch <row> --to FILE [--rank R] [--part P]   # a job's bundle: research run --send FILE
    verity-vllm records staged <repo> <revision> --at UTC --ref EVIDENCE --by LANE   # a checkpoint's staging event
    verity-vllm records switch <row> --comparison comparison.json --run R --trace art:… [--tier T]   # the row's Build becomes AOT

A switched row (`switch`: its `aot` in the index, `config_record.aot_of`) has the AOT Build on its record path
(`verity_vllm.program.predict.aot`): its pins are the predicted Programs, equal on every rank to the traced Build its comparison checked.
"""
from __future__ import annotations

import io
import json
import re
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from verity_vllm.config import RowId, UsageError, option
from verity_vllm.pipeline import config_record as CR
from verity_vllm.pipeline.store_data import data, fetch, listing, selected_from

INDEX = CR.INDEX
ART = re.compile(r"art:[0-9a-f]{64}")
RUN = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
TREE = re.compile(r"\S+@[0-9a-f]{7,40}")
REFS = ("build", "manifest", "match", "commit")
PROPERTY_KIND = "property-record/v1"
CONSTRUCTION = "construction_manifest.json"


def index() -> dict[str, dict[str, Any]]:
    """Row id -> its index entry: `record`, its record's art: id (None for a row pinned before it has one), and its circuit's pins."""
    return CR.load_index(INDEX)["rows"]


def art_of(row: str) -> str:
    e = index().get(row)
    if not (e or {}).get("record"):
        raise UsageError(f"{row}: not in the circuit database ({INDEX.name}{' pins its Programs but names no record' if e else ''}; "
                         f"`verity-vllm records publish` adds a row)")
    return e["record"]


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
    from verity.ml.definitions.boolean import load_program
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


def newest_capture(prop: str) -> str | None:
    """`prop`'s newest holding capture of record at its registered schema in the store (the latest capture run's), None when none.
    A store with nothing to select from (no index and an empty catalog: a `research run` job's) raises: `research data select`
    exits 1 there rather than answer none."""
    from verity_vllm.properties.record import REGISTRY
    if (want := REGISTRY.get(prop)) is None:
        return None
    held = [((r["manifest"].get("meta") or {}).get("run_id") or "", r["id"])
            for r in json.loads(data("select", "--kind", PROPERTY_KIND, "--where", f"meta.property={prop}", "--json") or "[]")
            if (r["manifest"].get("meta") or {}).get("ok") is True and r["manifest"]["meta"].get("schema") == want.schema]
    return max(held)[1] if held else None


def unindexed(prop: str) -> bool:
    """Whether `research data select` of `prop`'s captures here read only this store's own catalog, with no index (no
    $STORE_PRIVATE_READER_URL and no `research data index` build), as in a `research run` job's custody store.  There, a property
    missing from the catalog says nothing about the store."""
    return selected_from("--kind", PROPERTY_KIND, "--where", f"meta.property={prop}") == "the catalog"


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


def entry(rec: dict[str, Any], art: str, tier: str) -> dict[str, Any]:
    """A published record's index entry: its art: id, the tier of its rebuild, what it was pinned from and its circuit's pins, the
    Build's manifest digest and every rank's Program digests as `epoch.build_programs` reads them from the Build."""
    b = rec["build"]
    per = b.get("per_rank") or {"0": {"programs": b.get("programs") or {}, "workload_digest": b.get("workload_digest")}}
    programs = {r: {"step": p["programs"].get("step") or p["programs"].get("build_step"), "workload": p.get("workload_digest"),
                    "requests": {k: v for k, v in sorted(p["programs"].items()) if k == "request" or k.startswith("build_request")}}
                for r, p in sorted(per.items(), key=lambda kv: int(kv[0]))}
    return {"record": art, "tier": tier, "pinned_from": {"record": art, "build": rec["evidence"]["build"], "tree": rec["tree"]},
            "manifest_digest": b.get("manifest_digest"), "programs": programs}


def _tier(row: str, tier: str | None) -> str:
    if (tier := tier or (index().get(row) or {}).get("tier")) not in CR.TIERS:
        raise UsageError(f"{row}: a row new to {INDEX.name} needs --tier {'|'.join(CR.TIERS)}, the tier of its rebuild "
                         f"(`verity-vllm neutrality`)")
    return tier


def publish(d: str | Path, row: str, *, evidence: dict[str, Any], tree: str, by: str, summary: dict[str, Any] | None = None,
            tier: str | None = None, newest: bool = False) -> str:
    """Write the row's config record from its row dir `d` (the stages' files, or a copy of them) and its Commit of record's
    `summary`, refuse it unless it is PASS, the Commit committed against the Build's manifest and Programs, its Build artifact
    holds every Program it names, and every conformance property its Programs need (`due`) is cited by a holding capture of
    record (with `newest`, one `evidence` leaves out cites `newest_capture`) or named a gap, then label it `row=<id>`, preserve it
    with its label and index it with its Programs as the row's pins (`tier`: its neutrality rebuild's, by default the index's).
    A switched row's (`config_record.aot_of`) are its AOT Build's predicted Programs: a Build that traced one is refused, and the
    row keeps its `aot`.  The row it indexes is a row of record, so a row file not done only at its public cap
    (`config_record.index_refusal`) is refused before anything is written.  Returns its art: id."""
    if not TREE.fullmatch(tree):
        raise UsageError(f"--tree {tree!r}: REF@SHA, the code the row was built and served from")
    if why := CR.index_refusal(row):
        raise UsageError(f"{why}; publish would index it as one, so it publishes nothing")
    tier = _tier(row, tier)
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
    traced = sorted(f"rank{r} {p}" for r, ps in constructions(Path(d)).items() for p, c in ps.items() if c != "aot")
    if (aot := CR.aot_of(row, INDEX)) and traced:
        raise UsageError(f"{row}: a switched row's pins are its AOT Build's Programs (`aot` in {INDEX.name}), and this Build traced "
                         f"{len(traced)} of them ({', '.join(traced[:4])}): run its Build with --build-construction record")
    rec["conformance"] = due(evidence["build"], listed)
    cited, gaps = evidence.get("conformance") or {}, evidence.get("gaps") or {}
    if newest:
        left = [p for p in rec["conformance"]["due"] if p not in cited and p not in gaps]
        found = {p: a for p in left if (a := newest_capture(p))}
        if (missed := [p for p in left if p not in found]) and unindexed(missed[0]):
            raise UsageError(f"{row}: --newest-captures found no capture of {', '.join(missed)} in this store's catalog, and there is "
                             f"no store index here ($STORE_PRIVATE_READER_URL unset, no `research data index` build; a `research run` job "
                             f"has none), so the store may hold one: resolve each where the index is (the launcher: `research data select "
                             f"--kind {PROPERTY_KIND} --where meta.property=PROPERTY`) and pass it as --conformance PROPERTY=art:…")
        cited = dict(sorted({**found, **cited}.items()))
        rec["evidence"] = {**evidence, "conformance": cited}
    if silent := sorted(set(rec["conformance"]["due"]) - set(cited) - set(gaps)):
        raise UsageError(f"{row}: its Programs need {', '.join(silent)} ({rec['conformance']['map']}), which the record neither cites nor "
                         f"names a gap: --conformance PROPERTY=art:… (`verity-vllm records capture RUN`) or --gap PROPERTY=REASON")
    for prop, art in cited.items():
        check_capture(prop, art)
    refs = {**{k: evidence[k] for k in REFS if evidence.get(k)}, **cited}
    meta = {"row": row, "outcome": rec["outcome"], "world": rec["config"].get("tp"), "tree": tree, "tier": tier}
    with tempfile.TemporaryDirectory(prefix="config-record-") as tmp:
        p = Path(tmp) / CR.NAME
        p.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
        art = ART.findall(data("put", "--kind", CR.KIND, "--meta", json.dumps(meta, sort_keys=True), "--file", str(p),
                               *(f"--ref={k}={v}" for k, v in refs.items())))[-1]
    data("label", art, "row", row, "--by", by)
    data("push", art)
    doc = CR.load_index(INDEX)
    doc["rows"][row] = {**entry(rec, art, tier), **({"aot": aot} if aot else {})}
    CR.write_index(doc, INDEX)
    return art


def pull(art: str, tier: str | None = None) -> str:
    """Index `art`, a record `publish` preserved from another tree (a pod's `row chain`), in this tree's records.json, with the tier
    it was published with (`tier` overrides it), a switched row keeping its `aot`; returns its row.  A row whose file in this tree is
    not done only at its public cap (`config_record.index_refusal`) is refused before the index is touched."""
    m = json.loads(data("show", art, "--json"))["manifest"]
    meta = m.get("meta") or {}
    if m.get("kind") != CR.KIND or meta.get("outcome") != "PASS":
        raise UsageError(f"{art}: a {m.get('kind')} ({meta.get('outcome')}), not a PASS {CR.KIND}")
    rec = json.loads(fetch(art).read_text())
    row = rec["config"]["row"]
    if why := CR.index_refusal(row):
        raise UsageError(f"{why}; pull would index it as one, so it indexes nothing")
    doc = CR.load_index(INDEX)
    aot = (doc["rows"].get(row) or {}).get("aot")
    doc["rows"][row] = {**entry(rec, art, _tier(row, tier or meta.get("tier"))), **({"aot": aot} if aot else {})}
    CR.write_index(doc, INDEX)
    return row


STAGED = "staged"
UTC = "%Y-%m-%dT%H:%M:%SZ"


def checkpoint(repo: str, revision: str) -> str:
    """A checkpoint's label target in the store."""
    return f"ckpt:{repo}@{revision}"


def staged(repo: str, revision: str, *, at: str, ref: str, by: str) -> str:
    """Label `repo`@`revision` staged at `at` (UTC, `UTC`'s form), `ref` the evidence."""
    try:
        datetime.strptime(at, UTC)
    except ValueError:
        raise UsageError(f"--at {at!r}: a UTC time, YYYY-MM-DDTHH:MM:SSZ") from None
    return data("label", checkpoint(repo, revision), STAGED, at, "--by", by, "--ref", ref)


def constructions(d: Path) -> dict[str, dict[str, str]]:
    """Per rank of a row dir's Build (`build_summary.json`), each derived Program's construction: `aot` or `trace`."""
    s = json.loads((d / "build_summary.json").read_text()) if (d / "build_summary.json").is_file() else {}
    return {k.removeprefix("rank"): {p: v.get("construction", "trace") for p, v in rk.items() if p.startswith("build_")}
            for k, rk in (s.get("ranks") or {}).items()}


def switch(row: str, comparison: dict[str, Any], *, run: str, trace: str, tier: str | None = None) -> dict[str, Any]:
    """Switch `row` to the AOT Build: its index entry's `aot` names the comparison `run` (`verity-vllm predict compare`, its
    `comparison.json`) and the traced Build `trace` it compared, and its pins are the comparison's Programs.  Refused unless the
    comparison is of `row` and equal on every rank, or where a row already pinned would move: re-pinning is `verity-vllm neutrality
    --update`'s, or where the row's file is not done only at its public cap (`config_record.index_refusal`).  A row new to the index
    is pinned with no record, at `tier`."""
    if not RUN.fullmatch(run) or not ART.fullmatch(trace):
        raise UsageError(f"--run {run!r} --trace {trace!r}: the comparison's run id and the traced Build's art: id")
    if RowId.parse(row).world < 2:
        raise UsageError(f"{row}: the AOT Build is a TP row's (`row_tp.TpRow.build`); a single-GPU row's Build traces")
    if comparison.get("row") != row or comparison.get("equal") is not True:
        raise UsageError(f"{row}: the comparison is of {comparison.get('row')} and {'equal' if comparison.get('equal') else 'NOT equal'}: "
                         f"a row is switched only where its predicted Programs equal its traced Build's on every rank")
    if why := CR.index_refusal(row):
        raise UsageError(f"{why}; switch would index it as one, so it switches nothing")
    parts = lambda p: {n: v["predicted"] for n, v in p.items() if isinstance(v, dict) and "predicted" in v}   # noqa: E731
    programs = {r: {"step": ps.pop("step"), "workload": ps.pop("workload"), "requests": ps}
                for r, ps in ((r, parts(p)) for r, p in sorted(comparison["ranks"].items(), key=lambda kv: int(kv[0])))}
    doc = CR.load_index(INDEX)
    cur = doc["rows"].get(row)
    if cur is not None and cur.get("programs") != programs:
        raise UsageError(f"{row}: its pins are not the comparison's Programs: re-pin it first (`verity-vllm neutrality {row} --update`)")
    if cur is None and tier not in CR.TIERS:
        raise UsageError(f"{row}: a row new to {INDEX.name} needs --tier {'|'.join(CR.TIERS)}, the tier of its rebuild")
    doc["rows"][row] = {**(cur or {"record": None, "tier": tier, "pinned_from": {"build": trace, "run": run}, "manifest_digest": None,
                                     "programs": programs}), "aot": {"comparison": run, "trace": trace}}
    CR.write_index(doc, INDEX)
    return doc["rows"][row]


@dataclass(frozen=True, kw_only=True)
class SwitchOptions:
    """Switch a row to the AOT Build from a comparison equal on every rank (`verity-vllm predict compare`'s comparison.json)."""

    row: str = option("row")
    comparison: str = option("--comparison", help="the comparison run's comparison.json")
    run: str = option("--run", help="the comparison's run id")
    trace: str = option("--trace", help="the traced Build it compared, a vllm-build/v1 art: id holding every rank's derive dirs")
    tier: str | None = option("--tier", choices=CR.TIERS, default=None, help="a row new to the index: the tier of its rebuild")
    cmd: str = field(default="switch", init=False)


@dataclass(frozen=True, kw_only=True)
class PublishOptions:
    """Write a row's config record, preserve it labelled row=<id> and index it with its Programs' pins in manifests/records.json."""

    row_dir: str = option("row_dir", help="the row dir, or a copy of the stages' files the record reads")
    row: str = option("row")
    build: str = option("--build", help="the vllm-build/v1 holding every rank's Program descriptors")
    tree: str = option("--tree", help="the code the row was built and served from, REF@SHA")
    by: str = option("--by", help="the label's author (a lane)")
    tier: str | None = option("--tier", choices=CR.TIERS, default=None,
                              help="the tier of the row's neutrality rebuild (`verity-vllm neutrality`); default the index's")
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
    newest_captures: bool = option("--newest-captures", action="store_true", default=False,
                                   help="cite, for each property the Programs need that --conformance and --gap leave out, its newest "
                                        "holding capture of record in the store (where the store has no index, as in a job, one its "
                                        "catalog lacks is refused)")
    cmd: str = field(default="publish", init=False)


@dataclass(frozen=True, kw_only=True)
class PullOptions:
    """Index records published from another tree (a pod's `row chain`) in this tree's manifests/records.json."""

    art: list[str] = option("art", nargs="+", help="the records' art: ids")
    tier: str | None = option("--tier", choices=CR.TIERS, default=None, help="the tier of the rows' rebuild (default the record's)")
    cmd: str = field(default="pull", init=False)


@dataclass(frozen=True, kw_only=True)
class CaptureOptions:
    """Preserve a capture run's property records as property-record/v1 artifacts; print PROPERTY=art:… for --conformance."""

    run: str = option("run", help="the capture run (a run id with a run record)")
    property: list[str] = option("--property", action="append", default=[], help="these properties only (default every one it wrote)")
    cmd: str = field(default="capture", init=False)


@dataclass(frozen=True, kw_only=True)
class ShowOptions:
    """Print the rows of record and their records' art: ids, or a row's config record."""

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


@dataclass(frozen=True, kw_only=True)
class StagedOptions:
    """Record that a host finished staging a checkpoint's weights, every shard's sha256 checked: the label staged on ckpt:<repo>@<revision>."""

    repo: str = option("repo", help="the checkpoint's HF repo id")
    revision: str = option("revision")
    at: str = option("--at", help="when the shards' check passed, UTC (YYYY-MM-DDTHH:MM:SSZ)")
    ref: str = option("--ref", help="the evidence: the staging run, or host:path of its verify record")
    by: str = option("--by", help="the label's author (a lane)")
    cmd: str = field(default="staged", init=False)


COMMANDS = {"publish": PublishOptions, "pull": PullOptions, "capture": CaptureOptions, "show": ShowOptions, "programs": ProgramsOptions, "fetch": FetchOptions,
            "staged": StagedOptions, "switch": SwitchOptions}


def _pairs(flag: str, items: list[str], form: str) -> dict[str, str]:
    out = dict(c.split("=", 1) for c in items if "=" in c)
    if len(out) != len(items):
        raise UsageError(f"{flag} {items}: {form}, each property once")
    return out


def main(a: PublishOptions | PullOptions | CaptureOptions | ShowOptions | ProgramsOptions | FetchOptions | StagedOptions
         | SwitchOptions) -> int:
    if a.cmd == "switch":
        print(json.dumps(switch(a.row, json.loads(Path(a.comparison).read_text()), run=a.run, trace=a.trace, tier=a.tier)["aot"]))
        return 0
    if a.cmd == "publish":
        ev = evidence_of(a.build, manifest=a.manifest, match=a.match, match_run=a.match_run, commit=a.commit, commit_run=a.commit_run,
                         conformance=_pairs("--conformance", a.conformance, "PROPERTY=art:..."),
                         gaps=_pairs("--gap", a.gap, "PROPERTY=REASON"))
        summary = json.loads(Path(a.summary).read_text()) if a.summary else None
        print(publish(a.row_dir, a.row, evidence=ev, tree=a.tree, by=a.by, summary=summary, tier=a.tier, newest=a.newest_captures))
        return 0
    if a.cmd == "pull":
        for art in a.art:
            print(f"{pull(art, a.tier)} {art}")
        return 0
    if a.cmd == "capture":
        for prop, art in capture(a.run, a.property or None).items():
            print(f"{prop}={art}")
        return 0
    if a.cmd == "show":
        print(json.dumps(record(a.row) if a.row else {r: e.get("record") for r, e in index().items()}, indent=1, sort_keys=True))
        return 0
    if a.cmd == "fetch":
        print(bundle(a.row, Path(a.to), a.rank or None, part=a.part or None))
        return 0
    if a.cmd == "staged":
        print(staged(a.repo, a.revision, at=a.at, ref=a.ref, by=a.by), end="")
        return 0
    src = Path(a.bundle) if a.bundle else None
    named = parts(record(a.row, src))
    for r in a.rank or sorted(named):
        for part, p in programs(a.row, [r], part=a.part or None, bundle=src)[r].items():
            print(f"rank {r}  {part:28s} {named[r][part][:16]}  gates {p.gates}  definitions {len(p.defs)}", flush=True)
    return 0
