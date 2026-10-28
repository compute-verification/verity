"""Replay every recorded one-stage run through sampled proofs' Lean protocol and the current Python, against the verdict the
run recorded.

    python benchmarks/one_stage/replay.py --index INDEX.json --runs DIR [--fetch] --lean EXE --out DIR [--only RUN ...]

* **Index.** `one-stage-run/v0` rows (the store's selection of every run that holds one-stage registration or audit
  records): each run's art, the registration formats it holds and the Python verdicts it recorded.
* **Case.** For each recorded audit, the case `sampled-proofs-one-stage serve` reads: the verifier's parameters as
  `OneStage.paramsOfJson` reads them (its own program descriptor, query, law, scheme, ports, leaf layer, log, partition,
  delta), the registration record as `commitmentOf` reads it, and the evidence as `evidenceOf` reads it (the draw, the
  proved units, the instance count, the receipt, the statement, the verdicts, the sessions and their roots, the served
  draw), each from the run's own files or the verifier's own copies (the program rebuilt from the input set the run
  names, at the population program's version the record holds, v1 or v0; the domains derived again), and how each was
  found (`provenance`). Floats in verdicts and statements, which neither side reads, are written as strings. The
  registered roots the record's `registered` check reads are the verdicts' own, for the registration's own values
  (`audit.session_registered(verdicts, registration)`), on both sides.
* **The public file.** A0's check of the verifier's own public file against the record (`matches_public`: the scheme,
  the roots, their domains and the instance count in the header of `stage/pub-<n>.bin`), the honest one and each
  recorded refusal, by the verifier of record's Lean (`one_stage.verifier.lean`) and the current Python.
* **Negatives.** Each recorded refusal that is one of the protocol's checks (R1-R7, the draw's, the record's: proved,
  receipt, served, roots, registered; the verdicts'; the public file's) becomes the base case with the one change the
  run made; a refusal by a check the protocol does not make (`flock-verify`'s U1-U3 and roots, M0's selftests and
  bindings, a tampered set before the verifier of record) is listed as outside, with the reason, and where the protocol
  has a check of the same kind its analog is replayed and counted apart.
* **Verdicts.** Lean (`serve`), the current Python (`registration.check`, `draw.check_draw`, `audit.audit_record`, written
  as `OneStage.check` writes its outcome) and the recorded verdict, compared by verdict, stage (registration, draw,
  checks, verdict; public) and codes.
* **The whole record.** Where the registration passes and the draw is an object, the audit record itself by the
  verifier of record (`one_stage.verifier.audit_record`: the Lean's `record` beside Python's `audit_record`, equal under
  canonical JSON), on the case's evidence with its floats; whether they agree, and the fields where they don't.
* **No recorded verdict.** A run that recorded no verdict (a serving bundle no verifier audited in its own run, an A2
  or A4 run that stopped before its audit record) compares Lean with the current Python on the stages it holds: the whole
  audit where it kept the session, else the registration alone (its `stages`; each outcome read up to them, one that
  got past them a pass). Its own line in the counts (`no_recorded_verdict`); a run that holds nothing either side reads
  is listed with why.

Writes `replay.json` (every case's three verdicts and provenance, the runs neither reads or that miss an input, the
counts by driver and format, and the summary's lines) and `cases/<run>.jsonl` (the cases themselves) under `--out`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator

import verity_difftest as DT
from verity_difftest import protocol as PR
from verity.core.protocols.sampled_proofs.one_stage import audit, draw as DR, partition as P, registration as R, verifier as V
from verity.ml.circuits import codec, partition_object as PO
from verity.ml.commitments.identity import identity_digest, identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.definitions.registry import catalog

import r2_population as R2

SCHEME = "frame-v3-sha512/hm96-sha512"
ROW = "hm96-sha512/row/v1"
#: M0's set-bound frame domains: each tag with its identity digest (`a0.STAND_IN_BINDINGS`)
STAND_IN_BINDINGS = {"verity/flock-circuit/binding/v2": identity_digest_sha512,
                     "verity/flock-ir-frame/binding/v1": lambda tag, doc: bytes.fromhex(identity_digest(tag, doc))}
SERVED_BINDINGS = {"sha512": identity_digest_sha512, "v1": lambda tag, doc: bytes.fromhex(identity_digest(tag, doc))}
#: `registration.WINDOW_DOMAIN`, the current one and the one before it
SERVED_TAGS = (R.WINDOW_DOMAIN, "verity/one-stage/served-domain/v0")
STAGES = ("registration", "draw", "checks", "verdict")
REGISTRATION = ("registration",)
CHECKS = ("proved", "receipt", "served", "roots", "registered")
#: the evidence of a case that holds the registration alone: no draw, so both sides stop at the draw's check
NO_DRAW = {"draw": None, "instances": 0, "proved_units": [], "receipt": {}, "served": None, "session_roots": None,
           "sessions": [], "statement": {}, "verdicts": {}}


class Missing(Exception):
    """A run lacks a file the replay needs (the message names it)."""


class Gap(Exception):
    """The run's files hold the input but the replay cannot rebuild the verifier's copy of it (the message says which)."""


# --- programs ----------------------------------------------------------------------------------------------------------

def _json(value: Any) -> Any:
    return json.loads(DT.canonical(value))


def descriptor(program) -> dict:
    return program.desc if isinstance(program, codec.DecodedProgram) else _json(codec.encode_program(program))


def own_population(build: Callable[[int], Any], want: str) -> Any:
    """`build(version)`, the verifier's own population program, at the version whose program is the record's (`want`): v1,
    the builders' default, or v0, which every run before v1 recorded (`P.POPULATION_VERSIONS`); v1's when neither is."""
    first = None
    for version in P.POPULATION_VERSIONS:
        program = build(version)
        if PO.program_sha512(program) == want:
            return program
        first = program if first is None else first
    return first


def population_version(program) -> int:
    """The version of a population program (`TemplateInstances_vN{…}`, `TemplatePopulations_vN{…}`, built or a stored
    descriptor decoded), by its program function's id; v1 for another program."""
    fid = program.program_fn_id if isinstance(program, codec.DecodedProgram) else codec.descriptor_id(program.fn)
    m = re.match(r"Template(?:Instances|Populations)_v([0-9]+)\{", fid)
    return int(m[1]) if m else P.POPULATION_VERSIONS[0]


def decode(desc: dict):
    from verity.ml.circuits.defs import REGISTRY
    return codec.decode_program(desc, REGISTRY)


def decode_opaque(desc: dict) -> tuple[Any, list[str]]:
    """`decode`, each primitive the descriptor names that the current registry lacks standing in by the descriptor's own
    signature, never evaluated (the registration's checks and the draw read the program's shape only); and their ids."""
    from verity.ml.circuits.defs import REGISTRY, PrimitiveDefinition
    from verity.ml.circuits.types import type_from_json

    def opaque(*_: Any) -> int:
        raise RuntimeError("an opaque stand-in is never evaluated")
    stubs: dict[str, Any] = {}
    for fid, d in desc["definitions"].items():
        if d.get("form") != "primitive":
            continue
        try:
            codec.lookup_primitive(REGISTRY, fid)
        except KeyError:
            name, _, ver = fid.rpartition("_v")
            stubs[fid] = PrimitiveDefinition(name, int(ver), tuple((n, type_from_json(t)) for n, t in d["params"]),
                                             type_from_json(d["ret"]), opaque, register=False)

    class Registry:
        def __contains__(self, fid: str) -> bool:
            return fid in REGISTRY or fid in stubs

        def __getitem__(self, fid: str) -> Any:
            return REGISTRY[fid] if fid in REGISTRY else stubs[fid]
    return codec.decode_program(desc, Registry()), sorted(stubs)


class Programs:
    """Descriptors by SHA-512, from every `program.json` the runs stored."""

    def __init__(self) -> None:
        self.by_sha: dict[str, dict] = {}

    def add_file(self, path: Path) -> None:
        try:
            desc = json.loads(path.read_text())
            self.by_sha.setdefault(PO.program_sha512(desc), desc)
        except (ValueError, KeyError, TypeError):
            pass

    def get(self, sha: str):
        return decode(self.by_sha[sha]) if sha in self.by_sha else None


# --- the input sets ----------------------------------------------------------------------------------------------------

class Sets:
    """Input-set manifests by the SHA-256 of the `set.tar` a run was sent, each set under its directory name."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def dir(self, tar_sha: str) -> Path:
        return self.root / tar_sha[:16]

    def extract(self, tar: Path, tar_sha: str) -> None:
        out = self.dir(tar_sha)
        if out.exists():
            return
        out.mkdir(parents=True)
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if m.isfile() and Path(m.name).name == "manifest.json":
                    target = out / Path(m.name).parent.name / "manifest.json"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(t.extractfile(m).read())

    def has(self, tar_sha: str, name: str) -> bool:
        return (self.dir(tar_sha) / name / "manifest.json").exists()

    def holding(self, name: str) -> str | None:
        """The extracted tar (its directory name) with the fewest sets that holds set `name`."""
        tars = sorted((len(list(d.iterdir())), d.name) for d in self.root.iterdir() if (d / name / "manifest.json").exists()) \
            if self.root.exists() else []
        return tars[0][1] if tars else None

    def open(self, tar_sha: str, name: str):
        from verity_numerical.bench.input_sets import InputSet
        if not self.has(tar_sha, name):
            raise Missing(f"inputs/set.tar ({tar_sha[:16]}): {name}/manifest.json")
        return InputSet.open(str(self.dir(tar_sha) / name))

    def starting(self, tar_sha: str, prefix: str) -> str | None:
        """The first set (by directory name) in the tar whose name starts with `prefix`, as A4 picks a layout member's."""
        d = self.dir(tar_sha)
        return next((x.name for x in sorted(d.iterdir()) if x.name.startswith(prefix)), None) if d.exists() else None


class Records:
    """Registration records by digest, from every `audit.json` and `registration.json` the runs stored."""

    def __init__(self) -> None:
        self.by_digest: dict[str, tuple[dict, str]] = {}

    def add_file(self, path: Path, root: Path) -> None:
        try:
            body = json.loads(path.read_text())
            rec = body.get("registration", body) if path.name == "audit.json" else body
            self.by_digest.setdefault(R.digest(rec), (rec, str(path.relative_to(root))))
        except (ValueError, KeyError, TypeError, AttributeError):
            pass


# --- a run -------------------------------------------------------------------------------------------------------------

@dataclass
class Run:
    row: dict
    dir: Path
    files: list[dict]
    sets: Sets
    programs: Programs
    outside: list[dict] = field(default_factory=list)
    fetch: Callable[[str], None] | None = None
    records: Records = field(default_factory=Records)
    #: the replayed audit cases of the runs that recorded a verdict, by their record's digest
    audits: dict[str, list] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.row["art"][4:12]

    def path(self, rel: str) -> Path:
        """The run's file `rel`, fetched now when the art holds it and the replay fetches."""
        p = self.dir / rel
        if not p.exists() and self.fetch is not None and any(f["path"] == rel for f in self.files):
            self.fetch(rel)
        if not p.exists():
            raise Missing(rel)
        return p

    def json(self, rel: str) -> Any:
        return json.loads(self.path(rel).read_text())

    def has(self, rel: str) -> bool:
        return (self.dir / rel).exists() or any(f["path"] == rel for f in self.files)

    def file_sha256(self, rel: str) -> str | None:
        return next((f["sha256"] for f in self.files if f["path"] == rel), None)

    def commit(self) -> str | None:
        if self.has("commit"):
            return self.path("commit").read_text().strip()
        if not self.has("job.json"):
            return None
        j = self.json("job.json")
        return (j.get("computation") or {}).get("tree_sha") or ((j.get("store") or {}).get("conditions") or {}).get("source.commit")

    def input_set(self, name: str):
        """The run's own input set `name` (from the `set.tar` it was sent), else the one in the smallest other run's tar
        that holds it, with provenance. Only the set's subcircuit and name are read."""
        own = self.file_sha256("inputs/set.tar")
        if own is not None and self.sets.has(own, name):
            return self.sets.open(own, name), "the run's inputs/set.tar"
        other = self.sets.holding(name)
        if other is None:
            raise Missing(f"inputs/set.tar: {name}")
        why = "this run's does not hold it" if own else "this run's was not uploaded"
        return self.sets.open(other, name), f"inputs/set.tar {other} of another run ({why})"

    def pod_args(self) -> dict[str, str]:
        """The `KEY=VALUE` arguments the run passed `pod.sh`."""
        if not self.has("launch.json"):
            return {}
        argv = self.json("launch.json").get("argv") or []
        tail = argv[argv.index("--") + 1:] if "--" in argv else []
        return dict(x.split("=", 1) for x in tail if "=" in x and not x.startswith("-"))


def live_line(path: Path) -> dict | None:
    for line in path.read_text().splitlines():
        if line.startswith("LIVE\t"):
            return json.loads(line[5:])
    return None


def archive_keys(path: Path) -> dict[str, str]:
    """`flock-verify archive-put`'s stdout: the SHA-512 content key of each file it put, by the path it was put from."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{128}", parts[0]):
            out[parts[1]] = parts[0]
    return out


def sha512_file(path: Path) -> str:
    return hashlib.sha512(path.read_bytes()).hexdigest()


def public_digest(run: Run, pub: str, archive_log: str) -> tuple[str, str]:
    """The SHA-512 of the verifier's public file `pub`: `archive-put`'s key for it, else the file's own."""
    if run.has(archive_log):
        key = next((k for path, k in archive_keys(run.path(archive_log)).items() if path.endswith("/" + pub)), None)
        if key is not None:
            return key, f"archive-put's key of {pub} ({archive_log})"
    return sha512_file(run.path(pub)), f"SHA-512 of {pub}"


def stage_report(run: Run, log: str) -> dict:
    """`stage_subset.py`'s report, the last JSON line of its stdout in `log`: the staged selection's name and content
    digest, which M0's frame domains bind."""
    out = run.path(log).read_text().split("--- stdout", 1)[-1].split("--- stderr", 1)[0]
    lines = [x for x in out.splitlines() if x.startswith("{")]
    if not lines:
        raise Missing(f"{log}: stage_subset's report")
    return json.loads(lines[-1])


def tar_member(run: Run, tar: str, name: str) -> bytes:
    with tarfile.open(run.path(tar)) as t:
        m = next((m for m in t.getmembers() if m.isfile() and Path(m.name).name == name), None)
        if m is None:
            raise Missing(f"{tar}: {name}")
        return t.extractfile(m).read()


def flipped(body: bytes) -> bytes:
    """The public file with the first byte of its body (the first `b || c`) flipped, as A2's negative made it."""
    nl = body.index(b"\n") + 1
    return body[:nl] + bytes([body[nl] ^ 1]) + body[nl + 1:]


# --- the verifier's copies ---------------------------------------------------------------------------------------------

def partition_form(program, query: dict, rec: dict, era_templates: list[str]) -> tuple[dict | None, str]:
    """The verifier's partition object for `rec`: the canonical one over `program` and `query` (null); the canonical
    envelope over the era's template ids (Definition ids, before descriptor ids); or the pre-canonical `{program, query}`
    with the era's query form (`Q_template_instance(s)` v0 with `template(s)` and a `width_rule` at the top level)."""
    sha = PO.program_sha512(program)
    if P.digest(P.partition(program, query)) == rec["partition"]:
        return None, "canonical"
    lists = (list(era_templates), sorted(era_templates))
    envelope = [{"format": PO.FORMAT, "program": sha, "query": q} for q in
                [{"name": "Q_template_instance", "version": 0, "params": {"template": t}} for t in era_templates]
                + [{"name": "Q_template_instances", "version": 0, "params": {"templates": ts}} for ts in lists]]
    old = [{"program": sha, "query": q} for q in
           [{"name": "Q_template_instance", "template": t, "version": 0, "width_rule": None} for t in era_templates]
           + [{"name": "Q_template_instances", "templates": ts, "version": 0, "width_rule": None} for ts in lists]]
    for part, how in [*((p, "canonical envelope") for p in envelope), *((p, "pre-canonical") for p in old)]:
        if P.digest(part) == rec["partition"] and R.query_digest(part["query"]) == rec["query"]:
            return part, f"{how} {part['query']['name']} over {era_templates}"
    raise Gap(f"no partition form over {era_templates} gives the record's partition {rec['partition'][:16]}")


def with_program(partition: dict | None, program, drop_template: bool = False) -> dict | None:
    """The era's partition object over another program (a negative's), its last template dropped if `drop_template`."""
    if partition is None:
        return None
    q = dict(partition["query"])
    if drop_template and "templates" in q:
        q["templates"] = q["templates"][:-1]
    elif drop_template and "templates" in q.get("params", {}):
        q["params"] = {**q["params"], "templates": q["params"]["templates"][:-1]}
    return {**partition, "program": PO.program_sha512(program), "query": q}


def _domain(binding: bytes, leaves: int) -> str:
    return CommitmentDomain(binding, -1, RangeIndexedDomain(0, int(leaves)), hash="sha512").domain_id.hex()


def stand_in_layouts(s, n: int, prefix: str = "") -> dict[str, dict[str, tuple[str, int]]]:
    """The set's ports as M0 lays them out, before and after hidden outputs: `{layout: {port: (schema, leaves)}}`."""
    sc = s.subcircuit
    rows = {f"{prefix}{p.name}": (ROW, n) for p in sc.inputs}
    return {"words": {**rows, **{f"{prefix}{p.name}": ("u16", n * p.words) for p in sc.outputs}},
            "hidden": {**rows, **{f"{prefix}{p.name}": (ROW, n) for p in sc.outputs}}}


def stand_in_ports(name: str, content_digest: str, n: int, layout: dict[str, tuple[str, int]], tag: str, content: bool,
                   prefix: str = "") -> dict:
    """M0's frame domains bound to the staged set `name` (with its `content_digest`, before hidden outputs)."""
    out = {}
    for p, (schema, leaves) in layout.items():
        doc = {"set": name, **({"content_digest": content_digest} if content else {}), "lo": 0, "hi": n,
               "port": p[len(prefix):], "schema": schema}
        out[p] = {"domain": _domain(STAND_IN_BINDINGS[tag](tag, doc), leaves), "leaves": leaves}
    return out


def served_domain(form: tuple[str, str], program: str, partition: str, port: str, schema: str, leaves: int, run: str) -> str:
    """`registration.window_domain` under the era's tag and identity digest (`v1`, before 64-byte bindings, or `sha512`)."""
    tag, binding = form
    doc = {"program": program, "partition": partition, "port": port, "schema": schema, "leaves": int(leaves), "run": run}
    return _domain(SERVED_BINDINGS[binding](tag, doc), leaves)


SERVED_FORMS = [(tag, b) for tag in SERVED_TAGS for b in SERVED_BINDINGS]


def served_ports(form: tuple[str, str], part: dict, layout: dict[str, tuple[str, int]], run: str) -> dict:
    return {p: {"domain": served_domain(form, part["program"], P.digest(part), p, schema, leaves, run), "leaves": leaves}
            for p, (schema, leaves) in layout.items()}


def choose_served(part: dict, layout: dict[str, tuple[str, int]], run: str, rec: dict) -> tuple[dict, str, tuple[str, str]]:
    """The served domains under the era's form that the record's match (R3 passed in the run), else the current ones (a gap)."""
    for form in SERVED_FORMS:
        ports = served_ports(form, part, layout, run)
        if matches(ports, rec):
            return ports, f"served_domain {form[0]} {form[1]}", form
    return served_ports(SERVED_FORMS[0], part, layout, run), "GAP: no served domain form matches the record's; today's", \
        SERVED_FORMS[0]


def matches(ports: dict, rec: dict, only: Callable[[str], bool] = lambda p: True) -> bool:
    got = {r["port"]: {"domain": r["domain"], "leaves": r["leaves"]} for r in rec["roots"] if only(r["port"])}
    return got == ports


def choose(variants: Iterator[tuple[str, dict]], rec: dict, today: str, only: Callable[[str], bool] = lambda p: True
           ) -> tuple[dict, str]:
    """The era's derivation of the ports under which every one matches the record (R3 passed in the run); else the current one,
    named as a reconstruction gap."""
    seen = {}
    for name, ports in variants:
        seen[name] = ports
        if matches(ports, rec, only):
            return ports, name
    return seen[today], f"GAP: no derivation matches the record's ports; today's ({today})"


# --- cases -------------------------------------------------------------------------------------------------------------

def strings(x: Any) -> Any:
    """Floats, which neither side reads, as strings (canonical JSON has no floats)."""
    if isinstance(x, float):
        return repr(x)
    if isinstance(x, dict):
        return {k: strings(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [strings(v) for v in x]
    return x


def delta_of(x: float | str) -> str:
    f = Fraction(x) if isinstance(x, str) else Fraction(float(x))
    return str(f)


def recorded(verdict: str, stage: str | None = None, codes: list[str] | None = None, source: str = "") -> dict:
    return {"verdict": verdict, "stage": stage, "codes": sorted(codes) if codes is not None else None, "source": source}


@dataclass
class Case:
    run: str
    driver: str
    format: str
    name: str
    kind: str                 # audit | negative | analog
    program: Any
    params: dict
    rec: dict
    evidence: dict
    recorded: dict | None     # None: the run recorded no verdict
    provenance: dict = field(default_factory=dict)
    r7: bool = False
    stages: tuple[str, ...] = STAGES   # the stages of `OneStage.check` the run's files hold
    #: `matches_public`'s other arguments (`scheme`, `roots`, `domains`, `population`): a check of the public file, not
    #: `OneStage.check`
    public: dict | None = None

    def case(self) -> dict:
        if self.public is not None:
            return {"fn": "matches_public", "input": {"record": self.rec, **self.public}}
        return {"commitment": DT.canonical(self.rec).encode().hex(),
                "evidence": DT.canonical(strings(self.evidence)).encode().hex(), "params": _json(self.params)}

    def but(self, name: str, kind: str, recorded_: dict | None, *, params: dict | None = None, rec: dict | None = None,
            evidence: dict | None = None, program: Any = None, provenance: dict | None = None,
            public: dict | None = None) -> Case:
        return Case(self.run, self.driver, self.format, name, kind, program if program is not None else self.program,
                    {**self.params, **(params or {})}, rec if rec is not None else self.rec,
                    {**self.evidence, **(evidence or {})}, recorded_, {**self.provenance, **(provenance or {})}, self.r7,
                    self.stages, {**(self.public or {}), **public} if public is not None else self.public)


def params(program, query: dict, law: dict, ports: dict, leaf_layer: str | None, delta: str, *, partition: dict | None = None,
           log: list | None = None, anchors: list | None = None, work_table: dict | None = None,
           registered: dict | None = None, scheme: str = SCHEME) -> dict:
    return {"anchors": list(anchors or []), "delta": delta, "law": law, "leaf_layer": leaf_layer, "log": list(log or []),
            "partition": partition, "ports": ports, "program": descriptor(program), "query": query,
            "registered": registered, "scheme": scheme, "work_table": work_table}


def evidence(a: dict, *, instances: int, sessions: list[str], session_roots: dict | None, served: list[int] | None) -> dict:
    """The evidence of a recorded audit record `a` (what `audit_record` was handed)."""
    return {"draw": a["draw"], "instances": instances, "proved_units": a["proved_units"], "receipt": a["receipt"],
            "served": served, "session_roots": session_roots, "sessions": sessions, "statement": a["statement"],
            "verdicts": a["verdicts"]}


def recorded_audit(a: dict, source: str) -> dict:
    """The verdict the run's Python recorded for its audit record."""
    if a.get("accepted"):
        return recorded("accept", source=source)
    checks = a.get("checks") or {}
    if not checks.get("draw", {}).get("ok", True):
        return recorded("reject", "draw", checks["draw"].get("codes"), source)
    failed = [k for k, c in checks.items() if not c.get("ok")]
    if failed:
        return recorded("reject", "checks", failed, source)
    vs = a.get("verdicts") or {}
    if any(v.get("accepted") is not None and v.get("accepted") is not True for v in vs.values()):
        return recorded("reject", "verdict", source=source)
    return recorded("undecided", "verdict", source=source)


def recorded_codes(v: dict, stage: str, source: str) -> dict:
    return recorded("reject" if v.get("refused", True) else "accept", stage, v.get("codes"), source)


def recorded_failed(v: dict, source: str) -> dict:
    """An `audit_negatives` entry: the checks that failed, or none (the verdicts')."""
    if not v.get("refused"):
        return recorded("accept", source=source)
    return recorded("reject", "checks", v["failed"], source) if v["failed"] else recorded("undecided", "verdict", source=source)


# --- the two verdicts --------------------------------------------------------------------------------------------------

def public_outcome(v: dict) -> dict:
    """A `matches_public` verdict (`{"ok", "codes", "detail"}`) as an outcome: its stage `public`, the whole verdict kept."""
    if v["ok"]:
        return {"verdict": "accept", "check": v}
    return {"verdict": "reject", "reason": f"public: {', '.join(v['codes'])}", "check": v}


def lean_public(c: Case) -> dict:
    """The verifier of record's Lean (`sampled-proofs-verifier`) on a public-file case."""
    got = V.lean("matches_public", c.case()["input"])
    if "refused" in got:
        return {"verdict": "error", "reason": got["refused"]}
    return public_outcome(got)


def python(c: Case) -> dict:
    """Today's Python on the case, as `OneStage.check` writes its outcome (`test_difftest.check`, with Lean's reasons)."""
    if c.public is not None:
        x = c.public
        return public_outcome(R.matches_public(c.rec, x["scheme"], x["roots"], x["domains"], x["population"]).to_json())
    p, rec, e = c.params, c.rec, json.loads(DT.canonical(strings(c.evidence)))
    program = c.program
    digest = bytes.fromhex(R.digest(rec))
    delta = Fraction(p["delta"])
    out = lambda verdict, reason=None, public=None, profile=None: PR.outcome(  # noqa: E731
        R.PROTOCOL, verdict, reason, digest, public, profile)
    codes = R.check(rec, program, p["query"], p["law"], p["scheme"], p["ports"], p["leaf_layer"], p["anchors"], p["log"],
                    p["partition"], p["work_table"], p["registered"]).codes
    if codes:
        return out("reject", f"registration: {', '.join(codes)}")
    try:
        mine = DR.derive(program, p["query"], p["law"], p["work_table"])
    except (P.QueryRefused, ValueError) as err:
        return out("reject", f"registration: R1-law: {err}")
    dv = DR.check_draw(e["draw"], rec, mine)
    if not dv.ok:
        return out("reject", f"draw: {', '.join(dv.codes)}")
    a = audit.audit_record(rec, e["receipt"], e["draw"], mine, e["proved_units"], e["instances"], e["statement"],
                           e["verdicts"], delta, e["sessions"], e["served"], e["session_roots"],
                           audit.session_registered(e["verdicts"], rec))
    failed = [k for k in CHECKS if k in a["checks"] and not a["checks"][k]["ok"]]
    if failed:
        return out("reject", f"the audit's checks failed: {', '.join(failed)}")
    vs = sorted(e["verdicts"].items())
    if not a["accepted"]:
        bad = next((k for k, v in vs if v.get("accepted") is not None and v.get("accepted") is not True), None)
        if bad is not None:
            return out("reject", f"verdict {bad} did not accept")
        of_record = [(k, v) for k, v in vs if v.get("of_record")]
        if not of_record:
            return out("undecided", "no verifier of record")
        return out("undecided", f"the verifier of record {next(k for k, v in of_record if v.get('accepted') is None)} has not run")
    profile = audit.profile(rec, e["draw"], delta).to_json()
    wrong = a["wrong_units"]
    public = {"bound": wrong["bound"], "drawn": list(e["draw"]["units"]),
              "per_template": wrong.get("per_template"), "proved": DR.proved(e["draw"])}
    return out("accept", None, public, {**profile, "delta": str(profile["delta"])})


def whole_record(c: Case) -> dict | None:
    """The audit record by the verifier of record (`verifier.audit_record`, Lean beside Python), on the case's evidence
    with its floats, the law the verifier derives and the run's δ: whether the two records agree, each side's acceptance
    and failed checks, and where they disagree the fields and both sides' values. None where `OneStage.check` never
    reaches the record: a public-file case, a case that holds the registration alone, a registration R1-R7 refuse, a law
    that doesn't derive, a draw that isn't an object."""
    p, rec, e = c.params, c.rec, c.evidence
    if c.public is not None or c.stages != STAGES or not isinstance(e["draw"], dict):
        return None
    if R.check(rec, c.program, p["query"], p["law"], p["scheme"], p["ports"], p["leaf_layer"], p["anchors"], p["log"],
               p["partition"], p["work_table"], p["registered"]).codes:
        return None
    try:
        mine = DR.derive(c.program, p["query"], p["law"], p["work_table"])
    except (P.QueryRefused, ValueError):
        return None
    r = V.audit_record(rec, e["receipt"], e["draw"], mine, e["proved_units"], e["instances"], e["statement"], e["verdicts"],
                       float(Fraction(p["delta"])), e["sessions"], e["served"], e["session_roots"])
    v = r["verifiers"]
    out = {"agree": v["agree"], "lean": v["lean"], "python": v["python"]}
    if not v["agree"]:
        out["disagree"] = r["checks"]["verifiers"]["detail"]
    return out


def stage_of(outcome: dict) -> tuple[str | None, list[str] | None]:
    """The stage of `OneStage.check` an outcome's reason names, and its codes."""
    v, reason = outcome.get("verdict"), outcome.get("reason") or ""
    if v in ("accept", "pass"):
        return None, None
    if v == "undecided":
        return "verdict", None
    for prefix, stage in (("registration: ", "registration"), ("draw: ", "draw"), ("the audit's checks failed: ", "checks"),
                          ("public: ", "public")):
        if reason.startswith(prefix):
            rest = reason[len(prefix):]
            if stage == "registration" and rest.startswith("R1-law: "):
                return stage, ["R1-law"]
            return stage, sorted(rest.split(", "))
    if reason.startswith("verdict "):
        return "verdict", None
    return "decode", [reason]


def held(c: Case, outcome: dict) -> dict:
    """The outcome on the stages the case holds: one that got past them is a pass; a refusal at one of them, a decode
    failure or an error stays as it is."""
    if c.stages == STAGES:
        return outcome
    stage, _ = stage_of(outcome)
    if outcome.get("verdict") == "accept" or (stage in STAGES and stage not in c.stages):
        return {"verdict": "pass", "stages": list(c.stages)}
    return outcome


def against(c: Case, outcome: dict) -> str:
    """An outcome against the recorded verdict: the same verdict, stage and codes (`same`), the verdict only, or not."""
    outcome = held(c, outcome)
    stage, codes = stage_of(outcome)
    r = c.recorded
    if outcome.get("verdict") != r["verdict"]:
        return "differ"
    if c.kind == "analog" or (r["stage"] in (None, stage) and (r["codes"] is None or r["codes"] == codes)):
        return "same"
    return "same-verdict"


def compare(c: Case, lean: dict, py: dict) -> dict:
    hl, hp = held(c, lean), held(c, py)
    strip = lambda o: {k: v for k, v in o.items() if k != "reason"}  # noqa: E731
    lp = "same" if hl == hp else ("same-verdict" if strip(hl) == strip(hp) else "differ")
    stage, codes = stage_of(hl)
    out = {"lean_python": lp, "lean_stage": stage, "lean_codes": codes}
    if c.recorded is not None:
        out.update(lean_recorded=against(c, lean), python_recorded=against(c, py))
    return out


# --- drivers -----------------------------------------------------------------------------------------------------------

#: Negatives a run recorded that are not the protocol's checks, with why.
OUTSIDE = {
    "public-body-altered-leaf-layer-updated (Lean roots)": "flock-verify recomputing the roots from the public file",
    "statement-binds-another-partition": "M0's statement bindings (`a0.bindings`)",
    "m0-selftest-unit-draw-session": "M0's selftest",
    "prover-ignores-the-draw (M0 selftest)": "M0's selftest",
    "wrong-output-word": "the verifier of record (flock-verify, M0) on a tampered copy of the set, whose record the run did not keep",
    "lean-reads-another-root-of-W": "flock-verify's registered reads (`--registered`), opening each table row against the reads "
                                    "file's root; the protocol reads the roots the verdict opened against (replayed as "
                                    "registered-read-opened-against-another-root)",
    "lean-reads-moved-position": "flock-verify's registered reads (`--registered`): each unit's position against the verifier's "
                                 "read map from its own program; the protocol reads the verdict",
    "lean-weights-committed-afresh-read-as-registered": "flock-verify's registered reads (`--registered`) on a statement that "
                                                        "commits W under fresh salts, with no paths; the protocol reads the verdict",
}
#: The flock-verify draw checks a run recorded, each with the protocol's check of the same kind (an analog).
ANALOG = {
    "instances-not-the-draw (Lean U3)": "flock-verify U3; analog: the proved check on one instance more",
    "draw-dropped-from-record (Lean U1)": "flock-verify U1; analog: the draw check on a null draw",
}


def a0(run: Run) -> Iterator[Case]:
    """A0 (`a0.py`): one template's population program, a stand-in commitment under M0's set-bound domains."""
    base = "out/a0"
    a = run.json(f"{base}/audit.json")
    summary = run.json(f"{base}/summary.json")
    rec = a["registration"]
    n = summary["n"]
    s, set_from = run.input_set(rec["window"].get("input_set") or "rope101")
    template = s.subcircuit.definition()
    program = own_population(lambda v: P.population_program(template, n, version=v), rec["program"])
    prov: dict = {"commit": run.commit(), "set": set_from,
                  "program": f"P.population_program(set template, n, version={population_version(program)})"}
    if PO.program_sha512(program) != rec["program"]:
        stored = run.programs.get(rec["program"])
        if stored is None:
            raise Gap(f"the set's population program is {PO.program_sha512(program)[:16]}, the record's {rec['program'][:16]}")
        program, prov["program"] = stored, "a program.json the runs stored"
    query = P.template_instance_query(template.id)
    partition, prov["partition"] = partition_form(program, query, rec, [template.id])
    layouts = stand_in_layouts(s, n)
    ports, prov["ports"] = choose(((f"{tag} {layout}{' +content' if content else ''}",
                                    stand_in_ports(s.name, s.content_digest, n, layouts[layout], tag, content))
                                   for layout in ("words", "hidden") for tag in STAND_IN_BINDINGS for content in (True, False)),
                                  rec, "verity/flock-circuit/binding/v2 hidden")
    leaf_layer, prov["leaf_layer"] = public_digest(run, f"{base}/stage/pub-{n}.bin", f"{base}/session/archive-put.log")
    live = live_line(run.path(f"{base}/session/prove.log")) or {}
    sessions = (a.get("integrity_profile") or {}).get("evidence", {}).get("sessions") or session_digests(run, f"{base}/session")
    law = summary["law"]
    p = params(program, query, law, ports, leaf_layer, delta_of(summary["delta"]), partition=partition)
    e = evidence(a, instances=live.get("instances", -1), sessions=sessions, session_roots=live.get("roots") or {},
                 served=None)
    prov["evidence"] = "audit.json; instances and session roots from session/prove.log's LIVE line"
    c = Case(run.id, "a0", rec["format"], "audit", "audit", program, p, rec, e,
             recorded_audit(a, f"{base}/audit.json"), prov)
    yield c
    pc = None
    public = (a.get("checks_before_draw") or {}).get("public")
    if public is not None:
        pub = f"{base}/stage/pub-{n}.bin"
        with open(run.path(pub), "rb") as fh:
            header = json.loads(fh.readline())
        fv = header["frame_v3"]
        psrc = f"{base}/audit.json's checks_before_draw.public"
        pc = c.but("public", "audit", recorded("accept", source=psrc) if public["ok"] else
                   recorded("reject", "public", public["codes"], psrc),
                   public={"scheme": header["scheme"], "roots": fv["roots"], "domains": fv["domain_ids"],
                           "population": header["instances"]},
                   provenance={"public": f"the header of {pub}: its scheme, frame_v3's roots and domain_ids, its instances"})
        yield pc
    if not run.has(f"{base}/negatives.json"):
        return
    neg = run.json(f"{base}/negatives.json")
    src = f"{base}/negatives.json"
    draw = a["draw"]
    k = law["k"]
    smaller = R2.program([(template, n)], mixed=False, version=population_version(program))
    spart = with_program(partition, smaller)
    first = rec["roots"][0]
    cases: dict[str, Callable[[dict], Case]] = {} if pc is None else {
        "registration-root-mismatch": lambda v: pc.but(
            "registration-root-mismatch", "negative", recorded_codes(v, "public", src),
            rec={**rec, "roots": [{**first, "root": "00" * 64}, *rec["roots"][1:]]}),
        "public-file-domain-mismatch": lambda v: pc.but(
            "public-file-domain-mismatch", "negative", recorded_codes(v, "public", src),
            public={"domains": {**pc.public["domains"], first["port"]: "00" * 64}}),
    }
    cases |= {
        "registration-partition-mismatch": lambda v: c.but(
            "registration-partition-mismatch", "negative", recorded_codes(v, "registration", src), program=smaller,
            params={"program": descriptor(smaller), "partition": spart}),
        "registration-domain-not-the-layout's": lambda v: c.but(
            "registration-domain-not-the-layout's", "negative", recorded_codes(v, "registration", src),
            rec={**rec, "roots": [{**first, "domain": "00" * 64}, *rec["roots"][1:]]}),
        "registration-law-not-the-agreement's": lambda v: c.but(
            "registration-law-not-the-agreement's", "negative", recorded_codes(v, "registration", src),
            params={"law": {"k": k + 1, "law": "subset"}}),
        "second-registration-for-the-window": lambda v: c.but(
            "second-registration-for-the-window", "negative", recorded_codes(v, "registration", src),
            rec={**rec, "previous": R.digest(rec)}, params={"log": [rec]}),
        "registration-null-leaf-layer (R4)": lambda v: c.but(
            "registration-null-leaf-layer (R4)", "negative", recorded_codes(v, "registration", src),
            rec={**rec, "leaf_layer": None}),
        "draw-over-another-population": lambda v: c.but(
            "draw-over-another-population", "negative", recorded_codes(v, "draw", src),
            evidence={"draw": {**draw, "population": n + 1}}),
        "draw-outside-the-partition": lambda v: c.but(
            "draw-outside-the-partition", "negative", recorded_codes(v, "draw", src),
            evidence={"draw": {**draw, "units": draw["units"][:-1] + [n]}}),
        "draw-under-another-law": lambda v: c.but(
            "draw-under-another-law", "negative", recorded_codes(v, "draw", src),
            evidence={"draw": {**draw, "k": k - 1, "units": draw["units"][:-1]}}),
        "draw-altered-after-session (Lean U1)": lambda v: c.but(
            "draw-altered-after-session (Lean U1)", "negative",
            recorded("reject" if not v["proved-check"]["ok"] else "accept", "checks", ["proved"], f"{src} proved-check"),
            evidence={"draw": {**draw, "units": sorted(set(draw["units"][:-1]) | {next(u for u in range(n) if u not in draw["units"])})}}),
        "instances-not-the-draw (Lean U3)": lambda v: c.but(
            "instances-not-the-draw (Lean U3)", "analog", recorded("reject" if v["refused"] else "accept", None, None, src),
            evidence={"instances": len(draw["units"]) + 1}),
        "draw-dropped-from-record (Lean U1)": lambda v: c.but(
            "draw-dropped-from-record (Lean U1)", "analog", recorded("reject" if v["refused"] else "accept", None, None, src),
            evidence={"draw": None}),
    }
    yield from _negatives(run, c, neg, src, cases, base_instances=len(a["proved_units"]), session_roots=live.get("roots") or {})


def a0_lean(run: Run) -> Iterator[Case]:
    """A0 with the auditor's Lean program driving (`a0_lean.py`): A0's registration of a generated set (its name and
    content digest in `generate.log`), the draw `flock-audit`'s own, the session served at it (`served-draw.json`), and
    the audit record with `flock-audit` the verifier of record, taking the sessions, the served draw and the sessions'
    roots. The run kept neither the public file (its leaf layer is the record's, which the run's R4 matched against the
    launcher's own copy) nor the session's LIVE line (instances are the draw's proved units, which its passing proved
    check implies)."""
    base = "out"
    a = run.json(f"{base}/audit.json")
    state = run.json(f"{base}/state.json")
    rec = a["registration"]
    n = state["n"]
    gen = stage_report(run, f"{base}/generate.log") if "--- stdout" in run.path(f"{base}/generate.log").read_text() \
        else json.loads(run.path(f"{base}/generate.log").read_text())
    if gen.get("set") != rec["window"].get("input_set"):
        raise Gap(f"generate.log's set {gen.get('set')} is not the record's {rec['window'].get('input_set')}")
    s, set_from = run.input_set("rope101")
    if s.subcircuit.id != gen["subcircuit"]:
        raise Gap(f"rope101's subcircuit is {s.subcircuit.id}, the generated set's {gen['subcircuit']}")
    template = s.subcircuit.definition()
    program = own_population(lambda v: P.population_program(template, n, version=v), rec["program"])
    prov: dict = {"commit": run.commit(), "set": f"{gen['set']} (generate.log), its subcircuit {gen['subcircuit']} from "
                                                 f"rope101's relation ({set_from})",
                  "program": f"P.population_program(set template, n, version={population_version(program)})"}
    if PO.program_sha512(program) != rec["program"]:
        raise Gap(f"the set's population program is {PO.program_sha512(program)[:16]}, the record's {rec['program'][:16]}")
    query = P.template_instance_query(template.id)
    partition, prov["partition"] = partition_form(program, query, rec, [template.id])
    layouts = stand_in_layouts(s, n)
    ports, prov["ports"] = choose(((f"{tag} {layout}{' +content' if content else ''} ({gen['set']})",
                                    stand_in_ports(gen["set"], gen["content_digest"], n, layouts[layout], tag, content))
                                   for layout in ("words", "hidden") for tag in STAND_IN_BINDINGS for content in (True, False)),
                                  rec, "verity/flock-circuit/binding/v2 hidden")
    prov["leaf_layer"] = "the record's: the run kept no copy of the public file (stage and archive were scratch)"
    law = DR.parse_law(state["law"])
    served = run.json(f"{base}/served-draw.json")["units"]
    sessions = (a.get("integrity_profile") or {}).get("evidence", {}).get("sessions") or []
    p = params(program, query, law, ports, rec["leaf_layer"], delta_of(state["delta"]), partition=partition)
    e = evidence(a, instances=len(DR.proved(a["draw"])), sessions=sessions, session_roots=a["statement"].get("roots") or {},
                 served=served)
    prov["evidence"] = ("audit.json; served = served-draw.json; the sessions' roots the statement's; instances the draw's "
                        "proved units (inferred: the LIVE line was not kept, the run's proved check passed)")
    yield Case(run.id, "a0-lean", rec["format"], "audit", "audit", program, p, rec, e,
               recorded_audit(a, f"{base}/audit.json"), prov)


def _negatives(run: Run, c: Case, neg: dict, src: str, cases: dict[str, Callable[[dict], Case]], *, base_instances: int,
               session_roots: dict | None, served: list[int] | None = None,
               outside: dict[str, str] | None = None) -> Iterator[Case]:
    """The run's negatives: its own (`cases`), `audit_negatives`' (on the base audit as that function re-decided it:
    instances = the proved units, the sessions' roots, the served draw), and the ones outside the protocol (`outside`,
    the driver's own reasons, before `OUTSIDE`)."""
    an = c.but("", "negative", c.recorded, evidence={"instances": base_instances, "session_roots": session_roots,
                                                      "served": served})
    vs = c.evidence["verdicts"]
    port = next(iter(session_roots or {}), None)
    value = next(iter(audit.session_registered(vs, c.rec) or {}), None)
    other_root = {k: {**x, "registered": {p: {**r, "root": "00" * 64} if r["value"] == value else r
                                          for p, r in x["registered"].items()}}
                  if x.get("of_record") and isinstance(x.get("registered"), dict) else x for k, x in vs.items()}
    shared: dict[str, Callable[[dict], Case]] = {
        "verifier-of-record-did-not-run (C2)": lambda v: an.but(
            "verifier-of-record-did-not-run (C2)", "negative", recorded_failed(v, src),
            evidence={"verdicts": {k: {**x, "accepted": None} if x.get("of_record") else x for k, x in vs.items()}}),
        "no-verifier-of-record (C2)": lambda v: an.but(
            "no-verifier-of-record (C2)", "negative", recorded_failed(v, src),
            evidence={"verdicts": {k: x for k, x in vs.items() if not x.get("of_record")}}),
        "session-root-not-registered": lambda v: an.but(
            "session-root-not-registered", "negative", recorded_failed(v, src),
            evidence={"session_roots": {**(session_roots or {}), port: "00" * 64}}),
        "registered-read-opened-against-another-root": lambda v: an.but(
            "registered-read-opened-against-another-root", "negative", recorded_failed(v, src),
            evidence={"verdicts": other_root},
            provenance={"change": f"the verdicts of record report {value}'s reads opened against another root"}),
    }
    if served is not None:
        other = [u for u in range(c.rec["population"]) if u not in set(served)][:1]
        shared["served-share-not-the-verifier's-draw (C1)"] = lambda v: an.but(
            "served-share-not-the-verifier's-draw (C1)", "negative", recorded_failed(v, src),
            evidence={"served": sorted(served)[1:] + other})
    for name, v in neg.items():
        make = cases.get(name) or shared.get(name)
        if make is not None:
            yield make(v)
        else:
            run.outside.append({"run": c.run, "driver": c.driver, "name": name,
                                "why": (outside or {}).get(name) or OUTSIDE.get(name, "UNHANDLED: the replay has no case for it")})


@dataclass
class A2Copies:
    """A2's verifier copies of a record: the population program over the template set, the partition object the run
    passed, the served domains over the window's run, the leaf layer the SHA-512 of serving's public file."""
    template: Any
    program: Any
    query: dict
    part: dict
    layout: dict
    window_run: str
    ports: dict
    form: tuple[str, str]
    leaf_layer: str
    public: Callable[[], bytes]
    prov: dict


def a2_copies(run: Run, base: str, rec: dict, n: int) -> A2Copies:
    s, set_from = run.input_set("rope101")
    template = s.subcircuit.definition()
    program = own_population(lambda v: P.population_program(template, n, version=v), rec["program"])
    prov: dict = {"commit": run.commit(), "set": set_from,
                  "program": f"P.population_program(set template, n, version={population_version(program)})"}
    if PO.program_sha512(program) != rec["program"]:
        raise Gap(f"the set's population program is {PO.program_sha512(program)[:16]}, the record's {rec['program'][:16]}")
    query = P.template_instance_query(template.id)
    part = run.json(f"{base}/stage/partition.json")
    if P.digest(part) != rec["partition"] or part["program"] != rec["program"]:
        raise Gap("stage/partition.json is not the record's partition")
    prov["partition"] = f"{base}/stage/partition.json (the run passed its object)"
    layout = stand_in_layouts(s, n)["words"]
    window_run = rec["window"].get("run", "")
    ports, prov["ports"], form = choose_served(part, layout, window_run, rec)
    pub = f"{base}/stage/pub-{n}.bin"
    body: list[bytes] = []

    def public() -> bytes:
        if not body:
            try:
                body.append(run.path(pub).read_bytes())
            except Missing:
                body.append(tar_member(run, "inputs/served.tar", f"pub-{n}.bin"))
        return body[0]
    try:
        leaf_layer, prov["leaf_layer"] = public_digest(run, pub, f"{base}/session/archive-put.log")
    except Missing:
        leaf_layer = hashlib.sha512(public()).hexdigest()
        prov["leaf_layer"] = f"SHA-512 of serving's pub-{n}.bin in inputs/served.tar (the run did not keep {pub})"
    return A2Copies(template, program, query, part, layout, window_run, ports, form, leaf_layer, public, prov)


def a2(run: Run) -> Iterator[Case]:
    """A2 (`a2.py`): A0's template over serving's own commitment (vllm-serving-commit's rows): served domains recomputed
    from the record's program, partition and run, and the verifier's partition object passed as it is."""
    base = "out/a2"
    a = run.json(f"{base}/audit.json")
    summary = run.json(f"{base}/summary.json")
    rec = a["registration"]
    n = summary["n"]
    v2 = a2_copies(run, base, rec, n)
    program, part, prov = v2.program, v2.part, v2.prov
    live = live_line(run.path(f"{base}/session/prove.log")) or {}
    sessions = (a.get("integrity_profile") or {}).get("evidence", {}).get("sessions") or session_digests(run, f"{base}/session")
    law = {"k": summary["k"], "law": "subset"}
    p = params(program, v2.query, law, v2.ports, v2.leaf_layer, delta_of(summary["delta"]), partition=part)
    e = evidence(a, instances=live.get("instances", -1), sessions=sessions, session_roots=live.get("roots") or {},
                 served=None)
    prov["evidence"] = "audit.json; instances and session roots from session/prove.log's LIVE line"
    if summary.get("unbound"):
        prov["unbound"] = "the session ran on the verifier's re-headered copies for M0's unbound statement"
    c = Case(run.id, "a2", rec["format"], "audit", "audit", program, p, rec, e, recorded_audit(a, f"{base}/audit.json"), prov)
    yield c
    src = f"{base}/negatives.json"
    draw = a["draw"]
    smaller = R2.program([(v2.template, n)], mixed=False, version=population_version(program))
    cases: dict[str, Callable[[dict], Case]] = {
        "public-file-b||c-byte-flipped (R4)": lambda v: c.but(
            "public-file-b||c-byte-flipped (R4)", "negative", recorded_codes(v, "registration", src),
            params={"leaf_layer": hashlib.sha512(flipped(v2.public())).hexdigest()}),
        "domains-of-another-run (R3)": lambda v: c.but(
            "domains-of-another-run (R3)", "negative", recorded_codes(v, "registration", src),
            params={"ports": served_ports(v2.form, part, v2.layout, v2.window_run + "-other")}),
        "registration-partition-mismatch (R2)": lambda v: c.but(
            "registration-partition-mismatch (R2)", "negative", recorded_codes(v, "registration", src), program=smaller,
            params={"program": descriptor(smaller)}),
        "draw-outside-the-partition": lambda v: c.but(
            "draw-outside-the-partition", "negative", recorded_codes(v, "draw", src),
            evidence={"draw": {**draw, "units": draw["units"][:-1] + [n]}}),
    }
    yield from _negatives(run, c, run.json(src), src, cases, base_instances=len(a["proved_units"]),
                          session_roots=live.get("roots") or {})


def traced_argv(run: Run, script: str) -> list[str]:
    """The arguments `pod.sh` (under `set -x`) passed `script`, from the run's `stderr.log`."""
    for line in reversed(run.path("stderr.log").read_text().splitlines()):
        words = line.split()
        if words[:1] == ["+"] and script in words:
            return words[words.index(script) + 1:]
    raise Missing(f"stderr.log: {script}'s command line")


def option(argv: list[str], name: str, default: Any) -> Any:
    return argv[argv.index(name) + 1] if name in argv else default


#: the tools `a0.audit_session` names in its verdicts (neither side reads them)
A0_TOOLS = {"m0-verifier": "flock-circuit serve (M0, PR #83 eb90718f)",
            "lean-draw-u1-u3": "flock-verify draw-test (PR #113 9d0d53df)",
            "lean-verify": "flock-verify verify --archive (PR #118)"}


def copies_against(p: dict, rec: dict, run: Run) -> dict[str, str]:
    """Each replayed audit of the same record (by run): its copies equal `p`, or the parameters in which they differ."""
    out = {}
    for x in run.audits.get(R.digest(rec), []):
        differ = sorted(k for k in set(p) | set(x.params) if p.get(k) != x.params.get(k))
        out[x.run] = "equal" if not differ else f"differ in {', '.join(differ)}"
    return out


def a2_unaudited(run: Run) -> Iterator[Case]:
    """An A2 run that stopped before its audit record: the record served it (`inputs/served.tar`'s `registration.json`,
    the receipt's) under A2's copies, the law and δ the `--k` and `--delta` `pod.sh` passed `a2.py` (its defaults 256 and
    2^-20 where it passed none), and the evidence `a0.audit_session` handed `audit_record`, rebuilt from the session's
    files: the draw the session recorded (`session.json`'s `unit_draw`), the statement from the LIVE line and the
    session's coin seed, the verdicts (M0's from `session.json`, `flock-verify draw-test`'s DRAWCHECK lines, `flock-verify
    verify`'s VERDICT line as the verifier of record), the sessions' SHA-512, and the proved units through the public
    header's unit indices. The run recorded no verdict and no negatives."""
    base = "out/a2"
    rec = json.loads(tar_member(run, "inputs/served.tar", "registration.json"))
    receipt = run.json(f"{base}/receipt.json")
    if receipt.get("registration") != R.digest(rec):
        raise Gap(f"{base}/receipt.json does not name inputs/served.tar's registration.json")
    argv = traced_argv(run, "benchmarks/one_stage/a2.py")
    k, delta = int(option(argv, "--k", 256)), float(option(argv, "--delta", 2.0 ** -20))
    n = int(rec["population"])
    v2 = a2_copies(run, base, rec, n)
    prov = v2.prov
    prov["record"] = "inputs/served.tar's registration.json (the receipt's)"
    prov["law"] = f"subset:{k} and delta {delta!r}, a2.py's --k and --delta in stderr.log's trace (its defaults where absent)"
    sess = sorted(f["path"] for f in run.files if re.fullmatch(rf"{base}/session/sessions/[^/]+/session\.json", f["path"]))
    if not sess:
        raise Missing(f"{base}/session/sessions/*/session.json")
    record = run.json(sess[-1])
    draw = record.get("unit_draw")
    if draw is None:
        raise Gap(f"{sess[-1]}: the session recorded no draw")
    live = live_line(run.path(f"{base}/session/prove.log")) or {}
    statement = {"circuit_sha512": live.get("circuit_sha512"), "statement_digest": live.get("statement_digest"),
                 "public_sha512": live.get("public_sha512"), "sigma": live.get("sigma"), "roots": live.get("roots"),
                 "coin_commitment": (record.get("coin_seed") or {}).get("commitment"),
                 "roots_are_registered": live.get("roots") == {r["port"]: r["root"] for r in rec["roots"]}}
    m0 = record.get("verdict") or {}
    drawcheck = [json.loads(x.split(" ", 1)[1]) for x in run.path(f"{base}/session/u1-u3.lean.log").read_text().splitlines()
                 if x.startswith("DRAWCHECK ")]
    log = run.path(f"{base}/session/lean-verify.log").read_text()
    line = next((x for x in log.splitlines() if x.startswith("VERDICT ")), None)
    if line is None:
        raise Gap(f"{base}/session/lean-verify.log: no VERDICT line")
    lv = json.loads(line.split(" ", 1)[1])
    took = re.search(r"^# rc \S+ in ([0-9.]+) s$", log, re.M)
    verdicts = {
        "m0-verifier": {"accepted": m0.get("accepted") is True and statement["roots_are_registered"], "why": m0.get("why"),
                        "tool": A0_TOOLS["m0-verifier"]},
        "lean-draw-u1-u3": {"accepted": bool(drawcheck) and all(x["ok"] for x in drawcheck), "detail": drawcheck,
                            "tool": A0_TOOLS["lean-draw-u1-u3"]},
        "lean-verify": {"accepted": lv.get("accepted") is True, "why": lv.get("why"), "of_record": True,
                        "verify_s": float(took.group(1)) if took else None, "digest": hashlib.sha512(line.encode()).hexdigest(),
                        "tool": A0_TOOLS["lean-verify"]},
    }
    header = json.loads(v2.public().split(b"\n", 1)[0])
    proved = [header["units"]["indices"][u] for u in (live.get("unit_draw") or {}).get("units", [])]
    a = {"draw": draw, "proved_units": proved, "receipt": receipt, "statement": statement, "verdicts": verdicts}
    p = params(v2.program, v2.query, {"k": k, "law": "subset"}, v2.ports, v2.leaf_layer, delta_of(delta), partition=v2.part)
    e = evidence(a, instances=live.get("instances", -1), sessions=session_digests(run, f"{base}/session"),
                 session_roots=live.get("roots") or {}, served=None)
    prov["evidence"] = (f"rebuilt as a0.audit_session builds it: the draw from {sess[-1]}; the statement from "
                        "session/prove.log's LIVE line and the session's coin seed; the verdicts from session.json, "
                        "session/u1-u3.lean.log and session/lean-verify.log (verify_s from its timing line); the proved units "
                        f"through the unit indices of the header of {base}/stage/pub-{n}.bin (the re-headered copy keeps them); "
                        "instances and session roots from the LIVE line")
    if "--unbind" in argv:
        prov["unbound"] = "the session ran on the verifier's re-headered copies for M0's unbound statement"
    stopped = [x for x in run.path("out/a2.log").read_text().splitlines() if x.strip()] if run.has("out/a2.log") else []
    prov["stopped"] = f"out/a2.log: {stopped[-1].strip() if stopped else '?'}; the run wrote no audit.json or negatives.json"
    prov["copies_against"] = copies_against(p, rec, run)
    yield Case(run.id, "a2-unaudited", rec["format"], "audit", "audit", v2.program, p, rec, e, None, prov)


def _proved_check(v: dict, src: str) -> dict:
    """A negative the run decided with `draw.check_proved` alone, which the protocol makes as the audit's proved check."""
    return recorded("reject" if v["refused"] else "accept", "checks", ["proved"], f"{src} (check_proved)")


def a3(run: Run) -> Iterator[Case]:
    """A3 (`a3.py`): every template of one decoder layer as one population (`mixed_population_program` over
    `Q_template_instances`), each member a stand-in staged from its selection of the set (`stage_subset.py`), one
    registration (ports `<set>/<port>`, the leaf layer the SHA-512 of the map from set to its public file's SHA-512) and
    one draw over the union. With `--draw-file` the audited draw is the verifier's Lean draw and the sessions' union is
    the served draw; before it the audited draw is the union, and the audit took neither the served draw nor the
    sessions' roots."""
    base = "out/a3"
    a = run.json(f"{base}/audit.json")
    summary = run.json(f"{base}/summary.json")
    rec = a["registration"]
    members = summary["members"]
    prov: dict = {"commit": run.commit(), "sets": {}}
    mem = []
    for m in members:
        s, prov["sets"][m["set"]] = run.input_set(m["set"])
        mem.append((m["set"], s, s.subcircuit.definition(), m["n"]))
    program = own_population(lambda v: P.mixed_population_program([(d, cnt) for _, _, d, cnt in mem], version=v),
                             rec["program"])
    prov["program"] = f"P.mixed_population_program(each member's set template, n, version={population_version(program)})"
    if PO.program_sha512(program) != rec["program"]:
        stored = run.programs.get(rec["program"])
        if stored is None:
            raise Gap(f"the sets' population program is {PO.program_sha512(program)[:16]}, the record's {rec['program'][:16]} "
                      f"(templates today: {[P.template_id(d) for _, _, d, _ in mem]}; the run's: {[m['template'] for m in members]})")
        program, prov["program"] = stored, "a program.json the runs stored"
    query = P.template_instances_query([d for _, _, d, _ in mem])
    partition, prov["partition"] = partition_form(program, query, rec, [m["template"] for m in members])
    selections = rec["window"].get("sets") or {}
    ports, publics, prov["ports"], prov["leaf_layer"] = {}, {}, {}, {}
    for name, s, _, cnt in mem:
        prefix = f"{name}/"
        st = stage_report(run, f"{base}/stage-{name}/stage.log")
        if (selections.get(name) or {}).get("selection_digest") not in (None, st["content_digest"]):
            raise Gap(f"stage-{name}/stage.log's content digest is not the record's selection digest")
        layouts = stand_in_layouts(s, cnt, prefix)
        got, prov["ports"][name] = choose(
            ((f"{tag} {layout}{' +content' if content else ''} (selection {st['name']})",
              stand_in_ports(st["name"], st["content_digest"], cnt, layouts[layout], tag, content, prefix))
             for layout in ("words", "hidden") for tag in STAND_IN_BINDINGS for content in (True, False)),
            rec, "verity/flock-ir-frame/binding/v1 words +content", only=lambda p, pre=prefix: p.startswith(pre))
        ports.update(got)
        publics[name], prov["leaf_layer"][name] = public_digest(run, f"{base}/stage-{name}/pub-{cnt}.bin",
                                                                f"{base}/session-{name}/archive-put.log")
    leaf_layer = hashlib.sha512(DT.canonical(publics).encode()).hexdigest()
    split = a.get("split") or {}
    draw_file = "--draw-file" in split.get("rule", "")
    with_roots = "served_is_lean_draw" in split
    instances, session_roots, sessions = 0, {}, []
    for name, *_ in mem:
        if not split.get("counts", {}).get(name):
            continue
        live = live_line(run.path(f"{base}/session-{name}/prove.log")) or {}
        instances += live.get("instances", 0)
        session_roots.update({f"{name}/{p}": r for p, r in (live.get("roots") or {}).items()})
        sessions += session_digests(run, f"{base}/session-{name}")
    sessions = (a.get("integrity_profile") or {}).get("evidence", {}).get("sessions") or sessions
    if draw_file and not split.get("served_is_lean_draw"):
        raise Gap("the sessions' served units are not the Lean draw, and the run kept no other copy of them")
    served = sorted(a["draw"]["units"]) if draw_file else None
    law = {"k": summary["k"], "law": "subset"}
    p = params(program, query, law, ports, leaf_layer, delta_of(summary["delta"]), partition=partition)
    e = evidence(a, instances=instances, sessions=sessions, session_roots=session_roots if with_roots else None,
                 served=served)
    prov["evidence"] = ("audit.json; instances and session roots from each session's prove.log LIVE line"
                        + ("; served = the Lean draw (--draw-file)" if draw_file else
                           "; the era's audit took neither the served draw nor the sessions' roots"))
    c = Case(run.id, "a3", rec["format"], "audit", "audit", program, p, rec, e, recorded_audit(a, f"{base}/audit.json"), prov)
    yield c
    src = f"{base}/negatives.json"
    n = rec["population"]
    union = {"law": "subset", "population": n, "k": law["k"], "units": sorted(a["draw"]["units"])}
    proved = sorted(a["proved_units"])
    bases = {name: (lo, lo + cnt) for (name, *_), (_, lo, cnt) in zip(mem, P.ranges(program, query))}
    fewer = R2.program([(d, cnt) for _, _, d, cnt in mem], mixed=True, version=population_version(program))
    cases: dict[str, Callable[[dict], Case]] = {
        "registration-partition-mismatch": lambda v: c.but(
            "registration-partition-mismatch", "negative", recorded_codes(v, "registration", src), program=fewer,
            params={"program": descriptor(fewer), "partition": with_program(partition, fewer)}),
        "registration-query-drops-a-template": lambda v: c.but(
            "registration-query-drops-a-template", "negative", recorded_codes(v, "registration", src),
            params={"query": P.template_instances_query([d for _, _, d, _ in mem][:-1]),
                    "partition": with_program(partition, program, drop_template=True)}),
        "union-draw-outside-the-partition": lambda v: c.but(
            "union-draw-outside-the-partition", "negative", recorded_codes(v, "draw", src),
            evidence={"draw": {**union, "units": union["units"][:-1] + [n]}}),
        "proved-unit-not-drawn": lambda v: c.but(
            "proved-unit-not-drawn", "negative", _proved_check(v, src),
            evidence={"proved_units": sorted([next(u for u in range(n) if u not in union["units"]), *proved[1:]])}),
        "registration-null-leaf-layer (R4)": lambda v: c.but(
            "registration-null-leaf-layer (R4)", "negative", recorded_codes(v, "registration", src),
            rec={**rec, "leaf_layer": None}),
    }
    for name, (lo, hi) in bases.items():
        kept = [u for u in proved if not lo <= u < hi]
        cases[f"session-dropped ({name})"] = lambda v, name=name, kept=kept: c.but(
            f"session-dropped ({name})", "negative", _proved_check(v, src),
            evidence={"proved_units": kept, "instances": len(kept)})
    yield from _negatives(run, c, run.json(src), src, cases, base_instances=len(a["proved_units"]),
                          session_roots=session_roots if with_roots else None, served=served)


#: A commit on main whose `benchmarks/one_stage/a4_layout.json` is every A4 era's (the file never changed)
REPO = Path(__file__).resolve().parents[2]


def a4_layout(run: Run, name: str) -> tuple[dict, str]:
    """The layout of record `name` as the run read it: `pod.sh`'s `LAYOUT_FILE` in the served directory, else
    `benchmarks/one_stage/a4_layout.json` at the run's commit."""
    file = run.pod_args().get("LAYOUT_FILE")
    if file:
        return json.loads(tar_member(run, "inputs/served.tar", Path(file).name))[name], f"inputs/served.tar's {file}"
    commit = run.commit()
    p = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:benchmarks/one_stage/a4_layout.json"],
                       capture_output=True, text=True) if commit else None
    if p is None or p.returncode != 0:
        raise Missing(f"benchmarks/one_stage/a4_layout.json at {commit} (git history)")
    return json.loads(p.stdout)[name], f"benchmarks/one_stage/a4_layout.json at {commit[:9]}"


def a4_member_layouts(s, n: int, row_map: dict | None) -> dict[str, dict[str, tuple[str, int]]]:
    """`a4.member_ports`: a GEMM member's row ports are tables whose sizes the grid rule fixes, its outputs words."""
    if not row_map:
        return stand_in_layouts(s, n)
    groups = row_map["groups"]
    rows = {"x": max(g["x_base"] + g["tokens"] for g in groups), "w": max(g["w_base"] + g["columns"] for g in groups)}
    return {"words": {**{p.name: (ROW, rows[p.name]) for p in s.subcircuit.inputs},
                      **{p.name: ("u16", n * p.words) for p in s.subcircuit.outputs}}}


def a4_copies(run: Run, base: str, summary: dict, rec: dict) -> tuple[Any, dict, dict | None, dict, str, list, list[str], dict]:
    """A4's verifier copies of `rec` under the run's layout of record (`summary.json`'s layout and members): the
    program, query, partition, served domains and leaf layer; the members and their template ids; the provenance."""
    lay, layout_from = a4_layout(run, summary["layout"])
    prov: dict = {"commit": run.commit(), "layout": f"{summary['layout']} from {layout_from}", "sets": {}}
    own = run.file_sha256("inputs/set.tar")
    if own is None:
        raise Missing("inputs/set.tar")
    mem = []
    for k, mm in enumerate(lay["members"]):
        name = run.sets.starting(own, mm["set"])
        if name is None:
            raise Missing(f"inputs/set.tar: a set named {mm['set']}*")
        s, prov["sets"][str(k)] = run.input_set(name)
        mem.append((k, s, s.subcircuit.definition(), int(mm["count"]), mm.get("row_map")))
    tids = [m["template"] for m in summary["members"]]
    program = own_population(lambda v: P.mixed_population_program([(d, cnt) for _, _, d, cnt, _ in mem], version=v),
                             rec["program"])
    prov["program"] = f"P.mixed_population_program(each layout member's set template, count, version={population_version(program)})"
    if PO.program_sha512(program) != rec["program"]:
        stored = run.programs.get(rec["program"])
        if stored is None:
            raise Gap(f"the layout's population program is {PO.program_sha512(program)[:16]}, the record's {rec['program'][:16]}")
        program, prov["program"] = stored, "a program.json the runs stored"
    query = P.template_instances_query([d for _, _, d, _, _ in mem])
    partition, prov["partition"] = partition_form(program, query, rec, tids)
    pd = P.digest(partition if partition is not None else P.partition(program, query))
    run_id = rec["window"].get("run", "")

    def ports_of(form: tuple[str, str], kind: str) -> dict:
        out = {}
        for (k, s, _, cnt, row_map), tid in zip(mem, tids):
            layouts = a4_member_layouts(s, cnt, row_map)
            for p, (schema, leaves) in layouts.get(kind, layouts["words"]).items():
                out[f"{tid}/{p}"] = {"leaves": leaves,
                                     "domain": served_domain(form, rec["program"], pd, f"{tid}/{p}", schema, leaves, run_id)}
        return out
    ports, prov["ports"] = choose(((f"served_domain {f[0]} {f[1]} {kind}", ports_of(f, kind))
                                   for f in SERVED_FORMS for kind in ("words", "hidden")),
                                  rec, f"served_domain {SERVED_FORMS[0][0]} {SERVED_FORMS[0][1]} words")
    publics, prov["leaf_layer"] = {}, {}
    for (k, _, _, cnt, _), tid in zip(mem, tids):
        publics[tid], prov["leaf_layer"][tid] = public_digest(run, f"{base}/m{k}/pub-{cnt}.bin",
                                                              f"{base}/session-m{k}/archive-put.log")
    leaf_layer = hashlib.sha512(DT.canonical(publics).encode()).hexdigest()
    return program, query, partition, ports, leaf_layer, mem, tids, prov


def a4(run: Run) -> Iterator[Case]:
    """A4 (`a4.py`): #101's layer 0 on serving's own roots: the layout of record's members as one population (the
    canonical `Q_template_instances` partition), serving's registration (or, with `--reaudit`, a new record naming it under
    the agreement's law, the served record in the log), the served domains under `<template>/<port>`, the leaf layer the
    SHA-512 of the map from template to its public file's SHA-512, one draw by the verifier's law with each member served
    its share. Before 9515b4d4f the audit took neither the served draw nor the sessions' roots (and the run recorded no
    `audit_negatives`)."""
    base = "out/a4"
    a = run.json(f"{base}/audit.json")
    summary = run.json(f"{base}/summary.json")
    rec = a["registration"]
    program, query, partition, ports, leaf_layer, mem, tids, prov = a4_copies(run, base, summary, rec)
    neg = run.json(f"{base}/negatives.json")
    with_served = "verifier-of-record-did-not-run (C2)" in neg
    counts = a.get("draw_counts") or {}
    instances, session_roots, sessions = 0, {}, []
    for (k, *_), tid in zip(mem, tids):
        if not counts.get(tid):
            continue
        live = live_line(run.path(f"{base}/session-m{k}/prove.log")) or {}
        instances += live.get("instances", 0)
        session_roots.update({f"{tid}/{p}": r for p, r in (live.get("roots") or {}).items()})
        sessions += session_digests(run, f"{base}/session-m{k}")
    sessions = (a.get("integrity_profile") or {}).get("evidence", {}).get("sessions") or sessions
    if not a.get("served_is_lean_draw"):
        raise Gap("the sessions' served units are not the Lean draw, and the run kept no other copy of them")
    served = sorted(a["draw"]["units"]) if with_served else None
    stratified = (summary.get("law") or "").startswith("stratified")
    law = {"k": summary["k"], "law": "stratified" if stratified else "subset"}
    log: list = []
    served_rec = None
    if summary.get("reaudit"):
        of = summary["reaudit"]["of"]
        if of in run.records.by_digest:
            served_rec, where = run.records.by_digest[of]
            prov["log"] = f"the served record, as {where} holds it"
        else:
            served_rec = json.loads(tar_member(run, "inputs/served.tar", "registration.json"))
            prov["log"] = "the served record (inputs/served.tar's registration.json)"
        log = [served_rec]
    p = params(program, query, law, ports, leaf_layer, delta_of(summary["delta"]), partition=partition, log=log)
    e = evidence(a, instances=instances, sessions=sessions, session_roots=session_roots if with_served else None,
                 served=served)
    prov["evidence"] = ("audit.json; instances and session roots from each session's prove.log LIVE line"
                        + ("; served = the Lean draw" if with_served else
                           "; the era's audit took neither the served draw nor the sessions' roots"))
    c = Case(run.id, "a4", rec["format"], "audit", "audit", program, p, rec, e, recorded_audit(a, f"{base}/audit.json"), prov)
    yield c
    src = f"{base}/negatives.json"
    n = rec["population"]
    draw = a["draw"]
    proved = sorted(a["proved_units"])
    other = R2.program([(d, cnt) for _, _, d, cnt, _ in mem], mixed=True, version=population_version(program))
    cases: dict[str, Callable[[dict], Case]] = {
        "registration-partition-mismatch (R2)": lambda v: c.but(
            "registration-partition-mismatch (R2)", "negative", recorded_codes(v, "registration", src), program=other,
            params={"program": descriptor(other), "partition": with_program(partition, other)}),
        "domains-not-the-layout's (R3)": lambda v: c.but(
            "domains-not-the-layout's (R3)", "negative", recorded_codes(v, "registration", src),
            params={"ports": {nm: {**w, "domain": "00" * 64} for nm, w in ports.items()}}),
        "draw-outside-the-partition": lambda v: c.but(
            "draw-outside-the-partition", "negative", recorded_codes(v, "draw", src),
            evidence={"draw": {**draw, "units": draw["units"][:-1] + [n]}}),
        "proved-set-short-of-the-draw": lambda v: c.but(
            "proved-set-short-of-the-draw", "negative", _proved_check(v, src),
            evidence={"proved_units": proved[1:], "instances": len(proved) - 1}),
        "registration-null-leaf-layer (R4)": lambda v: c.but(
            "registration-null-leaf-layer (R4)", "negative", recorded_codes(v, "registration", src),
            rec={**rec, "leaf_layer": None}),
    }
    if stratified:
        strata = rec["law"]["strata"]
        big = max(strata, key=lambda s: s["k"])["template"]
        first = strata[0]["template"]
        ranges = {s["template"]: s["units"] for s in draw["strata"]}
        drawn = set(draw["units"])
        extra = next(u for lo, hi in ranges[first] for u in range(lo, hi) if u not in drawn)
        drop = next(u for u in draw["units"] if any(lo <= u < hi for lo, hi in ranges[big]))
        cases.update({
            "registered-law-with-a-stratum-k-changed (R1)": lambda v: c.but(
                "registered-law-with-a-stratum-k-changed (R1)", "negative", recorded_codes(v, "registration", src),
                rec={**rec, "law": {**rec["law"], "strata": [{**s, "k": s["k"] + 1} if s["template"] == big else s
                                                             for s in strata]}}),
            "draw-swaps-a-unit-into-a-second-of-the-first-stratum (strata-count)": lambda v: c.but(
                "draw-swaps-a-unit-into-a-second-of-the-first-stratum (strata-count)", "negative",
                recorded_codes(v, "draw", src), evidence={"draw": {**draw, "units": sorted((drawn - {drop}) | {extra})}}),
            "stratum-k-written-true (law, strata)": lambda v: c.but(
                "stratum-k-written-true (law, strata)", "negative", recorded_codes(v, "draw", src),
                evidence={"draw": {**draw, "strata": [{**s, "k": True} if s["template"] == first else s
                                                      for s in draw["strata"]]}}),
        })
    if served_rec is not None:
        cases["reaudit-reuses-the-window (R6)"] = lambda v: c.but(
            "reaudit-reuses-the-window (R6)", "negative", recorded_codes(v, "registration", src),
            rec={**rec, "window": served_rec["window"]})
    yield from _negatives(run, c, neg, src, cases, base_instances=len(a["proved_units"]),
                          session_roots=session_roots if with_served else None, served=served)


def a4_unaudited(run: Run) -> Iterator[Case]:
    """An A4 run refused before its draw: the record served it (`inputs/served.tar`'s `registration.json`, the
    receipt's) under A4's copies (`summary.json`'s layout, members, k and δ), the registration alone, since the run holds
    no draw. The run's own registration check (`summary.json`'s `checks.registration`) is the recorded result of that
    stage; its other refusals before the draw are M0's checks of each member's served files (pin, classes, bindings),
    which the protocol does not make."""
    base = "out/a4"
    summary = run.json(f"{base}/summary.json")
    rec = json.loads(tar_member(run, "inputs/served.tar", "registration.json"))
    if R.digest(rec) != summary.get("registration") or R.digest(rec) != run.json(f"{base}/receipt.json").get("registration"):
        raise Gap(f"inputs/served.tar's registration.json is not the one {base}/summary.json and receipt.json name")
    program, query, partition, ports, leaf_layer, _, _, prov = a4_copies(run, base, summary, rec)
    prov["record"] = "inputs/served.tar's registration.json (the receipt's)"
    stratified = (summary.get("law") or "").startswith("stratified")
    law = {"k": summary["k"], "law": "stratified" if stratified else "subset"}
    p = params(program, query, law, ports, leaf_layer, delta_of(summary["delta"]), partition=partition)
    checks = summary.get("checks") or {}
    reg, src = checks.get("registration"), f"{base}/summary.json's checks.registration"
    rv = None if reg is None else recorded("pass", source=src) if reg["ok"] else recorded("reject", "registration", reg["codes"], src)
    others = {k: v["codes"] for k, v in checks.items() if k not in ("registration", "source") and not v["ok"]}
    if others:
        prov["refused_before_draw"] = f"{others}: M0's checks of the members' served files, not the protocol's"
    prov["copies_against"] = copies_against(p, rec, run)
    yield Case(run.id, "a4-unaudited", rec["format"], "registration", "audit", program, p, rec, dict(NO_DRAW), rv, prov,
               stages=REGISTRATION)


def bundle_leaf_layer(run: Run) -> tuple[str, str | dict]:
    """A serving bundle's leaf layer: the SHA-512 of the map from each member's template to its public file's SHA-512
    (`members.json`, #101's layers), or of its one public file (`compare.json`); each SHA-512 the file's where the
    bundle holds it, else the one serving recorded of it."""
    if run.has("members.json"):
        publics, how = {}, {}
        for m in run.json("members.json")["members"]:
            f = f"m{m['k']}-pub-{m['n']}.bin"
            if run.has(f):
                publics[m["template"]], how[m["template"]] = sha512_file(run.path(f)), f"SHA-512 of {f}"
            else:
                publics[m["template"]], how[m["template"]] = m["pub"]["file_sha512"], "members.json's pub.file_sha512"
        return hashlib.sha512(DT.canonical(publics).encode()).hexdigest(), how
    return run.json("compare.json")["pub"]["file_sha512"], "compare.json's pub.file_sha512 (serving's SHA-512 of its public file)"


def serving_bundle(run: Run) -> Iterator[Case]:
    """A serving bundle (`vllm-serving-commit`'s files: `registration.json`, `window.json`, `index.json`, and
    `compare.json` for one template or `members.json` and the members' public files for #101's layers) that no verifier
    audited in its own run: the registration alone, under the copies of each replayed audit of the same record (its
    digest) with the leaf layer from the bundle itself, and no draw. Its `registration-v0.json`, the window's record in
    `verity/one-stage/registration/v0`, is a format neither side reads."""
    rec = run.json("registration.json")
    audits = run.audits.get(R.digest(rec)) or []
    if not audits:
        raise Gap(f"no replayed run audited registration {R.digest(rec)[:16]}, whose verifier's copies would read it")
    leaf_layer, how = bundle_leaf_layer(run)
    groups: dict[str, list[Case]] = {}
    for b in audits:
        groups.setdefault(DT.canonical({**b.params, "leaf_layer": leaf_layer}), []).append(b)
    for b, *rest in groups.values():
        prov = {"record": f"registration.json, the record run {b.run} audited (the same digest)",
                "copies": f"run {b.run}'s audit case" + (f" (equal to {', '.join(x.run for x in rest)}'s)" if rest else ""),
                "leaf_layer": how,
                "leaf_layer_against": f"equal to run {b.run}'s" if leaf_layer == b.params["leaf_layer"] else
                                      f"run {b.run}'s is {b.params['leaf_layer']}"}
        if run.has("registration-v0.json"):
            prov["registration-v0.json"] = "the window's record in verity/one-stage/registration/v0, which neither side reads"
        name = "registration" if len(groups) == 1 else f"registration ({b.run}'s copies)"
        yield Case(run.id, "serving-bundle", rec["format"], name, "audit", b.program, {**b.params, "leaf_layer": leaf_layer},
                   rec, dict(NO_DRAW), None, prov, b.r7, REGISTRATION)


def registered_gemm(run: Run) -> Iterator[Case]:
    """A4 of registered values (`a4.py` before A4 was #101's layer 0): `RegisteredGemm`, `n` Calls of the generated BF16
    set's GEMM coordinate, Call `i` reading its activation row `x[i]` and the weight column `W[pos[i]]` of the root
    parameter W (`pos` the reads of `stage/registered.json`). Windows register W in a `verity/registration/v2` record (R7:
    the verifier expects `{W: K}`, and a later window's log holds the earlier windows' records), baselines in a v1 record
    that registers nothing; every record also registers its statement's roots under M0's set-bound domains over the public
    file's shared-row tables. The audit took the sessions and their roots (the LIVE line's), not the served draw, and read
    the roots the Lean verdict opened registered reads against; `audit_negatives` re-decided it with the record's roots as
    the sessions'."""
    from verity_numerical.bench.input_sets import InputSet
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import coll_concat
    from verity.ml.circuits.types import Array
    base = "run"
    summary = run.json(f"{base}/summary.json")
    n, cols = summary["n"], summary["cols"]
    man = next((f["path"] for f in run.files if re.fullmatch(rf"{base}/set/sets/[^/]+/manifest\.json", f["path"])), None)
    if man is None:
        raise Missing(f"{base}/set/sets/*/manifest.json")
    sd = man.rsplit("/", 1)[0]
    for f in run.files:
        if f["path"].startswith(sd + "/"):
            run.path(f["path"])
    s = InputSet.open(str(run.dir / sd))
    template = s.subcircuit.definition()
    (port, rd), = run.json(f"{base}/stage/registered.json")["ports"].items()
    value, pos = rd["value"], rd["reads"]
    (_, xt), (_, wt) = template.signature()[0]
    ret = template.signature()[1]

    def sig(S):
        return (("x", Array(n, xt)), (value, Array(cols, wt))), Array(n, ret)

    def body(B, S, x, W):
        return coll_concat(sig(S)[1], [B.call(template, x[i], W[p]) for i, p in enumerate(pos)])
    program = Program(bind(CompositeDefinition("RegisteredGemm", 0, (), sig, body, register=False)))
    values = {value: summary["K"]}
    parameters = _json([[name, t.to_json()] for name, t in program.fn.signature()[0]])
    query = P.template_instance_query(template)
    era_tid = run.json(f"{base}/stage/partition.json")["query"]["params"]["template"]
    law = DR.parse_law(f"subset:{summary['k']}")
    prov0: dict = {"commit": run.commit(), "set": f"{sd} (the run's generated set)",
                   "program": f"RegisteredGemm over the set's template, W[pos] from {base}/stage/registered.json's reads",
                   "registered": f"{values} with the program function's parameters"}
    log: list[dict] = []
    for w in summary["windows"]:
        name, d = w["name"], f"{base}/{w['name']}"
        rec = run.json(f"{d}/registration.json")
        a = run.json(f"{d}/audit.json")
        if PO.program_sha512(program) != rec["program"]:
            raise Gap(f"RegisteredGemm is {PO.program_sha512(program)[:16]}, {name}'s record's {rec['program'][:16]}")
        prov = dict(prov0)
        partition, prov["partition"] = partition_form(program, query, rec, [era_tid])
        pub = f"{d}/stage/pub-{n}.bin"
        with open(run.path(pub), "rb") as fh:
            header = json.loads(fh.readline())
        layout = {p: (schema, header["shared_rows"][p] if schema == ROW else n) for p, schema in header["frame_v3"]["schemas"].items()}
        ports, prov["ports"] = choose(((f"{tag}{' +content' if content else ''} over {pub}'s shared rows",
                                        stand_in_ports(s.name, s.content_digest, n, layout, tag, content))
                                       for tag in STAND_IN_BINDINGS for content in (True, False)),
                                      rec, f"verity/flock-ir-frame/binding/v1 +content over {pub}'s shared rows")
        leaf_layer, prov["leaf_layer"] = public_digest(run, pub, f"{d}/session/archive-put.log")
        registered = bool(w.get("registered"))
        prior = list(log) if registered else []
        prov["log"] = f"the {len(prior)} earlier windows' records" if prior else "empty"
        live = live_line(run.path(f"{d}/session/prove.log")) or {}
        sessions = (a.get("integrity_profile") or {}).get("evidence", {}).get("sessions") or session_digests(run, f"{d}/session")
        delta = delta_of((a.get("profile") or {}).get("delta") or a["wrong_units"]["delta"])
        p = params(program, query, law, ports, leaf_layer, delta, partition=partition, log=prior,
                   registered=values if registered else None)
        if registered:
            p["parameters"] = parameters
        e = evidence(a, instances=live.get("instances", -1), sessions=sessions, session_roots=live.get("roots") or {},
                     served=None)
        prov["evidence"] = (f"{d}/audit.json; instances and session roots from session/prove.log's LIVE line; registered "
                            "reads' roots from the lean-verify verdict")
        c = Case(run.id, "registered-gemm", rec["format"], name, "audit", program, p, rec, e,
                 recorded_audit(a, f"{d}/audit.json"), prov, r7=registered)
        yield c
        if registered:
            log.append(rec)
        neg = w.get("negatives") or {}
        src = f"{base}/summary.json ({name}'s negatives)"
        cases: dict[str, Callable[[dict], Case]] = {}
        if registered and run.has(f"{d}/own-lean-reads-another-root-of-W.json"):
            other = run.json(f"{d}/own-lean-reads-another-root-of-W.json")["ports"][port]["root"]
            later = {**rec, "window": {**rec["window"], "window": 99}, "previous": R.digest(rec),
                     "registered": [{**x, "root": other} if x["value"] == value else x for x in rec["registered"]]}
            nm = "registration-re-registers-W-under-another-root (R7-held)"
            cases[nm] = lambda v, c=c, nm=nm, later=later, held=list(log): c.but(
                nm, "negative", recorded_codes(v, "registration", src), rec=later, params={"log": held},
                provenance={"change": f"window 99 after {name}, W under the run's second commitment of it (its root from "
                                      f"{d}/own-lean-reads-another-root-of-W.json)"})
        yield from _negatives(run, c, neg, src, cases, base_instances=len(a["proved_units"]),
                              session_roots={r["port"]: r["root"] for r in rec["roots"]})


#: what the era's code imports from a run's commit
ERA_PATHS = ("verity", "catalog/verity_catalog", "kernels/verity_kernels", "benchmarks/pouw/served_zk")


def era_tree(run: Run, commit: str) -> Path:
    """The run's commit's own code (`ERA_PATHS`, from git history), for a verifier's copy the current code can't build."""
    out = run.sets.root.parent / "_trees" / commit[:12]
    if (out / ".done").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    for sub in ERA_PATHS:
        a = subprocess.run(["git", "-C", str(REPO), "archive", commit, sub], capture_output=True)
        if a.returncode != 0:
            raise Missing(f"{sub} at {commit[:9]} (git history)")
        subprocess.run(["tar", "-x", "-C", str(out)], input=a.stdout, check=True)
    (out / ".done").write_text(commit)
    return out


def era_python(tree: Path, code: str, *args: str) -> Any:
    """`code` run on the era's tree, its last stdout line read as JSON."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(tree / p) for p in ("", "catalog", "kernels"))}
    p = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, env=env, cwd=tree)
    if p.returncode != 0:
        raise Gap(f"the code at {tree.name}: {(p.stderr.strip().splitlines() or ['?'])[-1]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


#: served_zk's verifier copies at each level, by the era's own `served_commit`
SERVED_ZK_ERA = r'''
import json, sys
sys.path.insert(0, "benchmarks/pouw/served_zk")
import served_commit as SC
from verity.protocols.verification.sampled_proofs.one_stage import partition as PT
from verity.primitives.circuits.codec import encode_program
params, levels = json.loads(sys.argv[1]), json.loads(sys.argv[2])
t = SC.template(params["k"])
out = {"template": t.id, "tid": PT.template_id(t), "levels": {}}
for ell in levels:
    n = SC.work_at(params, ell).n
    out["levels"][str(ell)] = {"n": n, "program": encode_program(PT.population_program(t, n)), "ports": SC.ports(params, ell)}
print(json.dumps(out))
'''
#: served_zk's registered scheme before the proof system (`served_commit.SCHEME`); in it, the service's (`SCHEME`)
SERVED_ZK_SCHEME = "verity/pouw/served-zk/window/hm96-sha512+frame-v3-sha512/v1"
SERVED_ZK_W_DOMAIN = "verity/pouw/served-zk/weights-domain/v0"
#: served windows' recorded refusals that are not the protocol's, by name, with why
SERVED_OUTSIDE = (
    (r"register-extra", "the service's public-items rule at registration (`public_problem`), before any record exists"),
    (r"receive-extra", "the service's public-items rule (`public_problem`): one-stage's record carries the window's items "
                       "without reading them"),
    (r"receive-foreign", "the service's public-context rule (`public_problem`), before one-stage's checks; analog: the "
                         "registration under the request's own copies, which R3 refuses (its served domain binds the context)"),
    (r"receive-raised", "PoUW's rule (`public_problem`: public-value, population), before one-stage's checks; analog: the "
                        "verifier's copies at the raised level, which R2 and R3 refuse"),
    (r"receive-open", "the stream's rule: one registration open at a time (`Stream.register`)"),
    (r"receive-(coin-.*|unrecorded)", "PoUW's epoch coins (`pouw.coins`), the challenger's, before one-stage's checks"),
    (r"select-again", "the service draws once per receipt and never redraws (`select`); the protocol checks the draw it is handed"),
    (r"receive-after-draw", "the challenger's record order (`Challenger.register`, late-registration): each window is received "
                            "before the epoch's draw coin is released"),
    (r"(outcome|deliver)-again|retry-.*", "the service's stream: one delivery, a final outcome, the failure budget"),
    (r"open-.*", "served_zk's opening check (a proof's rows are the drawn unit's), a verdict beside the verifier of record's"),
    (r"lean-.*", "the verifier of record (Lean `flock-verify verify --zk`) on one unit's session; the protocol reads its verdict"),
    (r"firewall-.*", "the firewall's checks on a delivered proof, before the outcome"),
)


def served_window(run: Run) -> Iterator[Case]:
    """served_zk's window through the proof system (`benchmarks/pouw/served_zk`): the request's registration of PoUW's
    tiles (`population_program` of the tile template over the registered level's N, `Q_template_instance`), received
    under the verifier's own copies (its `params.json`: the program at the record's level, the law of `pouw.window.sampling`,
    the window's served domain and W's domain, no leaf layer, no anchors, the stream's empty log), then one outcome per case
    (`outcome-<case>/`: honest, the selftest build's cases, the other request's proof), each its own audit record of the
    request's registration (a stream of its own, order 0). Before the proof system (`verity/one-stage/audit/v0`) the
    scheme is `served_commit.SCHEME` and the audit took no sessions; in it (`verity/proof-service/outcome/v0`) the scheme
    is the service's and the audit took the delivered sessions. Neither took the served draw or the sessions' roots, and
    no verdict of record reports registered reads."""
    from verity.core.protocols.pouw import window as Wd
    from verity.core.protocols.pouw.circuit import leaves as L, pc8
    from verity.ml.commitments.rowleaf import ROLE_W, SCHEMA_HM96_SHA512_ROW_V2
    pv = run.json("out/request/params.json")
    rec = run.json("out/request/registration.json")
    commit = run.commit()
    prov: dict = {"commit": commit, "params": "out/request/params.json (the verifier's own)"}
    outcomes = sorted(f["path"] for f in run.files if re.fullmatch(r"out/outcome-[^/]+/audit\.json", f["path"]))
    if not outcomes:
        raise Missing("out/outcome-*/audit.json")
    raised = run.json("out/registration-raised.json") if run.has("out/registration-raised.json") else None
    levels = [rec["window"]["public"]["work"]["level"]] + ([raised["window"]["public"]["work"]["level"]] if raised else [])
    tile_work = pv["k"]

    def law_at(ell: int) -> dict:
        return Wd.sampling(Fraction(pv["eps"]), Fraction(pv["delta"]), Wd.Work.at(ell, pv["grid_bits"]), tile_work,
                           pv["grid_bits"])["law"]
    n0 = Wd.Work.at(levels[0], pv["grid_bits"]).n
    tiles = {"pc8.side_tile": pc8.side_tile(pv["k"], (1, 1), "sm120"),
             "pc8's hiding tile": pc8.templates(pv["k"], (1, 1), "sm120", hidden=True)["tile"]}
    def own(x):
        return own_population(lambda v: P.population_program(x, n0, version=v), rec["program"])
    how, t = next(((h, x) for h, x in tiles.items() if PO.program_sha512(own(x)) == rec["program"]),
                  next(iter(tiles.items())))
    era = None
    first = own(t)
    version = population_version(first)
    if PO.program_sha512(first) != rec["program"]:
        if commit is None:
            raise Gap("today's tile template does not give the record's program, and the run names no commit")
        era = era_python(era_tree(run, commit), SERVED_ZK_ERA, json.dumps(pv), json.dumps(levels))
        prov["program"] = f"the era's `served_commit.template` at {commit[:9]} (today's code has no {era['template']})"
    else:
        prov["program"] = f"P.population_program({how} template, N at the registered level, version={version})"
    stubs: list[str] = []

    def program_at(ell: int):
        if era is None:
            return P.population_program(t, Wd.Work.at(ell, pv["grid_bits"]).n, version=version)
        prog, ids = decode_opaque(era["levels"][str(ell)]["program"])
        stubs.extend(x for x in ids if x not in stubs)
        return prog
    program = program_at(levels[0])
    if PO.program_sha512(program) != rec["program"]:
        raise Gap(f"the window's program is {PO.program_sha512(program)[:16]}, the record's {rec['program'][:16]}")
    tid = era["tid"] if era is not None else P.template_id(t)
    query = PO.template_instance_query(tid)
    partition, prov["partition"] = partition_form(program, query, rec, [tid])
    port = next(r["port"] for r in rec["roots"] if r["port"] != "W")

    def ports_at(ell: int, prog, part: dict | None, form: tuple[str, str], wb: str) -> dict:
        if era is not None:
            return era["levels"][str(ell)]["ports"]
        pd = P.digest(part if part is not None else P.partition(prog, query))
        wdoc = {"template": t.id, "port": "W", "schema": L.ROW_SCHEMA["bf16", ROLE_W], "leaves": pv["w_rows"],
                "deployment": pv["deployment"]}
        return {port: {"leaves": pv["entries"], "domain": served_domain(form, PO.program_sha512(prog), pd, port,
                                                                       SCHEMA_HM96_SHA512_ROW_V2, pv["entries"],
                                                                       pv["context"])},
                "W": {"leaves": pv["w_rows"], "domain": _domain(SERVED_BINDINGS[wb](SERVED_ZK_W_DOMAIN, wdoc), pv["w_rows"])}}
    if era is not None:
        form, wb = SERVED_FORMS[0], "sha512"
        ports = ports_at(levels[0], program, partition, form, wb)
        prov["ports"] = "the era's `served_commit.ports`"
        if not matches(ports, rec):
            prov["ports"] = "GAP: the era's ports are not the record's"
    else:
        found = [(f, wb) for f in SERVED_FORMS for wb in SERVED_BINDINGS
                 if matches(ports_at(levels[0], program, partition, f, wb), rec)]
        form, wb = found[0] if found else (SERVED_FORMS[0], "sha512")
        ports = ports_at(levels[0], program, partition, form, wb)
        prov["ports"] = (f"served_domain {form[0]} {form[1]} ({port}), W's domain {wb}" if found else
                         "GAP: no derivation matches the record's ports; today's")
    law = law_at(levels[0])
    first = run.json(outcomes[0])
    service = first.get("format") == "verity/proof-service/outcome/v0"
    scheme = SCHEME if service else SERVED_ZK_SCHEME
    prov["scheme"] = "the proof system's `SCHEME`" if service else "served_commit's `SCHEME` (before the proof system)"
    p = params(program, query, law, ports, None, delta_of(pv["delta"]), partition=partition, scheme=scheme)
    base = None
    for path in outcomes:
        body = run.json(path)
        a = body["audit"] if service else body
        if a is None:
            raise Gap(f"{path}: an outcome without an audit (late)")
        if service and body.get("order") != 0:
            raise Gap(f"{path}: order {body.get('order')}, a log the run did not keep")
        if R.digest(a["registration"]) != R.digest(rec):
            raise Gap(f"{path}: not the request's registration")
        sessions = list(body.get("sessions") or []) if service else []
        e = evidence(a, instances=len(a["proved_units"]), sessions=sessions, session_roots=None, served=None)
        name = path.split("/")[1]
        honest = name in ("outcome-honest", "outcome-st-honest")
        c = Case(run.id, "served-window", first.get("format", ""), name, "audit" if honest else "negative", program, p,
                 rec, e, recorded_audit(a, path),
                 {**prov, "evidence": f"{path}; instances the proved units; sessions "
                                      + ("the delivered ones" if service else "none (before the proof system)")})
        if stubs:
            c.provenance["opaque"] = stubs
        base = base or (c if name == "outcome-honest" else None)
        yield c
    if base is None:
        raise Missing("out/outcome-honest/audit.json")
    checks = run.json("out/summary.json").get("checks") or {}
    ours: list[str] = []
    if raised is not None and "receive-raised" in checks:
        ell = levels[1]
        prog_r = program_at(ell)
        part_r = with_program(partition, prog_r)
        out = run.path("out/receive-raised.out").read_text() if run.has("out/receive-raised.out") else ""
        got = json.loads(out.split(" ", 1)[1]) if out.startswith("REFUSED {") else {}
        ours = [x for x in got.get("codes") or [] if re.match(r"R\d", x)]
        rv = recorded("reject", "registration", ours, "out/receive-raised.out (its R codes)") if ours else \
            recorded("reject", None, None, "out/receive-raised.out")
        yield base.but("receive-raised", "negative" if ours else "analog", rv, program=prog_r, rec=raised,
                       params={"program": descriptor(prog_r), "partition": part_r, "law": law_at(ell),
                               "ports": ports_at(ell, prog_r, part_r, form, wb)},
                       provenance={"change": f"out/registration-raised.json under the verifier's copies at level {ell}"})
    if "receive-foreign" in checks and run.has("out/other/registration.json"):
        yield base.but("receive-foreign", "analog", recorded("reject", None, None, "out/receive-foreign.out"),
                       rec=run.json("out/other/registration.json"),
                       provenance={"change": "out/other/registration.json under the request's copies"})
    for name, v in sorted(checks.items()):
        if v.get("expected") not in ("REFUSED", "REJECT") or name.startswith("outcome-"):
            continue
        if name == "receive-raised" and ours:
            run.outside.append({"run": run.id, "driver": "served-window", "name": "receive-raised (public-value)",
                                "why": "PoUW's rule (`public_problem`) beside one-stage's R2 and R3, which are replayed"})
            continue
        why = next((w for pat, w in SERVED_OUTSIDE if re.fullmatch(pat, name)), "UNHANDLED: the replay has no case for it")
        run.outside.append({"run": run.id, "driver": "served-window", "name": name, "why": why})


#: served_zk's first window's registered scheme, law and δ (`served_commit.SCHEME`, `LAW`, `DELTA` before `params.json`)
SERVED_ZK_V0_SCHEME = "verity/pouw/served-zk/hm96-sha512+frame-v3-sha512/v0"
SERVED_ZK_V0_LAW = {"k": 1, "law": "subset"}
SERVED_ZK_V0_DELTA = 0.05


def served_audit(run: Run) -> Iterator[Case]:
    """served_zk's first window (`served_commit.py` and `job.sh` before the verifier's `params.json`): each side's
    (`request`, `other`) commitment of one served call, its registration received under the verifier's copies of that
    commitment (the tile template's population over the statement's tiles, `Q_template_instance`, `subset:1`, the scheme,
    each side's served domain over its row count and the request's run, no leaf layer), one Lean draw (`draw.json`), and
    `audit-<case>.json`, the audit record of the request's registration (`foreign-commit`: the other's), with one instance,
    no sessions and none of the later checks' inputs. Before `served_commit.run_of` the verifier took the run its served
    domains bind from the record's own window, not the request's binding, so the other request's registration passed R3
    (`receive-foreign` RECEIVED, the run's failed check)."""
    from verity.core.protocols.pouw.circuit import leaves as L, pc8
    from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X
    commit = run.commit()
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:benchmarks/pouw/served_zk/served_commit.py"],
                         capture_output=True, text=True) if commit else None
    if src is None or src.returncode != 0:
        raise Missing(f"benchmarks/pouw/served_zk/served_commit.py at {commit} (git history)")
    binding_run = "def run_of" in src.stdout
    prov0: dict = {"commit": commit, "program": "P.population_program(pc8 tile template, the statement's tiles, the record's "
                                                "version)",
                   "scheme": "served_commit's `SCHEME` (v0)", "run": "the served request's binding (`run_of`)" if binding_run
                   else "the record's own window (before `run_of`)"}
    forms: dict[str, tuple[str, str]] = {}

    def copies(side: str, rec: dict, run_id: str | None = None) -> tuple[Any, dict, dict, dict]:
        c = run.json(f"out/{side}/commitment.json")
        b, (rows, n) = c["binding"], c["statement"]["tiles"]
        t = pc8.templates(b["shape"][1], (1, 1), "sm120", hidden=True)["tile"]
        program = own_population(lambda v: P.population_program(t, rows * n, version=v), rec["program"])
        query = P.template_instance_query(t)
        part = P.partition(program, query)
        if run_id is None:
            run_id = f"{b['run']}/{b['manifest_sha256'][:16]}/matmul-{b['matmul']}" if binding_run else rec["window"].get("run", "")
        layout = {"a": (L.ROW_SCHEMA["fp8-p", ROLE_X], rows), "b": (L.ROW_SCHEMA["fp8-p", ROLE_W], n)}
        if side not in forms:
            own = run.json(f"out/{side}/registration.json")
            forms[side] = next((f for f in SERVED_FORMS if matches(served_ports(f, part, layout, run_id), own)), SERVED_FORMS[0])
        return program, query, served_ports(forms[side], part, layout, run_id), {"run_id": run_id, "form": forms[side]}
    audits = sorted(f["path"] for f in run.files if re.fullmatch(r"out/audit-[^/]+\.json", f["path"]))
    if not audits:
        raise Missing("out/audit-*.json")
    base = None
    for path in audits:
        name = path[len("out/audit-"):-len(".json")]
        side = "other" if name == "foreign-commit" else "request"
        a = run.json(path)
        rec = a["registration"]
        if R.digest(rec) != R.digest(run.json(f"out/{side}/registration.json")):
            raise Gap(f"{path}: not out/{side}/registration.json")
        program, query, ports, how = copies(side, rec)
        if PO.program_sha512(program) != rec["program"]:
            raise Gap(f"out/{side}/commitment.json's tiles give {PO.program_sha512(program)[:16]}, the record's {rec['program'][:16]}")
        p = params(program, query, SERVED_ZK_V0_LAW, ports, None, delta_of(SERVED_ZK_V0_DELTA), scheme=SERVED_ZK_V0_SCHEME)
        e = evidence(a, instances=1, sessions=[], session_roots=None, served=None)
        prov = {**prov0, "copies": f"out/{side}/commitment.json", "ports": f"served_domain {how['form'][0]} {how['form'][1]} "
                f"(run {how['run_id']})", "evidence": f"{path}; one instance, the draw's unit; no sessions"}
        c = Case(run.id, "served-audit", rec["format"], f"audit-{name}", "audit" if name in ("honest", "st-honest") else "negative",
                 program, p, rec, e, recorded_audit(a, path), prov)
        base = base or (c if name == "honest" else None)
        yield c
    if base is None:
        raise Missing("out/audit-honest.json")
    checks = run.json("out/summary.json").get("checks") or {}
    if "receive-foreign" in checks and run.has("out/receive-foreign.out"):
        other = run.json("out/other/registration.json")
        out = run.path("out/receive-foreign.out").read_text()
        got = json.loads(out.split(" ", 1)[1]) if " {" in out else {}
        received = out.startswith("RECEIVED")
        rv = recorded("accept", source="out/receive-foreign.out") if received else \
            recorded("reject", "registration", got.get("codes"), "out/receive-foreign.out")
        _, _, ports, how = copies("request", other)
        yield base.but("receive-foreign", "negative", rv, rec=other, params={"ports": ports},
                       evidence={"receipt": run.json("out/other/receipt.json")} if received else None,
                       provenance={"change": f"out/other/registration.json under the request's copies, its served domains "
                                             f"over run {how['run_id']}" + ("; the receipt the verifier writes for it (out/"
                                                                            "other/receipt.json), the honest audit otherwise"
                                                                            if received else "")})
        if not binding_run:
            _, _, ports_r, how_r = copies("request", other, f"{base.rec['window']['run']}")
            yield base.but("receive-foreign (the request's run)", "analog", recorded("reject", None, None, "out/summary.json "
                                                                                     "(receive-foreign expected REFUSED)"),
                           rec=other, params={"ports": ports_r},
                           provenance={"change": f"out/other/registration.json under the request's copies over the "
                                                 f"request's run {how_r['run_id']}, as `run_of` derives them"})
    for name, v in sorted(checks.items()):
        if v.get("expected") not in ("REFUSED", "REJECT") or name.startswith("audit-") or name == "receive-foreign":
            continue
        why = next((w for pat, w in SERVED_OUTSIDE if re.fullmatch(pat, name)), "UNHANDLED: the replay has no case for it")
        run.outside.append({"run": run.id, "driver": "served-audit", "name": name, "why": why})


def session_digests(run: Run, session: str) -> list[str]:
    return sorted(f["sha256"] and sha512_file(run.path(f["path"])) for f in run.files
                  if f["path"].startswith(f"{session}/sessions/") and f["path"].endswith("/session.json"))


def population(members: list[tuple[Any, int]], version: int = P.POPULATION_VERSIONS[0]):
    """The population program over `(template, n)` in order at `version`, and its query, as
    `serving_rows.population_program` and A6 build them: `TemplateInstances` and `Q_template_instance` for one template,
    `TemplatePopulations` and `Q_template_instances` for several."""
    if len(members) == 1:
        return P.population_program(*members[0], version=version), PO.template_instance_query(members[0][0])
    return P.mixed_population_program(members, version=version), PO.template_instances_query([t for t, _ in members])


def own_members(members: list[tuple[Any, int]], want: str):
    """`population(members)` at the version whose program is the record's (`own_population`), and that version."""
    program = own_population(lambda v: population(members, v)[0], want)
    version = population_version(program)
    return program, population(members, version)[1], version


def sessions_evidence(run: Run, a: dict, sessions: list[tuple[str, str]]) -> tuple[int, dict, list[str]]:
    """Instances and session roots from each `(session dir, port prefix)`'s `prove.log` LIVE line (roots named
    `<prefix>/<port>`), and the sessions' SHA-512 (the integrity profile's, else each `session.json`'s)."""
    instances, roots, digests = 0, {}, []
    for sdir, prefix in sessions:
        live = live_line(run.path(f"{sdir}/prove.log")) or {}
        instances += live.get("instances", 0)
        roots.update({f"{prefix}/{p}": r for p, r in (live.get("roots") or {}).items()})
        digests += session_digests(run, sdir)
    return instances, roots, (a.get("integrity_profile") or {}).get("evidence", {}).get("sessions") or digests


#: A5's and A6's recorded refusals that are not the protocol's checks, with why
A5_OUTSIDE = {"registration-of-another-rank (source)": "A5's check of the window's link (`source`: the stand-in's run, "
                                                       "`of` and `link` against the verifier's own), beside R1-R7"}
A6_OUTSIDE = {"statement-binds-another-partition": "A6's statement checks (`statement_checks`: META's partition binding)",
              "lean-verify-under-another-program": "flock-verify (`verify --zk --partition --program`) on a session under "
                                                   "another program's partition object; the protocol reads its verdict"}


def a5(run: Run) -> Iterator[Case]:
    """A5 (`a5.py`), and each rank's A5 of a TP Commit's stand-in (`out/standin-<I>/rank<r>/`): a deferred
    Commit's window, its registration over serving's rows of the Commit's values. The verifier's copies are its window of
    record as the run wrote it: the program its `program.json` defines (the record's; each template rebuilt from it into
    the population program, a primitive the current registry lacks an opaque stand-in), the canonical query over the members
    `summary.json` names, its law and δ, the served domains over the record's ports under the run root of record
    (`summary.json`'s `run_root`), and the leaf layer over each member's public body (`serving_rows.leaf_layer`, each
    file's header line dropped). One draw by the subset law, each member served its share."""
    bases = sorted(f["path"].rsplit("/", 1)[0] for f in run.files
                   if re.fullmatch(r"out/(a5|standin-[^/]+/rank\d+)/audit\.json", f["path"]))
    if not bases:
        raise Missing("out/a5/audit.json")
    for base in bases:
        yield from a5_audit(run, base)


def a5_audit(run: Run, base: str) -> Iterator[Case]:
    a = run.json(f"{base}/audit.json")
    summary = run.json(f"{base}/summary.json")
    rec = a["registration"]
    stored, stubs = decode_opaque(run.json(f"{base}/program.json"))
    if PO.program_sha512(stored) != rec["program"]:
        raise Gap(f"{base}/program.json is {PO.program_sha512(stored)[:16]}, the record's program {rec['program'][:16]}")
    members = sorted(summary["members"], key=lambda m: m["base"])
    tids = [m["template"] for m in members]
    if any(t not in stored.defs for t in tids):
        raise Gap(f"{base}/program.json defines no template {[t for t in tids if t not in stored.defs]}")
    defs = [(stored.defs[m["template"]], m["n"]) for m in members]
    program, query, version = own_members(defs, rec["program"])
    prov: dict = {"program": f"the population program (v{version}) over the templates {base}/program.json defines"
                             + (f" (opaque: {stubs})" if stubs else "")}
    if PO.program_sha512(program) != rec["program"]:
        raise Gap(f"the members' population program is {PO.program_sha512(program)[:16]}, the record's {rec['program'][:16]}")
    partition, prov["partition"] = partition_form(program, query, rec, tids)
    part = partition if partition is not None else P.partition(program, query)
    run_root = summary["run_root"]
    layout = {r["port"]: (ROW, r["leaves"]) for r in rec["roots"]}
    ports, how, form = choose_served(part, layout, run_root, rec)
    prov["ports"] = f"{how} over the record's ports and leaves, under {base}/summary.json's run_root"

    def leaf_layer_of(bodies: dict[str, bytes]) -> str:
        return hashlib.sha512(DT.canonical({t: hashlib.sha512(b).hexdigest() for t, b in bodies.items()}).encode()).hexdigest()
    bodies = {}
    for m in members:
        raw = run.path(f"{base}/m{m['k']}/pub-{m['n']}.bin").read_bytes()
        bodies[m["template"]] = raw[raw.index(b"\n") + 1:]
    leaf_layer = leaf_layer_of(bodies)
    prov["leaf_layer"] = f"serving_rows.leaf_layer over {base}/m<k>/pub-<n>.bin's bodies"
    counts = a.get("draw_counts") or {}
    instances, session_roots, sessions = sessions_evidence(
        run, a, [(f"{base}/session-m{m['k']}", m["template"]) for m in members if counts.get(m["template"])])
    if not summary.get("served_is_lean_draw"):
        raise Gap("the sessions' served units are not the Lean draw, and the run kept no other copy of them")
    served = sorted(a["draw"]["units"])
    law = summary["law"]
    p = params(program, query, law, ports, leaf_layer, delta_of(summary["delta"]), partition=partition)
    e = evidence(a, instances=instances, sessions=sessions, session_roots=session_roots, served=served)
    prov["evidence"] = (f"{base}/audit.json; instances and session roots from each session's prove.log LIVE line; "
                        "served = the Lean draw")
    c = Case(run.id, "a5", rec["format"], f"{base} audit", "audit", program, p, rec, e,
             recorded_audit(a, f"{base}/audit.json"), prov)
    yield c
    src = f"{base}/negatives.json"
    n = rec["population"]
    draw, proved = a["draw"], sorted(a["proved_units"])
    fewer, _ = population(R2.resized(defs), version)
    first = members[0]["template"]
    flipped_body = bodies[first][:-1] + bytes([bodies[first][-1] ^ 1])
    cases: dict[str, Callable[[dict], Case]] = {
        "registration-partition-mismatch (R2)": lambda v: c.but(
            "registration-partition-mismatch (R2)", "negative", recorded_codes(v, "registration", src),
            program=fewer, params={"program": descriptor(fewer), "partition": with_program(partition, fewer)}),
        "domains-bound-to-another-run-root (R3)": lambda v: c.but(
            "domains-bound-to-another-run-root (R3)", "negative", recorded_codes(v, "registration", src),
            params={"ports": served_ports(form, part, layout, "00" * 32)},
            provenance={"change": "the served domains under the run root 00…00"}),
        "public-body-not-the-registered-leaf-layer (R4)": lambda v: c.but(
            "public-body-not-the-registered-leaf-layer (R4)", "negative", recorded_codes(v, "registration", src),
            params={"leaf_layer": leaf_layer_of({**bodies, first: flipped_body})},
            provenance={"change": f"the leaf layer over {first}'s body with its last byte flipped"}),
        "draw-outside-the-partition": lambda v: c.but(
            "draw-outside-the-partition", "negative", recorded_codes(v, "draw", src),
            evidence={"draw": {**draw, "units": draw["units"][:-1] + [n]}}),
        "proved-set-short-of-the-draw": lambda v: c.but(
            "proved-set-short-of-the-draw", "negative", _proved_check(v, src),
            evidence={"proved_units": proved[1:], "instances": len(proved) - 1}),
    }
    for x in _negatives(run, c, run.json(src), src, cases, base_instances=len(proved), session_roots=session_roots,
                        served=served, outside=A5_OUTSIDE):
        yield replace(x, name=f"{base} {x.name}")


def a6(run: Run) -> Iterator[Case]:
    """A6 (`a6.py`): the W8A8 coordinates of an int8 decoder layer, one template per K, staged as members of at most
    `chunk_cap` units, sampled by the stratified law. The verifier's copies are A6's own: the population program and
    canonical `Q_template_instances` over `ScaledMmS8Coordinate` at each K of `summary.json`'s `members_spec` (the current
    Definition, else the templates the run's `program.json` defines), the law `stratified:k`, each member port's set-bound
    domain over its member's set (A6's row layout: x row-seg, w and o0 row-v2), and the leaf layer the SHA-512 of the
    map from member to its public file's SHA-512."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import gemm as OG
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2
    base = "out/a6"
    a = run.json(f"{base}/audit.json")
    summary = run.json(f"{base}/summary.json")
    rec = a["registration"]
    spec = [(int(k), int(x)) for k, x in (m.split(":") for m in summary["members_spec"].split(","))]
    defs = [(bind(OG.ScaledMmS8Coordinate, K=K, **OG.GEMM_SM120_S8), n) for K, n in spec]
    program, query, version = own_members(defs, rec["program"])
    prov: dict = {"program": f"P.mixed_population_program(ScaledMmS8Coordinate at each K of members_spec, version={version}), today's"}
    if PO.program_sha512(program) != rec["program"]:
        stored, stubs = decode_opaque(run.json(f"{base}/program.json"))
        tids = [P.template_id(d) for d, _ in defs]
        if PO.program_sha512(stored) != rec["program"] or any(t not in stored.defs for t in tids):
            raise Gap(f"neither today's A6 program nor {base}/program.json is the record's {rec['program'][:16]}")
        defs = [(stored.defs[t], n) for t, (_, n) in zip(tids, defs)]
        program, query, version = own_members(defs, rec["program"])
        prov["program"] = f"the population program over the templates {base}/program.json defines" + \
            (f" (opaque: {stubs})" if stubs else "")
    tids = [P.template_id(d) for d, _ in defs]
    partition, prov["partition"] = partition_form(program, query, rec, tids)
    k = int(summary["k"])
    law = DR.parse_law(f"stratified:{k}")
    def pub_of(m: dict) -> Path:
        K, j = re.fullmatch(r"s8-coord-k(\d+)(?:-c(\d+))?", m["name"]).groups()
        return run.path(f"{base}/stage-k{K}/{f'c{j}/' if j is not None else ''}pub-{m['n']}.bin")
    ports: dict = {}
    for m in summary["members"]:
        layout = {f"{m['name']}/{port}": (schema, m["n"]) for port, schema in (("x", SEG), ("w", V2), ("o0", V2))}
        set_name = m.get("set") or json.loads(pub_of(m).read_bytes().split(b"\n", 1)[0])["set"]
        ports.update(stand_in_ports(set_name, "", m["n"], layout, "verity/flock-circuit/binding/v2", False,
                                    prefix=f"{m['name']}/"))
    prov["ports"] = ("a6.frame_ports over each member's set (summary.json's, else its public file's; x row-seg, w and o0 "
                     "row-v2)" if matches(ports, rec) else "GAP: a6.frame_ports' domains are not the record's")
    publics = {m["name"]: sha512_file(pub_of(m)) for m in summary["members"]}
    leaf_layer_of = lambda pubs: hashlib.sha512(DT.canonical(pubs).encode()).hexdigest()  # noqa: E731
    leaf_layer = leaf_layer_of(publics)
    prov["leaf_layer"] = f"SHA-512 of the map from member to the SHA-512 of {base}/stage-k<K>[/c<j>]/pub-<n>.bin"
    counts = a.get("draw_counts") or {}
    instances, session_roots, sessions = sessions_evidence(
        run, a, [(f"{base}/session-{m['name']}", m["name"]) for m in summary["members"] if counts.get(m["name"])])
    if summary.get("served_is_lean_draw"):
        served, served_from = sorted(a["draw"]["units"]), "the Lean draw"
    else:
        served = sorted(m["base"] + u for m in summary["members"] for f in run.files
                        if re.fullmatch(rf"{base}/session-{re.escape(m['name'])}/sessions/[^/]+/session\.json", f["path"])
                        for u in (run.json(f["path"]).get("unit_draw") or {}).get("units", []))
        served_from = "each member's base plus its sessions' session.json unit_draw (the run's served units)"
    p = params(program, query, law, ports, leaf_layer, delta_of(summary["delta"]), partition=partition)
    e = evidence(a, instances=instances, sessions=sessions, session_roots=session_roots, served=served)
    prov["evidence"] = (f"{base}/audit.json; instances and session roots from each session's prove.log LIVE line; "
                        f"served = {served_from}")
    c = Case(run.id, "a6", rec["format"], "audit", "audit", program, p, rec, e, recorded_audit(a, f"{base}/audit.json"), prov)
    yield c
    src = f"{base}/negatives.json"
    n_all = rec["population"]
    draw, proved = a["draw"], sorted(a["proved_units"])
    drawn = set(draw["units"])
    fewer, _ = population(R2.resized(defs), version)
    ranges = P.ranges(program, query)
    first = summary["members"][0]["name"]
    cases: dict[str, Callable[[dict], Case]] = {
        "registration-partition-mismatch (R2)": lambda v: c.but(
            "registration-partition-mismatch (R2)", "negative", recorded_codes(v, "registration", src), program=fewer,
            params={"program": descriptor(fewer), "partition": with_program(partition, fewer)}),
        "registration-query-drops-a-template (R2)": lambda v: c.but(
            "registration-query-drops-a-template (R2)", "negative", recorded_codes(v, "registration", src),
            params={"query": PO.template_instances_query(tids[:-1]),
                    "partition": with_program(partition, program, drop_template=True)}),
        "registration-law-another-k (R1)": lambda v: c.but(
            "registration-law-another-k (R1)", "negative", recorded_codes(v, "registration", src),
            params={"law": DR.parse_law(f"stratified:{k + 1}")}),
        "registration-law-subset-not-stratified (R1)": lambda v: c.but(
            "registration-law-subset-not-stratified (R1)", "negative", recorded_codes(v, "registration", src),
            params={"law": DR.parse_law(f"subset:{k}")}),
        "public-file-not-the-registered-leaf-layer (R4)": lambda v: c.but(
            "public-file-not-the-registered-leaf-layer (R4)", "negative", recorded_codes(v, "registration", src),
            params={"leaf_layer": leaf_layer_of({**publics, first: "00" * 64})},
            provenance={"change": f"the leaf layer with {first}'s public file digest 00…00"}),
        "draw-outside-the-partition": lambda v: c.but(
            "draw-outside-the-partition", "negative", recorded_codes(v, "draw", src),
            evidence={"draw": {**draw, "units": draw["units"][:-1] + [n_all]}}),
        "proved-set-short-of-the-draw": lambda v: c.but(
            "proved-set-short-of-the-draw", "negative", _proved_check(v, src),
            evidence={"proved_units": proved[1:], "instances": len(proved) - 1}),
        "proved-unit-not-drawn": lambda v: c.but(
            "proved-unit-not-drawn", "negative", _proved_check(v, src),
            evidence={"proved_units": sorted([next(u for u in range(n_all) if u not in drawn), *proved[1:]]),
                      "instances": instances}),
    }
    if len(ranges) > 1:
        (_ta, a_lo, a_n), (_tb, b_lo, b_n) = ranges[:2]
        drop = next(u for u in draw["units"] if a_lo <= u < a_lo + a_n)
        add = next((u for u in range(b_lo, b_lo + b_n) if u not in drawn), None)
        if add is not None:
            cases["draw-moves-a-unit-across-strata"] = lambda v: c.but(
                "draw-moves-a-unit-across-strata", "negative", recorded_codes(v, "draw", src),
                evidence={"draw": {**draw, "units": sorted((drawn - {drop}) | {add})}})
    yield from _negatives(run, c, run.json(src), src, cases, base_instances=len(proved), session_roots=session_roots,
                          served=served, outside=A6_OUTSIDE)


#: A7's recorded refusals that are not the protocol's checks, by the suffix A7 names each with, and why
A7_OUTSIDE = (
    (r"\((PlanMatches|Carries|L4)\)$|free \(|\(dummy append or re-key\)$",
     "the unit's own clause: a witness the prover keeps consistent everywhere, refused only by the unit's ok row (0)"),
    (r"\(mapped\)$|^ledger-on-", "the remote bandwidth audit's link and ledger (`UnitsLink.mapped`, `audit.ledger`)"),
    (r"\(binding\)$", "A7's bindings of each unit's rows to the canonical commitments (`bindings`), beside R1-R7"),
    (r"\((v0 )?bridge[^)]*\)$", "v0's bridge between the remote session and the canonical rows (`v0_bridge`)"),
    (r"\(opening\)$", "A7's openings of each drawn unit's public and ok rows (`openings`)"),
    (r"\(Lean roots\)$", "flock-verify recomputing the roots from the public file"),
    (r"\((receipt|decode)[^)]*\)$|\((plan row|receipt path)\)$",
     "check 3's units and links, on cursor/zk-check3-unit-c7a9 (not on main)"),
)
A7_CHECK3 = ("check 3's audit record, over BlockReceiptCheck_v1 and BlockDecodeCheck_v1, whose Definitions are only on "
             "cursor/zk-check3-unit-c7a9 (not on main), so the replay can't rebuild the verifier's copies")


def a7(run: Run) -> Iterator[Case]:
    """A7 (`a7.py`): a stand-in requirement's plan and memory-map units (`requirement_units`), one member per template
    with a unit per step, sampled by the stratified law, beside the canonical commitments their rows bind to. The
    verifier's copies are A7's own: the population program and canonical `Q_template_instances` over `PlanMatch` and
    `MemoryMapStep` at the statics `summary.json` records (the current Definitions, else the templates the run's
    `program.json` defines), the law `stratified:k`, each member port's set-bound domain over its set (every row row-v2;
    the ports the Definition's rows and `o0`), each canonical commitment's over the rows it publishes, and the leaf
    layer the SHA-512 of the map from each member and `canon/<kind>` to its public file's SHA-512. Check 3's audit
    (`check3/`) is outside (`A7_CHECK3`)."""
    from types import SimpleNamespace

    from verity.ml.circuits import requirement_units as RU
    from verity.ml.circuits.defs import bind
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
    base, tag = "out/a7", "verity/flock-circuit/binding/v2"
    a = run.json(f"{base}/audit.json")
    summary = run.json(f"{base}/summary.json")
    rec = a["registration"]
    sp, sm = summary["standin"]["plan_statics"], summary["standin"]["map_statics"]
    steps = int(summary["steps"])
    defs = [(bind(RU.PlanMatch, **sp), steps), (bind(RU.MemoryMapStep, **sm), steps)]
    program, query, version = own_members(defs, rec["program"])
    prov: dict = {"program": f"P.mixed_population_program(PlanMatch, MemoryMapStep at summary.json's statics, version={version}), today's"}
    members = sorted(summary["members"], key=lambda m: m["base"])
    if PO.program_sha512(program) != rec["program"]:
        stored, stubs = decode_opaque(run.json(f"{base}/program.json"))
        tids = [m["template"] for m in members]
        if PO.program_sha512(stored) != rec["program"] or any(t not in stored.defs for t in tids):
            raise Gap(f"neither today's A7 program nor {base}/program.json is the record's {rec['program'][:16]}")
        defs = [(stored.defs[t], n) for t, (_, n) in zip(tids, defs)]
        program, query, version = own_members(defs, rec["program"])
        prov["program"] = f"the population program over the templates {base}/program.json defines" + \
            (f" (opaque: {stubs})" if stubs else "")
    tids = [P.template_id(d) for d, _ in defs]
    partition, prov["partition"] = partition_form(program, query, rec, tids)
    k = int(summary["k"])
    law = DR.parse_law(f"stratified:{k}")
    rows = {"plan": RU.plan_rows(SimpleNamespace(**sp)), "map": RU.memory_map_rows(SimpleNamespace(**sm))}
    ports: dict = {}
    publics: dict = {}
    for m in members:
        kind, n = m["name"].split("-")[0], m["n"]
        layout = {f"{m['name']}/{r}": (V2, n) for r in [r for r, _b, _ps in rows[kind]] + ["o0"]}
        ports.update(stand_in_ports(f"unit-a7-{m['name']}", "", n, layout, tag, False, prefix=f"{m['name']}/"))
        publics[m["name"]] = sha512_file(run.path(f"{base}/stage-{kind}/c0/pub-{n}.bin"))
    for kind in sorted(re.fullmatch(rf"{base}/canon-([a-z]+)\.bc", f["path"]).group(1) for f in run.files
                       if re.fullmatch(rf"{base}/canon-[a-z]+\.bc", f["path"])):
        path = run.path(f"{base}/canon-{kind}.bc")
        n = path.stat().st_size // 128
        ports.update(stand_in_ports(f"a7-canon-{kind}", "", n, {f"canon/{kind}": (V2, n)}, tag, False, prefix="canon/"))
        publics[f"canon/{kind}"] = sha512_file(path)
    prov["ports"] = ("each member's set-bound domains over unit-a7-<member> (the Definition's rows and o0) and each "
                     "canon-<kind>.bc's over a7-canon-<kind>" if matches(ports, rec) else
                     "GAP: A7's member and canonical domains are not the record's")
    leaf_layer = hashlib.sha512(codec.canonical_json(publics)).hexdigest()
    prov["leaf_layer"] = (f"SHA-512 of the map from member to {base}/stage-<kind>/c0/pub-<n>.bin's SHA-512 and from "
                          "canon/<kind> to canon-<kind>.bc's")
    counts = a.get("draw_counts") or {}
    instances, session_roots, sessions = sessions_evidence(
        run, a, [(f"{base}/session-{m['name']}", m["name"]) for m in members if counts.get(m["name"])])
    if not summary.get("served_is_lean_draw"):
        raise Gap("the sessions' served units are not the Lean draw, and the run kept no other copy of them")
    served = sorted(a["draw"]["units"])
    p = params(program, query, law, ports, leaf_layer, delta_of(summary["delta"]), partition=partition)
    e = evidence(a, instances=instances, sessions=sessions, session_roots=session_roots, served=served)
    prov["evidence"] = (f"{base}/audit.json; instances and session roots from each session's prove.log LIVE line; "
                        "served = the Lean draw")
    c = Case(run.id, "a7", rec["format"], "audit", "audit", program, p, rec, e, recorded_audit(a, f"{base}/audit.json"), prov)
    yield c
    src = f"{base}/negatives.json"
    n_all = rec["population"]
    draw, proved = a["draw"], sorted(a["proved_units"])
    fewer, _ = population(R2.resized(defs), version)
    cases: dict[str, Callable[[dict], Case]] = {
        "registration-partition-mismatch (R2)": lambda v: c.but(
            "registration-partition-mismatch (R2)", "negative", recorded_codes(v, "registration", src), program=fewer,
            params={"program": descriptor(fewer), "partition": with_program(partition, fewer)}),
        "registration-law-another-k (R1)": lambda v: c.but(
            "registration-law-another-k (R1)", "negative", recorded_codes(v, "registration", src),
            params={"law": DR.parse_law(f"stratified:{k // 2 if k > 1 else k + 1}")}),
        "registration-without-the-canonical-commitments (R3)": lambda v: c.but(
            "registration-without-the-canonical-commitments (R3)", "negative", recorded_codes(v, "registration", src),
            params={"ports": {q: w for q, w in ports.items() if not q.startswith("canon/")}}),
        "draw-outside-the-partition": lambda v: c.but(
            "draw-outside-the-partition", "negative", recorded_codes(v, "draw", src),
            evidence={"draw": {**draw, "units": draw["units"][:-1] + [n_all]}}),
    }
    neg = run.json(src)
    outside = {x: next(why for pat, why in A7_OUTSIDE if re.search(pat, x)) for x in neg
               if x not in cases and any(re.search(pat, x) for pat, _ in A7_OUTSIDE)}
    yield from _negatives(run, c, neg, src, cases, base_instances=len(proved), session_roots=session_roots,
                          served=served, outside=outside)
    if run.has(f"{base}/check3/audit.json"):
        run.outside.append({"run": run.id, "driver": "a7", "name": f"{base}/check3/audit.json", "why": A7_CHECK3})


DRIVERS: dict[str, Callable[[Run], Iterator[Case]]] = {"a0": a0, "a0-lean": a0_lean, "a2": a2, "a3": a3, "a4": a4,
                                                       "a5": a5, "a6": a6, "a7": a7,
                                                       "registered-gemm": registered_gemm, "served-audit": served_audit,
                                                       "served-window": served_window, "serving-bundle": serving_bundle,
                                                       "a2-unaudited": a2_unaudited, "a4-unaudited": a4_unaudited}
#: the runs that recorded no verdict, read after the others (a serving bundle reads under their copies)
READS_NOTHING = {"no-registration-record": "run/summary.json's windows is empty: the run timed the registered GEMM's "
                                           "sessions (run/zk/, M0 and zk, W registered in the Lean statement only) under "
                                           "no window, so it holds no registration record, the commitment OneStage.check "
                                           "reads first, and no draw"}
UNRECORDED = {"serving-bundle", "a2-unaudited", "a4-unaudited", "no-recorded-verdict", *READS_NOTHING}


# --- main --------------------------------------------------------------------------------------------------------------

SKIP = re.compile(r"(\.tar$|/archive/|/tables/|(^|/)circuit[^/]*\.(txt|bin)$|(^|/)inst-[0-9]+\.bin$)")


def wanted(f: dict) -> bool:
    """A file the replay may read: a record, a log or a small public file, never a circuit, a table or an archive."""
    return not SKIP.search(f["path"]) and f["bytes"] <= 8_000_000


def research_fetch(art: str, to: Path, paths: list[str] | None = None) -> None:
    subprocess.run(["research", "data", "fetch", art, "--to", str(to), *[x for p in paths or [] for x in ("--path", p)]],
                   check=True, capture_output=True, text=True)


def file_list(art: str) -> list[dict]:
    p = subprocess.run(["research", "data", "fetch", art, "--list", "--json"], check=True, capture_output=True, text=True)
    return json.loads(p.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", required=True, help="the one-stage-run/v0 index (JSON with `runs`)")
    ap.add_argument("--runs", required=True, help="the runs' files, one directory per run (the art's first 8 hex digits)")
    ap.add_argument("--fetch", action="store_true", help="fetch what --runs lacks from the store (`wanted` files, and each "
                                                         "distinct set.tar's manifests)")
    ap.add_argument("--lean", required=True, help="sampled-proofs-one-stage (the record and the public file go through "
                                                  "`one_stage.verifier`, which builds its own sampled-proofs-verifier)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="run ids (8 hex digits)")
    a = ap.parse_args()
    _, errors = catalog.load(("ops", "registry"))  # every Definition a recorded program may call
    if errors:
        raise ImportError("; ".join(errors))
    out = Path(a.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    runs_root = Path(a.runs)
    index = json.loads(Path(a.index).read_text())["runs"]
    rows = [r for r in index if a.only is None or r["art"][4:12] in a.only]
    lists: dict[str, list[dict]] = {}
    sets = Sets(runs_root / "_sets")
    for r in rows:
        rid = r["art"][4:12]
        lf = runs_root / "_lists" / f"{rid}.json"
        if not lf.exists():
            if not a.fetch:
                raise SystemExit(f"{lf}: no file list (--fetch)")
            lf.parent.mkdir(parents=True, exist_ok=True)
            lf.write_text(json.dumps(file_list(r["art"])))
        lists[rid] = json.loads(lf.read_text())
        d = runs_root / rid
        want = [f["path"] for f in lists[rid] if wanted(f) and not (d / f["path"]).exists()]
        if a.fetch and want:
            for i in range(0, len(want), 200):
                research_fetch(r["art"], d, want[i:i + 200])
    tars = {}
    for rid, fs in lists.items():
        for f in fs:
            if f["path"] == "inputs/set.tar":
                tars.setdefault(f["sha256"], rid)
    for sha, rid in sorted(tars.items()):
        if not sets.dir(sha).exists():
            tar = runs_root / rid / "inputs" / "set.tar"
            if not tar.exists():
                if not a.fetch:
                    raise SystemExit(f"{tar}: missing (--fetch)")
                research_fetch(next(r["art"] for r in rows if r["art"][4:12] == rid), runs_root / rid, ["inputs/set.tar"])
            sets.extract(tar, sha)
    programs, records = Programs(), Records()
    for f in sorted(runs_root.glob("*/**/program.json")):
        programs.add_file(f)
    for f in sorted([*runs_root.glob("*/**/audit.json"), *runs_root.glob("*/**/registration.json")]):
        records.add_file(f, runs_root)

    server = DT.Server([a.lean, "serve"])
    verifier = V.executable()
    results, runs_out, outside, audits = [], [], [], defaultdict(list)

    def replay(run: Run, info: dict) -> None:
        driver = info["driver"]
        if driver not in DRIVERS:
            info["status"] = driver
            if driver in READS_NOTHING:
                info["why"] = READS_NOTHING[driver]
            return
        cases: list[Case] = []
        try:
            cases.extend(DRIVERS[driver](run))
        except Missing as e:
            info["status"], info["missing"] = "missing-input", str(e)
        except Gap as e:
            info["status"], info["gap"] = "reconstruction-gap", str(e)
        else:
            info["status"] = "replayed"
        with open(out / "cases" / f"{run.id}.jsonl", "w") as f:
            for c in cases:
                case = c.case()
                try:
                    lean = lean_public(c) if c.public is not None else server.exchange(case)
                except DT.ProtocolError as err:
                    lean = {"verdict": "error", "reason": str(err)}
                try:
                    py = python(c)
                except Exception as err:  # noqa: BLE001 - a crash is a verdict here
                    py = {"verdict": "error", "reason": f"{type(err).__name__}: {err}"}
                try:
                    whole = whole_record(c)
                except Exception as err:  # noqa: BLE001
                    whole = {"agree": False, "error": f"{type(err).__name__}: {err}"}
                results.append(_row(c, lean, py, {**compare(c, lean, py), "record": whole}))
                f.write(json.dumps({"name": c.name, "r7": c.r7, "case": case}) + "\n")
                if c.kind == "audit" and c.public is None and driver not in UNRECORDED:
                    audits[R.digest(c.rec)].append(c)
        info["cases"] = len(cases)
        print(f"{run.id} {driver}: {info['status']} {len(cases)} cases", file=sys.stderr, flush=True)

    late: list[tuple[Run, dict]] = []
    for r in rows:
        rid = r["art"][4:12]
        run = Run(r, runs_root / rid, lists[rid], sets, programs, outside,
                  (lambda rel, art=r["art"], d=runs_root / rid: research_fetch(art, d, [rel])) if a.fetch else None, records,
                  audits)
        info = {"run": rid, "art": r["art"], "id": r.get("run"), "drivers": r.get("drivers"),
                "formats": r.get("registration_formats"), "python_verdicts": len(r.get("python_verdicts") or [])}
        info["driver"] = classify(run)
        runs_out.append(info)
        if info["driver"] in UNRECORDED:
            late.append((run, info))
        else:
            replay(run, info)
    for run, info in late:
        replay(run, info)
    server.close()
    report = {"format": "verity/one-stage-replay/v0", "index": a.index, "lean": a.lean,
              "verifier": {"path": str(verifier), "sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
                           "of_record": V.RECORD_VERDICT},
              "runs": runs_out, "cases": results, "outside": outside, "counts": counts(runs_out, results, outside)}
    report["summary"] = summary_lines(report["counts"])
    (out / "replay.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report["counts"], indent=1, sort_keys=True))
    print("\n".join(report["summary"]))
    return 0


def _row(c: Case, lean: dict | None, py: dict | None, cmp: dict) -> dict:
    return {"run": c.run, "driver": c.driver, "format": c.format, "name": c.name, "kind": c.kind, "r7": c.r7,
            "recorded": c.recorded, "lean": lean, "python": py, **cmp, "provenance": c.provenance,
            **({"stages": list(c.stages)} if c.stages != STAGES else {}), **({"public": True} if c.public is not None else {})}


def classify(run: Run) -> str:
    """The driver whose files the run holds, or why the replay reads none."""
    formats = run.row.get("registration_formats") or {}
    drivers = run.row.get("drivers") or []
    if formats and all(f == "verity/one-stage/registration/v0" for f in formats):
        return "neither-reads (registration v0)"
    if run.has("out/request/commitment.json") and not run.has("out/request/params.json") and \
            any(re.fullmatch(r"out/audit-[^/]+\.json", f["path"]) for f in run.files):
        return "served-audit"
    if not run.row.get("python_verdicts"):
        return unrecorded(run)
    if len(drivers) == 1 and drivers[0] in DRIVERS:
        return drivers[0]
    if run.has("out/flock-audit.argv.json"):
        return "a0-lean"
    if run.has("run/stage/registered.json"):
        return "registered-gemm"
    return "unhandled"


def unrecorded(run: Run) -> str:
    """The driver of a run that recorded no verdict, by the stages its files hold, or why neither side reads it."""
    if run.has("registration.json") and run.has("index.json"):
        return "serving-bundle"
    if run.has("out/a2/receipt.json") and not run.has("out/a2/audit.json"):
        return "a2-unaudited"
    if run.has("out/a4/summary.json") and not run.has("out/a4/audit.json"):
        return "a4-unaudited"
    if run.has("run/summary.json") and not run.json("run/summary.json").get("windows"):
        return "no-registration-record"
    return "no-recorded-verdict"


def _by_driver_format(runs: list[dict], results: list[dict], outcome: str) -> dict:
    by: dict = defaultdict(Counter)
    for r in results:
        by[f"{r['driver']} {r['format']}"][r.get(outcome) or "pending"] += 1
    for r in runs:
        if r["status"] != "replayed":
            by[f"{r['driver']} {','.join(sorted(r.get('formats') or {})) or '-'}"][r["status"]] += 1
    return {k: dict(v) for k, v in sorted(by.items())}


def counts(runs: list[dict], results: list[dict], outside: list[dict]) -> dict:
    """The runs that recorded a verdict against it (`runs` counts every run in the index), and those that recorded none
    as their own line (`no_recorded_verdict`, Lean against Python)."""
    mine = [r for r in results if r["driver"] not in UNRECORDED]
    none = [r for r in results if r["driver"] in UNRECORDED]
    none_runs = [r for r in runs if r["driver"] in UNRECORDED]
    return {"runs": len(runs), "cases": len(mine),
            "same_verdict": sum(1 for r in mine if r.get("lean_recorded") in ("same", "same-verdict")),
            "public": sum(1 for r in mine if r.get("public")),
            "records": _records(mine),
            "negatives": sum(1 for r in mine if r["kind"] == "negative"),
            "analogs": sum(1 for r in mine if r["kind"] == "analog"), "outside": len(outside),
            "lean_python": dict(Counter(r.get("lean_python", "pending") for r in mine)),
            "python_recorded": dict(Counter(r.get("python_recorded", "-") for r in mine)),
            "by_driver_format": _by_driver_format([r for r in runs if r["driver"] not in UNRECORDED], mine, "lean_recorded"),
            "no_recorded_verdict": {
                "runs": len(none_runs), "cases": len(none), "records": _records(none),
                "lean_python": dict(Counter(r.get("lean_python", "pending") for r in none)),
                "negatives": sum(1 for r in none if r["kind"] == "negative"),
                "registration_only": sum(1 for r in none if r.get("stages") == list(REGISTRATION)),
                "recorded_registration": dict(Counter(r["lean_recorded"] for r in none if r.get("lean_recorded"))),
                "reads_nothing": [{"run": r["run"], "why": r["why"]} for r in none_runs if r["driver"] in READS_NOTHING],
                "by_driver_format": _by_driver_format(none_runs, none, "lean_python")}}


def _records(rows: list[dict]) -> dict:
    """The whole-record comparison over `rows`: how many reach the record, how many agree, and each that doesn't."""
    compared = [r for r in rows if r.get("record") is not None]
    return {"compared": len(compared), "agree": sum(1 for r in compared if r["record"]["agree"]),
            "differ": [{"run": r["run"], "name": r["name"],
                        **({"error": r["record"]["error"]} if "error" in r["record"] else
                           {"fields": r["record"]["disagree"]["fields"]})}
                       for r in compared if not r["record"]["agree"]]}


def summary_lines(c: dict) -> list[str]:
    n = c["no_recorded_verdict"]
    return [f"{c['runs']} runs in the index; {c['cases']} cases, {c['lean_python'].get('same', 0)} Lean = Python, "
            f"{c['same_verdict']} Lean = recorded, {c['negatives']} negative controls, {c['analogs']} analogs, "
            f"{c['outside']} outside",
            f"{n['runs']} runs, {n['cases']} cases, {n['lean_python'].get('same', 0)} Lean = Python, {n['negatives']} "
            f"negative controls, no recorded verdict",
            f"whole audit records: {c['records']['agree']} of {c['records']['compared']} Lean = Python; with no recorded "
            f"verdict {n['records']['agree']} of {n['records']['compared']}"]


if __name__ == "__main__":
    raise SystemExit(main())
