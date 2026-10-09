"""Replay every recorded one-stage run through sampled proofs' Lean protocol and today's Python, against the verdict the
run recorded.

    python benchmarks/one_stage/replay.py --index INDEX.json --runs DIR [--fetch] --lean EXE [--r7-lean EXE] --out DIR \\
      [--only RUN ...]

* **Index.** `one-stage-run/v0` rows (`research data select` of every run that holds one-stage registration or audit
  records): each run's art, the registration formats it holds and the Python verdicts it recorded.
* **Case.** For each recorded audit, the case `sampled-proofs-one-stage serve` reads: the verifier's parameters as
  `OneStage.paramsOfJson` reads them (its own program descriptor, query, law, scheme, ports, leaf layer, log, partition,
  delta), the registration record as `commitmentOf` reads it, and the evidence as `evidenceOf` reads it (the draw, the
  proved units, the instance count, the receipt, the statement, the verdicts, the sessions and their roots, the served
  draw), each from the run's own files or the verifier's own copies (the program rebuilt from the input set the run
  names, the domains derived again), and how each was found (`provenance`). Floats in verdicts and statements, which
  neither side reads, are written as strings.
* **Negatives.** Each recorded refusal that is one of the protocol's checks (R1-R7, the draw's, the record's: proved,
  receipt, served, roots, registered; the verdicts') becomes the base case with the one change the run made; a refusal
  by a check the protocol does not make (the public file's own match, `flock-verify`'s U1-U3 and roots, M0's selftests
  and bindings, a tampered set before the verifier of record) is listed as outside, with the reason, and where the
  protocol has a check of the same kind its analog is replayed and counted apart.
* **Verdicts.** Lean (`serve`), today's Python (`registration.check`, `draw.check_draw`, `audit.audit_record`, written
  as `OneStage.check` writes its outcome) and the recorded verdict, compared by verdict, stage (registration, draw,
  checks, verdict) and codes.

Writes `replay.json` (every case's three verdicts and provenance, the runs neither reads or that miss an input, and the
counts by driver and format) and `cases/<run>.jsonl` (the cases themselves) under `--out`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pkgutil
import re
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator

import verity_difftest as DT
from verity_difftest import protocol as PR
from verity.core.protocols.sampled_proofs.one_stage import audit, draw as DR, partition as P, registration as R
from verity.ml.circuits import codec, partition_object as PO
from verity.ml.commitments.identity import identity_digest, identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain

SCHEME = "frame-v3-sha512/hm96-sha512"
ROW = "hm96-sha512/row/v1"
#: M0's set-bound frame domains: each tag with its identity digest (`a0.STAND_IN_BINDINGS`)
STAND_IN_BINDINGS = {"verity/flock-circuit/binding/v2": identity_digest_sha512,
                     "verity/flock-ir-frame/binding/v1": lambda tag, doc: bytes.fromhex(identity_digest(tag, doc))}
SERVED_BINDINGS = {"sha512": identity_digest_sha512, "v1": lambda tag, doc: bytes.fromhex(identity_digest(tag, doc))}
STAGES = ("registration", "draw", "checks", "verdict")
CHECKS = ("proved", "receipt", "served", "roots", "registered")


class Missing(Exception):
    """A run lacks a file the replay needs (the message names it)."""


class Gap(Exception):
    """The run's files hold the input but the replay cannot rebuild the verifier's copy of it (the message says which)."""


# --- programs ----------------------------------------------------------------------------------------------------------

def _registry() -> None:
    """Every Definition a recorded program may call: the circuit ops and vLLM's program registry."""
    for name in ("verity.ml.circuits.ops", "verity_vllm.program.registry"):
        pkg = importlib.import_module(name)
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            importlib.import_module(m.name)


def _json(value: Any) -> Any:
    return json.loads(DT.canonical(value))


def descriptor(program) -> dict:
    return program.desc if isinstance(program, codec.DecodedProgram) else _json(codec.encode_program(program))


def decode(desc: dict):
    from verity.ml.circuits.defs import REGISTRY
    return codec.decode_program(desc, REGISTRY)


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

    def open(self, tar_sha: str, name: str):
        from verity_numerical.bench.input_sets import InputSet
        d = self.dir(tar_sha) / name
        if not (d / "manifest.json").exists():
            raise Missing(f"inputs/set.tar ({tar_sha[:16]}): {name}/manifest.json")
        return InputSet.open(str(d))


# --- a run -------------------------------------------------------------------------------------------------------------

@dataclass
class Run:
    row: dict
    dir: Path
    files: list[dict]
    sets: Sets
    programs: Programs
    sets_fallback: str | None = None
    outside: list[dict] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.row["art"][4:12]

    def path(self, rel: str) -> Path:
        p = self.dir / rel
        if not p.exists():
            raise Missing(rel)
        return p

    def json(self, rel: str) -> Any:
        return json.loads(self.path(rel).read_text())

    def has(self, rel: str) -> bool:
        return (self.dir / rel).exists()

    def file_sha256(self, rel: str) -> str | None:
        return next((f["sha256"] for f in self.files if f["path"] == rel), None)

    def commit(self) -> str | None:
        if self.has("commit"):
            return self.path("commit").read_text().strip()
        if not self.has("job.json"):
            return None
        j = self.json("job.json")
        return (j.get("computation") or {}).get("tree_sha") or ((j.get("store") or {}).get("conditions") or {}).get("source.commit")

    def input_set(self, name: str, fallback: str | None = None):
        """The run's own input set `name` (from the `set.tar` it was sent), else the `fallback` tar's, with provenance."""
        own = self.file_sha256("inputs/set.tar")
        sha = own or fallback
        if sha is None:
            raise Missing("inputs/set.tar")
        return self.sets.open(sha, name), ("the run's inputs/set.tar" if own else f"inputs/set.tar {sha[:16]} of another run "
                                                                                  "(this run's was not uploaded)")


def live_line(path: Path) -> dict | None:
    for line in path.read_text().splitlines():
        if line.startswith("LIVE\t"):
            return json.loads(line[5:])
    return None


def archive_keys(path: Path) -> dict[str, str]:
    """`flock-verify archive-put`'s stdout: the SHA-512 content key of each file it put, by file name."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{128}", parts[0]):
            out[Path(parts[1]).name] = parts[0]
    return out


def sha512_file(path: Path) -> str:
    return hashlib.sha512(path.read_bytes()).hexdigest()


def public_digest(run: Run, pub: str, archive_log: str) -> tuple[str, str]:
    """The SHA-512 of the verifier's public file: the file's own when the run kept it, else `archive-put`'s key for it."""
    key = archive_keys(run.path(archive_log)).get(Path(pub).name) if run.has(archive_log) else None
    if run.has(pub):
        d = sha512_file(run.path(pub))
        if key is not None and key != d:
            raise Gap(f"{pub}: its SHA-512 is not archive-put's key")
        return d, f"SHA-512 of {pub}"
    if key is None:
        raise Missing(f"{pub} (or its key in {archive_log})")
    return key, f"archive-put's key of {Path(pub).name} ({archive_log})"


# --- the verifier's copies ---------------------------------------------------------------------------------------------

def template_ids(desc: dict) -> list[str]:
    """The templates a population program instantiates: its root's batch callees, in node order."""
    root = desc["definitions"][desc["root"]]
    ids = [desc["callees"][n["fn"]] for n in root["body"]["nodes"] if n.get("op") == "batch"]
    return [t if isinstance(t, str) else t.get("id", "") for t in ids]


def partition_form(program, query: dict, rec: dict, era_templates: list[str]) -> tuple[dict | None, str]:
    """The verifier's partition object for `rec`: the canonical one over `program` and `query` (null), or the
    pre-canonical `{program, query}` with the era's query form (`Q_template_instance(s)` v0 with `template(s)` and a
    `width_rule` at the top level) over the era's template ids."""
    sha = PO.program_sha512(program)
    if P.digest(P.partition(program, query)) == rec["partition"]:
        return None, "canonical"
    forms = [{"name": "Q_template_instance", "template": t, "version": 0, "width_rule": None} for t in era_templates]
    forms.append({"name": "Q_template_instances", "templates": list(era_templates), "version": 0, "width_rule": None})
    for q in forms:
        part = {"program": sha, "query": q}
        if P.digest(part) == rec["partition"] and R.query_digest(q) == rec["query"]:
            return part, f"pre-canonical {q['name']} over {era_templates}"
    raise Gap(f"no partition form over {era_templates} gives the record's partition {rec['partition'][:16]}")


def _domain(binding: bytes, leaves: int) -> str:
    return CommitmentDomain(binding, -1, RangeIndexedDomain(0, int(leaves)), hash="sha512").domain_id.hex()


def stand_in_layouts(s, n: int, prefix: str = "") -> dict[str, dict[str, tuple[str, int]]]:
    """The set's ports as M0 lays them out, before and after hidden outputs: `{layout: {port: (schema, leaves)}}`."""
    sc = s.subcircuit
    rows = {f"{prefix}{p.name}": (ROW, n) for p in sc.inputs}
    return {"words": {**rows, **{f"{prefix}{p.name}": ("u16", n * p.words) for p in sc.outputs}},
            "hidden": {**rows, **{f"{prefix}{p.name}": (ROW, n) for p in sc.outputs}}}


def stand_in_ports(s, n: int, layout: dict[str, tuple[str, int]], tag: str, content: bool, prefix: str = "") -> dict:
    out = {}
    for p, (schema, leaves) in layout.items():
        doc = {"set": s.name, **({"content_digest": s.content_digest} if content else {}), "lo": 0, "hi": n,
               "port": p[len(prefix):], "schema": schema}
        out[p] = {"domain": _domain(STAND_IN_BINDINGS[tag](tag, doc), leaves), "leaves": leaves}
    return out


def matches(ports: dict, rec: dict, only: Callable[[str], bool] = lambda p: True) -> bool:
    got = {r["port"]: {"domain": r["domain"], "leaves": r["leaves"]} for r in rec["roots"] if only(r["port"])}
    return got == ports


def choose(variants: Iterator[tuple[str, dict]], rec: dict, today: str, only: Callable[[str], bool] = lambda p: True
           ) -> tuple[dict, str]:
    """The era's derivation of the ports under which every one matches the record (R3 passed in the run); else today's,
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
    recorded: dict
    provenance: dict = field(default_factory=dict)
    r7: bool = False

    def case(self) -> dict:
        return {"commitment": DT.canonical(self.rec).encode().hex(),
                "evidence": DT.canonical(strings(self.evidence)).encode().hex(), "params": _json(self.params)}

    def but(self, name: str, kind: str, recorded_: dict, *, params: dict | None = None, rec: dict | None = None,
            evidence: dict | None = None, program: Any = None, provenance: dict | None = None) -> Case:
        return Case(self.run, self.driver, self.format, name, kind, program if program is not None else self.program,
                    {**self.params, **(params or {})}, rec if rec is not None else self.rec,
                    {**self.evidence, **(evidence or {})}, recorded_, {**self.provenance, **(provenance or {})}, self.r7)


def params(program, query: dict, law: dict, ports: dict, leaf_layer: str | None, delta: str, *, partition: dict | None = None,
           log: list | None = None, anchors: list | None = None, work_table: dict | None = None,
           registered: dict | None = None, scheme: str = SCHEME) -> dict:
    return {"anchors": list(anchors or []), "delta": delta, "law": law, "leaf_layer": leaf_layer, "log": list(log or []),
            "partition": partition, "ports": ports, "program": descriptor(program), "query": query,
            "registered": registered, "scheme": scheme, "work_table": work_table}


def evidence(a: dict, *, instances: int, sessions: list[str], session_roots: dict | None, served: list[int] | None) -> dict:
    """The evidence of a recorded audit record `a` (what `audit_record` was handed)."""
    return {"draw": a["draw"], "instances": instances, "proved_units": a["proved_units"], "receipt": a["receipt"],
            "served": served, "session_registered": audit.session_registered(a["verdicts"]),
            "session_roots": session_roots, "sessions": sessions, "statement": a["statement"], "verdicts": a["verdicts"]}


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

def python(c: Case) -> dict:
    """Today's Python on the case, as `OneStage.check` writes its outcome (`test_difftest.check`, with Lean's reasons)."""
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
                           e["verdicts"], delta, e["sessions"], e["served"], e["session_roots"], e["session_registered"])
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
    public = {"bound": wrong["bound"] if isinstance(wrong["bound"], int) else None, "drawn": list(e["draw"]["units"]),
              "per_template": wrong.get("per_template"), "proved": DR.proved(e["draw"])}
    return out("accept", None, public, {**profile, "delta": str(profile["delta"])})


def stage_of(outcome: dict) -> tuple[str | None, list[str] | None]:
    """The stage of `OneStage.check` an outcome's reason names, and its codes."""
    v, reason = outcome.get("verdict"), outcome.get("reason") or ""
    if v == "accept":
        return None, None
    if v == "undecided":
        return "verdict", None
    for prefix, stage in (("registration: ", "registration"), ("draw: ", "draw"), ("the audit's checks failed: ", "checks")):
        if reason.startswith(prefix):
            rest = reason[len(prefix):]
            if stage == "registration" and rest.startswith("R1-law: "):
                return stage, ["R1-law"]
            return stage, sorted(rest.split(", "))
    if reason.startswith("verdict "):
        return "verdict", None
    return "decode", [reason]


def compare(c: Case, lean: dict, py: dict) -> dict:
    strip = lambda o: {k: v for k, v in o.items() if k != "reason"}  # noqa: E731
    lp = "same" if lean == py else ("same-verdict" if strip(lean) == strip(py) else "differ")
    stage, codes = stage_of(lean)
    r = c.recorded
    if lean.get("verdict") != r["verdict"]:
        lr = "differ"
    elif c.kind == "analog" or (r["stage"] in (None, stage) and (r["codes"] is None or r["codes"] == codes)):
        lr = "same"
    else:
        lr = "same-verdict"
    return {"lean_python": lp, "lean_recorded": lr, "lean_stage": stage, "lean_codes": codes}


# --- drivers -----------------------------------------------------------------------------------------------------------

#: Negatives a run recorded that are not the protocol's checks, with why.
OUTSIDE = {
    "registration-root-mismatch": "the verifier's public file against the record (`registration.matches_public`), before the protocol",
    "public-file-domain-mismatch": "the verifier's public file against the record (`registration.matches_public`), before the protocol",
    "public-body-altered-leaf-layer-updated (Lean roots)": "flock-verify recomputing the roots from the public file",
    "statement-binds-another-partition": "M0's statement bindings (`a0.bindings`)",
    "m0-selftest-unit-draw-session": "M0's selftest",
    "prover-ignores-the-draw (M0 selftest)": "M0's selftest",
    "wrong-output-word": "the verifier of record (flock-verify, M0) on a tampered copy of the set, whose record the run did not keep",
}
#: The flock-verify draw checks a run recorded, each with the protocol's check of the same kind (an analog).
ANALOG = {
    "instances-not-the-draw (Lean U3)": "flock-verify U3; analog: the proved check on one instance more",
    "draw-dropped-from-record (Lean U1)": "flock-verify U1; analog: the draw check on a null draw",
    "public-file-b||c-byte-flipped (R4)": "R4 against the digest of a public file the run did not keep; analog: another digest",
}


def a0(run: Run) -> Iterator[Case]:
    """A0 (`a0.py`): one template's population program, a stand-in commitment under M0's set-bound domains."""
    base = "out/a0"
    a = run.json(f"{base}/audit.json")
    summary = run.json(f"{base}/summary.json")
    rec = a["registration"]
    n = summary["n"]
    s, set_from = run.input_set(rec["window"].get("input_set") or "rope101", run.sets_fallback)
    template = s.subcircuit.definition()
    program = P.population_program(template, n)
    prov: dict = {"commit": run.commit(), "set": set_from, "program": "P.population_program(set template, n)"}
    if PO.program_sha512(program) != rec["program"]:
        stored = run.programs.get(rec["program"])
        if stored is None:
            raise Gap(f"the set's population program is {PO.program_sha512(program)[:16]}, the record's {rec['program'][:16]}")
        program, prov["program"] = stored, "a program.json the runs stored"
    query = P.template_instance_query(template.id)
    partition, prov["partition"] = partition_form(program, query, rec, [template.id])
    layouts = stand_in_layouts(s, n)
    ports, prov["ports"] = choose(((f"{tag} {layout}{' +content' if content else ''}",
                                    stand_in_ports(s, n, layouts[layout], tag, content))
                                   for layout in ("words", "hidden") for tag in STAND_IN_BINDINGS for content in (True, False)),
                                  rec, "verity/flock-circuit/binding/v2 hidden")
    leaf_layer, prov["leaf_layer"] = public_digest(run, f"{base}/stage/pub-{n}.bin", f"{base}/session/archive-put.log")
    live = live_line(run.path(f"{base}/session/prove.log")) or {}
    sessions = (a.get("integrity_profile") or {}).get("evidence", {}).get("sessions") or sorted(
        sha512_file(f) for f in (run.dir / base / "session" / "sessions").glob("*/session.json"))
    law = summary["law"]
    p = params(program, query, law, ports, leaf_layer, delta_of(summary["delta"]), partition=partition)
    e = evidence(a, instances=live.get("instances", -1), sessions=sessions, session_roots=live.get("roots") or {},
                 served=None)
    prov["evidence"] = "audit.json; instances and session roots from session/prove.log's LIVE line"
    c = Case(run.id, "a0", rec["format"], "audit", "audit", program, p, rec, e,
             recorded_audit(a, f"{base}/audit.json"), prov)
    yield c
    if not run.has(f"{base}/negatives.json"):
        return
    neg = run.json(f"{base}/negatives.json")
    src = f"{base}/negatives.json"
    draw = a["draw"]
    k = law["k"]
    smaller = P.population_program(template, n - 1)
    spart, _ = (None, "") if partition is None else ({"program": PO.program_sha512(smaller), "query": partition["query"]}, "")
    first = rec["roots"][0]
    cases: dict[str, Callable[[dict], Case]] = {
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


def _negatives(run: Run, c: Case, neg: dict, src: str, cases: dict[str, Callable[[dict], Case]], *, base_instances: int,
               session_roots: dict, served: list[int] | None = None) -> Iterator[Case]:
    """The run's negatives: its own (`cases`), `audit_negatives`' (on the base audit as that function re-decided it:
    instances = the proved units, the sessions' roots, the served draw), and the ones outside the protocol."""
    an = c.but("", "negative", c.recorded, evidence={"instances": base_instances, "session_roots": session_roots,
                                                      "served": served})
    vs = c.evidence["verdicts"]
    port = next(iter(session_roots), None)
    shared: dict[str, Callable[[dict], Case]] = {
        "verifier-of-record-did-not-run (C2)": lambda v: an.but(
            "verifier-of-record-did-not-run (C2)", "negative", recorded_failed(v, src),
            evidence={"verdicts": {k: {**x, "accepted": None} if x.get("of_record") else x for k, x in vs.items()}}),
        "no-verifier-of-record (C2)": lambda v: an.but(
            "no-verifier-of-record (C2)", "negative", recorded_failed(v, src),
            evidence={"verdicts": {k: x for k, x in vs.items() if not x.get("of_record")}}),
        "session-root-not-registered": lambda v: an.but(
            "session-root-not-registered", "negative", recorded_failed(v, src),
            evidence={"session_roots": {**session_roots, port: "00" * 64}}),
    }
    if served is not None:
        other = next(u for u in range(c.rec["population"]) if u not in set(served))
        shared["served-share-not-the-verifier's-draw (C1)"] = lambda v: an.but(
            "served-share-not-the-verifier's-draw (C1)", "negative", recorded_failed(v, src),
            evidence={"served": sorted(served)[1:] + [other]})
    for name, v in neg.items():
        make = cases.get(name) or shared.get(name)
        if make is not None:
            yield make(v)
        else:
            run.outside.append({"run": c.run, "driver": c.driver, "name": name,
                                "why": OUTSIDE.get(name, "UNHANDLED: the replay has no case for it")})


DRIVERS: dict[str, Callable[[Run], Iterator[Case]]] = {"a0": a0}


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
    ap.add_argument("--lean", required=True, help="sampled-proofs-one-stage")
    ap.add_argument("--r7-lean", default=None, help="sampled-proofs-one-stage built with registered values (R7)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="run ids (8 hex digits)")
    a = ap.parse_args()
    _registry()
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
    programs = Programs()
    for f in sorted(runs_root.glob("*/**/program.json")):
        programs.add_file(f)
    # the rope101 set's tar (A0's), for the runs that were sent one but did not upload it
    fallback = next((sha for sha in sorted(tars) if (sets.dir(sha) / "rope101" / "manifest.json").exists()
                     and len(list(sets.dir(sha).iterdir())) == 1), None)

    servers = {False: DT.Server([a.lean, "serve"])}
    if a.r7_lean:
        servers[True] = DT.Server([a.r7_lean, "serve"])
    results, runs_out, outside = [], [], []
    for r in rows:
        rid = r["art"][4:12]
        run = Run(r, runs_root / rid, lists[rid], sets, programs, fallback, outside)
        info = {"run": rid, "art": r["art"], "id": r.get("run"), "drivers": r.get("drivers"),
                "formats": r.get("registration_formats"), "python_verdicts": len(r.get("python_verdicts") or [])}
        driver = classify(run)
        info["driver"] = driver
        if driver not in DRIVERS:
            info["status"] = driver
            runs_out.append(info)
            continue
        cases: list[Case] = []
        try:
            cases.extend(DRIVERS[driver](run))
        except Missing as e:
            info["status"], info["missing"] = "missing-input", str(e)
        except Gap as e:
            info["status"], info["gap"] = "reconstruction-gap", str(e)
        else:
            info["status"] = "replayed"
        with open(out / "cases" / f"{rid}.jsonl", "w") as f:
            for c in cases:
                if c.r7 and True not in servers:
                    results.append(_row(c, None, None, {"pending": "no --r7-lean"}))
                    continue
                case = c.case()
                try:
                    lean = servers[c.r7].exchange(case)
                except DT.ProtocolError as err:
                    lean = {"verdict": "error", "reason": str(err)}
                try:
                    py = python(c)
                except Exception as err:  # noqa: BLE001 - a crash is a verdict here
                    py = {"verdict": "error", "reason": f"{type(err).__name__}: {err}"}
                results.append(_row(c, lean, py, compare(c, lean, py)))
                f.write(json.dumps({"name": c.name, "case": case}) + "\n")
        info["cases"] = len(cases)
        runs_out.append(info)
        print(f"{rid} {driver}: {info['status']} {len(cases)} cases", file=sys.stderr, flush=True)
    for s in servers.values():
        s.close()
    report = {"format": "verity/one-stage-replay/v0", "index": a.index, "lean": a.lean, "r7_lean": a.r7_lean,
              "runs": runs_out, "cases": results, "outside": outside, "counts": counts(runs_out, results, outside)}
    (out / "replay.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report["counts"], indent=1, sort_keys=True))
    return 0


def _row(c: Case, lean: dict | None, py: dict | None, cmp: dict) -> dict:
    return {"run": c.run, "driver": c.driver, "format": c.format, "name": c.name, "kind": c.kind, "r7": c.r7,
            "recorded": c.recorded, "lean": lean, "python": py, **cmp, "provenance": c.provenance}


def classify(run: Run) -> str:
    """The driver whose files the run holds, or why the replay reads none."""
    formats = run.row.get("registration_formats") or {}
    drivers = run.row.get("drivers") or []
    if formats and all(f == "verity/one-stage/registration/v0" for f in formats):
        return "neither-reads (registration v0)"
    if not run.row.get("python_verdicts"):
        return "no-recorded-verdict"
    if drivers == ["a0"]:
        return "a0"
    return "unhandled"


def counts(runs: list[dict], results: list[dict], outside: list[dict]) -> dict:
    by: dict = defaultdict(Counter)
    for r in results:
        key = f"{r['driver']} {r['format']}"
        by[key][r["lean_recorded"] if r.get("lean_recorded") else "pending"] += 1
    for r in runs:
        if r["status"] != "replayed":
            by[f"{r['driver']} {','.join(sorted(r.get('formats') or {})) or '-'}"][r["status"]] += 1
    return {"runs": len(runs), "cases": len(results),
            "same_verdict": sum(1 for r in results if r.get("lean_recorded") in ("same", "same-verdict")),
            "negatives": sum(1 for r in results if r["kind"] == "negative"),
            "analogs": sum(1 for r in results if r["kind"] == "analog"), "outside": len(outside),
            "lean_python": dict(Counter(r.get("lean_python", "pending") for r in results)),
            "by_driver_format": {k: dict(v) for k, v in sorted(by.items())}}


if __name__ == "__main__":
    raise SystemExit(main())
