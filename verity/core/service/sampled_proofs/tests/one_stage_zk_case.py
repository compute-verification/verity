"""Sampled proofs' one-stage audit on the proofs themselves (`SampledProofs.OneStageZk`,
`verity/sampled-proofs/one-stage/v2`) in Python, on the cases it decides without a proof, for every difftest of a
protocol that runs it: the composition's integrity component (`verity/core/protocols/composition/tests/reference.py`)
and the network certifier's period check, which runs it on each link-window
(`verity/core/protocols/network/tests/period_case.py`).

It is v2's framing over v1's `one_stage` and decoders (this directory's `test_difftest.py`, loaded as
`one_stage_difftest`): `params` (`OneStageZk.paramsOfJson`), `evidence` (`evidenceOf`) and `check(blobs)`
(`checkBlobs` on evidence whose sessions' records are in no blob, where it needs no proof), `answer` the protocol on a
case and its ``blobs``. `CIRCUIT` and `PUBLIC` are the files the difftests hand v2 as blobs, and `undecided` an audit
that drew nothing, v2's best verdict without a session's proof."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from verity_difftest import Reject, canonical
from verity_difftest import protocol as P


def module(name: str, path: Path) -> ModuleType:
    """Another test module, loaded from its file under `name` once, with its directory on the path for the siblings it
    imports."""
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


#: Sampled proofs' one-stage difftest: `check` is v1's `OneStage.check` behind the decoders it reads, `SETUPS` its
#: programs, and its strategies draw audits.
ONE_STAGE = module("one_stage_difftest", Path(__file__).resolve().parent / "test_difftest.py")

ID = "verity/sampled-proofs/one-stage/v2"
ACCEPT, REJECT, UNDECIDED = "accept", "reject", "undecided"
DIGEST = re.compile(r"[0-9a-f]{128}")
REGISTERED_INPUT = re.compile(r"RegisteredInput[1-9][0-9]*_v1")
#: The files the difftests hand v2 as blobs: their cases name them as their circuit and public file.
CIRCUIT, PUBLIC = b"not a circuit file", b"not a public file"


def sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def params(j: Any) -> dict:
    """`OneStageZk.paramsOfJson`: v1's parameters with ``proofs``, exactly the auditor's circuit digest and the verifier's
    registered file, and the verifier's own partition object."""
    pr = j.get("proofs") if isinstance(j, dict) else None
    if not isinstance(pr, dict) or sorted(pr) != ["circuit", "registered"]:
        raise Reject("the parameters name no proofs, or other keys under them")
    if not isinstance(pr["circuit"], str) or not DIGEST.fullmatch(pr["circuit"]):
        raise Reject("proofs: circuit is not a SHA-512 digest")
    if pr["registered"] is not None and not isinstance(pr["registered"], dict):
        raise Reject("proofs: registered is not an object or null")
    base = ONE_STAGE.decode_params({k: v for k, v in j.items() if k != "proofs"})
    if base["partition"] is None:
        raise Reject("partition is null")
    return {"base": base, "program": j.get("program"), **pr}


def evidence(b: bytes) -> dict:
    """`OneStageZk.evidenceOf`: v1's evidence with no verdicts and with ``proofs``, the public file's digest and the
    sessions' files, named as the evidence's sessions."""
    j = P.canonical_json(b)
    if not isinstance(j, dict) or "verdicts" in j or "proofs" not in j:
        raise Reject("the evidence is not an object naming proofs and no verdicts")
    base = ONE_STAGE.decode_evidence(canonical({**{k: v for k, v in j.items() if k != "proofs"}, "verdicts": {}}).encode())
    pr = j["proofs"]
    if not isinstance(pr, dict) or sorted(pr) != ["public", "sessions"] or not isinstance(pr["public"], str) \
            or not isinstance(pr["sessions"], list):
        raise Reject("the evidence's proofs")
    for s in pr["sessions"]:
        if not isinstance(s, dict) or sorted(s) != ["name", "record", "reps"] or not isinstance(s["name"], str) \
                or not isinstance(s["record"], str) or not isinstance(s["reps"], dict) \
                or not all(isinstance(d, str) for d in s["reps"].values()):
            raise Reject("a session of the proofs")
    if [s["name"] for s in pr["sessions"]] != base["sessions"]:
        raise Reject("proofs: the sessions are not the evidence's sessions")
    return {"base": base, "public": pr["public"], "sessions": pr["sessions"]}


def check(blobs: list[bytes]):
    """`OneStageZk.checkBlobs` on `blobs`, for evidence whose sessions' records are in no blob, where it needs no proof:
    the circuit and the public file are blobs, a session whose record is in none is refused, and with no session the
    evidence proves no unit and v1 decides on no verdict (undecided at best: v2 accepts only on a session's proof)."""
    held = {sha(b) for b in blobs}

    def run(p: dict, c: tuple, e: dict) -> tuple:
        if p["registered"] is not None:
            raise ValueError("outside the mirror: the verifier's registered file (`OneStageZk.readsOf`)")
        definitions = p["program"].get("definitions") if isinstance(p["program"], dict) else None
        if not isinstance(definitions, dict) or any(REGISTERED_INPUT.fullmatch(k) for k in definitions):
            return REJECT, "registered: the program reads registered values"
        if p["circuit"] not in held or not DIGEST.fullmatch(e["public"]) or e["public"] not in held:
            return REJECT, "proofs: the circuit or the public file is in no blob"
        if e["sessions"]:
            if e["sessions"][0]["record"] in held:
                raise ValueError("outside the mirror: a session's record is a blob")
            return REJECT, "proofs: a session's record is in no blob"
        if e["base"]["proved_units"]:
            return REJECT, "proofs: the units the sessions proved are not the evidence's proved_units"
        return ONE_STAGE.check(p["base"], c, e["base"])
    return run


def answer(case: Any) -> dict:
    """`SampledProofs.OneStageZk.protocol` on a case and its ``blobs`` (`check`'s cases)."""
    return P.answer(ID, case, params=params, commitment=ONE_STAGE.decode_commitment, evidence=evidence,
                    digest=lambda c: c[1], check=check(case.get("blobs", []) if isinstance(case, dict) else []))


def undecided(setup, law: dict, record: dict, session_roots: dict | None = None) -> dict:
    """v2's evidence for `record`, a registration of `setup`'s program under `law`, a law that draws no unit (a subset
    of none, or Bernoulli at 0): the verifier's draw and receipt, no unit proved and no session, the public file
    `PUBLIC`; v1 is undecided on it for want of the verifier of record. `session_roots`: the roots the sessions'
    statements name (`audit_record` holds each to the registered root)."""
    d = {**ONE_STAGE.DR.derive(setup.program, setup.query, law, None), "population": setup.population, "units": []}
    return ONE_STAGE._json({
        "draw": d, "instances": 0, "proofs": {"public": sha(PUBLIC), "sessions": []}, "proved_units": [],
        "receipt": ONE_STAGE.R.receipt(record, "v", received=ONE_STAGE.RECEIVED), "served": None,
        "session_roots": session_roots, "sessions": [], "statement": {}})
