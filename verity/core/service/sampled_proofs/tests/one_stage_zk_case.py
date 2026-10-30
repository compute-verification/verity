"""Sampled proofs' one-stage audit on the proofs themselves (`SampledProofs.OneStageZk`,
`verity/sampled-proofs/one-stage/v2`) in Python, on the cases it decides without a proof, for every difftest of a
protocol that runs it: the network certifier's period check, which runs it on each link-window
(`verity/core/protocols/network/tests/period_case.py`).

It is v2's framing over v1's `one_stage` and decoders (this directory's `test_difftest.py`, loaded as
`one_stage_difftest`): `params` (`OneStageZk.paramsOfJson`), `evidence` (`evidenceOf`) and `check(blobs)`
(`checkBlobs` on evidence whose sessions' records are in no blob, where it needs no proof), `answer` the protocol on a
case and its ``blobs``. Each takes the single form and the group form (``proofs.members``: one circuit digest, registered
file, public file and sessions a member, the registration's ``leaf_layer`` the members' public files', `leaf_layer`).
`CIRCUIT` and `PUBLIC` are the files the difftests hand v2 as blobs, `undecided` an audit that sampled no proof block,
v2's best verdict without a session's proof, and `challenger` the auditor's challenger whose released coins each such
audit's sample reads."""

from __future__ import annotations

import hashlib
import importlib.util
import random
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


#: Sampled proofs' one-stage difftest: `check` is v1's `OneStage.check` behind the decoders it reads, and `SETUPS` its
#: programs.
ONE_STAGE = module("one_stage_difftest", Path(__file__).resolve().parent / "test_difftest.py")

ID = "verity/sampled-proofs/one-stage/v2"
ACCEPT, REJECT, UNDECIDED = "accept", "reject", "undecided"
DIGEST = re.compile(r"[0-9a-f]{128}")
REGISTERED_INPUT = re.compile(r"RegisteredInput[1-9][0-9]*_v1")
#: The files the difftests hand v2 as blobs: their cases name them as their circuit and public file.
CIRCUIT, PUBLIC = b"not a circuit file", b"not a public file"


def sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def _own(pr: dict) -> tuple:
    """`OneStageZk.circuitOf`: the auditor's circuit digest and the verifier's registered file (an object, or null)."""
    if not isinstance(pr["circuit"], str) or not DIGEST.fullmatch(pr["circuit"]):
        raise Reject("proofs: circuit is not a SHA-512 digest")
    if pr["registered"] is not None and not isinstance(pr["registered"], dict):
        raise Reject("proofs: registered is not an object or null")
    return pr["circuit"], pr["registered"]


def params(j: Any) -> dict:
    """`OneStageZk.paramsOfJson`: v1's parameters with ``proofs``, exactly the auditor's circuit digest and the verifier's
    registered file, or, in the group form, exactly ``members``, a non-empty list of exactly ``{circuit, name,
    registered}`` with distinct names; naming the auditor's challenger (`NO_CHALLENGER`), and the verifier's own
    partition object."""
    pr = j.get("proofs") if isinstance(j, dict) else None
    group = isinstance(pr, dict) and "members" in pr
    if not isinstance(pr, dict) or sorted(pr) != (["members"] if group else ["circuit", "registered"]):
        raise Reject("the parameters name no proofs, or other keys under them")
    base = ONE_STAGE.decode_params_v2(j)
    if base["partition"] is None:
        raise Reject("partition is null")
    if not group:
        circuit, registered = _own(pr)
        return {"base": base, "program": j.get("program"), "circuit": circuit, "registered": registered, "members": []}
    ms = pr["members"]
    if not isinstance(ms, list) or not ms:
        raise Reject("proofs: members is not a non-empty list")
    members = []
    for m in ms:
        if not isinstance(m, dict) or sorted(m) != ["circuit", "name", "registered"] or not isinstance(m["name"], str):
            raise Reject("proofs: a member is not exactly its name, circuit digest and registered file")
        circuit, registered = _own(m)
        members.append({"name": m["name"], "circuit": circuit, "registered": registered})
    if len({m["name"] for m in members}) != len(members):
        raise Reject("proofs: two members have one name")
    return {"base": base, "program": j.get("program"), "circuit": "", "registered": None, "members": members}


def _proofs(pr: Any) -> tuple:
    """`OneStageZk.proofsOf`: exactly the public file's digest and the sessions, each exactly its name, its record's
    digest and its rep files' digests."""
    if not isinstance(pr, dict) or sorted(pr) != ["public", "sessions"] or not isinstance(pr["public"], str) \
            or not isinstance(pr["sessions"], list):
        raise Reject("the evidence's proofs")
    for s in pr["sessions"]:
        if not isinstance(s, dict) or sorted(s) != ["name", "record", "reps"] or not isinstance(s["name"], str) \
                or not isinstance(s["record"], str) or not isinstance(s["reps"], dict) \
                or not all(isinstance(d, str) for d in s["reps"].values()):
            raise Reject("a session of the proofs")
    return pr["public"], pr["sessions"]


def evidence(b: bytes) -> dict:
    """`OneStageZk.evidenceOf`: v1's evidence with no verdicts and with ``proofs``, the public file's digest and the
    sessions' files, named as the evidence's sessions; or, in the group form, exactly ``members``, each exactly its
    name, public file's digest and sessions, the members' sessions in order the evidence's sessions."""
    j = P.canonical_json(b)
    if not isinstance(j, dict) or "verdicts" in j or "proofs" not in j:
        raise Reject("the evidence is not an object naming proofs and no verdicts")
    base = ONE_STAGE.decode_evidence(canonical({**{k: v for k, v in j.items() if k != "proofs"}, "verdicts": {}}).encode())
    pr = j["proofs"]
    if isinstance(pr, dict) and "members" in pr:
        if sorted(pr) != ["members"] or not isinstance(pr["members"], list):
            raise Reject("the evidence's proofs")
        members = []
        for m in pr["members"]:
            if not isinstance(m, dict) or sorted(m) != ["name", "public", "sessions"] or not isinstance(m["name"], str):
                raise Reject("a member of the evidence's proofs")
            public, sessions = _proofs({k: v for k, v in m.items() if k != "name"})
            members.append({"name": m["name"], "public": public, "sessions": sessions})
        if [s["name"] for m in members for s in m["sessions"]] != base["sessions"]:
            raise Reject("proofs: the members' sessions are not the evidence's sessions")
        return {"base": base, "public": "", "sessions": [], "members": members}
    public, sessions = _proofs(pr)
    if [s["name"] for s in sessions] != base["sessions"]:
        raise Reject("proofs: the sessions are not the evidence's sessions")
    return {"base": base, "public": public, "sessions": sessions, "members": []}


def leaf_layer(members: list[dict]) -> str:
    """`OneStageZk.leafLayer` (`scan_zk.audit`'s): the SHA-512 of the canonical object naming each member's public
    file's digest by its name."""
    return sha(canonical({m["name"]: m["public"] for m in members}).encode())


def check(blobs: list[bytes]):
    """`OneStageZk.checkBlobs` on `blobs`, for evidence whose sessions' records are in no blob, where it needs no proof:
    the sample's coins first (`coinsOf`, the challenger's release after the registration), then the circuit and the
    public file are blobs, a session whose record is in none is refused, and with no session the evidence proves no
    proof block and v1 decides on no verdict (undecided at best: v2 accepts only on a session's proof). The group form
    (`checkGroupAt`) first holds the registration's ``leaf_layer`` to the members' public files' and the evidence's
    members to the parameters', then runs each member so, in order."""
    held = {sha(b) for b in blobs}

    def own(program: Any, circuit: str, registered: Any, public: str, sessions: list) -> str | None:
        if registered is not None:
            raise ValueError("outside the mirror: the verifier's registered file (`OneStageZk.readsOf`)")
        definitions = program.get("definitions") if isinstance(program, dict) else None
        if not isinstance(definitions, dict) or any(REGISTERED_INPUT.fullmatch(k) for k in definitions):
            return "registered: the program reads registered values"
        if circuit not in held:
            return f"circuit: no blob has the SHA-512 {circuit}"
        if not DIGEST.fullmatch(public) or public not in held:
            return f"public file: no blob has the SHA-512 {public}" if DIGEST.fullmatch(public) else \
                "public file: not a SHA-512 digest"
        if sessions:
            s = sessions[0]
            if s["record"] in held:
                raise ValueError("outside the mirror: a session's record is a blob")
            return f"session {s['name']}'s record: no blob has the SHA-512 {s['record']}" \
                if DIGEST.fullmatch(s["record"]) else f"session {s['name']}'s record: not a SHA-512 digest"
        return None

    def run(p: dict, c: tuple, e: dict) -> tuple:
        try:
            ONE_STAGE._coins_of(p["base"], c, e["base"])
        except ONE_STAGE.CoinError as x:
            return REJECT, f"coins: {x}"
        if not p["members"] and not e["members"]:
            why = own(p["program"], p["circuit"], p["registered"], e["public"], e["sessions"])
            if why is not None:
                return REJECT, f"proofs: {why}"
            if e["base"]["proved_units"]:
                return REJECT, "proofs: the units the sessions proved are not the evidence's proved_units"
            return ONE_STAGE.check(p["base"], c, e["base"])
        if c[0].get("leaf_layer") != leaf_layer(e["members"]):
            return REJECT, "proofs: the registration's leaf_layer is not the members' public files' leaf layer"
        if [m["name"] for m in e["members"]] != [m["name"] for m in p["members"]]:
            return REJECT, "proofs: the members are not the parameters' members, in order"
        for pm, em in zip(p["members"], e["members"]):
            why = own(p["program"], pm["circuit"], pm["registered"], em["public"], em["sessions"])
            if why is not None:
                return REJECT, f"proofs: member {pm['name']}: {why}"
        if e["base"]["proved_units"]:
            return REJECT, "proofs: the units the members' sessions proved are not the evidence's proved_units, each once"
        return ONE_STAGE.check(p["base"], c, e["base"])
    return run


def answer(case: Any) -> dict:
    """`SampledProofs.OneStageZk.protocol` on a case and its ``blobs`` (`check`'s cases)."""
    return P.answer(ID, case, params=params, commitment=ONE_STAGE.decode_commitment, evidence=evidence,
                    digest=lambda c: c[1], check=check(case.get("blobs", []) if isinstance(case, dict) else []))


def undecided(setup, law: dict, record: dict, session_roots: dict | None = None) -> dict:
    """v2's evidence for `record`, a registration of `setup`'s program under `law`, a law that samples no proof block
    (a subset of none, or Bernoulli at 0): the verifier's sample and receipt, no proof block proved and no session, the
    public file `PUBLIC`; v1 is undecided on it for want of the verifier of record. `session_roots`: the roots the sessions'
    statements name (`audit_record` holds each to the registered root)."""
    d = {**ONE_STAGE.DR.derive(setup.program, setup.query, law, None), "population": setup.population, "units": []}
    return ONE_STAGE._json({
        "draw": d, "instances": 0, "proofs": {"public": sha(PUBLIC), "sessions": []}, "proved_units": [],
        "receipt": ONE_STAGE.R.receipt(record, "v", received=ONE_STAGE.RECEIVED), "served": None,
        "session_roots": session_roots, "sessions": [], "statement": {}})


def challenger(rng: random.Random, setup, law: dict, records: list[dict]) -> tuple[dict, list[dict]]:
    """The auditor's challenger for `records` in turn, registrations of `setup`'s program under `law` (`undecided`'s, a
    law that samples no proof block, so its sample reads the same bytes whatever they are): one withheld session of a
    ledger window (`Drawn`'s), releasing after each record's digest the coins its sample reads (`draw_first`). Returns
    the parameters that name it (``challenger`` and ``window_root``, which v2 requires) and each record's evidence
    ``coins``."""
    mine = ONE_STAGE.DR.derive(setup.program, setup.query, law, None)

    def sample(source) -> None:
        ONE_STAGE.draw_first(ONE_STAGE.OwnRandomness(source), mine, setup.population)
    read: list[int] = []
    sample(lambda k: read.append(k) or bytes(k))
    per = -(-sum(read) // ONE_STAGE.COIN_BYTES)
    digests = [bytes.fromhex(ONE_STAGE.R.digest(r)) for r in records]
    later = []

    def reads(coins) -> bytes:
        sample(coins)
        for after in digests[1:]:
            later.append(coins.commit(after))
            sample(coins)
        return b"".join(o.coin for s in coins.segments for o in s.openings)
    # the session opens up to two coins of the family before the first digest and two after the last
    d = ONE_STAGE.Drawn(rng, after=digests[0], reads=reads, size=4 + per * len(records))
    named = {"challenger": {"key": d.s.key.to_json(), "label": "draw"}, "window_root": d.ledger.window["root"]}
    return named, [{k: v for k, v in d.evidence(s.to_json()).items() if k != "key"} for s in [d.segment, *later]]
