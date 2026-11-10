"""Onsite's and remote's Lean (`onsite serve`, `remote serve`: each product's executable, the core CLI) against the
Python reference (`reference.py`: the trial's composition rule with reject dominance, over the harness's framing of the
core interface and the subprotocols' own Python). Hypothesis draws each subprotocol's case to accept, reject, be
undecided or fail to decode (from its own difftest's generator: PoUW's periods, made once from
`pouw_period_cases`, the network certifier's period `build`, the inference-only policy's `drawn`, which is
<<<<<<< HEAD
never undecided, remote's sampled proofs' audits under the form ``one-stage``, `form_case`: one-stage v2's over
`empty_draws` (an audit that drew nothing at its challenger's coins), in the one-statement form or the per-member form
(`_per_member`), with one of `V2_BREAKS` or the form's breaks, and
one-stage v1's `changed_audits`, which the form refuses by name, onsite's one-stage audits, `v1_empty_draws` and
`changed_audits`, the wipe's signed wipes, `wipe_protocol_case.case`; space's and bandwidth's from the small periods of
their hooks, `periods.py`, all on one committed `C*` but PoUS's run on another store, `periods.OTHER_STORE`), often
recommits the subprotocols each composite ties to one record (`one_record`: the policy to integrity's registration
record, on that audit's own declared circuit, `policy_on`, and at remote sometimes every window of the period too,
`windows_on`), hands every subprotocol the same blobs (`BLOBS`), then often breaks the composition's own inputs (a
component dropped, unknown, repeated or out of order, a case under other params, a wrong or short commitment,
non-canonical or malformed evidence, blobs missing or malformed) or the case itself (a field missing or not hex, not an
object), and every field of the outcome but its reason must agree exactly. Integrity's cases are aimed at undecided at
best (an audit that drew nothing), so no product accepts here. `product IDS serve` over each composite's subprotocols,
the composite `verity run`'s verifier runs, is held to the same reference, and `product IDS id` names remote and onsite
for their subprotocols and refuses every other set: the same two composites a configuration may list
(`verity.core.deploy.config.COMPOSITES`)."""
||||||| merged common ancestors
never undecided, sampled proofs' audits, `empty_draws` and `changed_audits`, the wipe's signed wipes,
`wipe_protocol_case.case`; space's and bandwidth's from the small periods of their hooks, `periods.py`, all on one
committed `C*` but PoUS's run on another store, `periods.OTHER_STORE`), often recommits the subprotocols each composite
ties to one record (`one_record`: the policy to the one-stage audit's registration record, on that audit's own
declared circuit, `policy_on`, and at remote sometimes every window of the period too, `windows_on`), hands every subprotocol the same blobs
(`BLOBS`), then often breaks the composition's own inputs (a component dropped, unknown, repeated or out of order, a
case under other params, a wrong or short commitment, non-canonical or malformed evidence, blobs missing or malformed)
or the case itself (a field missing or not hex, not an object), and every field of the outcome but its reason must agree
exactly. Integrity's cases are aimed at undecided at best (an audit that drew nothing), so no product accepts here.
`product IDS serve` over each composite's subprotocols, the composite `verity run`'s verifier runs, is held to the same
reference, and `product IDS id` names remote and onsite for their subprotocols and refuses every other set: the same two
composites a configuration may list (`verity.core.deploy.config.COMPOSITES`)."""
=======
never undecided, sampled proofs' audits under the form ``one-stage``, `form_case`: one-stage v2's over `empty_draws`
(an audit that drew nothing at its challenger's coins, which names no session) with one of `V2_BREAKS` or the form's
breaks, and one-stage v1's `changed_audits`, which the form refuses by name, the wipe's signed wipes,
`wipe_protocol_case.case`; space's and
bandwidth's from the small periods of their hooks, `periods.py`, all on one committed `C*` but PoUS's run on another
store, `periods.OTHER_STORE`), often recommits the subprotocols each composite ties to one record (`one_record`: the
policy to sampled proofs' registration record, on that audit's own declared circuit, `policy_on`, and at remote
sometimes every window of the period too, `windows_on`), hands every subprotocol the same blobs (`BLOBS`), then often
breaks the composition's own inputs (a component dropped, unknown, repeated or out of order, a case under other params,
a wrong or short commitment, non-canonical or malformed evidence, blobs missing or malformed) or the case itself (a
field missing or not hex, not an object), and every field of the outcome but its reason must agree exactly. Sampled
proofs' cases all reject: v2 refuses evidence naming no session (`oneSession`), and a session on the challenger's coin
trees is verified by `flock-verify`, which the mirror doesn't run. So every product rejects here, onsite often with a
component undecided beside the rejection. Remote's `windows` tie decides only where no component rejects, so this
difftest doesn't reach it; `CoreRemote`'s `#guard`s check which windows it finds stray. `product IDS
serve` over each composite's subprotocols, the composite `verity run`'s verifier runs, is held to the same reference,
and `product IDS id` names remote and onsite for their subprotocols and refuses every other set: the same two
composites a configuration may list (`verity.core.deploy.config.COMPOSITES`)."""
>>>>>>> 705be2b532

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import itertools
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

import periods
import reference as R
import verity_difftest as D
from verity_difftest import canonical

from verity.core.deploy import config as K
from verity.settings import REGISTRY as VERITY_SETTINGS
from verity.core.protocols.spot import parties as SPP

PROTOCOLS = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text(max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=3), inner, max_size=3),
                    max_leaves=6)


@functools.cache
def pouw_periods() -> dict[str, list[dict]]:
    """PoUW's periods by the verdict they aim at, made once (a period of the auditor's ledger and its windows is slow to
    make): a few honest ones, and one of each refusal of `credit.period_credit` and of each break of the protocol's
    framing. A period is accepted or rejected, never undecided."""
    PPC = R.POUW_PERIOD_CASES
    return {R.ACCEPT: [PPC.instance(random.Random(f"honest/{s}")) for s in range(3)],
            R.REJECT: [PPC.instance(random.Random(f"{n}/0"), n) for n in PPC.NAMES]}

OS = R.ONE_STAGE
#: The files every product case hands its components as blobs.
CIRCUIT, PUBLIC = b"not a circuit file", b"not a public file"
BLOBS = [CIRCUIT, PUBLIC]


<<<<<<< HEAD
@st.composite
def v1_empty_draws(draw) -> dict:
    """A valid one-stage v1 audit of a program that reads no registered value, whose draw drew no unit (a subset of none,
    or Bernoulli at 0): with no session and no proved unit, v1 is undecided on it for want of the verifier of record."""
    s = draw(st.sampled_from(OS.SETUPS))
    law = draw(st.sampled_from([{"k": 0, "law": "subset"}, {"law": "bernoulli", "p": "0/1"}]))
||||||| merged common ancestors
@st.composite
def empty_draws(draw) -> dict:
    """A valid audit of a program that reads no registered value, whose draw drew no unit (a subset of none, or Bernoulli
    at 0): with no session and no proved unit, v1 is undecided on it for want of the verifier of record."""
    s = draw(st.sampled_from(OS.SETUPS))
    law = draw(st.sampled_from([{"k": 0, "law": "subset"}, {"law": "bernoulli", "p": "0/1"}]))
=======
def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


@functools.cache
def _empty_audit(i: int) -> tuple[dict, dict, dict]:
    """One-stage v2's audit of `OS.SETUPS[i]`, a program that reads no registered value, its params but δ, its record
    and its evidence: its draw at its challenger's coins (`_challenger_draw`, a session whose challenger sampled no coin
    tree) drew no unit (a subset of none), so it names no session and no proved unit, and its params name the circuit
    and public files `BLOBS` hold. Made once each: a challenger's signed session is slow to make."""
    s, law = OS.SETUPS[i], {"k": 0, "law": "subset"}
>>>>>>> 705be2b532
    part = {"program": s.sha512, "query": s.query}
    rec = OS.R.record(s.program, s.query, law, OS.SCHEME, [{"port": p, "root": OS.ROOTS[p], **s.ports[p]} for p in s.ports],
                      OS.WINDOW, leaf_layer=OS.LEAF, partition=part)
    ch, d, coins = OS._challenger_draw(random.Random(f"composition/empty/{i}"), s, law, rec, trees=0)
    evidence = {"coins": coins, "draw": d, "instances": 0, "proofs": {"sessions": []}, "proved_units": [],
                "receipt": OS.R.receipt(rec, "v", received=OS.RECEIVED), "served": None, "session_roots": None,
                "sessions": [], "statement": {}}
    params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "law": law,
              "leaf_layer": OS.LEAF, "log": [], "partition": part, "ports": s.ports, "program": s.desc,
              "proofs": {"circuit": _sha(CIRCUIT), "public": _sha(PUBLIC), "registered": None}, "query": s.query,
              "registered": None, "scheme": OS.SCHEME, "window_root": ch.ledger.window["root"], "work_table": None}
    return params, rec, evidence


@st.composite
def empty_draws(draw) -> dict:
    """A one-stage v2 audit whose draw drew nothing (`_empty_audit`) at a drawn δ: it names no session, which v2 refuses
    at its challenger's coins before it reads a tree (`oneSession`: the drawn units are proved in one session)."""
    params, rec, evidence = _empty_audit(draw(st.integers(0, len(OS.SETUPS) - 1)))
    params = {**params, "delta": draw(st.sampled_from(["1/1000", "1/10", "1/2"]))}
    return OS._json({"params": params, "record": rec, "evidence": evidence})


<<<<<<< HEAD
#: Onsite's one-stage audit's cases by the verdict they aim at: an audit that drew nothing (undecided, for want of the
#: verifier of record), and an audit with one or two changes.
ONE_STAGE_CASES = {R.UNDECIDED: v1_empty_draws().map(OS._case),
                   R.REJECT: OS.changed_audits(most=2, least=1).map(OS._case)}


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


@functools.cache
def _empty_audit(i: int, after: str | None = None) -> tuple[dict, dict, dict]:
    """One-stage v2's audit of `OS.SETUPS[i]`, a program that reads no registered value, its params but δ, its record
    and its evidence: its draw at its challenger's coins (`_challenger_draw`, a session whose challenger sampled no coin
    tree) drew no unit (a subset of none), so it has no session and no proved unit, and `BLOBS` hold its circuit and
    public files. Its log is empty, or with `after` (a record's canonical JSON) that one record, which its record is
    chained right after. Made once each: a challenger's signed session is slow to make."""
    s, law = OS.SETUPS[i], {"k": 0, "law": "subset"}
    part = {"program": s.sha512, "query": s.query}
    log = [json.loads(after)] if after else []
    rec = OS.R.record(s.program, s.query, law, OS.SCHEME, [{"port": p, "root": OS.ROOTS[p], **s.ports[p]} for p in s.ports],
                      OS.WINDOW, leaf_layer=OS.LEAF, previous=OS.R.digest(log[-1]) if log else None, partition=part)
    ch, d, coins = OS._challenger_draw(random.Random(f"composition/empty/{i}"), s, law, rec, trees=0)
    evidence = {"coins": coins, "draw": d, "instances": 0, "proofs": {"public": _sha(PUBLIC), "sessions": []},
                "proved_units": [], "receipt": OS.R.receipt(rec, "v", received=OS.RECEIVED), "served": None,
                "session_roots": None, "sessions": [], "statement": {}}
    params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "law": law,
              "leaf_layer": OS.LEAF, "log": log, "partition": part, "ports": s.ports, "program": s.desc,
              "proofs": {"circuit": _sha(CIRCUIT), "registered": None}, "query": s.query, "registered": None,
              "scheme": OS.SCHEME, "window_root": ch.ledger.window["root"], "work_table": None}
    return params, rec, evidence


@st.composite
def empty_draws(draw) -> dict:
    """A valid one-stage v2 audit whose draw drew nothing (`_empty_audit`) at a drawn δ: with no session and no proved
    unit, v2 is undecided on it for want of the verifier of record."""
    params, rec, evidence = _empty_audit(draw(st.integers(0, len(OS.SETUPS) - 1)))
    params = {**params, "delta": draw(st.sampled_from(["1/1000", "1/10", "1/2"]))}
    return OS._json({"params": params, "record": rec, "evidence": evidence})


def _without(x: dict, *ks: str) -> dict:
    return {k: v for k, v in x.items() if k not in ks}


def _v1(a: dict) -> dict:
    """v2's audit `a` as one-stage v1's: its params without ``proofs``, its evidence with the verdicts of no session in
    place of ``proofs``, which v1 decides as v2 does."""
    return {**a, "params": _without(a["params"], "proofs"),
            "evidence": {**_without(a["evidence"], "proofs"), "verdicts": {}}}


#: The per-member form's one member over `empty_draws`' files: one unit, the circuit file `BLOBS` hold, no registered file,
#: one unit a proof block.
MEMBER = {"circuit": _sha(CIRCUIT), "registered": None, "units": [[0, 1]], "vus_per_block": 1}


def _per_member(a: dict, members: dict | None = None, statements: list | None = None) -> dict:
    """v2's audit `a` in the per-member form: the params' ``members`` (`MEMBER` as ``m0`` unless given) and the
    evidence's ``statements`` (none unless given: `a` drew nothing, so no member was drawn and none has a statement).
    With no statement and no proved unit it is undecided as `a` is."""
    members = {"m0": MEMBER} if members is None else members
    statements = [] if statements is None else statements
    return {**a, "params": {**a["params"], "proofs": {"members": members}},
            "evidence": {**a["evidence"], "proofs": {"statements": statements}}}


def _statement(member: str) -> dict:
    return {"member": member, "public": _sha(PUBLIC), "sessions": []}


#: One change to v2's audit (`empty_draws`) that sampled proofs refuses or rejects: the audit as v1's (refused by name
#: at the params), v1's evidence under v2's params (refused by name at the evidence), v2's evidence naming verdicts,
#: params naming no challenger or no partition, a circuit or public file no blob holds, proved units no session proved,
#: a session whose coins are no tree the challenger logged, coins on another window or none; and in the per-member form
#: (`_per_member`), a statement naming no member or of no session, two members' units meeting, a member's units, keys or
#: units a proof block malformed, a member's units outside the population (the registration's `link-points`), the
#: evidence's proofs in the other form than the params', and proved units no session proved. A statement's
#: other refusals (a member twice, statements out of the members' order, a circuit or public file in no blob, a public
#: file whose units are not exactly its member's) come after its one session's coins are held to a tree the challenger
#: logged, outside the mirror: sampled proofs' difftest holds them on the challenger's trees, and its `public_units`
#: piece holds the units rule to `proofs.units_exactly`.
V2_BREAKS = {
    "v1": _v1,
    "v1-evidence": lambda a: {**a, "evidence": _v1(a)["evidence"]},
    "verdicts": lambda a: {**a, "evidence": {**a["evidence"], "verdicts": {}}},
    "no-challenger": lambda a: {**a, "params": _without(a["params"], "challenger")},
    "no-partition": lambda a: {**a, "params": {**a["params"], "partition": None}},
    "circuit": lambda a: {**a, "params": {**a["params"], "proofs": {"circuit": _sha(b"another circuit"),
                                                                     "registered": None}}},
    "public": lambda a: {**a, "evidence": {**a["evidence"], "proofs": {"public": _sha(b"another public file"),
                                                                       "sessions": []}}},
    "proved": lambda a: {**a, "evidence": {**a["evidence"], "proved_units": [0]}},
    "session": lambda a: {**a, "evidence": {**a["evidence"], "sessions": ["s"], "proofs": {
        "public": _sha(PUBLIC), "sessions": [{"name": "s", "record": _sha(CIRCUIT), "reps": {}}]}}},
    "other-window": lambda a: {**a, "params": {**a["params"], "window_root": "ab" * 64}},
    "no-coins": lambda a: {**a, "evidence": _without(a["evidence"], "coins")},
    "member-unknown": lambda a: _per_member(a, statements=[_statement("m1")]),
    "member-no-session": lambda a: _per_member(a, statements=[_statement("m0")]),
    "members-meet": lambda a: _per_member(a, members={"m0": MEMBER, "m1": {**MEMBER, "units": [[0, 2]]}}),
    "member-units": lambda a: _per_member(a, members={"m0": {**MEMBER, "units": [[1, 1]]}}),
    "member-keys": lambda a: _per_member(a, members={"m0": _without(MEMBER, "units")}),
    "member-block": lambda a: _per_member(a, members={"m0": {**MEMBER, "vus_per_block": 0}}),
    "member-link-points": lambda a: _per_member(a, members={"m0": {**MEMBER, "units": [[0, 1000]]}}),
    "no-members": lambda a: _per_member(a, members={}),
    "member-evidence": lambda a: {**_per_member(a), "evidence": a["evidence"]},
    "member-params": lambda a: {**_per_member(a), "params": a["params"]},
    "member-proved": lambda a: {**_per_member(a), "evidence": {**_per_member(a)["evidence"], "proved_units": [0]}},
}


#: One break of sampled proofs' form that `VStar.Recursive.Sampled` refuses: params or evidence naming no form, another
#: form, or the recursive form with none of its files.
FORM_BREAKS = ["params-no-form", "params-form-other", "params-recursive", "evidence-no-form", "evidence-recursive"]


def _with_form(x, br: str | None, side: str):
    """Sampled proofs' form on the params or the evidence (`VStar.Recursive.Sampled`): ``one-stage``, unless `br` breaks
    it."""
    if not isinstance(x, dict) or br == f"{side}-no-form":
        return x
    return {**x, "form": {f"{side}-form-other": "two-stage", f"{side}-recursive": "recursive"}.get(br, "one-stage")}


def form_case(a: dict, br: str | None = None) -> dict:
    """An audit (`{"params", "record", "evidence"}`) as sampled proofs' case under the form ``one-stage`` (one-stage v2's,
    `Spot.OneStageZk`, or v1's, which the form refuses), with `br`, one of `FORM_BREAKS`, when given."""
    return OS._case({"params": _with_form(a["params"], br, "params"), "record": a["record"],
                     "evidence": _with_form(a["evidence"], br, "evidence")})


def broken_case(a: dict, br: str) -> dict:
    """v2's audit `a` with the change `V2_BREAKS[br]`, as sampled proofs' case."""
    return form_case(V2_BREAKS[br](a))


#: Remote's sampled proofs' cases by the verdict they aim at: v2's audit that drew nothing (undecided, for want of the
#: verifier of record), in either form, and v2's audit with one change or one break of the form, or v1's audit with up
#: to two changes (refused by name).
SAMPLED_CASES = {R.UNDECIDED: st.one_of(empty_draws().map(form_case), empty_draws().map(_per_member).map(form_case)),
                 R.REJECT: st.one_of(st.builds(broken_case, empty_draws(), st.sampled_from(list(V2_BREAKS))),
                                     st.builds(form_case, empty_draws(), st.sampled_from(FORM_BREAKS)),
                                     OS.changed_audits(most=2).map(form_case))}
||||||| merged common ancestors
#: The one-stage audit's cases by the verdict they aim at: an audit that drew nothing (undecided, for want of the
#: verifier of record), and an audit with one or two changes.
ONE_STAGE_CASES = {R.UNDECIDED: empty_draws().map(OS._case), R.REJECT: OS.changed_audits(most=2, least=1).map(OS._case)}
=======
def _without(x: dict, *ks: str) -> dict:
    return {k: v for k, v in x.items() if k not in ks}


def _v1(a: dict) -> dict:
    """v2's audit `a` as one-stage v1's: its params without ``proofs``, its evidence with the verdicts of no session in
    place of ``proofs``, on which v1 is undecided, for want of the verifier of record."""
    return {**a, "params": _without(a["params"], "proofs"),
            "evidence": {**_without(a["evidence"], "proofs"), "verdicts": {}}}


#: One change to v2's audit (`empty_draws`) that sampled proofs refuses at another step than its refusal of no
#: session: the audit as v1's (refused by name at the params), v1's evidence under v2's params (refused by name at the
#: evidence), v2's evidence naming verdicts, params naming no challenger or no partition, the public file named by the
#: evidence rather than the params, one session whose coins are no tree the challenger logged, coins on another window
#: or none. v2 reads no file and no proved unit before that refusal, so a change to those reaches it alone.
V2_BREAKS = {
    "v1": _v1,
    "v1-evidence": lambda a: {**a, "evidence": _v1(a)["evidence"]},
    "verdicts": lambda a: {**a, "evidence": {**a["evidence"], "verdicts": {}}},
    "no-challenger": lambda a: {**a, "params": _without(a["params"], "challenger")},
    "no-partition": lambda a: {**a, "params": {**a["params"], "partition": None}},
    "public-in-evidence": lambda a: {**a, "evidence": {**a["evidence"], "proofs": {
        "public": a["params"]["proofs"]["public"], "sessions": []}}},
    "session": lambda a: {**a, "evidence": {**a["evidence"], "sessions": ["s"], "proofs": {
        "sessions": [{"name": "s", "record": _sha(CIRCUIT), "reps": {}}]}}},
    "other-window": lambda a: {**a, "params": {**a["params"], "window_root": "ab" * 64}},
    "no-coins": lambda a: {**a, "evidence": _without(a["evidence"], "coins")},
}


#: One break of sampled proofs' form that `VStar.Recursive.Sampled` refuses: params or evidence naming no form, another
#: form, or the recursive form with none of its files.
FORM_BREAKS = ["params-no-form", "params-form-other", "params-recursive", "evidence-no-form", "evidence-recursive"]


def _with_form(x, br: str | None, side: str):
    """Sampled proofs' form on the params or the evidence (`VStar.Recursive.Sampled`): ``one-stage``, unless `br` breaks
    it."""
    if not isinstance(x, dict) or br == f"{side}-no-form":
        return x
    return {**x, "form": {f"{side}-form-other": "two-stage", f"{side}-recursive": "recursive"}.get(br, "one-stage")}


def form_case(a: dict, br: str | None = None) -> dict:
    """An audit (`{"params", "record", "evidence"}`) as sampled proofs' case under the form ``one-stage`` (one-stage v2's,
    `Spot.OneStageZk`, or v1's, which the form refuses), with `br`, one of `FORM_BREAKS`, when given."""
    return OS._case({"params": _with_form(a["params"], br, "params"), "record": a["record"],
                     "evidence": _with_form(a["evidence"], br, "evidence")})


def broken_case(a: dict, br: str) -> dict:
    """v2's audit `a` with the change `V2_BREAKS[br]`, as sampled proofs' case."""
    return form_case(V2_BREAKS[br](a))


#: Sampled proofs' cases, each refused: v2's audit that drew nothing (naming no session), the same with one change or
#: one break of the form, or v1's audit with up to two changes (refused by name).
SAMPLED_CASES = st.one_of(empty_draws().map(form_case),
                          st.builds(broken_case, empty_draws(), st.sampled_from(list(V2_BREAKS))),
                          st.builds(form_case, empty_draws(), st.sampled_from(FORM_BREAKS)),
                          OS.changed_audits(most=2).map(form_case))
>>>>>>> 705be2b532

NP = R.PERIOD
#: The period check's changes by the verdict they aim at (`case` breaks the line `serve` reads, as PoUW's `case-*`).
PERIOD_CASES = {v: tuple(m for m, x in NP.EXPECT.items() if x == v and m != "case")
                for v in (R.ACCEPT, R.REJECT, R.UNDECIDED)}

#: The inference-only policy's changes by the verdict they aim at: none, or one that rejects. It is never undecided.
INFERENCE_CASES = {v: tuple(m for m, (x, _) in R.INFERENCE.EXPECT.items() if x == v) for v in (R.ACCEPT, R.REJECT)}

#: A recorded component's kinds of case (`periods.build`), accepted ones first and thrice.
KINDS = [R.ACCEPT] * 3 + [R.REJECT, R.UNDECIDED, "other-setup", periods.OTHER_STORE, "broken-params", "broken-evidence"]


def kinds(C) -> list[str]:
    """The kinds of case `component_case` may aim `C` at when it is free to."""
    if isinstance(C, R.Recorded):
        return [k for k in KINDS if k in C.cases]
    if C is R.INFERENCE_ONLY:
        return list(INFERENCE_CASES)
    if C is R.SAMPLED_PROOFS:
<<<<<<< HEAD
        return list(SAMPLED_CASES)
    if C is R.ONE_STAGE_V1:
        return list(ONE_STAGE_CASES)
||||||| merged common ancestors
        return list(ONE_STAGE_CASES)
=======
        return [R.REJECT]
>>>>>>> 705be2b532
    if C is R.WIPE:
        return list(R.WIPE_CASES.KINDS)
    if C is R.POUW_PERIOD:
        return [R.ACCEPT, R.REJECT]
    return [R.ACCEPT, R.REJECT, R.UNDECIDED]


@st.composite
def component_case(draw, C, kind: str | None) -> dict:
    """`C`'s case aimed at `kind` (any of `kinds(C)` when it is None; sampled proofs' are all refused, whatever the
    aim), from its own difftest's generator or its recorded cases, with its digest as its Python reads it on `BLOBS` (the
    case's own commitment when it doesn't decode, which a rejection never binds)."""
    kind = kind or draw(st.sampled_from(kinds(C)))
    if C is R.POUS and kind == R.ACCEPT and draw(st.integers(0, 3)) == 0:
        kind = periods.OTHER_STORE
    if isinstance(C, R.Recorded):
        case = C.cases[kind][0]
    elif C is R.POUW_PERIOD:
        case = draw(st.sampled_from(pouw_periods()[kind]))
    elif C is R.CERTIFIER:
        case = NP.build(draw(st.sampled_from(PERIOD_CASES[kind])), draw(st.integers(0, 1 << 16)), small=True)
    elif C is R.INFERENCE_ONLY:
        case = draw(R.INFERENCE.drawn(draw(st.sampled_from(INFERENCE_CASES[kind]))))
    elif C is R.WIPE:
        case = R.WIPE_CASES.case(kind, draw(st.integers(0, 1)))
    else:
<<<<<<< HEAD
        cases = SAMPLED_CASES if C is R.SAMPLED_PROOFS else ONE_STAGE_CASES
        case = draw(cases[R.UNDECIDED if kind == R.ACCEPT else kind])
||||||| merged common ancestors
        case = draw(ONE_STAGE_CASES[R.UNDECIDED if kind == R.ACCEPT else kind])
=======
        case = draw(SAMPLED_CASES)
>>>>>>> 705be2b532
    return as_component(C, case)


def as_component(C, case: dict) -> dict:
    """`C`'s case as a composite's entry, with its digest as its Python reads it on `BLOBS` (the case's own commitment
    when it doesn't decode, which a rejection never binds)."""
    root = bytes.fromhex(case["commitment"])
    digest = C.answer({**case, "blobs": BLOBS} if getattr(C, "blobs", False) else case)["commitment"]
    return {"id": C.id, "params": case["params"], "root": root, "evidence": bytes.fromhex(case["evidence"]),
            "digest": bytes.fromhex(digest) if digest else root}


@functools.cache
def _declaration(program: str) -> tuple[dict, dict] | None:
    """The policy's params but the log, and its evidence, for the declared circuit `program` (descriptor text) of a
    program the one-stage audit knows (`ONE_STAGE.PROGRAMS`), as the provenance difftest's declarations make them
    (`baseline_difftest_vectors.build`): its program function's parameters, every gate in one inference block with Γ its
    gates and α over what any gate reads, and no registered information. None for another program."""
    NR = R.INFERENCE.NR
    digest = R.INFERENCE._digest(program)
    if isinstance(digest, Exception) or digest not in OS.PROGRAMS:
        return None
    desc = json.loads(program)
    gates = NR.codec.decode_program(desc, NR.declared(desc)).root.gates
    inputs, _, _ = NR.root_nodes(program)
    params = {"alpha": [64 * gates + 1, 1], "gamma": gates, "program": program,
              "parameters": [[n, t.to_json()] for n, t in OS.PROGRAMS[digest].fn.signature()[0]]}
    return params, {"owner": ["input" if i else 0 for i in inputs], "provenance": [], "theta": []}


def policy_on(anchor: dict) -> dict | None:
<<<<<<< HEAD
    """The policy's case, as `component_case` gives it, on the record and log of integrity's case `anchor`, declaring the
||||||| merged common ancestors
    """The policy's case, as `component_case` gives it, on the record and log of the one-stage audit's case `anchor`, declaring the
=======
    """The policy's case, as `component_case` gives it, on the record and log of sampled proofs' case `anchor`, declaring the
>>>>>>> 705be2b532
    audit's own circuit (`_declaration`), which the policy accepts when the record chains; None when the audit's params
    name no program it knows."""
    p = anchor["params"]
    if not isinstance(p, dict) or not isinstance(p.get("program"), dict) or not isinstance(p.get("log"), list):
        return None
    made = _declaration(canonical(p["program"]))
    if made is None:
        return None
    params, evidence = {**made[0], "log": p["log"]}, canonical(made[1]).encode()
    digest = R.INFERENCE_ONLY.answer({"commitment": anchor["root"].hex(), "evidence": evidence.hex(),
                                      "params": params})["commitment"]
    return {"id": R.INFERENCE_ONLY.id, "params": params, "root": anchor["root"], "evidence": evidence,
            "digest": bytes.fromhex(digest) if digest else anchor["root"]}


def windows_on(evidence: bytes, commit: bytes) -> bytes:
    """A period's evidence (JSON, as every period case's is) with every window's Commit record ``commit``; the evidence
    as it is when it holds no list of windows."""
    j = json.loads(evidence)
    ws = j.get("windows") if isinstance(j, dict) else None
    if not isinstance(ws, list):
        return evidence
    ws = [{**w, "commit": commit.hex()} if isinstance(w, dict) else w for w in ws]
    return bytes.fromhex(R.POUW_PERIOD_CASES._hex({**j, "windows": ws}))


def one_record(product: R.Same, cases: list[dict], windows: bool = False) -> list[dict]:
<<<<<<< HEAD
    """The cases with every subprotocol of ``product.same`` committing to integrity's registration record (remote's
    sampled proofs', onsite's one-stage audit's), each with its digest as its Python reads it on `BLOBS`, and the
    policy's case on the audit's own declared circuit where it can be (`policy_on`): cases that pass `Core.Composition.strays` and leave the verdict to the subprotocols.
||||||| merged common ancestors
    """The cases with every subprotocol of ``product.same`` committing to the one-stage audit's registration record,
    each with its digest as its Python reads it on `BLOBS`, and the policy's case on the audit's own declared circuit
    where it can be (`policy_on`): cases that pass `Core.Composition.strays` and leave the verdict to the subprotocols.
=======
    """The cases with every subprotocol of ``product.same`` committing to sampled proofs' registration record, each
    with its digest as its Python reads it on `BLOBS`, and the policy's case on the audit's own declared circuit
    where it can be (`policy_on`): cases that pass `Core.Composition.strays` and leave the verdict to the subprotocols.
>>>>>>> 705be2b532
    With ``windows``, every window of remote's PoUW period names that record too (`windows_on`), so that remote's
    `windows` tie holds; the period then refuses, since a window's Commit record registers its own tables. Without it,
    an honest period's windows name their own records, and the tie refuses."""
    first = next(c for c in cases if c["id"] in (R.SAMPLED_PROOFS.id, R.ONE_STAGE_V1.id))
    by_id = {C.id: C for C in product.components}

    def recommit(c: dict, root: bytes, evidence: bytes) -> dict:
        C = by_id[c["id"]]
        case = {"commitment": root.hex(), "evidence": evidence.hex(), "params": c["params"]}
        digest = C.answer({**case, "blobs": BLOBS} if getattr(C, "blobs", False) else case)["commitment"]
        return {**c, "root": root, "evidence": evidence, "digest": bytes.fromhex(digest) if digest else root}

    def on(c: dict) -> dict:
        if windows and c["id"] == R.POUW_PERIOD.id:
            return recommit(c, c["root"], windows_on(c["evidence"], first["root"]))
        if c["id"] not in product.same or c is first:
            return c
        if c["id"] == R.INFERENCE_ONLY.id and (made := policy_on(first)):
            return made
        return recommit(c, first["root"], c["evidence"])
    return [on(c) for c in cases]


#: The generic remote bandwidth audit's id: not a component of onsite, whose bandwidth is PoUB-P2 (`generic-bandwidth`
#: names PoUB-P2's case by it, its params and its entry, under a commitment that binds it).
GENERIC_BANDWIDTH = "verity/remote-bandwidth/v0"
#: The binding breaks come five times: with `commitment-size` they are the breaks that leave every component's case standing,
#: and so the way to a rejection that no component's own explains.
MUTATIONS = [None] * 20 + ["drop-param", "drop-case", "unknown-param", "unknown-case", "swap-params", "swap-cases",
                           "repeat-param", "repeat-case", "case-params", "commitment-size", "evidence-noncanonical",
                           "params-extra-key", "case-extra-key", "case-missing-key", "bad-hex", "garbage-evidence",
                           "garbage-params", "not-a-case", "no-blobs", "blobs-not-hex", "blobs-not-list",
                           "blob-extra", "generic-bandwidth"] + ["binding", "binding-id"] * 5
#: The breaks of the commitment alone, which leave every component's case standing.
BINDING = ["binding", "binding-id", "commitment-size"]


@st.composite
def composite(draw, product: R.Product, mutation: str | None = None) -> dict:
    """One case of `product`: its subprotocols' cases (one time in six every one aimed at acceptance, one in six the same
<<<<<<< HEAD
    but for one of those that can be undecided, aimed at undecided, so that a product of seven reaches its binding breaks
    and its undecided verdict, and each at any verdict otherwise), the subprotocols it ties recommitted to one record
    (`one_record`) two times in three when they are aimed at acceptance and one in three otherwise, then at most one
    break of the composition's own inputs (half the time a break of the commitment alone, `BINDING`, when every case is
    aimed at acceptance), or the break `mutation`, one of `MUTATIONS`, when given."""
||||||| merged common ancestors
    but for one of those that can be undecided, aimed at undecided, so that a product of seven reaches its binding breaks
    and its undecided verdict, and each at any verdict otherwise), the subprotocols it ties recommitted to one record
    (`one_record`) two times in three when they are aimed at acceptance and one in three otherwise, then at most one
    break of the composition's own inputs (half the time a break of the commitment alone, `BINDING`, when every case is
    aimed at acceptance)."""
=======
    but for one of those that can be undecided, when one can, aimed at undecided, so that a product of seven reaches its
    binding breaks with a component undecided, and each at any verdict otherwise), the subprotocols it ties recommitted
    to one record (`one_record`) two times in three when they are aimed at acceptance and one in three otherwise, then
    at most one break of the composition's own inputs (half the time a break of the commitment alone, `BINDING`, when
    every case is aimed at acceptance)."""
>>>>>>> 705be2b532
    mode = draw(st.integers(0, 5))
    undecided = [j for j, C in enumerate(product.components) if R.UNDECIDED in kinds(C)]
    pending = draw(st.sampled_from(undecided)) if undecided else None
    aims = [None if mode > 1 else R.UNDECIDED if mode == 1 and j == pending else R.ACCEPT
            for j in range(len(product.components))]
    cases = [draw(component_case(C, k)) for C, k in zip(product.components, aims)]
    if isinstance(product, R.Same) and draw(st.integers(0, 2)) < (2 if mode <= 1 else 1):
        cases = one_record(product, cases, R.WINDOWS in getattr(product, "ties", ()) and draw(st.booleans()))
    named = [{"id": c["id"], "params": c["params"]} for c in cases]
    entries = [{"commitment": c["root"].hex(), "evidence": c["evidence"].hex(), "id": c["id"], "params": c["params"]}
               for c in cases]
    commitment = R.bind(product.id, [(c["id"], c["digest"]) for c in cases])
    params: object = {"components": named}
    m = mutation or draw(st.sampled_from(BINDING if mode <= 1 and draw(st.booleans()) else MUTATIONS))
    i = draw(st.integers(0, len(cases) - 1))
    which = named if m and m.endswith("-param") or m in ("swap-params", "repeat-param") else entries
    if m in ("drop-param", "drop-case"):
        del which[i]
    elif m in ("unknown-param", "unknown-case"):
        stranger = dict(which[i], id=draw(st.sampled_from([GENERIC_BANDWIDTH, "verity/stub/nci/v0", "verity/example/v0",
                                                           "", "a"])))
        which.insert(draw(st.integers(0, len(which))), stranger)
    elif m in ("swap-params", "swap-cases"):
        which[0], which[-1] = which[-1], which[0]
    elif m in ("repeat-param", "repeat-case"):
        which.insert(i, dict(which[i]))
    elif m == "generic-bandwidth":
        for x in (*named, *entries):
            x["id"] = GENERIC_BANDWIDTH if x["id"] == R.BANDWIDTH.id else x["id"]
        commitment = R.bind(product.id, [(GENERIC_BANDWIDTH if c["id"] == R.BANDWIDTH.id else c["id"], c["digest"])
                                         for c in cases])
    elif m == "case-params":
        entries[i]["params"] = {"k": draw(st.integers(4, 6))}
    elif m == "binding":
        commitment = draw(st.binary(min_size=64, max_size=64).filter(lambda b: b != commitment))
    elif m == "binding-id":
        commitment = R.bind("verity/other/v0", [(c["id"], c["digest"]) for c in cases])
    elif m == "commitment-size":
        commitment = draw(st.binary(max_size=70).filter(lambda b: len(b) != 64))
    elif m == "params-extra-key":
        params = {"components": named, "x": 1}
    elif m == "case-extra-key":
        entries[i]["x"] = 1
    elif m == "case-missing-key":
        del entries[i][draw(st.sampled_from(["commitment", "evidence", "id", "params"]))]
    elif m == "bad-hex":
        entries[i]["commitment"] = draw(st.sampled_from(["G", "0", "A"])) + entries[i]["commitment"][1:]
    elif m == "garbage-params":
        params = draw(JSON)
    body: dict = {"components": entries}
    evidence = canonical(body).encode()
    if m == "evidence-noncanonical":
        evidence = json.dumps(body).encode()
    elif m == "garbage-evidence":
        evidence = draw(st.binary(max_size=24))
    if m == "not-a-case":
        return draw(st.sampled_from([{"commitment": commitment.hex(), "params": params},
                                     {"commitment": "zz", "evidence": evidence.hex(), "params": params},
                                     {"commitment": commitment.hex().upper(), "evidence": evidence.hex(), "params": params},
                                     {"commitment": commitment.hex(), "evidence": 7, "params": params}, []]))
    blobs: object = [b.hex() for b in BLOBS]
    if m == "blobs-not-hex":
        blobs = [*blobs[:1], draw(st.sampled_from(["zz", "A0", "0", 7]))]
    elif m == "blobs-not-list":
        blobs = draw(st.sampled_from([CIRCUIT.hex(), {}, None]))
    elif m == "blob-extra":
        blobs = [draw(st.binary(max_size=8)).hex(), *blobs]
    case = {"commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}
    return case if m == "no-blobs" else {**case, "blobs": blobs}


def reference(product: R.Product, seen: collections.Counter):
    """The Python side of `serve`: a case's outcome. Counts the verdicts it gives in `seen`, the rejections with an
    undecided component among them, and those no component's rejection explains."""
    def answer(case):
        out = product.answer(case)
        seen[out["verdict"]] += 1
        parts = product.verdicts(case)
        if out["verdict"] == R.REJECT and R.UNDECIDED in parts:
            seen["reject, a component undecided"] += 1
        if out["verdict"] == R.REJECT and parts and R.REJECT not in parts:
            seen["reject, no component rejected"] += 1
        if isinstance(product, R.Same) and (got := product.committed(case)) and None not in got:
            seen["two records" if len(set(got)) > 1 else "one record"] += 1
        if isinstance(product, R.TiedSame) and isinstance(case, dict) and isinstance(case.get("evidence"), str) \
                and GENERIC_BANDWIDTH.encode().hex() in case["evidence"]:
            assert out["verdict"] == R.REJECT, out
            seen["generic bandwidth"] += 1
        if isinstance(product, R.TiedSame):
            for what, one in product.tied(case):
                seen[f"one {what}" if one else f"two {what}s"] += 1
        return out
    return answer


<<<<<<< HEAD
class UndecidedPort(D.ProtocolPort):
    """A product port whose best verdict is undecided: integrity's best case is undecided (`ONE_STAGE_CASES`,
    `SAMPLED_CASES`), so the run's non-vacuity bar is a case every component left undecided or accepted."""
    accepted = R.UNDECIDED


||||||| merged common ancestors
class UndecidedPort(D.ProtocolPort):
    """A product port whose best verdict is undecided: integrity's best case is undecided (`ONE_STAGE_CASES`), so the
    run's non-vacuity bar is a case every component left undecided or accepted."""
    accepted = R.UNDECIDED


=======
>>>>>>> 705be2b532
class RejectingPort(D.ProtocolPort):
<<<<<<< HEAD
    """A product port whose sampled cases reject: every window of its PoUW period must name the policy's record, the
    sampled proofs', and a period accepts only when each window names its own Commit record, so the sampled cases
    don't reach undecided. Its bar is the two ways its subprotocols' records meet, counted in `seen`."""
||||||| merged common ancestors
    """A product port whose sampled cases reject: every window of its PoUW period must name the policy's record, the
    one-stage audit's, and a period accepts only when each window names its own Commit record, so the sampled cases
    don't reach undecided. Its bar is the two ways its subprotocols' records meet, counted in `seen`."""
=======
    """A product port whose cases reject: sampled proofs' do (`SAMPLED_CASES`), so the run's non-vacuity bar is a
    rejection, and its other bars are the ways its subprotocols' verdicts and records meet, counted in `seen`
    (`_agree`)."""
>>>>>>> 705be2b532
    accepted = R.REJECT


def port(name: str, product: R.Product, seen: collections.Counter, serve: tuple[str, ...] = ("serve",),
         package: str | None = None) -> D.ProtocolPort:
    """`name serve` (or `name SERVE`) of the Lake package in `package` (default `name`), on the product's cases and,
    one sample in ten, any JSON line."""
    return RejectingPort(name, reference(product, seen), st.one_of(*[composite(product)] * 9, JSON),
                         package=PROTOCOLS / (package or name), serve=serve, need_rejects=False)


def ids(product: R.Product) -> str:
    return ",".join(C.id for C in product.components)


SEEN = {"onsite": collections.Counter(), "remote": collections.Counter()}
ONSITE = port("onsite", R.ONSITE, SEEN["onsite"], UndecidedPort)
REMOTE = port("remote", R.REMOTE, SEEN["remote"], RejectingPort)
#: `product IDS serve` over onsite's subprotocols, which `verity run`'s verifier runs (remote v1's Pearl+ run has a
#: reference case only on a served run, `test_remote_v1_on_a_served_pearl_plus_run`, and H1's development composite is
#: no product)
PRODUCT = {R.ONSITE.id: port("product", R.ONSITE, SEEN["onsite"], UndecidedPort, serve=(ids(R.ONSITE), "serve"),
                             package="onsite")}
onsite_server = D.server_fixture(ONSITE)
remote_server = D.server_fixture(REMOTE)
product_onsite_server = D.server_fixture(PRODUCT[R.ONSITE.id])


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """Space's and bandwidth's cases, made once (their memory challenger's session logs under the suite's basetemp), and
    the wipe's test signer's files under it too while the module's tests draw wipes."""
    if not R.POUS.cases:
        periods.build(tmp_path_factory.mktemp("periods"))
    ts = R.WIPE_CASES.M.ts
    ts.DIR = tmp_path_factory.mktemp("test-signer")
    yield
    ts.DIR = None


def test_the_rule_changes_the_trial_only_where_a_rejection_meets_an_undecided_component():
    """Over every list of up to four verdicts, the composition's rule and the trial's agree but where a component rejects while
    another is undecided: the trial waits for the missing statement, and the port rejects."""
    for n in range(1, 5):
        for vs in itertools.product([R.ACCEPT, R.REJECT, R.UNDECIDED], repeat=n):
            vs = list(vs)
            if R.REJECT in vs and R.UNDECIDED in vs:
                assert (R.trial_verdict(vs), R.verdict(vs)) == (R.UNDECIDED, R.REJECT), vs
            else:
                assert R.trial_verdict(vs) == R.verdict(vs), vs


def _agree(p: D.ProtocolPort, server, seen: collections.Counter, examples: int,
           kinds: tuple[str, ...] = (R.REJECT,)) -> None:
    """Lean and Python agree on every case, and the cases reach each of `kinds` (by default a rejection: sampled proofs'
    cases are all refused, so no product accepts)."""
    seen.clear()
    stats = D.run(p, examples=examples, seed=0, server=server)
    print(f"{p.label}: {stats}, verdicts {dict(seen)}")
    assert seen[R.ACCEPT] == 0, dict(seen)
    assert stats.examples == examples and stats.counts == {v: seen[v] for v in (R.REJECT, R.UNDECIDED) if seen[v]}
    for kind in kinds:
        assert seen[kind] > examples // 50, (kind, dict(seen))


@pytest.mark.slow
def test_remote_agrees(remote_server):
    """Remote's components meet on one record (left to the components) and on two (refused), each often."""
    _agree(REMOTE, remote_server, SEEN["remote"], 1000, (R.REJECT, "one record", "two records"))


def test_remote_refuses_two_records_and_else_keeps_the_products_verdict():
    """On drawn cases remote's verdict is its product's without the record checks (`Core.Composition.product`,
    remote before them), except where its policy and integrity committed to two records, or a window of its PoUW period
    names another record than the policy's (`windows`), which it refuses: the checks change no other verdict. Python
    only: the difftest holds the Lean to it. Drawn from a fixed seed, as `verity_difftest.run` draws, so the counts it
    asks of the draw are the generator's, not a run's luck."""
    product = R.Product(R.REMOTE.id, R.REMOTE.components)
    seen = collections.Counter()

    @seed(0)
    @settings(max_examples=200, database=None, deadline=None, derandomize=True, suppress_health_check=list(HealthCheck))
    @given(composite(R.REMOTE))
    def check(case):
        got, was = R.REMOTE.answer(case), product.answer(case)
        committed = R.REMOTE.committed(case)
        two = bool(committed) and None not in committed and len(set(committed)) > 1 \
            or ("record", False) in R.REMOTE.tied(case)
        seen["two" if two else "one"] += 1
        if two:
            assert got["verdict"] == R.REJECT, got
            assert got["reason"].startswith("record: ") or was["verdict"] == R.REJECT, (got, was)
        else:
            assert {k: v for k, v in got.items() if k != "reason"} == {k: v for k, v in was.items() if k != "reason"}
    check()
    assert seen["two"] > 20 and seen["one"] > 20, dict(seen)


def remote_case(cases: list[dict], product: R.Product = R.REMOTE) -> dict:
    """Remote's case (or `product`'s) on its components' cases as they are, with no break of its own."""
    return {"commitment": R.bind(product.id, [(c["id"], c["digest"]) for c in cases]).hex(),
            "evidence": canonical({"components": [{"commitment": c["root"].hex(), "evidence": c["evidence"].hex(),
                                                   "id": c["id"], "params": c["params"]} for c in cases]}).encode().hex(),
            "params": {"components": [{"id": c["id"], "params": c["params"]} for c in cases]},
            "blobs": [b.hex() for b in BLOBS]}


def test_spot_refuses_one_stage_v1(remote_server):
    """Sampled proofs' one-stage form is one-stage v2: v1's audit under it, one v1 itself decides (undecided, on an
    empty draw at its challenger's coins), is refused by name (`VStar.Recursive.Sampled.V1_REFUSED`) at its params,
    and v1's evidence under v2's params at the evidence. Remote, its policy and period aimed at acceptance on the
    audit's record, rejects with that refusal as sampled proofs' reason, in Lean as in Python."""
    want = {"v1": f"params: the parameters name no proofs: {R.V1_REFUSED}",
            "v1-evidence": f"evidence: the evidence names no proofs: {R.V1_REFUSED}"}
    seen = collections.Counter()

    @settings(max_examples=3, database=None, deadline=None, suppress_health_check=list(HealthCheck))
    @given(component_case(R.INFERENCE_ONLY, R.ACCEPT), component_case(R.POUW_PERIOD, R.ACCEPT), empty_draws())
    def check(policy, period, audit):
        assert OS.answer(OS._case(_v1(audit)))["verdict"] == R.UNDECIDED
        for br, why in want.items():
            sampled = broken_case(audit, br)
            got = R.SAMPLED_PROOFS.answer({**sampled, "blobs": BLOBS})
            assert (got["verdict"], got["reason"]) == (R.REJECT, why), got
            case = remote_case(one_record(R.REMOTE, [policy, period, as_component(R.SAMPLED_PROOFS, sampled)]))
            assert R.REMOTE.answer(case)["verdict"] == R.REJECT
            out = remote_server.exchange(case)
            assert out["verdict"] == R.REJECT and f"{R.SAMPLED_PROOFS.id} reject: {why}" in out["reason"], out
            seen[br] += 1
    check()
    assert set(seen) == set(want), dict(seen)


def test_remote_v1_ties_pearl_plus_run_to_the_policys_record():
    """Remote v1's `run` tie, as `Core.Remote.V1`'s guards state it: a Pearl+ run naming the digest of the policy's
    record holds, one naming another record's is a stray naming both, and a run whose commitment doesn't decode (an
    extra key, a short digest, none) leaves Pearl+ to refuse it."""
    assert R.REMOTE_V1.ties == (R.RUN, R.WINDOW) and [s for s, _ in R.RUN.names] == [R.INFERENCE_ONLY.id, R.PEARL_PLUS.id]

    def record(n: int) -> bytes:
        return canonical({"format": "verity/registration/v2", "n": n, "registered": []}).encode()

    def digest(n: int) -> bytes:
        return R.policy_record(record(n), b"")[0]

    def run(on: bytes, **extra) -> bytes:
        window = {"dropped": [], "record": json.loads(record(9)), "rounds": None}
        return canonical({"manifests": [{}], "record": on.hex(), "window": window, **extra}).encode()

    def strays(policy: int, commitment: bytes) -> str | None:
        return R.RUN.strays([(R.INFERENCE_ONLY.id, "{}", record(policy), b""),
                             (R.PEARL_PLUS.id, "{}", commitment, b"")])

    assert len(digest(0)) == 64 and R.run_record(run(digest(0)), b"") == [digest(0)]
    assert strays(0, run(digest(0))) is None
    assert strays(1, run(digest(0))) == (f"record: {R.INFERENCE_ONLY.id} committed {digest(1).hex()}, "
                                         f"{R.PEARL_PLUS.id} committed {digest(0).hex()}")
    for undecoded in (b"", run(digest(0), x=1), run(digest(0)[:32]),
                      canonical({"manifests": [], "record": digest(0).hex()}).encode(),
                      canonical({"manifests": [{}], "record": digest(0).hex()}).encode()):
        assert not isinstance(R.run_record(undecoded, b""), list) and strays(1, undecoded) is None, undecoded


def test_remote_v1_ties_pearl_plus_window_record_to_the_one_the_policys_record_is_chained_after():
    """Remote v1's `window` tie, as `Core.Remote.V1`'s guards state it: the policy's record chained right after the
    window record W Pearl+'s check holds the passes to holds; one chained after another record, or after none, is a
    stray; and a window Pearl+'s decoder refuses (other keys, W not an object, rounds or dropped not lists of
    nonnegative integers) leaves Pearl+ to refuse it."""
    assert [s for s, _ in R.WINDOW.names] == [R.INFERENCE_ONLY.id, R.PEARL_PLUS.id]

    def record(n: int, **extra) -> dict:
        return {"format": "verity/registration/v2", "n": n, "registered": [], **extra}

    def digest(r: dict) -> bytes:
        return R.policy_record(canonical(r).encode(), b"")[0]

    def run(w: object, **window) -> bytes:
        return canonical({"manifests": [{}], "record": "00" * 64,
                          "window": {"dropped": [], "record": w, "rounds": None, **window}}).encode()

    def strays(policy: dict, commitment: bytes) -> str | None:
        return R.WINDOW.strays([(R.INFERENCE_ONLY.id, "{}", canonical(policy).encode(), b""),
                                (R.PEARL_PLUS.id, "{}", commitment, b"")])

    w = record(9)
    assert R.run_window(run(w), b"") == [digest(w)] == [bytes.fromhex(R.REG.digest(w))]
    assert R.policy_previous(canonical(record(0, previous=digest(w).hex())).encode(), b"") == [digest(w)]
    assert strays(record(0, previous=digest(w).hex()), run(w)) is None
    assert strays(record(0, previous=digest(record(8)).hex()), run(w)) == (
        f"window: {R.INFERENCE_ONLY.id} committed {digest(record(8)).hex()}, {R.PEARL_PLUS.id} committed {digest(w).hex()}")
    assert strays(record(0), run(w)) == f"window: {R.INFERENCE_ONLY.id} committed , {R.PEARL_PLUS.id} committed {digest(w).hex()}"
    for undecoded in (run([]), run(w, rounds=[-1]), run(w, rounds=[True]), run(w, dropped=None), run(w, x=1)):
        assert R.run_window(undecoded, b"") is None and strays(record(0), undecoded) is None, undecoded


#: A directory holding a served Pearl+ run (`test_remote_v1_on_a_served_pearl_plus_run`).
SERVED = VERITY_SETTINGS.VERITY_COMPOSITION_SERVED
#: The served run's window record W's digest and its run commitment (the dress rehearsal r20261105-082550-ed23's).
SERVED_W = ("50214c57da3ddafad760f1059eb3e2c1b8dd70e45c4906f308215f32589cc504"
            "1901e26b5d927f6249c4d813ba62b3d2a3a531f2652b61941dbab322d3803601")
SERVED_RUN = ("99841564d3beb02456ad3397c509ef236dd5bcd31e65b2d8187baca7e14b2f91"
              "e0e343524222acc79dbe4dbae9c4472a651fce81d831de6d9626f596e525d31e")


def _paths(j) -> list[str]:
    """Every file a pass's manifest names: each of its objects' ``path``."""
    if isinstance(j, dict):
        return ([j["path"]] if isinstance(j.get("path"), str) else []) + [p for v in j.values() for p in _paths(v)]
    return [p for v in j for p in _paths(v)] if isinstance(j, list) else []


@pytest.mark.slow
@pytest.mark.skipif(not SERVED, reason="VERITY_COMPOSITION_SERVED names no served run (its Lean check hashes 2.1 GB of "
                                       "distinct pass files and replays 470 tiles: about an hour on a CPU node)")
def test_remote_v1_on_a_served_pearl_plus_run(tmp_path, monkeypatch):
    """Remote v1 on a served Pearl+ run with live coins and the challenger's draw: Pearl+ accepts in Lean, both ties
    hold, and the composite is undecided only because sampled proofs is (`_empty_audit`, as everywhere here), in Lean
    as in Python. The run is the dress rehearsal r20261105-082550-ed23's (development, never the record): its four
    retained passes, the epoch coins, the coin record and the window record W the passes are tied to, with the verdict
    ``served_passes.py verify`` recorded on them (``verify.json``), which is Pearl+'s Python answer here
    (`reference.pearl_plus_answer`); the Lean replays the passes. `VERITY_COMPOSITION_SERVED` names a directory holding
    them, as ``research data fetch art:e76bdb5195f7d82bfbb073e50fa76b10dbd0b25417e20f430073e4d86a135ac0 --path
    out/served`` lays them out under ``out/served`` (run r20261105-155828-663b's record): ``passes/pass-0000{0..3}/``,
    ``passes/coins.json``, ``coin.json``, ``verify.json`` and ``window-record.json``.

    The registration record R the policy and sampled proofs commit to is constructed here, the empty audit's record
    chained right after W (``previous`` = digest(W), W the one record of its log), because the dress ran no follow-on
    and so registered no R. The run of record's own R, with its own run, replaces it.

    It catches what no drawn case can: the composite's `run` and `window` ties and Pearl+'s Lean decoder, opener and
    check on a real run's commitment, coin record and files handed over as blobs, which the synthetic cases (Pearl+'s
    commitments that don't decode, the ties' unit tests) never reach."""
    d = Path(SERVED)
    w = json.loads((d / "window-record.json").read_bytes())
    coin, verified = json.loads((d / "coin.json").read_bytes()), json.loads((d / "verify.json").read_bytes())
    assert OS.R.digest(w) == verified["tie"]["record"] == SERVED_W
    assert (coin["schema"], coin["draw"]["from"], coin["commitment"], verified["commitment"]) == (
        "verity-pouw/pearl-plus-coin/v2", "challenger", SERVED_RUN, SERVED_RUN)
    assert verified["answer"]["verdict"] == R.ACCEPT, verified["answer"]

    params, rec, evidence = _empty_audit(0, canonical(w))
    audit = as_component(R.SAMPLED_PROOFS, form_case(OS._json({"params": {**params, "delta": "1/2"}, "record": rec,
                                                               "evidence": evidence})))
    policy = policy_on(audit)
    record = R.policy_record(policy["root"], policy["evidence"])[0]
    assert R.policy_previous(policy["root"], policy["evidence"]) == [bytes.fromhex(SERVED_W)]

    passes = sorted((d / "passes").glob("pass-*"))
    manifests = [(p / "manifest.json").read_bytes() for p in passes]
    assert [hashlib.sha256(m).hexdigest() for m in manifests] == [x.rsplit("sha256:", 1)[1] for x in coin["passes"]]
    files: dict[str, Path] = {}
    names = []
    for p, m in zip(passes, manifests):
        named = {}
        for f in _paths(json.loads(m)):
            with open(p / f, "rb") as fh:
                named[f] = hashlib.file_digest(fh, "sha512").hexdigest()
            files.setdefault(named[f], p / f)
        names.append(named)
    commitment = canonical({"manifests": [json.loads(m) for m in manifests], "record": record.hex(),
                            "window": {"dropped": coin["dropped"], "record": w, "rounds": coin["rounds"]}}).encode()
    pearl = {"commitment": commitment.hex(), "evidence": canonical({"files": names}).encode().hex(),
             "params": {"coins": json.loads((d / "passes" / "coins.json").read_bytes()), "draws": verified["draws"],
                        "record": {k: coin[k] for k in ("commitment", "draw", "schema", "source")}}}
    monkeypatch.setitem(R.PEARL_PLUS.cases, "served", (pearl, R.pearl_plus_answer(commitment, verified)))

    case = remote_case([policy, as_component(R.PEARL_PLUS, pearl), audit], R.REMOTE_V1)
    assert R.REMOTE_V1.verdicts(case) == [R.ACCEPT, R.ACCEPT, R.UNDECIDED]
    assert R.REMOTE_V1.tied(case) == [("record", True), ("window", True)]
    want = R.REMOTE_V1.answer(case)
    assert want["verdict"] == R.UNDECIDED, want

    blobs = [tmp_path / "circuit", tmp_path / "public"]
    for b, data in zip(blobs, BLOBS):
        b.write_bytes(data)
    for name, data in (("params", canonical(case["params"]).encode()), ("commitment", bytes.fromhex(case["commitment"])),
                       ("evidence", bytes.fromhex(case["evidence"]))):
        (tmp_path / name).write_bytes(data)
    p = subprocess.run([D.build(PROTOCOLS / "remote", "remote-v1"), "check", tmp_path / "params", tmp_path / "commitment",
                        tmp_path / "evidence", *blobs, *files.values()], capture_output=True, text=True)
    got = json.loads(p.stdout)
    assert p.returncode == 2 and {k: v for k, v in got.items() if k != "reason"} == {
        k: v for k, v in want.items() if k != "reason"}, (p.returncode, got, p.stderr)
    assert got["reason"].startswith(f"{R.SAMPLED_PROOFS.id} undecided") and R.PEARL_PLUS.id not in got["reason"], got


def test_spot_refuses_one_stage_v1(remote_server):
    """Sampled proofs' one-stage form is one-stage v2: v1's audit under it, one v1 itself decides (undecided, on an
    empty draw at its challenger's coins), is refused by name (`VStar.Recursive.Sampled.V1_REFUSED`) at its params,
    and v1's evidence under v2's params at the evidence. Remote, its policy and period aimed at acceptance on the
    audit's record, rejects with that refusal as sampled proofs' reason, in Lean as in Python."""
    want = {"v1": f"params: the parameters name no proofs: {R.V1_REFUSED}",
            "v1-evidence": f"evidence: the evidence names no proofs: {R.V1_REFUSED}"}
    seen = collections.Counter()

    @settings(max_examples=3, database=None, deadline=None, suppress_health_check=list(HealthCheck))
    @given(component_case(R.INFERENCE_ONLY, R.ACCEPT), component_case(R.POUW_PERIOD, R.ACCEPT), empty_draws())
    def check(policy, period, audit):
        assert OS.answer(OS._case(_v1(audit)))["verdict"] == R.UNDECIDED
        for br, why in want.items():
            sampled = broken_case(audit, br)
            got = R.SAMPLED_PROOFS.answer({**sampled, "blobs": BLOBS})
            assert (got["verdict"], got["reason"]) == (R.REJECT, why), got
            case = remote_case(one_record(R.REMOTE, [policy, period, as_component(R.SAMPLED_PROOFS, sampled)]))
            assert R.REMOTE.answer(case)["verdict"] == R.REJECT
            out = remote_server.exchange(case)
            assert out["verdict"] == R.REJECT and f"{R.SAMPLED_PROOFS.id} reject: {why}" in out["reason"], out
            seen[br] += 1
    check()
    assert set(seen) == set(want), dict(seen)


@pytest.mark.slow
def test_onsite_agrees(onsite_server, recorded):
<<<<<<< HEAD
    """Onsite reaches each verdict but acceptance, its policy and integrity meet on one record (left to the
    subprotocols) and on two (refused), and PoUS and PoUB-P2 on one store and on two (refused unless onsite rejects
    already), each often."""
    _agree(ONSITE, onsite_server, SEEN["onsite"], 1000, (R.REJECT, R.UNDECIDED, "reject, a component undecided",
                                                       "reject, no component rejected", "one record",
||||||| merged common ancestors
    """Onsite reaches each verdict but acceptance, its policy and integrity meet on one record (left to the
    subprotocols) and on two (refused), and PoUS and PoUB-P2 on one store and on two (refused unless onsite rejects
    already), each often; and PoUB-P2's case named as the generic bandwidth audit's, which is no component, refused."""
    _agree(ONSITE, onsite_server, SEEN["onsite"], 1000, (R.REJECT, R.UNDECIDED, "reject, a component undecided",
                                                       "reject, no component rejected", "one record",
=======
    """Onsite rejects with a component undecided beside the rejection, its policy and integrity meet on one record
    (left to the subprotocols) and on two (refused), and PoUS and PoUB-P2 on one store and on two, each often; and
    PoUB-P2's case named as the generic bandwidth audit's, which is no component, refused."""
    _agree(ONSITE, onsite_server, SEEN["onsite"], 1000, (R.REJECT, "reject, a component undecided", "one record",
>>>>>>> 705be2b532
                                                       "two records", "one store", "two stores"))


@pytest.mark.slow
def test_onsite_refuses_p2_named_as_the_generic_bandwidth_audit(onsite_server, recorded):
    """PoUB-P2's case named as the generic bandwidth audit's, under a commitment that binds it, names no component of
    onsite: refused, in Lean as in Python, on every drawn case (`generic-bandwidth` on each, a break that
    `test_onsite_agrees`'s seeded draw needn't reach)."""
    seen = collections.Counter()
    p = RejectingPort("onsite", reference(R.ONSITE, seen), composite(R.ONSITE, "generic-bandwidth"),
                      package=PROTOCOLS / "onsite", need_rejects=False)
    stats = D.run(p, examples=10, seed=0, server=onsite_server)
    assert stats.counts == {R.REJECT: 10} and seen["generic bandwidth"] == 10, (stats, dict(seen))


def test_product_names_only_the_composites():
    """`product IDS id` is each composite over its subprotocols, in any order (onsite, and remote v1 with Pearl+'s
    run), exactly the configuration's composites (`verity.core.deploy.config.COMPOSITES`, by the ids `SUBPROTOCOLS`
    gives their names), and refuses every other set of their subprotocols, H1's development composite's among them,
    and the wipe (exit 64)."""
    exe = D.build(PROTOCOLS / "onsite", "product")

    def named(xs) -> tuple[int, str]:
        p = subprocess.run([exe, ",".join(xs), "id"], capture_output=True, text=True)
        return p.returncode, p.stdout.strip()

    want = {p.id: sorted(C.id for C in p.components) for p in (R.ONSITE, R.REMOTE_V1)}
    assert {cid: sorted(subs.values()) for cid, subs in SPP.SUBPROTOCOLS.items()} == want
    assert {cid: sorted(names) for cid, names in K.COMPOSITES.items()} == {
        cid: sorted(subs) for cid, subs in SPP.SUBPROTOCOLS.items()}
    for cid, ids in want.items():
        assert named(ids) == (0, cid)
        assert named(ids[::-1]) == (0, cid)
    assert named(C.id for C in R.REMOTE.components)[0] == 64
    every = sorted({i for ids in want.values() for i in ids} | {"verity/wipe/v0"})
    others = [s for k in range(len(every) + 1) for s in itertools.combinations(every, k)
              if sorted(s) not in want.values()]
    assert len(others) == 2 ** len(every) - len(want)
    for s in others:
        assert named(s)[0] == 64, s
    assert named([R.POUW_PERIOD.id, R.POUW_PERIOD.id, R.SAMPLED_PROOFS.id])[0] == 64
    assert named([R.INFERENCE_ONLY.id, "verity/pouw/v1", R.SAMPLED_PROOFS.id])[0] == 64


@pytest.mark.slow
<<<<<<< HEAD
||||||| merged common ancestors
def test_product_over_remotes_subprotocols_is_remote(product_remote_server):
    _agree(PRODUCT[R.REMOTE.id], product_remote_server, SEEN["remote"], 1000,
           (R.REJECT, "reject, a component undecided", "reject, no component rejected", "one record", "two records"))


@pytest.mark.slow
=======
def test_product_over_remotes_subprotocols_is_remote(product_remote_server):
    _agree(PRODUCT[R.REMOTE.id], product_remote_server, SEEN["remote"], 1000, (R.REJECT, "one record", "two records"))


@pytest.mark.slow
>>>>>>> 705be2b532
def test_product_over_onsites_subprotocols_is_onsite(product_onsite_server, recorded):
    _agree(PRODUCT[R.ONSITE.id], product_onsite_server, SEEN["onsite"], 200, (R.REJECT, "reject, a component undecided"))


def test_onsites_bandwidth_is_p2():
    """Onsite's bandwidth component is PoUB-P2, and the generic audit's id names none of its components."""
    ids = [C.id for C in R.ONSITE.components]
    assert R.BANDWIDTH.id == "verity/remote-bandwidth/p2/v0" and R.BANDWIDTH.id in ids and GENERIC_BANDWIDTH not in ids
    assert [s for s, _ in R.STORE.names] == [R.POUS.id, R.BANDWIDTH.id]


@pytest.mark.slow
def test_onsite_refuses_two_stores_and_else_keeps_its_verdict(recorded):
    """On drawn cases onsite's verdict is its verdict without the store tie (onsite before it), except where PoUS and
    PoUB-P2 committed to two stores (PoUS's run on another store, `periods.OTHER_STORE`), which it refuses: the tie
    changes no other verdict. Python only: the difftest holds the Lean to it. Drawn from a fixed seed, as
    `verity_difftest.run` draws."""
    untied = dataclasses.replace(R.ONSITE, ties=())
    seen = collections.Counter()

    @seed(0)
    @settings(max_examples=200, database=None, deadline=None, derandomize=True, suppress_health_check=list(HealthCheck))
    @given(composite(R.ONSITE))
    def check(case):
        got, was = R.ONSITE.answer(case), untied.answer(case)
        two = (R.STORE.what, False) in R.ONSITE.tied(case)
        if two:
            assert got["verdict"] == R.REJECT, got
        else:
            assert {k: v for k, v in got.items() if k != "reason"} == {k: v for k, v in was.items() if k != "reason"}
        seen[f"was {was['verdict']}, two" if two else "same"] += 1
    check()
    assert seen["same"] > 20 and seen["was reject, two"] > 0, dict(seen)


class Trial(R.TiedSame):
    """The trial's rule over the components' verdicts (`trial_verdict`): a component left undecided leaves the product
    undecided even when another rejects, once its ties hold. Everything else is the port's."""

    def check(self, ps, c, ev, blobs=()) -> tuple:
        answers = self.run(ps, ev, blobs)
        if not isinstance(answers, str) and not self.strays(answers) and not any(t.strays(ev) for t in self.ties) \
                and self.binds(answers, c) and R.trial_verdict([a["verdict"] for _, a in answers]) == R.UNDECIDED:
            return R.UNDECIDED, "a component is undecided"
        return super().check(ps, c, ev, blobs)


@pytest.mark.slow
def test_a_reference_without_reject_dominance_is_caught(onsite_server, recorded):
    """The difftest tells the composition's rule from the trial's: the case it shrinks to has a component that rejects and one that
    is undecided, under a commitment that binds their digests."""
    trial = Trial(R.ONSITE.id, R.ONSITE.components, R.ONSITE.same, R.ONSITE.ties)
    with pytest.raises(D.Mismatch) as e:
        D.run(D.ProtocolPort("onsite", trial.answer, ONSITE.inputs, package=PROTOCOLS / "onsite"), examples=1000,
              server=onsite_server)
    assert (e.value.lean[0], e.value.python[0]) == (R.REJECT, R.UNDECIDED), (e.value.lean, e.value.python)
    parts = R.ONSITE.verdicts(e.value.input)
    assert R.REJECT in parts and R.UNDECIDED in parts, parts
