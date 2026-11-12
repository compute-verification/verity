"""Onsite's and remote's Lean (`onsite serve`, `remote serve`: each product's executable, the core CLI) against the
Python reference (`reference.py`: the trial's composition rule, in which any rejection rejects the whole, over the harness's framing of the
core interface and the subprotocols' own Python). Hypothesis samples each subprotocol's case to accept, reject, be
undecided or fail to decode (from its own difftest's generator: PoUW's periods, made once from
`pouw_period_cases`, the network certifier's period `build`, the inference-only policy's `drawn`, which is
never undecided, sampled proofs' audits under the form ``one-stage``, `form_case`: one-stage v2's over `sessionless_draws`
(an audit whose sample at its challenger's coins is one proof block, which names no session) with one of `V2_BREAKS` or the form's
breaks, and one-stage v1's `changed_audits`, which the form refuses by name, the wipe's signed wipes,
`wipe_protocol_case.case`; space's and
bandwidth's from the small periods of their hooks, `periods.py`, all on one committed `C*` but PoUS's run on another
store, `periods.OTHER_STORE`), often recommits the subprotocols each composite ties to one record (`one_record`: the
policy to sampled proofs' registration record, on that audit's own registered circuit, `policy_on`, and sometimes every
window of the period too, `windows_on`), hands every subprotocol the same blobs (`BLOBS`), then often
breaks the composition's own inputs (a subprotocol dropped, unknown, repeated or out of order, a case under other params,
a wrong or short commitment, non-canonical or malformed evidence, blobs missing or malformed) or the case itself (a
field missing or not hex, not an object), and every field of the outcome but its reason must agree exactly. Sampled
proofs' cases all reject: v2 refuses evidence naming no session (`oneSession`), and a session on the challenger's coin
trees is verified by `flock-verify`, which the mirror doesn't run. So every product rejects here, onsite often with a
subprotocol undecided beside the rejection. Remote's `windows` tie decides only where no subprotocol rejects, so this
difftest doesn't reach it; `CoreRemote`'s `#guard`s check which windows it finds stray. Remote's case on one record
that reuses the window's registration of its tables (`reusing`) is one made case, which sampled proofs refuses at
`oneSession` too, before R1-R7 read the record; `CoreRemote`'s `#guard`s check what v2 decides on that record with a
session's proofs granted (R7's domains and outputs, R6's previous) and that remote credits the one window on it and
finds a window on the window's own registration stray. `product IDS serve` over each composite's subprotocols, the
composite `verity run`'s verifier runs, is held to the same reference, and `product IDS id` names remote, onsite and
onsite's two smaller lists (`R.NETWORK`, `R.COMPUTE`) for their subprotocols and refuses every other set: the same
composites a configuration may list (`verity.configuration.deploy.config.COMPOSITES`)."""

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
from verity_difftest import settings as opt

from verity.configuration.deploy import config as K
from verity.hardware import site as S
from verity.protocols.spot.verifier import parties as SPP
from verity.circuits.types import Array, Value

ROOT = Path(__file__).resolve().parents[6]
ONSITE_PACKAGE, REMOTE_PACKAGE = ROOT / "verity/protocols/onsite/verifier", ROOT / "verity/protocols/remote/verifier"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
#: the mark of a difftest that takes over five minutes on a node: it stays out of every check but the nightly cold one, which runs
#: each by name with VERITY_SLOW=1 (check.py's `OPT_IN`), and runs wherever else VERITY_SLOW asks
OPT_IN = pytest.mark.skipif(opt.get("VERITY_SLOW") != "1", reason="over five minutes on a node: VERITY_SLOW=1")

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
#: The files every product case hands its subprotocols as blobs.
CIRCUIT, PUBLIC = b"not a circuit file", b"not a public file"
BLOBS = [CIRCUIT, PUBLIC]


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


@functools.cache
def _sessionless_audit(i: int) -> tuple[dict, dict, dict]:
    """One-stage v2's audit of `OS.SETUPS[i]`, a program that reads no registered value, its params but δ, its record
    and its evidence: its sample at its challenger's coins (`_challenger_draw`, a session whose challenger sampled no coin
    tree) is one proof block (a subset of one, the least a subset law draws), its proved unit, and it names no session,
    and its params name the circuit and public files `BLOBS` hold. Made once each: a challenger's signed session is slow
    to make."""
    s, law = OS.SETUPS[i], {"k": 1, "law": "subset"}
    part = {"program": s.sha512, "query": s.query}
    rec = OS.R.record(s.program, s.query, law, OS.SCHEME, [{"port": p, "root": OS.ROOTS[p], **s.ports[p]} for p in s.ports],
                      OS.WINDOW, leaf_layer=OS.LEAF, partition=part)
    ch, d, coins = OS._challenger_draw(random.Random(f"composition/sessionless/{i}"), s, law, rec, trees=0)
    proved = OS.DR.proved(d)
    evidence = {"coins": coins, "draw": d, "instances": len(proved), "proofs": {"sessions": []}, "proved_units": proved,
                "receipt": OS.R.receipt(rec, "v", received=OS.RECEIVED), "served": None, "session_roots": None,
                "sessions": [], "statement": {}}
    params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "law": law,
              "leaf_layer": OS.LEAF, "log": [], "partition": part, "ports": s.ports, "program": s.desc,
              "proofs": {"circuit": _sha(CIRCUIT), "public": _sha(PUBLIC), "registered": None}, "query": s.query,
              "registered": None, "scheme": OS.SCHEME, "window_root": ch.ledger.window["root"], "work_table": None}
    return params, rec, evidence


@st.composite
def sessionless_draws(draw) -> dict:
    """A one-stage v2 audit whose sample is one proof block (`_sessionless_audit`) at a sampled δ: it names no session, which
    v2 refuses at its challenger's coins before it reads a tree (`oneSession`: the sampled proof blocks are proved in one
    session)."""
    params, rec, evidence = _sessionless_audit(draw(st.integers(0, len(OS.SETUPS) - 1)))
    params = {**params, "delta": draw(st.sampled_from(["1/1000", "1/10", "1/2"]))}
    return OS._json({"params": params, "record": rec, "evidence": evidence})


def _without(x: dict, *ks: str) -> dict:
    return {k: v for k, v in x.items() if k not in ks}


def _v1(a: dict) -> dict:
    """v2's audit `a` as one-stage v1's: its params without ``proofs``, its evidence with the verdicts of no session in
    place of ``proofs``, on which v1 is undecided, for want of the verifier."""
    return {**a, "params": _without(a["params"], "proofs"),
            "evidence": {**_without(a["evidence"], "proofs"), "verdicts": {}}}


#: One change to v2's audit (`sessionless_draws`) that sampled proofs refuses at another step than its refusal of no
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


#: One break of SPOT's form that `VStar.Recursive.Sampled` refuses: params or evidence naming no form or another form,
#: the recursive form's params (refused by name), or its evidence with none of its files.
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


#: Sampled proofs' cases, each refused: v2's audit naming no session (`sessionless_draws`), the same with one change or
#: one break of the form, or v1's audit with up to two changes (refused by name).
SAMPLED_CASES = st.one_of(sessionless_draws().map(form_case),
                          st.builds(broken_case, sessionless_draws(), st.sampled_from(list(V2_BREAKS))),
                          st.builds(form_case, sessionless_draws(), st.sampled_from(FORM_BREAKS)),
                          OS.changed_audits(most=2).map(form_case))

NP = R.PERIOD
#: The period check's changes by the verdict they aim at (`case` breaks the line `serve` reads, as PoUW's `case-*`).
PERIOD_CASES = {v: tuple(m for m, x in NP.EXPECT.items() if x == v and m != "case")
                for v in (R.ACCEPT, R.REJECT, R.UNDECIDED)}

#: The inference-only policy's changes by the verdict they aim at: none, or one that rejects. It is never undecided.
INFERENCE_CASES = {v: tuple(m for m, (x, _) in R.INFERENCE.EXPECT.items() if x == v) for v in (R.ACCEPT, R.REJECT)}

#: A recorded subprotocol's kinds of case (`periods.build`), accepted ones first and thrice.
KINDS = [R.ACCEPT] * 3 + [R.REJECT, R.UNDECIDED, "other-setup", periods.OTHER_STORE, "broken-params", "broken-evidence"]


def kinds(C) -> list[str]:
    """The kinds of case `component_case` may aim `C` at when it is free to."""
    if isinstance(C, R.Recorded):
        return [k for k in KINDS if k in C.cases]
    if C is R.INFERENCE_ONLY:
        return list(INFERENCE_CASES)
    if C is R.SAMPLED_PROOFS:
        return [R.REJECT]
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
        case = draw(SAMPLED_CASES)
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
    """The policy's params but the log, and its evidence, for the registered circuit `program` (descriptor text) of a
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
    """The policy's case, as `component_case` gives it, on the record and log of sampled proofs' case `anchor`, declaring the
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
    """The cases with every subprotocol of ``product.same`` committing to sampled proofs' registration record, each
    with its digest as its Python reads it on `BLOBS`, and the policy's case on the audit's own registered circuit
    where it can be (`policy_on`): cases that pass `Core.Composition.strays` and leave the verdict to the subprotocols.
    With ``windows``, every window of the product's PoUW period names that record too (`windows_on`), so that its
    `windows` tie holds; the period then refuses, since a window's Commit record registers its own tables. Without it,
    an honest period's windows name their own records, and the tie refuses."""
    first = next(c for c in cases if c["id"] == R.SAMPLED_PROOFS.id)
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


#: The generic remote bandwidth audit's id: not a subprotocol of onsite, whose bandwidth is PoUB-P2 (`generic-bandwidth`
#: names PoUB-P2's case by it, its params and its entry, under a commitment that binds it).
GENERIC_BANDWIDTH = "verity/remote-bandwidth/v0"
#: The binding breaks come five times: with `commitment-size` they are the breaks that leave every subprotocol's case standing,
#: and so the way to a rejection that no subprotocol's own explains. `generic-bandwidth` comes five times too, so that a
#: fixed seed's thousand cases reach it whatever the subprotocols' generators draw before it.
MUTATIONS = [None] * 20 + ["drop-param", "drop-case", "unknown-param", "unknown-case", "swap-params", "swap-cases",
                           "repeat-param", "repeat-case", "case-params", "commitment-size", "evidence-noncanonical",
                           "params-extra-key", "case-extra-key", "case-missing-key", "bad-hex", "garbage-evidence",
                           "garbage-params", "not-a-case", "no-blobs", "blobs-not-hex", "blobs-not-list",
                           "blob-extra"] + ["binding", "binding-id", "generic-bandwidth"] * 5
#: The breaks of the commitment alone, which leave every subprotocol's case standing.
BINDING = ["binding", "binding-id", "commitment-size"]


@st.composite
def composite(draw, product: R.Product) -> dict:
    """One case of `product`: its subprotocols' cases (one time in six every one aimed at acceptance, one in six the same
    but for one of those that can be undecided, when one can, aimed at undecided, so that a product of seven reaches its
    binding breaks with a subprotocol undecided, and each at any verdict otherwise), the subprotocols it ties recommitted
    to one record (`one_record`) two times in three when they are aimed at acceptance and one in three otherwise, then
    at most one break of the composition's own inputs (half the time a break of the commitment alone, `BINDING`, when
    every case is aimed at acceptance)."""
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
    m = draw(st.sampled_from(BINDING if mode <= 1 and draw(st.booleans()) else MUTATIONS))
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
    undecided subprotocol among them, and those no subprotocol's rejection explains."""
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


class RejectingPort(D.ProtocolPort):
    """A product port whose cases reject: sampled proofs' do (`SAMPLED_CASES`), so the run's non-vacuity bar is a
    rejection, and its other bars are the ways its subprotocols' verdicts and records meet, counted in `seen`
    (`_agree`)."""
    accepted = R.REJECT


def port(name: str, product: R.Product, seen: collections.Counter, package: Path,
         serve: tuple[str, ...] = ("serve",)) -> D.ProtocolPort:
    """`name serve` (or `name SERVE`) of the Lake package in `package`, on the product's cases and, one sample in ten,
    any JSON line."""
    return RejectingPort(name, reference(product, seen), st.one_of(*[composite(product)] * 9, JSON),
                         package=package, serve=serve, need_rejects=False)


def ids(product: R.Product) -> str:
    return ",".join(C.id for C in product.components)


SEEN = {n: collections.Counter() for n in ("onsite", "remote", "network", "compute")}
ONSITE = port("onsite", R.ONSITE, SEEN["onsite"], ONSITE_PACKAGE)
REMOTE = port("remote", R.REMOTE, SEEN["remote"], REMOTE_PACKAGE)
#: `product IDS serve` over each composite's subprotocols, which `verity run`'s verifier runs
PRODUCT = {p.id: port("product", p, SEEN[n], ONSITE_PACKAGE, serve=(ids(p), "serve"))
           for n, p in (("onsite", R.ONSITE), ("remote", R.REMOTE), ("network", R.NETWORK), ("compute", R.COMPUTE))}
onsite_server = D.server_fixture(ONSITE)
remote_server = D.server_fixture(REMOTE)
product_onsite_server = D.server_fixture(PRODUCT[R.ONSITE.id])
product_remote_server = D.server_fixture(PRODUCT[R.REMOTE.id])
product_network_server = D.server_fixture(PRODUCT[R.NETWORK.id])
product_compute_server = D.server_fixture(PRODUCT[R.COMPUTE.id])


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """Space's and bandwidth's cases, made once (their memory challenger's session logs under the suite's basetemp), and
    the wipe's test signer's files under it too while the module's tests sample wipes."""
    if not R.POUS.cases:
        periods.build(tmp_path_factory.mktemp("periods"))
    ts = R.WIPE_CASES.M.ts
    ts.DIR = tmp_path_factory.mktemp("test-signer")
    yield
    ts.DIR = None


def test_the_rule_changes_the_trial_only_where_a_rejection_meets_an_undecided_component():
    """Over every list of up to four verdicts, the composition's rule and the trial's agree but where a subprotocol rejects while
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
@OPT_IN
def test_remote_agrees(remote_server):
    """Remote's subprotocols meet on one record (left to the subprotocols) and on two (refused), each often."""
    _agree(REMOTE, remote_server, SEEN["remote"], 1000, (R.REJECT, "one record", "two records"))


def test_remote_refuses_two_records_and_else_keeps_the_products_verdict():
    """On sampled cases remote's verdict is its product's without the record checks (`Core.Composition.product`,
    remote before them), except where its policy and integrity committed to two records, or a window of its PoUW period
    names another record than the policy's (`windows`), which it refuses: the checks change no other verdict. Python
    only: the difftest holds the Lean to it. Sampled from a fixed seed, as `verity_difftest.run` samples, so the counts it
    asks of the sample are the generator's, not a run's luck."""
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


def product_case(product: R.Product, cases: list[dict]) -> dict:
    """The product's case on its subprotocols' cases as they are, with no break of its own."""
    return {"commitment": R.bind(product.id, [(c["id"], c["digest"]) for c in cases]).hex(),
            "evidence": canonical({"components": [{"commitment": c["root"].hex(), "evidence": c["evidence"].hex(),
                                                   "id": c["id"], "params": c["params"]} for c in cases]}).encode().hex(),
            "params": {"components": [{"id": c["id"], "params": c["params"]} for c in cases]},
            "blobs": [b.hex() for b in BLOBS]}


@pytest.mark.parametrize("name", ["remote", pytest.param("onsite", marks=pytest.mark.slow)])
def test_spot_refuses_one_stage_v1(name, request):
    """Sampled proofs' one-stage form is one-stage v2: v1's audit under it, one v1 itself decides (undecided, for want of
    the verifier, at its challenger's coins), is refused by name (`VStar.Recursive.Sampled.V1_REFUSED`) at its params,
    and v1's evidence under v2's params at the evidence. Remote and onsite, every other subprotocol aimed at acceptance
    on the audit's record, reject with that refusal as sampled proofs' reason, in Lean as in Python. Onsite's case is
    slow for its space and bandwidth cases (`recorded`)."""
    product = {"remote": R.REMOTE, "onsite": R.ONSITE}[name]
    if name == "onsite":
        request.getfixturevalue("recorded")
    server = request.getfixturevalue(f"{name}_server")
    want = {"v1": f"params: the parameters name no proofs: {R.V1_REFUSED}",
            "v1-evidence": f"evidence: the evidence names no proofs: {R.V1_REFUSED}"}
    others = [C for C in product.components if C is not R.SAMPLED_PROOFS]
    seen = collections.Counter()

    @settings(max_examples=3, database=None, deadline=None, suppress_health_check=list(HealthCheck))
    @given(st.tuples(*[component_case(C, R.ACCEPT) for C in others]), sessionless_draws())
    def check(drawn, audit):
        assert OS.answer(OS._case(_v1(audit)))["verdict"] == R.UNDECIDED
        for br, why in want.items():
            sampled = broken_case(audit, br)
            got = R.SAMPLED_PROOFS.answer({**sampled, "blobs": BLOBS})
            assert (got["verdict"], got["reason"]) == (R.REJECT, why), got
            by_id = {c["id"]: c for c in (*drawn, as_component(R.SAMPLED_PROOFS, sampled))}
            case = product_case(product, one_record(product, [by_id[C.id] for C in product.components]))
            assert product.answer(case)["verdict"] == R.REJECT
            out = server.exchange(case)
            assert out["verdict"] == R.REJECT and f"{R.SAMPLED_PROOFS.id} reject: {why}" in out["reason"], out
            seen[br] += 1
    check()
    assert set(seen) == set(want), dict(seen)


def _recursive(a: dict) -> dict:
    """v2's audit `a` as the recursive audit's case: its params v1's under the form ``recursive``, naming the recursive
    audit's own inputs (`VStar.Recursive.ownOf`'s keys: the circuit and public files `BLOBS` hold, a draw, no
    registration and no part), and its evidence under that form."""
    own = {"circuit": _sha(CIRCUIT), "draw": _sha(b"draw"), "parts": [], "public": _sha(PUBLIC), "registration": None}
    return OS._case({"params": {**_without(a["params"], "proofs"), "form": "recursive", "recursive": own},
                     "record": a["record"], "evidence": {**a["evidence"], "form": "recursive"}})


@pytest.mark.parametrize("name", ["remote", pytest.param("onsite", marks=pytest.mark.slow)])
def test_spot_refuses_the_recursive_form(name, request):
    """SPOT's soundness covers its one-stage form only, so it refuses the recursive form by name at its params
    (`VStar.Recursive.Sampled.RECURSIVE_REFUSED`), before it reads their rest: v2's audit under the recursive form,
    naming the recursive audit's own inputs (`_recursive`) or none of them (the form's break ``params-recursive``).
    Remote and onsite, every other subprotocol aimed at acceptance on the audit's record, reject with that refusal as
    SPOT's reason, in Lean as in Python. Onsite's case is slow for its space and bandwidth cases (`recorded`)."""
    product = {"remote": R.REMOTE, "onsite": R.ONSITE}[name]
    if name == "onsite":
        request.getfixturevalue("recorded")
    server = request.getfixturevalue(f"{name}_server")
    why = f"params: {R.RECURSIVE_REFUSED}"
    others = [C for C in product.components if C is not R.SAMPLED_PROOFS]
    seen = collections.Counter()

    @settings(max_examples=3, database=None, deadline=None, suppress_health_check=list(HealthCheck))
    @given(st.tuples(*[component_case(C, R.ACCEPT) for C in others]), sessionless_draws())
    def check(drawn, audit):
        for br, sampled in {"named": _recursive(audit), "unnamed": form_case(audit, "params-recursive")}.items():
            got = R.SAMPLED_PROOFS.answer({**sampled, "blobs": BLOBS})
            assert (got["verdict"], got["reason"]) == (R.REJECT, why), got
            by_id = {c["id"]: c for c in (*drawn, as_component(R.SAMPLED_PROOFS, sampled))}
            case = product_case(product, one_record(product, [by_id[C.id] for C in product.components]))
            assert product.answer(case)["verdict"] == R.REJECT
            out = server.exchange(case)
            assert out["verdict"] == R.REJECT and f"{R.SAMPLED_PROOFS.id} reject: {why}" in out["reason"], out
            seen[br] += 1
    check()
    assert set(seen) == {"named", "unnamed"}, dict(seen)


#: `reusing`'s R's law: a subset of one, so v2's audit of R samples R's one proof block at its challenger's coins
REUSING_LAW = {"k": 1, "law": "subset"}


def reusing(made: list[dict]):
    """A window's Commit record as `pouw_credit_cases.period_instance`'s `commit` makes it: R, the run's registration,
    chained after W (``previous``), the serving process's registration of the window's tables in a served window's form,
    and registering those tables under the Commit's names (`commit_names`) as output values that reuse W's entries
    (``outputs``), of a program whose parameters they are (registered on `ONE_STAGE.PROGRAMS`). Each call appends W, R,
    R's setup and its outputs to `made`."""
    CC = R.POUW_PERIOD_CASES.credit_cases

    def commit(true: dict[str, dict], _steps: int, circuit: str, open_root: str) -> str:
        w = OS._json({"format": OS.R.FORMAT_V2, "previous": None, "program": circuit, "protocol": OS.R.PROTOCOL,
                      "registered": [true[v] for v in sorted(true)],
                      "window": {"entry": len(made), "kind": "served", "open_root": open_root, "refused": []}})
        named = sorted((n, v) for v, n in CC.commit_names(true).items())
        params = tuple((n, Array(true[v]["leaves"] * true[v]["words"], Value(16))) for n, v in named)
        s = OS.RegisteredSetup(OS._registered_program("CoreRemoteReuse", 1, params, tuple(n for n, _ in named)),
                               [{n: true[v]["words"] for n, v in named}])
        OS.PROGRAMS[s.sha512] = s.program
        rec = OS._json(OS.R.record(s.program, s.query, REUSING_LAW, OS.SCHEME,
                                   [{"port": p, "root": OS.ROOTS[p], **s.ports[p]} for p in s.ports], OS.WINDOW,
                                   registered=[{**true[v], "value": n} for n, v in named], leaf_layer=OS.LEAF,
                                   anchors=[], previous=OS.R.digest(w),
                                   partition={"program": s.sha512, "query": s.query}))
        made.append({"w": w, "record": rec, "setup": s,
                     "outputs": {n: {"record": OS.R.digest(w), "value": v} for n, v in named}})
        return CC.Cr._canonical(rec).encode().hex()
    return commit


def reusing_audit(m: dict) -> dict:
    """Sampled proofs' one-stage v2 case on `reusing`'s R (`m`) over the log [W], with R's outputs: its sample at its
    challenger's coins (`_challenger_draw`, a session whose challenger sampled no coin tree) is R's one proof block, its
    proved unit, and it names no session, its params name the circuit and public files `BLOBS` hold, and its registered
    file names R's tables, which no proof block of R's program reads."""
    s, rec = m["setup"], m["record"]
    ch, d, coins = OS._challenger_draw(random.Random("remote/reusing"), s, REUSING_LAW, rec, trees=0)
    proved = OS.DR.proved(d)
    reads = {"format": "verity/registered-reads/v1", "ports": {
        e["value"]: {k: e[k] for k in ("domain", "leaves", "root", "value", "words")} | {"reads": [None]}
        for e in rec["registered"]}}
    params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "delta": "1/10",
              "law": REUSING_LAW, "leaf_layer": OS.LEAF, "log": [m["w"]], "outputs": m["outputs"],
              "parameters": s.signature, "partition": {"program": s.sha512, "query": s.query}, "ports": s.ports,
              "program": s.desc, "proofs": {"circuit": _sha(CIRCUIT), "public": _sha(PUBLIC), "registered": reads},
              "query": s.query, "registered": s.values[0], "scheme": OS.SCHEME,
              "window_root": ch.ledger.window["root"], "work_table": None}
    evidence = {"coins": coins, "draw": d, "instances": len(proved), "proofs": {"sessions": []}, "proved_units": proved,
                "receipt": OS.R.receipt(rec, "v", received=OS.RECEIVED), "served": None, "session_roots": None,
                "sessions": [], "statement": {}}
    return form_case(OS._json({"params": params, "record": rec, "evidence": evidence}))


#: v2's refusal of evidence naming no session (`OneStageZk.oneSession`)
NO_SESSION = "proofs: the evidence names 0 sessions: the drawn units are proved in one session"


def test_remote_on_one_record_that_reuses_the_windows_registration(remote_server):
    """A period of one window whose Commit record is R, the run's registration chained after W, the window's
    registration of its tables, reusing W's entries as R's outputs (`reusing`): the policy on the log [W] and R and the
    period accept, and the policy's record and the window's are R. Sampled proofs' one-stage v2 audit of R names no
    session, so v2 refuses it at `oneSession`, before R1-R7 read R, and remote rejects with that reason, in Lean as in
    Python. With a subprotocol rejecting, remote's `windows` tie decides nothing here: `CoreRemote`'s `#guard`s check v2
    on R with a session's proofs granted and the tie on a window on R or on W."""
    PPC, made = R.POUW_PERIOD_CASES, []
    period = as_component(R.POUW_PERIOD, PPC.serve(PPC.credit_cases.period_instance(
        random.Random("remote/reusing"), windows=1, commit=reusing(made), **PPC.COVERED)))
    reused = reusing_audit(made[0])
    audit = as_component(R.SAMPLED_PROOFS, reused)
    policy = policy_on(audit)
    record = OS.R.digest(made[0]["record"])
    assert [x.hex() for x in R.policy_record(policy["root"], policy["evidence"])] == [record]
    assert [x.hex() for x in R.window_records(period["root"], period["evidence"])] == [record]
    got = R.SAMPLED_PROOFS.answer({**reused, "blobs": BLOBS})
    assert (got["verdict"], got["reason"]) == (R.REJECT, NO_SESSION), got
    case = product_case(R.REMOTE, [policy, period, audit])
    assert R.REMOTE.verdicts(case) == [R.ACCEPT, R.ACCEPT, R.REJECT], R.REMOTE.verdicts(case)
    python, lean = R.REMOTE.answer(case), remote_server.exchange(case)
    assert {k: v for k, v in lean.items() if k != "reason"} == {k: v for k, v in python.items() if k != "reason"}
    assert python["verdict"] == R.REJECT and lean["reason"] == f"{R.SAMPLED_PROOFS.id} reject: {NO_SESSION}", lean


@pytest.mark.slow
@OPT_IN
def test_onsite_agrees(onsite_server, recorded):
    """Onsite rejects with a subprotocol undecided beside the rejection, its policy and integrity meet on one record
    (left to the subprotocols) and on two (refused), its period's windows on the policy's record and on others, and
    PoUS and PoUB-P2 on one store and on two, each often; and PoUB-P2's case named as the generic bandwidth audit's,
    which is no subprotocol, refused."""
    _agree(ONSITE, onsite_server, SEEN["onsite"], 1000, (R.REJECT, "reject, a component undecided", "one record",
                                                       "two records", "one store", "two stores"))
    assert SEEN["onsite"]["generic bandwidth"] > 0, dict(SEEN["onsite"])


def test_product_names_only_the_composites():
    """`product IDS id` is remote over remote's subprotocols, onsite over onsite's and each smaller list's composite
    over that list, in any order, exactly the configuration's composites
    (`verity.configuration.deploy.config.COMPOSITES`, by the ids `SUBPROTOCOLS` gives their names) and the composites a
    site's log is registered for (`verity.hardware.site.COMPOSITE_IDS`), and refuses every other set of their
    subprotocols (exit 64)."""
    exe = D.build(ONSITE_PACKAGE, "product")

    def named(xs) -> tuple[int, str]:
        p = subprocess.run([exe, ",".join(xs), "id"], capture_output=True, text=True)
        return p.returncode, p.stdout.strip()

    want = {p.id: sorted(C.id for C in p.components) for p in (R.ONSITE, R.REMOTE, R.NETWORK, R.COMPUTE)}
    assert all(p.id == "verity/composite/v0:" + ",".join(want[p.id]) for p in (R.NETWORK, R.COMPUTE))
    assert {cid: sorted(subs.values()) for cid, subs in SPP.SUBPROTOCOLS.items()} == want
    assert {cid: sorted(names) for cid, names in K.COMPOSITES.items()} == {
        cid: sorted(subs) for cid, subs in SPP.SUBPROTOCOLS.items()}
    assert sorted(map(list, S.COMPOSITE_IDS.values())) == sorted(want.values())
    for cid, ids in want.items():
        assert named(ids) == (0, cid)
        assert named(ids[::-1]) == (0, cid)
    every = sorted({i for ids in want.values() for i in ids} | {"verity/wipe/v0"})
    others = [s for k in range(len(every) + 1) for s in itertools.combinations(every, k)
              if sorted(s) not in want.values()]
    assert len(others) == 2 ** len(every) - 4
    for s in others:
        assert named(s)[0] == 64, s
    assert named([R.POUW_PERIOD.id, R.POUW_PERIOD.id, R.SAMPLED_PROOFS.id])[0] == 64
    assert named([R.INFERENCE_ONLY.id, "verity/pouw/v1", R.SAMPLED_PROOFS.id])[0] == 64


@pytest.mark.slow
@OPT_IN
def test_product_over_remotes_subprotocols_is_remote(product_remote_server):
    _agree(PRODUCT[R.REMOTE.id], product_remote_server, SEEN["remote"], 1000, (R.REJECT, "one record", "two records"))


@pytest.mark.slow
def test_product_over_onsites_subprotocols_is_onsite(product_onsite_server, recorded):
    _agree(PRODUCT[R.ONSITE.id], product_onsite_server, SEEN["onsite"], 200, (R.REJECT, "reject, a component undecided"))


@pytest.mark.slow
def test_product_over_the_network_list_is_its_composite(product_network_server, recorded):
    """The Lean's composite of network certification, sampled proofs and the wipe is the reference's (`R.NETWORK`): the
    three subprotocols alone, no tie. `product IDS id` shows only that the list is taken; this shows which
    subprotocols, tied how, the composite it names runs."""
    _agree(PRODUCT[R.NETWORK.id], product_network_server, SEEN["network"], 100)


@pytest.mark.slow
def test_product_over_the_compute_list_is_its_composite(product_compute_server, recorded):
    """The Lean's composite of the network list's three, the inference-only policy and PoUW's period is the reference's
    (`R.COMPUTE`): the policy and sampled proofs on one registration record and every window of the period on it, and
    no store's tie. `product IDS id` shows only that the list is taken."""
    _agree(PRODUCT[R.COMPUTE.id], product_compute_server, SEEN["compute"], 100, (R.REJECT, "one record", "two records"))


def test_onsites_bandwidth_is_p2():
    """Onsite's bandwidth subprotocol is PoUB-P2, and the generic audit's id names none of its subprotocols."""
    ids = [C.id for C in R.ONSITE.components]
    assert R.BANDWIDTH.id == "verity/remote-bandwidth/p2/v0" and R.BANDWIDTH.id in ids and GENERIC_BANDWIDTH not in ids
    assert [s for s, _ in R.STORE.names] == [R.POUS.id, R.BANDWIDTH.id]


def test_onsites_ties_are_remotes_windows_and_the_store():
    """Onsite holds remote's `windows` tie, the one value of it, before its store tie (`Core.Onsite.ties`)."""
    assert R.ONSITE.ties == (R.WINDOWS, R.STORE) and R.REMOTE.ties == (R.WINDOWS,)
    assert R.ONSITE.ties[0] is R.REMOTE.ties[0]


@pytest.mark.slow
def test_onsite_refuses_a_stray_tie_and_else_keeps_its_verdict(recorded):
    """On sampled cases onsite's verdict is its verdict without its ties (onsite before them), except where a window of
    its PoUW period names another record than the policy's, or PoUS and PoUB-P2 committed to two stores (PoUS's run on
    another store, `periods.OTHER_STORE`), which it refuses: the ties change no other verdict. Python only: the
    difftest holds the Lean to it. Sampled from a fixed seed, as `verity_difftest.run` samples."""
    untied = dataclasses.replace(R.ONSITE, ties=())
    seen = collections.Counter()

    @seed(0)
    @settings(max_examples=200, database=None, deadline=None, derandomize=True, suppress_health_check=list(HealthCheck))
    @given(composite(R.ONSITE))
    def check(case):
        got, was = R.ONSITE.answer(case), untied.answer(case)
        strays = [what for what, one in R.ONSITE.tied(case) if not one]
        if strays:
            assert got["verdict"] == R.REJECT, got
        else:
            assert {k: v for k, v in got.items() if k != "reason"} == {k: v for k, v in was.items() if k != "reason"}
        seen[f"was {was['verdict']}, two" if strays else "same"] += 1
        seen.update(f"two {what}s" for what in strays)
    check()
    assert seen["same"] > 20 and seen["was reject, two"] > 0, dict(seen)
    assert seen["two records"] > 0 and seen["two stores"] > 0, dict(seen)


class Trial(R.TiedSame):
    """The trial's rule over the subprotocols' verdicts (`trial_verdict`): a subprotocol left undecided leaves the product
    undecided even when another rejects, once its ties hold. Everything else is the port's."""

    def check(self, ps, c, ev, blobs=()) -> tuple:
        answers = self.run(ps, ev, blobs)
        if not isinstance(answers, str) and not self.strays(answers) and not any(t.strays(ev) for t in self.ties) \
                and self.binds(answers, c) and R.trial_verdict([a["verdict"] for _, a in answers]) == R.UNDECIDED:
            return R.UNDECIDED, "a component is undecided"
        return super().check(ps, c, ev, blobs)


@pytest.mark.slow
@OPT_IN
def test_a_reference_without_reject_dominance_is_caught(onsite_server, recorded):
    """The difftest tells the composition's rule from the trial's: the case it shrinks to has a subprotocol that rejects and one that
    is undecided, under a commitment that binds their digests."""
    trial = Trial(R.ONSITE.id, R.ONSITE.components, R.ONSITE.same, R.ONSITE.ties)
    with pytest.raises(D.Mismatch) as e:
        D.run(D.ProtocolPort("onsite", trial.answer, ONSITE.inputs, package=ONSITE_PACKAGE), examples=1000,
              server=onsite_server)
    assert (e.value.lean[0], e.value.python[0]) == (R.REJECT, R.UNDECIDED), (e.value.lean, e.value.python)
    parts = R.ONSITE.verdicts(e.value.input)
    assert R.REJECT in parts and R.UNDECIDED in parts, parts
