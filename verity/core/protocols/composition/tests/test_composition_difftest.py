"""Onsite's and remote's Lean (`onsite serve`, `remote serve`: each product's executable, the core CLI) against the
Python reference (`reference.py`: the trial's composition rule with reject dominance, over the harness's framing of the
core interface and the subprotocols' own Python). Hypothesis draws each subprotocol's case to accept, reject, be
undecided or fail to decode (from its own difftest's generator: PoUW's periods, made once from
`pouw_period_cases`, the network certifier's period `build`, the inference-only policy's `drawn`, which is
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
field missing or not hex, not an object), and every field of the outcome but its reason must agree exactly. The drawn
sampled proofs' cases all reject: v2 refuses evidence naming no session (`oneSession`). So the 1000-case runs reject,
onsite often with a component undecided beside the rejection. The one audit that accepts is recorded (`V2_CASE`, a
real `--zk` session on its challenger's coin tree, `accepted`): a case on it costs the session's verify, about 10 s of
Lean, so only small runs aim every component at acceptance on it (`composite`'s `acc`), where onsite accepts, is
undecided and rejects with no component rejecting, and remote, which never accepts, refuses by its `windows` tie.
`product IDS
serve` over each composite's subprotocols, the composite `verity run`'s verifier runs, is held to the same reference,
and `product IDS id` names remote and onsite for their subprotocols and refuses every other set: the same two
composites a configuration may list (`verity.core.deploy.config.COMPOSITES`)."""

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
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

import periods
import reference as R
import verity_difftest as D
from verity_difftest import canonical

from verity.core.deploy import config as K
from verity.core.protocols.sampled_proofs import parties as SPP
from verity.ml.circuits.types import Array, Value

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


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


@functools.cache
def _empty_audit(i: int) -> tuple[dict, dict, dict]:
    """One-stage v2's audit of `OS.SETUPS[i]`, a program that reads no registered value, its params but δ, its record
    and its evidence: its draw at its challenger's coins (`_challenger_draw`, a session whose challenger sampled no coin
    tree) drew no unit (a subset of none), so it names no session and no proved unit, and its params name the circuit
    and public files `BLOBS` hold. Made once each: a challenger's signed session is slow to make."""
    s, law = OS.SETUPS[i], {"k": 0, "law": "subset"}
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
    `SampledProofs.OneStageZk`, or v1's, which the form refuses), with `br`, one of `FORM_BREAKS`, when given."""
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

#: One-stage v2's audit of a real session, as `serve` reads its case (``case.json``, with its blobs) and Lean's answer to
#: it, an acceptance (``verdict.json``): `benchmarks/one_stage/v2_case.sh`'s, a RoPE d64 statement of 32 proof blocks
#: proved by C-Flock under `--zk`, its draw (all 32) and its coin tree from one withheld session of a test challenger.
V2_CASE = "art:227cd37ad55ff737bcdd1705f2fdbf7100789762fdde4ca7091565256d31b032"
VERIFIER = Path(__file__).resolve().parents[4] / "ml" / "flock" / "verifier"


@dataclasses.dataclass(frozen=True)
class Accepted:
    """Sampled proofs' audit that accepts (``{"params", "record", "evidence"}``, `V2_CASE`'s), and the blobs a product's
    case on it hands every component, and their hex: its files after `BLOBS`, so the drawn audits find theirs too."""

    audit: dict
    blobs: tuple[bytes, ...]
    hexed: tuple[str, ...]


def accepted_of(case: dict) -> Accepted:
    """`V2_CASE`'s case as an `Accepted`, its program (a decoded descriptor) put among the ones the one-stage audit
    knows (`ONE_STAGE.PROGRAMS`), which v1's decoders read and the policy declares (`policy_on`)."""
    desc = case["params"]["program"]
    OS.PROGRAMS[OS.PO.program_sha512(desc)] = OS.codec.decode_program(desc, R.INFERENCE.NR.declared(desc))
    audit = {"params": case["params"], "record": json.loads(bytes.fromhex(case["commitment"])),
             "evidence": json.loads(bytes.fromhex(case["evidence"]))}
    blobs = (*BLOBS, *map(bytes.fromhex, case["blobs"]))
    return Accepted(audit, blobs, tuple(b.hex() for b in blobs))

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
        return [R.REJECT]
    if C is R.WIPE:
        return list(R.WIPE_CASES.KINDS)
    if C is R.POUW_PERIOD:
        return [R.ACCEPT, R.REJECT]
    return [R.ACCEPT, R.REJECT, R.UNDECIDED]


@st.composite
def component_case(draw, C, kind: str | None, acc: Accepted | None = None) -> dict:
    """`C`'s case aimed at `kind` (any of `kinds(C)` when it is None; sampled proofs' are all refused, whatever the
    aim, but `acc`'s audit for an acceptance when given), from its own difftest's generator or its recorded cases, with
    its digest as its Python reads it on the blobs (`acc`'s, else `BLOBS`; the case's own commitment when it doesn't
    decode, which a rejection never binds)."""
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
    elif acc and kind == R.ACCEPT:
        case = form_case(acc.audit)
    else:
        case = draw(SAMPLED_CASES)
    return as_component(C, case, acc.blobs if acc else BLOBS)


def as_component(C, case: dict, blobs=BLOBS) -> dict:
    """`C`'s case as a composite's entry, with its digest as its Python reads it on `blobs` (the case's own commitment
    when it doesn't decode, which a rejection never binds)."""
    root = bytes.fromhex(case["commitment"])
    digest = C.answer({**case, "blobs": list(blobs)} if getattr(C, "blobs", False) else case)["commitment"]
    return {"id": C.id, "params": case["params"], "root": root, "evidence": bytes.fromhex(case["evidence"]),
            "digest": bytes.fromhex(digest) if digest else root}


@functools.cache
def _declaration(program: str) -> tuple[dict, dict] | None:
    """The policy's params but the log, and its evidence, for the declared circuit `program` (descriptor text) of a
    program the one-stage audit knows (`ONE_STAGE.PROGRAMS`), as the provenance difftest's declarations make them
    (`baseline_difftest_vectors.build`): its program function's parameters (for a program known only by its descriptor,
    `accepted_of`'s, one array a root input node, as `nci_reference.owners` lowers them), every gate in one inference
    block with Γ its gates and α over what any gate reads, and no registered information. None for another program."""
    NR = R.INFERENCE.NR
    digest = R.INFERENCE._digest(program)
    if isinstance(digest, Exception) or digest not in OS.PROGRAMS:
        return None
    desc = json.loads(program)
    decoded = NR.codec.decode_program(desc, NR.declared(desc))
    inputs, _, _ = NR.root_nodes(program)
    fn = getattr(OS.PROGRAMS[digest], "fn", None)
    if fn is not None:
        parameters = [[n, t.to_json()] for n, t in fn.signature()[0]]
    else:
        nodes = [n for n in decoded.root.body.nodes if NR.is_input(n.fn)]
        parameters = [[f"in{k}", Array(n.n if n.form in ("batch", "scan") else 1, Value(n.fn.ret.w)).to_json()]
                      for k, n in enumerate(nodes)]
    params = {"alpha": [64 * decoded.root.gates + 1, 1], "gamma": decoded.root.gates, "program": program,
              "parameters": parameters}
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


def one_record(product: R.Same, cases: list[dict], windows: bool = False, blobs=BLOBS) -> list[dict]:
    """The cases with every subprotocol of ``product.same`` committing to sampled proofs' registration record, each
    with its digest as its Python reads it on `blobs`, and the policy's case on the audit's own declared circuit
    where it can be (`policy_on`): cases that pass `Core.Composition.strays` and leave the verdict to the subprotocols.
    With ``windows``, every window of remote's PoUW period names that record too (`windows_on`), so that remote's
    `windows` tie holds; the period then refuses, since a window's Commit record registers its own tables. Without it,
    an honest period's windows name their own records, and the tie refuses."""
    first = next(c for c in cases if c["id"] == R.SAMPLED_PROOFS.id)
    by_id = {C.id: C for C in product.components}

    def recommit(c: dict, root: bytes, evidence: bytes) -> dict:
        C = by_id[c["id"]]
        case = {"commitment": root.hex(), "evidence": evidence.hex(), "params": c["params"]}
        digest = C.answer({**case, "blobs": list(blobs)} if getattr(C, "blobs", False) else case)["commitment"]
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
def composite(draw, product: R.Product, acc: Accepted | None = None) -> dict:
    """One case of `product`: its subprotocols' cases (one time in six every one aimed at acceptance, one in six the same
    but for one of those that can be undecided, when one can, aimed at undecided, so that a product of seven reaches its
    binding breaks with a component undecided, and each at any verdict otherwise), the subprotocols it ties recommitted
    to one record (`one_record`) two times in three when they are aimed at acceptance and one in three otherwise, then
    at most one break of the composition's own inputs (half the time a break of the commitment alone, `BINDING`, when
    every case is aimed at acceptance). With `acc`, on its blobs, every case is aimed at acceptance, sampled proofs' at
    `acc`'s audit, or all but one that can be undecided, half the time each, and the one break is of the commitment
    alone, one time in four: each such case costs the session's verify (about 10 s of Lean)."""
    mode = draw(st.integers(0, 1 if acc else 5))
    blobs = acc.blobs if acc else BLOBS
    undecided = [j for j, C in enumerate(product.components) if R.UNDECIDED in kinds(C)]
    pending = draw(st.sampled_from(undecided)) if undecided else None
    aims = [None if mode > 1 else R.UNDECIDED if mode == 1 and j == pending else R.ACCEPT
            for j in range(len(product.components))]
    cases = [draw(component_case(C, k, acc)) for C, k in zip(product.components, aims)]
    if isinstance(product, R.Same) and draw(st.integers(0, 2)) < (2 if mode <= 1 else 1):
        cases = one_record(product, cases, R.WINDOWS in getattr(product, "ties", ()) and draw(st.booleans()), blobs)
    named = [{"id": c["id"], "params": c["params"]} for c in cases]
    entries = [{"commitment": c["root"].hex(), "evidence": c["evidence"].hex(), "id": c["id"], "params": c["params"]}
               for c in cases]
    commitment = R.bind(product.id, [(c["id"], c["digest"]) for c in cases])
    params: object = {"components": named}
    if acc:
        m = draw(st.sampled_from(BINDING)) if draw(st.integers(0, 3)) == 0 else None
    else:
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
    hexed: object = list(acc.hexed) if acc else [b.hex() for b in blobs]
    if m == "blobs-not-hex":
        hexed = [*hexed[:1], draw(st.sampled_from(["zz", "A0", "0", 7]))]
    elif m == "blobs-not-list":
        hexed = draw(st.sampled_from([CIRCUIT.hex(), {}, None]))
    elif m == "blob-extra":
        hexed = [draw(st.binary(max_size=8)).hex(), *hexed]
    case = {"commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}
    return case if m == "no-blobs" else {**case, "blobs": hexed}


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


class RejectingPort(D.ProtocolPort):
    """A product port whose cases reject: sampled proofs' do (`SAMPLED_CASES`), so the run's non-vacuity bar is a
    rejection, and its other bars are the ways its subprotocols' verdicts and records meet, counted in `seen`
    (`_agree`). Remote's on `accepted`'s cases too: it never accepts (`windows`)."""
    accepted = R.REJECT


class AcceptingPort(D.ProtocolPort):
    """Onsite's port on `accepted`'s cases (`on_accepted`), every one aimed at acceptance: the run's non-vacuity bar is
    an acceptance, and some must not be accepted."""
    accepted = R.ACCEPT


def port(name: str, product: R.Product, seen: collections.Counter, serve: tuple[str, ...] = ("serve",),
         package: str | None = None) -> D.ProtocolPort:
    """`name serve` (or `name SERVE`) of the Lake package in `package` (default `name`), on the product's cases and,
    one sample in ten, any JSON line."""
    return RejectingPort(name, reference(product, seen), st.one_of(*[composite(product)] * 9, JSON),
                         package=PROTOCOLS / (package or name), serve=serve, need_rejects=False)


def ids(product: R.Product) -> str:
    return ",".join(C.id for C in product.components)


SEEN = {"onsite": collections.Counter(), "remote": collections.Counter()}
ONSITE = port("onsite", R.ONSITE, SEEN["onsite"])
REMOTE = port("remote", R.REMOTE, SEEN["remote"])
#: `product IDS serve` over each composite's subprotocols, which `verity run`'s verifier runs
PRODUCT = {p.id: port("product", p, SEEN[n], serve=(ids(p), "serve"), package="onsite")
           for n, p in (("onsite", R.ONSITE), ("remote", R.REMOTE))}


def on_accepted(p: D.ProtocolPort, product: R.TiedSame, seen: collections.Counter, acc: Accepted) -> D.ProtocolPort:
    """`p`'s executable on `product`'s cases aimed at acceptance on `acc` alone (`composite`'s `acc`): an
    `AcceptingPort` for onsite, a `RejectingPort` for remote."""
    kind = RejectingPort if R.WINDOWS in product.ties else AcceptingPort
    return kind(p.exe, reference(product, seen), composite(product, acc), package=p.package, serve=p.serve,
                need_rejects=kind is AcceptingPort)


onsite_server = D.server_fixture(ONSITE)
remote_server = D.server_fixture(REMOTE)
product_onsite_server = D.server_fixture(PRODUCT[R.ONSITE.id])
product_remote_server = D.server_fixture(PRODUCT[R.REMOTE.id])


@pytest.fixture(scope="module")
def accepted(tmp_path_factory) -> Accepted:
    """`V2_CASE`, fetched from the evidence store as `test_proofs.py` fetches its fixture (skipped when the store is not
    reachable), whose audit the reference accepts too."""
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        case = json.loads(ci.fetch(f"{V2_CASE}/case.json", tmp_path_factory.mktemp("store")).read_text())
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    acc = accepted_of(case)
    got = R.SAMPLED_PROOFS.answer({**form_case(acc.audit), "blobs": list(acc.blobs)})
    assert got["verdict"] == R.ACCEPT, got
    return acc


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
    """Lean and Python agree on every case, and the cases reach each of `kinds` (by default a rejection: the drawn
    sampled proofs' cases are all refused, so no product accepts unless `kinds` names an acceptance)."""
    seen.clear()
    stats = D.run(p, examples=examples, seed=0, server=server)
    print(f"{p.label}: {stats}, verdicts {dict(seen)}")
    assert R.ACCEPT in kinds or seen[R.ACCEPT] == 0, dict(seen)
    assert stats.examples == examples
    assert stats.counts == {v: seen[v] for v in (R.ACCEPT, R.REJECT, R.UNDECIDED) if seen[v]}
    for kind in kinds:
        assert seen[kind] > examples // 50, (kind, dict(seen))


@pytest.mark.slow
def test_remote_agrees(remote_server):
    """Remote's components meet on one record (left to the components) and on two (refused), each often."""
    _agree(REMOTE, remote_server, SEEN["remote"], 1000, (R.REJECT, "one record", "two records"))


@pytest.mark.slow
def test_remote_agrees_on_an_accepted_audit(remote_server, accepted):
    """On cases whose components are all aimed at acceptance, sampled proofs' at `accepted`'s audit, remote rejects
    with no component rejecting: by its `windows` tie, by two records, or by a commitment that doesn't bind them. Each
    case costs the session's verify (about 10 s of Lean): the drawn audits all reject, so only these reach the record
    checks and the binding with every component accepting."""
    _agree(on_accepted(REMOTE, R.REMOTE, SEEN["remote"], accepted), remote_server, SEEN["remote"], 10,
           (R.REJECT, "reject, no component rejected"))


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


def product_case(product: R.Product, cases: list[dict], blobs=BLOBS) -> dict:
    """`product`'s case on its components' cases as they are, with no break of its own."""
    return {"commitment": R.bind(product.id, [(c["id"], c["digest"]) for c in cases]).hex(),
            "evidence": canonical({"components": [{"commitment": c["root"].hex(), "evidence": c["evidence"].hex(),
                                                   "id": c["id"], "params": c["params"]} for c in cases]}).encode().hex(),
            "params": {"components": [{"id": c["id"], "params": c["params"]} for c in cases]},
            "blobs": [b.hex() for b in blobs]}


@pytest.mark.slow
def test_remote_refuses_two_records_where_its_product_accepts(remote_server, accepted):
    """With every component accepting, sampled proofs on `accepted`'s audit and the policy on a record of its own,
    remote's product without the record checks accepts, and remote refuses the two records, in Lean as in Python. One
    case, the session's verify (about 10 s of Lean): the drawn audits reject, so no other case shows a record check
    refusing what the product would accept."""
    product = R.Product(R.REMOTE.id, R.REMOTE.components)

    @settings(max_examples=1, database=None, deadline=None, suppress_health_check=list(HealthCheck))
    @given(component_case(R.INFERENCE_ONLY, R.ACCEPT), component_case(R.POUW_PERIOD, R.ACCEPT),
           component_case(R.SAMPLED_PROOFS, R.ACCEPT, accepted))
    def check(policy, period, audit):
        case = product_case(R.REMOTE, [policy, period, audit], accepted.blobs)
        assert product.answer(case)["verdict"] == R.ACCEPT, product.answer(case)
        for out in (R.REMOTE.answer(case), remote_server.exchange(case)):
            assert out["verdict"] == R.REJECT and out["reason"].startswith(f"record: {R.INFERENCE_ONLY.id} "), out
    check()


@pytest.mark.slow
def test_remote_refuses_a_period_window_on_another_record(remote_server, accepted):
    """With the policy accepting on sampled proofs' record, sampled proofs accepting on it (`accepted`) and an honest
    period accepted, every component accepts, but each of the period's windows names its own Commit record, not the
    policy's: remote refuses by its `windows` tie, naming the policy's record and the first window's, in Lean as in
    Python. Each case costs the session's verify (about 10 s of Lean); the drawn audits reject before the tie is
    read."""
    seen = collections.Counter()

    @settings(max_examples=2, database=None, deadline=None, suppress_health_check=list(HealthCheck))
    @given(component_case(R.INFERENCE_ONLY, R.ACCEPT), component_case(R.POUW_PERIOD, R.ACCEPT),
           component_case(R.SAMPLED_PROOFS, R.ACCEPT, accepted))
    def check(policy, period, audit):
        policy, period, audit = one_record(R.REMOTE, [policy, period, audit], blobs=accepted.blobs)
        case = product_case(R.REMOTE, [policy, period, audit], accepted.blobs)
        assert R.REMOTE.verdicts(case) == [R.ACCEPT] * 3, R.REMOTE.verdicts(case)
        record = R.policy_record(policy["root"], policy["evidence"])[0].hex()
        window = R.window_records(period["root"], period["evidence"])[0].hex()
        assert record != window
        for out in (R.REMOTE.answer(case), remote_server.exchange(case)):
            assert out["verdict"] == R.REJECT, out
            assert out["reason"] == (f"record: {R.INFERENCE_ONLY.id} committed {record}, "
                                     f"{R.POUW_PERIOD.id} committed {window}"), out
        seen[len(json.loads(period["evidence"])["windows"])] += 1
    check()
    assert sum(seen.values()) > 0, dict(seen)


def test_sampled_proofs_refuses_one_stage_v1(remote_server):
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
            case = product_case(R.REMOTE, one_record(R.REMOTE, [policy, period, as_component(R.SAMPLED_PROOFS, sampled)]))
            assert R.REMOTE.answer(case)["verdict"] == R.REJECT
            out = remote_server.exchange(case)
            assert out["verdict"] == R.REJECT and f"{R.SAMPLED_PROOFS.id} reject: {why}" in out["reason"], out
            seen[br] += 1
    check()
    assert set(seen) == set(want), dict(seen)


@pytest.mark.slow
def test_onsite_agrees(onsite_server, recorded):
    """Onsite rejects with a component undecided beside the rejection, its policy and integrity meet on one record
    (left to the subprotocols) and on two (refused), and PoUS and PoUB-P2 on one store and on two, each often; and
    PoUB-P2's case named as the generic bandwidth audit's, which is no component, refused."""
    _agree(ONSITE, onsite_server, SEEN["onsite"], 1000, (R.REJECT, "reject, a component undecided", "one record",
                                                       "two records", "one store", "two stores"))
    assert SEEN["onsite"]["generic bandwidth"] > 0, dict(SEEN["onsite"])


#: What onsite reaches on `accepted`'s cases, every component aimed at acceptance: an acceptance, undecided with one
#: component undecided, and a rejection no component's explains (a commitment that doesn't bind them, or PoUS and
#: PoUB-P2 on two stores).
ONSITE_ACCEPTED = (R.ACCEPT, R.UNDECIDED, "reject, no component rejected")


@pytest.mark.slow
def test_onsite_agrees_on_an_accepted_audit(onsite_server, recorded, accepted):
    """Onsite accepts, is undecided and rejects with no component rejecting (`ONSITE_ACCEPTED`), in Lean as in Python,
    on cases whose components are all aimed at acceptance but at most one at undecided, sampled proofs' at `accepted`'s
    audit. Each case costs the session's verify (about 10 s of Lean): the drawn audits all reject, so only these reach
    the product's acceptance, its undecided verdict and its binding with no component rejecting."""
    _agree(on_accepted(ONSITE, R.ONSITE, SEEN["onsite"], accepted), onsite_server, SEEN["onsite"], 16, ONSITE_ACCEPTED)


def test_product_names_only_the_two_composites():
    """`product IDS id` is remote over remote's subprotocols and onsite over onsite's, in any order, exactly the
    configuration's composites (`verity.core.deploy.config.COMPOSITES`, by the ids `SUBPROTOCOLS` gives their names),
    and refuses every other set of their subprotocols and the wipe (exit 64)."""
    exe = D.build(PROTOCOLS / "onsite", "product")

    def named(xs) -> tuple[int, str]:
        p = subprocess.run([exe, ",".join(xs), "id"], capture_output=True, text=True)
        return p.returncode, p.stdout.strip()

    want = {p.id: sorted(C.id for C in p.components) for p in (R.ONSITE, R.REMOTE)}
    assert {cid: sorted(subs.values()) for cid, subs in SPP.SUBPROTOCOLS.items()} == want
    assert {cid: sorted(names) for cid, names in K.COMPOSITES.items()} == {
        cid: sorted(subs) for cid, subs in SPP.SUBPROTOCOLS.items()}
    for cid, ids in want.items():
        assert named(ids) == (0, cid)
        assert named(ids[::-1]) == (0, cid)
    every = sorted({i for ids in want.values() for i in ids} | {"verity/wipe/v0"})
    others = [s for k in range(len(every) + 1) for s in itertools.combinations(every, k)
              if sorted(s) not in want.values()]
    assert len(others) == 2 ** len(every) - 2
    for s in others:
        assert named(s)[0] == 64, s
    assert named([R.POUW_PERIOD.id, R.POUW_PERIOD.id, R.SAMPLED_PROOFS.id])[0] == 64
    assert named([R.INFERENCE_ONLY.id, "verity/pouw/v1", R.SAMPLED_PROOFS.id])[0] == 64


@pytest.mark.slow
def test_product_over_remotes_subprotocols_is_remote(product_remote_server):
    _agree(PRODUCT[R.REMOTE.id], product_remote_server, SEEN["remote"], 1000, (R.REJECT, "one record", "two records"))


@pytest.mark.slow
def test_product_over_remotes_subprotocols_on_an_accepted_audit(product_remote_server, accepted):
    """As `test_remote_agrees_on_an_accepted_audit`, for `product IDS serve` (about 10 s of Lean a case)."""
    _agree(on_accepted(PRODUCT[R.REMOTE.id], R.REMOTE, SEEN["remote"], accepted), product_remote_server, SEEN["remote"],
           6, (R.REJECT, "reject, no component rejected"))


@pytest.mark.slow
def test_product_over_onsites_subprotocols_is_onsite(product_onsite_server, recorded):
    _agree(PRODUCT[R.ONSITE.id], product_onsite_server, SEEN["onsite"], 200, (R.REJECT, "reject, a component undecided"))


@pytest.mark.slow
def test_product_over_onsites_subprotocols_on_an_accepted_audit(product_onsite_server, recorded, accepted):
    """As `test_onsite_agrees_on_an_accepted_audit`, for `product IDS serve` (about 10 s of Lean a case)."""
    _agree(on_accepted(PRODUCT[R.ONSITE.id], R.ONSITE, SEEN["onsite"], accepted), product_onsite_server,
           SEEN["onsite"], 12, ONSITE_ACCEPTED)


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


@pytest.mark.slow
def test_onsite_refuses_two_stores_where_it_would_be_undecided(onsite_server, recorded, accepted):
    """With every component accepting but the certifier undecided, sampled proofs on `accepted`'s audit and the policy
    on its record, and PoUS's run on another store than PoUB-P2's (`periods.OTHER_STORE`), onsite without its store tie
    is undecided, and onsite refuses the two stores, in Lean as in Python. One case, the session's verify (about 10 s of
    Lean): the drawn audits reject, so no other case shows the tie refusing what onsite would leave undecided."""
    untied = dataclasses.replace(R.ONSITE, ties=())
    aims = {R.CERTIFIER: R.UNDECIDED, R.POUS: periods.OTHER_STORE}

    @settings(max_examples=1, database=None, deadline=None, suppress_health_check=list(HealthCheck))
    @given(st.tuples(*[component_case(C, aims.get(C, R.ACCEPT), accepted) for C in R.ONSITE.components]))
    def check(cases):
        case = product_case(R.ONSITE, one_record(R.ONSITE, list(cases), blobs=accepted.blobs), accepted.blobs)
        assert untied.answer(case)["verdict"] == R.UNDECIDED, untied.answer(case)
        for out in (R.ONSITE.answer(case), onsite_server.exchange(case)):
            assert out["verdict"] == R.REJECT and out["reason"].startswith(f"{R.STORE.what}: "), out
    check()


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
