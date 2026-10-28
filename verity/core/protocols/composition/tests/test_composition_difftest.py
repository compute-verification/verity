"""Onsite's and remote's Lean (`onsite serve`, `remote serve`: each product's executable, the core CLI) against the Python
reference (`reference.py`: the trial's composition rule with reject dominance, over the harness's framing of the
core interface and the components' own Python). Hypothesis draws each component's case to accept, reject, be undecided or
fail to decode (from its own difftest's generator: PoUW's `pouw_check_cases` and, made once, `pouw_period_cases`'
periods, the network certifier's period `build`, sampled proofs' audits as v2's cases, `v2_case`, the wipe's signed
wipes, `wipe_protocol_case.case`; PoUS's and the bandwidth audit's from the small periods of their hooks, `periods.py`),
and onsite's registration case from NCI's declarations (on v2's own registration record and log, mostly, else on NCI's
own, which onsite refuses: one registration digest in both), hands every component the same blobs (`BLOBS`, v2's
circuit and public file), then often breaks the composition's own inputs (a component dropped, unknown, repeated or out
of order, a case under other params, a wrong or short commitment, non-canonical or malformed evidence, blobs missing or
malformed; onsite's registration under another id or other params, its digest unbound, or left out) or the case itself
(a field missing or not hex, not an object), and every field of the outcome but its reason must agree exactly: the
components' certified quantities and profiles among them. v2 accepts only on a session's proof, which no case carries,
so sampled proofs is undecided at best and no product accepts here."""

from __future__ import annotations

import collections
import functools
import hashlib
import itertools
import json
import random
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import periods
import reference as R
import verity_difftest as D
from verity_difftest import canonical

PROTOCOLS = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text(max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=3), inner, max_size=3),
                    max_leaves=6)


#: PoUW's mutations of its window that leave the case's framing whole (`case-*` break the line `serve` reads, which a
#: composition's evidence can't carry).
POUW_REFUSALS = [n for n in R.POUW_CASES.NAMES if not n.startswith("case-") and n != "answer-undecided"]


@functools.cache
def pouw_periods() -> dict[str, list[dict]]:
    """PoUW's periods by the verdict they aim at, made once (a period of the auditor's ledger and its windows is slow to
    make): a few honest ones, and one of each refusal of `credit.period_credit` and of each break of the protocol's
    framing. A period is accepted or rejected, never undecided."""
    PPC = R.POUW_PERIOD_CASES
    return {R.ACCEPT: [PPC.instance(random.Random(f"honest/{s}")) for s in range(3)],
            R.REJECT: [PPC.instance(random.Random(f"{n}/0"), n) for n in PPC.NAMES]}


OS = R.ONE_STAGE
#: The files every product case hands its components as blobs: v2's cases name them as their circuit and public file.
CIRCUIT, PUBLIC = b"not a circuit file", b"not a public file"
BLOBS = [CIRCUIT, PUBLIC]


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


@st.composite
def empty_draws(draw) -> dict:
    """A valid audit of a program that reads no registered value, whose draw drew no unit (a subset of none, or Bernoulli
    at 0): with no session and no proved unit, v1 is undecided on it for want of the verifier of record."""
    s = draw(st.sampled_from(OS.SETUPS))
    law = draw(st.sampled_from([{"k": 0, "law": "subset"}, {"law": "bernoulli", "p": "0/1"}]))
    part = {"program": s.sha512, "query": s.query}
    rec = OS.R.record(s.program, s.query, law, OS.SCHEME, [{"port": p, "root": OS.ROOTS[p], **s.ports[p]} for p in s.ports],
                      OS.WINDOW, leaf_layer=OS.LEAF, partition=part)
    d = {**OS.DR.derive(s.program, s.query, law, None), "population": s.population, "units": []}
    evidence = {"draw": d, "instances": 0, "proved_units": [], "receipt": OS.R.receipt(rec, "v", received=OS.RECEIVED),
                "served": None, "session_roots": None, "sessions": [], "statement": {}, "verdicts": {}}
    params = {"anchors": [], "delta": draw(st.sampled_from(["1/1000", "1/10", "1/2"])), "law": law, "leaf_layer": OS.LEAF,
              "log": [], "partition": part, "ports": s.ports, "program": s.desc, "query": s.query, "registered": None,
              "scheme": OS.SCHEME, "work_table": None}
    return OS._json({"params": params, "record": rec, "evidence": evidence})


#: One change to a v2 case that v2 refuses: verdicts handed in, no proofs, a circuit that is no digest, a public file in
#: no blob, a session the proofs name that the evidence doesn't.
V2_BREAKS = ["verdicts", "no-proofs", "circuit", "public", "session"]


def v2_case(a: dict, br: str | None = None) -> dict:
    """An audit (`{"params", "record", "evidence"}`) as v2's case (`SampledProofs.OneStageZk`): the params naming `CIRCUIT`
    by its digest and no registered file, the evidence its sessions' files in place of verdicts (each record in no blob),
    with one of `V2_BREAKS`."""
    e = {k: v for k, v in a["evidence"].items() if k != "verdicts" or br == "verdicts"} \
        if isinstance(a["evidence"], dict) else a["evidence"]
    names = e.get("sessions") if isinstance(e, dict) and isinstance(e.get("sessions"), list) else []
    names = [*names, "extra"] if br == "session" else names
    proofs = {"public": _sha(b"other" if br == "public" else PUBLIC),
              "sessions": [{"name": n, "record": _sha(json.dumps(n).encode()), "reps": {}} for n in names]}
    if isinstance(e, dict):
        e = {**e, "proofs": proofs}
    p = a["params"]
    if isinstance(p, dict) and br != "no-proofs":
        p = {**p, "proofs": {"circuit": "ab" if br == "circuit" else _sha(CIRCUIT), "registered": None}}
    return OS._case({"params": p, "record": a["record"], "evidence": e})


#: v2's cases by the verdict they aim at: an audit that drew nothing (undecided, v2's best with no session's proof), and
#: any audit with up to two changes or one break of v2's own.
V2_CASES = {R.UNDECIDED: empty_draws().map(v2_case),
            R.REJECT: st.one_of(OS.changed_audits(most=2).map(v2_case),
                                st.builds(v2_case, empty_draws(), st.sampled_from(V2_BREAKS)))}

NP = R.PERIOD
#: The period check's changes by the verdict they aim at (`case` breaks the line `serve` reads, as PoUW's `case-*`).
PERIOD_CASES = {v: tuple(m for m, x in NP.EXPECT.items() if x == v and m != "case")
                for v in (R.ACCEPT, R.REJECT, R.UNDECIDED)}

#: NCI's changes by the verdict they aim at; NCI is never undecided (`kinds`).
NCI_CASES = {R.ACCEPT: ("honest",), R.REJECT: tuple(m for m in R.NCI.MUTATIONS if m != "honest")}

#: A recorded component's kinds of case (`periods.build`), accepted ones first and thrice.
KINDS = [R.ACCEPT] * 3 + [R.REJECT, R.UNDECIDED, "other-setup", "broken-params", "broken-evidence"]


def kinds(C) -> list[str]:
    """The kinds of case `component_case` may aim `C` at when it is free to."""
    if isinstance(C, R.Recorded):
        return [k for k in KINDS if k in C.cases]
    if C is R.SAMPLED_PROOFS:
        return list(V2_CASES)
    if C is R.WIPE:
        return list(R.WIPE_CASES.KINDS)
    if C is R.NCI_CHECK:
        return list(NCI_CASES)
    if C is R.POUW_PERIOD:
        return [R.ACCEPT, R.REJECT]
    return [R.ACCEPT, R.REJECT, R.UNDECIDED]


@st.composite
def component_case(draw, C, kind: str | None) -> dict:
    """`C`'s case aimed at `kind` (any of `kinds(C)` when it is None; v2's best for an acceptance), from its own
    difftest's generator or its recorded cases, with its digest as its Python reads it on `BLOBS` (the case's own
    commitment when it doesn't decode, which a rejection never binds)."""
    kind = kind or draw(st.sampled_from(kinds(C)))
    if isinstance(C, R.Recorded):
        case = C.cases[kind][0]
    elif C is R.POUW:
        name = {R.ACCEPT: "honest", R.UNDECIDED: "answer-undecided"}.get(kind) or draw(st.sampled_from(POUW_REFUSALS))
        case = R.POUW_CASES.instance(draw(st.randoms(use_true_random=False)), name)
    elif C is R.POUW_PERIOD:
        case = draw(st.sampled_from(pouw_periods()[kind]))
    elif C is R.CERTIFIER:
        case = NP.build(draw(st.sampled_from(PERIOD_CASES[kind])), draw(st.integers(0, 1 << 16)), small=True)
    elif C is R.WIPE:
        case = R.WIPE_CASES.case(kind, draw(st.integers(0, 1)))
    elif C is R.NCI_CHECK:
        case = draw(R.NCI.cases(NCI_CASES[kind]))
    else:
        case = draw(V2_CASES[R.UNDECIDED if kind == R.ACCEPT else kind])
    root = bytes.fromhex(case["commitment"])
    digest = C.answer({**case, "blobs": BLOBS} if getattr(C, "blobs", False) else case)["commitment"]
    return {"id": C.id, "params": case["params"], "root": root, "evidence": bytes.fromhex(case["evidence"]),
            "digest": bytes.fromhex(digest) if digest else root}


@st.composite
def registration_case(draw, committed: dict) -> dict:
    """NCI's case, as `component_case` gives it, on the registration ``committed`` (the committing component's case)
    commits to: an honest declaration with its θ and provenance dropped, which reads the record only through the log's
    chain, on that case's record and log, so that one registration digest is in both when the record chains; one time in
    four, NCI's own declaration on its own record at any verdict, which a registered composition refuses."""
    if draw(st.integers(0, 3)) == 0:
        return draw(component_case(R.NCI_CHECK, None))
    s = R.NCI.split(draw(st.sampled_from(R.NCI.seeds()[0])))
    log = committed["params"].get("log", []) if isinstance(committed["params"], dict) else []
    params = {**s["params"], "log": log}
    evidence = canonical({**s["evidence"], "theta": [], "provenance": []}).encode()
    digest = R.NCI_CHECK.answer({"commitment": committed["root"].hex(), "evidence": evidence.hex(),
                                 "params": params})["commitment"]
    return {"id": R.NCI_CHECK.id, "params": params, "root": committed["root"], "evidence": evidence,
            "digest": bytes.fromhex(digest) if digest else committed["root"]}


#: The binding breaks come five times: with `commitment-size` they are the breaks that leave every component's case standing,
#: and so the way to a rejection that no component's own explains.
MUTATIONS = [None] * 20 + ["drop-param", "drop-case", "unknown-param", "unknown-case", "swap-params", "swap-cases",
                           "repeat-param", "repeat-case", "case-params", "commitment-size", "evidence-noncanonical",
                           "params-extra-key", "case-extra-key", "case-missing-key", "bad-hex", "garbage-evidence",
                           "garbage-params", "not-a-case", "no-blobs", "blobs-not-hex", "blobs-not-list",
                           "blob-extra"] + ["binding", "binding-id"] * 5
#: The breaks of the commitment alone, which leave every component's case standing.
BINDING = ["binding", "binding-id", "commitment-size"]
#: A registered composition's breaks of its registration (`Core.Composition.registered`): its entry or its case under
#: another id, its case under other params, a commitment that doesn't bind its digest, and its entry or case left out.
REGISTRATION_MUTATIONS = [None] * 24 + ["registration-id", "registration-case-id", "registration-params",
                                        "registration-digest", "registration-dropped", "registration-case-dropped"]


@st.composite
def composite(draw, product: R.Product | R.Registered) -> dict:
    """One case of `product`: its components' cases (one time in six every one aimed at acceptance, one in six every one
    but one that is aimed at undecided, so that a product of six reaches its binding breaks and its undecided verdict,
    and each at any verdict otherwise), then at most one break of the composition's own inputs (half the time a break of
    the commitment alone, `BINDING`, when every case stands). A registered composition's case carries its registration's
    too (`registration_case`, on the registration the committing component's case commits to three times in four), with
    at most one break of its own (`REGISTRATION_MUTATIONS`)."""
    registered = product if isinstance(product, R.Registered) else None
    product = registered.product if registered else product
    mode = draw(st.integers(0, 5))
    pending = draw(st.sampled_from([j for j, C in enumerate(product.components) if R.UNDECIDED in kinds(C)]))
    aims = [None if mode > 1 else R.UNDECIDED if mode == 1 and j == pending else R.ACCEPT
            for j in range(len(product.components))]
    cases = [draw(component_case(C, k)) for C, k in zip(product.components, aims)]
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
        stranger = dict(which[i], id=draw(st.sampled_from(["verity/stub/nci/v0", "verity/example/v0", "", "a"])))
        which.insert(draw(st.integers(0, len(which))), stranger)
    elif m in ("swap-params", "swap-cases"):
        which[0], which[-1] = which[-1], which[0]
    elif m in ("repeat-param", "repeat-case"):
        which.insert(i, dict(which[i]))
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
    if registered:
        reg = draw(registration_case(cases[[C.id for C in product.components].index(registered.committer)]))
        rm = draw(st.sampled_from(REGISTRATION_MUTATIONS))
        rnamed = {"id": reg["id"], "params": reg["params"]}
        rentry = {"commitment": reg["root"].hex(), "evidence": reg["evidence"].hex(), "id": reg["id"],
                  "params": reg["params"]}
        rdigest = reg["digest"]
        if rm in ("registration-id", "registration-case-id"):
            (rnamed if rm == "registration-id" else rentry)["id"] = draw(st.sampled_from(
                [R.POUW.id, "verity/stub/nci/v0", ""]))
        elif rm == "registration-params":
            rentry["params"] = {"k": draw(st.integers(4, 6))}
        elif rm == "registration-digest":
            rdigest = draw(st.binary(min_size=64, max_size=64).filter(lambda b: b != reg["digest"]))
        if isinstance(params, dict) and "components" in params and rm != "registration-dropped":
            params = {**params, "registration": rnamed}
        if rm != "registration-case-dropped":
            body["registration"] = rentry
        commitment = draw(st.binary(max_size=63)) if m == "commitment-size" else commitment + rdigest
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


def reference(product: R.Product | R.Registered, seen: collections.Counter):
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
        return out
    return answer


class UndecidedPort(D.ProtocolPort):
    """A product port whose best verdict is undecided: integrity (v2) accepts only on a session's proof, which no case
    carries, so the run's non-vacuity bar is a case every component left undecided or accepted."""
    accepted = R.UNDECIDED


def port(name: str, product: R.Product | R.Registered, seen: collections.Counter) -> D.ProtocolPort:
    """`name serve` of the Lake package `name`, on the product's cases and, one draw in ten, any JSON line."""
    return UndecidedPort(name, reference(product, seen), st.one_of(*[composite(product)] * 9, JSON),
                         package=PROTOCOLS / name)


SEEN = {"onsite": collections.Counter(), "remote": collections.Counter()}
ONSITE = port("onsite", R.ONSITE, SEEN["onsite"])
REMOTE = port("remote", R.REMOTE, SEEN["remote"])
onsite_server = D.server_fixture(ONSITE)
remote_server = D.server_fixture(REMOTE)


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


def _agree(p: D.ProtocolPort, server, seen: collections.Counter, examples: int) -> None:
    """Lean and Python agree on every case, and the cases reach each verdict but acceptance: integrity (v2) accepts only
    on a session's proof, which no case carries, so no product accepts."""
    seen.clear()
    stats = D.run(p, examples=examples, seed=0, server=server)
    print(f"{p.label}: {stats}, verdicts {dict(seen)}")
    assert seen[R.ACCEPT] == 0, dict(seen)
    assert stats.examples == examples and stats.counts == {v: seen[v] for v in (R.REJECT, R.UNDECIDED)}
    for kind in (R.REJECT, R.UNDECIDED, "reject, a component undecided", "reject, no component rejected"):
        assert seen[kind] > examples // 50, (kind, dict(seen))


@pytest.mark.slow
def test_remote_agrees(remote_server):
    _agree(REMOTE, remote_server, SEEN["remote"], 1000)


@pytest.mark.slow
def test_onsite_agrees(onsite_server, recorded):
    _agree(ONSITE, onsite_server, SEEN["onsite"], 1000)


class Trial(R.Product):
    """The trial's rule over the components' verdicts (`trial_verdict`): a component left undecided leaves the product
    undecided even when another rejects. Everything else is the port's."""

    def check(self, ps, c, ev, blobs=()) -> tuple:
        answers = self.run(ps, ev, blobs)
        if not isinstance(answers, str) and self.binds(answers, c) and R.trial_verdict(
                [a["verdict"] for _, a in answers]) == R.UNDECIDED:
            return R.UNDECIDED, "a component is undecided"
        return super().check(ps, c, ev, blobs)


@pytest.mark.slow
def test_a_reference_without_reject_dominance_is_caught(onsite_server, recorded):
    """The difftest tells the composition's rule from the trial's: the case it shrinks to has a component that rejects and one that
    is undecided, under a commitment that binds their digests."""
    trial = R.Registered(R.ONSITE.id, R.ONSITE.registration, R.ONSITE.committer,
                         Trial(R.ONSITE.id, R.ONSITE.product.components))
    with pytest.raises(D.Mismatch) as e:
        D.run(D.ProtocolPort("onsite", trial.answer, ONSITE.inputs, package=PROTOCOLS / "onsite"), examples=1000,
              server=onsite_server)
    assert (e.value.lean[0], e.value.python[0]) == (R.REJECT, R.UNDECIDED), (e.value.lean, e.value.python)
    parts = R.ONSITE.verdicts(e.value.input)
    assert R.REJECT in parts and R.UNDECIDED in parts, parts
