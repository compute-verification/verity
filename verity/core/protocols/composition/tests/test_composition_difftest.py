"""Onsite's and remote's Lean (`onsite serve`, `remote serve`: each product's executable, the core CLI) against the Python
reference (`reference.py`: the trial's composition rule with top's reject dominance, over the harness's framing of the
core interface and mirrors of the stub components). Hypothesis draws each component's case to accept, reject, be
undecided or fail to decode, then often breaks the composition's own inputs (a component dropped, unknown, repeated or
out of order, a case under other params, a wrong or short commitment, non-canonical or malformed evidence) or the case
itself (a field missing or not hex, not an object), and every field of the outcome but its reason must agree exactly."""

from __future__ import annotations

import collections
import itertools
import json
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import reference as R
import verity_difftest as D
from verity_difftest import canonical

PROTOCOLS = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text(max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=3), inner, max_size=3),
                    max_leaves=6)


@st.composite
def stub_case(draw, cid: str) -> dict:
    """A stub's case that accepts, rejects, is undecided, or fails to decode (its params, root or evidence)."""
    kind = draw(st.sampled_from([R.ACCEPT] * 3 + [R.REJECT, R.UNDECIDED, "broken"]))
    k = draw(st.integers(0, 3))
    if kind == R.REJECT:
        drawn = draw(st.integers(1, 4))
        passed = draw(st.integers(0, drawn - 1))
    elif kind == R.UNDECIDED:
        k = draw(st.integers(1, 3))
        drawn = draw(st.integers(0, k - 1))
        passed = draw(st.integers(drawn, drawn + 1))
    else:
        drawn = draw(st.integers(k, k + 2))
        passed = draw(st.integers(drawn, drawn + 1))
    case = {"id": cid, "params": {"k": k}, "root": draw(st.binary(min_size=32, max_size=32)),
            "evidence": canonical({"drawn": drawn, "passed": passed}).encode()}
    if kind == "broken":
        how = draw(st.sampled_from(["params", "root", "evidence"]))
        if how == "params":
            case["params"] = draw(st.sampled_from([{}, {"k": -1}, {"k": True}, {"k": "2"}, [], None]) | JSON)
        elif how == "root":
            case["root"] = draw(st.binary(max_size=40).filter(lambda b: len(b) != 32))
        else:
            case["evidence"] = draw(st.sampled_from([b'{"passed":2,"drawn":2}', b'{"drawn": 2,"passed":2}',
                                                     canonical({"drawn": -1, "passed": 0}).encode(), b'{"drawn":2}'])
                                    | st.binary(max_size=12))
    return case


#: The binding breaks come thrice: with `commitment-size` they are the breaks that leave every component's case standing,
#: and so the way to a rejection that no component's own explains.
MUTATIONS = [None] * 10 + ["drop-param", "drop-case", "unknown-param", "unknown-case", "swap-params", "swap-cases",
                           "repeat-param", "repeat-case", "case-params", "commitment-size", "evidence-noncanonical",
                           "params-extra-key", "case-extra-key", "case-missing-key", "bad-hex", "garbage-evidence",
                           "garbage-params", "not-a-case"] + ["binding", "binding-id"] * 3


@st.composite
def composite(draw, product: R.Product) -> dict:
    """One case of `product`: its components' cases, then at most one break of the composition's own inputs."""
    cases = [draw(stub_case(C.id)) for C in product.components]
    named = [{"id": c["id"], "params": c["params"]} for c in cases]
    entries = [{"commitment": c["root"].hex(), "evidence": c["evidence"].hex(), "id": c["id"], "params": c["params"]}
               for c in cases]
    commitment = R.bind(product.id, [(c["id"], c["root"]) for c in cases])
    params: object = {"components": named}
    m = draw(st.sampled_from(MUTATIONS))
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
        commitment = R.bind("verity/other/v0", [(c["id"], c["root"]) for c in cases])
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
    evidence = canonical({"components": entries}).encode()
    if m == "evidence-noncanonical":
        evidence = json.dumps({"components": entries}).encode()
    elif m == "garbage-evidence":
        evidence = draw(st.binary(max_size=24))
    if m == "not-a-case":
        return draw(st.sampled_from([{"commitment": commitment.hex(), "params": params},
                                     {"commitment": "zz", "evidence": evidence.hex(), "params": params},
                                     {"commitment": commitment.hex().upper(), "evidence": evidence.hex(), "params": params},
                                     {"commitment": commitment.hex(), "evidence": 7, "params": params}, []]))
    return {"commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}


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
        return out
    return answer


def port(name: str, product: R.Product, seen: collections.Counter) -> D.ProtocolPort:
    """`name serve` of the Lake package `name`, on the product's cases and, one draw in ten, any JSON line."""
    return D.ProtocolPort(name, reference(product, seen), st.one_of(*[composite(product)] * 9, JSON),
                          package=PROTOCOLS / name)


SEEN = {"onsite": collections.Counter(), "remote": collections.Counter()}
ONSITE = port("onsite", R.ONSITE, SEEN["onsite"])
REMOTE = port("remote", R.REMOTE, SEEN["remote"])
onsite_server = D.server_fixture(ONSITE)
remote_server = D.server_fixture(REMOTE)


def test_the_rule_changes_the_trial_only_where_a_rejection_meets_an_undecided_component():
    """Over every list of up to four verdicts, top's rule and the trial's agree but where a component rejects while
    another is undecided: the trial waits for the missing statement, and the port rejects."""
    for n in range(1, 5):
        for vs in itertools.product([R.ACCEPT, R.REJECT, R.UNDECIDED], repeat=n):
            vs = list(vs)
            if R.REJECT in vs and R.UNDECIDED in vs:
                assert (R.trial_verdict(vs), R.verdict(vs)) == (R.UNDECIDED, R.REJECT), vs
            else:
                assert R.trial_verdict(vs) == R.verdict(vs), vs


def _agree(p: D.ProtocolPort, server, seen: collections.Counter, examples: int) -> None:
    seen.clear()
    stats = D.run(p, examples=examples, seed=0, server=server)
    print(f"{p.label}: {stats}, verdicts {dict(seen)}")
    assert stats.examples == examples and stats.counts == {v: seen[v] for v in (R.ACCEPT, R.REJECT, R.UNDECIDED)}
    for kind in (R.ACCEPT, R.REJECT, R.UNDECIDED, "reject, a component undecided", "reject, no component rejected"):
        assert seen[kind] > examples // 50, (kind, dict(seen))


def test_remote_agrees(remote_server):
    _agree(REMOTE, remote_server, SEEN["remote"], 1000)


def test_onsite_agrees(onsite_server):
    _agree(ONSITE, onsite_server, SEEN["onsite"], 1000)


class Trial(R.Product):
    """The trial's rule over the components' verdicts (`trial_verdict`): a component left undecided leaves the product
    undecided even when another rejects. Everything else is the port's."""

    def check(self, ps, c, ev) -> tuple:
        answers = self.run(ps, ev)
        if not isinstance(answers, str) and self.binds(answers, c) and R.trial_verdict(
                [a["verdict"] for _, a in answers]) == R.UNDECIDED:
            return R.UNDECIDED, "a component is undecided"
        return super().check(ps, c, ev)


def test_a_reference_without_reject_dominance_is_caught(onsite_server):
    """The difftest tells top's rule from the trial's: the case it shrinks to has a component that rejects and one that
    is undecided, under a commitment that binds their digests."""
    trial = Trial(R.ONSITE.id, R.ONSITE.components)
    with pytest.raises(D.Mismatch) as e:
        D.run(D.ProtocolPort("onsite", trial.answer, ONSITE.inputs, package=PROTOCOLS / "onsite"), examples=1000,
              server=onsite_server)
    assert (e.value.lean[0], e.value.python[0]) == (R.REJECT, R.UNDECIDED), (e.value.lean, e.value.python)
    parts = R.ONSITE.verdicts(e.value.input)
    assert R.REJECT in parts and R.UNDECIDED in parts, parts
