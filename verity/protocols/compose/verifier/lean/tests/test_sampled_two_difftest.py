"""SPOT's two-stage form's Lean (`sampled-two serve`, `VStar.Recursive.SampledTwo.protocol`) against its Python twin
(`sampled_two.py`) on parameters each refused at one step of the parse: the form's own steps, whose reasons must agree
word for word, and the decoders it hands its parts to, whose outcomes must agree but for the reason. The parameters are
the one-stage difftest's two-stage run's (`_two_stage_case`) in the form's shape, both registrations naming one challenger
key, each case one change to them. The twin decides no case past the interior's template circuits, so the form's
refusal of every audit while `FORM` is false is its Lean's (`checkBlobs_refuses`), not this test's."""

from __future__ import annotations

import copy
import functools
import json
import random
import shutil
import sys

import pytest
from hypothesis import strategies as st

import reference as R
import sampled_two as T
import verity_difftest as D
from verity_difftest import canonical

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

OS = R.ONE_STAGE
PORT = D.ProtocolPort("sampled-two", T.answer, st.nothing(), package=R.ROOT / "verity/protocols/proofs/recursion/verifier/recursive")
server = D.server_fixture(PORT)
DROP = object()


@functools.cache
def _key(seed: int) -> dict:
    """The challenger key of `seed`, at the ledger's parameters."""
    return sys.modules["test_challenger_difftest"]._ledger_session(random.Random(f"two-stage-form/{seed}"), seed=seed).key.to_json()


def _challenged(p: dict, seed: int = 0) -> dict:
    return {**p, "challenger": {"key": _key(seed), "label": "draw"}}


@functools.cache
def _base_text() -> str:
    c = OS._two_stage_case(0)
    it = c["interior"]
    return canonical({
        "form": T.TWO_STAGE, "boundary": _challenged(c["params"]), "challenger": c["challenger"], "delta": c["delta"],
        "interior": {"members": {}, "params": _challenged(it["params"]), "replay": it["replay"], "values": it["values"]},
        "outputs": [], "public": c["public"], "rule": c["rule"], "values": c["values"]})


def base() -> dict:
    """The form's parameters on the two-stage run, which the twin parses through the template circuits."""
    return json.loads(_base_text())


def _set(p: dict, path: tuple, v) -> dict:
    p = copy.deepcopy(p)
    *up, last = path
    at = p
    for k in up:
        at = at[k]
    if v is DROP:
        del at[last]
    else:
        at[last] = v
    return p


def _two_strata() -> dict:
    """A boundary over a setup whose proof blocks under its query are more than one stratum."""
    s = next(s for s in OS.SETUPS if len(OS.DR.strata_sizes(s.program, s.query)) > 1)
    b = base()["boundary"]
    return {**b, "law": {"k": 1, "law": "subset"}, "ports": s.ports, "program": s.desc, "query": s.query}


#: Each case's change to the base parameters.
CHANGES = {
    "not-an-object": lambda p: [],
    "no-form": lambda p: _set(p, ("form",), DROP),
    "form-not-a-string": lambda p: _set(p, ("form",), 2),
    "form-one-stage": lambda p: _set(p, ("form",), "one-stage"),
    "a-key-missing": lambda p: _set(p, ("delta",), DROP),
    "a-key-more": lambda p: _set(p, ("more",), 1),
    "outputs-not-a-list": lambda p: _set(p, ("outputs",), "credited"),
    "outputs": lambda p: _set(p, ("outputs",), ["credited"]),
    "rule-all": lambda p: _set(p, ("rule",), {"rule": "all"}),
    "rule-law": lambda p: _set(p, ("rule",), {"law": "rate:1/2", "rule": "law"}),
    **{f"rule-odd-{n}": (lambda r: lambda p: _set(p, ("rule",), r))(r) for n, r in enumerate(OS.ODD_RULES)},
    "challenger-not-a-string": lambda p: _set(p, ("challenger",), 3),
    "public-not-a-list": lambda p: _set(p, ("public",), "x"),
    "public-item": lambda p: _set(p, ("public",), [1]),
    "boundary-empty": lambda p: _set(p, ("boundary",), {}),
    "boundary-no-window": lambda p: _set(p, ("boundary", "window_root"), DROP),
    "boundary-two-strata": lambda p: _set(p, ("boundary",), _two_strata()),
    "interior-not-an-object": lambda p: _set(p, ("interior",), "x"),
    "interior-a-key-missing": lambda p: _set(p, ("interior", "replay"), DROP),
    "boundary-no-challenger": lambda p: _set(p, ("boundary", "challenger"), DROP),
    "interior-no-challenger": lambda p: _set(p, ("interior", "params", "challenger"), DROP),
    "interior-other-key": lambda p: _set(p, ("interior", "params", "challenger", "key"), _key(1)),
    "interior-params-empty": lambda p: _set(p, ("interior", "params"), {"challenger": p["interior"]["params"]["challenger"]}),
    "interior-no-window": lambda p: _set(p, ("interior", "params", "window_root"), DROP),
    "interior-proofs": lambda p: _set(p, ("interior", "params", "proofs"), {"templates": "x"}),
}


def case(params) -> dict:
    return {"commitment": "7b7d", "evidence": "7b7d", "params": params}


def test_base_parses_through_the_mirror() -> None:
    with pytest.raises(ValueError, match="outside the mirror"):
        T.params_of_json(base())


def test_the_form_s_own_refusals_are_reached() -> None:
    """Every refusal the form makes itself is some case's: a case list that missed one would hold nothing of it."""
    refusals = [T.refusal(change(base())) for change in CHANGES.values()]
    assert all(isinstance(e, D.Reject) for e in refusals), refusals
    reasons = {str(e) for e in refusals if not isinstance(e, T.Delegated)}
    for own in (T.ONE_DEFINITION, T.ONE_KEY, T.OUTPUTS_REFUSED, "the rule is not two-stage"):
        assert own in reasons


@pytest.mark.parametrize("name", list(CHANGES))
def test_refusal(server, name: str) -> None:
    c = case(CHANGES[name](base()))
    lean, py = PORT.lean(server, c), PORT.python(c)
    assert py[0] == "reject", py
    if T.reason_mirrored(c["params"]):
        assert lean == py
    else:
        assert PORT.key(lean) == PORT.key(py)


def test_two_definitions_are_refused_by_name(server) -> None:
    """A boundary population of replay blocks under two Definitions is refused by `ONE_DEFINITION`, in the Lean and the
    twin alike: the theorem counts replay blocks, a share by weight only where they are one Definition's."""
    c = case(CHANGES["boundary-two-strata"](base()))
    lean, py = PORT.lean(server, c), PORT.python(c)
    assert lean[0] == py[0] == "reject", (lean, py)
    assert json.loads(lean[1])["reason"] == json.loads(py[1])["reason"], (lean, py)
    assert T.ONE_DEFINITION in json.loads(lean[1])["reason"], lean


def test_the_base_is_refused(server) -> None:
    """Past the mirror the Lean still refuses: the members are none, and the form accepts nothing while `FORM` is false."""
    assert PORT.lean(server, case(base()))[0] == "reject"
