"""`answer`, the Python `Core.Answer`: its JSON is `Answer.toJson`'s, `of_json` reads back exactly what it writes in the
shapes `Protocol.answer` returns and refuses everything else, as `Answer.ofJson` does. The difftest suite holds these
bytes to the Lean's (`verity/core/difftest/tests/test_protocol_port.py`)."""

from __future__ import annotations

import pytest

from verity.core.protocols import answer as A

ID = "verity/example/v0"
ROOT = bytes(range(32))
PROFILE = {"covered": [], "delta": "1/1000000", "drawn": {"unit": 2},
           "levels": [{"draw": {"subset": 2}, "name": "unit", "query": None, "size": 2}], "linked": True,
           "prescribed": [], "programs": [ROOT.hex()]}
ANSWERS = [A.Answer.accept(ID, ROOT, {"passed": 2}, PROFILE), A.Answer.reject(ID, "1 of 2 drawn units failed", ROOT),
           A.Answer.undecided(ID, "1 of the 2 units drawn", ROOT), A.Answer.reject(ID, "params: k missing")]


@pytest.mark.parametrize("a", ANSWERS, ids=lambda a: f"{a.verdict}-{a.digest is not None}")
def test_an_answer_reads_back_from_its_json(a):
    j = a.to_json()
    assert tuple(j) == A.KEYS and j["verdict"] == a.verdict.value
    assert A.of_json(j) == a
    assert A.of_json(A.canonical_json(a.canonical().encode())) == a


def test_the_canonical_form_is_the_lean_one():
    a = A.Answer.undecided(ID, "é \"q\"", ROOT)
    assert a.canonical() == ('{"commitment":"' + ROOT.hex() + '","profile":null,"protocol":"verity/example/v0",'
                             '"public":null,"reason":"\\u00e9 \\"q\\"","verdict":"undecided"}')
    assert A.outcome(ID, "undecided", "é \"q\"", ROOT) == a.to_json()


@pytest.mark.parametrize(("edit", "why"), [
    (lambda j: {k: v for k, v in j.items() if k != "profile"}, "answer: profile missing"),
    (lambda j: j | {"extra": 1}, "answer: not in the form Answer.toJson writes"),
    (lambda j: j | {"verdict": "pending"}, "answer.verdict: not accept, reject or undecided"),
    (lambda j: j | {"verdict": 0}, "answer.verdict: not a string"),
    (lambda j: j | {"commitment": "AB"}, "not lowercase hex"),
    (lambda j: j | {"commitment": "abc"}, "odd-length hex"),
    (lambda j: j | {"reason": "why"}, "answer: an accept has no reason, and both its public outputs and its profile"),
    (lambda j: j | {"public": None}, "answer: an accept has no reason, and both its public outputs and its profile"),
    (lambda j: j | {"verdict": "reject"}, "answer: a reject has a reason, and neither public outputs nor a profile"),
    (lambda j: j | {"verdict": "undecided", "reason": "r"},
     "answer: a undecided has a reason, and neither public outputs nor a profile"),
])
def test_of_json_refuses_what_to_json_never_writes(edit, why):
    with pytest.raises(A.Refused) as e:
        A.of_json(edit(ANSWERS[0].to_json()))
    assert str(e.value) == why


def test_an_accept_has_its_public_outputs_and_profile():
    with pytest.raises(ValueError):
        A.Answer.accept(ID, ROOT, None, PROFILE)


@pytest.mark.parametrize("b", [b'{"b":1,"a":2}', b'{"a": 1}', b'{"a":1.5}', b"\xff", b"{", '"\u00e9"'.encode()])
def test_canonical_json_refuses_any_other_spelling(b):
    with pytest.raises(A.Refused):
        A.canonical_json(b)
