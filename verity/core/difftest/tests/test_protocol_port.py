"""The harness in protocol mode, against the core protocol interface's toy protocol (`CoreProtocol.Example`, its
`core-protocol-example serve`): its Python reference agrees on every drawn case, valid and malformed, and a reference wrong
in the verdict, the commitment or the profile is caught and shrunk, while one wrong only in the reason passes."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity_difftest import protocol as P

INTERFACE = Path(__file__).resolve().parents[2] / "protocols" / "interface"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ID = "verity/example/v0"


def _root(b: bytes) -> bytes:
    if len(b) != 32:
        raise D.Reject(f"a root is 32 bytes, not {len(b)}")
    return b


def _evidence(b: bytes) -> tuple[int, int]:
    j = P.canonical_json(b)
    return P.nat(j, "drawn"), P.nat(j, "passed")


def _check(k: int, root: bytes, e: tuple[int, int], delta: str = "1/1000000") -> tuple:
    drawn, passed = e
    if passed < drawn:
        return "reject", f"{drawn - passed} of {drawn} drawn units failed"
    if drawn < k:
        return "undecided", f"{drawn} of the {k} units drawn"
    profile = {"covered": [], "delta": delta, "drawn": {"unit": drawn},
               "levels": [{"draw": {"subset": k}, "name": "unit", "query": None, "size": drawn}],
               "linked": True, "prescribed": [], "programs": [root.hex()]}
    return "accept", {"passed": passed}, profile


def example(case, check=_check, commitment=_root) -> dict:
    """`CoreProtocol.Example` in Python: parameters `{"k"}`, a 32-byte root that is its own digest, and evidence
    `{"drawn", "passed"}` in canonical JSON."""
    return P.answer(ID, case, params=lambda j: P.nat(j, "k"), commitment=commitment, evidence=_evidence, check=check)


SMALL = st.integers(0, 4)
JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text(max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=3), inner, max_size=3),
                    max_leaves=6)
#: Evidence that doesn't decode: keys out of order, not compact, not naturals, a key missing, a float, not UTF-8, bytes.
BAD_EVIDENCE = st.one_of(
    st.builds(lambda d, p: f'{{"passed":{p},"drawn":{d}}}'.encode(), SMALL, SMALL),
    st.builds(lambda d, p: f'{{"drawn": {d}, "passed": {p}}}'.encode(), SMALL, SMALL),
    st.builds(lambda d, p: D.canonical({"drawn": d, "passed": p}).encode(), JSON, JSON),
    st.dictionaries(st.sampled_from(["drawn", "passed"]), SMALL, max_size=1).map(lambda d: D.canonical(d).encode()),
    st.just(b'{"drawn":1.0,"passed":1}'), st.just(b"\xff\xfe"), st.binary(max_size=8))
VALID = st.fixed_dictionaries({
    "commitment": st.binary(min_size=32, max_size=32).map(bytes.hex),
    "evidence": st.one_of(st.builds(lambda d, p: {"drawn": d, "passed": p}, SMALL, SMALL),
                          st.builds(lambda d, p, x: {"drawn": d, "passed": p, "x": x}, SMALL, SMALL, JSON))
                  .map(lambda e: D.canonical(e).encode().hex()),
    "params": st.fixed_dictionaries({"k": SMALL})})


def _upper(field):
    return lambda case, _: case | {field: case[field].upper()}


#: One change to a valid case: hex `Flock.unhex` refuses (uppercase, a space, odd length) at the size that would decode,
#: a short root, a field missing or not a string, evidence or parameters that don't decode.
MUTATIONS = [
    _upper("commitment"), _upper("evidence"),
    lambda case, _: case | {"commitment": case["commitment"][:2] + " " + case["commitment"][2:]},
    lambda case, _: case | {"evidence": case["evidence"] + "0"},
    lambda case, x: case | {"commitment": case["commitment"][: 2 * (x % 32)]},
    lambda case, x: {k: v for k, v in case.items() if k != ["commitment", "evidence", "params"][x % 3]},
    lambda case, x: case | {["commitment", "evidence"][x % 2]: x},
    lambda case, x: case | {"params": {"k": -x - 1}},
    lambda case, x: case | {"params": [x]},
]
MUTATED = st.builds(lambda case, m, x, bad, params: case | {"evidence": bad.hex()} if m == len(MUTATIONS) else
                    case | {"params": params} if m == len(MUTATIONS) + 1 else MUTATIONS[m](case, x),
                    VALID, st.integers(0, len(MUTATIONS) + 1), st.integers(0, 63), BAD_EVIDENCE, JSON)
CASES = st.one_of(VALID, VALID, MUTATED, JSON)


def port(reference=example, inputs=CASES) -> D.ProtocolPort:
    return D.ProtocolPort("core-protocol-example", reference, inputs, package=INTERFACE)


server = D.server_fixture(port())


def test_the_example_protocol_agrees(server):
    stats = D.run(port(), examples=500, server=server)
    assert stats.examples == 500 and min(stats.counts.get(v, 0) for v in ("accept", "reject", "undecided")) > 15, stats


def test_the_outcome_is_compared_field_by_field(server):
    root = bytes(range(32))
    case = {"commitment": root.hex(), "evidence": b'{"drawn":2,"passed":2}'.hex(), "params": {"k": 2}}
    kind, line = D.replay(port(), case, server)
    assert kind == "accept" and D.canonical(example(case)) == line


def test_the_reason_is_not_compared(server):
    def other_reasons(case):
        out = example(case)
        return out | {"reason": None if out["reason"] is None else "python's own words"}

    assert D.run(port(other_reasons), examples=200, server=server).counts["undecided"] > 0


def test_a_lenient_hex_decoder_is_caught(server):
    """A decoder that takes uppercase hex, as `bytes.fromhex` does, where `Flock.unhex` refuses it."""
    def lenient(case):
        if isinstance(case, dict) and isinstance(case.get("commitment"), str) and isinstance(case.get("evidence"), str):
            case = case | {"commitment": case["commitment"].lower(), "evidence": case["evidence"].lower()}
        return example(case)

    with pytest.raises(D.Mismatch) as e:
        D.run(port(lenient), examples=500, server=server)
    assert e.value.lean[0] == "reject" and '"commitment":null' in e.value.lean[1]
    assert any(c.isupper() or c == " " for c in e.value.input["commitment"] + e.value.input["evidence"]), e.value.input


def test_undecided_is_not_reject(server):
    """Verdict agreement is the reject/undecided parity."""
    def undecided_as_reject(k, root, e):
        out = _check(k, root, e)
        return ("reject", out[1]) if out[0] == "undecided" else out

    with pytest.raises(D.Mismatch) as e:
        D.run(port(lambda c: example(c, undecided_as_reject)), examples=500, server=server)
    assert (e.value.lean[0], e.value.python[0]) == ("undecided", "reject")
    assert e.value.input["params"] == {"k": 1} and e.value.input["evidence"] == b'{"drawn":0,"passed":0}'.hex()


def test_a_wrong_profile_is_caught(server):
    with pytest.raises(D.Mismatch) as e:
        D.run(port(lambda c: example(c, lambda k, r, ev: _check(k, r, ev, delta="1/1000"))), examples=500, server=server)
    assert e.value.lean[0] == e.value.python[0] == "accept"
    assert '"delta":"1/1000000"' in e.value.lean[1] and '"delta":"1/1000"' in e.value.python[1]


def test_a_wrong_commitment_is_caught(server):
    """A reference that decodes the commitment before the parameters reports a digest Lean doesn't."""
    def commitment_first(case):
        out = example(case)
        if out["reason"] and out["reason"].startswith("params: ") and len(P.unhex(case["commitment"])) == 32:
            return out | {"commitment": case["commitment"]}
        return out

    with pytest.raises(D.Mismatch) as e:
        D.run(port(commitment_first), examples=500, server=server)
    assert '"commitment":null' in e.value.lean[1] and e.value.python[0] == "reject"


def test_an_outcome_missing_a_field_is_an_error(server):
    def no_profile(case):
        out = example(case)
        del out["profile"]
        return out

    with pytest.raises(D.Mismatch) as e:
        D.run(port(no_profile), examples=20, server=server)
    assert e.value.python[0] == "error" and "fields" in e.value.python[1]
