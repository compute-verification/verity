"""The harness against its example port (`lean/DifftestExample.lean`, served by `Difftest.serve`): the right references agree
on every drawn input, a wrong one is caught and shrunk to a minimal input, and a reject on one side only is caught."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D

LEAN = Path(__file__).resolve().parents[1] / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

JSON = st.recursive(st.none() | st.booleans() | st.integers() | st.text(),
                    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=4), inner, max_size=4),
                    max_leaves=12)
#: Valid pairs, pairs with b = 0, pairs of anything, objects missing a key, and any JSON at all.
ARGS = st.one_of(st.fixed_dictionaries({"a": st.integers(), "b": st.integers()}),
                 st.fixed_dictionaries({"a": st.integers(), "b": st.just(0)}),
                 st.fixed_dictionaries({"a": JSON, "b": JSON}),
                 st.dictionaries(st.sampled_from("ab"), st.integers(), max_size=1),
                 JSON)


def _int(v, k: str) -> int:
    if not isinstance(v, dict) or not isinstance(v.get(k), int) or isinstance(v[k], bool):
        raise D.Reject(f"{k}: not an integer")
    return v[k]


def add(v) -> int:
    return _int(v, "a") + _int(v, "b")


def div(v) -> int:
    """Lean's `Int` division: the remainder is non-negative, so a negative divisor rounds up."""
    a, b = _int(v, "a"), _int(v, "b")
    if b == 0:
        raise D.Reject("division by zero")
    return a // b if b > 0 else -(a // -b)


def floor_div(v) -> int:
    a, b = _int(v, "a"), _int(v, "b")
    if b == 0:
        raise D.Reject("division by zero")
    return a // b


def total_div(v) -> int:
    _, b = _int(v, "a"), _int(v, "b")
    return 0 if b == 0 else div(v)


def port(fn, reference, inputs=ARGS, **kw) -> D.Port:
    return D.Port("difftest-example", fn, reference, inputs, package=LEAN, **kw)


server = D.server_fixture(port("add", add))


@pytest.mark.parametrize("fn, reference", [("add", add), ("div", div)])
def test_the_example_port_agrees(server, fn, reference):
    stats = D.run(port(fn, reference), examples=300, server=server)
    assert stats.examples == 300 and stats.outputs > 50 and stats.rejects > 50, stats


def test_echo_holds_the_wire_format_to_python_s_canonical_json(server):
    """`echo` returns what `Json.parse` read, as `Flock.canon` writes it: strings with control characters and non-ASCII,
    integers past 64 bits, nested objects with keys out of order."""
    stats = D.run(port("echo", lambda v: v, JSON, need_rejects=False), examples=300, server=server)
    assert stats.outputs == 300
    assert D.replay(port("echo", lambda v: v), {"z": "é\n\x01\x7f\u2028", "a": [-(2**100), True, None]}, server) == (
        "output", '{"a":[-1267650600228229401496703205376,true,null],"z":"é\\n\\u0001\x7f\u2028"}')


def test_a_wrong_reference_is_caught_and_shrunk(server):
    """Python's floor division rounds a negative divisor down where Lean's rounds up; Hypothesis shrinks to 1 / -2."""
    with pytest.raises(D.Mismatch) as e:
        D.run(port("div", floor_div), examples=300, server=server)
    assert e.value.input == {"a": 1, "b": -2}
    assert (e.value.lean, e.value.python) == (("output", "0"), ("output", "-1"))
    report = str(e.value)
    assert 'input:  {"a":1,"b":-2}' in report and "run(port, examples=300, seed=0)" in report, report


def test_with_lean_of_record_python_is_held_to_it(server):
    """Under `of_record="lean"` the same disagreement fails with the Lean's verdict as the one Python is held to; a side
    that is neither Lean nor Python is refused."""
    with pytest.raises(D.Mismatch) as e:
        D.run(port("div", floor_div, of_record="lean"), examples=300, server=server)
    assert e.value.input == {"a": 1, "b": -2} and e.value.of_record == "lean"
    assert str(e.value).startswith("difftest of 'div'") and "Python differs from the Lean, the verifier of record" in str(e.value)
    assert '  lean:   output 0  (of record)\n  python: output -1' in str(e.value)
    assert D.Mismatch("div", {}, ("output", "0"), ("output", "-1"), "lean").args[0] == (
        "div: Python output -1 where Lean, of record, gives output 0, on {}")
    stats = D.run(port("div", div, of_record="lean"), examples=100, server=server)
    assert stats.outputs and stats.rejects
    with pytest.raises(ValueError, match="of_record is 'rust'"):
        port("div", div, of_record="rust")


def test_the_run_is_deterministic(server):
    found = []
    for _ in range(2):
        with pytest.raises(D.Mismatch) as e:
            D.run(port("div", floor_div, st.fixed_dictionaries({"a": st.integers(), "b": st.integers(-9, -2)})),
                  examples=50, seed=7, server=server)
        found.append(e.value.input)
    assert found[0] == found[1]


def test_reject_parity_is_required(server):
    """Lean rejects b = 0 and this reference doesn't; and the other way round, a reference that rejects what Lean adds."""
    with pytest.raises(D.Mismatch) as e:
        D.run(port("div", total_div), examples=300, server=server)
    assert e.value.input == {"a": 0, "b": 0}
    assert (e.value.lean, e.value.python) == (("reject", "division by zero"), ("output", "0"))

    def picky_add(v):
        if _int(v, "a") < 0:
            raise D.Reject("a is negative")
        return add(v)

    with pytest.raises(D.Mismatch) as e:
        D.run(port("add", picky_add), examples=300, server=server)
    assert e.value.input == {"a": -1, "b": 0} and e.value.lean == ("output", "-1") and e.value.python[0] == "reject"


def test_reasons_need_not_match(server):
    def div_other_reasons(v):
        try:
            return div(v)
        except D.Reject as r:
            raise D.Reject(f"python says: {r}") from None

    assert D.run(port("div", div_other_reasons), examples=100, server=server).rejects > 0


def test_an_undeclared_exception_is_a_difference_not_a_reject(server):
    def keyed_add(v):
        if not isinstance(v, dict):
            raise D.Reject("not an object")
        return v["a"] + v["b"]

    with pytest.raises(D.Mismatch) as e:
        D.run(port("add", keyed_add), examples=300, server=server)
    assert e.value.input == {} and e.value.python[0] == "error" and "KeyError" in e.value.python[1]
    assert e.value.lean == ("reject", "a: not an integer")


def test_a_port_may_declare_its_own_rejects(server):
    def value_error_div(v):
        try:
            return div(v)
        except D.Reject as r:
            raise ValueError(str(r)) from None

    assert D.run(port("div", value_error_div, rejects=(ValueError,)), examples=100, server=server).rejects > 0


def test_an_unknown_fn_is_a_protocol_error_not_a_reject(server):
    with pytest.raises(D.Mismatch) as e:
        D.run(port("nope", add), examples=20, server=server)
    assert e.value.lean[0] == "error" and "unknown fn: nope" in e.value.lean[1]


def test_a_vacuous_run_fails(server):
    with pytest.raises(D.Vacuous, match="none of 50 inputs was rejected"):
        D.run(port("add", add, st.fixed_dictionaries({"a": st.integers(), "b": st.integers()})), examples=50, server=server)
    with pytest.raises(D.Vacuous, match="gave .output."):
        D.run(port("add", add, st.just({})), examples=5, server=server)
