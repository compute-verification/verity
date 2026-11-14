"""`BoundaryWindows.check`: the check template a serve was given, or the one its callable gives when it is first read."""
from __future__ import annotations

import pytest
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.types import Array
from verity.circuits.types import Value as Bits

from verity_vllm.pipeline import boundary_windows as BW

XOR8 = PrimitiveDefinition("BoundaryWindowsCheckTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
#: a check template: a proof block of one gate's worth
PAIR = bind(CompositeDefinition("BoundaryWindowsCheckTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))


@pytest.fixture
def stopped():
    """Each serve's boundary commitment a test adds here is stopped when the test ends, its commit thread joined."""
    made: list[BW.BoundaryWindows] = []
    yield made
    for bw in made:
        bw.stop()


def test_a_serve_given_its_template_holds_it(stopped, tmp_path):
    stopped.append(bw := BW.BoundaryWindows(tmp_path, PAIR, 0, 1, 10.0, clock=lambda: 0.0, serve="s"))
    assert bw.check is PAIR and bw.check is PAIR


def test_a_serve_given_a_callable_asks_it_once_when_the_template_is_first_read(stopped, tmp_path):
    asked = []

    def give():
        asked.append("asked")
        return PAIR

    stopped.append(bw := BW.BoundaryWindows(tmp_path, give, 0, 1, 10.0, clock=lambda: 0.0, serve="s"))
    assert asked == []
    assert bw.check is PAIR and bw.check is PAIR and asked == ["asked"]


def test_a_serve_made_from_its_configuration_looks_its_template_up_by_its_id_when_it_is_first_read(monkeypatch, stopped, tmp_path):
    asked = []

    def check_template(t: str):
        asked.append(t)
        return PAIR

    monkeypatch.setattr(BW, "check_template", check_template)
    stopped.append(bw := BW.BoundaryWindows.of({"check": "a-template-id"}, tmp_path))
    assert asked == []
    assert bw.check is PAIR and bw.check is PAIR and asked == ["a-template-id"]
