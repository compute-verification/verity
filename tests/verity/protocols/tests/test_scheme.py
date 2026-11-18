"""The resource scheme interface and its registry (`verity.protocols.resource_accounting.scheme`)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from verity.protocols.resource_accounting import scheme as S


def _toy(name: str = "toy-bandwidth", **kw) -> S.Scheme:
    return S.Scheme(
        name=name, protocol=f"verity/{name}/v0", resource="read",
        blocks=(S.Block("ToyReceipt_v1", lambda shape: 8 * shape["bytes"]),),
        assumptions=("ToyNoCompression",),
        log=(S.Order("payload", "key"), S.Order("receipts", "seed", within_ns=1_000_000, assumption="ToyNoCompression")),
        charges=(S.Charge("on-chip", "on_chip_bits"), S.Charge("rebuilt", "rebuilt_bits")),
        settings={"refresh": False}, requires={"coins-at-rate": "its own challenger"}, **kw)


@pytest.fixture(autouse=True)
def _empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(S, "SCHEMES", {})


def test_picked_by_name() -> None:
    a, b = S.register(_toy("pearl-plus")), S.register(_toy("dither"))
    assert S.get("pearl-plus") is a and S.get("dither") is b
    with pytest.raises(KeyError, match="registered: dither, pearl-plus"):
        S.get("pouw")
    with pytest.raises(KeyError, match="registered already"):
        S.register(_toy("dither"))


def test_settings_toggle() -> None:
    s = _toy()
    assert S.settings(s, {}) == {"refresh": False}
    assert S.settings(s, {"refresh": True}) == {"refresh": True}
    with pytest.raises(ValueError, match="no setting"):
        S.settings(s, {"lifetime": 3})
    with pytest.raises(ValueError, match="is a bool"):
        S.settings(s, {"refresh": 1})


def test_block_cost_and_unit() -> None:
    s = _toy()
    assert s.unit == "bits" and s.blocks[0].cost({"bytes": 4096}) == 32_768


def test_bound() -> None:
    s = _toy()
    b = S.bound(s, Fraction(1, 100), 65_536, {"on-chip": 10, "rebuilt": 6})
    assert b.uncredited == 65_552
    with pytest.raises(ValueError, match="charges are"):
        S.bound(s, Fraction(1, 100), 0, {"on-chip": 0})
    with pytest.raises(ValueError, match="δ lies"):
        S.bound(s, Fraction(1), 0, {"on-chip": 0, "rebuilt": 0})


@pytest.mark.parametrize("bad, match", [
    ({"name": "PoUB"}, "configuration's name"),
    ({"resource": "bandwidth"}, "resource is one of"),
    ({"requires": {"fiat-shamir": "x"}}, "not requirements"),
    ({"charges": (S.Charge("on-chip", "a"), S.Charge("on-chip", "b"))}, "charge twice"),
])
def test_refusals(bad: dict, match: str) -> None:
    base = {"name": "toy", "protocol": "verity/toy/v0", "resource": "read", "blocks": (), "assumptions": (), "log": (),
            "charges": ()}
    with pytest.raises(ValueError, match=match):
        S.Scheme(**{**base, **bad})


def test_hooks() -> None:
    assert _toy(hooks=("memory", "declared")).hooks == ("memory", "declared")
    with pytest.raises(ValueError, match="not hooks of a served run"):
        _toy(hooks=("memory", "pouw-gemm"))


def test_time_only_through_an_assumption() -> None:
    with pytest.raises(ValueError, match="only by an assumption"):
        S.Order("receipts", "seed", within_ns=1)
