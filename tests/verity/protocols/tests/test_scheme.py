"""The resource scheme interface and its registry (`verity.protocols.resource_accounting.scheme`)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from verity.protocols.resource_accounting import scheme as S


def _toy(name: str = "toy-bandwidth", **kw) -> S.Scheme:
    return S.Scheme(**{
        "name": name, "protocol": f"verity/{name}/v0", "resource": "read",
        "blocks": (S.Block("ToyReceipt_v1", lambda shape: 8 * shape["bytes"]),),
        "assumptions": ("ToyNoCompression",),
        "log": (S.Order("payload", "key"),
                S.Order("receipts", "seed", within_ns=1_000_000, assumption="ToyNoCompression")),
        "charges": (S.Charge("on-chip", "on_chip_bits"), S.Charge("rebuilt", "rebuilt_bits")),
        "settings": {"refresh": False}, "requires": {"coins-at-rate": "its own challenger"},
        "delta_scheme": Fraction(0), **kw})


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
    with pytest.raises(ValueError, match="only by an assumption"):
        S.Order("value roots", "seeds", "after", assumption="SAKMatmul")
    with pytest.raises(ValueError, match="positive time"):
        S.Order("value roots", "seeds", "after", within_ns=0, assumption="SAKMatmul")
    with pytest.raises(ValueError, match="not among its assumptions"):
        _toy(log=(S.Order("value roots", "seeds", "after", within_ns=30 * 10**9, assumption="SAKMatmul"),))


def test_order_direction() -> None:
    after = S.Order("value roots", "seeds", "after", within_ns=30 * 10**9, assumption="ToyNoCompression")
    assert _toy(log=(after,)).log[0].direction == "after"
    assert S.Order("start", "first-tick").direction == "before"
    with pytest.raises(ValueError, match="before or after"):
        S.Order("start", "first-tick", "beside")


def test_scheme_term_carried() -> None:
    s = _toy(delta_scheme=Fraction(1, 2**64))
    b = S.bound(s, Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0})
    assert b.delta_scheme == Fraction(1, 2**64) and b.probability == Fraction(1, 100) + Fraction(1, 2**64)
    assert S.bound(_toy(), Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0}).delta_scheme == 0
    with pytest.raises(ValueError, match="δscheme lies"):
        _toy(delta_scheme=Fraction(1))


def test_scheme_term_not_said_refuses_a_bound() -> None:
    s = _toy("silent", delta_scheme=None)
    assert S.register(s) is S.get("silent")
    with pytest.raises(ValueError, match=r"silent: its delta_scheme \(Verity.Scheme.δscheme\) is not said"):
        S.bound(s, Fraction(1, 100), 0, {"on-chip": 0, "rebuilt": 0})


def _dither_circuit() -> list[tuple[str, list[list[int]]]]:
    from verity.circuits.defs import bind
    from verity.commitments.periods import interface as I
    from verity.protocols.experimental.dither.prover import dither as D
    from verity.protocols.proofs.spot.verifier.one_stage import partition as P

    program, query = I.Circuit("toy", ((bind(D.PouwDitherBlock, L=32, FIRST=True), 3),
                                       (bind(D.PouwDitherLast, K=64, L=64), 1))).program()
    return P.strata(program, query)


def test_credited_resolves_a_declaration_in_a_registered_circuit() -> None:
    from verity.protocols.experimental.dither.scheme import DITHER

    strata = _dither_circuit()
    assert [t for t, _ in strata] == ["PouwDitherBlock_v1{L=32,FIRST=true}", "PouwDitherLast_v1{K=64,L=64}"]
    assert S.credited([DITHER], strata) == frozenset({0, 1, 2, 3})
    assert S.credited([], strata) == frozenset()
    with pytest.raises(ValueError, match="toy-bandwidth: its block ToyReceipt_v1 resolves to no block"):
        S.credited([DITHER, _toy()], strata)
    rival = _toy("rival", blocks=(S.Block("PouwDitherLast_v1", lambda shape: 0),))
    with pytest.raises(ValueError, match=r"PouwDitherLast_v1\{K=64,L=64\} would be credited twice"):
        S.credited([DITHER, rival], strata)


def test_unbounded_charge_refuses_a_bound() -> None:
    s = _toy(charges=(S.Charge("on-chip", "on_chip_bits"), S.Charge("padding", "Network.PaddingBound", bounded=False)))
    with pytest.raises(ValueError, match="nothing bounds yet: padding"):
        S.bound(s, Fraction(1, 100), 0, {"on-chip": 0, "padding": 0})


def test_ports() -> None:
    port = S.Port("scan state", "input", "ToyReceipt_v1", "state")
    assert _toy(ports=(port,)).ports == (port,)
    with pytest.raises(ValueError, match="not its blocks' Definitions"):
        _toy(ports=(S.Port("scan state", "input", "IngressScanBucket_v1", "state"),))


def test_every_declaration_is_on_the_one_list() -> None:
    from verity.protocols.compose import schemes

    root = Path(__file__).resolve().parents[4]
    declaring = {str(p.relative_to(root).with_suffix("")).replace("/", ".")
                 for d in ("verity/protocols/resource_accounting", "verity/protocols/experimental")
                 for p in (root / d).rglob("scheme.py") if "register(" in p.read_text()}
    declaring.discard("verity.protocols.resource_accounting.scheme")
    assert declaring == set(schemes.DECLARATIONS), "add each declaration module to compose.schemes.DECLARATIONS"


def test_a_scheme_attaches_nothing_by_default_and_registers_its_processes(monkeypatch) -> None:
    monkeypatch.setattr(S, "PROCESSES", {})
    assert _toy().requests(None) == () and _toy().attach(None, None) is None
    start = lambda **readers: readers  # noqa: E731
    monkeypatch.setattr(S, "SCHEMES", {})
    S.register(_toy(processes={"toy-beside": start}))
    assert S.PROCESSES == {"toy-beside": start}
