"""Lean's integer form of the cap circuit's tests (`security/Proofs/Pouw/PearlC/InertBound/IntForm.lean`) against the scheme:
its `IntPrices` are the devices' prices, and each kernel-checked pin in `InertBound/Pins.lean` is what Python computes at that
shape: `creditInt` is `PearlC(forming="v1").credit_of`, and `inertBoundPin` / `inertBoundPinTop` are `inert_bound._inert_bound_constants` at the device's
g, prices and cap (that test skips where `verity.protocols.accounting.pouw.prover.circuit.inert_bound` is absent)."""
from __future__ import annotations

import ast
import re
from fractions import Fraction
from pathlib import Path

import pytest

from verity.protocols.accounting.pouw.prover.schemes import pearl_c as C
from verity.protocols.accounting.pouw.prover.schemes.pearl_c_device import DEVICES
from verity.protocols.accounting.pouw.verifier.protocol import DomainError, Shape

CAP = Path(__file__).resolve().parents[6] / "security" / "Proofs" / "Pouw" / "PearlC" / "InertBound"
INT_FORM = (CAP / "IntForm.lean").read_text()
PINS = (CAP / "Pins.lean").read_text()
SHAPE = r"IntPrices\.(\w+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"


def _theorems(prefix: str) -> dict[str, str]:
    """Each `theorem <prefix>…` of Pins.lean: its name and its statement, up to `:= by decide`."""
    out = {m[1]: m[2] for m in re.finditer(rf"theorem ({prefix}\w+) :(.*?):= by decide", PINS, re.S)}
    assert out, prefix
    return out


def _shape(m: re.Match) -> tuple:
    return (m[1], *(int(x) for x in m.groups()[1:7]))


def test_int_prices_are_the_devices():
    """`IntPrices.<dev>` is `DEVICES[dev].prices`: `fadd`, `bf16_mac`, `forming_credited`, `forming_a_only`."""
    lean = {m[1]: tuple(int(x) for x in m[2].split(","))
            for m in re.finditer(r"def IntPrices\.(\w+) : IntPrices := ⟨([\d, ]+)⟩", INT_FORM)}
    assert set(lean) == {"sm120", "h100"}
    for dev, prices in lean.items():
        p = DEVICES[dev].prices
        assert prices == (p.fadd, p.bf16_mac, p.forming_credited, p.forming_a_only), dev


def test_credit_pins_are_credit_of():
    """Each `creditInt IntPrices.<dev> g m k n = v` pin is `credit_of(Shape(m, k, n))` on the device, at its g."""
    for name, stmt in _theorems("credit_").items():
        m = re.fullmatch(r"\s*creditInt IntPrices\.(\w+) (\d+) (\d+) (\d+) (\d+) = (\d+)\s*", stmt)
        assert m, name
        dev, g, rows, k, n, value = m[1], *(int(x) for x in m.groups()[1:])
        d = DEVICES[dev]
        assert g == d.g, name
        assert C.PearlC(forming="v1", device=d, hashing="h0").credit_of(Shape(rows, k, n)) == value, name


def test_inert_bound_pins_are_inert_bound_constants():
    """Each `inertBoundPin` pin is `_inert_bound_constants(k, tm, tn, n, rho, dev)` (`inertBoundPinTop`: with the table's last entry alone), on the
    device's g and at ρ = 1/rho its cap, with `InertBoundDomain` and `InertBoundFits` at the same shape; each refused shape raises."""
    inert_bound = pytest.importorskip("verity.protocols.accounting.pouw.prover.circuit.inert_bound", reason="the cap circuit is `verity.protocols.accounting.pouw.prover.circuit.inert_bound`'s")
    for name, stmt in _theorems("pin_").items():
        domain = re.search(r"InertBoundDomain (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)", stmt)
        fits = re.search(rf"InertBoundFits {SHAPE}", stmt)
        pin = re.search(rf"inertBoundPin(Top)? {SHAPE} = \((.*)\)\s*$", stmt, re.S)
        assert domain and fits and pin, name
        dev, g, k, tm, tn, n, rho = _shape(fits)
        assert tuple(int(x) for x in domain.groups()) == (g, k, tm, tn, n, rho), name
        assert (pin[2], *(int(x) for x in pin.groups()[2:8])) == (dev, g, k, tm, tn, n, rho), name
        assert g == DEVICES[dev].g and Fraction(1, rho) == DEVICES[dev].cap, name
        got = inert_bound._inert_bound_constants(k, tm, tn, n, rho, dev)
        want = ast.literal_eval(f"({pin[9]})")
        assert (*got[:5], got[5][-1] if pin[1] else list(got[5])) == want, name
    for name, stmt in _theorems("refuse_").items():
        fits = re.search(rf"¬ InertBoundFits {SHAPE}", stmt)
        assert fits, name
        dev, g, k, tm, tn, n, rho = _shape(fits)
        assert g == DEVICES[dev].g, name
        with pytest.raises(DomainError):
            inert_bound._inert_bound_constants(k, tm, tn, n, rho, dev)
