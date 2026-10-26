"""Lean's integer form of the cap circuit's tests (`Security/Proofs/Pouw/PearlC/Cap/IntForm.lean`) against the scheme:
its `IntPrices` are the devices' prices, and each kernel-checked pin in `Cap/Pins.lean` is what Python computes at that
shape: `creditInt` is `PearlC(forming="v1").credit_of`, and `capPin` / `capPinTop` are `cap._cap_constants` at the device's
g, prices and cap (that test skips where `verity.core.protocols.pouw.circuit.cap` is absent)."""
from __future__ import annotations

import ast
import re
from fractions import Fraction
from pathlib import Path

import pytest

from verity.core.protocols.pouw.protocol import DomainError, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES

CAP = Path(__file__).resolve().parents[5] / "Security" / "Proofs" / "Pouw" / "PearlC" / "Cap"
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
        assert C.PearlC(forming="v1", device=d).credit_of(Shape(rows, k, n)) == value, name


def test_cap_pins_are_cap_constants():
    """Each `capPin` pin is `_cap_constants(k, tm, tn, n, rho, dev)` (`capPinTop`: with the table's last entry alone), on the
    device's g and at ρ = 1/rho its cap, with `CapDomain` and `CapFits` at the same shape; each refused shape raises."""
    cap = pytest.importorskip("verity.core.protocols.pouw.circuit.cap", reason="the cap circuit is `verity.core.protocols.pouw.circuit.cap`'s")
    for name, stmt in _theorems("pin_").items():
        domain = re.search(r"CapDomain (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)", stmt)
        fits = re.search(rf"CapFits {SHAPE}", stmt)
        pin = re.search(rf"capPin(Top)? {SHAPE} = \((.*)\)\s*$", stmt, re.S)
        assert domain and fits and pin, name
        dev, g, k, tm, tn, n, rho = _shape(fits)
        assert tuple(int(x) for x in domain.groups()) == (g, k, tm, tn, n, rho), name
        assert (pin[2], *(int(x) for x in pin.groups()[2:8])) == (dev, g, k, tm, tn, n, rho), name
        assert g == DEVICES[dev].g and Fraction(1, rho) == DEVICES[dev].cap, name
        got = cap._cap_constants(k, tm, tn, n, rho, dev)
        want = ast.literal_eval(f"({pin[9]})")
        assert (*got[:5], got[5][-1] if pin[1] else list(got[5])) == want, name
    for name, stmt in _theorems("refuse_").items():
        fits = re.search(rf"¬ CapFits {SHAPE}", stmt)
        assert fits, name
        dev, g, k, tm, tn, n, rho = _shape(fits)
        assert g == DEVICES[dev].g, name
        with pytest.raises(DomainError):
            cap._cap_constants(k, tm, tn, n, rho, dev)
