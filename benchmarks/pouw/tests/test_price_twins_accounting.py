"""Pearl-C's Python accounting is the Lean's on every pinned FP8 price twin (`harness/price_twins.json`) and on the H100's
rev1 pins, as exact rationals: γ, the credit, and the debit cap the verifier's tile rule admits (`pearl_c_work.tile_cap`).
The verifier must admit no more than the rule its guarantee is proved for, rev1's (`DeviceRev1.lean`: `creditDevRev1`,
`tileDebitRev1`, `wrefDevRev1K`), so a credit or cap that drifts from it fails here, as the first promotion add of every
word did before rev1.  A pinned twin of a statement this file doesn't price (a hot start's, Pearl-C4's) fails too, until it
has a Python counterpart here."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from verity.core.protocols.pouw.protocol import Matmul, Shape
from verity.core.protocols.pouw.schemes import get
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_c_debit as D
from verity.core.protocols.pouw.schemes import pearl_c_work as W
from verity.core.protocols.pouw.schemes.pearl_c_device import SM120, SM120_UNPROMOTED

sys.path.insert(0, str(Path(__file__).parents[1] / "harness"))

import price_twins as P  # noqa: E402

GAMMA0 = Fraction(1, 400)            # TT_OUT's γ0, at every cap (`gammaFromTTOut`)
DEVICES = {"pearl-c-sm120/v1": SM120, "pearl-c-sm120/v2": SM120_UNPROMOTED}
#: the twins' price records by FP32 add price, each with the cast its W_ref already holds: Lean's `Prices.sm120`
#: ⟨8, 2, ⟨40, 1⟩⟩ at 8, and `Prices.sm120Loop` ⟨1047/125, 2, ⟨40, 2⟩⟩ (`DevicePricesLoop.lean`: the in-loop add, and the
#: A-only forming's 1.047 rounded up to 2) at 8.953, so that `wrefDevRev1K`'s c is the honest cast less it
#: (`DeviceSm120KernelGamma.lean`)
PRICES = {Fraction(8): (SM120.prices, Fraction(8)),
          Fraction(1047, 125): (replace(SM120.prices, fadd=Fraction(1047, 125), forming_a_only=2), Fraction(8953, 1000))}
TWINS = [t for t in P.load() if t["status"] == "pinned"]
CUBES = [t for t in TWINS if t["shape"] is not None]
#: each layout twin's linears, as `Definitions/Pouw/PearlC/Device.lean` lists them (`llama31_8bLinears`)
LAYOUTS = {"llama31_8b": ((6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336))}
H100_PINS = {f"Pouw.SecurityProofs.PearlC.{kind}H100Rev1_{n}": n
             for kind in ("Gamma", "Sampled") for n in (8192, 16384)}


def scheme_of(twin: dict) -> C.PearlC:
    """The twin's device record as a scheme: v1's G = 4 or v2's no promotion, at the twin's prices and cap."""
    d = DEVICES[twin["statement"]]
    return C.PearlC(forming="v1", device=replace(d, cap=Fraction(twin["cap"]), prices=PRICES[Fraction(twin["fadd"])][0]))


def cast_offset(twin: dict) -> Fraction:
    """`wrefDevRev1K`'s c: the priced cast beyond the record's own, per activation element."""
    return P.CAST_VALUES[twin["path"]] - PRICES[Fraction(twin["fadd"])][1]


def gamma(s: C.PearlC, n: int, c: Fraction, reading: str, kind: str) -> Fraction:
    """γ at n³ from the scheme's W_ref and credit, as the Lean states it per reading: the unit's or a tile's least credit
    over `wrefDevRev1K d c`.  A tile's credit is (1 − ρ)·credit·share whatever the cap reads, so a chain-cap tile's γ is
    the forming-credited one; the chain cap's unit (v2's `DeviceChainCap`) reads the debit against ρ·`chainCreditDev`."""
    sh = Shape(n, n, n)
    w, credit, rho = s.wref(sh) + c * n * n, Fraction(s.credit_of(sh)), s.cap
    if reading == "forming-credited" or (reading == "chain-cap" and kind == "tile"):
        least = (1 - rho) * credit
    elif reading == "chain-only":
        least = credit - s.prices.forming_credited * n * n - rho * credit
    elif reading == "chain-cap":
        least = credit - rho * s._chain_units(n, n, n)
    else:
        raise AssertionError(f"no Python reading {reading!r}")
    return 1 - (1 - GAMMA0) * least / w


def lean_credit(g: Fraction, n: int, rho: Fraction, qa: Fraction, c: Fraction) -> Fraction:
    """The Lean's credit at n³ solved from its forming-credited γ: γ = 1 − (1 − γ0)(1 − ρ)·C/(C + (qa + c)·n²)."""
    r = (1 - g) / ((1 - GAMMA0) * (1 - rho))
    return r * (qa + c) * n * n / (1 - r)


def test_every_pinned_twin_is_priced_here():
    assert len(TWINS) >= 20 and {t["statement"] for t in TWINS} <= set(DEVICES)
    assert {(t["reading"], t["kind"]) for t in TWINS} <= {(r, k) for r in ("forming-credited", "chain-only", "chain-cap")
                                                          for k in ("unit", "tile")}


@pytest.mark.parametrize("twin", CUBES, ids=[t["pin"].removeprefix("Pouw.PearlC.") for t in CUBES])
def test_gamma_is_the_pinned_twins(twin):
    s, n = scheme_of(twin), twin["shape"]
    assert gamma(s, n, cast_offset(twin), twin["reading"], twin["kind"]) == Fraction(twin["gamma"])


@pytest.mark.parametrize("twin", [t for t in CUBES if t["reading"] == "forming-credited"],
                         ids=lambda t: t["pin"].removeprefix("Pouw.PearlC."))
def test_credit_and_cap_are_the_pinned_twins(twin):
    """The Lean's credit, solved from the pinned γ alone, is `credit_of`; the unit's cap ρ·credit is its cells' and a
    tile's cap is ρ·credit·tileShare, `tile_cap`'s on v1 (v2 has no debit rule, so its tile cap is the cells' credit)."""
    s, n, c = scheme_of(twin), twin["shape"], cast_offset(twin)
    sh, rho = Shape(n, n, n), s.cap
    credit = lean_credit(Fraction(twin["gamma"]), n, rho, Fraction(s.prices.forming_a_only), c)
    assert s.wref(sh) - s.credit_of(sh) == s.prices.forming_a_only * n * n and s.credit_of(sh) == credit
    record = W.UnitRecord(Matmul(0, sh, 0), n, n)
    assert rho * W.cell_credit(s, record) * n * n == rho * credit
    rows, cols = 2, 3
    share = Fraction(rows * cols, n * n)
    if s.device.g:
        debit, cap = W.tile_cap(s, record, [[0] * n] * rows, [[0] * n] * cols)
        assert cap == rho * credit * share
        assert debit >= s.prices.fadd * rows * cols * (n // s.device.window - 1)
    else:
        assert rho * W.cell_credit(s, record) * rows * cols == rho * credit * share


@pytest.mark.parametrize("twin", [t for t in TWINS if t["shape"] is None], ids=lambda t: t["pin"].split(".")[-1])
def test_gamma_is_the_pinned_linear_and_layout_twins(twin):
    """A twin of one linear (n, k) states one γ at every batch m, and a layout's is the max over its linears: the unit's
    least credit (1 − ρ)·credit over `wrefDevRev1K d c` at m × n × k."""
    s, c = scheme_of(twin), cast_offset(twin)
    linears = LAYOUTS[twin["layout"]] if twin.get("layout") else ((twin["n"], twin["k"]),)
    assert twin["reading"] == "forming-credited"
    for m in (1, 32, 8192):
        gammas = []
        for n, k in linears:
            sh = Shape(m, k, n)
            gammas.append(1 - (1 - GAMMA0) * (1 - s.cap) * Fraction(s.credit_of(sh)) / (s.wref(sh) + c * m * k))
        assert max(gammas) == Fraction(twin["gamma"])


@pytest.mark.parametrize("pin,n", H100_PINS.items(), ids=list(H100_PINS))
def test_h100_gamma_is_rev1s_pin(pin, n):
    """`pearl-c-h100-v1`'s γ at the cap is `DeviceRev1Gamma.lean`'s, read off the repository's pin (its lifted
    guarantee's statement)."""
    audit = json.loads(P.REPO_AUDIT.read_text())
    sig, lifted = P._statement(pin, P.guarantees(audit)[pin], P.lifted_definitions(audit, P.REPO_AUDIT.parent))
    lit = P._GAMMA_LITERAL.search(sig) or P._GAMMA_SOURCE.search(sig)
    assert lifted and lit, sig
    s = get("pearl-c-h100-v1")
    assert gamma(s, n, Fraction(0), "forming-credited", "unit") == Fraction(int(lit[1]), int(lit[2]))
    assert s.credit_of(Shape(n, n, n)) == lean_credit(Fraction(int(lit[1]), int(lit[2])), n, s.cap,
                                                      Fraction(s.prices.forming_a_only), Fraction(0))


def test_the_zero_word_debits_every_later_promotion():
    """`tile_cap`'s debit holds `tileDebitRev1`'s later +0 promotions: a zero word's k/(32·G) − 1."""
    s, k = scheme_of(next(t for t in TWINS if t["statement"] == "pearl-c-sm120/v1")), 1024
    record = W.UnitRecord(Matmul(0, Shape(16, k, 16), 0), 16, k)
    atoms, zeros = D.chain_debit([[0] * k], [[0] * k], s.device)
    assert zeros == k // s.device.window - 1
    assert W.tile_cap(s, record, [[0] * k], [[0] * k])[0] == D.debit_units(atoms, 0, s.prices, s.device, zeros=zeros)
