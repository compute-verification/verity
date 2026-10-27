"""The card files (`verity.hardware`) as θ, the data sheet and the table route read them: each device's ceilings are its
card's, an expression reads only ceilings, and a number written beside its card expression is refused."""
from __future__ import annotations

import shutil
from fractions import Fraction

import pytest

from verity.hardware import load
from verity_machine_model import card as K
from verity_machine_model import eps as E
from verity_machine_model import sheet as S
from verity_machine_model import theta as T
from verity_machine_model.model import DATA, device

SM120, SM90 = DATA, device("sm_90")


def test_expressions_read_the_card_exactly():
    rtx, h200 = load("rtx-pro-6000-server"), load("h200-sxm-141gb")
    assert K.value(rtx, "rate.tensor / weight.nvf4") == 2048 and K.value(h200, "rate.tensor / weight.f64") == 128
    assert K.value(rtx, "rate.smem * 4") == 128 and K.number(h200, "memory.bandwidth") == 4.916736e12
    assert K.value(rtx, "host.amx * host.turbo * 1e9") == Fraction(39936, 10**4) * 10**12
    with pytest.raises(SystemExit, match="rate.alu is marked measurement, not ceiling"):
        K.value(rtx, "rate.alu")
    with pytest.raises(SystemExit, match="reads rate.nope, which the card file doesn't have"):
        K.value(rtx, "rate.nope")


@pytest.mark.parametrize("data, sku", [(SM120, "rtx-pro-6000-server"), (SM90, "h200-sxm-141gb")], ids=["sm_120", "sm_90"])
def test_each_device_reads_its_card(data, sku):
    c = load(sku)
    assert K.of(data)["sku"] == sku
    t = T.theta(data)
    assert t.sms == c["sm"]["count"]["value"] and t.clocks["fmax"].mhz == c["clock"]["sm"]["value"]
    assert t.rates["tensor.s8.k32"].sheet == c["rate"]["tensor"]["value"] and t.rates["issue"].sheet == 4
    assert t.rates["hbm"].sheet == c["memory"]["bandwidth"]["value"]
    assert t.capacities["hbm"].value == c["store"]["dram"]["value"]
    carded = [r for r in t.rates.values() if r.grade == T.CARD]
    assert len(carded) >= 15 and all(r.source.startswith(f"verity/hardware/cards/{sku}.toml: ") for r in carded)
    assert E.table(data).sms == c["sm"]["count"]["value"]
    if data == SM90:
        s = S.sheet(data)
        assert (s.sms.value, s.fmax.value, s.peaks["s8"].macs, s.pipes["alu"].value) == (132, 1980, 4096, 64)
        assert E.table(data).mhz == 1980


def test_a_number_beside_its_card_expression_is_refused(tmp_path):
    shutil.copytree(SM90, tmp_path / "d", ignore=shutil.ignore_patterns("census", "vectors", "designs"))
    p = tmp_path / "d" / T.THETA
    text = p.read_text()
    p.write_text(text.replace('card = "rate.issue"\n', 'card = "rate.issue"\nsheet = 4\n', 1))
    with pytest.raises(SystemExit, match="rate issue reads 'rate.issue' from the card, which holds its number and "
                                         "source; drop sheet"):
        T.theta(tmp_path / "d")
    p.write_text(text)
    m = tmp_path / "d" / "model.toml"
    m.write_text(m.read_text().replace('card = "h200-sxm-141gb"\n', ""))
    K.of.cache_clear()
    with pytest.raises(SystemExit, match="reads the card, and .*model.toml names none"):
        T.theta(tmp_path / "d")
