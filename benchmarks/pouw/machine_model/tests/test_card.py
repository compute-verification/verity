"""The card files (`verity.hardware`) as θ, the data sheet and the table route read them: each device's ceilings are its
card's, an expression reads only ceilings, and a number written beside its card expression is refused."""
from __future__ import annotations

import csv
import shutil
import tomllib
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


#: each DS-120 tensor kind's operand format (data/forms.toml gives its shape and whether it is 2:4 sparse)
DS120_FORMATS = {"MMA": "s8", "IMMA16832": "s8", "IMMA16816": "s8", "IMMA8816": "s8", "IMMASP16864": "s8",
                 "IMMASP16832": "s8", "HMMA16816": "f16", "HMMA16816F16": "f16", "HMMASP16832": "f16",
                 "HMMA1688T": "tf32", "HMMA1684T": "tf32", "QMMA16832": "e4m3", "QMMASF16832": "e4m3",
                 "QMMASP16864": "e4m3", "OMMASF16864": "nvf4"}


def test_ds120s_lines_are_the_cards():
    """sm_120 has no sheet.toml, so DS-120's census lines are held to the card here: a tensor form's MACs (half its
    logical m · n · k when 2:4 sparse) over a SMSP's quarter of its format's rate, a pipe's 128 lanes over its rate, and
    a shared-memory instruction one 128-byte request (DS_smem's description in rows.csv)."""
    c = K.of(SM120)
    forms = tomllib.loads((SM120 / "forms.toml").read_text())
    forms["MMA"] = forms["IMMA16832"]
    pipes = {"DS_imad": "rate.imad", "DS_shfl": "rate.shfl", "DS_xu": "rate.sfu"}
    seen = set()
    with open(SM120 / "census" / "DS-120.csv", newline="") as f:
        for x in csv.DictReader(f):
            got = Fraction(x["cycles"])
            if x["row"] == "DS_tensor":
                m, n, k = forms[x["kind"]]["shape"]
                macs = Fraction(m * n * k, 2 if forms[x["kind"]].get("sparse") else 1)
                assert got == macs / (K.value(c, f"rate.tensor / weight.{DS120_FORMATS[x['kind']]}") / 4), x
            elif x["row"] in pipes:
                assert got == 128 / K.value(c, pipes[x["row"]]), x
            elif x["row"] == "DS_smem" and x["kind"].startswith(("LDS", "STS")):
                assert got == 4 * 128 / K.value(c, "rate.smem * 4"), x
            else:
                continue
            seen.add(x["row"])
    assert seen == {"DS_tensor", "DS_smem", *pipes}


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
