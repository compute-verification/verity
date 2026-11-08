from math import log

import figure1 as f1
import figure4 as f
import pytest

ONE = {"configurations": {"tokens": {"label": "(A) Tokens only", "overhead": 1e10},
                          "per_expert": {"label": "(A′) the cut", "overhead": 1e9}},
       "taxes": {"tokens": {"tax": 0.02}, "per_expert": {"tax": 0.03}}, "served_bits_request": 144000}


@pytest.fixture(scope="module")
def out():
    return f.run("GLM-5.3", ONE)


def test_the_simple_form_is_b_ln_over_the_target(out):
    t = out["target"]["bits"]
    assert out["target"]["label"].startswith("GLM-5.3's weights at FP4")
    assert out["simple"][0] == pytest.approx(16 * log(2.0 ** 40) / t)
    assert out["simple"][-1] == 1.0


def test_the_exact_form_charges_location_bits_and_rises_with_the_scope(out):
    p, s = out["exact"]["datacenter year"], out["exact"]["world year"]
    assert (out["primary"], out["secondary"]) == (f1.PRIMARY, f1.SECONDARY)
    t = out["target"]["bits"]
    for bits, simple, a, b in zip(out["bits"], out["simple"], p, s):
        assert simple <= a["rate"] * (1 + bits / t) * (1 + 1e-9) and a["rate"] <= b["rate"] * (1 + 1e-9)
    assert p[0]["rate"] > 2 * out["simple"][0]
    assert p[0]["blocks"] == round(out["tokens"]["datacenter year"] * out["bits_per_token"] / 16)


def test_a_line_buys_beta_over_h_and_its_leeway_is_that_over_the_need(out):
    a, cut, target = out["bought"]
    assert a["rate"] == pytest.approx(0.08 / 1e10) and cut["rate"] == pytest.approx(0.07 / 1e9)
    assert (target["h"], target["config"]) == (f.TARGET_H, None)
    first = out["marks"][0]["label"]
    assert a["leeway"][out["primary"]][first] == pytest.approx(a["rate"] / out["need"][out["primary"]][first])
    assert a["largest_bits"][out["primary"]] is None
    widest = target["largest_bits"][out["primary"]]
    assert f.exact(widest, out["tokens"][out["primary"]], out["bits_per_token"], out["target"]["bits"])["rate"] \
        <= target["rate"] < f.exact(widest * 1.01, out["tokens"][out["primary"]], out["bits_per_token"],
                                    out["target"]["bits"])["rate"]


def test_the_figure_renders(out, tmp_path):
    assert "orders of magnitude below proving every block" in f.caption(out)
    assert [p.name for p in f.render(out, tmp_path)] == ["figure4.png", "figure4.svg"]
