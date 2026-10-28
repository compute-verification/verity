import math

import pytest

from golden import model


@pytest.mark.parametrize("key", ["b57f", "b921", "a1fc"])
def test_calibration_within_ten_percent(key):
    assert abs(model.calibrate()[key]["ratio"] - 1) < 0.10


@pytest.mark.parametrize("key", list(model.VALIDATION))
def test_out_of_sample_within_ten_percent(key):
    assert abs(model.validate()[key]["ratio"] - 1) < 0.10


def test_row_ands_matches_sampled_proofs():
    from verity.core.protocols.sampled_proofs import optimize
    for n in (1, 64, 1024, 2048, 4096, 16384, 32768):
        assert model.row_ands(n) == optimize.row_ands(n)


def test_unknown_keys_refused():
    with pytest.raises(KeyError):
        model.design(commit={"rowbytes": 4096})
    with pytest.raises(KeyError):
        model.J({"comit": {}})


def test_draws_are_lambda_over_eps_whatever_the_window():
    bound = math.log(2 ** 40) / 0.01
    for steps in (1, 1000):
        r = model.J(model.design(point={"steps_per_window": steps}))
        assert r["draws"]["values"] == pytest.approx(bound, rel=1e-3)
        assert r["draws"]["pouw"] == pytest.approx(bound, rel=1e-3)


def test_unnoised_pouw_earns_no_credit():
    assert not model.J(model.design(pouw={"scheme": "j10r"}))["pouw"]["credit"]["earns_credit"]
    assert model.pouw_credit(model.design(pouw={"L": 8192}))["credited_share_of_linear_macs"] == 0
    assert model.J(model.design())["pouw"]["credit"]["earns_credit"]
