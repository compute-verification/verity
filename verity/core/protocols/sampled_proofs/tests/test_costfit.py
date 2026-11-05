"""The cost fit (`costfit`): it recovers the coefficients it was generated from, holds them non-negative, pins what another
fit measured, and round-trips through its cost-model/v1 record into the planner's `CostModel`."""

from __future__ import annotations

import pytest

from verity.core.protocols.sampled_proofs.costfit import Fitted, Observation, fit, from_record
from verity.core.protocols.sampled_proofs.optimize import Candidate

TRUE = {"one": 0.6, "ands*drawn": 9.5e-11}


def _sweep(noise=(0.0,) * 6) -> list[Observation]:
    """A session per batch size from one statement of 1,024 proof blocks of a 14.8M-AND template."""
    ands = 14_843_669
    return [Observation(f"r{b}", "G", ands, 1024, b, prove=(TRUE["one"] + TRUE["ands*drawn"] * ands * b) * (1 + e))
            for b, e in zip((1, 4, 16, 64, 256, 1024), noise)]


def test_fit_recovers_exact_coefficients():
    f = fit(_sweep(), "prove", ["one", "ands*drawn"])
    assert f.coef == pytest.approx(TRUE, rel=1e-9) and f.sigma == pytest.approx(0.0, abs=1e-9) and f.points == 6


def test_fit_reports_its_relative_residual():
    f = fit(_sweep((0.05, -0.05, 0.05, -0.05, 0.05, -0.05)), "prove", ["one", "ands*drawn"])
    assert 0.03 < f.sigma < 0.08 and f.coef["ands*drawn"] == pytest.approx(TRUE["ands*drawn"], rel=0.1)


def test_fit_holds_coefficients_non_negative():
    obs = [Observation("a", "T", 10, 1, 1, verify=5.0), Observation("b", "T", 20, 1, 1, verify=4.0)]
    f = fit(obs, "verify", ["one", "ands*staged"])
    assert f.coef["ands*staged"] == 0.0 and 4.0 < f.coef["one"] < 5.0


def test_pinned_coefficients_and_the_sigma_floor():
    """Two zk sessions of two templates fix the session's fixed part once the per-proof-block slope is pinned from the sweep."""
    b = TRUE["ands*drawn"]
    obs = [Observation("a6", "K2048", 2_650_000, 4096, 11, prove=1.4), Observation("a6", "K8192", 9_030_000, 2048, 6,
                                                                                  prove=2.2)]
    f = fit(obs, "prove", ["one", "ands", "ands*drawn"], fixed={"ands*drawn": b})
    assert f.pinned == ("ands*drawn",) and f.sigma == 0.5
    assert [f(o) for o in obs] == pytest.approx([1.4, 2.2], rel=1e-9)
    with pytest.raises(ValueError, match="no observation"):
        fit(obs, "verify", ["one"])


def test_record_round_trips_into_the_planners_model():
    obs = _sweep() + [Observation("a6", "K2048", 2_650_000, 4096, 11, verify=32.0, stage=33.0),
                      Observation("a6", "K8192", 9_030_000, 2048, 6, verify=83.0, stage=69.0)]
    fitted = Fitted(fit(obs, "prove", ["one", "ands*drawn"]), fit(obs, "verify", ["one", "ands*staged"]),
                    fit(obs, "stage", ["ands*staged"]), cpu=1 / 55, key={"prover": "p", "verifier": "v", "gpu": "g"})
    rec = fitted.record(obs)
    assert rec["kind"] == "cost-model/v1" and rec["key"]["prover"] == "p" and len(rec["observations"]) == 8
    model = from_record(rec).model(source="art:x")
    c = Candidate("op", "K2048", 4096, 1, 16, ands=2_650_000, cap=4096)
    a0, a1 = model.statement(c)
    v, s = fitted.verify.coef, fitted.stage.coef
    assert a0 == pytest.approx(TRUE["one"] + v["one"] / 55, rel=1e-9)
    assert a0 + a1 * 4096 == pytest.approx(TRUE["one"] + (fitted.verify(obs[-2]) + fitted.stage(obs[-2])) / 55, rel=1e-9)
    assert model.unit(c) == pytest.approx(TRUE["ands*drawn"] * 2_650_000, rel=1e-9) and model.source == "art:x"
    with pytest.raises(ValueError, match="cost-model/v1"):
        from_record({"kind": "other"})


def test_every_feature_of_every_fit_reaches_the_model():
    """A prover that pays for its staged proof blocks and a verifier that pays for the drawn ones: each feature's rate is the
    prove coefficient plus the verify and stage ones at the price ratio."""
    p = {"one": 7.0, "ands": 2e-7, "ands*staged": 3e-10, "ands*drawn": 4e-9}
    v = {"one": 1.5, "ands*staged": 5e-9, "ands*drawn": 6e-8}
    s = {"one": 2.0, "ands*staged": 8e-9}
    obs = [Observation("z", f"K{ands}", ands, c, k, prove=sum(p[f] * o for f, o in zip(p, (1, ands, ands * c, ands * k))),
                       verify=v["one"] + v["ands*staged"] * ands * c + v["ands*drawn"] * ands * k,
                       stage=s["one"] + s["ands*staged"] * ands * c)
           for ands in (5_132_800, 15_772_800) for c in (1, 8, 64, 512) for k in sorted({1, 8, 64, c}) if k <= c]
    fitted = Fitted(fit(obs, "prove", list(p)), fit(obs, "verify", list(v)), fit(obs, "stage", list(s)), cpu=0.5)
    model = fitted.model()
    assert model.statement_fixed == pytest.approx(7.0 + (1.5 + 2.0) * 0.5, rel=1e-6)
    assert model.statement_per_and == pytest.approx(2e-7, rel=1e-6)
    assert model.staged_per_and == pytest.approx(3e-10 + (5e-9 + 8e-9) * 0.5, rel=1e-6)
    assert model.per_and == pytest.approx(4e-9 + 6e-8 * 0.5, rel=1e-6)
