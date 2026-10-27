"""The V4 proxy's error probe (`error_probe.py`) on tiny shapes: the sink's logit, the arms' errors and their parts, the
factors, and the V4-scaled arm it writes, which parses and runs."""
import math
import tomllib

import error_probe as P
import pytest
import spec
import torch
from test_dsv2 import tiny_dsv2

TINY = {"rep": 2, "d": 48, "dv": 32, "s": 96, "groups": 1}


@pytest.fixture(scope="module")
def v2(tmp_path_factory):
    return tiny_dsv2(tmp_path_factory.mktemp("dsv2"))


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(P, "ROWS", 64)
    monkeypatch.setattr(P, "CAL_ROWS", 32)
    monkeypatch.setattr(P, "N_CAP", 48)
    lin = {"deepseek-v2-lite": {c: [(64, 48)] for c in P.V2_CLASSES},
           "deepseek-v4-pro": {c: [(128, 48)] for c in P.V2_CLASSES}}
    lin["deepseek-v4-pro"]["q"] = [(128, 48), (64, 96)]
    monkeypatch.setattr(P, "LINEARS", lin)
    att = {"deepseek-v2-lite@2048": TINY, "deepseek-v4-pro-csa@2048": {**TINY, "s": 64, "d": 64, "dv": 64},
           "deepseek-v4-pro-hca@2048": {**TINY, "s": 32, "d": 64, "dv": 64}, "llama-3.1-8b@2048": TINY,
           "llama-3.1-8b@8192": {**TINY, "s": 192}}
    monkeypatch.setattr(P, "ATTENTION", att)
    arms = ["int8-all", "adopted-bd-lm-pv1024", "adopted-bd-lm-pv"]
    monkeypatch.setattr(P, "ATTN_ARMS", {k: arms for k in att} | {"llama-3.1-8b@2048": [
        "int8-all", "adopted-bd-lm-pv256", "adopted-bd-lm-pv512", "adopted-bd-lm-pv1024", "adopted-bd-lm-pv"]})


def test_the_sink_logit_is_beta():
    g = torch.Generator().manual_seed(0)
    q, k, v = P.attention_operands(TINY, 2.0, 7.0, g)
    assert q.shape == (1, 2, 96, 64) and k.shape == (1, 1, 96, 64) and v.shape == (1, 1, 96, 32)
    s = q[0] @ k[0, 0].T / math.sqrt(48)
    assert torch.allclose(s[..., 0], torch.full_like(s[..., 0], 7.0), atol=1e-4)
    assert abs(float(s[..., 1:].std()) - 2.0) < 0.1


def run_all(arm="adopted-bd-lm-pv"):
    rows = []
    for m in P.LINEARS:
        for c in P.LINEARS[m]:
            rows += P.task(("linear", (m, c), (), arm, (0,), 1))
    for s in P.ATTENTION:
        for th in ((1.0, 0.0), (2.0, 4.0)):
            rows += P.task(("attention", s, th, arm, (0,), 1))
    return {"arm": arm, "seeds": [0], "rows": rows}


def test_the_errors_and_their_parts(tiny):
    probe = run_all()
    att = {(r["shape"], r["tau"], r["beta"], r["arm"]): r["r"] for r in probe["rows"] if r["what"] == "attention"}
    for shape in P.ATTENTION:
        for th in ((1.0, 0.0), (2.0, 4.0)):
            whole, qk, pv = (att[(shape, *th, a)] for a in ("adopted-bd-lm-pv", "adopted-bd-lm-pv:qk", "adopted-bd-lm-pv:pv"))
            assert 0 < qk < whole and 0 < pv < whole and whole < 1.5 * math.hypot(qk, pv), (shape, th)
    lin = [r for r in probe["rows"] if r["what"] == "linear"]
    assert all(0 < r["r"] < 0.1 for r in lin if r["class"] != "head" or r["arm"] != "int8-all")
    q = P.lin(probe["rows"], "deepseek-v4-pro", "q", "adopted-bd-lm-pv")
    parts = [r["r"] for r in lin if r["model"] == "deepseek-v4-pro" and r["class"] == "q" and r["arm"] == "adopted-bd-lm-pv"]
    assert len(parts) == 2 and q == pytest.approx(math.hypot(*parts))


def test_the_factors_and_the_v4_arm(tiny, v2, ids, calib):
    from test_engine import evaluate, passed
    probe = run_all()
    rows = probe["rows"]
    measured = {a: P.attn(rows, "deepseek-v2-lite@2048", a, (2.0, 4.0)) for a in P.ATTN_ARMS["deepseek-v2-lite@2048"]}
    f = P.factors(probe, measured)
    assert (f["fit"]["tau"], f["fit"]["beta"]) == (2.0, 4.0) and f["fit"]["log_rms_misfit"] == 0
    assert set(f["linear"]) == set(P.V2_CLASSES) and all(v > 0 for v in f["linear"].values())
    assert f["qk"] > 0 and f["pv"] > 0 and set(f["llama_8b_check"]["predicted_8k"]) == set(P.LLAMA_8B["llama-3.1-8b@8192"])
    arm = spec.parse(tomllib.loads(P.spec_toml(f, "adopted-bd-lm-pv")), "v4")
    base = spec.load("adopted-bd-lm-pv")
    assert [s.defn.name for s in arm.linear] == ["scale_error"] * len(P.V2_CLASSES) + [s.defn.name for s in base.linear]
    assert [s.defn.name for s in arm.attention] == ["scale_error"] + [s.defn.name for s in base.attention]
    for name in ("self_attn.q_proj", "mlp.experts.5.down_proj", spec.HEAD):
        assert sum(s.defn.name == "scale_error" and s.applies(1, "linear", name) for s in arm.linear) == 1, name
    r = evaluate(v2, ids[:2], [arm], calib=calib)
    assert passed(r), r["gates"]
