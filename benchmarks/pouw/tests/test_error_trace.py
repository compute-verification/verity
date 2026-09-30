"""The error trace's arithmetic (`benchmarks/pouw/pearl_c_vllm/error_trace.py`), on the CPU."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

PATH = Path(__file__).resolve().parents[1] / "pearl_c_vllm" / "error_trace.py"
spec = importlib.util.spec_from_file_location("pearl_c_vllm_error_trace", PATH)
ET = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ET)


def operands(m=6, n=5, k=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g).to(torch.bfloat16)
    w = torch.randn(n, k, generator=g).to(torch.bfloat16)
    fp8 = lambda t: t.clamp(-448, 448).to(torch.float8_e4m3fn)  # noqa: E731
    epa, epb = fp8(torch.randn(m, 32, generator=g)), fp8(torch.randn(n, 32, generator=g))
    fa, fb = fp8(torch.randn(k, 32, generator=g)), fp8(torch.randn(k, 32, generator=g))
    al_a, al_b = torch.full((m,), 8.0), torch.full((n,), 16.0)
    at = ET.e4m3((al_a[:, None] * x.float()).double() + epa.double() @ fa.double().T)
    bt = ET.e4m3((al_b[:, None] * w.float()).double() + epb.double() @ fb.double().T)
    return x, w, at, bt, epa, epb, fa, fb, 1 / al_a, 1 / al_b


def test_the_split_sums_to_the_error_and_an_exact_device_leaves_only_the_cast():
    x, w, at, bt, epa, epb, fa, fb, ia_a, ia_b = operands()
    ref = x.float() @ w.float().T
    f = torch.float64
    exact = at.to(f) @ bt.to(f).T
    clean = (at.to(f) - epa.to(f) @ fa.to(f).T) @ (bt.to(f) - epb.to(f) @ fb.to(f).T).T
    s = ia_a.to(f)[:, None] * ia_b.to(f)[None, :]
    sp = ET.split(ref, clean * s, exact, clean, at, bt, epa, epb, fa, fb, ia_a, ia_b)
    assert all(float(sp["parts"][p].abs().max()) == 0.0 for p in ("chain", "peel", "out"))
    assert torch.equal(sp["parts"]["cast_noise"], sp["total"])
    noisy = ET.split(ref, (clean * s).to(torch.bfloat16), exact.float(), clean.float(), at, bt, epa, epb, fa, fb, ia_a, ia_b)
    assert float((noisy["total"] - sum(noisy["parts"].values())).abs().max()) < 1e-9 * float(ref.abs().max())
    assert float(noisy["parts"]["out"].abs().max()) > 0


def test_the_cast_error_of_the_clean_operands_is_e4m3s_and_the_noise_is_removed_exactly():
    x, w, at, bt, epa, epb, fa, fb, ia_a, ia_b = operands(m=32, n=24, k=256)
    ref = x.float() @ w.float().T
    sp = ET.split(ref, ref, ref.double(), ref.double(), at, bt, epa, epb, fa, fb, ia_a, ia_b)
    cast = ET.rel(sp["parts"]["cast_noise"], ref)
    assert 0.005 < cast < 0.2


def test_the_noise_gain_is_one_without_noise_and_two_at_the_signals_own_size():
    g = torch.Generator().manual_seed(1)
    x, w = torch.randn(64, 128, generator=g), torch.randn(32, 128, generator=g)
    same = ET.noise_gain(x, w, torch.zeros_like(x), torch.zeros_like(w))
    assert same == {"a": 1.0, "b": 1.0}
    doubled = ET.noise_gain(x, w, x.flip(0), w.flip(0))
    assert doubled["a"] == pytest.approx(2.0, rel=0.1) and doubled["b"] == pytest.approx(2.0, rel=0.1)


def test_the_noise_law_reads_rms_over_rho():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(256, 1024, generator=g)
    rho = (x[:, ::8] ** 2).mean(1).sqrt()
    noise = torch.randn(256, 1024, generator=g)
    noise = noise / (noise ** 2).mean(1, keepdim=True).sqrt() * rho[:, None]
    law = ET.noise_law(x, noise)
    assert law["p50"] == pytest.approx(1.0, abs=1e-5) and law["max_over_rho_p50"] > 2


def test_e4m3_saturates_and_rounds_to_nearest_even():
    v = ET.e4m3(torch.tensor([1000.0, -1000.0, 1.0625, 1.1875], dtype=torch.float64)).float().tolist()
    assert v == [448.0, -448.0, 1.0, 1.25]


def test_rows_separate_bos_and_the_summary_reads_every_linear():
    ref = torch.ones(4, 8)
    e = torch.zeros(4, 8)
    e[0] = 1.0
    r = ET.rows(e, ref)
    assert r["bos"] == pytest.approx(1.0) and r["no_bos"] == 0.0 and r["all"] == pytest.approx(0.5)

    def part(w, name, pc, f8):
        err = {m: {"all": 0.01, "no_bos": 0.01} for m in ET.METHODS}
        err["pearlc"], err["fp8"] = {"all": pc, "no_bos": pc}, {"all": f8, "no_bos": f8}
        return {"w": w, "name": name, "kind": ET.kind(name), "error": err,
                "one_sided": {m: {"all": 0.01, "no_bos": 0.01} for m in ET.ONE_SIDED},
                "share": {"cast_noise": 0.9, "chain": 0.05, "peel": 0.03, "out": 0.02},
                "expected": {"cast_noise_over_measured": 1.0}, "noise_law": {"a": {"p50": 1.0}, "b": {"p50": 1.0}}}
    s = ET.summarize([part(0, "model.layers.0.self_attn.qkv_proj", 0.06, 0.04), part(1, "model.layers.0.mlp.down_proj", 0.09, 0.03)])
    assert s["linears"] == 2 and set(s["by_kind"]) == {"qkv_proj", "down_proj"}
    assert s["all"]["pearlc_over_fp8_no_bos"]["max"] == pytest.approx(3.0) and s["worst_ratio"][0]["w"] == 1
