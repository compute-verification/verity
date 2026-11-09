"""The GLM-5.3 fused RMSNorm capture without a GPU: the parametrised model at its defaults is the twin and every alternative changes
its scale or its product order; `probe` separates every alternative from `DsaRMSNorm_v1` at both widths, on rows the reference
evaluator agrees with; the special rows are the reference's; the probe launch's operands keep the served layout; the PTX check
reads the code before the file table; a job takes the twin's words and finds a flipped one; the verdict needs every case and every
refutation; and `main` end to end (the GPU run is `tests/properties/glm53_fused_rmsnorm_capture_gpu.py --out ...` on an H200)."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import json
from pathlib import Path

import numpy as np
import pytest

from tests.properties import glm53_fused_rmsnorm_capture_gpu as X
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics import dsa_norm_rows as NR
from verity.ml.definitions.registry import dsa_norm as DN
from verity_vllm.properties.record import REGISTRY, digest_of

WIDTHS = [N for _, N, _, _ in X.NORMS]
FAKE_PTX = (".version 8.7\n.target sm_90a\n.visible .entry _fused_norm_rope_kernel(\n)\n.reqntid 128\n{\n"
            "\tld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];\n\tshfl.sync.bfly.b32\t%r5, %r1, 16, 31, -1;\n"
            "\tdiv.full.f32 \t%r6, %r5, 0f45000000;\n\tret;\n}\n\t.file\t1 \"/a/kernels.py\"\n")


def _rows(N: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([X.bf16_row(rng, N, 2.0 ** rng.uniform(-3, 3)) for _ in range(n)])


def _reference(N: int, x: np.ndarray, w: np.ndarray) -> list[list[int]]:
    fn = bind(DN.DsaRMSNorm, N=N, EPS=X.EPS)
    return [[int(v) for v in evaluate(fn, r.tolist(), w.tolist())] for r in x]


@pytest.mark.parametrize("N", WIDTHS)
def test_the_model_at_its_defaults_is_the_twin_and_each_alternative_changes_its_scale_or_product(N: int) -> None:
    x = np.concatenate([_rows(N, 64, N), X.special_rows(N, 1)])
    w = X.weight_row(np.random.default_rng(2), N)
    twin = NR.dsa_rmsnorm_rows(x, w, X.EPS)
    assert np.array_equal(X.model_r(x, X.EPS), twin["r"]) and np.array_equal(X.words(x, w, twin["r"], "rx_w"), twin["out"])
    alts = X.alternatives(N)
    wide = ["contiguous", "per_vector_chains"] if N > 1024 else []
    assert list(alts) == ["RMSNormTriton_v1", "f64_sum", "sqrt_epilogue", "warp_pairs", "x_times_rw", *wide]
    assert np.array_equal(alts["RMSNormTriton_v1"][0](x, X.EPS), DR.rmsnorm_triton_rows(x, w, X.EPS)["inv"])
    for name, (r, product) in alts.items():
        assert (product == "x_rw") == (name == "x_times_rw") and (product == "x_rw" or (r(x, X.EPS) != twin["r"]).any()), name


@pytest.mark.parametrize("N", WIDTHS)
def test_the_probe_separates_every_alternative_on_rows_the_reference_agrees_with(N: int) -> None:
    p = X.probe(N, X.EPS, 20261013 + N)
    x, w = p["x"], p["w"]
    twin = NR.dsa_rmsnorm_rows(x, w, X.EPS)["out"]
    assert _reference(N, x, w) == twin.tolist()
    cols = [t["col"] for t in p["targets"]]
    assert len(set(cols)) == len(cols) == X.PER_ALT * len(X.alternatives(N)) and len(x) < min(X.SPECIAL)
    for name, (r, product) in X.alternatives(N).items():
        theirs = X.words(x, w, r(x, X.EPS), product)
        assert int((theirs != twin).sum()) >= X.PER_ALT, name
        for t in (t for t in p["targets"] if t["alternative"] == name):
            assert (int(twin[t["row"], t["col"]]), int(theirs[t["row"], t["col"]]), int(w[t["col"]])) == (t["ours"], t["theirs"], t["w"])
            assert t["ours"] != t["theirs"] and t["w"] in X.PROBE_W
    again = X.probe(N, X.EPS, 20261013 + N)
    assert np.array_equal(again["x"], x) and np.array_equal(again["w"], w) and again["targets"] == p["targets"]


@pytest.mark.parametrize("N", WIDTHS)
def test_the_special_rows_are_the_references(N: int) -> None:
    """The overflowing row's r is +0 and the zero row's rsqrt(eps); a NaN anywhere makes the row NaN; an inf column is NaN (0 * inf)
    and the rest of its row +-0."""
    x = X.special_rows(N, 3)
    w = X.weight_row(np.random.default_rng(4), N)
    twin = NR.dsa_rmsnorm_rows(x, w, X.EPS)
    assert x.shape == (len(X.SPECIAL), N) and _reference(N, x, w) == twin["out"].tolist()
    assert twin["r"][1] == 0 and twin["r"][4] == DR.rsqrt_approx(DR.as_u32(np.float32([X.EPS])))[0]
    assert np.all(twin["out"][8] == 0x7FFF) and twin["out"][7][3] == 0x7FFF and np.all(np.delete(twin["out"][7], 3) & 0x7FFF == 0)


def test_the_probe_launch_keeps_the_served_layout(monkeypatch) -> None:
    """On CPU tensors laid out as `glm53_dsa_capture_gpu.operands` lays them: the special rows land in q_c and kv_c only, and the
    probe launch's rows and weights land in a copy, its views the served strides."""
    import torch

    T, DS = 40, X.DS
    qkv = torch.zeros(T, DS.Q_LORA + DS.KV_LORA + DS.DR, dtype=torch.bfloat16)
    o = {"qkv": qkv, "q_c": qkv[:, :DS.Q_LORA], "kv_c": qkv[:, DS.Q_LORA:DS.Q_LORA + DS.KV_LORA], "k_pe": qkv[:, DS.Q_LORA + DS.KV_LORA:],
         "q_rms_w": torch.ones(DS.Q_LORA, dtype=torch.bfloat16), "kv_rms_w": torch.ones(DS.KV_LORA, dtype=torch.bfloat16)}
    monkeypatch.setattr(DS, "operands", lambda seed: o)
    served = X.served_operands(7)
    words = DS._u16(served["qkv"])
    for norm, N, _, _ in X.NORMS:
        assert np.array_equal(DS._u16(served[norm])[list(X.SPECIAL)], X.special_rows(N, 7 + N))
    assert not words[:, DS.Q_LORA + DS.KV_LORA:].any() and not np.delete(words, list(X.SPECIAL), axis=0).any()
    probes = {norm: {"x": _rows(N, 3, N), "w": X.weight_row(np.random.default_rng(N), N)} for norm, N, _, _ in X.NORMS}
    p = X.probe_operands(served, probes)
    assert np.array_equal(DS._u16(served["qkv"]), words) and p["probe_rows"] == {"q_c": 3, "kv_c": 3}
    for norm, _, wkey, _ in X.NORMS:
        assert p[norm].stride() == served[norm].stride() and p[norm].data_ptr() != served[norm].data_ptr()
        assert np.array_equal(DS._u16(p[norm])[:3], probes[norm]["x"]) and np.array_equal(DS._u16(p[wkey]), probes[norm]["w"])
        assert np.array_equal(DS._u16(p[norm])[3:], DS._u16(served[norm])[3:])
    assert np.array_equal(DS._u16(p["k_pe"]), DS._u16(served["k_pe"]))


def test_the_ptx_check_pins_the_code_before_the_file_table(tmp_path, monkeypatch) -> None:
    rec = X.ptx_record(FAKE_PTX)
    assert rec["counts"] == {"ld.global.v4.b32": 1, "shfl.sync.bfly.b32": 1, "div.full.f32": 1} and rec["reqntid"] == ["128"]
    assert X.ptx_record(FAKE_PTX.replace("/a/", "/b/"))["code_sha256"] == rec["code_sha256"]
    assert X.ptx_record(FAKE_PTX.replace("16, 31", "8, 31"))["code_sha256"] != rec["code_sha256"]
    monkeypatch.setattr(X, "PTX_CODE_SHA256", rec["code_sha256"])
    monkeypatch.setattr(X, "PTX_COUNTS", rec["counts"])
    cache = tmp_path / "cache"
    for d, name, text in (("a", "_fused_norm_rope_kernel.ptx", FAKE_PTX), ("q", "_fused_q_kernel.ptx", "other")):
        (cache / d).mkdir(parents=True)
        (cache / d / name).write_text(text)
    assert X.ptx(cache, tmp_path / "out")["ok"] and len(list((tmp_path / "out" / "ptx").glob("*.ptx"))) == 1
    monkeypatch.setattr(X, "PTX_COUNTS", rec["counts"] | {"div.full.f32": 4})
    assert not X.ptx(cache, tmp_path / "out")["ok"]
    monkeypatch.setattr(X, "PTX_COUNTS", rec["counts"])
    (cache / "b").mkdir()
    (cache / "b" / "_fused_norm_rope_kernel.ptx").write_text(FAKE_PTX.replace("/a/", "/b/"))
    assert not X.ptx(cache, tmp_path / "out")["ok"]           # two specialisations


def _launch(work: Path, case: str, T: int, seed: int, flip: bool = False) -> list[dict]:
    """A launch's saved words with the twin as the kernel (the probe's rows and weights in the probe case; one word flipped)."""
    saved, probe_rows = {}, {}
    for norm, N, _, _ in X.NORMS:
        x = _rows(N, T, seed + N)
        x[list(X.SPECIAL)] = X.special_rows(N, seed)
        w = X.weight_row(np.random.default_rng(seed), N)
        if case == "probe":
            p = X.probe(N, X.EPS, 20261013 + N)
            x[:len(p["x"])], w, probe_rows[norm] = p["x"], p["w"], len(p["x"])
        out = NR.dsa_rmsnorm_rows(x, w, X.EPS)["out"]
        if flip and norm == "q_c":
            out[2, 5] ^= 1
        saved[norm] = {"x": x, "w": w, "out": out}
    return X.jobs_for(work, case, saved, probe_rows)


def test_a_job_takes_the_twins_words_and_finds_a_flipped_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(X, "CHUNK", 16)
    jobs = _launch(tmp_path / "a", "probe", 40, 1)
    assert [(j["norm"], j["span"]) for j in jobs] == [(n, s) for n, *_ in X.NORMS for s in ([0, 16], [16, 32], [32, 40])]
    res = [X.compare(j) for j in jobs]
    assert not any("error" in r for r in res)
    for norm, N, _, _ in X.NORMS:
        rs = [r for r in res if r["norm"] == norm]
        sampled = set(X.SAMPLE) & set(range(40)) | set(X.SPECIAL) | set(range(len(X.probe(N, X.EPS, 20261013 + N)["x"])))
        assert {k: sum(r[k] for r in rs) for k in X.TOTALS} == {"rows": 40, "words": 40 * N, "mismatch": 0, "reference": len(sampled),
                                                                 "twin_mismatch": 0}
        assert all(sum(r["refuted"][a] for r in rs) >= X.PER_ALT for a in X.alternatives(N))
    (job,) = [j for j in _launch(tmp_path / "b", "served", 40, 1, flip=True) if j["norm"] == "q_c" and j["span"][0] == 0]
    want = int(NR.dsa_rmsnorm_rows(np.load(job["paths"]["x"])[2:3], np.load(job["paths"]["w"]), X.EPS)["out"][0, 5])
    bad = X.compare(job)
    assert bad["mismatch"] == 1 and bad["first_mismatch"] == {"row": 2, "col": 5, "want": want, "kernel": want ^ 1}


def _rec(case: str, norm: str, **over) -> dict:
    N = dict((n, w) for n, w, _, _ in X.NORMS)[norm]
    return {"case": case, "norm": norm, "span": [0, 1], "first_mismatch": None, **{k: 1 for k in X.TOTALS}, "mismatch": 0,
            "twin_mismatch": 0, "refuted": {a: X.PER_ALT for a in X.alternatives(N)}} | over


def _doc(comparisons: list[dict]) -> dict:
    return {"gate": {"ok": True}, "served": {"ok": True}, "errors": {}, "ptx": {"ok": True}, "bind": {"ok": True},
            "launches": [{"case": c, "checks": {"deterministic": True, "mla_kv_c": True}} for c in X.CASES], "summary": X.summary(comparisons)}


def test_the_verdict_needs_every_case_and_every_refutation() -> None:
    good = [_rec(c, n) for c in X.CASES for n, *_ in X.NORMS]
    assert X._verdict(_doc(good))
    assert X.summary(good)["refuted"]["kv_c"] == {a: 2 * X.PER_ALT for a in X.alternatives(512)}
    for i in range(len(good)):
        assert not X._verdict(_doc(good[:i] + good[i + 1:])), i
        for over in ({"mismatch": 1}, {"twin_mismatch": 1}, {"reference": 0}):
            assert not X._verdict(_doc(good[:i] + [good[i] | over] + good[i + 1:])), (i, over)
    low = [r | {"refuted": r["refuted"] | {"warp_pairs": 0}} for r in good]
    low[0] = good[0] | {"refuted": good[0]["refuted"] | {"warp_pairs": X.PER_ALT - 1}}
    assert not X._verdict(_doc(low))
    for key, bad in (("ptx", {"ok": False}), ("bind", {"ok": False}), ("errors", {"compare/x": "boom"}), ("gate", {"ok": False}),
                     ("served", {"ok": False}), ("served", {}),
                     ("launches", [{"case": "served", "checks": {"deterministic": True, "mla_kv_c": True}}]),
                     ("launches", [{"case": c, "checks": {"deterministic": c != "probe", "mla_kv_c": True}} for c in X.CASES])):
        assert not X._verdict(_doc(good) | {key: bad}), key


def test_the_bindings_and_the_record() -> None:
    assert X.bind_check() == {"bindings": {"DsaRMSNorm_v1{N=2048,EPS=1e-05}": True, "DsaRMSNorm_v1{N=512,EPS=1e-05}": True}, "ok": True}
    assert REGISTRY["glm53_fused_rmsnorm_capture"].schema == X.SCHEMA


@pytest.mark.slow
@pytest.mark.parametrize("flip,code,line", [(False, 0, "GLM53-FUSED-RMSNORM GO"), (True, 1, "GLM53-FUSED-RMSNORM NO-GO")])
def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys, flip, code, line) -> None:
    """`main` with the GPU stood in for (the twin's words as the kernel's in a served and a probe launch of 40 tokens, a fake PTX
    in the Triton cache; one word flipped or none): the jobs through the spawned worker pool, the summary, the sealed record, the
    GO / NO-GO line and the exit status.  Slow (about 5 s): the spawned worker imports the evaluator and runs the reference on
    every sampled row at both widths."""
    rec = X.ptx_record(FAKE_PTX)
    monkeypatch.setattr(X, "PTX_CODE_SHA256", rec["code_sha256"])
    monkeypatch.setattr(X, "PTX_COUNTS", rec["counts"])
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X.DS, "environment", lambda work: ({}, {"VLLM_USE_DEEP_GEMM": "0"}))
    monkeypatch.setattr(X.DS, "served", lambda: {"VLLM_USE_DEEP_GEMM": True, "VLLM_BATCH_INVARIANT": False, "ok": True})

    def gpu(work, doc):
        (work / "triton_cache" / "k").mkdir(parents=True)
        (work / "triton_cache" / "k" / "_fused_norm_rope_kernel.ptx").write_text(FAKE_PTX)
        doc["launches"] += [{"case": c, "tokens": 40, "checks": {"deterministic": True, "mla_kv_c": True}} for c in X.CASES]
        return _launch(work, "served", 40, 3, flip=flip) + _launch(work, "probe", 40, 4)

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == code
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is (code == 0) and not doc["errors"] and doc["ptx"]["ok"]
    assert doc["env_cleared"] == {"VLLM_USE_DEEP_GEMM": "0"} and doc["served"]["ok"] is True
    cases = doc["summary"]["cases"]
    assert {k: t["rows"] for k, t in cases.items()} == {f"{c}/{n}": 40 for c in X.CASES for n, *_ in X.NORMS}
    assert cases["served/q_c"]["mismatch"] == int(flip) and sum(t["mismatch"] for t in cases.values()) == int(flip)
    assert all(n >= X.PER_ALT for alt in doc["summary"]["refuted"].values() for n in alt.values())
    printed = capsys.readouterr().out
    assert line in printed and " served=True " in printed and "launches_ok=2/2 ptx=True bind=True" in printed
