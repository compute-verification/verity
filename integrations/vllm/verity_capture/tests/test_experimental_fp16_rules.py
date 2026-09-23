"""[r8 lane E-cleanup 2026-09-18] unit tests for the fp16 CANDIDATE helpers (`verity_capture.experimental.fp16_rules`)."""
from __future__ import annotations

import json

import numpy as np

from verity_capture import analytic as A
from verity_capture.experimental.fp16_rules import (check_cos_sin_words, dtype_refusal_census, f32_to_words, served_dtype_labels,
                                                    words_to_f32)

C = {"H": 576, "NL": 30, "NH": 9, "KVH": 3, "D": 64, "I": 1536, "V": 49152, "ROPE_THETA": 100000.0, "MAX_POS": 512, "EPS": 1e-5}


def _table(dtype: str, C=C) -> np.ndarray:
    inv = A._inv_freq(C)
    t = np.arange(int(C["MAX_POS"]), dtype=np.float32)
    f = (t[:, None] * inv[None, :]).astype(np.float32)
    return np.concatenate([f32_to_words(np.cos(f, dtype=np.float32), dtype), f32_to_words(np.sin(f, dtype=np.float32), dtype)], axis=1)


def test_bfloat16_path_reproduces_the_frozen_check():
    tab = _table("bfloat16")
    frozen = A.check_cos_sin(C, tab)
    cand = check_cos_sin_words(C, tab, "bfloat16")
    assert cand["status"] == frozen["status"] == "pass"
    for k in ("n_diff", "n_outside_bound", "max_ulp", "n_elements", "ulp_histogram", "shape_ok"):
        assert cand[k] == frozen[k], k
    assert tab.dtype == np.uint16 and np.array_equal(tab, A.cos_sin_table(C))


def test_float16_rounding_roundtrips_and_the_float16_table_passes():
    x = np.array([0.0, 1.0, -1.0, 0.333251953125, 3.0517578125e-05, 65504.0], dtype=np.float32)
    w = f32_to_words(x, "float16")
    assert w.dtype == np.uint16 and np.array_equal(words_to_f32(w, "float16"), x)
    ties = np.array([1.0 + 2 ** -11, 1.0 + 3 * 2 ** -11, 1.0 + 2 ** -11 + 2 ** -12], dtype=np.float32)          # f16 ulp at 1.0 = 2^-10
    assert list(f32_to_words(ties, "float16")) == [0x3C00, 0x3C02, 0x3C01]                                       # RNE: ties to even
    r = check_cos_sin_words(C, _table("float16"), "float16")
    assert r["status"] == "pass" and r["n_outside_bound"] == 0 and r["dtype"] == "float16" and r["n_elements"] == 512 * 64


def test_a_wrong_theta_or_layout_fails_in_both_dtypes():
    for dt in ("bfloat16", "float16"):
        wrong = _table(dt, {**C, "ROPE_THETA": 10000.0})
        r = check_cos_sin_words(C, wrong, dt)
        assert r["status"] == "fail" and r["n_outside_bound"] > 0.5 * r["n_elements"], (dt, r["n_outside_bound"])
        swapped = np.concatenate([_table(dt)[:, 32:], _table(dt)[:, :32]], axis=1)          # [sin | cos]
        assert check_cos_sin_words(C, swapped, dt)["status"] == "fail"
    assert check_cos_sin_words(C, _table("float16"), "float32")["status"] == "error"
    assert check_cos_sin_words(C, _table("float16")[:100], "float16")["shape_ok"] is False


def test_refusal_census_separates_guard_messages_from_other_refusals():
    res = {"totals": {"unsupported": 6}, "unsupported_examples": [
        {"op": "_C.silu_and_mul.default", "why": "SiluMul_v1 is bf16"}, {"op": "_C.silu_and_mul.default", "why": "SiluMul_v1 is bf16"},
        {"op": "aten.embedding.default", "why": "embedding table is not a row-major bf16 [V,H] view"},
        {"op": "_C_cache_ops.reshape_and_cache_flash.default", "why": "KV move is bf16 → bf16 only"}]}
    c = dtype_refusal_census(res)
    assert c["all_dtype_refusals"] is True and c["distinct_reasons"] == 3 and c["by_reason"][0]["examples"] == 2
    res["unsupported_examples"].append({"op": "aten.argmax.default", "why": "argmax width is not the vocabulary"})
    c = dtype_refusal_census(res)
    assert c["all_dtype_refusals"] is False and [r["op"] for r in c["non_guard_reasons"]] == ["aten.argmax.default"]
    assert dtype_refusal_census({})["all_dtype_refusals"] is False


def _record(tmp_path, *, spec_dtype, manifest_dtype, served="float16"):
    import gzip
    (tmp_path / "spec.json").write_text(json.dumps({"dtype": spec_dtype, "hf_config": {"torch_dtype": "bfloat16"}}))
    (tmp_path / "run.json").write_text(json.dumps({"workload": {"engine_args_required": {"dtype": served}}, "engine_args_requested": {"dtype": served}}))
    (tmp_path / "capture").mkdir(exist_ok=True)
    (tmp_path / "capture" / "versions.json").write_text(json.dumps({"engine_kwargs": {"dtype": served}}))
    with gzip.open(tmp_path / "capture" / "log.jsonl.gz", "wt") as f:
        f.write(json.dumps({"$e": "RunHeader", "profile": {"dtype": manifest_dtype, "engine_kwargs": {"dtype": manifest_dtype}}}) + "\n")


def test_served_dtype_labels_flag_missing_and_wrong_labels(tmp_path):
    _record(tmp_path, spec_dtype=None, manifest_dtype="bfloat16")                 # the 2026-09-17 fp16 record's state (M12)
    r = served_dtype_labels(tmp_path)
    assert r["served_dtype_requested"] == "float16" and r["consistent"] is False
    assert r["mislabelled"] == ["manifest.dtype", "manifest.engine_kwargs.dtype"] and r["unlabelled"] == ["spec.dtype"]
    _record(tmp_path, spec_dtype="float16", manifest_dtype="float16")             # after experiment.build_spec + profile_manifest fixes
    r = served_dtype_labels(tmp_path)
    assert r["consistent"] is True and r["mislabelled"] == [] and r["unlabelled"] == []
    _record(tmp_path, spec_dtype="bfloat16", manifest_dtype="bfloat16", served=None)   # a bf16 run: nothing requested, nothing flagged
    r = served_dtype_labels(tmp_path)
    assert r["served_dtype_requested"] is None and r["consistent"] is True and r["mislabelled"] == []
