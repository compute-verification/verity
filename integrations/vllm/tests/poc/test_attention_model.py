"""CPU tests for W11's independent FlashAttention-2 model (``verity_vllm_numerics.fa2_model`` /
``fa2_relation``) on the saved fixtures ``fixtures/W11-40f0cebeb670-20260907T1800Z``.

No GPU: every expected output is a BF16 word the production kernel wrote on the RTX 4090 (natural
Qwen2.5-1.5B serving inputs; §11 stress inputs; one-bit mutations).  What is pinned down:

* the MUFU.EX2 / MUFU.RCP tables reconstruct from the shipped deltas to the hashed hardware dumps;
* natural decode and prefill rows, stress families and the mutated inputs reproduce the production
  words bit-for-bit (a sample per family to keep the runtime small; the full counts are in
  ``results/W11-40f0cebeb670-20260907T1800Z``);
* a prefill row checked as a single-query call (keys 0..t, the relation's view) equals the row the
  kernel produced inside the causal Lq x Lk call (the extra fully-masked blocks are no-ops);
* a one-bit flip of a K / V / Q word changes the model's output where it changed production's, and
  the model follows the mutated input;
* the coverage gate: ``independent_attention`` predicts only inside the measured family and raises
  ``AttentionUnresolved`` (``unresolved: ...``) elsewhere, including on non-finite operands.

Needs g++ (or $CXX) for ``cpp/fa2_model.cpp``; skips where it cannot be built.
Run: ``cd vllm-poc && python -m pytest tests/test_attention_model.py -q``
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT.parent / "src"):  # vllm-poc/ and the veritor package (relations.py imports veritor.core.silicon)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FIX = ROOT / "fixtures" / "W11-40f0cebeb670-20260907T1800Z"
os.environ.setdefault("VERITY_MUFU_TABLES", str(FIX / "mufu"))

from verity_vllm_numerics import fa2_model as FA  # noqa: E402
from verity_vllm_numerics import fa2_relation as FR  # noqa: E402

pytestmark = pytest.mark.skipif(not FIX.exists(), reason="fixture directory missing")


def _have_cpp() -> bool:
    try:
        FA.lib()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_cpp = pytest.mark.skipif(not _have_cpp(), reason="cpp/fa2_model.cpp could not be built (no g++)")


@pytest.fixture(scope="module")
def tables() -> FA.MufuTables:
    return FA.MufuTables.load(FIX / "mufu")


@pytest.fixture(scope="module")
def natural() -> dict:
    return json.loads((FIX / "natural" / "index.json").read_text())


@pytest.fixture(scope="module")
def stress() -> dict:
    return json.loads((FIX / "stress" / "index.json").read_text())


def _consts(idx: dict) -> dict:
    c = dict(idx["consts"])
    c.setdefault("gpu", idx.get("gpu"))
    return c


def _load_call(base: Path, c: dict):
    q = np.load(base / c["q_file"])
    out = np.load(base / c["out_file"])
    k = np.load(base / c["k_file"])[: c["Lk"]]
    v = np.load(base / c["v_file"])[: c["Lk"]]
    return q, k, v, out


# --- MUFU tables ---------------------------------------------------------------------------------------


def test_mufu_tables_reconstruct(tables):
    """The int8 deltas rebuild the hardware dumps (sha256 of the raw u32 tables measured on the 4090)."""
    assert tables.ex2.shape == (1 << 23,) and tables.rcp.shape == (1 << 23,)
    assert hashlib.sha256(tables.ex2.astype("<u4").tobytes()).hexdigest() == "b2a42c4a626469c986e33e43f16f41bde9d84de94347cd9ceaa1bd68d36bcdf0"
    assert hashlib.sha256(tables.rcp.astype("<u4").tobytes()).hexdigest() == "c408381446132e07fa5581e930dc33fb790bea7ec4276ef3621e322df2492a55"
    assert int(tables.ex2[0]) == 0x3F800000 and int(tables.rcp[0]) == 0x3F800000


@needs_cpp
def test_mufu_special_values(tables):
    x = np.array([0.0, -0.0, 1.0, -1.0, 23.0, -23.0, 128.0, -150.0, np.inf, -np.inf], np.float32)
    y = FA.mufu_ex2(x, tables)
    assert y[0] == 1.0 and y[1] == 1.0 and y[2] == 2.0 and y[3] == 0.5 and y[4] == 2.0**23 and y[5] == 2.0**-23
    assert np.isinf(y[6]) and y[7] == 0.0 and np.isinf(y[8]) and y[9] == 0.0
    r = FA.mufu_rcp(np.array([1.0, 2.0, 4.0, 0.5, 3.0], np.float32), tables)
    assert r[0] == 1.0 and r[1] == 0.5 and r[2] == 0.25 and r[3] == 2.0
    assert abs(r[4] - 1 / 3) <= 2 * np.finfo(np.float32).eps


# --- natural fixture ------------------------------------------------------------------------------------


@needs_cpp
def test_natural_decode_rows_exact(natural, tables):
    base = FIX / "natural"
    consts = _consts(natural)
    H, KVH, D = consts["num_heads"], consts["num_kv_heads"], consts["head_size"]
    scale = float(consts["softmax_scale"])
    decode = [c for c in natural["calls"] if c["phase"] == "decode"]
    picks = decode[:: max(1, len(decode) // 24)][:24]
    checked = 0
    for c in picks:
        assert FA.coverage(c["Lq"], c["Lk"], H, KVH, D, consts) is None
        q, k, v, out = _load_call(base, c)
        res = FA.attention_call(q, k, v, np.asarray(c["rows"]), c["Lq"], c["Lk"], scale, tables)
        assert not res.nonfinite.any()
        assert np.array_equal(res.out, out), f"decode seq {c['seq']} L{c['layer']} Lk={c['Lk']}"
        checked += out.size
    assert checked >= 24 * H * D


@needs_cpp
def test_natural_prefill_rows_exact(natural, tables):
    base = FIX / "natural"
    consts = _consts(natural)
    scale = float(consts["softmax_scale"])
    prefill = [c for c in natural["calls"] if c["phase"] == "prefill"]
    assert prefill
    for c in prefill[:3]:
        q, k, v, out = _load_call(base, c)
        rows = np.asarray(c["rows"])[:40]
        res = FA.attention_call(q[:40], k, v, rows, c["Lq"], c["Lk"], scale, tables)
        assert np.array_equal(res.out, out[:40]), f"prefill seq {c['seq']} Lq={c['Lq']} Lk={c['Lk']}"


@needs_cpp
def test_prefill_row_equals_single_query_call(natural, tables):
    """The relation's view of a prefill row (one query, keys 0..t) is bit-identical to the row the kernel
    computed inside the causal Lq x Lk call: the extra fully-masked blocks are arithmetic no-ops."""
    base = FIX / "natural"
    consts = _consts(natural)
    scale = float(consts["softmax_scale"])
    c = next(c for c in natural["calls"] if c["phase"] == "prefill" and c["Lq"] >= 300)
    q, k, v, out = _load_call(base, c)
    rows = list(c["rows"])
    for i in [0, 1, len(rows) // 2, len(rows) - 2, len(rows) - 1]:
        r = rows[i]
        t = r + c["Lk"] - c["Lq"]
        single = FA.attention_call(q[i : i + 1], k[: t + 1], v[: t + 1], np.array([0]), 1, t + 1, scale, tables)
        assert np.array_equal(single.out[0], out[i]), f"row {r} (t={t})"


# --- stress fixture -------------------------------------------------------------------------------------


@needs_cpp
def test_stress_families_exact(stress, tables):
    base = FIX / "stress"
    consts = _consts(stress)
    H, KVH, D = consts["num_heads"], consts["num_kv_heads"], consts["head_size"]
    scale = float(consts["softmax_scale"])
    seen = {}
    for c in stress["calls"]:
        fam = c["family"]
        if seen.get(fam, 0) >= 3:
            continue
        q, k, v, out = _load_call(base, c)
        if FA.nonfinite_words(q, k, v):
            continue  # gated: covered by test_nonfinite_operands_unresolved
        assert FA.coverage(c["Lq"], c["Lk"], H, KVH, D, consts) is None, c["case"]
        rows = np.asarray(c["rows"])
        res = FA.attention_call(q, k, v, rows, c["Lq"], c["Lk"], scale, tables)
        finite = ~res.nonfinite
        if not finite.all():
            assert fam == "extremes", c["case"]  # v near BF16 max: FP32 overflow -> unresolved rows
        assert np.array_equal(res.out[finite], out[finite]), c["case"]
        seen[fam] = seen.get(fam, 0) + 1
    assert set(seen) >= {"zeros", "cancellation", "rounding_boundaries", "subnormals", "max_shifts", "near_uniform", "tile_boundaries", "beyond_natural_range"}


@needs_cpp
def test_cast_ties_land_on_bf16_ties(stress, tables):
    """Q = 0, Lk = 2: every pre-cast FP32 value is x.5 with 8 significant bits (an exact BF16 tie) and RNE picks
    the even word, as production did."""
    base = FIX / "stress"
    consts = _consts(stress)
    c = next(c for c in stress["calls"] if c["case"] == "cast_ties_n2")
    q, k, v, out = _load_call(base, c)
    res = FA.attention_call(q, k, v, np.array([0]), 1, 2, float(consts["softmax_scale"]), tables)
    f32 = res.out_f32.reshape(-1).view(np.float32)
    assert np.all(f32 * 2 == np.round(f32 * 2)) and np.all(f32 != np.round(f32))  # exactly half-integers
    assert np.all((res.out.reshape(-1) & 1) == 0)  # RNE: ties to even
    assert np.array_equal(res.out, out)


@needs_cpp
def test_mutations(stress, tables):
    base = FIX / "stress"
    consts = _consts(stress)
    scale = float(consts["softmax_scale"])
    muts = json.loads((base / "mutations.json").read_text())
    by_seq = {c["seq"]: c for c in stress["calls"]}
    n_rej = n_track = n_done = 0
    for m in muts:
        if m.get("base_fixture"):
            continue  # natural-fixture mutations: same code path, kept out of the unit test for time
        c = by_seq[m["seq"]]
        q, k, v, _ = _load_call(base, c)
        rows = np.asarray(m["rows"])
        prod_mut = np.load(base / m["production_out_mutated_file"])
        which, idx = m["operand"], tuple(m["index"])
        qm, km, vm = q.copy(), k.copy(), v.copy()
        {"q": qm, "k": km, "v": vm}[which][idx] = m["new_word"]  # q is indexed in the saved-row space
        if FA.nonfinite_words(qm, km, vm):
            with pytest.raises(FR.AttentionUnresolved):
                FR.independent_attention(_node(consts), {"q": qm[0].reshape(-1).astype("<u2").tobytes(), "k": km, "v": vm})
            continue
        orig = FA.attention_call(q, k, v, rows, c["Lq"], c["Lk"], scale, tables).out
        mut_res = FA.attention_call(qm, km, vm, rows, c["Lq"], c["Lk"], scale, tables)
        if mut_res.nonfinite.any():  # e.g. an exponent-bit flip drives QK^T past FP32 max: the relation declines
            if c["Lq"] == 1:
                with pytest.raises(FR.AttentionUnresolved, match="non-finite intermediate"):
                    FR.independent_attention(_node(consts), {"q": qm[0].reshape(-1).astype("<u2").tobytes(), "k": km, "v": vm})
            continue
        mut = mut_res.out
        assert np.array_equal(mut, prod_mut), f"model does not follow the mutation {m['case']} {which}{list(idx)} bit {m['bit']}"
        if m["production_output_words_changed"]:
            assert (orig != prod_mut).any(), "mutated production output would have been accepted"
            n_rej += 1
        n_track += 1
        n_done += 1
        if n_done >= 40:
            break
    assert n_done >= 30 and n_rej >= 15


# --- the relation and its gate ---------------------------------------------------------------------------


def _node(consts: dict, **over) -> dict:
    c = {"num_heads": consts["num_heads"], "num_kv_heads": consts["num_kv_heads"], "head_size": consts["head_size"],
         "block_size": consts.get("block_size", 16), "fa_version": 2, "num_splits": 1, "sliding_window": [-1, -1],
         "logits_soft_cap": 0, "kv_cache_dtype": "auto", "scale": consts["softmax_scale"], "gpu": consts.get("gpu", "NVIDIA GeForce RTX 4090")}
    c.update(over)
    return {"op_type": "attention", "extra": {"consts": c}}


def _rows_as_values(arr: np.ndarray) -> list[bytes]:
    return [row.reshape(-1).astype("<u2").tobytes() for row in arr]


@needs_cpp
def test_relation_predicts_natural_decode_row(natural):
    base = FIX / "natural"
    consts = _consts(natural)
    c = next(c for c in natural["calls"] if c["phase"] == "decode" and 100 < c["Lk"] < 400)
    q, k, v, out = _load_call(base, c)
    pred = FR.independent_attention(_node(consts), {"q": q[0].reshape(-1).astype("<u2").tobytes(), "k": _rows_as_values(k), "v": _rows_as_values(v)})
    assert pred == out[0].reshape(-1).astype("<u2").tobytes()
    cov = FR.coverage(_node(consts), {"q": q[0].reshape(-1).astype("<u2").tobytes(), "k": k, "v": v})
    assert cov["covered"] and cov["n"] == c["Lk"]


@needs_cpp
def test_relation_arg_cols_slice(natural):
    """W1 records the V input as a column slice of the qkv add output (arg_cols v = [1792, 2048])."""
    base = FIX / "natural"
    consts = _consts(natural)
    c = next(c for c in natural["calls"] if c["phase"] == "decode" and c["Lk"] < 64)
    q, k, v, out = _load_call(base, c)
    KVH, D = consts["num_kv_heads"], consts["head_size"]
    wide = [np.concatenate([np.zeros(1792, np.uint16), row.reshape(-1)]).astype("<u2").tobytes() for row in v]
    node = _node(consts)
    node["extra"]["arg_cols"] = {"v": [1792, 1792 + KVH * D]}
    pred = FR.independent_attention(node, {"q": q[0].reshape(-1).astype("<u2").tobytes(), "k": _rows_as_values(k), "v": wide})
    assert pred == out[0].reshape(-1).astype("<u2").tobytes()


def test_gate_unresolved_outside_measured_family(natural):
    consts = _consts(natural)
    H, KVH, D = consts["num_heads"], consts["num_kv_heads"], consts["head_size"]
    assert FA.coverage(1, 100, H, KVH, D, consts) is None
    assert "head_dim" in FA.coverage(1, 100, 9, 3, 64, consts)
    assert "num_splits" in FA.coverage(1, 100, H, KVH, D, {**consts, "num_splits": 4})
    assert "sliding window" in FA.coverage(1, 100, H, KVH, D, {**consts, "window_size": [64, 0]})
    assert "softcap" in FA.coverage(1, 100, H, KVH, D, {**consts, "softcap": 30.0})
    assert "seqlen_k" in FA.coverage(1, 4097, H, KVH, D, consts)
    assert "seqlen_q" in FA.coverage(200, 100, H, KVH, D, consts)
    assert "not measured" in FA.coverage(1, 100, 32, 8, D, consts)
    assert "block_size" in FA.coverage(1, 100, H, KVH, D, {**consts, "block_size": 32})
    saved_env = os.environ.pop("VERITY_ARCH", None)
    try:
        assert "arch not declared" in FA.coverage(1, 100, H, KVH, D, {**consts, "gpu": None, "arch": None})
    finally:
        if saved_env is not None:
            os.environ["VERITY_ARCH"] = saved_env
    assert "arch sm_80" in FA.coverage(1, 100, H, KVH, D, {**consts, "arch": "sm_80"})
    assert "kv_cache_dtype" in FA.coverage(1, 100, H, KVH, D, {**consts, "kv_cache_dtype": "fp8"})


def test_gate_relation_raises_unresolved(natural):
    consts = _consts(natural)
    H, KVH, D = consts["num_heads"], consts["num_kv_heads"], consts["head_size"]
    q = np.zeros(H * D, np.uint16).tobytes()
    k = np.zeros((5, KVH * D), np.uint16)
    with pytest.raises(FR.AttentionUnresolved, match="unresolved: num_splits"):
        FR.independent_attention(_node(consts, num_splits=2), {"q": q, "k": k, "v": k})
    with pytest.raises(FR.AttentionUnresolved, match="unresolved: arch sm_80"):
        FR.independent_attention(_node(consts, arch="sm_80"), {"q": q, "k": k, "v": k})
    with pytest.raises(FR.AttentionRelationError, match="K rows but"):
        FR.operands(_node(consts), {"q": q, "k": k, "v": k[:4]})


@needs_cpp
def test_nonfinite_operands_unresolved(stress):
    base = FIX / "stress"
    consts = _consts(stress)
    for name in ("q_inf_n64", "k_nan_n64"):
        c = next(c for c in stress["calls"] if c["case"] == name)
        q, k, v, out = _load_call(base, c)
        assert FA.nonfinite_words(q, k, v) == 1
        with pytest.raises(FR.AttentionUnresolved, match="non-finite operand"):
            FR.independent_attention(_node(consts), {"q": q[0].reshape(-1).astype("<u2").tobytes(), "k": k, "v": v})
    # FP32 overflow inside the chain (V near BF16 max): production wrote inf words; the model declines
    c = next(c for c in stress["calls"] if c["case"] == "v_near_max_n64")
    q, k, v, out = _load_call(base, c)
    with pytest.raises(FR.AttentionUnresolved, match="non-finite intermediate"):
        FR.independent_attention(_node(consts), {"q": q[0].reshape(-1).astype("<u2").tobytes(), "k": k, "v": v})


def test_registered_with_w2():
    try:
        from verity_vllm import relations as R
    except ImportError:
        pytest.skip("W2 registry not importable")
    rel = R.REGISTRY.get("attention.bf16.v1")
    assert rel is not None and rel.implemented and rel.klass == R.INDEPENDENT and rel.owner == "W11"
