"""The torch-free Gumbel-max recomputation (`bench/gen_sampling_rng.py`) and the sampling workloads."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.config import RowEngine, workload_file
from verity_vllm.engine import static_schedule
from verity_vllm.program import prefix_plan as PP
from verity_vllm.program import sampling_rng as R

PROFILES = Path(__file__).resolve().parents[2] / "workloads"
WORKLOADS = ["workload_b0_sampling_temp07", "workload_b0_sampling_topk50_topp09", "workload_b0_sampling_n2", "workload_b0_sampling_2req_seeds"]


# --- Philox4x32-10: Random123 known-answer vectors (tests/kat_vectors; Triton's philox_impl is the same permutation) ---

@pytest.mark.parametrize("ctr,key,expect", [
    ((0, 0, 0, 0), (0, 0), "6627e8d5 e169c58d bc57ac4c 9b00dbd8"),
    ((0xffffffff,) * 4, (0xffffffff,) * 2, "408f276d 41c83b0e a20bc7c6 6d5451fd"),
    ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0), "d16cfe09 94fdcceb 5001e420 24126ea1"),
])
def test_philox_known_answers(ctr, key, expect):
    out = R.philox4x32_10(*[np.array([c], dtype=np.uint32) for c in ctr], key[0], key[1])
    assert " ".join(f"{int(x[0]):08x}" for x in out) == expect


def test_randint_is_philox_of_seed_key_and_offset_counter():
    # seed 0, offset 0 == the all-zero KAT's first word; the offset is the counter's low word, the seed the key
    assert int(R.tl_randint(0, np.array([0], dtype=np.int64))[0]) == 0x6627E8D5
    a = R.tl_randint(0, np.arange(8, dtype=np.int64))
    b = R.tl_randint(1, np.arange(8, dtype=np.int64))
    assert a.dtype == np.uint32 and len(set(a.tolist())) == 8 and not np.array_equal(a, b)
    # negative int64 seeds are reinterpreted as uint64 (key hi word != 0) and differ from the 32-bit-truncated key
    neg = R.tl_randint(-1, np.array([5], dtype=np.int64))[0]
    trunc = R.tl_randint(0xFFFFFFFF, np.array([5], dtype=np.int64))[0]
    assert int(neg) != int(trunc)
    # offsets above 2**32 use the counter's high word
    assert int(R.tl_randint(0, np.array([2**33 + 1], dtype=np.int64))[0]) != int(R.tl_randint(0, np.array([1], dtype=np.int64))[0])


def test_rand_uniform_fp32_in_unit_interval():
    u = R.tl_rand(7, np.arange(4096, dtype=np.int32))
    assert u.dtype == np.float32 and float(u.min()) >= 0.0 and float(u.max()) < 1.0
    assert abs(float(u.mean()) - 0.5) < 0.03
    assert np.array_equal(u, R.tl_rand(7, np.arange(4096, dtype=np.int32)))     # pure function


def test_gumbel_noise_stream_keys_and_distribution():
    V = 49152
    g = R.gumbel_noise(0, 32, V)
    assert g.dtype == np.float32 and g.shape == (V,)
    assert abs(float(g.mean()) - 0.5772) < 0.03      # Euler-Mascheroni: standard Gumbel mean
    # the stream key is (seed, position): a different position or a different seed (the n=2 child: seed + 1) is a new draw
    assert R.gumbel_seed_of(0, 32) != R.gumbel_seed_of(0, 33) != R.gumbel_seed_of(1, 33)
    assert not np.array_equal(g, R.gumbel_noise(0, 33, V)) and not np.array_equal(g, R.gumbel_noise(1, 32, V))
    assert np.array_equal(g, R.gumbel_noise(0, 32, V))
    # the same vocab index draws the same noise whatever the vocab size (keys = vocab index)
    assert np.array_equal(g[:1000], R.gumbel_noise(0, 32, 1000))
    # drafting salt moves the stream
    assert R.gumbel_seed_of(0, 32, is_drafting=True) == R.gumbel_seed_of(0, 32 + (1 << 30))


def test_noised_argmax_tie_break_and_margin():
    logits = np.array([1.0, 3.0, 3.0, 2.0], dtype=np.float32)
    r = R.noised_argmax(logits, np.zeros(4, dtype=np.float32))
    assert r["token"] == 1 and r["runner_up"] == 2 and r["margin"] == 0.0 and r["margin_ulps"] == 0.0
    r = R.noised_argmax(logits, np.array([0.0, 0.0, 0.0, 5.0], dtype=np.float32))
    assert r["token"] == 3 and r["margin"] == pytest.approx(4.0) and r["margin_ulps"] > 1e6
    r = R.noised_argmax(np.array([0.7, 1.4], dtype=np.float32), np.zeros(2, dtype=np.float32), temperature=0.7)
    assert r["token"] == 1 and r["value"] == pytest.approx(2.0)


def test_sampled_token_paths():
    rng = np.random.default_rng(0)
    logits = (rng.standard_normal(49152) * 4).astype(np.float32)
    greedy = R.sampled_token(logits, 0, 32, temperature=0.0, apply_temperature=False)
    assert greedy["token"] == int(np.argmax(logits))
    a = R.sampled_token(logits, 0, 32, temperature=0.7, apply_temperature=True)          # kernel divides
    b = R.sampled_token((logits / np.float32(0.7)).astype(np.float32), 0, 32, temperature=0.7, apply_temperature=False)   # pre-divided
    assert a["token"] == b["token"] and a["gumbel_seed"] == b["gumbel_seed"] == R.gumbel_seed_of(0, 32)
    c = R.sampled_token(logits, 0, 33, temperature=0.7, apply_temperature=True)
    assert 0 <= c["token"] < 49152 and c["margin"] >= 0.0
    d = R.sampled_token(logits, 0, 32, temperature=1.0, apply_temperature=False, use_fp64=True)
    assert 0 <= d["token"] < 49152


# --- the workloads + `load_workload`'s sampling parameters (the adapter imports torch; exec the pure function instead) ---

def _adapter_fn(name: str):
    src = (Path(__file__).resolve().parents[2] / "verity_vllm" / "engine" / "vllm_adapter.py").read_text()
    tree = ast.parse(src)
    ns: dict = {"json": json, "Path": Path, "hashlib": hashlib, "Any": object, "RowEngine": RowEngine, "PP": PP,
                "static_schedule": static_schedule, "workload_file": workload_file}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "ACCEPTED_TOKENS":
            exec(compile(ast.Module([node], []), "vllm_adapter", "exec"), ns)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (name, "execution_of_workload"):
            exec(compile(ast.Module([node], []), "vllm_adapter", "exec"), ns)
    return ns[name]


@pytest.mark.parametrize("name", WORKLOADS)
def test_sampling_workloads_schema(name):
    doc = json.loads((PROFILES / f"{name}.json").read_text())
    assert doc["sampling"]["variant"] in doc["variants"]
    assert "greedy" in doc["variants"] and doc["variants"]["greedy"]["temperature"] == 0.0
    assert all(len(r["prompt_token_ids"]) == 32 and r["max_tokens"] == 16 and "accepted_token_ids" not in r for r in doc["requests"])
    ids = [r["request_id"] for r in doc["requests"]]
    assert len(set(ids)) == len(ids)


def test_load_workload_honours_sampling_block_and_keeps_greedy_default():
    load = _adapter_fn("load_workload")
    g = load(PROFILES / "workload_32x16_1req.json")
    assert g["sampling"]["greedy"] and g["sampling"]["seed"] is None and g["sampling"]["n"] == 1 and g["sampling"]["variant"] == "greedy"
    assert g["requests"][0]["seed"] is None and g["requests"][0]["n"] == 1
    t = load(PROFILES / "workload_b0_sampling_temp07.json")
    assert t["sampling"] == {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "logprobs": None, "seed": 0, "n": 1, "variant": "temp07", "greedy": False}
    assert t["requests"][0]["seed"] == 0 and t["accepted"].get(t["requests"][0]["request_id"]) is None
    k = load(PROFILES / "workload_b0_sampling_topk50_topp09.json")["sampling"]
    assert (k["temperature"], k["top_p"], k["top_k"], k["seed"]) == (1.0, 0.9, 50, 0)
    n2 = load(PROFILES / "workload_b0_sampling_n2.json")
    assert n2["sampling"]["n"] == 2 and n2["requests"][0]["n"] == 2 and n2["requests"][0]["seed"] == 0
    s2 = load(PROFILES / "workload_b0_sampling_2req_seeds.json")
    assert s2["sampling"]["seed"] is None and [r["seed"] for r in s2["requests"]] == [0, 1]


def test_load_workload_rejects_unknown_variant(tmp_path):
    load = _adapter_fn("load_workload")
    doc = json.loads((PROFILES / "workload_b0_sampling_temp07.json").read_text())
    doc["sampling"]["variant"] = "nope"
    p = tmp_path / "w.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(KeyError):
        load(p)


def test_sampling_params_uses_request_seed_and_n(monkeypatch):
    fn = _adapter_fn("sampling_params")
    calls = []

    class SP:
        def __init__(self, **kw):
            calls.append(kw)
    import sys
    import types
    fake = types.ModuleType("vllm")
    fake.SamplingParams = SP
    monkeypatch.setitem(sys.modules, "vllm", fake)
    fn({"max_tokens": 16, "seed": 3, "n": 2}, {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "seed": 0, "n": 1})
    fn({"max_tokens": 16}, {"temperature": 0.0, "top_p": 1.0, "top_k": 0})
    assert calls[0]["seed"] == 3 and calls[0]["n"] == 2 and calls[0]["temperature"] == 0.7
    assert calls[1]["seed"] is None and calls[1]["n"] == 1 and calls[1]["max_tokens"] == 16
