"""[ov-sampling] the B1 (Qwen2.5-1.5B) sampling workloads and the per-request variant (mixed batch) of the SHARED-CODE
`load_workload` / `sampling_params` change.  The adapter imports torch, so the pure functions are exec'd from source."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from tests.observe.test_gen_sampling import _adapter_fn

PROFILES = Path(__file__).resolve().parents[2] / "workloads"
B1 = ["temp07", "topk50", "topp09", "topk50_topp09", "n2", "2req_seeds", "noseed", "combo", "mixed"]


@pytest.mark.parametrize("name", B1)
def test_b1_sampling_workloads_schema(name):
    doc = json.loads((PROFILES / f"workload_b1_sampling_{name}.json").read_text())
    assert doc["case"] == "B1" and doc["model"] == "Qwen/Qwen2.5-1.5B"
    assert doc["sampling"]["variant"] in doc["variants"] and doc["variants"]["greedy"]["temperature"] == 0.0
    base = json.loads((PROFILES / "workload_qwen15_32x16_1req.json").read_text())["requests"][0]["prompt_token_ids"]
    for r in doc["requests"]:
        assert r["prompt_token_ids"] == base and r["max_tokens"] == 16 and "accepted_token_ids" not in r
        if "variant" in r:
            assert r["variant"] in doc["variants"]
    ids = [r["request_id"] for r in doc["requests"]]
    assert len(set(ids)) == len(ids)


def test_b1_workloads_load_to_the_intended_sampling():
    load = _adapter_fn("load_workload")
    t = load(PROFILES / "workload_b1_sampling_temp07.json")["sampling"]
    assert (t["temperature"], t["top_p"], t["top_k"], t["seed"], t["n"], t["greedy"]) == (0.7, 1.0, 0, 0, 1, False)
    k = load(PROFILES / "workload_b1_sampling_topk50.json")["sampling"]
    assert (k["temperature"], k["top_p"], k["top_k"], k["seed"]) == (1.0, 1.0, 50, 0)
    p = load(PROFILES / "workload_b1_sampling_topp09.json")["sampling"]
    assert (p["temperature"], p["top_p"], p["top_k"], p["seed"]) == (1.0, 0.9, 0, 0)
    c = load(PROFILES / "workload_b1_sampling_combo.json")["sampling"]
    assert (c["temperature"], c["top_p"], c["top_k"], c["seed"]) == (0.7, 0.9, 50, 12345)
    n2 = load(PROFILES / "workload_b1_sampling_n2.json")
    assert n2["sampling"]["n"] == 2 and n2["requests"][0]["n"] == 2 and n2["requests"][0]["seed"] == 0
    s2 = load(PROFILES / "workload_b1_sampling_2req_seeds.json")
    assert s2["sampling"]["seed"] is None and [r["seed"] for r in s2["requests"]] == [0, 1]
    ns = load(PROFILES / "workload_b1_sampling_noseed.json")
    assert ns["sampling"]["seed"] is None and ns["requests"][0]["seed"] is None and not ns["sampling"]["greedy"]
    # a non-greedy workload never inherits accepted tokens
    for name in B1:
        w = load(PROFILES / f"workload_b1_sampling_{name}.json")
        assert all(w["accepted"].get(r["request_id"]) is None for r in w["requests"])


def test_mixed_batch_per_request_variant():
    load = _adapter_fn("load_workload")
    w = load(PROFILES / "workload_b1_sampling_mixed.json")
    r0, r1 = w["requests"]
    # request 0 pins the greedy variant: its own temperature / top_p / top_k travel with the request record
    assert r0["variant"] == "greedy" and (r0["temperature"], r0["top_p"], r0["top_k"]) == (0.0, 1.0, 0)
    assert r0["seed"] == 0                                         # inherits the workload seed (irrelevant for a T=0 row)
    # request 1 has no override: no per-request params, the workload's variant (temp1, seed 0) applies
    assert "temperature" not in r1 and r1["seed"] == 0
    assert w["sampling"]["variant"] == "temp1" and w["sampling"]["temperature"] == 1.0
    assert w["sampling"]["greedy"] is False                        # one sampled row makes the run non-greedy
    assert w["sampling"]["per_request"] == [{"request_id": r0["request_id"], "variant": "greedy", "temperature": 0.0,
                                             "top_p": 1.0, "top_k": 0, "seed": 0}]
    # the existing greedy workloads are unchanged: no per_request key, greedy True
    g = load(PROFILES / "workload_qwen15_32x16_1req.json")
    assert "per_request" not in g["sampling"] and g["sampling"]["greedy"] is True


def test_mixed_batch_rejects_unknown_request_variant(tmp_path):
    load = _adapter_fn("load_workload")
    doc = json.loads((PROFILES / "workload_b1_sampling_mixed.json").read_text())
    doc["requests"][0]["variant"] = "nope"
    p = tmp_path / "w.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(KeyError):
        load(p)


def test_sampling_params_prefers_the_request_params(monkeypatch):
    fn = _adapter_fn("sampling_params")
    calls = []

    class SP:
        def __init__(self, **kw):
            calls.append(kw)
    fake = types.ModuleType("vllm")
    fake.SamplingParams = SP
    monkeypatch.setitem(sys.modules, "vllm", fake)
    samp = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 0, "n": 1}
    fn({"max_tokens": 16, "seed": None, "n": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 0}, samp)   # the pinned greedy workload
    fn({"max_tokens": 16, "seed": 0, "n": 1}, samp)                                                     # the sampled workload
    fn({"max_tokens": 16, "seed": 7, "n": 1, "temperature": 0.7, "top_p": 0.9, "top_k": 50}, samp)
    assert (calls[0]["temperature"], calls[0]["seed"]) == (0.0, 0)          # no request seed -> the workload's seed
    assert (calls[1]["temperature"], calls[1]["seed"], calls[1]["top_k"]) == (1.0, 0, 0)
    assert (calls[2]["temperature"], calls[2]["top_p"], calls[2]["top_k"], calls[2]["seed"]) == (0.7, 0.9, 50, 7)


# ---------------------------------------------------------------------------------------------------------
# registry Definition `GumbelArgmaxBf16_v1` (torch-free): the circuit against the NumPy libdevice oracle
# ---------------------------------------------------------------------------------------------------------


def _program_output_gate(P):
    """The composite returns `step[0][1]`: the `idx` slot of the last scan step's state (y, idx, count)."""
    steps = [c for c in P.circuit.root_calls() if not c.is_primitive and c.fn.name == "GumbelArgmaxStep"]
    return steps[-1].returns[1]


def _evaluate_gumbel(V: int, seed: int, pos: int, words: list[int]) -> int:
    from verity.ml.circuits import Program, bind
    from verity.ml.circuits.evaluate import evaluate_region
    from verity.ml.definitions.registry.sampling import GumbelArgmaxBf16
    P = Program(bind(GumbelArgmaxBf16, V=V, SEED=seed, POS=pos))
    C = P.circuit
    gates = [g for _, _, g in P.input_coll(0).refs]
    assert len(gates) == V
    t = dict(zip(gates, words))
    evaluate_region(C, C[0:C.size], t)
    return int(t[_program_output_gate(P)])


def _oracle(V: int, seed: int, pos: int, words: list[int]) -> int:
    import numpy as np
    from verity_vllm.program.libdevice_sampling import gumbel_noise_exact
    logits = (np.asarray(words, dtype=np.uint32) << 16).view(np.float32)
    y = logits + gumbel_noise_exact(seed, pos, V)      # f32 add, round-to-nearest (the kernel's `logits + gumbel`)
    return int(np.argmax(y))                            # lowest index on ties, like the kernel's block-max scan


@pytest.mark.parametrize("seed,pos", [(0, 0), (12345, 3), (2**63 - 1, 17), (0xDEADBEEF, 1)])
def test_gumbel_argmax_definition_matches_the_libdevice_oracle(seed, pos):
    import numpy as np
    rng = np.random.default_rng(seed % 1000 + pos)
    V = 96
    mag = rng.integers(0x3E00, 0x4100, size=V)                          # bf16 magnitudes in [0.125, 4)
    words = [int(w) | (0x8000 if s else 0) for w, s in zip(mag, rng.random(V) < 0.3)]   # ~30% negative
    assert _evaluate_gumbel(V, seed, pos, words) == _oracle(V, seed, pos, words)


def test_gumbel_argmax_definition_is_a_function_of_the_seed_and_position():
    words = [0x3F80] * 32                                              # all-equal logits: the noise alone decides
    a, b, c = (_evaluate_gumbel(32, s, p, words) for s, p in [(1, 0), (2, 0), (1, 1)])
    assert a == _oracle(32, 1, 0, words) and b == _oracle(32, 2, 0, words) and c == _oracle(32, 1, 1, words)
    assert len({a, b, c}) > 1


def test_gumbel_lane_noise_is_bit_exact_against_the_gpu_probe():
    """The pod probe dumped `_gumbel_sample_kernel`'s noise (tl.rand -> log1pf/logf) for seeds/positions of the B1 runs;
    the lane's NumPy transcription and the Definition's per-lane primitive must reproduce every word."""
    import numpy as np
    from verity_vllm.program.libdevice_sampling import gumbel_noise_exact
    from verity.ml.definitions.registry.sampling import GumbelNoiseLane, GumbelStreamKey
    probe = Path(__file__).resolve().parents[2] / "out/gen/lanes/ov-sampling/runs/probe"
    cmp = probe / "compare.json"
    if not cmp.exists():
        pytest.skip("probe artifacts not fetched")
    rep = json.loads(cmp.read_text())
    assert rep["all_bit_exact"] is True, rep
    dumps = sorted(probe.glob("*.npz"))
    if not dumps:
        pytest.skip("probe .npz not fetched (compare.json is the recorded verdict)")
    d = np.load(dumps[0])
    seed, pos = int(d["seed"]), int(d["pos"])
    gpu = np.asarray(d["gumbel"], dtype=np.float32)
    ours = gumbel_noise_exact(seed, pos, gpu.shape[0])
    assert np.array_equal(gpu.view(np.uint32), ours.view(np.uint32))
    key = GumbelStreamKey.evaluate(seed & ((1 << 64) - 1), pos & ((1 << 64) - 1))
    for v in (0, 1, 7, gpu.shape[0] - 1):
        assert GumbelNoiseLane.evaluate(key, v) == int(gpu.view(np.uint32)[v])


# ---------------------------------------------------------------------------------------------------------
# recognizer `sample.gumbel_argmax` on the synthetic gumbel decode step (SmolLM2 geometry, pinned profile + ours)
# ---------------------------------------------------------------------------------------------------------


def _lane_ctx(events, start, sampling, seed=None, sample=None):
    import dataclasses
    from verity_vllm.observe.fold.resolver import StepContext
    from tests.observe import test_patterns_synthetic as S
    from verity_vllm.observe.fold.profiles.gen_ov_sampling_patterns import ov_sampling_patterns
    hdr = S.header(sampling)
    if seed is not None:
        hdr.workload["requests"][0]["seed"] = seed
    prof = dataclasses.replace(S.PROF.PROFILE, patterns=ov_sampling_patterns(S.C) + list(S.PROF.PROFILE.patterns))
    roots: dict = {}
    for r in S.weights_roots():
        roots.setdefault(r.view.alloc, []).append(r)
    return StepContext(hdr, start, sample, [], roots, S.C, prof)


SAMPLED = {"greedy": False, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "logprobs": None, "seed": None}


def test_seeded_gumbel_row_is_one_gumbel_argmax_compute():
    from verity_vllm.observe.fold.resolver import Compute, PerRow, Unsupported, resolve_step
    from tests.observe import test_patterns_synthetic as S
    from verity_vllm.observe.fold.tree import step_events
    events, start = S.decode_step(sampler="gumbel")
    se = step_events(events)
    res = resolve_step(se, _lane_ctx(events, start, SAMPLED, seed=4242))
    assert not [r for r in res if isinstance(r, Unsupported)], [r for r in res if isinstance(r, Unsupported)]
    cs = [m for r in res for m in S.members(r) if isinstance(m, Compute) and m.spec.name == "GumbelArgmaxBf16"]
    assert len(cs) == 1
    c = cs[0]
    assert c.spec.id == f"GumbelArgmaxBf16_v1{{V={S.V},SEED=4242,POS=32}}"
    assert c.operands == (PerRow(S.bf(310, (1, S.V))),) and c.outputs == (PerRow(S.i64(311, (1, 1))),)
    assert c.pattern == "ov_sampling.gumbel_token_sample" and c.attribution == (("r0", 32),)
    assert "_gumbel_sample_kernel" in c.kernels
    by_op = {root.op: r for root, r in zip(se.roots, res)}
    assert by_op["vllm.v1.worker.gpu.sample.gumbel._gumbel_sample_kernel"].effect == "noop"
    assert by_op["aten.argmax.default"].effect == "noop"
    assert "TokenSelect_v1" not in " ".join(S.by_spec(res))


def test_greedy_rows_are_left_to_the_pinned_token_select():
    from verity_vllm.observe.fold.resolver import Unsupported, resolve_step
    from tests.observe import test_patterns_synthetic as S
    from verity_vllm.observe.fold.tree import step_events
    events, start = S.decode_step(sampler="gumbel")
    se = step_events(events)
    res = resolve_step(se, _lane_ctx(events, start, S.GREEDY, sample=se.sample))
    assert not [r for r in res if isinstance(r, Unsupported)]
    assert list(S.by_spec(res)) .count(f"TokenSelect_v1{{V={S.V}}}") == 1
    assert not [k for k in S.by_spec(res) if k.startswith("GumbelArgmax")]


def test_unseeded_sampled_row_is_unsupported_not_a_constant():
    from verity_vllm.observe.fold.resolver import Unsupported, resolve_step
    from tests.observe import test_patterns_synthetic as S
    from verity_vllm.observe.fold.tree import step_events
    events, start = S.decode_step(sampler="gumbel")
    se = step_events(events)
    res = resolve_step(se, _lane_ctx(events, start, SAMPLED, seed=None))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert uns and uns[0].op.endswith("_gumbel_sample_kernel") and "unseeded" in uns[0].reason
    assert not [k for k in S.by_spec(res) if k.startswith("GumbelArgmax") or k.startswith("TokenSelect")]


@pytest.mark.parametrize("sampling,needle", [
    ({**SAMPLED, "temperature": 0.7}, "temperature"),
    ({**SAMPLED, "top_k": 50}, "top_k/top_p"),
    ({**SAMPLED, "top_p": 0.9}, "top_k/top_p"),
])
def test_filtered_or_tempered_rows_are_unsupported(sampling, needle):
    from verity_vllm.observe.fold.resolver import Unsupported, resolve_step
    from tests.observe import test_patterns_synthetic as S
    from verity_vllm.observe.fold.tree import step_events
    events, start = S.decode_step(sampler="gumbel")
    se = step_events(events)
    res = resolve_step(se, _lane_ctx(events, start, sampling, seed=1))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert uns and uns[0].op.endswith("_gumbel_sample_kernel") and needle in uns[0].reason
    assert not [k for k in S.by_spec(res) if k.startswith("GumbelArgmax")]


def test_tempered_row_is_gumbel_argmax_temp_over_the_bf16_source():
    """A `_temperature_kernel` launch on the float32 widening ahead of the sampler (T=0.7): the launch is a claimed noop and
    the sampled row is `GumbelArgmaxTempBf16_v1{V,SEED,POS,T}` over the bf16 logits (T = the request's temperature, fp32 bits)."""
    import numpy as np
    from verity_vllm.observe.events import TritonLaunch
    from verity_vllm.observe.fold.resolver import Compute, PerRow, Unsupported, resolve_step
    from tests.observe import test_patterns_synthetic as S
    from verity_vllm.observe.fold.tree import step_events
    events, start = S.decode_step(sampler="gumbel", f32_logits=True)
    k = next(i for i, e in enumerate(events) if isinstance(e, TritonLaunch) and e.fn.endswith("_gumbel_sample_kernel"))
    g = events[k]
    l32 = g.args["logits_ptr"]
    assert l32.dtype == "float32"
    temp = TritonLaunch(g.id - 1 + 500000, None, g.step, "sample", "vllm.v1.worker.gpu.sample.gumbel._temperature_kernel", "sha",
                        {"logits_ptr": l32, "temperature_ptr": g.args["temp_ptr"], "expanded_idx_mapping_ptr": g.args["expanded_idx_mapping_ptr"],
                         "logits_stride": S.V, "vocab_size": S.V, "BLOCK_SIZE": 8192}, {"BLOCK_SIZE": 8192}, [1, 6], 4, 3, 7)
    events = events[:k] + [temp] + events[k:]
    se = step_events(events)
    res = resolve_step(se, _lane_ctx(events, start, {**SAMPLED, "temperature": 0.7}, seed=1))
    assert not [r for r in res if isinstance(r, Unsupported)], [r for r in res if isinstance(r, Unsupported)]
    cs = [m for r in res for m in S.members(r) if isinstance(m, Compute) and m.spec.name.startswith("GumbelArgmax")]
    assert len(cs) == 1
    tb = int(np.float32(0.7).view(np.uint32))
    assert cs[0].spec.id == f"GumbelArgmaxTempBf16_v1{{V={S.V},SEED=1,POS=32,T={tb}}}"
    assert cs[0].operands == (PerRow(S.bf(310, (1, S.V))),)          # the bf16 logits, not the float32 quotient
    by_op = {root.op: r for root, r in zip(se.roots, res)}
    t = by_op["vllm.v1.worker.gpu.sample.gumbel._temperature_kernel"]
    assert t.effect == "noop" and t.kernels == ("_temperature_kernel",)
    # without the temperature launch the same header is refused: nobody divided
    res2 = resolve_step(step_events(events[:k] + events[k + 1:]), _lane_ctx(events, start, {**SAMPLED, "temperature": 0.7}, seed=1))
    uns = [r for r in res2 if isinstance(r, Unsupported)]
    assert uns and "non-unit temperature" in uns[0].reason


def test_gumbel_argmax_temp_definition_matches_the_div_full_oracle():
    """div.full.f32 quotient (F32Mul(DivFullScaleA(x,T), DivFullRcp(T))) + noise, lowest-index argmax."""
    import numpy as np
    from verity.ml.circuits import Program, bind
    from verity.ml.circuits.evaluate import evaluate_region
    from verity.ml.definitions.registry import prims as P
    from verity.ml.definitions.registry.sampling import GumbelArgmaxTempBf16
    from verity_vllm.program.libdevice_sampling import gumbel_noise_exact
    V, seed, pos = 80, 11, 5
    for T in (0.7, 1.3):
        tb = int(np.float32(T).view(np.uint32))
        rng = np.random.default_rng(int(T * 10))
        mag = rng.integers(0x3E00, 0x4100, size=V)
        words = [int(w) | (0x8000 if s else 0) for w, s in zip(mag, rng.random(V) < 0.3)]
        Pg = Program(bind(GumbelArgmaxTempBf16, V=V, SEED=seed, POS=pos, T=tb))
        C = Pg.circuit
        t = dict(zip([g for _, _, g in Pg.input_coll(0).refs], words))
        evaluate_region(C, C[0:C.size], t)
        steps = [c for c in C.root_calls() if not c.is_primitive and c.fn.name == "GumbelArgmaxTempStep"]
        rcp = np.uint32(P.DivFullRcp.evaluate(tb)).view(np.float32)
        x = (np.asarray(words, dtype=np.uint32) << 16).view(np.float32)
        y = (x * rcp).astype(np.float32) + gumbel_noise_exact(seed, pos, V)
        assert int(t[steps[-1].returns[1]]) == int(np.argmax(y))
