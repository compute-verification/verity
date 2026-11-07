"""gap.py: a captured step evaluated as the class G decoder writes it, and the refusals counted instead of raised."""
from __future__ import annotations

import json
from collections import Counter

import pytest

import gap

CLASS_G_STEP = {"tokens": 8, "reqs": 8, "new": 0, "prefix_hit_tokens": 0, "prefill": [], "decode": 8,
                "decode_ctx": [100 + i for i in range(8)], "decode_ctx_max": 107, "sampling": {"greedy": 8},
                "cg": {"cg_mode": "NONE", "num_tokens": 8}}
DECODE_PADDED = {"tokens": 200, "reqs": 200, "new": 0, "prefix_hit_tokens": 0, "prefill": [], "decode": 200,
                 "decode_ctx": [300 + 150 * i for i in range(200)], "decode_ctx_max": 300 + 150 * 199,
                 "sampling": {"greedy": 120, "random": 80, "top_p": 80}, "cg": {"cg_mode": "FULL", "num_tokens": 208}}
MIXED_8192 = {"tokens": 8192, "reqs": 33, "new": 2, "prefix_hit_tokens": 512,
              "prefill": [[2000, 0, 2000], [4066, 16000, 20480], [2096, 0, 4000]], "decode": 30,
              "decode_ctx": [500 + 10 * i for i in range(30)], "decode_ctx_max": 790,
              "sampling": {"greedy": 21, "seeded": 4, "random": 8, "top_p": 12, "top_k": 2}, "cg": {"cg_mode": "NONE", "num_tokens": 8192}}


@pytest.fixture(scope="module")
def launches():
    return gap.Launches(gap.phi_of())


def _capture(out, rows, sampler="unset"):
    cap = out / "capture"
    cap.mkdir(parents=True)
    (cap / "steps.rank0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "engine_env.txt").write_text(f"VLLM_USE_FLASHINFER_SAMPLER={sampler}\n")
    return cap


def _check(ids, failing=()):
    return {"targets": [{"target": i, "failures": ["unprobed table"] if i in failing else []} for i in ids]}


def test_a_class_g_step_has_no_gap(tmp_path, launches):
    """The step class G registers (8 greedy decodes, eager, short contexts) needs only registered Definitions: no refusal and no
    engine fact, and one failing id is reported as circuit-check's, with circuits as its owner."""
    need, facts = launches.step(CLASS_G_STEP, flashinfer_sampler=True)
    assert all(n.gap is None for n in need) and not facts
    ids = sorted({n.id for n in need})
    g = gap.gap(_capture(tmp_path, [CLASS_G_STEP]), _check(ids), phi=launches.phi)
    assert g["gaps"] == [] and g["tokens"] == 8 and g["blocks"] == sum(need.values())
    g = gap.gap(_capture(tmp_path / "failing", [CLASS_G_STEP]), _check(ids, failing={ids[0]}), phi=launches.phi)
    assert [(r["definition"], r["status"], r["owner"]) for r in g["gaps"]] == [(ids[0].split("{")[0], "fails circuit-check", "circuits")]


def test_production_steps_name_each_gap(tmp_path, launches):
    """A padded FULL-graph decode batch with long contexts and an 8,192-token mixed step: each family the production limits add
    is its own row, with its refusal, its owner and its statics' range."""
    need, _ = launches.step(CLASS_G_STEP, flashinfer_sampler=False)
    g = gap.gap(_capture(tmp_path, [CLASS_G_STEP, DECODE_PADDED, MIXED_8192]), _check({n.id for n in need}), phi=launches.phi)
    rows = {(r["definition"], r["status"]): r for r in g["gaps"]}
    radix = rows[("radix-topk", "radix top-k")]
    assert radix["owner"] == "circuits + kernels" and radix["exists"] == "no" and radix["statics"] == "TOPK=2048, KEYS=2100..30150 (188 values)"
    assert rows[("DsaMqaLogits_v3", "unregistered")]["owner"] == "circuits"
    assert rows[("allreduce-pynccl_comm", "no capture")]["statics"] == "N=6144, M=8192"
    assert rows[("cublas-linear", "no capture")]["owner"] == "kernels"
    assert rows[("prefill-call", "dense-MHA prefill")]["statics"] == "ROWS=2000, KEYS=2000"
    assert rows[("prefill-call", "sparse-MQA prefill")]["statics"].startswith("ROWS=2096..4066")
    assert rows[("MoeRouterSigmoidBias_v1", "unregistered")]["statics"].endswith("KERNEL='warp'")
    assert rows[("select-random", "unseeded sampling")]["owner"] == "vllm"
    assert rows[("select-top_k", "top-k sampling")]["owner"] == "circuits"
    assert ("flashinfer-sampler", "FlashInfer sampler") in rows
    assert set(g["engine"]) >= {"CUDA graphs (eager in class G)", "launch padded past its rows", "prefix-cache hit (off in class G)",
                                "more tokens than class G's max_num_batched_tokens", "a sequence past class G's max_model_len"}
    assert g["engine"]["launch padded past its rows"]["tokens"] == 200
    assert "| dsa-topk | radix-topk (radix top-k) |" in gap.markdown(g)


def test_the_sampler_fallback_turns_its_row_off(tmp_path, launches):
    cap = _capture(tmp_path, [DECODE_PADDED], sampler="0")
    assert not gap.flashinfer_sampler_on(cap)
    (tmp_path / "engine_env.txt").write_text("VLLM_USE_FLASHINFER_SAMPLER=unset\nfallback: ... restarted with VLLM_USE_FLASHINFER_SAMPLER=0\n")
    assert not gap.flashinfer_sampler_on(cap)
    (tmp_path / "engine_env.txt").write_text("VLLM_USE_FLASHINFER_SAMPLER=unset\n")
    assert gap.flashinfer_sampler_on(cap)


@pytest.mark.slow
@pytest.mark.parametrize("M", [8, 64])
def test_a_step_counts_the_decoders_calls(M, launches):
    """The class G decoder's own Program for one engine step of M rows (`build_program(..., "step")`: one call of M rows with no
    history, the last row sampled greedily) calls each Definition exactly as often as gap.py counts for that step: the FlashInfer
    linears and fused all-reduce + norm at 8, DeepGEMM, the multimem all-reduce and the fused norm at 64. This is what holds
    gap.py's evaluation to the decoder; no cheaper test compares the two. About 20 s, nearly all of it the decoder's build."""
    from verity_vllm.program.predict import dsa
    from verity_vllm.program.predict.build import build_program
    seen: Counter[str] = Counter()
    orig = dsa._Model.call

    def call(self, fn, *refs):
        seen[fn.id] += 1
        return orig(self, fn, *refs)

    dsa._Model.call = call
    try:
        build_program(launches.phi, "step", M=M, pos0=8)
    finally:
        dsa._Model.call = orig
    need, facts = launches.step({"tokens": M, "reqs": 1, "prefill": [[M, 0, M]], "decode": 0, "decode_ctx": [], "decode_ctx_max": 0,
                                 "sampling": {"greedy": 1}, "prefix_hit_tokens": 0}, flashinfer_sampler=False)
    assert all(n.gap is None for n in need)
    assert Counter({n.id: c for n, c in need.items()}) == Counter({k: v for k, v in seen.items() if not k.startswith("Const")})
