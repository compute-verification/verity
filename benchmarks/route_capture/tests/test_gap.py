"""gap.py: captured steps evaluated as the class G decoder writes them (`step_needs`), the refusals rows of the table."""
from __future__ import annotations

import json

import pytest

import gap
from verity_vllm.program.predict.step_needs import StepNeeds

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


def _warmup_step(n, prefill):
    """One step of vLLM's kernel warm-up as the plugin records it: n requests, each with every feature of the warm-up's sampling."""
    return {"tokens": 2 * n if prefill else n, "reqs": n, "new": n if prefill else 0, "prefix_hit_tokens": 0,
            "prefill": [[2, 0, 2]] * n if prefill else [], "decode": 0 if prefill else n, "decode_ctx": [] if prefill else [3] * n,
            "decode_ctx_max": 0 if prefill else 3, "sampling": {k: n for k in gap.WARMUP_SAMPLING},
            "cg": {"cg_mode": "NONE", "num_tokens": 2 * n if prefill else n}}


WARMUP = [_warmup_step(256, True), _warmup_step(256, False), _warmup_step(2, False), _warmup_step(1, False)]


@pytest.fixture(scope="module")
def launches():
    return StepNeeds(gap.class_g_phi())


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
    need = gap.needs(launches, CLASS_G_STEP, flashinfer_sampler=True)
    assert all(n.gap is None for n in need) and not gap.engine_facts(CLASS_G_STEP)
    ids = sorted({n.id for n in need})
    g = gap.gap(_capture(tmp_path, [CLASS_G_STEP]), _check(ids), phi=launches.phi)
    assert g["gaps"] == [] and g["tokens"] == 8 and g["blocks"] == sum(need.values())
    g = gap.gap(_capture(tmp_path / "failing", [CLASS_G_STEP]), _check(ids, failing={ids[0]}), phi=launches.phi)
    assert [(r["definition"], r["status"], r["owner"]) for r in g["gaps"]] == [(ids[0].split("{")[0], "fails circuit-check", "circuits")]


def test_production_steps_name_each_gap(tmp_path, launches):
    """A padded FULL-graph decode batch with long contexts and an 8,192-token mixed step: each family the production limits add
    is its own row, with its refusal, its owner and its statics' range."""
    need = gap.needs(launches, CLASS_G_STEP, flashinfer_sampler=False)
    g = gap.gap(_capture(tmp_path, [CLASS_G_STEP, DECODE_PADDED, MIXED_8192]), _check({n.id for n in need}), phi=launches.phi)
    rows = {(r["definition"], r["status"]): r for r in g["gaps"]}
    radix = rows[("radix-topk", "radix top-k")]
    assert radix["owner"] == "circuits + kernels" and radix["exists"] == "no" and radix["statics"] == "TOPK=2048, KEYS=2100..30150 (188 values)"
    assert rows[("DsaMqaLogits_v3", "unregistered")]["owner"] == "circuits"
    assert rows[("allreduce-pynccl_comm", "no capture")]["statics"] == "N=6144, M=8192"
    assert rows[("prefill-call", "dense-MHA prefill")]["statics"] == "ROWS=2000, KEYS=2000"
    assert rows[("prefill-call", "sparse-MQA prefill")]["statics"].startswith("ROWS=2096..4066")
    assert sum(n["blocks"] for n in g["needs"]) == g["blocks"] == sum(r["blocks"] for r in g["gaps"] + g["covered"])
    assert sum(n["blocks"] for n in g["needs"] if n["id"].startswith("radix-topk{")) == radix["blocks"]
    assert rows[("MoeRouterSigmoidBias_v1", "unregistered")]["statics"].endswith("KERNEL='warp'")
    assert rows[("select-random", "unseeded sampling")]["owner"] == "vllm"
    assert rows[("select-top_k", "top-k sampling")]["owner"] == "circuits"
    assert ("flashinfer-sampler", "FlashInfer sampler") in rows
    assert set(g["engine"]) >= {"CUDA graphs (eager in class G)", "launch padded past its rows", "prefix-cache hit (off in class G)",
                                "more tokens than class G's max_num_batched_tokens", "a sequence past class G's max_model_len"}
    assert g["engine"]["launch padded past its rows"]["tokens"] == 200
    assert "| dsa-topk | radix-topk (radix top-k) |" in gap.markdown(g)


def test_vllms_warmup_is_not_served(tmp_path, launches):
    """vLLM's kernel warm-up runs its dummy requests through `execute_model` before serving: the capture's leading warm-up steps
    are left out and counted, with none of their sampler refusals, and a served step like them later is kept."""
    g = gap.gap(_capture(tmp_path, [*WARMUP, CLASS_G_STEP]), _check([]), phi=launches.phi)
    assert g["warmup"] == {"steps": 4, "tokens": 512 + 256 + 2 + 1} and g["steps"] == 1 and g["tokens"] == 8
    assert not {r["definition"] for r in g["gaps"]} & {"select-top_k", "select-min_p", "select-penalties", "flashinfer-sampler"}
    g = gap.gap(_capture(tmp_path / "late", [*WARMUP, CLASS_G_STEP, WARMUP[2]]), _check([]), phi=launches.phi)
    assert g["warmup"]["steps"] == 4 and g["steps"] == 2 and g["tokens"] == 10
    assert "select-top_k" in {r["definition"] for r in g["gaps"]}


def test_the_sampler_fallback_turns_its_row_off(tmp_path, launches):
    cap = _capture(tmp_path, [DECODE_PADDED], sampler="0")
    assert not gap.flashinfer_sampler_on(cap)
    (tmp_path / "engine_env.txt").write_text("VLLM_USE_FLASHINFER_SAMPLER=unset\nfallback: ... restarted with VLLM_USE_FLASHINFER_SAMPLER=0\n")
    assert not gap.flashinfer_sampler_on(cap)
    (tmp_path / "engine_env.txt").write_text("VLLM_USE_FLASHINFER_SAMPLER=unset\n")
    assert gap.flashinfer_sampler_on(cap)
    assert gap.engine_had_flashinfer(cap) is None


def test_an_engine_without_flashinfer_is_said(tmp_path, launches):
    """A pod whose vLLM finds neither FlashInfer's prebuilt kernels nor nvcc serves other kernels than production's image; the
    gap's rows stay the decoder's for production, and the report says the captured engine differed."""
    cap = _capture(tmp_path, [DECODE_PADDED])
    (tmp_path / "engine_env.txt").write_text("VLLM_USE_FLASHINFER_SAMPLER=unset\nhas_flashinfer=False flashinfer_cubin=False\n")
    g = gap.gap(cap, _check([]), phi=launches.phi)
    assert g["engine_had_flashinfer"] is False and g["flashinfer_sampler"]
    assert "The captured engine had no FlashInfer" in gap.markdown(g)
    (tmp_path / "engine_env.txt").write_text("has_flashinfer=True flashinfer_cubin=True\n")
    assert gap.engine_had_flashinfer(cap) is True
