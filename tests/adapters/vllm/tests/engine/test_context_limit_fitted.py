"""Engine class G at the context limit vLLM fits to the KV pool, on a stand-in with a small pool.

The stand-in is vLLM's own: its model config for a checkpoint whose own limit is 1,048,576 positions, launched at -1 (what
``--max-model-len auto`` parses to); its fit of that launch to a pool of 64 blocks of 16 tokens (`get_kv_cache_configs`); and its
scheduler on that pool, driven as the engine core drives it, with the model runner replaced by one that samples a fixed id. No
weights and no GPU. Stock is that scheduler alone; ours carries the class's admission and the step trace. Both schedule the
same rows for a request that ends at the fitted limit; the admission's limit and the trace's header are the fitted number; and
the class's check of the built engine takes it for a launch at -1 and for no other stated length. Skipped where vLLM is not
installed."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from verity_vllm.engine.class_admission import Admission
from verity_vllm.engine.profiles import engine_classes as EC
from verity_vllm.engine.step_trace import StepTrace
from verity_vllm.errors import Refused

pytest.importorskip("vllm")

BLOCK, BLOCKS, OWN = 16, 64, 1048576
#: one block of the pool is its null block, so the longest request it holds is a block short of the pool
FITTED = (BLOCKS - 1) * BLOCK
#: a request that ends at the limit: its prompt, the outputs it asks for, and the outputs the limit leaves it
PROMPT, ASKED, LEFT = FITTED - 32, 64, 32


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    d = tmp_path_factory.mktemp("checkpoint")
    (d / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_hidden_layers": 2, "vocab_size": 1024, "max_position_embeddings": OWN}))
    return d


def scheduler(checkpoint):
    """vLLM's scheduler on the pool, at the limit vLLM fitted the launch at -1 to; and its config."""
    import torch
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.v1.core import kv_cache_utils as K
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.structured_output import StructuredOutputManager

    mc = ModelConfig(model=str(checkpoint), dtype="bfloat16", seed=0, skip_tokenizer_init=True, max_model_len=-1)
    assert (mc.max_model_len, mc.original_max_model_len) == (OWN, -1)
    sc = SchedulerConfig(max_num_seqs=256, max_num_batched_tokens=2048, max_model_len=mc.max_model_len,
                         long_prefill_token_threshold=256, enable_chunked_prefill=True, is_encoder_decoder=False)
    cc = CacheConfig(block_size=BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto", enable_prefix_caching=False)
    cfg = VllmConfig(model_config=mc, scheduler_config=sc, cache_config=cc, device_config=DeviceConfig(device="cpu"))
    cc.kv_cache_layout, cc.num_gpu_blocks_override = "LBNHC", BLOCKS
    spec = FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    kvc = K.generate_scheduler_kv_cache_config(K.get_kv_cache_configs(cfg, [{"layer": spec}], [1 << 30]))
    cc.num_gpu_blocks = kvc.num_blocks
    register_all_kvcache_specs(cfg)
    return Scheduler(vllm_config=cfg, kv_cache_config=kvc, block_size=BLOCK, log_stats=True,
                     structured_output_manager=StructuredOutputManager(cfg)), cfg


def request(rid: str, prompt: int, max_tokens: int):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    return Request(request_id=rid, prompt_token_ids=[11 + i % 97 for i in range(prompt)],
                   sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0.0), pooling_params=None)


def serve(sched, requests):
    """Drive the scheduler as the engine core does, a row that reaches its request's last token sampling the id 7: each step's
    rows (request, tokens), and each request's output count and finish reason."""
    from vllm.v1.outputs import ModelRunnerOutput

    for r in requests:
        sched.add_request(r)
    done: dict[str, tuple[int, str | None]]
    rows, done = [], {}
    for _ in range(4 * FITTED):
        so = sched.schedule()
        ids = list(so.num_scheduled_tokens)
        if not ids:
            break
        sampled = [[7] if sched.requests[r].num_computed_tokens == sched.requests[r].num_tokens else [] for r in ids]
        rows.append([(r, so.num_scheduled_tokens[r]) for r in ids])
        outs = sched.update_from_output(so, ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                                              sampled_token_ids=sampled))
        for eco in outs.values():
            for o in eco.outputs:
                n, finish = done.get(o.request_id, (0, None))
                done[o.request_id] = (n + len(o.new_token_ids), finish if o.finish_reason is None else str(o.finish_reason))
    assert not sched.requests, f"the scheduler holds {sorted(sched.requests)} and runs none of them"
    return rows, done


def load():
    return [request("ends-at-the-limit", PROMPT, ASKED), request("short", 100, 8)]


def test_vllm_fits_a_launch_at_minus_one_to_the_pool_less_its_null_block(checkpoint):
    sched, cfg = scheduler(checkpoint)
    assert cfg.cache_config.num_gpu_blocks == BLOCKS
    assert cfg.model_config.max_model_len == sched.max_model_len == FITTED
    # the launch's -1 stays beside the fitted limit, in the config vLLM's scheduler holds
    assert sched.vllm_config.model_config.original_max_model_len == -1


def test_stock_and_ours_schedule_the_same_rows_for_a_request_that_ends_at_the_fitted_limit(checkpoint):
    stock, _ = scheduler(checkpoint)
    stock_rows, stock_done = serve(stock, load())
    assert stock_done == {"ends-at-the-limit": (LEFT, "length"), "short": (8, "length")}
    assert sum(n for step in stock_rows for r, n in step if r == "ends-at-the-limit") == FITTED - 1

    ours, _ = scheduler(checkpoint)
    adm = Admission("G").install(NS(llm_engine=NS(engine_core=NS(engine_core=NS(scheduler=ours)))))
    trace = StepTrace().install(ours)
    rows, done = serve(ours, load())
    trace.uninstall()
    adm.uninstall()
    assert (rows, done) == (stock_rows, stock_done)
    assert adm.limit == FITTED and adm.refused is None
    header = trace.records[0]["config"]
    assert (header["max_model_len"], header["num_gpu_blocks"], header["block_size"]) == (FITTED, BLOCKS, BLOCK)
    served = [[(row[0], row[2]) for row in r["rows"]] for r in trace.records if r["kind"] == "step"]
    assert served == stock_rows


def test_the_class_refuses_a_prompt_over_the_fitted_limit_before_vllms_scheduler_holds_it(checkpoint):
    sched, _ = scheduler(checkpoint)
    adm = Admission("G").install(NS(llm_engine=NS(engine_core=NS(engine_core=NS(scheduler=sched)))))
    with pytest.raises(Refused, match=rf"a prompt of {FITTED + 1} tokens, over the {FITTED} a class-G window admits"):
        sched.add_request(request("over", FITTED + 1, 8))
    assert not sched.requests
    adm.uninstall()


def test_the_built_engines_fitted_limit_is_taken_for_a_launch_at_minus_one_only(checkpoint):
    _, cfg = scheduler(checkpoint)
    built = {"tensor_parallel_size": EC.CLASS_G["world"], **EC.CLASS_G_SERVED, "enable_prefix_caching": False,
             "max_model_len": cfg.model_config.max_model_len}
    env = EC.env("G")
    assert EC.effective_refusals("G", built, env, -1) == []
    assert EC.effective_refusals("G", built, env, FITTED) == []
    assert EC.effective_refusals("G", built, env, OWN) == [
        f"the built engine's max_model_len is {FITTED}, not the launch's {OWN}"]
