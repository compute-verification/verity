"""CB-B probe: can the runner's OWN request-state path (`GPUModelRunner.add_requests` → `gather_batch_req_state` →
`prepare_inputs` → `prepare_attn`) produce the per-step inputs (positions, query_start_loc, seq_lens, logits_indices,
block tables, slot mappings) without `load_model` (no weights)?  Records the exact first refusal or the produced values.

Usage: python -m verity_capture.experimental.cb_b.probe_prepare_inputs --model <snapshot> --prompt-len 8 --max-tokens 4 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from types import SimpleNamespace

import torch

from . import fake_exec as fx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fx.apply_env()
    rep: dict = {"stages": [], "stubs": []}

    def stage(name, fn):
        try:
            r = fn()
            rep["stages"].append({"stage": name, "ok": True, "result": r})
            return r
        except Exception as e:  # noqa: BLE001
            rep["stages"].append({"stage": name, "ok": False, "refusal": f"{type(e).__name__}: {str(e)[:700]}",
                                  "traceback": traceback.format_exc()[-2500:]})
            raise

    try:
        cfg = stage("engine config", lambda: fx.build_vllm_config(a.model))
        stage("distributed init", lambda: fx.init_distributed(cfg))
        model = stage("instantiate model on meta", lambda: fx.instantiate_model(cfg))
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        dev = torch.device("cuda")
        runner = stage("GPUModelRunner.__init__(cuda) [no load_model]", lambda: GPUModelRunner(cfg, dev))
        # KV-cache config via the engine-core path already used by the builder (meta allocation + runner attention groups)
        tr = fx.Trace()
        layers = cfg.compilation_config.static_forward_context
        stage("bind_kv_caches (engine-core config, runner attn groups, meta KV)",
              lambda: list(fx.bind_kv_caches(tr, {n: l for n, l in layers.items() if hasattr(l, "kv_cache")}, cfg,
                                             (a.prompt_len + a.max_tokens) // 16 + 2))) and None
        prep = tr.runner_prep
        # attributes `add_requests` / `gather_batch_req_state` / `prepare_inputs` read that `load_model` / `initialize_kv_cache` set
        class _Stub:                       # stands in for load_model()'s ModelState / Sampler: bookkeeping no-ops, one sampled token per step
            num_new_sampled_tokens_per_step = 1

            def __getattr__(self, k):
                return lambda *x, **kw: None

        stubs = {"model_state": _Stub(), "decode_query_len": 1, "sampler": _Stub(), "prompt_logprobs_worker": _Stub(),
                 "block_tables": prep.block_tables, "kv_cache_config": tr.kv_cache_config, "attn_groups": prep.attn_groups}
        for k, v in stubs.items():
            if getattr(runner, k, None) is None or k in ("block_tables", "kv_cache_config", "attn_groups"):
                setattr(runner, k, v)
                rep["stubs"].append(k)
        from vllm import SamplingParams
        from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
        from vllm.config import CUDAGraphMode

        P, T = a.prompt_len, a.max_tokens
        nblocks = (P + T) // 16 + 1
        block_ids = tuple([list(range(nblocks))] * len(tr.kv_cache_config.kv_cache_groups))
        toks = [0] * P                                                  # placeholder token VALUES (circuit values; never used)
        new_req = NewRequestData(req_id="r0", prompt_token_ids=toks, mm_features=[], sampling_params=SamplingParams(max_tokens=T, temperature=0.0),
                                 pooling_params=None, block_ids=block_ids, num_computed_tokens=0, lora_request=None, prefill_token_ids=toks)
        so = SchedulerOutput(scheduled_new_reqs=[new_req], scheduled_cached_reqs=CachedRequestData.make_empty(),
                             num_scheduled_tokens={"r0": P}, total_num_scheduled_tokens=P, scheduled_spec_decode_tokens={},
                             scheduled_encoder_inputs={}, num_common_prefix_blocks=[0] * len(block_ids), finished_req_ids=set(),
                             free_encoder_mm_hashes=[])
        stage("runner.add_requests(scheduler_output)", lambda: runner.add_requests(so))
        stage("block_tables.apply_staged_writes", lambda: runner.block_tables.apply_staged_writes())
        brs, uniform = stage("runner.gather_batch_req_state", lambda: runner.gather_batch_req_state(so, False))
        bd = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=P, num_reqs=None)
        ib = stage("runner.prepare_inputs (prefill)", lambda: runner.prepare_inputs(so, brs, bd))
        torch.cuda.synchronize()
        vals = {"positions": ib.positions[:P].tolist(), "query_start_loc": ib.query_start_loc.tolist(), "seq_lens": ib.seq_lens.tolist(),
                "logits_indices": ib.logits_indices.tolist(), "input_ids[:4]": ib.input_ids[:4].tolist(), "num_tokens": ib.num_tokens}
        rep["prefill_inputs"] = vals
        bts, sms = stage("runner.prepare_attn (prefill)", lambda: runner.prepare_attn(ib))
        rep["prefill_attn"] = {"block_tables": [bt[0][:nblocks + 1].tolist() for bt in bts], "slot_mappings": sms[:, :P].tolist()}
        # the runner's own post-step bookkeeping (execute_model → postprocess_num_computed_tokens; the sampled token itself
        # is a circuit value the runner would write into last_sampled_tokens — irrelevant to positions/seq_lens/slots)
        stage("runner.postprocess_num_computed_tokens (after prefill)", lambda: runner.postprocess_num_computed_tokens(ib))
        # one decode step through the runner's cached-request path
        cached = CachedRequestData(req_ids=["r0"], resumed_req_ids=set(), new_token_ids=[[]], all_token_ids={}, new_block_ids=[None],
                                   num_computed_tokens=[P], num_output_tokens=[1]) if "num_output_tokens" in CachedRequestData.__dataclass_fields__ \
            else CachedRequestData(req_ids=["r0"], resumed_req_ids=set(), new_token_ids=[[]], all_token_ids={}, new_block_ids=[None],
                                   num_computed_tokens=[P])
        so2 = SchedulerOutput(scheduled_new_reqs=[], scheduled_cached_reqs=cached, num_scheduled_tokens={"r0": 1}, total_num_scheduled_tokens=1,
                              scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={}, num_common_prefix_blocks=[0] * len(block_ids),
                              finished_req_ids=set(), free_encoder_mm_hashes=[])
        stage("runner.update_requests (decode 1)", lambda: runner.update_requests(so2))
        brs2, _ = stage("runner.gather_batch_req_state (decode 1)", lambda: runner.gather_batch_req_state(so2, False))
        ib2 = stage("runner.prepare_inputs (decode 1)", lambda: runner.prepare_inputs(so2, brs2, BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=1, num_reqs=None)))
        torch.cuda.synchronize()
        rep["decode1_inputs"] = {"positions": ib2.positions[:1].tolist(), "query_start_loc": ib2.query_start_loc.tolist(),
                                 "seq_lens": ib2.seq_lens.tolist(), "logits_indices": ib2.logits_indices.tolist()}
        bts2, sms2 = stage("runner.prepare_attn (decode 1)", lambda: runner.prepare_attn(ib2))
        rep["decode1_attn"] = {"block_tables": [bt[0][:nblocks + 1].tolist() for bt in bts2], "slot_mappings": sms2[:, :1].tolist()}
        rep["verdict"] = "runner prepare_inputs/prepare_attn ran without load_model (stubs listed)"
    except Exception:  # noqa: BLE001
        rep["verdict"] = "refused: see last stage"
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "stages"}, indent=1, default=str)[:3000])
    for s in rep["stages"]:
        print(("OK   " if s["ok"] else "FAIL ") + s["stage"] + ("" if s["ok"] else "  -> " + s["refusal"][:400]))
    return 0 if rep["verdict"].startswith("runner") else 1


if __name__ == "__main__":
    sys.exit(main())
