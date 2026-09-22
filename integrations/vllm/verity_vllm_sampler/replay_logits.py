"""Replay-side instrument: dump the sampler-input logits row per (request, position) and force
the recorded trajectory (contract §3 "Replay reference").

vLLM 0.28.0 (the latest PyPI release) does NOT ship `enable_trace_replay` /
`SamplingParams.trace_decode_token_ids` (they exist only on `main`, in
vllm/v1/worker/gpu/sample/trace_replay.py, and overwrite the *sampled* token after sampling).
We reproduce exactly that mechanism without patching vLLM: the replay process runs the engine
in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, same GPU work) and wraps
`vllm.v1.worker.gpu.sample.sampler.Sampler.__call__` (Model Runner V2, the default runner for
this model) so that, for every sampling row:

  * the raw logits row handed to the sampler (`model.compute_logits(...)` output, bf16 here) is
    copied to host, keyed by (request_id, pos) where pos = input_batch.positions[logits_indices]
    is the absolute position of the input token whose logits these are (= prompt_len - 1 + k for
    output token k);
  * the sampler runs unmodified (so `sampled_before_force` is the sampler's own choice on the
    replayed logits, and logprobs stay real);
  * `sampler_output.sampled_token_ids[row]` is overwritten in place with the recorded output
    token k, which is what the model runner feeds back as the next input token.  Stop conditions
    are left to the engine: the forced sequence is the recorded one, so it stops where production
    stopped.

Production is never run with this module imported.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m verity_vllm_sampler.replay_logits \
        --production prod.json --out-dir /workspace/results/<exp>/logits_topk_topp_s1 [--force-json overrides.json]

`--force-json` (N08) maps request_id -> {position_k: token} to replace tokens in the forced
trajectory; the dump records both the original recorded token and the forced one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

from .production import engine_kwargs, profile_snapshot
from .workload import SampledRequest


class LogitsDump:
    def __init__(self, trajectories: dict[str, list[int]], prompt_lens: dict[str, int],
                 forced: dict[str, list[int]] | None = None, id_map: dict[str, str] | None = None,
                 do_force: bool = True):
        self.trajectories = trajectories  # request_id -> recorded output ids (what production emitted)
        self.forced = forced or trajectories  # request_id -> tokens actually injected
        self.prompt_lens = prompt_lens
        # internal (InputBatch.req_ids) -> external request id; filled by production.run_requests
        self.id_map: dict[str, str] = id_map if id_map is not None else {}
        # False when vLLM's own trace replay does the forcing (nightly): the hook only records
        self.do_force = do_force
        self.rows: dict[str, dict[int, np.ndarray]] = {}  # request_id -> {pos: bf16 row (uint16)}
        self.sampled_before_force: dict[str, dict[int, int]] = {}
        self.calls = 0
        self.rows_seen = 0
        self.rows_skipped_prefill = 0
        self.rows_unknown_request = 0
        self.logits_dtype: str | None = None
        self.vocab_size: int | None = None
        self.forcing_events = 0
        self.beyond_trajectory = 0
        self.trace_applied_rows = 0        # trace_replay mode: rows where vLLM changed the sampled token
        self.trace_followed_rows = 0       # trace_replay mode: rows whose output token == forced[k]
        self.trace_token_mismatches = 0    # trace_replay mode: rows whose output token != forced[k]
        self.trace_changed_rows: list[dict[str, Any]] = []  # identity of rows vLLM's trace replay changed (first 32)
        self.post_force_cpu: list[int] = []

    def install(self):
        import torch
        from vllm.v1.worker.gpu.sample import sampler as sampler_mod

        dump = self
        orig = sampler_mod.Sampler.__call__

        def patched(self_s, logits, input_batch):
            dump.calls += 1
            out = orig(self_s, logits, input_batch)
            try:
                dump._record_and_force(logits, input_batch, out, torch)
            except Exception as e:  # noqa: BLE001
                print(f"[replay_logits] hook error: {e!r}", file=sys.stderr, flush=True)
                raise
            return out

        sampler_mod.Sampler.__call__ = patched  # type: ignore[method-assign]
        self._orig = orig
        self._mod = sampler_mod
        # Nightly trace replay overwrites `sampled` INSIDE Sampler.__call__ (after self.sample, before
        # logprobs).  To keep the sampler's own choice we snapshot the tensor on its way into
        # TraceReplayState.apply_trace; the __call__ wrap then reads it as `sampled_before_force`.
        self._pre_force = None
        self._tr_mod = None
        try:
            from vllm.v1.worker.gpu.sample import trace_replay as tr_mod
        except ImportError:
            tr_mod = None
        if tr_mod is not None:
            orig_apply = tr_mod.TraceReplayState.apply_trace

            def patched_apply(self_t, sampled, idx_mapping):
                dump._pre_force = sampled.detach().clone()
                return orig_apply(self_t, sampled, idx_mapping)

            tr_mod.TraceReplayState.apply_trace = patched_apply  # type: ignore[method-assign]
            self._tr_mod = tr_mod
            self._orig_apply = orig_apply

    def uninstall(self):
        self._mod.Sampler.__call__ = self._orig  # type: ignore[method-assign]
        if self._tr_mod is not None:
            self._tr_mod.TraceReplayState.apply_trace = self._orig_apply  # type: ignore[method-assign]

    def _record_and_force(self, logits, input_batch, out, torch):
        if self.logits_dtype is None:
            self.logits_dtype = str(logits.dtype).replace("torch.", "")
            self.vocab_size = int(logits.shape[1])
        pos = input_batch.positions[input_batch.logits_indices].tolist()
        cu = input_batch.cu_num_logits_np
        req_ids = input_batch.req_ids
        n_rows = logits.shape[0]
        # rows -> request index
        row_req = np.searchsorted(cu, np.arange(n_rows), side="right") - 1
        logits_cpu = logits.detach().to("cpu")
        if logits_cpu.dtype == torch.bfloat16:
            rows_np = logits_cpu.view(torch.int16).numpy().view(np.uint16)
        else:
            rows_np = logits_cpu.numpy()
        post_cpu = out.sampled_token_ids.view(-1).tolist()  # after vLLM's trace replay, if any
        if self._pre_force is not None and self._pre_force.numel() == len(post_cpu):
            sampled_cpu = self._pre_force.view(-1).tolist()  # the sampler's own choice
            for r, (a, b) in enumerate(zip(sampled_cpu, post_cpu)):
                if a != b:
                    self.trace_applied_rows += 1
                    if len(self.trace_changed_rows) < 32:
                        b_ = int(row_req[r]) if r < len(row_req) else -1
                        rid_ = req_ids[b_] if 0 <= b_ < len(req_ids) else None
                        ext = self.id_map.get(rid_, rid_) if rid_ is not None else None
                        p_ = int(pos[r]) if r < len(pos) else None
                        plen_ = self.prompt_lens.get(ext) if ext in self.prompt_lens else None
                        self.trace_changed_rows.append({
                            "call": self.calls, "row": r, "request_id": ext, "pos": p_, "prompt_len": plen_,
                            "kind": ("mid_prefill" if (plen_ is not None and p_ is not None and p_ < plen_ - 1)
                                     else "sampled_position" if ext in self.trajectories else "unknown_request"),
                            "sampler_choice": int(a), "trace_token": int(b)})
        else:
            sampled_cpu = post_cpu  # hook mode: not yet forced (we force below)
        self._pre_force = None
        self.post_force_cpu = post_cpu
        force_idx: list[int] = []
        force_tok: list[int] = []
        for r in range(n_rows):
            b = int(row_req[r])
            if b < 0 or b >= len(req_ids):
                continue
            rid = self.id_map.get(req_ids[b], req_ids[b])
            if rid not in self.trajectories:
                self.rows_unknown_request += 1
                continue
            p = int(pos[r])
            plen = self.prompt_lens[rid]
            if p < plen - 1:
                self.rows_skipped_prefill += 1  # mid-prefill chunk: no token is sampled from it
                continue
            k = p - (plen - 1)
            self.rows_seen += 1
            self.rows.setdefault(rid, {})[p] = rows_np[r].copy()
            self.sampled_before_force.setdefault(rid, {})[p] = int(sampled_cpu[r])
            forced = self.forced[rid]
            if k < len(forced):
                force_idx.append(r)
                force_tok.append(int(forced[k]))
                if not self.do_force:
                    # vLLM's trace replay must have put forced[k] into the output row
                    if int(self.post_force_cpu[r]) == int(forced[k]):
                        self.trace_followed_rows += 1
                    else:
                        self.trace_token_mismatches += 1
            else:
                self.beyond_trajectory += 1
        if force_idx and self.do_force:
            idx_t = torch.tensor(force_idx, dtype=torch.long, device=out.sampled_token_ids.device)
            tok_t = torch.tensor(force_tok, dtype=out.sampled_token_ids.dtype, device=out.sampled_token_ids.device)
            out.sampled_token_ids.view(-1)[idx_t] = tok_t
            self.forcing_events += len(force_idx)

    def save(self, out_dir: str, meta: dict[str, Any]) -> dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        index: dict[str, Any] = {"requests": {}}
        for rid, rows in self.rows.items():
            poss = sorted(rows)
            arr = np.stack([rows[p] for p in poss])
            path = os.path.join(out_dir, f"{rid}.npy")
            np.save(path, arr)
            index["requests"][rid] = {
                "positions": poss,
                "sampled_before_force": [self.sampled_before_force[rid][p] for p in poss],
                "recorded": self.trajectories[rid],
                "forced": self.forced[rid],
                "prompt_len": self.prompt_lens[rid],
                "file": os.path.basename(path),
                "shape": list(arr.shape),
            }
        index.update({
            "logits_dtype": self.logits_dtype, "vocab_size": self.vocab_size,
            "hook": "vllm.v1.worker.gpu.sample.sampler.Sampler.__call__ (Model Runner V2); rows = compute_logits output "
                    "before any sampler processing; pos = input_batch.positions[input_batch.logits_indices]",
            "sampler_calls": self.calls, "rows_seen": self.rows_seen,
            "rows_skipped_prefill": self.rows_skipped_prefill, "rows_unknown_request": self.rows_unknown_request,
            "forcing_events": self.forcing_events, "rows_beyond_trajectory": self.beyond_trajectory,
            "do_force_in_hook": self.do_force,
            "trace_replay_rows_changed_by_vllm": self.trace_applied_rows,
            "trace_replay_changed_rows": self.trace_changed_rows,
            "trace_replay_rows_followed": self.trace_followed_rows,
            "trace_replay_rows_token_mismatch": self.trace_token_mismatches,
        })
        index.update(meta)
        with open(os.path.join(out_dir, "index.json"), "w") as f:
            json.dump(index, f, indent=1)
        return index


def trace_replay_available() -> bool:
    """Does the installed wheel actually have SamplingParams.trace_decode_token_ids (contract §15.3)?"""
    from vllm import SamplingParams
    try:
        sp = SamplingParams(max_tokens=1, trace_decode_token_ids=[1])
    except TypeError:
        return False
    return getattr(sp, "trace_decode_token_ids", None) == [1]


def vllm_commit() -> str | None:
    import vllm
    for attr in ("__commit__", "__version_tuple__"):
        v = getattr(vllm, attr, None)
        if v:
            return str(v)
    ver = vllm.__version__
    return ver.split("+g", 1)[1] if "+g" in ver else None


def sampled_requests_from_production(prod: dict[str, Any]) -> list[SampledRequest]:
    out = []
    for rec in prod["records"]:
        out.append(SampledRequest(
            request_id=rec["request_id"], base_request_id=rec["base_request_id"], index=rec["index"],
            prompt_len=rec["prompt_len"], max_tokens=rec["max_tokens"], prompt_token_ids=rec["prompt_token_ids"],
            variant=rec["variant"], seed=rec["seed"], sampling=rec["sampling"]))
    return out


def main() -> None:
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        print("replay_logits must run with VLLM_ENABLE_V1_MULTIPROCESSING=0 (in-process worker)", file=sys.stderr)
        sys.exit(2)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--production", required=True, help="production record JSON")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force-json", default=None, help="N08 overrides: {request_id: {k: token}}")
    ap.add_argument("--max-num-seqs", type=int, default=None, help="default: same as production")
    ap.add_argument("--only", default=None, help="comma list of request indices to replay")
    ap.add_argument("--exclude", default=None, help="comma list of request indices to skip")
    ap.add_argument("--warmup-throwaway", type=int, default=0,
                    help="run N throwaway prompts (no shared 16-token block with the workload) to completion in the "
                         "replay engine before the replayed pass; the hook sees them as rows_unknown_request")
    ap.add_argument("--kv-gib", type=float, default=None,
                    help="fixed KV cache size for the replay engine (default: production's); a smaller cache "
                         "lets replay share the card with a production run and does not change the computation "
                         "unless it forces preemption, which the record would show (running_trace)")
    ap.add_argument("--force-mode", default="auto", choices=["auto", "hook", "trace_replay", "none"],
                    help="how the recorded trajectory is injected: `trace_replay` = vLLM's own "
                         "SamplingParams.trace_decode_token_ids (+ enable_trace_replay engine flag) when the wheel "
                         "has it; `hook` = overwrite sampled_token_ids in the Sampler.__call__ wrap (0.28.0); "
                         "`auto` = trace_replay if available else hook; `none` = dump only, replay samples freely")
    args = ap.parse_args()

    with open(args.production) as f:
        prod = json.load(f)
    sreqs = sampled_requests_from_production(prod)
    if args.only:
        keep = {int(x) for x in args.only.split(",")}
        sreqs = [s for s in sreqs if s.index in keep]
    if args.exclude:
        drop = {int(x) for x in args.exclude.split(",")}
        sreqs = [s for s in sreqs if s.index not in drop]
    traj = {r["request_id"]: list(r["output_token_ids"]) for r in prod["records"]}
    plens = {r["request_id"]: int(r["prompt_len"]) for r in prod["records"]}
    forced = {k: list(v) for k, v in traj.items()}
    overrides: dict[str, Any] = {}
    if args.force_json:
        with open(args.force_json) as f:
            overrides = json.load(f)
        for rid, m in overrides.items():
            for k, tok in m.items():
                forced[rid][int(k)] = int(tok)

    import vllm
    from vllm import LLM

    from .production import make_sampling_params, run_requests

    has_trace_replay = trace_replay_available()
    mode = args.force_mode
    if mode == "auto":
        mode = "trace_replay" if has_trace_replay else "hook"
    if mode == "trace_replay" and not has_trace_replay:
        print(f"vllm {vllm.__version__} has no SamplingParams.trace_decode_token_ids; use --force-mode hook",
              file=sys.stderr)
        sys.exit(3)

    id_map: dict[str, str] = {}
    dump = LogitsDump(traj, plens, forced, id_map=id_map, do_force=(mode == "hook"))
    dump.install()
    t0 = time.perf_counter()
    kw = engine_kwargs(prod["engine_kwargs"]["model"], args.max_num_seqs or prod["max_num_seqs"],
                       prod["engine_kwargs"]["max_model_len"])
    if args.kv_gib is not None:
        kw["kv_cache_memory_bytes"] = int(args.kv_gib * 1024**3)
        kw["gpu_memory_utilization"] = 0.4  # only the startup free-memory check uses it once KV is fixed
    params_override: dict[str, Any] | None = None
    if mode == "trace_replay":
        kw["enable_trace_replay"] = True
        params_override = {}
        for s in sreqs:
            # the forced trajectory (recorded, or with N08 overrides) is what the engine must follow
            extra = dict(prod.get("sampling_extra") or {})
            extra["trace_decode_token_ids"] = list(forced[s.request_id])
            params_override[s.request_id] = make_sampling_params(s, True, extra)
    llm = LLM(**kw)
    t_init = time.perf_counter() - t0
    snap = profile_snapshot(llm)
    from .production import shared_prefix_pairs, warmup_throwaway
    warm = warmup_throwaway(llm, sreqs, args.warmup_throwaway) if args.warmup_throwaway > 0 else None
    calls_before_pass = dump.calls
    unknown_before_pass = dump.rows_unknown_request
    # logprobs=1 so the (real) logprob of the forced token is available too
    res = run_requests(llm, sreqs, record=True, abort=None, sampling_extra=prod.get("sampling_extra") or None,
                       id_map=id_map, params_override=params_override)
    dump.uninstall()
    prod_cached = {r["request_id"]: r.get("num_cached_tokens") for r in prod["records"]}
    meta = {
        "schema": "verity-vllm/replay-logits/v1", "mode": "replay_reference",
        "force_mode": mode, "vllm_version": vllm.__version__, "vllm_commit": vllm_commit(),
        "trace_replay_available_in_wheel": has_trace_replay,
        "production_file": os.path.abspath(args.production), "overrides": overrides,
        "only": args.only, "exclude": args.exclude,
        "engine_kwargs": kw, "engine_kwargs_differ_from_production": {k: (prod["engine_kwargs"].get(k), v) for k, v in kw.items() if prod["engine_kwargs"].get(k) != v},
        "profile": snap, "timing_init_s": t_init, "timing": res["timing"], "running_trace": res["running_trace"],
        "peak_concurrency_achieved": res["peak_concurrency_achieved"], "engine_steps": res["engine_steps"],
        "warmup": warm, "sampler_calls_before_replayed_pass": calls_before_pass,
        "rows_unknown_request_before_replayed_pass": unknown_before_pass,
        "prefix_cache_totals": res["prefix_cache_totals"],
        "shared_prefix_pairs_in_pass": shared_prefix_pairs(sreqs),
        # prefix-cache witness on both sides, per request: production's recorded num_cached_tokens vs this replay's
        "num_cached_tokens": {r["request_id"]: {"index": r["index"], "production": prod_cached.get(r["request_id"]),
                                                "replay": r.get("num_cached_tokens")} for r in res["records"]},
        "replay_requests_with_cached_prefix": res["requests_with_cached_prefix"],
        "replay_outputs": {r["request_id"]: {"output_token_ids": r["output_token_ids"], "finish_reason": r["finish_reason"],
                                             "stop_reason": r["stop_reason"], "chosen_logprobs": r.get("chosen_logprobs"),
                                             "num_cached_tokens": r.get("num_cached_tokens")}
                           for r in res["records"]},
    }
    # §15.3 field 1: did replay follow the supplied trajectory?  (output ids == forced ids, per request)
    mism = 0
    followed = 0
    for r in res["records"]:
        f_ids = forced[r["request_id"]]
        if r["output_token_ids"] == f_ids:
            followed += 1
        else:
            mism += 1
    meta["trajectory_followed"] = {"requests": followed, "attempted": len(res["records"]), "mismatched_requests": mism,
                                   "all_followed": mism == 0}
    index = dump.save(args.out_dir, meta)
    print(f"REPLAY_DONE {args.out_dir} mode={mode} vllm={vllm.__version__} rows={index['rows_seen']} "
          f"calls={index['sampler_calls']} forced={index['forcing_events']} "
          f"trace_followed_rows={index['trace_replay_rows_followed']} changed_by_vllm={index['trace_replay_rows_changed_by_vllm']} "
          f"trajectory_followed={followed}/{len(res['records'])} dtype={index['logits_dtype']} "
          f"replay_cached_prefix_indices={res['requests_with_cached_prefix']}", flush=True)
    from .production import shutdown_engine
    shutdown_engine(llm)


if __name__ == "__main__":
    main()
