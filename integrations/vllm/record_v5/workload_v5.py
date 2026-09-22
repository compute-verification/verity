"""[A9 contract 2026-09-10] The auditor's request rows -- fixed BEFORE the first commitment (WR-5: ``epoch.inputs`` is built from these,
never from the prover's ``boundary.json``) and served unchanged by ``serve_v5``.

The workload: the B0 workload's prompts (``e2e/workload-B0-small.json``) truncated to ``prompt_len`` for shapes <= 32, or synthetic seeded
random prompts (D4's 6.5K practice) for longer shapes; ``max_tokens = STEPS + 1`` slots; ``requests`` requests; greedy.  Written once to
``REC/workload.json`` (schema ``verity-vllm/workload/v1`` so ``verity_vllm_adapter.production`` loads it) by ``instance_v5``; ``serve_v5``
refuses to build another.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from record_v5 import POC
from record_v5 import config as cfgmod

WORKLOAD_B0 = POC / "e2e" / "workload-B0-small.json"
VOCAB = 49152


def make_workload(cfg: cfgmod.RunConfig) -> dict[str, Any]:
    from verity_vllm_sampler.workload import base_request_id

    base = json.loads(WORKLOAD_B0.read_text())
    reqs, origins = [], []
    # [E10] `workload.request_indices` (run config) names the B0 rows in epoch order; default 0..requests-1 (unchanged for every existing config).
    rows = list(cfg.request_indices) if cfg.request_indices else list(range(cfg.requests))
    for i, src_i in enumerate(rows):
        if cfg.prompt_len <= 32:
            if src_i >= len(base["requests"]):
                raise ValueError(f"workload.request_indices[{i}] = {src_i}: workload-B0-small.json has {len(base['requests'])} requests")
            src = base["requests"][src_i]
            ids = list(src["prompt_token_ids"])[:cfg.prompt_len]
            full = cfg.prompt_len == 32
            text, rt, seed = (src["prompt_text"] if full else ""), (full and src["text_roundtrip"]), src["corpus_seed"]
            origin = f"workload-B0-small.json request {src_i}" + ("" if full else f" truncated to {cfg.prompt_len} tokens")
        else:
            rng = random.Random(src_i + 1)
            ids = [rng.randrange(VOCAB) for _ in range(cfg.prompt_len)]
            text, rt, seed = "", False, src_i + 1
            origin = f"synthetic: random.Random({src_i + 1}).randrange({VOCAB}) x {cfg.prompt_len} (D4's 6.5K practice)"
        reqs.append({"request_id": base_request_id(cfg.model, ids, cfg.max_tokens), "index": i, "prompt_len": len(ids), "max_tokens": cfg.max_tokens,
                     "prompt_token_ids": ids, "prompt_text": text, "text_roundtrip": bool(rt), "corpus_seed": seed})
        origins.append({"request_id": reqs[-1]["request_id"], "origin": origin})
    # [E1 NIGHT 2026-09-12] `model` names the run config's served model (was B0's label copied from workload-B0-small.json; nothing in the
    # adapter reads it at serving time -- the declared inputs are the token ids -- but the record must say which checkpoint served them).
    # The prompt ids are B0's SmolLM2 tokenizer's ids (prompt_text "" when truncated): valid inputs for any vocabulary >= 49152.
    return {**{k: v for k, v in base.items() if k not in ("requests", "prompt_lengths", "max_tokens")}, "model": cfg.model,
            "tokenizer_note": f"prompt_token_ids come from workload-B0-small.json (tokenizer {base.get('model')} @ {base.get('tokenizer_revision')}); the served model is `model`",
            "prompt_lengths": [cfg.prompt_len], "max_tokens": [cfg.max_tokens],
            "requests": reqs, "request_origins": origins, "note": f"record_v5 workload {cfg.workload}: {cfg.requests} requests, prompt {cfg.prompt_len}, {cfg.max_tokens} slots (STEPS+1), greedy; "
                                      "the auditor's submission (WR-5): epoch.inputs is derived from these rows"
                                      + (f"; B0 rows {rows} (workload.request_indices)" if cfg.request_indices else "")}


def write_once(cfg: cfgmod.RunConfig, path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    wl = make_workload(cfg)
    path.write_text(json.dumps(wl, indent=1) + "\n")
    return wl


def request_rows(wl: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for ``protocol_instance.declared_inputs_section`` (request_id, prompt_token_ids); seed / sampling params are not pinned here
    (the serving lane's exact ``sampling_params`` dict is not known before serving; greedy is declared in the run config)."""
    return [{"request_id": r["request_id"], "prompt_token_ids": list(r["prompt_token_ids"])} for r in wl["requests"]]
