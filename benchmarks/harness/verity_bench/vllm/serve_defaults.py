"""`vllm serve MODEL`'s defaults at the pinned vLLM (`DEFAULTS`), the values an arm's `presets.VLLM` settings take on a card
and checkpoint (`context`, `resolve`), and where an engine's settings differ from `vllm serve`'s (`differences`), which
every arm's engine reports and a benchmark result records (`runner.run_benchmark`, `table.render`)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_bench.vllm.presets import VLLM

#: The vLLM commit `DEFAULTS` was read at: a test holds it to the wheel `pod_bootstrap.sh` installs, so a vLLM bump fails
#: until `DEFAULTS` is read again at the new commit.
VLLM_COMMIT = "d9105ea8001e0a6d77a96327d17515bb5791fb36"
TORCH = "v2.13.0"
GiB = 1 << 30

#: Who a difference is: one the harness could set otherwise, the requests every arm serves alike, or the adapter's
HARNESS, WORKLOAD, ADAPTER = "differs", "differs, the workload's", "differs, the adapter's"
#: A default resolved for the card (`card`) or read from the checkpoint (`checkpoint`)
CARD, CHECKPOINT = "the card's", "the checkpoint's"
NEUTRAL = "vLLM's neutral"
#: The `generation_config` keys `vllm serve` applies to a request that doesn't set them, and `GenerationConfig`'s own
#: defaults, which its `to_diff_dict` leaves out (vllm/config/model.py:1720-1762)
GENERATION = {"repetition_penalty": 1.0, "temperature": 1.0, "top_k": 50, "top_p": 1.0, "min_p": None, "max_new_tokens": None}
#: vLLM's own sampling where the checkpoint's `generation_config` says nothing (vllm/sampling_params.py)
NEUTRAL_SAMPLING = {"temperature": 1.0, "top_p": 1.0}
#: vLLM's own resolution of a config's context length (vllm/transformers_utils/model_arch_config_convertor.py:391-424)
MAX_LEN_KEYS = ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "model_max_length",
                "max_target_positions", "max_sequence_length", "max_seq_length", "seq_len")
#: RoPE types whose scaling vLLM doesn't apply to the context length (vllm/config/model.py:2455-2479), and the absent one
UNSCALED_ROPE = (None, "default", "llama3")
NOTE = ("the requests are the workload's, the same for every arm (its prompts and output lengths, greedy, ignore_eos, each "
        "seeded by its index), so this arm doesn't take vllm serve's sampling defaults: the checkpoint's generation_config, "
        "ignore_eos off, an unseeded request and the client's max_tokens")


@dataclass(frozen=True)
class Default:
    value: Any
    where: str
    whose: str = HARNESS


_SERVE_LIMITS = "vllm/engine/arg_utils.py:2652-2681, OPENAI_API_SERVER"
DEFAULTS: dict[str, Default] = {
    "enforce_eager": Default(False, "vllm/config/model.py:241"),
    "compilation_config": Default("VLLM_COMPILE, FULL_AND_PIECEWISE", "vllm/config/vllm.py:308, 1442-1444"),
    "VLLM_BATCH_INVARIANT": Default("0", "vllm/envs.py:631"),
    "max_num_seqs": Default(CARD, _SERVE_LIMITS),
    "max_num_batched_tokens": Default(CARD, _SERVE_LIMITS),
    "max_model_len": Default(CHECKPOINT, "vllm/config/model.py:214, 2394-2500"),
    "gpu_memory_utilization": Default(0.92, "vllm/config/cache.py:112"),
    "kv_cache_memory_bytes": Default(None, "vllm/config/cache.py:234-242"),
    "enable_prefix_caching": Default(True, "vllm/engine/arg_utils.py:527, 2750-2751"),
    "quantization": Default(None, "vllm/config/model.py:229"),
    "dtype": Default("auto", "vllm/config/model.py:169"),
    "seed": Default(0, "vllm/config/model.py:179"),
    "TORCHINDUCTOR_FX_GRAPH_CACHE": Default(None, f"torch {TORCH}'s own"),
    "TORCHINDUCTOR_AUTOGRAD_CACHE": Default(None, f"torch {TORCH}'s own"),
    "temperature": Default(CHECKPOINT, "vllm/config/model.py:329, 1740-1762; vllm/sampling_params.py:252", WORKLOAD),
    "top_p": Default(CHECKPOINT, "vllm/config/model.py:329, 1740-1762; vllm/sampling_params.py:256", WORKLOAD),
    "top_k": Default(CHECKPOINT, "vllm/config/model.py:329, 1740-1762", WORKLOAD),
    "min_p": Default(CHECKPOINT, "vllm/config/model.py:329, 1740-1762", WORKLOAD),
    "repetition_penalty": Default(CHECKPOINT, "vllm/config/model.py:329, 1740-1762", WORKLOAD),
    "max_tokens": Default(CHECKPOINT, "the client's; vllm/config/model.py:1740-1762 (max_new_tokens)", WORKLOAD),
    "ignore_eos": Default(False, "vllm/sampling_params.py:275", WORKLOAD),
    "request_seed": Default(None, "vllm/sampling_params.py:266", WORKLOAD),
    "detokenize": Default(True, "vllm/sampling_params.py:309", WORKLOAD),
    "VLLM_ENABLE_V1_MULTIPROCESSING": Default("1", "vllm/envs.py:1408-1410", ADAPTER),
    "VLLM_USE_FLASHINFER_SAMPLER": Default("1", "vllm/envs.py:868-872", ADAPTER),
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": Default("0", "vllm/envs.py:2032-2034", ADAPTER),
    "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK": Default("1", f"torch {TORCH} torch/_inductor/config.py:372", ADAPTER),
    "VLLM_USE_V2_MODEL_RUNNER": Default("1", "unset, and vllm/config/vllm.py:655-688 takes V2 for a decoder on CUDA",
                                        ADAPTER),
    "enable_flashinfer_autotune": Default(True, "vllm/config/vllm.py:312", ADAPTER),
    "VLLM_DISABLE_COMPILE_CACHE": Default("0", "vllm/envs.py:375-376", ADAPTER),
}


def card(memory_bytes: int, name: str) -> dict[str, int]:
    """`vllm serve`'s scheduler limits on a card of `memory_bytes` named `name`, and the in-process `LLM`'s
    max_num_batched_tokens, which an engine the harness builds takes where it sets none (`_SERVE_LIMITS`)."""

    if memory_bytes >= 160 * GiB:
        seqs, tokens, llm = 1024, 16384, 16384
    elif memory_bytes >= 70 * GiB and "a100" not in name.lower():
        seqs, tokens, llm = 1024, 8192, 16384
    else:
        seqs, tokens, llm = 256, 2048, 8192
    return {"max_num_seqs": seqs, "max_num_batched_tokens": tokens, "llm_max_num_batched_tokens": llm}


def max_model_len(config: Mapping[str, Any], tokenizer_config: Mapping[str, Any] | None = None) -> int:
    """The context length vLLM derives from a checkpoint's `config.json` and `tokenizer_config.json`; a config whose
    length vLLM scales or guesses is refused, to be read by hand."""

    text = config.get("text_config") or config
    found = [text[k] for k in MAX_LEN_KEYS if text.get(k) is not None]
    if not found:
        raise ValueError(f"the checkpoint's config has none of {MAX_LEN_KEYS}: vLLM would assume 2048")
    length = text["model_max_length"] if text.get("model_max_length") else min(found)
    if tokenizer_config and tokenizer_config.get("model_max_length") is not None:
        length = min(length, tokenizer_config["model_max_length"])
    rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
    if rope.get("rope_type", rope.get("type")) not in UNSCALED_ROPE:
        raise ValueError(f"vLLM scales this checkpoint's context length by its RoPE ({rope}): read its max_model_len by hand")
    return int(length)


def generation(generation_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """The sampling `vllm serve` takes from a checkpoint's `generation_config.json` for a request that sets none."""

    return {k: v for k, v in (generation_config or {}).items() if k in GENERATION and v is not None and v != GENERATION[k]}


def _json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def context(memory_bytes: int, name: str, config: Path | None, checkpoint: Path | None) -> dict[str, Any]:
    """What `vllm serve`'s defaults resolve to for this card and checkpoint: `config` its `config.json` of record and
    `checkpoint` its directory, for its tokenizer and generation configs (each absent when the directory lacks it)."""

    tokenizer = _json(checkpoint / "tokenizer_config.json") if checkpoint else None
    gen = _json(checkpoint / "generation_config.json") if checkpoint else None
    out: dict[str, Any] = {"card": name, "card_memory_bytes": memory_bytes, **card(memory_bytes, name),
                           "generation": generation(gen), "generation_config": gen is not None}
    try:
        if config is None:
            raise ValueError("no config.json of the checkpoint was found")
        out["max_model_len"] = max_model_len(json.loads(config.read_text()), tokenizer)
    except ValueError as e:
        out["max_model_len_unknown"] = str(e)
    return out


def _serve(key: str, ctx: Mapping[str, Any]) -> Any:
    if key in ("max_num_seqs", "max_num_batched_tokens", "max_model_len"):
        if key not in ctx:
            raise ValueError(f"vLLM's {key} is unknown here: {ctx.get('max_model_len_unknown', 'no card was read')}")
        return ctx[key]
    return {"kv_cache_memory_bytes": None, "gpu_memory_utilization": DEFAULTS["gpu_memory_utilization"].value,
            "enable_prefix_caching": True, "quantization": None, "dtype": DEFAULTS["dtype"].value}[key]


def resolve(engine: Mapping[str, Any], ctx: Mapping[str, Any]) -> dict[str, Any]:
    """An arm's engine overrides with each `VLLM` value `vllm serve`'s on this card and checkpoint; None is a setting
    `vllm serve` leaves unset, which `engine.engine_args` drops."""

    return {k: _serve(k, ctx) if v == VLLM else v for k, v in engine.items()}


def resolve_sampling(sampling: Mapping[str, Any]) -> dict[str, Any]:
    return {k: None if v == VLLM else v for k, v in sampling.items()}


def settings(engine: Mapping[str, Any], sampling: Mapping[str, Any], env: Mapping[str, str], ctx: Mapping[str, Any], *,
             compile_fresh: bool) -> dict[str, Any]:
    """An engine's value of each of `DEFAULTS`: `engine` the arguments it is built with (`engine.engine_args`, its arm's
    overrides applied), `sampling` its requests' `SamplingParams` but each request's own seed and max_tokens, `env` the
    adapter's switches (`verity_vllm.engine.env.ENGINE`), and `compile_fresh` whether the harness turned Inductor's
    caches off for it. A key the harness leaves to the adapter takes the adapter's value."""

    eager = bool(engine["enforce_eager"])
    execution = engine.get("execution")
    graphs = engine.get("compilation_config")
    return {
        "enforce_eager": eager,
        "compilation_config": "none (eager)" if eager else (
            DEFAULTS["compilation_config"].value if graphs is None else
            f"mode {graphs.get('mode')}, {graphs.get('cudagraph_mode')}"),
        "VLLM_BATCH_INVARIANT": "1" if execution is None or execution.startswith("bi-") else "0",
        "max_num_seqs": engine["max_num_seqs"],
        "max_num_batched_tokens": engine.get("max_num_batched_tokens", ctx.get("llm_max_num_batched_tokens")),
        "max_model_len": engine["max_model_len"],
        "gpu_memory_utilization": engine["gpu_memory_utilization"],
        "kv_cache_memory_bytes": engine.get("kv_cache_memory_bytes"),
        "enable_prefix_caching": bool(engine.get("enable_prefix_caching", False)),
        "quantization": engine.get("quantization"),
        "dtype": engine.get("dtype", "bfloat16"),
        "seed": engine.get("seed", 0),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "0" if compile_fresh else None,
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "0" if compile_fresh else None,
        "temperature": sampling.get("temperature", NEUTRAL_SAMPLING["temperature"]),
        "top_p": sampling.get("top_p", NEUTRAL_SAMPLING["top_p"]),
        "top_k": NEUTRAL,
        "min_p": NEUTRAL,
        "repetition_penalty": NEUTRAL,
        "max_tokens": "the workload's output length",
        "ignore_eos": sampling.get("ignore_eos", False),
        "request_seed": sampling["seed"] if "seed" in sampling else "the request's index",
        "detokenize": sampling.get("detokenize", True),
        **{k: env[k] for k in env if k in DEFAULTS},
        "enable_flashinfer_autotune": False,
        "VLLM_DISABLE_COMPILE_CACHE": "1",
    }


def _default(key: str, ctx: Mapping[str, Any]) -> Any:
    value = DEFAULTS[key].value
    if value == CARD or key == "max_model_len":
        return ctx.get(key, value)
    if value != CHECKPOINT:
        return value
    gen = ctx.get("generation") or {}
    if key == "max_tokens":
        return gen.get("max_new_tokens", "the client's, else to the end of the context")
    return gen.get(key, NEUTRAL_SAMPLING.get(key, NEUTRAL))


def differences(values: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each of `values` (`settings`) that differs from `vllm serve`'s default at `VLLM_COMMIT` on this card and
    checkpoint: the setting, this engine's value, `vllm serve`'s, whose the difference is, and where vLLM sets it."""

    return [{"setting": k, "arm": v, "vllm_serve": d, "whose": DEFAULTS[k].whose, "where": DEFAULTS[k].where}
            for k, v in values.items() if v != (d := _default(k, ctx))]
