"""φ — the base context known before execution, from which the predictor builds a Program without tracing.

    Phi.model     the pinned checkpoint: repo, revision and its `config.json` (the HF config the model class is built from)
    Phi.target    the declared `TargetProfile` (capability, SM count, batch invariance, numerics profile, FA version knob), and the
                  `device_name` its GPU reports where the row's class names one (`engine.profiles.devices`: vLLM's tuned configs)
    Phi.tp        {"world", "rank"}
    Phi.sampling  the row's declared sampling policy (`pipeline.build.sampling_policy` shape: greedy | stochastic)
    Phi.engine    the engine args (block_size, max_num_seqs, max_num_batched_tokens, chunked prefill, prefix caching,
                  disable_custom_all_reduce) and the `env` its engines run under (the all-reduce switches)
    Phi.workload  the workload manifest's requests (request_id, prompt length, max_tokens, arrive_step, seed, n)
    Phi.execution the row's execution label (`bi-eager`, `eager`): the target's batch invariance and execution mode

`score.phi_of(row)` reads φ from a Build row's DECLARED inputs only: the row id, its workload manifest and the pinned
checkpoint's `config.json`.  It never reads a Build output (descriptor, result, derive report): what the Build derived is the
ground truth the prediction is scored against, not an input to it.

`served_phi` is φ of an engine that served, from its run's record: the one formula every compile of a served engine's
requests reads φ by.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from verity_vllm.records.schema_ids import PHI_SCHEMA

__all__ = ["Phi"]


@dataclass
class Phi:
    schema: str = PHI_SCHEMA
    model: dict[str, Any] = field(default_factory=dict)       # {"repo", "revision", "hf_config"}
    target: dict[str, Any] = field(default_factory=dict)      # TargetProfile.to_json()
    tp: dict[str, Any] = field(default_factory=lambda: {"world": 1, "rank": 0})
    sampling: dict[str, Any] = field(default_factory=lambda: {"kind": "greedy"})
    engine: dict[str, Any] = field(default_factory=dict)
    workload: dict[str, Any] = field(default_factory=dict)
    execution: str = "bi-eager"
    profile: str = "b1-eager-v3"

    def to_json(self) -> dict:
        return {"schema": self.schema, "model": self.model, "target": self.target, "tp": self.tp, "sampling": self.sampling,
                "engine": self.engine, "workload": self.workload, "execution": self.execution, "profile": self.profile}

    @classmethod
    def from_json(cls, d: dict) -> Phi:
        if d.get("schema") != PHI_SCHEMA:
            raise ValueError(f"phi schema {d.get('schema')!r} != {PHI_SCHEMA!r}")
        engine, workload = d.get("engine"), d.get("workload")   # a φ that leaves one out declares no engine argument, no request
        return cls(model=dict(d["model"]), target=dict(d["target"]), tp=dict(d.get("tp") or {"world": 1, "rank": 0}),
                   sampling=dict(d.get("sampling") or {"kind": "greedy"}), engine=dict(engine) if engine else {},
                   workload=dict(workload) if workload else {}, execution=d.get("execution", "bi-eager"), profile=d.get("profile", "b1-eager-v3"))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def hf_config(self) -> dict:
        config = self.model.get("hf_config")   # a model block that carries no config.json has an empty one
        return dict(config) if config else {}

    @property
    def requests(self) -> list[dict]:
        declared = self.workload.get("requests")   # a workload that states none has no request
        return list(declared) if declared else []

    def target_profile(self):
        """The declared `TargetProfile`; where the target block does not declare `batch_invariant`, the one the execution label names
        (`TargetProfile.execution_label`: a `bi-*` row runs the batch-invariant kernels, an `eager` row vLLM's defaults)."""
        from verity_vllm.program.frontend.target_profile import TargetProfile
        names = {f.name for f in fields(TargetProfile)}
        tp = TargetProfile.default().with_overrides(**{k: (tuple(v) if k == "compute_capability" else v) for k, v in self.target.items()
                                                       if k in names})
        tp = tp if tp.numerics_profile == self.profile else tp.with_overrides(numerics_profile=self.profile)
        if "batch_invariant" in self.target:
            return tp
        return next((t for t in (tp.with_overrides(batch_invariant=b) for b in (True, False)) if t.execution_label == self.execution), tp)

    def request_shapes(self) -> list[tuple[int, int]]:
        """The distinct (LP, T) request shapes of the workload in first-occurrence order (T = max_tokens - 1 decode steps)."""
        seen: list[tuple[int, int]] = []
        for r in self.requests:
            s = (_prompt_len(r), _cap(r) - 1)
            if s not in seen:
                seen.append(s)
        return seen


def _prompt_len(r: dict) -> int:
    if r.get("prompt_token_ids") is not None:
        return len(r["prompt_token_ids"])
    return int(r.get("prompt_len") or r["input_len"])


def _cap(r: dict) -> int:
    params = r.get("sampling_params")   # where a request may state its `max_tokens` instead; one that states it nowhere has 0
    return int(r.get("max_tokens") or (params.get("max_tokens") if params else None) or 0)


def load_phi(path: str | Path) -> Phi:
    return Phi.from_json(json.loads(Path(path).read_text()))


#: the engine args a served engine's φ carries from its engine record: the ones class G states (`engine_classes.CLASS_G_LIMITS`,
#: the prefix cache). A request Program's length rule reads ``max_model_len`` (`steps.compile_steps`), so two compiles of one
#: served engine's requests that read them differently compile different Programs
REGISTERED_ENGINE = ("max_num_batched_tokens", "max_model_len", "long_prefill_token_threshold", "max_num_seqs", "enable_prefix_caching")
#: the ``kv_cache_dtype`` values a launch's request Programs are stated at: vLLM's default, and bf16 by name, the default's
#: format for a model computed in bf16
STATED_KV_CACHE = ("auto", "bfloat16")
#: the engine args that turn speculative decoding on: the config, and vLLM's short forms of its method, model and token count
SPECULATIVE = ("speculative_config", "spec_method", "spec_model", "spec_tokens")
#: the KV cache formats vLLM reads off a ``modelopt`` checkpoint's `quantization_config` (its `MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP`)
MODELOPT_KV_CACHE = {"fp8": "fp8_e4m3", "nvfp4": "nvfp4"}


def checkpoint_kv_cache(hf_config: Mapping[str, Any]) -> str | None:
    """The KV cache format vLLM serves a checkpoint on where the launch leaves ``kv_cache_dtype`` at its default, ``auto``, and
    the checkpoint's config turns a quantized one on; None where the cache stays the default's. It is vLLM's
    `get_kv_cache_quant_algo_string`: a `quantization_config` whose ``quant_method`` starts with ``modelopt`` and that names a
    KV cache scheme (``kv_cache_scheme`` or ``kv_cache_quant_algo``, under its ``quantization`` block first), an 8-bit static
    float scheme or the name ``fp8`` an FP8 cache, a 4-bit float one or ``nvfp4`` an NVFP4 cache. A scheme vLLM doesn't know
    leaves the default."""
    q = hf_config.get("quantization_config")
    if not isinstance(q, Mapping) or not str(q.get("quant_method", "")).startswith("modelopt"):
        return None
    inner = q.get("quantization", q)
    inner = inner if isinstance(inner, Mapping) else q
    algo = inner.get("kv_cache_scheme") or q.get("kv_cache_scheme") or inner.get("kv_cache_quant_algo") or q.get("kv_cache_quant_algo")
    if isinstance(algo, Mapping):
        if algo.get("dynamic") is False and algo.get("num_bits") == 8 and algo.get("type") == "float":
            algo = "fp8"
        elif algo.get("num_bits") == 4 and algo.get("type") == "float":
            algo = "nvfp4"
        else:
            return None
    return MODELOPT_KV_CACHE.get(algo.lower()) if isinstance(algo, str) else None


def _refuse_unstated(kw: Mapping[str, Any]) -> None:
    """A ValueError for engine args that serve what no request Program of this engine states: a KV cache format the launch
    names (a ``kv_cache_dtype`` outside `STATED_KV_CACHE`), whose cache other kernels than the default's may write and read, or
    speculative decoding (an arg of `SPECULATIVE`), whose decode rows hold draft positions. φ carries neither
    (`REGISTERED_ENGINE`): a request Program holds draft rows only under a registration that names a stand-in for the
    model's draft layer (`draft_rows`), which no engine ran. So the compile of such a launch would be Programs it did not run."""
    cache = kw.get("kv_cache_dtype")
    if cache is not None and cache not in STATED_KV_CACHE:
        raise ValueError(f"engine kv_cache_dtype={cache!r}: a request Program models the KV cache of a launch at kv_cache_dtype "
                         f"{' or '.join(STATED_KV_CACHE)} and no other format (an FP8 KV cache is not modelled), so the compile of "
                         "this launch would be Programs it did not run. Serve with kv_cache_dtype at vLLM's default to compile it")
    for k in SPECULATIVE:
        if kw.get(k) is not None:
            raise ValueError(f"engine {k}={kw[k]!r}: a served engine's request Programs model no draft tokens (speculative "
                             "decoding): no Definition of the model's draft layer is registered, so the compile of this launch "
                             "would be Programs it did not run. Serve with speculative decoding off to compile it")


def engine_of(meta: Mapping[str, Any], rank: int) -> tuple[dict, dict]:
    """A served engine's tensor-parallel world at ``rank`` and its engine args (`REGISTERED_ENGINE`), from the run's engine record
    (``meta``'s ``engine_record``: `LLM.engine_record`'s ``engine_kwargs``; where the launch left ``max_model_len`` to vLLM, -1,
    the length the built engine reads back, ``effective``), and, from a record of the environment its build set
    (``env_flags``), the DSA top-k's (`engine_classes.DSA_TOPK_ENV`) as set or not, so `predict.dsa` refuses a row past INDEX_TOPK
    the engine served in the kernels' own order; a ValueError for a rank the world doesn't have, and for a launch that names
    another KV cache format or decodes speculatively (`_refuse_unstated`)."""
    from verity_vllm.engine.profiles.engine_classes import DSA_TOPK_ENV

    rec = meta.get("engine_record")
    kw = None if rec is None else rec.get("engine_kwargs")
    if kw is not None:
        _refuse_unstated(kw)
    world = 1 if kw is None else int(kw.get("tensor_parallel_size") or 1)
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank}: the run's engine has tensor-parallel world {world}")
    engine = {} if kw is None else {k: kw[k] for k in REGISTERED_ENGINE if k in kw}
    if engine.get("max_model_len") == -1:
        effective = rec.get("effective")
        engine["max_model_len"] = None if effective is None else effective.get("max_model_len")
    if rec is not None and "env_flags" in rec:
        flags = rec["env_flags"]
        engine["env"] = {} if flags is None else {k: flags[k] for k in DSA_TOPK_ENV if k in flags}
    return {"world": world, "rank": rank}, engine


def checkpoint_dtype(hf_config: Mapping[str, Any]) -> str | None:
    """The checkpoint dtype φ states (``model["dtype"]``, in `config.DTYPES`' names, as `predict.score` states a served row's from its
    id), read from the pinned config's `quantization_config`: a block-FP8 checkpoint (``quant_method: fp8``, ``fmt: e4m3``) is
    ``float8_e4m3fn``, the dtype `predict.dsa`'s domain requires. Any other config states none, and the decoder takes the dtype its
    config's shape implies (bfloat16, or int8)."""
    from verity_vllm.config import DTYPES

    q = hf_config.get("quantization_config")
    return DTYPES["fp8"] if q is not None and q.get("quant_method") == "fp8" and q.get("fmt", "e4m3") == "e4m3" else None


def served_phi(meta: Mapping[str, Any], hf_config: Mapping[str, Any], rank: int = 0) -> Phi:
    """φ of an engine that served, from its run's record ``meta`` (its ``model`` and ``revision``, its ``target`` profile, its
    ``execution`` label, ``eager`` where it states none, and its ``engine_record``): the model at its pinned config ``hf_config``,
    with the checkpoint dtype that config states (`checkpoint_dtype`), on the run's target at its execution, at the engine's
    tensor-parallel world at ``rank`` with the engine args class G states (`engine_of`, whose ValueError it raises); no workload.
    A served window's registered compiler and the Builds its row dir holds both compile under it, so the Programs the one
    compiles are the Programs the other wrote. A ValueError too for a checkpoint whose config turns a quantized KV cache on
    at vLLM's default (`checkpoint_kv_cache`) under a launch that leaves ``kv_cache_dtype`` there: the launch's args don't name
    the cache, and no request Program states it. A launch of such a checkpoint that names the bf16 cache keeps that arg in φ."""
    tp, engine = engine_of(meta, rank)
    if (cache := checkpoint_kv_cache(hf_config)) is not None:
        stated = ((meta.get("engine_record") or {}).get("engine_kwargs") or {}).get("kv_cache_dtype") or "auto"
        if stated == "auto":
            raise ValueError(f"the checkpoint's quantization_config turns a {cache} KV cache on at kv_cache_dtype auto (a modelopt "
                             "checkpoint that names a KV cache scheme): a request Program models the cache of a launch at "
                             f"{' or '.join(STATED_KV_CACHE)} and no quantized one, so the compile of this launch would be "
                             "Programs it did not run")
        engine["kv_cache_dtype"] = stated
    dtype = checkpoint_dtype(hf_config)
    return Phi(model={"repo": meta["model"], "revision": meta["revision"], "hf_config": dict(hf_config), **({"dtype": dtype} if dtype else {})},
               target=dict(meta["target"]), execution=meta.get("execution") or "eager", tp=tp, engine=engine)


def save_phi(phi: Phi, path: str | Path) -> None:
    Path(path).write_text(json.dumps(phi.to_json(), indent=1, sort_keys=True) + "\n")
