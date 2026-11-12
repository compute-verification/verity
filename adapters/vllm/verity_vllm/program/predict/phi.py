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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PHI_SCHEMA = "verity-predict/phi/v1"


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
    def from_json(cls, d: dict) -> "Phi":
        if d.get("schema") != PHI_SCHEMA:
            raise ValueError(f"phi schema {d.get('schema')!r} != {PHI_SCHEMA!r}")
        return cls(model=dict(d["model"]), target=dict(d["target"]), tp=dict(d.get("tp") or {"world": 1, "rank": 0}),
                   sampling=dict(d.get("sampling") or {"kind": "greedy"}), engine=dict(d.get("engine") or {}),
                   workload=dict(d.get("workload") or {}), execution=d.get("execution", "bi-eager"), profile=d.get("profile", "b1-eager-v3"))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def hf_config(self) -> dict:
        return dict(self.model.get("hf_config") or {})

    @property
    def requests(self) -> list[dict]:
        return list(self.workload.get("requests") or [])

    def target_profile(self):
        """The declared `TargetProfile`; where the target block does not declare `batch_invariant`, the one the execution label names
        (`TargetProfile.execution_label`: a `bi-*` row runs the batch-invariant kernels, an `eager` row vLLM's defaults)."""
        from dataclasses import fields
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
    return int(r.get("prompt_len") or r.get("input_len"))


def _cap(r: dict) -> int:
    return int(r.get("max_tokens") or (r.get("sampling_params") or {}).get("max_tokens") or 0)


def load_phi(path: str | Path) -> Phi:
    return Phi.from_json(json.loads(Path(path).read_text()))


#: the engine args a served engine's φ carries from its engine record: the ones class G states (`engine_classes.CLASS_G_LIMITS`,
#: the prefix cache). A request Program's length rule reads ``max_model_len`` (`steps.compile_steps`), so two compiles of one
#: served engine's requests that read them differently compile different Programs
REGISTERED_ENGINE = ("max_num_batched_tokens", "max_model_len", "long_prefill_token_threshold", "max_num_seqs", "enable_prefix_caching")


def engine_of(meta: Mapping[str, Any], rank: int) -> tuple[dict, dict]:
    """A served engine's tensor-parallel world at ``rank`` and its engine args (`REGISTERED_ENGINE`), from the run's engine record
    (``meta``'s ``engine_record``: `LLM.engine_record`'s ``engine_kwargs``; where the launch left ``max_model_len`` to vLLM, -1,
    the length the built engine reads back, ``effective``), and, from a record of the environment its build set
    (``env_flags``), the DSA top-k's (`engine_classes.DSA_TOPK_ENV`) as set or not, so `predict.dsa` refuses a row past INDEX_TOPK
    the engine served in the kernels' own order; a ValueError for a rank the world doesn't have."""
    from verity_vllm.engine.profiles.engine_classes import DSA_TOPK_ENV

    rec = meta.get("engine_record")
    kw = None if rec is None else rec.get("engine_kwargs")
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
    compiles are the Programs the other wrote."""
    tp, engine = engine_of(meta, rank)
    dtype = checkpoint_dtype(hf_config)
    return Phi(model={"repo": meta["model"], "revision": meta["revision"], "hf_config": dict(hf_config), **({"dtype": dtype} if dtype else {})},
               target=dict(meta["target"]), execution=meta.get("execution") or "eager", tp=tp, engine=engine)


def save_phi(phi: Phi, path: str | Path) -> None:
    Path(path).write_text(json.dumps(phi.to_json(), indent=1, sort_keys=True) + "\n")
