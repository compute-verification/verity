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
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
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
    stated: Any = r.get("prompt_len") or r.get("input_len")   # a request that states neither has no length: `int`'s TypeError
    return int(stated)


def _cap(r: dict) -> int:
    params = r.get("sampling_params")   # where a request may state its `max_tokens` instead; one that states it nowhere has 0
    return int(r.get("max_tokens") or (params.get("max_tokens") if params else None) or 0)
