"""Engine specs: a `verity_vllm.LLM` described by a strict config v0 document (`verity_config`).

    schema = "verity-vllm/engine-spec/v0"
    model = "unsloth/Llama-3.1-8B"
    revision = "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"

    [engine]
    quantization = "fp8"
    max_model_len = 8384
    enforce_eager = false
    compilation_config = { mode = 0, cudagraph_mode = "FULL_DECODE_ONLY", cudagraph_capture_sizes = [1, 8, 32, 128] }

``[engine]`` takes exactly `LLM`'s supported options (`pipeline.llm.SUPPORTED`), each typed; an unknown or missing key,
or a value of the wrong type, raises `verity_config.ConfigError` before any engine is built.  A key left out keeps
`LLM`'s default, as when a caller doesn't pass it.  `EngineSpec.digest` is the spec's canonical SHA-256
(`verity_config.sha256`, identity form), which `LLM.from_spec` records as ``engine_record["spec_digest"]``;
``protocols_config`` is left out of it, as `LLM` never hashes it.  Stdlib and `verity_config` only.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import verity_config

SCHEMA = "verity-vllm/engine-spec/v0"


@dataclass(frozen=True)
class Engine:
    """`LLM`'s options; None is "not set", so the default applies."""

    tensor_parallel_size: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    enforce_eager: bool | None = None
    dtype: str | None = None
    quantization: str | None = None
    compilation_config: Mapping[str, Any] | None = None
    seed: int | None = None
    protocols: tuple[str, ...] | None = None
    protocols_config: Mapping[str, Mapping[str, Any]] | None = field(default=None, metadata={"identity": False})


@dataclass(frozen=True)
class EngineSpec:
    schema: str
    model: str
    revision: str
    engine: Engine = Engine()

    @property
    def digest(self) -> str:
        return verity_config.sha256(self, identity=True)

    def options(self) -> dict[str, Any]:
        """The `LLM` keyword options this spec sets."""
        return {f.name: _plain(getattr(self.engine, f.name)) for f in fields(Engine) if getattr(self.engine, f.name) is not None}


def _plain(v: Any) -> Any:
    """Decoded TOML arrays are tuples; vLLM's settings take lists."""
    if isinstance(v, Mapping):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    return v


def from_mapping(raw: Mapping[str, Any], *, path: str = "$") -> EngineSpec:
    spec = verity_config.from_mapping(EngineSpec, dict(raw), path=path)
    if spec.schema != SCHEMA:
        raise verity_config.ConfigError(f"{path}: schema must be {SCHEMA!r}, got {spec.schema!r}")
    return spec


def load(path: str | Path) -> EngineSpec:
    spec = verity_config.load(path, EngineSpec)
    if spec.schema != SCHEMA:
        raise verity_config.ConfigError(f"{path}: schema must be {SCHEMA!r}, got {spec.schema!r}")
    return spec
