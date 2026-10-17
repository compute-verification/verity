"""Strict benchmark specs and deterministic request sweeps for vLLM."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from verity_config import Admission, ConfigError, gpus, load, sha256

VLLM_PIN = "d9105ea8001e0a6d77a96327d17515bb5791fb36"
SPEC_ROOT = Path(__file__).resolve().parents[3] / "specs"
GPUS = gpus()


@dataclass(frozen=True)
class Preset:
    """One reproducible model, precision, machine, and engine configuration."""

    id: str
    model: str
    precision: str
    gpu_id: str
    gpu_count: int
    revision: str | None = None
    case: str | None = None
    engine_settings: Mapping[str, Any] = field(default_factory=dict)
    vllm_pin: str = VLLM_PIN
    model_digest: str | None = None
    unavailable_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSpec:
    """One runnable benchmark configuration before its named sweep is resolved."""

    schema: str
    backend: str
    id: str
    arms: tuple[str, ...]
    sweep: str
    reps: int
    eager: bool
    preset: Preset
    arm_modules: tuple[str, ...] = ()
    admission: Admission = Admission()


@dataclass(frozen=True)
class Sweep:
    schema: str
    id: str
    batches: tuple[int, ...]
    prompt_tokens: tuple[int, ...]
    traffic: tuple[str, ...]
    output_tokens: Mapping[str, int]


@dataclass(frozen=True)
class ResolvedSpec:
    schema: str
    backend: str
    id: str
    arms: tuple[str, ...]
    sweep: Sweep
    reps: int
    eager: bool
    preset: Preset
    arm_modules: tuple[str, ...] = ()
    admission: Admission = Admission()

    @property
    def config_sha256(self) -> str:
        return sha256(self, identity=True)


def _spec_path(spec: str | Path) -> Path:
    path = Path(spec)
    if path.is_file():
        return path
    named = SPEC_ROOT / f"{spec}.toml"
    if named.is_file():
        return named
    raise ConfigError(f"no benchmark spec {spec!r}")


def load_spec(spec: str | Path) -> BenchmarkSpec:
    path = _spec_path(spec)
    doc = load(path, BenchmarkSpec)
    if doc.schema != "verity-bench/vllm-spec/v0" or doc.backend != "vllm":
        raise ConfigError(f"{path}: expected a verity-bench/vllm-spec/v0 document")
    if doc.id != doc.preset.id:
        raise ConfigError(f"{path}: spec id {doc.id!r} differs from preset id {doc.preset.id!r}")
    if doc.preset.gpu_id not in GPUS:
        raise ConfigError(f"{path}: unknown GPU id {doc.preset.gpu_id!r}")
    if doc.preset.gpu_count < 1 or doc.preset.gpu_count > GPUS[doc.preset.gpu_id].count_per_node:
        raise ConfigError(f"{path}: gpu_count is outside {doc.preset.gpu_id!r}'s per-node capacity")
    if doc.reps < 1 or not doc.arms:
        raise ConfigError(f"{path}: arms must be nonempty and reps must be positive")
    return doc


def load_sweep(name: str) -> Sweep:
    path = SPEC_ROOT / f"{name}.toml"
    doc = load(path, Sweep)
    if doc.schema != "verity-bench/vllm-sweep/v0" or doc.id != name:
        raise ConfigError(f"{path}: expected sweep {name!r}")
    if not doc.batches or not doc.prompt_tokens or not doc.traffic:
        raise ConfigError(f"{path}: sweep axes must be nonempty")
    missing = sorted(set(doc.traffic) - set(doc.output_tokens))
    if missing:
        raise ConfigError(f"{path}: output_tokens has no value for {missing}")
    return doc


def resolve_spec(spec: str | Path) -> ResolvedSpec:
    doc = load_spec(spec)
    return ResolvedSpec(
        schema=doc.schema,
        backend=doc.backend,
        id=doc.id,
        arms=doc.arms,
        sweep=load_sweep(doc.sweep),
        reps=doc.reps,
        eager=doc.eager,
        preset=doc.preset,
        arm_modules=doc.arm_modules,
        admission=doc.admission,
    )


def served(preset: Preset, eager: bool) -> tuple[Preset, bool]:
    """The preset as its engines are built, and whether they run eagerly: when asked, or when the preset requires it
    (`enforce_eager`), and with CUDA graphs exactly when not."""

    eager = bool(eager or preset.engine_settings.get("enforce_eager", False))
    settings = {**preset.engine_settings, "cuda_graphs": not eager, "enforce_eager": eager}
    return replace(preset, engine_settings=settings), eager


def local_checkpoint(preset: Preset, path: str | Path) -> Preset:
    """Bind a machine-local checkpoint to a preset by hashing all of its files."""

    root = Path(path)
    files = sorted(p for p in root.iterdir() if p.is_file())
    if not files:
        raise ValueError(f"benchmark checkpoint {root} has no files")
    digest = hashlib.sha256()
    for file in files:
        name = file.name.encode()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(file.stat().st_size.to_bytes(8, "big"))
        content = hashlib.sha256()
        with file.open("rb") as stream:
            while chunk := stream.read(8 << 20):
                content.update(chunk)
        digest.update(content.digest())
    return replace(
        preset,
        model=str(root),
        revision=None,
        case=None,
        model_digest=digest.hexdigest(),
        unavailable_reason=None,
    )


@dataclass(frozen=True)
class RequestShape:
    prompt_tokens: int
    output_tokens: int
    arrive_step: int

    def to_json(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SweepPoint:
    """One batch, prompt envelope, and traffic mix."""

    batch: int
    prompt_tokens: int
    traffic: str
    output_tokens: int

    @property
    def id(self) -> str:
        return f"b{self.batch}-p{self.prompt_tokens}-{self.traffic}"

    @property
    def headline(self) -> bool:
        return self.batch == 32 and self.traffic == "mixed"

    def requests(self) -> tuple[RequestShape, ...]:
        if self.traffic == "decode-heavy":
            return tuple(RequestShape(self.prompt_tokens, self.output_tokens, 0) for _ in range(self.batch))
        if self.traffic == "prefill-heavy":
            return tuple(RequestShape(self.prompt_tokens, self.output_tokens, 0) for _ in range(self.batch))
        if self.traffic != "mixed":
            raise ValueError(f"unknown traffic shape {self.traffic!r}")
        wave = max(1, self.batch // 4)
        return tuple(
            RequestShape(
                self.prompt_tokens if i % 2 == 0 else max(16, self.prompt_tokens // 4),
                self.output_tokens if i % 3 else max(8, self.output_tokens // 2),
                i // wave,
            )
            for i in range(self.batch)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "batch": self.batch,
            "prompt_tokens": self.prompt_tokens,
            "traffic": self.traffic,
            "output_tokens": self.output_tokens,
            "staggered_arrivals": any(r.arrive_step for r in self.requests()),
        }


def expand_sweep(name: str) -> tuple[SweepPoint, ...]:
    """Expand a strict sweep document into its fixed Cartesian product."""

    sweep = load_sweep(name)
    return tuple(
        SweepPoint(batch, prompt, traffic, sweep.output_tokens[traffic])
        for batch in sweep.batches
        for prompt in sweep.prompt_tokens
        for traffic in sweep.traffic
    )
