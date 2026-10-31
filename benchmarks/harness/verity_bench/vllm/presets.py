"""Strict benchmark specs and deterministic request sweeps for vLLM."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from verity_config import Admission, ConfigError, gpus, load, sha256

VLLM_PIN = "d9105ea8001e0a6d77a96327d17515bb5791fb36"
SPEC_ROOT = Path(__file__).resolve().parents[3] / "specs"
SOURCE_ROOT = SPEC_ROOT.parents[1]
GPUS = gpus()
ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
# What the dispatcher, the queue or the launcher sets: a spec naming one would be overridden or would override them.
RESERVED_ENV = re.compile(r"VERITY_BENCH_\w*|RESEARCH_\w*|GPU_LEASE_\w*|CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER|PYTHONPATH|PATH")
#: A sweep point's sampling, as vLLM `SamplingParams` fields. Every request also carries its own seed (its index in the
#: point), so stochastic tokens are a function of the engine's numerics and an arm is gated against its control on them.
SAMPLING: Mapping[str, Mapping[str, float]] = {
    "greedy": {"temperature": 0.0},
    "stochastic": {"temperature": 0.8, "top_p": 0.95},
}


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

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


#: Engine precisions an arm may name (`runner.engine_args`). ``int8_exact`` is the eager exact-int8 engine on an int8 pin
#: (`TargetProfile.int8_exact` ``"all"``: every linear and every attention layer on the plain exact-int8 arm)
PRECISIONS = ("bfloat16", "fp8", "int8_exact")
#: The keys a preset's `engine_settings` may set: those `runner.engine_args` reads, and the two `served` sets from the
#: spec's `eager` (``cuda_graphs``, ``enforce_eager``), which a spec's own value doesn't change
ENGINE_SETTINGS = (
    "max_model_len", "max_num_seqs", "max_num_batched_tokens", "resident_gpu_memory_utilization",
    "block_gpu_memory_utilization", "resident_kv_cache_memory_bytes", "block_kv_cache_memory_bytes", "capture_sizes",
    "serving_defaults", "attention_backend", "attention_config", "cuda_graphs", "enforce_eager",
)
#: A protocol arm's `control` that gives it none: its tokens are checked by no other arm's, and the table names it
NO_CONTROL = "none"
#: The executions a served variant of a built-in arm (`BUILTIN@LABEL`, `arms.served_variant`) builds its engine in: vLLM's
#: serving defaults, eager or compiled (`verity_vllm.engine.served.EXECUTIONS`)
SERVED_EXECUTIONS = ("eager", "compiled")


@dataclass(frozen=True)
class ArmSettings:
    """A protocol arm's own engine (`[arm_settings."pouw:SCHEME"]`): the precision its engine loads, its control (an arm
    the spec names, or `NO_CONTROL`; default stock FP8), and the protocols config handed to `verity_vllm.LLM`
    (``{protocol: {...}}``), where a string value ``"$NAME"`` is the point's environment variable NAME, which the spec's
    setup exports. A served variant (`[arm_settings."stock-fp8@compiled"]`) takes only its control (default its
    built-in) and its own checkpoint, `model` at `revision` (default the preset's), for a preset whose checkpoint its
    precision can't load (stock FP8 against the int8 pin, from the pin's BF16 base). All of it is in the point's
    identity."""

    precision: str | None = None
    control: str | None = None
    protocols_config: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    model: str | None = None
    revision: str | None = None


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
    arm_settings: Mapping[str, ArmSettings] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    setup: str | None = None
    admission: Admission = Admission()
    warmups: int = 1
    max_warmups: int | None = None


@dataclass(frozen=True)
class Sweep:
    schema: str
    id: str
    batches: tuple[int, ...]
    prompt_tokens: tuple[int, ...]
    traffic: tuple[str, ...]
    output_tokens: Mapping[str, int]
    sampling: tuple[str, ...] = ("greedy",)


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
    arm_settings: Mapping[str, ArmSettings] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    setup: str | None = None
    admission: Admission = Admission()
    warmups: int = 1
    max_warmups: int | None = None

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
    unknown = sorted(set(doc.preset.engine_settings) - set(ENGINE_SETTINGS))
    if unknown:
        raise ConfigError(f"{path}: unknown engine_settings {unknown}; known: {list(ENGINE_SETTINGS)}")
    if doc.reps < 1 or doc.warmups < 1 or not doc.arms:
        raise ConfigError(f"{path}: arms must be nonempty and reps and warmups must be positive")
    if doc.max_warmups is not None and doc.max_warmups < doc.warmups:
        raise ConfigError(f"{path}: max_warmups must be at least warmups")
    bad = sorted(name for name in doc.env if not ENV_NAME.fullmatch(name) or RESERVED_ENV.fullmatch(name))
    if bad:
        raise ConfigError(f"{path}: env names must be upper-case and not the dispatcher's, queue's or launcher's: {bad}")
    if doc.setup is not None:
        setup = PurePosixPath(doc.setup)
        if setup.is_absolute() or ".." in setup.parts or not (SOURCE_ROOT / setup).is_file():
            raise ConfigError(f"{path}: setup {doc.setup!r} must name a script in the source tree, relative to its root")
    for name, arm in doc.arm_settings.items():
        variant = name.partition("@")[2] in SERVED_EXECUTIONS
        if name not in doc.arms or not (":" in name or variant):
            raise ConfigError(f"{path}: arm_settings {name!r} must name a protocol arm (protocol:scheme) or a served "
                              f"variant (BUILTIN@{'|'.join(SERVED_EXECUTIONS)}) in arms")
        if variant and (arm.precision is not None or arm.protocols_config):
            raise ConfigError(f"{path}: arm_settings {name!r}: a served variant takes only its control and checkpoint")
        if not variant and (arm.model is not None or arm.revision is not None):
            raise ConfigError(f"{path}: arm_settings {name!r}: only a served variant names its own checkpoint")
        if (arm.model is None) != (arm.revision is None):
            raise ConfigError(f"{path}: arm_settings {name!r}: a checkpoint is a model at a pinned revision: set both")
        if arm.precision is not None and arm.precision not in PRECISIONS:
            raise ConfigError(f"{path}: arm_settings {name!r}: precision is one of {', '.join(PRECISIONS)}")
        if arm.control is not None and arm.control != NO_CONTROL and (arm.control not in doc.arms or arm.control == name):
            raise ConfigError(f"{path}: arm_settings {name!r}: control must be another arm in arms, or {NO_CONTROL!r}")
    return doc


def load_sweep(name: str) -> Sweep:
    path = SPEC_ROOT / f"{name}.toml"
    doc = load(path, Sweep)
    if doc.schema != "verity-bench/vllm-sweep/v0" or doc.id != name:
        raise ConfigError(f"{path}: expected sweep {name!r}")
    if not doc.batches or not doc.prompt_tokens or not doc.traffic or not doc.sampling:
        raise ConfigError(f"{path}: sweep axes must be nonempty")
    missing = sorted(set(doc.traffic) - set(doc.output_tokens))
    if missing:
        raise ConfigError(f"{path}: output_tokens has no value for {missing}")
    unknown = sorted(set(doc.sampling) - set(SAMPLING))
    if unknown:
        raise ConfigError(f"{path}: unknown sampling {unknown}; known: {sorted(SAMPLING)}")
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
        arm_settings=doc.arm_settings,
        env=doc.env,
        setup=doc.setup,
        admission=doc.admission,
        warmups=doc.warmups,
        max_warmups=doc.max_warmups,
    )


def served(preset: Preset, eager: bool) -> tuple[Preset, bool]:
    """The preset as its engines are built, and whether they run eagerly: when asked, and with CUDA graphs exactly when
    not."""

    eager = bool(eager)
    settings = {**preset.engine_settings, "cuda_graphs": not eager, "enforce_eager": eager}
    return replace(preset, engine_settings=settings), eager


def checkpoint_files(root: Path) -> list[Path]:
    """The files of a local checkpoint that bind it: all but a Hub snapshot's repository metadata (dot-files such as
    ``.gitattributes``, and Markdown), which no loader reads and which a re-fetched snapshot can add (a node's
    llama-3.1-8b-instruct gained ``.gitattributes`` and ``README.md`` with every weight unchanged)."""

    return sorted(p for p in root.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix != ".md")


def local_checkpoint(preset: Preset, path: str | Path) -> Preset:
    """Bind a machine-local checkpoint to a preset by hashing its files (`checkpoint_files`)."""

    root = Path(path)
    files = checkpoint_files(root)
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
    """One batch, prompt envelope, traffic mix, and sampling."""

    batch: int
    prompt_tokens: int
    traffic: str
    output_tokens: int
    sampling: str = "greedy"

    @property
    def id(self) -> str:
        suffix = "" if self.sampling == "greedy" else f"-{self.sampling}"
        return f"b{self.batch}-p{self.prompt_tokens}-{self.traffic}{suffix}"

    @property
    def headline(self) -> bool:
        return self.batch == 32 and self.traffic == "mixed" and self.sampling == "greedy"

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
            "sampling": self.sampling,
            "staggered_arrivals": any(r.arrive_step for r in self.requests()),
        }


def expand_sweep(name: str) -> tuple[SweepPoint, ...]:
    """Expand a strict sweep document into its fixed Cartesian product."""

    sweep = load_sweep(name)
    return tuple(
        SweepPoint(batch, prompt, traffic, sweep.output_tokens[traffic], sampling)
        for batch in sweep.batches
        for prompt in sweep.prompt_tokens
        for traffic in sweep.traffic
        for sampling in sweep.sampling
    )
