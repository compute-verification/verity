"""Named machines and deterministic request sweeps for ``verity-bench vllm``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

VLLM_PIN = "d9105ea8001e0a6d77a96327d17515bb5791fb36"

# Stable ids are kept in this one table until the shared GPU registry lands.
GPUS: dict[str, str] = {
    "rtx-pro-6000": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "h200": "NVIDIA H200",
}

PRESET_FACTS: dict[str, dict[str, Any]] = json.loads(
    Path(__file__).with_name("presets.json").read_text()
)


@dataclass(frozen=True)
class Preset:
    """One reproducible model, precision, machine, and engine configuration."""

    id: str
    model: str
    revision: str | None
    case: str | None
    precision: str
    gpu_id: str
    gpu_count: int
    engine_settings: Mapping[str, Any] = field(default_factory=dict)
    vllm_pin: str = VLLM_PIN
    model_digest: str | None = None
    unavailable_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


COMMON = {
    "cuda_graphs": True,
    "capture_sizes": [1, 8, 32, 128],
    "max_model_len": 8_384,
    "resident_gpu_memory_utilization": 0.40,
    "block_gpu_memory_utilization": 0.80,
}

PRESETS: dict[str, Preset] = {
    name: Preset(
        id=name,
        **{
            **facts,
            "engine_settings": {
                **COMMON,
                **facts.get("engine_settings", {}),
            },
        },
    )
    for name, facts in PRESET_FACTS.items()
}

unknown_gpus = sorted({preset.gpu_id for preset in PRESETS.values()} - GPUS.keys())
if unknown_gpus:
    raise ValueError(f"benchmark presets name unknown GPU ids: {unknown_gpus}")


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown benchmark preset {name!r} (known: {', '.join(PRESETS)})") from None


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
    """Expand ``quick`` (two batches, one mixed shape) or the fixed full panel."""

    if name == "quick":
        return tuple(SweepPoint(b, 512, "mixed", 64) for b in (8, 32))
    if name != "full":
        raise ValueError("benchmark sweep must be 'quick' or 'full'")
    output = {"decode-heavy": 128, "prefill-heavy": 8, "mixed": 64}
    return tuple(
        SweepPoint(batch, prompt, traffic, output[traffic])
        for batch in (1, 8, 32, 128)
        for prompt in (512, 8192)
        for traffic in ("decode-heavy", "prefill-heavy", "mixed")
    )
