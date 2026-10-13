"""Named machines and deterministic request sweeps for ``verity-vllm bench``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from verity_vllm.engine.profiles.benchmark import PRESET_FACTS

VLLM_PIN = "d9105ea8001e0a6d77a96327d17515bb5791fb36"


@dataclass(frozen=True)
class Preset:
    """One reproducible model, precision, machine, and engine configuration."""

    id: str
    model: str
    revision: str | None
    case: str | None
    precision: str
    gpu_type: str
    gpu_count: int
    engine_settings: Mapping[str, Any] = field(default_factory=dict)
    vllm_pin: str = VLLM_PIN
    unavailable_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


COMMON = {
    "cuda_graphs": True,
    "capture_sizes": [1, 8, 32, 128],
    "max_num_batched_tokens": 16_384,
    "max_model_len": 8_384,
    "resident_gpu_memory_utilization": 0.40,
    "block_gpu_memory_utilization": 0.90,
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


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown benchmark preset {name!r} (known: {', '.join(PRESETS)})") from None


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
