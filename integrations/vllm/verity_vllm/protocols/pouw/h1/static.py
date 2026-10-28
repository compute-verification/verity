"""A static H1 window's declaration and its u (`pouw_h1`'s static layout, the ``u`` of
`benchmarks/pouw/period_cost.py`): each step's rows a function of the declaration alone, so its u is public."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence


def utilization(step_rows: Sequence[int], rows: int, steps: int) -> Fraction:
    """The window's u: its served rows over its declared rows, Σ_s m_s / (S·T)."""
    if len(step_rows) > steps or any(not 0 <= m <= rows for m in step_rows):
        raise ValueError(f"{list(step_rows)} is not a window of {steps} steps of at most {rows} rows")
    return Fraction(sum(step_rows), rows * steps)


@dataclass(frozen=True)
class Static:
    """A static H1 window's declaration (the static schedule's ``sampling.ignore_eos``): requests of ``prompt_lens`` tokens, all admitted
    at step 0 and prefilled there in one step, each running exactly ``max_tokens`` steps (no early stop, no preemption, no host
    prefix cache), at T ``rows`` a call and S ``steps``. Each step's rows are then a function of the declaration alone
    (`schedule`), and so is the window's u."""

    prompt_lens: tuple[int, ...]
    max_tokens: int
    rows: int
    steps: int

    def __post_init__(self) -> None:
        if self.rows % 32 or self.rows <= 0:
            raise ValueError(f"T = {self.rows} rows is not a positive multiple of 32")
        if not self.prompt_lens or min(self.prompt_lens) < 1 or sum(self.prompt_lens) > self.rows:
            raise ValueError(f"prompts of {list(self.prompt_lens)} tokens don't prefill in one step of {self.rows} rows")
        if not 1 <= self.max_tokens <= self.steps:
            raise ValueError(f"a cap of {self.max_tokens} tokens doesn't fit a window of {self.steps} steps")

    @property
    def slots(self) -> int:
        return len(self.prompt_lens)

    def schedule(self) -> list[int]:
        """Each step's rows: the prompts' tokens at step 0, one row a request for its other ``max_tokens`` - 1 steps, then the
        window's empty steps."""
        return [sum(self.prompt_lens)] + [self.slots] * (self.max_tokens - 1) + [0] * (self.steps - self.max_tokens)

    @property
    def u(self) -> Fraction:
        return utilization(self.schedule(), self.rows, self.steps)

    def engine_args(self) -> dict[str, Any]:
        """vLLM's scheduler limits for the declaration: every request in flight at once, T tokens a step, no prefix cache."""
        return {"max_num_seqs": self.slots, "max_num_batched_tokens": self.rows, "enable_chunked_prefill": True,
                "enable_prefix_caching": False}

    def check(self, realized: Sequence[int]) -> list[str]:
        """Each step whose served rows aren't the declared ones, by name; empty when the window is the declaration."""
        want = self.schedule()
        if len(realized) > len(want):
            return [f"{len(realized)} served steps past the window's {len(want)}"]
        got = list(realized) + [0] * (len(want) - len(realized))
        return [f"step {s}: {m} rows served, {d} declared ({'fewer' if m < d else 'more'}: an early stop, a preemption or an "
                f"admission off the declaration decided it, not the public schedule)"
                for s, (m, d) in enumerate(zip(got, want)) if m != d]


def bind_all(calls: Sequence[Any], bound: Mapping[int, Mapping[str, Any]]) -> list[tuple[bytes, bytes] | None]:
    """Each call's (root(A), root(B)) from what its listeners bound (``root_a``, ``root_b``, hex), None where either is missing."""
    out: list[tuple[bytes, bytes] | None] = []
    for c in calls:
        b = bound.get(c.index, {})
        out.append((bytes.fromhex(b["root_a"]), bytes.fromhex(b["root_b"])) if "root_a" in b and "root_b" in b else None)
    return out


def graph_pad(m: int, batch: int, sizes: Sequence[int]) -> int:
    """The rows a CUDA graph pads a step of ``batch`` scheduled rows with, when the call ran ``m``: m - batch when m is the
    smallest capture size (``sizes``) at least the batch, vLLM's padding, else 0, so any other mismatch stays a gap."""
    fit = [z for z in sorted(sizes) if z >= batch]
    return m - batch if batch < m and fit and fit[0] == m else 0


#: 64 one-token prompts at cap 16, every row of the 16 served steps filled (u = 16/18)
FILL = Static((1,) * 64, 16, 64, 18)


__all__ = ["FILL", "Static", "bind_all", "graph_pad", "utilization"]
