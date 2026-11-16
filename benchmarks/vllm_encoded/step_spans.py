"""Per-part timing spans of the encoded stack's steps: where a step's time goes, for ablations. Off unless an arm asks
(`bench_arm`'s ``spans``); no verdict reads them.

A mark (`paged.cu`'s ``pg_span``, one thread) reads the GPU's global timer and adds the time since its stream's last
mark to the part it names, then counts it. The stack's marks are captured into vLLM's CUDA graphs with the kernels
around them, so a replayed step accumulates its spans on the device with no host sync, and a read (one sync, between
runs) restarts the stream's clock so the gap before the next run is in no part.

~~~text
main stream   one step's timeline from its first mark to its last, each interval in the part of the mark that ends it:
  between        vLLM's sampling, scheduling and input preparation since the last step's last mark
  commit         the last pass's receipts closed: the selected KV slots' host read, their compaction and the step tree
  append_wait    the step waiting for the last step's deferred KV append (second stream)
  pass           the pass's start: its seed, signs and the KV receipts' pass state
  model          vLLM's own ops between the stack's: norms, RoPE, activation quantization, SiLU, residual adds
  linear         a decode's weight linears (M <= 128 rows): stack, codec decode + FP8 MMA + sketch receipts in one fused
                 launch; pstack, the plain GEMM. Then their chunk sums reduced
  attention      the decode attention over the paged cache (stack, its KV words decoded inside it), or FlashAttention
                 over the gathered pages
  kv_receipts    the KV receipts of the layer's selected slots
  gather         a prefill step's pages decoded (stack) or copied into its scratch, its own tokens written over them
  prefill_decode a prefill linear's weight decoded into its N x K matrix (stack) or unpermuted (pstack)
  prefill_gemm   that linear's CUTLASS GEMM
  lm_head        the lm_head: fused decode + BF16 MMA + sketch receipts (stack), the plain GEMM (pstack)
second stream
  append         the deferred KV append: every layer's new tokens encoded (stack) or copied into the pages
~~~

The main stream's parts sum to its steps' time. The append and the top-up (`bench_arm`) run beside it, so they overlap
it. A fused launch is one part: the codec decode of a decode step's weights and KV is inside ``linear``, ``lm_head``
and ``attention`` with their MMA and sketches, and only the audit's span over the plain arm's (the same part) is its
cost. Each mark's own time is in the part it ends; ``marks`` counts them.
"""
from __future__ import annotations

from typing import Any

PARTS = ("between", "commit", "append_wait", "pass", "model", "linear", "attention", "kv_receipts", "gather",
         "prefill_decode", "prefill_gemm", "lm_head", "idle", "append")
MAIN = PARTS[:PARTS.index("idle")]
P = {name: i for i, name in enumerate(PARTS)}
#: a step replayed from vLLM's full CUDA graph, or run eagerly (prefill, a mixed step, a decode past the captured sizes)
KINDS = ("graph", "eager")
#: an accumulator: the stream's last mark (ns, 0: none since the last read), the current kind, then each kind's parts'
#: nanoseconds and marks
SLOTS = 2 + len(KINDS) * 2 * len(PARTS)


def account(acc: list[int], part: int, kind: int, now: int) -> None:
    """``pg_span``'s arithmetic on an accumulator at global time ``now`` (ns): the kernel's reference."""
    if kind >= 0:
        acc[1] = kind
    if acc[0]:
        base = 2 + acc[1] * 2 * len(PARTS)
        acc[base + part] += now - acc[0]
        acc[base + len(PARTS) + part] += 1
    acc[0] = now


def decode(main: list[int], side: list[int], steps: list[int], tokens: list[int]) -> dict[str, Any]:
    """The accumulators' totals as the arm's cumulative spans: each kind's steps, tokens, main-stream parts (ms), marks
    and second-stream append (ms)."""
    if len(main) != SLOTS or len(side) != SLOTS:
        raise ValueError(f"a span accumulator has {SLOTS} slots")
    out: dict[str, Any] = {}
    for k, kind in enumerate(KINDS):
        base = 2 + k * 2 * len(PARTS)
        out[kind] = {
            "steps": steps[k],
            "tokens": tokens[k],
            "ms": {name: main[base + P[name]] / 1e6 for name in MAIN},
            "marks": {name: main[base + len(PARTS) + P[name]] for name in MAIN},
            "append_ms": side[base + P["append"]] / 1e6,
        }
    return out


def per_step(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    """One run's spans a step, from the cumulative spans before and after it: each kind's parts (ms), their sum
    (``step_ms``), marks and append, for every kind the run stepped."""
    out: dict[str, float] = {}
    for kind in KINDS:
        a, b = after.get(kind), before.get(kind, {})
        if not a:
            continue
        steps = a["steps"] - b.get("steps", 0)
        if steps <= 0:
            continue
        bms, bmarks = b.get("ms", {}), b.get("marks", {})
        ms = {name: (a["ms"][name] - bms.get(name, 0.0)) / steps for name in MAIN}
        for name, value in ms.items():
            out[f"{kind}.{name}_ms"] = value
        out[f"{kind}.step_ms"] = sum(ms.values())
        out[f"{kind}.marks"] = sum(a["marks"][name] - bmarks.get(name, 0) for name in MAIN) / steps
        out[f"{kind}.append_ms"] = (a["append_ms"] - b.get("append_ms", 0.0)) / steps
        out[f"{kind}.steps"] = float(steps)
        out[f"{kind}.tokens_per_step"] = (a["tokens"] - b.get("tokens", 0)) / steps
    return out


class Spans:
    """The stack's two accumulators on the device (the main stream's, the deferred append's) and the host's step
    counts. ``mark`` launches on the caller's stream, captured or not."""

    def __init__(self, lib: Any, torch: Any) -> None:
        self.lib, self.torch = lib, torch
        self.main = torch.zeros(SLOTS, dtype=torch.int64, device="cuda")
        self.side = torch.zeros(SLOTS, dtype=torch.int64, device="cuda")
        self.steps = [0] * len(KINDS)
        self.tokens = [0] * len(KINDS)

    def mark(self, part: int, kind: int = -1, s: int | None = None) -> None:
        self._launch(self.main, part, kind, s)

    def mark_side(self, part: int, kind: int, s: int) -> None:
        self._launch(self.side, part, kind, s)

    def _launch(self, acc: Any, part: int, kind: int, s: int | None) -> None:
        stream = self.torch.cuda.current_stream().cuda_stream if s is None else s
        self.lib.call("pg_span", acc.data_ptr(), part, kind, len(PARTS), len(KINDS), stream)

    def step(self, kind: int, tokens: int) -> None:
        self.steps[kind] += 1
        self.tokens[kind] += tokens

    def read(self) -> dict[str, Any]:
        """The totals so far (one device sync), each stream's clock restarted."""
        torch = self.torch
        torch.cuda.synchronize()
        main, side = self.main.tolist(), self.side.tolist()
        self.main[0].zero_()
        self.side[0].zero_()
        torch.cuda.synchronize()
        return decode(main, side, self.steps, self.tokens)


__all__ = ["KINDS", "MAIN", "P", "PARTS", "SLOTS", "Spans", "account", "decode", "per_step"]
