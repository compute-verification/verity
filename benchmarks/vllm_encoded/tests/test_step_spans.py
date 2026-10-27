"""The step spans' accounting (`step_spans.py`) on an injected global timer, and, on a GPU with the arms' paged library
built, `paged.cu`'s pg_span against it."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_spans as SS

P = SS.P
GRAPH, EAGER = SS.KINDS.index("graph"), SS.KINDS.index("eager")


def decode_marks(layers: int = 32) -> list[tuple[str, int]]:
    """A decode step's marks in the stack's order: the pass's start (hook_passes.begin), then each layer's qkv linear,
    attention and its KV receipts, o, gate_up and down, then the lm_head; the first sets the step's kind."""
    marks = [("between", GRAPH), ("commit", -1), ("append_wait", -1), ("pass", -1)]
    for _ in range(layers):
        marks += [("model", -1), ("linear", -1), ("model", -1), ("attention", -1), ("kv_receipts", -1)]
        marks += [("model", -1), ("linear", -1)] * 3
    return marks + [("model", -1), ("lm_head", -1)]


def prefill_marks(layers: int = 32) -> list[tuple[str, int]]:
    marks = [("between", EAGER), ("commit", -1), ("append_wait", -1), ("pass", -1)]
    for _ in range(layers):
        marks += [("model", -1), ("prefill_decode", -1), ("prefill_gemm", -1), ("model", -1), ("gather", -1),
                  ("kv_receipts", -1), ("gather", -1), ("attention", -1)]
        marks += [("model", -1), ("prefill_decode", -1), ("prefill_gemm", -1)] * 3
    return marks + [("model", -1), ("lm_head", -1)]


class Timeline:
    """The main stream's accumulator and the host's step counts, driven by an injected global timer: each mark lands
    a seeded gap after the stream's last one."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.now = 10**12
        self.main = [0] * SS.SLOTS
        self.side = [0] * SS.SLOTS
        self.side_free = 0
        self.steps, self.tokens = [0] * len(SS.KINDS), [0] * len(SS.KINDS)

    def step(self, marks: list[tuple[str, int]], tokens: int) -> list[int]:
        """The step's marks; returns each mark's time."""
        kind = marks[0][1]
        self.steps[kind] += 1
        self.tokens[kind] += tokens
        times = []
        for part, k in marks:
            self.now += self.rng.randrange(1_000, 400_000)
            SS.account(self.main, P[part], k, self.now)
            times.append(self.now)
        return times

    def append(self, kind: int) -> None:
        """The deferred append on the second stream once the step's forward is done and its last append ended, beside
        the next step: in no main-stream part."""
        start = max(self.now, self.side_free)
        SS.account(self.side, P["idle"], kind, start)
        self.side_free = start + 2_500_000
        SS.account(self.side, P["append"], kind, self.side_free)

    def read(self) -> dict:
        """`Spans.read`: the totals, then each stream's clock restarted."""
        out = SS.decode(self.main, self.side, self.steps, self.tokens)
        self.main[0] = self.side[0] = 0
        return out


def test_a_runs_parts_sum_to_its_steps_and_the_gap_between_runs_is_in_none():
    t = Timeline(1)
    for _ in range(3):
        t.step(decode_marks(), 8)
    before = t.read()
    t.now += 7 * 10**9                       # the harness between two runs: in no part
    starts, last = [], 0
    for _ in range(20):
        times = t.step(decode_marks(), 8)
        starts.append(times[0])
        last = times[-1]
        t.append(GRAPH)
    after = t.read()
    got = SS.per_step(before, after)
    assert got["graph.steps"] == 20 and got["graph.tokens_per_step"] == 8
    # the parts telescope: a step's (its first mark to the next's) to the float, the last ending at its last mark
    steps = [b - a for a, b in zip(starts, starts[1:])] + [last - starts[-1]]
    assert got["graph.step_ms"] == pytest.approx(sum(steps) / 20 / 1e6, rel=1e-12)
    assert got["graph.step_ms"] < 1000      # the 7 s between the runs is in no part
    assert got["graph.marks"] == pytest.approx(len(decode_marks()) - 1 / 20)   # the run's first only starts the clock
    assert got["graph.append_ms"] == pytest.approx(2.5)
    assert "eager.steps" not in got
    assert all(got[f"graph.{part}_ms"] > 0 for part in ("between", "commit", "pass", "model", "linear", "attention",
                                                        "kv_receipts", "lm_head"))
    assert got["graph.gather_ms"] == got["graph.prefill_decode_ms"] == got["graph.prefill_gemm_ms"] == 0


def test_graph_and_eager_steps_are_kept_apart():
    t = Timeline(2)
    before = t.read()
    t.step(prefill_marks(), 4096)
    t.append(EAGER)
    for _ in range(5):
        t.step(decode_marks(), 32)
        t.append(GRAPH)
    after = t.read()
    got = SS.per_step(before, after)
    assert (got["eager.steps"], got["graph.steps"]) == (1, 5)
    assert got["eager.tokens_per_step"] == 4096 and got["graph.tokens_per_step"] == 32
    assert got["eager.prefill_decode_ms"] > 0 and got["eager.gather_ms"] > 0 and got["graph.prefill_gemm_ms"] == 0
    assert got["eager.linear_ms"] == 0
    total = sum(after[kind]["ms"][part] for kind in SS.KINDS for part in SS.MAIN)
    assert total == pytest.approx(got["eager.step_ms"] + 5 * got["graph.step_ms"])


def test_the_accumulator_layout_and_the_reference_arithmetic():
    acc = [0] * SS.SLOTS
    SS.account(acc, P["model"], GRAPH, 100)    # starts the clock: nothing accumulated
    assert acc[0] == 100 and sum(acc[2:]) == 0
    SS.account(acc, P["linear"], -1, 175)
    SS.account(acc, P["between"], EAGER, 200)  # the kind is set before the interval is counted
    width = 2 * len(SS.PARTS)
    assert acc[2 + GRAPH * width + P["linear"]] == 75 and acc[2 + GRAPH * width + len(SS.PARTS) + P["linear"]] == 1
    assert acc[2 + EAGER * width + P["between"]] == 25 and acc[1] == EAGER
    with pytest.raises(ValueError):
        SS.decode(acc[:-1], acc, [0, 0], [0, 0])


@pytest.mark.skipif(not os.environ.get("VERITY_VLLM_BENCH_PAGED_LIB"), reason="needs the arms' paged library and a GPU")
def test_pg_span_on_the_gpu_telescopes_and_counts_its_captured_marks():
    """pg_span is `account` on the device: its parts sum to its stream's time between marks to the nanosecond, and
    marks captured into a CUDA graph count once a replay. Reads the global timer only through the marks."""
    import torch

    import paged

    spans = SS.Spans(paged.PagedLib("p1024", os.environ["VERITY_VLLM_BENCH_PAGED_LIB"]), torch)
    x = torch.randn(1 << 22, device="cuda")
    spans.mark(P["between"], GRAPH)
    torch.cuda.synchronize()
    first = int(spans.main[0].item())
    for _ in range(4):
        x.mul_(1.0001)
        spans.mark(P["model"])
        spans.mark(P["linear"])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x.mul_(1.0001)
        spans.mark(P["attention"])
        spans.mark(P["kv_receipts"])
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    raw = spans.main.tolist()
    base = 2 + GRAPH * 2 * len(SS.PARTS)
    assert sum(raw[base:base + len(SS.PARTS)]) == raw[0] - first
    got = spans.read()["graph"]
    assert (got["marks"]["model"], got["marks"]["linear"]) == (4, 4)
    assert (got["marks"]["attention"], got["marks"]["kv_receipts"]) == (3, 3)
    assert got["marks"]["between"] == 0 and got["ms"]["attention"] > 0
    assert int(spans.main[0].item()) == 0        # the read restarted the clock
