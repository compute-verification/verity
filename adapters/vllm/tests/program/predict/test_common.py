from __future__ import annotations

import gc
import re
from types import SimpleNamespace

import pytest
from verity.circuits.refs import Affine, Coll, runs
from verity.circuits.types import Array, Value

from tests.program.predict.test_predict import TINY, _phi
from verity_vllm.program.predict.build import build_program
from verity_vllm.program.predict.common import (
    Unexpressible,
    cyclic_gc_off,
    drive,
    launch_rows,
    request_passes,
)

REFUSED = "vocabulary cannot express: "
EDGE = " with its rows a window's edge (a request's, on the sparse-attention decoder)"
INSIDE = " (one that stops inside its prompt samples nothing: only a request a window leaves there has a Program, its rows the edge's)"
CHUNKS = " (a request's chunks are its {} prefill rows in turn, each at least one)"
# (wrapper, LP, T, M, C, chunks, dense, edge) -> (resumed, its prefill chunks, each pass's rows, whether each samples)
PASSES = [
    (("step", None, None, 3, 0, None, True, False), (False, (), [3], [True])),
    (("step", None, None, 3, 0, None, False, False), (False, (), [3], [True])),
    (("request", 4, 2, 1, 0, None, True, False), (False, (), [4, 1, 1], [True, True, True])),
    (("request", 4, 0, 1, 0, None, True, False), (False, (), [4], [True])),
    # a history of C rows: the prefill is the prompt's other rows, down to one
    (("request", 4, 2, 1, 1, None, True, False), (False, (), [3, 1, 1], [True, True, True])),
    (("request", 4, 2, 1, 3, None, True, False), (False, (), [1, 1, 1], [True, True, True])),
    # a prefill in several launches: a pass a chunk, the last alone sampling; one chunk is no chunking
    (("request", 4, 2, 1, 0, (1, 3), True, False), (False, (1, 3), [1, 3, 1, 1], [False, True, True, True])),
    (("request", 4, 2, 1, 1, (2, 1), False, False), (False, (2, 1), [2, 1, 1, 1], [False, True, True, True])),
    (("request", 4, 2, 1, 0, (4,), True, False), (False, (), [4, 1, 1], [True, True, True])),
    # off the dense decoder, a history reaching the prompt's end or past it resumes at decode C
    (("request", 4, 2, 1, 4, None, False, False), (True, (), [1, 1], [True, True])),
    (("request", 4, 3, 1, 5, None, False, False), (True, (), [1, 1], [True, True])),
    (("request", 4, 3, 1, 6, None, False, False), (True, (), [1], [True])),
    # a request a window leaves: its rows the edge's, and one that stops inside its prompt samples nothing
    (("request", 4, 2, 1, 0, None, False, True), (False, (), [4, 1, 1], [True, True, True])),
    (("request", 4, 2, 1, 5, None, False, True), (True, (), [1], [True])),
    (("request", 6, -2, 1, 0, None, False, True), (False, (), [4], [False])),
    (("request", 6, -2, 1, 2, None, False, True), (False, (), [2], [False])),
    (("request", 3, -2, 1, 0, None, False, True), (False, (), [1], [False])),
    (("request", 6, -2, 1, 0, (1, 3), False, True), (False, (1, 3), [1, 3], [False, False])),
]
NO_PASSES = [
    (("request", 4, 2, 1, 4, None, True, False), "a history of 4 rows for a prompt of 4 (0 < C < LP)"),
    (("request", 4, 2, 1, -1, None, True, False), "a history of -1 rows for a prompt of 4 (0 < C < LP)"),
    (("request", 4, 2, 1, 6, None, False, False), "a history of 6 rows for a prompt of 4 and 2 decodes (0 < C < LP + T)"),
    (("request", 6, -2, 1, 4, None, False, True), "a history of 4 rows for a prompt of 6 and -2 decodes (0 < C < LP + T)"),
    (("step", None, None, 1, 2, None, False, False), "a history of 2 rows for a prompt of None and None decodes (0 < C < LP + T)"),
    (("step", None, None, 1, 0, None, False, True), "a step to position 0 of a None-token prompt" + EDGE),
    (("request", 4, 2, 1, 0, None, True, True), "a request to position 6 of a 4-token prompt" + EDGE),
    (("request", 6, -6, 1, 0, None, False, True), "a request to position 0 of a 6-token prompt" + EDGE),
    (("request", 6, -2, 1, 0, None, False, False), "a request to position 4 of a 6-token prompt" + INSIDE),
    (("request", 6, -2, 1, 0, None, True, False), "a request to position 4 of a 6-token prompt" + INSIDE),
    (("step", None, None, 2, 0, (2,), False, False), "prefill chunks [2] of a step" + CHUNKS.format(0)),
    (("request", 4, 3, 1, 5, (1, 1), False, False), "prefill chunks [1, 1] of a request" + CHUNKS.format(0)),
    (("request", 0, 2, 1, 0, (1,), False, False), "prefill chunks [1] of a request" + CHUNKS.format(0)),
    (("request", 4, 1, 1, 0, (0, 4), False, False), "prefill chunks [0, 4] of a request" + CHUNKS.format(4)),
    (("request", 4, 1, 1, 0, (1, 2), False, False), "prefill chunks [1, 2] of a request" + CHUNKS.format(4)),
    (("request", 4, 1, 1, 1, (1, 3), False, False), "prefill chunks [1, 3] of a request" + CHUNKS.format(3)),
    (("request", 6, -2, 1, 0, (1, 5), False, True), "prefill chunks [1, 5] of a request" + CHUNKS.format(4)),
]


def test_cyclic_gc_off_holds_the_collector_off_inside_and_restores_its_state():
    @cyclic_gc_off
    def inside(fail: bool = False) -> bool:
        if fail:
            raise ValueError("boom")
        return gc.isenabled()

    was = gc.isenabled()
    try:
        gc.enable()
        assert inside() is False and gc.isenabled()
        with pytest.raises(ValueError):
            inside(fail=True)
        assert gc.isenabled()
        gc.disable()
        assert inside() is False and not gc.isenabled()
    finally:
        (gc.enable if was else gc.disable)()


@pytest.mark.parametrize(("shape", "passes"), PASSES, ids=[str(shape) for shape, _ in PASSES])
def test_a_wrappers_passes_are_its_prefill_in_its_chunks_and_a_decode_each(shape, passes):
    """`request_passes`: a step is one pass of its M rows; a request is its prefill of the prompt's rows after its history, a pass
    a chunk, and then a pass a decode, or only the decodes left where its history reaches them; every pass samples but a
    chunk before the prefill's last, and a request that stops inside its prompt samples nothing."""
    assert request_passes(*shape) == passes


@pytest.mark.parametrize(("shape", "why"), NO_PASSES, ids=[str(shape) for shape, _ in NO_PASSES])
def test_a_wrapper_with_no_passes_is_refused_by_name(shape, why):
    """A history that is not inside the prompt (the dense decoder) or the request, an edge on a step or on the dense
    decoder, a request that stops inside its prompt with no edge or at its first row, and chunks that are not a request's
    prefill rows in turn."""
    with pytest.raises(Unexpressible, match="^" + re.escape(REFUSED + why) + "$"):
        request_passes(*shape)


class _Forwards:
    """A decoder that keeps what `drive` hands it: each forward's rows and first position and the launch step it runs at, and
    each sample's hidden row, index and position.  A forward's hidden rows are its positions, and sample k is row k of an
    Input of its own."""

    def __init__(self) -> None:
        self.forwards: list[tuple[list[Coll], int]] = []
        self.samples: list[tuple[int, int, int]] = []
        self.launch = SimpleNamespace(step=None)
        self.steps: list[int | None] = []

    def forward(self, rows: list[Coll], pos: int) -> list[int]:
        self.forwards.append((list(rows), pos))
        self.steps.append(self.launch.step)
        return list(range(pos, pos + len(rows)))

    def sample(self, hid: int, k: int, pos: int, policy: None) -> Coll:
        self.samples.append((hid, k, pos))
        return Coll(Array(1, Value(32)), Affine("p", 1, k, 1))


@pytest.mark.parametrize(("shape", "passes"), PASSES, ids=[str(shape) for shape, _ in PASSES])
def test_drive_runs_a_wrappers_passes_and_feeds_each_decode_the_sample_before_it(shape, passes):
    """`drive` over a decoder runs `request_passes`' passes in order, each at the position after the one before: a prefill
    pass reads the prompt's rows there, a decode one row, the sample before it as a u32 (a resumed request's first, the token
    it is handed).  A pass that samples does at its last row, and the samples in order are what `drive` returns.  In a batch,
    a request's k-th pass runs at the launch's step k; a step, and a request with no batch, is its own launch."""
    wrapper, LP, T, M, C, _, _, _ = shape
    resumed, chunks, sizes, sampling = passes
    pos0 = 5
    prompt = [Coll(Value(32), Affine("p", 0, i, 1)) for i in range(M if wrapper == "step" else LP)]
    last = Coll(Array(1, Value(32)), Affine("p", 2, 0, 1))
    model = _Forwards()
    toks = drive(model, wrapper, prompt, M=M, pos0=pos0, LP=LP, T=T, policy=None, batch=None, C=C, chunks=chunks, last=last)
    at = [(pos0 if wrapper == "step" else C) + sum(sizes[:k]) for k in range(len(sizes))]
    assert [(len(rows), pos) for rows, pos in model.forwards] == list(zip(sizes, at))
    assert model.samples == [(a + n - 1, k, a + n - 1) for k, (a, n) in enumerate((a, n) for a, n, s in zip(at, sizes, sampling) if s)]
    assert len(toks) == len(model.samples) and [runs(t.refs) for t in toks] == [runs(Affine("p", 1, k, 1)) for k in range(len(toks))]
    assert model.steps == [None] * len(sizes)
    batched = _Forwards()
    drive(batched, wrapper, prompt, M=M, pos0=pos0, LP=LP, T=T, policy=None, batch=[], C=C, chunks=chunks, last=last)
    assert batched.steps == ([None] if wrapper == "step" else list(range(len(sizes)))) and batched.samples == model.samples
    decodes = 0 if wrapper == "step" else len(sizes) if resumed else max(T, 0)
    fed = ([last] if resumed else []) + toks
    for k, (rows, pos) in enumerate(model.forwards):
        if k < len(sizes) - decodes:
            assert all(a is b for a, b in zip(rows, prompt[pos - (pos0 if wrapper == "step" else 0):])), "a prefill pass reads the prompt's rows"
        else:
            (row,), before = rows, fed[k - (len(sizes) - decodes)]
            assert row.type == Value(32) and runs(row.refs) == runs(before.refs)


def test_a_step_runs_under_no_launch_context():
    """A request runs under its launch's context (`launch_rows`), read from the workload's requests at its (LP, T): here a
    prompt of 4 and 2 decodes, whose second launch holds the other request's 6-row prefill.  A step states neither and reads
    none: its Program is the same with the workload's requests and without them."""
    requests = [{"request_id": f"u{i}", "index": i, "prompt_len": lp, "max_tokens": out, "arrive_step": i} for i, (lp, out) in enumerate([(4, 3), (6, 2)])]
    assert launch_rows(_phi(TINY, requests), 4, 2) == {"max_seqlen_q_by_step": {0: 4, 1: 6, 2: 1}, "conflict": False, "declared": True}
    assert launch_rows(_phi(TINY), 4, 2) is None
    assert build_program(_phi(TINY, requests), "step", M=2, pos0=3).digest == build_program(_phi(TINY), "step", M=2, pos0=3).digest
