"""`engine.sampler_capture` on stand-ins for vLLM's V2 model runner: its sampler, input batch and two TP ranks, on CPU torch.

A target's rows are kept only at the steps that sample a token of it (its scheduled tokens reach its prompt's end), as the
bf16 bits the sampler took, and `pipeline.interior_rows.captured` reads them back as its token-select span. Ranks that took
other bits are refused, as are a sharded sampler and logits that are not one bf16 row a request. The wrap is a patch of
`verity_vllm.engine.hooks`, and `verity_vllm.engine.sampler_capture.uninstall` puts the runner's own sampler back."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from verity_vllm.engine import hooks
from verity_vllm.engine import sampler_capture as SC
from verity_vllm.pipeline import interior_rows as IR

V = 64
TSEL = "TokenSelect_v1{V=64}"
#: the replay's step trace: rows (request, start, n), and the tokens each step emitted
STEPS: list[dict[str, Any]] = [
    {"step": 0, "rows": [["a", 0, 4], ["b", 0, 2]], "emitted": {"b": 1}},
    {"step": 1, "rows": [["a", 4, 2], ["b", 2, 1]], "emitted": {"a": 1, "b": 1}},
    {"step": 2, "rows": [["a", 6, 1]], "emitted": {"a": 1}}]
PREFILL = {"a": 6, "b": 2}


class Sampler:
    """Greedy, as a token select is; ``vocab`` stands for the attributes a runner reads through the wrapper."""

    vocab = V

    def __call__(self, logits, input_batch):
        return SimpleNamespace(sampled_token_ids=logits.float().argmax(-1, keepdim=True).to(torch.int64))


def _batch(step: dict) -> SimpleNamespace:
    rows = step["rows"]
    return SimpleNamespace(req_ids=[r[0] for r in rows] + ["pad"], num_reqs=len(rows), num_draft_tokens=0,
                           num_computed_tokens_np=np.array([r[1] for r in rows]),
                           num_scheduled_tokens=np.array([r[2] for r in rows]),
                           prefill_len_np=np.array([PREFILL[r[0]] for r in rows]))


def _worker(rank: int) -> SimpleNamespace:
    return SimpleNamespace(rank=rank, model_runner=SimpleNamespace(sampler=Sampler(), batch_sharder=None))


@pytest.fixture(autouse=True)
def no_capture_stays_on():
    """A test's captures come off with it, so the process's live patches are what they were."""
    before = hooks.live()
    yield
    for p in reversed(hooks.live(SC.TAG)):
        p.uninstall()
    assert hooks.live() == before


def _serve(workers, flip_at: int | None = None):
    """Each step's logits through every worker's sampler; rank 1's flipped at step ``flip_at``. The logits rows, by step."""
    g = torch.Generator().manual_seed(7)
    sent = []
    for s in STEPS:
        logits = torch.randn(len(s["rows"]), V, generator=g).to(torch.bfloat16)
        sent.append(logits.clone())
        for w in workers:
            x = logits.clone()
            if w.rank == 1 and s["step"] == flip_at:
                x[0, 0] = -x[0, 0] + 1
            w.model_runner.sampler(x, _batch(s))
    return sent


def test_a_target_s_rows_are_its_sampled_steps_bits_and_read_back_as_its_token_select_span(tmp_path):
    workers = [_worker(0), _worker(1)]
    assert [SC.install(w, ["a"]) for w in workers] == [0, 1]
    assert workers[0].model_runner.sampler.vocab == V
    sent = _serve(workers)
    spans = SC.spans([SC.take(w, str(tmp_path / "ranks")) for w in workers], STEPS)
    want = torch.stack([sent[1][0], sent[2][0]]).view(torch.int16).numpy().view("<u2")
    assert set(spans) == {"a"} and spans["a"]["steps"] == [1, 2]
    assert np.array_equal(spans["a"]["logits"], want)
    ids = [int(sent[k][0].float().argmax()) for k in (1, 2)]
    assert spans["a"]["tokens"] == ids and spans["a"]["program"] == SC.launch("a", STEPS)
    SC.write(tmp_path / "capture", TSEL, spans)
    got = IR.captured(tmp_path / "capture", "a", ids)
    assert np.array_equal(got.data["rows"]["in0"], want) and [e[3] for e in got.entries] == [1, 2]


def test_ranks_that_took_other_bits_and_runners_a_token_select_cannot_read_are_refused(tmp_path):
    workers = [_worker(0), _worker(1)]
    for w in workers:
        SC.install(w, ["a"])
    _serve(workers, flip_at=2)
    with pytest.raises(ValueError, match="rank 1 took other logits or ids than rank 0"):
        SC.spans([SC.take(w, str(tmp_path / "ranks")) for w in workers], STEPS)
    sharded = _worker(0)
    sharded.model_runner.batch_sharder = object()
    with pytest.raises(TypeError, match="shards sampling across ranks"):
        SC.install(sharded, ["a"])
    w = _worker(0)
    SC.install(w, ["a"])
    with pytest.raises(TypeError, match="one bf16 row a token"):
        w.model_runner.sampler(torch.zeros(2, V), _batch(STEPS[1]))


def test_uninstall_puts_the_runner_s_own_sampler_back_and_the_rows_stay_with_the_capture(tmp_path):
    w, before = _worker(0), hooks.live()
    own = w.model_runner.sampler
    assert SC.install(w, ["a"]) == 0
    capture = w.model_runner.sampler
    (patch,) = hooks.live(SC.TAG)
    assert isinstance(capture, SC.Capture) and capture.inner is own and hooks.live() == (*before, patch)
    assert (patch.owner, patch.name, patch.value, patch.prior) == (w.model_runner, "sampler", capture, own)
    _serve([w])
    took = SC.take(w, str(tmp_path / "ranks"))["spans"]["a"]
    assert SC.uninstall(w) == 0
    assert w.model_runner.sampler is own and vars(w.model_runner)["sampler"] is own and hooks.live() == before
    assert (len(capture.rows["a"]), capture.ids["a"]) == (2, took["ids"])
    w.model_runner.sampler(torch.zeros(1, V).to(torch.bfloat16), _batch(STEPS[2]))
    assert len(capture.rows["a"]) == 2
    with pytest.raises(TypeError, match="the runner's sampler is a Sampler, not a capture `install` put there"):
        SC.uninstall(w)


def test_two_captures_on_one_runner_come_off_newest_first_and_another_s_patch_on_top_is_not_taken_for_one():
    w = _worker(0)
    own = w.model_runner.sampler
    SC.install(w, ["a"])
    SC.install(w, ["b"])
    assert w.model_runner.sampler.targets == {"b"} and w.model_runner.sampler.inner.targets == {"a"}
    SC.uninstall(w)
    assert w.model_runner.sampler.targets == {"a"} and w.model_runner.sampler.inner is own
    with hooks.patched(w.model_runner, "sampler", Sampler()), pytest.raises(TypeError, match="not a capture `install` put there"):
        SC.uninstall(w)
    SC.uninstall(w)
    assert w.model_runner.sampler is own and hooks.live(SC.TAG) == ()
