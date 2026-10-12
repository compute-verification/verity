"""`verity_vllm/observe/fold/fold.py`: served vs discarded samples under chunked prefill — the fold classifies each SampleEnd row by vLLM's
own rule `StepStart.seq_lens[r] >= prompt_len_r`; the toy log of test_fold_synthetic prefilled in two chunks is the reproducer
(b0_chunked_200p: 15 sampled, 12 served, 3 discarded at steps 0-2)."""
from __future__ import annotations

import dataclasses

from verity_vllm.observe.events import StepLink, StepStart
from verity_vllm.observe.fold.fold import fold, instance_nodes
from tests.observe import test_fold_synthetic as TFS
from verity.ml.circuits.types import Array, Value

memory = TFS.memory  # the fixture (real memory model when importable, else the in-test stub)


def _chunked_toy_log(seq_len_at_step0: int | None = None):
    """PROMPT [5, 7] prefilled in two chunks: step 0 = position 0 only (seq_len 1 < prompt_len 2 -> the sample, token 11, is
    dropped by the engine), step 1 = position 1 (prompt complete -> served, SAMPLED[0]), step 2 = decode of the served token
    (StepLink to step 1).  `seq_len_at_step0` overrides the step-0 `seq_lens` to show the rule is driven by that fact alone."""
    ids = TFS.Ids()
    evs = TFS.preamble()
    step0 = TFS.step_events(0, [0], ids, 20, 11)
    if seq_len_at_step0 is not None:
        step0 = [dataclasses.replace(e, seq_lens={TFS.REQ: seq_len_at_step0}) if isinstance(e, StepStart) else e for e in step0]
    evs += step0
    evs += [e for e in TFS.step_events(1, [1], ids, 40, TFS.SAMPLED[0]) if not isinstance(e, StepLink)]   # a prompt chunk: rows bind by position
    evs += TFS.step_events(2, [2], ids, 60, TFS.SAMPLED[1])
    return evs


def test_fold_discards_partial_prefill_samples(memory):
    r = fold(_chunked_toy_log(), TFS.toy_profile(), strict=True)
    assert not r.errors and r.report.unsupported == 0
    # served: steps 1 and 2 only; the step-0 sample is recorded as discarded with its logits position
    assert r.sampled == {TFS.REQ: [(1, TFS.SAMPLED[0]), (2, TFS.SAMPLED[1])]}
    assert r.discarded == {TFS.REQ: [(0, 11, 0)]}
    assert r.body.ret_type == Array(2, Value(32))
    ts = {m["step"]: m["node"] for m in instance_nodes(r, spec_prefix="TokenSelect_v1")}
    assert sorted(ts) == [0, 1, 2]                                   # the discarded sample's TokenSelect instance IS in the Program
    assert list(r.body.ret) == [("n", ts[1], 0), ("n", ts[2], 0)]    # ... but is not returned
    consumers = [k for k, nd in enumerate(r.body.nodes) if any(("n", ts[0], 0) in list(a.refs) for a in nd.args)]
    assert consumers == []                                           # ... and nothing reads it (no StepLink fed it back)
    emb2 = instance_nodes(r, spec_prefix="Embedding_v1", step=2)[0]["node"]
    assert list(r.body.nodes[emb2].args[0].refs) == [("n", ts[1], 0)]  # the decode row carries the SERVED token
    assert any("discarded by the engine" in n for n in r.notes)


def test_fold_served_rule_is_seq_lens_not_positions(memory):
    # negative: the same events with StepStart.seq_lens claiming the prompt complete after step 0 -> the step-0 sample is SERVED
    # (seq_len 2 >= prompt 2 + 0 served), although position 0 alone was fed: the decision is seq_lens', not the positions'.
    # [R9 ref 2026-09-18] the rule is vLLM's full one, `seq_len >= prompt_len + served so far` (`num_tokens` counts the output
    # tokens; identical to `seq_len >= prompt_len` on every run without preemption): step 1 then claims seq_len 2 < 2 + 1 and
    # its sample is discarded (the engine would drop a chunk that ends inside already-generated tokens); step 2's is served.
    r = fold(_chunked_toy_log(seq_len_at_step0=2), TFS.toy_profile(), strict=True)
    assert r.discarded == {TFS.REQ: [(1, TFS.SAMPLED[0], 1)]} and [s for s, _ in r.sampled[TFS.REQ]] == [0, 2]
    assert r.body.ret_type == Array(2, Value(32))
    # and the single-chunk toy log (prefill completes at step 0) is unchanged by the rule
    r0 = fold(TFS.toy_log(), TFS.toy_profile(), strict=True)
    assert r0.discarded == {} and r0.sampled == {TFS.REQ: [(0, 9), (1, 3)]} and r0.body.ret_type == Array(2, Value(32))
