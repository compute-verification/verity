"""`served_record.SamplerInputsSource` on a `NativeHostCommitter`: each sampled step binds the seed and the position each
row's sample read (Model Runner V2's ``seeds[idx_mapping]`` and ``positions[logits_indices]``) and keeps the step's rows,
widens the arena's step bound by them, and refuses by name a step whose input batch it can't read or whose rows don't
match the samples."""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from verity_vllm.commit.committer.native_host import NativeHostCommitter
from verity_vllm.commit.served_record import SAMPLER_INPUTS, SamplerInputsSource, served_rows


def committer(seeds):
    states = types.SimpleNamespace(seeds=types.SimpleNamespace(gpu=torch.tensor(seeds, dtype=torch.int64)))
    runner = types.SimpleNamespace(sampler=types.SimpleNamespace(sampling_states=states))
    com = NativeHostCommitter(types.SimpleNamespace(model=torch.nn.Module(), runner=runner), run_id="sampler-inputs", arena_bytes=None,
                              geo_digest=b"g" * 32)
    src = SamplerInputsSource(com)
    com.attach_source(src)
    got = {}
    com._acquire = lambda name, t, job: got.__setitem__(name, t.tolist())
    return com, src, got


def batch():
    """Two requests: one decoding at position 9 (state slot 2), one a 3-token prefill chunk from position 4 (slot 0)."""
    return types.SimpleNamespace(num_reqs=2, req_ids=["b-0123abcd", "a-0123abcd"], idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
                                 positions=torch.tensor([9, 4, 5, 6, 0]), logits_indices=torch.tensor([0, 3]),
                                 query_start_loc_np=np.array([0, 1, 4]), num_computed_tokens_np=np.array([9, 4]),
                                 prefill_len_np=np.array([5, 12]))


def test_binds_the_draws_seed_and_position_and_keeps_the_rows():
    com, src, got = committer([11, 0, 33])
    src.on_sample(batch(), 2, types.SimpleNamespace(step=3))
    assert got == {SAMPLER_INPUTS[0]: [33, 11], SAMPLER_INPUTS[1]: [9, 6]}
    assert served_rows(com) == {3: [("b-0123abcd", 0, 1, 9, 5), ("a-0123abcd", 1, 4, 4, 12)]}
    assert not com.errors and com.name.endswith("+sampler_inputs")
    src.on_sample(batch(), 2, types.SimpleNamespace(step=0))          # a new window's first step
    assert list(served_rows(com)) == [0]


def test_the_inputs_widen_the_step_bound_and_stay_out_of_module_names():
    _com, src, _got = committer([1])
    assert src.extra_step_bytes({}, None) == {"per_seq_bytes": 16, "tensors_per_step": 2}
    assert src.own_names == set(SAMPLER_INPUTS)


@pytest.mark.parametrize("ib, n, problem", [(None, 2, "no input batch"), (batch(), 3, "samples 3 rows for 2 requests")])
def test_refuses_a_step_it_cannot_bind(ib, n, problem):
    com, src, got = committer([1, 2, 3])
    src.on_sample(ib, n, types.SimpleNamespace(step=0))
    assert not got and not served_rows(com)
    assert any(problem in e for e in com.errors), com.errors
