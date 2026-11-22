"""The cross-rank collective-output pass: picks are rank-agnostic and seeded; the definition is bit-exact per kind;
a received output that is not the definition over the committed partials fails BY NAME; a missing member is a named FAIL, never a pass."""
from __future__ import annotations

from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")


SITES = ["model.layers.0.self_attn.o_proj", "model.layers.0.mlp.down_proj", "model.embed_tokens"]


def _layouts(rank: int, steps: int = 6, prefill_rows: int = 40, hidden: int = 8):
    lay = {}
    for s in range(steps):
        rows = prefill_rows if s == 0 else 3
        ms = []
        for site in SITES:
            ms.append((f"{site}/part_rank{rank}", (rows, hidden)))
            ms.append((f"{site}/0", (rows, hidden)))
        ms.append((f"lm_head/shard_rank{rank}", (rows, hidden // 2)))
        ms.append(("lm_head/0", (rows, hidden)))
        lay[s] = ms
    return lay
















def _chain(parts, order):
    acc = parts[order[0]]
    for r in order[1:]:
        acc = (acc.float() + parts[r].float()).to(torch.bfloat16)
    return acc










def test_flip_site_alters_the_collective_input_but_not_the_committed_partial():
    """The TP-12 negative: TPPartialSource binds x (commits it) and then hands the collective a different tensor."""
    from verity_vllm.acquire.sources import partial_source as TPS

    class _C:  # a committer stub: records what was bound
        _cur = object()

        def __init__(self):
            self.bound = []
            self.errors = []

        def _acquire(self, member, x, job):
            self.bound.append((member, x.clone()))

    src = TPS.TPPartialSource.__new__(TPS.TPPartialSource)
    src.c, src.rank, src.world, src.label_rank, src.omit_site, src.reference = _C(), 1, 2, 1, None, False
    src.flip_site, src.flipped, src.stats = "layers.0.mlp.down_proj", 0, {"unattributed": 0, "launches": 0, "bytes": 0, "acquire_s": 0.0, "shapes": {}, "kinds": {}}
    src.flip_mode = "gross"
    src.cur = ("model.layers.0.mlp.down_proj", "RowParallelLinear")
    src.ref, src.ref_in, src._cur_in, src.ref_reexec_steps = {}, {}, None, 0
    seen = {}

    def orig(x):
        seen["x"] = x.clone()
        return x * 2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.ones(2, 4, dtype=torch.bfloat16, device=dev)
    if dev == "cpu":
        # _bind only binds CUDA tensors; on CPU we exercise the flip branch through the wrapper with a device-typed stub
        class _X(torch.Tensor):
            pass
        pytest.skip("flip branch needs a CUDA tensor (exercised on the pod by neg_flip_partial)")
    w = src._wrap("all_reduce", orig)
    w(x)
    assert src.flipped == 1
    assert torch.equal(src.c.bound[0][1], torch.ones(2, 4, dtype=torch.bfloat16, device=dev))      # committed = genuine
    exp0 = float(torch.ones((), dtype=torch.bfloat16, device=dev) * 2 + 1024.0)                   # bf16(1*2+1024) = 1024.0 (spacing 8 at 2^10)
    assert float(seen["x"].view(-1)[0]) == exp0 != 1.0 and float(seen["x"].view(-1)[1]) == 1.0      # the collective saw a different tensor
    # `:ulp` mode: element 0 moves by exactly one bf16 ulp (1.0 -> 1.0078125), nothing else; the committed partial stays genuine
    src.flip_mode, src.c.bound = "ulp", []
    w(torch.ones(2, 4, dtype=torch.bfloat16, device=dev))
    assert src.flipped == 2
    assert torch.equal(src.c.bound[0][1], torch.ones(2, 4, dtype=torch.bfloat16, device=dev))
    assert float(seen["x"].view(-1)[0]) == 1.0078125 and float(seen["x"].view(-1)[1]) == 1.0


def test_flip_partial_spec_parses_the_ulp_mode():
    from verity_vllm.acquire.sources import partial_source as TPS
    for spec, site, mode in (("layers.3.mlp.down_proj", "layers.3.mlp.down_proj", "gross"), ("layers.3.mlp.down_proj:ulp", "layers.3.mlp.down_proj", "ulp"), (None, None, "gross")):
        assert TPS.TPPartialSource.parse_flip(spec) == (site, mode)


