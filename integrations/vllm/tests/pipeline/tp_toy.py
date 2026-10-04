"""A stand-in for one rank of a W-rank vLLM model, built on the construction's own `CollectiveBus`: a vocab-parallel embedding
(`verity_tp::embedding_shard` + all-reduce), a column- then row-parallel MLP (all-reduce) and column-parallel logits (all-gather).
Its rank enters the graph where a vLLM rank's does (the own partial's index in each collective, the shard's START/END), so a test can
derive any rank of it on CPU without vLLM."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from verity_vllm.program.frontend import vllm_meta as vm
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.export_ops import CollectiveBus, ensure_ops
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset

BF = torch.bfloat16
V, H, FF, TOKENS = 32, 8, 16, 3


def _param(*shape) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=BF, device="meta"), requires_grad=False)


class ToyRank(nn.Module):
    DERIVE_SOURCE = "verity-vllm/test-tp-toy-rank/v1"

    def __init__(self, bus: CollectiveBus) -> None:
        super().__init__()
        W, r = bus.world, bus.rank
        self.bus, self.VS = bus, V // W
        self.start, self.end = r * self.VS, (r + 1) * self.VS
        self.embed = _param(self.VS, H)
        self.up = _param(FF // W, H)
        self.down = _param(H, FF // W)
        self.lm_head = _param(V // W, H)

    def forward(self, prompt):
        x = torch.ops.verity_tp.embedding_shard(prompt.reshape(-1), self.embed, self.start, self.end)
        x = self.bus.all_reduce(x.view(*prompt.shape, H))
        h = self.bus.all_reduce(F.linear(F.linear(x, self.up), self.down))
        return self.bus.all_gather(F.linear(h, self.lm_head))


def embedding_shards(world: int, rank: int) -> list[dict]:
    """The `embedding_shards` record of the toy's rank (the shape `export_ops.bind_vocab_parallel_embeddings` writes)."""
    VS = V // world
    return [{"module": "inner.embed", "START": rank * VS, "END": (rank + 1) * VS, "VS": VS, "H": H, "tp_rank": rank, "tp_size": world}]


def rank_source(world: int, rank: int):
    """(the rank's traced wrapper, its example inputs, the probed collectives, the toy model)."""
    ensure_ops()
    bus = CollectiveBus(rank=rank, world=world)
    inner = ToyRank(bus)
    prompt = torch.empty(TOKENS, dtype=torch.int32, device="meta")
    bus.reset("probe")
    with torch.no_grad(), torch.device("meta"):
        torch.export.export(inner, (prompt,), strict=False)
    record = list(bus.record)
    mod, inputs = vm.with_peers(inner, (prompt,), bus)
    return mod, inputs, record, inner


def derive_rank(world: int, rank: int, session: list | None = None):
    """Rank `rank`'s derive report, as `build._construct` derives it (profile mode, the vLLM rules, the rank coordinate)."""
    mod, inputs, record, inner = rank_source(world, rank)
    rep = derive_report(mod, inputs, mode="profile", profile="b1-eager-v3", rules=vllm_ruleset(), name="vLLMStep[ToyRank]",
                        check_registry=True, rank=(world, rank), session=session)
    assert rep.ok, rep.unsupported
    return rep, mod, record, inner
