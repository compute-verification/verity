"""[R13 lane tp, P4 step 2] Tensor-parallel construction scaffolding for Bet A: the collectives of ONE rank's forward as explicit
Verity ops over (this rank's partial, the peer rank's partial), so that `derive_step --tp 2 --rank r` exports a TP-aware Program.

Established first (R13 tp step 1, `out/gen/sweep/evidence/tp/tp2_semantics/RESULT.md`): under the admitted BI profile at TP2 the
row-parallel linears compute two K/2-sharded `Gemm_v1` partials whose bf16 outputs are summed by ONE bf16 addition (the pynccl
all-reduce, `bf16(f32(a)+f32(b))`, order-free for two ranks) — a different exact computation from TP1's single GEMM; column-parallel
GEMMs, the vocab-parallel embedding (masked shard lookup + the same sum) and the lm_head all-gather (a concatenation) are
bit-identical slices of the TP1 definitions.  The registered Definitions for these are `verity_vllm.registry.b1_tp2.{AllReduce2,
AllGather2, EmbeddingShard}` (exact-model-tested on B1 TP2 by the GEN tp2 lane; re-confirmed bit-exactly on Llama by step 1).

Mechanism (no vLLM source is modified; generic — nothing here names a model):
  * `TPStub` replaces vLLM's `_TP` GroupCoordinator for the meta instantiation + export of rank r at world size TP: the parallel
    layers size their shards from `world_size` / `rank_in_group` exactly as the engine's workers do; the collectives route to
    `CollectiveBus`.
  * `CollectiveBus` — `probe` mode: one meta forward records every collective (kind, shape, dtype, call order); `trace` mode: the k-th
    collective consumes the k-th PEER INPUT of the wrapper (the other rank's partial, an ordinary Input of the Program) and emits
    `verity_tp::all_reduce2(part_rank0, part_rank1)` / `verity_tp::all_gather2(part_rank0, part_rank1)` in RANK ORDER.  A received
    collective output is therefore never a fresh trusted input: it is `AllReduce2{N}` of this rank's committed partial and the peer's
    committed partial; `tp_links.json` (written by derive_step) binds peer input k of rank r to collective k of rank 1-r, whose
    partial is a committed value of THAT rank's Program.
  * `VocabParallelEmbedding.forward` at tp_size > 1 is bound per instance to `verity_tp::embedding_shard(ids, shard, START, END)`
    (= what vLLM's fused `_C.vocab_parallel_embedding` computes: `shard[id-START]` if START <= id < END else 0) followed by the
    all-reduce — the fused CUDA op has no meta kernel and no Definition of its own; padding / added-vocab shards are refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

_lib_ready = False


def ensure_ops() -> None:
    """Register the `verity_tp` custom ops once (real impls for the CPU/eager check, fake impls for export on meta)."""
    global _lib_ready
    if _lib_ready:
        return

    @torch.library.custom_op("verity_tp::all_reduce2", mutates_args=())
    def all_reduce2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b  # bf16 add: f32 add, one RNE rounding == ncclSum bf16 of two terms

    @all_reduce2.register_fake
    def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)

    @torch.library.custom_op("verity_tp::all_gather2", mutates_args=())
    def all_gather2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=-1)

    @all_gather2.register_fake
    def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty(tuple(a.shape[:-1]) + (a.shape[-1] + b.shape[-1],), dtype=a.dtype, device=a.device)

    @torch.library.custom_op("verity_tp::embedding_shard", mutates_args=())
    def embedding_shard(ids: torch.Tensor, table: torch.Tensor, start: int, end: int) -> torch.Tensor:
        own = (ids >= start) & (ids < end)
        local = torch.where(own, ids - start, torch.zeros_like(ids))
        rows = table[local.long()]
        return torch.where(own.unsqueeze(-1), rows, torch.zeros_like(rows))

    @embedding_shard.register_fake
    def _(ids: torch.Tensor, table: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return torch.empty(tuple(ids.shape) + (table.shape[-1],), dtype=table.dtype, device=table.device)

    _lib_ready = True


@dataclass
class CollectiveBus:
    """The collectives of one rank's forward, in call order.  `mode`: `probe` (record; no peers) or `trace` (consume peers, emit ops)."""
    rank: int
    world: int
    mode: str = "probe"
    peers: list[torch.Tensor] = field(default_factory=list)
    record: list[dict] = field(default_factory=list)
    cursor: int = 0

    def _peer(self, kind: str, x: torch.Tensor, out_shape: tuple) -> torch.Tensor | None:
        k = self.cursor
        self.cursor += 1
        if self.mode == "probe":
            self.record.append({"k": k, "kind": kind, "shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", ""), "out_shape": list(out_shape)})
            return None
        if k >= len(self.peers):
            raise RuntimeError(f"collective #{k} ({kind}, {tuple(x.shape)}) has no peer input: the probe recorded {len(self.peers)} collectives")
        p = self.peers[k]
        if tuple(p.shape) != tuple(x.shape) or p.dtype != x.dtype:
            raise RuntimeError(f"collective #{k} ({kind}): peer {tuple(p.shape)} {p.dtype} != partial {tuple(x.shape)} {x.dtype}")
        return p

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.world != 2:
            raise NotImplementedError(f"AllReduce2 binds world size 2; world {self.world} (NCCL tree order becomes part of the Definition)")
        p = self._peer("all_reduce", x, tuple(x.shape))
        if p is None:
            return torch.empty_like(x)
        ensure_ops()
        a, b = (x, p) if self.rank == 0 else (p, x)
        return torch.ops.verity_tp.all_reduce2(a, b)

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world != 2:
            raise NotImplementedError("AllGather2 binds world size 2")
        if dim not in (-1, x.ndim - 1):
            raise NotImplementedError(f"all_gather along dim {dim} (AllGather2 concatenates the trailing axis)")
        out_shape = tuple(x.shape[:-1]) + (x.shape[-1] * self.world,)
        p = self._peer("all_gather", x, out_shape)
        if p is None:
            return torch.empty(out_shape, dtype=x.dtype, device=x.device)
        ensure_ops()
        a, b = (x, p) if self.rank == 0 else (p, x)
        return torch.ops.verity_tp.all_gather2(a, b)

    def example_peers(self, device: str = "meta") -> tuple[torch.Tensor, ...]:
        return tuple(torch.empty(tuple(r["shape"]), dtype=getattr(torch, r["dtype"]), device=device) for r in self.record)

    def reset(self, mode: str, peers: list[torch.Tensor] | None = None) -> None:
        self.mode, self.cursor = mode, 0
        self.peers = list(peers or [])
        if mode == "probe":
            self.record = []


class TPStub:
    """What the model classes and the collective helpers read from vLLM's TP `GroupCoordinator`, for rank `rank` of `world`."""

    def __init__(self, rank: int, world: int, bus: CollectiveBus) -> None:
        self.rank_in_group, self.world_size, self.rank, self.local_rank = int(rank), int(world), int(rank), int(rank)
        self.ranks = list(range(world))
        self.unique_name = "tp:0"
        self.use_custom_op_call = False
        self.device_communicator = None
        self.cpu_group = self.device_group = None
        self.bus = bus

    @property
    def is_first_rank(self) -> bool:
        return self.rank_in_group == 0

    @property
    def is_last_rank(self) -> bool:
        return self.rank_in_group == self.world_size - 1

    @property
    def first_rank(self) -> int:
        return 0

    @property
    def last_rank(self) -> int:
        return self.world_size - 1

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return self.bus.all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return self.bus.all_gather(input_, dim)

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        raise NotImplementedError("broadcast is not a bound TP collective")

    def reduce_scatter(self, *a: Any, **k: Any) -> torch.Tensor:
        raise NotImplementedError("reduce_scatter is not a bound TP collective")

    def barrier(self) -> None:
        return None


class tp_group_override:
    """Context: vLLM's `parallel_state._TP` := `TPStub(rank, world, bus)` (the world-1 groups stay initialised for PP/world lookups)."""

    def __init__(self, rank: int, world: int, bus: CollectiveBus) -> None:
        self.stub = TPStub(rank, world, bus)

    def __enter__(self) -> TPStub:
        from vllm.distributed import parallel_state as ps
        self._saved = ps._TP
        ps._TP = self.stub
        return self.stub

    def __exit__(self, *exc: Any) -> None:
        from vllm.distributed import parallel_state as ps
        ps._TP = self._saved


def bind_vocab_parallel_embeddings(model: torch.nn.Module) -> list[dict]:
    """Bind every `VocabParallelEmbedding` (exact class; `ParallelLMHead` is applied through the LogitsProcessor, not called) with
    tp_size > 1 to the `embedding_shard` + all-reduce forward.  Refuses padded / added-vocab shards (not what `EmbeddingShard` states)."""
    import types

    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    ensure_ops()
    bound = []
    for name, m in model.named_modules():
        if type(m) is not VocabParallelEmbedding or getattr(m, "tp_size", 1) <= 1:
            continue
        si = m.shard_indices
        if si.num_org_vocab_padding != 0 or si.added_vocab_end_index != si.added_vocab_start_index:
            raise ValueError(f"{name}: padded ({si.num_org_vocab_padding}) or added-vocab shard; EmbeddingShard{{VS,H,START}} binds plain contiguous shards only")
        start, end, H = int(si.org_vocab_start_index), int(si.org_vocab_end_index), int(m.embedding_dim)

        def fwd(self, input_, _start=start, _end=end, _H=H):
            part = torch.ops.verity_tp.embedding_shard(input_.reshape(-1), self.weight, _start, _end)
            return tensor_model_parallel_all_reduce(part.view(*input_.shape, _H))

        m.forward = types.MethodType(fwd, m)
        bound.append({"module": name, "START": start, "END": end, "VS": end - start, "H": H, "tp_rank": int(m.tp_rank), "tp_size": int(m.tp_size)})
    return bound


__all__ = ["ensure_ops", "CollectiveBus", "TPStub", "tp_group_override", "bind_vocab_parallel_embeddings"]
