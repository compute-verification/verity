"""verity_vllm.program.registry.quarantine.collective — lane tp2 (window 2026-09-17): provisional definitions for the cross-rank
collectives of tensor-parallel serving (Qwen2.5-1.5B, `tensor_parallel_size=2`, vLLM d9105ea80, PyNccl path, custom
all-reduce disabled).

One definition here: `allreduce.AllReduceSumBf16{N, R}` — the NCCL `ncclAllReduce(…, ncclBfloat16, ncclSum)` body over the
R ranks' operands, written as a chain of one-rounding bf16 additions (`Bf16ToF32`, `F32Add`, `F2fpBf16`).  Only R = 2 is defined:
there is exactly one addition per element, so the order question does not arise and the body coincides with
`registry.b1_tp2.AllReduce2` (`Bf16Add`) on every non-NaN word.  R > 2 is refused; its hop order is `b1_tp2.allreduce_order` and
its all-reduce `b1_tp2.all_reduce(R, N)` (see the module docstring for what is known and what is not).

The all-gather needs no definition (a reference rearrangement: `b1_tp2.AllGather2` is a `reshape`), and the rank-local masked
lookup `_C.vocab_parallel_embedding` is `b1_tp2.EmbeddingShard`.

Fold-side use is NOT wired in this window: `fold.py` / `patterns.py` are frozen (WINDOW rule 7); `verity_vllm/tp/collective_link.py`
produces the per-launch linkage record (which per-rank launch produced each operand, which launches consume the result) that a future
pattern would turn into an instance of this definition whose output is a journaled write on every rank.
"""
from __future__ import annotations

from verity_vllm.program.registry.quarantine.collective import allreduce  # noqa: F401

ALL_DEFINITIONS = [allreduce.AllReduceSumBf16Elem, allreduce.AllReduceSumBf16]
ALL_PRIMITIVES: list = []
