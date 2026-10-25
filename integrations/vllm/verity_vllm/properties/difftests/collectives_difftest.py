"""Differential-test adapter (`verity_vllm.properties.admission`) for the two-rank collectives a TP2 Program binds, `AllReduce2_v1{N}`
and `AllGather2_v1{N}` (`registry.b1_tp2`), against NCCL (`ncclAllReduce(ncclSum, ncclBfloat16)` / `ncclAllGather`) between two GPUs
of one host, through `torch.cuda.nccl` (one process driving both devices: the same NCCL collectives vLLM's PyNccl
communicator calls, with no rank processes to spawn):

    # pod with two GPUs, torch venv (the transport is the caller's: NCCL_P2P_DISABLE=1 for hosts without NVLink):
    verity-vllm properties-admission produce --adapter verity_vllm.properties.difftests.collectives_difftest --n 30 --seed 5 --out DIR
    verity-vllm properties-admission check   --adapter verity_vllm.properties.difftests.collectives_difftest --dir DIR --out RUN/tp2.json

Rank r contributes operand r; every rank must receive the same words (asserted), and rank 0's are the production output."""
from __future__ import annotations

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry.b1_tp2 import AllGather2, AllReduce2
from verity.ml.definitions.registry.prims import BF16

SEMANTIC_ID = "tp2/Collectives2@v1"
KINDS = ("random", "zeros", "max", "subnormal", "alternating")
SHAPES = (2048, 576, 4096, 1536, 64)
OPS = ("all_reduce", "all_gather")


def cases(rng, i: int) -> dict:
    op = OPS[i % len(OPS)]
    N = SHAPES[(i // 2) % len(SHAPES)]
    k0 = KINDS[(i // 2) % len(KINDS)]
    k1 = k0 if i % 4 < 2 else "random"
    return {"statics": {"op": op, "N": N},
            "inputs": [{"shape": [N], "dtype": "bfloat16", "kind": k0}, {"shape": [N], "dtype": "bfloat16", "kind": k1}],
            "edge": k0 != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    import torch.cuda.nccl as nccl

    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"the two-rank collectives need 2 CUDA devices, found {torch.cuda.device_count()}")
    # through the host: a device-to-device copy goes over P2P, which returns zeros on hosts whose P2P is broken (the reason for
    # NCCL_P2P_DISABLE; 2x RTX PRO 6000 across NUMA nodes)
    parts = [x.detach().cpu().to(f"cuda:{r}") for r, x in enumerate(inputs)]
    for r in range(2):
        torch.cuda.synchronize(r)
    if statics["op"] == "all_reduce":
        nccl.all_reduce(parts)
        got = parts
    else:
        got = [torch.empty(2 * p.numel(), dtype=p.dtype, device=p.device) for p in parts]
        nccl.all_gather(parts, got)
    for r in range(2):
        torch.cuda.synchronize(r)
    if not torch.equal(got[1].cpu().view(torch.int16), got[0].cpu().view(torch.int16)):
        raise AssertionError(f"{statics['op']}: rank 1 received other words than rank 0")
    statics["nccl"] = ".".join(map(str, nccl.version()))
    return [got[0].to(inputs[0].device)]


def _difftest_body(B, S, a, b):
    """The collective of rank 0's operand `a` and rank 1's `b`, as the TP2 Program binds it: `parts = [a, b]`."""
    return B.call(bind(AllReduce2 if S.OP == "all_reduce" else AllGather2, N=S.N), array_of([a, b]))


Collectives2Difftest = CompositeDefinition(
    "Collectives2Difftest", 1, ("OP", "N"),
    lambda S: ((("a", Array(S.N, BF16)), ("b", Array(S.N, BF16))), Array(S.N if S.OP == "all_reduce" else 2 * S.N, BF16)),
    _difftest_body, doc=_difftest_body.__doc__ or "", conformance="derived: difftest", register=False)


def spec(statics: dict):
    return bind(Collectives2Difftest, OP=str(statics["op"]), N=int(statics["N"]))
