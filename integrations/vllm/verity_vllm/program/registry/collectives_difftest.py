"""Differential-test adapter (`verity_vllm.properties.admission`) for the two-rank collectives a TP2 Program binds, `AllReduce2_v1{N}`
and `AllGather2_v1{N}` (`registry.b1_tp2`), against NCCL (`ncclAllReduce(ncclSum, ncclBfloat16)` / `ncclAllGather`) between two GPUs
of one host, one process per rank:

    # pod with two GPUs, torch venv (the transport is the caller's: NCCL_P2P_DISABLE=1 for hosts without NVLink):
    verity-vllm properties-admission produce --adapter verity_vllm.program.registry.collectives_difftest --n 30 --seed 5 --out DIR
    verity-vllm properties-admission check   --adapter verity_vllm.program.registry.collectives_difftest --dir DIR --out RUN/tp2.json

Rank r contributes operand r; every rank must receive the same words (asserted), and rank 0's are the production output."""
from __future__ import annotations

from verity.ir.defs import CompositeDefinition, bind
from verity.ir.refs import array_of
from verity.ir.types import Array
from verity_vllm.program.registry.b1_tp2 import AllGather2, AllReduce2
from verity_vllm.program.registry.prims import BF16

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


def _rank_main(rank: int, port: int, op: str, cpu_inputs: list, results) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", store=dist.TCPStore("127.0.0.1", port, 2, is_master=False), rank=rank, world_size=2)
    t = cpu_inputs[rank].to(f"cuda:{rank}")
    if op == "all_reduce":
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        got = t
    else:
        got = torch.empty(2 * t.numel(), dtype=t.dtype, device=t.device)
        dist.all_gather_into_tensor(got, t)
    torch.cuda.synchronize()
    results.put((rank, got.cpu()))
    dist.destroy_process_group()


def production(statics: dict, inputs: list) -> list:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"the two-rank collectives need 2 CUDA devices, found {torch.cuda.device_count()}")
    master = dist.TCPStore("127.0.0.1", 0, 2, is_master=True, wait_for_workers=False)
    ctx = mp.get_context("spawn")
    results = ctx.SimpleQueue()
    cpu_inputs = [x.detach().cpu() for x in inputs]
    procs = [ctx.Process(target=_rank_main, args=(r, int(master.port), statics["op"], cpu_inputs, results)) for r in range(2)]
    for p in procs:
        p.start()
    outs = dict(results.get() for _ in procs)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"{statics['op']} rank process exited with {p.exitcode}")
    if not torch.equal(outs[1].view(torch.int16), outs[0].view(torch.int16)):
        raise AssertionError(f"{statics['op']}: rank 1 received other words than rank 0")
    statics["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    del master
    return [outs[0].to(inputs[0].device)]


def _difftest_body(B, S, a, b):
    """The collective of rank 0's operand `a` and rank 1's `b`, as the TP2 Program binds it: `parts = [a, b]`."""
    return B.call(bind(AllReduce2 if S.OP == "all_reduce" else AllGather2, N=S.N), array_of([a, b]))


Collectives2Difftest = CompositeDefinition(
    "Collectives2Difftest", 1, ("OP", "N"),
    lambda S: ((("a", Array(S.N, BF16)), ("b", Array(S.N, BF16))), Array(S.N if S.OP == "all_reduce" else 2 * S.N, BF16)),
    _difftest_body, doc=_difftest_body.__doc__ or "", conformance="derived: difftest", register=False)


def spec(statics: dict):
    return bind(Collectives2Difftest, OP=str(statics["op"]), N=int(statics["N"]))
