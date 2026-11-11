"""difftest adapter for `tp2/AllReduceSumBf16@v1`: production = `ncclAllReduce(…, ncclBfloat16, ncclSum)` over R = 2 CUDA devices.

The protected `difftest.produce` is single-process and hands `production` the R operand tensors on `cuda:0`.  An NCCL all-reduce needs
one process per rank, so `production` spawns R processes (torch.multiprocessing, NCCL backend over loopback), gives rank r its operand
on `cuda:r`, all-reduces with `ReduceOp.SUM`, and returns rank 0's result; every other rank's result is asserted bit-equal (the "every
rank receives the same word" fact of the module docstring).  Needs a host with R visible GPUs (a 2×4090 pod).

    verity-vllm properties-admission produce --adapter verity_vllm.program.quarantine.collective.allreduce_difftest --n 20 --seed 7 --out DIR
    verity-vllm properties-admission check   --adapter verity_vllm.program.quarantine.collective.allreduce_difftest --dir DIR --out quarantine.json

`torch.distributed.all_reduce` and vLLM's `PyNcclCommunicator.all_reduce` both end in `ncclAllReduce` with the same (op, dtype); NCCL may
pick RING instead of the captured TREE for a stand-alone communicator — for R = 2 both perform the same single addition per element.
"""
from __future__ import annotations

SEMANTIC_ID = "tp2/AllReduceSumBf16@v1"
KINDS = ("random", "zeros", "max", "subnormal", "alternating")
SHAPES = (1536, 64, 4096)   # Qwen2.5-1.5B hidden size (every captured all-reduce row), plus small / larger
WORLD = 2


def cases(rng, i: int) -> dict:
    kind = KINDS[i % len(KINDS)]
    N = SHAPES[(i // len(KINDS)) % len(SHAPES)]
    peer_kind = "random" if kind in ("zeros", "subnormal") else kind   # zeros + zeros / subnormal + subnormal are trivial: mix
    return {"statics": {"N": N, "R": WORLD},
            "inputs": [{"shape": [N], "dtype": "bfloat16", "kind": kind}] + [{"shape": [N], "dtype": "bfloat16", "kind": peer_kind}] * (WORLD - 1),
            "edge": kind != "random"}


def _rank_main(rank: int, world: int, port: int, cpu_inputs: list, results) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    store = dist.TCPStore("127.0.0.1", port, world, is_master=False)
    dist.init_process_group("nccl", store=store, rank=rank, world_size=world)
    t = cpu_inputs[rank].to(f"cuda:{rank}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    results.put((rank, t.cpu()))
    dist.destroy_process_group()


def production(statics: dict, inputs: list) -> list:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    world = int(statics["R"])
    if torch.cuda.device_count() < world:
        raise RuntimeError(f"AllReduceSumBf16 difftest needs {world} CUDA devices, found {torch.cuda.device_count()}")
    # the rendezvous store lives in this (parent) process on an ephemeral port; the rank processes attach to it
    master = dist.TCPStore("127.0.0.1", 0, world, is_master=True, wait_for_workers=False)
    port = int(master.port)
    ctx = mp.get_context("spawn")
    results = ctx.SimpleQueue()
    cpu_inputs = [x.detach().cpu() for x in inputs]
    procs = [ctx.Process(target=_rank_main, args=(r, world, port, cpu_inputs, results)) for r in range(world)]
    for p in procs:
        p.start()
    outs = {}
    for _ in range(world):
        r, t = results.get()
        outs[r] = t
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"all-reduce rank process exited with {p.exitcode}")
    for r in range(1, world):
        if not torch.equal(outs[r].view(torch.int16), outs[0].view(torch.int16)):
            raise AssertionError(f"rank {r} result differs from rank 0 (NCCL contract: every rank receives the same words)")
    del master
    return [outs[0].to(inputs[0].device)]


def spec(statics: dict):
    from verity_vllm.program.quarantine.collective.allreduce import all_reduce_sum_bf16
    return all_reduce_sum_bf16(int(statics["N"]), int(statics["R"]))
