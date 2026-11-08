"""An 8-GPU NCCL bring-up probe: peer access between every pair, then one torch.distributed NCCL all-reduce.

    python spot-send/nccl_probe.py RENDEZVOUS_FILE      (NCCL_DEBUG=INFO for NCCL's own account)
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def rank_main(rank: int, n: int, rdv: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{rdv}", rank=rank, world_size=n, device_id=torch.device("cuda", rank))
    x = torch.full((1 << 20,), float(rank + 1), device="cuda")
    dist.all_reduce(x)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"nccl_probe: all_reduce ok, {x[0].item()} (want {n * (n + 1) / 2})", flush=True)
    dist.destroy_process_group()


def main() -> None:
    n = torch.cuda.device_count()
    print(f"nccl_probe: torch {torch.__version__} nccl {torch.cuda.nccl.version()} devices {n}", flush=True)
    print("nccl_probe: NCCL_ env " + " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("NCCL_")), flush=True)
    denied = [(i, j) for i in range(n) for j in range(n) if i != j and not torch.cuda.can_device_access_peer(i, j)]
    print(f"nccl_probe: peer access denied for {len(denied)} of {n * (n - 1)} pairs {denied[:8]}", flush=True)
    a = torch.arange(1 << 20, device="cuda:0", dtype=torch.float32)
    b = a.to("cuda:1")
    torch.cuda.synchronize()
    print(f"nccl_probe: peer copy 0->1 equal {bool(torch.equal(a.cpu(), b.cpu()))}", flush=True)
    mp.spawn(rank_main, args=(n, sys.argv[1]), nprocs=n, join=True)


if __name__ == "__main__":
    main()
