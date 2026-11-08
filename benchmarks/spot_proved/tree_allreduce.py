"""One node's TP all-reduce arithmetic on another: the all-reduces the serving node's NVSwitch reduced are summed in its multimem
reduction tree, in software, so a replay on a node whose NVSwitch reduces in another order computes the served bits.

On the served H200 node, `CudaCommunicator.all_reduce` hands a bf16 all-reduce FlashInfer's 0.5 MiB doesn't take to torch's
symmetric memory (``multimem.ld_reduce ... acc::f32``, to 64 MiB) and past that to NCCL's NVLS; the node's capture found both reduce
every element in one tree of f32 adds, then round to bf16 (`AllReduceMultimem_v1` at the node's TREES, FTZ off). This plugin keeps
FlashInfer's route, the custom all-reduce and the fused all-reduce + RMSNorm as they are (their order is in software, the same on every node),
and replaces the other two with an NCCL all-gather (a copy) and that tree's f32 adds on every rank, rounded once to bf16. It
changes what the replay node computes, never the served record: a replay under it says so.

    python tree_allreduce.py site DIR --tree '(((((((0,2),3),1),5),6),4),7)' --source WHERE
        # DIR on PYTHONPATH registers the plugin (vLLM general plugin ``verity_tree_allreduce``) with that tree
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

CONFIG = "tree_allreduce.json"
Tree = int | tuple


def parse_tree(s: str) -> Tree:
    """A tree as the probe prints it, ``(((0,2),3),1)``: nested pairs of ranks."""
    t = ast.literal_eval(s)

    def check(x: Any) -> Tree:
        if isinstance(x, int):
            return x
        if isinstance(x, tuple) and len(x) == 2:
            return (check(x[0]), check(x[1]))
        raise ValueError(f"{s!r}: a tree is a rank or a pair of trees")
    return check(t)


def leaves(t: Tree) -> list[int]:
    return [t] if isinstance(t, int) else leaves(t[0]) + leaves(t[1])


def reduce_gathered(gathered: Any, world: int, t: Tree) -> Any:
    """A rank-major all-gather of a bf16 tensor (``world * n`` words) reduced in ``t``, a node its two subtrees' f32 sum, then
    rounded once to bf16."""
    import torch
    rows = gathered.view(world, -1)

    def go(x: Tree) -> Any:
        return rows[x].to(torch.float32) if isinstance(x, int) else go(x[0]) + go(x[1])
    return go(t).to(torch.bfloat16)


def config(d: Path | None = None) -> dict | None:
    p = (d or Path(__file__).resolve().parent) / CONFIG
    return json.loads(p.read_text()) if p.is_file() else None


_TREE: Tree | None = None


def register() -> None:
    """The ``vllm.general_plugins`` entry: wraps `CudaCommunicator.all_reduce` with the configured tree."""
    global _TREE
    cfg = config()
    if cfg is None or _TREE is not None:
        return
    _TREE = parse_tree(cfg["tree"])
    from vllm.distributed.device_communicators import cuda_communicator as cc
    cc.CudaCommunicator.all_reduce = wrap(cc.CudaCommunicator.all_reduce, _TREE)
    print(f"[tree-allreduce] symmetric-memory and NCCL all-reduces sum in {cfg['tree']} ({cfg.get('source', '')})",
          file=sys.stderr, flush=True)


CHUNK = 1 << 22  # words a rank gathers at once: elementwise, so only the transient memory depends on it
UNMEASURED = ("qr_comm", "fi_pcie_ipc_ar_comm", "aiter_ar_comm")


def wrap(orig: Any, t: Tree) -> Any:
    """`CudaCommunicator.all_reduce` with the calls vLLM hands torch's symmetric memory or NCCL summed in ``t``. FlashInfer's and
    the custom all-reduce's calls (an order in software, the same on every node) and other dtypes go to vLLM; a route the tree
    wasn't measured with (NCCL's symmetric memory, quick reduce, FlashInfer's PCIe IPC, aiter) is refused."""
    def all_reduce(self, input_):
        import torch
        fi, ca = self.fi_ar_comm, self.ca_comm
        if input_.dtype != torch.bfloat16 or (fi is not None and not fi.disabled and fi.should_use_fi_ar(input_)) \
                or (ca is not None and not ca.disabled and ca.should_custom_ar(input_)):
            return orig(self, input_)
        from vllm.distributed.device_communicators.pynccl_allocator import is_symmetric_memory_enabled
        on = [k for k in UNMEASURED if getattr(self, k, None) is not None and not getattr(getattr(self, k), "disabled", False)]
        if is_symmetric_memory_enabled():
            on.append("NCCL's symmetric memory")
        if on:
            raise RuntimeError(f"tree-allreduce: {', '.join(on)} on, a route the tree wasn't measured with")
        if sorted(leaves(t)) != list(range(self.world_size)):
            raise RuntimeError(f"tree-allreduce: tree {t} is not over the group's {self.world_size} ranks")
        nc, w = self.pynccl_comm, self.world_size
        flat = input_.contiguous().view(-1)
        out = torch.empty_like(flat)
        for i in range(0, flat.numel(), CHUNK):
            x = flat[i:i + CHUNK]
            g = torch.empty(w * x.numel(), dtype=x.dtype, device=x.device)
            if nc is not None and not nc.disabled:
                nc.all_gather(g, x)
            else:
                torch.distributed.all_gather_into_tensor(g, x, group=self.device_group)
            out[i:i + x.numel()] = reduce_gathered(g, w, t)
        return out.view(input_.shape)
    all_reduce.__wrapped__ = orig
    return all_reduce


def site(d: Path, tree: str, source: str) -> Path:
    """A directory that, on PYTHONPATH, registers this file as the ``vllm.general_plugins`` entry ``verity_tree_allreduce``."""
    import shutil
    parse_tree(tree)
    info = d / "verity_tree_allreduce-0.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-tree-allreduce\nVersion: 0.0\n")
    (info / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_tree_allreduce = tree_allreduce:register\n")
    shutil.copyfile(__file__, d / "tree_allreduce.py")
    (d / CONFIG).write_text(json.dumps({"tree": tree, "source": source}, indent=1))
    return d


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="tree_allreduce.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("site", help="write the plugin's site directory with its tree")
    s.add_argument("dir", type=Path)
    s.add_argument("--tree", required=True, help="the serving node's multimem reduction tree, as its probe printed it")
    s.add_argument("--source", required=True, help="where the tree was read (the probe's record)")
    a = ap.parse_args(argv)
    print(site(a.dir, a.tree, a.source))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
