"""One node's TP all-reduce arithmetic on another: every all-reduce FlashInfer's MNNVL route doesn't take is summed in the serving
node's multimem reduction tree, in software, so a replay on a node whose NVSwitch reduces in another order computes the served bits.

On the served H200 node, `CudaCommunicator.all_reduce` hands a bf16 all-reduce FlashInfer's 0.5 MiB doesn't take to torch's
symmetric memory (``multimem.ld_reduce ... acc::f32``, to 64 MiB) and past that to NCCL's NVLS; the node's capture found both reduce
every element in one tree of f32 adds, then round to bf16 (`AllReduceMultimem_v1` at the node's TREES, FTZ off). This plugin keeps
FlashInfer's route and the fused all-reduce + RMSNorm as they are (their order is the ranks', on every node), and replaces the
rest with an NCCL all-gather (a copy) and that tree's f32 adds on every rank, rounded once to bf16. It changes what the replay
node computes, never the served record: a replay under it says so.

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


def tree_sum(parts: Any, t: Tree) -> Any:
    """``parts[r]`` (f32, one row per rank) added in the tree's order: a node is its two subtrees' f32 sum."""
    return parts[t] if isinstance(t, int) else tree_sum(parts, t[0]) + tree_sum(parts, t[1])


def reduce_gathered(gathered: Any, world: int, t: Tree) -> Any:
    """A rank-major all-gather of a bf16 tensor (``world * n`` words) reduced in ``t``: the f32 sum, rounded once to bf16."""
    import torch
    return tree_sum(gathered.view(world, -1).to(torch.float32), t).to(torch.bfloat16)


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
    print(f"[tree-allreduce] all-reduces past FlashInfer's route sum in {cfg['tree']} ({cfg.get('source', '')})",
          file=sys.stderr, flush=True)


def wrap(orig: Any, t: Tree) -> Any:
    def all_reduce(self, input_):
        import torch
        fi, nc = self.fi_ar_comm, self.pynccl_comm
        if (fi is not None and not fi.disabled and fi.should_use_fi_ar(input_)) or input_.dtype != torch.bfloat16 \
                or nc is None or nc.disabled:
            return orig(self, input_)
        if sorted(leaves(t)) != list(range(self.world_size)):
            raise RuntimeError(f"tree {t} is not over the group's {self.world_size} ranks")
        flat = input_.contiguous().view(-1)
        gathered = torch.empty(self.world_size * flat.numel(), dtype=flat.dtype, device=flat.device)
        nc.all_gather(gathered, flat)
        return reduce_gathered(gathered, self.world_size, t).view(input_.shape)
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
