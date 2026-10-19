"""The MoE align schedule of the Marlin experts on one rank, as an acquisition source: per MoE layer and step, `MoeAlignRanks_v1`'s
rank and start and its `MoeAlignBlocks_v1` call boundary's blk and par (`registry.marlin_moe`, emitted by `rules/vllm_marlin_moe`),
committed as `<experts>/align.rank`, `align.start`, `align.blk` and `align.par` and bound to those ports (`PORTS`).

`fused_marlin_moe` (experts/marlin_moe.py: DeepSeek-V4's MXFP4 experts on Hopper) calls its module's `moe_align_block_size(topk_ids,
block_size_m, global_num_experts, expert_map, ignore_invalid_experts=True)`.  The kernel writes each pair p = t * TOPK + k routed to an
expert e in [0, E) at sorted_token_ids[offset(e) + rank], the rank in the order atomicAdd handed out, pads each expert's pairs to whole
blocks of BM with numel, and writes each block's expert and num_tokens_post_padded; no tensor holds a pair's rank or block.  The source
wraps that name: after the call it takes each pair's position in sorted_token_ids[:num_tokens_post_padded] as its key (a pair absent
from it keys after every present one, by index) and computes, on the device, what the Definitions compute over that key (`align_values`):
rank = the pairs of its expert with a smaller (key, index), start = e in [0, E) and rank % BM == 0, blk = the starts of the pairs of
smaller experts plus rank // BM, par = the starts' sum.  In the same pass it holds them to the kernel's own output: every pair of [0, E)
placed once, at sorted_token_ids[blk * BM + rank % BM], in a block of expert e, every other written slot numel, and par * BM =
num_tokens_post_padded; the verdicts are read once, at the step's end (`on_step_end`), and a failed one is a collector error naming
the layer.  The rank Program's `schedule` input, the key `MoeAlignRanks_v1` reads, is bound to the committed rank
(`check.replay.evaluate`): any key that orders each expert's pairs gives ranks of the kernel's form, and the rank orders them as the
GPU did.  An `expert_map` (expert parallelism) is refused.  Each value is committed as the 32-bit words of its i32 tensor, a float32
view (`call_boundary_plan.WORD_DTYPES` "f32").
"""
from __future__ import annotations

from typing import Any, Mapping

from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources.moe_source import moe_layers_of
from verity_vllm.engine import hooks
from verity_vllm.query.manifest.bind import Port

MARLIN_MODULE = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
ALIGN = "moe_align_block_size"
RANKS, BLOCKS = "MoeAlignRanks_v1", "MoeAlignBlocks_v1"
MEMBERS = ("align.rank", "align.start", "align.blk", "align.par")
#: `bind.Binder`'s declared labels: the ranks are the experts module's MoeAlignRanks_v1 ports, the blocks its MoeAlignBlocks_v1 call boundary's
PORTS = {MEMBERS[0]: Port(frozenset({RANKS}), "0"), MEMBERS[1]: Port(frozenset({RANKS}), "1"),
         MEMBERS[2]: Port(frozenset({BLOCKS}), "0", call_boundary=True), MEMBERS[3]: Port(frozenset({BLOCKS}), "1", call_boundary=True)}
WORD_BYTES = 4


def _spec_family(r: Mapping[str, Any]) -> str:
    return str(r.get("spec") or "").split("{", 1)[0]


def manifest_names_it(manifest: Mapping[str, Any] | None) -> bool:
    """Whether a manifest's rows name a `MoeAlignRanks_v1` or `MoeAlignBlocks_v1` Call."""
    return any(_spec_family(r) in (RANKS, BLOCKS) for r in (manifest or {}).get("identities") or ())


def claims(r: Mapping[str, Any]) -> str | None:
    """`call_boundary_source.CLAIMS`: a `MoeAlignBlocks_v1` call boundary is this source's."""
    return MoeAlignSource.short_name if r.get("family") == CB.FAMILY and _spec_family(r) == BLOCKS else None


if claims not in CB.CLAIMS:
    CB.CLAIMS.append(claims)


def align_values(topk_ids: Any, sorted_ids: Any, expert_ids: Any, num_post_padded: Any, num_experts: int, block: int) -> tuple[Any, ...]:
    """(rank, start, blk, par, ok) of one `moe_align_block_size` call from its kernel's outputs: int32 [P] x 3, int32 [1], and a bool
    scalar, whether those outputs are the ones the kernel writes for that schedule (module docstring).  On the inputs' device, no sync."""
    import torch

    dev, i64 = topk_ids.device, torch.int64
    e = topk_ids.reshape(-1).to(i64)
    P, L = int(e.numel()), int(sorted_ids.numel())
    idx, pos = torch.arange(P, device=dev), torch.arange(L, device=dev)
    s = sorted_ids.to(i64)
    live = pos < num_post_padded.reshape(-1)[:1].to(i64)
    hit = live & (s >= 0) & (s < P)
    tgt = torch.where(hit, s, P)
    seen = torch.zeros(P + 1, dtype=i64, device=dev).scatter_add_(0, tgt, torch.ones(L, dtype=i64, device=dev))[:P]
    first_pos = torch.full((P + 1,), L, dtype=i64, device=dev).scatter_reduce_(0, tgt, pos, reduce="amin")[:P]
    key = torch.where(seen > 0, first_pos, L + idx)
    by_key = torch.argsort(key, stable=True)
    o = by_key[torch.argsort(e[by_key], stable=True)]
    es = e[o]
    new = torch.ones(P, dtype=torch.bool, device=dev)
    new[1:] = es[1:] != es[:-1]
    head = torch.cummax(torch.where(new, idx, 0), 0).values
    rank_o = idx - head
    valid_o = (es >= 0) & (es < num_experts)
    start_o = (valid_o & (rank_o % block == 0)).to(i64)
    blk_o = (torch.cumsum(start_o, 0) - start_o)[head] + rank_o // block
    rank, start, blk = (torch.empty_like(x).scatter_(0, o, x) for x in (rank_o, start_o, blk_o))
    par = start_o.sum().reshape(1)
    valid = (e >= 0) & (e < num_experts)
    nb = int(expert_ids.numel())
    ok = torch.stack([
        (seen == valid.to(i64)).all(),
        (~live | hit | (s == P)).all(),
        (~valid | (key == blk * block + rank % block)).all(),
        (~valid | (expert_ids.to(i64)[blk.clamp(0, max(nb - 1, 0))] == e)).all() if nb else ~valid.any(),
        (par * block == num_post_padded.reshape(-1)[:1].to(i64)).all(),
    ]).all()
    i32 = torch.int32
    return rank.to(i32), start.to(i32), blk.to(i32), par.to(i32), ok


class MoeAlignSource:
    short_name = "moe_align"

    def __init__(self, committer: Any, model: Any) -> None:
        import torch

        self.torch, self.c = torch, committer
        self.layers = moe_layers_of(model)
        if not self.layers:
            raise ValueError("moe align: the model has no MoE module")
        self.modules = [(n, m) for n, m in model.named_modules() if n in self.layers]
        self.cur: str | None = None
        self._calls: dict[str, int] = {}
        self._verdicts: list[tuple[str, Any]] = []
        self._held: list[Any] = []
        self._handles: list[Any] = []
        self._hooks = hooks.Hooks()
        self.stats: dict[str, Any] = {"calls": 0, "pairs": 0, "layers": len(self.layers), "refused": 0, "unattributed": 0}
        self.active = False

    def install(self) -> None:
        import importlib

        for name, m in self.modules:
            self._handles.append(m.register_forward_pre_hook(self._enter(name)))
            self._handles.append(m.register_forward_hook(self._leave(name), always_call=True))
        mod = importlib.import_module(MARLIN_MODULE)
        if not callable(getattr(mod, ALIGN, None)):
            raise RuntimeError(f"moe align: {MARLIN_MODULE} has no {ALIGN}")
        self._hooks.wrap(mod, ALIGN, self.wrap)
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._hooks.uninstall()
        self.active = False

    def on_step_begin(self) -> None:
        self.cur, self._calls, self._verdicts, self._held = None, {}, [], []

    def on_step_end(self, job: Any) -> None:
        if not self._verdicts:
            return
        bad = (~self.torch.stack([ok for _n, ok in self._verdicts])).nonzero().flatten().tolist()
        for i in bad[:4]:
            self.error(f"moe align: {self._verdicts[i][0]} at engine step {int(job.step)}: moe_align_block_size's outputs are not the "
                       f"kernel's for the schedule recomputed from its sorted_token_ids")
        self._verdicts = []

    def _enter(self, name: str) -> Any:
        def pre(_module: Any, _args: tuple) -> None:
            self.cur = name
        return pre

    def _leave(self, name: str) -> Any:
        def post(_module: Any, _args: tuple, _out: Any) -> None:
            if self.cur == name:
                self.cur = None
        return post

    def error(self, msg: str) -> None:
        self.c.errors.append(msg[:300])

    def wrap(self, orig: Any) -> Any:
        me = self

        def moe_align_block_size(topk_ids: Any, block_size: int, num_experts: int, expert_map: Any = None, *a: Any, **k: Any) -> Any:
            out = orig(topk_ids, block_size, num_experts, expert_map, *a, **k)
            job = getattr(me.c, "_cur", None)
            if job is not None:
                me.bind(topk_ids, int(block_size), int(num_experts), expert_map, out, job)
            return out

        return moe_align_block_size

    def bind(self, topk_ids: Any, block: int, num_experts: int, expert_map: Any, out: tuple, job: Any) -> None:
        op = self.cur
        if op is None:
            self.stats["unattributed"] += 1
            return self.error(f"moe align: {ALIGN} called outside every MoE module")
        self._calls[op] = self._calls.get(op, 0) + 1
        if self._calls[op] > 1:
            return self.error(f"moe align: {op} called {ALIGN} {self._calls[op]} times in one step")
        if expert_map is not None:
            self.stats["refused"] += 1
            return self.error(f"moe align: {op} passed an expert_map (expert parallelism); {RANKS} ranks every expert of [0, E)")
        sorted_ids, expert_ids, ntpp = out
        rank, start, blk, par, ok = align_values(topk_ids, sorted_ids, expert_ids, ntpp, num_experts, block)
        f32 = self.torch.float32
        for member, t in zip(MEMBERS, (rank, start, blk, par)):
            v = t.view(f32)
            self._held.append(v)        # alive until the step ends: the committer may stage it asynchronously
            self.c._acquire(f"{op}/{member}", v, job)
        self._verdicts.append((op, ok))
        self.stats["calls"] += 1
        self.stats["pairs"] += int(rank.numel())

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        """rank, start and blk per pair (TOPK pairs per token), par once per layer."""
        topk = sum(int(getattr(m, "top_k", 0) or 0) for _n, m in self.modules)
        return {"per_token_bytes": 3 * WORD_BYTES * topk, "per_seq_bytes": WORD_BYTES * len(self.modules),
                "tensors_per_step": len(MEMBERS) * len(self.modules)}

    def declaration_classes(self) -> list[dict]:
        return []

    def ports(self) -> dict[str, Port]:
        return PORTS


def attach(com: Any, model: Any, manifest: Mapping[str, Any] | None) -> MoeAlignSource | None:
    """Attach the source to `com` when the rank's manifest rows name the align Definitions; None otherwise."""
    if not manifest_names_it(manifest):
        return None
    src = MoeAlignSource(com, model)
    com.attach_source(src)
    return src


__all__ = ["MoeAlignSource", "align_values", "manifest_names_it", "claims", "attach", "MEMBERS", "PORTS", "RANKS", "BLOCKS",
           "MARLIN_MODULE", "ALIGN"]
