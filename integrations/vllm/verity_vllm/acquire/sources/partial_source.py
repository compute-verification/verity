"""tp_partial_source — bind THIS rank's rank-local inputs of every tensor-parallel collective.

Under TP the module forward hooks of the native collector see the collective OUTPUT of a `RowParallelLinear` (o_proj / down_proj:
`quant_method.apply` -> `tensor_model_parallel_all_reduce`), of the `VocabParallelEmbedding` (masked partial -> all_reduce) and of the
`LogitsProcessor` (lm_head shard -> all_gather).  The TP-aware Program (derive_step --tp W --rank r) requires the rank-LOCAL values those
collectives consume — `part_rank_r` = `Gemm_v1{K/W}` on the rank's weight shard, the masked embedding partial, the logits shard — as
values of THIS rank (the received output is a function of both ranks' partials, never a fresh input).  Committing only the outputs
would leave the rank-local GEMM partials unbound; this source snapshots the INPUT tensor of every collective call into the
same packed step stream (same run root, same `chunk-leaf-v1` layout) BEFORE the collective runs, attributed to the calling module by a
forward pre-hook, named `<module path>/part_rank<r>` (the all-gather input: `<module path>/shard_rank<r>`) -- bind's identity vocabulary: op_path = the issuing module, output_member = the rank-local partial member.

Expected population (install time, not from the collector): every hooked collective site produces exactly ONE partial per step; the
committer's `identity_check` verifies it per step (`tp_sites_bad`).  Bytes/step for the arena bound: `extra_step_bytes(envelope)`.
The same seam as `tp2_worker.tp2_coll_install` (module-global collective names in vLLM's linear / vocab_parallel_embedding /
logits_processor modules); the computation is unchanged (the wrapper snapshots `x` and calls the original).  A model that fuses
the all-reduce into the RMSNorm after it (`fused_allreduce_rms_norm`) adds one site per norm it passes there, bound by `allreduce_norm`.
"""
from __future__ import annotations

import importlib
import time
from typing import Any

from verity_vllm.acquire.sources import oracle_witness as OW
from verity_vllm.acquire.sources.allreduce_norm import AllReduceNorm
from verity_vllm.commit.opened import OpenedReader, OpeningNotVerified
from verity_vllm.engine import hooks
from verity_vllm.engine.collectives.collective_sites import MOE_COLLECTIVE_CLASSES as MOE_SITE_CLASSES, MOE_COLLECTIVE_MODULES

SITES = (("vllm.model_executor.layers.linear", "tensor_model_parallel_all_reduce", "all_reduce"),
         ("vllm.model_executor.layers.vocab_parallel_embedding", "tensor_model_parallel_all_reduce", "all_reduce"),
         ("vllm.model_executor.layers.logits_processor", "tensor_model_parallel_all_gather", "all_gather"),
         ("vllm.model_executor.layers.logits_processor", "tensor_model_parallel_gather", "gather"),
         # the fused-MoE block's final all-reduce is issued by MoERunner (`model.layers.<L>.mlp.experts`, the
         # module the rank Program names `.../mlp.experts/all_reduce2[k]`): its input is the rank-local fused expert output (width H per
         # routed token row) -- the block's `sum` plane on THIS rank's expert shards, before the ranks are summed
         *((m, "tensor_model_parallel_all_reduce", "all_reduce") for m in MOE_COLLECTIVE_MODULES))
COVERS_FAMILIES = ["Gemm_v1 (rank shard, K/W) -> AllReduce2_v1 input", "EmbeddingShard_v1 -> AllReduce2_v1 input", "Gemm_v1 (lm_head shard) -> AllGather2_v1 input",
                   "MoeSum_v1 (rank-local fused expert output, MoERunner) -> AllReduce2_v1 input"]


def _moe_reduces(m: Any, world: int) -> bool:
    """A fused-MoE module that all-reduces its fused output across TP ranks (vLLM: reduce_results and not moe_config.skip_final_all_reduce)."""
    cfg = getattr(m, "moe_config", None) or getattr(m, "moe", None)
    if cfg is not None and getattr(cfg, "skip_final_all_reduce", False):
        return False
    if getattr(m, "reduce_results", None) is False:
        return False
    tp = getattr(m, "tp_size", None)
    if tp is None and cfg is not None:
        tp = getattr(cfg, "tp_size", None)
    return int(tp or world) > 1


def _bytes(t: Any) -> bytes:
    import torch

    t = t.detach()
    return (t.contiguous().reshape(-1).view(torch.uint8) if t.dtype != torch.uint8 else t.contiguous().reshape(-1)).cpu().numpy().tobytes()


def _committed(reader: Any, step: int, m: Any) -> bytes | None:
    """Committed bytes of layout meta `m`, opened from the committer and verified against the run root by `reader`
    (`commit.opened.OpenedReader`).  None = not committed at the step or not retained; a read that does not verify raises
    `OpeningNotVerified` (the caller fails it by name)."""
    return None if m is None else reader.read_meta(step, m)


def _ref_key(member: str, occ: int) -> str:
    """Reference-tap key of the `occ`-th commit of `member` in a step: the member itself for the first (an ordinary site commits each
    member once per step), `<member>#<occ>` for a mid-module site's later calls (OlmoeAttention's k gather is `#1`)."""
    return member if occ == 0 else f"{member}#{occ}"


def _split_ref_key(key: str) -> tuple[str, int]:
    if "#" in key:
        m, o = key.rsplit("#", 1)
        return m, int(o)
    return key, 0


def _metas_by_name(layouts: list) -> dict[str, list]:
    """Layout metas of one step grouped by committed name IN COMMIT ORDER (the binder's occurrence k = position k)."""
    out: dict[str, list] = {}
    for m in layouts:
        out.setdefault(m.name, []).append(m)
    return out


def _nth(by_name: dict[str, list], name: str, k: int):
    ms = by_name.get(name) or []
    return ms[k] if k < len(ms) else None


class TPPartialSource:
    short_name = "tp_partial"
    ns_name = "tp_partial"

    def __init__(self, committer, model, *, rank: int, world: int, relabel_rank: int | None = None, omit_site: str | None = None,
                 flip_site: str | None = None) -> None:
        self.c = committer
        self.rank, self.world = int(rank), int(world)
        # NEGATIVE: at ONE site this rank COMMITS its genuine partial (the GEMM's output on the shard -- the per-rank
        # sampled replay stays clean) but feeds the collective a DIFFERENT tensor (element 0 of the partial moved by 2x+1024): the received
        # all-reduce output on every rank is then NOT the definition over the committed partials -> the cross-rank collective-output pass
        # must FAIL naming the site.  A per-rank check cannot see this (each rank's record is self-consistent).  Never set on the honest path.
        # `<site>` = element 0 moved by 2x+1024 (gross: tokens change, several checks fire); `<site>:ulp` = element 0 moved by ONE bf16 ulp (the
        # DISCRIMINATING case: the received all-reduce differs from the definition in one element of one row at one site; tokens are
        # expected to be unchanged, so token equality / attribution are blind and only a check that evaluates the collective output
        # against the committed partials can name it)
        self.flip_site, self.flip_mode = self.parse_flip(flip_site)
        self.flipped = 0
        # NEGATIVE ("relabel rank 0 as rank 1"): tag this rank's partials with another rank's member name.  The identity's
        # `rank` field is fixed by the committing rank (the binding map is built with rank=self.rank); only the MEMBER name lies, so the
        # manifest's `part_rank<self.rank>` identities find no binding -> coverage FAIL naming this rank.  Never set on the honest path.
        self.label_rank = int(relabel_rank) if relabel_rank is not None else self.rank
        # NEGATIVE: drop the partial of ONE site only (module name suffix match, e.g. "layers.3.mlp.down_proj") -- the rest
        # of the family is bound; per-identity coverage must FAIL naming this rank and that site.  Never set on the honest path.
        self.omit_site = str(omit_site) if omit_site else None
        # optional reference tap for the value check: CPU clones of every partial at bind time (+ the module input of the
        # first `ref_reexec_steps` steps for a re-execution of the rank-local GEMM on the shard) -- one extra instrumented run only
        self.reference = False
        self.ref_reexec_steps = 4
        self.ref: dict[int, dict[str, bytes]] = {}
        self.ref_in: dict[int, dict[str, Any]] = {}
        self._cur_in: Any = None
        # the collective-issuing modules of THIS rank's model = the expected sites (one partial each per step)
        self.sites: list[tuple[str, str]] = []
        self.reexec_sites: list[str] = []                                    # the sites the value check's (R) re-executes
        for n, m in model.named_modules():
            cls = type(m).__name__
            row = cls == "RowParallelLinear" and getattr(m, "reduce_results", True) and int(getattr(m, "tp_size", self.world)) > 1
            if row or cls in ("VocabParallelEmbedding", "LogitsProcessor") or (cls in MOE_SITE_CLASSES and _moe_reduces(m, self.world)):
                self.sites.append((n, cls))
            if row:
                self.reexec_sites.append(n)
        # one fused-MoE site per block: when a fused-MoE wrapper holds a fused-MoE runner submodule
        # (SharedFusedMoE -> .runner), the block's all-reduce is attributed to the OUTERMOST hooked MoE module = the Program's module path
        # (tp_links: 'layers.N.mlp.experts (MoERunner)'; the MoeSum row's path), never to both.  On Qwen3-30B-A3B / OLMoE the runner IS
        # `.experts` (no nesting) and this drops nothing.  Nested non-MoE sites (a RowParallelLinear inside a hooked module) stay their
        # own site: `cur` is a stack, so a collective issued by the outer module after the inner returns is attributed to the outer.
        moe_paths = [n for n, c in self.sites if c in MOE_SITE_CLASSES]
        nested = [(n, c) for n, c in self.sites if c in MOE_SITE_CLASSES and any(n.startswith(p + ".") for p in moe_paths)]
        self.sites = [s for s in self.sites if s not in nested]
        self.nested_moe_sites_dropped = [n for n, _c in nested]
        # MID-module sites: model classes that call a collective name bound in THEIR OWN python module
        # (OlmoeAttention: q and k all-gathered inside `self_attn`, twice per layer).  `collective_sites.mid_module_sites` is the one finder
        # (shared with the Match recorder); this source hooks the modules for attribution, patches their modules' collective names, binds
        # each call's rank-local input (`<module>/shard_rank<r>` | `/part_rank<r>`) AND received output (`<module>/out`; no layer module
        # hook binds it) -- the k-th call of the step is the binder's occurrence k = the manifest's invocation k (`collective_site_aliases`).
        # A call its instance's flags rule out (Qwen3MoeSparseMoeBlock's sequence-parallel all_gather) is no expected partial; it is listed
        # (`mid_sites_ruled_out`, the identity report's `tp_sites_ruled_out`) and its module stays patched, so a call that runs is named.
        from verity_vllm.engine.collectives.collective_sites import mid_module_site_census
        census = mid_module_site_census(model, self.world)
        self.mid_sites: dict[str, dict[str, Any]] = {s["name"]: s for s in census["sites"]}
        self.mid_sites_ruled_out: list[dict[str, Any]] = census["ruled_out"]
        self.sites += [(s["name"], s["cls"]) for s in self.mid_sites.values() if s["name"] not in {n for n, _c in self.sites}]
        self._occ: dict[tuple[int, str], int] = {}                       # (step, member) -> calls bound so far (mid-module sites commit a member twice)
        self.modules = [(n, m) for n, m in model.named_modules() if n in {s[0] for s in self.sites}]
        self.cur: tuple[str, str] | None = None
        self._cur_stack: list[tuple[str, str]] = []
        self._handles: list = []
        self._hooks = hooks.Hooks()
        self.active = False
        self.stats = {"launches": 0, "bytes": 0, "sites": len(self.sites), "acquire_s": 0.0, "unattributed": 0, "shapes": {}, "kinds": {}}
        self.far = AllReduceNorm(self, model)

    # -- expected population / arena bound -----------------------------------------------------------------------------------
    def site_names(self) -> list[str]:
        return [n for n, _c in self.sites] + self.far.site_names()

    def received_site_names(self) -> list[str]:
        """The sites whose RECEIVED collective output this source binds as `<module>/out` (fused-MoE blocks and mid-module sites; a
        linear's / embedding's / logits processor's output is the native collector's module hook) -- expected modules of every step for
        `identity_check`."""
        return [n for n, c in self.sites if c in MOE_SITE_CLASSES or n in self.mid_sites]

    def site_counts(self) -> dict[str, int]:
        """Rank-local partials this source binds per site per step: 1 (one collective per module forward), or a mid-module site's static
        call count (OlmoeAttention: 2) -- what identity_check requires exactly (`tp_sites_bad` otherwise)."""
        return {**{n: int(self.mid_sites[n]["per_step"]) if n in self.mid_sites else 1 for n, _c in self.sites}, **dict.fromkeys(self.far.site_names(), 1)}

    def extra_step_bytes(self, envelope: dict[str, Any], model: Any) -> dict[str, int]:
        """Bytes this source adds to ONE step (prefill worst case): every RowParallelLinear partial is (tokens, output_size) bf16,
        the embedding partial (tokens, hidden) bf16, the logits shard (sequences, vocab/W) of the lm_head dtype."""
        per_token = 0
        per_seq = 0
        n_moe_out = 0
        for n, m in self.modules:
            cls = type(m).__name__
            isz = 2
            try:
                isz = next(m.parameters()).element_size()
            except StopIteration:
                pass
            if cls == "RowParallelLinear":
                per_token += int(getattr(m, "output_size", 0) or 0) * isz
            elif cls == "VocabParallelEmbedding":
                per_token += int(getattr(m, "embedding_dim", 0) or 0) * isz
            elif cls == "LogitsProcessor":
                vocab = int(getattr(m, "vocab_size", 0) or getattr(m, "org_vocab_size", 0) or 0)
                per_seq += -(-vocab // max(1, self.world)) * 2
            elif cls in MOE_SITE_CLASSES:
                # the fused expert output before the all-reduce: (tokens, hidden) in the activation dtype (H from the module / its config)
                # -- and the RECEIVED all-reduced block output (`<module>/out`, same shape), which this source also binds
                cfg = getattr(m, "moe_config", None) or getattr(m, "moe", None)
                h = int(getattr(m, "hidden_size", 0) or getattr(cfg, "hidden_dim", 0) or getattr(cfg, "hidden_size", 0) or 0)
                per_token += 2 * h * isz
                n_moe_out += 1
            elif n in self.mid_sites:
                # per call: the rank-local input (<= H per token) and the received output (<= H per token: a gathered q/k row is the
                # module's full head width); H from the module, else its config -- an upper bound for the arena, exact bytes are the run's
                cfg = getattr(m, "config", None)
                h = int(getattr(m, "hidden_size", 0) or (int(getattr(m, "total_num_heads", 0) or 0) * int(getattr(m, "head_dim", 0) or 0))
                        or getattr(cfg, "hidden_size", 0) or 0)
                per_token += int(self.mid_sites[n]["per_step"]) * 2 * h * isz
                n_moe_out += 2 * int(self.mid_sites[n]["per_step"]) - 1                  # per call: partial + out; the site itself counted once below
        far_pt, far_n = self.far.extra_step_bytes()
        return {"per_token_bytes": per_token + far_pt, "per_seq_bytes": per_seq, "tensors_per_step": len(self.sites) + n_moe_out + far_n}

    # -- install / uninstall -------------------------------------------------------------------------------------------------
    def begin_run(self) -> None:
        """Per-run state: the occurrence counter is keyed by (step, member) and steps restart at 0 every run, so a counter carried over
        from an earlier run would key this run's first commit as occurrence N (every reference tap then misses its committed meta)."""
        self._occ = {}

    def install(self) -> None:
        self.begin_run()
        me = self

        def pre(name: str, cls: str):
            def h(mod, args):
                if me.cur is not None:
                    me._cur_stack.append(me.cur)
                me.cur = (name, cls)
                me._cur_in = args[0] if (me.reference and args and hasattr(args[0], "device")) else None
            return h

        def post(mod, args, output):
            me.cur = me._cur_stack.pop() if me._cur_stack else None
            me._cur_in = None

        for n, m in self.modules:
            self._handles.append(m.register_forward_pre_hook(pre(n, type(m).__name__)))
            self._handles.append(m.register_forward_hook(post))
        targets: list[tuple[Any, str, str]] = []
        for modname, attr, kind in SITES:
            try:
                targets.append((importlib.import_module(modname), attr, kind))
            except Exception:  # noqa: BLE001
                continue
        if self.mid_sites or self.mid_sites_ruled_out:
            from verity_vllm.engine.collectives.collective_sites import patch_targets
            targets += patch_targets(list(self.mid_sites.values()) + self.mid_sites_ruled_out)   # the model classes' own imports
        done: set[tuple[int, str]] = set()
        for mod, attr, kind in targets:
            if (id(mod), attr) in done or not hasattr(mod, attr):
                continue
            done.add((id(mod), attr))
            self._hooks.wrap(mod, attr, lambda orig, kind=kind: self._wrap(kind, orig))
        for mod, attr in self.far.targets:
            self._hooks.wrap(mod, attr, self.far.wrap)
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._hooks.uninstall()
        self.far.uninstall()
        self.active = False

    def on_step_begin(self) -> None:
        self.cur = None
        self._cur_stack = []
        self.far.close_step()

    def on_step_end(self, job: Any) -> None:
        self.far.close_step()

    # -- acquisition -------------------------------------------------------------------------------------------------------
    def _bind(self, kind: str, x: Any) -> None:
        """The collective INPUT x (this rank's partial / shard), staged before the collective runs.  LIFETIME ASSUMPTION:
        under the gather-mode row committer the staged reference is copied at the next flush (o_proj / down_proj hook right after the
        collective returns; layer 0's qkv_proj for the embedding partial; step end for the logits shard).  That is sound only because
        vLLM's tensor_model_parallel_all_reduce / all_gather are OUT OF PLACE (custom all-reduce writes a fresh out tensor; pynccl,
        which NCCL_P2P_DISABLE=1 forces, all-reduces into torch.empty_like(x)) and nothing writes into x afterwards: x is the GEMM's
        (or the masked embedding lookup's / the logits GEMM's) own output, consumed only by the collective.  An in-place all-reduce
        would overwrite x before the flush and the committed partial would equal the received sum -- a fault the cross-rank provenance
        check (received == AllReduce2(part_rank0, part_rank1)) cannot see; the value check's hook-time clone (self.reference) can."""
        job = getattr(self.c, "_cur", None)
        if job is None or x is None or not hasattr(x, "device") or x.device.type != "cuda":
            return
        if self.cur is None:
            self.stats["unattributed"] += 1
            self.c.errors.append(f"tp partial source: {kind} issued outside a hooked collective module forward (no site attribution)"[:200])
            self.c.x03_expected = getattr(self.c, "x03_expected", 0) + 1
            return
        t0 = time.perf_counter()
        name, _cls = self.cur
        if self.omit_site and name.endswith(self.omit_site):
            self.stats["omitted_site"] = self.stats.get("omitted_site", 0) + 1
            return
        what = "shard" if kind in ("all_gather", "gather") else "part"
        member = f"{name}/{what}_rank{self.label_rank}"
        self.c._acquire(member, x, job)      # op_path = the issuing module, output_member = part_rank<r> | shard_rank<r>
        step = int(getattr(job, "step", len(self.ref)))
        occ = self._occ.get((step, member), 0)                          # a mid-module site commits this member once per CALL (occurrence k)
        self._occ[(step, member)] = occ + 1
        if self.reference:
            self.ref.setdefault(step, {})[_ref_key(member, occ)] = _bytes(x)   # independent clone of the collective INPUT at the moment it exists
            if step < self.ref_reexec_steps and self._cur_in is not None and _cls == "RowParallelLinear":
                self.ref_in.setdefault(step, {})[member] = (name, self._cur_in.detach().clone())
        nb = x.numel() * x.element_size()
        self.stats["launches"] += 1
        self.stats["bytes"] += nb
        self.stats["acquire_s"] += time.perf_counter() - t0
        key = f"{kind}:{tuple(x.shape)}:{str(x.dtype).replace('torch.', '')}"
        self.stats["shapes"][key] = self.stats["shapes"].get(key, 0) + 1
        self.stats["kinds"][kind] = self.stats["kinds"].get(kind, 0) + 1

    def _bind_received(self, kind: str, y: Any) -> None:
        """The fused-MoE block's all-reduce OUTPUT as received by this rank, bound as
        `<module>/out` -- the manifest's AllReduce2 row under `model.layers.<L>.mlp.experts` (the received value is a required
        identity of every rank).  The native collector acquires module outputs for ACQUIRE_CLASSES only (linears, norms, embedding,
        logits); MoERunner / FusedMoE are not among them and the MoE block collector binds the six block-stream planes and never a module
        output, so this source is the ONE binder of that name (a second binder would be refused by the binding map's duplicate rule)."""
        job = getattr(self.c, "_cur", None)
        if job is None or y is None or self.cur is None:
            return
        # a MID-module site's received output too (any kind: OLMoE's gathered q / k rows are `self_attn/out`, the
        # module's k-th collective identity; no layer-module hook binds them) -- the same lifetime rule as the MoE block output below: the
        # gathered row is consumed by q_norm / k_norm (out of place) but its storage is the module's to reuse, so flush right here.
        mid = self.cur[0] in self.mid_sites
        if not mid and (kind != "all_reduce" or self.cur[1] not in MOE_SITE_CLASSES):
            return
        if not hasattr(y, "device") or y.device.type != "cuda":
            return
        name, _cls = self.cur
        if self.omit_site and name.endswith(self.omit_site):
            return
        member = f"{name}/out"
        self.c._acquire(member, y, job)     # snapshot AFTER the collective (stream-ordered behind the NCCL call on the compute stream)
        # LIFETIME: under the row committer (native_collect gather mode) _acquire STAGES a held reference that is copied
        # at the next legal collection point (GATHER_FLUSH_LEAVES = qkv_proj / o_proj / down_proj hooks + step end).  A fused-MoE block has
        # no down_proj hook, and its received output y IS the next layer's fused-norm input, which vLLM's fused_add_rms_norm overwrites IN
        # PLACE before layer L+1's qkv_proj flush (the last layer's y meets model.norm before step end) -> the committed /out would be the
        # normalised value, bit-identical to <L+1>.input_layernorm/0, never the AllReduce2 output.  So this acquisition is made a legal
        # collection point: flush right here (one gather launch per MoE layer, the same cost class as dense's down_proj flush).
        nc = getattr(self.c, "_nc", None)
        if getattr(self.c, "gather", False) and nc is not None and hasattr(nc, "flush"):
            nc.flush()
            self.stats["received_flushes"] = self.stats.get("received_flushes", 0) + 1
        step = int(getattr(job, "step", len(self.ref)))
        occ = self._occ.get((step, member), 0)
        self._occ[(step, member)] = occ + 1
        if self.reference:
            self.ref.setdefault(step, {})[_ref_key(member, occ)] = _bytes(y)
        self.stats["received_bound"] = self.stats.get("received_bound", 0) + 1
        self.stats["bytes"] += y.numel() * y.element_size()

    @staticmethod
    def parse_flip(spec: str | None) -> tuple[str | None, str]:
        """`<site>` -> (site, "gross"); `<site>:ulp` -> (site, "ulp"); None -> (None, "gross")."""
        fs = str(spec) if spec else None
        if fs and fs.endswith(":ulp"):
            return fs[:-4], "ulp"
        return fs, "gross"

    def _wrap(self, kind: str, orig):
        me = self

        def w(x, *a, **k):
            me._bind(kind, x)            # snapshot BEFORE the collective (D2D on the compute stream: stream-ordered ahead of the NCCL call)
            if me.flip_site and me.cur is not None and me.cur[0].endswith(me.flip_site) and hasattr(x, "device") and x.device.type == "cuda":
                # the flip negative: the committed partial above is genuine; the collective's INPUT is not it
                x = x.clone()
                import torch
                if me.flip_mode == "ulp" and x.dtype == torch.bfloat16:
                    v = x.view(-1)[0:1]
                    v.copy_((v.view(torch.int16) + 1).view(torch.bfloat16))      # one bf16 ulp on element 0 (magnitude +1 ulp)
                else:
                    x.view(-1)[0] = x.view(-1)[0] * 2 + 1024.0
                me.flipped += 1
            y = orig(x, *a, **k)
            me._bind_received(kind, y)   # MoE sites only: the received all-reduced block output (`<module>/out`), see _bind_received
            return y
        return w

    # -- value correspondence (C2) -----------------------------------------------------------------------------------------
    def compare(self, reader: Any) -> dict[str, Any]:
        """After finalize(): for every partial this source bound, the committed bytes (opened and verified against the
        rank's run root by `reader`, an `commit.opened.OpenedReader`) must equal (T) the hook-time reference clone taken in `_bind`, and,
        for the RowParallelLinear sites of the first `ref_reexec_steps` steps, (R) a re-execution of the rank-local GEMM on the shard from
        the tapped module input (`quant_method.apply(layer, x)` -- the exact relation `part_rank_r = Gemm_v1{K/W}(x_r, W_r)` the rank
        Program declares).  Bit-identity is what VLLM_BATCH_INVARIANT promises.  A partial the committer did not bind is
        `missing_in_commit` (FAIL); one whose opening does not verify is `not_verified` (FAIL)."""
        import torch

        t0 = time.perf_counter()
        res: dict[str, Any] = {"rank": self.rank, "steps": len(self.ref), "tensors": 0, "tap_equal": 0, "tap_mismatch": [], "reexec_tensors": 0,
                               "reexec_equal": 0, "reexec_mismatch": [], "reexec_skipped": {}, "missing_in_commit": [], "not_verified": [],
                               "by_member": {}}
        mods = dict(self.modules)
        for step, refs in sorted(self.ref.items()):
            by_name = _metas_by_name(self.c._layouts.get(step, []))
            for rkey, ref in refs.items():
                member, occ = _split_ref_key(rkey)                        # the occ-th commit of the member (mid-module sites)
                kind = member.rsplit("/", 1)[-1].split("_rank")[0]
                bm = res["by_member"].setdefault(kind, {"tensors": 0, "tap_equal": 0, "reexec": 0, "reexec_equal": 0})
                m = _nth(by_name, member, occ)
                try:
                    committed = _committed(reader, step, m)
                except OpeningNotVerified as e:
                    res["not_verified"].append((step, member, str(e)[:200]))
                    continue
                if committed is None:
                    res["missing_in_commit"].append((step, member))
                    continue
                res["tensors"] += 1
                bm["tensors"] += 1
                if committed == ref:
                    res["tap_equal"] += 1
                    bm["tap_equal"] += 1
                else:
                    res["tap_mismatch"].append((step, member, len(ref)))
                if member in self.ref_in.get(step, {}):
                    name, x = self.ref_in[step][member]
                    mod = mods.get(name)
                    try:
                        with torch.inference_mode():
                            bias = None if (int(getattr(mod, "tp_rank", 0)) > 0 or getattr(mod, "skip_bias_add", False)) else getattr(mod, "bias", None)
                            if not getattr(mod, "input_is_parallel", True):
                                raise RuntimeError("input_is_parallel=False: the module splits its input; re-exec skipped")
                            out2 = mod.quant_method.apply(mod, x.clone(), bias=bias)
                    except Exception as e:  # noqa: BLE001
                        res["reexec_skipped"][name] = repr(e)[:160]
                        continue
                    res["reexec_tensors"] += 1
                    bm["reexec"] += 1
                    if committed == _bytes(out2):
                        res["reexec_equal"] += 1
                        bm["reexec_equal"] += 1
                    else:
                        res["reexec_mismatch"].append((step, member, len(committed)))
        res["tap_mismatch_n"], res["reexec_mismatch_n"] = len(res["tap_mismatch"]), len(res["reexec_mismatch"])
        res["missing_n"], res["not_verified_n"] = len(res["missing_in_commit"]), len(res["not_verified"])
        res["tap_mismatch"], res["reexec_mismatch"], res["missing_in_commit"] = res["tap_mismatch"][:20], res["reexec_mismatch"][:20], res["missing_in_commit"][:20]
        res["not_verified"] = res["not_verified"][:20]
        res["value_source"] = reader.record()
        res["wall_s"] = time.perf_counter() - t0
        # (R) exists only for RowParallelLinear sites; a rank whose partials are all norm sites, the embedding or the logits shard
        # (GLM-5.3: o_proj / MoE leave their output un-reduced for fused_allreduce_rms_norm) has no GEMM here to re-execute
        res["reexec_due"] = bool(self.reexec_sites)
        res["verdict"] = "PASS" if (res["tensors"] > 0 and res["tap_mismatch_n"] == 0 and res["reexec_mismatch_n"] == 0 and res["missing_n"] == 0
                                    and res["not_verified_n"] == 0 and (res["reexec_tensors"] > 0 or not res["reexec_due"])) else "FAIL"
        self.ref, self.ref_in = {}, {}
        return res

    def _norm_partials_under(self, layout: list, recorded: set) -> dict[str, list]:
        """{recorded module: its all-reduce + norm sites' `<norm>/part_rank<r>` metas in commit order}, each partial under its nearest
        recorded ancestor module (a decoder layer, not the model that holds every layer)."""
        norms, tail = set(self.far.site_names()), f"/part_rank{self.rank}"
        out: dict[str, list] = {}
        for m in layout:
            site = m.name[:-len(tail)] if m.name.endswith(tail) else None
            owners = [o for o in recorded if o and site in norms and site.startswith(o + ".")] if site else []
            if owners:
                out.setdefault(max(owners, key=len), []).append(m)
        return out

    def compare_match_oracle(self, capture_dir: str, reader: Any) -> dict[str, Any]:
        """Every committed TP partial (`<site>/part_rank<r>` | `/shard_rank<r>`) and received
        fused-MoE output (`<site>/out`) of THIS rank -- opened and verified against the rank's run root by `reader` (an
        `commit.opened.OpenedReader`; a member whose opening does not verify is `not_verified`, FAIL) -- compared bit-for-bit against the
        INDEPENDENT Match observation of the same workload:
        `tp2_match --arm capture`'s per-rank recorder stored every collective's input and output per step in call order
        (`coll.rank<r>.jsonl` = (step, k, kind, module), `coll.rank<r>.step<s>.pt` = {part[k], out[k]}).  The oracle is a different
        process and a different run (batch-invariant execution promises bit-identity); a committed member the oracle does not name is
        `committed_not_in_oracle`, an oracle collective the committer did not bind is `missing_in_commit`, a step without a stored
        oracle is `steps_without_oracle` -- each of them FAILS (exhaustive over the family or not a record).  Covers the manifest's
        `tp_rank_partials` family only (this Match is collective-level; instance outputs have no TP Match oracle).

        Each MoE align call's committed witness (`<experts>/align.*`, `moe_align_source`) is paired with the capture's
        (`align.rank<r>.jsonl`) by (step, module), exhaustively; a differing one FAILS by name with what it implies
        (`oracle_witness.implied`), and no tensor is exempt from the value compare (`oracle_witness.judge`)."""
        import json
        from pathlib import Path

        import torch

        from verity_vllm.engine.collectives import fused_norm as FN

        t0 = time.perf_counter()
        cap = Path(capture_dir)
        res: dict[str, Any] = {"rank": self.rank, "oracle": str(cap), "steps_compared": 0, "steps_without_oracle": [], "oracle_steps_not_committed": [],
                               "duplicate_module_steps": [], "oracle_inconsistent": [], "per_tensor": [],
                               "missing_in_commit": [], "not_verified": [], "committed_not_in_oracle": [],
                               "witnesses": {"equal": 0, "differs": [], "unpaired": []},
                               "families_covered": ["tp_rank_partials"], "families_not_covered": ["instance_outputs", "fa2_hidden_m1_stream", "input_token_ids", "weights"],
                               # the pairing key is (step, MODULE): the committed member is named by its issuing module (one partial per
                               # site per step is what identity_check requires), the oracle record by (step, k, module); a module recorded twice in one
                               # step is refused (a second oracle collective would be compared against the same committed member)
                               "rule": ("committed bytes == tp2_match capture recorder's stored collective input (part_rank/shard_rank) / output (MoE /out), "
                                        "paired by (step, module) with k recorded, same rank; exhaustive in BOTH directions (every committed step has an oracle "
                                        "step and every oracle step a committed step); a module recorded twice in a step or a truncated .pt is a named FAIL; "
                                        "an MoE align witness that differs between the runs FAILS by name, nothing exempt (witness_rule)")}
        recp = cap / f"coll.rank{self.rank}.jsonl"
        if not recp.exists():
            res["verdict"], res["why"] = "FAIL", f"no oracle records {recp}"
            return res
        by_step: dict[int, list[dict]] = {}
        for l in recp.open():
            if l.strip():
                x = json.loads(l)
                by_step.setdefault(int(x["step"]), []).append(x)
        recorded: dict[int, dict[str, dict]] = {}
        if (wp := cap / OW.witness_file(self.rank)).exists():
            for l in wp.open():
                if l.strip():
                    x = json.loads(l)
                    w = recorded.setdefault(int(x["step"]), {}).setdefault(x["module"], {"k": int(x["k"]), "sha256": {}})
                    w["sha256"][x["member"]] = x["sha256"]
        received = set(self.received_site_names())
        # exhaustive in both directions: an oracle step the committer never reached (a Commit that stopped early) FAILs
        res["oracle_steps_not_committed"] = sorted(s for s in by_step if s not in self.c._layouts)
        mid_counts = {n: int(s["per_step"]) for n, s in self.mid_sites.items()}
        for step in sorted(self.c._layouts):
            by_name = _metas_by_name(self.c._layouts.get(step, []))
            fam = {n for n in by_name if n.endswith((f"/part_rank{self.rank}", f"/shard_rank{self.rank}")) or (n.endswith("/out") and n.rsplit("/", 1)[0] in received)}
            pt = cap / f"coll.rank{self.rank}.step{step}.pt"
            if not pt.exists() or step not in by_step:
                res["steps_without_oracle"].append(step)
                continue
            v = torch.load(pt)
            xs = sorted(by_step[step], key=lambda x: int(x["k"]))
            # a truncated / inconsistent oracle (the .pt holds fewer tensors than the jsonl names) is a named verdict, not a crash string
            n_part, n_out = len(v.get("part") or []), len(v.get("out") or [])
            if n_part < len(xs) or (any(x.get("module") in received for x in xs) and n_out < len(xs)):
                res["oracle_inconsistent"].append({"step": step, "jsonl_names": len(xs), "pt_part": n_part, "pt_out": n_out})
                continue
            # one oracle record per module per step, else refused for this step -- except a MID-module site (OlmoeAttention
            # gathers q then k), which is recorded exactly its static call count of times and paired by (step, module, occurrence j) with the
            # committer's j-th commit of the member (the binder's occurrence j); any other multiplicity is still refused
            # an all-reduce + norm site's partial is `<norm>/part_rank<r>` while the recorder names the collective's enclosing module (a
            # GLM decoder layer runs two: input_layernorm's and post_attention_layernorm's): such a module's records pair, in call order,
            # with the norm-site partials whose nearest recorded ancestor it is, in commit order, when the two counts agree
            mods = [x.get("module") for x in xs]
            under = self._norm_partials_under(self.c._layouts.get(step, []), set(mods))
            dups = sorted({m for m in mods if mods.count(m) > 1 and mods.count(m) != mid_counts.get(m)
                           and len(under.get(m, ())) != mods.count(m)})
            if dups:
                res["duplicate_module_steps"].append({"step": step, "modules": dups[:6]})
                continue
            seen: set[str] = set()
            occ_of: dict[str, int] = {}
            for k, x in enumerate(xs):
                mod = x.get("module")
                j = occ_of.get(mod, 0)
                occ_of[mod] = j + 1
                fused = x["kind"] == FN.KIND                   # its recorded part is stack([partial, residual])
                part_t = v["part"][k][0] if fused else v["part"][k]
                pname = f"{mod}/{'part' if fused or x['kind'] == 'all_reduce' else 'shard'}_rank{self.rank}"
                if pname not in by_name and len(under.get(mod, ())) == mods.count(mod):
                    pairs = [(under[mod][j].name, part_t, "partial", under[mod][j])]
                else:
                    pairs = [(pname, part_t, "partial", _nth(by_name, pname, j))]
                if mod in received:
                    pairs.append((f"{mod}/out", v["out"][k], "out", _nth(by_name, f"{mod}/out", j)))
                for member, oracle_t, _what, meta in pairs:
                    try:
                        committed = _committed(reader, step, meta)
                    except OpeningNotVerified as e:
                        seen.add(member)
                        res["not_verified"].append((step, k, member, str(e)[:200]))
                        continue
                    if committed is None:
                        res["missing_in_commit"].append((step, k, member))
                        continue
                    seen.add(member)
                    ob = _bytes(oracle_t)
                    n_diff = 0
                    if committed != ob:
                        n_diff = None
                        if len(committed) == len(ob):
                            a = torch.frombuffer(bytearray(committed), dtype=torch.uint8)
                            b = torch.frombuffer(bytearray(ob), dtype=torch.uint8)
                            n_diff = int((a != b).sum())
                    res["per_tensor"].append([step, k, member, x["kind"], n_diff, len(committed), len(ob)])
            for n in sorted(fam - seen):
                res["committed_not_in_oracle"].append((step, n))
            self._pair_witnesses(step, by_name, recorded.pop(step, {}), reader, res)
            res["steps_compared"] += 1
        for step, mods in sorted(recorded.items()):
            res["witnesses"]["unpaired"] += [{"step": step, "module": m, "in_commit": False, "in_capture": True} for m in sorted(mods)]
        wit = res["witnesses"]
        res["missing_n"], res["not_in_oracle_n"] = len(res["missing_in_commit"]), len(res["committed_not_in_oracle"])
        res["not_verified_n"], wit["unpaired_n"] = len(res["not_verified"]), len(wit["unpaired"])
        res["missing_in_commit"], res["committed_not_in_oracle"] = res["missing_in_commit"][:20], res["committed_not_in_oracle"][:20]
        res["not_verified"], wit["unpaired"] = res["not_verified"][:20], wit["unpaired"][:20]
        res["witness_counts"] = {"equal": wit["equal"], "differs": len(wit["differs"]), "unpaired": wit["unpaired_n"]}
        res["steps_committed"] = len(self.c._layouts)
        res["value_source"] = reader.record()
        res["wall_s"] = time.perf_counter() - t0
        return OW.judge(res)

    def _pair_witnesses(self, step: int, by_name: dict[str, list], recorded: dict[str, dict], reader: Any, res: dict[str, Any]) -> None:
        """This rank's committed MoE align witnesses of ``step`` against the capture's (``recorded``: module -> {k, sha256 per member}),
        by module: equal, differing (its members named, at the capture's position k, with what they imply) or unpaired (on one side only)."""
        wit = res["witnesses"]
        committed = {n.rsplit("/", 1)[0] for n in by_name if n.rsplit("/", 1)[-1] in OW.MEMBERS}
        for mod in sorted(committed | set(recorded)):
            rec = recorded.get(mod)
            if mod not in committed or rec is None:
                wit["unpaired"].append({"step": step, "module": mod, "in_commit": mod in committed, "in_capture": rec is not None})
                continue
            differs: list[str] = []
            for m in OW.MEMBERS:
                name = f"{mod}/{m}"
                try:
                    b = _committed(reader, step, _nth(by_name, name, 0))
                except OpeningNotVerified as e:
                    res["not_verified"].append((step, rec["k"], name, str(e)[:200]))
                    differs = []
                    break
                if b is None or m not in rec["sha256"]:
                    wit["unpaired"].append({"step": step, "module": mod, "member": m, "in_commit": b is not None, "in_capture": m in rec["sha256"]})
                    differs = []
                    break
                if OW.digest(b) != rec["sha256"][m]:
                    differs.append(m)
            else:
                if differs:
                    wit["differs"].append({"rank": self.rank, "step": step, "k": rec["k"], "module": mod, "members": differs, "implies": OW.implied(differs)})
                else:
                    wit["equal"] += 1

    # -- declaration -------------------------------------------------------------------------------------------------------
    def declaration_classes(self) -> list[dict]:
        return [{"class": "tp_rank_partials", "how": "bound-as-leaf", "covers_families": COVERS_FAMILIES, "rule": None, "count": None,
                 "bytes": self.stats["bytes"], "tensors": self.stats["launches"], "rank": self.rank, "world": self.world,
                 "sites": len(self.site_names()), "site_names": self.site_names(),
                 "received_site_names": self.received_site_names(), "nested_moe_sites_dropped": self.nested_moe_sites_dropped,
                 "mid_module_sites": {n: {"cls": s["cls"], "calls": s["calls"]} for n, s in self.mid_sites.items()},
                 "mid_module_sites_ruled_out": self.mid_sites_ruled_out,
                 "allreduce_norm": self.far.record(),
                 "evidence": ("TP partial source: wraps tensor_model_parallel_all_reduce (linear, vocab_parallel_embedding) and "
                              "tensor_model_parallel_all_gather/gather (logits_processor) in THIS rank's process, plus the same names in the "
                              "python module of every model class that calls one directly (mid-module sites: OlmoeAttention's q/k "
                              "gathers); the collective INPUT "
                              f"(rank {self.rank}'s partial GEMM / masked embedding / logits shard / mid-module operand) is snapshotted into the packed step "
                              "stream before the collective, attributed to the issuing module by a forward pre-hook; identity_check requires "
                              f"exactly the declared count of partials per site per step ({len(self.sites)} sites; 1 each, a mid-module site its "
                              "static call count); value correspondence: tp/match.py cross-rank "
                              "provenance (received == AllReduce2/AllGather2(part_rank0, part_rank1) bit-exact; part_rank_r == Gemm_v1{K/W} "
                              f"on the shard); kinds seen {self.stats['kinds']}; shapes seen {self.stats['shapes']}")}]


__all__ = ["TPPartialSource", "SITES", "COVERS_FAMILIES"]


def late_read_check(com: Any, rc: Any, rank: int) -> dict[str, Any]:
    """Late-read detector for the fused-MoE received all-reduce output: the committed bytes of
    `model.layers.<L>.mlp.experts/out` vs the committed `model.layers.<L+1>.input_layernorm/0` (last layer: `model.norm/0`) in the
    same step, for EVERY step and layer of the last finalized run.  The /out is staged (a held reference, copied at the next gather
    flush); vLLM's fused_add_rms_norm writes the next norm's output INTO that tensor in place, so a late read commits bytes equal
    to the next norm's /0.  Any equal pair = late read (result False, count named); result True needs > 0 pairs compared and 0
    equal / 0 missing."""
    import hashlib
    import re

    layouts = getattr(com, "_layouts", {}) or {}
    rd = OpenedReader(com, rc)
    pairs: list[dict] = []
    equal = missing = compared = not_verified = 0
    steps = 0
    for step in sorted(layouts):
        metas = {m.name: m for m in layouts[step]}
        outs = sorted((int(re.search(r"layers\.(\d+)\.", n).group(1)), n) for n in metas if n.endswith(".mlp.experts/out"))
        if not outs:
            continue
        steps += 1
        last = max(L for L, _n in outs)
        for L, n in outs:
            nxt = n.rsplit(".layers.", 1)[0] + (f".layers.{L + 1}.input_layernorm/0" if L < last else ".norm/0")
            try:
                a = _committed(rd, step, metas.get(n))
                b = _committed(rd, step, metas.get(nxt))
            except OpeningNotVerified as e:
                not_verified += 1
                if len(pairs) < 240:
                    pairs.append({"step": step, "layer": L, "out": n, "next_norm": nxt, "not_verified": str(e)[:200]})
                continue
            if a is None or b is None:
                missing += 1
                if len(pairs) < 240:
                    pairs.append({"step": step, "layer": L, "out": n, "next_norm": nxt, "missing": [x for x, v in (("out", a), ("next_norm", b)) if v is None]})
                continue
            compared += 1
            eq = a == b
            equal += int(eq)
            if eq or step < 2 or len(pairs) < 96:
                pairs.append({"step": step, "layer": L, "out": n, "next_norm": nxt, "out_sha": hashlib.sha256(a).hexdigest()[:16],
                              "next_norm_sha": hashlib.sha256(b).hexdigest()[:16], "nbytes": len(a), "equal": eq})
    # a dense TP workload (Llama: no fused-MoE `/out` member in any step) has nothing for this detector to compare -- record NOT
    # APPLICABLE (result None, applicable False) rather than a False that reads as a detected late read; on a MoE workload the rule is
    # the MoE one: True only with > 0 pairs compared, 0 equal, 0 missing
    applicable = steps > 0
    return {"schema": "verity/commit-integ/t6_4-check/v1", "rank": int(rank), "applicable": applicable,
            "result": (bool(compared > 0 and equal == 0 and missing == 0 and not_verified == 0) if applicable else None),
            "steps": steps, "layers_compared": compared, "equal_pairs": equal, "missing_pairs": missing, "not_verified_pairs": not_verified,
            "value_source": rd.record(), "pairs": pairs[:240],
            "why": (None if applicable else "no fused-MoE received all-reduce member (*.mlp.experts/out) in any committed step: dense TP row, detector not applicable"),
            "rule": "committed <L>.mlp.experts/out == committed <L+1>.input_layernorm/0 (last: model.norm/0) in the same step => late read"}
