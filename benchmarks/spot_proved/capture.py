"""A vLLM general plugin that records one replayed step's interior, per TP rank, in an engine that serves with CUDA graphs.

    python capture.py site SITE --capture DIR --steps s[,s2] [--open dense:1,moe:1] [--mla-tokens 8] [--keep all [--keep-steps s,...]] [--sampler]
    PYTHONPATH=SITE <a vLLM engine>

Registered as a ``vllm.general_plugins`` entry point (`site` writes the dist-info, and the capture's settings into
``SITE/spot_capture.json``, which the plugin reads beside itself), so it loads in every worker. It runs on
the V2 model runner, vLLM's default at the integration's pin. A step is counted when a worker's ``execute_model`` has
scheduled tokens and is not a dummy run. At step s (0-based), when the step dispatches to a CUDA graph (FULL, PIECEWISE or
breakable), the runner's graph call is preceded by one more forward of the same model on the same input buffers, attention
metadata, slot mappings and padded size, with the forward context's graph mode set to NONE; that forward is the one
recorded. A step the engine runs without a graph is recorded as it runs. What is recorded:

* every dispatched op (`Recorder`, a TorchDispatchMode): over the whole forward when the model is not compiled (the pin's
  default for GLM-5 and DeepSeek-V3.2, whose breakable CUDA graphs turn compilation off), or inside each
  ``PiecewiseBackend.__call__`` when it is, where values that pass only between two Inductor Triton kernels are not
  recorded. Inside vLLM's Python custom ops (``vllm::``) the ops they dispatch are recorded too, each naming its
  ``parent``. Each tensor argument is recorded before the call, and each return and written argument after it. A parameter
  is named, not copied. A tensor above ``BIG`` bytes (a KV cache) is recorded by shape only. An op whose ``Tensor[]``
  return can hold an undefined tensor (`OUTSIDE_MODE`: FA3's ``fwd``) is recorded the same way by a wrapper that runs it
  outside the mode, which couldn't hand such a list back;
* calls no dispatcher sees: each Triton fused-MoE launch (``invoke_fused_moe_triton_kernel``: A, its scales, the expert
  ids, the sorted token ids, the routed weights, the config, and C after the launch), and each sparse MLA kernel call
  (FlashMLA's ``_bf16_flash_mla_kernel``, or FA3's ``flash_attn_varlen_func`` in the FlashAttention sparse backend: the
  queries, the global top-k rows, the outputs, and the KV rows those indices read for ``--mla-tokens`` tokens: FlashMLA's
  first ones, FA3's with the most keys (``seqused_k``), the ones with the most splits, named in ``kv_gathered_rows``), and
  each other kernel `PY_KERNELS` lists, launched from Python (GLM-5.3's fused q/kv norm and RoPE with its cache writes, its
  fused query RoPE and quantization, the router's GEMM at up to 16 tokens, and the sparse backend's conversion of each
  token's top-k positions to global slots through its request's page table): its arguments by name, a paged cache by
  shape only, and after the call its returns and the arguments it writes.

Then the graph call runs as served, and its hidden states must equal the recorded forward's bit for bit (``check``). The
extra forward writes the same KV values to the same slots, so the step and every later one are unchanged.

Each recorded tensor is hashed when recorded; its bytes are held in memory until the coins have opened layers, and only the
opened layers' tensors are written, once each, under ``DIR/rank<r>/blobs/<sha512>`` (``--keep all`` writes every one up
to ``--keep-max-mib`` as it is recorded, at every step or at the ``--keep-steps``, until the disk would fall below
``--min-free-gib``; the rest are written only in the opened layers). ``DIR/rank<r>/step<s>/ops.jsonl`` lists
the events in order. ``step<s>/meta.json`` holds the step (tokens, padding, graph mode, the scheduled requests and their
token counts, the batch's request ids in its order), the check, and the rank's interior root for the step: SHA-512 over the SHA-512s of ``ops.jsonl``, of
``params.json``, which gives each parameter a recorded op read (its shape, dtype and SHA-512 over its bytes), and of the
batch (``batch_sha512``: the request ids in order and their scheduled tokens). Every
recorded event names its blobs by SHA-512 and its decoder layer, so the root binds every recorded byte. With
``--allreduce-tree``, ``step<s>/allreduce_tree.json`` beside it holds the node's multimem reduction tree and where it was
read, for the node's registration to commit and the match gate to bind. Then
`_commit_and_open`: the step's root over the ranks' roots, the coins rank 0 samples only after that, and the layers they
open, whose parameters each rank writes beside the step (``commit.json``, ``open/``).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

BIG = 1 << 30
#: with ``--keep all``, every parameter the step read up to this size is written too (``commit.json``'s ``kept_params``: the
#: norms' and biases' weights of the layers the coins didn't open, which a gate over every call reads)
SMALL_PARAM = 1 << 20
#: aten ops that only allocate, view or restride: their outputs alias or are uninitialized, never a value of the step
SKIP = {"empty", "empty_strided", "empty_like", "as_strided", "view", "_unsafe_view", "reshape", "alias", "detach", "t",
        "permute", "transpose", "expand", "unsqueeze", "squeeze", "select", "slice", "_reinterpret_tensor", "lift_fresh",
        "split", "split_with_sizes", "unbind", "chunk", "narrow", "view_as", "resize_", "set_", "record_stream"}
#: the paged KV caches an op takes whole: hashed into the root but never written, since the rows a call reads are the sparse
#: MLA tap's ``kv_gathered`` and no gate reads the rest
CACHES = {"_vllm_fa3_C::fwd": ("k", "v"), "_C_cache_ops::cp_gather_indexer_k_quant_cache": ("kv_cache",)}


#: the capture's settings, which `site` writes beside its copy of this file
CONFIG = "spot_capture.json"
#: the node's multimem reduction tree and where it was read, beside each step's ``params.json`` (stage.py's `TREE_FILE`)
TREE_FILE = "allreduce_tree.json"


def reduction_tree(text: str) -> list:
    """A binary reduction tree over ranks 0..W-1, as JSON pairs or as kernels' probe writes it, ``(((0,2),3),1)``."""
    tree = json.loads(text.replace("(", "[").replace(")", "]"))

    def leaves(x: Any) -> list[int]:
        if isinstance(x, int) and not isinstance(x, bool):
            return [x]
        if not (isinstance(x, list) and len(x) == 2):
            raise ValueError(f"allreduce tree {text}: a node is a rank or a pair of nodes, not {x!r}")
        return leaves(x[0]) + leaves(x[1])
    got = leaves(tree)
    if sorted(got) != list(range(len(got))):
        raise ValueError(f"allreduce tree {text}: its leaves must be the ranks 0..{len(got) - 1}, each once")
    return tree


def config() -> dict | None:
    """`site`'s settings, or None where none sit beside this file: the plugin then leaves the engine as it is."""
    p = Path(__file__).with_name(CONFIG)
    return json.loads(p.read_text()) if p.is_file() else None


class _State:
    def __init__(self, cfg: dict) -> None:
        self.dir = Path(cfg["capture"])
        self.targets = set(cfg["steps"])
        self.mla_tokens = int(cfg["mla_tokens"])
        self.sampler = bool(cfg.get("sampler"))
        self.allreduce_tree: dict | None = cfg.get("allreduce_tree")
        self.sample_at: int | None = None
        self.smp: dict | None = None
        self.open = dict(cfg["open"])
        self.step = -1
        self.active = False
        self.recording = False
        self.runner: Any = None
        self.sched: Any = None
        self.input_batch: Any = None
        self.attn: Any = None
        self.slot_maps: Any = None
        self.params: dict[tuple, str] = {}
        self.storages: dict[int, tuple[int, str]] = {}
        self.param_of: dict[str, Any] = {}
        self.used_params: set[str] = set()
        self.seen: set[str] = set()
        self.pending: dict[str, bytes] = {}
        self.keep = cfg["keep"]
        self.keep_steps = None if cfg.get("keep_steps") is None else set(cfg["keep_steps"])
        self.keep_max = int(cfg.get("keep_max_mib", 64)) << 20
        self.min_free = int(cfg.get("min_free_gib", 40)) << 30
        self.floor_hit = False
        self.events: list[dict] = []
        self.piece = -1
        self.parents: list[int] = []
        self.layer: int | None = None
        self.quiet = False
        self.whole = False
        self.check: dict | None = None
        self.t0 = time.time()
        self.rank = 0
        self.errors: list[str] = []
        self.bytes = 0

    def root(self) -> Path:
        return self.dir / f"rank{self.rank}"

    @property
    def keep_all(self) -> bool:
        """Whether this step writes every recorded tensor: ``--keep all``, at the ``--keep-steps`` (default every step)."""
        return self.keep == "all" and (self.keep_steps is None or self.step in self.keep_steps)

    def write_now(self, nbytes: int) -> bool:
        """Whether a blob is written as it is recorded: on a step kept whole, up to ``--keep-max-mib``, while the capture's
        disk keeps ``--min-free-gib`` free. Any other blob waits for the coins and is written only in an opened layer."""
        if not self.keep_all or nbytes > self.keep_max or self.floor_hit:
            return False
        import shutil
        self.root().mkdir(parents=True, exist_ok=True)
        self.floor_hit = shutil.disk_usage(self.root()).free - nbytes < self.min_free
        return not self.floor_hit


S: _State | None = None


def register() -> None:
    global S
    cfg = config()
    if cfg is None or S is not None:
        return
    S = _State(cfg)
    try:
        _install()
    except Exception:  # noqa: BLE001 -- a plugin that fails to install must say so and leave the engine as served
        print("[spot-capture] install failed:\n" + traceback.format_exc(), file=sys.stderr, flush=True)


def _note(what: str) -> None:
    print(f"[spot-capture] {what}", file=sys.stderr, flush=True)


def _wrap(owner: Any, name: str, make) -> None:
    orig = getattr(owner, name)
    setattr(owner, name, make(orig))


def _install() -> None:
    from vllm.compilation import piecewise_backend as pb
    from vllm.v1.worker.gpu import cudagraph_utils as cgu
    from vllm.v1.worker.gpu import model_runner as mr

    def execute_model(orig):
        def run(self, scheduler_output, *a, **kw):
            dummy = kw.get("dummy_run", a[1] if len(a) > 1 else False)
            reqs = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
            real = ((not dummy) and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0
                    and not any(str(r).startswith("_warmup") for r in reqs))
            if real:
                S.step += 1
                S.active = S.step in S.targets
                S.runner, S.sched = self, scheduler_output
                S.sample_at = S.step if S.active and S.sampler else None
                if S.active:
                    _bind_model_state(self)
            try:
                return orig(self, scheduler_output, *a, **kw)
            finally:
                if S.active and S.events:
                    try:
                        _finish(self)
                    except Exception:  # noqa: BLE001
                        print(f"[spot-capture] rank {S.rank} step {S.step}: the capture failed, serving goes on\n"
                              f"{traceback.format_exc()}", flush=True)
                        S.events, S.errors, S.check = [], [], None
                        S.pending.clear()
                        S.seen.clear()
                        S.floor_hit = False
                        S.used_params.clear()
                S.active = False
        return run

    def prepare_inputs(orig):
        def run(self, *a, **kw):
            out = orig(self, *a, **kw)
            if S.active:
                S.input_batch = out
            return out
        return run

    def slot_maps(orig):
        def run(*a, **kw):
            out = orig(*a, **kw)
            if S.active:
                S.slot_maps = out
            return out
        return run

    def run_fullgraph(orig):
        def run(self, batch_desc, *a, **kw):
            if not S.active:
                return orig(self, batch_desc, *a, **kw)
            mine = _eager(S.runner, None, "FULL", batch_desc)
            out = orig(self, batch_desc, *a, **kw)
            _check(mine, out, "FULL", batch_desc)
            return out
        return run

    def run_pw_graph(orig):
        def run(self, model, model_inputs, *a, **kw):
            if not S.active:
                return orig(self, model, model_inputs, *a, **kw)
            mine = _eager(S.runner, model_inputs, "PIECEWISE", None)
            out = orig(self, model, model_inputs, *a, **kw)
            _check(mine, out, "PIECEWISE", None)
            return out
        return run

    def pb_call(orig):
        def call(self, *args):
            if not S.recording or S.whole:
                return orig(self, *args)
            S.piece = int(getattr(self, "piecewise_compile_index", -1))
            with Recorder():
                return orig(self, *args)
        return call

    if S.sampler:
        _install_sampler()
    _wrap(mr.GPUModelRunner, "execute_model", execute_model)
    _wrap(mr.GPUModelRunner, "prepare_inputs", prepare_inputs)
    _wrap(mr, "build_slot_mappings_by_layer", slot_maps)
    manager = getattr(cgu, "ModelCudaGraphManager", cgu.CudaGraphManager)
    _wrap(manager, "run_fullgraph", run_fullgraph)
    _wrap(manager, "run_pw_graph", run_pw_graph)
    _wrap(pb.PiecewiseBackend, "__call__", pb_call)
    _install_moe()
    _install_mla()
    _install_py_kernels()
    _install_outside_mode()


def _install_sampler() -> None:
    """Rank 0's sampler at each recorded step (``--sampler``), read only: the logits it takes, the logits entering top-k/top-p
    (after penalties and temperature) and leaving it, the Gumbel sample's per-row seed, position and temperature, and the chosen
    tokens, under ``step<s>/sampler/`` with their SHA-512s in ``sampler.json``, outside the step's root."""
    import torch
    from vllm.v1.worker.gpu.sample import sampler as smp

    def host(t: Any) -> Any:
        return None if t is None else t.detach().to("cpu", copy=True)

    def call(orig):
        def run(self, logits, input_batch, *a, **kw):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if S.sample_at is None or rank != 0:
                return orig(self, logits, input_batch, *a, **kw)
            S.smp = {"logits": host(logits)}
            out = None
            try:
                out = orig(self, logits, input_batch, *a, **kw)
                return out
            finally:
                rec, step, S.smp, S.sample_at = S.smp, S.sample_at, None, None
                try:
                    rec["chosen"] = host(getattr(out, "sampled_token_ids", None))
                    _write_sampler(step, rec, list(getattr(input_batch, "req_ids", None) or []))
                except Exception:  # noqa: BLE001
                    _note(f"step {step}: the sampler record failed, serving goes on\n{traceback.format_exc()}")
        return run

    def top_k_top_p(orig):
        def run(logits, top_k, top_p, *a, **kw):
            if S.smp is None:
                return orig(logits, top_k, top_p, *a, **kw)
            S.smp.update(topp_in=host(logits), top_k=host(top_k), top_p=host(top_p))
            out = orig(logits, top_k, top_p, *a, **kw)
            S.smp["topp_out"] = host(out)
            return out
        return run

    def gumbel(orig):
        def run(logits, expanded_idx_mapping, temperature, seed, pos, *a, **kw):
            out = orig(logits, expanded_idx_mapping, temperature, seed, pos, *a, **kw)
            if S.smp is not None:
                idx = expanded_idx_mapping.long()
                S.smp.update(row_state=host(idx), seed=host(seed[idx]), temperature=host(temperature[idx]), pos=host(pos),
                             gumbel_out=host(out), gumbel_args={k: v for k, v in kw.items() if isinstance(v, (bool, int, float))})
            return out
        return run

    _wrap(smp.Sampler, "__call__", call)
    _wrap(smp, "apply_top_k_top_p", top_k_top_p)
    _wrap(smp, "gumbel_sample", gumbel)


def _write_sampler(step: int, rec: dict, req_ids: list[str]) -> None:
    import torch
    d = S.dir / "rank0" / f"step{step}" / "sampler"
    d.mkdir(parents=True, exist_ok=True)
    if rec.get("topp_out") is not None:
        rec["topp_kept"] = rec["topp_out"] != float("-inf")
    doc: dict = {"step": step, "req_ids": req_ids, "gumbel_args": rec.pop("gumbel_args", None), "tensors": {},
                 "path": "gumbel" if "gumbel_out" in rec else "flashinfer (no per-request seed in the batch)"}
    for k, t in rec.items():
        if t is None:
            continue
        raw = t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        (d / f"{k}.bin").write_bytes(raw)
        doc["tensors"][k] = {"shape": list(t.shape), "dtype": str(t.dtype), "sha512": hashlib.sha512(raw).hexdigest()}
    (d / "sampler.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    _note(f"step {step}: the sampler's {len(doc['tensors'])} tensors over {len(req_ids)} requests -> {d}")


def _bind_model_state(runner: Any) -> None:
    ms = runner.model_state
    if getattr(ms, "_spot_bound", False):
        return
    orig = ms.prepare_attn

    def prepare_attn(*a, **kw):
        out = orig(*a, **kw)
        if S.active:
            S.attn = out
        return out
    ms.prepare_attn = prepare_attn
    ms._spot_bound = True
    model = runner.model
    import re
    for name, mod in model.named_modules():
        if m := re.search(r"(?:^|\.)layers\.(\d+)$", name):
            def pre(_mod, _args, n=int(m[1])):
                S.layer = n

            def post(_mod, _args, _out):
                S.layer = None
            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)
    fwd = model.forward

    def forward(*a, **kw):
        if not S.active or S.recording or _graph_mode() != "NONE":
            return fwd(*a, **kw)
        out = _recorded(runner, "NONE", lambda: fwd(*a, **kw))
        S.check = {"mode": "NONE", "recorded_forward": out is not None, "served_forward": True}
        return out if out is not None else fwd(*a, **kw)
    model.forward = forward


def _graph_mode() -> str:
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return "NONE"
    mode = get_forward_context().cudagraph_runtime_mode
    return getattr(mode, "name", str(mode))


def _install_moe() -> None:
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    try:
        from vllm.model_executor.layers.fused_moe.experts import triton_moe
    except ImportError as e:
        _note(f"no fused_moe.experts.triton_moe ({e}): only fused_moe's own launch is wrapped")
        triton_moe = None
    names = ("A", "B", "C", "A_scale", "B_scale", "topk_weights", "sorted_token_ids", "expert_ids", "num_tokens_post_padded",
             "mul_routed_weight", "top_k", "config")

    def expert(orig):
        def launch(*a, **kw):
            if not S.recording:
                return orig(*a, **kw)
            args = dict(zip(names, a))
            args.update({k: v for k, v in kw.items() if k in names or k == "block_shape"})
            ins = {k: _ref(v) for k, v in args.items() if k != "C"}
            out = orig(*a, **kw)
            _event("moe_launch", ins, {"C": _ref(args["C"])})
            return out
        return launch

    for mod in (fm, triton_moe):
        if mod is not None and hasattr(mod, "invoke_fused_moe_triton_kernel"):
            _wrap(mod, "invoke_fused_moe_triton_kernel", expert)


def _install_mla() -> None:
    try:
        from vllm.v1.attention.backends.mla import flashmla_sparse as fs
    except ImportError as e:
        _note(f"no sparse MLA backend ({e}): no sparse MLA kernel call is recorded")
        return
    import torch

    def kernel(orig):
        def run(self, q, kv_rows, topk_indices, *a, **kw):
            if not S.recording:
                return orig(self, q, kv_rows, topk_indices, *a, **kw)
            rest = dict(zip(("topk_length", "actual_num_heads"), a))
            rest.update(kw)
            ins = {"q": _ref(q), "kv_rows": _ref(kv_rows), "topk_indices": _ref(topk_indices), **{k: _ref(v) for k, v in rest.items()}}
            n = min(S.mla_tokens, int(topk_indices.shape[0]))
            if n:
                with _quiet():
                    idx = topk_indices[:n].long().clamp(min=0)
                    rows = kv_rows.reshape(-1, kv_rows.shape[-1])[idx.reshape(-1)].reshape(n, idx.shape[1], -1)
                ins["kv_gathered"] = _ref(rows)
                ins["kv_gathered_tokens"] = n
            out = orig(self, q, kv_rows, topk_indices, *a, **kw)
            outs = out if isinstance(out, tuple) else (out,)
            _event("sparse_mla_kernel", ins, {f"r{i}": _ref(t) for i, t in enumerate(outs) if isinstance(t, torch.Tensor)})
            return out
        return run

    if hasattr(fs.FlashMLASparseImpl, "_bf16_flash_mla_kernel"):
        _wrap(fs.FlashMLASparseImpl, "_bf16_flash_mla_kernel", kernel)
    try:
        from vllm.v1.attention.backends.mla import flashattn_mla_sparse as fa
    except ImportError as e:
        _note(f"no FlashAttention sparse MLA backend ({e}): only FlashMLA's kernel is recorded")
        return

    def varlen(orig):
        def run(*a, **kw):
            if not S.recording:
                return orig(*a, **kw)
            ins = {k: _ref(v) for k, v in kw.items() if k not in ("k", "v")}
            k, v, table = kw["k"], kw["v"], kw["block_table"]
            ins["k"] = {"shape": list(k.shape), "dtype": str(k.dtype), "kv_rows": True}
            n = min(S.mla_tokens, int(table.shape[0]))
            if n:
                with _quiet():
                    used = kw.get("seqused_k")
                    pick = (torch.sort(torch.argsort(used[:table.shape[0]].long(), descending=True, stable=True)[:n]).values
                            if used is not None else torch.arange(n, device=table.device))
                    idx = table[pick].long().clamp(min=0).reshape(-1)
                    rows = k.reshape(k.shape[0], -1), v.reshape(v.shape[0], -1)
                    gathered = torch.cat([rows[1][idx], rows[0][idx]], dim=-1).reshape(n, table.shape[1], -1)
                ins["kv_gathered"] = _ref(gathered)
                ins["kv_gathered_tokens"] = n
                ins["kv_gathered_rows"] = [int(x) for x in pick.tolist()]
            out = orig(*a, **kw)
            outs = out if isinstance(out, tuple) else (out,)
            _event("fa3_sparse_mla", ins, {f"r{i}": _ref(t) for i, t in enumerate(outs) if isinstance(t, torch.Tensor)})
            return out
        return run

    _wrap(fa, "flash_attn_varlen_func", varlen)


#: kernels the served forward launches from Python, which no dispatcher sees: the module its caller looks the name up in, the
#: name (also the event's op), its positional arguments' names, the arguments it writes, and the paged caches it writes
#: whole, recorded by shape only
PY_KERNELS = (
    ("vllm.models.deepseek_v32.attention", "fused_norm_rope",
     ("positions", "q_c", "q_rms_norm_w", "q_rms_eps", "kv_c", "kv_rms_norm_w", "kv_rms_eps", "k_pe", "k_rope_cos_sin_cache",
      "index_k", "index_k_layer_norm_w", "index_k_layer_norm_bias", "index_k_layer_norm_eps", "index_k_rope_cos_sin_cache",
      "topk_indices_buffer"),
     ("q_c_out", "kv_c_out", "k_pe_out", "index_k_out", "topk_indices_buffer"), ("mla_kv_cache", "indexer_k_cache")),
    ("vllm.models.deepseek_v32.attention", "fused_q",
     ("positions", "q_pe", "q_pe_cos_sin_cache", "index_q", "index_q_cos_sin_cache", "ql_nope", "q_scale", "index_weights",
      "index_weights_softmax_scale", "index_weights_head_scale"), (), ()),
    ("vllm.model_executor.kernels.linear.cute_dsl.ll_bf16", "ll_bf16_gemm", ("hidden_states", "router_weight", "output_dtype"),
     (), ()),
    ("vllm.v1.attention.backends.mla.flashattn_mla_sparse", "triton_convert_req_index_to_global_index",
     ("req_id", "block_table", "token_indices"), (), ()),
)


def _py_kernel(op: str, names: tuple[str, ...], written: tuple[str, ...], caches: tuple[str, ...]):
    def make(orig):
        def run(*a, **kw):
            if not S.recording:
                return orig(*a, **kw)
            args = {**dict(zip(names, a)), **kw}
            ins = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "kv_rows": True} if k in caches and v is not None
                   else _ref(v) for k, v in args.items()}
            out = orig(*a, **kw)
            outs = {f"r{i}": _ref(t) for i, t in enumerate(out if isinstance(out, tuple) else (out,))}
            outs.update({k: _ref(args[k]) for k in written if args.get(k) is not None})
            _event(op, ins, outs)
            return out
        return run
    return make


def _install_py_kernels() -> None:
    import importlib
    for mod, name, names, written, caches in PY_KERNELS:
        try:
            owner = importlib.import_module(mod)
        except ImportError as e:
            _note(f"no {mod} ({e}): {name} is not recorded")
            continue
        if hasattr(owner, name):
            _wrap(owner, name, _py_kernel(name, names, written, caches))


def _recorder_base():
    from torch.utils._python_dispatch import TorchDispatchMode
    return TorchDispatchMode


class Recorder(_recorder_base()):
    """Records each dispatched op, and inside vLLM's Python custom ops (``vllm::``) the ops they dispatch, with ``parent``."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        import torch
        kwargs = kwargs or {}
        packet = getattr(func, "_overloadpacket", None)
        base = getattr(packet, "__name__", str(func)).split(".")[-1]
        ns = getattr(func, "namespace", "aten")
        if S.quiet or (ns == "aten" and base in SKIP):
            return func(*args, **kwargs)

        def call(seq: int) -> Any:
            if ns == "vllm" and torch._C._dispatch_has_kernel_for_dispatch_key(func.name(), "CUDA"):
                S.parents.append(seq)
                try:
                    with self:
                        return func.redispatch(torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA), *args, **kwargs)
                finally:
                    S.parents.pop()
            return func(*args, **kwargs)
        return _record_op(f"{ns}::{base}", func._schema, args, kwargs, call)


def _record_op(op: str, schema: Any, args: tuple, kwargs: dict, call) -> Any:
    """One op's event: its arguments by schema name before ``call(seq)``, and its returns and written arguments after it."""
    flat = list(args) + [kwargs.get(a.name) for a in schema.arguments[len(args):]]
    cache = CACHES.get(op, ())
    ins = {a.name: _ref(v, write=a.name not in cache) for a, v in zip(schema.arguments, flat)}
    seq = _event(op, ins, {})
    out = call(seq)
    outs = {f"ret{i}": _ref(t) for i, t in enumerate(out if isinstance(out, (tuple, list)) else (out,))}
    for a, v in zip(schema.arguments, flat):
        if a.alias_info is not None and a.alias_info.is_write:
            outs[a.name] = _ref(v, write=a.name not in cache)
    S.events[seq]["out"] = outs
    return out


#: ops returning a ``Tensor[]`` that can hold an undefined tensor (FA3's split-KV accumulators when a call doesn't split), which
#: a Python dispatch mode can't hand back: it sees the undefined one as None, and C++ refuses a None in a ``Tensor[]``
#: ("Unable to cast Python instance of type <class 'list'>"). Each is recorded as `Recorder` would, by a wrapper on
#: ``torch.ops.<ns>.<name>`` that runs the op with the Python modes off.
OUTSIDE_MODE = (("_vllm_fa3_C", "fwd"),)


def _install_outside_mode() -> None:
    import torch
    from torch.utils._python_dispatch import _disable_current_modes, _get_current_dispatch_mode
    for ns, name in OUTSIDE_MODE:
        space = getattr(torch.ops, ns)
        try:
            packet = getattr(space, name)
        except AttributeError as e:
            _note(f"no {ns}::{name} ({e}): it is recorded under the mode")
            continue

        def run(*a, _packet=packet, _op=f"{ns}::{name}", **kw):
            if S.quiet or not isinstance(_get_current_dispatch_mode(), Recorder):
                return _packet(*a, **kw)

            def call(seq: int) -> Any:
                with _disable_current_modes():
                    return _packet(*a, **kw)
            return _record_op(_op, _packet.default._schema, a, kw, call)
        setattr(space, name, run)


def _params(runner: Any) -> None:
    if S.params:
        return
    for name, t in list(runner.model.named_parameters()) + list(runner.model.named_buffers()):
        if t.numel() == 0:
            continue
        S.params[(t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype))] = name
        S.param_of[name] = t
        lo = t.untyped_storage().data_ptr()
        S.storages.setdefault(lo, (lo + t.untyped_storage().nbytes(), name))


def _param_view(v: Any) -> dict | None:
    """A tensor that is a parameter, or a view of one's storage (a transposed or reshaped weight), named with its view."""
    key = (v.data_ptr(), tuple(v.shape), tuple(v.stride()), str(v.dtype))
    if key in S.params:
        S.used_params.add(S.params[key])
        return {"param": S.params[key], "shape": list(v.shape), "dtype": str(v.dtype)}
    if v.numel() == 0:
        return None
    hit = S.storages.get(v.untyped_storage().data_ptr())
    if hit is None:
        return None
    S.used_params.add(hit[1])
    base = S.param_of[hit[1]]
    return {"param": hit[1], "shape": list(v.shape), "dtype": str(v.dtype), "stride": list(v.stride()),
            "byte_offset": v.data_ptr() - base.data_ptr()}


class _quiet:
    """Tensor work of the capture's own (copies, gathers): not recorded."""

    def __enter__(self):
        self.was, S.quiet = S.quiet, True

    def __exit__(self, *exc):
        S.quiet = self.was


def _ref(v: Any, write: bool = True) -> Any:
    with _quiet():
        return _ref_quiet(v, write)


def _write_blob(h: str, raw: bytes) -> None:
    p = S.root() / "blobs" / h[:2] / h
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)


def _blobs(v: Any):
    if isinstance(v, dict):
        if isinstance(v.get("sha512"), str) and "shape" in v:
            yield v["sha512"]
        for x in v.values():
            yield from _blobs(x)
    elif isinstance(v, list):
        for x in v:
            yield from _blobs(x)


def _ref_quiet(v: Any, write: bool = True) -> Any:
    import torch
    if isinstance(v, torch.Tensor):
        if v.device.type == "cuda" and (pv := _param_view(v)) is not None:
            return pv
        nb = v.numel() * v.element_size()
        d = {"shape": list(v.shape), "dtype": str(v.dtype), "stride": list(v.stride())}
        if v.device.type != "cuda" and v.device.type != "cpu":
            return {**d, "device": v.device.type}
        if nb > BIG:
            return {**d, "big": nb}
        raw = v.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes() if nb else b""
        h = hashlib.sha512(raw).hexdigest()
        if write and h not in S.seen:
            S.seen.add(h)
            S.bytes += len(raw)
            if S.write_now(len(raw)):
                _write_blob(h, raw)
            else:
                S.pending[h] = raw
        return {**d, "sha512": h}
    if isinstance(v, (list, tuple)):
        return [_ref(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _ref(x) for k, x in v.items()}
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    return {"repr": repr(v)[:200]}


def _event(kind: str, ins: dict, outs: dict) -> int:
    seq = len(S.events)
    S.events.append({"seq": seq, "piece": S.piece, "layer": S.layer, "parent": S.parents[-1] if S.parents else None, "op": kind,
                     "in": ins, "out": outs})
    return seq


def _recorded(runner: Any, mode: str, forward) -> Any:
    """Runs ``forward()`` with recording on: under a `Recorder` for the whole forward unless the model is compiled (then
    `PiecewiseBackend` enters one per piece). Returns the hidden states, or None if the forward raised."""
    import contextlib
    import torch
    S.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    _params(runner)
    S.t0 = time.time()
    S.mode = mode
    cmode = runner.vllm_config.compilation_config.mode
    S.whole = getattr(cmode, "name", str(cmode)) != "VLLM_COMPILE"
    S.recording = True
    try:
        with Recorder() if S.whole else contextlib.nullcontext():
            out = forward()
        hidden = out[0] if isinstance(out, tuple) else out
        S.piece = -1
        _event("model_output", {}, {"hidden": _ref(hidden)})
        return out
    except Exception:  # noqa: BLE001
        S.errors.append(traceback.format_exc()[-4000:])
        return None
    finally:
        S.recording = False


def _eager(runner: Any, model_inputs: dict | None, mode: str, batch_desc: Any) -> Any:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
    ib = S.input_batch

    def forward():
        if model_inputs is None:
            inputs = {"input_ids": ib.input_ids, "positions": ib.positions, "inputs_embeds": None, "intermediate_tensors": None,
                      **runner.model_state.prepare_inputs(ib, runner.req_states)}
            with set_forward_context(S.attn, runner.vllm_config, num_tokens=ib.num_tokens_after_padding,
                                     cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                     batch_descriptor=BatchDescriptor(num_tokens=ib.num_tokens_after_padding,
                                                                      has_lora=runner.lora_config is not None,
                                                                      num_active_loras=getattr(batch_desc, "num_active_loras", 0)),
                                     slot_mapping=S.slot_maps, is_padding=getattr(ib, "is_padding", None)):
                return runner.model(**inputs)
        fc = get_forward_context()
        was = fc.cudagraph_runtime_mode
        fc.cudagraph_runtime_mode = CUDAGraphMode.NONE
        try:
            return runner.model(**model_inputs)
        finally:
            fc.cudagraph_runtime_mode = was

    out = _recorded(runner, mode, forward)
    if out is None:
        return None
    return (out[0] if isinstance(out, tuple) else out).clone()


def _check(mine: Any, out: Any, mode: str, batch_desc: Any) -> None:
    import torch
    got = out[0] if isinstance(out, tuple) else out
    ib = S.input_batch
    n = int(getattr(ib, "num_tokens", got.shape[0] if got is not None else 0))
    S.check = {"mode": mode, "tokens": n, "padded": int(getattr(ib, "num_tokens_after_padding", 0)),
               "recorded_forward": mine is not None}
    if mine is not None and isinstance(got, torch.Tensor):
        S.check.update(equal_tokens=bool(torch.equal(mine[:n], got[:n])), equal_padded=bool(torch.equal(mine, got)),
                       differ_words=int((mine[:n].view(torch.int16) != got[:n].view(torch.int16)).sum().item())
                       if mine.dtype == torch.bfloat16 else None)
        if not S.check["equal_tokens"] and mine.shape == got.shape:
            rows = (mine[:n] != got[:n]).reshape(n, -1).any(dim=1).nonzero().reshape(-1).tolist()
            S.check["differ_rows"] = [int(r) for r in rows]


def _batch_sha512(req_ids: list[str], scheduled: dict[str, int]) -> str:
    """SHA-512 over the step's batch, which the rank root commits: its request ids in batch order and each request's scheduled
    tokens (`stage.batch_sha512` recomputes it)."""
    doc = json.dumps({"req_ids": req_ids, "scheduled": scheduled}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha512(f"verity/spot/batch\0{doc}".encode()).hexdigest()


def _finish(runner: Any) -> None:
    import torch
    d = S.root() / f"step{S.step}"
    d.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in S.events).encode()
    (d / "ops.jsonl").write_bytes(lines)
    sched = S.sched
    reqs = {str(k): int(v) for k, v in getattr(sched, "num_scheduled_tokens", {}).items()}
    from concurrent.futures import ThreadPoolExecutor

    def digest(name: str) -> tuple[str, dict]:
        t = S.param_of[name]
        raw = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        return name, {"shape": list(t.shape), "dtype": str(t.dtype), "sha512": hashlib.sha512(memoryview(raw)).hexdigest()}
    with ThreadPoolExecutor(8) as pool:
        params = dict(pool.map(digest, sorted(S.used_params)))
    (d / "params.json").write_text(json.dumps(params, indent=0, sort_keys=True))
    if S.allreduce_tree is not None:
        (d / TREE_FILE).write_text(json.dumps(S.allreduce_tree, sort_keys=True))
    ops_sha, params_sha = hashlib.sha512(lines).hexdigest(), hashlib.sha512((d / "params.json").read_bytes()).hexdigest()
    ids = [str(x) for x in getattr(S.input_batch, "req_ids", None) or []]
    batch_sha = _batch_sha512(ids, reqs)
    meta = {"rank": S.rank, "step": S.step, "check": S.check, "scheduled": reqs, "req_ids": ids, "events": len(S.events),
            "blob_bytes": S.bytes, "errors": S.errors, "seconds": round(time.time() - S.t0, 1),
            "ops_sha512": ops_sha, "params_sha512": params_sha, "batch_sha512": batch_sha,
            "rank_root": hashlib.sha512(f"verity/spot/rank-root\0{ops_sha}\0{params_sha}\0{batch_sha}".encode()).hexdigest()}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    commit = _commit_and_open(runner, d, meta["rank_root"])
    print(f"[spot-capture] rank {S.rank} step {S.step}: {len(S.events)} events, {S.bytes >> 20} MiB, check {meta['check']}, "
          f"errors {len(S.errors)}, root {commit.get('root', '')[:16]}, opened layers {commit.get('layers')}", flush=True)
    S.events, S.errors, S.check = [], [], None


def _commit_and_open(runner: Any, d: Path, rank_root: str) -> dict:
    """The step's interior root over every rank's, then the coins, flipped by rank 0 only after every rank's root is fixed, and
    the layers they open: ``--open`` (default ``dense:1,moe:1``) layers of each kind, whose parameters each rank
    writes under ``open/`` (the bytes its sampled proof blocks read). Written to ``commit.json``."""
    import torch
    import torch.distributed as dist
    group = None
    try:
        from vllm.distributed.parallel_state import get_world_group
        group = get_world_group().cpu_group
    except Exception as e:  # noqa: BLE001
        _note(f"no vLLM world group ({e!r}): the roots and coins go over torch.distributed's default group")
        group = None
    world = dist.get_world_size() if dist.is_initialized() else 1
    roots = [None] * world
    if world > 1:
        dist.all_gather_object(roots, rank_root, group=group)
    else:
        roots = [rank_root]
    root = hashlib.sha512(("verity/spot/step-root\0" + "\0".join(roots)).encode()).hexdigest()
    t_root = time.time()
    coins = [os.urandom(64).hex() if S.rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(coins, src=dist.get_global_rank(group, 0) if group is not None else 0, group=group)
    coins_hex = coins[0]
    hf = getattr(runner.model_config, "hf_text_config", None) or runner.model_config.hf_config
    n_layers = int(getattr(hf, "num_hidden_layers"))
    dense_k = min(int(getattr(hf, "first_k_dense_replace", 0) or 0), n_layers)
    pools = {"dense": list(range(dense_k)), "moe": list(range(dense_k, n_layers))}
    want = S.open
    layers: dict[str, list[int]] = {}
    for kind, n in want.items():
        pool = list(pools.get(kind, []))
        layers[kind] = []
        for i in range(min(int(n), len(pool))):
            j = int.from_bytes(hashlib.sha512(bytes.fromhex(coins_hex) + f"open/{kind}/{i}".encode()).digest(), "big") % len(pool)
            layers[kind].append(pool.pop(j))
    opened = sorted(x for v in layers.values() for x in v)
    marks = tuple(f"layers.{x}." for x in opened)
    kept = {h for e in S.events if e.get("layer") in opened for h in _blobs([e.get("in"), e.get("out")])}
    for h in sorted(kept):
        if h in S.pending:
            _write_blob(h, S.pending[h])
    kept_bytes = sum(len(S.pending[h]) for h in kept if h in S.pending)
    withheld = [len(raw) for h, raw in S.pending.items() if h not in kept]
    S.pending.clear()
    S.seen.clear()
    od = d / "open"
    od.mkdir(exist_ok=True)
    files = {}
    for name, t in sorted(S.param_of.items()):
        if not any(m in f".{name}" for m in marks):
            continue
        raw = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        (od / f"{name}.bin").write_bytes(memoryview(raw))
        files[name] = {"shape": list(t.shape), "dtype": str(t.dtype), "stride": list(t.stride()),
                       "sha512": hashlib.sha512(memoryview(raw)).hexdigest()}
    small = {}
    if S.keep_all:
        for name in sorted(S.used_params - set(files)):
            t = S.param_of[name]
            if t.numel() * t.element_size() > SMALL_PARAM:
                continue
            raw = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            (od / f"{name}.bin").write_bytes(memoryview(raw))
            small[name] = {"shape": list(t.shape), "dtype": str(t.dtype), "stride": list(t.stride()),
                           "sha512": hashlib.sha512(memoryview(raw)).hexdigest()}
    commit = {"rank": S.rank, "step": S.step, "rank_roots": roots, "root": root, "t_root": t_root, "coins": coins_hex,
              "t_coins_received": time.time(), "coins_by": "rank 0's os.urandom(64), drawn after every rank's root was gathered",
              "layers": layers, "opened": files, "kept_params": small,
              "kept": ("every recorded tensor" if S.keep_all and not withheld else
                       f"every recorded tensor but {len(withheld)} outside the opened layers, {sum(withheld)} bytes "
                       f"({'past the disk floor' if S.floor_hit else 'each over the keep cap'})" if S.keep_all else
                       f"the opened layers' {len(kept)} tensors, {kept_bytes} bytes"),
              # every layer's calls can be gated: a withheld tensor is one over the keep cap, never one lost to the floor
              "whole": S.keep_all and not S.floor_hit, "keep_max": S.keep_max if S.keep_all else None}
    S.floor_hit = False
    S.used_params.clear()
    (d / "commit.json").write_text(json.dumps(commit, indent=1, sort_keys=True))
    return commit


def site(d: Path, cfg: dict) -> Path:
    """A directory that, on PYTHONPATH, registers this file as the ``vllm.general_plugins`` entry ``verity_spot_capture``,
    with ``cfg`` (`main`'s settings) beside it as `CONFIG`."""
    import shutil
    info = d / "verity_spot_capture-0.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-spot-capture\nVersion: 0.0\n")
    (info / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_spot_capture = spot_capture:register\n")
    shutil.copyfile(__file__, d / "spot_capture.py")
    (d / CONFIG).write_text(json.dumps(cfg, indent=1, sort_keys=True))
    return d


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="capture.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("site", help="write the plugin's site directory, its settings beside it")
    s.add_argument("dir", type=Path)
    s.add_argument("--capture", required=True, help="where each rank's capture goes")
    s.add_argument("--steps", required=True, help="the 0-based steps to record, comma-separated")
    s.add_argument("--open", default="dense:1,moe:1", help="how many layers of each kind the coins open, KIND:N,...")
    s.add_argument("--mla-tokens", type=int, default=8, help="how many of a sparse MLA call's tokens have their gathered KV rows recorded (FA3: those with the most keys)")
    s.add_argument("--keep", choices=("opened", "all"), default="opened", help="all: write every recorded tensor")
    s.add_argument("--keep-steps", help="with --keep all, the steps that write every tensor (default: every step); the "
                   "others write the opened layers'")
    s.add_argument("--keep-max-mib", type=int, default=64,
                   help="with --keep all, a larger tensor is written only in an opened layer (hashed into the root either way)")
    s.add_argument("--min-free-gib", type=int, default=40,
                   help="with --keep all, a step stops writing outside the opened layers once its disk would fall below this")
    s.add_argument("--sampler", action="store_true", help="also record rank 0's sampler at each step (outside the step's root)")
    s.add_argument("--allreduce-tree", metavar="TREE",
                   help=f"the node's multimem reduction tree as its probe read it, written at each step as {TREE_FILE}")
    s.add_argument("--allreduce-tree-source", default="", help="where TREE was read (the probe's record)")
    a = ap.parse_args(argv)
    cfg = {"capture": str(Path(a.capture).resolve()), "steps": sorted({int(x) for x in a.steps.split(",") if x.strip()}),
           "open": {k: int(n) for k, n in (x.split(":") for x in a.open.split(",") if x)}, "mla_tokens": a.mla_tokens,
           "keep": a.keep, "keep_steps": None if a.keep_steps is None else sorted({int(x) for x in a.keep_steps.split(",") if x.strip()}),
           "keep_max_mib": a.keep_max_mib, "min_free_gib": a.min_free_gib, "sampler": a.sampler,
           "allreduce_tree": {"tree": reduction_tree(a.allreduce_tree), "source": a.allreduce_tree_source}
           if a.allreduce_tree else None}
    print(site(a.dir, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
