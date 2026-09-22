"""``CompiledCapture``: turn the record sink's per-step buffers into the capture format (RUs, leaves,
boundaries, weights, description) - rung (b) of CONTRACT §16.1.

Granularity is the compiled graph's: one node per kernel launch of the compiled piece (GEMM, fused
add+rmsnorm, silu*mul, rope, embedding+rmsnorm), the attention op (outside the graph, its output is a
piece input), ``lm_head.gemm`` and ``sampler.select`` (eager, hooked around ``compute_logits`` /
``sample``). Node names follow the eager adapter (``layers.L.self_attn.qkv_proj.gemm`` ...) so the
verifier's family chains and capture predicates apply; the description's ``interception`` field and
every node's ``relation``/``runtime_impl`` say what was actually run.

Duplicates: a piece's ``hidden_states`` input is the previous piece's residual output (same buffer).
It is recorded (so rung (a) can bit-compare it) but never written as a leaf; the comparison count is
reported (``residual_dup``).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verity_vllm.capture import Boundary, RunWriter, StepRecord
from verity_vllm.identity import value_id
from verity_vllm_adapter.capture import layer_of, node_sort_key, normalize_module
from verity_vllm_adapter.compiled_fx import kernels as K
from verity_vllm_adapter.compiled_fx import program as P
from verity_vllm_adapter.compiled_fx.fx_pass import ID_STRIDE, FxIdTable

INTERCEPTION = "compiled-fx-record-pass"
RELATIONS_MODULE = "verity_vllm_adapter.compiled_fx.relations"
PATCHES = [
    "replay process only: vllm.compilation.passes.pass_manager.PostGradPassManager -> RecordingPassManager (runs verity RecordInsertionPass after fix_functionalization; uuid extended)",
    "replay process only: PiecewiseCompileInterpreter.call_module / PiecewiseBackend.__call__ wrapped to publish the piece index",
    "replay process only: model.compute_logits and runner.sample / execute_model wrapped for lm_head / sampler rows and RU bookkeeping",
    "production: none",
]


class CompiledCapture:
    def __init__(self, *, runner: Any, model: torch.nn.Module, profile_id: str, run_dir: Path, writer: RunWriter, table: FxIdTable,
                 prompt_lens: dict[str, int], log: Any = print, cache_dir: Path | None = None, shard: int | None = None, world_size: int = 1) -> None:
        self.runner = runner
        self.model = model
        self.profile_id = profile_id
        self.run_dir = Path(run_dir)
        self.writer = writer
        self.table = table
        self.prompt_lens = prompt_lens
        self.log = log
        # [COMP, CX1] tensor parallelism: this capture runs inside rank `shard`'s worker (compiled_fx/tp_worker.py); every value and
        # weight id carries s=<shard> (capture-format field s, as the eager TP capture), tp_merge joins the ranks
        self.shard = shard
        self.world_size = int(world_size)
        import os

        self.cache_dir = Path(cache_dir or os.environ["TORCHINDUCTOR_CACHE_DIR"])
        self.programs: dict[int, P.Program] = {}
        self.templates: dict[int, list[P.NodeSpec]] = {}
        self.port_of: dict[int, tuple[str, int]] = {}  # template record id -> (template op, port)
        self.inputs_info: dict[int, dict[str, Any]] = {}
        self.nodes: dict[str, dict[str, Any]] = {}  # instantiated description nodes by op
        self.node_phases: dict[str, set[str]] = defaultdict(set)
        self.node_instances: dict[str, int] = defaultdict(int)
        self.req_records: dict[str, list[StepRecord]] = defaultdict(list)
        self.req_steps: dict[str, int] = defaultdict(int)
        self.req_outputs: dict[str, list[int]] = defaultdict(list)
        self.leaf_count = 0
        self.bytes_captured = 0
        self.ru_count = 0
        self.residual_dup = {"compared": 0, "equal": 0, "unequal": 0}
        self.errors: list[dict[str, Any]] = []
        self.timing = {"emit_s": 0.0}
        self.cur_logits: torch.Tensor | None = None
        self.cur_logits_in: torch.Tensor | None = None
        self.attn_consts_cache: dict[str, dict[str, Any]] = {}
        self.ids_unmapped: dict[int, int] = defaultdict(int)
        self.non_token_rows: dict[int, int] = defaultdict(int)
        self.lm_head_param = self._lm_head_param()
        self.params = self._params()

    # ---------------------------------------------------------------- setup
    def begin(self) -> None:
        self.programs = P.load_programs(self.cache_dir)
        for piece, prog in sorted(self.programs.items()):
            info = self.table.pieces.get(piece)
            if info is None:
                self.errors.append({"where": "begin", "error": f"no table piece for program {piece}"})
                continue
            nodes, port_of, inputs_info = P.build_nodes(prog, self.table, info)
            self.templates[piece] = nodes
            self.port_of.update(port_of)
            self.inputs_info[piece] = inputs_info
        self.log(f"compiled capture: {len(self.programs)} template programs {sorted(self.programs)}; nodes per template "
                 f"{ {p: len(n) for p, n in self.templates.items()} }; ports mapped {len(self.port_of)}")
        # hooks for the eager tail (lm_head)
        cap = self
        orig_compute_logits = self.model.compute_logits

        def compute_logits(hidden_states, *a, **kw):
            out = orig_compute_logits(hidden_states, *a, **kw)
            try:
                cap.cur_logits_in = hidden_states.detach().clone()
                cap.cur_logits = out.detach().clone() if out is not None else None
            except Exception as e:  # noqa: BLE001
                cap.errors.append({"where": "compute_logits", "error": repr(e)})
            return out

        self.model.compute_logits = compute_logits

    def _lm_head_param(self) -> str:
        names = {p.data_ptr(): n for n, p in self.model.named_parameters()}
        lm = getattr(self.model, "lm_head", None)
        w = getattr(lm, "weight", None)
        if w is not None and w.data_ptr() in names:
            return normalize_module(names[w.data_ptr()])
        return "lm_head.weight"

    def _params(self) -> list[tuple[str, torch.Tensor]]:
        seen: set[int] = set()
        out = []
        for n, p in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            if p.data_ptr() in seen or p.numel() == 0:
                continue
            seen.add(p.data_ptr())
            out.append((normalize_module(n), p))
        return out

    # ---------------------------------------------------------------- identity
    def resolve(self, record_id: int, runtime_piece: int) -> tuple[str, int, int] | None:
        """template record id + runtime piece -> (op, port, shift) for the actual layer."""
        template = record_id // ID_STRIDE
        shift = runtime_piece - template
        hit = self.port_of.get(record_id)
        if hit is not None:
            return P.shift_layers(hit[0], shift), hit[1], shift
        info = self.inputs_info.get(template, {})
        ao = info.get("attention_output")
        if ao and ao["record_id"] == record_id:
            return P.shift_layers(f"{ao['module_path']}.attn", shift), 0, shift
        return None

    def residual_producer(self, runtime_piece: int) -> tuple[str, int]:
        if runtime_piece <= 1:
            # TP: the residual entering layer 0 is the all-reduced embedding (embed_tokens.all_reduce), not the rank's partial
            if self.shard is not None and any(p[0] == "embed_tokens.all_reduce" for p in self.port_of.values()):
                return "embed_tokens.all_reduce", 0
            return "embed_tokens.embedding", 0
        return f"layers.{runtime_piece - 1}.input_layernorm.rmsnorm", 1

    def ensure_node(self, spec: P.NodeSpec, shift: int, runtime_piece: int) -> dict[str, Any]:
        op = P.shift_layers(spec.op, shift)
        node = self.nodes.get(op)
        if node is not None:
            return node
        args = []
        for a in spec.args:
            a = dict(a)
            if a.get("source") == "node":
                if a["op"] == "<residual_in>":
                    a["op"], a["port"] = self.residual_producer(runtime_piece)
                else:
                    a["op"] = P.shift_layers(a["op"], shift)
            if a.get("source") == "weight":
                a["param"] = P.shift_layers(a["param"], shift)
                if a["param"].endswith("rotary_emb.cos_sin_cache"):
                    a["param"] = "layers.0.self_attn.rotary_emb.cos_sin_cache"  # one shared buffer (eager convention)
            args.append(a)
        p0 = spec.ports[0]
        node = {
            "op": op, "layer": layer_of(P.shift_layers(spec.module_path, shift)), "op_type": spec.op_type, "dtype": p0.dtype, "shape": list(p0.row_shape),
            "ports": len(spec.ports), "outputs": [{"port": p.port, "dtype": p.dtype, "shape": list(p.row_shape)} for p in spec.ports if p.port != 0],
            "phases": [], "positions": "all", "args": args, "relation": spec.relation, "origin": "captured",
            "decompositions": [{"id": "row", "interface": {"input_bytes": 0, "output_bits": int(math.prod(p0.row_shape)) * _dtype_bits(p0.dtype), "range": "row"}}],
            "runtime_impl": spec.runtime_impl, "family": spec.family, "runtime_op": f"inductor:{spec.kernel_name}", "consts": dict(spec.consts),
            "arg_cols": dict(spec.arg_cols), "unbound_args": [l["arg"] for l in spec.launch if l.get("unbound")],
            "module_path": P.shift_layers(spec.module_path, shift), "instances_observed": 0, "arg_violations": [],
            "compiled": {"template_piece": spec_piece(spec, self), "kernel_var": spec.kernel_var, "launch_line": spec.line},
        }
        if node["unbound_args"]:
            node["relation"] = "unresolved"
        self.nodes[op] = node
        return node

    def ensure_attn_node(self, op: str, module_path: str, row_shape: list[int], dtype: str, layer: str) -> dict[str, Any]:
        node = self.nodes.get(op)
        if node is not None:
            return node
        consts = self._attn_consts(module_path)
        if "num_heads" not in consts:
            self.errors.append({"where": "ensure_attn_node", "op": op, "error": f"no attention module found for {module_path!r}: {consts}"})
        H, KVH = int(consts.get("num_heads", 0) or 0), int(consts.get("num_kv_heads", 0) or 0)
        D = int(consts.get("head_size", 0) or 0)
        src = self._qkv_sources(int(layer), H, KVH, D)
        if src is None:  # B0 layout: q and k are the two ports of one rope kernel, v a column range of the qkv GEMM output
            src = {"q": (f"layers.{layer}.self_attn.rotary_emb.rope", 0), "k": (f"layers.{layer}.self_attn.rotary_emb.rope", 1),
                   "v": (f"layers.{layer}.self_attn.qkv_proj.gemm", 0)}
        node = {
            "op": op, "layer": layer, "op_type": "attention", "dtype": dtype, "shape": list(row_shape), "ports": 1, "outputs": [], "phases": [], "positions": "all",
            "args": [{"name": "q", "source": "node", "op": src["q"][0], "port": src["q"][1]}, {"name": "k", "source": "node", "op": src["k"][0], "port": src["k"][1], "t": "all_le"},
                     {"name": "v", "source": "node", "op": src["v"][0], "port": src["v"][1], "t": "all_le"}],
            "relation": "attention.same_runtime.v1", "origin": "captured",
            "decompositions": [{"id": "row", "interface": {"input_bytes": 0, "output_bits": int(math.prod(row_shape)) * _dtype_bits(dtype), "range": "row"}}],
            "runtime_impl": "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl (vllm::unified_attention_with_output, outside the compiled pieces; identical in production and replay)",
            "family": "attn", "runtime_op": "vllm.unified_attention_with_output", "consts": consts, "arg_cols": {"v": [(H + KVH) * D, (H + 2 * KVH) * D]}, "unbound_args": [],
            "module_path": module_path, "instances_observed": 0, "arg_violations": [], "compiled": {"template_piece": None, "note": "piece input placeholder (attention output)"},
        }
        self.nodes[op] = node
        return node

    def _qkv_sources(self, layer: int, H: int, KVH: int, D: int) -> dict[str, tuple[str, int]] | None:
        """Which description ports feed layer ``layer``'s attention op, from the template node specs.

        The pieces are split at the attention op, so q, k and the qkv buffer that v is a column range of are
        outputs of the piece before it. Their producers differ by model: B0 (no qkv bias) roped q and k in one
        kernel (2 ports) and v is a column range of the GEMM output; B1 (Qwen2, qkv bias) ropes k inside the
        bias-add kernel (ports: bias-added qkv, k) and q in a second kernel, v is a column range of the
        bias-added qkv. Identify by row size and module: q = H*D rows-elements produced in the rotary module,
        k = KVH*D likewise (q before k in graph order when H == KVH), v = the (H + 2*KVH)*D buffer.
        """
        if not (H and KVH and D) or not self.templates:
            return None
        pieces = sorted(self.templates)
        body = [p for p in pieces if p != pieces[-1]] if len(pieces) > 1 else pieces  # the last template is the tail (no attention after it)
        cands = [p for p in body if p <= layer]
        if not cands:
            return None
        template = max(cands)
        shift = layer - template
        q: list[tuple[int, str, int]] = []
        k: list[tuple[int, str, int]] = []
        v: list[tuple[int, str, int]] = []
        for spec in self.templates[template]:
            for p in spec.ports:
                e = self.table.entries.get(p.record_id)
                if e is None:
                    continue
                numel = int(math.prod(p.row_shape)) if p.row_shape else 0
                op = P.shift_layers(spec.op, shift)
                if numel == (H + 2 * KVH) * D:
                    v.append((e.k, op, p.port))
                elif "rotary_emb" in (e.module_path or ""):
                    if numel == H * D:
                        q.append((e.k, op, p.port))
                    if numel == KVH * D:
                        k.append((e.k, op, p.port))
        if H == KVH:  # MHA: same size; q is roped first
            both = sorted(set(q) | set(k))
            if len(both) >= 2:
                q, k = [both[0]], [both[1]]
        if len(q) != 1 or len(k) != 1 or len(v) < 1:
            self.errors.append({"where": "_qkv_sources", "layer": layer, "template": template, "q": q, "k": k, "v": v})
            return None
        v0 = sorted(v)[-1]  # the latest producer of a (H+2KVH)*D buffer (bias-added qkv if there is one, else the GEMM output)
        return {"q": (q[0][1], q[0][2]), "k": (k[0][1], k[0][2]), "v": (v0[1], v0[2])}

    def ensure_tail_nodes(self, logits_dtype: str, V: int, H: int) -> None:
        tp = self.world_size > 1
        VS = V // self.world_size if tp else V
        if "lm_head.gemm" not in self.nodes:
            self.nodes["lm_head.gemm"] = {
                "op": "lm_head.gemm", "layer": "lm_head", "op_type": "lm_head", "dtype": logits_dtype, "shape": [VS], "ports": 1, "outputs": [], "phases": [], "positions": "sampled",
                "args": [{"name": "x", "source": "node", "op": "norm.rmsnorm", "port": 0}, {"name": "w", "source": "weight", "param": self.lm_head_param}],
                "relation": "lm_head.same_runtime.v1", "origin": "captured",
                "decompositions": [{"id": "row", "interface": {"input_bytes": H * 2, "output_bits": VS * _dtype_bits(logits_dtype), "range": "row"}}],
                "runtime_impl": "eager tail: vllm ...batch_invariant.matmul_persistent via linear_batch_invariant (compute_logits runs outside the compiled pieces)"
                                + (" - the rank's shard, taken as its slice of the gathered row (the all-gather is a copy)" if tp else ""),
                "family": "gemm", "runtime_op": "verity.bi_mm", "consts": {}, "arg_cols": {}, "unbound_args": [], "module_path": "lm_head", "instances_observed": 0, "arg_violations": [],
                "compiled": {"template_piece": None, "note": "eager tail"},
            }
        if tp and "lm_head.all_gather" not in self.nodes:
            # [COMP, CX1] the gathered logits row every rank's compute_logits returns (eager TP's lm_head.all_gather node)
            self.nodes["lm_head.all_gather"] = {
                "op": "lm_head.all_gather", "layer": "lm_head", "op_type": "other", "dtype": logits_dtype, "shape": [V], "ports": 1, "outputs": [], "phases": [], "positions": "sampled",
                "args": [{"name": "p0", "source": "node", "op": "lm_head.gemm", "port": 0}],
                "relation": "allgather.copy.v1", "origin": "captured",
                "decompositions": [{"id": "row", "interface": {"input_bytes": V * _dtype_bits(logits_dtype) // 8, "output_bits": V * _dtype_bits(logits_dtype), "range": "row"}}],
                "runtime_impl": "eager tail: torch.ops.vllm.all_gather of the ranks' lm_head shards (LogitsProcessor._gather_logits, use_all_gather)",
                "family": "all_gather", "runtime_op": "vllm.all_gather", "consts": {}, "arg_cols": {}, "unbound_args": [], "module_path": "lm_head", "instances_observed": 0, "arg_violations": [],
                "compiled": {"template_piece": None, "note": "eager tail"},
            }
        if "sampler.select" not in self.nodes:
            self.nodes["sampler.select"] = {
                "op": "sampler.select", "layer": "sampler", "op_type": "sampler", "dtype": "int32", "shape": [], "ports": 1, "outputs": [], "phases": [], "positions": "sampled",
                "args": [{"name": "logits", "source": "node", "op": "lm_head.all_gather" if tp else "lm_head.gemm", "port": 0}, {"name": "sampling_params", "source": "boundary", "field": "sampling_params"},
                         {"name": "seed", "source": "boundary", "field": "seed"}],
                "relation": "sampler.select.v1", "origin": "captured", "decompositions": [{"id": "row", "interface": {"input_bytes": 6, "output_bits": 32, "range": "row"}}],
                "runtime_impl": "vllm.v1.worker.gpu.sample.sampler (greedy argmax; trace replay overwrites the emitted token)", "family": "sampler", "runtime_op": "vllm.v1.worker.gpu.sample",
                "consts": {}, "arg_cols": {}, "unbound_args": [], "module_path": "", "instances_observed": 0, "arg_violations": [], "compiled": {"template_piece": None, "note": "eager tail"},
            }

    def _attn_consts(self, module: str) -> dict[str, Any]:
        if module in self.attn_consts_cache:
            return self.attn_consts_cache[module]
        out: dict[str, Any] = {}
        try:
            for name, mod in self.model.named_modules():
                if normalize_module(name) == module:
                    impl = mod.impl
                    out = {
                        "num_heads": int(impl.num_heads), "num_kv_heads": int(impl.num_kv_heads), "head_size": int(impl.head_size),
                        "scale": float(impl.scale), "sliding_window": getattr(impl, "sliding_window", None),
                        "logits_soft_cap": getattr(impl, "logits_soft_cap", None), "kv_cache_dtype": str(getattr(impl, "kv_cache_dtype", "auto")),
                        "fa_version": getattr(impl, "vllm_flash_attn_version", None), "num_splits": 1,
                        "block_size": int(self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size),
                        "backend": getattr(getattr(mod, "attn_backend", None), "__name__", None),
                    }
                    break
        except Exception as e:  # noqa: BLE001
            out["error"] = repr(e)
        self.attn_consts_cache[module] = out
        return out

    # ---------------------------------------------------------------- per step
    def is_sampled_position(self, rid: str, t: int) -> bool:
        plen = self.prompt_lens.get(rid)
        return plen is None or t >= plen - 1

    def on_step(self, ctx: dict[str, Any]) -> None:
        t0 = time.perf_counter()
        recs = ctx["records"]
        req_ids: list[str] = ctx["req_ids"]
        rows_req = ctx["rows_req"]
        rows_pos = ctx["rows_pos"]
        qsl = ctx["qsl"]
        nct = ctx["nct"]
        n_tokens = int(ctx["num_tokens"])
        sampled = ctx.get("sampled")
        step = int(ctx["step_index"])
        # group records by request rows
        per_req: dict[str, list[tuple[Any, int, int, str, int]]] = defaultdict(list)  # rid -> (rec, row, t, op, port)
        by_key: dict[tuple[int, int], Any] = {}
        for r in recs:
            by_key[(r.runtime_piece, r.id)] = r
        for r in recs:
            if r.host is None:
                continue
            template = r.id // ID_STRIDE
            info = self.inputs_info.get(template, {})
            ri = info.get("residual_in")
            if ri and ri["record_id"] == r.id:
                # duplicate of the previous piece's residual output in this same step: compare, don't write
                prod_op, prod_port = self.residual_producer(r.runtime_piece)
                prev = self._find_record_for(recs, prod_op, prod_port, r.runtime_piece - 1)
                self.residual_dup["compared"] += 1
                if prev is not None and prev.host is not None and prev.host.shape == r.host.shape and torch.equal(prev.host, r.host):
                    self.residual_dup["equal"] += 1
                else:
                    self.residual_dup["unequal"] += 1
                continue
            res = self.resolve(r.id, r.runtime_piece)
            if res is None:
                self.ids_unmapped[r.id] += 1
                continue
            op, port, shift = res
            if r.shape[0] != n_tokens:
                self.non_token_rows[r.id] += 1
                continue
            spec = self._spec_for(r.id)
            if spec is not None:
                node = self.ensure_node(spec, shift, r.runtime_piece)
            else:
                e = self.table.entries[r.id]
                layer = layer_of(op)
                node = self.ensure_attn_node(op, op[: -len(".attn")], [int(d) for d in e.shape[1:]], e.dtype, layer)  # op is already shifted
            for i in range(n_tokens):
                rid = req_ids[rows_req[i]]
                per_req[rid].append((r, i, int(rows_pos[i]), node["op"], port))
        # lm_head rows (eager tail): logits at each request's last scheduled position
        logits_map = ctx["logits_map"]
        lm_rows: dict[str, tuple[int, int]] = {}
        if self.cur_logits is not None and self.cur_logits.shape[0] >= len(logits_map):
            V = int(self.cur_logits.shape[-1])
            self.ensure_tail_nodes(str(self.cur_logits.dtype).replace("torch.", ""), V, int(self.model.config.hidden_size))
            for i, (rid, t) in enumerate(logits_map):
                if self.is_sampled_position(rid, t):
                    lm_rows[rid] = (i, t)
        logits_host = self.cur_logits.cpu() if self.cur_logits is not None else None
        for i, rid in enumerate(req_ids):
            positions = [int(nct[i] + k) for k in range(int(qsl[i + 1] - qsl[i]))]
            plen = self.prompt_lens.get(rid)
            phase = "prefill" if plen is None or positions[0] < plen else "decode"
            rstep = self.req_steps[rid]
            self.req_records[rid].append(StepRecord(rstep, phase, tuple(positions)))
            self.req_steps[rid] = rstep + 1
            ru = (self.writer.ru(rid, rstep, phase=phase, positions=positions, on_duplicate="flag") if self.writer is not None
                  else LeafSidecar(self.run_dir, rid, rstep, phase, positions))
            for r, row, t, op, port in per_req[rid]:
                node = self.nodes[op]
                cid = str(value_id(self.profile_id, rid, t=t, phase=phase, layer=node["layer"], op=op, port=port, shard=self.shard))
                hostrow = r.host[row].reshape(-1)
                data = hostrow.view(torch.uint8).numpy().tobytes()
                ru.add(cid, str(r.dtype).replace("torch.", ""), list(r.shape[1:]), data)
                self.leaf_count += 1
                self.bytes_captured += len(data)
                self.node_phases[op].add(phase)
                self.node_instances[op] += 1
            if rid in lm_rows and logits_host is not None:
                li, t = lm_rows[rid]
                node = self.nodes["lm_head.gemm"]
                full = logits_host[li].reshape(-1).contiguous()
                V = int(full.shape[0])
                if self.world_size > 1:
                    # the rank's shard is its slice of the gathered row (AllGather2 is a copy); the row itself is lm_head.all_gather
                    VS = V // self.world_size
                    r0 = int(self.shard or 0) * VS
                    data = full[r0:r0 + VS].contiguous().view(torch.uint8).numpy().tobytes()
                    cid = str(value_id(self.profile_id, rid, t=t, phase=phase, layer="lm_head", op="lm_head.gemm", port=0, shard=self.shard))
                    ru.add(cid, node["dtype"], [VS], data)
                    self.leaf_count += 1
                    self.bytes_captured += len(data)
                    self.node_phases["lm_head.gemm"].add(phase)
                    self.node_instances["lm_head.gemm"] += 1
                    data = full.view(torch.uint8).numpy().tobytes()
                    cid = str(value_id(self.profile_id, rid, t=t, phase=phase, layer="lm_head", op="lm_head.all_gather", port=0, shard=self.shard))
                    ru.add(cid, node["dtype"], [V], data)
                    self.node_phases["lm_head.all_gather"].add(phase)
                    self.node_instances["lm_head.all_gather"] += 1
                else:
                    data = full.view(torch.uint8).numpy().tobytes()
                    cid = str(value_id(self.profile_id, rid, t=t, phase=phase, layer="lm_head", op="lm_head.gemm", port=0))
                    ru.add(cid, node["dtype"], [V], data)
                    self.node_phases["lm_head.gemm"].add(phase)
                    self.node_instances["lm_head.gemm"] += 1
                self.leaf_count += 1
                self.bytes_captured += len(data)
                if sampled is not None:
                    tok = int(sampled[i])
                    cid = str(value_id(self.profile_id, rid, t=t, phase=phase, layer="sampler", op="sampler.select", port=0, shard=self.shard))
                    ru.add(cid, "int32", [], np.array(tok, dtype=np.int32).tobytes())
                    self.leaf_count += 1
                    self.req_outputs[rid].append(tok)
                    self.node_phases["sampler.select"].add(phase)
                    self.node_instances["sampler.select"] += 1
            ru.finish()
            self.ru_count += 1
        self.cur_logits = None
        self.cur_logits_in = None
        self.timing["emit_s"] += time.perf_counter() - t0

    def _spec_for(self, record_id: int) -> P.NodeSpec | None:
        template = record_id // ID_STRIDE
        for spec in self.templates.get(template, []):
            if any(p.record_id == record_id for p in spec.ports):
                return spec
        return None

    def _find_record_for(self, recs: list[Any], op: str, port: int, runtime_piece: int) -> Any:
        for r in recs:
            if r.runtime_piece != runtime_piece:
                continue
            res = self.resolve(r.id, r.runtime_piece)
            if res is not None and res[0] == op and res[1] == port:
                return r
        return None

    # ---------------------------------------------------------------- finish
    def boundaries(self, requests: dict[str, dict[str, Any]]) -> list[Boundary]:
        out = []
        for rid, info in requests.items():
            steps = self.req_records.get(rid, [])
            out.append(Boundary.build(rid, info["prompt_token_ids"], info["output_token_ids"], info["sampling_params"], seed=info.get("seed"),
                                      finish_reason=info.get("finish_reason"), steps=steps, max_tokens=info["sampling_params"].get("max_tokens")))
        return out

    def description(self, model_repo: str, symbols: dict[str, int], kernel_identity: dict[str, Any] | None) -> dict[str, Any]:
        nodes = []
        for op in sorted(self.nodes, key=node_sort_key):
            n = dict(self.nodes[op])
            n["phases"] = sorted(self.node_phases.get(op, set())) or ["prefill", "decode"]
            n["instances_observed"] = self.node_instances.get(op, 0)
            # interface input bytes from the args' producers
            in_bytes = 0
            for a in n["args"]:
                if a.get("source") == "node" and a["op"] in self.nodes:
                    src = self.nodes[a["op"]]
                    port = a.get("port", 0)
                    shp = src["shape"] if port == 0 else next((o["shape"] for o in src["outputs"] if o["port"] == port), src["shape"])
                    dt = src["dtype"] if port == 0 else next((o["dtype"] for o in src["outputs"] if o["port"] == port), src["dtype"])
                    cols = n.get("arg_cols", {}).get(a["name"])
                    elems = (cols[1] - cols[0]) if cols else int(math.prod(shp)) if shp else 1
                    in_bytes += elems * _dtype_bits(dt) // 8
                elif a.get("source") == "token":
                    in_bytes += 4
            n["decompositions"][0]["interface"]["input_bytes"] = in_bytes
            nodes.append(n)
        doc = {
            "schema": "verity-vllm/description/v1", "profile_id": self.profile_id, "model": model_repo, "symbols": symbols,
            "interception": INTERCEPTION, "relations_module": RELATIONS_MODULE, "patches": PATCHES,
            "kernel_identity": None if kernel_identity is None else {k: kernel_identity[k] for k in ("kernels_identical", "production", "replay", "common_distinct_kernels") if k in kernel_identity},
            "nodes": nodes,
        }
        if self.shard is not None:
            doc["shard"] = int(self.shard)  # one rank's view; tp_merge joins the ranks' descriptions
        return doc

    def write_kernels(self) -> dict[str, str]:
        """``kernels/<hash>.py``: the exact Triton source of every Inductor kernel a node's relation launches."""
        d = self.run_dir / "kernels"
        d.mkdir(exist_ok=True)
        written: dict[str, str] = {}
        for prog in self.programs.values():
            for var, src in prog.kernel_src.items():
                h = prog.kernel_hash[var]
                if h in written:
                    continue
                (d / f"{h}.py").write_text(src)
                written[h] = prog.kernel_name[var]
        (d / "index.json").write_text(json.dumps({"kernels": written, "hash": "sha256[:16] of the name-normalised source (compiled_fx.kernels.normalise_kernel)"}, indent=1) + "\n")
        return written

    def finish(self, reqs: list[Any], cp: dict[str, Any], weights_values: str, kernel_identity: dict[str, Any] | None = None) -> dict[str, Any]:
        from verity_vllm import required as vrequired
        from verity_vllm.description import Description

        from verity_vllm_adapter.replay import PARTITION

        requests_info = {r.request_id: {"prompt_token_ids": r.prompt_token_ids, "output_token_ids": r.output_token_ids, "sampling_params": r.sampling_params,
                                        "seed": r.sampling_params.get("seed"), "finish_reason": r.finish_reason} for r in reqs}
        boundary_problems = []
        for b in self.boundaries(requests_info):
            probs = b.validate()
            if probs:
                boundary_problems.append(f"{b.request_id}: {probs}")
            elif self.writer is not None:
                self.writer.write_boundary(b)
        if self.writer is None:  # reference (hash-only) run: sidecars only, no weights/description/required
            summary = self.summary()
            summary.update({"boundary_problems": boundary_problems, "mode": "reference"})
            (self.run_dir / "capture_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
            self.log(f"reference sidecars: leaves {self.leaf_count}, RUs {self.ru_count}, residual dup {self.residual_dup}, unmapped ids {dict(self.ids_unmapped)}")
            return summary
        t_w0 = time.perf_counter()
        ww = self.writer.weights(store_values=(weights_values == "store"))
        n_params = 0
        for name, p in sorted(self.params, key=lambda kv: kv[0]):
            data = p.detach().contiguous().cpu().reshape(-1)
            # under TP a rank's parameter is its shard of the checkpoint tensor (or a replica): identified per rank (w=<param>/s=<r>)
            ww.add(name, str(p.dtype).replace("torch.", ""), list(p.shape), data.view(torch.uint8).numpy().tobytes(), shard=self.shard)
            n_params += 1
        ww.finish()
        weights_info = {"parameters": n_params, "seconds": time.perf_counter() - t_w0, "values_stored": weights_values == "store", "shard": self.shard}
        symbols = {"H": int(self.model.config.hidden_size), "V": int(getattr(self.model.config, "vocab_size", 0))}
        if self.world_size > 1:
            symbols["TP"] = self.world_size
        kernels_written = self.write_kernels()
        description = self.description(cp["repo"], symbols, kernel_identity)
        self.writer.write_description(description)
        self.writer.finish()
        desc_obj = Description(json.loads((self.run_dir / "description.json").read_text()))
        required_n: int | str
        try:
            required = vrequired.freeze_required(self.run_dir / "required.json", desc_obj, PARTITION)
            required_n = len(required["interfaces"])
        except Exception as e:  # noqa: BLE001
            if self.shard is None:
                raise
            required_n = f"not frozen for the per-rank run: {e!r}"  # as the eager TP run (the required-set rule predates sharded nodes)
        summary = self.summary()
        summary.update({"weights": weights_info, "boundary_problems": boundary_problems, "description_nodes": len(desc_obj.nodes),
                        "required_interfaces": required_n, "kernels_written": kernels_written})
        (self.run_dir / "capture_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
        self.log(f"description: {len(desc_obj.nodes)} nodes, required interfaces {required_n}, leaves {self.leaf_count}, RUs {self.ru_count}, "
                 f"residual dup {self.residual_dup}, unmapped ids {dict(self.ids_unmapped)}")
        return summary

    def summary(self) -> dict[str, Any]:
        rel_hist: dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            rel_hist[n["relation"]] += 1
        return {
            "replay_units": self.ru_count, "leaves": self.leaf_count, "bytes_captured": self.bytes_captured, "nodes": len(self.nodes),
            "relations": dict(rel_hist), "unresolved_nodes": [op for op, n in self.nodes.items() if n["relation"] == "unresolved"][:20],
            "residual_dup": self.residual_dup, "ids_unmapped": dict(self.ids_unmapped), "non_token_rows": dict(self.non_token_rows),
            "templates": {str(p): [n.op for n in ns] for p, ns in self.templates.items()}, "errors": self.errors[:20], "timing": self.timing,
            "interception": INTERCEPTION, "patches": PATCHES,
        }


class LeafSidecar:
    """Hash-only replay unit (``ru/<rid>/<step>/leaves.json``), the reference-run format
    ``verity_vllm_adapter.compare`` reads; same ``add``/``finish`` surface as ``CaptureWriter``."""

    def __init__(self, run_dir: Path, rid: str, step: int, phase: str, positions: list[int]) -> None:
        from verity_vllm import hashing

        self._hashing = hashing
        self.dir = run_dir / "ru" / rid / f"{step:06d}"
        self.meta = {"request_id": rid, "step": step, "phase": phase, "positions": list(positions)}
        self.leaves: dict[str, dict[str, Any]] = {}

    def add(self, cid: str, dtype: str, shape: list[int], data: bytes) -> None:
        self.leaves[cid] = {"dtype": dtype, "shape": list(shape), "nbytes": len(data), "leaf": self._hashing.leaf_hash(cid, dtype, shape, data).hex()}

    def finish(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "leaves.json").write_text(json.dumps({**self.meta, "leaves": self.leaves}))


def spec_piece(spec: P.NodeSpec, cap: CompiledCapture) -> int | None:
    for piece, specs in cap.templates.items():
        if spec in specs:
            return piece
    return None


def _dtype_bits(dtype: str) -> int:
    return {"bfloat16": 16, "float16": 16, "float32": 32, "int64": 64, "int32": 32, "int8": 8, "uint8": 8, "bool": 8}.get(dtype, 16)
