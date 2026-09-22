"""[R15 compiled, DECISIONS-2 §7 Stage 1/2 — M-0389] The compiled row's Program is P_compiled, derived from the PRODUCTION compiled
implementation's artifacts (Inductor output code: one registered relation per generated kernel, the compiled numerics read off the kernel
source), and validated against the finest NONINVASIVE checkpoints the implementation already materializes: every buffer a kernel writes.

Mechanism (no change to what Inductor compiles): the production compile with Inductor's own wrapper-level debug printer seam
(`aot_inductor.debug_intermediate_value_printer`, level 1) routed to `run_intermediate_hooks` instead of torch.save -- the generated
Triton kernels are byte-identical to production (kernel_set digest compared), only the Python wrapper gains a hook line before and after
each launch.  The observation run uses cudagraph_mode=NONE so the wrapper's Python runs on every step (graphs replay the same kernel
binaries; they do not change numerics) -- noninterference of that choice and of the hooks is checked by comparing tokens and top-k
logprobs bit-for-bit with a PIECEWISE production run of the same process arguments (--reference).

Stage-2 replay, in this process, from the materialized inputs of every launch (weights taken from the live model by reference):
  matmul_kernel_persistent      vLLM's batch-invariant GEMM (the same Triton kernel eager uses)          -> torch.mm under the same override
  fused_add_rms_norm (A)        r = f32(x0)+f32(x1); out = bf16((r*rsqrt(mean(r^2)+eps))*w)              (the graph's bf16 casts elided)
  fused_add_rms_norm (B)        r = f32(x0)+(f32(x1)+f32(x2)); residual = bf16(r); out as (A)            (residual recomputed unrounded)
  fused_add_rms_norm (final)    (B) without the residual store
  _to_copy_embedding_rms_norm   x = table[ids]; residual = x; out = bf16((f32(x)*rsqrt(mean+eps))*w)
  mul_silu_slice                out = bf16((g/(exp(-g)+1))*u)   g = f32(x[:, :H]), u = f32(x[:, H:])
  RoPE combo (poi_fused q,k)    neox rotate-half from the bf16 cos/sin cache row, f32 math, one rounding (fma candidates tried)
The reduction order of the norm kernels (tl.sum over the autotuned R0_BLOCK / num_warps layout) is not known from the source alone: a
family of Triton-order models is tried and the exact-match count of each is reported; the best is the candidate Definition.
Every output is compared bit-for-bit; per-kernel exact rates, mismatched element counts and max |d| are the report."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REC: dict = {"enabled": False, "step": -1, "piece": None, "cur": None, "launches": [], "pieces": [], "max_clone_numel": 5_000_000,
             "gpu_pending": [], "prefill_pieces": None, "hook_calls": 0, "skipped": 0,
             # [R16 compiled, B8 S2] weights/caches are identified by STORAGE (the live model's parameters + buffers), not by size: a large
             # prefill ACTIVATION (B8: 3680 tokens x 16384 = 60M elements) is materialised like any other (up to max_clone_numel_hard), and a
             # tensor above the hard cap that is not a weight is recorded as UNMATERIALISED (its launch is not checked -- never 'exact').
             "param_storages": set(), "param_storages_n": 0, "weight_refs": 0, "assumed_weight_refs": 0, "unmaterialised": 0, "max_clone_numel_hard": 120_000_000,
             # captured path (PIECEWISE graphs): the hook lines run once at CAPTURE; there they issue a device copy into a preallocated slot,
             # which the graph replays every step; the host drains the slots after the last piece of each observed step.
             "slots": {}, "cap_launch": -1, "cap_last_b": -1, "in_cgw": False, "capture_copies": 0, "drained_steps": 0}


# ----------------------------------------------------------------------------------------------------------------------------------- hooks
def install_wrapper_hooks() -> dict:
    """Route Inductor's wrapper-level intermediate-value seam to run_intermediate_hooks (before + after every kernel launch)."""
    import torch
    from torch._inductor import config as icfg
    from torch._inductor.codegen import debug_utils as du

    icfg.aot_inductor.debug_intermediate_value_printer = "1"

    def save(self, args_to_save, kernel_name, before_launch=True, arg_signatures=None):
        from torch._inductor.virtualized import V

        tag = "b" if before_launch else "a"
        for i, arg in enumerate(args_to_save):
            if not isinstance(arg, str):
                continue
            if arg_signatures is not None and i < len(arg_signatures) and not isinstance(arg_signatures[i], torch.dtype):
                continue
            V.graph.wrapper_code.writeline(f"run_intermediate_hooks({kernel_name + '|' + str(i) + '|' + arg + '|' + tag!r}, {arg})")

    du.DebugPrinterManager.codegen_intermediate_tensor_value_save = save

    from vllm.compilation import piecewise_backend as pb

    orig_call = pb.PiecewiseBackend.__call__

    def _enter_piece(idx: int, first: bool) -> None:
        if REC["enabled"] and first:
            _close_launch()
            REC["step"] += 1
        REC["piece"] = idx
        REC["cap_launch"] = -1
        REC["cap_last_b"] = -1

    def call(self, *args):
        if REC["in_cgw"]:
            return orig_call(self, *args)
        _enter_piece(self.piecewise_compile_index, self.is_first_graph)
        if REC["enabled"]:
            REC["pieces"].append({"step": REC["step"], "piece": self.piecewise_compile_index, "inputs": [_meta(a) for a in args]})
        out = orig_call(self, *args)
        if REC["enabled"]:
            outs = out if isinstance(out, (tuple, list)) else (out,)
            REC["pieces"][-1]["outputs"] = [_meta(o) for o in outs]
            if self.is_last_graph:
                _close_launch()
                _flush()
        return out

    pb.PiecewiseBackend.__call__ = call
    try:
        from vllm.compilation import cuda_graph as cg

        orig_cgw = cg.CUDAGraphWrapper.__call__

        def cgw_call(self, *args, **kwargs):
            pbk = getattr(self, "runnable", None)
            idx = getattr(pbk, "piecewise_compile_index", None)
            if idx is None:
                return orig_cgw(self, *args, **kwargs)
            _enter_piece(idx, bool(getattr(pbk, "is_first_graph", False)))
            if REC["enabled"]:
                REC["pieces"].append({"step": REC["step"], "piece": idx, "inputs": [_meta(a) for a in args]})
            REC["in_cgw"] = True
            try:
                out = orig_cgw(self, *args, **kwargs)
            finally:
                REC["in_cgw"] = False
            if REC["enabled"]:
                outs = out if isinstance(out, (tuple, list)) else (out,)
                REC["pieces"][-1]["outputs"] = [_meta(o) for o in outs]
                if getattr(pbk, "is_last_graph", False):
                    _close_launch()
                    n = next((o.shape[0] for o in outs if hasattr(o, "shape") and o.dim() >= 1), None)
                    if n in REC["capture_sizes"]:                                      # replayed step: the slots hold this step's values
                        _drain_slots(n)
                    _flush()
            return out

        cg.CUDAGraphWrapper.__call__ = cgw_call
        cgw = "CUDAGraphWrapper.__call__"
    except Exception as e:  # noqa: BLE001
        cgw = f"none: {e}"
    from torch._inductor import hooks as ih

    ih.INTERMEDIATE_HOOKS.append(_hook)
    return {"debug_printer_level": "1", "piece_hook": "PiecewiseBackend.__call__", "graph_hook": cgw, "intermediate_hooks": len(ih.INTERMEDIATE_HOOKS)}


def _meta(t) -> dict | None:
    import torch

    if not isinstance(t, torch.Tensor):
        return {"value": repr(t)[:40]}
    return {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype).replace("torch.", ""), "ptr": t.data_ptr(), "numel": t.numel()}


def _is_weight(t) -> bool | None:
    """True: t lives in a storage of the model's parameters/buffers (unchanged during the run: kept by reference).  None: the parameter
    storages could not be collected (size rule falls back, recorded as assumed_weight_refs)."""
    if not REC["param_storages"]:
        return None
    try:
        return t.untyped_storage().data_ptr() in REC["param_storages"]
    except Exception:  # noqa: BLE001
        return None


def _live_or_clone(t, *, where: str):
    """Reference for weights/caches, a device clone for everything else; {'unmaterialised': meta} above the hard cap."""
    import torch

    w = _is_weight(t)
    if w is True:
        REC["weight_refs"] += 1
        return {"live": t, "weight": True}
    if t.numel() > REC["max_clone_numel"]:
        if w is None:                            # parameters not yet identified (graph CAPTURE runs inside engine build, before the model is
            REC["assumed_weight_refs"] += 1      # reachable): kept by reference, RESOLVED after build (resolve_deferred_weights)
            return {"live": t, "assumed_weight": True, "deferred": True}
        if t.numel() > REC["max_clone_numel_hard"]:
            REC["unmaterialised"] += 1
            return {"unmaterialised": _meta(t), "where": where}
    return None


def _clone(t):
    import torch

    if not isinstance(t, torch.Tensor):
        return None
    r = _live_or_clone(t, where="eager")
    if r is not None:
        return r
    c = t.detach().clone()
    REC["gpu_pending"].append(c)
    return c


def resolve_deferred_weights() -> dict:
    """After build (parameters reachable): every capture-time reference taken by the size rule is re-classified by storage -- a weight stays a
    reference (weight: True); a tensor that is NOT a weight becomes UNMATERIALISED (its launch will not be checked)."""
    n_w = n_u = 0
    for key, slot in list(REC["slots"].items()):
        if isinstance(slot, dict) and slot.get("deferred"):
            t = slot["live"]
            w = _is_weight(t)
            if w is True:
                REC["slots"][key] = {"live": t, "weight": True}
                n_w += 1
            elif w is False:
                REC["slots"][key] = {"unmaterialised": _meta(t), "where": "capture", "why": "not a parameter storage; above max_clone_numel at capture"}
                n_u += 1
                REC["unmaterialised"] += 1
    REC["assumed_weight_refs"] = sum(1 for s_ in REC["slots"].values() if isinstance(s_, dict) and s_.get("assumed_weight"))
    REC["weight_refs"] += n_w
    return {"resolved_weights": n_w, "resolved_unmaterialised": n_u, "still_assumed": REC["assumed_weight_refs"]}


def collect_param_storages(llm) -> int:
    """Storage pointers of the live model's parameters + buffers (+ KV caches when reachable), read from the in-process engine.  Several
    attribute paths are tried (vLLM V1 UniProcExecutor layouts); 0 when none resolves (the size rule then applies, recorded)."""
    import torch

    cands = []
    paths = (("llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker", "worker", "model_runner"),
             ("llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker", "model_runner"),
             ("llm_engine", "model_executor", "driver_worker", "worker", "model_runner"),
             ("llm_engine", "model_executor", "driver_worker", "model_runner"))
    for path in paths:
        o = llm
        for a in path:
            o = getattr(o, a, None)
            if o is None:
                break
        if o is not None:
            cands.append(o)
    ptrs: set[int] = set()
    for mr in cands:
        model = getattr(mr, "model", None) or getattr(mr, "get_model", lambda: None)()
        if model is None:
            continue
        if "rms_eps" not in REC:
            from verity_capture.experimental.commit_integ.compiled_relations import rms_eps_of
            REC["rms_eps"], REC["rms_eps_source"] = rms_eps_of(model)         # the norm relation's epsilon is the MODEL's (Qwen 1e-6), never a default
        for p in list(model.parameters()) + list(model.buffers()):
            if isinstance(p, torch.Tensor) and p.device.type == "cuda":
                ptrs.add(p.untyped_storage().data_ptr())
        kv = getattr(mr, "kv_caches", None)
        for t in (kv or []) if isinstance(kv, (list, tuple)) else (list(kv.values()) if isinstance(kv, dict) else []):
            if isinstance(t, torch.Tensor):
                ptrs.add(t.untyped_storage().data_ptr())
        if ptrs:
            break
    REC["param_storages"] = ptrs
    REC["param_storages_n"] = len(ptrs)
    return len(ptrs)


def _record_prefill_piece(step: int, piece: int) -> bool:
    pp = REC["prefill_pieces"]
    return step > 0 or pp is None or piece in pp


def _hook(name: str, val) -> None:
    import torch

    if not isinstance(val, torch.Tensor):
        return
    if torch.cuda.is_current_stream_capturing():
        # graph CAPTURE: record a device copy into a persistent slot (replayed with the graph); slots are never freed (their blocks stay
        # live in the graph pool).  Weights and caches are kept by reference.
        kernel, idx, arg, tag = name.rsplit("|", 3)
        if tag == "b" and int(idx) <= REC["cap_last_b"]:
            REC["cap_launch"] += 1
        elif tag == "b" and REC["cap_last_b"] == -1:
            REC["cap_launch"] += 1
        if tag == "b":
            REC["cap_last_b"] = int(idx)
        else:
            REC["cap_last_b"] = -1
        key = (REC["piece"], REC["cap_launch"], name, tuple(val.shape))
        r = _live_or_clone(val, where="capture")
        if r is not None:
            REC["slots"][key] = r
            return
        slot = REC["slots"].get(key)
        if not isinstance(slot, torch.Tensor):
            slot = torch.empty_like(val)
            REC["slots"][key] = slot
        slot.copy_(val)
        REC["capture_copies"] += 1
        return
    if not REC["enabled"]:
        return
    REC["hook_calls"] += 1
    kernel, idx, arg, tag = name.rsplit("|", 3)
    if not _record_prefill_piece(REC["step"], REC["piece"]):
        REC["skipped"] += 1
        return
    cur = REC["cur"]
    if tag == "b" and (cur is None or cur["closed"] or int(idx) <= cur["last_b_idx"]):
        _close_launch()
        cur = REC["cur"] = {"step": REC["step"], "piece": REC["piece"], "kernel": kernel, "args": [], "before": {}, "after": {}, "closed": False, "last_b_idx": -1}
    if tag == "b":
        cur["last_b_idx"] = int(idx)
        cur["args"].append((int(idx), arg))
        cur["before"][arg] = _clone(val)
    else:
        cur["after"][arg] = _clone(val)
        cur["closed"] = True


def triton_launch_configs(cache_dir: str) -> dict:
    """Launch configuration of every Triton kernel compiled into this process's Inductor cache (metadata written by Triton): the
    num_warps / num_stages the generated kernels were built with — the artifact facts a kernel Definition's reduction order derives from."""
    import glob
    import json as _json
    import os

    out: dict = {}
    for f in glob.glob(os.path.join(cache_dir, "triton", "**", "*.json"), recursive=True):
        try:
            d = _json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or "num_warps" not in d:
            continue
        name = d.get("name") or os.path.basename(f)
        out.setdefault(name, set()).add((d.get("num_warps"), d.get("num_stages"), d.get("num_ctas")))
    return {k: sorted(v) for k, v in sorted(out.items())}


def _drain_slots(n) -> None:
    """After the last piece of a replayed step: read every slot of the replayed batch shape into launch records for this step."""
    import torch

    torch.cuda.synchronize()
    groups: dict = {}
    for (piece, launch, name, shape), slot in REC["slots"].items():
        kernel, idx, arg, tag = name.rsplit("|", 3)
        if n is not None and len(shape) >= 1 and shape[0] != n and shape[0] in REC.get("capture_sizes", ()):
            continue                                                                   # the other capture size's slots
        g = groups.setdefault((piece, launch), {"step": REC["step"], "piece": piece, "kernel": kernel, "args": [], "before": {}, "after": {},
                                                "closed": True, "last_b_idx": -1, "captured": True})
        if tag == "b":
            g["args"].append((int(idx), arg))
        val = slot if isinstance(slot, dict) else slot.cpu()
        g[("before" if tag == "b" else "after")][arg] = val
    for (piece, launch) in sorted(groups):
        g = groups[(piece, launch)]
        g["flushed"] = True
        REC["launches"].append(g)
    REC["drained_steps"] += 1


def _close_launch() -> None:
    cur = REC["cur"]
    if cur is not None:
        REC["launches"].append(cur)
        REC["cur"] = None


def _flush() -> None:
    """End of a step: move this step's clones to the host (the GPU is shared; a prefill step's clones are hundreds of MB)."""
    import torch

    torch.cuda.synchronize()
    REC["gpu_pending"].clear()
    for L in REC["launches"]:
        if L.get("flushed"):
            continue
        for side in ("before", "after"):
            L[side] = {n: (t.cpu() if isinstance(t, torch.Tensor) else t) for n, t in L[side].items()}
        L["flushed"] = True
    for P in REC["pieces"]:
        if P.get("flushed") or "out_clones" not in P:
            continue
        P["out_clones"] = [(t.cpu() if isinstance(t, torch.Tensor) else t) for t in P["out_clones"]]
        P["flushed"] = True


# ---------------------------------------------------------------------------------------------------------------------------- relations
def _tensor(x):
    import torch

    if isinstance(x, dict) and "live" in x:
        return x["live"]
    if isinstance(x, dict):
        return None                              # unmaterialised
    return x.cuda() if isinstance(x, torch.Tensor) else None


from verity_capture.experimental.commit_integ.compiled_relations import _fma, _triton_sum_models, _triton_sum, relation  # noqa: E402,F401
def replay(report: dict) -> None:
    import torch

    per: dict[str, dict] = {}
    variant_hits: dict[str, dict[str, int]] = {}
    examples: list = []
    for L in REC["launches"]:
        args = sorted(L["args"])
        names = [a for _, a in args]
        if ("matmul_kernel_persistent" in L["kernel"] and len(names) < 3) or not names:
            report.setdefault("malformed", []).append({"step": L["step"], "piece": L["piece"], "kernel": L["kernel"], "captured": bool(L.get("captured")),
                                                       "args": names, "before": {a: _meta(v) if hasattr(v, "shape") else "live" for a, v in L["before"].items()},
                                                       "after": {a: _meta(v) if hasattr(v, "shape") else "live" for a, v in L["after"].items()}})
            continue
        # a launch whose activation input or output was NOT materialised (above the hard cap, or -- size rule -- taken for a weight) is not
        # checkable: counted as unmaterialised, never as exact or mismatched
        def _unmat(x) -> bool:
            return isinstance(x, dict) and "unmaterialised" in x

        for a in names:                                                      # weight args still taken by the size rule: informational
            for side in ("before", "after"):
                v = L[side].get(a)
                if isinstance(v, dict) and v.get("assumed_weight"):
                    report["assumed_weight_args"] = report.get("assumed_weight_args", 0) + 1

        if "matmul_kernel_persistent" in L["kernel"] and (_unmat(L["before"].get(names[0])) or _unmat(L["after"].get(names[2]))
                                                          or isinstance(L["after"].get(names[2]), dict)):
            u = report.setdefault("unmaterialised_launches", {"n": 0, "examples": []})
            u["n"] += 1
            if len(u["examples"]) < 12:
                u["examples"].append({"step": L["step"], "piece": L["piece"], "kernel": L["kernel"],
                                      "A": (L["before"].get(names[0]) if isinstance(L["before"].get(names[0]), dict) else "materialised"),
                                      "C": (L["after"].get(names[2]) if isinstance(L["after"].get(names[2]), dict) else "materialised")})
            continue
        before = [_tensor(L["before"].get(a)) for a in names]
        # outputs = args rewritten by the launch (after != before bitwise); weights/live refs are never outputs
        out_names = []
        for a in names:
            b, af = L["before"].get(a), L["after"].get(a)
            if _unmat(af):
                u = report.setdefault("unmaterialised_launches", {"n": 0, "examples": []})
                u["n"] += 1
                if len(u["examples"]) < 12:
                    u["examples"].append({"step": L["step"], "piece": L["piece"], "kernel": L["kernel"], "arg": a, "after": af})
                continue
            if isinstance(af, dict) or af is None or isinstance(b, dict):
                continue
            if b is None or b.shape != af.shape or not torch.equal(b, af):
                out_names.append(a)
        if "matmul_kernel_persistent" in L["kernel"]:
            out_names = [names[2]]
            ins = [before[0], before[1]]
        else:
            ins = [before[i] for i, a in enumerate(names) if a not in out_names]
        outs = [_tensor(L["after"][a]) for a in out_names]
        outs_meta = [{"shape": list(o.shape), "dtype": str(o.dtype)} for o in outs]
        labels = None
        if "rms_norm" in L["kernel"]:
            seen = per.get("_norm_seen", 0)
            per["_norm_seen"] = seen + 1
            if seen >= 40:                                                    # after a budget of launches, keep the 4 best-scoring orders
                hits = {}
                for kd in ("add1_norm", "add2_norm", "final_norm", "embed_norm"):
                    for lb, h in variant_hits.get(kd, {}).items():
                        hits[lb] = hits.get(lb, 0) + h
                labels = set(lb for lb, _ in sorted(hits.items(), key=lambda kv: -kv[1])[:4]) or None
        try:
            rel = relation(L["kernel"], ins, outs_meta, eps=float(REC.get("rms_eps", 1e-5)), labels=labels)
        except Exception as e:  # noqa: BLE001
            rel = {"kind": "error", "outputs": None, "error": f"{type(e).__name__}: {e}"}
        kind = rel["kind"]
        p = per.setdefault(kind, {"launches": 0, "outputs": 0, "exact_outputs": 0, "elements": 0, "mismatched_elements": 0, "max_abs": 0.0,
                                 "kernels": set(), "steps": set(), "errors": []})
        p["launches"] += 1
        p["kernels"].add(L["kernel"])
        p["steps"].add(L["step"])
        if rel.get("error"):
            p["errors"].append(rel["error"])
            continue
        cands = rel.get("variants") or ({"single": rel["outputs"]} if rel.get("outputs") is not None else {})
        if not cands:
            continue
        # score every variant; the launch's outputs are judged by the best variant (and every variant's hit count is kept)
        best = None
        for label, pred in cands.items():
            if pred is None or len(pred) != len(outs):
                continue
            ex = sum(1 for pr, o in zip(pred, outs) if pr.shape == o.shape and torch.equal(pr.to(o.dtype), o))
            vh = variant_hits.setdefault(kind, {})
            vh[label] = vh.get(label, 0) + ex
            if best is None or ex > best[1]:
                best = (label, ex, pred)
        if best is None:
            p["errors"].append(f"no variant matched shapes for {L['kernel']}: pred {len(next(iter(cands.values())) or [])} outs {len(outs)}")
            continue
        label, ex, pred = best
        for pr, o in zip(pred, outs):
            p["outputs"] += 1
            p["elements"] += o.numel()
            pr = pr.to(o.dtype)
            eq = torch.equal(pr, o)
            p["exact_outputs"] += int(eq)
            if not eq:
                d = (pr.float() - o.float()).abs()
                p["mismatched_elements"] += int((d > 0).sum().item())
                p["max_abs"] = max(p["max_abs"], float(d.max().item()))
                if len(examples) < 12:
                    examples.append({"kind": kind, "kernel": L["kernel"], "step": L["step"], "piece": L["piece"], "variant": label,
                                     "mismatched": int((d > 0).sum().item()), "numel": o.numel(), "max_abs": float(d.max().item())})
    per.pop("_norm_seen", None)
    for k, p in per.items():
        p["kernels"] = sorted(p["kernels"])
        p["steps"] = [min(p["steps"]), max(p["steps"])] if p["steps"] else []
        p["exact_rate"] = (p["exact_outputs"] / p["outputs"]) if p["outputs"] else None
        p["errors"] = p["errors"][:5]
    report["replay"] = {"per_kind": per, "variant_hits": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:8]) for k, v in variant_hits.items()},
                        "mismatch_examples": examples}


# --------------------------------------------------------------------------------------------------------------------------------- run
def run(a) -> None:
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    hooks = install_wrapper_hooks() if a.hooks == "on" else {"disabled": True}
    from vllm import SamplingParams

    from verity_capture import vllm_adapter as va
    from verity_capture.experimental.commit_integ.compiled_source import compile_settings

    wl = json.load(open(a.workload))
    eng_args = dict(wl.get("engine_args_required") or {})
    for kv in a.engine_arg or []:
        k, v = kv.split("=", 1)
        try:
            eng_args[k] = json.loads(v)
        except Exception:  # noqa: BLE001
            eng_args[k] = v
    eng_args["enforce_eager"] = False
    cc = dict(eng_args.get("compilation_config") or {})
    cc["cudagraph_mode"] = a.cudagraph
    eng_args["compilation_config"] = cc
    if a.prefill_pieces:
        REC["prefill_pieces"] = set(int(x) for x in a.prefill_pieces.split(","))
    t0 = time.time()
    eng = va.build_engine(case=a.case, max_num_seqs=int(wl.get("sweep", {}).get("batch") or 1), engine_args=eng_args)
    comp = compile_settings(eng.llm)
    facts = va.compilation_facts(eng.llm)
    REC["capture_sizes"] = set(int(x) for x in (comp.get("cudagraph_capture_sizes") or []))
    REC["max_clone_numel"] = int(a.max_clone_numel)
    REC["max_clone_numel_hard"] = int(a.max_clone_numel_hard)
    n_ps = collect_param_storages(eng.llm)
    deferred = resolve_deferred_weights() if n_ps else {"resolved_weights": 0, "resolved_unmaterialised": 0, "still_assumed": REC["assumed_weight_refs"]}
    REC["deferred_resolution"] = deferred
    print(f"[s2] weights identified by storage: {n_ps} parameter/buffer/cache storages; capture-time references resolved: {deferred}"
          + ("" if n_ps else " -- NONE: size rule (> max_clone_numel = assumed weight) applies, recorded"))
    sp = SamplingParams(temperature=0.0, max_tokens=int(wl["sweep"]["max_output"]), logprobs=int(a.topk), ignore_eos=True)
    prompts = [{"prompt_token_ids": r["prompt_token_ids"]} for r in wl["requests"]]
    for _ in range(int(a.warmup)):
        eng.llm.generate(prompts, sp, use_tqdm=False)
    REC["enabled"] = a.hooks == "on"
    t1 = time.time()
    outs = eng.llm.generate(prompts, sp, use_tqdm=False)
    _close_launch()
    _flush()
    REC["enabled"] = False
    gen_s = time.time() - t1
    rec = {"schema": "verity/compiled-s2-checkpoints/v1", "workload": a.workload, "engine_args": eng_args, "compilation": comp, "compilation_facts": facts,
           "hooks": hooks, "build_s": round(time.time() - t0 - gen_s, 1), "observed_generate_s": round(gen_s, 3), "topk": int(a.topk), "requests": []}
    for o in outs:
        rec["requests"].append({"request_id": o.request_id, "token_ids": list(o.outputs[0].token_ids),
                                "logprobs": [{str(t): float(l.logprob) for t, l in step_lp.items()} for step_lp in (o.outputs[0].logprobs or [])]})
    rec["generated_kernels"] = va.generated_kernels_of_this_process()
    rec["triton_launch_configs"] = triton_launch_configs(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_root"))
    try:
        from vllm.model_executor.determinism import batch_invariant as bi

        rec["batch_invariant_override_active"] = bool(bi.vllm_is_batch_invariant()) if hasattr(bi, "vllm_is_batch_invariant") else None
    except Exception as e:  # noqa: BLE001
        rec["batch_invariant_override_active"] = f"unknown: {e}"
    steps = sorted({L["step"] for L in REC["launches"]})
    rec["observation"] = {"steps": len(steps), "pieces_per_step": len({p["piece"] for p in REC["pieces"] if p["step"] == steps[0]}) if steps else 0,
                          "launches": len(REC["launches"]), "hook_calls": REC["hook_calls"], "skipped_prefill_tensors": REC["skipped"],
                          "capture_slots": len(REC["slots"]), "capture_copies": REC["capture_copies"], "drained_steps": REC["drained_steps"],
                          "captured_launches": sum(1 for L in REC["launches"] if L.get("captured")),
                          "weights_by_storage": {"param_storages": REC["param_storages_n"], "weight_refs": REC["weight_refs"],
                                                 "assumed_weight_refs_by_size": REC["assumed_weight_refs"], "unmaterialised_tensors": REC["unmaterialised"],
                                                 "deferred_resolution": REC.get("deferred_resolution"),
                                                 "max_clone_numel": REC["max_clone_numel"], "max_clone_numel_hard": REC["max_clone_numel_hard"]},
                          "rms_eps": {"value": REC.get("rms_eps"), "source": REC.get("rms_eps_source")},
                          "launches_per_kernel": {}}
    for L in REC["launches"]:
        rec["observation"]["launches_per_kernel"][L["kernel"]] = rec["observation"]["launches_per_kernel"].get(L["kernel"], 0) + 1
    # noninterference: tokens + top-k logprobs against the PIECEWISE production reference (same process args except the graph mode)
    if a.reference:
        ref = json.load(open(a.reference))
        cmp = {"reference": a.reference, "reference_compilation": ref.get("compilation"), "tokens_equal": True, "steps": 0, "steps_bit_identical": 0, "logprob_max_abs": 0.0}
        for ra, rb in zip(rec["requests"], ref["requests"]):
            if ra["token_ids"] != rb["token_ids"]:
                cmp["tokens_equal"] = False
            for la, lb in zip(ra["logprobs"], rb["logprobs"]):
                cmp["steps"] += 1
                if set(la) == set(lb) and all(la[t] == lb[t] for t in la):
                    cmp["steps_bit_identical"] += 1
                for t in set(la) & set(lb):
                    cmp["logprob_max_abs"] = max(cmp["logprob_max_abs"], abs(la[t] - lb[t]))
        rec["noninterference"] = cmp
    t2 = time.time()
    replay(rec)
    rec["replay"]["replay_s"] = round(time.time() - t2, 2)
    # piece boundary summary (what is materialized between graphs)
    if REC["pieces"]:
        p0 = [p for p in REC["pieces"] if p["step"] == steps[-1]]
        rec["piece_boundaries_last_step"] = [{"piece": p["piece"], "inputs": [m for m in p["inputs"] if m and "shape" in m and m["numel"] <= REC["max_clone_numel"]],
                                              "outputs": p.get("outputs")} for p in p0]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    if a.dump:
        import torch

        keep = [L for L in REC["launches"] if L["step"] in set(int(x) for x in a.dump_steps.split(","))]
        torch.save([{k: (v if k not in ("before", "after") else {n: (t.cpu() if hasattr(t, "cpu") else t) for n, t in v.items() if not isinstance(t, dict)})
                     for k, v in L.items()} for L in keep], a.dump)
    r = rec["replay"]["per_kind"]
    print(f"[s2] compilation={comp} kernel_set={rec['generated_kernels'].get('kernel_set_sha256', '')[:16]} launches={len(REC['launches'])} steps={len(steps)}")
    for k, p in r.items():
        print(f"[s2] {k:12s} launches {p['launches']:5d} outputs {p['outputs']:5d} exact {p['exact_outputs']:5d} ({p['exact_rate']}) mismatched el {p['mismatched_elements']} max|d| {p['max_abs']:.3e} {p['errors'][:1]}")
    print(f"[s2] unmaterialised launches (not checked): {(rec.get('unmaterialised_launches') or {}).get('n', 0)}; weight args by size rule: {rec.get('assumed_weight_args', 0)}; "
          f"weights by storage: {rec['observation']['weights_by_storage']}")
    print(f"[s2] variant hits: {json.dumps(rec['replay']['variant_hits'])[:1200]}")
    if a.reference:
        print(f"[s2] noninterference vs {a.reference}: {rec['noninterference']}")
    print(f"[s2] -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--case", default="LLAMA32_1B")
    ap.add_argument("--hooks", choices=["on", "off"], default="on", help="off: bisect -- production compile without the wrapper hook lines")
    ap.add_argument("--cudagraph", default="NONE", help="graph mode of the OBSERVATION run (NONE: the wrapper's Python runs every step)")
    ap.add_argument("--engine-arg", action="append")
    ap.add_argument("--topk", default=8)
    ap.add_argument("--warmup", default=1)
    ap.add_argument("--prefill-pieces", default=None, help="comma list of piece indices to record on the prefill step (default all)")
    ap.add_argument("--max-clone-numel", default=5_000_000, help="above this, a tensor that is NOT a model parameter/buffer is still cloned (activation); "
                    "only when the parameter storages could not be collected does the size rule take it for a weight (recorded)")
    ap.add_argument("--max-clone-numel-hard", default=120_000_000, help="above this an activation is recorded as unmaterialised (its launch is not checked)")
    ap.add_argument("--reference", help="compiled_ab_logprobs JSON of the PIECEWISE production run (tokens + top-k logprobs)")
    ap.add_argument("--dump", help="torch.save of the recorded launches for --dump-steps")
    ap.add_argument("--dump-steps", default="1,2,31")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
