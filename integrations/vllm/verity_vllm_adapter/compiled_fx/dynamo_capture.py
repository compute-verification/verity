"""[R9 CB-C 2026-09-18] Capture vLLM's **Dynamo FX graph of the model forward before Inductor** (and before
vLLM's piecewise split), by wrapping ``vllm.compilation.backends.VllmBackend.__call__``.

``VllmBackend.__call__(graph, example_inputs)`` is the single point where Dynamo hands vLLM the whole
traced forward of the model class (custom ops as ``torch.ops.vllm.*`` / ``torch.ops._C.*`` nodes when the
``custom_ops`` profile keeps them, ``torch.*`` functions otherwise). Everything downstream (``split_graph``,
AOTAutograd, vLLM's post-grad passes, Inductor lowering / fusion) happens *after* this graph exists, so it
is the pre-fusion structure of what production runs; the same object the W1d post-grad pass sees, one
stage earlier and without the AOT functionalisation.

The hook serialises (1) the Dynamo graph as it arrives, (2) after the original call, the split graph
(top-level ``call_module`` sequence + each piece + which pieces are splitting graphs), (3) the example
input descriptions, (4) the Python stack that triggered compilation (what forward vLLM ran to compile),
and writes ``dynamo_graphs.json`` + readable ``*.py`` dumps into ``VERITY_CBC_OUT``. It never modifies the
graph. Structure source only; no numerics.
"""

from __future__ import annotations

import json
import operator
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import fx

SCHEMA = "verity-cbc/dynamo-graphs/v1"
STATE: dict[str, Any] = {"calls": [], "installed": False, "out": None}


# --------------------------------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------------------------------

def target_str(t: Any) -> str:
    if isinstance(t, str):
        return t
    if isinstance(t, torch._ops.OpOverload):
        return f"{t.namespace}.{t._schema.name.split('::')[-1]}.{t._overloadname}"
    if isinstance(t, torch._ops.OpOverloadPacket):
        return t._qualified_op_name.replace("::", ".")
    if isinstance(t, torch._ops.HigherOrderOperator):
        return f"hop.{t.__name__}"
    if t is operator.getitem:
        return "operator.getitem"
    mod = getattr(t, "__module__", None) or ""
    name = getattr(t, "__qualname__", None) or getattr(t, "__name__", None) or repr(t)
    return f"{mod}.{name}" if mod else str(name)


def _is_symbolic(d: Any) -> bool:
    if isinstance(d, int):
        return False
    try:
        from torch.fx.experimental.symbolic_shapes import free_symbols

        return bool(free_symbols(d))
    except Exception:  # noqa: BLE001
        return True


def val_desc(v: Any) -> Any:
    """Shape/dtype description of a node's example value (never the values)."""
    if isinstance(v, torch.Tensor):
        return {"kind": "tensor", "shape": [str(d) if _is_symbolic(d) else int(d) for d in v.shape],
                "dtype": str(v.dtype).replace("torch.", ""), "device": str(v.device),
                "stride": [str(s) if _is_symbolic(s) else int(s) for s in v.stride()] if v.layout == torch.strided else None}
    if isinstance(v, torch.SymInt):
        return {"kind": "symint", "expr": str(v)}
    if isinstance(v, torch.SymBool):
        return {"kind": "symbool", "expr": str(v)}
    if isinstance(v, torch.SymFloat):
        return {"kind": "symfloat", "expr": str(v)}
    if isinstance(v, (list, tuple)):
        return {"kind": type(v).__name__, "items": [val_desc(x) for x in v]}
    if v is None:
        return None
    if isinstance(v, (int, float, bool, str)):
        return {"kind": type(v).__name__, "value": v}
    return {"kind": type(v).__name__}


def enc_arg(a: Any) -> Any:
    if isinstance(a, fx.Node):
        return {"$": a.name}
    if isinstance(a, (list, tuple)):
        return {"seq": type(a).__name__, "items": [enc_arg(x) for x in a]}
    if isinstance(a, dict):
        return {"dict": {str(k): enc_arg(v) for k, v in a.items()}}
    if isinstance(a, slice):
        return {"slice": [enc_arg(a.start), enc_arg(a.stop), enc_arg(a.step)]}
    if isinstance(a, torch.dtype):
        return {"dtype": str(a).replace("torch.", "")}
    if isinstance(a, torch.device):
        return {"device": str(a)}
    if isinstance(a, torch.layout):
        return {"layout": str(a)}
    if isinstance(a, torch.memory_format):
        return {"memory_format": str(a)}
    if isinstance(a, (torch.SymInt, torch.SymBool, torch.SymFloat)):
        return {"sym": str(a)}
    if isinstance(a, torch.Tensor):
        return {"tensor_const": val_desc(a)}
    if a is None or isinstance(a, (bool, int, float, str)):
        return a
    if a is Ellipsis:
        return {"ellipsis": True}
    return {"repr": repr(a)[:200], "type": type(a).__name__}


def _stack_paths(node: fx.Node) -> list[str]:
    st = node.meta.get("nn_module_stack") or {}
    out = []
    for _k, v in st.items():
        if isinstance(v, (tuple, list)):
            path, cls = v[0], v[1] if len(v) > 1 else None
            cls_name = getattr(cls, "__qualname__", None) or str(cls)
            out.append(f"{path}:{cls_name}")
        else:
            out.append(str(v))
    return out


def _source_fn(node: fx.Node) -> list[str]:
    sf = node.meta.get("source_fn_stack") or []
    out = []
    for item in sf:
        try:
            nm, fn = item
            out.append(f"{nm}:{target_str(fn)}")
        except Exception:  # noqa: BLE001
            out.append(str(item))
    return out


def _stack_tail(node: fx.Node) -> str:
    st = node.meta.get("stack_trace") or ""
    lines = [ln for ln in st.strip().splitlines() if ln.strip()]
    return " | ".join(lines[-2:]) if lines else ""


def serialize_graph(gm: fx.GraphModule, *, name: str) -> dict[str, Any]:
    nodes = []
    for n in gm.graph.nodes:
        val = n.meta.get("example_value", n.meta.get("val"))
        entry: dict[str, Any] = {
            "name": n.name, "op": n.op, "target": target_str(n.target),
            "args": [enc_arg(a) for a in n.args], "kwargs": {k: enc_arg(v) for k, v in n.kwargs.items()},
            "val": val_desc(val), "users": [u.name for u in n.users],
            "module_stack": _stack_paths(n), "source_fn": _source_fn(n), "stack_tail": _stack_tail(n),
        }
        if n.op == "get_attr":
            try:
                attr = gm.get_parameter(n.target) if n.target in dict(gm.named_parameters()) else gm.get_buffer(n.target) if n.target in dict(gm.named_buffers()) else None
                entry["attr_val"] = val_desc(attr)
            except Exception as e:  # noqa: BLE001
                entry["attr_val"] = {"error": repr(e)}
        nodes.append(entry)
    try:
        readable = gm.print_readable(print_output=False)
    except Exception as e:  # noqa: BLE001
        readable = f"# print_readable failed: {e!r}\n{gm.graph}"
    return {"name": name, "node_count": len(nodes), "nodes": nodes, "readable": readable,
            "submodules": [nm for nm, _ in gm.named_children()]}


def _vllm_frames(limit: int = 40) -> list[str]:
    out = []
    for fr in traceback.extract_stack()[:-2]:
        if "/vllm/" in fr.filename or "verity" in fr.filename:
            out.append(f"{fr.filename.split('/site-packages/')[-1]}:{fr.lineno} {fr.name}")
    return out[-limit:]


# --------------------------------------------------------------------------------------------------
# the hook
# --------------------------------------------------------------------------------------------------

def install(out_dir: str | Path | None = None) -> list[str]:
    """Wrap ``VllmBackend.__call__``. Idempotent. Returns the patched qualnames."""
    if STATE["installed"]:
        return ["vllm.compilation.backends.VllmBackend.__call__"]
    out = Path(out_dir or os.environ.get("VERITY_CBC_OUT", "/tmp/verity_cbc"))
    out.mkdir(parents=True, exist_ok=True)
    STATE["out"] = out
    import vllm.compilation.backends as be

    orig = be.VllmBackend.__call__

    def wrapped(self, graph: fx.GraphModule, example_inputs):
        t0 = time.perf_counter()
        idx = len(STATE["calls"])
        cc = self.compilation_config
        rec: dict[str, Any] = {
            "index": idx, "prefix": getattr(self, "prefix", None), "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trigger_stack": _vllm_frames(),
            "compilation_config": {
                "mode": str(getattr(cc, "mode", None)), "backend": getattr(cc, "backend", None), "custom_ops": list(getattr(cc, "custom_ops", []) or []),
                "splitting_ops": list(getattr(cc, "splitting_ops", []) or []), "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
                "use_inductor_graph_partition": getattr(cc, "use_inductor_graph_partition", None),
                "compile_ranges_endpoints": list(getattr(cc, "compile_ranges_endpoints", []) or []),
                "ir_enable_torch_wrap": getattr(cc, "ir_enable_torch_wrap", None),
                "enabled_custom_ops": dict(getattr(cc, "enabled_custom_ops", {}) or {}), "disabled_custom_ops": dict(getattr(cc, "disabled_custom_ops", {}) or {}),
                "pass_config": {k: v for k, v in vars(getattr(cc, "pass_config", object())).items() if not k.startswith("_")} if hasattr(cc, "pass_config") else None,
            },
            "example_inputs": [val_desc(x) for x in example_inputs],
            "dynamo_graph": serialize_graph(graph, name=f"dynamo_{idx}"),
        }
        STATE["calls"].append(rec)
        _flush()
        try:
            result = orig(self, graph, example_inputs)
        finally:
            try:
                split_gm = getattr(self, "split_gm", None)
                pieces = getattr(self, "piecewise_graphs", None)
                if split_gm is not None:
                    rec["split_graph"] = serialize_graph(split_gm, name=f"split_{idx}")
                    rec["pieces"] = []
                    for item in pieces or []:
                        sub = getattr(split_gm, item.submod_name)
                        rec["pieces"].append({"submod_name": item.submod_name, "graph_id": item.graph_id, "is_splitting_graph": bool(item.is_splitting_graph),
                                              "graph": serialize_graph(sub, name=f"split_{idx}.{item.submod_name}")})
                rec["cache_dir"] = getattr(cc, "cache_dir", None)
                rec["local_cache_dir"] = getattr(cc, "local_cache_dir", None)
                rec["elapsed_s"] = time.perf_counter() - t0
            except Exception as e:  # noqa: BLE001
                rec["post_error"] = repr(e)
            _flush()
        return result

    be.VllmBackend.__call__ = wrapped
    STATE["installed"] = True
    return ["vllm.compilation.backends.VllmBackend.__call__"]


def capture_callable(name: str, fn, *args, dynamic: bool = True, **kwargs) -> dict[str, Any]:
    """Dynamo-trace an arbitrary Python callable (the parts of the vLLM step that live *outside* the
    ``@support_torch_compile`` model graph: ``compute_logits``, the sampler) with a recording backend, run it once
    on the given (real or fake) arguments, and store the graph(s) as ``aux:<name>`` entries. Structure only."""
    import torch._dynamo as dynamo

    graphs: list[dict[str, Any]] = []

    def backend(gm: fx.GraphModule, example_inputs):
        graphs.append({"example_inputs": [val_desc(x) for x in example_inputs], "graph": serialize_graph(gm, name=f"aux_{name}_{len(graphs)}")})
        return gm.forward

    err = None
    try:
        dynamo.reset()
        compiled = dynamo.optimize(backend, dynamic=dynamic)(fn)
        with torch.no_grad():
            compiled(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        import traceback

        err = {"type": type(e).__name__, "msg": str(e)[:3000], "tb": traceback.format_exc()[-6000:]}
    finally:
        dynamo.reset()
    rec = {"index": f"aux:{name}", "kind": "aux", "name": name, "graphs": graphs, "error": err, "n_graphs": len(graphs),
           "callable": f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', repr(fn)))}"}
    STATE.setdefault("aux", []).append(rec)
    _flush()
    return rec


def _flush() -> None:
    if STATE["out"] is None:
        STATE["out"] = Path(os.environ.get("VERITY_CBC_OUT", "/tmp/verity_cbc"))
    out: Path = STATE["out"]
    out.mkdir(parents=True, exist_ok=True)
    doc = {"schema": SCHEMA, "calls": [], "aux": []}
    for rec in STATE.get("aux", []):
        slim = dict(rec)
        gs = []
        for i, g in enumerate(rec["graphs"]):
            (out / f"aux_{rec['name']}_{i}.py").write_text(g["graph"]["readable"])
            gs.append({**g, "graph": {k: v for k, v in g["graph"].items() if k != "readable"}})
        slim["graphs"] = gs
        doc["aux"].append(slim)
    for rec in STATE["calls"]:
        slim = dict(rec)
        dg = slim.get("dynamo_graph")
        if dg:
            (out / f"dynamo_{rec['index']}.py").write_text(dg["readable"])
            slim["dynamo_graph"] = {k: v for k, v in dg.items() if k != "readable"}
        sg = slim.get("split_graph")
        if sg:
            (out / f"split_{rec['index']}.py").write_text(sg["readable"])
            slim["split_graph"] = {k: v for k, v in sg.items() if k != "readable"}
        if slim.get("pieces"):
            ps = []
            for p in slim["pieces"]:
                g = p["graph"]
                (out / f"split_{rec['index']}.{p['submod_name']}.py").write_text(g["readable"])
                ps.append({**p, "graph": {k: v for k, v in g.items() if k != "readable"}})
            slim["pieces"] = ps
        doc["calls"].append(slim)
    tmp = out / "dynamo_graphs.json.tmp"
    tmp.write_text(json.dumps(doc, indent=None, default=str) + "\n")
    tmp.replace(out / "dynamo_graphs.json")
