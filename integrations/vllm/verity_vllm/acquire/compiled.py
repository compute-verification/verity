"""Required-value manifest of an `Execution=compiled` workload from P_compiled x query.

P_compiled is read from the PRODUCTION process's compiler artifacts: the Inductor output code of every piecewise piece (the wrapper's
`call()` = the ordered kernel launches with their buffer arguments and shape constants).  Identities = one row per (step, piece, launch,
non-weight argument role) in the vocabulary of required_manifest.py / binding.py (`op_path` = `c<piece>.<kernel>`, `output_member` = the
role, element_range = the member's whole extent for the step's token count).  Nothing here reads a checkpoint: the manifest is the
artifact x the workload's steps.  program_digest = sha256(kernel_set_sha256 || the composed launch table || widths).

    verity-vllm compiled build --inductor-dir <control process cache> --pieces 17 --steps 32 \
        --prefill-tokens 256 --request r0 --out manifest.json
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from verity_vllm.config import UsageError, option, workload_arg
from verity_vllm.engine.arrivals import realized_arrive_steps
from verity_vllm.query.manifest.format import SCHEMA, manifest_digest as _manifest_digest
from verity_vllm.program.compiled_relations import OUTPUT_ROLES, kernel_key, pointer_arity, roles_of

# the consumer (binding.load_required_manifest) admits ONE manifest schema and recomputes `manifest_digest`
# over the rows with required_manifest.manifest_digest: the compiled manifest is that schema (its P_compiled-specific content lives under
# `program`, `identity_key` and the rows' `family`/`engine_step`/`kind`/`piece` annotations, which the digest formula ignores)
KIND = "verity/compiled-required-manifest/v1"
IDENTITY_KEY = ("program_digest", "request_id", "step", "invocation", "op_path", "output_member", "element_range", "rank")
FAMILY = "compiled_kernel_buffers"
_RUN = re.compile(r"^\s*(\w+)\.run\((.*),\s*stream=\w+\)\s*$")


def _split_args(s: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out


_SHAPE_ALLOC = re.compile(r"^\s*(\w+) = empty_strided_cuda\(\(([^()]*)\),")
_SHAPE_ASSERT = re.compile(r"^\s*assert_size_stride\((\w+), \(([^()]*)\),")
_REINTERP = re.compile(r"^reinterpret_tensor\((\w+), \(([^()]*)\),")
_ALIAS = re.compile(r"^\s*(\w+) = (\w+)\s*;\s*del \w+")                         # Inductor's buffer reuse: `buf6 = buf0; del buf0  # reuse`


def _dims(s: str) -> list[str]:
    return [d.strip() for d in s.split(",") if d.strip()]


def _width(dims: list[str]) -> int | None:
    """Elements per token row: the product of the numeric dims; the leading symbolic dim (s72...) is the token dim.  None = no token dim."""
    if not dims or not re.fullmatch(r"s\d+", dims[0]):
        return None
    w = 1
    for d in dims[1:]:
        if not re.fullmatch(r"\d+", d):
            return None
        w *= int(d)
    return w


_REINTERP_FULL = re.compile(r"^reinterpret_tensor\((\w+), \(([^()]*)\), \(([^()]*)\), ([^()]+)\)$")
_REUSE = re.compile(r"^\s*(\w+) = (?:reinterpret_tensor\((\w+),[^;]*|(\w+))\s*;\s*del \w+\s*#\s*reuse")   # a NEW buffer on a dead buffer's storage
_VIEW_ALIAS = re.compile(r"^\s*(\w+) = reinterpret_tensor\((\w+), \(([^()]*)\), \(([^()]*)\), ([^()]+)\)\s*$")  # a view of a live buffer
_ARGPOS = re.compile(r"^arg(\d+)_1$")
_RUN_ANY = re.compile(r"^\s*\w+\.run\(")


def _base_view(expr: str) -> tuple[str, dict[str, Any] | None]:
    """A launch/return argument expression -> (base buffer name, view) where view = the reinterpret_tensor (dims, strides, offset) as
    written (symbolic in s72) or None when the buffer is passed whole."""
    m = _REINTERP_FULL.match(expr)
    if m:
        return m.group(1), {"dims": _dims(m.group(2)), "strides": _dims(m.group(3)), "offset": m.group(4).strip()}
    return expr, None


def _src_of(base: str, producer: dict[str, Any], view: dict[str, Any] | None) -> dict[str, Any]:
    """Where an input argument's VALUE comes from: an earlier launch's output role (`from: [kernel, role]`), a piece input
    (`piece_input`, positional index in the wrapper's args), or nothing known (a reuse-alias never written = storage handoff)."""
    src: dict[str, Any]
    if base in producer:
        src = {"from": list(producer[base])}
    else:
        m = _ARGPOS.match(base)
        src = {"from": "piece_input", "pos": int(m.group(1))} if m else {"from": "undefined", "base": base}
    if view is not None:
        src["view"] = view
    return src


def parse_wrapper(path: str) -> dict[str, Any]:
    """The runtime `Runner.call(self, args)` program of one Inductor wrapper (the dynamic-shape body; the example-value autotune stubs
    earlier in the file are skipped): launches [{kernel, args (expressions), ints, widths (per arg, elements per token or None),
    edges [{arg, role, src}] (the DATAFLOW: every non-weight input's producer)}], the wrapper's positional args with their kinds
    (tensor | int, from get_args()), the return tuple with each position's producer, and the count of `.run(` lines the parser did not
    read (must be 0: an unparsed launch is an undeclared part of the implementation)."""
    src = open(path).read()
    if "    def call(self, args):" not in src:
        return {"launches": [], "args": [], "arg_kinds": {}, "returns": [], "unparsed_runs": 0, "signatures": {}}
    signatures = kernel_signatures(src)
    body = src.split("    def call(self, args):", 1)[1]
    body = body.split("\ndef benchmark_compiled_module", 1)[0]
    shapes: dict[str, list[str]] = {}
    producer: dict[str, tuple[str, str]] = {}                                  # buffer name -> (kernel, output role) of the launch that wrote it
    out: list[dict[str, Any]] = []
    args_names: list[str] = []
    returns: list[dict[str, Any]] = []
    unparsed = 0
    for line in body.splitlines():
        if not args_names and line.strip().endswith("= args"):
            args_names = [a.strip() for a in line.split("= args")[0].split(",") if a.strip()]
            continue
        m = _SHAPE_ALLOC.match(line) or _SHAPE_ASSERT.match(line)
        if m:
            shapes[m.group(1)] = _dims(m.group(2))
            continue
        m = _REUSE.match(line)
        if m:
            other = m.group(2) or m.group(3)
            if other in shapes:
                shapes[m.group(1)] = shapes[other]
            mv = re.match(r"^\s*(\w+) = reinterpret_tensor\((\w+), \(([^()]*)\),", line)
            if mv:
                shapes[m.group(1)] = _dims(mv.group(3))
            producer.pop(m.group(1), None)                                     # new storage, no value yet
            continue
        m = _VIEW_ALIAS.match(line)
        if m:
            shapes[m.group(1)] = _dims(m.group(3))
            if m.group(2) in producer:
                producer[m.group(1)] = producer[m.group(2)]
            continue
        rs = line.strip()
        if rs.startswith("return (") and rs.endswith(")"):
            for pos, e in enumerate(_split_args(rs[len("return ("):-1])):
                if not e:
                    continue
                base, view = _base_view(e)
                if re.fullmatch(r"\w+", base):
                    r = {"pos": pos, "kind": "tensor", **_src_of(base, producer, view)}
                    if r["from"] == "undefined":
                        r["from"] = "handoff"                                  # storage the piece hands to the eager attention to write
                else:
                    r = {"pos": pos, "kind": "int", "expr": e}
                returns.append(r)
            continue
        m = _RUN.match(line)
        if not m or "*args" in m.group(2):
            if _RUN_ANY.match(line):
                unparsed += 1
            continue
        args = _split_args(m.group(2))
        ints = [int(a) for a in args if re.fullmatch(r"-?\d+", a)]
        widths = []
        for a in args:
            r = _REINTERP.match(a)
            if r:
                widths.append(_width(_dims(r.group(2))))
            elif a in shapes:
                widths.append(_width(shapes[a]))
            else:
                widths.append(None)
        kernel = m.group(1)
        # roles by (name prefix, pointer arity of the generated kernel's signature): the same name carries
        # different argument lists across models (Llama RoPE vs Qwen RoPE+bias); a launch whose arity has no entry is unmodelled (refused)
        sig = signatures.get(kernel)
        roles, wargs = roles_of(kernel, sig)
        edges = []
        for i, a in enumerate(args):
            role = roles.get(i)
            if i in wargs:
                # WEIGHT ROUTING is part of the Program: which served parameter a launch consumes (the wrapper's
                # positional input -> the split graph names it) -- two P_compiled routing the same weights to different norms must differ
                base, view = _base_view(a)
                edges.append({"arg": i, "role": role or f"weight{i}", "weight": True, **_src_of(base, producer, view)})
                continue
            if role is None:
                continue
            base, view = _base_view(a)
            if role in OUTPUT_ROLES:
                continue
            edges.append({"arg": i, "role": role, **_src_of(base, producer, view)})
        for i, a in enumerate(args):                                           # outputs exist after the launch
            role = roles.get(i)
            if role is not None and role in OUTPUT_ROLES:
                base, _v = _base_view(a)
                producer[base] = (kernel, role)
        out.append({"kernel": kernel, "nargs": len(args), "args": args, "ints": ints, "widths": widths, "edges": edges,
                    "roles": dict(roles), "wargs": sorted(wargs), "signature": sig, "pointer_arity": pointer_arity(sig)})
    kinds: dict[str, str] = {}
    if "def get_args():" in src:
        for line in src.rsplit("def get_args():", 1)[1].split("\n\n", 1)[0].splitlines():          # the LAST get_args = the Runner's
            mk = re.match(r"^\s*(arg\d+_1) = (rand_strided|-?\d+)", line)
            if mk:
                kinds[mk.group(1)] = "tensor" if mk.group(2) == "rand_strided" else "int"
    return {"launches": out, "args": args_names, "arg_kinds": [kinds.get(a, "?") for a in args_names], "returns": returns, "unparsed_runs": unparsed,
            "signatures": signatures}


_ASYNC_TRITON = re.compile(r"^(\w+) = async_compile\.triton\('(\w+)', '''", re.M)
_KERNEL_DEF = re.compile(r"^def (\w+)\((.*?)\):", re.M | re.S)


def kernel_signatures(src: str) -> dict[str, list[str]]:
    """{launch-site name: parameter names of the generated kernel's `def`} from the wrapper's inline `X = async_compile.triton('name',
    '''<source>''')` blocks (the generated Triton source is embedded in the wrapper; the def's name may be the bare `triton_`)."""
    out: dict[str, list[str]] = {}
    starts = list(_ASYNC_TRITON.finditer(src))
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(src)
        block = src[m.end(): end]
        d = _KERNEL_DEF.search(block)
        if d:
            params = "\n".join(l.split("#", 1)[0] for l in d.group(2).splitlines())      # a commented-out parameter is not a parameter
            out[m.group(1)] = [p.strip() for p in params.replace("\n", " ").split(",") if p.strip()]
    return out


def _roles(L: dict[str, Any]) -> dict[int, str]:
    """The launch's resolved role map (parse_wrapper stored it with the signature's arity); integer keys (JSON round-trips make them str)."""
    r = L.get("roles")
    if r is None:
        return roles_of(L["kernel"], L.get("signature"))[0]
    return {int(k): v for k, v in r.items()}


def launches_of_wrapper(path: str) -> list[dict[str, Any]]:
    return parse_wrapper(path)["launches"]


# ---------------------------------------------------------------------------------------------------------------- the split graph
_CALL = re.compile(r"^\s*(submod_\d+) = self\.(submod_\d+)\((.*?)\);?")
_GETITEM = re.compile(r"^\s*(getitem(?:_\d+)?) = (submod_\d+)\[(\d+)\]")
_CLASS = re.compile(r"^\s*class (submod_\d+)\(torch\.nn\.Module\):")
_FWD = re.compile(r"^\s*def forward\(self, (.*)\):\s*$")


def _param_kinds(sig: str) -> list[tuple[str, str]]:
    """`name: "bf16[s72, 2048]", s72: "Sym(s72)", tmp : vllm_utils_torch_utils_LayerName` -> [(name, tensor|int|layer)]."""
    out = []
    for part in _split_args(sig):
        if ":" not in part:
            out.append((part.strip(), "?"))
            continue
        name, ann = part.split(":", 1)
        ann = ann.strip().strip('"')
        kind = "int" if ann.startswith("Sym(") else "layer" if "LayerName" in ann else "tensor"
        out.append((name.strip(), kind))
    return out


def _weight_name(n: str) -> str:
    """`l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_` -> `layers.0.mlp.down_proj.weight`."""
    s = re.sub(r"^l_self_", "", n).rstrip("_")
    s = s.replace("modules_", "").replace("parameters_", "").replace("buffers_", "")
    return re.sub(r"_(\d+)_", r".\1.", s).replace("_proj", "_proj").replace("layers_", "layers.").replace("mlp_", "mlp.").replace("self_attn_", "self_attn.").replace("norm_weight", "norm.weight").replace("proj_weight", "proj.weight").replace("emb_cos", "emb.cos").replace("tokens_weight", "tokens.weight")


def split_graph_wiring(path: str) -> dict[str, Any]:
    """vLLM's split graph (`computation_graph.py`, the backbone's `split_gm.print_readable()`): the ordered compiled pieces (submodules
    without the attention splitting op), each attention op, and for every piece argument WHERE ITS VALUE COMES FROM -- another piece's
    return position, the attention op's output buffer, a weight, positions, input ids, or a symbolic int.  This is the wiring the
    Inductor wrappers cannot show (they see `argN_1`)."""
    src = open(path).read()
    top, *classes = re.split(r"\n(?=    class submod_\d+\(torch\.nn\.Module\):)", src)
    sigs: dict[str, list[tuple[str, str]]] = {}
    is_attn: dict[str, bool] = {}
    for c in classes:
        name = _CLASS.match(c).group(1)
        fwd = next((l for l in c.splitlines() if _FWD.match(l)), None)
        sigs[name] = _param_kinds(_FWD.match(fwd).group(1)) if fwd else []
        is_attn[name] = "unified_attention_with_output" in c
    calls: list[tuple[str, list[str]]] = []
    getitems: dict[str, tuple[str, int]] = {}
    for line in top.splitlines():
        m = _CALL.match(line)
        if m:
            calls.append((m.group(2), [a.strip() for a in _split_args(m.group(3))]))
            continue
        m = _GETITEM.match(line)
        if m:
            getitems[m.group(1)] = (m.group(2), int(m.group(3)))
    pieces: list[str] = [n for n, _a in calls if not is_attn.get(n, False)]
    piece_index = {n: i for i, n in enumerate(pieces)}
    attn_index = {n: i for i, n in enumerate(n for n, _a in calls if is_attn.get(n, False))}

    def source(a: str) -> dict[str, Any]:
        if a in getitems:
            q, m = getitems[a]
            if q in piece_index:
                return {"kind": "piece_ret", "piece": piece_index[q], "ret": m}
            return {"kind": "attn_ret", "layer": attn_index.get(q), "ret": m}
        if a == "s72" or re.fullmatch(r"s\d+", a):
            return {"kind": "sym", "name": a}
        if a.startswith("l_input_ids"):
            return {"kind": "input_ids"}
        if a.startswith("l_positions"):
            return {"kind": "positions"}
        if a.startswith("synthetic_local"):
            return {"kind": "layer_name"}
        if a.startswith("l_self_") or a.startswith("l_"):
            return {"kind": "weight", "name": _weight_name(a)}
        return {"kind": "unknown", "name": a}

    wiring = []
    attention = []
    for n, args in calls:
        srcs = [source(a) for a in args]
        kinds = [k for _p, k in sigs.get(n, [])]
        if is_attn.get(n, False):
            names = [p for p, _k in sigs.get(n, [])]
            role = {}
            for pn, sname in zip(names, srcs):
                key = "q" if pn.startswith("query") else "k" if pn.startswith("key") else "v" if pn.startswith("value") else "out" if pn.startswith("output") else None
                if key:
                    role[key] = sname
            attention.append({"layer": attn_index[n], "submod": n, **role})
        else:
            wiring.append({"piece": piece_index[n], "submod": n, "args": srcs, "raw_args": args, "arg_kinds": kinds,
                           "n_ret": 1 + max([m for (q, m) in getitems.values() if q == n], default=-1)})
    return {"path": path, "pieces": len(pieces), "wiring": wiring, "attention": attention}


def _sig(launches: list[dict[str, Any]]) -> tuple:
    return tuple((L["kernel"], L["nargs"], tuple(L["widths"])) for L in launches)


def widths_of(launches: list[dict[str, Any]], _max_tokens=None) -> dict[tuple[str, str], int]:
    """Per (kernel, role) elements per token, from the wrapper's own buffer shapes (empty_strided_cuda / assert_size_stride /
    reinterpret_tensor dims with the symbolic token dim leading)."""
    w: dict[tuple[str, str], int] = {}
    for L in launches:
        roles = _roles(L)
        for i, role in roles.items():
            if i < len(L["widths"]) and L["widths"][i] is not None:
                w[(L["kernel"], role)] = L["widths"][i]
    return w


def compose_pieces(wrappers: list[dict[str, Any]], pieces: int) -> list[list[dict[str, Any]]]:
    """The wrapper set (one per DISTINCT piece code) -> the per-piece launch table for `pieces` pieces: first (embedding), middle x
    (pieces-2) (all middle wrappers must have the same launch signature), last (final norm)."""
    first = [w for w in wrappers if any("embedding" in L["kernel"] for L in w["launches"])]
    last = [w for w in wrappers if any(L["kernel"].endswith("rms_norm_2") for L in w["launches"])]
    middle = [w for w in wrappers if w not in first and w not in last and w["launches"]]
    if len(first) != 1 or len(last) != 1 or not middle:
        raise SystemExit(f"compiled_manifest: cannot classify the wrappers: first {len(first)} middle {len(middle)} last {len(last)}")
    sigs = {_sig(w["launches"]) for w in middle}
    if len(sigs) != 1:
        raise SystemExit(f"compiled_manifest: middle pieces differ in their launch programs ({len(sigs)} signatures) -- not one P_compiled")
    return [first[0]["launches"]] + [middle[0]["launches"]] * (pieces - 2) + [last[0]["launches"]]


def schedule_of_workload(wl: dict[str, Any]) -> list[dict[str, Any]]:
    """[B>=2] the query's request schedule from the sweep workload: request `r<i>` (workload order, = commit_delta's ids) arrives at
    `arrive_step` under the engine's idle-gap rule (`engine.arrivals.realized_arrive_steps`) with its whole prompt (the eager
    Program's model: one prefill step, then one token per step for max_tokens-1 steps)."""
    reqs = [{"arrive_step": int(r.get("arrive_step") or 0), "max_tokens": int(r.get("max_tokens") or wl.get("max_tokens") or 1)} for r in wl["requests"]]
    return [{"request_id": f"r{i}", "arrive": a, "prompt_tokens": len(r["prompt_token_ids"]), "max_tokens": q["max_tokens"]}
            for i, (r, q, a) in enumerate(zip(wl["requests"], reqs, realized_arrive_steps(reqs)))]


def tokens_at(schedule: list[dict[str, Any]], step: int) -> list[tuple[str, int]]:
    """[(request_id, tokens this step)] for the requests active at `step` (prefill = the prompt at the arrival step, then 1)."""
    out = []
    for r in schedule:
        k = step - r["arrive"]
        if k == 0:
            out.append((r["request_id"], r["prompt_tokens"]))
        elif 0 < k < r["max_tokens"]:
            out.append((r["request_id"], 1))
    return out


def assign_wrappers(wrappers: list[dict[str, Any]], graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Each compiled piece of the split graph -> the ONE Inductor wrapper whose positional signature matches it (argument count and
    tensor/int kinds = the submodule's forward() placeholders, number of returned values = the values the graph takes from it).
    Ambiguity or no match is a refusal: the artifact set does not describe the graph.  A layer-name placeholder is an argument of
    no kind get_args() names ("?")."""
    out = []
    for pc in graph["wiring"]:
        kinds = ["?" if k == "layer" else k for k in pc["arg_kinds"]]
        cands = [w for w in wrappers if len(w["args"]) == len(pc["args"]) and w["arg_kinds"] == kinds and len(w["returns"]) == pc["n_ret"]]
        sigs = {_sig(w["launches"]) for w in cands}
        if not cands or len(sigs) != 1:
            raise SystemExit(f"compiled_manifest: piece {pc['piece']} ({pc['submod']}, {len(pc['args'])} args, {pc['n_ret']} returns): "
                             f"{len(cands)} wrapper candidates with {len(sigs)} launch programs -- the wrapper set does not describe the split graph")
        out.append(cands[0])
    return out


def resolve_dataflow(piece_wrappers: list[dict[str, Any]], graph: dict[str, Any] | None) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Every non-weight input of every launch of every piece -> where its VALUE comes from, in the manifest's vocabulary:
    {kind: launch, piece, kernel, role, view?, view_ret?} (an earlier launch's committed output, same piece or across the eager attention
    boundary through the split graph's wiring), {kind: attention_out, layer} (the eager splitting op wrote the buffer), input_ids,
    positions, sym, weight, or undefined (named).  Also the attention ops' operands (q, k, v) resolved to launch outputs."""
    attn_out_of: dict[tuple[int, int], int] = {}
    if graph:
        for a in graph["attention"]:
            o = a.get("out")
            if o and o.get("kind") == "piece_ret":
                attn_out_of[(o["piece"], o["ret"])] = a["layer"]

    def resolve_ret(q: int, m: int) -> dict[str, Any]:
        rets = piece_wrappers[q]["returns"]
        r = next((r for r in rets if r["pos"] == m), None)
        if r is None:
            return {"kind": "undefined", "why": f"piece {q} has no return {m}"}
        if r["kind"] == "int":
            return {"kind": "sym"}
        if r["from"] == "handoff":
            L = attn_out_of.get((q, m))
            return {"kind": "attention_out", "layer": L} if L is not None else {"kind": "undefined", "why": f"piece {q} return {m} is a storage handoff no attention op writes"}
        if r["from"] == "piece_input":
            return {"kind": "undefined", "why": f"piece {q} returns its own input {r['pos']}"}
        out = {"kind": "launch", "piece": q, "kernel": r["from"][0], "role": r["from"][1]}
        if r.get("view"):
            out["view_ret"] = r["view"]
        return out

    per_piece = []
    for p, w in enumerate(piece_wrappers):
        rows = []
        for L in w["launches"]:
            es = []
            for e in L["edges"]:
                if e.get("weight"):
                    if e["from"] == "piece_input" and graph is not None and e["pos"] < len(graph["wiring"][p]["args"]):
                        a = graph["wiring"][p]["args"][e["pos"]]
                        wsrc = dict(a) if a.get("kind") == "weight" else {"kind": "undefined", "why": f"weight argument {e['arg']} is piece input {e['pos']} of kind {a.get('kind')}"}
                    elif e["from"] == "piece_input":
                        wsrc = {"kind": "piece_input", "pos": e["pos"], "wiring": "not declared"}
                    else:
                        wsrc = {"kind": "undefined", "why": f"weight argument {e['arg']} is not a piece input ({e['from']})"}
                    es.append({"role": e["role"], "arg": e["arg"], "weight": True, "src": wsrc})
                    continue
                if e["from"] == "piece_input":
                    if graph is None:
                        src = {"kind": "piece_input", "pos": e["pos"], "wiring": "not declared"}
                    else:
                        a = graph["wiring"][p]["args"][e["pos"]] if e["pos"] < len(graph["wiring"][p]["args"]) else {"kind": "undefined"}
                        src = resolve_ret(a["piece"], a["ret"]) if a["kind"] == "piece_ret" else dict(a)
                elif e["from"] == "undefined":
                    src = {"kind": "undefined", "base": e.get("base")}
                else:
                    src = {"kind": "launch", "piece": p, "kernel": e["from"][0], "role": e["from"][1]}
                if e.get("view"):
                    src["view"] = e["view"]
                es.append({"role": e["role"], "arg": e["arg"], "src": src})
            rows.append({"kernel": L["kernel"], "edges": es})
        per_piece.append(rows)
    attention = []
    if graph:
        for a in graph["attention"]:
            ops = {}
            for k in ("q", "k", "v", "out"):
                sname = a.get(k)
                ops[k] = resolve_ret(sname["piece"], sname["ret"]) if sname and sname.get("kind") == "piece_ret" else (sname or {"kind": "undefined"})
            attention.append({"layer": a["layer"], **ops})
    return per_piece, attention


def build(inductor_dir: str, *, pieces: int, steps: int, prefill_tokens: int, request_id: str = "r0", rank: int = 0,
          kernel_set_sha256: str | None = None, max_tokens: int | None = None, schedule: list[dict[str, Any]] | None = None,
          split_graph: str | None = None, weights_program_digest: str | None = None, weights_in_dataflow: bool = True) -> dict[str, Any]:
    """`weights_in_dataflow` False reproduces the v1 digest (records built before weight routing entered program_digest); the default
    True is v2: every launch's weight argument -> the served parameter the split graph routes to it, in the digest."""
    paths = sorted(glob.glob(os.path.join(inductor_dir, "*", "*.py")))
    wrappers = []
    unparsed = 0
    for p in paths:
        w = parse_wrapper(p)
        if w["launches"] or w["unparsed_runs"]:
            w["path"] = os.path.relpath(p, inductor_dir)
            wrappers.append(w)
            unparsed += w["unparsed_runs"]
    if not wrappers:
        raise SystemExit(f"compiled_manifest: no Inductor wrappers with a call() under {inductor_dir}")
    if unparsed:
        raise SystemExit(f"compiled_manifest: {unparsed} kernel launch line(s) the parser did not read (`*args` or an unknown form) -- an "
                         f"undeclared part of the implementation; refusing")
    graph = split_graph_wiring(split_graph) if split_graph else None
    if graph is not None:
        if graph["pieces"] != pieces:
            raise SystemExit(f"compiled_manifest: the split graph has {graph['pieces']} compiled pieces, --pieces says {pieces}")
        piece_wrappers = assign_wrappers(wrappers, graph)
        table = [w["launches"] for w in piece_wrappers]
        sigs_mid = {_sig(pl) for pl in table[1:-1]}
        if len(sigs_mid) != 1:
            raise SystemExit(f"compiled_manifest: middle pieces differ in their launch programs ({len(sigs_mid)} signatures) -- not one P_compiled")
    else:
        table = compose_pieces(wrappers, pieces)
        first = next(w for w in wrappers if w["launches"] is table[0]); last = next(w for w in wrappers if w["launches"] is table[-1])
        mid = next(w for w in wrappers if w["launches"] is table[1])
        piece_wrappers = [first] + [mid] * (pieces - 2) + [last]
    dataflow, attention = resolve_dataflow(piece_wrappers, graph)
    if not weights_in_dataflow:
        dataflow = [[{"kernel": r["kernel"], "edges": [e for e in r["edges"] if not e.get("weight")]} for r in rows] for rows in dataflow]
    n_weight_edges = sum(1 for rows in dataflow for r in rows for e in r["edges"] if e.get("weight"))
    mt = max_tokens
    if mt is None:                                                              # the norm kernels' xnumel hint = the compile range's max tokens
        for L in table[1]:
            if "rms_norm" in L["kernel"]:
                mt = L["ints"][0]
                break
    widths_by_piece = [widths_of(pl) for pl in table]                          # kernel NAMES repeat across piece kinds with other shapes
    widths = {}
    for kind, pl in (("first", table[0]), ("middle", table[1]), ("last", table[-1])):
        widths.update({(kind + ":" + k, r): v for (k, r), v in widths_of(pl).items()})
    if kernel_set_sha256 is None:
        raise SystemExit("compiled_manifest: --kernel-set (the run's inductor_kernels.json kernel_set_sha256) is required: the Program's "
                         "identity is the generated kernels' sources, not the wrapper bytes")
    launch_table = [[{"kernel": L["kernel"], "roles": [r for i, r in sorted(_roles(L).items())]} for L in pl] for pl in table]
    # the resolved role maps (arg index -> role, weight args, the generated kernel's pointer arity) per piece kind and
    # launch site -- OUTSIDE the digest (the launch table above keeps its earlier form, so the recorded v1/v2 digests are reproduced); the replay
    # (compiled_kernel_check) reads them so that a kernel name shared by two argument lists is never re-roled by name
    role_maps = {kind: [{"kernel": L["kernel"], "roles": {str(i): r for i, r in sorted(_roles(L).items())}, "weight_args": L.get("wargs"),
                         "pointer_arity": L.get("pointer_arity"), "signature": L.get("signature")} for L in pl]
                 for kind, pl in (("first", table[0]), ("middle", table[1]), ("last", table[-1]))}
    undefined = sum(1 for rows in dataflow for r in rows for e in r["edges"] if e["src"]["kind"] == "undefined")
    # program identity = the kernel sources + every piece's launch program + the DATAFLOW (every input's producer, within a piece and
    # across the attention boundary through the split graph) + the attention ops' operands; wrapper bytes/comments do not enter
    pdig = hashlib.sha256(json.dumps({"kernel_set": kernel_set_sha256, "pieces": launch_table, "dataflow": dataflow, "attention": attention,
                                      "wiring_declared": graph is not None, **({"weights_declared": True} if weights_in_dataflow else {}),
                                      "widths": {f"{k[0]}/{k[1]}": v for k, v in sorted(widths.items())}}, sort_keys=True).encode()).hexdigest()
    identities: list[dict[str, Any]] = []
    unknown: dict[str, int] = {}
    if schedule is None:                                        # single request: the whole prompt at step 0, one token per later step
        schedule = [{"request_id": request_id, "arrive": 0, "prompt_tokens": prefill_tokens, "max_tokens": steps}]
    arrive_of = {r["request_id"]: int(r["arrive"]) for r in schedule}
    for step in range(steps):
        # [B>=2] one identity per (request, launch, role): the request's token rows of the step's batched buffer (request-local element
        # range [0, tokens x width); the binding map splits the committed step buffer by the engine's row order, binding.build_binding_map).
        # `step` is REQUEST-RELATIVE (the gprog reading binding.load_required_manifest applies to every per-request manifest: engine step =
        # manifest step + the request's arrival step); `engine_step` records the absolute step the row was derived for.
        for rid, toks in tokens_at(schedule, step):
            rel_step = step - arrive_of[rid]
            for piece, pl in enumerate(table):
                for L in pl:
                    k = L["kernel"]
                    roles = _roles(L)
                    if not roles:
                        unknown[k] = unknown.get(k, 0) + 1
                        continue
                    wargs_L = {int(i) for i in (L.get("wargs") or [])}
                    for _i, role in sorted(roles.items()):
                        if _i in wargs_L:
                            continue        # a WEIGHT argument (registered input, weights-pinned) is never a VU identity: the hook skips it too (Qwen: 'bias')
                        width = widths_by_piece[piece].get((k, role))
                        numel = (toks * width) if width else None
                        identities.append({"family": FAMILY, "program_digest": pdig, "request_id": rid, "step": rel_step, "invocation": 0,
                                           "op_path": f"c{piece:02d}.{k}", "output_member": role, "element_range": [0, numel], "rank": rank,
                                           "kind": "output" if role in OUTPUT_ROLES else "input", "piece": piece, "engine_step": step})
    fam = {FAMILY: len(identities)}
    n_edges = sum(len(r["edges"]) for rows in dataflow for r in rows)
    n_launch_edges = sum(1 for rows in dataflow for r in rows for e in r["edges"] if e["src"]["kind"] == "launch")
    # P_compiled has no weights struct of its own: its GEMM / norm / RoPE launches take the served module
    # parameters as arguments (C2 replays every sampled launch with the LIVE parameter and matches the committed output bit-for-bit), and
    # those parameters are the eager Program's fields.  The row's `bi-eager` counterpart Build of record is named here as the component
    # whose weights-of-record applies; commit_delta reads components.*.program_digest as the Programs of record for the weights pin.
    components = ({"weights_struct": {"program_digest": str(weights_program_digest),
                                      "role": "served-parameter fields (weights-of-record record; the compiled kernels' weight arguments)",
                                      "source": "the row's bi-eager counterpart Build of record"}} if weights_program_digest else {})
    return {"schema": SCHEMA, "kind": KIND, "identity_key": list(IDENTITY_KEY), "program_digest": pdig, "kernel_set_sha256": kernel_set_sha256,
            "components": components,
            "program": {"kind": "P_compiled (Inductor kernel granularity)", "inductor_dir": inductor_dir, "wrappers": [w["path"] for w in wrappers],
                        "piece_wrappers": [w["path"] for w in piece_wrappers],
                        "pieces": pieces, "launch_table": {"first": launch_table[0], "middle": launch_table[1], "last": launch_table[-1]},
                        "role_maps": role_maps,
                        "widths_per_token": {f"{k[0]}/{k[1]}": v for k, v in sorted(widths.items())}, "max_tokens_hint": mt,
                        "dataflow": {"edges": dataflow, "attention": attention, "split_graph": split_graph, "wiring_declared": graph is not None,
                                     "weights_declared": weights_in_dataflow, "n_weight_edges": n_weight_edges, "digest_version": 2 if weights_in_dataflow else 1,
                                     "n_edges": n_edges, "n_launch_edges": n_launch_edges, "n_undefined": undefined,
                                     "why": "every non-weight input of every launch -> its producer (launch output, same piece or through the split "
                                            "graph across the eager attention op); part of program_digest; checked bytes-equal per sampled step"}},
            "query": {"profile": "compiled-kernel-buffers", "why": "every non-weight argument of every kernel launch of P_compiled is a VU boundary; "
                                                                    "weights are registered inputs"},
            "workload": {"request_id": request_id, "steps": steps, "prefill_tokens": prefill_tokens, "schedule": schedule,
                         "step_basis": "request-relative (engine step = step + arrive; binding.load_required_manifest)"},
            "requests": [r["request_id"] for r in schedule], "ranks": [rank], "steps": list(range(steps)),
            "populations": {"identities": len(identities), "by_family": fam, "steps": steps, "per_step": len(identities) // max(steps, 1)},
            "complete": not unknown and undefined == 0, "unmodelled": unknown, "manifest_digest": _manifest_digest(identities), "identities": identities}


@dataclass(frozen=True, kw_only=True)
class BuildOptions:
    inductor_dir: str = option("--inductor-dir")
    pieces: int = option("--pieces", type=int)
    steps: int | None = option("--steps", type=int, default=None,
                               help="engine steps (default: the workload's expected_engine_steps)")
    prefill_tokens: int | None = option("--prefill-tokens", type=int, default=None,
                                        help="single-request form (no --workload): the prompt length")
    workload: str | None = option("--workload", type=workload_arg, default=None,
                                  help="sweep workload json: the request schedule (B>=2: r<i> arrive_step / prompt / max_tokens)")
    request: str = option("--request", default="r0")
    kernel_set: str = option("--kernel-set",
                             help="the run's inductor_kernels.json kernel_set_sha256 (the Program identity's kernel sources)")
    split_graph: str | None = option("--split-graph", default=None,
                                     help="vLLM's computation_graph.py (compilation_config.cache_dir/rank_0_0/backbone): the piece wiring")
    weights_program_digest: str | None = option("--weights-program-digest", default=None,
                                                help="the eager Program of record whose served-parameter fields the compiled kernels read (the workload's bi-eager counterpart's "
                                                     "Build digest): named as components.weights_struct for the weights pin")
    digest_v1: bool = option("--digest-v1", default=False, action="store_true",
                             help="reproduce the v1 program_digest (weight routing NOT in the digest)")
    out: str = option("--out")
    cmd: str = field(default="build", init=False)


@dataclass(frozen=True, kw_only=True)
class VerifyOptions:
    """The manifest a compiled workload's treatment USED (built from the control arm's Inductor output code) vs the
    same derivation REBUILT from the treatment's own output code: digests of rows + Program"""

    manifest: str = option("--manifest", help="the manifest the treatment ran with (compiled_manifest.json, control-derived)")
    rebuilt: str = option("--rebuilt", help="compiled_manifest.json derived from the TREATMENT process's own output code")
    out: str = option("--out")
    cmd: str = field(default="verify", init=False)


COMMANDS = {"build": BuildOptions, "verify": VerifyOptions}


def main(a: BuildOptions | VerifyOptions) -> int:
    if a.cmd == "verify":
        used, rebuilt = json.load(open(a.manifest)), json.load(open(a.rebuilt))
        rows_digest = _manifest_digest(used.get("identities") or [])
        rec = {"schema": "verity-sweep/manifest-verify/v1", "manifest": a.manifest, "used_digest": used.get("manifest_digest"), "used_rows_digest": rows_digest,
               "rebuilt_digest": rebuilt.get("manifest_digest"), "used_program_digest": used.get("program_digest"), "rebuilt_program_digest": rebuilt.get("program_digest"),
               "used_identities": len(used.get("identities") or []), "rebuilt_identities": (rebuilt.get("populations") or {}).get("identities"),
               "rebuilt_complete": rebuilt.get("complete"), "rebuilt_from": a.rebuilt,
               "used_artifact": (used.get("artifact") or {}).get("digest"), "rebuilt_artifact": (rebuilt.get("artifact") or {}).get("digest"),
               "method": "compiled rows (P_compiled): no derive_step Program dir exists; the rebuild is compiled_manifest.build over the TREATMENT "
                         "process's own Inductor output code + split graph x the workload schedule (the manifest used was built the same way from "
                         "the CONTROL process's) -- equal program_digest = the treatment executed the control's Program (kernel set x launch "
                         "tables x dataflow), equal manifest_digest = the same required-value population; a served Build's compile artifact "
                         "(served_artifact) equal = the treatment loaded the pinned kernels and autotune winners"}
        rec["ok"] = bool(rec["used_digest"] == rows_digest == rec["rebuilt_digest"] and rec["used_program_digest"] == rec["rebuilt_program_digest"]
                         and rebuilt.get("complete") is True and rec["used_artifact"] == rec["rebuilt_artifact"])
        if not rec["ok"]:
            why = []
            if rec["used_digest"] != rows_digest:
                why.append(f"manifest_digest {str(rec['used_digest'])[:16]} != digest of its rows {rows_digest[:16]} (edited after digesting)")
            if rows_digest != rec["rebuilt_digest"]:
                why.append(f"rows digest {rows_digest[:16]} != the treatment-derived manifest {str(rec['rebuilt_digest'])[:16]} "
                           f"({rec['used_identities']} vs {rec['rebuilt_identities']} identities)")
            if rec["used_program_digest"] != rec["rebuilt_program_digest"]:
                why.append(f"program_digest used {str(rec['used_program_digest'])[:16]} != derived from the treatment's output code {str(rec['rebuilt_program_digest'])[:16]}")
            if rebuilt.get("complete") is not True:
                why.append(f"treatment-derived manifest not complete: unmodelled {rebuilt.get('unmodelled')}")
            if rec["used_artifact"] != rec["rebuilt_artifact"]:
                why.append(f"compile artifact used {str(rec['used_artifact'])[:16]} != the treatment's {str(rec['rebuilt_artifact'])[:16]}")
            rec["why"] = "; ".join(why)
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[compiled_manifest verify] ok={rec['ok']} used {str(rec['used_digest'])[:16]} rows {rows_digest[:16]} rebuilt {str(rec['rebuilt_digest'])[:16]} "
              f"program {str(rec['used_program_digest'])[:16]} / {str(rec['rebuilt_program_digest'])[:16]}{(' -- ' + rec['why']) if not rec['ok'] else ''} -> {a.out}")
        return 0 if rec["ok"] else 3
    schedule = None
    steps, prefill = a.steps, a.prefill_tokens
    if a.workload:
        wl = json.load(open(a.workload))
        schedule = schedule_of_workload(wl)
        steps = steps or int(wl["expected_engine_steps"])
        prefill = prefill or schedule[0]["prompt_tokens"]
    if steps is None or prefill is None:
        raise UsageError("--steps and --prefill-tokens (or --workload) are required")
    m = build(a.inductor_dir, pieces=a.pieces, steps=steps, prefill_tokens=prefill, request_id=a.request, kernel_set_sha256=a.kernel_set, schedule=schedule,
              split_graph=a.split_graph, weights_program_digest=a.weights_program_digest, weights_in_dataflow=not a.digest_v1)
    with open(a.out, "w") as f:
        json.dump(m, f, indent=1)
    df = m["program"]["dataflow"]
    print(f"[compiled_manifest] {m['populations']['identities']} identities ({m['populations']['per_step']}/step) complete={m['complete']} "
          f"dataflow edges {df['n_edges']} (launch->launch {df['n_launch_edges']}, undefined {df['n_undefined']}, wiring "
          f"{'declared' if df['wiring_declared'] else 'NOT declared'}) program {m['program_digest'][:16]} manifest {m['manifest_digest'][:16]} -> {a.out}")
    return 0
