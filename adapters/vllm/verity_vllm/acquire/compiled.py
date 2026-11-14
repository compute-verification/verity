"""The Inductor artifacts of an `Execution=compiled` workload, read: the launch program of a wrapper (`parse_wrapper`), the split
graph's wiring (`split_graph_wiring`), the wrapper of each piece (`assign_wrappers`) and the workload's request schedule
(`schedule_of_workload`, `tokens_at`), which the served Build (`acquire.served`) is derived from.  Nothing here reads a checkpoint.

    verity-vllm compiled verify --manifest USED --rebuilt REBUILT --out verify.json

holds the Build a treatment ran with to the Build that process rederived from its own compile artifact.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import option
from verity_vllm.engine.arrivals import realized_arrive_steps
from verity_vllm.program.compiled_relations import OUTPUT_ROLES, pointer_arity, roles_of
from verity_vllm.query.manifest.format import manifest_digest as _manifest_digest
from verity_vllm.records.schema_ids import MANIFEST_VERIFY_SCHEMA

IDENTITY_KEY = ("program_digest", "request_id", "step", "invocation", "op_path", "output_member", "element_range", "rank")
_RUN = re.compile(r"^\s*(\w+)\.run\((.*),\s*stream=\w+\)\s*$")


def _split_args(s: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    depth = 0
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
    src = Path(path).read_text()
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
            view_as = _REINTERP.match(a)
            if view_as:
                widths.append(_width(_dims(view_as.group(2))))
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
            params = "\n".join(ln.split("#", 1)[0] for ln in d.group(2).splitlines())      # a commented-out parameter is not a parameter
            out[m.group(1)] = [p.strip() for p in params.replace("\n", " ").split(",") if p.strip()]
    return out


# ---------------------------------------------------------------------------------------------------------------- the split graph
_CALL = re.compile(r"^\s*(submod_\d+) = self\.(submod_\d+)\((.*?)\);?")
_GETITEM = re.compile(r"^\s*(getitem(?:_\d+)?) = (submod_\d+)\[(\d+)\]")
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
    src = Path(path).read_text()
    top, *classes = re.split(r"\n    class (submod_\d+)\(torch\.nn\.Module\):", src)
    sigs: dict[str, list[tuple[str, str]]] = {}
    is_attn: dict[str, bool] = {}
    for name, c in zip(classes[::2], classes[1::2]):
        fwd = next(filter(None, map(_FWD.match, c.splitlines()[1:])), None)        # the first line is what follows the class's own header
        sigs[name] = _param_kinds(fwd.group(1)) if fwd else []
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


@dataclass(frozen=True, kw_only=True)
class VerifyOptions:
    """The Build a compiled workload's treatment USED (the served Build of record) vs the same derivation REBUILT by the
    treatment from its own compile artifact: digests of rows + Program + artifact"""

    manifest: str = option("--manifest", help="the Build the treatment ran with (the served Build of record)")
    rebuilt: str = option("--rebuilt", help="the Build the TREATMENT process rederived from its own compile artifact (build_rebuilt.json)")
    out: str = option("--out")


COMMANDS = {"verify": VerifyOptions}


def main(a: VerifyOptions) -> int:
    used, rebuilt = (json.loads(Path(p).read_text()) for p in (a.manifest, a.rebuilt))
    # a Build may leave out its rows, its populations and its compile artifact, or hold each empty: then it has none
    rows = used.get("identities")
    rows = rows if rows else []
    rows_digest = _manifest_digest(rows)
    counted, used_artifact, rebuilt_artifact = rebuilt.get("populations"), used.get("artifact"), rebuilt.get("artifact")
    rec = {"schema": MANIFEST_VERIFY_SCHEMA, "manifest": a.manifest, "used_digest": used.get("manifest_digest"), "used_rows_digest": rows_digest,
           "rebuilt_digest": rebuilt.get("manifest_digest"), "used_program_digest": used.get("program_digest"), "rebuilt_program_digest": rebuilt.get("program_digest"),
           "used_identities": len(rows), "rebuilt_identities": counted.get("identities") if counted else None,
           "rebuilt_complete": rebuilt.get("complete"), "rebuilt_from": a.rebuilt,
           "used_artifact": used_artifact.get("digest") if used_artifact else None,
           "rebuilt_artifact": rebuilt_artifact.get("digest") if rebuilt_artifact else None,
           "method": "served-compiled rows: no derive_step Program dir exists; the rebuild is the served Build the TREATMENT process "
                     "rederived from its own compile artifact x the workload schedule (pipeline.served_match) -- equal program_digest = the "
                     "treatment executed the Program of record, equal manifest_digest = the same required-value population, equal compile "
                     "artifact (served_artifact) = the treatment loaded the pinned kernels and autotune winners"}
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


__all__ = ["COMMANDS", "IDENTITY_KEY", "assign_wrappers", "main", "parse_wrapper", "schedule_of_workload", "split_graph_wiring", "tokens_at"]
