"""The served-compiled Program and its required-value manifest: the computation vLLM serves at its defaults (torch.compile with
Inductor, FULL_AND_PIECEWISE CUDA graphs, no batch invariance), read from the serving process's own compile artifact.

Each compiled piece is its Inductor wrapper's `call()`, read line by line: a line is one of the forms `_parse_piece` names or the
Build refuses it.  A Triton launch binds through `acquire.inductor_templates` (its generated source is a template, its launch
config the one config the process ran, an autotune's winner included); an `extern_kernels.mm` is a cuBLAS GEMM of an activation
against a served weight; vLLM's opaque `moe_forward` op is the fused-MoE block, expanded into the Calls it runs (`served_moe`, its
values committed from inside the op); the buffer each writes is a committed value, `run_intermediate_hooks('verity|<op>|<buf>')`
(`ServedCompiledSource`), named `c<piece>.<op>/<buf>`, and a write no hook names is refused.  A flush,
`run_intermediate_hooks(FLUSH, None)` (`inductor_hooks`, under `batch_copies`), issues the copies its hooks recorded and is no
call.  vLLM's split graph
(`computation_graph.py`) wires the pieces: a piece argument is another piece's return, an attention op's output, a served weight
(its placeholder's module path under the backbone), the token ids or the positions (`c00.input/ids`, `c00.input/positions`).
After the piece whose RoPE (or q/k norm-RoPE) wrote its query, each attention
op is the device's attention kernel (`served_device`: FA3 on cc 9.0, FA2 on cc 12.0) over its request's cached keys and values (the
committed k and v rows of all its tokens so far), named `c<piece>.attention/<layer>` with the layer the module path of the GEMM its
query was projected by.  The backbone's output feeds
the logits, a cuBLAS GEMM of each request's last row against the lm_head (`logits_processor`), and `TokenSelect_v1`
(`runner.sampler/sampled_token_ids`).

The steps are the workload's schedule (`compiled.schedule_of_workload`: a request's whole prompt at its arrival step, then a token a
step) under vLLM's dispatch: a uniform decode step of at most `max_num_seqs` tokens replays a FULL graph, padded to the next of
its capture sizes; another step of at most the largest capture size replays PIECEWISE graphs, padded likewise; a larger step runs
uncaptured.  Each step names the Definitions its batch selects:

  * a GEMM of M rows (the padded size under a graph, the step's tokens without; the logits' M is the step's requests) is the
    cuBLAS route the serving device takes at (M, K, N) (`routes`): `Gemm_v2` unsplit, `CublasSplitKGemm_v1` split-K (cc 12.0's
    bf16 workspace `CublasSplitKGemmBf16Partials_v1`, its slices through the output `CublasSerialSplitKGemm_v1`), with the
    device's k16 step, or cc 12.0's M = 1 gemvx `CublasGemv_v1`;
  * a query row i of a request's qlen rows over klen keys sees T = klen - qlen + 1 + i keys in the device's attention launch
    (`served_device.launch`).  FA3's tile (kBlockM, kBlockN) is the batch's max query length's; its scheduler (vllm-flash-attn's
    `num_splits_heuristic` and its varlen split, statically capped at `split_cap` when the step has at most the largest capture
    size's tokens) splits it NS ways: NS = 1 is `Attention_v4` with the tile's MASKED_FROM, NS > 1 (decode only)
    `Fa3AttentionSplit_v1`.  FA2 splits a decode step `fa2_num_splits` ways (`Attention_v5`, `Fa2AttentionSplit_v1`).

The manifest's identities are compiled.build's: one per (step, request, committed value), the request's rows of the value
(element_range [0, tokens x width); the logits and the sampled id one row per request), `step` request-relative.  program_digest
is over the Program (every piece's calls with their Definitions and operands, the attention ops, the logits) and the step plans.
The Build carries the compile artifact it read (`artifact`, `served_artifact.record`), the workload's pin.

    verity-vllm served build --inductor-dir DIR --split-graph computation_graph.py --launch-configs launch_configs.json \\
        --routes routes.json --workload workload.json --capture-sizes 1,2,4,8,16 --max-num-seqs 8 --cc 9.0 --sms 132 --out manifest.json
    verity-vllm served routes --needs manifest.json --out routes.json          (on the serving GPU: the GEMM routes a Build lacks)
    verity-vllm compiled verify --manifest USED --rebuilt REBUILT --out verify.json
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verity_vllm.acquire import compiled as C
from verity_vllm.acquire import inductor_templates as T
from verity_vllm.acquire import served_artifact as SA
from verity_vllm.acquire import served_device as SD
from verity_vllm.acquire import served_moe as SM
from verity_vllm.acquire.inductor_templates import Unmodelled, bound
from verity_vllm.acquire.sources.inductor_hooks import FLUSH
from verity_vllm.config import UsageError, option, workload_arg
from verity.ml.definitions.registry import b1, cublas_bf16, cublas_gemv
from verity.ml.definitions.registry import inductor as IR
from verity_vllm.query.manifest.bind import split_name
from verity_vllm.query.manifest.format import SCHEMA, manifest_digest
from verity_vllm.query.manifest.format import SERVED_FAMILY, SERVED_IDS as IDS, SERVED_LOGITS as LOGITS
from verity_vllm.query.manifest.format import SERVED_POSITIONS as POSITIONS, SERVED_SAMPLED as SAMPLED

KIND = "verity/served-required-manifest/v1"
FAMILY = SERVED_FAMILY
#: vllm-flash-attn's `_DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH`: the static split count of a step a full graph may hold
SPLIT_CAP = 32
#: the Definitions whose output an attention op reads as its query and keys (statics NHEADS, D)
ROPES = (IR.InductorRoPE.id, IR.InductorQKNormRoPE.id)


# ------------------------------------------------------------------------------------------------------------- one wrapper
_SKIP = [re.compile(p) for p in (r"^#", r"^del \w+(?:, \w+)*$", r"^args\.clear\(\)$", r"^with torch\.cuda\._DeviceGuard\(\d+\):$",
                                 r"^torch\.cuda\.set_device\(\d+\)$", r"^raw_stream\d+ = get_raw_stream\(\d+\)$",
                                 r"^(\w+) = copy_if_misaligned\(\1\)$", r"^assert_alignment\(\w+, \d+, '[^']*'\)$",
                                 rf"^run_intermediate_hooks\({re.escape(repr(FLUSH))}, None\)$")]
_ARGS = re.compile(r"^(\w+(?:, \w+)*),? = args$")
_SYM = re.compile(r"^(s\d+) = (\w+)$")
_ASSERT = re.compile(r"^assert_size_stride\((\w+), \(([^()]*)\), \(([^()]*)\), 'input'\)$")
_META = re.compile(r"^assert_(?:size_stride|tensor_metadata)\((\w+), \(([^()]*)\), \(([^()]*)\)(?:, torch\.(\w+))?, '([^']*)'\)$")
_MOE = re.compile(r"^(\w+) = (torch\.ops\.\w+\.moe_forward\.default)\((.*)\)$")
_ALIAS = re.compile(r"^(\w+) = (\w+)$")
_ALLOC = re.compile(r"^(\w+) = empty_strided_cuda\(\(([^()]*)\), \(([^()]*)\), torch\.(\w+)\)$")
_MM = re.compile(r"^extern_kernels\.mm\((.*), out=(\w+)\)$")
_HOOK = re.compile(r"^run_intermediate_hooks\('verity\|(\w+)\|(\w+)', (\w+)\)$")
_NUMEL = re.compile(r"^(\w+_xnumel(?:_\d+)?) = (.+)$")
_RUN = re.compile(r"^(\w+)\.run\((.*), stream=raw_stream\d+\)$")
_REUSE = re.compile(r"^(\w+) = (?:reinterpret_tensor\((\w+), \(([^()]*)\), \(([^()]*)\), (\d+)\)|(\w+)); del (\w+)  # reuse$")
_VIEW = re.compile(r"^(\w+) = reinterpret_tensor\((\w+), \(([^()]*)\), \(([^()]*)\), (\d+)\)$")
_REINTERP = re.compile(r"^reinterpret_tensor\((\w+), \(([^()]*)\), \(([^()]*)\), (\d+)\)$")
_RETURN = re.compile(r"^return \((.*)\)$")
_TOKENS_EXPR = re.compile(r"^(?:(\d+)\*)?(s\d+)$")


def _prod(dims: list[str]) -> int:
    return int(np.prod([int(d) for d in dims], dtype=np.int64)) if dims else 1


def _contiguous(dims: list[str]) -> list[str]:
    out, acc = [], 1
    for d in reversed(dims):
        out.append(str(acc))
        acc *= int(d)
    return out[::-1]


def call_body(src: str) -> str | None:
    """The wrapper's `call(self, args)` body (its lines up to the next one at column 0), or None for a module without one."""
    if "    def call(self, args):" not in src:
        return None
    return re.split(r"\n(?=\S)", src.split("    def call(self, args):", 1)[1], maxsplit=1)[0]


class _Piece:
    """One wrapper's `call()` as calls over refs: {"arg": k} (the k-th positional argument), {"value": i} (the i-th buffer a
    launch wrote), {"empty": buf} (storage no launch wrote, a handoff), with "width" (elements per token row) or "shape" (a
    buffer without a token dimension), and "cols" [lo, hi) (a column range of each row), "t" (a weight read transposed) or
    "row_of" (the row of a table each element of another ref indexes)."""

    def __init__(self, src: str, configs: dict[str, Any] | None, where: str, cc: tuple[int, int]) -> None:
        self.where, self.configs, self.cc = where, configs, cc
        self.kernels = T.kernel_sources(src)
        self.args: list[str] = []
        self.syms: dict[str, str] = {}
        self.shape: dict[str, tuple[list[str], list[str]]] = {}
        self.cur: dict[str, dict[str, Any]] = {}
        self.numels: dict[str, str] = {}
        self.values: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.kernel_records: dict[str, dict[str, Any]] = {}
        self.extern: dict[str, str] = {}                    # a buffer an extern op returned -> the op's qualified name
        self.returns: list[dict[str, Any]] | None = None
        body = call_body(src)
        if body is None:
            raise Unmodelled(f"{where}: no call()")
        for line in body.splitlines():
            line = line.strip()
            if line and not any(p.match(line) for p in _SKIP):
                self._line(line)
        if self.returns is None:
            raise Unmodelled(f"{where}: call() has no return")
        unnamed = [v["buf"] for v in self.values if "name" not in v]
        if unnamed:
            raise Unmodelled(f"{where}: buffers {unnamed} are written but no run_intermediate_hooks names them (not committed)")

    def fail(self, why: str) -> Unmodelled:
        return Unmodelled(f"{self.where}: {why}")

    def _token(self, dims: list[str]) -> bool:
        return bool(dims) and dims[0] in self.syms

    def _register(self, name: str, dims: list[str], strides: list[str]) -> None:
        inner = dims[1:] if self._token(dims) else dims
        want = ([str(_prod(inner))] if self._token(dims) else []) + _contiguous(inner)
        if strides != want or any(not d.isdigit() for d in inner):
            raise self.fail(f"{name} {dims} strides {strides}: not a row-major buffer of token rows or a constant shape")
        self.shape[name] = (dims, strides)

    def _base(self, name: str, base: dict[str, Any]) -> dict[str, Any]:
        dims = self.shape[name][0]
        return {**base, "width": _prod(dims[1:])} if self._token(dims) else {**base, "shape": [int(d) for d in dims]}

    def ref(self, expr: str) -> dict[str, Any]:
        m = _REINTERP.match(expr)
        if m:
            return self._view(m.group(1), C._dims(m.group(2)), C._dims(m.group(3)), int(m.group(4)))
        if expr in self.syms:
            return {"sym": True}
        if expr not in self.cur:
            raise self.fail(f"{expr!r} is not a buffer or argument of the wrapper")
        return dict(self.cur[expr])

    def _view(self, name: str, dims: list[str], strides: list[str], off: int) -> dict[str, Any]:
        base = self.ref(name)
        if name not in self.shape or any(k in base for k in ("cols", "t", "row_of")):
            raise self.fail(f"a view of {name}, itself a view or of no known shape")
        bd = self.shape[name][0]
        if self._token(bd):
            W, inner = _prod(bd[1:]), dims[1:]
            if (not self._token(dims) or any(not d.isdigit() for d in inner) or strides != [str(W)] + _contiguous(inner)
                    or not 0 <= off <= off + _prod(inner) <= W):
                raise self.fail(f"view {dims} {strides} +{off} of {name} {bd}: not a column range of its rows")
            w = _prod(inner)
            return base if (off, w) == (0, W) else {**base, "width": w, "cols": [off, off + w]}
        if len(bd) == 2 and dims == [bd[1], bd[0]] and strides == ["1", bd[1]] and off == 0:
            return {**base, "t": True}
        raise self.fail(f"view {dims} {strides} +{off} of {name} {bd}: only a weight's transpose is modelled")

    def _new_value(self, buf: str, op: str) -> dict[str, Any]:
        if buf not in self.shape or not self._token(self.shape[buf][0]):
            raise self.fail(f"{op} writes {buf}, not a buffer of token rows")
        if self.cur.get(buf, {}).get("arg") is not None:
            raise self.fail(f"{op} writes its argument {buf} in place")
        self.values.append({"buf": buf, "width": _prod(self.shape[buf][0][1:])})
        self.cur[buf] = self._base(buf, {"value": len(self.values) - 1})
        return self.cur[buf]

    def _scalar(self, kernel: str, param: str, expr: str, want: tuple[str, int]) -> None:
        expr = self.numels.get(expr, expr)
        kind, k = want
        m = _TOKENS_EXPR.match(expr)
        ok = (kind == "const" and expr == str(k)) or (kind == "tokens" and m is not None and m.group(2) in self.syms
                                                      and int(m.group(1) or 1) == k)
        if not ok:
            raise self.fail(f"{kernel}: {param} = {expr}, the template's is {k}{'' if kind == 'const' else ' x tokens'}")

    def _line(self, line: str) -> None:
        if (m := _ARGS.match(line)) and not self.args:
            self.args = [a.strip() for a in m.group(1).split(",")]
            for k, a in enumerate(self.args):
                self.cur[a] = {"arg": k}
        elif m := _SYM.match(line):
            src = m.group(2)
            if not (src in self.syms or (src in self.cur and "arg" in self.cur[src] and not self.syms)):
                raise self.fail(f"symbol {line!r}: one token count is modelled")
            self.syms[m.group(1)] = self.syms.get(src, src)
        elif m := _ASSERT.match(line):
            self._register(m.group(1), C._dims(m.group(2)), C._dims(m.group(3)))
            self.cur[m.group(1)] = self._base(m.group(1), self.cur[m.group(1)])
        elif m := _META.match(line):
            buf, dims, strides, dtype, op = m.group(1), C._dims(m.group(2)), C._dims(m.group(3)), m.group(4), m.group(5)
            if self.extern.get(buf) != op or self.shape.get(buf) != (dims, strides) or dtype not in (None, "bfloat16"):
                raise self.fail(f"{line!r}: not the {self.shape.get(buf)} bf16 buffer {self.extern.get(buf)} returned")
        elif m := _MOE.match(line):
            self._moe(m.group(1), m.group(2), C._split_args(m.group(3)))
        elif (m := _ALIAS.match(line)) and m.group(2) in self.extern:
            new, old = m.groups()                           # an extern op's return renamed; the old name is dead
            self.shape[new], self.extern[new], self.cur[new] = self.shape.pop(old), self.extern.pop(old), self.cur.pop(old)
        elif m := _ALLOC.match(line):
            if m.group(4) != "bfloat16":
                raise self.fail(f"{line!r}: a {m.group(4)} buffer")
            self._register(m.group(1), C._dims(m.group(2)), C._dims(m.group(3)))
            self.cur[m.group(1)] = self._base(m.group(1), {"empty": m.group(1)})
        elif m := _REUSE.match(line):
            new, old = m.group(1), m.group(2) or m.group(6)
            if old != m.group(7) or old not in self.shape:
                raise self.fail(f"{line!r}: reuse of an unknown buffer")
            if m.group(2):
                dims, strides = C._dims(m.group(3)), C._dims(m.group(4))
                if int(m.group(5)) != 0 or _prod(dims[1:]) > _prod(self.shape[old][0][1:]):
                    raise self.fail(f"{line!r}: reuse beyond the dead buffer's rows")
            else:
                dims, strides = self.shape[old]
            self._register(new, dims, strides)
            prev = self.cur.pop(old)
            kept = "value" in prev and "cols" not in prev and _prod(dims[1:]) == prev.get("width")
            self.cur[new] = self._base(new, {"value": prev["value"]} if kept else {"empty": new})
        elif m := _VIEW.match(line):
            self.cur[m.group(1)] = self._view(m.group(2), C._dims(m.group(3)), C._dims(m.group(4)), int(m.group(5)))
        elif m := _HOOK.match(line):
            op, buf, var = m.groups()
            v = self.cur.get(var, {}).get("value")
            if buf != var or v is None or "name" in self.values[v]:
                raise self.fail(f"{line!r}: hooks a buffer no launch of this wrapper wrote, or twice")
            self.values[v]["name"] = f"{op}/{buf}"
        elif m := _NUMEL.match(line):
            self.numels[m.group(1)] = m.group(2).strip()
        elif m := _MM.match(line):
            self._mm(C._split_args(m.group(1)), m.group(2))
        elif m := _RUN.match(line):
            self._launch(m.group(1), C._split_args(m.group(2)))
        elif m := _RETURN.match(line):
            self.returns = [self.ref(a) for a in C._split_args(m.group(1)) if a]
        else:
            raise self.fail(f"wrapper line not modelled: {line!r}")

    def _mm(self, args: list[str], out: str) -> None:
        if len(args) != 2:
            raise self.fail(f"extern_kernels.mm with {len(args)} operands")
        x, w = self.ref(args[0]), self.ref(args[1])
        if "width" not in x or "cols" in x or not w.get("t") or "shape" not in w:
            raise self.fail(f"mm({args[0]}, {args[1]}): an activation's whole rows against a transposed weight is modelled")
        N, K = w["shape"]
        if x["width"] != K:
            raise self.fail(f"mm: activation width {x['width']} against a weight of {K} columns")
        y = self._new_value(out, "mm")
        if y["width"] != N:
            raise self.fail(f"mm writes {out} of width {y['width']}, the weight has {N} rows")
        self.calls.append({"op": "mm", "gemm": [K, N], "reads": [x, w], "writes": [y["value"]]})

    def _moe(self, out: str, op: str, args: list[str]) -> None:
        """`out = moe_forward(x, r, None, None, layer, 0)`: the fused-MoE block on x's rows, its gate inside the op on x (the runner
        holds the gate and rebinds the router-logits argument r to its output, so r is never read: x itself, or a buffer of x's
        shape that Inductor gives the op beside the x it mutates), no shared-expert input or token ids, no unpadded width; `layer`
        an argument (vLLM's opaque layer name) or a string.  The op returns a new buffer of x's shape and mutates x, which no later
        line may read."""
        if (len(args) != 6 or args[2:4] != ["None", "None"] or args[5] != "0" or args[0] not in self.shape
                or self.shape.get(args[1]) != self.shape[args[0]]):
            raise self.fail(f"{op}({', '.join(args)}): x, x or a buffer of its shape, None, None, a layer, 0 on a buffer of token rows "
                            "is modelled")
        x = self.ref(args[0])
        lit = re.fullmatch(r"'([^']*)'", args[4])
        layer = {"name": lit.group(1)} if lit else self.ref(args[4])
        if "width" not in x or "cols" in x or not ("name" in layer or set(layer) == {"arg"}):
            raise self.fail(f"{op}: hidden states {x}, layer {args[4]}: whole token rows and a layer argument are modelled")
        self._register(out, *self.shape[args[0]])
        y = self._new_value(out, "moe_forward")
        self.extern[out] = op
        self.calls.append({"op": "moe_forward", "layer": layer, "reads": [x], "writes": [y["value"]]})
        self.cur[args[0]] = self._base(args[0], {"empty": args[0]})

    def _launch(self, name: str, args: list[str]) -> None:
        src = self.kernels.get(name)
        if src is None:
            raise self.fail(f"{name}.run: no async_compile.triton block defines it")
        cfg = T.launch_config(self.configs, src) if self.configs is not None and T.source_sha256(src) in self.configs else None
        rec = T.recognize(name, src, cfg, cc=self.cc)
        self.kernel_records[name] = {k: rec[k] for k in ("source_sha256", "template", "literals", "config")}
        if len(args) != len(rec["params"]):
            raise self.fail(f"{name}.run with {len(args)} arguments, its source takes {rec['params']}")
        ptr = {}
        for p, a in zip(rec["params"], args):
            if p in rec["scalars"]:
                self._scalar(name, p, a, tuple(rec["scalars"][p]))
            else:
                ptr[p] = a
        local: dict[str, dict[str, Any]] = {}
        for c in rec["calls"]:
            reads = []
            for r in c["reads"]:
                ref = local.get(r["param"]) or self.ref(ptr[r["param"]])
                if "cols" in r:
                    lo, hi = r["cols"]
                    if "width" not in ref or "cols" in ref or not 0 <= lo < hi <= ref["width"]:
                        raise self.fail(f"{name}: columns {r['cols']} of {ptr[r['param']]}")
                    ref = {**ref, "width": hi - lo, "cols": [lo, hi]}
                if "row_of" in r:
                    idx = local.get(r["row_of"]) or self.ref(ptr[r["row_of"]])
                    if "shape" not in ref or len(ref["shape"]) != 2 or idx.get("width") != 1:
                        raise self.fail(f"{name}: {r['param']} indexed by {r['row_of']}: a 2-d table by one index a token")
                    ref = {**ref, "row_of": idx}
                got = ref["shape"][1] if "row_of" in ref else ref.get("width", _prod([str(d) for d in ref.get("shape", [])]))
                if got != r["numel"]:
                    raise self.fail(f"{name}: {c['definition']} reads {r['numel']} elements per row from {r['param']}, which has {got}")
                reads.append(ref)
            writes = []
            for wr in c["writes"]:
                buf = ptr[wr["param"]]
                if buf not in self.shape:
                    raise self.fail(f"{name} writes {buf}: not a whole buffer")
                y = self._new_value(buf, name)
                if y["width"] != wr["numel"]:
                    raise self.fail(f"{name}: {c['definition']} writes {wr['numel']} elements per row to {buf} of width {y['width']}")
                local[wr["param"]] = y
                writes.append(y["value"])
            self.calls.append({"op": name, "definition": c["definition"], "base": c["base"], "statics": c["statics"], "reads": reads,
                               "writes": writes})

    def program(self) -> dict[str, Any]:
        return {"args": len(self.args), "values": [{"name": v["name"], "width": v["width"]} for v in self.values], "calls": self.calls,
                "returns": self.returns, "kernels": self.kernel_records}


def parse_piece(path: str, configs: dict[str, Any] | None, cc: tuple[int, int]) -> dict[str, Any]:
    """One Inductor wrapper's `call()` (its kernels compiled for compute capability `cc`) as a piece program (`_Piece`); `Unmodelled`
    names the first line or launch no form models."""
    return _Piece(open(path).read(), configs, os.path.basename(path), cc).program()


# ------------------------------------------------------------------------------------------------------------- the pieces, wired
def served_weight(placeholder: str, backbone: str) -> str:
    """`l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_` under backbone `model` ->
    `model.layers.0.mlp.down_proj.weight` (the served parameter's name)."""
    s = re.sub(r"^[lL]_self_", "", placeholder).rstrip("_")
    parts = [p for p in re.split(r"(?:^|_)(?:modules|parameters|buffers)_", s) if p]
    return ".".join(([backbone] if backbone else []) + parts)


_GRAPH_RETURN = re.compile(r"^\s*return \((\w+),\)\s*$", re.M)


def _graph_output(path: str) -> str:
    top = open(path).read().split("\n    class submod_", 1)[0]
    m = _GRAPH_RETURN.search(top)
    if m is None:
        raise Unmodelled(f"{path}: the split graph returns no single value")
    return m.group(1)


def _merge(env: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    """A piece-local ref of an argument, through the argument's global ref."""
    out = {k: v for k, v in env.items()}
    if "cols" in local:
        lo, hi = local["cols"]
        base = env.get("cols", [0, env.get("width")])
        if env.get("width") is None or hi > base[1] - base[0]:
            raise Unmodelled(f"columns {local['cols']} of an argument {env}")
        out["cols"], out["width"] = [base[0] + lo, base[0] + hi], hi - lo
    elif "width" in local and env.get("width") != local["width"]:
        raise Unmodelled(f"an argument of width {local['width']} wired to {env}")
    if "shape" in local and env.get("weight") is not None:
        out["shape"] = local["shape"]
    if local.get("t"):
        out["t"] = True
    return out


def wire(programs: list[dict[str, Any]], graph: dict[str, Any], *, backbone: str) -> dict[str, Any]:
    """The piece programs through the split graph: global calls over named values, the attention ops, the backbone's output."""
    att_of_out = {(a["out"]["piece"], a["out"]["ret"]): a for a in graph["attention"]
                  if (a.get("out") or {}).get("kind") == "piece_ret"}
    if len(att_of_out) != len(graph["attention"]):
        raise Unmodelled("an attention op's output buffer is not a piece's return")
    values: dict[str, dict[str, Any]] = {IDS: {"width": 1, "piece": 0}, POSITIONS: {"width": 1, "piece": 0}}
    producer: dict[str, dict[str, Any]] = {}
    att_value: dict[tuple[int, int], str] = {}
    rets: list[list[dict[str, Any]]] = []
    calls: list[dict[str, Any]] = []
    attention: list[dict[str, Any]] = []
    for p, (prog, wiring) in enumerate(zip(programs, graph["wiring"])):
        env = []
        for k, (src, raw) in enumerate(zip(wiring["args"], wiring["raw_args"])):
            kind = src["kind"]
            if kind == "piece_ret":
                key = (src["piece"], src["ret"])
                env.append({"value": att_value[key], "width": values[att_value[key]]["width"]} if key in att_value
                           else rets[src["piece"]][src["ret"]])
            elif kind in ("input_ids", "positions"):
                env.append({"value": IDS if kind == "input_ids" else POSITIONS, "width": 1})
            elif kind == "weight":
                env.append({"weight": served_weight(raw, backbone)})
            elif kind == "sym":
                env.append({"sym": True})
            elif kind == "layer_name":
                env.append({"layer_name": True})
            else:
                raise Unmodelled(f"piece {p} argument {k} ({raw}): a {kind} argument")
        names = [f"c{p:02d}.{v['name']}" for v in prog["values"]]

        def g(ref: dict[str, Any]) -> dict[str, Any]:
            if "value" in ref:
                out = {"value": names[ref["value"]], **{k: ref[k] for k in ("width", "cols") if k in ref}}
            elif "arg" in ref:
                out = _merge(env[ref["arg"]], ref)
            elif "empty" in ref:
                out = {"empty": True, "width": ref["width"]}
            else:
                out = dict(ref)
            if "row_of" in ref:
                out["row_of"] = g(ref["row_of"])
            return out

        for name, v in zip(names, prog["values"]):
            if name in values:
                raise Unmodelled(f"{name}: two values of one name")
            values[name] = {"width": v["width"], "piece": p}
        for c in prog["calls"]:
            gc = {**c, "piece": p, "reads": [g(r) for r in c["reads"]], "writes": [names[i] for i in c["writes"]]}
            if any("empty" in r or "sym" in r or "layer_name" in r for r in gc["reads"]):
                raise Unmodelled(f"piece {p} {c['op']} reads storage no launch wrote")
            if "layer" in c:
                gc["layer"] = g(c["layer"])
                if not ("name" in gc["layer"] or gc["layer"] == {"layer_name": True}):
                    raise Unmodelled(f"piece {p} {c['op']}: its layer argument is {gc['layer']}, not a layer name")
            calls.append(gc)
            for w in gc["writes"]:
                producer[w] = gc
        rets.append([g(r) for r in prog["returns"]])
        for a in graph["attention"]:
            if (a.get("q") or {}).get("piece") != p:
                continue
            ops = {}
            for k in ("q", "k", "v"):
                s = a.get(k) or {}
                if s.get("kind") != "piece_ret":
                    raise Unmodelled(f"attention {a['layer']}: its {k} is not a piece's return")
                ops[k] = rets[s["piece"]][s["ret"]]
                if "value" not in ops[k]:
                    raise Unmodelled(f"attention {a['layer']}: its {k} is no value")
            out = rets[a["out"]["piece"]][a["out"]["ret"]]
            if "width" not in out or "cols" in out:
                raise Unmodelled(f"attention {a['layer']}: its output {out} is not a whole buffer of token rows")
            heads = {}
            for k in ("q", "k"):
                rope = producer.get(ops[k]["value"])
                if rope is None or rope.get("base") not in ROPES:
                    raise Unmodelled(f"attention {a['layer']}: its {k} is not a RoPE's output")
                heads[k] = rope["statics"]
            NH, KVH, D = heads["q"]["NHEADS"], heads["k"]["NHEADS"], heads["q"]["D"]
            if heads["k"]["D"] != D or ops["v"]["width"] != KVH * D or out["width"] != NH * D or NH % KVH:
                raise Unmodelled(f"attention {a['layer']}: q {NH}x{D}, k {KVH}x{heads['k']['D']}, v width {ops['v']['width']}, "
                                 f"output width {out['width']}")
            name = f"c{p:02d}.attention/{_layer_of(ops['q']['value'], producer)}"
            att_value[(a["out"]["piece"], a["out"]["ret"])] = name
            values[name] = {"width": NH * D, "piece": p}
            attention.append({"piece": p, "layer": a["layer"], "name": name, **ops, "NH": NH, "KVH": KVH, "D": D})
    return {"values": values, "calls": calls, "attention": attention, "returns": rets}


def _layer_of(value: str, producer: dict[str, dict[str, Any]]) -> str:
    """The attention module whose query `value` is: up its producers' first operands to the GEMM that projected it."""
    seen = value
    while seen in producer:
        c = producer[seen]
        if c["op"] == "mm":
            return c["reads"][1]["weight"].rsplit(".", 2)[0] + ".attn"
        seen = c["reads"][0].get("value")
    raise Unmodelled(f"{value}: no GEMM projects it")


# ------------------------------------------------------------------------------------------------------------- the steps
def gemm_definition(route: dict[str, Any], K: int, N: int, dot: Any) -> dict[str, Any]:
    kind = route["kernel"]
    if kind == "nosplit":
        return bound(b1.GemmV2, K=K, N=N, DOT=dot)
    if kind == "gemv":
        return bound(cublas_gemv.CublasGemvWord, K=K, N=N, LANES=int(route["LANES"]))
    split = {"splitk": cublas_bf16.CublasSplitKGemmBf16PartialsWord if route.get("partial") == "bf16" else cublas_bf16.CublasSplitKGemmWord,
             "serialk": cublas_bf16.CublasSerialSplitKGemmWord}.get(kind)
    if split is None:
        raise Unmodelled(f"cuBLAS route {route}")
    return bound(split, K=K, N=N, SPLIT_K=int(route["SPLIT_K"]), DOT=dot)


def dispatch(n: int, uniform_decode: bool, capture_sizes: list[int], max_num_seqs: int) -> tuple[str, int]:
    """vLLM's CUDA-graph dispatch of a step of n tokens: (FULL | PIECEWISE | NONE, the padded token count)."""
    full = [s for s in capture_sizes if s <= max_num_seqs]
    if uniform_decode and full and n <= max(full):
        return "FULL", min(s for s in full if s >= n)
    if capture_sizes and n <= max(capture_sizes):
        return "PIECEWISE", min(s for s in capture_sizes if s >= n)
    return "NONE", n


def step_plans(schedule: list[dict[str, Any]], steps: int, gemms: list[tuple[int, int]], logits_kn: tuple[int, int], heads: tuple[int, int, int],
               routes: dict[str, Any], *, capture_sizes: list[int], max_num_seqs: int, split_cap: int, device: SD.DeviceBindings,
               moe: Any = None) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """Each engine step's batch, CUDA-graph mode and the Definitions it selects on `device` (`moe`(M): the fused-MoE expert GEMMs' at M
    rows, `served_moe.expert_plans`); and the (M, K, N) GEMM routes `routes` lacks."""
    arrive = {r["request_id"]: r["arrive"] for r in schedule}
    prompt = {r["request_id"]: r["prompt_tokens"] for r in schedule}
    plans, missing = [], set()

    def route(M: int, K: int, N: int) -> dict[str, Any] | None:
        r = routes.get(f"{M},{K},{N}")
        if r is None:
            missing.add((M, K, N))
            return None
        return gemm_definition(r, K, N, device.dot)

    for step in range(steps):
        reqs = C.tokens_at(schedule, step)
        if not reqs:
            continue
        n = sum(t for _r, t in reqs)
        mode, padded = dispatch(n, all(t == 1 for _r, t in reqs), capture_sizes, max_num_seqs)
        M = padded if mode != "NONE" else n
        klens = [prompt[r] + (step - arrive[r]) if step > arrive[r] else prompt[r] for r, _t in reqs]
        ns, rows, launch = SD.launch(device, [t for _r, t in reqs], klens, heads, mode=mode, padded=padded, split_cap=split_cap,
                                     capture_sizes=capture_sizes)
        plans.append({"step": step, "tokens": n, "mode": mode, "padded": padded, "gemm_m": M,
                      "requests": [{"request_id": r, "tokens": t, "klen": k, "NS": s, "attention": a}
                                   for (r, t), k, s, a in zip(reqs, klens, ns, rows)],
                      **launch, "gemms": {f"{K},{N}": route(M, K, N) for K, N in gemms},
                      "logits": route(len(reqs), *logits_kn), **({"moe": moe(M)} if moe else {})})
    return plans, sorted(map(list, missing))


# ------------------------------------------------------------------------------------------------------------- the Build
def identities_of(plans: list[dict[str, Any]], values: dict[str, dict[str, Any]], V: int, schedule: list[dict[str, Any]], pdig: str,
                  rank: int) -> list[dict[str, Any]]:
    """One identity per (engine step, request, value): the request's rows of each of the Program's `values`, its logits row and
    its sampled id."""
    arrive = {r["request_id"]: r["arrive"] for r in schedule}
    out = []
    for plan in plans:
        for r in plan["requests"]:
            rid, toks = r["request_id"], r["tokens"]
            for name in list(values) + [LOGITS, SAMPLED]:
                op, member = split_name(name)
                if name == LOGITS:
                    numel, piece = V, None
                elif name == SAMPLED:
                    numel, piece = 1, None
                else:
                    numel, piece = toks * values[name]["width"], values[name]["piece"]
                out.append({"family": FAMILY, "program_digest": pdig, "request_id": rid, "step": plan["step"] - arrive[rid],
                            "invocation": 0, "op_path": op, "output_member": member, "element_range": [0, numel], "rank": rank,
                            "kind": "input" if name in (IDS, POSITIONS) else "output", "piece": piece, "engine_step": plan["step"]})
    return out


def build(inductor_dir: str, *, split_graph: str, launch_configs: dict[str, Any], routes: dict[str, Any], schedule: list[dict[str, Any]],
          steps: int, capture_sizes: list[int], max_num_seqs: int, device: SD.DeviceBindings, split_cap: int = SPLIT_CAP, backbone: str = "model",
          lm_head: str = "lm_head.weight", rank: int = 0, moe: dict[str, Any] | None = None) -> dict[str, Any]:
    """The served Build (module docstring) on `device` (`served_device.device_of`); `moe` the serving process's fused-MoE record
    (`served_moe.expand`), required when a piece calls the fused-MoE op."""
    graph = C.split_graph_wiring(split_graph)
    wrappers = []
    for path in sorted(glob.glob(os.path.join(inductor_dir, "*", "*.py"))):
        w = C.parse_wrapper(path)
        if w["launches"] or w["unparsed_runs"]:
            wrappers.append({**w, "path": path})
    if not wrappers:
        raise Unmodelled(f"no Inductor wrapper with a call() under {inductor_dir}")
    chosen = C.assign_wrappers(wrappers, graph)
    parsed: dict[str, dict[str, Any]] = {}
    for w in chosen:
        if w["path"] not in parsed:
            parsed[w["path"]] = parse_piece(w["path"], launch_configs, device.cc)
            for other in wrappers:                          # every wrapper of the piece's compiled code computes the same program
                if other["path"] != w["path"] and C._sig(other["launches"]) == C._sig(w["launches"]) and \
                        parse_piece(other["path"], launch_configs, device.cc) != parsed[w["path"]]:
                    raise Unmodelled(f"{os.path.basename(other['path'])} and {os.path.basename(w['path'])}: one launch signature, two programs")
    programs = [parsed[w["path"]] for w in chosen]
    wired, moe_plans = SM.expand(wire(programs, graph, backbone=backbone), moe)
    kernels = {rec["source_sha256"]: rec for prog in programs for rec in prog["kernels"].values()}
    return assemble(wired, moe_plans, _final_value(split_graph, graph, wired), kernels, SA.winners(inductor_dir),
                    source={"inductor_dir": inductor_dir, "split_graph": split_graph,
                            "piece_wrappers": [os.path.relpath(w["path"], inductor_dir) for w in chosen]},
                    schedule=schedule, steps=steps, routes=routes.get("routes") or {}, capture_sizes=capture_sizes,
                    max_num_seqs=max_num_seqs, device=device, split_cap=split_cap, lm_head=lm_head, rank=rank)


def assemble(wired: dict[str, Any], moe_plans: Any, final: dict[str, Any], kernels: dict[str, dict[str, Any]], won: dict[str, Any], *,
             source: dict[str, Any], schedule: list[dict[str, Any]], steps: int, routes: dict[str, Any], capture_sizes: list[int],
             max_num_seqs: int, device: SD.DeviceBindings, split_cap: int, lm_head: str, rank: int) -> dict[str, Any]:
    """The Build of a wired Program (`wire`, `served_moe.expand`) with its kernel records and autotune winners `won`: its step plans,
    identities and artifact; `source` says what the Program was read from (the trace's wrappers, or `predict.compiled`'s pins)."""
    if not wired["attention"]:
        raise Unmodelled("the split graph has no attention op")
    a0 = wired["attention"][0]
    heads = (a0["NH"], a0["KVH"], a0["D"])
    if any((a["NH"], a["KVH"], a["D"]) != heads for a in wired["attention"]):
        raise Unmodelled("attention ops of different head geometry")
    embed = next((c for c in wired["calls"] if c.get("base") == b1.Embedding.id), None)
    if embed is None:
        raise Unmodelled("no embedding call: the vocabulary is unknown")
    V, H = embed["statics"]["V"], wired["values"][final["value"]]["width"]
    gemms = sorted({tuple(c["gemm"]) for c in wired["calls"] if c["op"] == "mm"})
    plans, missing = step_plans(schedule, steps, gemms, (H, V), heads, routes, capture_sizes=capture_sizes,
                                max_num_seqs=max_num_seqs, split_cap=split_cap, device=device, moe=moe_plans)
    program = {"calls": wired["calls"], "attention": wired["attention"], "final": final,
               "logits": {"name": LOGITS, "x": final, "w": {"weight": lm_head, "shape": [V, H]}, "gemm": [H, V]},
               "sampled": {"name": SAMPLED, **bound(b1.TokenSelect, V=V)},
               "kernels": dict(sorted(kernels.items())), "values": wired["values"],
               "dispatch": device.dispatch(capture_sizes, max_num_seqs, split_cap)}
    pdig = hashlib.sha512(json.dumps({"program": program, "steps": plans}, sort_keys=True).encode()).hexdigest()
    names = list(wired["values"])
    identities = identities_of(plans, wired["values"], V, schedule, pdig, rank)
    return {"schema": SCHEMA, "kind": KIND, "identity_key": list(C.IDENTITY_KEY), "program_digest": pdig,
            "program": {"kind": "P_served (the served-compiled computation)", **source, **program},
            "steps_plan": plans, "routes_missing": missing,
            "workload": {"schedule": schedule, "steps": steps,
                         "step_basis": "request-relative (engine step = step + arrive; binding.load_required_manifest)"},
            "requests": [r["request_id"] for r in schedule], "ranks": [rank], "steps": list(range(steps)),
            "populations": {"identities": len(identities), "by_family": {FAMILY: len(identities)}, "steps": steps,
                            "values_per_request_step": len(names) + 2},
            "complete": not missing, "unmodelled": {"routes_missing": len(missing)} if missing else {},
            "artifact": SA.artifact_of(program["kernels"], won), "manifest_digest": manifest_digest(identities), "identities": identities}


def _final_value(split_graph: str, graph: dict[str, Any], wired: dict[str, Any]) -> dict[str, Any]:
    """The backbone's output: the split graph's return, a piece's return that is a value."""
    out = _graph_output(split_graph)
    src = open(split_graph).read().split("\n    class submod_", 1)[0]
    m = re.search(rf"^\s*{out} = (submod_\d+)\[(\d+)\]", src, re.M)
    pieces = [w["submod"] for w in graph["wiring"]]
    if m is None or m.group(1) not in pieces:
        raise Unmodelled(f"the split graph's output {out} is not a piece's return")
    ref = wired["returns"][pieces.index(m.group(1))][int(m.group(2))]
    if "value" not in ref or "cols" in ref:
        raise Unmodelled(f"the backbone's output {ref} is not a whole value")
    return ref


# ------------------------------------------------------------------------------------------------------------- the routes
def probe_routes(shapes: list[list[int]]) -> dict[str, Any]:
    """cuBLAS's route at each (M, K, N) on this GPU, read from the words of `torch.mm(x, w.t(), out=y)` as the served wrapper calls
    it (W rows past the probe's zero): x all ones against `bf16_gemm_check.splitk_count_probe`'s rows reads the split counts a
    partition could have (`fit_cublas_split_k`); of the routes those leave (`cublas_routes_at`), the one whose twin
    (`cublas_route_rows`) computes every word of `route_probe_rows`' spread and cancelling rows (drawn from the generator of
    seed (M, K, N)) is the route, and none or more than one is refused, naming each route's missed words."""
    import torch

    from verity_vllm.commit import challenge as CH
    from verity.ml.definitions.registry import bf16_gemm_check as BC

    to = lambda a: torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).cuda()

    def mm(M: int, K: int, N: int, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
        w = torch.zeros((N, K), dtype=torch.bfloat16, device="cuda")
        w[:len(rows)] = to(rows)
        y = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        torch.mm(to(x), w.t(), out=y)
        return y[:, :len(rows)].cpu().view(torch.int16).numpy().view(np.uint16)

    out: dict[str, Any] = {}
    for M, K, N in shapes:
        probe, x = BC.splitk_count_probe(K, N), np.full((M, K), 0x3F80, np.uint16)
        fits = BC.fit_cublas_split_k(x[:1], probe, mm(M, K, N, x, probe)[:1], out="bf16")
        rows = sorted({0, M // 2, M - 1})
        tests = [(xr[rows], w, mm(M, K, N, xr, w)[rows])
                 for xr, w in BC.route_probe_rows(M, K, min(N, 256), CH.generator([M, K, N]))]
        missed = {json.dumps(r): sum(int(np.count_nonzero(BC.cublas_route_rows(r, a, w) != y)) for a, w, y in tests)
                  for r in BC.cublas_routes_at(fits)}
        exact = [r for r, n in missed.items() if n == 0]
        if len(exact) != 1:
            raise Unmodelled(f"cuBLAS at M={M} K={K} N={N}: words missed by each route {missed}: no one route models it")
        out[f"{M},{K},{N}"] = json.loads(exact[0])
    return {"device": torch.cuda.get_device_name(), "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
            "routes": out}


# ------------------------------------------------------------------------------------------------------------- the command
def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


@dataclass(frozen=True, kw_only=True)
class BuildOptions:
    inductor_dir: str = option("--inductor-dir", help="the serving process's Inductor output code (one directory per wrapper)")
    split_graph: str = option("--split-graph", help="vLLM's computation_graph.py (compilation_config.cache_dir/rank_0_0/backbone)")
    launch_configs: str = option("--launch-configs", help="ServedCompiledSource.launch_configs() of the serving process (json)")
    routes: str | None = option("--routes", default=None, help="cuBLAS routes of the serving GPU (`served routes`); without, every GEMM is missing")
    workload: str = option("--workload", type=workload_arg, help="sweep workload json: the request schedule")
    steps: int | None = option("--steps", type=int, default=None, help="engine steps (default: the workload's expected_engine_steps)")
    capture_sizes: str = option("--capture-sizes", help="compilation_config.cudagraph_capture_sizes, comma-separated")
    max_num_seqs: int = option("--max-num-seqs", type=int)
    cc: str = option("--cc", help="the GPU's compute capability, e.g. 12.0 (served_device: its bindings)")
    sms: int = option("--sms", type=int, help="the GPU's multiprocessor count (the attention scheduler's)")
    max_model_len: int | None = option("--max-model-len", type=int, default=None, help="the engine's max_model_len (FA2's full-graph split count)")
    split_cap: int = option("--split-cap", type=int, default=SPLIT_CAP, help="FA3's static split count under a full graph")
    backbone: str = option("--backbone", default="model", help="the compiled backbone's module path (the served weights' prefix)")
    lm_head: str = option("--lm-head", default="lm_head.weight")
    moe: str | None = option("--moe", default=None, help="the serving process's fused-MoE record (served-match's moe.json)")
    out: str = option("--out")
    cmd: str = field(default="build", init=False)


@dataclass(frozen=True, kw_only=True)
class RoutesOptions:
    needs: str = option("--needs", help="a served Build whose routes_missing to probe")
    out: str = option("--out")
    cmd: str = field(default="routes", init=False)


COMMANDS = {"build": BuildOptions, "routes": RoutesOptions}


def main(a: BuildOptions | RoutesOptions) -> int:
    if a.cmd == "routes":
        shapes = json.load(open(a.needs)).get("routes_missing") or []
        rec = probe_routes(shapes)
        json.dump(rec, open(a.out, "w"), indent=1)
        print(f"[served routes] {len(rec['routes'])} routes on {rec['device']} -> {a.out}")
        return 0
    wl = json.load(open(a.workload))
    steps = a.steps or int(wl.get("expected_engine_steps") or 0)
    if not steps:
        raise UsageError("--steps (or the workload's expected_engine_steps) is required")
    try:
        device = SD.device_of(a.cc.split("."), a.sms, max_model_len=a.max_model_len)
        m = build(a.inductor_dir, split_graph=a.split_graph, launch_configs=json.load(open(a.launch_configs)),
                  routes=json.load(open(a.routes)) if a.routes else {}, schedule=C.schedule_of_workload(wl), steps=steps,
                  capture_sizes=_ints(a.capture_sizes), max_num_seqs=a.max_num_seqs, device=device, split_cap=a.split_cap,
                  backbone=a.backbone, lm_head=a.lm_head, moe=json.load(open(a.moe)) if a.moe else None)
    except Unmodelled as e:
        print(f"[served build] refused: {e}")
        return 3
    with open(a.out, "w") as f:
        json.dump(m, f, indent=1)
    missing = f" (routes missing: {len(m['routes_missing'])})" if not m["complete"] else ""
    print(f"[served build] program {m['program_digest'][:16]} manifest {m['manifest_digest'][:16]} identities {len(m['identities'])} "
          f"steps {len(m['steps_plan'])} kernels {len(m['program']['kernels'])} complete {m['complete']}{missing} -> {a.out}")
    return 0 if m["complete"] else 4
