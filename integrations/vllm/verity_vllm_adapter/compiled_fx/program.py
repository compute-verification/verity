"""From Inductor's generated wrapper to description nodes.

The wrapper (``call``) of a compiled piece is a straight-line program: buffer allocations, kernel
launches (Inductor Triton kernels ``triton_*.run(...)``, the user Triton GEMM
``matmul_kernel_persistent_N.run(...)``), the inserted ``torch.ops.verity.record.default(buf, id)`` lines
and ``del``s. Every recorded buffer is written by exactly one launch, so *the launch is the node*:
its output buffers are the node's ports, its input buffers/placeholders are the node's args, its
Triton source (by normalised hash) is the relation's identity. This module parses that program and
turns it into description nodes in the eager adapter's naming convention
(``layers.L.self_attn.qkv_proj.gemm``, ``layers.L.post_attention_layernorm.rmsnorm``, ...), so the
existing verifier machinery (family chains, capture predicates, Table 2 rows) applies unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm_adapter.compiled_fx import kernels as K
from verity_vllm_adapter.compiled_fx.fx_pass import ID_STRIDE, FxIdTable, IdEntry, PieceInfo

_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_LAUNCH = re.compile(r"^\s*([A-Za-z0-9_]+)\.run\((.*)\)\s*$")
# [COMP, CX1] an extern custom op the graph keeps as a node (Inductor's FallbackKernel): `buf1 = torch.ops.vllm.all_reduce.default(buf0, 'tp')`
# (functional, assignment form) or `torch.ops._C.some_op.default(buf0, ...)` (mutating, statement form). Not a Triton launch, but
# the same shape of node: the buffers it writes are the ports, the buffers it reads the args. The eager TP capture records
# these ops through the dispatcher; the compiled capture sees them here.
_EXTERN = re.compile(r"^\s*(?:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*)?torch\.ops\.(?!verity\.record)([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\.(?:default|out)\((.*)\)\s*$")
_RECORD = re.compile(r"^\s*torch\.ops\.verity\.record\.default\(([A-Za-z0-9_]+),\s*(\d+)\)\s*$")
# [COMP, CX1] a plain buffer alias: Inductor names an extern op's functional result twice (`buf3 = torch.ops.vllm.all_reduce.default(...)`,
# `buf4 = buf3`) and records / consumers use the second name. Not the storage-reuse form (`buf22 = buf14; del buf14  # reuse`),
# which introduces a fresh logical buffer. Without this the all-reduce outputs looked unwritten and the next Triton launch
# claimed them as its own outputs (the first cx1 commit attempt: a fused norm named layers.1.mlp.down_proj.rmsnorm).
_ALIAS = re.compile(r"^\s*(buf\d+)\s*=\s*(buf\d+)\s*$")
_REUSE = re.compile(r"^\s*(buf\d+)\s*=\s*(buf\d+)\s*;\s*del\s+\2\s*#\s*reuse\s*$")
_DEF_PARAMS = re.compile(r"^def\s+[A-Za-z0-9_]+\(([^)]*)\)", re.M)
_RETURN = re.compile(r"^\s*return\s*\((.*)\)\s*$")
_REINTERP = re.compile(r"^reinterpret_tensor\((.*)\)$")
_LAYER = re.compile(r"layers\.(\d+)")

FAMILY_BY_KERNEL = [
    ("matmul_kernel_persistent", "gemm", "gemm"),
    ("embedding", "embedding", "embedding"),
    ("rms_norm", "rmsnorm", "rmsnorm"),
    ("silu", "act_mul", "silu_mul"),
    # extern custom ops (kernel_name = "<namespace>.<op>"); the TP collectives and vLLM's vocab-parallel embedding
    ("vllm.all_reduce", "all_reduce", "allreduce"),
    ("vllm.all_gather", "all_gather", "other"),
    ("vocab_parallel_embedding", "embedding", "embedding"),
]


def split_args(s: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


@dataclass
class Launch:
    index: int
    kernel_var: str  # e.g. triton_per_fused_fused_add_rms_norm_2 / matmul_kernel_persistent_0
    kernel_name: str  # name in async_compile.triton(...)
    args: list[str]
    line: str
    outputs: list[str] = field(default_factory=list)  # buffers first written here (in arg order)
    inputs: list[str] = field(default_factory=list)
    # [COMP, CX1] in-place args: output buffer token -> the recorded buffer whose storage it reuses and whose value the kernel READS
    # first (Inductor `in_out_ptr` params; `buf23 = buf19; del buf19  # reuse` then `kernel.run(buf23, ...)` with in_out_ptr0 = buf23:
    # the final norm under TP reads the down_proj all-reduce output in place and overwrites it with the normed row)
    inout: dict[str, str] = field(default_factory=dict)


@dataclass
class Program:
    piece: int
    wrapper: str
    launches: list[Launch]
    record_buffer: dict[int, str]  # id -> buffer/placeholder token
    buffer_record: dict[str, int]
    syms: dict[str, str]  # assignment name -> expression (in terms of s72 & literals)
    kernel_src: dict[str, str]  # kernel_var -> source
    kernel_hash: dict[str, str]  # kernel_var -> normalised hash
    kernel_name: dict[str, str]
    returned: list[str]
    m_symbol: str = "s72"


def _select_call_body(txt: str) -> str:
    """The dynamic-shape ``call`` (the one assigning ``s72 = argN_1``); wrappers can hold a static copy too."""
    parts = re.split(r"\n\s*def call\((?:self, )?args\):", txt)
    bodies = parts[1:]
    if not bodies:
        return ""
    for b in bodies:
        if re.search(r"^\s*s\d+ = arg\d+_1", b, re.M):
            return _cut_body(b)
    return _cut_body(bodies[-1])


def _cut_body(b: str) -> str:
    """Stop at the end of ``call`` (before the module-level tail: Runner, get_args, benchmark_*)."""
    for marker in ("\nrunner = Runner", "\ndef get_args", "\ndef benchmark_compiled_module", "\ndef benchmark_all_configs", "\nclass ", "\ndef "):
        if marker in b:
            b = b.split(marker, 1)[0]
    return b


def _resolve(expr: str, syms: dict[str, str], depth: int = 0) -> str:
    if depth > 8:
        return expr

    def sub(m: re.Match) -> str:
        name = m.group(0)
        if re.fullmatch(r"arg\d+_1", name) or (name in syms and re.fullmatch(r"arg\d+_1", syms[name].strip())):
            return name  # the symbolic row count stays symbolic (the caller substitutes M); aliases (s80 = s72) resolve
        if name in syms:
            return "(" + _resolve(syms[name], syms, depth + 1) + ")"
        return name

    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sub, expr)


def parse_program(path: Path) -> Program | None:
    txt = path.read_text()
    body = _select_call_body(txt)
    if not body or "verity.record" not in body:
        return None
    kernel_src: dict[str, str] = {}
    kernel_name: dict[str, str] = {}
    for m in re.finditer(r"^([A-Za-z0-9_]+) = async_compile\.triton\('([A-Za-z0-9_]+)',\s*'''(.*?)'''", txt, re.S | re.M):
        kernel_src.setdefault(m.group(1), m.group(3))
        kernel_name.setdefault(m.group(1), m.group(2))
    launches: list[Launch] = []
    record_buffer: dict[int, str] = {}
    syms: dict[str, str] = {}
    written: set[str] = set()
    returned: list[str] = []
    alias: dict[str, str] = {}  # alias name -> the written buffer it names
    reuse_of: dict[str, str] = {}  # fresh logical buffer -> the (canonical) written buffer whose storage it takes over

    def params_of(kvar: str) -> list[str]:
        src = kernel_src.get(kvar, "")
        m = _DEF_PARAMS.search(src)
        return [x.split(":")[0].strip() for x in m.group(1).split(",")] if m else []

    def canon(tok: str) -> str:
        return re.sub(r"\bbuf\d+\b", lambda m: alias.get(m.group(0), m.group(0)), tok)

    piece = -1
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("del ") or s.startswith("assert_size_stride") or s.startswith("assert_alignment") or s.startswith("with ") or s.startswith("torch.cuda."):
            continue
        mr = _RECORD.match(s)
        if mr:
            rid = int(mr.group(2))
            record_buffer[rid] = canon(mr.group(1))
            piece = rid // ID_STRIDE
            continue
        mal = _ALIAS.match(s)
        if mal and mal.group(2) in written:
            alias[mal.group(1)] = alias.get(mal.group(2), mal.group(2))
            written.add(mal.group(1))
            continue
        mre = _REUSE.match(s)
        if mre and canon(mre.group(2)) in written:
            reuse_of[mre.group(1)] = canon(mre.group(2))
            # falls through: the assignment is also a sym-table entry, as before
        ml = _LAUNCH.match(s)
        if ml:
            args = [canon(a) for a in split_args(ml.group(2)) if not a.startswith("stream=") and not a.startswith("grid=")]
            L = Launch(index=len(launches), kernel_var=ml.group(1), kernel_name=kernel_name.get(ml.group(1), ml.group(1)), args=args, line=s)
            params = params_of(ml.group(1))
            for ai, a in enumerate(args):
                base = _base_token(a)
                if base is None:
                    continue
                if base.startswith("buf") and base not in written:
                    L.outputs.append(base)
                    written.add(base)
                    if ai < len(params) and params[ai].startswith("in_out_ptr") and base in reuse_of:
                        L.inout[base] = reuse_of[base]
                        L.inputs.append(reuse_of[base])
                else:
                    L.inputs.append(base)
            launches.append(L)
            continue
        me = _EXTERN.match(s)
        if me:
            assigned, ns, opname, inner = me.group(1), me.group(2), me.group(3), me.group(4)
            qual = f"{ns}.{opname}"
            args = [canon(a) for a in split_args(inner) if "=" not in a.split("(")[0]]  # drop keyword args
            # the functional result comes first in the launch's arg list so it is port 0 (an output-variant op's `out` buffer is
            # an unwritten buf among the args, handled like a Triton launch's)
            if assigned is not None and re.fullmatch(r"buf\d+", assigned):
                args = [assigned] + args
            L = Launch(index=len(launches), kernel_var=f"extern_{qual.replace('.', '_')}", kernel_name=qual, args=args, line=s)
            for a in args:
                base = _base_token(a)
                if base is None:
                    continue
                if base.startswith("buf") and base not in written:
                    L.outputs.append(base)
                    written.add(base)
                else:
                    L.inputs.append(base)
            launches.append(L)
            continue
        mret = _RETURN.match(s)
        if mret:
            returned = [_base_token(canon(a)) or canon(a) for a in split_args(mret.group(1))]
            continue
        ma = _ASSIGN.match(s)
        if ma:
            name, expr = ma.group(1), ma.group(2)
            if expr.startswith("empty_strided") or expr.startswith("copy_if_misaligned") or expr.startswith("get_raw_stream") or "=" in expr:
                continue
            syms[name] = expr
    if piece < 0:
        return None
    # normalise every sym to an expression in the M symbol: s80 = s72 etc.
    m_sym = next((n for n, e in syms.items() if re.fullmatch(r"arg\d+_1", e)), "s72")
    resolved = {}
    for n, e in syms.items():
        r = _resolve(e, {k: v for k, v in syms.items() if not re.fullmatch(r"arg\d+_1", v)})
        resolved[n] = r
    return Program(
        piece=piece, wrapper=str(path), launches=launches, record_buffer=record_buffer,
        buffer_record={v: k for k, v in record_buffer.items()}, syms=resolved, kernel_src=kernel_src,
        kernel_hash={k: K._sha(K.normalise_kernel(v)) for k, v in kernel_src.items()}, kernel_name=kernel_name, returned=returned, m_symbol=m_sym,
    )


def _base_token(a: str) -> str | None:
    m = _REINTERP.match(a)
    if m:
        inner = split_args(m.group(1))
        return inner[0] if inner else None
    if re.fullmatch(r"(buf\d+|arg\d+_1)", a):
        return a
    return None


def load_programs(cache_dir: Path) -> dict[int, Program]:
    out: dict[int, Program] = {}
    for p in sorted(cache_dir.rglob("*.py")):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        if "verity.record" not in txt or "def call(" not in txt:
            continue
        prog = parse_program(p)
        if prog is not None and prog.piece not in out:
            out[prog.piece] = prog
    return out


# --------------------------------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------------------------------


@dataclass
class PortSpec:
    port: int
    record_id: int  # template id
    dtype: str
    row_shape: list[int]
    launch_index: int  # position in launch args


@dataclass
class NodeSpec:
    """A node template for one compiled piece (layer numbers relative to the template piece)."""

    op: str
    module_path: str
    family: str
    op_type: str
    ports: list[PortSpec]
    args: list[dict[str, Any]]  # description args (template op names; shifted at instantiation)
    launch: list[dict[str, Any]]
    kernel_var: str
    kernel_name: str
    kernel_hash: str
    consts: dict[str, Any]
    relation: str
    runtime_impl: str
    arg_cols: dict[str, list[int]] = field(default_factory=dict)
    line: str = ""


def shift_layers(s: str, shift: int) -> str:
    if shift == 0:
        return s
    return _LAYER.sub(lambda m: f"layers.{int(m.group(1)) + shift}", s)


def family_of(kernel_name: str, entries: list[IdEntry]) -> tuple[str, str]:
    for needle, fam, op_type in FAMILY_BY_KERNEL:
        if needle in kernel_name:
            return fam, op_type
    # generic pointwise kernel: rope if it reads the cos_sin_cache / produces the rotary module's outputs
    if any("rotary_emb" in e.module_path for e in entries):
        return "rope", "rope"
    return "fused", "other"


OP_SUFFIX = {"gemm": "gemm", "embedding": "embedding", "rmsnorm": "rmsnorm", "rope": "rope", "act_mul": "act_mul", "attn": "attn", "fused": "fused",
             "all_reduce": "all_reduce", "all_gather": "all_gather"}


def build_nodes(prog: Program, table: FxIdTable, piece_info: PieceInfo) -> tuple[list[NodeSpec], dict[int, tuple[str, int]], dict[str, Any]]:
    """Nodes of one template piece; also ``record id -> (op, port)`` and the piece's input bindings."""
    binding = {b["postgrad"]: b for b in piece_info.placeholder_binding}
    entries = {e.id: e for e in table.entries.values() if e.piece == prog.piece}
    id_of_buf = prog.buffer_record
    nodes: list[NodeSpec] = []
    port_of: dict[int, tuple[str, int]] = {}
    # which launch wrote each buffer
    producer: dict[str, tuple[NodeSpec, int]] = {}
    inputs_info = {"attention_output": None, "residual_in": None}
    for tok, b in binding.items():
        if b.get("kind") == "activation" and tok in id_of_buf:
            e = entries[id_of_buf[tok]]
            if b.get("name", "").startswith("output"):
                inputs_info["attention_output"] = {"record_id": e.id, "module_path": e.consumers[0]["module_path"] if e.consumers else "", "shape": e.shape[1:], "dtype": e.dtype}
            else:
                inputs_info["residual_in"] = {"record_id": e.id, "shape": e.shape[1:], "dtype": e.dtype}
    for L in prog.launches:
        out_entries = [entries[id_of_buf[o]] for o in L.outputs if o in id_of_buf]
        if not out_entries:
            continue
        fam, op_type = family_of(L.kernel_name, out_entries)
        # port order: rmsnorm-family -> the normed output (aten.mul) first, the residual (convert_element_type) after;
        # rope -> larger row first (q before k); otherwise launch order
        def port_key(e: IdEntry) -> tuple:
            if fam == "rmsnorm":
                return (1 if e.target.startswith("prims.convert_element_type") else 0, e.k)
            if fam == "rope":
                return (-_numel(e.shape[1:]), e.node)
            if fam == "embedding":
                return (1 if "layernorm" in e.module_path else 0, e.k)
            return (e.k,)

        ordered = sorted(out_entries, key=port_key)
        head = ordered[0]
        module_path = head.module_path or (head.producer_module_path or "")
        op = f"{module_path}.{OP_SUFFIX.get(fam, fam)}" if module_path else f"piece{prog.piece}.launch{L.index}.{fam}"
        ports = [PortSpec(port=i, record_id=e.id, dtype=e.dtype, row_shape=[int(d) for d in e.shape[1:]], launch_index=L.args.index(_arg_token_for(L, prog.record_buffer[e.id])))
                 for i, e in enumerate(ordered)]
        args: list[dict[str, Any]] = []
        launch: list[dict[str, Any]] = []
        arg_cols: dict[str, list[int]] = {}
        n_in = 0
        for ai, a in enumerate(L.args):
            base = _base_token(a)
            reinterp = _reinterp_meta(a, prog)
            if base is None:
                # scalar: literal int, string literal (an extern op's group name) or symbolic expression
                if re.fullmatch(r"-?\d+", a):
                    launch.append({"kind": "const", "value": int(a)})
                elif re.fullmatch(r"'[^']*'|\"[^\"]*\"|True|False|None", a):
                    launch.append({"kind": "const", "value": a.strip("'\"") if a[0] in "'\"" else a})
                else:
                    launch.append({"kind": "sym", "expr": _resolve(a, prog.syms).replace(prog.m_symbol, "M")})
                continue
            if base in [prog.record_buffer[e.id] for e in ordered]:
                port = next(p.port for p in ports if prog.record_buffer[p.record_id] == base)
                if base in L.inout and L.inout[base] in producer:
                    # in-place: the kernel reads the reused buffer's recorded value before overwriting it
                    name = f"in{n_in}"
                    n_in += 1
                    pnode, pport = producer[L.inout[base]]
                    args.append({"name": name, "source": "node", "op": pnode.op, "port": pport, "in_place_with_output": port})
                    launch.append({"kind": "output", "port": port, "in_place_input": name})
                    continue
                launch.append({"kind": "output", "port": port})
                continue
            name = f"in{n_in}"
            n_in += 1
            if base.startswith("buf"):
                rid = id_of_buf.get(base)
                if rid is None or base not in producer:
                    args.append({"name": name, "source": "const", "value": f"unrecorded:{base}"})
                    launch.append({"kind": "input", "arg": name, "unbound": True})
                    continue
                pnode, pport = producer[base]
                args.append({"name": name, "source": "node", "op": pnode.op, "port": pport})
                launch.append({"kind": "input", "arg": name, **({"reinterpret": reinterp} if reinterp else {})})
            else:  # placeholder
                b = binding.get(base, {})
                kind = b.get("kind")
                if kind in ("param", "buffer"):
                    args.append({"name": name, "source": "weight", "param": _param_name(b["name"])})
                    launch.append({"kind": "input", "arg": name, **({"reinterpret": reinterp} if reinterp else {})})
                elif kind == "input" and b.get("name") == "positions":
                    args.append({"name": name, "source": "const", "value": "positions"})
                    launch.append({"kind": "positions", "arg": name})
                elif kind == "activation":
                    rid = id_of_buf.get(base)
                    if b.get("name", "").startswith("output"):
                        # attention output: produced by the attention op outside the graph
                        e = entries[rid]
                        attn_mod = e.consumers[0]["module_path"] if e.consumers else ""
                        args.append({"name": name, "source": "node", "op": f"{attn_mod}.attn", "port": 0})
                    elif b.get("name") in ("input_ids",) or (rid is None and entries and False):
                        args.append({"name": name, "source": "token", "which": "input"})
                    else:
                        args.append({"name": name, "source": "node", "op": "<residual_in>", "port": 0})
                    launch.append({"kind": "input", "arg": name, **({"reinterpret": reinterp} if reinterp else {})})
                elif kind == "sym":
                    launch.append({"kind": "sym", "expr": str(b.get("sym") or base).replace(prog.m_symbol, "M")})
                else:
                    # int inputs (input_ids) are not recorded: token source
                    if b.get("dtype") in ("int64", "int32") and (b.get("shape") or [None])[0] == prog.m_symbol:
                        args.append({"name": name, "source": "token", "which": "input"})
                        launch.append({"kind": "token", "arg": name, "dtype": b.get("dtype")})
                    else:
                        args.append({"name": name, "source": "const", "value": f"unbound:{base}:{b}"})
                        launch.append({"kind": "input", "arg": name, "unbound": True})
        extern = L.kernel_var.startswith("extern_")
        relation = f"inductor.{op_type}.same_runtime.v1"
        if extern:
            # [COMP, CX1] an extern custom op inside the compiled graph: the same runtime op the eager capture records through the
            # dispatcher (the collectives' relations are the eager TP row's; tp_merge names their per-rank operands)
            relation = {"all_reduce": "allreduce.bf16.v1", "all_gather": "allgather.copy.v1", "embedding": "embedding.vocab_parallel.v1"}.get(fam, relation)
            runtime_impl = f"cuda:torch.ops.{L.kernel_name} (extern node of the Inductor graph, not lowered; the same custom op as eager execution)"
        else:
            runtime_impl = (f"inductor:{L.kernel_name} (Triton, source sha {prog.kernel_hash.get(L.kernel_var, '?')})" if fam != "gemm"
                            else f"inductor:user-triton {L.kernel_name} (vllm ...batch_invariant.matmul_kernel_persistent through triton_kernel_wrapper)")
        node = NodeSpec(op=op, module_path=module_path, family=fam, op_type=op_type, ports=ports, args=args, launch=launch, kernel_var=L.kernel_var,
                        kernel_name=L.kernel_name, kernel_hash=prog.kernel_hash.get(L.kernel_var, ""), consts={}, relation=relation, runtime_impl=runtime_impl,
                        arg_cols=arg_cols, line=L.line)
        node.consts = {"kernel": ({"name": L.kernel_name, "extern": True} if extern else
                                  {"name": L.kernel_name, "sha256_16": node.kernel_hash, "file": f"kernels/{node.kernel_hash}.py"}),
                       "launch": launch, "m_symbol": "M", "outputs": [{"port": p.port, "dtype": p.dtype, "row_shape": p.row_shape} for p in ports]}
        nodes.append(node)
        for p in ports:
            producer[prog.record_buffer[p.record_id]] = (node, p.port)
            port_of[p.record_id] = (node.op, p.port)
    return nodes, port_of, inputs_info


def _arg_token_for(L: Launch, buf: str) -> str:
    for a in L.args:
        if _base_token(a) == buf:
            return a
    raise KeyError(buf)


def _reinterp_meta(a: str, prog: Program) -> dict[str, Any] | None:
    m = _REINTERP.match(a)
    if not m:
        return None
    parts = split_args(m.group(1))
    if len(parts) < 4:
        return None

    def tup(s: str) -> list[Any]:
        s = s.strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
        return [x.strip().replace(prog.m_symbol, "M") for x in s.split(",") if x.strip()]

    return {"shape": tup(parts[1]), "stride": tup(parts[2]), "offset": parts[3].strip()}


def _param_name(name: str) -> str:
    name = name.replace("modules_", "")
    if name.startswith("modules."):
        name = name[len("modules."):]
    return name


def _numel(shape: list[Any]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n
