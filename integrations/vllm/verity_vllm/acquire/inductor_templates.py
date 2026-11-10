"""Inductor's generated Triton kernels in a served-compiled Program, as templates of their source: the only way the served Build
(`acquire.served`) gets from a kernel to Definitions.

A kernel that a served wrapper launches binds to Definitions when its generated source (the text of its `async_compile.triton`
block from `def <name>(` on, with the name read as KERNEL) is one of the templates below (their texts are `acquire.inductor_texts`),
with each `{NAME}` a numeric literal. The first occurrence binds NAME and every later one must repeat it, and the template's
relations among the names must hold. Its decorator must be the template's heuristic and declare exactly the template's signature
(pointer element types included), fp fusion on, ftz not disabled and compute capability 9.0: the PTX that `registry.inductor` was
read from. A reduction kernel's launch config (XBLOCK, R0_BLOCK, num_warps) gives its norm statics, read from the serving process's
record (`ServedCompiledSource.launch_configs`), which must name exactly one config its autotuners ran (the only candidate, or the
winner an autotune timed in that process: the Build is of the process that served, and a Commit process whose autotune chose
otherwise rebuilds another Program). Any other source is refused by name, since no Definition models it.

Each template lists its calls: a Definition with its statics, the kernel parameters it reads (whole, a column range of each row,
or the row of a table that each element of another parameter indexes) and the parameters it writes.

  EMBED_NORM       Embedding_v1{V, H=N}(in_ptr0 ids, in_ptr1 table) -> out_ptr0; InductorRMSNorm_v1(out_ptr0, in_ptr2) -> out_ptr2
  ADD_NORM         InductorAddRMSNorm_v1(in_ptr0, in_ptr1, in_ptr2) -> out_ptr1
  ADD_NORM_COPY    ADD_NORM storing its rows twice: InductorAddRMSNorm_v1(in_ptr0, in_ptr1, in_ptr2) -> out_ptr1, and again -> out_ptr2
  ADD2_NORM        InductorAdd2RMSNorm_v1(in_ptr0, in_ptr1, in_ptr2, in_ptr3) -> (out_ptr1, out_ptr2)
  ADD2_NORM_FINAL  InductorAdd2RMSNormFinal_v1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2) -> in_out_ptr0
  SILU_MUL         InductorSiluMul_v1{I}(in_ptr0) -> out_ptr0
  ROPE_QK          InductorRoPE_v1{NHEADS=KVH, D}(in_ptr0 columns [H*D, (H+KVH)*D), in_ptr2 row in_ptr1) -> out_ptr0, and
                   InductorRoPE_v1{NHEADS=H, D}(in_ptr0 columns [0, H*D), in_ptr2 row in_ptr1) -> out_ptr1
  QK_NORM_ROPE     InductorQKNormRoPE_v1{NHEADS=H, D}(in_ptr0 columns [0, H*D), in_ptr1, in_ptr3 row in_ptr2) -> out_ptr1, and
                   InductorQKNormRoPE_v1{NHEADS=KVH, D}(in_ptr0 columns [H*D, (H+KVH)*D), in_ptr4, in_ptr3 row in_ptr2) -> out_ptr3
"""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array, Tuple, Value
from verity_vllm.acquire.inductor_texts import (ADD2_NORM, ADD2_NORM_FINAL, ADD_NORM, ADD_NORM_COPY, EMBED_NORM, QK_NORM_ROPE, ROPE_QK,
                                                SILU_MUL)
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import inductor as IR


class Unmodelled(ValueError):
    """A generated kernel no template models, or one whose decorator or launch config the Definitions were not read for."""


def bound(defn: Any, **statics: Any) -> dict[str, Any]:
    """A Definition at its statics as a step plan names it: its id, its family's id and the statics (a Definition static by id)."""
    fn = bind(defn, **statics)
    return {"definition": fn.id, "base": defn.id, "statics": {k: getattr(v, "id", v) for k, v in statics.items()}}


@dataclass(frozen=True)
class Read:
    param: str
    cols: Callable[[dict[str, Any]], tuple[int, int]] | None = None
    row_of: str | None = None


@dataclass(frozen=True)
class Call:
    fn: Any
    statics: Callable[[dict[str, Any]], dict[str, Any]]
    reads: tuple[Read, ...]
    writes: tuple[str, ...]


@dataclass(frozen=True)
class Template:
    name: str
    heuristic: str
    signature: dict[str, str]
    text: str
    relations: Callable[[dict[str, Any]], bool]
    calls: tuple[Call, ...]
    scalars: Callable[[dict[str, Any]], dict[str, tuple[str, int]]]
    config: bool = False
    pattern: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern", _pattern(self.text.rstrip()))


_NUM = r"-?\d+(?:\.\d*)?(?:e[-+]?\d+)?"
_HOLE = re.compile(r"\{([A-Z][A-Z0-9]*)\}")


def _pattern(text: str) -> re.Pattern:
    out, seen, pos = [], set(), 0
    for m in _HOLE.finditer(text):
        out.append(re.escape(text[pos:m.start()]))
        name = m.group(1)
        out.append(f"(?P={name})" if name in seen else f"(?P<{name}>{_NUM})")
        seen.add(name)
        pos = m.end()
    out.append(re.escape(text[pos:]))
    return re.compile("".join(out))


def _num(s: str) -> int | float:
    return int(s) if re.fullmatch(r"-?\d+", s) else float(s)


def _sig(ptrs: dict[str, str], scalars: tuple[str, ...], constexprs: tuple[str, ...]) -> dict[str, str]:
    return {**ptrs, **{s: "i32" for s in scalars}, **{c: "constexpr" for c in constexprs}}


_BF16 = "*bf16"
_RED = ("xnumel", "r0_numel")
_RED_C = ("XBLOCK", "R0_BLOCK")


def _norm_statics(lit: dict[str, Any]) -> dict[str, Any]:
    return {"N": lit["N"], "EPS": float(lit["EPS"]), "XBLOCK": lit["XBLOCK"], "R0_BLOCK": lit["R0_BLOCK"], "NUM_WARPS": lit["NUM_WARPS"]}


def _red_scalars(lit: dict[str, Any]) -> dict[str, tuple[str, int]]:
    return {"xnumel": ("tokens", 1), "r0_numel": ("const", lit["N"])}


def _norm_relations(lit: dict[str, Any]) -> bool:
    return isinstance(lit["N"], int) and lit["N"] > 0 and float(lit["NF"]) == lit["N"]


def _rope_relations(lit: dict[str, Any]) -> bool:
    D, H, KVH = lit["D"], lit["H"], lit["KVH"]
    return (all(isinstance(lit[k], int) and lit[k] > 0 for k in ("D", "H", "KVH", "HALF", "P")) and D == 2 * lit["HALF"]
            and lit["QLEN"] == H * D and lit.get("KLEN", KVH * D) == KVH * D and lit["KHI"] == H * D + lit["HALF"]
            and lit["QKV"] == (H + 2 * KVH) * D)


def _head_norm_statics(heads: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda lit: {"NHEADS": lit[heads], "D": lit["D"], "EPS": float(lit["EPS"]), "XBLOCK": lit["XBLOCK"],
                        "R0_BLOCK": lit["R0_BLOCK"], "NUM_WARPS": lit["NUM_WARPS"]}


TEMPLATES = (
    Template("EMBED_NORM", "reduction",
             _sig({"in_ptr0": "*i32", "in_ptr1": _BF16, "in_ptr2": _BF16, "out_ptr0": _BF16, "out_ptr2": _BF16}, _RED, _RED_C),
             EMBED_NORM, lambda lit: _norm_relations(lit) and isinstance(lit["V"], int) and lit["V"] > 0,
             (Call(b1.Embedding, lambda lit: {"V": lit["V"], "H": lit["N"]}, (Read("in_ptr0"), Read("in_ptr1")), ("out_ptr0",)),
              Call(IR.InductorRMSNorm, _norm_statics, (Read("out_ptr0"), Read("in_ptr2")), ("out_ptr2",))), _red_scalars, config=True),
    Template("ADD_NORM", "reduction", _sig({"in_ptr0": _BF16, "in_ptr1": _BF16, "in_ptr2": _BF16, "out_ptr1": _BF16}, _RED, _RED_C),
             ADD_NORM, _norm_relations,
             (Call(IR.InductorAddRMSNorm, _norm_statics, (Read("in_ptr0"), Read("in_ptr1"), Read("in_ptr2")), ("out_ptr1",)),), _red_scalars,
             config=True),
    Template("ADD_NORM_COPY", "reduction",
             _sig({"in_ptr0": _BF16, "in_ptr1": _BF16, "in_ptr2": _BF16, "out_ptr1": _BF16, "out_ptr2": _BF16}, _RED, _RED_C),
             ADD_NORM_COPY, _norm_relations,
             tuple(Call(IR.InductorAddRMSNorm, _norm_statics, (Read("in_ptr0"), Read("in_ptr1"), Read("in_ptr2")), (o,))
                   for o in ("out_ptr1", "out_ptr2")), _red_scalars, config=True),
    Template("ADD2_NORM", "reduction",
             _sig({"in_ptr0": _BF16, "in_ptr1": _BF16, "in_ptr2": _BF16, "in_ptr3": _BF16, "out_ptr1": _BF16, "out_ptr2": _BF16}, _RED, _RED_C),
             ADD2_NORM, _norm_relations,
             (Call(IR.InductorAdd2RMSNorm, _norm_statics, (Read("in_ptr0"), Read("in_ptr1"), Read("in_ptr2"), Read("in_ptr3")),
                   ("out_ptr1", "out_ptr2")),), _red_scalars, config=True),
    Template("ADD2_NORM_FINAL", "reduction",
             _sig({"in_out_ptr0": _BF16, "in_ptr0": _BF16, "in_ptr1": _BF16, "in_ptr2": _BF16}, _RED, _RED_C),
             ADD2_NORM_FINAL, _norm_relations,
             (Call(IR.InductorAdd2RMSNormFinal, _norm_statics, (Read("in_out_ptr0"), Read("in_ptr0"), Read("in_ptr1"), Read("in_ptr2")),
                   ("in_out_ptr0",)),), _red_scalars, config=True),
    Template("SILU_MUL", "pointwise", _sig({"in_ptr0": _BF16, "out_ptr0": _BF16}, ("xnumel",), ("XBLOCK",)),
             SILU_MUL, lambda lit: isinstance(lit["I"], int) and lit["I"] > 0 and lit["I2"] == 2 * lit["I"],
             (Call(IR.InductorSiluMul, lambda lit: {"I": lit["I"]}, (Read("in_ptr0"),), ("out_ptr0",)),),
             lambda lit: {"xnumel": ("tokens", lit["I"])}),
    Template("ROPE_QK", "pointwise",
             _sig({"in_ptr0": _BF16, "in_ptr1": "*i64", "in_ptr2": _BF16, "out_ptr0": _BF16, "out_ptr1": _BF16}, ("xnumel_0", "xnumel_1"), ("XBLOCK",)),
             ROPE_QK, _rope_relations,
             (Call(IR.InductorRoPE, lambda lit: {"NHEADS": lit["KVH"], "D": lit["D"]},
                   (Read("in_ptr0", cols=lambda lit: (lit["QLEN"], lit["QLEN"] + lit["KLEN"])), Read("in_ptr2", row_of="in_ptr1")), ("out_ptr0",)),
              Call(IR.InductorRoPE, lambda lit: {"NHEADS": lit["H"], "D": lit["D"]},
                   (Read("in_ptr0", cols=lambda lit: (0, lit["QLEN"])), Read("in_ptr2", row_of="in_ptr1")), ("out_ptr1",))),
             lambda lit: {"xnumel_0": ("tokens", lit["KLEN"]), "xnumel_1": ("tokens", lit["QLEN"])}),
    Template("QK_NORM_ROPE", "reduction",
             _sig({"in_ptr0": _BF16, "in_ptr1": _BF16, "in_ptr2": "*i64", "in_ptr3": _BF16, "in_ptr4": _BF16, "out_ptr1": _BF16,
                   "out_ptr3": _BF16}, ("xnumel_0", "xnumel_1"), _RED_C),
             QK_NORM_ROPE, lambda lit: _rope_relations(lit) and float(lit["NF"]) == lit["D"],
             (Call(IR.InductorQKNormRoPE, _head_norm_statics("H"),
                   (Read("in_ptr0", cols=lambda lit: (0, lit["QLEN"])), Read("in_ptr1"), Read("in_ptr3", row_of="in_ptr2")), ("out_ptr1",)),
              Call(IR.InductorQKNormRoPE, _head_norm_statics("KVH"),
                   (Read("in_ptr0", cols=lambda lit: (lit["QLEN"], lit["QLEN"] + lit["KVH"] * lit["D"])), Read("in_ptr4"),
                    Read("in_ptr3", row_of="in_ptr2")), ("out_ptr3",))),
             lambda lit: {"xnumel_0": ("tokens", lit["H"]), "xnumel_1": ("tokens", lit["KVH"])}, config=True),
)
TEMPLATE = {t.name: t for t in TEMPLATES}

_BLOCK = re.compile(r"^\w+ = async_compile\.triton\('(\w+)', '''(.*?)''', device_str='cuda'\)$", re.M | re.S)
_JIT = re.compile(r"^@triton\.jit\ndef (\w+)\(", re.M)
#: the triton_meta a kernel of each compute capability carries; a cc not here has no binding
FLAGS = {cc: ("'enable_fp_fusion': True", "'disable_ftz': False", f"cc={cc[0]}{cc[1]}") for cc in ((9, 0), (10, 0), (12, 0))}


def kernel_sources(wrapper_src: str) -> dict[str, str]:
    """{launch-site name: generated source} of a wrapper's `async_compile.triton` blocks (a name's last block, the Runner's)."""
    return {m.group(1): m.group(2) for m in _BLOCK.finditer(wrapper_src)}


def source_sha256(source: str) -> str:
    """The kernel module's content hash: Inductor writes the block's text as the module it loads."""
    return hashlib.sha256(source.encode()).hexdigest()


def launch_config(record: dict[str, Any] | None, source: str) -> dict[str, Any]:
    """The one launch config the serving process ran the kernel with: the config every settled autotuner of it holds (an
    autotuned kernel's winner), one of its candidates."""
    rec = (record or {}).get(source_sha256(source))
    if rec is None:
        raise Unmodelled(f"no launch config recorded for kernel source {source_sha256(source)[:16]}")
    cands, ran = rec.get("candidates") or [], rec.get("executed") or []
    if len(ran) != 1 or (cands and ran[0] not in cands):
        raise Unmodelled(f"kernel source {source_sha256(source)[:16]}: {len(cands)} candidate configs, its autotuners hold {ran}; one "
                         f"config every instance ran is required")
    return ran[0]


def _function_text(rest: str) -> str:
    """The jit function's text: its lines up to the first one at column 0 (a combo kernel's module goes on with its benchmark
    harness, host code that never runs in the served process)."""
    lines = rest.split("\n")
    end = next((i for i, l in enumerate(lines[1:], 1) if l and not l[0].isspace()), len(lines))
    return "\n".join(lines[:end]).rstrip()


def recognize(name: str, source: str, config: dict[str, Any] | None = None, *, cc: tuple[int, int]) -> dict[str, Any]:
    """The template a generated kernel source of compute capability `cc` is, its literals, and its calls with bound Definition ids;
    `Unmodelled` otherwise."""
    if cc not in FLAGS:
        raise Unmodelled(f"{name}: Triton kernels for cc={cc[0]}{cc[1]} have no binding (FLAGS: cc {sorted(FLAGS)})")
    m = _JIT.search(source)
    if m is None:
        raise Unmodelled(f"{name}: no @triton.jit def")
    head, body = source[:m.start()], "def KERNEL(" + _function_text(source[m.end():])
    t, got = next(((t, g) for t in TEMPLATES if (g := t.pattern.fullmatch(body))), (None, None))
    if t is None:
        raise Unmodelled(f"{name}: its source is none of the templates {[t.name for t in TEMPLATES]}")
    heur = re.search(r"^@triton_heuristics\.(\w+)\(", head, re.M)
    sig = re.search(r"'signature': (\{[^{}]*\})", head)
    if heur is None or heur.group(1) != t.heuristic:
        raise Unmodelled(f"{name} ({t.name}): heuristic {heur and heur.group(1)}, the template's is {t.heuristic}")
    if sig is None or ast.literal_eval(sig.group(1)) != t.signature:
        raise Unmodelled(f"{name} ({t.name}): signature {sig and sig.group(1)} is not {t.signature}")
    missing = [f for f in FLAGS[cc] if f not in head]
    if missing:
        raise Unmodelled(f"{name} ({t.name}): its triton_meta lacks {missing}")
    return {**instantiate(name, t, {k: _num(v) for k, v in got.groupdict().items()}, config), "source_sha256": source_sha256(source)}


def holes(t: Template) -> list[str]:
    """The template's literal names, in the order its text first names them."""
    return list(dict.fromkeys(m.group(1) for m in _HOLE.finditer(t.text)))


def instantiate(name: str, t: Template, lit: dict[str, Any], config: dict[str, Any] | None) -> dict[str, Any]:
    """Template `t` at literals `lit` (its holes) and launch config: its calls with bound Definition ids (`recognize` less the source,
    which `predict.compiled` reads from a model's config)."""
    lit = dict(lit)
    if not t.relations(lit):
        raise Unmodelled(f"{name} ({t.name}): literals {lit} break the template's relations")
    if t.config:
        if config is None:
            raise Unmodelled(f"{name} ({t.name}): its launch config is a static and none was given")
        kw = config.get("kwargs") or {}
        lit.update(XBLOCK=kw.get("XBLOCK"), R0_BLOCK=kw.get("R0_BLOCK"), NUM_WARPS=config.get("num_warps"))
    calls = []
    for c in t.calls:
        try:
            spec = bind(c.fn, **c.statics(lit))
            if "NUM_WARPS" in (b := spec.bindings):
                IR.norm_layout(b["N"] if "N" in b else b["D"], b["EPS"], b["XBLOCK"], b["R0_BLOCK"], b["NUM_WARPS"])
        except (TypeError, ValueError) as e:
            raise Unmodelled(f"{name} ({t.name}): {e}") from None
        ins, out = spec.signature()
        reads = []
        for r, (_n, ty) in zip(c.reads, ins):
            d: dict[str, Any] = {"param": r.param, "numel": numel(ty)}
            if r.cols is not None:
                d["cols"] = list(r.cols(lit))
            if r.row_of is not None:
                d["row_of"] = r.row_of
            reads.append(d)
        outs = out.fields if isinstance(out, Tuple) else (out,)
        if len(ins) != len(c.reads) or len(outs) != len(c.writes):
            raise Unmodelled(f"{name} ({t.name}): {spec.id} takes {len(ins)} inputs and gives {len(outs)} outputs, the template "
                             f"reads {len(c.reads)} and writes {len(c.writes)}")
        calls.append({"definition": spec.id, "base": spec.base_id, "statics": c.statics(lit), "reads": reads,
                      "writes": [{"param": p, "numel": numel(ty)} for p, ty in zip(c.writes, outs)]})
    return {"template": t.name, "literals": lit, "config": config,
            "params": [p for p, ty in t.signature.items() if ty != "constexpr"],
            "scalars": {p: list(v) for p, v in t.scalars(lit).items()}, "calls": calls}


def numel(ty: Any) -> int:
    """Elements of a Definition's per-row input or output type (an array of arrays of values)."""
    if isinstance(ty, Array):
        return ty.n * numel(ty.elem)
    if isinstance(ty, Value):
        return 1
    raise Unmodelled(f"type {ty} is not an array of values")
