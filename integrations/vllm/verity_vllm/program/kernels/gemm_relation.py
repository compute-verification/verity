"""``independent`` GEMM relations for the replay checker (contract §7, Table 2 row class
``the same GEMM inside the replayed model``).

``independent_gemm(node, opened_inputs) -> bytes`` predicts the committed bytes of one GEMM
output row (the default VU: one node instance at one position) from the description node and
the authenticated openings of its inputs, using nothing but the explicit numerical model
measured in this lane:

* per instruction: Ampere ``mma.sync.m16n8k16`` BF16 x BF16 -> FP32 semantics
  (``veritor.core.silicon.AMPERE_BF16_M16N8K16`` == ``verity_vllm.program.kernels.mma.AMPERE_BF16``;
  Ada's BF16 instruction is parameter-identical, ``verity.ml.circuits.silicon.models.ADA_BF16_M16N8K16``);
* across K: one chain over the K/16 tiles in ascending k order from a zero FP32 accumulator
  (vLLM's batch-invariant ``matmul_kernel_persistent``) -- or the slice/split-K description the
  node carries for a cuBLAS kernel that needs one;
* epilogue: ``none`` | ``bias_fp32_in_kernel`` (``aten::addmm`` override, cuBLASLt bias epilogue)
  | ``bias_bf16_after`` (``aten::linear`` override: RNE to BF16, then BF16 + bias as one FP32 add,
  RNE again); then round-to-nearest-even to BF16.

The checker never runs the runtime; it is C++/OpenMP over the opened words (``cpu_model``), or
pure Python for the scalar gate ``independent_gemm_coordinate`` (one output coordinate from the
opened input row and one opened weight column).

Interface.  With W2's harness importable (``verity_vllm.program.kernels.relations``) this module fills the
registered slots ``gemm.bf16.v1`` and ``lm_head.bf16.v1`` at import time and adds
``gemm.bf16.coordinate.v1``; the checkers then receive a ``NodeInstance`` and a mapping of
argument names to authenticated ``Value`` objects.  Without W2's package the same functions
accept a plain node mapping and ``opened_inputs`` keyed by argument name *or* canonical id, with
values as ``Value``-likes, ``numpy`` ``uint16`` arrays or raw little-endian bytes.

How the epilogue is chosen (the model must be told, it must not guess by trying both):

1. a ``const`` arg named ``epilogue`` on the node (``"none"``, ``"bias_fp32_in_kernel"``,
   ``"bias_bf16_after"``), ``node.extra["consts"]["epilogue"]`` (W1's consts), or
   ``attrs.epilogue`` on a plain node mapping;
2. else the node's ``runtime_op`` (W1: ``node.extra["runtime_op"]``): ``verity.bi_mm`` /
   ``aten.mm`` / ``aten.matmul`` never carry a bias (``linear_batch_invariant`` adds it afterwards
   as a separate ``aten.add`` node) so ``epilogue = none``; ``aten.addmm`` -> ``bias_fp32_in_kernel``;
   ``aten.linear`` -> ``bias_bf16_after``; else the same by the ``runtime_impl`` string;
3. else ``bias_bf16_after`` (what ``torch.nn.functional.linear(x, w, b)`` does under
   ``enable_batch_invariant_mode()`` in eager mode -- the path this lane measured).

W1's GEMM node (``verity.bi_mm``) opens ``x`` = the producer's row (with ``extra["arg_cols"]["x"]``
= the column slice when the producer is wider) and ``w`` = the *parameter* leaf in its own
``[N, K]`` layout (the runtime saw ``weight.t()``; W1's own checker transposes back).  The row VU
is ``M = 1``: one opened input row -> the ``N`` BF16 words of one output row.

A slice/split-K cuBLAS kernel is described by a ``const`` arg ``split_k`` (or ``attrs.split_k``)
``{"nsplit": S, "bounds": [...], "order": [...]|null, "reduce_mode": 0|1|2}`` in K/16-tile units,
e.g. the ``sliced1x2`` 64x64 kernel = ``verity_vllm.program.kernels.gemm_production.sliced_k_spec(K, 2, 2, ...)``.
A node whose kernel is unresolved must not carry this relation (checker ``none``).

Coverage (what the checker claims).  ``independent_gemm`` predicts only on the families this
lane measured bit-exact on the A100 -- :func:`coverage` decides from the node alone:

* batch-invariant ``matmul_kernel_persistent`` (W1's ``verity.bi_mm`` / ``runtime_impl``
  naming ``matmul_persistent``): (K, N) in :data:`VALIDATED_BI_FAMILIES` (the five Qwen2.5-1.5B
  projections and the synthetic K=1040 N=256 bias case) with an epilogue measured on that family;
* torch's default cuBLAS/cuBLASLt path: only the launch signatures in ``kernel_zoo.ENTRIES``,
  each at the K value(s) and M range where it was observed exact (the node must declare ``M``);
* a node that declares ``split_k`` itself is taken at its word (the description is the claim).

Everything else raises :class:`GemmUnresolved` (``unresolved: <why>``), which W2's runner records
as ``error`` with that detail -- never ``exact``, never ``mismatch``.  Non-finite operands are
outside the domain (``GemmRelationError``).  ``model_gemm_bytes`` is the ungated arithmetic for
tests and diagnosis.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


from verity_vllm.program.kernels import cpu_model as CM
from verity_vllm.program.kernels import kernel_zoo as Z
from verity_vllm.program.kernels.mma import AMPERE_BF16, TCModel, tc_dot_chain


def _f32_to_bf16(bits):
    """FP32 words -> BF16 words, nearest-even, NaN -> 0x7FC0: the vectorised form of `verity.ml.circuits.silicon.cast.f32_to_bf16_word`
    (veritor's `core.silicon.f32_to_bf16`, kept here because verity.ml.circuits.ops is stdlib-only)."""
    bits = np.asarray(bits).astype(np.uint32, copy=False)
    with np.errstate(over="ignore"):
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> 16
    exponent_all_ones = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    is_nan = exponent_all_ones & ((bits & np.uint32(0x007FFFFF)) != 0)
    return np.where(is_nan, np.uint32(0x7FC0), rounded).astype(np.uint16)

MODEL_VERSION = "gemm.bf16.v1 = ampere_bf16_m16n8k16 chain (silicon.AMPERE_BF16_M16N8K16: groups (8,8), width 25, exponent floor -132) + epilogue; A100 validation 2026-09-07"
EPILOGUES = {
    "none": 0,
    "bias_fp32_in_kernel": 1,  # vLLM matmul_kernel_persistent via aten::addmm; cuBLASLt bias epilogue
    "bias_acc_init": 2,
    "bias_bf16_after": 3,  # vLLM linear_batch_invariant: matmul, then output + bias in BF16
}
DEFAULT_EPILOGUE = "bias_bf16_after"
ARG_ALIASES = {"x": ("x", "input", "a"), "w": ("w", "weight"), "bias": ("bias", "b")}


class GemmRelationError(ValueError):
    """Inputs the model cannot be applied to (dtype, shape, unresolved kernel)."""


# --- input adaptation -------------------------------------------------------------------------------


def _words(value: Any) -> np.ndarray:
    """Any opened BF16 leaf -> flat uint16 words (little-endian device bytes)."""
    if isinstance(value, np.ndarray):  # before the duck-typed Value check: ndarrays have .data/.dtype/.shape too
        if value.dtype == np.uint16:
            return value.reshape(-1)
        if value.dtype == np.int16:
            return value.view(np.uint16).reshape(-1)
        raise GemmRelationError(f"numpy opening must be uint16 words, got {value.dtype}")
    if hasattr(value, "data") and hasattr(value, "dtype") and hasattr(value, "shape"):  # W2 Value
        if str(value.dtype) not in ("bfloat16", "torch.bfloat16"):
            raise GemmRelationError(f"{getattr(value, 'canonical_id', '?')}: expected bfloat16, got {value.dtype}")
        return np.frombuffer(value.data, dtype="<u2")
    if hasattr(value, "array") and hasattr(value, "dtype"):
        if str(value.dtype) not in ("bfloat16", "torch.bfloat16"):
            raise GemmRelationError(f"{getattr(value, 'canonical_id', '?')}: expected bfloat16, got {value.dtype}")
        return np.ascontiguousarray(value.array()).view(np.uint16).reshape(-1)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype="<u2")
    raise GemmRelationError(f"cannot read an opening of type {type(value).__name__}")


def _node_of(node_or_inst: Any) -> Any:
    return getattr(node_or_inst, "node", node_or_inst)


def _arg_names(node: Any) -> list[str]:
    args = getattr(node, "args", None)
    if args is not None and not isinstance(args, Mapping):
        return [a.name for a in args]
    if isinstance(node, Mapping):
        a = node.get("args") or node.get("inputs") or {}
        return list(a.keys()) if isinstance(a, Mapping) else [x["name"] for x in a]
    return []


def _arg_value(node: Any, inputs: Mapping[str, Any], role: str) -> Any:
    """The opened value for role ``x`` / ``w`` / ``bias``: by argument name first, then by the
    canonical id a plain node mapping names for it."""
    for name in ARG_ALIASES[role]:
        if name in inputs:
            return inputs[name]
    if isinstance(node, Mapping):
        args = node.get("args") or node.get("inputs") or {}
        if isinstance(args, Mapping):
            for name in ARG_ALIASES[role]:
                ref = args.get(name)
                if isinstance(ref, Mapping):
                    ref = ref.get("id") or ref.get("canonical_id")
                if ref is not None and ref in inputs:
                    return inputs[ref]
    return None


def _extra(node: Any) -> Mapping[str, Any]:
    """W1's per-node extras on a W2 ``Node`` (``runtime_op``, ``consts``, ``arg_cols``) or on a mapping."""
    ex = getattr(node, "extra", None)
    if isinstance(ex, Mapping):
        return ex
    if isinstance(node, Mapping):
        ex = node.get("extra")
        if isinstance(ex, Mapping):
            return ex
    return {}


def _const(node: Any, inputs: Mapping[str, Any], name: str) -> Any:
    v = inputs.get(name)
    if v is not None and not hasattr(v, "array"):
        if isinstance(v, Mapping) and "const" in v:
            v = v["const"]
        return v
    consts = _extra(node).get("consts")
    if isinstance(consts, Mapping) and name in consts:
        return consts[name]
    if isinstance(node, Mapping):
        attrs = node.get("attrs") or node.get("attributes") or {}
        return attrs.get(name)
    return None


def _arg_cols(node: Any, name: str) -> tuple[int, int] | None:
    """The producer's column slice W1 records for an input (``extra.arg_cols[name] = [c0, c1]``)."""
    cols = (_extra(node).get("arg_cols") or {}).get(name)
    return (int(cols[0]), int(cols[1])) if cols else None


def runtime_op_of(node_or_inst: Any) -> str:
    node = _node_of(node_or_inst)
    op = _extra(node).get("runtime_op")
    if op is None and isinstance(node, Mapping):
        op = node.get("runtime_op")
    return str(op or "")


def epilogue_for(node_or_inst: Any, inputs: Mapping[str, Any], has_bias: bool) -> str:
    node = _node_of(node_or_inst)
    if not has_bias:
        return "none"
    explicit = _const(node, inputs, "epilogue")
    if explicit is not None:
        if explicit not in EPILOGUES:
            raise GemmRelationError(f"unknown epilogue {explicit!r}; one of {sorted(EPILOGUES)}")
        return str(explicit)
    rop = runtime_op_of(node)
    if rop in ("verity.bi_mm", "aten.mm", "aten.matmul"):
        raise GemmRelationError(f"{rop} carries no bias operand (the bias add is a separate aten.add node); got one")
    if rop == "aten.addmm":
        return "bias_fp32_in_kernel"
    if rop == "aten.linear":
        return "bias_bf16_after"
    impl = getattr(node, "runtime_impl", None) or (node.get("runtime_impl") if isinstance(node, Mapping) else None) or ""
    impl = str(impl).lower()
    if "addmm" in impl:
        return "bias_fp32_in_kernel"
    if "linear" in impl:
        return "bias_bf16_after"
    return DEFAULT_EPILOGUE


def spec_for(node_or_inst: Any, inputs: Mapping[str, Any], has_bias: bool) -> CM.GemmModelSpec:
    node = _node_of(node_or_inst)
    bias_mode = EPILOGUES[epilogue_for(node_or_inst, inputs, has_bias)]
    split = _const(node, inputs, "split_k")
    if split:
        return CM.GemmModelSpec(
            AMPERE_BF16.name,
            nsplit=int(split["nsplit"]),
            bounds=tuple(int(v) for v in split["bounds"]) if split.get("bounds") else None,
            reduce_mode=int(split.get("reduce_mode", 0)),
            order=tuple(int(v) for v in split["order"]) if split.get("order") else None,
            bias_mode=bias_mode,
            note=str(split.get("note", "split/slice-K from the description")),
        )
    return CM.GemmModelSpec(AMPERE_BF16.name, bias_mode=bias_mode, note="single ascending chain")


def per_instruction_model(node_or_inst: Any, inputs: Mapping[str, Any]) -> TCModel:
    node = _node_of(node_or_inst)
    arch = _const(node, inputs, "arch") or "sm_80"
    if arch not in ("sm_80", "sm_86", "sm_89"):
        raise GemmRelationError(f"no validated per-instruction model for arch {arch!r} in this relation")
    # Ada BF16 mma.sync is parameter-identical to Ampere's (verity.ml.circuits.silicon.models.ADA_BF16_M16N8K16), so the
    # same chain model applies on sm_89; the arch is recorded in the row, not in the arithmetic.
    return AMPERE_BF16


def operands(node_or_inst: Any, inputs: Mapping[str, Any]):
    node = _node_of(node_or_inst)
    xv, wv, bv = (_arg_value(node, inputs, r) for r in ("x", "w", "bias"))
    if xv is None or wv is None:
        raise GemmRelationError(f"gemm node must open its 'x' and 'w' arguments (got {sorted(k for k in inputs if not k.startswith('__'))})")
    x, w = _words(xv), _words(wv)
    cols = _arg_cols(node, "x")
    if cols is not None:  # the producer's output row is wider than this input (W1: arg_cols)
        x = x[cols[0] : cols[1]]
    K = _const(node, inputs, "K")
    wshape = getattr(wv, "shape", None)
    wshape = tuple(int(d) for d in wshape) if wshape is not None and len(wshape) == 2 else None
    if K is None:
        if cols is not None:
            K = cols[1] - cols[0]
        elif wshape is not None:
            # The opened weight leaf is the parameter in its own [N, K] layout (W1 opens the
            # parameter even though the runtime sees weight.t(); its own checker transposes).
            # A leaf that can only be [K, N] (first dim = the row width, second dim != it) is
            # accepted transposed; a square weight is taken as [N, K].
            K = wshape[1]
            xs = getattr(xv, "shape", None)
            if xs is not None and len(xs) >= 1 and int(xs[-1]) != wshape[1] and int(xs[-1]) == wshape[0]:
                K = wshape[0]
        else:
            K = int(x.size)  # a single opened row
    K = int(K)
    if K % AMPERE_BF16.k:
        raise GemmRelationError(f"K={K} is not a multiple of the instruction K={AMPERE_BF16.k}")
    if x.size % K or w.size % K:
        raise GemmRelationError(f"openings ({x.size} and {w.size} words) are not multiples of K={K}")
    x = x.reshape(-1, K)
    if wshape is not None and wshape[0] == K and wshape[1] != K:
        w = np.ascontiguousarray(w.reshape(K, wshape[1]).T)  # a [K, N] leaf -> [N, K]
    else:
        w = w.reshape(-1, K)
    b = None if bv is None else _words(bv)
    if b is not None and b.size != w.shape[0]:
        raise GemmRelationError(f"bias has {b.size} words, weight has {w.shape[0]} rows")
    # domain: finite BF16 operands only.  The tensor core treats inf/NaN operands specially
    # (inf x 0 -> NaN, NaN -> 0x7fffffff: overflow_diagnosis.py) and the integer-exact product
    # model would read an exponent field of 0xFF as a large finite number.
    for name, arr in (("x", x), ("w", w), ("bias", b)):
        if arr is not None and np.any((arr & 0x7F80) == 0x7F80):
            raise GemmRelationError(f"non-finite BF16 word in {name}: outside the model's domain (finite operands only)")
    return x, w, b, K, int(w.shape[0])


# --- coverage: exactly the families the model reproduced bit-exactly ------------------------------------


class GemmUnresolved(GemmRelationError):
    """The node is outside the validated coverage: the checker declines instead of predicting.

    W2's runner records the raised ``RelationError`` as status ``error`` with this message
    (prefix ``unresolved:``); it never counts as exact or as a mismatch.
    """


# (K, N, bias?) shape families of Qwen2.5-1.5B (merged projections as vLLM runs them) plus the
# synthetic masked-K-tail shape, each bit-exact under vLLM's batch-invariant
# ``matmul_kernel_persistent`` on the A100 for every natural and stress case that was run
# (results/numerics-a100-20260907T0615Z/table2.jsonl, profile a100-vllm-bi1; M in 1..2048).
# ``epilogues`` lists the epilogues measured on that family (the GEMM inside the engine,
# ``verity.bi_mm``, never carries a bias: the add is a separate node).
VALIDATED_BI_FAMILIES: dict[tuple[int, int], dict[str, Any]] = {
    (1536, 2048): {"name": "qkv_proj", "epilogues": ("none", "bias_fp32_in_kernel", "bias_bf16_after"), "M": (1, 2048)},
    (1536, 1536): {"name": "o_proj", "epilogues": ("none",), "M": (1, 2048)},
    (1536, 17920): {"name": "gate_up_proj", "epilogues": ("none",), "M": (1, 2048)},
    (8960, 1536): {"name": "down_proj", "epilogues": ("none",), "M": (1, 2048)},
    (1536, 151936): {"name": "lm_head", "epilogues": ("none",), "M": (1, 2048)},
    (1040, 256): {"name": "synthetic_k1040_n256_bias", "epilogues": ("bias_fp32_in_kernel", "bias_bf16_after"), "M": (64, 64)},
}
BI_KERNEL_MARKERS = ("matmul_kernel_persistent", "matmul_persistent", "batch_invariant", "verity.bi_mm")
BI_RUNTIME_OPS = ("verity.bi_mm", "aten.mm", "aten.addmm", "aten.linear", "aten.matmul")
VALIDATED_ARCHS = ("sm_80",)  # where the production GEMM itself was measured
ASSUMED_ARCHS = ("sm_86", "sm_89")  # parameter-identical BF16 mma.sync (verity.ml.circuits.silicon.models.ADA_BF16_M16N8K16); production GEMM not measured by this lane


def _runtime_impl_of(node: Any) -> str:
    node = _node_of(node)
    impl = getattr(node, "runtime_impl", None)
    if impl is None and isinstance(node, Mapping):
        impl = node.get("runtime_impl")
    return str(impl or "")


def _declared_M(node: Any, inputs: Mapping[str, Any]) -> int | None:
    node = _node_of(node)
    m = _const(node, inputs, "M")
    if m is not None:
        return int(m)
    shapes = _extra(node).get("arg_shapes") or {}
    xs = shapes.get("x")
    if xs and len(xs) == 2:
        return int(xs[0])
    return None


def coverage(node: Any, opened_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Does the model claim this node?  Returns ``{"covered": bool, "family": ..., "kernel_path":
    ..., "spec": GemmModelSpec | None, "reason": str}`` without running the chain.

    Covered iff (a) the operands are finite BF16 with K % 16 == 0, (b) the kernel path is the
    batch-invariant Triton kernel and (K, N, epilogue) is one of :data:`VALIDATED_BI_FAMILIES`,
    or the kernel path is a cuBLAS launch signature the kernel zoo resolved for this K and the
    declared M (``kernel_zoo.ENTRIES``, each tied to the (K, M) range where it was exact), and
    (c) the arch is sm_80 (measured) or sm_86/89 (assumed parameter-identical; flagged).
    """

    n = _node_of(node)
    x, w, b, K, N = operands(n, opened_inputs)  # raises GemmRelationError on dtype/shape/non-finite
    has_bias = b is not None
    epi = epilogue_for(n, opened_inputs, has_bias)
    impl = _runtime_impl_of(n)
    rop = runtime_op_of(n)
    arch = str(_const(n, opened_inputs, "arch") or "sm_80")
    arch_note = "" if arch in VALIDATED_ARCHS else (f"; arch {arch} assumed parameter-identical to sm_80 (production GEMM not measured there by this lane)" if arch in ASSUMED_ARCHS else None)
    if arch_note is None:
        return {"covered": False, "family": None, "kernel_path": impl, "spec": None, "reason": f"unresolved: no validated per-instruction model for arch {arch!r}"}
    explicit_split = _const(n, opened_inputs, "split_k")
    # Capture format v2 (W3c): a coordinate VU opens one *row block* of the parameter (a column
    # of W^T), so the opened N is the block height, not the launch's N.  The description declares
    # the launch's N (``consts.N`` or the node's output width) and the family is looked up there;
    # the arithmetic of one column of a single-chain kernel is the same chain.
    n_declared = _const(n, opened_inputs, "N")
    slice_note = ""
    if n_declared is not None and int(n_declared) != N and N < int(n_declared):
        slice_note = f"; opened a {N}-row block of the N={int(n_declared)} weight"
        N = int(n_declared)
    bi = any(m in impl for m in BI_KERNEL_MARKERS) or (rop in BI_RUNTIME_OPS and not impl) or rop == "verity.bi_mm"
    if bi and not explicit_split:
        fam = VALIDATED_BI_FAMILIES.get((K, N))
        if fam is None:
            return {"covered": False, "family": None, "kernel_path": "matmul_kernel_persistent", "spec": None, "reason": f"unresolved: shape family K={K} N={N} not validated (validated: {sorted(VALIDATED_BI_FAMILIES)})"}
        if epi not in fam["epilogues"]:
            return {"covered": False, "family": fam["name"], "kernel_path": "matmul_kernel_persistent", "spec": None, "reason": f"unresolved: epilogue {epi!r} not validated on {fam['name']} (validated: {fam['epilogues']})"}
        M = _declared_M(n, opened_inputs)
        if M is not None and not (fam["M"][0] <= M <= fam["M"][1]):
            return {"covered": False, "family": fam["name"], "kernel_path": "matmul_kernel_persistent", "spec": None, "reason": f"unresolved: M={M} outside the measured range {fam['M']} for {fam['name']}"}
        spec = CM.GemmModelSpec(AMPERE_BF16.name, bias_mode=EPILOGUES[epi], note="single ascending chain (matmul_kernel_persistent)")
        return {"covered": True, "family": fam["name"], "kernel_path": "matmul_kernel_persistent", "spec": spec, "reason": f"validated family {fam['name']} (K={K}, N={N}, epilogue {epi}){slice_note}{arch_note}"}
    # cuBLAS / cuBLASLt: only launch signatures the zoo resolved, at the K and M where they were exact
    M = _declared_M(n, opened_inputs)
    if explicit_split:
        spec = spec_for(n, opened_inputs, has_bias)
        return {"covered": True, "family": "declared split/slice-K", "kernel_path": impl or "declared", "spec": spec, "reason": f"split_k declared by the description ({spec.note}){arch_note}"}
    if not impl:
        return {"covered": False, "family": None, "kernel_path": "", "spec": None, "reason": "unresolved: node names no runtime_impl and no batch-invariant runtime_op"}
    if M is None:
        return {"covered": False, "family": None, "kernel_path": impl, "spec": None, "reason": "unresolved: cuBLAS launch without a declared M (consts.M or arg_shapes.x): the kernel zoo entries are tied to M ranges"}
    zoo = Z.resolve(impl, K, M, has_bias)
    if zoo is None:
        return {"covered": False, "family": None, "kernel_path": impl, "spec": None, "reason": f"unresolved: {Z.why_unresolved(impl)} (K={K}, M={M})"}
    # the zoo entry fixes the epilogue it was validated with; a description that declares another one is not covered
    declared = _const(n, opened_inputs, "epilogue")
    if has_bias and declared is not None and EPILOGUES.get(str(declared)) != zoo.bias_mode:
        return {"covered": False, "family": "cuBLAS zoo", "kernel_path": impl, "spec": None, "reason": f"unresolved: declared epilogue {declared!r} differs from the validated zoo epilogue (bias_mode {zoo.bias_mode}) for this launch"}
    return {"covered": True, "family": "cuBLAS zoo", "kernel_path": impl, "spec": zoo, "reason": f"kernel zoo entry ({zoo.note}) for K={K}, M={M}{arch_note}"}


# --- the relations ------------------------------------------------------------------------------------


def model_gemm_bytes(node: Any, opened_inputs: Mapping[str, Any], spec: CM.GemmModelSpec | None = None) -> bytes:
    """The arithmetic without the coverage gate: BF16 bytes of the output row(s) under ``spec``
    (default: the node's epilogue/split description).  Used by the coverage-gated relations and
    by tests on shapes that were never measured."""

    x, w, b, K, N = operands(node, opened_inputs)
    if spec is None:
        spec = spec_for(node, opened_inputs, b is not None)
    out, _ = CM.gemm_model(per_instruction_model(node, opened_inputs), x, w, b, spec, want_f32=False)
    return out.astype("<u2").tobytes()


def independent_gemm(node: Any, opened_inputs: Mapping[str, Any]) -> bytes:
    """Predicted BF16 bytes of the GEMM output row(s) for the opened input row(s), weight and bias.

    Only for nodes :func:`coverage` claims (validated kernel path x shape family x epilogue);
    anything else raises :class:`GemmUnresolved` (``unresolved: ...``) -- the checker never
    guesses a description it did not measure.
    """

    cov = coverage(node, opened_inputs)
    if not cov["covered"]:
        raise GemmUnresolved(cov["reason"])
    return model_gemm_bytes(node, opened_inputs, cov["spec"])


def independent_gemm_coordinate(node: Any, opened_inputs: Mapping[str, Any], j: int | None = None) -> bytes:
    """The scalar gate: one output coordinate from the opened input row and the opened weight *column*.

    Pure Python (``veritor.core.silicon.tc_dot`` chain + the epilogue in numpy fp32), no C++.
    ``j`` (or a ``const`` arg ``j``) selects the column when the weight opening is the whole
    matrix; a ``[K]`` opening is the column itself and ``j`` indexes the bias only.
    """

    x, w, b, K, N = operands(node, opened_inputs)
    if j is None:
        j = _const(_node_of(node), opened_inputs, "j")
    x = x[0]
    if N == 1:
        col = w[0]
        bias_word = None if b is None else int(b[int(j) if j is not None and b.size > 1 else 0])
    else:
        if j is None:
            raise GemmRelationError("j is required when the weight opening is the whole matrix")
        col = w[int(j)]
        bias_word = None if b is None else int(b[int(j)])
    spec = spec_for(node, opened_inputs, bias_word is not None)
    if spec.nsplit != 1:
        raise GemmRelationError("coordinate gate for split-K descriptions: use independent_gemm")
    acc = tc_dot_chain(AMPERE_BF16, 0, x.tolist(), col.tolist())
    v = np.array([acc], np.uint32).view(np.float32)
    if spec.bias_mode == 1:
        v = (v + np.array([bias_word << 16], np.uint32).view(np.float32)).astype(np.float32)
        out = _f32_to_bf16(v.view(np.uint32))
    elif spec.bias_mode == 3:
        r = _f32_to_bf16(v.view(np.uint32))
        rv = (r.astype(np.uint32) << 16).view(np.float32)
        out = _f32_to_bf16((rv + np.array([bias_word << 16], np.uint32).view(np.float32)).astype(np.float32).view(np.uint32))
    else:
        out = _f32_to_bf16(v.view(np.uint32))
    return out.astype("<u2").tobytes()


RELATION_IDS = {
    "gemm.bf16.v1": ("gemm", independent_gemm, "Hawkeye-derived Ampere mma.sync chain (silicon.AMPERE_BF16_M16N8K16) over ascending K tiles + declared epilogue, RNE to BF16; validated bit-exact against vLLM's batch-invariant matmul_kernel_persistent on the A100 (numerics.md). Coverage-gated: predicts only on the validated (kernel path, K, N, epilogue) families, else RelationError('unresolved: ...')."),
    "lm_head.bf16.v1": ("lm_head", independent_gemm, "gemm.bf16.v1 at the vocab shape (K=1536, N=151936, no bias); same coverage gate."),
    "gemm.bf16.coordinate.v1": ("gemm", independent_gemm_coordinate, "Scalar gate: one coordinate of gemm.bf16.v1 from the opened input row and weight column, pure-Python tc_dot chain."),
}
RELATIONS: dict[str, Any] = {rid: fn for rid, (_, fn, _) in RELATION_IDS.items()}


def register_with_w2(replace: bool = False) -> list[str]:
    """Fill W2's slots (``verity_vllm.program.kernels.relations``); returns the ids registered, [] if W2 is absent."""

    try:
        from verity_vllm.program.kernels import relations as R
    except ImportError:
        return []
    done = []
    for rid, (op_type, fn, doc) in RELATION_IDS.items():
        existing = R.REGISTRY.get(rid)
        if existing is not None and existing.implemented and not replace:
            continue

        def _wrap(inst, inputs, _fn=fn):
            try:
                return _fn(inst, inputs)
            except GemmRelationError as error:
                raise R.RelationError(str(error)) from error

        R.register(rid, klass=R.INDEPENDENT, op_type=op_type, includes_full_op=False, version=MODEL_VERSION, doc=doc, owner="W4", replace=replace)(_wrap)
        done.append(rid)
    return done


def register(registry: Any) -> None:
    """Install into a dict-like registry or one exposing ``.register(id, fn)`` (standalone use)."""

    for rid, fn in RELATIONS.items():
        if hasattr(registry, "register"):
            registry.register(rid, fn)
        else:
            registry[rid] = fn


REGISTERED_WITH_W2 = register_with_w2()

# W11: the independent attention relation (FlashAttention-2 model) registers itself alongside the GEMM
# relations when this module is imported; kept in its own module (fa2_relation.py).
from verity_vllm.program.kernels import fa2_relation as _fa2_relation  # noqa: E402,F401
