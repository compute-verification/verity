"""VU boundary census: every boundary value of the canonical `W_max = 32` VU partition of one captured Program, with its
logical role, its production materialization class, and the physical cost of exposing it (window 2026-09-17, post-window
decision block item 1 and its addendum in `out/gen/WINDOW-20260917.md`).

    python -m verity_capture.bench.vu_census RUN_DIR [--profile sm89] [--out OUT_DIR] [--g8] [--samples 3]

`RUN_DIR` holds `program.json` (the fold descriptor), `instances.jsonl` (the per-instance runtime views) and, when
`--g8` is given, receives `query.json` + `query_use.json` so `verity_capture.gates.gate_G8` can be evaluated on it.

What is counted.  The canonical query (`verity_capture.vu_canonical`) is evaluated on the Program (`gateset.evaluate`);
its `classes()` are the layout classes of the family -- every set of a class is the same body entry at a different
activation / member, so one `boundary.explain(representative)` gives the out gates of every set of the class.  For each
out gate the census records (width, producing primitive) and decides, by walking the return chain of the body up to the
root node, whether the value is

    root-input        a member of a root `Input<w>` batch (weights, prompt tokens);
    instance-output   a returned leaf of the root-body node = an element of the runtime Instance's output tensor;
    interior          consumed by another VU inside the same Instance, never returned by it;
    constant          a `Const<w>` gate (fixed by the Program text; carries no run-time information).

Physical location (eager production, the captured profile) is assigned STRICTLY (addendum item 2): `HBM-materialized`
only with positive evidence -- root inputs and instance outputs are runtime tensors named by `instances.jsonl`
(`operand_views` / `output_views`, whose element counts this tool checks against the Program's leaf counts); interior
values of a library kernel whose physical representation is not exposed (FA2 `flash_fwd_splitkv_kernel`) are
`opaque/library-internal`; every other interior value is `unknown` (positive evidence would be the compiled kernel's
IR / SASS -- not inspected here).  Disappearance under fusion is never taken as register residency.

`f_free` is reported as a range (addendum item 3): lower = established-native / total, upper = (established-native +
unknown) / total; the same for added HBM traffic.  Logical bytes (the VU interface) and dynamic bytes (per request /
per token / per step; weights once per model) are kept apart (addendum item 4).

The production (compiled, `torch.compile` + CUDA graphs) column is a separate, weaker classification: `HBM (extern
boundary)` only for values that are operands or outputs of GEMM / attention / sampler kernels -- extern kernels under
every pinned compiled configuration of the recon (`docs/vllm-poc/production-vllm-recon-2026-09-17.md` §3, W1c) -- and
`unknown` for everything the fused Inductor graph may or may not materialise.  Nothing in this column is measured.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

VERSION = "2026-09-18.1"
SCHEMA = "verity-gen/vu-census/v1"

# --------------------------------------------------------------------------------------------------------------------
# kernel table: the eager kernel that executes each Instance family (from the run's census / resolution: pattern ids and
# the kernel names of `census/census.json.run_histogram`), and the class its INTERIOR values get.
# --------------------------------------------------------------------------------------------------------------------

#: pattern prefix -> (kernel name (as in the census histogram), kernel origin, interior class, note)
KERNELS: dict[str, tuple[str, str, str, str]] = {
    "attention.fa2": ("flash_fwd_splitkv_kernel", "library (vllm_flash_attn FA2, CUTLASS)", "opaque/library-internal",
                      "scores, running max / sum, P and the PV accumulators live inside one kernel; no API exposes their representation"),
    "gemm.matmul_kernel_persistent": ("matmul_kernel_persistent", "Triton (vLLM batch-invariant GEMM)", "unknown",
                                      "the canonical GEMM VU is one output word; no interior boundary value"),
    "rmsnorm.fused_add_rms_norm": ("fused_add_rms_norm_kernel", "vLLM CUDA (csrc/layernorm_kernels.cu)", "unknown",
                                   "source shows a block reduction in registers / shared memory; register residency of the compiled kernel not established (no SASS inspected)"),
    "rmsnorm.triton_rms_norm_kernel": ("_rms_norm_kernel", "Triton (vLLM batch-invariant rms_norm)", "unknown",
                                       "Triton values are registers unless spilled; TTGIR / PTX not inspected"),
    "rope.rotary_embedding": ("rotary_embedding_kernel", "vLLM CUDA (csrc/pos_encoding_kernels.cu)", "unknown", "outputs only under the canonical query"),
    "act.silu_and_mul": ("act_and_mul_kernel", "vLLM CUDA (csrc/activation_kernels.cu)", "unknown", "outputs only under the canonical query"),
    "embedding.aten_embedding": ("index_elementwise_kernel / indexSelectSmallIndex", "ATen", "unknown", "outputs only under the canonical query"),
    "sample.gumbel_two_stage": ("_gumbel_sample_kernel + ArgMaxOps reduce", "Triton + ATen", "unknown", "the whole TokenSelect call is one VU (out = the token id)"),
    "root": ("(root-level call)", "-", "unknown", "a whole root-body call as one VU: its out is the Instance output"),
    "input": ("(root input batch)", "-", "unknown", "root inputs"),
}

#: root families whose operands AND outputs are extern-kernel tensors under the pinned compiled configurations
#: (recon §3 / W1c: GEMM = cuBLAS or the BI Triton custom op; attention = FA2 custom op; sampler = separate kernels).
COMPILED_EXTERN_FAMILIES = ("Gemm_v1", "Gemm_v2", "Attention_v3", "Attention_v1", "TokenSelect_v1")
#: root families whose OUTPUT is an operand of an extern kernel (so materialised as its input buffer)
COMPILED_FEEDS_EXTERN = {"RMSNormFusedCuda_v2": "normalized output feeds the QKV / gate-up GEMM (extern); the residual return does not",
                         "RMSNormFusedCuda_v1": "same", "RMSNormTriton_v1": "final norm output feeds the lm_head GEMM (extern)",
                         "RoPE_v1": "q / k feed the attention custom op (extern)", "SiluMul_v1": "feeds the down GEMM (extern)"}

#: descriptive role of an out gate, by producing primitive (attention / norm bodies); tables keep the primitive id too
ROLES: dict[str, str] = {
    "AmpereBF16TcDot16_v1": "PV accumulator partial (k16 MMA step, fp32)",
    "HopperBF16WgmmaDot16_v1": "PV accumulator partial (k16 wgmma step, fp32)",
    "MufuEx2Ftz_v1": "P = exp2(scaled score - max) (fp32)",
    "F32FmaSubFtz_v1": "scaled score minus running max (fp32)",
    "F32Max_v1": "running row-max chain (fp32)",
    "F32AddFtz_v1": "row-sum / lane-sum partial (fp32)",
    "F32MulFtz_v1": "scale / rescale factor (fp32)",
    "F2fpBf16_v1": "fp32 -> bf16 conversion",
    "Bf16ToF32_v1": "bf16 -> fp32 conversion",
    "F32Mul_v1": "fp32 multiply (square / normalize path)",
    "F32Add_v1": "fp32 add (reduction partial)",
    "F32Fma_v1": "fp32 fma (reduction partial)",
    "F32Div_v1": "fp32 divide (variance / N)",
    "DivFullRcp_v1": "reciprocal (rsqrt path)",
    "MufuSqrtFtz_v1": "sqrt (rsqrt path)",
    "Fa2InvSum_v1": "1 / row-sum (FA2 epilogue)",
    "SiluMulBf16_v1": "silu(gate) * up (bf16 output word)",
    "RopeOut_v1": "rotary output word (bf16)",
    "RopeOutAdd_v1": "rotary output word (bf16)",
    "Bf16AddF2fp_v1": "residual + x (bf16 residual output word)",
    "GatherBf16x49152_v1": "embedding row element (bf16)",
    "SelectI32_v1": "sampled token id (int32)",
    "Input16_v1": "weight element (bf16)",
    "Input32_v1": "prompt token id (int32)",
}


#: roles decided by what the VU is (its items), before the out primitive is consulted
VU_ROLES: dict[str, str] = {
    "call DotBf16_v1{…}": "attention score s = q·k[c] (fp32; the dot's last MMA step is the out gate)",
    "call DotBf16_v2{…}": "attention score s = q·k[c] (fp32)",
    "zip[AmpereBF16TcDot16_v1]": "PV accumulator o[d] after one 128-key block (fp32; consumed by the next block / epilogue)",
    "zip[HopperBF16WgmmaDot16_v1]": "PV accumulator o[d] after one key block (fp32)",
    "zip[F2fpBf16_v1+F32Mul_v1]": "normalized output word (bf16)",
    "call TokenSelect_v1{…}": "sampled token id (int32)",
}


def _role(prim: str, vu_kind: str = "") -> str:
    if vu_kind in VU_ROLES:
        return VU_ROLES[vu_kind]
    base = prim.split("{")[0]
    if base.startswith("Const"):
        return "Program constant"
    if base.startswith("Input"):
        return ROLES.get(base, "root input")
    if base.startswith("GatherBf16x"):
        return "embedding row element (bf16)"
    return ROLES.get(base, base)


# --------------------------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------------------------


def load_program(run_dir: Path):
    """(Program, descriptor dict, program.json record) for a fold `program.json` (`verity-gen/program/v1`)."""
    import verity_vllm.registry.b1  # noqa: F401  (registers the primitives the descriptor names)
    from verity.ir.codec import decode_program, program_digest
    from verity.ir.defs import REGISTRY
    from verity.ir.program import Program
    rec = json.loads((run_dir / "program.json").read_text())
    desc = rec.get("program", rec)
    dec = decode_program(desc, REGISTRY)
    P = Program.__new__(Program)
    P.fn = dec.program_fn_functions[0] if dec.program_fn_functions else None
    P.annotations = dec.annotations
    P.root = dec.root
    P._circuit = dec.circuit
    digest = program_digest(desc)
    if rec.get("digest") not in (None, digest):
        raise SystemExit(f"program.json digest {rec.get('digest')} != codec.program_digest {digest}")
    return P, desc, rec, digest


def load_instances(run_dir: Path, n_input_nodes: int) -> tuple[dict[int, tuple[str, str]], dict[str, dict]]:
    """root node index -> (spec id, pattern); spec id -> one representative instance record (views)."""
    by_node: dict[int, tuple[str, str]] = {}
    rep: dict[str, dict] = {}
    with (run_dir / "instances.jsonl").open() as f:
        header = json.loads(next(f))
        assert header.get("schema", "").startswith("verity-gen/instances/"), header
        for line in f:
            o = json.loads(line)
            by_node[n_input_nodes + o["node"]] = (o["spec"], o["pattern"])
            rep.setdefault(o["spec"], o)
    return by_node, rep


def kernel_of(pattern: str) -> tuple[str, tuple[str, str, str, str]]:
    for prefix, row in KERNELS.items():
        if pattern.startswith(prefix):
            return prefix, row
    return "?", ("?", "?", "unknown", f"pattern {pattern!r} not in the kernel table")


# --------------------------------------------------------------------------------------------------------------------
# the census
# --------------------------------------------------------------------------------------------------------------------


def _returned_to_root(scope, key, leaf) -> bool:
    """Whether the leaf `(key, leaf)` produced inside activation `scope` is, through the chain of re-returns, a returned
    leaf of the ROOT-BODY node that contains `scope` (= an element of the Instance's output).  Mirrors the leaf
    translation of `boundary._ret_consumed` one level at a time."""
    from verity_vllm.ir.boundary import _scan_shape, consumer_index
    while True:
        parent = scope.parent
        if parent is None:                      # the root activation itself: handled by the caller
            return True
        positions = consumer_index(scope.spec).ret.positions(key, leaf)
        if not positions:
            return False
        if parent.parent is None:               # `scope` is the activation of a root-body node
            return True
        k, m = scope.node_idx, scope.member
        node = parent.spec.body.nodes[k]
        pos = positions[0]
        if node.form == "call":
            key, leaf = ("n", k), pos
        elif node.form == "batch":
            key, leaf = ("n", k), m * node.fn.out_leaves + pos
        else:
            c, y = _scan_shape(node)
            if pos < c:
                key, leaf = (("s", k), m * c + pos) if m < node.n - 1 else (("n", k), pos)
            else:
                key, leaf = ("n", k), c + m * y + (pos - c)
        scope = parent


def _root_node(scope) -> int | None:
    """Index of the root-body node whose activation contains `scope` (None when `scope` is the root activation)."""
    if scope.parent is None:
        return None
    while scope.parent.parent is not None:
        scope = scope.parent
    return scope.node_idx


def _fanout(part, key, leaf) -> str:
    """Structural fan-out class of the out leaf `(key, leaf)` of `part` inside its body: how many consumer ITEMS outside
    the part read it (+1 when the body returns it).  `1` means the value has a single consumer, so a coarser partition
    that merges producer and consumer keeps W_out unchanged and removes the value from the boundary ("absorbable");
    `>=2` means it is a boundary value under EVERY <= w_max partition ("intrinsic").  A composite consumer (a call /
    scan whose body the family refines further) may hide several consumer VUs behind one item: `1?` = one composite
    reader (undetermined); `?` = the index could not resolve the readers."""
    from verity_vllm.ir.boundary import (
        _Aff,
        _Aff2,
        _All,
        _contains,
        _Lazy,
        _One,
        _scan_shape,
        consumer_index,
    )
    from verity.ir.parts import n_members

    def readers(scope, key, leaf, inside: dict[int, set[int]]) -> tuple[int, bool, bool]:
        """(consumer items outside `inside` in this body, any of them composite, unresolved) for (key, leaf); a leaf the
        body returns is followed into the parent body (the producing activation is then the only `inside` item)."""
        body = scope.spec.body
        ix = consumer_index(scope.spec)
        n, comp, unres = 0, False, False
        for jp, d in ix.consumers(key):
            node = body.nodes[jp]
            ins = inside.get(jp, set())
            is_comp = node.form in ("call", "scan") or (node.form == "batch" and getattr(node.fn, "body", None) is not None)
            if isinstance(d, _All):
                if _contains(d.ivs, leaf):
                    k = n_members(node) - len(ins)
                    n += k
                    comp |= is_comp and k > 0
            elif isinstance(d, (_Aff, _Aff2)):
                m = d.reader(leaf)
                if m is not None and m not in ins:
                    n += 1
                    comp |= is_comp
            elif isinstance(d, _One):
                if _contains(d.ivs, leaf) and d.m not in ins:
                    n += 1
                    comp |= is_comp
            elif isinstance(d, _Lazy) and _contains(d.union, leaf):
                unres = True
        positions = ix.ret.positions(key, leaf)
        if positions:
            parent = scope.parent
            if parent is None or parent.parent is None:
                n += 1                                   # designated output / Instance output: one external consumer at least
            else:
                k, m = scope.node_idx, scope.member
                node = parent.spec.body.nodes[k]
                for pos in positions:
                    if node.form == "call":
                        pk, pl = ("n", k), pos
                    elif node.form == "batch":
                        pk, pl = ("n", k), m * node.fn.out_leaves + pos
                    else:
                        c, y = _scan_shape(node)
                        if pos < c:
                            pk, pl = (("s", k), m * c + pos) if m < node.n - 1 else (("n", k), pos)
                        else:
                            pk, pl = ("n", k), c + m * y + (pos - c)
                    n2, _c2, u2 = readers(parent, pk, pl, {k: {m}})
                    # the parent context of the representative activation is one of possibly several (the same block body
                    # is the last block of a short head and a middle block of a long one): a returned leaf's reader
                    # count is context-dependent, so a single reader is reported as undetermined (`1?`), never as `1`
                    n, comp, unres = n + n2, True, unres | u2
        return n, comp, unres

    inside: dict[int, set[int]] = {}
    for j, m in part.items:
        inside.setdefault(j, set()).add(m)
    n, composite, unresolved = readers(part.scope, key, leaf, inside)
    if unresolved:
        return "?"
    if n >= 2:
        return ">=2"
    if n == 1:
        return "1?" if composite else "1"
    return "0"


def _profile(part, root_body) -> tuple[tuple[int, str, str, str], ...]:
    """Sorted (width, prim, logical class, fan-out class) of the out gates of `part`; raises on Unsupported."""
    from verity_vllm.ir import boundary as BD
    bgs = BD.explain(part)
    if not isinstance(bgs, list):
        raise RuntimeError(f"explain unsupported: {bgs}")  # noqa: TRY004 -- an Unsupported answer, not a type error
    body = part.scope.spec.body
    out = []
    for bg in bgs:
        base = bg.prim.split("{")[0]
        fo = "-"
        if part.scope.parent is None:           # a root-level part: an Input batch member or a whole root call
            node = root_body.nodes[bg.node]
            if node.form == "batch" and node.fn.id.startswith("Input"):
                cls = "root-input"
            else:
                cls = "instance-output"
        elif base.startswith("Const"):
            cls = "constant"
        else:
            if bg.kind == "carry":
                c, _ = BD._scan_shape(body.nodes[bg.node])
                key, leaf = ("s", bg.node), bg.member * c + bg.leaf
            else:
                key, leaf = ("n", bg.node), bg.leaf
            if _returned_to_root(part.scope, key, leaf):
                cls = "instance-output"
            else:
                cls = "interior"
                fo = _fanout(part, key, leaf)
        out.append((bg.width, bg.prim, cls, fo))
    return tuple(sorted(out))


def _vu_kind(kind: str, nodes: list[tuple[str, str]]) -> str:
    """A short structural name of the VU: what the part IS (its items), independent of which gate happens to be out."""
    ids = [fid.split("{")[0] for _, fid in nodes]
    if kind == "zip":
        return "zip[" + "+".join(sorted(set(ids))) + "]"
    if len(nodes) == 1:
        form, fid = nodes[0]
        return ("call " if form == "call" else "") + fid.split("{")[0] + ("{…}" if "{" in fid else "")
    return "+".join(sorted(set(ids)))


def run_census(P, by_node: dict[int, tuple[str, str]], fam, *, samples: int = 3, log=print) -> dict:
    from verity_vllm.ir import boundary as BD
    root_body = P.root.body
    t0 = time.time()
    classes = fam.classes()
    log(f"  classes: {len(classes)}  sets: {sum(c.total for c in classes):,} (== family count {fam.count():,})")
    rows: list[dict] = []
    nonuniform: list[dict] = []
    unsupported: list[dict] = []
    for ci, c in enumerate(classes):
        part = c.representative.parts[0]
        k = _root_node(part.scope)
        if k is None:                            # root-level entry: the items name root-body nodes directly
            j = next(iter(part.items))[0]
            node = root_body.nodes[j]
            root_spec, pattern = node.fn.id, ("input" if node.fn.id.startswith("Input") else "root")
        else:
            root_spec, pattern = by_node.get(k, (root_body.nodes[k].fn.id, "?"))
        try:
            prof = _profile(part, root_body)
        except RuntimeError as e:
            unsupported.append({"class": ci, "body": c.body, "entry": c.entry, "kind": c.kind, "total": c.total, "why": str(e)})
            continue
        ins = BD.in_gates(part)
        n_in = len(ins) if isinstance(ins, frozenset) else None
        # member uniformity: other members of the representative activation (first / middle / last)
        uniform = True
        if c.per_activation > 1 and samples > 0:
            idx0 = fam.index_of(c.representative)
            if idx0 is not None:
                probe = sorted({c.per_activation // 2, c.per_activation - 1} | ({1} if samples > 2 else set()))
                for off in probe[:samples]:
                    try:
                        p2 = _profile(fam.by_index(idx0 + off).parts[0], root_body)
                    except RuntimeError as e:
                        unsupported.append({"class": ci, "member": off, "why": str(e)})
                        continue
                    if p2 != prof:
                        uniform = False
                        nonuniform.append({"class": ci, "body": c.body, "kind": c.kind, "member": off, "rep": prof, "other": p2, "total": c.total})
                        break
        body = part.scope.spec.body
        nodes = sorted({(body.nodes[j].form, body.nodes[j].fn.id) for j, _ in part.items})
        rows.append({"class": ci, "body": c.body, "spec": part.scope.spec.id, "entry": c.entry, "kind": c.kind,
                     "root_spec": root_spec, "pattern": pattern, "per_activation": c.per_activation, "activations": c.activations,
                     "total": c.total, "items": len(part.items), "nodes": nodes, "vu_kind": _vu_kind(c.kind, nodes),
                     "n_in": n_in, "out": [list(x) for x in prof], "uniform": uniform})
    log(f"  explained {len(rows)} classes in {time.time() - t0:.1f} s; unsupported {len(unsupported)}; non-uniform {len(nonuniform)}")
    return {"rows": rows, "nonuniform": nonuniform, "unsupported": unsupported}


# --------------------------------------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------------------------------------


def _family(spec: str) -> str:
    return spec.split("{")[0]


def _eager_location(cls: str, pattern: str) -> tuple[str, str]:
    """(physical class, evidence) of an out value in the captured eager profile."""
    if cls == "root-input":
        return "HBM-materialized (native)", "runtime tensor: instances.jsonl operand_views (weights alloc / prompt ids alloc)"
    if cls == "instance-output":
        return "HBM-materialized (native)", "runtime tensor: instances.jsonl output_views of the Instance (leaf count == view element count)"
    if cls == "constant":
        return "n/a (Program constant)", "fixed by the Program text; no run-time information"
    _, (kname, origin, interior_cls, note) = kernel_of(pattern)
    return interior_cls, f"{kname} ({origin}): {note}"


def _compiled_location(cls: str, root_spec: str, prim: str) -> tuple[str, str]:
    fam = _family(root_spec)
    if cls == "root-input":
        return "HBM (extern boundary)", "weights / prompt ids are graph inputs"
    if cls == "constant":
        return "n/a (Program constant)", ""
    if cls == "instance-output":
        if fam in COMPILED_EXTERN_FAMILIES:
            return "HBM (extern boundary)", f"{fam} is an extern kernel under every pinned compiled configuration (recon §3 / W1c)"
        if fam in COMPILED_FEEDS_EXTERN:
            if fam.startswith("RMSNormFusedCuda") and prim.startswith("Bf16AddF2fp"):
                return "unknown", "residual return: consumed by a fused kernel; Inductor may materialise or recompute (not established)"
            return "HBM (extern boundary)", COMPILED_FEEDS_EXTERN[fam]
        return "unknown", f"{fam} output consumed by a fused pointwise / reduction kernel; materialisation not established"
    # interior
    if fam.startswith("Attention"):
        return "opaque/library-internal", "FA2 stays a custom op under compile"
    return "unknown", "fused Inductor kernel; IR not inspected"


def aggregate(census: dict, *, n_tokens: int, n_steps: int, n_prompt: int) -> dict:
    per_key: dict[tuple, dict] = {}
    for r in census["rows"]:
        for w, prim, cls, fo in r["out"]:
            eager_cls, eager_ev = _eager_location(cls, r["pattern"])
            comp_cls, comp_ev = _compiled_location(cls, r["root_spec"], prim)
            key = (_family(r["root_spec"]), cls, r["vu_kind"], prim, w, eager_cls, comp_cls, fo)
            d = per_key.setdefault(key, {"root_family": key[0], "logical_class": cls, "vu_kind": r["vu_kind"], "prim": prim, "role": _role(prim, r["vu_kind"]), "width_bits": w,
                                         "eager_class": eager_cls, "eager_evidence": eager_ev, "compiled_class": comp_cls,
                                         "compiled_evidence": comp_ev, "kernel": kernel_of(r["pattern"])[1][0] if cls == "interior" else "",
                                         "fanout": fo, "count": 0, "bits": 0, "classes": 0})
            d["count"] += r["total"]
            d["bits"] += r["total"] * w
            d["classes"] += 1
    groups = sorted(per_key.values(), key=lambda d: -d["bits"])
    for g in groups:
        g["bytes"] = g["bits"] / 8

    def tot(pred) -> tuple[int, float]:
        c = sum(g["count"] for g in groups if pred(g))
        b = sum(g["bits"] for g in groups if pred(g)) / 8
        return c, b

    is_weight = lambda g: g["logical_class"] == "root-input" and g["prim"].startswith("Input16")
    is_prompt = lambda g: g["logical_class"] == "root-input" and not g["prim"].startswith("Input16")
    is_const = lambda g: g["logical_class"] == "constant"
    is_out = lambda g: g["logical_class"] == "instance-output"
    is_int = lambda g: g["logical_class"] == "interior"
    buckets = {
        "weights (root Input16; once per model)": tot(is_weight),
        "prompt token ids (root Input32; once per request)": tot(is_prompt),
        "instance outputs (native HBM)": tot(is_out),
        "interior: opaque/library-internal (FA2)": tot(lambda g: is_int(g) and g["eager_class"].startswith("opaque")),
        "interior: unknown (Triton / vLLM CUDA kernel interiors)": tot(lambda g: is_int(g) and g["eager_class"] == "unknown"),
        "constants (Const32 gates; no commitment needed)": tot(is_const),
    }
    # per-request population (weights excluded, constants excluded) -- the serving-cost quantity
    native = buckets["prompt token ids (root Input32; once per request)"][1] + buckets["instance outputs (native HBM)"][1]
    opaque = buckets["interior: opaque/library-internal (FA2)"][1]
    unknown = buckets["interior: unknown (Triton / vLLM CUDA kernel interiors)"][1]
    const = buckets["constants (Const32 gates; no commitment needed)"][1]
    weights = buckets["weights (root Input16; once per model)"][1]
    req_total = native + opaque + unknown

    def frange(nat, unk, total):
        return {"lower": nat / total if total else None, "upper": (nat + unk) / total if total else None}

    f_free = {
        "per_request_excl_weights_excl_constants": frange(native, unknown, req_total),
        "per_request_excl_weights_incl_constants": frange(native, unknown, req_total + const),
        "all_in_incl_weights_excl_constants": frange(native + weights, unknown, req_total + weights),
    }
    # added HBM traffic if every canonical boundary value had to be exposed (write) and read once for hashing
    added_write = {"lower_bytes": opaque, "upper_bytes": opaque + unknown,
                   "note": "lower = only the opaque values need a new write; upper = also every unknown interior value"}
    read_for_hash = req_total
    # compiled column totals
    comp = collections.Counter()
    for g in groups:
        if is_weight(g) or is_const(g):
            continue
        comp[g["compiled_class"]] += g["bits"] / 8
    per_token = {k: v / n_tokens for k, v in {"native": native, "opaque": opaque, "unknown": unknown, "total": req_total}.items()}
    per_step = {k: v / n_steps for k, v in {"native": native, "opaque": opaque, "unknown": unknown, "total": req_total}.items()}
    # attention share and its T-dependence
    attn_bits = sum(g["bits"] for g in groups if g["root_family"].startswith("Attention"))
    attn_int_bits = sum(g["bits"] for g in groups if g["root_family"].startswith("Attention") and is_int(g))
    # interior values by structural fan-out: absorbable (one consumer: a coarser <= w_max partition removes them) vs intrinsic
    fan = {}
    for lab, pred in (("fan-out 1: absorbable by a coarser <=w_max partition", lambda g: g["fanout"] == "1"),
                      ("fan-out 1 via a composite consumer or a nested return (context-dependent): undetermined", lambda g: g["fanout"] == "1?"),
                      ("fan-out >=2 (or returned + consumed): intrinsic boundary", lambda g: g["fanout"] == ">=2"),
                      ("fan-out unresolved", lambda g: g["fanout"] in ("?", "0"))):
        c, b = tot(lambda g, p=pred: is_int(g) and p(g))
        fan[lab] = {"count": c, "bytes": b}
    return {
        "interior_by_fanout": fan,
        "groups": groups,
        "buckets": {k: {"count": c, "bytes": b} for k, (c, b) in buckets.items()},
        "logical_bytes": {"total_incl_weights_and_constants": sum(g["bits"] for g in groups) / 8, "weights": weights, "constants": const,
                          "per_request_excl_weights_excl_constants": req_total, "native": native, "opaque": opaque, "unknown": unknown},
        "f_free": f_free,
        "dynamic_bytes": {"workload": {"tokens": n_tokens, "steps": n_steps, "prompt_tokens": n_prompt, "decode_tokens": n_tokens - n_prompt},
                          "per_request": {"native": native, "opaque": opaque, "unknown": unknown, "total": req_total},
                          "per_token_avg": per_token, "per_step_avg": per_step,
                          "added_hbm_write_if_all_exposed": added_write, "read_for_hashing_bytes": read_for_hash,
                          "weights_once_per_model_bytes": weights},
        "compiled_column_bytes_per_request": dict(comp),
        "attention": {"bytes": attn_bits / 8, "interior_bytes": attn_int_bits / 8, "share_of_per_request_total": (attn_bits / 8) / req_total if req_total else None,
                      "note": "attention boundary bytes grow with the visible KV length T per (layer, position): sum_T over the workload"},
    }


# --------------------------------------------------------------------------------------------------------------------
# G8
# --------------------------------------------------------------------------------------------------------------------


def run_g8(P, q, run_dir: Path, w_max: int, digest: str, log=print) -> dict:
    """Validate the partition, write the query / use documents bound to the run's fold digest, evaluate gate G8."""
    from verity_capture import gates as G
    from verity_capture import vu_canonical as VC
    from verity_vllm.ir import partition as PT
    from verity_vllm.ir import query_artifact as QA
    t0 = time.time()
    v = PT.validate_vu_partition(P, q, w_max)
    log(f"  validate_vu_partition: status={v.status} partition={v.partition.status} width={v.width.status} max_seen={v.width.max_seen} "
        f"count={v.count:,} readiness32={v.readiness_32.ok} ({time.time() - t0:.1f} s)")
    # bind to the fold's digest (== codec.program_digest(descriptor), checked in load_program); the decoded shell Program
    # cannot be re-encoded, so the digest string is passed rather than the Program object
    qdoc, udoc = QA.vu_use_document(q, digest, v, w_max)
    cdoc = VC.canonical_query_document("sm89")
    qdoc["provenance"] = cdoc["provenance"]
    assert cdoc["query_id"] == qdoc["query_id"]
    (run_dir / "query.json").write_text(json.dumps(qdoc, indent=1, sort_keys=True))
    (run_dir / "query_use.json").write_text(json.dumps(udoc, indent=1, sort_keys=True))
    res = G.gate_G8(G.RunArtifacts(run_dir))
    log(f"  G8: {res.status}  {res.reason or ''}")
    return {"verdict": v.to_json(), "query_id": qdoc["query_id"], "binding_id": qdoc["binding_id"], "program_digest": qdoc["program_digest"],
            "g8": {"status": res.status, "reason": res.reason, "evidence": res.evidence, "detail": res.detail}}


# --------------------------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------------------------


def _mb(b: float) -> str:
    return f"{b / 1e6:,.1f} MB"


def _kb(b: float) -> str:
    return f"{b / 1e3:,.1f} kB"


def render_md(out: dict) -> str:
    A = out["aggregate"]
    W = out["workload"]
    L = A["logical_bytes"]
    D = A["dynamic_bytes"]
    F = A["f_free"]
    lines = [f"# VU boundary census — `{out['run']}` (canonical `W_max = {out['w_max']}` query, {out['utc']})", ""]
    lines.append(f"Program `{out['program_digest'][:16]}…`, profile `{out['profile']}`: {out['gates']:,} gates, {out['root_nodes']:,} root nodes "
                 f"({out['n_instances']:,} Instances + {out['n_input_nodes']} Input batches), **{out['vu_count']:,} canonical VUs** in "
                 f"{out['n_classes']:,} layout classes ({out['n_nonuniform']} non-uniform, {out['n_unsupported']} unsupported).  "
                 f"Workload: {W['prompt_tokens']} prompt + {W['decode_tokens']} generated tokens = {W['tokens']} tokens, {W['steps']} engine steps, "
                 f"{W['layers']} layers.  Tool `verity_capture/bench/vu_census.py` v{VERSION}; wall {out['seconds']:.1f} s.")
    lines.append("")
    if out.get("g8"):
        g = out["g8"]
        v = g["verdict"]
        lines.append(f"**G8:** `{g['g8']['status']}` — partition `{v['partition']['status']}`, width `{v['width']['status']}` (max_seen {v['width']['max_seen']}, "
                     f"violators {len(v['width']['violators'])}, unsupported {len(v['width']['unsupported'])}), readiness-32 `{v['readiness_32']['ok']}`, "
                     f"count {v['count']:,}; `query_id {g['query_id'][:16]}…`, `binding_id {g['binding_id'][:16]}…` bound to fold digest `{g['program_digest'][:16]}…` "
                     f"(`query.json` + `query_use.json` written beside `program.json`).")
        lines.append("")
    lines += ["## 1. Population by logical class (logical bytes = the VU interface)", "",
              "| bucket | values | logical bytes | share of per-request total |", "|---|---:|---:|---:|"]
    req = L["per_request_excl_weights_excl_constants"]
    for k, b in A["buckets"].items():
        share = "" if ("weights" in k or "constants" in k) else f"{100 * b['bytes'] / req:.1f} %"
        lines.append(f"| {k} | {b['count']:,} | {_mb(b['bytes'])} | {share} |")
    lines.append(f"| **per-request total (excl. weights, excl. constants)** | | **{_mb(req)}** | 100 % |")
    lines.append("")
    lines += ["## 2. `f_free` (fraction of canonical boundary bytes already materialised natively) — a range while unknowns remain", "",
              "| denominator | lower = established native / total | upper = (native + unknown) / total |", "|---|---:|---:|"]
    for k, r in F.items():
        lines.append(f"| {k.replace('_', ' ')} | {100 * r['lower']:.1f} % | {100 * r['upper']:.1f} % |")
    aw = D["added_hbm_write_if_all_exposed"]
    lines.append("")
    lines.append(f"Added HBM write traffic if the complete canonical population were exposed, per request: **{_mb(aw['lower_bytes'])} (lower: opaque only) … "
                 f"{_mb(aw['upper_bytes'])} (upper: opaque + unknown)**; bytes read once for hashing: {_mb(D['read_for_hashing_bytes'])} (+ {_mb(D['weights_once_per_model_bytes'])} weights once per model).")
    lines.append("")
    lines += ["## 3. Dynamic bytes (serving-cost quantity; this workload)", "",
              "| scope | native (HBM) | opaque (FA2 interior) | unknown (other interiors) | total |", "|---|---:|---:|---:|---:|"]
    for k, lab in (("per_request", "per request (169 tokens)"), ("per_token_avg", "per token (avg over prefill + decode)"), ("per_step_avg", "per engine step (avg)")):
        d = D[k]
        lines.append(f"| {lab} | {_mb(d['native']) if k == 'per_request' else _kb(d['native'])} | {_mb(d['opaque']) if k == 'per_request' else _kb(d['opaque'])} | "
                     f"{_mb(d['unknown']) if k == 'per_request' else _kb(d['unknown'])} | {_mb(d['total']) if k == 'per_request' else _kb(d['total'])} |")
    at = A["attention"]
    lines.append("")
    lines.append(f"Attention accounts for {_mb(at['bytes'])} of the per-request total ({100 * at['share_of_per_request_total']:.1f} %), {_mb(at['interior_bytes'])} of it interior; "
                 f"{at['note']}.")
    lines.append("")
    lines += ["## 4. Grouped culprits — every (root family, logical class, VU kind, out primitive) group, largest first", "",
              "VU kind = what the part is (its body items); out primitive = the gate whose value crosses the VU boundary.", "",
              ("Fan-out (interior values only): `1` = one consumer VU item — a coarser ≤ w_max partition that merges producer and consumer removes the value "
              "from the boundary without widening any VU; `>=2` = read by two or more VUs (or returned and read) — a boundary value under every ≤ w_max partition; "
              "`1?` = one composite consumer whose refined body may hold several readers."), "",
              "| root family | logical class | VU kind | out primitive (role) | w | fan-out | values | bytes | eager physical class | compiled column |", "|---|---|---|---|---:|---:|---:|---:|---|---|"]
    for g in A["groups"]:
        lines.append(f"| {g['root_family']} | {g['logical_class']} | `{g['vu_kind']}` | `{g['prim']}` ({g['role']}) | {g['width_bits']} | {g['fanout']} | {g['count']:,} | {_mb(g['bytes'])} | "
                     f"{g['eager_class']}{(' — ' + g['kernel']) if g['kernel'] else ''} | {g['compiled_class']} |")
    lines.append("")
    lines += ["### 4a. Interior values by structural fan-out", "", "| class | values | bytes | share of interior |", "|---|---:|---:|---:|"]
    int_total = L["opaque"] + L["unknown"]
    for k, b in A["interior_by_fanout"].items():
        lines.append(f"| {k} | {b['count']:,} | {_mb(b['bytes'])} | {100 * b['bytes'] / int_total if int_total else 0:.1f} % |")
    lines.append("")
    lines += ["## 5. Compiled-production column (structural, unmeasured)", "", "| class | bytes per request |", "|---|---:|"]
    for k, b in sorted(A["compiled_column_bytes_per_request"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {_mb(b)} |")
    lines.append("")
    lines += ["## 6. Evidence and classification rules", "",
              ("- `HBM-materialized (native)`: root inputs and Instance outputs are runtime tensors named in `instances.jsonl` "
              f"(`operand_views` / `output_views`); for every one of the {out['n_specs']} specializations the Program's output leaf count equals the "
              "view element count (checked)."),
              ("- `opaque/library-internal`: interior values of `Attention_v3` — executed by `flash_fwd_splitkv_kernel` (vllm_flash_attn FA2); "
              "the logical values (scores, running max / sum, P, PV partials) exist only inside the kernel and no API exposes their representation."),
              ("- `unknown`: interior values of the vLLM CUDA / Triton kernels (`fused_add_rms_norm_kernel`, `_rms_norm_kernel`).  The kernel sources "
              "keep them in registers / shared memory, but the compiled kernel's IR / SASS was not inspected, so register residency is NOT claimed."),
              "- `constant`: `Const32[...]` gates that the canonical `default(32)` makes their own VUs; their value is fixed by the Program text.",
              "- Disappearance of a tensor boundary under fusion was never used as evidence of register residency.",
              ("- The compiled column applies only the recon's structural facts (GEMM / attention / sampler are extern kernels under every pinned "
              "compiled configuration); everything a fused Inductor kernel may or may not materialise is `unknown`.  Nothing in that column was measured.")]
    if out["nonuniform"]:
        lines += ["", "## 7. Non-uniform classes (members of one activation with different out profiles)", ""]
        for nu in out["nonuniform"][:20]:
            lines.append(f"- class {nu['class']} `{nu['body'][:80]}` ({nu['kind']}, {nu['total']:,} sets): member {nu['member']} differs from the representative")
    if out["unsupported"]:
        lines += ["", "## 8. Unsupported classes", ""]
        for u in out["unsupported"][:20]:
            lines.append(f"- {u}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--profile", default="sm89")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: RUN_DIR/census_vu)")
    ap.add_argument("--g8", action="store_true", help="validate the partition, write query.json / query_use.json, evaluate gate G8")
    ap.add_argument("--samples", type=int, default=3, help="extra members probed per class for uniformity (0 = none)")
    ap.add_argument("--w-max", type=int, default=32)
    a = ap.parse_args(argv)
    run_dir: Path = a.run_dir
    out_dir: Path = a.out or (run_dir / "census_vu")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(s, flush=True)
    log(f"[census] {run_dir}")
    P, _desc, rec, digest = load_program(run_dir)
    n_in = int(rec.get("n_input_nodes", 0))
    by_node, rep = load_instances(run_dir, n_in)
    root_body = P.root.body
    log(f"  program: {P.circuit.size:,} gates, {len(root_body.nodes):,} root nodes, digest {digest[:16]}…")
    # positive materialization evidence: output leaf count == output view element count, per specialization
    mism = []
    for spec, o in rep.items():
        node = root_body.nodes[n_in + o["node"]]
        leaves = node.fn.out_leaves
        elems = sum(math.prod(v["shape"]) if v["shape"] else 1 for v in o["output_views"])
        if leaves != elems:
            mism.append((spec, leaves, elems))
    if mism:
        log(f"  WARNING: {len(mism)} specializations whose output leaves != output view elements: {mism[:3]}")
    from verity_capture import vu_canonical as VC
    from verity.ir import query_ast as GS
    q = VC.canonical_vu_query(a.profile)
    fam = GS.evaluate(P, q)
    n = fam.count()
    log(f"  canonical VUs: {n:,}")
    g8 = run_g8(P, q, run_dir, a.w_max, digest, log) if a.g8 else None
    census = run_census(P, by_node, fam, samples=a.samples, log=log)
    # workload facts from the run (spec.json / experiment.json when present; else from the Program)
    n_prompt = root_body.nodes[0].n if root_body.nodes and root_body.nodes[0].fn.id.startswith("Input32") else None
    n_tokens = sum(1 for k, (s, _) in by_node.items() if s.startswith("Embedding_v1"))
    n_steps = sum(1 for k, (s, _) in by_node.items() if s.startswith("TokenSelect_v1"))
    n_layers = sum(1 for k, (s, _) in by_node.items() if s.startswith("Attention_v3") and s.split("T=")[1].split(",")[0] == "1")
    agg = aggregate(census, n_tokens=n_tokens, n_steps=n_steps, n_prompt=n_prompt or 0)
    out = {"schema": SCHEMA, "version": VERSION, "utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "run": run_dir.name, "run_dir": str(run_dir), "profile": rec.get("profile"), "program_digest": digest, "gates": P.circuit.size,
           "root_nodes": len(root_body.nodes), "n_instances": rec.get("n_instances"), "n_input_nodes": n_in, "n_specs": len(rep),
           "query_profile": a.profile, "query_text": VC.canonical_query_text(a.profile), "query_id": VC.canonical_query_id(a.profile), "w_max": a.w_max,
           "vu_count": n, "n_classes": len(census["rows"]) + len(census["unsupported"]), "n_nonuniform": len(census["nonuniform"]),
           "n_unsupported": len(census["unsupported"]),
           "workload": {"tokens": n_tokens, "prompt_tokens": n_prompt, "decode_tokens": (n_tokens - n_prompt) if n_prompt else None, "steps": n_steps, "layers": n_layers},
           "leaf_view_mismatches": mism, "g8": g8, "aggregate": agg, "classes": census["rows"], "nonuniform": census["nonuniform"],
           "unsupported": census["unsupported"], "kernel_table": {k: list(v) for k, v in KERNELS.items()}, "seconds": time.time() - t0}
    (out_dir / "census.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    summary = {k: v for k, v in out.items() if k != "classes"}          # the per-class rows are the bulk (tens of MB)
    summary["classes_omitted"] = len(out["classes"])
    (out_dir / "census.summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    (out_dir / "census.md").write_text(render_md(out))
    log(f"  wrote {out_dir / 'census.json'}, census.summary.json and census.md ({time.time() - t0:.1f} s)")
    L = agg["logical_bytes"]
    F = agg["f_free"]["per_request_excl_weights_excl_constants"]
    log(f"  per-request logical bytes (excl. weights / constants): {L['per_request_excl_weights_excl_constants'] / 1e6:,.1f} MB = native {L['native'] / 1e6:,.1f} "
        f"+ opaque {L['opaque'] / 1e6:,.1f} + unknown {L['unknown'] / 1e6:,.1f};  f_free in [{100 * F['lower']:.1f} %, {100 * F['upper']:.1f} %]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
