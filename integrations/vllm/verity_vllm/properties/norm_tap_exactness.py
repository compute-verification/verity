"""Norm-tap exactness: the tapped norm kernels the norm-scale source launches compute exactly the installed kernels' outputs, and each
row's scale word is the circuit's scale.

`acquire.sources.norm_scale_source` launches, in place of vLLM's kernels, the tapped CUDA op `verity_norm_tap.fused_add_rms_norm_scale`
(`ops/pod_norm_tap.sh`) and the Triton copy `acquire/norm_tap_src/rms_norm_tap.py`.  One GPU process checks, per kernel and case
(`cases`: bf16 rows at the widths the served models normalise over, and `SPECIAL` rows of zeros, subnormals, huge values, infinities
and NaN):

  (a) the outputs are bit-identical to the installed kernel's (CUDA: `input` and `residual`, both written in place; Triton: `output`),
      and for CUDA to the same build with the tap off; Triton's direct launch is also checked against vLLM's `rms_norm_batch_invariant`;
  (b) every row's scale word is written: two tapped launches over scale buffers filled with `FILLS` agree on every word;
  (c) every scale word equals the circuit's scale bit for bit (`RMSNormFusedCuda_v2`'s `RsqrtApprox` gate, `RMSNormTriton_v1`'s `inv`
      gate): every row against the vectorised twin (`program.kernels.derived_rows`, gate-checked in tests/program/test_derived_rows.py),
      and the first `IR_ROWS` rows of each case, every special row included, against the circuit's own evaluation (`ir_scale`: the gate's
      value in the `verity.ml.circuits.evaluation.evaluate_call` transcript);
  (d) Triton: the installed kernel's source is the pinned one, the copy is that source plus exactly `rms_norm_tap.TAP_LINES`
      (`source_diff`), and at every case's specialization the two kernels' PTX has the same sequence of floating-point, conversion and
      shuffle instructions (`arith_ops`), the copy adding one global store;
  (e) the source (`norm_scale_source`) on vLLM's own norm modules -- `RMSNorm` without and with a residual, a q/k-norm `RMSNorm` over
      [tokens, heads, head_dim], `GemmaRMSNorm` -- leaves every output bit-identical, acquires nothing outside a committed step,
      acquires one `<module>/norm_scale` per module inside one, each equal to the model's scale, and restores every patched site.

This module is the property and its record `norm_tap_exactness.json` (environment, both builds' sha256, per-case results, `ok`), whose
`digest` is the sha256 of the canonical JSON of every other field; `verify_record` re-derives the digest and the verdict without a GPU.
The GPU run is `tests/properties/norm_tap_exactness_gpu.py`.
"""
from __future__ import annotations

import difflib
from typing import Any

from verity_vllm.properties.record import digest_of, seal as seal_record

SCHEMA = "verity-vllm/norm-tap-exactness/v1"
RECORD = "norm_tap_exactness.json"
KERNELS = ("cuda", "triton")
#: two scale-buffer fills (NaN payloads the kernels never write): a word both launches leave at their fill was not written
FILLS = (0x7FC00001, 0x7FC00002)
#: rows per case checked against the circuit's own evaluation (every special row besides)
IR_ROWS = 3
#: (hidden width N, epsilon) of the served norms: SmolLM2, Qwen2.5-1.5B, Llama-3.2-1B / OLMoE / Qwen3-30B-A3B, Gemma-2-2B, Qwen3-4B,
#: Mistral-7B; and the q/k-norm head widths (64, 128, 256)
WIDTHS = ((576, 1e-5), (1536, 1e-6), (2048, 1e-5), (2304, 1e-6), (2560, 1e-6), (4096, 1e-5), (64, 1e-6), (128, 1e-6), (256, 1e-6))
ROWS = (1, 7, 257)
#: special rows, each a pattern over a random row: name -> (bf16 word written, which elements: "all" | "one")
SPECIAL: dict[str, tuple[int, str]] = {"zeros": (0x0000, "all"), "neg_zeros": (0x8000, "all"), "subnormal": (0x0001, "all"),
                                       "huge": (0x7F7F, "all"), "one_huge": (0x7F7F, "one"), "one_inf": (0x7F80, "one"),
                                       "one_neg_inf": (0xFF80, "one"), "one_nan": (0x7FC0, "one")}
QUICK = ("cuda_n2048_r7", "triton_n2048_r7", "cuda_n128_special", "triton_n128_special")


def _case(kernel: str, N: int, eps: float, rows: int, *, weight: bool = True, special: bool = False, residual_special: bool = False) -> dict[str, Any]:
    name = f"{kernel}_n{N}_" + ("special" + ("_res" if residual_special else "") if special else f"r{rows}") + ("" if weight else "_noweight")
    return {"name": name, "kernel": kernel, "N": N, "eps": eps, "rows": len(SPECIAL) if special else rows, "weight": weight, "special": special,
            "residual_special": residual_special}


def cases(quick: bool = False) -> list[dict[str, Any]]:
    """Every (kernel, width, rows) of `WIDTHS` x `ROWS`; the special rows at a head width and a hidden width (for CUDA also in the
    residual operand); and one case per kernel without a weight (`HAS_WEIGHT` false / `weight` None)."""
    cs = [_case(k, N, eps, r) for k in KERNELS for N, eps in WIDTHS for r in ROWS]
    for k in KERNELS:
        for N, eps in ((128, 1e-6), (2048, 1e-5)):
            cs.append(_case(k, N, eps, 0, special=True))
            if k == "cuda":
                cs.append(_case(k, N, eps, 0, special=True, residual_special=True))
        cs.append(_case(k, 2048, 1e-5, 7, weight=False))
    return [c for c in cs if c["name"] in QUICK] if quick else cs


# ------------------------------------------------------------------------------------------------------------------------ the circuit
def _scale_gate(kernel: str, call: Any) -> int:
    prims = [c for c in call.children() if c.is_primitive]
    if kernel == "cuda":
        return next(c for c in prims if c.name == "RsqrtApprox").lo
    div = next(c for c in prims if c.name == "DivFullScaleA")
    return next(c for c in prims if c.name == "F32Mul" and div.lo in c.args).lo


def ir_scale(kernel: str, N: int, eps: float, x: list[int], res: list[int] | None, w: list[int]) -> int:
    """The circuit's scale word of one row: evaluate `RMSNormFusedCuda_v2{N,EPS}` (x, res, w) or `RMSNormTriton_v1{N,EPS}` (x, w) as a
    standalone Call and read its scale gate from the transcript."""
    from verity.ml.circuits.evaluation.reference import standalone_call
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate_call
    from verity.ml.definitions.registry import b1

    spec = bind(b1.RMSNormFusedCuda if kernel == "cuda" else b1.RMSNormTriton, N=N, EPS=eps)
    prog, call = standalone_call(spec)
    t: dict[int, int] = {}
    evaluate_call(prog.circuit, call, [int(v) for v in x] + ([int(v) for v in res] if kernel == "cuda" else []) + [int(v) for v in w], transcript=t)
    return int(t[_scale_gate(kernel, call)])


def twin_scales(kernel: str, x: Any, res: Any, w: Any, eps: float) -> Any:
    """Every row's circuit scale word by the vectorised twin (uint32): `derived_rows.rmsnorm_fused_v2_rows(...)["s"]` / `rmsnorm_triton_rows(...)["inv"]`."""
    import numpy as np

    from verity.ml.definitions.numerics import derived_rows as D

    if kernel == "cuda":
        return np.asarray(D.rmsnorm_fused_v2_rows(x, res, w, eps)["s"], dtype=np.uint32)
    return np.asarray(D.rmsnorm_triton_rows(x, w, eps)["inv"], dtype=np.uint32)


# ------------------------------------------------------------------------------------------------------------------------ Triton
def source_diff(installed: str, tapped: str, tap_lines: tuple[str, ...]) -> dict[str, Any]:
    """The line diff of the installed kernel's source against the copy's: `ok` when the copy only inserts `tap_lines`
    (`rms_norm_tap.TAP_LINES`), in order."""
    a, b = installed.splitlines(keepends=True), tapped.splitlines(keepends=True)
    ops = [(tag, a[i1:i2], b[j1:j2]) for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != "equal"]
    inserted = [line for tag, _old, new in ops if tag == "insert" for line in new]
    return {"changes": [{"op": tag, "removed": old, "added": new} for tag, old, new in ops], "inserted": inserted,
            "ok": all(tag == "insert" for tag, _o, _n in ops) and tuple(inserted) == tuple(tap_lines)}


_FLOAT_TYPES = (".f32", ".f64", ".f16", ".bf16")
_ARITH = frozenset({"add", "sub", "mul", "mad", "fma", "div", "rcp", "sqrt", "rsqrt", "ex2", "lg2", "sin", "cos", "tanh", "max", "min",
                    "neg", "abs", "cvt", "selp", "setp", "testp", "copysign"})


def _instructions(ptx: str) -> list[str]:
    out = []
    for line in ptx.splitlines():
        s = line.strip()
        if not s or s[0] in "/.{}$" or s.endswith(":"):
            continue
        if s[0] == "@":
            s = s.split(None, 1)[1] if " " in s else ""
        if s:
            out.append(s.split(None, 1)[0].rstrip(";"))
    return out


def arith_ops(ptx: str) -> list[str]:
    """The floating-point arithmetic of a PTX listing, in order: every instruction on a float type (add, fma, div, sqrt, rcp, cvt, selp,
    setp, ...) and every warp shuffle; loads, stores, moves, branches and integer address arithmetic are left out."""
    return [op for op in _instructions(ptx) if op.startswith("shfl") or (op.split(".", 1)[0] in _ARITH and any(t in op for t in _FLOAT_TYPES))]


def global_stores(ptx: str) -> int:
    return sum(1 for op in _instructions(ptx) if op.startswith("st.global"))


def ptx_check(installed: str, tapped: str) -> dict[str, Any]:
    a, b = arith_ops(installed), arith_ops(tapped)
    return {"arith_ops": len(a), "arith_equal": a == b, "extra_global_stores": global_stores(tapped) - global_stores(installed),
            "first_arith_difference": next(({"at": i, "installed": x, "tapped": y} for i, (x, y) in enumerate(zip(a, b)) if x != y), None)}


# ------------------------------------------------------------------------------------------------------------------------ record
def case_ok(e: dict[str, Any]) -> bool:
    ok = (not e.get("error") and e.get("out_eq_installed") is True and e.get("unwritten") == 0 and e.get("scale_ne_twin") == 0
          and e.get("scale_ne_ir") == 0 and int(e.get("ir_rows") or 0) > 0)
    if e.get("kernel") == "cuda":
        return ok and e.get("out_eq_untapped") is True
    ptx = e.get("ptx") or {}
    return ok and e.get("direct_eq_vllm") is True and ptx.get("arith_equal") is True and ptx.get("extra_global_stores") == 1


def triton_source_ok(doc: dict[str, Any]) -> bool:
    t = doc.get("triton") or {}
    return t.get("installed_sha256") == t.get("pinned_sha256") and bool(t.get("pinned_sha256")) and (t.get("diff") or {}).get("ok") is True


def source_ok(s: dict[str, Any]) -> bool:
    return (not s.get("error") and s.get("outputs_equal") is True and s.get("outside_step_acquired") == 0 and s.get("names_ok") is True
            and s.get("scales_equal") is True and s.get("restored") is True and s.get("errors") == 0)


def _verdict(doc: dict[str, Any]) -> bool:
    cs = doc.get("cases") or []
    return ({c.get("kernel") for c in cs} >= set(KERNELS) and all(map(case_ok, cs)) and triton_source_ok(doc)
            and source_ok(doc.get("source") or {}))


def seal(doc: dict[str, Any]) -> dict[str, Any]:
    """`doc` with `ok` (cases of both kernels, every one ok, the Triton source pinned and diffed, the source check ok) and its `digest`."""
    return seal_record(doc, _verdict(doc))


def verify_record(doc: dict[str, Any]) -> list[str]:
    """What is wrong with a record: its schema, its digest, and every `ok` re-derived from the fields beside it (empty = intact and
    consistent)."""
    problems = [] if doc.get("schema") == SCHEMA else [f"schema {doc.get('schema')!r}, not {SCHEMA}"]
    if doc.get("digest") != digest_of(doc):
        problems.append("digest does not match the record")
    for e in doc.get("cases") or []:
        if bool(e.get("ok")) != case_ok(e):
            problems.append(f"case {e.get('name')}: ok={e.get('ok')} but its fields say {case_ok(e)}")
    s = doc.get("source") or {}
    if s and bool(s.get("ok")) != source_ok(s):
        problems.append(f"source: ok={s.get('ok')} but its fields say {source_ok(s)}")
    if bool(doc.get("ok")) != _verdict(doc):
        problems.append(f"ok={doc.get('ok')} but the entries say {_verdict(doc)}")
    return problems
