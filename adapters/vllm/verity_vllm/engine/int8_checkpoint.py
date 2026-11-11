"""Int8 checkpoints for the exact-int8 linears, made offline from a pinned checkpoint: the recipes ``verity-int8-channel127/v1``
and ``/v2`` from a BF16 base, ``/v3`` from an FP8-block base.

    verity-vllm int8-checkpoint make --role ROLE [--hf-home DIR] [--jobs N]
                                                                          re-make a registered int8 pin (under its own recipe) and check
                                                                          every file's sha256
    verity-vllm int8-checkpoint pin --base-role ROLE --role ROLE --repo ID [--recipe ID] [--hf-home DIR] [--jobs N]
                                                                          make one and print its new manifest entry
    verity-vllm int8-checkpoint report --role ROLE --out FILE [--hf-home DIR] [--jobs N]
                                                                          a made pin's error against its base, per tensor (v3)

The weights the plain step reads (``engine.int8_exact``, ``TargetProfile.int8_exact``): ``benchmarks/ppl/steps.py``'s
``quant_rows(w.float(), 127)``, int8 with every entry in [-127, 127] and one float32 scale per output row.  Like
``engine.fp8_checkpoint``'s recipe, on its container (shard layout, header, index, copied files, pinned revision), it is
data-free and a pure function of the base checkpoint's bytes; the linear weights are the same tensors (every 2-D tensor none of
whose dimensions is ``vocab_size``).  For every such W [N, K]:

* W is read exactly as float32 (BF16, F16 and F32 widen exactly);
* for every row n: a = max |W[n, :]|, and s = max(a, RN32(1e-30)) / 127 in float32 (one round-to-nearest-even division);
* the stored weight is q = clamp(rint(W / s), -127, 127): the float32 quotient (round-to-nearest-even), rounded to the nearest
  integer, ties to even, as int8;
* stored as ``<name>`` I8 [N, K] and ``<module>.weight_scale`` F32 [N, 1]; the dequantized weight is q * s.

N and K must be multiples of 8 (``torch._int_mm``'s shape rule on CUDA), else the checkpoint is refused whole.  config.json gets
``quantization_config = {"modules_to_not_convert": ["lm_head"], "quant_method": "verity_int8", "weight_max": 127}``, the vLLM
method ``engine.int8_vllm`` registers.  The pin's ``int8_of`` names its base.  Int8 pins live in
``manifests/checkpoints-int8.json`` (`config.SIBLING_MANIFESTS`), which ``config.checkpoint_entries`` reads with ``checkpoints.json``.

A mixture of experts whose experts are stored as 2-D tensors (OLMoE's ``mlp.experts.<e>.{gate,up,down}_proj.weight``) is made too
(``fp8_checkpoint``'s ``moe``): each expert's projections are linears like any other, served by ``engine.int8_vllm``'s
``Int8ExpertsMethod``, and the routers, the [num_experts, hidden_size] tensors vLLM builds with no quantization config, are
copied as they are (bf16).

``verity-int8-channel127/v2`` is v1 plus the LM head: ``lm_head.weight`` [vocab_size, K] is made by the same row rule (an int8
``lm_head.weight`` and an F32 ``lm_head.weight_scale`` [vocab_size, 1]), and ``modules_to_not_convert`` is ``[]``, so vLLM's
``ParallelLMHead`` serves the int8 linear too.  The embedding stays as the base has it (a gather, not a product).  A config with
``tie_word_embeddings`` true or absent is refused, as is a checkpoint without an ``lm_head.weight`` of its own.

``verity-int8-channel127/v3`` reads a block-FP8 base (``quantization_config``: ``quant_method: fp8``, e4m3, a square
``weight_block_size`` [G, G]; GLM-5.3, a mixture of experts, is the first), on the same container.  Its linears are read off the
structure: every F8_E4M3 ``<name>`` [N, K] beside its F32 ``<name>_scale_inv`` [ceil(N/G), ceil(K/G)] in the same shard, the routed
and shared experts' included (the base stores each expert's projections as their own 2-D tensors).

* W = e4m3(q) * s, q's value times its block's scale.  That product is not exact in float32 for such a base (GLM-5.3's scales
  are amax / 448, full 24-bit significands, and f32 rounds about three products in four), but it is exact in float64: an e4m3
  value has at most 4 significant bits and a float32 at most 24, so the product has at most 28, and its exponent is far inside
  float64's range.  v3 reads W exactly, in float64.
* The row rule is v1's on that exact W: s = RN32(max(a, RN32(1e-30)) / 127) with a = max |W[n, :]| (exact), and
  q = clamp(rint(W / s), -127, 127) with the quotient's nearest integer, ties to even.  Both are computed in float64 and are the
  exact values' roundings: a / 127 rounded to float64 then to float32 equals its one rounding to float32, and rint of the float64
  quotient equals rint of the exact one, because an exact quotient that is not itself a float32 midpoint (resp. a half-integer)
  lies at least 2^-32 (resp. 2^-42) of its size from one, farther than float64's half ulp 2^-53 (W, s and the midpoint all sit
  on lattices fixed by their 24- and 28-bit significands).
* The LM head is made as v2 makes it (the BF16 ``lm_head.weight`` widened exactly, then the same exact rule): int8 with row scales.
* Two FP8 linears are not products vLLM serves as an int8 GEMM, so v3 stores them in BF16 = RN_bf16(RN32(e4m3(q) * s)), the words
  vLLM itself upcasts them to from this base (``check.fp8_ue8m0.served_words``), and lists them in ``modules_to_not_convert``:
  - MLA's ``kv_b_proj`` (input width ``kv_lora_rank``): vLLM absorbs it into the attention (``W_UK_T``, ``W_UV``), where its
    output rows are contracted against the query, so a scale per output row is not a GEMM output scale there;
  - the DSA indexer's ``wk`` (output width ``index_head_dim``): vLLM fuses it with the BF16 ``weights_proj`` into one BF16 linear.
* Every other tensor is copied byte for byte, so what the base keeps out of FP8 stays as it is: the embedding, the norms, the
  router (``mlp.gate`` and its ``e_score_correction_bias``), the indexer's ``weights_proj`` and ``k_norm``, and the MTP layer's
  ``eh_proj`` and norms.
* N and K of every int8 tensor must be multiples of 8 (v1's rule), a 3-D tensor, an F8 tensor without its scale (or a scale
  without its weight), an e4m3 NaN code and a non-finite scale are refused, as are tied embeddings and a base without its own
  ``lm_head.weight``.
* config.json's ``quantization_config`` is v1's with ``modules_to_not_convert`` = the module of every 2-D float ``.weight`` the made
  checkpoint holds, sorted.

``report`` reads a v3 pin beside its base and writes, per made tensor, max |q·s − W|, that over max |W| and the Frobenius ratio
||q·s − W|| / ||W||, the worst rows by their own ratio, how many products of W are inexact in float32, and how many stored words
differ from the rule recomputed (0, or the pin is not the recipe's).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import config as C
from verity_vllm.config import option
from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine import int8_exact as E
from verity_vllm.program.frontend.target_profile import TargetProfile

RECIPE = "verity-int8-channel127/v1"
RECIPE_V2 = "verity-int8-channel127/v2"
RECIPE_V3 = "verity-int8-channel127/v3"
QUANTIZATION = {"quant_method": TargetProfile.INT8_EXACT_QUANTIZATION, "weight_max": E.W_MAX}


def channel_scales(w: np.ndarray) -> np.ndarray:
    """s [N, 1] float32 of a float32 [N, K]: max |W| per row, floored at 1e-30, / 127 (steps.py's `quant_rows`)."""
    amax = np.abs(w).max(axis=1, keepdims=True)
    if not np.isfinite(amax).all():
        raise ValueError("non-finite weight")
    return (np.maximum(amax, np.float32(E.SCALE_FLOOR)) / np.float32(E.W_MAX)).astype(np.float32)


def quantize(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """q [N, K] int8 of a float32 [N, K] under its row scales `s` [N, 1]."""
    return np.clip(np.rint(w / s), -E.W_MAX, E.W_MAX).astype(np.int8)


def _int8_refusal(n: int, k: int) -> str | None:
    return "not a multiple of 8 on both axes" if n % 8 or k % 8 else None


INT8 = F.Recipe(id=RECIPE, weight_dtype="I8", scale_suffix="weight_scale", rows_per_scale=1, scale_shape=lambda n, k: [n, 1],
                refusal=_int8_refusal, scales=channel_scales, quantize=quantize, quantization=QUANTIZATION, of="int8_of", moe=True)
INT8_V2 = replace(INT8, id=RECIPE_V2, lm_head=True)


# --------------------------------------------------------------------------------------------------------- v3: an FP8-block base

#: e4m3fn code -> its value, exact in float64 (NaN at 0x7F and 0xFF)
E4M3_VALUES = F.e4m3fn_value(np.arange(256, dtype=np.uint8)).astype(np.float64)
FLOOR = np.float64(np.float32(E.SCALE_FLOOR))
#: an FP8 linear's block scales, beside it: `<name>` + this
SCALE_INV = "_scale_inv"


def exact_scales(w: np.ndarray) -> np.ndarray:
    """s [N, 1] float32 of an exact float64 W [N, K]: RN32(max(max |W[n, :]|, RN32(1e-30)) / 127), one rounding of the exact value."""
    amax = np.abs(w).max(axis=1, keepdims=True)
    if not np.isfinite(amax).all():
        raise ValueError("non-finite weight")
    return (np.maximum(amax, FLOOR) / np.float64(E.W_MAX)).astype(np.float32)


def exact_quantize(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """q [N, K] int8 of an exact float64 W [N, K] under float32 row scales `s` [N, 1]: the exact quotient's nearest integer."""
    return np.clip(np.rint(w / s.astype(np.float64)), -E.W_MAX, E.W_MAX).astype(np.int8)


def _block_rows(s: np.ndarray, lo: int, rows: int, k: int, block: int) -> np.ndarray:
    """The block scale of every entry of rows lo..lo+rows of W [N, k], as [rows, k]."""
    return np.repeat(s[np.arange(lo, lo + rows) // block], block, axis=1)[:, :k]


def fp8_exact(codes: np.ndarray, s: np.ndarray, lo: int, block: int) -> np.ndarray:
    """W [r, K] float64, exact, of e4m3 codes [r, K] (rows lo.. of the tensor) under block scales s float32 [ceil(N/G), ceil(K/G)]."""
    v = E4M3_VALUES[codes]
    if np.isnan(v).any():
        raise ValueError("an e4m3 NaN code")
    return v * _block_rows(s, lo, *codes.shape, block).astype(np.float64)


def fp8_served_bf16(codes: np.ndarray, s: np.ndarray, lo: int, block: int) -> np.ndarray:
    """bf16 words [r, K] of e4m3 codes under block scales: RN_bf16(RN32(e4m3(q) * s)), vLLM's upcast of an FP8 linear to BF16."""
    v = F.e4m3fn_value(codes)
    if np.isnan(v).any():
        raise ValueError("an e4m3 NaN code")
    u = np.ascontiguousarray(v * _block_rows(s, lo, *codes.shape, block), dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


@dataclass(frozen=True)
class Fp8Base:
    """What v3 reads off a block-FP8 base's config.json: the vocabulary, the block G, and the widths that place MLA's absorbed
    `kv_b_proj` (its input, `kv_lora_rank`) and the DSA indexer's `wk` (its output, `index_head_dim`); None where the model has none."""

    vocab: int
    block: int
    kv_lora_rank: int | None
    index_head_dim: int | None


def fp8_base(cfg: dict[str, Any], recipe: F.Recipe) -> Fp8Base:
    qc = cfg.get("quantization_config") or {}
    bs = qc.get("weight_block_size")
    if (qc.get("quant_method") != "fp8" or qc.get("fmt", "e4m3") != "e4m3" or not isinstance(bs, list) or len(bs) != 2
            or bs[0] != bs[1] or not isinstance(bs[0], int) or bs[0] <= 0):
        raise ValueError(f"{recipe.id} reads a block-FP8 base (quantization_config: quant_method fp8, e4m3, a square weight_block_size); "
                         f"this one has quant_method {qc.get('quant_method')!r}, fmt {qc.get('fmt')!r}, weight_block_size {bs!r}")
    if not isinstance(cfg.get("vocab_size"), int):
        raise ValueError("config.json names no integer vocab_size")
    if cfg.get("tie_word_embeddings", True) is not False:
        raise ValueError(f"tie_word_embeddings is {cfg.get('tie_word_embeddings', '<absent>')!r}: {recipe.id} quantizes the LM head, "
                         "which a tied checkpoint shares with the embedding")
    return Fp8Base(cfg["vocab_size"], bs[0], cfg.get("kv_lora_rank"), cfg.get("index_head_dim"))


def fp8_base_layout(header: dict[str, Any], plan: Fp8Base, recipe: F.Recipe) -> list[tuple[str, str, list[int], str, str]]:
    """The made shard's tensors in file order: (name, dtype, shape, kind, source), kind "copy" | "weight" | "scale" | "dequant"."""
    out, scales = [], set()
    G = plan.block
    for name, info in header.items():
        if name == "__metadata__":
            continue
        shape, dtype = [int(d) for d in info["shape"]], info["dtype"]
        if len(shape) > 2:
            raise ValueError(f"{name}: a {len(shape)}-D tensor: {recipe.id} reads 2-D linears only")
        if dtype == "F8_E4M3":
            if len(shape) != 2 or not name.endswith(".weight"):
                raise ValueError(f"{name}: an F8_E4M3 tensor that is not a 2-D `.weight`")
            n, k = shape
            want, si = [-(-n // G), -(-k // G)], header.get(name + SCALE_INV)
            if si is None or si["dtype"] != "F32" or [int(d) for d in si["shape"]] != want:
                got = None if si is None else (si["dtype"], si["shape"])
                raise ValueError(f"{name}: no F32 {name + SCALE_INV} {want} beside it in this shard (got {got})")
            scales.add(name + SCALE_INV)
            if k == plan.kv_lora_rank or n == plan.index_head_dim:
                out.append((name, "BF16", shape, "dequant", name))
                continue
        elif name == F.LM_HEAD:
            if dtype not in F._WIDEN or len(shape) != 2 or shape[0] != plan.vocab:
                raise ValueError(f"{name}: {dtype} {shape} is not a float [vocab_size {plan.vocab}, K]")
        else:
            if dtype not in F._ELEM:
                raise ValueError(f"{name}: unknown safetensors dtype {dtype}")
            out.append((name, dtype, shape, "copy", name))
            continue
        n, k = shape
        if why := recipe.refusal(n, k):
            raise ValueError(f"{name}: [{n}][{k}] is {why}")
        out.append((name, recipe.weight_dtype, shape, "weight", name))
        out.append((F.scale_name(name, recipe), "F32", recipe.scale_shape(n, k), "scale", name))
    stray = sorted(n for n, d, _, kind, _ in out if kind == "copy" and n.endswith(SCALE_INV) and n not in scales)
    if stray:
        raise ValueError(f"block scales with no F8_E4M3 weight beside them: {stray[:3]}")
    out = [t for t in out if t[0] not in scales]
    names = [t[0] for t in out]
    if len(set(names)) != len(names) or set(names) & scales:
        raise ValueError(f"a made tensor's name collides with a base tensor: {sorted({n for n in names if names.count(n) > 1})[:3]}")
    return sorted(out, key=lambda t: (-F._ELEM[t[1]], t[0]))


class _Base:
    """Rows of a base shard's 2-D tensors, read exactly (float64), or as their raw words."""

    def __init__(self, f, base: int, header: dict[str, Any], plan: Fp8Base):
        self.f, self.base, self.header, self.plan, self._scales = f, base, header, plan, {}

    def raw(self, name: str, lo: int, hi: int) -> bytes:
        info = self.header[name]
        k, e = int(info["shape"][1]), F._ELEM[info["dtype"]]
        a, _ = info["data_offsets"]
        return F._read(self.f, self.base + a + lo * k * e, self.base + a + hi * k * e)

    def block_scales(self, name: str) -> np.ndarray:
        if name not in self._scales:
            si = self.header[name + SCALE_INV]
            a, b = si["data_offsets"]
            s = np.frombuffer(F._read(self.f, self.base + a, self.base + b), dtype="<f4").reshape(si["shape"])
            if not np.isfinite(s).all():
                raise ValueError(f"{name + SCALE_INV}: a non-finite block scale")
            self._scales[name] = s
        return self._scales[name]

    def codes(self, name: str, lo: int, hi: int) -> np.ndarray:
        return np.frombuffer(self.raw(name, lo, hi), dtype=np.uint8).reshape(hi - lo, int(self.header[name]["shape"][1]))

    def exact(self, name: str, lo: int, hi: int) -> np.ndarray:
        info = self.header[name]
        if info["dtype"] == "F8_E4M3":
            return fp8_exact(self.codes(name, lo, hi), self.block_scales(name), lo, self.plan.block)
        return F._WIDEN[info["dtype"]](self.raw(name, lo, hi)).reshape(hi - lo, int(info["shape"][1])).astype(np.float64)

    def served_bf16(self, name: str, lo: int, hi: int) -> np.ndarray:
        return fp8_served_bf16(self.codes(name, lo, hi), self.block_scales(name), lo, self.plan.block)


def _stripes(n: int):
    for lo in range(0, n, F.STRIPE):
        yield lo, min(lo + F.STRIPE, n)


def fp8_base_shard(src: Path, dst: Path, plan: Fp8Base, recipe: F.Recipe) -> tuple[dict[str, Any], dict[str, str], int, int]:
    """Write v3's shard of `src` to `dst`: (file digest, tensor -> shard name, vocabulary-sized tensors, data bytes)."""
    base, header = F.read_header(src)
    layout = fp8_base_layout(header, plan, recipe)
    head, offsets = F.encode_header(layout)
    with open(src, "rb") as f:
        rd = _Base(f, base, header, plan)
        scales = {source: np.concatenate([recipe.scales(rd.exact(source, lo, hi)) for lo, hi in _stripes(shape[0])])
                  for _, _, shape, kind, source in layout if kind == "scale"}
        sink = F._Sink(dst)
        sink.write(head)
        for name, _, shape, kind, source in layout:
            if kind == "scale":
                sink.write(scales[source].astype("<f4").tobytes())
            elif kind == "weight":
                for lo, hi in _stripes(shape[0]):
                    sink.write(recipe.quantize(rd.exact(source, lo, hi), scales[source][lo:hi]).tobytes())
            elif kind == "dequant":
                for lo, hi in _stripes(shape[0]):
                    sink.write(rd.served_bf16(source, lo, hi).astype("<u2").tobytes())
            else:
                a, b = header[source]["data_offsets"]
                for at in range(a, b, F._CHUNK):
                    sink.write(F._read(f, base + at, base + min(at + F._CHUNK, b)))
        digest = sink.close()
    n_vocab = sum(1 for n, i in header.items() if n != "__metadata__" and len(i["shape"]) == 2 and plan.vocab in i["shape"])
    return digest, {name: dst.name for name, *_ in layout}, n_vocab, sum(b - a for a, b in offsets.values())


def fp8_base_config(base_config: bytes, recipe: F.Recipe, tensors: dict[str, tuple[str, list[int]]]) -> bytes:
    cfg = json.loads(base_config)
    fp8_base(cfg, recipe)
    kept = sorted(n[: -len(".weight")] for n, (dt, shape) in tensors.items() if len(shape) == 2 and dt in F._WIDEN and n.endswith(".weight"))
    cfg["quantization_config"] = dict(sorted(dict(recipe.quantization, modules_to_not_convert=kept).items()))
    return (json.dumps(cfg, indent=2, ensure_ascii=False) + "\n").encode()


INT8_V3 = replace(INT8, id=RECIPE_V3, lm_head=True, scales=exact_scales, quantize=exact_quantize, plan=fp8_base, shard=fp8_base_shard,
                  config=fp8_base_config)
RECIPES = {r.id: r for r in (INT8, INT8_V2, INT8_V3)}


# ------------------------------------------------------------------------------------------------------------- v3's error report

def _tensor_error(w: np.ndarray, d: np.ndarray, acc: dict[str, Any], lo: int) -> None:
    """Fold rows lo.. of W (exact) and their made value `d` into a tensor's running error."""
    e = d - w
    ae, aw = np.abs(e), np.abs(w)
    row_e, row_w = np.sqrt((e * e).sum(axis=1)), np.sqrt((w * w).sum(axis=1))
    acc["max_abs"] = max(acc["max_abs"], float(ae.max()))
    acc["amax"] = max(acc["amax"], float(aw.max()))
    acc["sq_e"] += float((e * e).sum())
    acc["sq_w"] += float((w * w).sum())
    acc["inexact_f32"] += int((w.astype(np.float32).astype(np.float64) != w).sum())
    ratio = np.divide(row_e, row_w, out=np.zeros_like(row_e), where=row_w > 0)
    for i in np.argsort(-ratio, kind="stable")[:3]:
        acc["rows"].append({"row": lo + int(i), "rel_fro": float(ratio[i]), "max_abs": float(ae[i].max()), "amax": float(aw[i].max())})
    acc["rows"] = sorted(acc["rows"], key=lambda r: (-r["rel_fro"], r["row"]))[:3]


def report_shard(base_path: Path, made_path: Path, plan: Fp8Base, recipe: F.Recipe) -> dict[str, dict[str, Any]]:
    """Per made weight of one shard: its error against the base's exact W (`report`'s fields)."""
    bstart, bheader = F.read_header(base_path)
    mstart, mheader = F.read_header(made_path)
    layout = fp8_base_layout(bheader, plan, recipe)
    out = {}
    with open(base_path, "rb") as bf, open(made_path, "rb") as mf:
        rd, md = _Base(bf, bstart, bheader, plan), _Base(mf, mstart, mheader, plan)
        for name, dtype, shape, kind, source in layout:
            if kind not in ("weight", "dequant"):
                continue
            if [mheader[name]["dtype"], [int(d) for d in mheader[name]["shape"]]] != [dtype, shape]:
                raise ValueError(f"{made_path.name}: {name} is {mheader[name]['dtype']} {mheader[name]['shape']}, not {recipe.id}'s {dtype} {shape}")
            acc = {"kind": "int8" if kind == "weight" else "bf16", "base_dtype": bheader[source]["dtype"], "shape": shape, "max_abs": 0.0,
                   "amax": 0.0, "sq_e": 0.0, "sq_w": 0.0, "inexact_f32": 0, "mismatch": 0, "rows": []}
            if kind == "weight":
                sn = F.scale_name(name, recipe)
                a, b = mheader[sn]["data_offsets"]
                s = np.frombuffer(F._read(mf, mstart + a, mstart + b), dtype="<f4").reshape(shape[0], 1)
            for lo, hi in _stripes(shape[0]):
                w = rd.exact(source, lo, hi)
                if kind == "weight":
                    q = np.frombuffer(md.raw(name, lo, hi), dtype=np.int8).reshape(hi - lo, shape[1])
                    s_rule = recipe.scales(w)
                    acc["mismatch"] += int((s_rule != s[lo:hi]).sum()) + int((recipe.quantize(w, s_rule) != q).sum())
                    d = q.astype(np.float64) * s[lo:hi].astype(np.float64)
                else:
                    u = np.frombuffer(md.raw(name, lo, hi), dtype="<u2").reshape(hi - lo, shape[1])
                    acc["mismatch"] += int((rd.served_bf16(source, lo, hi) != u).sum())
                    d = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                _tensor_error(w, d, acc, lo)
            sq_e, sq_w = acc.pop("sq_e"), acc.pop("sq_w")
            acc["rel_fro"] = float(np.sqrt(sq_e / sq_w)) if sq_w else 0.0
            acc["rel_max"] = acc["max_abs"] / acc["amax"] if acc["amax"] else 0.0
            acc["elements"] = shape[0] * shape[1]
            out[name] = acc
    return out


def summarize(tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The report's headline: totals, and the worst tensors by each measure, per kind."""
    out: dict[str, Any] = {"tensors": len(tensors), "elements": sum(t["elements"] for t in tensors.values()),
                           "inexact_f32": sum(t["inexact_f32"] for t in tensors.values()),
                           "mismatch": sum(t["mismatch"] for t in tensors.values())}
    for kind in sorted({t["kind"] for t in tensors.values()}):
        ts = {n: t for n, t in tensors.items() if t["kind"] == kind}
        rel = np.array([t["rel_fro"] for t in ts.values()])
        out[kind] = {"tensors": len(ts), "rel_fro_quantiles": {q: float(np.quantile(rel, float(q))) for q in ("0.5", "0.9", "0.99", "1.0")},
                     **{f"worst_{m}": [dict(name=n, **{k: ts[n][k] for k in ("rel_fro", "rel_max", "max_abs", "amax")})
                                       for n in sorted(ts, key=lambda n: (-ts[n][m], n))[:5]] for m in ("rel_fro", "rel_max", "max_abs")}}
    return out


def report(entry: dict[str, Any], entries: list[dict[str, Any]], hf_home: str | None, jobs: int = 1) -> dict[str, Any]:
    recipe = recipe_of(entry)
    if recipe.shard is not fp8_base_shard:
        raise C.UsageError(f"{entry.get('role')}: `report` reads a v3 pin (an FP8-block base); this one is {recipe.id}")
    base = F.base_entry(entry, entries, recipe)
    bdir = F.snapshot_dir(hf_home, base["repo"], base["revision"]) if hf_home else Path(base["local_path"])
    mdir = F.snapshot_dir(hf_home, entry["repo"], entry["revision"]) if hf_home else Path(entry["local_path"])
    plan = fp8_base(json.loads((bdir / F.CONFIG).read_bytes()), recipe)
    shards = sorted(base["safetensors_sha256"])
    tensors: dict[str, dict[str, Any]] = {}
    for part in F.map_shards(report_shard, [(bdir / fn, mdir / fn, plan, recipe) for fn in shards], jobs):
        tensors.update(part)
    return {"schema": "verity-vllm/int8-error-report/v1", "role": entry["role"], "recipe": recipe.id, "revision": entry["revision"],
            "base": {"repo": base["repo"], "revision": base["revision"]}, "summary": summarize(tensors),
            "tensors": dict(sorted(tensors.items()))}


def recipe_of(entry: dict) -> F.Recipe:
    """The int8 recipe a pin was made by (its `recipe.id`)."""
    rid = (entry.get("recipe") or {}).get("id")
    if rid not in RECIPES:
        raise C.UsageError(f"{entry.get('role')}: recipe {rid!r} is none of the int8 recipes {sorted(RECIPES)}")
    return RECIPES[rid]


@dataclass(frozen=True, kw_only=True)
class MakeOptions:
    """Re-make a registered int8 pin (`recipe` in checkpoints-int8.json) from its base pin and check every file's sha256."""

    role: str = option("--role", help="the int8 pin's role in checkpoints-int8.json")
    checkpoints: str = option("--checkpoints", default=str(C.CHECKPOINTS))
    hf_home: str | None = option("--hf-home", env="HF_HOME", default=None,
                                 help="HF cache holding the base snapshot; the int8 snapshot goes beside it (default: the pins' local_path)")
    jobs: int = option("--jobs", type=int, default=1, help="shards made at a time (the bytes do not depend on it)")


@dataclass(frozen=True, kw_only=True)
class PinOptions:
    """Make an int8 checkpoint from a base pin and print its new entry (to append to checkpoints-int8.json, append-only)."""

    base_role: str = option("--base-role", help="the base pin's role, or its repo id when the role is prose")
    role: str = option("--role", help="the new int8 pin's role")
    repo: str = option("--repo", help="the new pin's repo id (a local namespace: no hub repo holds these bytes)")
    checkpoints: str = option("--checkpoints", default=str(C.CHECKPOINTS))
    hf_home: str | None = option("--hf-home", env="HF_HOME", default=None)
    by: str = option("--by", default="int8-checkpoint pin", help="resolved_by of the entry")
    at: str = option("--at", help="resolved_at of the entry (UTC, ISO 8601)")
    recipe: str = option("--recipe", default=RECIPE, choices=sorted(RECIPES),
                         help="the int8 recipe (v2 also makes the LM head; v3 reads an FP8-block base)")
    jobs: int = option("--jobs", type=int, default=1, help="shards made at a time (the bytes do not depend on it)")


@dataclass(frozen=True, kw_only=True)
class ReportOptions:
    """Write a v3 pin's per-tensor error against its base's exact FP8 dequant (JSON)."""

    role: str = option("--role", help="the int8 pin's role in checkpoints-int8.json")
    out: str = option("--out", help="the report's path (JSON)")
    checkpoints: str = option("--checkpoints", default=str(C.CHECKPOINTS))
    hf_home: str | None = option("--hf-home", env="HF_HOME", default=None,
                                 help="HF cache holding both snapshots (default: the pins' local_path)")
    jobs: int = option("--jobs", type=int, default=1, help="shards read at a time")


COMMANDS = {"make": MakeOptions, "pin": PinOptions, "report": ReportOptions}


def main(a: MakeOptions | PinOptions | ReportOptions) -> int:
    if isinstance(a, PinOptions):
        return F.run(a, RECIPES[a.recipe], "int8-checkpoint")
    entries = C.checkpoint_entries(a.checkpoints)
    entry = next((c for c in entries if c.get("role") == a.role), None)
    if isinstance(a, ReportOptions):
        if entry is None:
            raise C.UsageError(f"no checkpoints.json entry with role {a.role!r}")
        doc = report(entry, entries, a.hf_home, a.jobs)
        Path(a.out).write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
        print(json.dumps(doc["summary"], indent=1, sort_keys=True))
        return 1 if doc["summary"]["mismatch"] else 0
    return F.run(a, recipe_of(entry) if entry is not None else INT8, "int8-checkpoint")
