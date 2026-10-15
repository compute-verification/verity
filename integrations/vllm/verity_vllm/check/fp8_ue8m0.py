"""The block-FP8 weights vLLM rewrites at load for its DeepGEMM linears, as the weights of record hold them: a function of the
checkpoint's words that `weights_of_record.check` compares with the served parameters like any other field.

vLLM d9105ea80's `DeepGemmFp8BlockScaledMMKernel.process_weights_after_loading` (the fallback `FlashInferFp8DeepGEMMDynamicBlockScaled
Kernel` delegates to) runs `requant_weight_ue8m0_inplace` on each rank's block-FP8 linear when DeepGEMM's e8m0 scales are in use:

    w_dq = f32(w) * s (scale repeated over its G x G block);  amax = max |w_dq| over the block, clamped to 1e-4;
    s' = 2 ^ ceil(log2(amax / 448));  w' = e4m3(w_dq * (1 / s'))  (per_block_cast_to_fp8, use_ue8m0=True)

so the served weight and `weight_scale_inv` are (w', s'), not the checkpoint's words.  The Build names the linears it binds this way
(derive-report `weight_roles`: `fp8_w_requant_ue8m0` / `fp8_ws_requant_ue8m0`).  MLA's `kv_b_proj` is one such linear the Program does
not read: `MLAAttention.process_weights_after_loading` (mla_attention.py:1154-1252) dequantises it through the kernel on an identity
(`get_and_maybe_dequant_weights`; the identity's ue8m0 input quant is exact, and the one non-zero product of each output is exact in
the f32 accumulator, which turns -0 into +0) into bf16 `W_UK_T` [NH][DN][L] and `W_UV` [NH][L][DV], the fields of role `mla.absorb`;
so a layer with those roles has its kv_b_proj requantised too.  Blocks are cut per rank, on the rank's shard, as vLLM does.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Any, Callable

import numpy as np

REQUANT_ROLES = ("fp8_w_requant_ue8m0", "fp8_ws_requant_ue8m0")
ABSORB_ROLE = "mla.absorb"
SCALE = ".weight_scale_inv"


def build_roles(program_dir: str | None) -> dict[str, dict[str, Any]]:
    """The Build's `weight_roles` (field -> {rule, role, ...}) from the component's derive report, or {} without one."""
    for name in ("derive-report.json", "derive-report.json.gz"):
        p = os.path.join(program_dir or "", name)
        if program_dir and os.path.isfile(p):
            with (gzip.open(p, "rt") if name.endswith(".gz") else open(p)) as f:
                doc = json.load(f)
            stack = [doc]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    if isinstance(x.get("weight_roles"), dict):
                        return x["weight_roles"]
                    stack.extend(x.values())
    return {}


def served_words(ck: Any, src: str, bits: int, rows: tuple[int, int] | None = None) -> tuple[np.ndarray, str] | None:
    """A bf16 field whose checkpoint tensor is e4m3 beside a `weight_scale_inv`: vLLM's upcast at load (deepseek_v2
    `_try_load_fp8_indexer_wk`: GLM-5.3's indexer wk, fused bf16 with weights_proj) = bf16_RNE(f32(w) * scale over its block); else None."""
    scale = src + "_scale_inv"
    if bits != 16 or ck.dtype(src) != "F8_E4M3" or not src.endswith(".weight") or not ck.has(scale):
        return None
    from verity_vllm.program.kernels.fp8_twins import np_e4m3_to_f32
    w, s = ck.array(src), ck.array(scale).astype(np.float32)
    full = np.repeat(np.repeat(s, w.shape[0] // s.shape[0], axis=0), w.shape[1] // s.shape[1], axis=1)
    u = np.ascontiguousarray(np_e4m3_to_f32(w) * full, dtype=np.float32).view(np.uint32)
    out = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return (out if rows is None else out[rows[0]:rows[1]]), f"e4m3 x {scale} -> bf16 (RNE; vLLM's FP8 indexer wk upcast)"


def requant(w: np.ndarray, s: np.ndarray, G: int) -> tuple[np.ndarray, np.ndarray]:
    """(e4m3 words [N, K], f32 words [cdiv(N, G), K / G]) -> the ue8m0 requantised pair, as `requant_weight_ue8m0_inplace`."""
    from verity_vllm.program.kernels.fp8_twins import np_e4m3_to_f32, np_f32_to_e4m3_sat
    N, K = w.shape
    NB, KB = -(-N // G), -(-K // G)
    sf = s.view(np.float32).reshape(NB, KB)
    full = np.repeat(np.repeat(sf, G, axis=0), G, axis=1)[:N, :K]
    pad = np.zeros((NB * G, KB * G), dtype=np.float32)
    pad[:N, :K] = np_e4m3_to_f32(w) * full
    amax = np.maximum(np.abs(pad).reshape(NB, G, KB, G).max(axis=(1, 3)), np.float32(1e-4))
    s2 = np.exp2(np.ceil(np.log2(amax / np.float32(448.0)))).astype(np.float32)
    inv = np.repeat(np.repeat(np.float32(1.0) / s2, G, axis=0), G, axis=1)
    return np_f32_to_e4m3_sat(pad * inv)[:N, :K], s2.view(np.uint32)


def absorbed(w: np.ndarray, s: np.ndarray, G: int, NH: int, DN: int, DV: int, which: str) -> np.ndarray:
    """bf16 words of `W_UK_T` [NH][DN][L] or `W_UV` [NH][L][DV] from the rank's requantised kv_b_proj (w', s')."""
    from verity_vllm.program.kernels.fp8_twins import np_e4m3_to_f32
    full = np.repeat(np.repeat(s.view(np.float32), G, axis=0), G, axis=1)[:w.shape[0], :w.shape[1]]
    deq = np.where((w & 0x7F) == 0, np.float32(0.0), np_e4m3_to_f32(w) * full).view(np.uint32)
    if np.any(deq & 0xFFFF):   # an e4m3 word times a power of two is exact in bf16; anything else is not this rule
        raise ValueError("kv_b_proj's dequantised words are not exact in bf16 (the requantised scales are not powers of two)")
    heads = (deq >> 16).astype(np.uint16).reshape(NH, DN + DV, -1)
    return np.ascontiguousarray(heads[:, :DN, :] if which == "W_UK_T" else heads[:, DN:, :].transpose(0, 2, 1))


class Rewrites:
    """The fields the record composes through the load-time rewrite: `raw(field)` is the checkpoint's composition of a Program field
    (shard included), `config` the checkpoint's config.json."""

    def __init__(self, fields: list[str], roles: dict[str, dict[str, Any]], raw: Callable[[str], tuple[np.ndarray, dict]], config: dict):
        self.raw, self.config, self._pair = raw, config, {}
        bs = (config.get("quantization_config") or {}).get("weight_block_size") or [0]
        self.G = int(bs[0])
        names = set(fields)
        self.requant = {f for f in fields if (roles.get(f) or {}).get("role") in REQUANT_ROLES}
        self.absorb = {f: f.rsplit(".", 1)[0] + ".kv_b_proj.weight" for f in fields if (roles.get(f) or {}).get("role") == ABSORB_ROLE}
        if self.requant:
            self.requant |= {k + sfx for k in self.absorb.values() for sfx in ("", SCALE[len(".weight"):]) if k + sfx in names}

    def covers(self, field: str) -> bool:
        return self.G > 0 and (field in self.requant or field in self.absorb)

    def _requanted(self, weight: str) -> tuple[np.ndarray, np.ndarray, dict]:
        if weight not in self._pair:
            (w, nw), (s, ns) = self.raw(weight), self.raw(weight[:-len(".weight")] + SCALE)
            self._pair = {weight: (*requant(w, s, self.G), {**nw, "sources": nw["sources"] + ns["sources"]})}
        return self._pair[weight]

    def compose(self, field: str, shape: list[int]) -> tuple[np.ndarray, dict]:
        rule = f"vLLM load-time ue8m0 requant (DeepGEMM linear, block {self.G})"
        if field in self.absorb:
            if not self.requant:
                raise KeyError(f"{field}: mla.absorb without the DeepGEMM requant path has no composition rule")
            w, s, note = self._requanted(self.absorb[field])
            c = self.config
            arr = absorbed(w, s, self.G, int(shape[0]), int(c["qk_nope_head_dim"]), int(c["v_head_dim"]), field.rsplit(".", 1)[1])
            rule = f"{rule}, then MLA's bf16 {field.rsplit('.', 1)[1]} of kv_b_proj (identity GEMM)"
        else:
            weight = field if field.endswith(".weight") else field[:-len(SCALE)] + ".weight"
            w, s, note = self._requanted(weight)
            arr = w if field == weight else s
        if list(arr.shape) != list(shape):
            raise ValueError(f"{field}: rewritten shape {list(arr.shape)} != Program shape {shape}")
        return arr, {**note, "rule": rule}
