"""Capture the encoded stack's KV-cache append on sm_120 and set it against ``KvCacheAppendE4m3_v1``.

The encoded stack (`benchmarks/vllm_encoded/stack.py`) serves Llama-3.1-8B from the encoded store and appends
each step's new BF16 key and value rows to vLLM's paged pool as word-codec words (`paged.cu`'s deferred
``pg_append_multi``).  This tool installs the stack exactly as serving does (`serve.py`'s ``Install``, KV receipts
off), wraps the deferred append, and for one prefill and three decode steps captures the first and last layers' input
BF16 rows, their slots and every cache byte the append wrote for them.  It then checks:

- **the stack's own reference**: every written byte equals ``kv.encode_word`` of the exact BF16 payloads at the
  layer's store indices (``store_base + (slot * layers + layer) * TLW + pg_wi``), the CPU form of what the kernel
  states it writes — the encoded stack's append matched bit for bit;
- **the 8-bit Definition**: ``KvCacheAppendE4m3_v1``'s CPU reference (`kv_cache_append_e4m3_word`) applied to the
  same inputs at the engine's key and value scales, element by element against what the stack stored.

The second comparison is the diagnosis, not a gate: the encoded stack has no 8-bit cache mode today (its attention
raises on ``kv_cache_dtype != "auto"``, and its payloads are the exact BF16 words), a deliberate interim choice
(FP8 KV where served is planned; the 8B model serves 16-bit for now).  The report records that the
Definition applies nowhere, localizes the first differing element, and counts the elements the Definition's
quantization would change, so the day the 8-bit mode lands this capture is re-pointed, not rewritten.

Captured on the RTX PRO 6000: one prefill and three decode appends, layers 0 and 31,
172,032/172,032 cache bytes bit-exact against the codec reference; 81,324 of 86,016 elements would change under the
Definition's quantization, first difference localized, both scales FP32 1.0.

Run from the repository root on the node (`stack.sh` MODE=capture builds the libraries and launches it)::

    python benchmarks/vllm_encoded/tests/properties/encoded_kv_append_capture_gpu.py \
      --model /workspace/pouw/mvp-e2e/models/llama-3.1-8b-instruct \
      --wt-lib wt.so --reader-lib reader.so --paged-lib paged.so --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
VE = HERE.parents[1]                         # benchmarks/vllm_encoded
sys.path.insert(0, str(VE.parent / "word_codec"))

import kv as KV            # noqa: E402  the codec's byte-exact CPU reference
import kv_bench as KB      # noqa: E402  the stack's key derivation (SEED, lib_key)

from verity.ml.circuits.ops.kv_cache import kv_cache_append_e4m3_word  # noqa: E402

SCHEMA = "verity/vllm-encoded/kv-append-capture/v1"
WANT_CC = (12, 0)
CAPTURES = 4
F32_ONE = 0x3F800000


# --- the CPU references (pure, tested without a GPU) -------------------------------------------------------------------

def pg_wi(g: int, j: int, hw: int, head_words: int, half_words: int) -> int:
    """`paged.cu`'s logical word of physical word j of head g: K of every head, then V of every head."""
    return (0 if j < hw // 2 else half_words - hw // 2) + g * head_words + j


def payload_words(key_row: np.ndarray, value_row: np.ndarray, f: KV.Field) -> list[int]:
    """One token-layer's payloads from its BF16 rows (uint16, [KV heads * head dim] each): K then V, little-endian."""
    return KV.payloads(key_row.astype("<u2").tobytes() + value_row.astype("<u2").tobytes(), f)


def encode_reference(key_row: np.ndarray, value_row: np.ndarray, slot: int, layer: int, layers: int,
                     store_base: int, key: KV.Key) -> list[int]:
    """The words the append must write for one token at a physical slot, in logical word order."""
    tlw = KV.token_layer_words(key.field)
    w0 = store_base + (slot * layers + layer) * tlw
    return [KV.encode_word(m, w0 + w, key) for w, m in enumerate(payload_words(key_row, value_row, key.field))]


def bf16_value(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (w & 0xFFFF) << 16))[0]


def e4m3_value(b: int) -> float:
    """The value of an E4M3 code (NaN for 0x7f/0xff)."""
    s, e, m = -1.0 if b & 0x80 else 1.0, (b >> 3) & 0xF, b & 7
    if e == 0xF and m == 7:
        return float("nan")
    return s * (m / 8.0 * 2.0 ** (1 - 7) if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))


def definition_report(key_row: np.ndarray, value_row: np.ndarray, k_scale: int, v_scale: int) -> dict:
    """`KvCacheAppendE4m3_v1`'s CPU reference on the same inputs, against the BF16 words the stack stores exactly.

    ``lossy`` counts the elements whose Definition byte, dequantized (its E4M3 value times the scale), differs from
    the stored BF16 word's value: the information the 8-bit cache would drop and the 16-bit cache keeps.
    ``first_difference`` is the first of them.  The byte streams themselves share no format (16-bit words against
    8-bit codes), which is the finding."""
    lossy, first, total = 0, None, 0
    for name, row, scale in (("key", key_row, k_scale), ("value", value_row, v_scale)):
        sv = struct.unpack("<f", struct.pack("<I", scale))[0]
        for at, x in enumerate(row.reshape(-1).tolist()):
            total += 1
            b = kv_cache_append_e4m3_word(int(x), scale)
            bx, ex = bf16_value(int(x)), e4m3_value(b) * sv
            if not (bx == ex or (bx != bx and ex != ex)):
                lossy += 1
                if first is None:
                    first = {"which": name, "element": at, "stored_bf16_word": f"0x{int(x):04x}",
                             "stored_value": bx, "definition_byte": f"0x{b:02x}", "definition_value": ex}
    return {"elements": total, "lossy_under_definition": lossy, "first_difference": first}


# --- the capture -------------------------------------------------------------------------------------------------------

class Capture:
    """Wraps the stack's deferred append (`flush_eager` for an eager step, `flush_graph` for a replayed one) and, once
    armed, snapshots the captured layers' input rows and slots before the launch and the cache words after it."""

    def __init__(self, out: Path, att: Any, want: tuple[int, ...], torch) -> None:
        self.out, self.att, self.want, self.torch = out, att, set(want), torch
        self.armed = False
        self.steps = 0
        self.rows: list[dict] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.key = KB.lib_key(att.lib, att.st.k)
        sh = att.lib.shape
        self.hw, self.wb, self.heads, self.head_dim = sh["hw"], sh["wb"], sh["kv_heads"], sh["head_dim"]
        self.scales = (F32_ONE, F32_ONE)

    def wrap_eager(self, orig):
        cap = self

        def fn(att):
            take = cap.armed and cap.steps < CAPTURES and att.pending
            snap = {}
            if take:
                for key, val, slots, n, cache, _base, _nl, idx in att.pending:
                    if idx in cap.want:
                        snap[idx] = (cap._u16(key[:n]), cap._u16(val[:n]),
                                     slots.detach().cpu().numpy().astype(np.int64, copy=True).reshape(-1), cache)
            r = orig(att)
            if snap:
                cap.torch.cuda.synchronize()
                cap._record("prefill" if cap.steps == 0 else f"eager-{cap.steps}", snap)
            return r
        return fn

    def wrap_graph(self, orig):
        cap = self

        def fn(att, rows):
            take = cap.armed and cap.steps < CAPTURES and att.staged
            snap = {}
            if take:
                cap.torch.cuda.synchronize()   # the replay wrote the stage buffers
                for idx, (cache, _base, _nl, _i) in att.staged.items():
                    if idx in cap.want:
                        snap[idx] = (cap._u16(att.stage_k[idx]), cap._u16(att.stage_v[idx]),
                                     att.stage_s[idx].detach().cpu().numpy().astype(np.int64, copy=True), cache)
            r = orig(att, rows)
            if snap:
                cap.torch.cuda.synchronize()
                cap._record(f"decode-{cap.steps}", snap)
            return r
        return fn

    def _u16(self, t) -> np.ndarray:
        return t.detach().contiguous().view(self.torch.uint16).cpu().numpy().copy()

    def _cache_words(self, cache, slots: np.ndarray) -> np.ndarray:
        """The written words of each slot in logical word order: uint8 [tokens, TLW, word bytes]."""
        torch, att = self.torch, self.att
        lay, we = att.lay, self.wb // cache.element_size()
        flat = torch.as_strided(cache, (lay.blocks * lay.sb * we,), (1,))
        half = KV.token_layer_words(self.key.field) // 2
        head_words = self.head_dim * 2 // self.wb
        out = []
        for u in (int(s) for s in slots):
            words: list[Any] = [None] * KV.token_layer_words(self.key.field)
            for g in range(self.heads):
                for j in range(self.hw):
                    p = (u // lay.bs) * lay.sb + g * lay.sh + (u % lay.bs) * lay.ss + j
                    words[pg_wi(g, j, self.hw, head_words, half)] = flat[p * we:(p + 1) * we]
            out.append(torch.stack(words))
        got = torch.stack(out).cpu()
        return got.view(self.torch.uint8).numpy().reshape(len(slots), KV.token_layer_words(self.key.field), self.wb)

    def _record(self, phase: str, snap: dict) -> None:
        att, s = self.att, self.key.field.word_bytes
        for idx, (keys, vals, slots, cache) in sorted(snap.items()):
            live = np.flatnonzero(slots >= 0)
            k_rows, v_rows, u = keys[live], vals[live], slots[live]
            got = self._cache_words(cache, u)
            want = np.zeros_like(got)
            for t in range(len(u)):
                ref = encode_reference(k_rows[t], v_rows[t], int(u[t]), idx, att.nl, att.st.words, self.key)
                for w, y in enumerate(ref):
                    want[t, w] = np.frombuffer(y.to_bytes(s, "little"), dtype=np.uint8)
            bad = np.argwhere((got != want).any(axis=2))
            first = None
            if len(bad):
                t, w = (int(x) for x in bad[0])
                first = {"token": t, "slot": int(u[t]), "word": w,
                         "store_index": att.st.words + (int(u[t]) * att.nl + idx) * KV.token_layer_words(self.key.field) + w,
                         "got": got[t, w].tobytes().hex(), "want": want[t, w].tobytes().hex()}
            i = len(self.rows)
            self.arrays.update({
                f"capture_{i}_layer_{idx}_key_bf16": k_rows, f"capture_{i}_layer_{idx}_value_bf16": v_rows,
                f"capture_{i}_layer_{idx}_slots": u,
                f"capture_{i}_layer_{idx}_cache_words": got, f"capture_{i}_layer_{idx}_reference_words": want,
            })
            self.rows.append({
                "phase": phase, "layer": idx, "tokens": int(len(u)),
                "bytes_checked": int(got.size), "byte_mismatches": int(np.count_nonzero(got != want)),
                "first_mismatch": first,
                "e4m3_definition": definition_report(k_rows, v_rows, *self.scales),
            })
        self.steps += 1

    def save(self, metadata: dict) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.out / "capture.npz", **self.arrays)
        checked = sum(r["bytes_checked"] for r in self.rows)
        mismatches = sum(r["byte_mismatches"] for r in self.rows)
        lossy = sum(r["e4m3_definition"]["lossy_under_definition"] for r in self.rows)
        elements = sum(r["e4m3_definition"]["elements"] for r in self.rows)
        result = {
            **metadata, "captures": self.rows, "bytes_checked": checked, "mismatches": mismatches,
            "e4m3_definition": {
                "applies": False,
                "reason": "the encoded stack has no 8-bit cache mode: its append stores the exact BF16 words as codec "
                          "payloads (16-bit for now; FP8 KV where served is planned), so the Definition's E4M3 byte "
                          "is written nowhere",
                "elements": elements, "lossy_under_definition": lossy,
            },
            "verdict": "GO" if self.steps == CAPTURES and mismatches == 0 else "NO-GO",
        }
        (self.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wt-lib", required=True)
    ap.add_argument("--reader-lib", required=True)
    ap.add_argument("--paged-lib", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--select-trials", type=int, default=2)
    a = ap.parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    sys.path.insert(0, str(VE))
    sys.path.insert(0, str(VE.parent / "codec_tensor"))
    import torch

    cc = torch.cuda.get_device_capability()
    if cc != WANT_CC:
        raise RuntimeError(f"capture requires sm_120, got compute capability {cc}")

    import serve as VS  # vllm_encoded's
    import vllm
    from verity_vllm.engine import handle
    from verity_vllm.engine.hooks import Hooks
    from vllm import LLM, SamplingParams

    args = argparse.Namespace(arm="stack", k=a.k, wt_lib=a.wt_lib, reader_lib=a.reader_lib, paged_lib=a.paged_lib,
                              select_trials=a.select_trials, kv_receipts=False)
    hooks = Hooks(tag="mem7-encoded-kv-append-capture")
    install = VS.Install(args)
    VS.hook_runner(hooks, install)
    llm = LLM(
        model=str(a.model),
        quantization="fp8",
        max_num_seqs=2,
        max_model_len=64,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        seed=0,
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]},
    )
    st = install.stack
    att = st.attention
    import stack as S   # the module Install imported

    capture = Capture(a.out, att, (0, att.nl - 1), torch)
    runner = handle.model_runner(llm)
    model = VS.unwrap(runner.model)
    for _name, m in model.named_modules():
        ks, vs = getattr(m, "_k_scale", None), getattr(m, "_v_scale", None)
        if ks is not None and vs is not None:
            capture.scales = tuple(struct.unpack("<I", struct.pack("<f", float(t.detach().item())))[0]
                                   for t in (ks, vs))
            break
    hooks.wrap(S.Attention, "flush_eager", capture.wrap_eager)
    hooks.wrap(S.Attention, "flush_graph", capture.wrap_graph)
    capture.armed = True
    prompts = ["The capital of France is", "Write one word for the color of grass"]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True, seed=0),
                           use_tqdm=False)
    capture.armed = False
    hooks.uninstall()

    props = torch.cuda.get_device_properties(0)
    result = capture.save({
        "schema": SCHEMA,
        "model": str(a.model),
        "arm": "stack", "k": st.k, "kv_receipts": False,
        "field": capture.key.field.name, "masked": capture.key.mask is not None,
        "store_base": st.words, "model_layers": att.nl, "layers_captured": [0, att.nl - 1],
        "scales": [f"0x{w:08x}" for w in capture.scales],
        "append": "BF16 key and value rows stored exactly as the payloads of word-codec words "
                  "(K of every head then V of every head), each word encoded at its store index; no quantization",
        "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": props.name,
        "compute_capability": list(cc),
        "outputs": [list(o.outputs[0].token_ids) for o in outputs],
    })
    print(json.dumps(result, indent=1))
    return 0 if result["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
