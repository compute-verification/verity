"""Capture vLLM's Llama-3.1-8B BF16 -> E4M3 KV-cache append on sm_120.

The pinned vLLM d9105ea80 engine chooses its attention backend normally.  This
tool wraps each candidate backend's ``do_kv_cache_update`` after engine startup,
captures the first attention layer's input K/V words and the cache bytes written
for one prefill and three decode steps, and checks every byte against
``KvCacheAppendE4m3_v1``'s CPU reference.

Run from the repository root with the node-2 vLLM environment::

    python integrations/vllm/tests/properties/fp8_kv_cache_capture_gpu.py \
      --model /workspace/pouw/mvp-e2e/models/llama-3.1-8b-instruct --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verity.ml.circuits.ops.kv_cache import kv_cache_append_e4m3_word
from verity_vllm.engine.hooks import Hooks

SCHEMA = "verity-vllm/fp8-kv-cache-capture/v1"
WANT_CC = (12, 0)
CAPTURES = 4


def _u16(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().view(torch.uint16).cpu().numpy().copy()


def _u8(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().view(torch.uint8).cpu().numpy().copy()


def _u32_scalar(t: torch.Tensor) -> int:
    return struct.unpack("<I", struct.pack("<f", float(t.detach().item())))[0]


def _scale_sources(model: Path) -> list[str]:
    """Checkpoint tensor names that can supply per-tensor K/V cache scales."""
    index = model / "model.safetensors.index.json"
    if not index.is_file():
        return []
    names = json.loads(index.read_text()).get("weight_map", {})
    suffixes = (".k_scale", ".v_scale", ".kv_scale", ".k_proj.output_scale", ".v_proj.output_scale")
    return sorted(n for n in names if n.endswith(suffixes))


def _reference(words: np.ndarray, scale: int) -> np.ndarray:
    flat = words.reshape(-1)
    return np.fromiter((kv_cache_append_e4m3_word(int(x), scale) for x in flat), dtype=np.uint8, count=flat.size).reshape(words.shape)


class Capture:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.hooks = Hooks(tag="fp8-kv-cache-capture")
        self.layer: Any = None
        self.backend: str | None = None
        self.rows: list[dict[str, Any]] = []
        self.arrays: dict[str, np.ndarray] = {}

    def _wrapper(self, backend: str):
        def make(original):
            def wrapped(impl, layer, key, value, kv_cache, slot_mapping, *args, **kwargs):
                take = len(self.rows) < CAPTURES and (self.layer is None or layer is self.layer)
                before = None
                if take:
                    if self.layer is None:
                        self.layer, self.backend = layer, backend
                    slots_all = slot_mapping.detach().cpu().numpy().astype(np.int64, copy=True).reshape(-1)
                    row_ids = np.flatnonzero(slots_all >= 0)
                    before = {
                        "slots": slots_all[row_ids],
                        "rows": row_ids,
                        "key": _u16(key[row_ids]),
                        "value": _u16(value[row_ids]),
                        "k_scale": _u32_scalar(layer._k_scale),
                        "v_scale": _u32_scalar(layer._v_scale),
                    }
                result = original(impl, layer, key, value, kv_cache, slot_mapping, *args, **kwargs)
                if before is not None:
                    torch.cuda.synchronize()
                    head_size = int(impl.head_size)
                    k_cache, v_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
                    block_size = int(k_cache.shape[1])
                    slots = before["slots"]
                    kout = torch.stack([k_cache[int(s) // block_size, int(s) % block_size] for s in slots])
                    vout = torch.stack([v_cache[int(s) // block_size, int(s) % block_size] for s in slots])
                    self._record(before, _u8(kout), _u8(vout), tuple(kv_cache.shape), block_size)
                return result

            return wrapped

        return make

    def install(self) -> None:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        from vllm.v1.attention.backends.flashinfer import FlashInferImpl
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

        for cls in (FlashInferImpl, FlashAttentionImpl, TritonAttentionImpl):
            self.hooks.wrap(cls, "do_kv_cache_update", self._wrapper(f"{cls.__module__}.{cls.__name__}"))

    def _record(self, before: dict[str, Any], kout: np.ndarray, vout: np.ndarray, cache_shape: tuple[int, ...], block_size: int) -> None:
        i = len(self.rows)
        kref, vref = _reference(before["key"], before["k_scale"]), _reference(before["value"], before["v_scale"])
        km, vm = int(np.count_nonzero(kout != kref)), int(np.count_nonzero(vout != vref))
        prefix = f"capture_{i}"
        self.arrays.update({
            f"{prefix}_rows": before["rows"],
            f"{prefix}_slots": before["slots"],
            f"{prefix}_key_bf16": before["key"],
            f"{prefix}_value_bf16": before["value"],
            f"{prefix}_key_cache_e4m3": kout,
            f"{prefix}_value_cache_e4m3": vout,
            f"{prefix}_key_reference_e4m3": kref,
            f"{prefix}_value_reference_e4m3": vref,
        })
        first = None
        for name, got, want in (("key", kout, kref), ("value", vout, vref)):
            bad = np.argwhere(got != want)
            if len(bad):
                at = tuple(int(x) for x in bad[0])
                first = {"which": name, "index": at, "got": int(got[at]), "want": int(want[at])}
                break
        self.rows.append({
            "phase": "prefill" if i == 0 else f"decode-{i}",
            "tokens": int(len(before["slots"])),
            "elements_per_tensor": int(before["key"].size),
            "key_scale_word": f"0x{before['k_scale']:08x}",
            "v_scale_word": f"0x{before['v_scale']:08x}",
            "cache_shape": list(cache_shape),
            "block_size": block_size,
            "key_words_checked": int(kout.size),
            "value_words_checked": int(vout.size),
            "key_mismatches": km,
            "value_mismatches": vm,
            "first_mismatch": first,
        })

    def save(self, metadata: dict[str, Any]) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.out / "capture.npz", **self.arrays)
        words = sum(r["key_words_checked"] + r["value_words_checked"] for r in self.rows)
        mismatches = sum(r["key_mismatches"] + r["value_mismatches"] for r in self.rows)
        result = {**metadata, "backend": self.backend, "captures": self.rows, "words_checked": words,
                  "mismatches": mismatches, "verdict": "GO" if len(self.rows) == CAPTURES and mismatches == 0 else "NO-GO"}
        (self.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    cc = torch.cuda.get_device_capability()
    if cc != WANT_CC:
        raise RuntimeError(f"capture requires sm_120, got compute capability {cc}")

    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(a.model),
        quantization="fp8",
        kv_cache_dtype="fp8",
        max_num_seqs=2,
        max_model_len=64,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=0,
    )
    capture = Capture(a.out)
    capture.install()
    prompts = ["The capital of France is", "Write one word for the color of grass"]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True, seed=0), use_tqdm=False)
    capture.hooks.uninstall()

    props = torch.cuda.get_device_properties(0)
    result = capture.save({
        "schema": SCHEMA,
        "model": str(a.model),
        "checkpoint_scale_tensors": _scale_sources(a.model),
        "scale_rule": "checkpoint per-tensor k_scale/v_scale when present; otherwise both default to FP32 1.0",
        "quantization": "BF16 widened exactly, divided by the FP32 dequantization scale with div.rn.f32, then RN-even "
                        "satfinite conversion to E4M3; NaN -> 0x7f, infinity/overflow -> signed 0x7e",
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "compute_capability": list(cc),
        "outputs": [list(o.outputs[0].token_ids) for o in outputs],
    })
    print(json.dumps(result, indent=1))
    return 0 if result["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
