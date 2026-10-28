"""Calibrate the int8 attention's K scales (`engine.int8_k_scales`, the provisional recipe) on one GPU.

The int8 pin of `--role` (as `int8_exact_gpu.py` finds or makes it) is served with its int8 linears and vLLM's own attention
(`TargetProfile.int8_exact = "linears"`); `Calibrator` (wrapping each attention impl's forward) keeps max |k| per (layer, KV
head) over every token of the perplexity bench's calibration windows (`benchmarks/ppl/methodology.CALIBRATION`: C4 windows 100..115 at 2048 tokens, checked
against `windows.lock.json`), each a one-token greedy request.  Writes `<out>/k_scales_entry.json` (the table entry) and
`<out>/int8-k-scales.json` (the registered table with that entry in place: a job copies it over its clone's
`manifests/int8-k-scales.json` before `int8_exact_gpu.py`).

    cd integrations/vllm; PY=/workspace/jobs/venv312/bin/python3     # PYTHONPATH=.:../..:../../tools/research/src
    $PY tests/engine/int8_kscale_gpu.py --out DIR --data DIR_WITH_C4 --at ISO_TIME --by LANE
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tests.engine import ppl_oracle  # noqa: F401  (puts benchmarks/ppl on sys.path)
from tests.engine.int8_exact_gpu import pin, scratch_checkpoints
from verity_vllm.engine import int8_k_scales as K

import methodology
import windows as W


class Calibrator:
    """amax [L, G] of K over every token an attention layer is handed, kept by wrapping its impl's forward."""

    def __init__(self, model: Any):
        self.mods = {n: m for n, m in model.named_modules() if hasattr(m, "impl") and hasattr(m.impl, "num_kv_heads")}
        if not self.mods:
            raise ValueError("no attention layers to calibrate")
        g = {m.impl.num_kv_heads for m in self.mods.values()}
        if len(g) != 1:
            raise ValueError(f"attention layers with different KV head counts {sorted(g)}")
        self.amax = torch.zeros(1 + max(K.layer_index(n) for n in self.mods), g.pop(), dtype=torch.float32)
        self.tokens = 0
        self.saved = {n: m.impl.forward for n, m in self.mods.items()}
        for n, m in self.mods.items():
            m.impl.forward = self._wrap(K.layer_index(n), self.saved[n])

    def _wrap(self, layer: int, forward):
        def wrapped(lay, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
            if attn_metadata is not None:
                n = attn_metadata.num_actual_tokens
                k = key[:n].float().abs().amax(dim=(0, 2)).cpu()
                self.amax[layer] = self.amax[layer].maximum(k)
                if layer == 0:
                    self.tokens += n
            return forward(lay, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
        return wrapped

    def close(self) -> np.ndarray:
        for n, m in self.mods.items():
            m.impl.forward = self.saved[n]
        return self.amax.numpy()


def run(a: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from verity_vllm.engine.build import build_engine, target_mismatches
    from verity_vllm.program.frontend.target_profile import TargetProfile

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    entry, how = pin(a)
    tok = AutoTokenizer.from_pretrained(entry["local_path"])
    wins, rec = W.build(methodology.CALIBRATION["set"], methodology.CALIBRATION["ctx"], *W.encoder(tok), Path(a.data), calibration=True)
    W.check(a.tokenizer_family, methodology.CALIBRATION["set"], methodology.CALIBRATION["ctx"], wins, calibration=True)
    print(f"[kscale] pin {entry['revision']} ({how}); windows {rec}", flush=True)
    target = TargetProfile.from_device().with_overrides(int8_exact="linears")
    with tempfile.TemporaryDirectory() as scratch:
        eng = build_engine(scratch_checkpoints(a.checkpoints, entry, Path(scratch)), case=a.role, target=target, max_num_seqs=len(wins),
                           engine_args=json.loads(a.engine_args))
    assert not target_mismatches(eng), target_mismatches(eng)
    cal = Calibrator(eng.model)
    eng.llm.generate([TokensPrompt(prompt_token_ids=w.tolist()) for w in wins], SamplingParams(temperature=0.0, max_tokens=1),
                     use_tqdm=False)
    amax = cal.close()
    calibration = {"set": methodology.CALIBRATION["set"], "ctx": methodology.CALIBRATION["ctx"],
                   "windows": list(methodology.CALIBRATION["windows"]), "windows_sha256": rec["sha256"], "tokens": cal.tokens,
                   "served": "int8 linears (verity_int8) and vLLM's FlashAttention, bf16 KV cache", "at": a.at, "by": a.by,
                   "run": os.environ.get("RESEARCH_RUN_ID")}
    new = K.entry(entry["revision"], entry["repo"], amax, calibration)
    (out / "k_scales_entry.json").write_text(json.dumps(new, indent=1) + "\n")
    doc = json.loads(K.MANIFEST.read_text()) if K.MANIFEST.is_file() else None
    (out / "int8-k-scales.json").write_text(json.dumps(K.with_entry(doc, new), indent=1) + "\n")
    s = K.table(entry["revision"], out / "int8-k-scales.json")
    print(f"[kscale] tokens {cal.tokens} (want {len(wins) * methodology.CALIBRATION['ctx']}); amax min {amax.min():.4g} max {amax.max():.4g}; "
          f"s_k min {s.min():.4g} max {s.max():.4g}", flush=True)
    ok = cal.tokens >= wins.size and bool((amax > 0).all())
    print(f"INT8-KSCALE {'OK' if ok else 'FAIL'}", flush=True)
    return {"ok": ok}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="calibrate the int8 attention's K scales (one per layer and KV head) on this GPU")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True, help="directory holding the bench's C4 file (benchmarks/ppl/datasets.json)")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA31_8B")
    ap.add_argument("--role", default="LLAMA31_8B_INT8")
    ap.add_argument("--repo", default="verity-int8/Llama-3.1-8B-INT8")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--tokenizer-family", default="llama3", choices=sorted(W.lock()),
                    help="the windows.lock.json family the calibration windows are checked against (qwen2.5 for Qwen2.5 and Qwen3)")
    ap.add_argument("--engine-args", default="{}", help='JSON engine arguments over the defaults, e.g. \'{"gpu_memory_utilization": 0.95}\' '
                    "for a model whose int8 weights fill most of the GPU (Llama-3.1-70B)")
    ap.add_argument("--at", required=True)
    ap.add_argument("--by", required=True)
    return 0 if run(ap.parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
