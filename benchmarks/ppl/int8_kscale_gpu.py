"""Calibrate the int8 attention's K scales (`engine.int8_k_scales`, the provisional recipe) on one GPU.

The int8 pin of `--role` (as `int8_exact_gpu.py` finds or makes it) is served with its int8 linears and vLLM's own attention
(`TargetProfile.int8_exact = "linears"`); `Calibrator` (wrapping each attention impl's forward) keeps max |k| per (layer, KV
head) over every token of the perplexity bench's calibration windows (`benchmarks/ppl/methodology.CALIBRATION`: C4 windows 100..115 at 2048 tokens, checked
against `windows.lock.json`), each a one-token greedy request.  Writes `<out>/k_scales_entry.json` (the table entry) and
`<out>/int8-k-scales.json` (the registered table with that entry in place: a job copies it over its clone's
`manifests/int8-k-scales.json` before `int8_exact_gpu.py`).

    ROOT=$(git rev-parse --show-toplevel); cd "$ROOT/adapters/vllm"; PY=/workspace/jobs/venv312/bin/python3
    PYTHONPATH="$ROOT/adapters/vllm:$ROOT:$ROOT/tools/research/src" \
      $PY "$ROOT/benchmarks/ppl/int8_kscale_gpu.py" --out DIR --data DIR_WITH_C4 --at ISO_TIME --by LANE
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from tests.engine.int8_exact_gpu import pin, scratch_checkpoints
from tests.engine.k_scale_calibrator import Calibrator
from verity_vllm.engine import int8_k_scales as K

PPL = Path(__file__).resolve().parent
if str(PPL) not in sys.path:
    sys.path.insert(0, str(PPL))

import methodology  # noqa: E402
import settings  # noqa: E402
import windows as W  # noqa: E402


def run(a: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    from verity_vllm.engine.build import build_engine, target_mismatches
    from verity_vllm.engine.stock import sampling
    from verity_vllm import TargetProfile

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
    eng.llm.generate([{"prompt_token_ids": w.tolist()} for w in wins], sampling(temperature=0.0, max_tokens=1), use_tqdm=False)
    amax = cal.close()
    calibration = {"set": methodology.CALIBRATION["set"], "ctx": methodology.CALIBRATION["ctx"],
                   "windows": list(methodology.CALIBRATION["windows"]), "windows_sha256": rec["sha256"], "tokens": cal.tokens,
                   "served": "int8 linears (verity_int8) and vLLM's FlashAttention, bf16 KV cache", "at": a.at, "by": a.by,
                   "run": settings.given("RESEARCH_RUN_ID")}
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
    ap.add_argument("--hf-home", default=settings.given("HF_HOME"))
    ap.add_argument("--tokenizer-family", default="llama3", choices=sorted(W.lock()),
                    help="the windows.lock.json family the calibration windows are checked against (qwen2.5 for Qwen2.5 and Qwen3)")
    ap.add_argument("--engine-args", default="{}", help='JSON engine arguments over the defaults, e.g. \'{"gpu_memory_utilization": 0.95}\' '
                    "for a model whose int8 weights fill most of the GPU (Llama-3.1-70B)")
    ap.add_argument("--at", required=True)
    ap.add_argument("--by", required=True)
    return 0 if run(ap.parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
