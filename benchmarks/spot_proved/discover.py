"""A GLM-5.3 engine at the integration's pin with dummy weights and a few layers, served with CUDA graphs and run with
`capture.py` on a few steps: a rehearsal of the capture before it runs on served values. It answers which kernels the
served engine runs where the census types sit, and whether the recorded forward equals the graph's bit for bit.

    python discover.py --config DIR --out OUT --capture DIR [--tp 2] [--layers 4] [--steps 0,3] [--prompts 8] [--max-tokens 8] [--prompt-len 24,64]

``--config`` holds GLM-5.3's config and tokenizer, without weights (``load_format=dummy``). The engine takes vLLM's
defaults (for GLM-5 at the pin: breakable CUDA graphs, compilation off, the V2 model runner). ``OUT/summary.json`` gives,
per rank and captured step, the check, the events by op, and each piece's ops in order.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def summarize(cap: Path) -> dict:
    out = {}
    for meta in sorted(cap.glob("rank*/step*/meta.json")):
        m = json.loads(meta.read_text())
        ops = [json.loads(x) for x in (meta.parent / "ops.jsonl").read_text().splitlines()]
        pieces: dict[int, list[str]] = {}
        for e in ops:
            pieces.setdefault(e.get("layer") if e.get("layer") is not None else -1, []).append(e["op"])
        out[f"{meta.parent.parent.name}/{meta.parent.name}"] = {
            "check": m["check"], "events": m["events"], "blob_mib": m["blob_bytes"] >> 20, "seconds": m["seconds"],
            "errors": [x[-600:] for x in m["errors"]], "scheduled": m["scheduled"],
            "ops": dict(Counter(e["op"] for e in ops).most_common()),
            "layers": {str(k): v for k, v in sorted(pieces.items())}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture", required=True, help="the capture directory, as the site's settings name it")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", default="0,3")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--prompt-len", default="24,64", help="the prompts' lengths, uniform in LO,HI (long ones make FA3 split)")
    ap.add_argument("--summarize", action="store_true", help="only summarize --capture")
    a = ap.parse_args()
    out = Path(a.out)
    if a.summarize:
        (out / "summary.json").write_text(json.dumps(summarize(Path(a.capture)), indent=1))
        return 0
    try:
        import spot_capture
    except ImportError:
        raise SystemExit("discover.py: put capture.py's site on PYTHONPATH (run_discover.sh)") from None
    cfg = spot_capture.config() or {}
    if cfg.get("capture") != str(Path(a.capture).resolve()) or cfg.get("steps") != sorted({int(s) for s in a.steps.split(",")}):
        raise SystemExit(f"discover.py: the site's capture settings {cfg} are not --capture {a.capture} --steps {a.steps}")
    import random

    from vllm import LLM, SamplingParams

    llm = LLM(model=a.config, load_format="dummy", tensor_parallel_size=a.tp, trust_remote_code=True, seed=0, dtype="bfloat16",
              hf_overrides={"num_hidden_layers": a.layers}, max_model_len=4096, max_num_seqs=256, gpu_memory_utilization=0.85)
    rng = random.Random(0)
    lo, hi = map(int, a.prompt_len.split(","))
    prompts = [{"prompt_token_ids": [rng.randrange(1000, 100000) for _ in range(rng.randrange(lo, hi))]} for _ in range(a.prompts)]
    got = llm.generate(prompts, SamplingParams(max_tokens=a.max_tokens, temperature=0.0, ignore_eos=True))
    (out / "tokens.json").write_text(json.dumps([list(g.outputs[0].token_ids) for g in got]))
    cfg = llm.llm_engine.vllm_config
    (out / "config.txt").write_text(f"{cfg.compilation_config}\n\nuse_v2_model_runner={getattr(cfg, 'use_v2_model_runner', None)}\n")
    del llm
    (out / "summary.json").write_text(json.dumps(summarize(Path(a.capture)), indent=1))
    print(json.dumps({k: {x: v[x] for x in ("check", "events", "blob_mib", "errors")} for k, v in summarize(Path(a.capture)).items()},
                     indent=1)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
