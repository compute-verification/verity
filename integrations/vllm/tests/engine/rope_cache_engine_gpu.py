"""On one GPU: an int8 Llama engine on vLLM's FlashAttention (`TargetProfile.int8_exact` "linears", the int8 pin of `--role`),
whose build installs the fused rope and KV-cache write (`engine.rope_cache`), against the same engine with it uninstalled.

One process writes `<out>/rope_cache_engine_gpu.json`:

  install   what `build_engine` installed (`eng.rope_cache`) and each decoder layer's attention impl.
  kernels   the rope and cache kernels a short generation launches in each arm (the profiler's names): `rope_cache_kernel`
            alone fused, vLLM's two unfused.
  equal     greedy generations in three passes, fused, unfused, fused again: PROMPTS batched, then a `--long`-token prompt
            alone.  Each pass's token ids, their logprobs and, by SHA-256, every layer's attention output (o_proj's input) at
            every call are the first pass's.
  timed     at each of `--ts`, a prefill of T tokens as one step (max_tokens 1), the arms interleaved, `--reps` each: ms a
            request between CUDA events on either side of `generate`.

    cd integrations/vllm; PY=/workspace/jobs/venv312/bin/python3     # PYTHONPATH=.:../..:../../tools/research/src
    $PY tests/engine/rope_cache_engine_gpu.py --out DIR --at ISO_TIME --by LANE [--role R --base-role B --repo REPO] [--ts 8192,32768]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from tests.engine import int8_exact_gpu as X

RECORD = "rope_cache_engine_gpu.json"


def layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "self_attn") and hasattr(m.self_attn, "o_proj")]


class Digests:
    """SHA-256 of each layer's attention output, call by call, while `on`."""

    def __init__(self, model: torch.nn.Module):
        self.on = False
        self.got: list[tuple[str, str]] = []
        self.handles = [m.self_attn.o_proj.register_forward_hook(self._hook(n)) for n, m in layers(model)]

    def _hook(self, name: str):
        def hook(mod, args, out):
            if self.on:
                x = args[0].contiguous()
                self.got.append((name, hashlib.sha256(x.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()))
        return hook

    def take(self) -> list[tuple[str, str]]:
        got, self.got = self.got, []
        return got


def generate(llm: Any, prompts: list[Any], max_tokens: int) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0, logprobs=1), use_tqdm=False)
    return [{"token_ids": list(o.outputs[0].token_ids),
             "logprobs": [float(lp[t].logprob) for t, lp in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)]} for o in outs]


def kernels(llm: Any, prompts: list[Any]) -> list[str]:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        generate(llm, prompts, 2)
        torch.cuda.synchronize()
    return sorted({e.name.split("<")[0].split("(")[0].replace("void ", "") for e in prof.events()
                   if e.device_type == torch.autograd.DeviceType.CUDA
                   and ("rope" in e.name or "rotary" in e.name or "reshape_and_cache" in e.name)})


def run(a: argparse.Namespace) -> dict[str, Any]:
    from vllm import SamplingParams

    from verity_vllm.engine import rope_cache as R
    from verity_vllm.engine.build import build_engine
    from verity_vllm.program.frontend.target_profile import TargetProfile

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    target = TargetProfile.from_device().with_overrides(int8_exact="linears")
    doc: dict[str, Any] = {"argv": sys.argv, "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = X.pin(a)
        doc["pin"] = {"repo": entry["repo"], "revision": entry["revision"], "source": how}
        print(f"[rope] pin {json.dumps(doc['pin'])}", flush=True)
        checkpoints = X.scratch_checkpoints(a.checkpoints, entry, Path(scratch))
        ts = [int(t) for t in a.ts.split(",")]
        eng = build_engine(checkpoints, case=a.role, target=target, max_num_seqs=len(X.PROMPTS),
                           engine_args={"max_num_batched_tokens": max(ts), "max_model_len": max(ts) + 256})
    model = eng.model
    doc["install"] = {"rope_cache": eng.rope_cache, "layers": len(layers(model)),
                      "impls": sorted({type(m.self_attn.attn.impl).__name__ for _, m in layers(model)}),
                      "enforce_eager": eng.kwargs.get("enforce_eager")}
    print(f"[rope] install {json.dumps(doc['install'])}", flush=True)
    installed = eng.rope_cache is not None and eng.rope_cache["layers"] == doc["install"]["layers"]

    g = torch.Generator().manual_seed(0)
    long_ids = torch.randint(100, 30000, (a.long,), generator=g).tolist()
    arms = {"fused": lambda: R.install(model), "unfused": R.uninstall}
    doc["kernels"] = {}
    for arm in ("fused", "unfused"):
        R.uninstall()
        arms[arm]()
        doc["kernels"][arm] = kernels(eng.llm, X.PROMPTS)
    print(f"[rope] kernels {json.dumps(doc['kernels'])}", flush=True)
    dig = Digests(model)
    passes = []
    for arm in ("fused", "unfused", "fused"):
        R.uninstall()
        arms[arm]()
        dig.on = True
        batched = generate(eng.llm, X.PROMPTS, a.max_tokens)
        b_dig = dig.take()
        long = generate(eng.llm, [{"prompt_token_ids": long_ids}], a.max_tokens)
        l_dig = dig.take()
        dig.on = False
        passes.append({"arm": arm, "batched": batched, "long": long, "batched_digests": b_dig, "long_digests": l_dig})
    ref = passes[0]
    doc["equal"] = [{"arm": p["arm"], "calls": len(p["batched_digests"]) + len(p["long_digests"]),
                     "tokens": [g["token_ids"] for g in p["batched"] + p["long"]] == [g["token_ids"] for g in ref["batched"] + ref["long"]],
                     "logprobs": [g["logprobs"] for g in p["batched"] + p["long"]] == [g["logprobs"] for g in ref["batched"] + ref["long"]],
                     "attention": p["batched_digests"] == ref["batched_digests"] and p["long_digests"] == ref["long_digests"]}
                    for p in passes]
    doc["generations"] = [{"arm": p["arm"], "batched": p["batched"], "long": p["long"]} for p in passes]
    for e in doc["equal"]:
        print(f"[rope] equal {json.dumps(e)}", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=1, seed=0)
    doc["timed"] = []
    for t in ts:
        ids = torch.randint(100, 30000, (t,), generator=g).tolist()
        ms: dict[str, list[float]] = {"fused": [], "unfused": []}
        for r in range(a.warmup + a.reps):
            for arm in (("fused", "unfused") if r % 2 else ("unfused", "fused")):
                R.uninstall()
                arms[arm]()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                eng.llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
                end.record()
                torch.cuda.synchronize()
                if r >= a.warmup:
                    ms[arm].append(start.elapsed_time(end))
        row = {"t": t, "reps": a.reps, "ms": ms, "median_ms": {k: statistics.median(v) for k, v in ms.items()}}
        row["saved_ms"] = row["median_ms"]["unfused"] - row["median_ms"]["fused"]
        doc["timed"].append(row)
        print(f"[rope] timed {json.dumps({k: v for k, v in row.items() if k != 'ms'})}", flush=True)
    R.uninstall()
    arms["fused"]()

    doc["ok"] = bool(installed and doc["install"]["impls"] == ["FlashAttentionImpl"]
                     and doc["kernels"]["fused"] == ["rope_cache_kernel"]
                     and doc["kernels"]["unfused"] == ["vllm::reshape_and_cache_flash_kernel", "vllm::rotary_embedding_kernel"]
                     and all(e["tokens"] and e["logprobs"] and e["attention"] and e["calls"] > 0 for e in doc["equal"]))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    saved = " ".join(f"T{r['t']}={r['saved_ms']:.3f}" for r in doc["timed"])
    print(f"ROPE-CACHE-ENGINE-GPU {'OK' if doc['ok'] else 'FAIL'} layers={doc['install']['layers']} saved_ms {saved}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the fused rope and KV-cache write in an int8 Llama engine, against it uninstalled")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA32_1B")
    ap.add_argument("--role", default="LLAMA32_1B_INT8")
    ap.add_argument("--repo", default="verity-int8/Llama-3.2-1B-INT8")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--at", required=True, help="resolved_at of the pin entry (UTC, ISO 8601)")
    ap.add_argument("--by", required=True, help="resolved_by of the pin entry")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--long", type=int, default=3000, help="tokens of the long prompt")
    ap.add_argument("--ts", default="8192", help="tokens of each timed prefill, comma-separated")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    return 0 if run(ap.parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
