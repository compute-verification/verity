"""On one GPU: decode TPOT of the int8 Llama-3.1-8B pin served with `TargetProfile.int8_exact` "all" (eager, the
llama31-8b-1gpu-pouw-h1 spec's stock-int8 engine: max_num_batched_tokens 64), at batch ``--batch`` (8), with its attention
``--attention batched`` (`engine.int8_ops.paged_attention`, one kv_quant_batch and one window_rows_batch a layer) or
``reference`` (`int8_attention.reference_paged`: the same values request by request and KV head by KV head, the loop it
replaced); ``linears`` serves `int8_exact` "linears" (vLLM's own bf16 attention, other tokens), the floor the attention's share
of TPOT sits on.

TPOT is (t(N2) - t(N1)) / (N2 - N1) over two greedy generate calls of N1 and N2 output tokens (ignore_eos), so prefill and
the per-call overhead cancel; one warm-up call, then ``--reps`` pairs.  One process serves one arm and writes
``<out>/int8_tpot_<arm>.json`` (its TPOTs and the N2 call's token ids); ``--compare A B`` writes ``<out>/int8_tpot_compare.json``:
whether every request's token ids are equal, and the two median TPOTs.

    cd integrations/vllm; . ../../benchmarks/harness/env.sh
    $VENV/bin/python -m tests.engine.int8_attention_tpot_gpu --attention batched --out DIR
    $VENV/bin/python -m tests.engine.int8_attention_tpot_gpu --compare DIR/int8_tpot_reference.json DIR/int8_tpot_batched.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA = "verity-vllm/int8-attention-tpot-gpu/v1"
TEXTS = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
         "The three primary colors are", "Water boils at a temperature of", "Once upon a time, in a small village,",
         "The largest planet in the solar system is", "To make a cup of tea, first", "In 1969, the first humans to walk on the Moon"]


def use_reference() -> None:
    from verity_vllm.engine import int8_attention as A
    from verity_vllm.engine import int8_ops

    def reference(query, key_cache, value_cache, query_start_loc, seq_lens, block_table, k_scales, scale, output, host=None):
        A.reference_paged(query, key_cache, value_cache, query_start_loc, seq_lens, block_table, k_scales, scale, output)

    int8_ops.paged_attention = reference


def generate(llm: Any, prompts: list[str], n: int) -> tuple[float, list[list[int]]]:
    from vllm import SamplingParams
    torch = __import__("torch")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=n, min_tokens=n, ignore_eos=True, seed=0), use_tqdm=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, [list(o.outputs[0].token_ids) for o in outs]


def serve(a: argparse.Namespace) -> dict[str, Any]:
    import torch

    from tests.engine.int8_exact_gpu import pin, scratch_checkpoints
    from verity_vllm.engine.build import build_engine, target_mismatches
    from verity_vllm.program.frontend.kernel_readback import int8_exact_of
    from verity_vllm.program.frontend.target_profile import TargetProfile

    if a.attention == "reference":
        use_reference()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exact = "linears" if a.attention == "linears" else "all"
    target = TargetProfile.from_device().with_overrides(int8_exact=exact)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                           "attention": a.label or a.attention, "batch": a.batch, "n1": a.n1, "n2": a.n2, "mnbt": a.mnbt}
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = pin(a)
        doc["pin"] = {"repo": entry["repo"], "revision": entry["revision"], "source": how}
        eng = build_engine(scratch_checkpoints(a.checkpoints, entry, Path(scratch)), case=a.role, target=target, max_num_seqs=a.batch,
                           engine_args={"max_num_batched_tokens": a.mnbt, "enable_chunked_prefill": True, "enable_prefix_caching": False})
    doc["engine"] = {"target_mismatches": target_mismatches(eng), "int8_exact": int8_exact_of(eng.model),
                     "enforce_eager": eng.kwargs.get("enforce_eager"), "attention_backend": eng.kwargs.get("attention_backend")}
    print(f"[tpot] engine {json.dumps(doc['engine'], default=str)}", flush=True)
    prompts = [TEXTS[i % len(TEXTS)] for i in range(a.batch)]
    generate(eng.llm, prompts, a.n1)
    tpots, ids = [], None
    for _ in range(a.reps):
        t1, _ids1 = generate(eng.llm, prompts, a.n1)
        t2, ids = generate(eng.llm, prompts, a.n2)
        tpots.append(1e3 * (t2 - t1) / (a.n2 - a.n1))
        print(f"[tpot] {doc['attention']} t({a.n1})={t1:.3f}s t({a.n2})={t2:.3f}s tpot={tpots[-1]:.2f} ms", flush=True)
    doc.update(tpot_ms=tpots, tpot_median_ms=statistics.median(tpots), token_ids=ids)
    doc["ok"] = bool(not doc["engine"]["target_mismatches"] and doc["engine"]["int8_exact"] == exact and ids and all(ids))
    (out / f"int8_tpot_{doc['attention']}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"INT8-TPOT attention={doc['attention']} b={a.batch} tpot_median_ms={doc['tpot_median_ms']:.2f} ok={doc['ok']}", flush=True)
    return doc


def compare(a: argparse.Namespace) -> dict[str, Any]:
    x, y = (json.loads(Path(p).read_text()) for p in a.compare)
    equal = [i == j for i, j in zip(x["token_ids"], y["token_ids"])]
    doc = {"schema": SCHEMA, "arms": [x["attention"], y["attention"]], "batch": x["batch"], "tpot_median_ms": [x["tpot_median_ms"], y["tpot_median_ms"]],
           "speedup": x["tpot_median_ms"] / y["tpot_median_ms"], "requests_equal": equal,
           "equal": len(equal) == len(x["token_ids"]) == len(y["token_ids"]) and all(equal)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "int8_tpot_compare.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"INT8-TPOT-COMPARE {doc['arms'][0]}={doc['tpot_median_ms'][0]:.2f} ms {doc['arms'][1]}={doc['tpot_median_ms'][1]:.2f} ms "
          f"speedup={doc['speedup']:.1f}x token_ids_equal={doc['equal']}", flush=True)
    return {**doc, "ok": doc["equal"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="int8_exact decode TPOT, batched attention against the per-request reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--attention", choices=["batched", "reference", "linears"], default="batched")
    ap.add_argument("--compare", nargs=2, metavar=("REFERENCE", "BATCHED"), help="judge two records instead of serving")
    ap.add_argument("--label", help="the record's arm name (default --attention), e.g. main for a run in main's tree")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n1", type=int, default=16)
    ap.add_argument("--n2", type=int, default=48)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mnbt", type=int, default=64, help="vLLM's max_num_batched_tokens")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA31_8B")
    ap.add_argument("--role", default="LLAMA31_8B_INT8")
    ap.add_argument("--repo", default="verity-int8/Llama-3.1-8B-INT8")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--at", default="2026-10-17T00:00:00Z", help="resolved_at of a new pin entry (UTC, ISO 8601)")
    ap.add_argument("--by", default="vllm-int8-attn", help="resolved_by of a new pin entry")
    a = ap.parse_args(argv)
    doc = compare(a) if a.compare else serve(a)
    return 0 if doc.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
