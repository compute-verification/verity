"""GPU run of the vocabulary-range exactness property (`verity_vllm.properties.vocab_range_exactness`) on a TP2 engine: in every rank's
worker, the vocabulary-range source's committed values against the circuit's `EmbeddingShard_v1` values and the fused kernel's own output, on
edge ids and live; write the sealed record `<out>/vocab_range_exactness.json`.  Exit status 1 unless ok; `--verify RECORD` re-derives a
record's digest and verdict without a GPU.

It lives under tests/ because it seeds random generators and sets the engine's environment for its own process (lints P3, P7).  In the
pod venv on a two-GPU pod, with the tree on PYTHONPATH as for gate (b):

    $PY tests/properties/vocab_range_exactness_gpu.py [--case tiny | ROLE] --out DIR

`--case tiny` (the default) serves a tiny random-weight Llama-architecture model with the B0 checkpoint's tokenizer (4 heads, so it shards
over two ranks; B0's vocabulary splits into unpadded halves): vLLM's own `VocabParallelEmbedding`, no checkpoint download.  A checkpoint
role serves that checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from verity_vllm.properties import vocab_range_exactness as X
from verity_vllm.query import vocab_range as VR

PROMPTS = ("The capital of France is", "def fibonacci(n):\n    ", "Tensor parallelism splits the vocabulary so that", "Zebra quartz vixen")


def _w_install(worker) -> dict[str, Any]:
    """Rank side: find the embedding site, check edge ids through vLLM's own forward with the source in a committed step, and leave the
    source installed with a recorder that keeps every launch's ids and partial (for the live check)."""
    import random
    import types

    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    from verity_vllm.acquire.sources.vocab_range_source import VocabRangeSource
    from verity_vllm.engine import hooks

    class Recorder:
        def __init__(self) -> None:
            self._cur: Any = None
            self.errors: list[str] = []
            self.acquired: list[tuple[str, Any]] = []

        def _acquire(self, name: str, t: Any, job: Any) -> None:
            self.acquired.append((name, t.detach().clone()))

    rank = int(get_tensor_model_parallel_rank())
    model = worker.model_runner.get_model()
    embeds = {n: m for n, m in model.named_modules() if type(m) is VocabParallelEmbedding}
    (name, emb), = embeds.items()
    si = emb.shard_indices
    start, vs = int(si.org_vocab_start_index), int(si.org_vocab_end_index - si.org_vocab_start_index)
    vocab = int(emb.org_vocab_size)
    site = torch.ops._C.vocab_parallel_embedding
    rec = Recorder()
    src = VocabRangeSource(rec, model, {name: {"start": start, "shard_rows": vs}})
    captured: list[tuple[Any, Any]] = []
    inner_box: list[Any] = []

    def recording(out, ids, weight, s0, s1, pad, a0, a1):
        r = inner_box[0](out, ids, weight, s0, s1, pad, a0, a1)
        captured.append((ids.detach().reshape(-1).to(torch.int64).cpu(), out.detach().clone()))
        return r

    rng = random.Random(1000)                         # the same ids on every rank: the embedding's forward all-reduces the partials
    edge = sorted(set(X.edge_ids(0, vs, vocab) + X.edge_ids(vs, vs, vocab))) + [rng.randrange(vocab) for _ in range(40)]
    out: dict[str, Any] = {"rank": rank, "site": name, "start": start, "shard_rows": vs, "vocab": vocab, "edge_ids": edge}
    with torch.no_grad():
        bare = {dt: emb(torch.tensor(edge, dtype=dt, device="cuda")).clone() for dt in (torch.int32, torch.int64)}
    src.install()
    h = hooks.Hooks()
    inner_box.append(torch.ops._C.vocab_parallel_embedding)
    h.patch(torch.ops._C, "vocab_parallel_embedding", recording)
    with torch.no_grad():
        emb(torch.tensor(edge, dtype=torch.int64, device="cuda"))
        out["outside_step_acquired"] = len(rec.acquired)
        rec._cur = types.SimpleNamespace(step=0)
        got = {dt: emb(torch.tensor(edge, dtype=dt, device="cuda")).clone() for dt in (torch.int32, torch.int64)}
    torch.cuda.synchronize()
    out["edge_out_equal"] = all(torch.equal(bare[dt].view(torch.int16), got[dt].view(torch.int16)) for dt in bare)
    worker._vr = {"src": src, "rec": rec, "captured": captured, "hooks": h, "site": site, "emb": emb, "start": start, "vs": vs,
                  "edge_n": len(rec.acquired), "edge_captured": len(captured), "forwards": [0],
                  "handle": model.register_forward_pre_hook(lambda _m, _a: worker._vr["forwards"].__setitem__(0, worker._vr["forwards"][0] + 1))}
    return out


def _check(pairs: list[tuple[Any, Any]], vals: list[Any], emb: Any, start: int, vs: int) -> dict[str, Any]:
    """Every committed row against the circuit's values of its id, and the kernel's partial against the committed range bits."""
    import torch

    cache: dict[int, tuple[int, int, int]] = {}
    rows = ne = disagree = 0
    first: list[Any] = []
    W = emb.weight.detach()
    for (ids, part), v in zip(pairs, vals):
        v = v.cpu()
        for i, tok in enumerate(ids.tolist()):
            if tok not in cache:
                cache[tok] = X.ir_values(vs, start, tok)
            want = cache[tok]
            got = tuple(int(x) for x in v[i].tolist())
            rows += 1
            if got != want:
                ne += 1
                if len(first) < 4:
                    first.append({"tok": tok, "committed": got, "ir": want})
            row = part[i].view(torch.int16)
            ref = W[want[0]].view(torch.int16) if want[1] and want[2] else torch.zeros_like(row)
            if not torch.equal(row, ref):
                disagree += 1
    return {"rows": rows, "ne_ir": ne, "kernel_disagrees": disagree, "first_ne": first, "ids_evaluated": len(cache)}


def _w_edge_result(worker) -> dict[str, Any]:
    st = worker._vr
    n = st["edge_n"]
    acquired = [t for _n, t in st["rec"].acquired[:n]]
    pairs = st["captured"][1:st["edge_captured"]]                # the first captured launch is the one outside a step
    r = _check(pairs, acquired, st["emb"], st["start"], st["vs"])
    st["live_from"] = (n, st["edge_captured"])
    st["forwards"][0] = 0
    return {"edge_rows": r["rows"], "edge_ne_ir": r["ne_ir"], "edge_kernel_disagrees": r["kernel_disagrees"], "edge_first_ne": r["first_ne"],
            "edge_names": sorted({nm for nm, _t in st["rec"].acquired[:n]})}


def _w_live_result(worker) -> dict[str, Any]:
    import torch

    from verity_vllm.engine import hooks

    st = worker._vr
    st["handle"].remove()
    st["hooks"].uninstall()
    st["src"].uninstall()
    a0, c0 = st["live_from"]
    acquired = st["rec"].acquired[a0:]
    pairs = st["captured"][c0:]
    r = _check(pairs, [t for _n, t in acquired], st["emb"], st["start"], st["vs"])
    restored = torch.ops._C.vocab_parallel_embedding is st["site"] and not any(getattr(p, "owner", None) is torch.ops._C for p in hooks.live())
    return {"start": st["start"], "shard_rows": st["vs"], "rows": r["rows"], "rows_ne_ir": r["ne_ir"], "kernel_disagrees": r["kernel_disagrees"],
            "first_ne": r["first_ne"], "ids_evaluated": r["ids_evaluated"], "forwards": st["forwards"][0], "launches": len(pairs),
            "one_per_step": len(acquired) == len(pairs) == st["forwards"][0] > 0, "restored": restored,
            "names": sorted({nm for nm, _t in acquired}), "errors": len(st["rec"].errors), "error_msgs": st["rec"].errors[:4]}


def make_tiny(tok_dir: str, out_dir: str, seed: int) -> dict[str, Any]:
    import shutil

    import torch
    import transformers

    base = json.loads(Path(tok_dir, "config.json").read_text())
    cfg = transformers.LlamaConfig(vocab_size=int(base["vocab_size"]), hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=2048, tie_word_embeddings=False,
                                   **{k: base.get(k) for k in ("bos_token_id", "eos_token_id", "pad_token_id")})
    torch.manual_seed(seed)
    transformers.AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16).save_pretrained(out_dir, safe_serialization=True)
    for f in Path(tok_dir).iterdir():
        if f.is_file() and f.name != "config.json" and f.suffix in (".json", ".txt", ".model") and not f.name.endswith(".index.json"):
            shutil.copy(f, Path(out_dir, f.name))
    return {"model": "tiny random-weight LlamaForCausalLM", "vocab": cfg.vocab_size, "heads": 4, "kv_heads": 2, "layers": 2, "hidden": 128, "seed": seed,
            "tokenizer": tok_dir}


def run(case: str, out: str, max_tokens: int = 8) -> dict[str, Any]:
    from verity_vllm.engine import engine_profile as prof

    env_flags = prof.apply_env()
    import torch
    from vllm import LLM, SamplingParams

    if case == "tiny":
        tok = prof.checkpoint_for("B0", "manifests/checkpoints.json")["local_path"]
        d = str(Path(out, "tiny-llama"))
        cp = {**make_tiny(tok, d, 20260926), "repo": "tiny", "revision": None}
        kw = {"model": d, "tokenizer": d, "dtype": "bfloat16", "seed": 0, "max_model_len": 512, "gpu_memory_utilization": 0.5,
              "tensor_parallel_size": X.WORLD, "enforce_eager": True}
    else:
        cp = prof.checkpoint_for(case, "manifests/checkpoints.json")
        kw = prof.engine_kwargs(cp, replay=False, seed=prof.ENGINE_SEED, enforce_eager=True, tensor_parallel_size=X.WORLD)
    kw.update(enable_prefix_caching=False, max_num_seqs=4)
    doc: dict[str, Any] = {"schema": X.SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv, "case": case,
                           "model": cp.get("repo") if case != "tiny" else cp, "revision": cp.get("revision"), "env_flags": env_flags,
                           "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                                           "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                                           "vllm": __import__("vllm").__version__},
                           "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")}}
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    try:
        bare = [list(o.outputs[0].token_ids) for o in llm.generate(list(PROMPTS), sp, use_tqdm=False)]
        installed = llm.collective_rpc(_w_install)
        edge = llm.collective_rpc(_w_edge_result)
        served = [list(o.outputs[0].token_ids) for o in llm.generate(list(PROMPTS), sp, use_tqdm=False)]
        live = llm.collective_rpc(_w_live_result)
        doc["ranks"] = []
        for a, e in zip(installed, edge):
            r = {**a, **e}
            r["reference_ne_ir"] = sum(1 for t in a["edge_ids"] if VR.values(t, a["start"], a["shard_rows"]) != X.ir_values(a["shard_rows"], a["start"], t))
            r["ok"] = X.rank_ok(r)
            doc["ranks"].append(r)
        doc["live"] = {"prompts": len(PROMPTS), "max_tokens": max_tokens, "tokens_bare": bare, "tokens_served": served, "tokens_equal_bare": bare == served,
                       "ranks": [{**v, "outside_step_acquired": a["outside_step_acquired"]} for a, v in zip(installed, live)]}
        doc["live"]["ok"] = X.live_ok(doc["live"])
    except Exception as e:  # noqa: BLE001
        doc["error"] = f"{type(e).__name__}: {e}"[:600]
    for r in doc.get("ranks") or []:
        print(f"[vocab-range] rank {r['rank']} site {r['site']} [{r['start']}, {r['start'] + r['shard_rows']}): edge rows {r['edge_rows']} "
              f"!=ir {r['edge_ne_ir']} kernel-disagrees {r['edge_kernel_disagrees']} out== {r['edge_out_equal']} -> {'OK' if r['ok'] else 'FAIL'}", flush=True)
    lv = doc.get("live") or {}
    for r in lv.get("ranks") or []:
        print(f"[vocab-range] live rank start {r['start']}: rows {r['rows']} !=ir {r['rows_ne_ir']} kernel-disagrees {r['kernel_disagrees']} "
              f"forwards {r['forwards']} launches {r['launches']} one/step {r['one_per_step']} restored {r['restored']} errors {r['errors']}", flush=True)
    print(f"[vocab-range] tokens equal bare {lv.get('tokens_equal_bare')} {doc.get('error') or ''}", flush=True)
    doc = X.seal(doc)
    path = Path(out) / X.RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"VOCAB-RANGE-EXACTNESS {'OK' if doc['ok'] else 'FAIL'} {doc['digest']} {path}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vocabulary-range exactness on a TP2 engine: the committed range values are the IR's, per rank")
    ap.add_argument("--case", default="tiny", help="`tiny` (a random-weight model, B0's tokenizer) or a checkpoint role whose heads and vocabulary "
                    "split over two ranks, the vocabulary into unpadded halves")
    ap.add_argument("--out", help=f"directory for {X.RECORD}")
    ap.add_argument("--verify", metavar="RECORD", help="check a record's digest and verdict (no GPU)")
    a = ap.parse_args(argv)
    if a.verify:
        doc = json.loads(Path(a.verify).read_text())
        problems = X.verify_record(doc)
        print(f"{a.verify}: {'intact, ok=' + str(doc.get('ok')) if not problems else '; '.join(problems)} (digest {doc.get('digest')})")
        return 1 if problems or not doc.get("ok") else 0
    if not a.out:
        ap.error("--out is required (or --verify RECORD)")
    out = str(Path(a.out).resolve())
    os.chdir(Path(__file__).resolve().parents[2])
    return 0 if run(a.case, out)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
