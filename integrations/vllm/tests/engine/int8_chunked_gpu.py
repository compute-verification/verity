"""On one GPU: the int8 Llama-3.1-8B pin (as `tests/engine/int8_exact_gpu.py` finds or makes it) served with
`TargetProfile.int8_exact` (linears and attention) at ``--mnbt`` = vLLM's max_num_batched_tokens, greedy, on PROMPTS: two
512-token prompts, a 300-token one and a short one, batched, then the first 512-token prompt alone.  At 64 every long prompt is
prefilled in 64-row chunks (`engine.int8_ops.paged_attention` serves a request's m rows as the last m of its s-token window); at
2048 every prompt is prefilled whole.

One process serves one ``--mnbt`` and writes ``<out>/int8_chunked_<mnbt>.json``: the token ids, and how many int8 attention
requests (of ``verity_int8::window_rows_batch``, every layer) were a decode row (m = 1), a whole prompt (m = s) or a chunk (1 < m < s).
``--compare A B`` writes ``<out>/int8_chunked_compare.json``: whether each generation's token ids are equal, and where they
first differ.  A chunk is its own window: an earlier chunk's rows read V scaled over the tokens cached by its end, not over the
whole prompt, so the two serves are equal only where that changes no greedy token.

    cd integrations/vllm; . ../../benchmarks/harness/env.sh
    $VENV/bin/python -m tests.engine.int8_chunked_gpu --mnbt 64 --out DIR --at ISO_TIME --by LANE
    $VENV/bin/python -m tests.engine.int8_chunked_gpu --compare DIR/int8_chunked_64.json DIR/int8_chunked_2048.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "verity-vllm/int8-chunked-gpu/v1"
CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Norway", "Oslo"), ("Egypt", "Cairo"),
            ("Chile", "Santiago"), ("India", "New Delhi"), ("Canada", "Ottawa"), ("Ghana", "Accra"), ("Spain", "Madrid"),
            ("Vietnam", "Hanoi"), ("Poland", "Warsaw"), ("Cuba", "Havana"), ("Iran", "Tehran"), ("Fiji", "Suva")]


def _texts() -> list[str]:
    caps = "Countries and their capitals:\n" + "".join(f"{i + 1}. The capital of {c} is {k}.\n" for i, (c, k) in enumerate(CAPITALS * 8))
    code = "".join(f"def f{i}(x):\n    \"\"\"Return {i} x plus {i * i}.\"\"\"\n    return {i} * x + {i * i}\n\n\n" for i in range(60))
    story = ("Once upon a time, in a small village by the sea, there lived an old fisherman and his granddaughter. Every morning "
             "they rowed out past the lighthouse, and every evening they counted the day's catch on the pier. ") * 20
    return [caps, code, story]


def prompts(tokenizer: Any) -> list[list[int]]:
    """Token ids: the capitals and the code cut to 512 tokens, the story to 300, and a short prompt."""
    caps, code, story = (tokenizer.encode(t) for t in _texts())
    short = tokenizer.encode("The capital of France is")
    if min(len(caps), len(code)) < 512 or len(story) < 300:
        raise SystemExit(f"prompt texts are too short: {len(caps)}, {len(code)}, {len(story)} tokens")
    return [caps[:512], code[:512], story[:300], short]


def generate(llm: Any, ids: list[list[int]], max_tokens: int) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0),
                        use_tqdm=False)
    return [{"prompt_tokens": len(p), "text": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids)} for p, o in zip(ids, outs)]


def count_calls(shapes: Counter):
    """Wrap `int8_ops.window_rows_batch` (which `paged_attention` calls by its module name) to count each request's (m, s)."""
    from verity_vllm.engine import int8_ops
    real = int8_ops.window_rows_batch

    def counted(query, kq, vq, sv, v0, k_scales, qsl, lens, scale):
        for r, s in enumerate(lens):
            if s:
                shapes[(qsl[r + 1] - qsl[r], s)] += 1
        return real(query, kq, vq, sv, v0, k_scales, qsl, lens, scale)

    int8_ops.window_rows_batch = counted


def kinds(shapes: Counter) -> dict[str, int]:
    out = {"decode": 0, "whole": 0, "chunk": 0}
    for (m, s), n in shapes.items():
        out["decode" if m == 1 else "whole" if m == s else "chunk"] += n
    return out


def serve(a: argparse.Namespace) -> dict[str, Any]:
    import torch

    from tests.engine.int8_exact_gpu import pin, scratch_checkpoints
    from verity_vllm.engine.build import build_engine, target_mismatches
    from verity_vllm.program.frontend.kernel_readback import int8_exact_of
    from verity_vllm.program.frontend.target_profile import TargetProfile

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = TargetProfile.from_device().with_overrides(int8_exact="all")
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                           "mnbt": a.mnbt}
    shapes: Counter = Counter()
    count_calls(shapes)
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = pin(a)
        doc["pin"] = {"repo": entry["repo"], "revision": entry["revision"], "source": how}
        eng = build_engine(scratch_checkpoints(a.checkpoints, entry, Path(scratch)), case=a.role, target=target, max_num_seqs=8,
                           engine_args={"max_num_batched_tokens": a.mnbt, "enable_chunked_prefill": True, "enable_prefix_caching": False})
    sc = eng.llm.llm_engine.vllm_config.scheduler_config
    doc["engine"] = {"target_mismatches": target_mismatches(eng), "int8_exact": int8_exact_of(eng.model),
                     "max_num_batched_tokens": sc.max_num_batched_tokens, "enable_chunked_prefill": sc.enable_chunked_prefill,
                     "enforce_eager": eng.kwargs.get("enforce_eager"), "attention_backend": eng.kwargs.get("attention_backend")}
    print(f"[chunked] engine {json.dumps(doc['engine'], default=str)}", flush=True)
    ids = prompts(eng.llm.get_tokenizer())
    shapes.clear()
    doc["batched"] = generate(eng.llm, ids, a.max_tokens)
    doc["batched_calls"] = kinds(shapes)
    shapes.clear()
    doc["solo"] = generate(eng.llm, ids[:1], a.max_tokens)
    doc["solo_calls"] = kinds(shapes)
    doc["chunk_shapes"] = sorted({f"{m}/{s}" for (m, s) in shapes if 1 < m < s}, key=lambda x: int(x.split("/")[1]))
    doc["ok"] = bool(not doc["engine"]["target_mismatches"] and doc["engine"]["int8_exact"] == "all"
                     and all(g["token_ids"] for g in doc["batched"] + doc["solo"]))
    (out / f"int8_chunked_{a.mnbt}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"INT8-CHUNKED mnbt={a.mnbt} ok={doc['ok']} batched_calls={doc['batched_calls']} solo_calls={doc['solo_calls']}", flush=True)
    return doc


def _first_diff(x: list[int], y: list[int]) -> int | None:
    return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None if len(x) == len(y) else min(len(x), len(y)))


def compare(a: argparse.Namespace) -> dict[str, Any]:
    chunked, whole = (json.loads(Path(p).read_text()) for p in a.compare)
    rows = []
    for phase in ("batched", "solo"):
        for i, (c, w) in enumerate(zip(chunked[phase], whole[phase])):
            d = _first_diff(c["token_ids"], w["token_ids"])
            rows.append({"phase": phase, "prompt": i, "prompt_tokens": c["prompt_tokens"], "tokens": len(c["token_ids"]),
                         "equal": d is None, "first_diff": d, "chunked_text": c["text"], "whole_text": w["text"]})
    doc = {"schema": SCHEMA, "chunked": {"mnbt": chunked["mnbt"], "calls": chunked["batched_calls"], "solo_calls": chunked["solo_calls"]},
           "whole": {"mnbt": whole["mnbt"], "calls": whole["batched_calls"], "solo_calls": whole["solo_calls"]},
           "served_chunks": chunked["batched_calls"]["chunk"] > 0 and chunked["solo_calls"]["chunk"] > 0,
           "whole_unchunked": whole["batched_calls"]["chunk"] == 0 and whole["solo_calls"]["chunk"] == 0,
           "generations": rows, "equal": all(r["equal"] for r in rows)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "int8_chunked_compare.json").write_text(json.dumps(doc, indent=1) + "\n")
    for r in rows:
        print(f"[compare] {r['phase']} prompt {r['prompt']} ({r['prompt_tokens']} tokens): equal={r['equal']} first_diff={r['first_diff']}")
    print(f"INT8-CHUNKED-COMPARE equal={doc['equal']} served_chunks={doc['served_chunks']} whole_unchunked={doc['whole_unchunked']} "
          f"chunked_calls={doc['chunked']['calls']} whole_calls={doc['whole']['calls']}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="int8_exact served with chunked prefill against the unchunked serve, token for token")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mnbt", type=int, default=64, help="vLLM's max_num_batched_tokens")
    ap.add_argument("--compare", nargs=2, metavar=("CHUNKED", "WHOLE"), help="judge two records instead of serving")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA31_8B")
    ap.add_argument("--role", default="LLAMA31_8B_INT8")
    ap.add_argument("--repo", default="verity-int8/Llama-3.1-8B-INT8")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--at", default="2026-10-17T00:00:00Z", help="resolved_at of a new pin entry (UTC, ISO 8601)")
    ap.add_argument("--by", default="vllm-int8-chunk", help="resolved_by of a new pin entry")
    a = ap.parse_args(argv)
    doc = compare(a) if a.compare else serve(a)
    return 0 if doc.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
