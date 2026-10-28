"""commit-arb's shared-prefix traffic (`pipeline.commit_arb.traffic`: system prompts, conversations of several turns) sent to a
running ``serve-openai --prefix-caching`` over HTTP, for its GPU arm: one thread a conversation, each turn a streamed
``/v1/completions`` of token ids with ``return_token_ids``, the next turn's prompt this turn's prompt, its streamed ids and the
next question. ``top_p`` is left out (serve-openai refuses it below 1).

    python -m tests.pipeline.serve_prefix_gpu --url URL --out FILE [--requests 32] [--seed 0] [--prompt-range 16:512]
        [--tokens 64] [--system-prompts 4] [--system-tokens 384] [--turns 2] [--max-model-len 2048]
    (from integrations/vllm, the server already listening)

writes one JSON line a turn to FILE (its conversation, turn, HTTP id, prompt ids, streamed ids, finish) and prints the counts;
`compare` then holds them to the server's ``requests.jsonl``: every turn checked and passed there, with the ids the client
saw."""
from __future__ import annotations

import argparse
import json
import threading
import urllib.request
from pathlib import Path
from typing import Any

from verity_vllm.pipeline import commit_arb as CA


def post(url: str, body: dict) -> Any:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    return urllib.request.urlopen(req)


def complete(url: str, model: str, prompt: list[int], sampling: dict) -> dict[str, Any]:
    body = {"model": model, "prompt": prompt, "stream": True, "return_token_ids": True,
            **{k: v for k, v in sampling.items() if k != "top_p"}}
    ids, http_id, finish = [], None, None
    with post(url + "/v1/completions", body) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[len(b"data: "):])
            http_id = http_id or chunk["id"]
            for c in chunk["choices"]:
                ids.extend(c.get("token_ids") or [])
                finish = c.get("finish_reason") or finish
    return {"http_id": http_id, "streamed_ids": ids, "finish": finish}


def conversations(reqs: list[dict]) -> list[list[dict]]:
    by_parent = {q["follows"]: q for q in reqs if "follows" in q}
    out = []
    for q in reqs:
        if "follows" in q:
            continue
        chain = [q]
        while chain[-1]["request_id"] in by_parent:
            chain.append(by_parent[chain[-1]["request_id"]])
        out.append(chain)
    return out


def serve(url: str, model: str, chain: list[dict], max_model_len: int, sink: list[dict]) -> None:
    prompt, streamed = None, None
    for t, q in enumerate(chain):
        prompt = q["prompt"] if t == 0 else prompt + streamed + q["suffix"]
        if len(prompt) + q["sampling"]["max_tokens"] > max_model_len:
            sink.append({"request_id": q["request_id"], "turn": t, "skipped": "past --max-model-len"})
            return
        got = complete(url, model, prompt, q["sampling"])
        streamed = got["streamed_ids"]
        sink.append({"request_id": q["request_id"], "turn": t, "prompt_ids": prompt, **got})


def compare(client: list[dict], server: list[dict]) -> dict[str, Any]:
    """Every turn the client was served is a request the server checked and passed, with the same prompt and streamed ids."""
    by_http = {r["http_id"]: r for r in server}
    served = [c for c in client if "http_id" in c]
    missing = [c["request_id"] for c in served if c["http_id"] not in by_http]
    differ = [c["request_id"] for c in served if c["http_id"] in by_http
              and (by_http[c["http_id"]]["prompt_ids"], by_http[c["http_id"]]["streamed_ids"]) != (c["prompt_ids"], c["streamed_ids"])]
    failed = [c["request_id"] for c in served if c["http_id"] in by_http and not by_http[c["http_id"]]["check"]["passed"]]
    hits = [by_http[c["http_id"]]["check"].get("cache_hit") or 0 for c in served if c["http_id"] in by_http]
    return {"turns": len(served), "later_turns": sum(1 for c in served if c["turn"]), "skipped": len(client) - len(served),
            "missing": missing[:10], "differ": differ[:10], "failed": failed[:10], "cache_hits": sum(1 for h in hits if h),
            "cached_positions": sum(hits),
            "passed": bool(served) and not missing and not differ and not failed and any(hits)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-requests", help="the server's requests.jsonl: compare and print the verdict instead of serving")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-range", default="16:512")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--system-prompts", type=int, default=4)
    ap.add_argument("--system-tokens", type=int, default=384)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=2048)
    a = ap.parse_args(argv)
    if a.server_requests:
        client = [json.loads(x) for x in Path(a.out).read_text().splitlines()]
        server = [json.loads(x) for x in Path(a.server_requests).read_text().splitlines()]
        got = compare(client, server)
        print(f"SERVE-PREFIX {'PASS' if got['passed'] else 'FAIL'}: {json.dumps(got)}", flush=True)
        return 0 if got["passed"] else 1
    with urllib.request.urlopen(a.url + "/v1/models") as r:
        model = json.loads(r.read())["data"][0]["id"]

    def encode(text: str) -> list[int]:
        with post(a.url + "/tokenize", {"model": model, "prompt": text}) as r:
            return json.loads(r.read())["tokens"]

    reqs = CA.traffic(encode, a.requests, seed=a.seed, prompt_tokens=CA.prompt_range(a.prompt_range), max_tokens=a.tokens,
                      rate=a.rate, system_prompts=a.system_prompts, system_tokens=a.system_tokens, turns=a.turns)
    sink: list[dict] = []
    threads = [threading.Thread(target=serve, args=(a.url, model, chain, a.max_model_len, sink)) for chain in conversations(reqs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in sorted(sink, key=lambda x: x["request_id"])))
    print(json.dumps({"turns": len(sink), "served": sum(1 for x in sink if "http_id" in x)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
