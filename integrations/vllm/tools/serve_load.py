"""A short varied load on a `verity-vllm serve-openai` endpoint, in bursts the server's idle closes into Commit windows.

    python tools/serve_load.py --port 18084 --model unsloth/Llama-3.2-1B [--bursts 8] [--per-burst 4] [--seed 0] --out FILE

Each burst is ``--per-burst`` concurrent ``/v1/completions`` requests, each with its own ``user`` (``b<burst>-<i>``), a prompt of
distinct token ids (BOS, then 16 to 200 ids, so no two share a cached block), top_p 1, a temperature of 0, 0.7 or 1.0 (half of
the stochastic ones with a client seed) and ``max_tokens`` 4 to 40; the next burst starts once every request of this one has
finished and the engine has idled. One JSON line a request goes to ``--out``: its user, body and response.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.request


def body(rng: random.Random, model: str, user: str) -> dict:
    temperature = rng.choice([0.0, 0.7, 1.0])
    b = {"model": model, "user": user, "prompt": [128000] + [rng.randrange(1000, 120000) for _ in range(rng.randint(16, 200))],
         "max_tokens": rng.randint(4, 40), "temperature": temperature, "top_p": 1.0, "return_token_ids": True}
    if temperature and rng.random() < 0.5:
        b["seed"] = rng.randrange(1 << 31)
    return b


def post(port: int, b: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(b).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:  # noqa: S310  (the local endpoint)
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bursts", type=int, default=8)
    ap.add_argument("--per-burst", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--idle-s", type=float, default=2.0, help="the pause after a burst, for the window to close")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    with open(a.out, "a") as fh:
        for k in range(a.bursts):
            bodies = [body(rng, a.model, f"b{k}-{i}") for i in range(a.per_burst)]
            got: list = [None] * len(bodies)

            def go(i: int) -> None:
                try:
                    got[i] = post(a.port, bodies[i])
                except Exception as e:  # noqa: BLE001  (recorded with the request)
                    got[i] = {"error": f"{type(e).__name__}: {e}"}

            ts = [threading.Thread(target=go, args=(i,)) for i in range(len(bodies))]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            for b, r in zip(bodies, got):
                fh.write(json.dumps({"burst": k, "user": b["user"], "body": b, "response": r}) + "\n")
            fh.flush()
            print(f"[serve-load] burst {k}: {sum('error' not in (r or {}) for r in got)}/{len(got)} ok", flush=True)
            time.sleep(a.idle_s)  # the engine idles; the server closes the window at its next idle
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
