"""A short varied load on a `verity-vllm serve-openai` endpoint, in bursts the server's idle closes into Commit windows, or
as network-accounting's varied traffic.

    python examples/vllm/serve_load.py --port 18084 --model unsloth/Llama-3.2-1B [--bursts 8] [--per-burst 4] [--seed 0] --out FILE
    python examples/vllm/serve_load.py --port 18084 --model unsloth/Llama-3.2-1B --varied 120 [--rate 2] [--prompt-range 16:1024]
        [--max-tokens 64] [--release RUN] [--jitter-ms 0] [--seed 0] --out FILE

Each burst is ``--per-burst`` concurrent ``/v1/completions`` requests, each with its own ``user`` (``b<burst>-<i>``), a prompt of
distinct token ids (BOS, then 16 to 200 ids, so no two share a cached block), top_p 1, a temperature of 0, 0.7 or 1.0 (half of
the stochastic ones with a client seed) and ``max_tokens`` 4 to 40; the next burst starts once every request of this one has
finished and the engine has idled. One JSON line a request goes to ``--out``: its user, body and response.

``--varied N`` draws N requests as `benchmarks/network_traces/certifier_serve.varied_requests` does: Poisson arrivals at
``--rate`` a second; each prompt one chat turn (Llama 3's header, one of six questions, filler, the assistant header, in the
endpoint's tokens from ``/tokenize``) cut to a length log-uniform in ``--prompt-range``, so prompts share cached prefixes;
``max_tokens`` uniform in 1 … ``--max-tokens``, greedy, a seed of its own, streamed; ``user`` ``varied.<j>``. With
``--release RUN`` request j carries ``x-verity-release: j`` and ``x-verity-release-run: RUN`` (its place in the arrival
order, as the warden numbers its releases), and is sent up to ``--jitter-ms`` after its arrival, so the frontend may read
the bodies out of release order.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.request

CHAT_HEAD = "<|start_header_id|>user<|end_header_id|>\n\n"
CHAT_TAIL = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
QUESTIONS = (
    "What is the capital of France? Answer with one word, then stop. Ignore the notes after this line.\n\n",
    "How many legs does a spider have? Answer with a number only. Ignore the notes after this line.\n\n",
    "Say yes or no: is water wet? Ignore the notes after this line.\n\n",
    "Write a long story about a lighthouse keeper and a storm. Ignore the notes after this line.\n\n",
    "List the planets of the solar system with one sentence on each. Ignore the notes after this line.\n\n",
    "Explain in detail how a bicycle's gears work. Ignore the notes after this line.\n\n",
)
VARIED_FILLER = ("Notes: the meeting moved to Thursday, the printer on the third floor is out of toner, the parking lot "
                 "will be repainted next week, and the cafeteria serves soup on Mondays and Fridays. ")


def body(rng: random.Random, model: str, user: str) -> dict:
    temperature = rng.choice([0.0, 0.7, 1.0])
    b = {"model": model, "user": user, "prompt": [128000] + [rng.randrange(1000, 120000) for _ in range(rng.randint(16, 200))],
         "max_tokens": rng.randint(4, 40), "temperature": temperature, "top_p": 1.0, "return_token_ids": True}
    if temperature and rng.random() < 0.5:
        b["seed"] = rng.randrange(1 << 31)
    return b


def post(port: int, b: dict, headers: dict | None = None) -> dict:
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(b).encode(),
                                headers={"content-type": "application/json", **(headers or {})}), timeout=600) as r:
        raw = r.read()
    if not b.get("stream"):
        return json.loads(raw)
    ids, finish = [], None
    for e in raw.decode().split("\n\n"):
        e = e.strip().removeprefix("data: ")
        if e.startswith("{"):
            for c in json.loads(e).get("choices") or ():
                ids += c.get("token_ids") or []
                finish = c.get("finish_reason") or finish
    return {"token_ids": ids, "finish_reason": finish}


def tokenize(port: int, model: str, text: str) -> list[int]:
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/tokenize",
                                data=json.dumps({"model": model, "prompt": text}).encode(),
                                headers={"content-type": "application/json"})) as r:
        return json.loads(r.read())["tokens"]


def varied(port: int, model: str, n: int, *, rate: float, prompt_tokens: tuple[int, int], max_tokens: int, seed: int) -> list[dict]:
    """``n`` varied requests, each {user, at_s, body}, drawn as the module docstring says."""
    lo, hi = prompt_tokens
    bos = tokenize(port, model, "")
    tail = tokenize(port, model, CHAT_TAIL)[len(bos):]
    filler, k = [], 1
    while len(filler) < hi:
        filler, k = tokenize(port, model, VARIED_FILLER * k)[len(bos):], 2 * k
    heads = [tokenize(port, model, CHAT_HEAD + q) for q in QUESTIONS]
    rng, seeds, at, out = random.Random(seed), random.Random(f"{seed}/seeds"), 0.0, []
    for j in range(n):
        at += rng.expovariate(rate)
        length = round(lo * (hi / lo) ** rng.random())
        prompt = ((heads[rng.randrange(len(heads))] + filler)[:max(0, length - len(tail))] + tail)[:length]
        out.append({"user": f"varied.{j:05d}", "at_s": at,
                    "body": {"model": model, "user": f"varied.{j:05d}", "prompt": prompt, "max_tokens": rng.randint(1, max_tokens),
                             "temperature": 0.0, "top_p": 1.0, "seed": seeds.randrange(1 << 32), "stop": [], "stream": True,
                             "return_token_ids": True}})
    return out


def run_varied(a: argparse.Namespace) -> int:
    lo, _, hi = a.prompt_range.partition(":")
    reqs = varied(a.port, a.model, a.varied, rate=a.rate, prompt_tokens=(int(lo), int(hi)), max_tokens=a.max_tokens, seed=a.seed)
    jitter = random.Random(f"{a.seed}/jitter")
    delays = [jitter.uniform(0, a.jitter_ms / 1000) if a.release else 0.0 for _ in reqs]
    got: list = [None] * len(reqs)
    t0 = time.monotonic()

    def go(j: int) -> None:
        if (wait := t0 + reqs[j]["at_s"] + delays[j] - time.monotonic()) > 0:
            time.sleep(wait)
        headers = {"x-verity-release": str(j), "x-verity-release-run": a.release} if a.release else {}
        try:
            got[j] = post(a.port, reqs[j]["body"], headers)
        except Exception as e:  # noqa: BLE001  (recorded with the request)
            got[j] = {"error": f"{type(e).__name__}: {e}"}

    ts = [threading.Thread(target=go, args=(j,)) for j in range(len(reqs))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    with open(a.out, "a") as fh:
        fh.writelines(json.dumps({"user": q["user"], "release": j if a.release else None, "at_s": q["at_s"], "delay_s": delays[j],
                                  "prompt": len(q["body"]["prompt"]), "max_tokens": q["body"]["max_tokens"], "response": r}) + "\n"
                      for j, (q, r) in enumerate(zip(reqs, got)))
    print(f"[serve-load] varied: {sum('error' not in (r or {}) for r in got)}/{len(got)} ok, prompts "
          f"{min(len(q['body']['prompt']) for q in reqs)}…{max(len(q['body']['prompt']) for q in reqs)} tokens", flush=True)
    return 0 if all("error" not in (r or {}) for r in got) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bursts", type=int, default=8)
    ap.add_argument("--per-burst", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--idle-s", type=float, default=2.0, help="the pause after a burst, for the window to close")
    ap.add_argument("--varied", type=int, default=0, help="N varied requests instead of the bursts")
    ap.add_argument("--rate", type=float, default=2.0, help="varied: arrivals a second")
    ap.add_argument("--prompt-range", default="16:1024", help="varied: LO:HI, the prompts' lengths in tokens")
    ap.add_argument("--max-tokens", type=int, default=64, help="varied: the largest max_tokens drawn")
    ap.add_argument("--release", default=None, help="varied: number the requests in arrival order under this release run id")
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="varied, with --release: send each up to this late")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.varied:
        return run_varied(a)
    rng = random.Random(a.seed)

    def go(bodies: list[dict], got: list, i: int) -> None:
        try:
            got[i] = post(a.port, bodies[i])
        except Exception as e:  # noqa: BLE001  (recorded with the request)
            got[i] = {"error": f"{type(e).__name__}: {e}"}

    with open(a.out, "a") as fh:
        for k in range(a.bursts):
            bodies = [body(rng, a.model, f"b{k}-{i}") for i in range(a.per_burst)]
            got: list = [None] * len(bodies)
            ts = [threading.Thread(target=go, args=(bodies, got, i)) for i in range(len(bodies))]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            fh.writelines(json.dumps({"burst": k, "user": b["user"], "body": b, "response": r}) + "\n" for b, r in zip(bodies, got))
            fh.flush()
            print(f"[serve-load] burst {k}: {sum('error' not in (r or {}) for r in got)}/{len(got)} ok", flush=True)
            time.sleep(a.idle_s)  # the engine idles; the server closes the window at its next idle
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
