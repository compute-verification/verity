"""A short varied load on a `verity-vllm serve-openai` endpoint, in bursts the server's idle closes into Commit windows, or
as network-accounting's varied traffic.

    python demos/vllm/serve_load.py --port 18084 --model unsloth/Llama-3.2-1B [--bursts 8] [--per-burst 4] [--seed 0] --out FILE
    python demos/vllm/serve_load.py --port 18084 --model unsloth/Llama-3.2-1B --varied 120 [--rate 2] [--prompt-range 16:1024]
        [--max-tokens 64] [--release RUN] [--jitter-ms 0] [--seed 0] --out FILE
    python demos/vllm/serve_load.py --port 18084 --model unsloth/Llama-3.2-1B --fixed 4 [--window-steps 16]
        [--prompt-range 16:1024] [--release RUN] [--seed 0] --out FILE

Each burst is ``--per-burst`` concurrent ``/v1/completions`` requests, each with its own ``user`` (``b<burst>-<i>``), a prompt of
distinct token ids (BOS, then 16 to 200 ids, so no two share a cached block), top_p 1, a temperature of 0, 0.7 or 1.0 (half of
the stochastic ones with a client seed) and ``max_tokens`` 4 to 40; the next burst starts once every request of this one has
finished and the engine has idled. One JSON line a request goes to ``--out``: its user, body and response.

``--varied N`` draws N requests as `benchmarks/network_traces/certifier_serve.varied_requests` does: Poisson arrivals at
``--rate`` a second; each prompt one chat turn (Llama 3's header, one of six questions, filler, the assistant header, in the
endpoint's tokens from ``/tokenize``) cut to a length log-uniform in ``--prompt-range``, so prompts share cached prefixes;
``max_tokens`` uniform in 1 … ``--max-tokens``, greedy, a seed of its own, streamed; ``user`` ``varied.<j>``. With
``--release RUN`` request j carries ``x-verity-release: j`` and ``x-verity-release-run: RUN`` (its place in the arrival
order, as the network certifier numbers its releases), and is sent up to ``--jitter-ms`` after its arrival, so the frontend may read
the bodies out of release order.

``--fixed N`` is one full window of fixed length: N requests whose prompts are drawn as ``--varied`` draws them
(``--prompt-range L:L`` fixes their length), each with ``ignore_eos`` and ``max_tokens`` S (``--window-steps``), all arriving
at 0, streamed; ``user`` ``fixed.<j>``. With ``--release RUN`` request 0 goes out only once the endpoint has answered every
other one's head: under ``--admit release-order`` serve-openai answers a streamed request's head once it holds it, and holds
each until every earlier-released one is there, so request 0 brings all N into the engine together. They are prefilled at
its first step and each runs exactly S steps.

A varied or fixed run also writes its timings beside ``--out`` (`timings_path`: load.jsonl's are load.timings.json), one JSON
document: ``start_unix``, the wall clock at the release run's start (the server's step ends, step_trace.jsonl's ``t_end``,
are on that clock), and for each request in arrival order its ``user``, ``release``, ``at_s`` (its drawn arrival),
``send_s``, ``first_token_s`` (when the first streamed event carrying output ids was read), ``done_s`` (when its stream
ended), ``tokens`` (the output ids it streamed) and ``ok``, each time in seconds on the monotonic clock from the release
run's start. They are the client's measurements of synthetic load; nothing checks them.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

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


def post(port: int, b: dict, headers: dict | None = None, times: dict | None = None,
         clock: Callable[[], float] = time.monotonic, opened: Callable[[], None] | None = None) -> dict:
    """The response to ``b``, a streamed one as its output ids and finish. With ``times``, its ``clock`` readings: ``send``
    as the request goes out, ``first_token`` as the first event carrying output ids is read (streamed only) and ``done``
    once the response is read. ``opened`` is called once the response's head is read."""
    def stamp(k: str) -> None:
        if times is not None:
            times[k] = clock()

    stamp("send")
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(b).encode(),
                                headers={"content-type": "application/json", **(headers or {})}), timeout=600) as r:
        if opened is not None:
            opened()
        if not b.get("stream"):
            got = json.loads(r.read())
            stamp("done")
            return got
        ids, finish = [], None
        for line in r:
            e = line.decode().strip().removeprefix("data: ")
            if e.startswith("{"):
                for c in json.loads(e).get("choices") or ():
                    if c.get("token_ids") and not ids:
                        stamp("first_token")
                    ids += c.get("token_ids") or []
                    finish = c.get("finish_reason") or finish
    stamp("done")
    return {"token_ids": ids, "finish_reason": finish}


def timings_path(out: str | Path) -> Path:
    """The varied or fixed run's timings file beside its record ``out``."""
    out = Path(out)
    return out.with_name(out.name.removesuffix(".jsonl") + ".timings.json")


def timings(reqs: list[dict], got: list, times: list[dict], t0: float, *, release: str | None, start_unix: float) -> dict:
    """The varied or fixed run's timings document (the module docstring): each request's ``times`` less ``t0``, the release
    run's start."""
    return {"clock": "monotonic seconds from the release run's start", "release_run": release, "start_unix": start_unix,
            "requests": [{"user": q["user"], "release": j if release else None, "at_s": q["at_s"],
                          **{f"{k}_s": t[k] - t0 if k in t else None for k in ("send", "first_token", "done")},
                          "tokens": len((r or {}).get("token_ids") or ()), "ok": "error" not in (r or {})}
                         for j, (q, r, t) in enumerate(zip(reqs, got, times))]}


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


def fixed(port: int, model: str, n: int, *, window_steps: int, prompt_tokens: tuple[int, int], seed: int) -> list[dict]:
    """``n`` requests of one full window of ``window_steps`` steps, each {user, at_s, body}, as the module docstring says."""
    return [{"user": f"fixed.{j:05d}", "at_s": 0.0,
             "body": {**q["body"], "user": f"fixed.{j:05d}", "max_tokens": window_steps, "ignore_eos": True}}
            for j, q in enumerate(varied(port, model, n, rate=1.0, prompt_tokens=prompt_tokens, max_tokens=window_steps, seed=seed))]


def run_varied(a: argparse.Namespace, *, clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
               sleep: Callable[[float], None] = time.sleep) -> int:
    lo, _, hi = a.prompt_range.partition(":")
    if a.fixed:
        reqs = fixed(a.port, a.model, a.fixed, window_steps=a.window_steps, prompt_tokens=(int(lo), int(hi)), seed=a.seed)
    else:
        reqs = varied(a.port, a.model, a.varied, rate=a.rate, prompt_tokens=(int(lo), int(hi)), max_tokens=a.max_tokens, seed=a.seed)
    jitter = random.Random(f"{a.seed}/jitter")
    delays = [jitter.uniform(0, a.jitter_ms / 1000) if a.release and not a.fixed else 0.0 for _ in reqs]
    got: list = [None] * len(reqs)
    times: list[dict] = [{} for _ in reqs]
    heads = [threading.Event() for _ in reqs]
    t0, start_unix = clock(), wall()

    def go(j: int) -> None:
        if a.fixed and a.release and j == 0:
            for h in heads[1:]:
                h.wait()
        if (wait := t0 + reqs[j]["at_s"] + delays[j] - clock()) > 0:
            sleep(wait)
        headers = {"x-verity-release": str(j), "x-verity-release-run": a.release} if a.release else {}
        try:
            got[j] = post(a.port, reqs[j]["body"], headers, times[j], clock, heads[j].set)
        except Exception as e:  # noqa: BLE001  (recorded with the request)
            got[j] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            heads[j].set()

    ts = [threading.Thread(target=go, args=(j,)) for j in range(len(reqs))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    with open(a.out, "a") as fh:
        fh.writelines(json.dumps({"user": q["user"], "release": j if a.release else None, "at_s": q["at_s"], "delay_s": delays[j],
                                  "prompt": len(q["body"]["prompt"]), "max_tokens": q["body"]["max_tokens"], "response": r}) + "\n"
                      for j, (q, r) in enumerate(zip(reqs, got)))
    timings_path(a.out).write_text(json.dumps(timings(reqs, got, times, t0, release=a.release, start_unix=start_unix), indent=1))
    print(f"[serve-load] {'fixed' if a.fixed else 'varied'}: {sum('error' not in (r or {}) for r in got)}/{len(got)} ok, prompts "
          f"{min(len(q['body']['prompt']) for q in reqs)}…{max(len(q['body']['prompt']) for q in reqs)} tokens", flush=True)
    return 0 if all("error" not in (r or {}) for r in got) else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bursts", type=int, default=8)
    ap.add_argument("--per-burst", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--idle-s", type=float, default=2.0, help="the pause after a burst, for the window to close")
    ap.add_argument("--varied", type=int, default=0, help="N varied requests instead of the bursts")
    ap.add_argument("--rate", type=float, default=2.0, help="varied: arrivals a second")
    ap.add_argument("--fixed", type=int, default=0, help="N requests of one full window of fixed length instead")
    ap.add_argument("--window-steps", type=int, default=16, help="fixed: the window's steps S, each request's max_tokens")
    ap.add_argument("--prompt-range", default="16:1024", help="varied or fixed: LO:HI, the prompts' lengths in tokens")
    ap.add_argument("--max-tokens", type=int, default=64, help="varied: the largest max_tokens drawn")
    ap.add_argument("--release", default=None,
                    help="varied or fixed: number the requests in arrival order under this release run id")
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="varied, with --release: send each up to this late")
    ap.add_argument("--out", required=True)
    return ap


def main() -> int:
    ap = parser()
    a = ap.parse_args()
    if a.varied and a.fixed:
        ap.error("--varied or --fixed, not both")
    if a.varied or a.fixed:
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
