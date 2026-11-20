"""The generators: each a pure function of (seed, parameters) to records, by name and version in `GENERATORS`."""
from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any


def varied_completions(seed: int, p: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``count`` records ``{"at_s", "body"}``: a `/v1/completions` body and its arrival in seconds from the start, the gaps
    exponential at ``rate`` a second, drawn between the body's other draws. Each prompt a chat head of ``heads`` (in the
    endpoint's tokens, BOS first), then ``filler``, cut so that ``tail`` ends it, to a length log-uniform in
    ``prompt_tokens`` (LO, HI); ``max_tokens`` uniform in 1 … ``max_tokens``, greedy, streamed, a seed of its own."""
    lo, hi = p["prompt_tokens"]
    heads, tail, filler = p["heads"], p["tail"], p["filler"]
    rng, seeds, at, out = random.Random(seed), random.Random(f"{seed}/seeds"), 0.0, []
    for j in range(p["count"]):
        at += rng.expovariate(p["rate"])
        length = round(lo * (hi / lo) ** rng.random())
        prompt = ((heads[rng.randrange(len(heads))] + filler)[:max(0, length - len(tail))] + tail)[:length]
        out.append({"at_s": at, "body": {"model": p["model"], "user": f"varied.{j:05d}", "prompt": prompt,
                                         "max_tokens": rng.randint(1, p["max_tokens"]), "temperature": 0.0, "top_p": 1.0,
                                         "seed": seeds.randrange(1 << 32), "stop": [], "stream": True,
                                         "return_token_ids": True}})
    return out


def fixed_window_completions(seed: int, p: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``count`` `/v1/completions` bodies of one window of ``window_steps`` steps: the varied load's bodies at rate 1 and
    ``max_tokens`` ``window_steps`` (their arrival times dropped), each with ``ignore_eos`` and ``max_tokens``
    ``window_steps``."""
    steps = p["window_steps"]
    return [{**r["body"], "user": f"fixed.{j:05d}", "max_tokens": steps, "ignore_eos": True}
            for j, r in enumerate(varied_completions(seed, {**p, "rate": 1.0, "max_tokens": steps}))]


GENERATORS: dict[tuple[str, int], Callable[[int, Mapping[str, Any]], list[Any]]] = {
    ("fixed-window-completions", 1): fixed_window_completions,
    ("varied-completions", 1): varied_completions,
}
