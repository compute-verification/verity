"""The generators: each a pure function of (seed, parameters) to records, by name and version in `GENERATORS`."""
from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any


def fixed_window_completions(seed: int, p: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``count`` `/v1/completions` bodies of one window of ``window_steps`` steps: each prompt a chat head of ``heads``
    (in the endpoint's tokens, BOS first), then ``filler``, cut so that ``tail`` ends it, to a length log-uniform in
    ``prompt_tokens`` (LO, HI); greedy, ``ignore_eos``, ``max_tokens`` ``window_steps``, streamed, a seed of its own."""
    lo, hi = p["prompt_tokens"]
    heads, tail, filler = p["heads"], p["tail"], p["filler"]
    rng, seeds, out = random.Random(seed), random.Random(f"{seed}/seeds"), []
    for j in range(p["count"]):
        rng.expovariate(1.0)  # the draws are those of the varied load, whose arrival times a fixed window drops
        length = round(lo * (hi / lo) ** rng.random())
        prompt = ((heads[rng.randrange(len(heads))] + filler)[:max(0, length - len(tail))] + tail)[:length]
        rng.randint(1, p["window_steps"])
        user = f"fixed.{j:05d}"
        out.append({"model": p["model"], "user": user, "prompt": prompt, "max_tokens": p["window_steps"],
                    "temperature": 0.0, "top_p": 1.0, "seed": seeds.randrange(1 << 32), "stop": [], "stream": True,
                    "return_token_ids": True, "ignore_eos": True})
    return out


GENERATORS: dict[tuple[str, int], Callable[[int, Mapping[str, Any]], list[Any]]] = {
    ("fixed-window-completions", 1): fixed_window_completions,
}
