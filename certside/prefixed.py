#!/usr/bin/env python3
"""`warden_serve.py` with shared prefixes over the heads of its id requests' prompts, so that a server with its prefix
cache on hits it (VLLM-368). The tree is TREE, else the directory above this file's when that is a tree's root (the
file as `certside/prefixed.py`, in the place of `certside/bounded.py`), else the working directory (the file sent
with a job that runs at a tree's root):

    python prefixed.py rehearse ...            # as `python benchmarks/network_traces/warden_serve.py rehearse ...`
    python prefixed.py groups TRAFFIC.json --seed S [--block N] [--out GROUPS.json]

`id_traffic`'s requests stay the seed's: the arrivals, each prompt's length, `max_tokens`, the sampling, the seeds and
every prompt id a prefix doesn't cover are what `warden_serve.py rehearse --ids` sends at the same seed. Over them, by
the trace corpus's rule (`benchmarks/network_e2e/traces/traffic.py`: PREFIXES, SHARED, MIN_BODY, read from the tree):
with probability SHARED a request is in one of len(PREFIXES) groups, group g with weight 1 / (g + 1), or, when its
prompt is shorter than that group's prefix and MIN_BODY ids of its own, in the group of the longest prefix that leaves
them (in none when none does). Group g's prefix is PREFIXES[g] ids, each uniform below 128,256, drawn once from
(seed, g); it replaces the prompt's first PREFIXES[g] ids. Every prefix is whole 16-token blocks, the unit vLLM's
prefix cache keeps, so a request of a group whose prefix the cache holds hits PREFIXES[g] / 16 blocks.

The groups' draws come from a generator of their own (the seed "prefixed:S"), two a request whatever they come to, so a
request's group is the seed's alone, whatever the count and the rate. Only the process that writes the traffic reads
this: `rehearse` runs `serve`, `client` and `check` from `warden_serve.py` itself, on the traffic.json it wrote.

groups reads a traffic.json back: each request's group (the prefix its prompt starts with), and what a cache that never
drops a block would hit: every request of a group after its first, PREFIXES[g] / 16 blocks each. It is a count over the
traffic, not a measurement: the engine's own hits are in its record. `--block N` is an engine whose block is N tokens
(GLM-5.3's is 64): a prefix then hits its whole blocks, PREFIXES[g] // N, and one shorter than a block hits none.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

ABOVE = Path(__file__).resolve().parents[1]
TREE = Path(os.environ.get("TREE") or (ABOVE if (ABOVE / "benchmarks/network_traces/warden_serve.py").is_file() else Path.cwd()))
DRAWN_BELOW = 128_256


def rule():
    """The trace corpus's traffic module, for its PREFIXES, SHARED, MIN_BODY and BLOCK."""
    path = TREE / "benchmarks/network_e2e/traces/traffic.py"
    spec = importlib.util.spec_from_file_location("corpus_traffic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if any(p % mod.BLOCK for p in mod.PREFIXES):
        raise SystemExit(f"prefixed: {path}'s PREFIXES are not whole {mod.BLOCK}-token blocks")
    return mod


def heads(seed: int, T) -> list[list[int]]:
    return [[r.randrange(DRAWN_BELOW) for _ in range(p)]
            for p, r in ((p, random.Random(f"prefixed:{seed}:{g}")) for g, p in enumerate(T.PREFIXES))]


def prefixed(traffic: list[dict], seed: int, T) -> list[dict]:
    rng, own = random.Random(f"prefixed:{seed}"), heads(seed, T)
    weights = [1 / (g + 1) for g in range(len(T.PREFIXES))]
    for item in traffic:
        prompt = item["ids"]["prompt"]
        shared, group = rng.random() < T.SHARED, rng.choices(range(len(T.PREFIXES)), weights)[0]
        if not shared:
            continue
        if T.PREFIXES[group] + T.MIN_BODY > len(prompt):
            fits = [g for g in range(len(T.PREFIXES)) if T.PREFIXES[g] + T.MIN_BODY <= len(prompt)]
            if not fits:
                continue
            group = max(fits, key=lambda g: (T.PREFIXES[g], -g))
        prompt[:T.PREFIXES[group]] = own[group]
    return traffic


def groups(traffic: list[dict], seed: int, T, block: int | None = None) -> dict:
    own, block = heads(seed, T), block or T.BLOCK
    longest = sorted(range(len(own)), key=lambda g: -len(own[g]))
    seen: dict[int, int] = {}
    rows, tokens, hit_tokens, hit_blocks, followers = [], 0, 0, 0, 0
    for i, item in enumerate(traffic):
        prompt = item["ids"]["prompt"]
        g = next((g for g in longest if prompt[:len(own[g])] == own[g]), None)
        tokens += len(prompt)
        if g is not None:
            if g in seen:
                followers += 1
                hit_tokens += len(own[g])
                hit_blocks += len(own[g]) // block
            seen[g] = seen.get(g, 0) + 1
        rows.append(g)
    n = len(traffic)
    return {"seed": seed, "requests": n, "prefixes": list(T.PREFIXES), "shared": T.SHARED, "min_body": T.MIN_BODY,
            "in_a_group": sum(g is not None for g in rows), "by_group": [seen.get(g, 0) for g in range(len(own))],
            "after_their_groups_first": followers, "share_of_requests": round(followers / n, 4) if n else None,
            "block": block, "their_prefix_blocks": hit_blocks, "their_prefix_tokens": hit_tokens, "prompt_tokens": tokens,
            "share_of_prompt_tokens": round(hit_tokens / tokens, 4) if tokens else None, "group": rows}


def main(argv: list[str]) -> int:
    T = rule()
    if argv[:1] == ["groups"]:
        ap = argparse.ArgumentParser(prog="prefixed.py groups")
        ap.add_argument("traffic", type=Path)
        ap.add_argument("--seed", type=int, required=True)
        ap.add_argument("--out", type=Path)
        ap.add_argument("--block", type=int, help="the engine's block, in tokens (16 unless given)")
        a = ap.parse_args(argv[1:])
        got = groups(json.loads(a.traffic.read_text()), a.seed, T, a.block)
        if a.out:
            a.out.write_text(json.dumps(got))
        print(json.dumps({k: v for k, v in got.items() if k != "group"}))
        return 0
    sys.path.insert(0, str(TREE / "benchmarks/network_traces"))
    import warden_serve

    drawn = warden_serve.id_traffic

    def id_traffic(*args, **kwargs) -> list[dict]:
        return prefixed(drawn(*args, **kwargs), kwargs.get("seed", 0), T)

    warden_serve.id_traffic = id_traffic
    return warden_serve.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
