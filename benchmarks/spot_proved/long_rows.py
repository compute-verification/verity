"""A replay's comparison with its serve, split at DSA's long rows: the rows attending to more than ``--keys`` (2,048) keys,
whose top-k keeps only some of them, and the rest.

    long_rows.py --tree TREE --arm-dir ARM --load NAME --replay OUT/replay [--keys 2048] [--out FILE]

TREE holds ``benchmarks/serve_tax/replay.py``, whose `window` reads the served window from the record arm. A row
``(request, first position, tokens)`` attends to ``first position + tokens`` keys at its last token, and output token i of a
request was sampled from the row attending to ``len(prompt) + i`` keys. For each side of the split it counts the served
steps (a step is long when any of its rows is), the rows and the sampled tokens, and how many of each the replay
(``replay_steps.jsonl``, ``replay_tokens.jsonl.gz``) reproduced. Requests the serve aborted are listed apart: the client
received none of their tokens, so only their rows are compared. It also gives the served rows' longest keys, decode and
prefill, and how many rows pass each power of two from 16,384 keys.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def split(w: dict, mine_steps: dict[int, list], mine_tokens: dict[str, list[int]], keys: int) -> dict:
    served = {s["step"]: s["rows"] for s in w["steps"]}
    aborted = {rid for _, rid in w.get("aborts", ())}
    side = lambda long: "long" if long else "short"  # noqa: E731
    res: dict = {"keys_past": keys, **{k: {"steps": 0, "steps_differing": 0, "rows": 0, "rows_differing": 0, "tokens": 0,
                                            "tokens_differing": 0, "requests": 0, "requests_differing": 0} for k in ("long", "short")}}
    for step in sorted(set(served) | set(mine_steps)):
        a, b = served.get(step, []), mine_steps.get(step, [])
        long = any(r[1] + r[2] > keys for r in a + b)
        c = res[side(long)]
        c["steps"] += 1
        c["steps_differing"] += sorted(map(tuple, a)) != sorted(map(tuple, b))
        for r in set(map(tuple, a)) | set(map(tuple, b)):
            res[side(r[1] + r[2] > keys)]["rows"] += 1
            res[side(r[1] + r[2] > keys)]["rows_differing"] += (list(r) in a) != (list(r) in b)
    for q in w["requests"]:
        if q["engine_id"] in aborted:
            continue
        want, got, n = q["served"], mine_tokens.get(q["engine_id"], []), len(q["prompt_ids"])
        differ = {"long": False, "short": False}
        for i in range(max(len(want), len(got))):
            s = side(n + i > keys)
            res[s]["tokens"] += 1
            if i >= len(want) or i >= len(got) or want[i] != got[i]:
                res[s]["tokens_differing"] += 1
                differ[s] = True
        for s in ("long", "short"):
            if any(side(n + i > keys) == s for i in range(len(want))):
                res[s]["requests"] += 1
                res[s]["requests_differing"] += differ[s]
    res["aborted"] = [{"request": rid, "after_step": step, "prompt": len(next(q["prompt_ids"] for q in w["requests"] if q["engine_id"] == rid)),
                       "rows": sum(r[0] == rid for s in served.values() for r in s),
                       "rows_reproduced": sum(r[0] == rid and r in mine_steps.get(st, []) for st, s in served.items() for r in s)}
                      for step, rid in w.get("aborts", ())]
    dec = sorted(r[1] + r[2] for s in served.values() for r in s if r[2] == 1)
    pre = sorted(r[1] + r[2] for s in served.values() for r in s if r[2] > 1)
    full = [len(q["prompt_ids"]) + len(q["served"]) - 1 for q in w["requests"] if q["engine_id"] not in aborted and q["served"]]
    res["served"] = {"decode_rows": len(dec), "decode_max_keys": dec[-1] if dec else None,
                     "prefill_rows": len(pre), "prefill_max_keys": pre[-1] if pre else None,
                     "longest_request_decoded_to_keys": max(full) if full else None,
                     "decode_rows_past": {k: sum(x > k for x in dec) for k in (keys, 16384, 32768, 65536, 131072)},
                     "prefill_rows_past": {k: sum(x > k for x in pre) for k in (keys, 16384, 32768, 65536, 131072)}}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--load", required=True)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--keys", type=int, default=2048)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.tree / "benchmarks" / "serve_tax"))
    import replay

    w = replay.window(a.arm_dir, a.load, a.rep)
    steps = {s["step"]: s["rows"] for s in map(json.loads, (a.replay / "replay_steps.jsonl").read_text().splitlines()) if s}
    with gzip.open(a.replay / "replay_tokens.jsonl.gz", "rt") as f:
        tokens = {r["request"]: r["tokens"] for r in map(json.loads, f) if r}
    res = split(w, steps, tokens, a.keys)
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
