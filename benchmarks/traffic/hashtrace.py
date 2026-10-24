"""Rules 2 and 3 on a prefix-hash trace (Mooncake's format: per request ``input_length``, ``output_length`` and
``hash_ids``, one id per 512-token block, equal ids meaning equal prefixes up to that block).

- **Registered prefix (rule 2).** A block is a *fork* when at least ``fork_min`` distinct full blocks follow it somewhere in
  the trace: many different contexts continue from it, which is what a shared system prompt or tool definition looks like,
  and a session's own history doesn't (it continues one way, apart from a regenerated reply). A request's registered prefix
  ends at the deepest fork on its path. Block granularity: up to 511 tokens of a shared prompt that end mid-block count as
  unregistered.
- **Sessions (rule 3).** Request B continues the earlier request A when A's last full block lies on B's path deeper than
  B's registered prefix (the latest such A at the deepest such block). B's new tokens are then
  ``B.input - A.input - A.output``: what B adds beyond A's context and A's reply. When B holds less than A's whole reply
  (a reply cut or replaced in history, or a parallel call), ``B.input - A.input`` is used instead, which can only overcount.
  A request with no such A starts a session, with new = input - registered. A session that began before the trace window
  starts with its whole context counted as new (overcounts).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BLOCK = 512


def read(path: Path) -> list[dict]:
    return read_rows([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def read_rows(rows: list[dict]) -> list[dict]:
    for i, r in enumerate(rows):
        r.setdefault("timestamp", i)
        r.setdefault("output_length", 0)
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def forks(rows: list[dict], fork_min: int) -> set[int]:
    full_children: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        h = r["hash_ids"]
        nfull = r["input_length"] // BLOCK
        for i in range(1, min(nfull, len(h))):
            full_children[h[i - 1]].add(h[i])
    return {node for node, ch in full_children.items() if len(ch) >= fork_min}


def analyze(rows: list[dict], fork_min: int = 8) -> dict[str, np.ndarray]:
    fk = forks(rows, fork_min)
    n = len(rows)
    ctx = np.array([r["input_length"] for r in rows], dtype=np.int64)
    out = np.array([r["output_length"] for r in rows], dtype=np.int64)
    reg = np.zeros(n, dtype=np.int64)
    new = np.zeros(n, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    capped = 0
    end_at: dict[int, int] = {}  # last full block -> latest request ending there
    for j, r in enumerate(rows):
        h = r["hash_ids"]
        depth = -1
        for i, x in enumerate(h):
            if x in fk:
                depth = i
        reg[j] = min(ctx[j], BLOCK * (depth + 1))
        a = -1
        for i in range(len(h) - 1, depth, -1):
            if h[i] in end_at:
                a = end_at[h[i]]
                break
        if a >= 0:
            parent[j] = a
            d = ctx[j] - ctx[a] - out[a]
            if d < 0:
                capped += 1
                d = max(0, ctx[j] - ctx[a])
            new[j] = min(d, ctx[j] - reg[j]) if ctx[j] >= reg[j] else 0
        else:
            new[j] = ctx[j] - reg[j]
        nfull = ctx[j] // BLOCK
        if nfull >= 1 and nfull <= len(h):
            end_at[h[nfull - 1]] = j
    sess = np.arange(n, dtype=np.int64)
    turn = np.zeros(n, dtype=np.int64)
    for j in range(n):
        if parent[j] >= 0:
            sess[j] = sess[parent[j]]
            turn[j] = turn[parent[j]] + 1
    return {"sess": sess, "turn": turn, "ctx": ctx, "reg": reg, "new": new, "out": out,
            "stats": {"requests": n, "forks": len(fk), "linked": int((parent >= 0).sum()), "reply_not_held": capped,
                      "sessions": int(len(np.unique(sess))), "fork_min": fork_min}}
