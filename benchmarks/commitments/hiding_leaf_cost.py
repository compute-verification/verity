#!/usr/bin/env python3
"""Per-row cost of ``hm96-sha256/v1`` hiding leaves against today's unsalted leaves: stored bytes, hashed bytes, native
compressions, in-circuit AND and XOR gates, and measured single-core host time per row.

    python benchmarks/commitments/hiding_leaf_cost.py                      # rows of 4, 256, 1536, 3072, 4096, 8192 bytes
    python benchmarks/commitments/hiding_leaf_cost.py --row-bytes 3072 --rows 16384 --reps 7 --out /tmp/hlc

Leaf rules, one row per leaf:

    blake3-keyed/row/v2        today's research row leaf, BLAKE3(row, key=role key); unsalted
    vllm-v1/pos-leaf           today's serving leaf, SHA-256("verity/pos-leaf/v0" || u64be len || row); unsalted
    hm96/pos-leaf              hm96-sha256/v1 over the serving leaf (what the host committer's opt-in commits)
    hm96/sha256-row            hm96-sha256/v1 over sha256/row/v1 (a 64-byte public prefix block: the circuit-friendly inner digest)
    vllm-v1-sha512/pos-leaf    the serving leaf through SHA-512; unsalted
    sha512/row/v1              the SHA-512 row digest (a 128-byte public prefix block); unsalted
    hm96-sha512/pos-leaf       hm96-sha512/v1 over the SHA-512 serving leaf
    hm96-sha512/sha512-row     hm96-sha512/v1 over sha512/row/v1
    salted-blake3              baseline only: BLAKE3(salt || row, key) with a stored 32-byte salt; hides in the random-oracle model

In-circuit gates use the ripple-carry model (a w-bit addition is w - 1 ANDs, Ch and Maj w each): a SHA-256 compression is 64
rounds of 7 additions, Ch and Maj, 48 schedule words of 3 additions and 8 feed-forward additions (w = 32); a SHA-512 compression is
80 rounds, 64 schedule words and 8 feed-forward additions (w = 64); a BLAKE3 compression is 7 rounds of 8 G of 6 additions.  A
public prefix block is a constant midstate; every other compression is private.  hm96's XORs are those of ``b = x XOR M y`` for the
pinned key; the hashes' own XORs are not counted.  Host times run the integration's paths on one core:
``verity_vllm.commit.scheme.pos_leaf`` (under ``hashlib.sha512`` for SHA-512), ``verity.ml.windows.hiding.HidingLeaves``
(salts from the OS generator included) and the ``blake3`` package; the median of ``--reps``.

Writes ``hiding_leaf_cost.json`` and ``result.json`` (``research/result/v0.1``) to ``$RESEARCH_RUN_DIR`` or ``--out`` when either is
set; prints a table either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs.optimize import SHA512_ANDS
from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import ROLE_X, blake3_row_key, sha256_row_prefix, sha512_row_prefix
from verity_vllm.commit import scheme
from verity.ml.windows.hiding import HidingLeaves

try:
    import blake3 as _blake3
except ModuleNotFoundError:
    _blake3 = None

RULES = ("blake3-keyed/row/v2", "vllm-v1/pos-leaf", "hm96/pos-leaf", "hm96/sha256-row", "vllm-v1-sha512/pos-leaf", "sha512/row/v1",
         "hm96-sha512/pos-leaf", "hm96-sha512/sha512-row", "salted-blake3")
DEFAULT_ROW_BYTES = (4, 256, 1536, 3072, 4096, 8192)
RESULT_SCHEMA = "research/result/v0.1"
POS_LEAF_PREFIX = len(scheme.POS_LEAF_TAG) + 8
SALTED_BLAKE3_SALT = 32

ADD = 31
SHA256_ANDS = 64 * (7 * ADD + 32 + 32) + 48 * 3 * ADD + 8 * ADD
BLAKE3_ANDS = 7 * 8 * 6 * ADD
_SHA = {"sha256": (64, 9, SHA256_ANDS), "sha512": (128, 17, SHA512_ANDS)}
_HM = {"sha256": hm96.SHA256, "sha512": hm96.SHA512}


def sha_compressions(h: str, message_bytes: int) -> int:
    block, pad, _ = _SHA[h]
    return (message_bytes + pad + block - 1) // block


def blake3_compressions(message_bytes: int) -> int:
    chunks = [min(1024, message_bytes - o) for o in range(0, message_bytes, 1024)] or [0]
    return sum(max(1, -(-c // 64)) for c in chunks) + len(chunks) - 1


def _sha_rule(h: str, inner: str, hiding: bool, r: int) -> dict[str, Any]:
    """A SHA-2 leaf rule: the inner digest (``pos`` with its 26-byte prefix, or ``row`` behind a constant prefix block), wrapped in
    hm96 when ``hiding``."""

    block, _, ands = _SHA[h]
    if inner == "pos":
        n_inner, hashed = sha_compressions(h, POS_LEAF_PREFIX + r), POS_LEAF_PREFIX + r
    else:
        n_inner, hashed = sha_compressions(h, block + r) - 1, r
    zero = {"sha256": 0, "sha512": 0, "blake3": 0}
    if not hiding:
        return {**zero, "stored_bytes": 0, "hashed_bytes": hashed, h: n_inner, "ands": n_inner * ands, "xors": 0, "public_bytes": block // 2}
    inst = _HM[h]
    salt = sha_compressions(h, block + inst.salt_bytes) - 1               # after the salt prefix's constant midstate
    leaf = sha_compressions(h, block + inst.proven_bytes) - 1             # after the leaf prefix's constant midstate
    return {**zero, "stored_bytes": inst.salt_bytes, "hashed_bytes": hashed + inst.salt_bytes + inst.proven_bytes, h: n_inner + salt + leaf,
            "ands": (n_inner + salt) * ands, "xors": inst.xor_count(inst.default_key), "public_bytes": inst.proven_bytes}


_SHA_RULES = {"vllm-v1/pos-leaf": ("sha256", "pos", False), "hm96/pos-leaf": ("sha256", "pos", True), "hm96/sha256-row": ("sha256", "row", True),
              "vllm-v1-sha512/pos-leaf": ("sha512", "pos", False), "sha512/row/v1": ("sha512", "row", False),
              "hm96-sha512/pos-leaf": ("sha512", "pos", True), "hm96-sha512/sha512-row": ("sha512", "row", True)}


def analytic(rule: str, r: int) -> dict[str, Any]:
    """Stored bytes beyond the row, payload bytes hashed, native compressions, in-circuit gates and public bytes, per row."""

    if rule in _SHA_RULES:
        return _sha_rule(*_SHA_RULES[rule], r)
    if rule == "blake3-keyed/row/v2":
        n = blake3_compressions(r)
        return {"stored_bytes": 0, "hashed_bytes": r, "blake3": n, "sha256": 0, "sha512": 0, "ands": n * BLAKE3_ANDS, "xors": 0, "public_bytes": 32}
    if rule == "salted-blake3":
        n = blake3_compressions(SALTED_BLAKE3_SALT + r)
        return {"stored_bytes": SALTED_BLAKE3_SALT, "hashed_bytes": SALTED_BLAKE3_SALT + r, "blake3": n, "sha256": 0, "sha512": 0,
                "ands": n * BLAKE3_ANDS, "xors": 0, "public_bytes": 32}
    raise ValueError(f"unknown rule {rule!r}")


def _timer(fn: Callable[[], Any], reps: int) -> float:
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def measure(r: int, n: int, reps: int) -> tuple[dict[str, float | None], bool]:
    """Median host seconds per row for each rule over ``n`` rows of ``r`` bytes, and whether the batch path matched the reference."""

    data = memoryview(os.urandom(n * r))
    rows = [data[i * r:(i + 1) * r] for i in range(n)]
    hl, hl512 = HidingLeaves(), HidingLeaves(instance=hm96.SHA512)
    row_prefix = hashlib.sha256(sha256_row_prefix(ROLE_X, 8, max(r, 1)))
    row512_prefix = hashlib.sha512(sha512_row_prefix(ROLE_X, 8, max(r, 1)))

    def pos() -> list[bytes]:
        return [scheme.pos_leaf(v) for v in rows]

    def pos512() -> list[bytes]:
        return [scheme.pos_leaf(v, hashlib.sha512) for v in rows]

    def prefixed(prefix) -> list[bytes]:
        out = []
        for v in rows:
            h = prefix.copy()
            h.update(v)
            out.append(h.digest())
        return out

    out: dict[str, float | None] = {
        "vllm-v1/pos-leaf": _timer(pos, reps),
        "hm96/pos-leaf": _timer(lambda: hl.leaves(pos()), reps),
        "hm96/sha256-row": _timer(lambda: hl.leaves(prefixed(row_prefix)), reps),
        "vllm-v1-sha512/pos-leaf": _timer(pos512, reps),
        "sha512/row/v1": _timer(lambda: prefixed(row512_prefix), reps),
        "hm96-sha512/pos-leaf": _timer(lambda: hl512.leaves(pos512()), reps),
        "hm96-sha512/sha512-row": _timer(lambda: hl512.leaves(prefixed(row512_prefix)), reps),
    }
    inner, inner512 = pos(), pos512()
    out["hm96 only"] = _timer(lambda: hl.leaves(inner), reps)
    out["hm96-sha512 only"] = _timer(lambda: hl512.leaves(inner512), reps)
    if _blake3 is not None:
        key = blake3_row_key(ROLE_X)
        out["blake3-keyed/row/v2"] = _timer(lambda: [_blake3.blake3(v, key=key, max_threads=1).digest() for v in rows], reps)

        def salted() -> list[bytes]:
            salts = memoryview(os.urandom(SALTED_BLAKE3_SALT * n))
            res = []
            for i, v in enumerate(rows):
                h = _blake3.blake3(key=key, max_threads=1)
                h.update(salts[i * SALTED_BLAKE3_SALT:(i + 1) * SALTED_BLAKE3_SALT])
                h.update(v)
                res.append(h.digest())
            return res

        out["salted-blake3"] = _timer(salted, reps)
    else:
        out["blake3-keyed/row/v2"] = out["salted-blake3"] = None
    checked = True
    for h, ins in ((hl, inner), (hl512, inner512)):
        leaves, salts = h.leaves(ins)
        y = h.instance.salt_bytes
        checked &= all(leaves[i] == h.instance.tree_leaf(h.key, h.instance.commit_string(h.key, ins[i], salts[y * i:y * (i + 1)]))
                       for i in {0, n // 2, n - 1})
    return {k: (v / n if v is not None else None) for k, v in out.items()}, checked


def rows_for(r: int, n: int, reps: int) -> tuple[list[dict[str, Any]], bool]:
    per_row, ok = measure(r, n, reps)
    base = {rule: analytic(rule, r) for rule in RULES}
    out = []
    for rule in RULES:
        a = base[rule]
        row = {"rule": rule, "row_bytes": r, **a, "us_per_row": None if per_row[rule] is None else per_row[rule] * 1e6,
               "stored_over_row": a["stored_bytes"] / r,
               "ands_over_blake3_row": a["ands"] / base["blake3-keyed/row/v2"]["ands"],
               "ands_over_pos_leaf": a["ands"] / base["vllm-v1/pos-leaf"]["ands"],
               "sha256_over_pos_leaf": (a["sha256"] / base["vllm-v1/pos-leaf"]["sha256"]) if a["sha256"] else None,
               "sha512_over_pos_leaf": (a["sha512"] / base["vllm-v1-sha512/pos-leaf"]["sha512"]) if a["sha512"] else None}
        if rule.startswith("hm96/"):
            row["hm96_only_us_per_row"] = per_row["hm96 only"] * 1e6
        if rule.startswith("hm96-sha512/"):
            row["hm96_only_us_per_row"] = per_row["hm96-sha512 only"] * 1e6
        out.append(row)
    for row in out:
        t, tp, tb = row["us_per_row"], per_row["vllm-v1/pos-leaf"], per_row["blake3-keyed/row/v2"]
        row["time_over_pos_leaf"] = t / (tp * 1e6) if t is not None else None
        row["time_over_blake3_row"] = t / (tb * 1e6) if t is not None and tb else None
    return out, ok


def table(rows: list[dict[str, Any]]) -> str:
    head = ("| row B | rule | stored B | hashed B | SHA-256 | SHA-512 | BLAKE3 | ANDs | ANDs vs BLAKE3 row | XORs | µs/row (1 core) "
            "| time vs pos-leaf |")
    lines = [head, "|" + "---|" * 12]
    for x in rows:
        us = "—" if x["us_per_row"] is None else f"{x['us_per_row']:.2f}"
        tv = "—" if x["time_over_pos_leaf"] is None else f"{x['time_over_pos_leaf']:.2f}×"
        lines.append(f"| {x['row_bytes']} | {x['rule']} | {x['stored_bytes']} | {x['hashed_bytes']} | {x['sha256']} | {x['sha512']} | {x['blake3']} | "
                     f"{x['ands']:,} | {x['ands_over_blake3_row']:.2f}× | {x['xors']:,} | {us} | {tv} |")
    return "\n".join(lines)


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--row-bytes", type=int, action="append", help=f"repeatable; default {' '.join(map(str, DEFAULT_ROW_BYTES))}")
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None, help="directory for result.json (default $RESEARCH_RUN_DIR)")
    ns = ap.parse_args(argv)
    ns.row_bytes = ns.row_bytes or list(DEFAULT_ROW_BYTES)
    if ns.rows <= 0 or ns.reps <= 0 or any(r <= 0 for r in ns.row_bytes):
        ap.error("--rows, --reps and every --row-bytes must be positive")
    return ns


def main(argv: list[str] | None = None) -> int:
    ns = parse(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    rows, checks = [], []
    for r in ns.row_bytes:
        got, ok = rows_for(r, ns.rows, ns.reps)
        rows += got
        checks.append(ok)
    print(table(rows))
    out = ns.out or os.environ.get("RESEARCH_RUN_DIR")
    if out:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "hiding_leaf_cost.json").write_text(json.dumps(rows, indent=1) + "\n")
        result = {
            "schema": RESULT_SCHEMA,
            "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
            "kind": "bench-result/v1",
            "workload_fingerprint": {"workload": "hiding_leaf_cost", "rules": list(RULES), "row_bytes": ns.row_bytes, "rows": ns.rows,
                                     "reps": ns.reps, "blake3_package": getattr(_blake3, "__version__", None), "cpu_count": os.cpu_count(),
                                     "python": sys.version.split()[0]},
            "validation": {"status": "passed" if all(checks) else "failed",
                           "checker": "hm96-sha256 and hm96-sha512 batch leaves equal core's reference leaves at three rows per row size",
                           "detail": ", ".join(f"{r}={'ok' if ok else 'FAILED'}" for r, ok in zip(ns.row_bytes, checks)), "evidence": []},
            "measurements": [{"name": f"{x['rule']}.{x['row_bytes']}.{k}", "value": x[k], "unit": u}
                             for x in rows for k, u in (("us_per_row", "us"), ("ands", "AND"), ("stored_bytes", "B"), ("sha256", "compression"),
                                                        ("sha512", "compression"), ("blake3", "compression")) if x[k] is not None]
                            + [{"name": "workload_wall", "value": time.perf_counter() - started, "unit": "s"}],
            "measurement_files": ["hiding_leaf_cost.json"],
            "artifacts": [],
        }
        tmp = d / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        os.replace(tmp, d / "result.json")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
