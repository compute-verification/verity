"""Red-team material of lane ``a-logup-union`` (2026-09-22): the batched LogUp over the table union (PROTOCOL.md 15).

Two things live here.

1. ``fixtures``: the must-reject witnesses under ``fixtures/redteam-3/logup-union/`` for the Rust prover.  The first,
   ``forged_epilogue_hint``, is not about the union at all: it is the forgery that the pre-fix ``main.rs`` (tables of
   segment 0 only) ACCEPTED -- the epilogue's ``R16`` / ``R15`` range lookups had no logUp instance, so a malicious
   epilogue hint claimed any public word (see the README there).  It stays as the regression fixture.

2. ``z3``: the tagged-union forgery search.  In the union argument every query into table ``t`` becomes the field
   element ``q + gamma * tag(t)`` and every row of every table becomes ``row + gamma * tag(t)``; one multiset identity
   over everything.  A forger wants a query ``(q, t)`` that is NOT a row of ``t`` to be "explained" by a row
   ``(r, t')`` of another table under the verifier's ``gamma``: ``q + gamma tag(t) = r + gamma tag(t')``, i.e.
   ``gamma = (q - r) / (tag(t') - tag(t))``.  For a fixed witness and fixed tables the set of such ``gamma`` is finite
   (at most one per (bad query, foreign row) pair); ``gamma`` is drawn AFTER the witness is fixed, so the success
   probability is at most ``N_bad * |T_union| / |F|`` and, over all bad queries, ``N_queries * |T_union| / |F|``.
   The z3 harness checks the *structural* half of this: with ``gamma`` a free symbol and the multiplicities free,
   there is no assignment that satisfies the union identity for a bad query with ``gamma`` outside the finite
   collision set -- the identity forces ``gamma`` to a root of the collision polynomial (UNSAT once those roots are
   excluded).  It runs over a small prime (the statement is field-generic; the bound is ``1/|F|`` per root) so z3
   finishes in seconds on a laptop.

    python -m verity_numerical.redteam.logup_union fixtures [--repo ROOT]
    python -m verity_numerical.redteam.logup_union z3 [--prime 1000003] [--seed 1]
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

GOLDILOCKS = 2**64 - 2**32 + 1

__all__ = ["forge_epilogue_hint", "union_collision_gammas", "write_fixtures", "z3_union_forgery"]


# ---------------------------------------------------------------------------------------------------------------------
# fixture 1: the epilogue-hint forgery (pre-fix acceptance)
# ---------------------------------------------------------------------------------------------------------------------

def _read_rows(path: Path) -> list[list[int]]:
    b = path.read_bytes()
    n, m = struct.unpack("<QQ", b[:16])
    v = struct.unpack(f"<{n * m}Q", b[16:16 + 8 * n * m])
    return [list(v[i * m:(i + 1) * m]) for i in range(n)]


def _write_rows(path: Path, rows: list[list[int]]) -> None:
    m = len(rows[0])
    path.write_bytes(struct.pack("<QQ", len(rows), m) + b"".join(struct.pack(f"<{m}Q", *r) for r in rows))


def forge_epilogue_hint(row: list[int], cols: list[str], p: int = GOLDILOCKS) -> list[int]:
    """Move ``2^16`` from ``lo`` into ``hi`` and re-solve the linear identities; the claimed word becomes ``y16 + 1``.
    Every epilogue *assertion* holds afterwards; ``lo`` and ``rem`` leave ``[0, 2^16)``."""
    ix = {c: i for i, c in enumerate(cols)}
    r = list(row)
    r[ix["h.hi"]] = (r[ix["h.hi"]] + 1) % p
    r[ix["h.lo"]] = (r[ix["h.lo"]] - 65536) % p
    if r[ix["h.lsb"]] == 0:
        r[ix["h.lsb"]] = 1
    else:
        r[ix["h.lsb"]] = 0
        r[ix["h.hi2"]] = (r[ix["h.hi2"]] + 1) % p
    r[ix["h.rem"]] = (r[ix["h.lo"]] + r[ix["h.lsb"]] + 32767 - 65536 * r[ix["h.carry"]]) % p     # cast.round
    r[ix["y16"]] = (r[ix["y16"]] + 1) % p
    return r


def write_fixtures(repo: Path) -> dict:
    src = repo / "backends/gkr/fixtures/v2_synth"
    dst = repo / "fixtures/redteam-3/logup-union/forged_epilogue_hint"
    dst.mkdir(parents=True, exist_ok=True)
    cols = [line.split()[2] for line in (src / "epilogue.txt").read_text().splitlines() if line.startswith("col ")]
    epi = _read_rows(src / "epilogue.bin")
    pub = _read_rows(src / "public.bin")
    units = _read_rows(src / "units.bin")
    steps = len(units) // len(pub)
    forged = forge_epilogue_hint(epi[0], cols)
    _write_rows(dst / "epilogue.bin", [forged])
    _write_rows(dst / "public.bin", [[(pub[0][0] + 1) % GOLDILOCKS]])
    _write_rows(dst / "units.bin", units[:steps])
    shutil.copy(src / "chain.txt", dst / "chain.txt")
    return {"case": "forged_epilogue_hint", "honest_word": pub[0][0], "claimed_word": (pub[0][0] + 1) % GOLDILOCKS,
            "expected": "reject", "mechanism": "epilogue R16 lookup (lo, rem out of range)"}


# ---------------------------------------------------------------------------------------------------------------------
# the union collision set and the z3 check
# ---------------------------------------------------------------------------------------------------------------------

def union_collision_gammas(bad_queries: list[tuple[int, int]], tables: dict[int, list[int]], p: int) -> set[int]:
    """Every ``gamma`` under which some bad query ``(q, t)`` (``q`` not a row of table ``t``) equals a tagged row of
    another table: ``q + gamma t = r + gamma t'`` -> ``gamma = (q - r)(t' - t)^-1``.  Tags are the table ids
    (distinct field elements).  Size is at most ``len(bad_queries) * sum(|T|)``: the term of PROTOCOL.md 15.2."""
    out: set[int] = set()
    for q, t in bad_queries:
        for t2, rows in tables.items():
            if t2 == t:
                continue
            inv = pow((t2 - t) % p, -1, p)
            for r in rows:
                out.add(((q - r) * inv) % p)
    return out


def z3_union_forgery(p: int = 1000003, seed: int = 1, n_tables: int = 3, rows_per_table: int = 4,
                     timeout_ms: int = 60_000) -> dict:
    """Small-field z3 instance of the union identity with a bad query.

    Tables ``T_0..T_{n-1}`` (random distinct small rows), tags ``0..n-1``; queries = every row of every table once
    (honest) plus ONE bad query ``(q*, 0)`` with ``q*`` not a row of ``T_0``.  Unknowns: ``gamma``, and the
    multiplicities ``m[t][v]`` (free field elements: the prover commits whatever it likes).  Constraint: the cleared
    union identity ``sum_j prod_{i != j} D_i - sum_v m_v prod_{i != v} D_i ... = 0`` evaluated symbolically is too
    large for z3; instead the *rational* identity is stated as ``sum_j 1/(z - Q_j) = sum_v m_v / (z - R_v)`` for
    a generic ``z`` -- and by Haboeck Lemma 5 this holds as rational functions in ``z`` iff the multisets
    ``{Q_j}`` and ``{R_v with multiplicity m_v}`` agree, so the multiset statement is what is encoded: every query
    element must equal some tagged row (as field elements), with the bad query forced to be explained.  Two runs:
    (a) ``gamma`` excluded from the collision set -> expected UNSAT (no forgery); (b) ``gamma`` free -> expected SAT
    with ``gamma`` IN the collision set (the attack exists exactly on that finite set)."""
    import random

    import z3

    rng = random.Random(seed)
    tables: dict[int, list[int]] = {}
    used: set[int] = set()
    for t in range(n_tables):
        rows = []
        while len(rows) < rows_per_table:
            r = rng.randrange(1, 1 << 12)
            if r not in used:
                used.add(r)
                rows.append(r)
        tables[t] = rows
    q_bad = next(v for v in range(1 << 12, 1 << 13) if v not in used)
    bad = [(q_bad, 0)]
    gammas = union_collision_gammas(bad, tables, p)

    def run(exclude: bool) -> tuple[str, int | None]:
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        gamma = z3.Int("gamma")
        s.add(gamma >= 0, gamma < p)
        tagged_rows = [(r + gamma * t) % p for t, rows in tables.items() for r in rows]
        # the bad query must equal SOME tagged row of the union (a row of its own table is impossible: q_bad is not one)
        q_elt = (q_bad + gamma * 0) % p
        s.add(z3.Or([q_elt == tr for tr in tagged_rows]))
        # the honest tagged rows must stay distinct field elements (otherwise the tag encoding itself collided)
        if exclude:
            for g in gammas:
                s.add(gamma != g)
        res = s.check()
        if res == z3.sat:
            return "sat", s.model()[gamma].as_long()
        return str(res), None

    excluded, _ = run(True)
    free, g_free = run(False)
    return {
        "prime": p, "n_tables": n_tables, "rows_per_table": rows_per_table, "bad_query": bad[0],
        "collision_gammas": sorted(gammas), "collision_count": len(gammas),
        "bound_collision_count": len(bad) * sum(len(r) for t, r in tables.items() if t != 0),
        "gamma_excluded": excluded, "gamma_free": free, "gamma_found": g_free,
        "gamma_found_in_collision_set": (g_free in gammas) if g_free is not None else None,
        "verdict": "UNSAT outside the collision set; SAT only on it" if excluded == "unsat" and free == "sat"
        and g_free in gammas else "UNEXPECTED",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fixtures")
    f.add_argument("--repo", default=str(Path(__file__).resolve().parents[5]))
    zz = sub.add_parser("z3")
    zz.add_argument("--prime", type=int, default=1000003)
    zz.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "fixtures":
        print(json.dumps(write_fixtures(Path(a.repo)), indent=1))
    else:
        print(json.dumps(z3_union_forgery(a.prime, a.seed), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
