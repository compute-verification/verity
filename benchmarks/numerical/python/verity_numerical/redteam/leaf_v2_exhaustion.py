"""leaf/v2 randomizer exhaustion: what a verifier learns from the SECOND proof against the same commitment.

Status 2026-09-22 (lane soundness-fixes): the *systematic leak* this module demonstrated (payload words read in the
clear from the first session's openings, ``payload_words_in_clear`` > 0) is FIXED -- the codeword now lives on the
coset ``31 * H_n``, disjoint from the message domain, so ``payload_words_in_clear`` is 0 up to 2^-15 coincidences.
The *exhaustion* finding below (a payload-equality oracle once the union of opened columns exceeds ``k - l``) is
unchanged and stays true for a long-lived commitment; its fix is re-randomisation per session (a protocol-owner
decision, LEAF_V2.md 6).

LEAF_V2.md 6 states the budget: the union of every column set ``Q`` ever opened against one chunk must stay
``<= k - l`` (256 at the campaign's code; ``t = 191`` per proof, so one proof per commitment).  This module makes
the consequence of exceeding it concrete, at the reference implementation's reduced code (``n=1024, k=256, l=192``:
64 randomizers, ``t = 48`` per session):

* a row's opened symbols at ``u`` distinct columns are ``u`` linear equations in the ``k`` message symbols
  (``l`` payload words + ``k - l`` randomizers) with public coefficients (the RS encoding matrix);
* with ``u <= k - l`` any payload is consistent with the openings (the randomizer block has full row rank: hiding,
  AHIV22 Lemma 4.15);
* with ``u > k - l`` exactly ``u - (k - l)`` independent linear relations on the payload survive: a verifier holding
  a *candidate* payload row (a guessed token's activations, a public weight row) decides it with false-positive
  probability ``p^-(u - (k - l))`` -- a payload-equality oracle after two sessions, and full recovery (the whole
  message is determined) once ``u >= k``.

The demonstration solves the system over BabyBear by Gaussian elimination (rank test); it needs no proof machinery,
only the committed chunk's column openings the verifier legitimately receives in each session.  The table
``sessions_table`` projects the same count to the campaign parameters (``n = 16384, k - l = 256, t = 191``).

    python -m verity_numerical.redteam.leaf_v2_exhaustion [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys

from verity_numerical.reference.leaf_v2 import (
    BABYBEAR,
    Code,
    commit_tensor,
    coset_ntt,
    intt,
)

__all__ = ["demo", "encoding_matrix", "rank_mod_p", "sessions_table"]

P = BABYBEAR


def encoding_matrix(code: Code) -> list[list[int]]:
    """``M[c][j]``: codeword symbol ``c`` of the unit message ``e_j`` (RS: message on ``H_k``, codeword on the coset
    ``shift * H_n``; before the 2026-09-22 fix the codeword lived on ``H_n`` itself and ``M`` had unit rows)."""
    k, n = code.k, code.n
    cols = []
    for j in range(k):
        e = [0] * k
        e[j] = 1
        # message on the k-subgroup -> coefficients -> evaluations on the codeword domain
        cols.append(coset_ntt(intt(e) + [0] * (n - k), code.shift))
    return [[cols[j][c] for j in range(k)] for c in range(n)]


def rank_mod_p(rows: list[list[int]]) -> int:
    m = [r[:] for r in rows]
    rank = 0
    ncols = len(m[0]) if m else 0
    for col in range(ncols):
        piv = next((i for i in range(rank, len(m)) if m[i][col] % P), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        inv = pow(m[rank][col], P - 2, P)
        m[rank] = [x * inv % P for x in m[rank]]
        for i in range(len(m)):
            if i != rank and m[i][col] % P:
                f = m[i][col]
                m[i] = [(a - f * b) % P for a, b in zip(m[i], m[rank])]
        rank += 1
    return rank


def consistent(M: list[list[int]], code: Code, opened: dict[int, int], candidate: list[int]) -> bool:
    """Do randomizers exist that make ``candidate`` (payload) encode to the opened symbols at the opened columns?"""
    l = code.l
    rows = []
    for c, o in sorted(opened.items()):
        a_r = M[c][l:]
        b = (o - sum(M[c][j] * candidate[j] for j in range(l))) % P
        rows.append(a_r + [b])
    r_aug = rank_mod_p(rows)
    r_a = rank_mod_p([r[:-1] for r in rows])
    return r_aug == r_a


def demo(seed: int = 7) -> dict:
    rng = random.Random(seed)
    code = Code(n=1024, k=256, l=192, R=4)
    t = 48
    words = [rng.randrange(2**16) for _ in range(code.l)]
    tensor = commit_tensor("model.layers.0.w", "bf16", [1, code.l], words, code, seed=b"\x01" * 32)
    row = tensor.tableau_row(0)
    M = encoding_matrix(code)
    true_payload = tensor.payload_rows[0]
    other = true_payload[:]
    other[5] ^= 1  # one word, one bit off
    out: dict = {"code": code.text, "t": t, "randomizers": code.randomizers, "sessions": []}
    opened: dict[int, int] = {}
    for s in range(1, 7):
        Q = rng.sample(range(code.n), t)
        for c in Q:
            opened[c] = row[c]
        u = len(opened)
        # the former systematic positions: column (n/k) j (j < l) WAS payload word j before the coset fix; count how
        # many opened columns still equal the payload word the old map would have exposed (chance only: ~2^-15 each)
        stride = code.n // code.k
        clear = {c // stride: row[c] for c in opened if c % stride == 0 and c // stride < code.l and row[c] == true_payload[c // stride]}
        a_r = [M[c][code.l:] for c in sorted(opened)]
        rk = rank_mod_p(a_r)
        ok_true = consistent(M, code, opened, true_payload)
        ok_other = consistent(M, code, opened, other)
        # payload relations that survive: u - rank(randomizer block)
        out["sessions"].append({
            "session": s, "union_columns": u, "randomizer_rank": rk, "payload_relations": max(0, u - rk),
            "payload_words_in_clear": len(clear), "clear_words_verified_equal_payload": len(clear) > 0,
            "true_payload_consistent": ok_true, "one_bit_off_payload_consistent": ok_other,
            "oracle": (ok_true and not ok_other),
        })
        if u >= code.k:
            break
    return out


def sessions_table(n: int = 16384, k: int = 4096, l: int = 3840, t: int = 191, word_bits: int = 16, max_sessions: int = 30) -> list[dict]:
    """Expected union of ``m`` independent uniform ``t``-subsets of ``n`` columns and the surviving payload relations."""
    rows = []
    r = k - l
    for m in range(1, max_sessions + 1):
        u = n * (1 - (1 - t / n) ** m)
        rel = max(0.0, u - r)
        rows.append({
            "sessions": m, "expected_union": round(u, 1), "payload_relations_F_p": round(rel, 1),
            "leaked_bits_per_row": round(rel * math.log2(BABYBEAR), 1), "payload_bits_per_row": l * word_bits,
            "fraction": round(rel * math.log2(BABYBEAR) / (l * word_bits), 4), "full_recovery": u >= k,
        })
        if u >= k:
            break
    return rows


def clear_words_table(n: int = 16384, k: int = 4096, l: int = 3840, t: int = 191, R: int = 64, fixed: bool = True) -> dict:
    """Systematic leak at the campaign code BEFORE the coset fix: opened column ``(n/k) j`` with ``j < l`` was payload
    word ``j`` verbatim, for every one of the chunk's ``R`` rows, whatever the randomizers.  With ``fixed`` (the
    current code: codeword on ``31 H_n``, disjoint from ``H_k``) no column is a message symbol and the row is 0."""
    frac = l / n
    per_row = 0.0 if fixed else t * frac
    return {"t": t, "encoding": "coset 31*H_n (non-systematic)" if fixed else "H_n (systematic)",
            "systematic_fraction_of_columns": 0.0 if fixed else frac, "expected_clear_words_per_row_per_proof": round(per_row, 2),
            "expected_clear_words_per_chunk_per_proof": round(per_row * R, 1),
            "prob_no_clear_word_in_a_proof": 1.0 if fixed else (1 - frac) ** t, "budget_irrelevant": not fixed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    d = demo()
    d["campaign_projection"] = sessions_table()
    d["campaign_clear_words"] = clear_words_table()
    d["campaign_clear_words_before_fix"] = clear_words_table(fixed=False)
    for s in d["sessions"]:
        print(f"session {s['session']}: union {s['union_columns']:4} cols, randomizer rank {s['randomizer_rank']:3}, payload relations {s['payload_relations']:3}, "
              f"words in clear {s['payload_words_in_clear']:3}, "
              f"true consistent={s['true_payload_consistent']}, one-bit-off consistent={s['one_bit_off_payload_consistent']} -> oracle={s['oracle']}")
    for r in d["campaign_projection"][:6] + d["campaign_projection"][-1:]:
        print(f"campaign: {r['sessions']:2} sessions -> union {r['expected_union']:8}, {r['payload_relations_F_p']:7} F_p relations = {r['leaked_bits_per_row']:8} bits of {r['payload_bits_per_row']} per row ({r['fraction']:.1%}); full={r['full_recovery']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(d, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
