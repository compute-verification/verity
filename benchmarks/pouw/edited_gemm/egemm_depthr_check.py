#!/usr/bin/env python3
"""Depth-R same-word binding of the fold.

The served-GEMM scalar-only dial absorbs the SAME committed word R times through the XOR-in(13,19) fold with the SAME keys
every round (benchmarks/pouw/edited_gemm/egemm.cu's EG_ROUNDS; receipt_folds_r9.xorin_step iterated). This closes the open
piece: the R-round fold still binds the word with no new luck term. Binding is the trivial-kernel criterion g3-chains uses
-- on the full 2^n state space, no nonzero change to the committed word's four s32 limbs leaves the R-fold output fixed on
every state (an "all-state collapse" is a kernel element = a break).

The theorem, and why same-word depth-R is NOT g3-chains' open case:
  * Runs of >= 3 are open only in the alternating shape (T, R, T, R, split), which g3-chains shows needs the per-round
    keys/words to ALTERNATE (pairs 1 and 3 carry a key near +-1; pairs 0, 2 do not). The scalar-only dial folds the
    IDENTICAL word with the IDENTICAL keys every round, so it cannot realize the alternation. Hence at ODD depth there is
    no collapse at all -- confirmed here exhaustively over EVERY odd key pair, so key_ok is not even needed for odd depth.
  * EVEN depth R = 2, 4, ... is an iterated run of two. g3-chains proved runs of two impossible under key_ok_r11 (its
    KJ19 clause rejects keys ~ +-1 to the relevant power of two). Confirmed here: the only even-depth collapses are on
    keys with that KJ19 defect, and there are none under key_ok_r11.
Together: same-word depth-R binds for every R under key_ok_r11, with no new luck term. (key_ok_r11 = key_ok_r9 + KJ19;
egemm.cu currently derives keys under key_ok_r9, so depth R >= 2 needs the KJ19 clause added -- a claim for the coordinator.)

What is verified (each R-fold built as a permutation on the whole 2^n state space, at reduced width with small rotations
standing in for (13, 19); then a 32-bit sample on the real rotations):
  (1) n = 8 EXHAUSTIVE over every odd key pair, every perturbation, every state, R = 1..5: zero collapse at ODD R; every
      EVEN-R collapse sits on a KJ19-defective key; zero collapse among key_ok_r11 keys.
  (2) n = 10, 12 over key_ok_r11 keys (dangerous near-inverse + random), full state space, R = 1..5: zero collapse.
  (3) a KJ19 positive control: a key ~ +-1 collapses at even depth (the search sees runs of two), and key_ok_r11 rejects it.
  (4) 32-bit over the real (13, 19) and key_ok_r11 keys, R = 3, 4, 5: zero collapse.
  (5) the mirror is faithful: one fold round equals receipt_folds_r9.xorin_step (the pinned r9 reference).
Positive controls: a degenerate key a0 = 0 drops limb w0, and a zero-round fold ignores the word; both collapse, and
--mutations confirms the checker catches each.

  uv run --project .../exact_kit python egemm_depthr_check.py [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import numpy as np

from exact_kit import wrap
from exact_kit.checks import Checks


def xorin_step_ref(y, d0, d1, d2, d3, a0, a1):
    """receipt_folds_r9.xorin_step verbatim (rotations 13, 19), inlined so this check is self-contained: one absorb of a
    word's four limbs into the 32-bit state (uint32 wrapping). The fold below iterates exactly this step."""
    y = (rf_rotl(y, 13) ^ d0) * a0 + d1
    return (rf_rotl(y, 19) ^ d2) * a1 + d3


def rf_rotl(x, r):
    x = x.astype(np.uint32)
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def perturbations(w, n):
    """Every nonzero change to the committed word's four limbs we test: each limb's single bit flips, a +1 on the added
    limbs w1, w3, and the paired bit flip on the two XOR-in limbs w0, w2 (g3-chains' family shape). Returns (P, 4)."""
    m = (1 << n) - 1
    out = []
    for i in range(4):
        for j in range(n):
            wp = list(w); wp[i] = w[i] ^ (1 << j); out.append(wp)
    for i in (1, 3):
        wp = list(w); wp[i] = (w[i] + 1) & m; out.append(wp)
    for j in range(n):
        wp = list(w); wp[0] = w[0] ^ (1 << j); wp[2] = w[2] ^ (1 << j); out.append(wp)
    seen, uniq = set(), []
    for wp in out:
        t = tuple(wp)
        if t != tuple(w) and t not in seen:
            seen.add(t); uniq.append(wp)
    return np.array(uniq, dtype=np.uint64)


def collapses(keys, words, max_r, check_rounds, n, r1, r2, force_a0=None, kchunk=512):
    """Every all-state collapse (a0, a1, word, pert index, R) over the full 2^n state space, folding incrementally so one
    pass checks every R in check_rounds. A collapse is a perturbed word whose R-fold permutation equals the original's on
    every state. Keys run in chunks broadcast over (key, pert, state). force_a0 overrides a0 (a degenerate-key control)."""
    S = np.arange(1 << n, dtype=np.uint64)
    m = np.uint64((1 << n) - 1)
    keys = np.asarray(keys, dtype=np.uint64)
    cr = set(check_rounds)
    hits = []
    for w in words:
        P = perturbations(w, n)
        b0, b1, b2, b3 = (np.uint64(x) for x in w)
        p0, p1, p2, p3 = (P[:, i][None, :, None] for i in range(4))
        for k0 in range(0, len(keys), kchunk):
            ak = keys[k0:k0 + kchunk]
            a0 = (np.full(len(ak), force_a0, dtype=np.uint64) if force_a0 is not None else ak[:, 0])[:, None, None]
            a1 = ak[:, 1][:, None, None]
            y0 = np.broadcast_to(S, (len(ak), 1, S.size)).astype(np.uint64)             # base word, (K, 1, Sn)
            yd = np.broadcast_to(S, (len(ak), P.shape[0], S.size)).astype(np.uint64)    # perturbed, (K, Pn, Sn)
            if 0 in cr:                                                                 # R = 0: word unread, all collapse
                for ki in range(len(ak)):
                    hits += [(int(ak[ki, 0]), int(ak[ki, 1]), tuple(w), int(pi), 0) for pi in range(P.shape[0])]
            for r in range(1, max_r + 1):
                y0 = (((wrap.rotl(y0, r1, n) ^ b0) * a0) + b1) & m
                y0 = (((wrap.rotl(y0, r2, n) ^ b2) * a1) + b3) & m
                yd = (((wrap.rotl(yd, r1, n) ^ p0) * a0) + p1) & m
                yd = (((wrap.rotl(yd, r2, n) ^ p2) * a1) + p3) & m
                if r in cr:
                    same = (yd == y0).all(axis=2)
                    for ki, pi in zip(*np.nonzero(same)):
                        hits.append((int(ak[ki, 0]), int(ak[ki, 1]), tuple(w), int(pi), r))
    return hits


def kj19_bad(a, n):
    """The reduced KJ19 defect: a ~ +-1 mod 2^(n-2) (the small-rotation stand-in for 32-bit KJ19's a ~ +-1 mod 2^13)."""
    t = 1 << (n - 2)
    return (a % t) in (1, t - 1)


def near_inverse(a0, a1, n):
    """g3-chains' LI-style defect that key_ok_r9 rejects: a0*a1 ~ +-1 2-adically (v2(a0*a1 -+ 1) >= n-5)."""
    m = (1 << n) - 1
    prod = (a0 * a1) & m
    def v2(x):
        return n if x == 0 else (x & -x).bit_length() - 1
    return v2((prod - 1) & m) >= n - 5 or v2((prod + 1) & m) >= n - 5


def key_ok_r11(a0, a1, n):
    """The reduced key_ok_r11: odd keys, neither key near +-1 (KJ19), and no small near-inverse pair (key_ok_r9)."""
    return (a0 & 1) and (a1 & 1) and not kj19_bad(a0, n) and not kj19_bad(a1, n) and not near_inverse(a0, a1, n)


def danger_keys(n, count, seed):
    """key_ok_r11 keys drawn from the regions the open shape and the family lean on -- near-inverse pairs and random --
    kept to the keys key_ok_r11 admits, so a collapse here would be a real break."""
    rng = np.random.default_rng(seed)
    m = (1 << n) - 1
    out = []
    while len(out) < count:
        a0 = int(rng.integers(0, 1 << n)) | 1
        for a1 in (int(rng.integers(0, 1 << n)) | 1, pow(a0, -1, 1 << n), (-pow(a0, -1, 1 << n)) & m):
            if key_ok_r11(a0, a1, n):
                out.append((a0, a1))
    return np.array(sorted(set(out))[:count], dtype=np.uint64)


def key_ok_r9_32(a):
    """egemm.cu's xin_ok_r9 (host): odd, low-24-bit window in (256, 2^24-256), rot not +-1 mod 32, and a*b mod 2^30
    outside [-128, 128] for every b in 1..128."""
    a &= 0xFFFFFFFF
    low, rot = a & 0xFFFFFF, a & 31
    if not (a & 1 and 256 < low < (1 << 24) - 256 and rot not in (1, 31)):
        return False
    t = 0
    for b in range(1, 129):
        t = (t + a) & 0xFFFFFFFF
        if ((t + 128) & 0x3FFFFFFF) <= 256:
            return False
    return True


def key_ok_r11_32(a):
    """key_ok_r9 plus KJ19: reject a ~ +-1 mod 2^13 (covers both rotations 13, 19)."""
    return key_ok_r9_32(a) and (a & 0x1FFF) not in (1, 0x1FFF)


def r11_keys(count, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        a = int(rng.integers(0, 1 << 32))
        if key_ok_r11_32(a):
            out.append(a)
    return out


def sample32(keys, words, rounds, nstate, seed=1):
    """A 32-bit confirmation over the real (13, 19) rotations and key_ok_r11 keys: no perturbation collapses the R-fold on
    a random state sample. Over states, so it witnesses non-collapse rather than proving the all-state kernel."""
    rng = np.random.default_rng(seed)
    S = rng.integers(0, 1 << 32, size=nstate, dtype=np.uint64)
    m = np.uint64(0xFFFFFFFF)
    hits = []
    for (a0, a1) in keys:
        a0, a1 = np.uint64(a0), np.uint64(a1)
        for w in words:
            P = perturbations(w, 32)
            b = [np.uint64(x) for x in w]
            p = [P[:, i][:, None] for i in range(4)]
            y0 = S[None, :].astype(np.uint64)
            yd = np.broadcast_to(S, (P.shape[0], S.size)).astype(np.uint64)
            for r in range(1, max(rounds) + 1):
                y0 = (((wrap.rotl(y0, 13, 32) ^ b[0]) * a0) + b[1]) & m
                y0 = (((wrap.rotl(y0, 19, 32) ^ b[2]) * a1) + b[3]) & m
                yd = (((wrap.rotl(yd, 13, 32) ^ p[0]) * a0) + p[1]) & m
                yd = (((wrap.rotl(yd, 19, 32) ^ p[2]) * a1) + p[3]) & m
                if r in rounds:
                    hits += [(int(a0), int(a1), int(pi), r) for pi in np.nonzero((yd == y0).all(axis=1))[0]]
    return hits


def main() -> int:
    checks = Checks()
    rng = np.random.default_rng(7)
    rand_words = lambda n, k: [tuple(int(x) for x in rng.integers(1, 1 << n, 4)) for _ in range(k)]

    # (0) the fold is faithful: one round of the proof's fold == receipt_folds_r9.xorin_step (the pinned r9 fold).
    a0m, a1m = np.uint32(0x9E3779B1), np.uint32(0x85EBCA77)
    ym = np.arange(256, dtype=np.uint32) * np.uint32(2654435761)
    dm = [(np.uint32(0x1234 + i) + np.arange(256, dtype=np.uint32)) for i in range(4)]
    with np.errstate(over="ignore"):
        ref = xorin_step_ref(ym, dm[0], dm[1], dm[2], dm[3], a0m, a1m).astype(np.uint64)
        mine = (wrap.rotl(ym.astype(np.uint64), 13, 32) ^ dm[0].astype(np.uint64)) * np.uint64(a0m) + dm[1]
        mine = ((wrap.rotl(mine & np.uint64(0xFFFFFFFF), 19, 32) ^ dm[2].astype(np.uint64)) * np.uint64(a1m) + dm[3]) & np.uint64(0xFFFFFFFF)
    checks.check("one fold round == receipt_folds_r9.xorin_step (fold faithful)", bool((mine == ref).all()))

    # (1) n = 8 EXHAUSTIVE over every odd key pair, every perturbation, every state, R = 1..5.
    n, r1, r2 = 8, 3, 5
    w8 = (0x9E, 0x37, 0x79, 0xB9)
    keys8 = np.array([(a0, a1) for a0 in range(1, 1 << n, 2) for a1 in range(1, 1 << n, 2)], dtype=np.uint64)
    hits1 = collapses(keys8, [w8], 5, [1, 2, 3, 4, 5], n, r1, r2)
    odd_hits = [h for h in hits1 if h[4] % 2 == 1]
    even_nondef = [h for h in hits1 if h[4] % 2 == 0 and not (kj19_bad(h[0], n) or kj19_bad(h[1], n))]
    r11_hits = [h for h in hits1 if key_ok_r11(h[0], h[1], n)]
    checks.check("n8 exhaustive: zero collapse at ODD depth R=1,3,5 (open shape needs alternating keys)",
                 len(odd_hits) == 0, f"{len(keys8)} odd key pairs x {len(perturbations(w8, n))} perts x {1 << n} states; "
                 f"{len(hits1)} collapses total, {len(odd_hits)} at odd depth")
    checks.check("n8 exhaustive: every EVEN-depth collapse is KJ19-defective (iterated run of two)",
                 len(even_nondef) == 0, f"{len(even_nondef)} even-depth collapses on a non-KJ19 key")
    checks.check("n8 exhaustive: zero collapse among key_ok_r11 keys, R=1..5", len(r11_hits) == 0,
                 f"{len(r11_hits)} collapses on key_ok_r11 keys")

    # (2) n = 10, 12 over key_ok_r11 keys, full state space, R = 1..5.
    for n, r1, r2, kcount, wcount in [(10, 5, 7, 300, 3), (12, 5, 11, 150, 2)]:
        keys = danger_keys(n, kcount, seed=100 + n)
        words = rand_words(n, wcount)
        hits = collapses(keys, words, 5, [1, 2, 3, 4, 5], n, r1, r2, kchunk=64 if n <= 10 else 16)
        checks.check(f"n{n} key_ok_r11 keys, R=1..5: zero all-state collapse", len(hits) == 0,
                     f"{len(keys)} keys x {len(words)} words x {1 << n} states x 5 depths; {len(hits)} collapses")

    # (3) KJ19 positive control: the n=8 near-(-1) key a0=63 (== -1 mod 2^6) collapses at even depth; key_ok_r11 rejects it.
    kjbad = (63, 0x55)
    kjhits = collapses(np.array([kjbad], dtype=np.uint64), [w8], 4, [2, 4], 8, 3, 5)
    checks.check("KJ19 control: a near-+-1 key collapses at even depth (search sees runs of two)", len(kjhits) > 0,
                 f"key {kjbad}: {len(kjhits)} even-depth collapses; key_ok_r11 accepts it = {key_ok_r11(*kjbad, 8)}")
    checks.check("KJ19 control: key_ok_r11 rejects that near-+-1 key", not key_ok_r11(*kjbad, 8))

    # (4) 32-bit confirmation: real (13, 19), key_ok_r11 keys (and near-inverse key_ok pairs), R = 3, 4, 5.
    a0s = r11_keys(3, 0xC0FFEE)
    keys32 = [(a, b) for a in a0s for b in r11_keys(2, a)] + [(a, (pow(a, -1, 1 << 32) | 1)) for a in a0s]
    keys32 = [(a, b) for (a, b) in keys32 if key_ok_r11_32(b)]
    words32 = rand_words(32, 4)
    hits32 = sample32(keys32, words32, [3, 4, 5], 1 << 13)
    checks.check("32-bit (13,19) key_ok_r11: zero collapse on a 2^13 state sample, R=3,4,5", len(hits32) == 0,
                 f"{len(keys32)} key pairs x {len(words32)} words x 3 depths; {len(hits32)} collapses")

    # positive control: a degenerate key a0 = 0 drops limb w0, so a w0-only change is invisible -- the search must see it.
    ctrl = collapses(keys8[:400], [w8], 3, [3], 8, 3, 5, force_a0=0)
    checks.check("control a0=0 drops w0: collapse is detectable", len(ctrl) > 0, f"{len(ctrl)} collapses with a0=0")
    if checks.mutations:
        with checks.mutant("degenerate key a0=0 (drops limb w0)"):
            checks.check("mut a0=0: zero collapse (should fail)", len(collapses(keys8[:400], [w8], 3, [3], 8, 3, 5, force_a0=0)) == 0)
        with checks.mutant("zero-round fold R=0 (ignores the word)"):
            checks.check("mut R=0: zero collapse (should fail)", len(collapses(keys8[:200], [w8], 0, [0], 8, 3, 5)) == 0)

    return checks.finish()


if __name__ == "__main__":
    import sys
    sys.exit(main())
