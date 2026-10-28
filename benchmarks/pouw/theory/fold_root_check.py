#!/usr/bin/env python3
"""FOLD-R: the same-word R-fold's whole all-state kernel K_R by commuting, and K_R = K_1 = P/X/N under the scaled KJ19.

`fold_root_kernel.c` finds K_R from the fact that F_{w'}^R = F_w^R forces F_{w'} to commute with F_w^R, a meet in the
middle with 2^(2n) half-round maps a side (`fold_kernel.c` enumerates all 2^(4n) words). This checks:
  (1) at n = 6 it finds exactly what fold_kernel.c finds, at every (key, R), over every odd key pair (--quick: a sample)
      and two words, apart from the near-±1 keys where Φ^R is so structured that the join is cut (it says so);
  (2) it reproduces the three known counterexamples at n = 8 and 10;
  (3) `fold_root_sweep.kj_scaled` read at n = 32 with r2 = 19 is receipt_folds.kj19, the 32-bit rule of record, and
      `kj_last` is egemm_depthr_check.key_ok_r11_32's KJ19 clause;
  (4) at n = 8 over every odd key pair (--quick: a sample), one random word each, every pair where K_R = K_1 = P/X/N fails
      for some R <= 6 has a key ≡ ±1 mod 2^r1, which KJ19's last clause rejects; so at n = 8 that clause alone, with no
      pair clause, admits 4,096 pairs and K_R = K_1 = P/X/N holds at every one;
  (5) the exception no key rule removes: when 2·d1 ≡ a0 − 1 and 2·d3 ≡ a1 − 1 (mod 2^n), both half-rounds commute with
      bitwise NOT, and w with d0 complemented has F_{w'} = NOT ∘ F_w, so it fixes every even fold. At n = 8 and 10 under
      the scaled KJ that orbit is all depth adds (|K_even| = 32), and at 32 bits it holds at key_ok_r11 keys;
  (6) at n = 10, K_R for R = 1, 2, 3 is the part of the 2R-pair chain group that keeps the word the same every round;
  (7) the enumerator's -p and -c settle a join cut at a structured Φ^R (two of the filtered n = 10 sweep's cut jobs).
Mutations: a brute-force comparison against a different word is caught; a kernel that drops P1's doubling is caught;
a NOT-symmetric word with d1 one off keeps K_2 = K_1.

  uv run python benchmarks/pouw/theory/fold_root_check.py [--quick] [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import collections
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fold_root_sweep as sw
from exact_kit.checks import Checks

THREADS = str(min(8, len(os.sched_getaffinity(0))))


def build(tmp: Path, src: str = "fold_root_kernel.c", edit: tuple[str, str] | None = None) -> Path:
    """Compile src into tmp; edit = (old, new) compiles a copy with that one line replaced."""
    path, exe = HERE / src, tmp / Path(src).stem
    if edit:
        text = path.read_text()
        assert text.count(edit[0]) == 1, edit[0]
        path, exe = tmp / f"mut_{src}", tmp / f"mut_{exe.name}"
        path.write_text(text.replace(*edit))
    subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(path)], check=True)
    return exe


def root_kernels(exe: Path, n: int, r1: int, r2: int, rmax: int, jobs, flags=()) -> tuple[dict, dict]:
    """({job: [|K_1|, .., pxn, g3bad, degenerate]}, {(job, R): set of w' != w in K_R}) for jobs (a0, a1, *w). The
    enumerator's K lines name only the keys, so each call gets jobs with distinct key pairs."""
    summ, elems, pending = {}, collections.defaultdict(set), [tuple(j) for j in dict.fromkeys(map(tuple, jobs))]
    while pending:
        batch, rest = {}, []
        for j in pending:
            (rest.append(j) if j[:2] in batch else batch.__setitem__(j[:2], j))
        inp = "".join(" ".join(map(str, j)) + "\n" for j in batch.values())
        out = subprocess.run([str(exe), "-a", *flags, str(n), str(r1), str(r2), str(rmax)], input=inp, capture_output=True,
                             text=True, check=True, env=dict(os.environ, OMP_NUM_THREADS=THREADS)).stdout
        for p in map(str.split, out.splitlines()):
            if p[0] == "S":
                summ[tuple(map(int, p[1:7]))] = list(map(int, p[7:]))
            elif p[0] == "K":
                elems[(batch[(int(p[1]), int(p[2]))], int(p[3]))].add(tuple(map(int, p[4:8])))
        pending = rest
    return summ, elems


def brute_kernels(exe: Path, n: int, r1: int, r2: int, rmax: int, word, keys) -> dict:
    """fold_kernel.c's {(a0, a1, R): set of w' != w in K_R}."""
    out = subprocess.run([str(exe), str(n), str(r1), str(r2), str(rmax), *map(str, word)],
                         input="".join(f"{a} {b}\n" for a, b in keys), capture_output=True, text=True, check=True,
                         env=dict(os.environ, OMP_NUM_THREADS=THREADS)).stdout
    elems = collections.defaultdict(set)
    for p in map(str.split, out.splitlines()):
        if p[0] == "K":
            elems[(int(p[1]), int(p[2]), int(p[3]))].add(tuple(map(int, p[4:8])))
    return elems


def agree(root, brute, word, keys, rmax, degenerate) -> list:
    return [(a, b, r) for a, b in keys if (a, b) not in degenerate for r in range(1, rmax + 1)
            if root[((a, b) + tuple(word), r)] != brute[(a, b, r)]]


def good(row, rmax) -> bool:
    ks, pxn, g3bad, degen = row[:rmax], row[rmax], row[rmax + 1], row[rmax + 2]
    return bool(pxn and all(k == ks[0] for k in ks) and g3bad == 0 and degen == 0)


def main() -> int:
    checks = Checks()
    quick = "--quick" in checks.args
    rng = random.Random(20261017)
    sys.path.insert(0, str(HERE.parents[0] / "receipts"))
    sys.path.insert(0, str(HERE.parents[0] / "edited_gemm"))
    import egemm_depthr_check as fold_ref
    import receipt_folds as rf

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        exe, brute = build(tmp), build(tmp, "fold_kernel.c")

        # (1) n = 6 against the brute force
        n, r1, r2, rmax = 6, 2, 4, 5
        odd = list(range(1, 1 << n, 2))
        keys = [(a, b) for a in odd for b in odd]
        if quick:
            keys = rng.sample(keys, 96)
        for word in [(0x1E, 0x37, 0x39, 0x2B), (32, 45, 3, 59)]:
            summ, root = root_kernels(exe, n, r1, r2, rmax, [k + word for k in keys])
            ref = brute_kernels(brute, n, r1, r2, rmax, word, keys)
            degen = {(a, b) for (a, b, *_), row in summ.items() if row[rmax + 2]}
            bad = agree(root, ref, word, keys, rmax, degen)
            checks.check(f"n6 word {word}: K_R by commuting == fold_kernel.c's brute force at every (key, R), R = 1..{rmax}",
                         not bad, f"{len(keys)} keys, {sum(len(ref[(a, b, r)]) for a, b in keys if (a, b) not in degen for r in range(1, rmax + 1))} "
                         f"elements; {len(degen)} keys cut short; mismatches {bad[:3]}")
            checks.check(f"n6 word {word}: every key where the join is cut has a key ≡ ±1 mod 2^r1 (KJ19's last clause)",
                         all(not (sw.kj_last(a, r1) and sw.kj_last(b, r1)) for a, b in degen), f"{sorted(degen)[:6]}")
            if checks.mutations:
                other = tuple((x + 1) & 63 for x in word)
                ref2 = brute_kernels(brute, n, r1, r2, rmax, other, keys[:24])
                checks.mutation(f"n6 word {word}: compared against the brute force at another word",
                                agree(root, ref2, word, keys[:24], rmax, degen))

        # (2) the three known counterexamples
        s8, _ = root_kernels(exe, 8, 3, 5, 6, [(97, 179, 255, 48, 87, 75), (83, 225, 144, 166, 44, 160)])
        big, even = s8[(97, 179, 255, 48, 87, 75)], s8[(83, 225, 144, 166, 44, 160)]
        checks.check("n8 (97, 179), word (255, 48, 87, 75): |K_R| = 64 at every R <= 6, 48 elements break (G3), K_1 isn't P/X/N",
                     big[:6] == [64] * 6 and big[6] == 0 and big[7] == 48, str(big))
        checks.check("n8 (83, 225), word (144, 166, 44, 160): K_1 = P/X/N (16), |K_2| = 32, |K_4| = 64, |K_6| = 32, odd R adds none",
                     even[:6] == [16, 32, 16, 64, 16, 32] and even[6] == 1 and even[7] == 0, str(even))
        s10, _ = root_kernels(exe, 10, 4, 6, 3, [(321, 411, 827, 208, 250, 366)])
        checks.check("n10 (321, 411), word (827, 208, 250, 366): |K_1| = 32", s10[(321, 411, 827, 208, 250, 366)][0] == 32,
                     str(s10[(321, 411, 827, 208, 250, 366)]))
        checks.check("each of the three pairs has a key ≡ ±1 mod 2^r1, so KJ19's last clause rejects it",
                     not any(sw.kj_last(a, r) and sw.kj_last(b, r) for a, b, r in [(97, 179, 3), (83, 225, 3), (321, 411, 4)]))
        if checks.mutations:
            mexe = build(tmp, edit=("for (int p = 0; p < 4; p++)", "for (int p = 0; p < 2; p++)"))
            m8, _ = root_kernels(mexe, 8, 3, 5, 6, [(83, 225, 144, 166, 44, 160)])
            checks.mutation("a kernel that drops P1's doubling of each match", m8[(83, 225, 144, 166, 44, 160)][:6] != even[:6],
                            str(m8[(83, 225, 144, 166, 44, 160)][:6]))

        # (3) the scaling at 32 bits
        sample = [rng.randrange(1 << 32) | 1 for _ in range(20000)]
        sample += [(s + (k << e)) & 0xFFFFFFFF for e in range(12, 31) for k in range(1, 9) for s in (1, -1, 3, -3)]
        sample = [a | 1 for a in sample]
        checks.check("kj_scaled at n = 32, r2 = 19 is receipt_folds.kj19 (the 32-bit KJ19 of record)",
                     all(sw.kj_scaled(a, 32, 19) == rf.kj19(a) for a in sample),
                     f"{len(sample)} odd keys, {sum(not rf.kj19(a) for a in sample)} rejected")
        checks.check("kj_last at r1 = 13 is egemm_depthr_check.key_ok_r11_32's KJ19 clause (a ≢ ±1 mod 2^13)",
                     all(sw.kj_last(a, 13) == ((a & 0x1FFF) not in (1, 0x1FFF)) for a in sample))
        checks.check("kj19 implies kj_last at 32 bits", all(sw.kj_last(a, 13) for a in sample if rf.kj19(a)))
        checks.check("egemm_depthr_check's reduced near-inverse clause has no 32-bit counterpart in key_ok_r11: a pair with "
                     "a0 a1 ≡ 1 mod 2^27 passes key_ok_r11 at both keys",
                     any(rf.key_ok_r11(a) and rf.key_ok_r11(pow(a, -1, 1 << 32)) for a in sample[:2000]))

        # (4) n = 8, every odd key pair, one random word each
        n, (r1, r2), rmax = 8, sw.ROT[8], 6
        odd = list(range(1, 1 << n, 2))
        keys = [(a, b) for a in odd for b in odd]
        if quick:
            keys = rng.sample(keys, 2048) + [(97, 179), (83, 225)]
        jobs = [k + tuple(rng.randrange(1 << n) for _ in range(4)) for k in keys]
        summ, _ = root_kernels(exe, n, r1, r2, rmax, jobs)
        badk = [j for j, row in summ.items() if not good(row, rmax)]
        admitted = [j for j in summ if sw.kj_last(j[0], r1) and sw.kj_last(j[1], r1)]
        checks.check(f"n8, {len(keys)} key pairs, a random word each: K_R = K_1 = P/X/N fails somewhere (the search sees it)",
                     len(badk) > 0, f"{len(badk)} pairs")
        checks.check(f"n8: every pair where K_R = K_1 = P/X/N fails for some R <= {rmax} has a key ≡ ±1 mod 2^{r1}",
                     all(not (sw.kj_last(a, r1) and sw.kj_last(b, r1)) for a, b, *_ in badk),
                     f"{len(admitted)} pairs pass KJ19's last clause at both keys, "
                     f"{sum(not good(summ[j], rmax) for j in admitted)} of them fail; "
                     f"{sum(sw.near_inverse(a, b, n) for a, b, *_ in admitted)} of the admitted are near-inverse pairs")

        # (5) the NOT-symmetric words open K_2 at every key
        for n in (8, 10):
            r1, r2 = sw.ROT[n]
            ok = [a for a in range(1, 1 << n, 2) if sw.kj_scaled(a, n, r2)]
            jobs = sw.jobs_for("symmetric", n, rng, 6 if quick else 40, 0)
            settle = ("-p", str(64 // n), "-c", "28")
            summ, elems = root_kernels(exe, n, r1, r2, 6, jobs, settle)
            twins = all(elems[(j, 2)] - elems[(j, 1)] >=
                        {((j[2] ^ ((1 << n) - 1)), j[3], j[4], j[5])} for j in jobs)
            checks.check(f"n{n}, {len(jobs)} NOT-symmetric words at pairs passing the scaled KJ ({len(ok)} keys): "
                         "|K_1| = |K_odd| = 16 = P/X/N, |K_even| = 32, and the added words include w with d0 complemented",
                         all(summ[j] == [16, 32, 16, 32, 16, 32, 1, 0, 0] for j in jobs) and twins,
                         str(sorted({tuple(summ[j]) for j in jobs})))
            if checks.mutations:
                off = [j[:3] + ((j[3] + 1) % (1 << n),) + j[4:] for j in jobs]
                ms, _ = root_kernels(exe, n, r1, r2, 6, off, settle)
                checks.mutation(f"n{n}: the same words with d1 one off (2 d1 ≠ a0 − 1)",
                                not all(ms[j] == [16, 32, 16, 32, 16, 32, 1, 0, 0] for j in off),
                                str(sorted({tuple(ms[j]) for j in off})))
        fails = []
        for _ in range(8 if quick else 64):
            while True:
                a0, a1 = rng.randrange(1 << 32) | 1, rng.randrange(1 << 32) | 1
                if rf.key_ok_r11(a0) and rf.key_ok_r11(a1):
                    break
            h = rng.choice((0, 1 << 31)), rng.choice((0, 1 << 31))
            w = (rng.randrange(1 << 32), ((a0 - 1) // 2 + h[0]) % (1 << 32), rng.randrange(1 << 32),
                 ((a1 - 1) // 2 + h[1]) % (1 << 32))
            w2 = (w[0] ^ 0xFFFFFFFF,) + w[1:]
            for y in (rng.randrange(1 << 32) for _ in range(256)):
                p, q = fold32(y, w, a0, a1), fold32(y, w2, a0, a1)
                if q != p ^ 0xFFFFFFFF or fold32(q, w2, a0, a1) != fold32(p, w, a0, a1):
                    fails.append((a0, a1, w, y))
                    break
        checks.check("32 bits, rotations (13, 19), keys passing key_ok_r11, NOT-symmetric words: F_{w'} = NOT ∘ F_w and "
                     "F_{w'}^2 = F_w^2 with w' = w with d0 complemented, so K_2 ⊋ K_1 at every key",
                     not fails, f"{8 if quick else 64} key pairs × 256 states; failures {fails[:2]}")

        # (6) K_R is the part of the 2R-pair chain group that keeps the word the same every round
        n, (r1, r2) = 10, sw.ROT[10]
        ok = [a for a in range(1, 1 << n, 2) if sw.kj_scaled(a, n, r2)]
        jobs = []
        for _ in range(3 if quick else 12):
            a0, a1 = rng.choice(ok), rng.choice(ok)
            jobs.append((a0, a1) + tuple(rng.randrange(1 << n) for _ in range(4)))
            jobs.append((a0, a1, rng.randrange(1 << n), (a0 - 1) // 2, rng.randrange(1 << n), (a1 - 1) // 2))
        summ, elems = root_kernels(exe, n, r1, r2, 3, jobs)
        bad = [(j, r) for j in jobs for r in (1, 2, 3)
               if elems[(j, r)] | {j[2:]} != chain_group_orbit(n, j[0], j[1], j[2:], r)]
        if checks.mutations:
            off = [(j, r) for j in jobs for r in (1, 2, 3)
                   if elems[(j, r)] | {j[2:]} != chain_group_orbit(n, j[0], j[1], j[2:], r, drop_n=True)]
        checks.check(f"n10, {len(jobs)} words (half NOT-symmetric) at pairs passing the scaled KJ: K_R for R = 1, 2, 3 is "
                     "exactly the set of words w' whose R-fold repetition is w's under an element of the 2R-pair chain "
                     "group (P_p, X_p, N_p)", not bad, f"mismatches {bad[:2]}")
        if checks.mutations:
            checks.mutation("the chain group without its N_p generators", off, f"{len(off)} mismatches")

        # (7) a join cut at a structured Φ^R settles with more probe states and a higher cap
        cut = [(77, 219, 0, 0, 0, 0), (409, 681, 1023, 1023, 1023, 1023)]
        before, _ = root_kernels(exe, n, r1, r2, 6, cut)
        after, _ = root_kernels(exe, n, r1, r2, 6, cut, ("-p", "6", "-c", "28"))
        checks.check("n10, two structured words at scaled-KJ pairs that the filtered sweep cut: cut with 3 probes and "
                     "cap 2^16", all(before[j][-1] == 1 for j in cut), str([before[j] for j in cut]))
        checks.check("the same with 6 probes and cap 2^28: uncut, and K_R = K_1 = P/X/N for R = 1..6",
                     all(good(after[j], 6) and after[j][0] == 16 for j in cut), str([after[j] for j in cut]))
    return checks.finish()


def chain_group_orbit(n: int, a0: int, a1: int, w, R: int, drop_n: bool = False) -> set:
    """The words w' such that w' repeated R times is w repeated R times moved by an element of the 2R-pair chain group:
    P_p = (c_p ⊕ t, z_p + t), X_p = (z_p + t, c_(p+1) ⊕ rotl(t, r_(p+1))), N_p = (¬c_p, a_p − 1 − z_p, ¬c_(p+1))."""
    r1, r2 = sw.ROT[n]
    m, t, S = (1 << n) - 1, 1 << (n - 1), 2 * R
    rot = lambda x, r: ((x << r) | (x >> (n - r))) & m
    gens = [("P", p) for p in range(S)] + [(k, p) for p in range(S - 1) for k in ("X",) + (() if drop_n else ("N",))]
    out = set()
    for bits in range(1 << len(gens)):
        L = [[w[0], w[1]] if p % 2 == 0 else [w[2], w[3]] for p in range(S)]
        for i, (k, p) in enumerate(gens):
            if not bits >> i & 1:
                continue
            if k == "P":
                L[p][0] ^= t
                L[p][1] = (L[p][1] + t) & m
            elif k == "X":
                L[p][1] = (L[p][1] + t) & m
                L[p + 1][0] ^= rot(t, r1 if (p + 1) % 2 == 0 else r2)
            else:
                L[p][0] ^= m
                L[p][1] = ((a0 if p % 2 == 0 else a1) - 1 - L[p][1]) & m
                L[p + 1][0] ^= m
        if all(L[p] == L[p % 2] for p in range(S)):
            out.add((L[0][0], L[0][1], L[1][0], L[1][1]))
    return out


def fold32(y: int, w, a0: int, a1: int) -> int:
    """One round of the XOR-in(13, 19) fold at 32 bits."""
    m = 0xFFFFFFFF
    y = ((((y << 13) | (y >> 19)) & m ^ w[0]) * a0 + w[1]) & m
    return ((((y << 19) | (y >> 13)) & m ^ w[2]) * a1 + w[3]) & m


if __name__ == "__main__":
    sys.exit(main())
