"""Does a receipt fold at the fan-in floor bind its words? (CMP-118, for BET-FOLD CMP-128)

A 3-input instruction reads the chain state and at most two words, so a fold costs at least ½ lane instruction a word
when every word is read by one (O-BITS section 9, BET-BANDWIDTH's ISA-FANIN). The obvious fold at that floor alternates
IMAD y ← y·w_a + w_b with LOP3 y ← y ⊕ w_c ⊕ w_d, the multiplier a data word, no keys. main's li
(`segment._li_absorb`) instead spends an input on a key, y ← ((y ⊕ w_a ⊕ K)·A + w_b) with A odd, one instruction a word.

The check: flip one bit of the first word of a 32-word window, fold both windows, and count how often the receipt is
unchanged (a collision). Words are uniform 32-bit, as at A = B = 0 under fresh noise. An even multiplier shifts the
state's top bits out, so a high-bit change survives only while every later multiplier is odd.

    python3 benchmarks/pouw/accel_score/fold_floor.py
"""
from __future__ import annotations

import random

M32 = (1 << 32) - 1


def floor_fold(ws: list[int]) -> int:
    y = 0x6A09E667
    for i in range(0, len(ws), 4):
        y = (y * ws[i] + ws[i + 1]) & M32
        y ^= ws[i + 2] ^ ws[i + 3]
    return y


def li_fold(ws: list[int], keys: list[tuple[int, int]]) -> int:
    y = 0x6A09E667
    for i in range(0, len(ws), 2):
        k, a = keys[i // 2]
        y = (((y ^ ws[i] ^ k) * a) + ws[i + 1]) & M32
    return y


def collisions(fold, bit: int, word: int, trials: int, rng: random.Random) -> float:
    hits = 0
    for _ in range(trials):
        ws = [rng.getrandbits(32) for _ in range(32)]
        ws2 = list(ws)
        ws2[word] ^= 1 << bit
        hits += fold(ws) == fold(ws2)
    return hits / trials


def main() -> None:
    rng = random.Random(118)
    keys = [(rng.getrandbits(32), rng.getrandbits(32) | 1) for _ in range(16)]
    trials = 20000
    print(f"collision rate when one bit of a word in a 32-word window flips ({trials} windows each)")
    print(f"{'flipped':24s} {'floor fold (½ instr/word)':>26s} {'li (1 instr/word)':>18s}")
    for word, bit in [(0, 31), (0, 16), (0, 0), (1, 31), (1, 0), (2, 31), (29, 31), (30, 31)]:
        f = collisions(floor_fold, bit, word, trials, rng)
        g = collisions(lambda w: li_fold(w, keys), bit, word, trials, rng)
        print(f"word {word:2d} bit {bit:2d}{'':10s} {f:26.4f} {g:18.4f}")


if __name__ == "__main__":
    main()
