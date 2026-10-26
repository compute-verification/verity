"""SKIP-FIX (CMP-137): v2's served math under SEG-COINS's coin schedule, and every route that skips work at A = B = 0.

Exact CPU reference on small shapes in `hyb.py`'s style (BET-HYBRID-2): every value is an int64 array, words are
mod 2^32, and the verifier recomputes every cell in the clear (in the protocol that recomputation is a zero-knowledge
proof on drawn cells). `python segcoins.py [--out DIR]` runs every check, prints one line each, and writes
`segcoins.json` to DIR.

~~~text
segments  lengths L_1..L_S along K (v2: all L; SEG-COINS: long ones of L, then a final one of L_f)
coins     V (m x L) on ±96, U (n x L) on ±80, a permutation π_s of the L noise columns a segment, dither bits
          b_s (m x L_s), c_s (n x L_s). v2 draws them all after A and B are committed. SEG-COINS draws the final
          segment's b_f, c_f only after the root of every long-segment word.
operands  X_s = wrap8(A_s + b_s + V[:, π_s[:L_s]]),  W_s = wrap8(B_s + c_s + U[:, π_s[:L_s]]);  wX_s, wW_s their wraps
words     T_s = Σ_{s' <= s} X_s' W_s'ᵀ mod 2^32
peel      Z_X = Σ_s X_s π_sᵀ (m x L), Y_W = Σ_s (B + c)_s π_sᵀ (n x L): Z_X Uᵀ and V Y_Wᵀ, two base-128 digits each
          (ρ = 4 products of inner size L), as `nofold.RHO_V2`
served    O = T_S − Z_X Uᵀ − V Y_Wᵀ + 256 (Σ_s wX_s (B + c)_sᵀ + Σ_s X_s wW_sᵀ) = (A + b)(B + c)ᵀ
~~~

Costs are per output cell in INT8 MAC-times: a product of inner size ℓ costs ℓ, a 0/1 product (b1 `wgmma` on the H200,
0/1 NVFP4 on sm_120) ℓ/s_b, a base-128 digit product L unless that digit is zero in every entry. Honest's kernel is
data-oblivious and pays K + 4L.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

M32 = (1 << 32) - 1
UX, UW = 96, 80                       # nofold.U_X, nofold.U_W
RHO = 4
S_B = {"H200": 8, "RTX PRO 6000": 2}  # nofold.V2_CARDS


@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    n: int
    L: int                     # the noise period, and v2's segment length
    lengths: tuple[int, ...]   # segment lengths along K
    late: int = 0              # trailing segments whose dither is drawn after the long-segment root (SEG-COINS)

    @property
    def K(self) -> int:
        return sum(self.lengths)

    @property
    def S(self) -> int:
        return len(self.lengths)

    @property
    def offsets(self) -> list[int]:
        return [sum(self.lengths[:s]) for s in range(self.S)]


def wrap8(v):
    return ((v + 128) & 0xFF) - 128


def signed32(v):
    v = v & M32
    return np.where(v >= 1 << 31, v - (1 << 32), v)


def stream(seed: bytes, label: str, n: int) -> np.ndarray:
    """n uint32 from SHA-512 in counter mode (the stand-in for the challenger's live coins)."""
    out, ctr = bytearray(), 0
    while len(out) < 4 * n:
        out += hashlib.sha512(seed + label.encode() + struct.pack("<Q", ctr)).digest()
        ctr += 1
    return np.frombuffer(bytes(out[:4 * n]), dtype="<u4").astype(np.int64)


def uniform(seed, label, shape, u):
    return (stream(seed, label, int(np.prod(shape))) % (2 * u + 1) - u).reshape(shape)


def bits(seed, label, rows, cols):
    return stream(seed, label, rows * cols).reshape(rows, cols) & 1


def early_coins(seed: bytes, sh: Shape) -> dict:
    long = sh.S - sh.late
    return dict(V=uniform(seed, "V", (sh.m, sh.L), UX), U=uniform(seed, "U", (sh.n, sh.L), UW),
                pi=[np.argsort(stream(seed, f"pi{s}", sh.L), kind="stable") for s in range(sh.S)],
                b=[bits(seed, f"b{s}", sh.m, sh.lengths[s]) for s in range(long)],
                c=[bits(seed, f"c{s}", sh.n, sh.lengths[s]) for s in range(long)])


def late_coins(seed: bytes, sh: Shape) -> dict:
    late = range(sh.S - sh.late, sh.S)
    return dict(b=[bits(seed, f"b{s}", sh.m, sh.lengths[s]) for s in late],
                c=[bits(seed, f"c{s}", sh.n, sh.lengths[s]) for s in late])


def root(words: list[np.ndarray]) -> str:
    """The commitment's stand-in: SHA-512 over the words in order (the protocol uses NF3's hiding leaves and tree)."""
    h = hashlib.sha512()
    for T in words:
        h.update((T & M32).astype("<u4").tobytes())
    return h.hexdigest()


def segment(A, B, co, sh: Shape, s: int):
    """Segment s's operands X_s, W_s, wraps wX_s, wW_s, and (B + c)_s."""
    o, ln = sh.offsets[s], sh.lengths[s]
    cols = co["pi"][s][:ln]
    xs = A[:, o:o + ln] + co["b"][s] + co["V"][:, cols]
    e = B[:, o:o + ln] + co["c"][s]
    ws = e + co["U"][:, cols]
    X, W = wrap8(xs), wrap8(ws)
    return X, W, (xs - X) // 256, (ws - W) // 256, e


def scatter(Y, cols, L):
    out = np.zeros((Y.shape[0], L), dtype=np.int64)
    out[:, cols] += Y
    return out


def digit_product(Z, M):
    """Z Mᵀ through two base-128 digits of Z, as the INT8 kernel computes it; and which digits are nonzero."""
    lo, hi = Z % 128, Z // 128
    assert np.abs(hi).max() < 128, "Z needs a third digit"
    return lo @ M.T + 128 * (hi @ M.T), [bool(lo.any()), bool(hi.any())]


def peel(A, B, co, sh: Shape, segs):
    """Z_X Uᵀ + V Y_Wᵀ over the given segments, the nonzero digits of each, and Σ_s wX_s E_sᵀ + X_s wW_sᵀ."""
    ZX = sum(scatter(segment(A, B, co, sh, s)[0], co["pi"][s][:sh.lengths[s]], sh.L) for s in segs)
    YW = sum(scatter(segment(A, B, co, sh, s)[4], co["pi"][s][:sh.lengths[s]], sh.L) for s in segs)
    pz, dz = digit_product(ZX, co["U"])
    py, dy = digit_product(YW, co["V"])
    wraps = 0
    for s in segs:
        X, _, wX, wW, e = segment(A, B, co, sh, s)
        wraps = wraps + wX @ e.T + X @ wW.T
    return pz + py.T, dz + dy, wraps


def bit_product(b, c):
    """b cᵀ for 0/1 matrices as `wgmma .and.popc` computes it: popcount of the AND of packed rows."""
    assert set(np.unique(b)) <= {0, 1} and set(np.unique(c)) <= {0, 1}
    pb, pc = np.packbits(b.astype(np.uint8), axis=1), np.packbits(c.astype(np.uint8), axis=1)
    pop = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    return pop[pb[:, None, :] & pc[None, :, :]].sum(-1).astype(np.int64)


def merged(early, late):
    return dict(early, b=early["b"] + late["b"], c=early["c"] + late["c"])


def run(prover, A, B, sh: Shape, seed: bytes, **kw) -> dict:
    """The challenger: early coins, the prover's long-segment words, their root, then the late coins."""
    early, log = early_coins(seed, sh), {}

    def draw_late(long_words):
        assert len(long_words) == sh.S - sh.late
        log["root"] = root(long_words)
        return late_coins(seed + b"/late", sh)

    words, O, cost, co = prover(A, B, sh, early, draw_late, **kw)
    assert root(words[:sh.S - sh.late]) == log["root"], "opened words differ from the root committed before late coins"
    return dict(words=words, O=O, cost=cost, root=log["root"], coins=co)


def honest(A, B, sh: Shape, early, draw_late):
    co, words, T = early, [], 0
    for s in range(sh.S):
        if s == sh.S - sh.late:
            co = merged(early, draw_late(words))
        X, W, *_ = segment(A, B, co, sh, s)
        T = (T + X @ W.T) & M32
        words.append(T)
    if sh.late == 0:
        draw_late(words)
    P, _, wraps = peel(A, B, co, sh, range(sh.S))
    O = signed32(words[-1] - P + 256 * wraps)
    return words, O, dict(int8=sh.K + RHO * sh.L, bit=0), co


def derive(A, B, sh: Shape, early, draw_late, s_star: int, zero_digits: bool):
    """R1 at A = B = 0: the word of segment s* from its prefix's bit product and peel; O from the bit product."""
    assert not A.any() and not B.any() and s_star < sh.S - sh.late
    co, words, T = early, [], 0
    prefix = range(s_star + 1)
    P, digits, wraps = peel(A, B, co, sh, prefix)
    assert not np.any(wraps), "A = B = 0 never wraps: |b + V| <= 97, |c + U| <= 81"
    bp = sum(bit_product(co["b"][s], co["c"][s]) for s in prefix)
    for s in range(sh.S):
        if s == sh.S - sh.late:
            co = merged(early, draw_late(words))
        if s == s_star:
            T = (bp + P) & M32
        else:
            X, W, *_ = segment(A, B, co, sh, s)
            T = (T + X @ W.T) & M32
        words.append(T)
    if sh.late == 0:
        draw_late(words)
    O = bp + sum(bit_product(co["b"][s], co["c"][s]) for s in range(s_star + 1, sh.S))
    n_digit = sum(digits) if zero_digits else RHO
    return words, O, dict(int8=sh.K - sh.lengths[s_star] + n_digit * sh.L, bit=sh.K), co


def bit_route(A, B, sh: Shape, early, draw_late):
    """R2 at A = B = 0 (PD2-1's O-route): every word honest, O from the bit product, no peel."""
    assert not A.any() and not B.any()
    words, _, _, co = honest(A, B, sh, early, draw_late)
    O = sum(bit_product(co["b"][s], co["c"][s]) for s in range(sh.S))
    return words, O, dict(int8=sh.K, bit=sh.K), co


def zero_digit(A, B, sh: Shape, early, draw_late):
    """R3 at A = B = 0 (PD2-1): honest, but skipping the peel digit products that are zero in every entry."""
    words, O, _, co = honest(A, B, sh, early, draw_late)
    _, digits, _ = peel(A, B, co, sh, range(sh.S))
    return words, O, dict(int8=sh.K + sum(digits) * sh.L, bit=0, digits=digits), co


def verify(A, B, sh: Shape, t: dict) -> bool:
    """Every cell of every word and of O, recomputed from the operands and the coins (drawn cells in the protocol)."""
    co, prev = t["coins"], 0
    for s in range(sh.S):
        X, W, *_ = segment(A, B, co, sh, s)
        if not np.array_equal((t["words"][s] - prev) & M32, (X @ W.T) & M32):
            return False
        prev = t["words"][s]
    P, _, wraps = peel(A, B, co, sh, range(sh.S))
    return bool(np.array_equal(t["O"] & M32, (t["words"][-1] - P + 256 * wraps) & M32))


def served_exact(A, B, sh: Shape, t: dict) -> bool:
    co = t["coins"]
    Ab = A + np.concatenate(co["b"], axis=1)
    Bc = B + np.concatenate(co["c"], axis=1)
    return bool(np.array_equal(t["O"], Ab @ Bc.T))


def workloads(sh: Shape, rng):
    m, n, K = sh.m, sh.n, sh.K
    heavy = lambda r: np.clip(np.round(rng.standard_t(3, size=(r, K)) * 12), -128, 127).astype(np.int64)
    return {"heavy-tailed": (heavy(m), heavy(n)),
            "A = B = 0": (np.zeros((m, K), np.int64), np.zeros((n, K), np.int64)),
            "A = B = 127": (np.full((m, K), 127, np.int64), np.full((n, K), 127, np.int64)),
            "uniform": (rng.integers(-128, 128, size=(m, K)), rng.integers(-128, 128, size=(n, K)))}


SHAPES = [
    Shape("v2 schedule, S 4 of L 256", 32, 32, 256, (256,) * 4),
    Shape("SEG-COINS, one long 7K/8 + K/8", 32, 32, 896, (896, 128), late=1),
    Shape("SEG-COINS, two long 7K/16 + K/8", 32, 32, 448, (448, 448, 128), late=1),
]


def saving(sh: Shape, hon: dict, att: dict, s_b: int) -> float:
    c = lambda d: d["int8"] + d["bit"] / s_b
    return (c(hon) - c(att)) / c(hon)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rng, out, fails = np.random.default_rng(137), [], 0

    def line(ok: bool, name: str, detail: str, **extra):
        nonlocal fails
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
        out.append(dict(ok=ok, check=name, detail=detail, **extra))

    for sh in SHAPES:
        seed = hashlib.sha512(sh.name.encode()).digest()
        for wl, (A, B) in workloads(sh, rng).items():
            t = run(honest, A, B, sh, seed)
            line(served_exact(A, B, sh, t) and verify(A, B, sh, t), f"{sh.name} | honest | {wl}",
                 f"K {sh.K}, O = (A+b)(B+c)ᵀ exactly, every word and O verify")
        A, B = workloads(sh, rng)["A = B = 0"]
        hon = run(honest, A, B, sh, seed)
        bad = dict(hon, words=[w.copy() for w in hon["words"]])
        bad["words"][0][3, 5] ^= 1
        line(not verify(A, B, sh, bad), f"{sh.name} | control", "one bit of one word flipped is caught")
        same = lambda t: all(np.array_equal(x, y) for x, y in zip(t["words"], hon["words"])) and \
            np.array_equal(t["O"], hon["O"]) and t["root"] == hon["root"]

        routes = [(f"R1, derive segment {s}{' + R3' if zd else ''}", derive, dict(s_star=s, zero_digits=zd))
                  for s in range(sh.S - sh.late) for zd in (False, True)]
        routes += [("R2, O by the bit product, no peel", bit_route, {}), ("R3, skip zero digits", zero_digit, {})]
        for name, prover, kw in routes:
            t = run(prover, A, B, sh, seed, **kw)
            ok = same(t) and verify(A, B, sh, t)
            sav = {card: saving(sh, hon["cost"], t["cost"], s_b) for card, s_b in S_B.items()}
            digits = t["cost"].get("digits")
            extra = f", nonzero digits (Z_X lo, hi, Y_W lo, hi) {digits}" if digits else ""
            line(ok, f"{sh.name} | A = B = 0 | {name}",
                 "words, O and the pre-late root equal honest's bit for bit, all verify; saves "
                 + ", ".join(f"{c} {v:+.1%}" for c, v in sav.items()) + " of honest's cost" + extra,
                 saving=sav, cost=t["cost"], honest_cost=hon["cost"], K=sh.K, L=sh.L, lengths=list(sh.lengths))
    print(f"{len(out) - fails}/{len(out)} checks pass")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "segcoins.json").write_text(json.dumps(out, indent=1))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
