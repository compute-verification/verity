"""Poseidon2 over BabyBear, pure-Python reference (Plonky3 ``p3-poseidon2`` + ``p3-baby-bear``, main branch
f676a57c, 2026-09-21; the permutation is identical in ``p3-*-0.4.3-succinct`` except for the internal diagonal).

Permutation (``permute``): initial external linear layer M_E; 4 external rounds (add RC, x^7 on every lane, M_E);
``RP`` internal rounds (add RC to lane 0, x^7 on lane 0, M_I); 4 external rounds.  M_E = circ(2 M4, M4, ..., M4)
with M4 = [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]] (``Poseidon2ExternalMatrixGeneral`` / ``MDSMat4``).  M_I =
J + diag(V) with Plonky3's BabyBear V (``INTERNAL_DIAG``: -2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 2^-8, ...).  Round
numbers at 128-bit security (``poseidon2_round_numbers_128``): width 16 -> (8, 13); width 24 -> (8, 21).

Two round-constant sets:

* ``grain``: Plonky3's hard-coded default constants (``BABYBEAR_POSEIDON2_RC_16_*``, Grain LFSR), canonical values.
* ``rng1``: the constants ``Poseidon2::new_from_rng_128(Xoroshiro128Plus::seed_from_u64(1))`` draws, re-derived
  here (``rng_constants``: rand_core's PCG32 ``seed_from_u64``, rand_xoshiro's Xoroshiro128Plus with ``next_u32 =
  next_u64 >> 32``, Plonky3's ``StandardUniform`` for MontyField31: reject ``next_u32 >> 1 >= P`` and take the
  value as the *Montgomery* form, i.e. the field element is ``v * 2^-32 mod P``).  The two Plonky3 test vectors
  (``PLONKY3_VECTORS``) use these constants, so ``check_vectors`` validates the permutation end to end.

Digest conventions of this package (``compress``, ``leaf_blocks``): a digest is 8 field elements; ``compress(a, b)``
= first 8 lanes of ``permute(a || b || 0...)`` (Plonky3's ``TruncatedPermutation<_, 2, 8, WIDTH>``); a column of
``R`` elements is hashed as a binary tree over its ``ceil(R/8)`` zero-padded 8-element blocks with the rightmost
odd node promoted (``leaf_blocks``), the same tree rule as the Merkle levels.
"""
from __future__ import annotations

from collections.abc import Sequence

P = 0x78000001
R32 = 1 << 32
R32_INV = pow(R32, P - 2, P)      # 2^-32 mod P
ROUNDS = {16: (8, 13), 24: (8, 21)}
DIGEST = 8

# Plonky3 BabyBearInternalLayerParameters::internal_layer_mat_mul (V; lane 0 is handled as -2 there too).
_INV = lambda k: pow(pow(2, k, P), P - 2, P)
INTERNAL_DIAG = {
    16: [P - 2, 1, 2, _INV(1), 3, 4, P - _INV(1), P - 3, P - 4, _INV(8), _INV(2), _INV(3), _INV(27), P - _INV(8),
         P - _INV(4), P - _INV(27)],
    24: [P - 2, 1, 2, _INV(1), 3, 4, P - _INV(1), P - 3, P - 4, _INV(8), _INV(2), _INV(3), _INV(4), _INV(7), _INV(9),
         _INV(27), P - _INV(8), P - _INV(2), P - _INV(3), P - _INV(4), P - _INV(5), P - _INV(6), P - _INV(7),
         P - _INV(27)],
}
M4 = ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))


def external_matrix(width: int) -> list[list[int]]:
    """M_E = circ(2 M4, M4, ..., M4) as a dense width x width integer matrix (entries in 1..6)."""
    m = [[0] * width for _ in range(width)]
    for bi in range(width // 4):
        for bj in range(width // 4):
            f = 2 if bi == bj else 1
            for i in range(4):
                for j in range(4):
                    m[4 * bi + i][4 * bj + j] = f * M4[i][j]
    return m


def sbox7(x: int) -> int:
    x2 = x * x % P
    x4 = x2 * x2 % P
    return x4 * x2 % P * x % P


def mds_external(s: list[int]) -> list[int]:
    w = len(s)
    t = [0] * w
    for b in range(0, w, 4):
        x = s[b:b + 4]
        for i in range(4):
            t[b + i] = sum(M4[i][j] * x[j] for j in range(4))
    sums = [sum(t[k::4]) for k in range(4)]
    return [(t[i] + sums[i % 4]) % P for i in range(w)]


def mds_internal(s: list[int]) -> list[int]:
    w = len(s)
    total = sum(s)
    v = INTERNAL_DIAG[w]
    return [(total + v[i] * s[i]) % P for i in range(w)]


class Poseidon2:
    """One parameter set: width, round numbers, round constants (canonical field values)."""

    def __init__(self, width: int, ext_init: Sequence[Sequence[int]], ext_final: Sequence[Sequence[int]],
                 internal: Sequence[int], name: str = "") -> None:
        rf, rp = ROUNDS[width]
        if len(ext_init) != rf // 2 or len(ext_final) != rf // 2 or len(internal) != rp:
            raise ValueError("round-constant shape does not match the round numbers")
        self.width, self.rf, self.rp, self.name = width, rf, rp, name
        self.ext_init = [list(r) for r in ext_init]
        self.ext_final = [list(r) for r in ext_final]
        self.internal = list(internal)

    def permute(self, state: Sequence[int]) -> list[int]:
        s = [x % P for x in state]
        if len(s) != self.width:
            raise ValueError("state width")
        s = mds_external(s)
        for rc in self.ext_init:
            s = mds_external([sbox7((a + c) % P) for a, c in zip(s, rc)])
        for c in self.internal:
            s[0] = sbox7((s[0] + c) % P)
            s = mds_internal(s)
        for rc in self.ext_final:
            s = mds_external([sbox7((a + c) % P) for a, c in zip(s, rc)])
        return s

    def compress(self, left: Sequence[int], right: Sequence[int]) -> list[int]:
        s = list(left) + list(right)
        s += [0] * (self.width - len(s))
        return self.permute(s)[:DIGEST]

    def leaf_blocks(self, column: Sequence[int]) -> list[int]:
        """Leaf of a column: binary compression tree over its zero-padded 8-element blocks (rightmost odd node
        promoted).  A single block is compressed with a zero block so the leaf is never raw data."""
        col = list(column)
        col += [0] * (-len(col) % DIGEST)
        level = [col[i:i + DIGEST] for i in range(0, len(col), DIGEST)]
        if len(level) == 1:
            level.append([0] * DIGEST)
        while len(level) > 1:
            nxt = [self.compress(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        return level[0]


# ---------------------------------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------------------------------

def _splitmix64(seed: int, n: int) -> list[int]:
    """rand_xoshiro ``Xoroshiro128Plus::seed_from_u64`` seeds via SplitMix64 (``from_splitmix!``): two u64 words."""
    m = (1 << 64) - 1
    out = []
    x = seed & m
    for _ in range(n):
        x = (x + 0x9E3779B97F4A7C15) & m
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
        out.append(z ^ (z >> 31))
    return out


class Xoroshiro128Plus:
    def __init__(self, seed_u64: int) -> None:
        self.s0, self.s1 = _splitmix64(seed_u64, 2)

    def next_u64(self) -> int:
        m = (1 << 64) - 1
        s0, s1 = self.s0, self.s1
        r = (s0 + s1) & m
        s1 ^= s0
        self.s0 = (((s0 << 24) | (s0 >> 40)) & m) ^ s1 ^ ((s1 << 16) & m)
        self.s1 = ((s1 << 37) | (s1 >> 27)) & m
        return r

    def next_u32(self) -> int:
        return self.next_u64() >> 32


def _sample_babybear(rng: Xoroshiro128Plus) -> int:
    """Plonky3 ``StandardUniform`` for MontyField31: rejection-sample a u31 and take it as the Montgomery form."""
    while True:
        v = rng.next_u32() >> 1
        if v < P:
            return v * R32_INV % P


def rng_constants(width: int, seed: int = 1) -> Poseidon2:
    rf, rp = ROUNDS[width]
    rng = Xoroshiro128Plus(seed)
    ext = [[_sample_babybear(rng) for _ in range(width)] for _ in range(rf)]
    internal = [_sample_babybear(rng) for _ in range(rp)]
    return Poseidon2(width, ext[: rf // 2], ext[rf // 2:], internal, f"rng{seed}")


def grain_constants(width: int) -> Poseidon2:
    from . import poseidon2_grain as g

    ei = getattr(g, f"GRAIN_EXT_INIT_{width}")
    ef = getattr(g, f"GRAIN_EXT_FINAL_{width}")
    it = getattr(g, f"GRAIN_INT_{width}")
    return Poseidon2(width, [ei[i:i + width] for i in range(0, len(ei), width)],
                     [ef[i:i + width] for i in range(0, len(ef), width)], it, "grain")


def instance(width: int = 16, constants: str = "grain") -> Poseidon2:
    return grain_constants(width) if constants == "grain" else rng_constants(width)


#: Plonky3 ``baby-bear/src/poseidon2.rs`` tests (``test_poseidon2_width_{16,24}_random``, rng1 constants).
PLONKY3_VECTORS = {
    16: ([894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831, 120589055, 19776022,
          42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920, 1783754913],
         [1255099308, 941729227, 93609187, 112406640, 492658670, 1824768948, 812517469, 1055381989, 670973674,
          1407235524, 891397172, 1003245378, 1381303998, 1564172645, 1399931635, 1005462965]),
    24: ([886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108, 355597330, 1799586834,
          59617783, 790334801, 1968791836, 559272107, 31054313, 1042221543, 474748436, 135686258, 263665994,
          1962340735, 1741539604, 449439011, 1131357108, 50869465, 1589724894],
         [249424342, 562262148, 757431114, 354243402, 57767055, 976981973, 1393169022, 1774550827, 1527742125,
          1019514605, 1776327602, 266236737, 1412355182, 1070239213, 426390978, 1775539440, 1527732214, 1101406020,
          1417710778, 1699632661, 413672313, 820348291, 1067197851, 1669055675]),
}


def check_vectors() -> None:
    for w, (inp, exp) in PLONKY3_VECTORS.items():
        got = rng_constants(w).permute(inp)
        if got != exp:
            raise AssertionError(f"Poseidon2 width {w}: {got[:3]}... != Plonky3 {exp[:3]}...")


if __name__ == "__main__":       # pragma: no cover
    check_vectors()
    print("Plonky3 vectors OK (width 16, 24)")
