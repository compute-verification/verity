"""PoUW's receipt folds, the one reference: each fold's step, its key rule, the start states and keys of the schedules that
run it, and what is proved about it (`FOLDS`). `receipt_folds.cuh` is the same in C++ for CUDA and host code, `vectors.json`
the published test vectors (`receipt_conform.py --write` makes them from this file), and `receipt_conform.py` checks an
implementation against them. Stdlib only; every word is a uint32 and all arithmetic wraps mod 2^32. The steps, `rotl` and
`mix32` also take numpy uint32 arrays (pfill_ref.py runs them over a tile's lanes at once).

A fold absorbs an MMA's four words (d0, d1, d2, d3), a lane's (row g, row g + 8) x (column 2t, 2t + 1), into a state:

    ram    y <- (rotl(y, 7) + d0) a0 + d1;   y <- (rotl(y, 19) + d2) a1 + d3
    xin13  y <- (rotl(y, 13) ^ d0) a0 + d1;  y <- (rotl(y, 19) ^ d2) a1 + d3        (XOR-in with rotations 13, 19)
    li     y <- (y ^ d0 ^ k0) a0 + d1;       y <- (y ^ d2 ^ k1) a1 + d3             (LOP3+IMAD)
    duo    per word d with keys (k, a, r): u = y ^ d ^ k;  y <- u a + z;  z <- rotl(u, r) + y
    xin7   xin13 with rotations (7, 19): retired, still read by older copies

Schedules (which words a chain absorbs, its start, its keys): `pfill_*` (pfill.cuh's lane chains), `served_*` (fgemm.cu's
duo, which EGEMM's duo kernels and E2E's PouwSegCheck share), `egemm_*` (EGEMM's x13_l2o) and `e2e_*` (E2E's xin). The
steps are shared; the schedules are not, and `SCHEDULES` says who runs which.
"""
from __future__ import annotations

M32 = 0xFFFFFFFF
GOLDEN = 0x9E3779B9


def rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & M32


def mix32(x: int) -> int:
    """fgemm.cu's mix32."""
    x = x & M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


# ---- steps ---------------------------------------------------------------------------------------------------------------

def ram_step(y: int, d0: int, d1: int, d2: int, d3: int, a0: int, a1: int) -> int:
    y = ((rotl(y, 7) + d0) * a0 + d1) & M32
    return ((rotl(y, 19) + d2) * a1 + d3) & M32


def xin13_step(y: int, d0: int, d1: int, d2: int, d3: int, a0: int, a1: int) -> int:
    y = ((rotl(y, 13) ^ d0) * a0 + d1) & M32
    return ((rotl(y, 19) ^ d2) * a1 + d3) & M32


def xin7_step(y: int, d0: int, d1: int, d2: int, d3: int, a0: int, a1: int) -> int:
    y = ((rotl(y, 7) ^ d0) * a0 + d1) & M32
    return ((rotl(y, 19) ^ d2) * a1 + d3) & M32


def li_step(y: int, d0: int, d1: int, d2: int, d3: int, a0: int, a1: int, k0: int, k1: int) -> int:
    y = ((y ^ d0 ^ k0) * a0 + d1) & M32
    return ((y ^ d2 ^ k1) * a1 + d3) & M32


def duo_word(y: int, z: int, d: int, k: int, a: int, r: int) -> tuple[int, int]:
    u = y ^ d ^ k
    y = (u * a + z) & M32
    return y, (rotl(u, r) + y) & M32


#: pfill's duo: words d0..d3 with keys (k0, a0), (k1, a1), (k0, a0), (k1, a1) and these rotations
PFILL_DUO_ROT = (5, 11, 17, 29)


def duo_step(y: int, z: int, d0: int, d1: int, d2: int, d3: int, a0: int, a1: int, k0: int, k1: int) -> tuple[int, int]:
    """pfill's duo step over one MMA's four words."""
    for d, k, a, r in zip((d0, d1, d2, d3), (k0, k1, k0, k1), (a0, a1, a0, a1), PFILL_DUO_ROT):
        y, z = duo_word(y, z, d, k, a, r)
    return y, z


def duo_final(y: int, z: int) -> int:
    """A served duo chain's word."""
    return y ^ rotl(z, 16)


# ---- key rules -----------------------------------------------------------------------------------------------------------

def key_ok(a: int) -> bool:
    """pfill's rule: a odd, a mod 2^24 at least 2^8 + 1 away from 0, a not ±1 mod 32."""
    m, r = a & 0xFFFFFF, a & 31
    return bool(a & 1) and 256 < m < (1 << 24) - 256 and r not in (1, 31)


def int8_ratio(lam: int, bits: int = 30) -> bool:
    """lam b = b' mod 2^bits for some int8 b != 0 and b' (the r9 Λ8 membership)."""
    return any(((lam * b + 128) % (1 << bits)) < 256 for b in range(-128, 128) if b)


def lambda8_ok(a: int) -> bool:
    """a odd and neither a nor -a in Λ8 mod 2^30: a b mod 2^30 outside [-128, 128] for every b in 1..128. E2E's xin key
    rule (segment.key_ok) is this alone, without `key_ok`."""
    return bool(a & 1) and all(((a * b + 128) & 0x3FFFFFFF) > 256 for b in range(1, 129))


def key_ok_r9(a: int) -> bool:
    """The r9 rule (receipt_folds_r9.key_ok_r9): `key_ok` and `lambda8_ok`."""
    return key_ok(a) and lambda8_ok(a)


def kj19(a: int) -> bool:
    """The r11 KJ19: for each j < 19, the signed residue of a mod 2^(31 - j) is at least 2^(19 - j) in size. For odd a,
    no 12 consecutive bits [19 - j, 30 - j] of a are all equal."""
    for j in range(19):
        mod = 1 << (31 - j)
        r = a % mod
        if r >= mod >> 1:
            r -= mod
        if abs(r) < 1 << (19 - j):
            return False
    return True


def key_ok_r11(a: int) -> bool:
    """The r11 rule: `key_ok_r9` and `kj19`, the premise of its
    Theorem 1 (XOR-in-13's one-MMA kernel is exactly the group). No implementation enforces it yet."""
    return key_ok_r9(a) and kj19(a)


KEY_RULES = {"key_ok": key_ok, "lambda8": lambda8_ok, "key_ok_r9": key_ok_r9, "kj19": kj19, "key_ok_r11": key_ok_r11}


# ---- schedules -----------------------------------------------------------------------------------------------------------

#: pfill.cuh's keys when fewer than two of the key block's rows pass the rule
PFILL_KEY_DEFAULTS = (0x9E3779B9, 0x7F4A7C15)


def pfill_keys(rows: list[int], r9: bool = False) -> tuple[int, int, int, int]:
    """pf_keys: (a0, a1) the first two of the key block's 16 row words that pass `key_ok` (`key_ok_r9` in a -DPFILL_LI or
    -DPFILL_XIN13 build, for every fold of that build), else the defaults; (k0, k1) rows 14 and 15."""
    rule = key_ok_r9 if r9 else key_ok
    a = [int(x) for x in rows if rule(int(x))][:2]
    a += list(PFILL_KEY_DEFAULTS[len(a):])
    return a[0], a[1], int(rows[14]), int(rows[15])


def pfill_start(pos: int) -> tuple[int, int]:
    """pf_y0, pf_z0 of chain c of lane l of warp w, pos = (32 w + l) 4 + c."""
    return pos & M32, (pos * GOLDEN) & M32


#: fgemm.cu's tile: threads, warp tiles (mi, ni), chains a thread, and the tile's columns
NCT, MI, NI, NCH, BN, WM = 512, 2, 4, 8, 128, 32
#: fgemm.cu's lane-0 duo keys at position t = 4 mi + ni: key_mul(0, t), key_xor(0, t), rot_of(DUO, 0, t)
SERVED_KA = tuple(mix32(0xA0000000 + 2 * t) | 1 for t in range(MI * NI))
SERVED_KK = tuple(mix32(0xB0000000 + 2 * t + 1) for t in range(MI * NI))
SERVED_ROT = tuple(((t * 7 + 5) % 31) + 1 for t in range(MI * NI))


def served_entry(ctid: int, q: int) -> tuple[int, int]:
    """Thread ctid's accumulator entry q = 4 t + e (t = 4 mi + ni) as (row, column) of the 128 x 128 tile."""
    warp, lane = divmod(ctid, 32)
    wm, wn, g, qq = warp // (BN // 32), warp % (BN // 32), lane >> 2, lane & 3
    t, e = divmod(q, 4)
    mi, ni = divmod(t, NI)
    return wm * WM + 16 * mi + g + 8 * (e >> 1), wn * 32 + 8 * ni + 2 * qq + (e & 1)


def served_start(seed: int, tile: int, ctid: int, w: int = 0, nw: int = 1) -> tuple[list[int], list[int]]:
    """The duo chains' (y, z) of thread ctid of tile at window w of nw (EGEMM and fgemm: w = 0, nw = 1)."""
    y = [mix32(seed ^ mix32((((tile * NCT + ctid) * nw + w) * NCH + j) & M32)) for j in range(NCH)]
    return y, [mix32(v + GOLDEN) for v in y]


def served_duo_absorb(y: list[int], z: list[int], p: list[int]) -> tuple[list[int], list[int]]:
    """One segment's 32 words: chain 4 (t % 2) + e absorbs entry 4 t + e with position t's keys, t = 0..7 in order."""
    y, z = list(y), list(z)
    for t in range(MI * NI):
        for e in range(4):
            c = 4 * (t % 2) + e
            y[c], z[c] = duo_word(y[c], z[c], p[4 * t + e], SERVED_KK[t], SERVED_KA[t], SERVED_ROT[t])
    return y, z


#: EGEMM's key stream: candidate i is mix32(seed ^ (EGEMM_KEY_BASE + i))
EGEMM_KEY_BASE = 0x5A5A0000


def egemm_xin_keys(seed: int, rule=key_ok_r9) -> tuple[int, int, int]:
    """egemm.cu's xin_keys: the first two candidates that pass `rule`, and how many `key_ok` candidates it skipped."""
    out, skipped, i = [], 0, 0
    while len(out) < 2:
        a = mix32((seed ^ (EGEMM_KEY_BASE + i)) & M32)
        if rule(a):
            out.append(a)
        elif key_ok(a):
            skipped += 1
        i += 1
    return out[0], out[1], skipped


def egemm_xin13_start(tile: int, ctid: int) -> list[int]:
    """x13_l2o's four chains of thread ctid: y = (tile 512 + ctid) 4 + c."""
    return [((tile * NCT + ctid) * 4 + c) & M32 for c in range(4)]


def egemm_xin13_absorb(y: list[int], p: list[int], a0: int, a1: int) -> list[int]:
    """One segment's 32 words: chain t % 4 absorbs warp tile t's entries 4 t .. 4 t + 3, t = 0..7 in order."""
    y = list(y)
    for t in range(MI * NI):
        y[t % 4] = xin13_step(y[t % 4], *p[4 * t:4 * t + 4], a0, a1)
    return y


#: E2E's key stream: candidate i is mix32(E2E_KEY_BASE + i) | 1
E2E_KEY_BASE = 0xC0000000


def e2e_xin_keys(rule=lambda8_ok) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """segment.XA0, XA1: the first 16 candidates that pass `rule`, alternately chain t's a0 and a1."""
    out, i = [], 0
    while len(out) < 2 * MI * NI:
        a = mix32(E2E_KEY_BASE + i) | 1
        i += 1
        if rule(a):
            out.append(a)
    return tuple(out[0::2]), tuple(out[1::2])


def e2e_xin_absorb(y: list[int], p: list[int], xa0, xa1) -> list[int]:
    """One segment's 32 words: chain t absorbs entries 4 t .. 4 t + 3 with its own keys."""
    return [xin13_step(y[t], *p[4 * t:4 * t + 4], xa0[t], xa1[t]) for t in range(MI * NI)]


# ---- the folds and what is proved about each -------------------------------------------------------------------------------

#: Per fold: its step, its state, the keys it reads, its rotations into an XOR (None: words enter by addition), the key rule
#: each schedule draws its multipliers under, and its soundness status (``status`` one of proved, open, pairs-only,
#: retired; ``needs`` the open premises; ``ref`` where it is stated).
FOLDS = {
    "duo": {"step": duo_step, "state": ("y", "z"), "keys": ("a0", "a1", "k0", "k1"), "xor_rotations": (),
            "key_rule": {"pfill": "key_ok (key_ok_r9 in an r9 build)", "served": "fixed public odd keys SERVED_KA"},
            "status": "open", "needs": ("D",),
            "ref": "(D): different runs of a chain's words leave different final states on some start state. Proved for "
                   "one, two and three words at every width (Lean Pouw.SecurityProofs.Receipt.DuoWordInjective, "
                   "DuoPairNoKernel, DuoTripleNoKernel); false at four words for odd keys with r1 + r3 = 0 mod 32, "
                   "a2 = -1 mod 2^r1 and a3 = ±1 mod 2^(33 - r1), which key_ok and both schedules' rotations exclude. "
                   "At four words a kernel needs a2 = -1 mod 2^r1, for odd a1, a2, a3, 0 < r2, r3 < n and "
                   "0 < r1 <= n - 3 (Lean DuoQuadNoKernel), so both schedules' keys have none. At any length, two "
                   "runs with the same final states differ at two words at least four positions apart "
                   "(DuoWindowNoKernel), a run checkpointed every four words has no kernel (DuoCheckpointNoKernel), "
                   "and every run of more than 2*4^n words has one (DuoLongRunKernel); open for five or more words "
                   "at the deployed keys, and for the served receipt y ^ rotl(z, 16) at any length "
                   "(note:20261017T1000Z-draft-fold-r §6.5, §8)"},
    "xin13": {"step": xin13_step, "state": ("y",), "keys": ("a0", "a1"), "xor_rotations": (13, 19),
              "key_rule": {"pfill": "key_ok_r9", "egemm": "key_ok_r9", "e2e": "lambda8 (weaker: no key_ok)"},
              "status": "open", "needs": ("RT", "G3"),
              "ref": "RS-E-REF Part IX (note:20261015T1931Z-draft-theorem-e-referee): (RT) open, no lemma removes it; "
                     "(G₃) suffices for (G₄). P2-FF r11 (note:20261016T1549Z-draft-fullfloor-s-r11): pricing can't "
                     "replace (RT); (G₃) proved at one MMA under key_ok_r11 (Theorem 1), and for chains apart from runs "
                     "of two or more non-XOR δ"},
    "li": {"step": li_step, "state": ("y",), "keys": ("a0", "a1", "k0", "k1"), "xor_rotations": (0,),
           "key_rule": {"pfill": "key_ok_r9"},
           "status": "open", "needs": ("RT", "G3", "HO"),
           "ref": "as xin13 (RS-E-REF Part IX, P2-FF r11), and (HO): no hand-off (d1, d2) or (d3, next d0) joins two "
                  "words whose rows or columns are equal or complementary (note:20261016T1117Z-draft-fullfloor-s-r9 §2). "
                  "With every absorbed word pinned and a checkpoint a segment (the XOR of the chains' states), none of "
                  "them is read: a half step is a bijection and a checkpoint is injective in each word, for odd "
                  "multipliers alone (Lean: Pouw.SecurityProofs.Receipt.LiHalfBijective, LiCheckpointWordInjective, "
                  "LiCheckpointLowBit; note:20261017T1848Z-draft-li-alt)"},
    "ram": {"step": ram_step, "state": ("y",), "keys": ("a0", "a1"), "xor_rotations": None,
            "key_rule": {"pfill": "key_ok (key_ok_r9 in an r9 build)"},
            "status": "pairs-only", "needs": (),
            "ref": "a step is rotl(y, 7) a0 + Q: it reads a pair only through Q = a0 d0 + d1, one product a pair "
                   "(note:20261016T0917Z-draft-fullfloor-s-r8 §1.2)"},
    "xin7": {"step": xin7_step, "state": ("y",), "keys": ("a0", "a1"), "xor_rotations": (7, 19),
             "key_rule": {"pfill": "key_ok"},
             "status": "retired", "needs": (),
             "ref": "±128 is an int8 ratio at its step boundary (note:20261016T1117Z-draft-fullfloor-s-r9 §1); "
                    "pfill.cuh refuses -DPFILL_XIN"},
}

#: Who runs which schedule, and the build that selects the fold
SCHEDULES = {
    "pfill-lane": {"folds": ("ram", "duo", "xin13", "li"), "where": "pfill.cuh pf_absorb, pfill_ref.absorb",
                   "build": {"ram": (4, None), "duo": (2, None), "xin13": (4, "-DPFILL_XIN13"), "li": (4, "-DPFILL_LI"),
                             "xin7": (4, "-DPFILL_XIN")}},
    "served-duo": {"folds": ("duo",), "where": "fgemm.cu duo, EGEMM derive.tile_receipts, E2E segment._duo_absorb"},
    "egemm-xin13": {"folds": ("xin13",), "where": "egemm.cu x13_l2o, egemm_ref.xin13_receipts"},
    "e2e-xin": {"folds": ("xin13",), "where": "E2E segment._xin_absorb (PouwXinAbsorb_v1)"},
}


def step(fold: str, state, d, keys: dict):
    """One step of `fold` from state (y, or (y, z) for duo) over d = (d0, d1, d2, d3) with keys {a0, a1, k0, k1}."""
    f = FOLDS[fold]
    args = [keys[k] for k in f["keys"]]
    if fold == "duo":
        return f["step"](*state, *d, *args)
    return f["step"](state, *d, *args)


def chain(fold: str, start, words, keys: dict) -> list:
    """The states after each step of `words` (a list of 4-word steps) from `start`."""
    out, s = [], start
    for d in words:
        s = step(fold, s, d, keys)
        out.append(s)
    return out
