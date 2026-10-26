"""(e) The receipt folds, counted: the reference kernel's implementation and every alternative exact one tried, each
checked against `segment.py` bit for bit.

A lane's state is li's 8 chain words, or duo's 8 y then 8 z (`segment.fold_keys`). One absorb takes a set of 32 words
(`segment.absorb`); the warp checkpoint after a segment is `segment.warp_checkpoint`.

- **li** (`segment._li_absorb`): half step y <- (y ^ d0 ^ k)·a + d1, one LOP3 and one IMAD a pair. No alternative is
  cheaper in instructions: the fan-in floor is 1/2 a word (3-input instructions), and `theory/fold_routes.py`,
  `fold_super.py` and `fold_cegis.py` find nothing between. The ones run here: `li` (reference) and `li_kimad` (the
  key's XOR as an IMAD-pipe program, `theory/fold_min.py`'s identity: the word's XOR stays a LOP3, so it only adds).
- **duo** (`segment._duo_absorb`): u = y ^ d ^ k, y' = u·a + z, z' = rotl(u, r) + y'. The rotate-add has four exact
  forms: LEA.HI (one ALU instruction, the reference), IMAD.HI then IMAD (two on the IMAD pipe), SHF.L.W then IADD3 (two
  ALU) and IMAD.WIDE then IADD3. A per-word mix of forms is exact, so the cheapest priced duo is an LP over them.
- **checkpoints**: the in-lane XOR of the 8 chains (4 LOP3, the fan-in floor for 8 words) and then the warp's XOR: REDUX
  with its result moved to a register and a quarter STG.128 (the reference, LI-CP-W's cp-seg4), or a red.global.xor
  (cp-red), or 3 LOP3 and two red.global.xor (cp-red2, which balances ALU against issue on DS-90), or 5 SHFL.BFLY and
  5 LOP3 (LI-CP-W's first form).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from verity.core.protocols.pouw.circuit import segment as SG

from .isa import Kernel, U32

NCH = SG.NCH
ROTADD = ("leahi", "imad", "shf", "wide")


def li_absorb(k: Kernel, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """li on every lane at once: y (lanes, 8), p (lanes, 32)."""
    y = y.copy()
    for ch in range(NCH):
        e = SG.li_entries(ch)
        for h in range(2):
            u = k.xor3(y[:, ch], p[:, e[2 * h]], SG.LK[ch][h])
            y[:, ch] = k.imad(u, SG.LA[ch][h], p[:, e[2 * h + 1]])
    return y


def _xor_key_imad(k: Kernel, x: np.ndarray, K: int, a: int, c: np.ndarray) -> np.ndarray:
    """(x ^ K)·a + c on the IMAD pipe alone (`theory/fold_min.py`): x ^ K = (1 - 2K_0) x + K + sum_b tau_b 2^(b+1) (x >> b)
    over the bits b in 1..30 where K's bit changes; x >> b is IMAD.HI(x, 2^(32 - b), 0)."""
    m = 0xFFFFFFFF
    acc = k.imad(x, ((1 - 2 * (K & 1)) * a) & m, (c + U32((K * a) & m)).astype(U32))
    for b in range(1, 31):
        if (K >> b & 1) != (K >> (b - 1) & 1):
            tau = -1 if K >> b & 1 else 1
            s = k.imad_hi(x, 1 << (32 - b), 0)
            acc = k.imad(s, (tau * a * (1 << (b + 1))) & m, acc)
    return acc


def li_kimad_absorb(k: Kernel, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """li with each key's XOR on the IMAD pipe: x = y ^ d0 (a LOP3, two inputs), then (x ^ k)·a + d1 as IMADs. The key's
    constant K·a + d1 rides the first IMAD's addend; adding K·a to d1 is free only because the wgmma accumulator of d1's
    entry could start at K·a (once a tile), which the count grants."""
    y = y.copy()
    for ch in range(NCH):
        e = SG.li_entries(ch)
        for h in range(2):
            x = k.xor2(y[:, ch], p[:, e[2 * h]])
            y[:, ch] = _xor_key_imad(k, x, SG.LK[ch][h], SG.LA[ch][h], p[:, e[2 * h + 1]])
    return y


def rotadd(k: Kernel, form: str, u: np.ndarray, r: int, y: np.ndarray) -> np.ndarray:
    """rotl(u, r) + y in one of its exact forms."""
    if form == "leahi":
        return k.lea_hi_rot(u, r, y)
    if form == "imad":
        return k.imad(u, 1 << r, k.imad_hi(u, 1 << r, y))
    if form == "shf":
        return k.iadd3(k.shf_l_w(u, r), y)
    if form == "wide":
        w = k.imad_wide(u, 1 << r, np.zeros(u.shape, dtype=np.uint64))
        return k.iadd3((w & np.uint64(0xFFFFFFFF)).astype(U32), (w >> np.uint64(32)).astype(U32), y)
    raise ValueError(form)


def duo_absorb(k: Kernel, st: np.ndarray, p: np.ndarray, form: Callable[[int], str] = lambda q: "leahi") -> np.ndarray:
    """duo on every lane at once: st (lanes, 16), p (lanes, 32); ``form(q)`` names entry q's rotate-add form."""
    y, z = st[:, :NCH].copy(), st[:, NCH:].copy()
    for t in range(SG.MI * SG.NI):
        h = 4 * (t % 2)
        for e in range(4):
            q, c = 4 * t + e, h + e
            u = k.xor3(y[:, c], p[:, q], SG.KK[t])
            y[:, c] = k.imad(u, SG.KA[t], z[:, c])
            z[:, c] = rotadd(k, form(q), u, SG.ROT[t], y[:, c])
    return np.concatenate([y, z], axis=1)


def mixed(share_imad: float) -> Callable[[int], str]:
    """Entry q takes the IMAD form for a ``share_imad`` fraction of the 32 entries (spread evenly), LEA.HI otherwise."""
    m = round(share_imad * 32)
    pick = {int(i * 32 / m) for i in range(m)} if m else set()
    return lambda q: "imad" if q in pick else "leahi"


def _lane_xor8(k: Kernel, v: np.ndarray) -> np.ndarray:
    """XOR of a lane's 8 chain words in 4 LOP3 (3-input LOP3 removes two words each: the fan-in floor)."""
    a = k.xor3(v[:, 0], v[:, 1], v[:, 2])
    a = k.xor3(a, v[:, 3], v[:, 4])
    a = k.xor3(a, v[:, 5], v[:, 6])
    return k.xor2(a, v[:, 7])


def checkpoint(k: Kernel, st: np.ndarray, route: str = "redux") -> np.ndarray:
    """The warp checkpoint of every warp, (warps, parts), as `segment.warp_checkpoint`."""
    parts = st.shape[1] // NCH
    lanes = st.shape[0]
    out = []
    for j in range(parts):
        a = _lane_xor8(k, st[:, NCH * j:NCH * j + NCH])
        if route == "redux":
            w = k.redux_xor(a)
            k.add("IMAD", lanes)          # IMAD.U32 R, RZ, RZ, UR: the uniform result back to a register
            k.stg(lanes // 4, lanes // 32 * 4)   # a quarter STG.128 a warp: four segments' words in one store
        elif route == "redg":
            w = np.bitwise_xor.reduce(a.reshape(-1, 32), axis=1).astype(U32)
            k.add("REDG", lanes)          # red.global.xor.b32 from every lane into the checkpoint word
        elif route == "redg2":
            v = st[:, NCH * j:NCH * j + NCH]
            a = k.xor3(k.xor3(v[:, 0], v[:, 1], v[:, 2]), v[:, 3], v[:, 4])
            b = k.xor3(v[:, 5], v[:, 6], v[:, 7])
            w = np.bitwise_xor.reduce((a ^ b).reshape(-1, 32), axis=1).astype(U32)
            k.add("REDG", 2 * lanes)      # 3 LOP3 leave two words a lane; each goes by its own red.global.xor
        elif route == "shfl":
            for m in (16, 8, 4, 2, 1):
                a = k.xor2(a, k.shfl_bfly(a, m))
            w = a[::32]
            k.stg(lanes // 32, lanes // 32 * 4)
        else:
            raise ValueError(route)
        out.append(w)
    return np.stack(out, axis=1)
