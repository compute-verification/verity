"""Hiding through forced work: N-LIT's counter-mode pad and its keyed form against hm96 with N-HIDE's forced salt (a), and
hm96's mask M_key·y computed as an int8 tensor-core product whose k32 partial sums are committed (FM). Checks and prices,
stdlib and verity.ml.commitments.hm96 only, no GPU.

    python3 benchmarks/pouw/theory/salt.py OUT_JSON [--salts N]

The pad family. ks is 64 bytes from the OS generator, committed once a window under its own hm96 leaf before the window's
coins; S = f(IV, ks ‖ salt_w) is the window's midstate (ks ‖ salt_w is one 128-byte block); pos is a public digest's frame-v3
domain, rank and position, distinct under one ks.
  pad (P1, N-LIT's): the public digest is b = x ⊕ f(S, pos ‖ padding), one compression a digest and b's 16 XORs;
  keyed digest (P2, N-HIDE's e3 shape): the public digest is f(S, pos ‖ x ‖ padding), one compression, no XOR;
  keyed leaf (P3): the public digest is x's own hash started from S instead of from the IV, its first block
    row-bound's (salt, forward, position) block with the length (without pos, equal rows give equal digests), which
    row-bound already spends; a frame-v3 node's message already carries (domain, level, index). Nothing is added a
    digest, and the first block makes the messages prefix-free.
Each binds through ks: ks's hm96 leaf fixes ks, so the pad (or key) is fixed and x is unique under cr/sha-512, for any ks
the prover picks. Each hides at SHA-512's PRF property: P1 and P2 on one block from S (exactly the generator convention of
hm96 PROTOCOL §3, as today's served salts SHA-512(Ks ‖ pos ‖ r) read it), P3 on a prefix-free cascade of blocks
(Bellare-Canetti-Krawczyk's reduction to the same property, which prefix-freeness protects from length extension). A drawn unit's proof opens ks, computes S and checks the digest.

The forced mask (FM) keeps hm96-sha512/v1 bit for bit. y is the salt as 48 little-endian 32-bit words h_w; each word is
formed into eight registers R_{w,k} = h_w & (0x01010101 << k), one LOP3 each, so int8 lane e of R_{w,k} is 2^k·y_j for
j = 32w + 8e + k (−128·y_j at k = 7). The public B operand holds M'[i][j] = M_key[i][j]·s_k, s_0 = −128,
s_k = 2^(7−k) for k = 1..6, s_7 = 1, so every product is ±128·M[i][j]·y_j: the int32 sum has bits 0..6 zero and bit 7 equal to
(M_key·y)_i over GF(2). Word w's 32 lanes are k32 step w, so each output has 48 partials P[i][w], all committed in a later
instance (an A instance's in its forward's words instance, a words instance's in the next forward's). Each partial is affine
in y's bits with coefficients ±128·M[i][j] on the bits of words ≤ w; if the 24,576 prefixes are pairwise distinct and nonzero
(checked on the pinned key), every partial is a distinct, non-free word, which DistinctWrites charges 32: the k32 step's 32
MACs. It needs (a): with one fixed y every partial is a constant.

Honest-only work a public digest, W1 at the floor (32 a register, 116,224 a compression) and alone (64.13, 232,888):
"premise-free" counts every hash and register the design adds; "credited" lets ShaWrites credit forced hashing as the lead's
set does ((a)'s PRF and c, the pad's compression). The pad's 16 XORs are charged unless b's words are read into
DistinctWrites' set of committed words (they are committed, distinct and non-free), which credits 32 each. Salt counts: L1 is N-HIDE's (one public digest a call for A_j and one a
forward for its words); L0 is one a K-byte activation row a call and one an R-byte row of ssn_1l's words.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from verity.ml.commitments.hm96 import SHA512  # noqa: E402

N_OUT, L_SALT = SHA512.rows, SHA512.cols            # 512 mask bits, 1,536 salt bits
WORDS = L_SALT // 32                                 # k32 steps a salt
SCALE = [-128, 64, 32, 16, 8, 4, 2, 1]              # s_k: lane 2^k (−128 at k = 7) times s_k is ±128
SIGN = [-1, 1, 1, 1, 1, 1, 1, -1]
POS = sum(1 << (8 * e + k) for e in range(4) for k in range(8) if SIGN[k] > 0)
NEG = sum(1 << (8 * e + k) for e in range(4) for k in range(8) if SIGN[k] < 0)

LAYERS = 16
LINEARS = {"qkv": (3072, 2048), "o": (2048, 2048), "gate_up": (16384, 2048), "down": (2048, 8192)}   # (n, K)
CALLS = 4 * LAYERS
MACS_TOKEN = LAYERS * sum(n * k for n, k in LINEARS.values())
WORD_BYTES_TOKEN = LAYERS * sum(n * 4 * (1 + math.ceil((k - 28) / 31)) for n, k in LINEARS.values())   # ssn_1l
A_BYTES_TOKEN = LAYERS * sum(k for _, k in LINEARS.values())
POINTS = {"T 8,192": 8192, "T 512, 512 rows": 512, "T 512, 64 rows": 64, "decode, 16 rows": 16, "decode, 1 row": 1}
PRICE_REG = {"floor": 32.0, "alone": 64.13}
PRICE_COMPRESSION = {"floor": 116_224, "alone": 232_888}
SALT = {"unforced": {"floor": 791_892, "alone": 1_586_849}, "forced (a)": {"floor": 238_080, "alone": 477_127}}
TARGET = 0.001
DIGEST_REGS = N_OUT // 32                            # a 512-bit digest in 32-bit registers
FORM = 8 * WORDS                                     # FM's formed registers R_{w,k}
GATHER = 3 * N_OUT // 4 + 7 * N_OUT // 32            # FM: byte 0 of four partials by three PRMTs, eight registers by seven LEA.HI
ANDS_COMPRESSION = 58_120                            # a SHA-512 compression in-circuit
ANDS_FROM_8 = 53_568                                 # one from round 8 (the PRF's)
KS_OPEN = 5 * ANDS_COMPRESSION                       # ks's hm96 leaf opened in-circuit: inner digest 1, c 2, leaf 2
H200_MACS = 9.89e14                                  # the one basis for pooled figures (N-CARD): stock held fixed at MFU_SERVE
MFU_SERVE = 0.4
DATACENTER_H200 = 100_000
WINDOW_S = 600
WORLD_MACS_YEAR = 9.36e30 / 2
AND_PER_MAC = 2079 / 16
PERIOD_S = {"window": WINDOW_S, "week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}


def words_of(salt: bytes) -> list[int]:
    return [int.from_bytes(salt[4 * w:4 * w + 4], "little") for w in range(WORDS)]


def s8(v: int) -> int:
    return v - 256 if v >= 128 else v


def fm_lanes(salt: bytes) -> list[int]:
    """FM's A operand in k-order κ = 32w + 4k + e, each lane the s8 byte e of R_{w,k}."""
    return [s8(((h & (0x01010101 << k)) >> (8 * e)) & 0xFF) for h in words_of(salt) for k in range(8) for e in range(4)]


def fm_weights(rows: tuple[int, ...]) -> list[list[int]]:
    order = [(32 * w + 8 * e + k, k) for w in range(WORDS) for k in range(8) for e in range(4)]
    return [[((r >> j) & 1) * SCALE[k] for j, k in order] for r in rows]


def fm_literal(wts: list[list[int]], salt: bytes) -> tuple[bytes, bool, bool]:
    """The int8 product MAC by MAC: the mask from bit 7 of each final sum, whether every k32 partial has bits 0..6 zero, and
    whether every partial fits int32."""
    a = fm_lanes(salt)
    mask, low_zero, fits = 0, True, True
    for i, wi in enumerate(wts):
        acc = 0
        for t in range(WORDS):
            acc += sum(x * y for x, y in zip(a[32 * t:32 * t + 32], wi[32 * t:32 * t + 32]))
            low_zero &= acc % 128 == 0
            fits &= -(1 << 31) <= acc < (1 << 31)
        mask |= ((acc >> 7) & 1) << i
    return mask.to_bytes(N_OUT // 8, "little"), low_zero, fits


def fm_partials(rows: tuple[int, ...], salt: bytes) -> list[list[int]]:
    """P[i][w] in closed form: 128·(popcount of the row's positive-class bits of y − its negative ones), cumulated."""
    hw = words_of(salt)
    out = []
    for r in rows:
        acc, ps = 0, []
        for w, h in enumerate(hw):
            b = (r >> (32 * w)) & 0xFFFFFFFF & h
            acc += 128 * ((b & POS).bit_count() - (b & NEG).bit_count())
            ps.append(acc)
        out.append(ps)
    return out


def check_fm(n_salts: int, n_literal: int, seed: int) -> dict:
    key = SHA512.default_key
    rows = SHA512.rows_of(key)
    rng = random.Random(seed)
    wts = fm_weights(rows)
    lit = {"salts": n_literal, "match": 0, "low_zero": True, "fits_int32": True}
    for _ in range(n_literal):
        s = rng.randbytes(SHA512.salt_bytes)
        m, lz, fits = fm_literal(wts, s)
        lit["match"] += m == SHA512.mask(key, s)
        lit["low_zero"] &= lz
        lit["fits_int32"] &= fits
    fast = {"salts": n_salts, "match": 0, "low_zero": True, "max_abs": 0}
    edge = [bytes(SHA512.salt_bytes), b"\xff" * SHA512.salt_bytes]
    for t in range(n_salts):
        s = edge[t] if t < len(edge) else rng.randbytes(SHA512.salt_bytes)
        p = fm_partials(rows, s)
        m = sum(((pi[-1] >> 7) & 1) << i for i, pi in enumerate(p)).to_bytes(N_OUT // 8, "little")
        fast["match"] += m == SHA512.mask(key, s)
        fast["low_zero"] &= all(v % 128 == 0 for pi in p for v in pi)
        fast["max_abs"] = max(fast["max_abs"], max(abs(v) for pi in p for v in pi))
    # mutation: s_0 = +128 is not s8, and s_0 = 127 breaks the parity
    bad = [[((r >> j) & 1) * (127 if k == 0 else SCALE[k]) for j, k in
            [(32 * w + 8 * e + k, k) for w in range(WORDS) for k in range(8) for e in range(4)]] for r in rows[:64]]
    s = rng.randbytes(SHA512.salt_bytes)
    a = fm_lanes(s)
    mut = sum(((sum(x * y for x, y in zip(a, wi)) >> 7) & 1) << i for i, wi in enumerate(bad))
    mutation_caught = mut != int.from_bytes(SHA512.mask(key, s), "little") & ((1 << 64) - 1)
    prefixes = [r & ((1 << (32 * (w + 1))) - 1) for r in rows for w in range(WORDS)]
    blocks = [(r >> (32 * w)) & 0xFFFFFFFF for r in rows for w in range(WORDS)]
    rw = [r.bit_count() for r in rows]
    mrows = list(rows)
    mrows[1] = mrows[0] & ((1 << 64) - 1) | (mrows[1] >> 64 << 64)
    mpre = [r & ((1 << (32 * (w + 1))) - 1) for r in mrows for w in range(WORDS)]
    dist = {"partials": len(prefixes), "distinct": len(set(prefixes)), "nonzero": all(prefixes),
            "min_block_bits": min(b.bit_count() for b in blocks), "rows_distinct": len(set(rows)) == len(rows),
            "min_row_weight": min(rw), "max_row_weight": max(rw), "mutation_caught": len(set(mpre)) < len(mpre)}
    ok = (lit["match"] == n_literal and lit["low_zero"] and lit["fits_int32"] and fast["match"] == n_salts and fast["low_zero"]
          and fast["max_abs"] < (1 << 31) and mutation_caught and dist["distinct"] == dist["partials"] and dist["nonzero"]
          and dist["rows_distinct"] and dist["mutation_caught"])
    return {"literal": lit, "closed_form": fast, "parity_mutation_caught": mutation_caught, "distinct": dist, "ok": ok}


def check_pad(seed: int) -> dict:
    """The pad's inputs: ks ‖ salt_w is exactly one block, every pad input is 144 bytes (so the set is prefix-free and each pad
    is one compression from S), today's served salt input is one block, and equal x at two positions give unrelated b."""
    rng = random.Random(seed)
    ks, salt_w = rng.randbytes(64), rng.randbytes(64)

    def blocks(n: int) -> int:
        return (n + 17 + 127) // 128

    def pos(domain: int, rank: int, position: int) -> bytes:
        return domain.to_bytes(8, "big") + rank.to_bytes(3, "big") + position.to_bytes(5, "big")

    inputs = [ks + salt_w + pos(d, r, p) for d in range(3) for r in range(2) for p in range(50)]
    x = rng.randbytes(64)
    pads = [hashlib.sha512(m).digest() for m in inputs]
    bs = [bytes(u ^ v for u, v in zip(x, p)) for p in pads]
    return {"key_block_bytes": len(ks + salt_w), "pad_input_bytes": sorted({len(m) for m in inputs}),
            "compressions_a_pad_after_midstate": blocks(len(inputs[0])) - 1,
            "served_salt_input_compressions": blocks(64 + 8 + 8),
            "equal_x_distinct_b": len(set(bs)) == len(bs),
            "ok": len(ks + salt_w) == 128 and {len(m) for m in inputs} == {144} and blocks(144) - 1 == 1
            and len(set(bs)) == len(bs)}


def sha512_blocks(m: bytes) -> list[bytes]:
    padded = m + b"\x80" + b"\0" * ((111 - len(m)) % 128) + (8 * len(m)).to_bytes(16, "big")
    return [padded[i:i + 128] for i in range(0, len(padded), 128)]


def check_kleaf(seed: int) -> dict:
    """The keyed leaf: the digest is SHA-512(ks ‖ salt_w ‖ head ‖ row), head a 128-byte block carrying (forward, position,
    length) as row-bound's does, so it is the cascade from S with exactly one compression more than SHA-512(head ‖ row) from
    the IV, and that one is S, shared a window; equal rows at distinct positions give distinct digests; and head fixes the
    length, so the padded block sequences after S are prefix-free (Salt.lean's prefix_free_of_head)."""
    rng = random.Random(seed)
    ks, salt_w = rng.randbytes(64), rng.randbytes(64)

    def head(forward: int, position: int, n: int, length: bool = True) -> bytes:
        return (forward.to_bytes(8, "big") + position.to_bytes(8, "big")
                + (n.to_bytes(8, "big") if length else b"")).ljust(128, b"\0")

    def prefix_free(length: bool) -> bool:
        seqs = [tuple(sha512_blocks(ks + salt_w + head(f, p, len(r), length) + r)[1:])
                for f in range(2) for p in range(3) for r in rows]
        return all(a == b or a[:len(b)] != b for a in seqs for b in seqs)

    rows = [rng.randbytes(n) for n in (0, 1, 100, 111, 112, 127, 128, 239, 240, 1000, 16384)]
    r0 = rows[2]
    rows.append(r0 + sha512_blocks(ks + salt_w + head(0, 0, 0, False) + r0)[-1][len(r0) % 128:] + rng.randbytes(40))
    msgs = [head(f, p, len(r)) + r for f in range(2) for p in range(3) for r in rows]
    extra = {len(sha512_blocks(ks + salt_w + m)) - len(sha512_blocks(m)) for m in msgs}
    same_row = [hashlib.sha512(ks + salt_w + head(f, p, len(r0)) + r0).digest() for f in range(2) for p in range(3)]
    pf, pf_mutant = prefix_free(True), prefix_free(False)
    ok = len(ks + salt_w) == 128 and extra == {1} and pf and not pf_mutant and len(set(same_row)) == len(same_row)
    return {"key_block_bytes": len(ks + salt_w), "extra_compressions": sorted(extra), "messages": len(msgs),
            "prefix_free_after_s": pf, "length_free_head_extension_caught": not pf_mutant,
            "equal_rows_distinct_digests": len(set(same_row)) == len(same_row), "ok": ok}


def digests_per_forward(layout: str, rows: int) -> float:
    if layout == "L1":
        return CALLS + 1
    r = {"L0, 16 KiB rows": 16384, "L0, 256 KiB rows": 262144}[layout]
    return rows * (CALLS + WORD_BYTES_TOKEN / r)


def designs() -> dict:
    """W1 a public digest: honest-only (premise-free, and with forced hashing credited), and the hashing it adds."""
    reg, comp = PRICE_REG, PRICE_COMPRESSION
    xor = {p: DIGEST_REGS * v for p, v in reg.items()}
    fm_unc = {p: (FORM + GATHER + DIGEST_REGS) * v for p, v in reg.items()}
    fm_com = {p: (FORM + GATHER + DIGEST_REGS) * (v - reg["floor"]) for p, v in reg.items()}
    return {
        "hm96 unforced (today's gadget)": {"layouts": ["L1"], "premise_free": SALT["unforced"], "credited": SALT["unforced"],
                                           "compressions": 5, "leaf": "hm96"},
        "hm96 + (a) (N-HIDE)": {"layouts": ["L1"], "premise_free": SALT["unforced"], "credited": SALT["forced (a)"],
                                "compressions": 5, "leaf": "hm96"},
        "hm96 + (a) + FM, interface uncommitted": {"layouts": ["L1"], "premise_free": {p: SALT["unforced"][p] - SALT["forced (a)"][p]
                                                                                           + fm_unc[p] for p in reg},
                                                   "credited": fm_unc, "compressions": 5, "leaf": "hm96"},
        "hm96 + (a) + FM, interface committed": {"layouts": ["L1"], "premise_free": {p: SALT["unforced"][p] - SALT["forced (a)"][p]
                                                                                         + fm_com[p] for p in reg},
                                                 "credited": fm_com, "compressions": 5, "leaf": "hm96"},
        "pad (P1)": {"layouts": ["L1", "L0, 256 KiB rows", "L0, 16 KiB rows"],
                     "premise_free": {p: comp[p] + xor[p] for p in reg}, "credited": xor, "compressions": 1, "leaf": "pad"},
        "pad (P1), b in DistinctWrites' set": {"layouts": ["L1", "L0, 256 KiB rows", "L0, 16 KiB rows"],
                                               "premise_free": {p: comp[p] + xor[p] for p in reg},
                                               "credited": {p: DIGEST_REGS * (v - reg["floor"]) for p, v in reg.items()},
                                               "compressions": 1, "leaf": "pad"},
        "keyed digest (P2)": {"layouts": ["L1", "L0, 256 KiB rows", "L0, 16 KiB rows"],
                              "premise_free": dict(comp), "credited": {p: 0.0 for p in reg}, "compressions": 1, "leaf": "keyed"},
        "keyed leaf (P3)": {"layouts": ["L1", "L0, 256 KiB rows", "L0, 16 KiB rows"],
                            "premise_free": {p: 0.0 for p in reg}, "credited": {p: 0.0 for p in reg}, "compressions": 0,
                            "leaf": "keyed"},
    }


def window_extra() -> dict:
    """What every pad-family design adds once a window: ks's hm96 leaf (unforced) and S."""
    w = {p: SALT["unforced"][p] + PRICE_COMPRESSION[p] for p in PRICE_REG}
    return {"w1": w, "share_one_h200": {p: v / (H200_MACS * MFU_SERVE * WINDOW_S) for p, v in w.items()}}


def shares(ds: dict) -> dict:
    out = {}
    for name, d in ds.items():
        for lay in d["layouts"]:
            out[f"{name}, {lay}"] = {pt: {basis: {p: digests_per_forward(lay, rows) * d[basis][p] / (rows * MACS_TOKEN)
                                                  for p in PRICE_REG} for basis in ("premise_free", "credited")}
                                     for pt, rows in POINTS.items()}
    return out


def in_circuit() -> dict:
    """ANDs a drawn unit's proof spends hiding one public digest beyond x itself (hm96's gadget: c's 2 compressions and
    the leaf's 2 over private (b, c), M_key·y XORs only), and once a proof (ks's leaf and S). The keyed leaf's first
    block, row-bound's, is public data hashed from the secret S, so it moves one compression into the circuit."""
    hm96 = 4 * ANDS_COMPRESSION
    return {"hm96 today": {"a digest": hm96, "once": 0},
            "hm96 + (a)": {"a digest": hm96 + 3 * ANDS_FROM_8, "once": KS_OPEN},
            "hm96 + (a) + FM": {"a digest": hm96 + 3 * ANDS_FROM_8 + N_OUT * WORDS * (26 + 11), "once": KS_OPEN},
            "pad (P1)": {"a digest": ANDS_COMPRESSION, "once": KS_OPEN + ANDS_COMPRESSION},
            "keyed digest (P2)": {"a digest": ANDS_COMPRESSION, "once": KS_OPEN + ANDS_COMPRESSION},
            "keyed leaf (P3)": {"a digest": ANDS_COMPRESSION, "once": KS_OPEN + ANDS_COMPRESSION}}


def per_period(share: float) -> dict:
    out = {}
    for scope, macs_s in (("one H200", H200_MACS * MFU_SERVE), ("datacenter", DATACENTER_H200 * H200_MACS * MFU_SERVE),
                          ("world", WORLD_MACS_YEAR / PERIOD_S["year"])):
        out[scope] = {p: share * macs_s * s * AND_PER_MAC for p, s in PERIOD_S.items()}
    return out


def main() -> None:
    args = sys.argv[1:]
    n_salts = int(args[args.index("--salts") + 1]) if "--salts" in args else 256
    out_path = Path(args[0])
    fm = check_fm(n_salts, 4, 20261029)
    pad = check_pad(20261029)
    kleaf = check_kleaf(20261029)
    ds = designs()
    table = shares(ds)
    pooled = {k: {pt: {b: per_period(table[k][pt][b]["floor"]) for b in ("premise_free", "credited")}
                  for pt in ("T 512, 64 rows", "decode, 1 row")}
              for k in ("hm96 + (a) (N-HIDE), L1", "hm96 + (a) + FM, interface uncommitted, L1", "pad (P1), L1",
                        "pad (P1), L0, 16 KiB rows", "keyed leaf (P3), L0, 16 KiB rows")}
    passes = {k: {pt: {b: all(v <= TARGET for v in r[b].values()) for b in r} for pt, r in t.items()} for k, t in table.items()}
    res = {"schema": "np-salt/result/v1", "checks": {"fm": fm, "pad": pad, "kleaf": kleaf}, "designs": ds, "shares": table, "passes": passes,
           "window_extra": window_extra(), "in_circuit_ands": in_circuit(), "pooled_and_gates": pooled,
           "bases": {"macs_token": MACS_TOKEN, "word_bytes_token": WORD_BYTES_TOKEN, "a_bytes_token": A_BYTES_TOKEN,
                     "bytes_a_mac": (WORD_BYTES_TOKEN + A_BYTES_TOKEN) / MACS_TOKEN,
                     "fm_tensor_macs_a_salt": N_OUT * L_SALT, "fm_committed_bytes_a_salt": 4 * N_OUT * WORDS,
                     "fm_interface_registers": {"form": FORM, "gather": GATHER, "xor": DIGEST_REGS}}}
    res["ok"] = fm["ok"] and pad["ok"] and kleaf["ok"]
    out_path.write_text(json.dumps(res, indent=1))
    print(f"FM: literal {fm['literal']['match']}/{fm['literal']['salts']}, closed form {fm['closed_form']['match']}/"
          f"{fm['closed_form']['salts']} against hm96-sha512/v1's mask; bits 0..6 zero {fm['closed_form']['low_zero']}; "
          f"max |P| {fm['closed_form']['max_abs']}; parity mutation caught {fm['parity_mutation_caught']}")
    d = fm["distinct"]
    print(f"FM distinct: {d['distinct']}/{d['partials']} prefixes, nonzero {d['nonzero']}, min block bits {d['min_block_bits']}, "
          f"row weight {d['min_row_weight']}..{d['max_row_weight']}, mutation caught {d['mutation_caught']}")
    print(f"pad: {pad}")
    print(f"keyed leaf: {kleaf}")
    for k, t in table.items():
        print(f"{k}: " + "; ".join(f"{pt} {r['premise_free']['floor']:.4%}/{r['premise_free']['alone']:.4%} free, "
                                   f"{r['credited']['floor']:.4%}/{r['credited']['alone']:.4%} cr" for pt, r in t.items()))
    print("ok" if res["ok"] else "FAILED")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
