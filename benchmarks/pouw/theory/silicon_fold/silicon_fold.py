"""SI-FOLD: price a fused accumulate-path transcript, tau <- rho(tau xor absorbed accumulator words), for each rho.

The design this prices (FT, "fused transcript"):
- The MAC's linear accumulator is unchanged, so the useful output stays exact and is denoised exactly as today.
- A separate transcript register beside the accumulator write-back absorbs every instruction's accumulator words through
  rho. Its initial state is the call's live coin and the chain's id (one a tile, refused by the verifier if it repeats
  under a coin, so tiles with equal words don't share a chain). It never feeds back into the output.
- Every instruction that can advance the transcript costs at least one credited k-step (the trigger rule), and the
  transcript absorbs the instruction's format and shape tag.
- At the call's end each transcript state is sealed with one SHA-512 under a hiding leaf.

Inputs: `synth.json` from synth.py (areas, depths, AND counts, all checked against rho.py). Software costs of each rho
in W1 register writes (best known, word-wise or bit-sliced, and the round-structured counting floor) are stated in
SOFTWARE with their derivations. Margins and T1 are priced twice: in W1 (a write is 32 MACs) and at sm_120's measured
issue prices (a LOP3-type write 16.2 FP8 MACs, an integer add 8.29), where a software bypass is cheaper. Experiments run
bit for bit on small instances:
- E1, usefulness: the split output equals the int8 GEMM exactly after linear denoising, and in-line rho does not.
- E2, the affine lemma: an affine rho's transcript at zero data is a row term plus a column term (`decomp` reproduces it
  exactly), and every non-linear rho fails that separability test.
- E3, placement: tile-end absorption is reproduced by one C-injected instruction from `decomp`'s final words.
- E4, C-injection at per-instruction placement reproduces the transcript only by issuing every instruction.

    python3 silicon_fold.py SYNTH.json --json OUT.json [--mutations] [--verdict V.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import rho

M32 = rho.M32
TC_MACS_PER_CYCLE = 1024  # one tensor core's int8 MACs a cycle at the write-back port (H100/H200 class, estimate)
MACS_PER_WORD = 32  # a k32 step: each 32-bit accumulator word absorbs 32 MACs
DECOMP_FORM = 0.109  # COUNTER's decomp: writes a word to form a zero-data word (note 20261029T2120Z-report-hash-fold-rounds §2)
# Writes a word the cheater pays to form the words it absorbs. FT needs no operand noise, and without it zero data makes
# every word free (0, the default); with the lead's noise it is decomp's 0.109 (--form 0.109)
FORM = 0.0
W1_PER_SHA_BYTE = 883.0  # a chained SHA-512 compression's 3,532 registers over 128 B, x32 units (pouw_space price.sha-3532)
H_REST = 0.272  # the lead's A pre-noise commitment (0.203) and bound blocks (0.069), unchanged (HASH card, checked)
H_A = 0.203
H_LEAD = 127.50
H_R80X = (17.72, 18.10)
AND_PER_MAC_S8 = 2079 / 16  # the card's int8 MAC in AND gates
SHA512_MC_COMPRESSION = 57947  # Bristol Fashion sha512.txt: ANDs a compression (literature)
LEAD_MACS_PER_WORD = 27.75
CYCLE_DEPTH = 0  # gate levels a cycle; 0 takes the s8 MAC's own depth
# sm_120's measured issue prices in FP8 MACs a lane (kb/sm120 W1 price table, r20260930-060338-26b9): LOP3, PRMT, SHF
# 16.2 and IMAD 16.3 against W1's 32 a write; INT32 adds 8.29 as ptxas merges them, which also prices decomp's forming
SM120_WRITE = 16.2
SM120_ADD = 8.29
# SI-ALG's shared baseline, GE a MAC with the accumulator register (note compute-research/20261029T2215Z-report-silicon-
# baseline, run r20261029-220924-1126): s8 4-wide into s32; FP8 E4M3 into FP32 1-wide (conservative); exact FP8 4-wide
SI_ALG = {"s8": 817.9, "fp8": 2285.5, "fp8x4_k64": 841.7, "run": "r20261029-220924-1126"}


@dataclass(frozen=True)
class Rho:
    key: str
    title: str
    modules: tuple[str, ...]  # synth modules a unit holds (logic, ROM)
    state_bits: int
    absorb_words: int  # 32-bit words absorbed an invocation, natural
    c_sw: float  # best-known software writes an invocation (W1)
    c_floor: float  # counting floor in the round-structured bitwise/word model, writes an invocation
    sw_note: str
    premise: str
    rank: str
    standard: bool
    and_mc: int  # multiplicative complexity an invocation (literature or exact count), for the proof
    nonlinear: bool = True
    sw_sm120: float = 0.0  # the cheapest software invocation at sm_120's prices, FP8 MACs; 0 means c_sw LOP3-type writes

    def sm120(self) -> float:
        return self.sw_sm120 or self.c_sw * SM120_WRITE


SOFTWARE = [
    Rho("keccak1600", "Keccak-f[1600], 1 round per 1,024-bit absorption", ("keccak1600_rnd", "keccak_rcrom"), 1600, 32,
        120.0, 50.0,
        "bit-sliced over 32 instances: theta parity 640 + (A xor D) 1,600 + chi 1,600 LOP3, rho/pi renamed, iota folded "
        "into chi's truth table = 3,840 / 32 = 120; word-wise 32-bit interleaved about 175. Floor: chi's 1,600 output "
        "bits need >= 50 32-bit writes",
        "KeccakRound-1.75: no program evaluates n chained Keccak-f[1600] rounds with 1,024 absorbed bits each in under "
        "28.5 n writes (1.75x under chi's counting floor, 4.2x under the best known)", "4a (2 in the round-structured model)", True, 1600),
    Rho("keccak800", "Keccak-p[800], 1 round per 512-bit absorption", ("keccak800_rnd", "keccak_rcrom"), 800, 16,
        60.0, 25.0, "bit-sliced 1,920 / 32 = 60; word-wise about 90. Floor 800 / 32 = 25",
        "KeccakRound-1.75 at b = 800 (Keccak-p[800] is FIPS 202's family; less studied than b = 1600)", "4a- (2 structured)", True, 800),
    Rho("keccak400", "Keccak-p[400], 1 round per 256-bit absorption", ("keccak400_rnd", "keccak_rcrom"), 400, 8,
        30.0, 12.5, "bit-sliced 960 / 32 = 30. Floor 400 / 32 = 12.5",
        "KeccakRound-1.75 at b = 400", "4a- (2 structured)", True, 400),
    Rho("keccak200", "Keccak-p[200], 1 round per 128-bit absorption", ("keccak200_rnd", "keccak_rcrom"), 200, 4,
        15.0, 6.25, "bit-sliced 480 / 32 = 15. Floor 200 / 32 = 6.25",
        "KeccakRound-1.75 at b = 200", "4a- (2 structured)", True, 200),
    Rho("xoodoo", "Xoodoo, 1 round per 384-bit absorption", ("xoodoo_rnd",), 384, 12,
        28.0, 12.0, "bit-sliced (P 128 + theta/rho-west 384 + chi 384) / 32 = 28; word-wise about 53. Floor 384 / 32 = 12",
        "XoodooRound: no program evaluates chained Xoodoo rounds with 12 absorbed words each under 10.7 writes a round "
        "(1.12x under chi's floor)", "4a- (2 structured)", True, 384),
    Rho("ascon", "Ascon-p, 1 round per 320-bit absorption", ("ascon_rnd",), 320, 10,
        22.0, 10.0, "bit-sliced (S-box about 6 LOP3 a bit position x 64 + linear 320) / 32 = 22; word-wise about 53. Floor 10",
        "AsconRound under 8.9 writes a round (1.12x under the S-box floor)", "4a- (2 structured)", True, 320),
    Rho("chacha", "ChaCha double round per 16-word absorption", ("chacha_dround",), 512, 16,
        96.0, 96.0, "8 QRs x 12 writes (IADD3, LOP3, SHF or PRMT a step; no sm_120 instruction fuses two of add, xor, "
        "rotate); bit-sliced also 12 a QR",
        "ChaChaRound: no program evaluates chained ChaCha double rounds with 16 absorbed words under 14.3 writes "
        "(6.7x under the best known)", "4a (3 in the word model)", True, 8 * 4 * 31,
        sw_sm120=64 * SM120_WRITE + 32 * SM120_ADD),
    Rho("sha512", "SHA-512 round per 2-word absorption (R80's round, hardware)", ("sha512_rnd", "sha512_krom"), 512, 2,
        28.0, 28.0, "R80X's W1 count, 28 a round (S1 8, S0 8, Ch 2, Maj 2, four IADD3 pairs 8), equal to its restricted-"
        "model bound; bit-sliced 39.75",
        "ShaRound-14: no program evaluates chained SHA-512 rounds with a 64-bit word each in under 1.78 writes a round "
        "(15.7x under R80X's count and its restricted-model bound)", "4a (3 in R80X's restricted model)", True, 7 * 63 + 128,
        sw_sm120=20 * SM120_WRITE + 8 * SM120_ADD),
    Rho("aes", "AES round with the absorbed 128 bits as round key", ("aes_rnd",), 128, 4,
        40.0, 16.0, "T-tables: 16 index extractions + 16 shared-memory loads + 8 LOP3; bit-sliced about 45, the cheaper "
        "on sm_120 where an LDS lookup costs 66. Floor: 16 S-box outputs need >= 16 writes",
        "AesRound under 3.6 writes a round; and the chip holds AES elsewhere (copy engines' AES-GCM)", "4a-", True, 16 * 32,
        sw_sm120=45 * SM120_WRITE),
    Rho("speck64", "Speck64 round per word (the cheapest standard ARX)", ("speck64_rnd",), 64, 1,
        4.0, 3.0, "word-wise 5 (SHF/PRMT, IADD3, LOP3, SHF, LOP3); bit-sliced 4 (add 2 with the key xor fused, xor 1, "
        "rotations renamed). Floor 3",
        "SpeckRound under 0.89 writes a round (4.5x under the best known)", "4a- (ISO/IEC 29167-22; little studied for cost)", True, 31),
    Rho("simon64", "Simon64 round per word", ("simon64_rnd",), 64, 1,
        2.0, 2.0, "bit-sliced 2 LOP3 a bit plane ((a & b) xor c, then xor y xor k); word-wise 5",
        "SimonRound under 0.89 writes a round (2.2x under the best known)", "4a- (ISO/IEC 29167-21)", True, 32),
    Rho("nv_arx", "our own per-lane ARX, ((t + w) <<< 7) xor t", ("nv_arx",), 32, 1,
        3.0, 2.0, "IADD3, SHF, LOP3; bit-sliced add 2 + xor 1", "NvArx: our own mixer has no shortcut", "5", False, 31,
        sw_sm120=2 * SM120_WRITE + SM120_ADD),
    Rho("affine", "affine control, t * a + w mod 2^32", ("affine",), 32, 1,
        1.0, 0.0, "one IMAD; and `decomp` evaluates it at zero data with no per-cell work (E2)",
        "none: refuted by E2", "refuted", False, 0, nonlinear=False, sw_sm120=16.3),
]


def _ge_dff(synth: dict) -> float:
    return synth["meta"]["ge_dff"]


def mac_baselines(synth: dict) -> dict:
    m = synth["modules"]
    dff = _ge_dff(synth)
    s8 = (m["mac4_s8"]["ge"] + 32 * dff) / 4
    fp8 = m["fp8_mac"]["ge"] + 32 * dff
    base = {
        "s8_ge_per_mac": round(s8, 2),
        "fp8_ge_per_mac": round(fp8, 2),
        "s8_depth": m["mac4_s8"]["depth"],
        "fp8_depth": m["fp8_mac"]["depth"],
        "label": "checked RTL, synthesized (yosys + ABC, generic gates, Nangate45 GE); SI-ALG's baseline supersedes it",
    }
    if synth["meta"].get("flow") == "si-alg":
        base.update({
            "s8_ge_per_mac": SI_ALG["s8"], "fp8_ge_per_mac": SI_ALG["fp8"], "fp8x4_k64_ge_per_mac": SI_ALG["fp8x4_k64"],
            "own_rtl_s8_ge_per_mac": round(s8, 2), "own_rtl_fp8_ge_per_mac": round(fp8, 2),
            "label": f"SI-ALG's baseline (run {SI_ALG['run']}); every rho mapped in SI-ALG's flow (generic.lib), checked",
        })
    return base


def adopt_lib_flow(synth: dict, lib: dict) -> dict:
    """Take each module's area from SI-ALG's flow; depth and AND counts stay from the generic-gate mapping."""
    for name, m in synth["modules"].items():
        if name not in lib["modules"]:
            sys.exit(f"silicon_fold.py: {name} missing from the SI-ALG-flow synthesis")
        lm = lib["modules"][name]
        m["ge_own_flow"], m["ge"] = m["ge"], lm["ge"]
        m["checked"] = m["checked"] and lm["checked"]
    synth["meta"]["flow"] = "si-alg"
    synth["meta"]["selftest"] = synth["meta"]["selftest"] + lib["meta"]["selftest"]
    return synth


def price(r: Rho, synth: dict, base: dict, margin: float) -> dict:
    """Area, timing, margins and h for one rho, every word absorbed and at a common margin."""
    m = synth["modules"]
    dff = _ge_dff(synth)
    logic = sum(m[x]["ge"] for x in r.modules if not x.endswith("rom"))
    rom = sum(m[x]["ge"] for x in r.modules if x.endswith("rom"))
    depth = max(m[x]["depth"] for x in r.modules if not x.endswith("rom"))
    stages = max(1, math.ceil(depth / (CYCLE_DEPTH or base["s8_depth"])))
    unit = logic + rom + stages * r.state_bits * dff
    need = 1.0 - FORM  # software writes an absorbed MAC-word the cheater must exceed
    words_per_cycle = TC_MACS_PER_CYCLE / MACS_PER_WORD
    every = {
        "units_per_tc": words_per_cycle / r.absorb_words,
        "ge_per_mac": unit * words_per_cycle / r.absorb_words / TC_MACS_PER_CYCLE,
        "margin_known": r.c_sw / (r.absorb_words * need),
        "margin_floor": r.c_floor / (r.absorb_words * need) if r.c_floor else 0.0,
    }
    need_sm120 = MACS_PER_WORD - FORM * SM120_ADD  # the same saving in sm_120 MACs
    every["margin_known_sm120"] = r.sm120() / (r.absorb_words * need_sm120)
    every["margin_floor_sm120"] = r.c_floor * r.sm120() / r.c_sw / (r.absorb_words * need_sm120) if r.c_floor else 0.0

    def at(covered: float) -> dict:
        prefold = max(0.0, covered - r.absorb_words) * 32 * 2.0  # an XOR2 a bit to pre-fold extra words
        return {"covered_words": covered, "ge_per_mac": (unit + prefold) / (covered * MACS_PER_WORD)}

    at_m = at(r.c_sw / (margin * need))  # MAC-words an invocation may cover at this margin
    at_m_sm120 = at(r.sm120() / (margin * need_sm120))
    for d in (every, at_m, at_m_sm120):
        d["pct_s8"] = 100 * d["ge_per_mac"] / base["s8_ge_per_mac"]
        d["pct_fp8"] = 100 * d["ge_per_mac"] / base["fp8_ge_per_mac"]
    return {
        "ge_logic": round(logic, 1),
        "ge_rom": round(rom, 1),
        "depth": depth,
        "stages": stages,
        "ge_unit": round(unit, 1),
        "ge_per_write": round(unit / r.c_sw, 1),
        "every_word": {k: round(v, 4) for k, v in every.items()},
        f"margin_{margin:g}": {k: round(v, 4) for k, v in at_m.items()},
        f"margin_{margin:g}_sm120": {k: round(v, 4) for k, v in at_m_sm120.items()},
        "and_per_mac_every_word": round(r.and_mc / (r.absorb_words * MACS_PER_WORD), 3),
        "ands_synth": sum(m[x]["ands"] for x in r.modules if not x.endswith("rom")),
    }


def h_of(seal_state_bytes: int, macs_per_state: float, a_in_hw: bool) -> dict:
    seal = seal_state_bytes * W1_PER_SHA_BYTE / macs_per_state
    rest = H_REST - (H_A if a_in_hw else 0.0)
    return {"h": round(rest + seal, 4), "seal": round(seal, 6), "rest": rest}


def t1_attacks(r: Rho, covered_words: float, placement: str, k: int, fast_ratio: float, trigger_rule: bool) -> dict:
    """The cheapest cheater a credited MAC-word, in W1 writes (honest: 1, the k-step's own write, rho free in hardware).

    Strategies: honest; zero data, the words formed at FORM a word, and rho in software; C-injection through the MAC
    path; the tile-end skip (one C-injected instruction a tile); a faster-format trigger when the trigger rule is absent;
    linear batching when rho is affine (E2)."""
    honest = 1.0
    s = {"honest": honest}
    s["zero data+software"] = FORM + r.c_sw / covered_words if r.nonlinear else FORM
    s["c-inject"] = 1.0 + 1.0 / MACS_PER_WORD  # the instruction plus materializing C (1 write a word over 32 MACs' credit)
    if placement in ("tile-end", "call-end"):
        s["skip (one injected instruction a tile)"] = (FORM + 1.0) * MACS_PER_WORD / k
    if not trigger_rule:
        s["faster-format trigger"] = 1.0 / fast_ratio
    best = min(s, key=s.get)
    return {"strategies": {x: round(v, 4) for x, v in s.items()}, "best": best, "t1": round(max(0.0, honest - s[best]), 4)}


def t1_sm120(r: Rho, covered_words: float) -> dict:
    """Per-instruction placement with the trigger rule at sm_120's prices, in FP8 MACs a credited word (honest: 32)."""
    honest = float(MACS_PER_WORD)
    form = FORM * SM120_ADD
    s = {"honest": honest, "zero data+software": form + (r.sm120() / covered_words if r.nonlinear else 0.0),
         "c-inject": honest * (1.0 + 1.0 / MACS_PER_WORD)}
    best = min(s, key=s.get)
    return {"strategies": {x: round(v, 4) for x, v in s.items()}, "best": best,
            "t1": round(max(0.0, 1.0 - s[best] / honest), 4)}


# ---------------------------------------------------------------------------------------------------------------------
# Experiments (bit for bit)

def _zero_data_words(rng: np.random.Generator, T: int, rows: int, cols: int) -> np.ndarray:
    """The lead's zero-data words w_t(i, j) = R_t(i) + C_t(class(i), j) mod 2^32, two row-parity classes."""
    R = rng.integers(0, 1 << 32, size=(T, rows), dtype=np.uint64)
    C = rng.integers(0, 1 << 32, size=(T, 2, cols), dtype=np.uint64)
    w = (R[:, :, None] + C[:, np.arange(rows) % 2, :]) & M32
    return w.astype(np.uint64), R, C


def _chain(rho_key: str, words: list[int]) -> int:
    """A per-cell chain of `rho_key` over one cell's words (each absorbed in lane 0), from a fixed IV; 32-bit digest."""
    if rho_key == "affine":
        t = 0x243F6A88
        for w in words:
            t = rho.affine(t, w)
        return t
    if rho_key == "nv_arx":
        t = 0x243F6A88
        for w in words:
            t = rho.nv_arx(t, w)
        return t
    if rho_key in ("speck64", "simon64"):
        f = rho.speck64_round if rho_key == "speck64" else rho.simon64_round
        x, y = 0x243F6A88, 0x85A308D3
        for w in words:
            x, y = f(x, y, w)
        return x
    if rho_key == "chacha":
        s = list(range(16))
        for w in words:
            s[0] ^= w
            s = rho.chacha_diagonal_round(rho.chacha_column_round(s))
        return s[0]
    if rho_key == "sha512":
        st = list(rho.SHA512_IV)
        for t, w in enumerate(words):
            st = rho.sha512_round(st, w, rho.SHA512_K[t % 80])
        return st[0] & M32
    if rho_key == "aes":
        s = bytes(16)
        for w in words:
            s = rho.aes_round(s, w.to_bytes(4, "little") + bytes(12))
        return int.from_bytes(s[:4], "little")
    if rho_key == "ascon":
        x = [1, 2, 3, 4, 5]
        for t, w in enumerate(words):
            x[0] ^= w
            x = rho.ascon_round(x, ((0xF - t % 12) << 4) | (t % 12))
        return x[0] & M32
    if rho_key == "xoodoo":
        a = [[0] * 4 for _ in range(3)]
        for t, w in enumerate(words):
            a[0][0] ^= w
            a = rho.xoodoo_round(a, rho.XOODOO_RC[t % 12])
        return a[0][0]
    if rho_key.startswith("keccak"):
        b = int(rho_key[6:])
        lw = b // 25
        lanes = [0] * 25
        for t, w in enumerate(words):
            lanes[0] ^= w & ((1 << lw) - 1)
            lanes[1] ^= (w >> lw) & ((1 << lw) - 1) if lw < 32 else 0
            lanes = rho.keccak_round(lanes, rho.KECCAK_RC64[t % 24], lw)
        return lanes[0] & M32
    raise KeyError(rho_key)


def e2_affine_lemma(rng: np.random.Generator, T: int = 24, rows: int = 8, cols: int = 6) -> dict:
    """An affine rho's zero-data transcript is f(i) + g(j): decomp computes it from T(rows + cols) terms, exactly.
    Separability test on every rho: tau(i,j) - tau(i,0) - tau(0,j) + tau(0,0) == 0 for all (i, j) of one class."""
    w, R, C = _zero_data_words(rng, T, rows, cols)
    out = {}
    a = rho.AFFINE_A
    # decomp for affine: tau_T = a^T iv + sum_t a^(T-1-t) R_t(i) + sum_t a^(T-1-t) C_t(class, j)
    iv = 0x243F6A88
    pw = [pow(a, T - 1 - t, 1 << 32) for t in range(T)]
    fr = [sum(pw[t] * int(R[t, i]) for t in range(T)) % (1 << 32) for i in range(rows)]
    gc = [[sum(pw[t] * int(C[t, c, j]) for t in range(T)) % (1 << 32) for j in range(cols)] for c in range(2)]
    base = pow(a, T, 1 << 32) * iv % (1 << 32)
    exact = 0
    for i in range(rows):
        for j in range(cols):
            honest = _chain("affine", [int(w[t, i, j]) for t in range(T)])
            exact += honest == (base + fr[i] + gc[i % 2][j]) % (1 << 32)
    out["affine_decomp_exact"] = f"{exact}/{rows * cols}"
    out["affine_decomp_terms"] = T * (rows + 2 * cols)
    out["honest_terms"] = T * rows * cols
    sep = {}
    for r in SOFTWARE:
        tau = [[_chain(r.key, [int(w[t, i, j]) for t in range(T)]) for j in range(cols)] for i in range(rows)]
        bad = 0
        n = 0
        for i in range(2, rows, 2):  # same class as row 0
            for j in range(1, cols):
                n += 1
                bad += (tau[i][j] - tau[i][0] - tau[0][j] + tau[0][0]) % (1 << 32) != 0
        sep[r.key] = {"non_separable": f"{bad}/{n}", "separable": bad == 0}
    out["separability"] = sep
    out["checks"] = {
        "affine reproduced exactly by decomp": exact == rows * cols,
        "affine separable": sep["affine"]["separable"],
        "every non-linear rho non-separable on every tested cell": all(
            v["non_separable"].split("/")[0] == v["non_separable"].split("/")[1] for k, v in sep.items() if k != "affine"
        ),
    }
    return out


def _keccak_transcript(words_by_step: list[list[int]], iv_coin: int, chain_id: int = 0) -> list[int]:
    """Cross-lane FT-Keccak: each step absorbs up to 32 words (1,024 bits) into lanes 0..15, then one round."""
    lanes = [0] * 25
    lanes[23] = chain_id & rho.M64
    lanes[24] = iv_coin & rho.M64
    for t, ws in enumerate(words_by_step):
        for q in range(0, len(ws), 32):
            chunk = ws[q : q + 32] + [0] * (32 - len(ws[q : q + 32]))
            for k in range(16):
                lanes[k] ^= chunk[2 * k] | (chunk[2 * k + 1] << 32)
            lanes = rho.keccak_round(lanes, rho.KECCAK_RC64[t % 24], 64)
    return lanes


def e1_usefulness(rng: np.random.Generator, M: int = 16, N: int = 8, K: int = 128) -> dict:
    """int8 GEMM with additive operand noise, denoised linearly, beside a split FT-Keccak transcript; and in-line rho."""
    A = rng.integers(-128, 128, size=(M, K), dtype=np.int64)
    B = rng.integers(-128, 128, size=(N, K), dtype=np.int64)
    EA = rng.integers(-128, 128, size=(M, K), dtype=np.int64)
    EB = rng.integers(-128, 128, size=(N, K), dtype=np.int64)
    An, Bn = A + EA, B + EB  # noised operands (wider than s8 in this model; the lead's noise layout differs)
    acc = np.zeros((M, N), dtype=np.int64)
    inl = np.zeros((M, N), dtype=np.int64)
    steps = []
    for t in range(K // 32):
        P = An[:, 32 * t : 32 * t + 32] @ Bn[:, 32 * t : 32 * t + 32].T
        acc = (acc + P) % (1 << 32)
        steps.append([int(x) for x in acc.ravel()])
        inl = (inl + P) % (1 << 32)
        inl = np.vectorize(lambda v: rho.nv_arx(int(v), 0x9E3779B9))(inl).astype(np.int64)
    tau = _keccak_transcript(steps, 0x0123456789ABCDEF)
    corr = (EA @ B.T + A @ EB.T + EA @ EB.T) % (1 << 32)
    y = (acc - corr) % (1 << 32)
    exact = (A @ B.T) % (1 << 32)
    y_inl = (inl - corr) % (1 << 32)
    tau2 = _keccak_transcript(steps[:-1] + [[s ^ (1 if i == 0 else 0) for i, s in enumerate(steps[-1])]], 0x0123456789ABCDEF)
    tau3 = _keccak_transcript(steps, 0x0123456789ABCDEE)
    return {
        "split_output_exact": f"{int((y == exact).sum())}/{M * N}",
        "inline_output_exact": f"{int((y_inl == exact).sum())}/{M * N}",
        "transcript_changes_with_one_word": tau2 != tau,
        "transcript_changes_with_the_coin": tau3 != tau,
        "checks": {
            "split output exact after linear denoising": bool((y == exact).all()),
            "in-line rho breaks the output": int((y_inl == exact).sum()) == 0,
            "transcript binds a word flip (sanity; binding is the seal's job)": tau2 != tau,
            "transcript depends on the live coin": tau3 != tau,
        },
    }


def e3_placement(rng: np.random.Generator, T: int = 64, rows: int = 4, cols: int = 4) -> dict:
    """Tile-end absorption: the cheater forms the final words by decomp and issues one instruction with C = w_T.
    Per-instruction absorption: the cheater must issue T instructions (every step's word is absorbed)."""
    w, R, C = _zero_data_words(rng, T, rows, cols)
    final = [int(w[T - 1, i, j]) for i in range(rows) for j in range(cols)]
    honest_tile_end = _keccak_transcript([final], 7)
    cheat_words = [(int(R[T - 1, i]) + int(C[T - 1, i % 2, j])) & M32 for i in range(rows) for j in range(cols)]
    cheat_tile_end = _keccak_transcript([cheat_words], 7)
    every = [[int(w[t, i, j]) for i in range(rows) for j in range(cols)] for t in range(T)]
    honest_every = _keccak_transcript(every, 7)
    short = _keccak_transcript([cheat_words], 7)  # one injected instruction in place of T
    return {
        "tile_end_cheat_exact": honest_tile_end == cheat_tile_end,
        "tile_end_instructions": {"honest": T, "cheater": 1},
        "every_step_one_instruction_matches": honest_every == short,
        "checks": {
            "tile-end placement reproduced by one injected instruction (skip)": honest_tile_end == cheat_tile_end,
            "per-instruction placement not reproduced by one instruction": honest_every != short,
        },
    }


def e4_cinject(rng: np.random.Generator, T: int = 16, rows: int = 4, cols: int = 4) -> dict:
    """C-injection at per-instruction placement: A = B = 0, C = decomp's word each step reproduces the transcript, with
    T instructions issued, the honest count; it saves no instruction."""
    w, R, C = _zero_data_words(rng, T, rows, cols)
    honest = _keccak_transcript([[int(w[t, i, j]) for i in range(rows) for j in range(cols)] for t in range(T)], 9)
    inj = []
    for t in range(T):
        inj.append([(int(R[t, i]) + int(C[t, i % 2, j])) & M32 for i in range(rows) for j in range(cols)])
    cheat = _keccak_transcript(inj, 9)
    return {"cinject_exact": honest == cheat, "instructions": {"honest": T, "cheater": T},
            "checks": {"C-injection reproduces the transcript only with every instruction issued": honest == cheat}}


def e5_keccak_bitsliced(rng: np.random.Generator) -> dict:
    """The cheater's best-known software for FT-Keccak, run: Keccak-f[1600]'s round bit-sliced over 32 instances, one
    32-bit register a (lane, z) bit, every 3-input LOP3 counted as one write, rho/pi as renaming, iota folded into chi's
    truth table. Its output must equal rho.keccak_round on every instance."""
    n = 32
    states = [[int(x) for x in rng.integers(0, 1 << 63, size=25, dtype=np.uint64)] for _ in range(n)]
    states = [[x | ((i & 1) << 63) for x in s] for i, s in enumerate(states)]
    msgs = [[int(x) for x in rng.integers(0, 1 << 63, size=16, dtype=np.uint64)] for _ in range(n)]
    t = 5
    bits = {}
    for x in range(5):
        for y in range(5):
            for z in range(64):
                bits[(x, y, z)] = sum(((states[i][x + 5 * y] >> z) & 1) << i for i in range(n))
    mb = {(i5 % 5, i5 // 5, z): sum(((msgs[i][i5] >> z) & 1) << i for i in range(n)) for i5 in range(16) for z in range(64)}
    writes = {"absorb": 0, "theta_parity": 0, "theta_apply": 0, "chi": 0}
    for key, v in mb.items():
        bits[key] ^= v
        writes["absorb"] += 1
    C = {}
    for x in range(5):
        for z in range(64):
            C[(x, z)] = bits[(x, 0, z)] ^ bits[(x, 1, z)] ^ bits[(x, 2, z)] ^ bits[(x, 3, z)] ^ bits[(x, 4, z)]
            writes["theta_parity"] += 2
    A = {}
    for x in range(5):
        for y in range(5):
            for z in range(64):
                A[(x, y, z)] = bits[(x, y, z)] ^ C[((x - 1) % 5, z)] ^ C[((x + 1) % 5, (z - 1) % 64)]
                writes["theta_apply"] += 1
    B = {}
    for x in range(5):
        for y in range(5):
            r = rho.KECCAK_ROT[x][y]
            for z in range(64):
                B[(y, (2 * x + 3 * y) % 5, (z + r) % 64)] = A[(x, y, z)]
    full = (1 << n) - 1
    rc = rho.KECCAK_RC64[t]
    out = {}
    for x in range(5):
        for y in range(5):
            for z in range(64):
                v = B[(x, y, z)] ^ (~B[((x + 1) % 5, y, z)] & full & B[((x + 2) % 5, y, z)])
                if x == 0 and y == 0 and (rc >> z) & 1:
                    v ^= full
                out[(x, y, z)] = v
                writes["chi"] += 1
    ok = 0
    for i in range(n):
        lanes = [states[i][k] ^ (msgs[i][k] if k < 16 else 0) for k in range(25)]
        want = rho.keccak_round(lanes, rc, 64)
        got = [sum(((out[(k % 5, k // 5, z)] >> i) & 1) << z for z in range(64)) for k in range(25)]
        ok += got == want
    round_only = (writes["theta_parity"] + writes["theta_apply"] + writes["chi"]) / n
    return {
        "instances_exact": f"{ok}/{n}",
        "writes": writes,
        "writes_per_instance_round": round_only,
        "writes_per_instance_with_absorb": round_only + writes["absorb"] / n,
        "checks": {
            "bit-sliced Keccak round equals the reference on all 32 instances": ok == n,
            "bit-sliced round is 120 writes an instance (the c_sw the margin uses)": round_only == 120.0,
        },
    }


# ---------------------------------------------------------------------------------------------------------------------


def run(synth: dict, margin: float, k: int, fast_ratio: float) -> dict:
    base = mac_baselines(synth)
    rng = np.random.default_rng(20261029)
    designs = {}
    for r in SOFTWARE:
        p = price(r, synth, base, margin)
        cov_every = r.absorb_words
        t1_every = t1_attacks(r, cov_every, "per-instruction", k, fast_ratio, trigger_rule=True)
        t1_tile = t1_attacks(r, cov_every, "tile-end", k, fast_ratio, trigger_rule=True)
        t1_notrig = t1_attacks(r, cov_every, "per-instruction", k, fast_ratio, trigger_rule=False)
        # seal: one state a tensor-core transcript a tile; a CTA tile 128 x 256 at K, split over 4 tensor cores
        tile_macs = 128 * 256 * k
        h = h_of(r.state_bits // 8 * 4, tile_macs, a_in_hw=False)
        h_hw = h_of(r.state_bits // 8 * 4, tile_macs, a_in_hw=True)
        designs[r.key] = {
            "title": r.title,
            "premise": r.premise,
            "rank": r.rank,
            "standard": r.standard,
            "software": {"c_sw": r.c_sw, "c_floor": r.c_floor, "note": r.sw_note},
            "area": p,
            "t1": {"every_word": t1_every, "tile_end": t1_tile, "no_trigger_rule": t1_notrig,
                   "every_word_sm120": t1_sm120(r, cov_every)},
            "h": h,
            "h_with_hw_A_hash": h_hw,
        }
    models = {"w1": (1.0, 1.0 - FORM), "sm120": (SM120_WRITE, MACS_PER_WORD - FORM * SM120_ADD)}

    def at_margin(r: Rho, model: str, M: float) -> dict:
        """One invocation covering as many words as margin M allows (pre-folded by XOR2s past its natural width)."""
        cost = r.c_sw if model == "w1" else r.sm120()
        words = cost / (M * models[model][1])
        unit = designs[r.key]["area"]["ge_unit"]
        g = (unit + max(0.0, words - r.absorb_words) * 32 * 2.0) / (words * MACS_PER_WORD)
        return {"words_per_round": round(words, 1), "ge_per_mac": round(g, 2),
                "pct_s8": round(100 * g / base["s8_ge_per_mac"], 2), "pct_fp8": round(100 * g / base["fp8_ge_per_mac"], 2)}

    kr = SOFTWARE[0]
    floor_one = kr.c_sw / kr.c_floor  # the known margin at which the counting floor's margin is exactly 1
    trade = {f"{model} {M:.2f}": at_margin(kr, model, M) for model in models for M in (1.1, floor_one, 4.0)}
    every = 120.0 / 32 / (1 - FORM)
    trade[f"w1 {every:.2f} (every word)"] = at_margin(kr, "w1", every)
    cheapest = {}
    for model in models:
        pts = {r.key: at_margin(r, model, 1.1) for r in SOFTWARE if r.standard}
        best = min(pts, key=lambda x: pts[x]["ge_per_mac"])
        cheapest[model] = {"rho": best, **pts[best], "all": pts}
    lead_and = AND_PER_MAC_S8 + SHA512_MC_COMPRESSION / 32 / LEAD_MACS_PER_WORD
    exps = {
        "e1_usefulness": e1_usefulness(rng),
        "e2_affine_lemma": e2_affine_lemma(rng),
        "e3_placement": e3_placement(rng),
        "e4_cinject": e4_cinject(rng),
        "e5_keccak_bitsliced": e5_keccak_bitsliced(rng),
    }
    return {
        "baseline": base,
        "params": {"margin": margin, "K": k, "fast_ratio": fast_ratio, "tc_macs_per_cycle": TC_MACS_PER_CYCLE,
                   "form": FORM, "decomp_form": DECOMP_FORM, "w1_per_sha_byte": W1_PER_SHA_BYTE, "h_rest": H_REST,
                   "sm120_write": SM120_WRITE, "sm120_add": SM120_ADD},
        "proof": {"lead_and_per_mac": round(lead_and, 2), "mac_and_per_mac": round(AND_PER_MAC_S8, 2)},
        "designs": designs,
        "keccak_margin_trade": trade,
        "cheapest_at_margin_1.1": cheapest,
        "experiments": exps,
    }


def checks(res: dict, synth: dict) -> list[tuple[str, bool]]:
    c = []
    c.append(("every module's netlist equals rho.py on its vectors", all(m["checked"] for m in synth["modules"].values())))
    c.append(("rho.py's standard rounds equal their primitives", all(ok for _, ok in synth["meta"]["selftest"])))
    for e in res["experiments"].values():
        c += list(e["checks"].items())
    d = res["designs"]
    c.append(("keccak1600 every-word T1 is 0", d["keccak1600"]["t1"]["every_word"]["t1"] == 0))
    c.append(("tile-end placement fails T1 (> 90% at K)", d["keccak1600"]["t1"]["tile_end"]["t1"] > 0.9))
    c.append(("no trigger rule fails T1 (faster format)", d["keccak1600"]["t1"]["no_trigger_rule"]["t1"] > 0.4))
    c.append(("affine control fails T1", d["affine"]["t1"]["every_word"]["t1"] > 0.85))
    c.append(("every standard rho's h under R80X's", all(v["h"]["h"] < H_R80X[0] for v in d.values() if v["standard"])))
    c.append(("keccak margin above 1 at its counting floor", d["keccak1600"]["area"]["every_word"]["margin_floor"] > 1))
    c.append(("keccak1600 every-word margin above 1 at sm_120's prices",
              d["keccak1600"]["area"]["every_word"]["margin_known_sm120"] > 1))
    c.append(("keccak1600 every-word T1 is 0 at sm_120's prices", d["keccak1600"]["t1"]["every_word_sm120"]["t1"] == 0))
    c.append(("affine control fails T1 at sm_120's prices", d["affine"]["t1"]["every_word_sm120"]["t1"] > 0.4))
    # dominance: keccak1600 every-word is the least area among standard rho with margin_known >= 4 every-word, or least at the common margin
    mk = f"margin_{res['params']['margin']:g}"
    top = {k: v for k, v in d.items() if v["rank"].split(" ")[0] == "4a"}
    least = min(top, key=lambda k: top[k]["area"][mk]["ge_per_mac"])
    c.append((f"least area at the common margin among rank-4a rho is keccak1600 ({least})", least == "keccak1600"))
    std = {k: v for k, v in d.items() if v["standard"]}
    floor = min(v["area"][mk]["ge_per_mac"] for v in std.values())
    c.append(("keccak1600 within 10% of the least area of any standard rho at the common margin",
              d["keccak1600"]["area"][mk]["ge_per_mac"] <= 1.1 * floor))
    return c


def mutations(synth: dict, margin: float, k: int, fast_ratio: float) -> list[tuple[str, bool]]:
    """Each mutation must make some check fail."""
    out = []
    import copy

    s2 = copy.deepcopy(synth)
    s2["modules"]["keccak1600_rnd"]["checked"] = False
    out.append(("an unchecked netlist", not all(ok for _, ok in checks(run(s2, margin, k, fast_ratio), s2))))
    saved = SOFTWARE[0]
    SOFTWARE[0] = Rho(**{**saved.__dict__, "nonlinear": False})
    r = run(synth, margin, k, fast_ratio)
    out.append(("keccak treated as linear", not all(ok for _, ok in checks(r, synth))))
    SOFTWARE[0] = saved
    saved_aff = SOFTWARE[-1]
    SOFTWARE[-1] = Rho(**{**saved_aff.__dict__, "nonlinear": True, "c_sw": 30.0})
    r = run(synth, margin, k, fast_ratio)
    out.append(("affine credited as non-linear", not all(ok for _, ok in checks(r, synth))))
    SOFTWARE[-1] = saved_aff
    r = run(synth, margin, 32, fast_ratio)
    out.append(("K = 32 makes tile-end look sound", not all(ok for _, ok in checks(r, synth))))
    return out


def table(res: dict) -> str:
    mk = f"margin_{res['params']['margin']:g}"
    L = ["| rho | GE unit | depth | stages | GE a forced write | every word: GE/MAC, % s8, % FP8, margin known / floor | "
         f"at margin {res['params']['margin']:g}: GE/MAC, % s8 | sm_120 prices: margin known / floor, at margin "
         f"{res['params']['margin']:g} % s8, T1 every word | h | T1 every word / tile-end / no trigger rule | AND/MAC | rank |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["designs"].items():
        a = v["area"]
        e = a["every_word"]
        m = a[mk]
        ms = a[mk + "_sm120"]
        t = v["t1"]
        L.append(
            f"| {k} | {a['ge_unit']:.0f} | {a['depth']} | {a['stages']} | {a['ge_per_write']:.0f} | "
            f"{e['ge_per_mac']:.1f}, {e['pct_s8']:.1f}%, {e['pct_fp8']:.1f}%, {e['margin_known']:.2f} / {e['margin_floor']:.2f} | "
            f"{m['ge_per_mac']:.1f}, {m['pct_s8']:.1f}% | "
            f"{e['margin_known_sm120']:.2f} / {e['margin_floor_sm120']:.2f}, {ms['pct_s8']:.1f}%, "
            f"{100 * t['every_word_sm120']['t1']:.1f}% | {v['h']['h']:.3f} | "
            f"{100 * t['every_word']['t1']:.1f}% / {100 * t['tile_end']['t1']:.1f}% / {100 * t['no_trigger_rule']['t1']:.1f}% | "
            f"{a['and_per_mac_every_word']:.2f} | {v['rank']} |"
        )
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("synth")
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--fast-ratio", type=float, default=2.0)
    ap.add_argument("--cycle-depth", type=int, default=0)
    ap.add_argument("--mutations", action="store_true")
    ap.add_argument("--verdict")
    ap.add_argument("--form", type=float, default=0.0, help="writes a word to form absorbed words (0.109: the lead's noise)")
    ap.add_argument("--synth-lib", help="synth.py --lib output: price every area in SI-ALG's flow against its baseline")
    a = ap.parse_args()
    synth = json.loads(Path(a.synth).read_text())
    if a.synth_lib:
        synth = adopt_lib_flow(synth, json.loads(Path(a.synth_lib).read_text()))
    global CYCLE_DEPTH, FORM
    CYCLE_DEPTH, FORM = a.cycle_depth, a.form
    res = run(synth, a.margin, a.K, a.fast_ratio)
    cs = checks(res, synth)
    res["checks"] = [{"name": n, "ok": ok} for n, ok in cs]
    if a.mutations:
        ms = mutations(synth, a.margin, a.K, a.fast_ratio)
        res["mutations"] = [{"name": n, "caught": ok} for n, ok in ms]
    tab = table(res)
    print(json.dumps(res["baseline"]), file=sys.stderr)
    print(tab)
    for n, ok in cs:
        print(("ok    " if ok else "FAIL  ") + n)
    for m in res.get("mutations", []):
        print(("caught " if m["caught"] else "MISSED ") + m["name"])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    if a.md:
        Path(a.md).write_text(tab + "\n")
    ok = all(ok for _, ok in cs) and all(m["caught"] for m in res.get("mutations", []))
    if a.verdict:
        Path(a.verdict).write_text(json.dumps({"ok": ok, "checks": len(cs), "failed": [n for n, o in cs if not o],
                                               "mutations": res.get("mutations", [])}, indent=1))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
