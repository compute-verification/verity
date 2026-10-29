"""The premise ledger for PoUW's frontier designs, their minimal premise sets, and the price of each premise class.

Every premise any frontier design uses or could use is one row of `LEDGER`: its statement, rank (0 theorem, 1 measured
hardware floor, 2 CPU-checkable fact, 3 model faithfulness or protocol premise, 4a algorithmic knowledge about
SHA-512, 4b about our own noised operands, 5 hardness conjecture; an open premise ranks 5 today), how to falsify or
measure it on the H200, its status, its owner and the T1 terms it closes.

T1 is CORE §2a's additive gap in W1 units, as a share of stock's int8 MACs. Each design's terms come from SW-NOISE's
calculator (`noise_sweep.gap_terms`: forming, de-noise, the final word, duplicates, the coins). Added here:

- **hashing**: SHA-512 of every committed byte, (row blocks + 12) compressions an hm96 leaf, at a compression's W1
  floor (3,632 registers written at 32) or its measured price alone. Under ShaSpeed and hash-compose the honest and the
  adversary pay the same and the term is 0, to within δ. Without them the best rank ≤ 3 lower bound is the
  compression's 16 output registers (H32 with DistinctWrites), so the term is h·(1 − 16·32/W1).
- **pre-noise hashing**: A's commitment before its noise, A's bytes a useful MAC (1/n), at the same price. The atomic
  call puts it inside the call; ShaSpeed prices it.
- **hiding**: each leaf's salt side (the PRF, M_key·y and c), honest-only, at SW-HASH's measured share of a leaf's hash
  time (SW-PRICE's card), for the committed words' leaves and A's pre-noise leaves (one activation row a leaf on the
  card; a call's A packed into big leaves under the big-rows option, at the smallest leaf size that passes).
- **coin branch**: the adversary may branch on coin bits, and a branch writes no register, so W1 prices it at 0.
  `DistinctWritesMH100On` then charges each path only the words distinct on that path's salts. For each row-step it
  computes the noise class once and skips every row-step whose n words equal an earlier one at this salt. The term is
  α times the expected share of row-steps skipped, simulated exactly on the words for ssn_1l and its hardened variant
  ssn_1l_hw (16 code-lane coin bits, step coins a_t ∈ {1, 3} against public column signs).

Other sections: ShaSpeed's precision at each point and what Daniel's 02:01Z ruling covers, what rank ≤ 3 overall would
take, G1 against advice, energy as a pricing (excluded by AGENTS.md; reported only as what it would change), per-period
totals. `--json` writes every number; the exit code is 0 iff every check passes; `--mutations` runs mutations that the
checks must catch.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_sweep as NS  # noqa: E402

H32 = 32
W1_COMPRESSION = {"floor": 116_224, "alone": 232_888}       # SW-HASH: 3,632 registers at 32; measured alone
SHA_INSTRUCTIONS = 3_646
SHA_OUT_REGS = 16                                           # a compression's 512-bit output, 32-bit registers
ANDS_COMPRESSION = 58_120
ANDS_MAC = 2079 / 16
NCP_BYTES_MAC = 3 * 4 / 32
ROWS = {"16 KiB": 16_384, "256 KiB": 262_144, "1 MiB": 1_048_576}
CARD_ROW = "256 KiB"
#: SW-HASH's measured hm96 leaves, as SW-PRICE's card holds them (`hash_rows`): a leaf's hash time in ns, the salt
#: side's hiding (PRF, M_key·y and c) and M_key·y alone as shares of it, and the data kernel's ns a compression. A leaf
#: under 16 KiB hides at 16 KiB's per-leaf price (SW-PRICE's estimate); past 1 MiB a leaf's hiding grows by its last
#: measured step a 4× (estimate)
HIDE_ROWS = (16_384, 65_536, 262_144, 1_048_576)
HIDE_NS_LEAF = (31.787, 125.18, 494.424, 1989.076)
HIDE_SHARE = (0.0553, 0.0151, 0.0043, 0.0012)
MASK_SHARE = (0.0345, 0.0096, 0.0029, 0.0008)
COMP_NS = 0.2255
BIG_ROWS = tuple(2 ** e for e in range(18, 31))          # 256 KiB … 1 GiB, the big-rows option's leaf sizes
HBM_BYTES = 141e9
TDP_W, INT8_MAC_S, INT32_OPS_S = 700.0, 989.5e12, 132 * 64 * 1.98e9    # data sheet; INT32 lanes at f_max
ALLOW = 0.01
POINTS = {"llama_T512": (NS.LLAMA, 64, 1.0), "llama_512rows": (NS.LLAMA, 512, 1.0), "llama_T8192": (NS.LLAMA, 8192, 1.0),
          "v4d_T512": (NS.V4_DENSE, 64, 1.0), "v4d_T8192": (NS.V4_DENSE, 8192, 1.0),
          "v4e_T8192": (NS.V4_EXPERT, 8192, NS.V4_EXPERT_SHARE)}
RANK_ORDER = ["0", "1", "2", "3", "4a", "4b", "5", "F"]

# --------------------------------------------------------------------------------------------------------- the ledger
# (id, statement, rank once settled, status, rank today, falsify / measure on the H200, owner, T1 terms it closes)
LEDGER = [
    ("H32", "Every instruction costs at least 32 W1 units per 32-bit register it writes", "1", "measured", "1",
     "per-instruction probes through the SASS gate (h200Measured = h100Floor, 232 probes)", "CORE-H32",
     "base of every word's credit"),
    ("H32-multi", "No multi-output instruction (IMAD.WIDE, IADD3 with carry out, HMMA pairs) writes two target "
     "registers for one instruction's 32", "1", "open", "5",
     "probe IMAD.WIDE and IADD3.X writing two live registers a cycle; price per register written", "N-W1",
     "base: if q target registers come at one price, credit per word falls to 32/q"),
    ("FloorTc", "No tensor instruction writes an s8 partial-sum register under 32; k32 is the finest full-rate step",
     "1", "measured", "1", "mma.sync and wgmma N 8–256 probes", "CORE-H32", "k32 commitment at no premium"),
    ("W1", "Every program the H200 runs is a well-formed MH100w program costing at least its h200Measured entries: "
     "every register written is an Op100 output at its class floor, moves are oblivious, operands carry free values. "
     "MH100w programs are straight-line: a branch writes no register and is priced 0, so a program that branches on "
     "the coins is charged per path (coin-branch)",
     "3", "stated", "3", "every class's probes; opcode census; per-SM scaling", "SW-SEM / N-W1",
     "base: DistinctWritesMH100Floor is about the model without it"),
    ("ISA-census", "The opcode census of sm_90a is complete: no unit, opcode or mode outside it",
     "3", "open", "5", "census of nvdisasm's opcode list, each class probed; undocumented encodings stay unprovable "
     "(the vendor is untrusted), so completeness stays rank 3", "N-W1", "base (part of W1 and CardWithinAccel)"),
    ("CardWithinAccel", "The card refines the ACCEL instance: no hidden units, 132 SMs, every opcode", "3", "locked",
     "3", "per-SM scaling, census", "lock", "base"),
    ("Sm90Sem", "Every class computes its documented function (tensor, lanes, memory path), stated per class",
     "3", "stated", "3", "class captures; opcode list incomplete (IMNMX, SEL, VABSDIFF4, IDP.2A, I2F/F2I, MUFU, "
     "FLO/BREV/BMSK/SGXT, ISETP/PLOP3, uniform datapath, IMAD.WIDE)", "SW-SEM",
     "raises the rank ≤ 3 credit a word from 32 to 48–60 (L 64–256) with AlgSuffL"),
    ("TensorSemantics", "A tensor output register is the captured exact s8 k32 dot of its tiles plus its accumulator",
     "3", "stated", "3", "re-capture; an output ≠ intDot falsifies", "SW-SEM", "forming (with every class semantic)"),
    ("LaneSemantics", "Each lane instruction computes its documented function exactly", "3", "partly", "3",
     "lane captures per opcode", "SW-SEM", "forming (with every class semantic)"),
    ("FanIn", "An instruction reads at most 3 register sources plus immediates; an MMA reads its tiles", "2", "stated",
     "2", "SASS encoding census", "SW-SEM", "ingredient of Sm90Sem; closes no frontier term alone"),
    ("MemOblivious", "TMA, LDGSTS and copies move whole words without transforming them; the memory path computes "
     "only atomics and reductions at ≥ 32 a word", "3", "partly", "3", "memory-side probes", "CORE-H32",
     "ingredient of TensorSemantics; no free forming in the memory path"),
    ("AlgSuffL", "MH100w programs meeting every class's semantics cost at least the least bilinear accumulator program "
     "with the same outputs", "3", "stated", "3", "a nonlinear-lane route cheaper than bilinear falsifies", "SW-SEM",
     "BamFloor's 64 − 1024/L credit a word at rank 3"),
    ("BamFloor", "At interval L a committed word costs at least 64 − 1024/L (one merge a word, its auxiliary register "
     "its own)", "2", "sketch", "5", "CPU check of the merge step", "SW-SEM", "uncommitted steps at L > 32, rank 3"),
    ("PureLanes", "The adversary's tensor lanes hold noise-pure operands (each lane one operand entry plus noise)",
     "4b", "stated", "4b", "a mixed-lane route cheaper than CoverCount's", "SW-SEM",
     "with CoverCount: the full L credit a word up to L 128"),
    ("CoverCount", "On PureLanes, a committed L-sum needs L/32 tensor writes", "0", "sketch", "5",
     "a proof", "SW-SEM", "uncommitted steps at L ≤ 128 (with PureLanes, rank 4b)"),
    ("CellPure", "Each tensor cell computes one noised operand pair's product", "4b", "stated", "4b",
     "a cell route mixing pairs", "SW-SEM", "with PureCount: the full L credit a word"),
    ("PureCount", "On CellPure, the L-sum's products are counted one a cell", "0", "sketch", "5", "a proof", "SW-SEM",
     "uncommitted steps (with CellPure, rank 4b)"),
    ("NoSharingGain", "A committed L-sum of int7 data costs at least L/32 tensor writes of 32", "5", "open", "5",
     "a cheaper L-sum route (likely false from L 256: Hopcroft–Kerr ties at L 128)", "SW-SEM",
     "uncommitted steps at any L"),
    ("SAK-U", "No program computes a product of uniform operands with less work than the best known", "4b", "stated",
     "4b", "a faster algorithm on uniform matrices", "N-SAKU", "matmul part of SAK-U designs; nothing in ssn_1l"),
    ("SAKMatmul", "Symmetric algorithmic knowledge on our noise distribution", "4b", "locked", "4b", "—", "lock",
     "the baseline; nothing in ssn_1l"),
    ("ssn-distinct", "ssn_1l's words D_t − β + Σ a₁C + Σ a_t are pairwise distinct and non-free as functions of the "
     "coin bits, for every A and B, each lane read from its own bit of the row's coin words",
     "0", "proved (scratch)", "0",
     "— (SW-LEAN's ssn1lBit_charged and ssnABit_charged, straight-line programs, n ≤ 28,559; no lock)",
     "SW-LEAN / N-LEAN", "ssn_1l's and ssn_a's word credit"),
    ("ssn-hw-distinct", "ssn_1l_hw's words (a₁ ∈ {0..15}⁴ from the nibble mask on coin word 0; step lanes 1 + 2b_t "
     "against public column signs s_t[j]) are pairwise distinct and non-free as functions of the coin bits",
     "0", "open", "5", "CPU-checked on 7 data cases (this file); a proof extends ssnTargetG to nibble lanes on a "
     "shared coin word and to affine step lanes", "N-LEAN", "ssn_1l_hw's word credit"),
    ("ssn-success-set", "Fixing g coin bits succeeds with probability 2^−g a try and merges ≤ g·n words", "0", "open",
     "5", "a proof; it bounds a program fixed in advance, not one that branches on the bits (coin-branch)", "N-LEAN",
     "grinding for a fixed program"),
    ("coin-branch", "The expected words a coin-branching program skips (rows-steps equal at its salt) is bounded by "
     "the design's per-salt collision share", "0", "open", "5",
     "this file's exact per-salt simulation; a proof of the per-path bound", "N-PREM",
     "the coin-branch term (prover-chosen identical or zero rows)"),
    ("final-word", "The free values are the call's own words, not their span; the final words are then targets",
     "0", "proved (scratch)", "0", "—", "SW-LEAN", "final word (exact-output designs), given atomic-call"),
    ("formed-charged", "A formed register that moves with the salt, equals no coin word and differs pairwise is "
     "charged 32", "0", "proved (scratch)", "0", "—", "SW-LEAN", "the forced part of forming (C1)"),
    ("weights-distinct", "The registered weights have no zero or equal columns where a design's words need them distinct",
     "0", "open", "5", "a check of the committed weights inside the proof (measured: none on Llama-3.2-1B)", "SW-PRICE",
     "os2's and per2's word credit on degenerate weights"),
    ("per2-distinct", "per2's words are pairwise distinct and non-free except the final word", "0", "open", "5",
     "a proof", "N-LEAN", "per2's word credit"),
    ("ncp-indep", "NCP's checked words are linearly independent modulo the free values", "0", "locked (A2WordsIff)",
     "0", "—", "lock", "NCP's word credit"),
    ("OneInstructionMinimal", "No single sm_90 integer lane instruction forms NCP's packed X register from A and a raw "
     "XOF word", "2", "checked (55/55)", "2", "CPU check over integer lane families", "SW-PREMISES",
     "NCP's derivation (with a sharing lemma)"),
    ("SignedFormMinimal", "No chain of two lane instructions forms a packed signed B + F register", "2",
     "checked (LOP3→IADD3 only)", "5", "CPU check over pairs", "SW-PREMISES", "NCP's sign fix"),
    ("P1Int", "An s8 MMA's dot is the exact integer sum in s32 in any order (determinism)", "1", "measured", "1",
     "captures, 10.5 M elements, 0 mismatches", "lock", "none (completeness)"),
    ("P2Local", "An MMA output changes only with its row, column and accumulator", "0", "proved", "0", "—", "lock",
     "ingredient of TensorSemantics"),
    ("SheetClock", "0 < clock ≤ f_max = 1,980 MHz", "1", "locked", "1", "clock under load", "lock",
     "none in T1 (time only)"),
    ("G1", "size + held ≤ mem (HBM 141 GB, L2 50 MiB, registers, shared memory)", "1", "locked", "1",
     "capacity probe", "lock", "none: M0 frees advice of any size (this file's G1 section)"),
    ("G2", "Each pipe's use ≤ rate × wall; ≤ 4 warp instructions an SM-clock", "1", "locked", "1", "rate probes",
     "lock", "none in T1 (time only)"),
    ("G3", "Oracle calls × use ≤ use", "1", "locked", "1", "SHA-512 throughput", "lock", "hashing's accounting"),
    ("RegFileBW", "About 256 operand reads an SM-clock", "3", "stated", "3", "operand-read probes", "CORE-TIME",
     "none in T1 (time only)"),
    ("ShaSpeed", "No program computes SHA-512 cheaper than the best implementation, to within δ", "4a", "locked", "4a",
     "a faster SHA-512 kernel or a cheaper W1 program", "lock",
     "hashing, credited coins, pre-noise hashing, forced PRF"),
    ("hash-compose", "A program hashing N bytes and writing M committed noise-dependent words costs ≥ N(1 − δ)·price "
     "+ 32·M: hashing shares no work with the words", "4a", "stated", "4a",
     "a SHA-512 whose written words double as targets", "N-COMPOSE", "every credited byte, with ShaSpeed"),
    ("SHA-LB", "A rank-2 lower bound on SHA-512's W1 cost tight to δ* (a fraction of an instruction a compression)",
     "2", "open", "5", "CPU-checked count in Sm90Sem", "N-SHALB",
     "would replace ShaSpeed; the best rank ≤ 3 bound today is 16 output registers, 0.44%"),
    ("mask-optimal", "No program computes M_key·y with less work than the best honest program (same memory)", "4a",
     "stated", "4a", "a 384 MiB table over 16-bit chunks gives 96 lookups a leaf; as worded likely false", "N-HIDE",
     "hiding's M·y with the PRF forced"),
    ("forced-PRF", "The proof checks the salt PRF against a committed seed, so the PRF is committed work", "0",
     "design", "0", "—", "N-HIDE", "hiding's PRF part (then priced under ShaSpeed)"),
    ("hiding-ruled", "Hiding work is credited by ruling (a choice of target, not a premise)", "R", "pending Daniel",
     "R", "—", "Daniel", "hiding"),
    ("big-rows", "hm96 leaves of R ≥ R* bytes, so honest-only hiding ≤ the allowance left", "0", "design", "0",
     "arithmetic; costs a leaf's opening", "N-HIDE", "hiding, at no premise"),
    ("atomic-call", "Each A·B is atomic: work done before the noise inside the call is charged", "3",
     "ruled (Daniel 4:49 PM)", "3", "— (a scoping of the game)", "Daniel",
     "final word (with final-word), pre-noise hashing (with ShaSpeed and hash-compose); T1 needs only its own-card "
     "part, so not PeriodInventory"),
    ("live-coins", "The window's salt is drawn by the challenger after the roots it reads are fixed", "3", "rule",
     "3", "— (protocol structure)", "AGENTS.md", "every noise term"),
    ("RootAfterCoin", "A root committed after a coin arrives no earlier than coin + the work's wall time", "3",
     "locked", "3", "certifier timestamps", "lock", "none in priced work"),
    ("SeedRO", "seed_j = SHA-512(tag ‖ salt ‖ … ‖ root(A_j) ‖ root(B_j)) behaves as fresh coins: μ = 1/2 a bit", "C",
     "ruled (AGENTS.md exception)", "C", "—", "AGENTS.md", "grinding's μ (crypto axis, every design)"),
    ("PeriodInventory", "Every device the prover runs in the period is counted on both sides", "3", "stated", "3",
     "inventory and egress", "SW-PREMISES",
     "none in T1: the egress layer's (devices off the card, with OutsourceWithinEgress)"),
    ("OutsourceWithinEgress", "Off-card compute is bounded by the egress the certifiers see", "3", "locked", "3",
     "certifiers", "lock", "none in T1 (egress layer)"),
    ("inputs-natural", "The prover does not choose its inputs (no batches of identical or zero rows)", "F", "false",
     "F", "a self-dealing prover submits its own requests", "—", "coin-branch on prover-chosen data"),
    ("energy", "A measured J per op per class prices work", "X", "excluded (AGENTS.md)", "X",
     "— (no meter enters a premise)", "—", "none: forming and hashing ×r, credit unchanged (this file)"),
]
FIELDS = ["id", "statement", "rank", "status", "rank_today", "falsify", "owner", "closes"]


def ledger() -> list[dict]:
    return [dict(zip(FIELDS, row)) for row in LEDGER]


def rank_max(ranks) -> str:
    ranked = [r for r in ranks if r in RANK_ORDER]
    return max(ranked, key=RANK_ORDER.index) if ranked else "0"


# -------------------------------------------------------------------------------------------- hashing and its prices
def blocks(r: int) -> int:
    return (r + 17 + 127) // 128


def comps_byte(r: int) -> float:
    return (blocks(r) + 12) / r


def weighted_alpha(design, shapes: dict, share: float = 1.0) -> float:
    w = sum(n * k for n, k in shapes.values())
    return sum(n * k * design.alpha(k) for n, k in shapes.values()) / w


def a_bytes_mac(shapes: dict) -> float:
    w = sum(n * k for n, k in shapes.values())
    return sum(n * k / n for n, k in shapes.values()) / w


def hash_terms(design, shapes: dict, m: int, c1: bool, row: str = CARD_ROW) -> dict:
    """The design's hashed bytes a useful MAC, and hashing's W1 work a useful MAC at the floor and alone: the words on
    `row` leaves, A before its noise on one activation row (K bytes) a leaf, as SW-PRICE's r256k card lays it out."""
    w = sum(n * k for n, k in shapes.values())
    words_b = sum(n * k * NS.hash_bytes_per_mac(design, m, n, k, c1) for n, k in shapes.values()) / w
    pre_b = a_bytes_mac(shapes)
    pre_c = sum(k * comps_byte(k) for n, k in shapes.values()) / w
    words_c = words_b * comps_byte(ROWS[row])
    out = {"bytes_mac": words_b, "pre_bytes_mac": pre_b}
    for p, wc in W1_COMPRESSION.items():
        h = (words_c + pre_c) * wc
        out[f"h_{p}"] = h
        out[f"words_{p}"] = words_c * wc
        out[f"pre_{p}"] = pre_c * wc
        out[f"rank3_residual_{p}"] = h * (1 - SHA_OUT_REGS * H32 / wc)
    out["h_ands"] = (words_c + pre_c) * ANDS_COMPRESSION / ANDS_MAC
    return out


def leaf_hiding(leaf: float, share=HIDE_SHARE) -> float:
    """A leaf's salt side (or with MASK_SHARE, its M_key·y) in compressions, interpolated in log2 of the leaf size."""
    per = [ns * s / COMP_NS for ns, s in zip(HIDE_NS_LEAF, share)]
    x = math.log2(max(leaf, HIDE_ROWS[0]))
    xs = [math.log2(r) for r in HIDE_ROWS]
    if x >= xs[-1]:
        return per[-1] + (per[-1] - per[-2]) * (x - xs[-1]) / 2
    i = max(j for j in range(len(xs)) if xs[j] <= x)
    return per[i] + (per[i + 1] - per[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])


def hiding_terms(bytes_mac: float, shapes: dict, m: int, row: int = ROWS[CARD_ROW], a_pack: int | None = None) -> dict:
    """Honest-only hiding a useful MAC as a share of stock: the committed words on `row` leaves, and A's pre-noise leaves,
    one activation row a leaf (`a_pack` None, the card's layout) or a call's A packed in leaves of `a_pack` bytes."""
    w = sum(n * k for n, k in shapes.values())
    out = {}
    for name, share in (("unforced", HIDE_SHARE), ("my", MASK_SHARE)):
        words = bytes_mac / row * leaf_hiding(row, share)
        if a_pack is None:
            a = sum(leaf_hiding(k, share) for n, k in shapes.values()) / w
        else:
            a = sum(math.ceil(m * k / a_pack) * leaf_hiding(min(a_pack, m * k), share) for n, k in shapes.values()) / (m * w)
        for p, wc in W1_COMPRESSION.items():
            out[f"{name}_{p}"] = (words + a) * wc
            out[f"{name}_words_{p}"] = words * wc
            out[f"{name}_a_{p}"] = a * wc
    return out


# --------------------------------------------------------------------------------------------- the coin-branch term
_CODE_FULL = {}


def _code(n: int, n_full: int = 2048) -> np.ndarray:
    """The column code at n columns spread evenly over a layer of n_full, so every digit varies as in the layer."""
    if n_full not in _CODE_FULL:
        _CODE_FULL[n_full] = NS.ssn_a_w0(n_full, NS.DATA_PER_STEP)[0]
    cols = np.linspace(0, n_full - 1, n).round().astype(int)
    return _CODE_FULL[n_full][:, cols]


_KEY = np.random.default_rng(99).integers(1, 2 ** 62, (64, 2), dtype=np.int64)


def _unique_rows(words: np.ndarray) -> int:
    """Distinct rows, by two 64-bit random linear keys (wrapping); a false merge has probability about 2^-120."""
    k = words @ _KEY[:words.shape[1]]
    return np.unique(k.view([("a", np.int64), ("b", np.int64)]).ravel()).shape[0]


@functools.lru_cache(maxsize=None)
def branch_collapse(variant: str, case: str, m: int, s: int, n: int = 16, salts: int = 3, seed: int = 7,
                    data_sd: float = 300.0) -> float:
    """Expected share of row-steps whose n words equal an earlier row-step's at the same salt (all rows of the call).

    Words are computed exactly: data dot D[i, t, j] plus the noise part. ssn_1l: a₁ ∈ {0,1}⁴ against the column code,
    a_t ∈ {0,1} against 1. ssn_1l_hw: a₁ ∈ {0..15}⁴ against the code, a_t = 1 + 2b_t against public signs s_t[j]. ssn_a: 4 bits a step,
    step t's lane (j + t) mod 4 against 1 in column j.
    Mutations: ssn_1l_hw_even (a_t = b_t, signs kept), ssn_1l_hw_nosign (a_t = 1 + 2b_t, all signs +1).
    Cases: identical (every row the same random data), zero (every row zero), pair (rows identical in pairs), random.
    """
    rng = np.random.default_rng(seed)
    code = _code(n)
    sign_rng = np.random.default_rng(12345)
    signs = sign_rng.choice(np.array([-1, 1]), size=(s, n))
    if variant.endswith("nosign"):
        signs = np.ones((s, n), dtype=np.int64)
    if case == "zero":
        d = np.zeros((m, s, n), dtype=np.int64)
    elif case == "identical":
        d = np.broadcast_to(np.rint(rng.normal(0, data_sd, (1, s, n))).astype(np.int64), (m, s, n))
    elif case == "pair":
        base = np.rint(rng.normal(0, data_sd, ((m + 1) // 2, s, n))).astype(np.int64)
        d = np.repeat(base, 2, axis=0)[:m]
    else:
        d = np.rint(rng.normal(0, data_sd, (m, s, n))).astype(np.int64)
    dcum = d.cumsum(axis=1)
    shares = []
    for _ in range(salts):
        if variant == "ssn_a":
            a1 = rng.integers(0, 2, (m, 4))
            bits = rng.integers(0, 2, (m, s, 4))
            lane = (np.arange(n)[None, :] + np.arange(s)[:, None]) % 4
            step = np.take_along_axis(bits, np.broadcast_to(lane[None], (m, s, n)), axis=2)
            step[:, 0] = 0
        elif variant == "ssn_1l":
            a1 = rng.integers(0, 2, (m, 4))
            at = rng.integers(0, 2, (m, s))
            at[:, 0] = 0
            step = at[:, :, None] * np.ones((1, 1, n), dtype=np.int64)
        else:
            a1 = rng.integers(0, 16, (m, 4))
            b = rng.integers(0, 2, (m, s))
            at = b if variant.endswith("even") else 1 + 2 * b
            at[:, 0] = 0
            step = at[:, :, None] * signs[None]
        noise = (a1 @ code)[:, None, :] + step.cumsum(axis=1)
        words = (dcum + noise).reshape(m * s, n).astype(np.int64)
        uniq = _unique_rows(words)
        shares.append(1 - uniq / (m * s))
    return float(np.mean(shares))


def pair_collision_ssn1l(s: int) -> float:
    """ssn_1l, two identical rows: the mean over steps of P(their words coincide at a salt): a₁ equal (1/16) and
    equal step-bit sums, C(2(t−1), t−1)/4^(t−1)."""
    return sum(math.comb(2 * (t - 1), t - 1) / 4 ** (t - 1) for t in range(1, s + 1)) / s / 16


def coin_branch_terms(data: dict) -> dict:
    """The coin-branch share of stock at each point: natural data (WikiText scopes) and prover-chosen batches."""
    d = NS.designs()
    ssn = next(x for x in d if x.name == "ssn_1l")
    ssa = next(x for x in d if x.name == "ssn_a")
    out = {}
    for point, (shapes, m, share) in POINTS.items():
        rows = max(1, round(m * share))
        alpha = weighted_alpha(ssn, shapes)
        scope = NS.DATA_SCOPE[min(NS.DATA_SCOPE, key=lambda r: abs(math.log(max(rows, 1) / r)))]
        summ = data["summary"][scope] if data else {"zero_block_share": 0.0, "identical_row_share": 0.0}
        w = sum(n * k for n, k in shapes.values())
        pairc = sum(n * k * pair_collision_ssn1l(NS.steps1l(k)) for n, k in shapes.values()) / w
        natural = alpha * (0.5 * summ["zero_block_share"] + 0.5 * summ["identical_row_share"] * pairc)
        e = {"rows": rows, "alpha": alpha, "scope": scope, "natural_ssn_1l": natural, "natural_ssn_1l_hw": 0.0}
        for variant, design, steps in (("ssn_1l", ssn, NS.steps1l), ("ssn_1l_hw", ssn, NS.steps1l),
                                       ("ssn_a", ssa, NS.steps28)):
            for case in ("identical", "zero"):
                acc = sum(n_ * k * branch_collapse(variant, case, rows, steps(k), salts=2) for n_, k in shapes.values())
                e[f"chosen_{case}_{variant}"] = weighted_alpha(design, shapes) * acc / w
        out[point] = e
    return out


def hw_coin_words(s: int) -> int:
    """ssn_1l_hw's coin words a row: a₁'s 16 nibble bits share word 0 with 16 step bits, then S − 1 step bits in all."""
    return math.ceil((16 + s - 1) / 32)


def hw_noise_std(k: int, n: int = 2048) -> dict:
    """Served noise std (accumulator units) for ssn_1l and ssn_1l_hw: code lanes' variance plus the steps'."""
    code = _code(min(n, 28_560)).astype(float)
    c2 = (code ** 2).sum(axis=0)
    s = NS.steps1l(k)
    v1l = c2 / 4 + (s - 1) / 4
    vhw = c2 * (16 ** 2 - 1) / 12 + (s - 1) * 1.0
    return {"K": k, "ssn_1l": float(np.sqrt(v1l).mean()), "ssn_1l_hw": float(np.sqrt(vhw).mean())}


def hw_distinct_check(salts: int = 24, m: int = 4, n: int = 6, s: int = 4, seed: int = 3) -> dict:
    """ssn_1l_hw's words as functions on a salt grid (SW-NOISE's DistinctOn reading) on each data case."""
    rng = np.random.default_rng(seed)
    code = _code(n)
    signs = np.random.default_rng(12345).choice(np.array([-1, 1]), size=(s, n))
    res = {}
    for case in NS.CASES:
        a, b = NS._data(case, m, n, 32 * s, rng)
        dd = np.stack([(a[:, 32 * t:32 * t + 31] @ b[32 * t:32 * t + 31]) for t in range(s)], axis=1).cumsum(axis=1)
        ws = []
        for _ in range(salts):
            a1 = rng.integers(0, 16, (m, 4))
            at = 1 + 2 * rng.integers(0, 2, (m, s))
            at[:, 0] = 0
            ws.append(dd + (a1 @ code)[:, None, :] + (at[:, :, None] * signs[None]).cumsum(axis=1))
        words = np.stack(ws)
        raw = rng.integers(0, 2 ** 32, (salts, m, s))
        res[case] = NS._distinct([words], [raw])
    return res


# ------------------------------------------------------------------------------------------ hitting sets per design
DESIGNS = [  # (label, noise_sweep name, model, C1, base premises (word credit), coin-branch applies)
    ("ssn_1l", "ssn_1l", "M0", False, ("H32", "W1", "ssn-distinct", "ssn-success-set"), "ssn_1l"),
    ("ssn_1l+C1", "ssn_1l", "M0", True, ("H32", "W1", "ssn-distinct", "ssn-success-set", "formed-charged"), "ssn_1l"),
    ("ssn_1l_hw", "ssn_1l", "M0", False, ("H32", "W1", "ssn-hw-distinct", "ssn-success-set", "coin-branch"),
     "ssn_1l_hw"),
    ("ssn_a", "ssn_a", "M0", False, ("H32", "W1", "ssn-distinct", "ssn-success-set"), "ssn_a"),
    ("per2 M1+C1", "per2", "M1", True, ("H32", "W1", "per2-distinct", "formed-charged"), None),
    ("os2 M1+C1", "os2", "M1", True, ("H32", "W1", "weights-distinct", "formed-charged"), None),
    ("ncp3 M1+C1", "ncp3", "M1", True, ("H32", "W1", "ncp-indep", "formed-charged"), None),
    ("nsf", "nsf", "M0", False, ("H32", "W1"), "data"),
]
#: the options a minimal set chooses among: each closes terms; ranks are the ledger's settled ranks
OPTIONS = {
    "final-word+atomic-call": (("final-word", "atomic-call"), ("final_word",)),
    "ShaSpeed+hash-compose": (("ShaSpeed", "hash-compose"), ("hashing", "coins", "pre_noise")),
    "atomic-call (pre-noise)": (("atomic-call",), ()),
    "hiding: ruled": (("hiding-ruled",), ("hiding",)),
    "hiding: forced PRF + mask-optimal": (("forced-PRF", "mask-optimal", "ShaSpeed"), ("hiding",)),
    "hiding: big rows": (("big-rows",), ("hiding",)),
    "inputs-natural": (("inputs-natural",), ("coin_branch_chosen",)),
}


def design_terms(label: str, point: str, data: dict, cb: dict) -> dict:
    name, model, c1 = next((n, mo, c) for lab, n, mo, c, _, _ in DESIGNS if lab == label)
    branch = next(b for lab, _, _, _, _, b in DESIGNS if lab == label)
    d = next(x for x in NS.designs() if x.name == name)
    shapes, m, share = POINTS[point]
    t = NS.weighted(d, shapes, m, model, c1, share)
    rows = max(1, round(m * share))
    h = hash_terms(d, shapes, rows, c1)
    hid = hiding_terms(h["bytes_mac"], shapes, rows)
    big = {r: hiding_terms(h["bytes_mac"], shapes, rows, row=r, a_pack=r)["unforced_floor"] for r in BIG_ROWS}
    e = cb[point]
    if branch == "data":
        # a data-dependent design repeats every later word of an all-zero row in any program (estimate: zero rows)
        w = sum(n * k for n, k in shapes.values())
        chosen, natural = sum(n * k * d.alpha(k) * (1 - H32 / k) for n, k in shapes.values()) / w, 0.0
    elif branch:
        chosen = max(e.get(f"chosen_identical_{branch}", 0.0), e.get(f"chosen_zero_{branch}", 0.0))
        natural = e.get(f"natural_{branch}", 0.0)
    else:
        chosen = natural = 0.0
    return {"forming": t["forming_x"] + t["forming_y"], "denoise": t["denoise"], "final_word": t["final_word"],
            "dup": t["dup"], "coins": t["coins"],
            "hashing": h["words_floor"] * (1 - SHA_OUT_REGS * H32 / W1_COMPRESSION["floor"]), "pre_noise": h["pre_floor"],
            "hiding": hid["unforced_floor"], "coin_branch_natural": natural, "coin_branch_chosen": chosen,
            "_h_floor": h["h_floor"], "_h_alone": h["h_alone"], "_bytes_mac": h["bytes_mac"], "_hiding": hid,
            "_hiding_big": big}


def minimal_sets(label: str, data: dict, cb: dict) -> dict:
    base = next(b for lab, _, _, _, b, _ in DESIGNS if lab == label)
    terms = {p: design_terms(label, p, data, cb) for p in POINTS}
    opts = list(OPTIONS)
    passing = []
    for r in range(len(opts) + 1):
        for combo in itertools.combinations(opts, r):
            closed = set(itertools.chain.from_iterable(OPTIONS[o][1] for o in combo))
            prem = set(itertools.chain.from_iterable(OPTIONS[o][0] for o in combo)) | set(base)
            if "pre_noise" in closed and "atomic-call" not in prem:
                closed.discard("pre_noise")
            res = {p: sum(v for k, v in t.items() if not k.startswith("_") and k not in closed) for p, t in terms.items()}
            leaf = None
            if "hiding: big rows" in combo:
                # big rows close hiding only at a leaf size where what remains of it fits: the smallest such size
                leaf = next((r for r in BIG_ROWS if all(res[p] + terms[p]["_hiding_big"][r] <= ALLOW for p in res)),
                            None)
                if leaf is None:
                    continue
                res = {p: v + terms[p]["_hiding_big"][leaf] for p, v in res.items()}
            if all(v <= ALLOW for v in res.values()):
                if any(set(c) <= set(combo) for c, *_ in passing):
                    continue
                passing.append((combo, res, leaf))
    led = {r["id"]: r for r in ledger()}
    out = []
    for combo, res, leaf in passing:
        prem = sorted(set(itertools.chain.from_iterable(OPTIONS[o][0] for o in combo)) | set(base))
        out.append({"options": [f"{o} ({leaf / 2 ** 20:g} MiB leaves, A packed a call)" if o == "hiding: big rows"
                                else o for o in combo], "premises": prem,
                    "rank_settled": rank_max(led[p]["rank"] for p in prem),
                    "rank_today": rank_max(led[p]["rank_today"] for p in prem),
                    "matmul_rank_settled": rank_max(led[p]["rank"] for p in base),
                    "matmul_rank_today": rank_max(led[p]["rank_today"] for p in base),
                    "big_rows_leaf": leaf, "residual": res})
    out.sort(key=lambda x: (RANK_ORDER.index(x["rank_settled"]) if x["rank_settled"] in RANK_ORDER else 9,
                            len(x["premises"])))
    every = set(itertools.chain.from_iterable(o[1] for o in OPTIONS.values()))
    best = {p: sum(v for k, v in t.items() if not k.startswith("_") and k not in every) for p, t in terms.items()}
    return {"terms": terms, "minimal_sets": out, "every_option_residual": best}


# --------------------------------------------------------------------------------------------- the other classes
def shaspeed_precision(data: dict, cb: dict) -> dict:
    out = {}
    for point in POINTS:
        t = design_terms("ssn_1l", point, data, cb)
        rest = t["forming"] + t["denoise"] + t["final_word"] + t["dup"] + t["coin_branch_natural"]
        budget = ALLOW - rest
        e = {"budget": budget}
        for p in W1_COMPRESSION:
            h = t[f"_h_{p}"]
            e[f"h_{p}"] = h
            e[f"delta_{p}"] = budget / h
            e[f"instr_compression_{p}"] = budget / h * SHA_INSTRUCTIONS
        out[point] = e
    return out


def rank3_routes() -> dict:
    """What rank ≤ 3 overall would take for a word-commit design: hashing in T1 with only rank ≤ 3 bounds on SHA-512."""
    wc = W1_COMPRESSION["floor"]
    lb = SHA_OUT_REGS * H32 / wc
    cb = comps_byte(ROWS[CARD_ROW])
    w1_byte = cb * wc
    max_bytes = 0.0099 / (w1_byte * (1 - lb))
    credit = {c: 4 / c * w1_byte for c in (32, 48, 60, 64, 128, 256)}
    # every design commits its outputs (one 4 B word a dot product, K MACs): hashing them alone, and the δ ShaSpeed
    # then needs with nothing else in T1
    outputs = {}
    for point, (shapes, _, _) in POINTS.items():
        w = sum(n * k for n, k in shapes.values())
        h = sum(n * 4 * w1_byte for n, k in shapes.values()) / w
        outputs[point] = {"hash_share": h, "delta_star": ALLOW / h, "instr_compression": ALLOW / h * SHA_INSTRUCTIONS}
    return {"rank3_sha_lb_share": lb, "w1_byte_floor": w1_byte, "max_bytes_mac_rank3": max_bytes,
            "min_macs_per_4B_word_rank3": 4 / max_bytes, "hash_of_stock_at_credit": credit,
            "outputs_only": outputs, "state_lemma_share_hypothetical": (80 * 4 + 64 * 2 + SHA_OUT_REGS) / 3_632}


def g1_advice() -> dict:
    """Advice sizes against HBM: one call's data dots and outputs, and a whole forward's, at the lead's points."""
    out = {}
    d = next(x for x in NS.designs() if x.name == "ssn_1l")
    for point, (shapes, m, share) in POINTS.items():
        rows = max(1, round(m * share))
        layers = NS.LLAMA_LAYERS if shapes is NS.LLAMA else 1
        call_dt = max(rows * n * NS.steps1l(k) * 4 for n, k in shapes.values())
        fwd_dt = sum(rows * n * NS.steps1l(k) * 4 for n, k in shapes.values()) * layers
        call_out = max(rows * n * 4 for n, _ in shapes.values())
        fwd_out = sum(rows * n * 4 for n, _ in shapes.values()) * layers
        out[point] = {"call_data_dots_B": call_dt, "forward_data_dots_B": fwd_dt, "call_outputs_B": call_out,
                      "forward_outputs_B": fwd_out, "hbm_B": HBM_BYTES,
                      "fits": {"call_dots": call_dt <= HBM_BYTES, "forward_dots": fwd_dt <= HBM_BYTES,
                               "call_out": call_out <= HBM_BYTES, "forward_out": fwd_out <= HBM_BYTES},
                      "alpha": weighted_alpha(d, shapes)}
    return out


def energy_reweight(data: dict, cb: dict) -> dict:
    """Energy as a pricing (excluded by AGENTS.md; estimate from the data sheet, both pipes at TDP): an INT32 register
    write costs r·32 MAC-energies; the tensor's 32 a register is unchanged, so the per-word credit is unchanged and the
    scalar terms (forming, hashing) scale by r."""
    j_mac, j_int = TDP_W / INT8_MAC_S, TDP_W / INT32_OPS_S
    r = j_int / j_mac / H32
    t = design_terms("ssn_1l", "llama_T8192", data, cb)
    return {"J_mac_pJ": j_mac * 1e12, "J_int32_pJ": j_int * 1e12, "r": r, "forming_w1": t["forming"],
            "forming_energy": t["forming"] * r, "h_floor_w1": t["_h_floor"], "h_floor_energy": t["_h_floor"] * r,
            "delta_w1": (ALLOW - t["forming"]) / t["_h_floor"], "delta_energy": (ALLOW - t["forming"] * r) /
            (t["_h_floor"] * r), "credit_per_word": H32}


def periods(data: dict, cb: dict) -> dict:
    """Per monitoring period, SW-NOISE §7's basis: one H200 at the tensor peak covers 989.5e12/α stock MACs a second; a
    datacenter is 100,000 of them; the world is 9.36e30 FLOP a year at 2 a MAC, over α. Each W1 share of stock is shown as
    that share of stock's AND gates (2,079/16 a MAC); the hashing's own AND count is its own row."""
    t = design_terms("ssn_1l", "llama_T8192", data, cb)
    alpha = cb["llama_T8192"]["alpha"]
    h = hash_terms(next(x for x in NS.designs() if x.name == "ssn_1l"), NS.LLAMA, 8192, False)
    rows = {"allowance (1%)": ALLOW, "hashing ShaSpeed must price (W1 floor)": h["h_floor"],
            "hashing in AND gates (hm96's AND model, not W1)": h["h_ands"],
            "rank ≤ 3 residual (hashing without ShaSpeed)": h["rank3_residual_floor"],
            "ShaSpeed's slack at δ*": ALLOW - t["forming"],
            "hiding at 256 KiB, unforced (W1 floor)": t["hiding"],
            "coin branch, WikiText": t["coin_branch_natural"],
            "coin branch, identical rows at T 8,192": t["coin_branch_chosen"]}
    mac_s = {"one H200": INT8_MAC_S / alpha, "datacenter": INT8_MAC_S / alpha * NS.DATACENTER}
    out = {}
    for scope, rate in mac_s.items():
        for name, share in rows.items():
            out.setdefault(scope, {})[name] = {p: share * rate * s * ANDS_MAC for p, s in NS.PERIOD_S.items()}
    world_s = NS.WORLD_MAC_YEAR / alpha / NS.PERIOD_S["year"]
    for name, share in rows.items():
        out.setdefault("world", {})[name] = {p: share * world_s * s * ANDS_MAC for p, s in NS.PERIOD_S.items()}
    return out


# --------------------------------------------------------------------------------------------------- checks, report
def checks(rep: dict) -> list[dict]:
    c = []

    def ok(name, cond, detail=""):
        c.append({"check": name, "pass": bool(cond), "detail": detail})

    led = rep["ledger"]
    ids = [r["id"] for r in led]
    ok("ledger ids unique", len(ids) == len(set(ids)))
    ok("every ledger rank known", all(r["rank"] in RANK_ORDER + ["R", "C", "X"] for r in led))
    ok("open premises rank 5 today", all(r["rank_today"] == "5" for r in led if r["status"] in ("open", "sketch")))
    ok("every option names ledger rows", all(p in ids for o in OPTIONS.values() for p in o[0]))
    ok("every design base names ledger rows", all(p in ids for d in DESIGNS for p in d[4]))
    sp = rep["shaspeed"]["llama_T8192"]
    ok("ShaSpeed δ* at the lead is below 1e-4 (floor)", sp["delta_floor"] < 1e-4, f"{sp['delta_floor']:.2e}")
    ok("ShaSpeed slack under one instruction a compression", sp["instr_compression_floor"] < 1,
       f"{sp['instr_compression_floor']:.3f}")
    r3 = rep["rank3"]
    ok("rank ≤ 3 needs ≥ 1,000× fewer bytes than k32", r3["max_bytes_mac_rank3"] * 1000 < 4 / 32,
       f"{r3['max_bytes_mac_rank3']:.2e} B a MAC")
    ok("hashing the outputs alone exceeds the allowance at every point, so no design is rank ≤ 3 overall with hashing "
       "in T1", all(v["hash_share"] * (1 - r3["rank3_sha_lb_share"]) > ALLOW for v in r3["outputs_only"].values()),
       str({p: round(v["hash_share"], 3) for p, v in r3["outputs_only"].items()}))
    cb = rep["coin_branch"]
    ok("ssn_1l collapses on identical rows at T 8,192", cb["llama_T8192"]["chosen_identical_ssn_1l"] > 0.5,
       f"{cb['llama_T8192']['chosen_identical_ssn_1l']:.3f}")
    ok("ssn_1l collapses on zero rows at T 512", cb["llama_T512"]["chosen_zero_ssn_1l"] > 0.1,
       f"{cb['llama_T512']['chosen_zero_ssn_1l']:.3f}")
    ok("ssn_1l_hw ≤ 1% on identical rows at every point",
       all(v["chosen_identical_ssn_1l_hw"] <= ALLOW for v in cb.values() if "chosen_identical_ssn_1l_hw" in v),
       str({k: round(v.get("chosen_identical_ssn_1l_hw", -1), 5) for k, v in cb.items()}))
    ok("ssn_1l_hw ≤ 1% on zero rows at every point",
       all(v["chosen_zero_ssn_1l_hw"] <= ALLOW for v in cb.values() if "chosen_zero_ssn_1l_hw" in v))
    hw = rep["hw_distinct"]
    ok("ssn_1l_hw words distinct and non-free on every data case", all(v["bad_share"] == 0 for v in hw.values()),
       str({k: v["bad_share"] for k, v in hw.items()}))
    ok("random data never collapses (ssn_1l)", branch_collapse("ssn_1l", "random", 64, 67) == 0)
    ok("pair-collision formula matches simulation", abs(
        branch_collapse("ssn_1l", "pair", 4096, 8, salts=4) - 0.5 * pair_collision_ssn1l(8)) < 0.004)
    lead = rep["hitting"]["ssn_1l"]["minimal_sets"]
    ok("the lead has a passing minimal set", len(lead) > 0)
    ok("every passing set for the lead takes ShaSpeed (rank 4a)", all("ShaSpeed" in s["premises"] for s in lead))
    ok("no design passes at rank ≤ 3 overall", all(
        s["rank_settled"] not in ("0", "1", "2", "3") for d in rep["hitting"].values() for s in d["minimal_sets"]))
    hwset = rep["hitting"]["ssn_1l_hw"]["minimal_sets"]
    ok("ssn_1l_hw passes without inputs-natural", any("inputs-natural" not in s["premises"] for s in hwset))
    ok("ssn_1l passes only with inputs-natural (false)", all("inputs-natural" in s["premises"] for s in lead))
    ok("nsf passes only with inputs-natural (false)",
       all("inputs-natural" in s["premises"] for s in rep["hitting"]["nsf"]["minimal_sets"]))
    ks = sorted({k for shapes, _, _ in POINTS.values() for _, k in shapes.values()})
    ok("ssn_1l_hw's S + 15 coin bits a row take as many coin words as ssn_1l's S + 3 at every priced K, so its "
       "coin term is ssn_1l's", all(hw_coin_words(NS.steps1l(k)) == NS.ssn_1l_words(NS.steps1l(k)) for k in ks),
       str({k: (hw_coin_words(NS.steps1l(k)), NS.ssn_1l_words(NS.steps1l(k))) for k in ks}))
    hid = rep["hiding"]["ssn_1l"]["card"]["llama_T8192"]
    ok("ssn_1l's card hiding, alone, is within 15% of SW-PRICE's 1.18x stock (words, A's rows; measured price)",
       abs(hid["unforced_alone"] / 1.18 - 1) < 0.15, f"{hid['unforced_alone']:.3f}")
    ok("A's per-row leaves alone exceed the allowance at the floor on Llama, so big word rows alone can't close hiding",
       hid["unforced_a_floor"] > ALLOW, f"{hid['unforced_a_floor']:.2%}")
    ok("every passing set with big rows names its leaf size", all(
        s["big_rows_leaf"] for d in rep["hitting"].values() for s in d["minimal_sets"]
        if any(o.startswith("hiding: big rows") for o in s["options"])))
    g1 = rep["g1"]
    ok("one call's outputs fit HBM at every point", all(v["fits"]["call_out"] for v in g1.values()))
    en = rep["energy"]
    ok("energy keeps the credit a word and scales forming by r", en["credit_per_word"] == H32 and
       abs(en["forming_energy"] / en["forming_w1"] - en["r"]) < 1e-9)
    return c


def mutations() -> list[dict]:
    """Each mutation must flip a check: the hardened variant without odd coins (zero rows merge step t into t − 1),
    without signs (zero rows' steps coincide across steps), and the pair formula on ssn_1l_hw (it must not match)."""
    res = []
    even = branch_collapse("ssn_1l_hw_even", "zero", 512, 67, salts=2)
    res.append({"mutation": "ssn_1l_hw with a_t ∈ {0,1}", "zero-row collapse": even, "caught": even > ALLOW})
    nosign = branch_collapse("ssn_1l_hw_nosign", "zero", 8192, 67, salts=1)
    res.append({"mutation": "ssn_1l_hw with every sign +1", "zero-row collapse": nosign, "caught": nosign > ALLOW})
    hwpair = branch_collapse("ssn_1l_hw", "pair", 4096, 8, salts=2)
    res.append({"mutation": "pair formula applied to ssn_1l_hw", "sim": hwpair, "formula": 0.5 * pair_collision_ssn1l(8),
                "caught": abs(hwpair - 0.5 * pair_collision_ssn1l(8)) > 0.004})
    return res


def report(data_path: str | None) -> dict:
    data = {}
    if data_path:
        NS.load_data(data_path)
        data = NS.DATA
    cb = coin_branch_terms(data)
    rep = {"ledger": ledger(), "coin_branch": cb, "hw_noise_std": [hw_noise_std(k) for k in (2048, 8192, 7168)],
           "hw_distinct": hw_distinct_check()}
    rep["hitting"] = {lab: minimal_sets(lab, data, cb) for lab, *_ in DESIGNS}
    rep["shaspeed"] = shaspeed_precision(data, cb)
    rep["rank3"] = rank3_routes()
    rep["g1"] = g1_advice()
    rep["energy"] = energy_reweight(data, cb)
    rep["periods"] = periods(data, cb)
    rep["hiding"] = hiding_summary(rep["hitting"])
    rep["params"] = {"W1_compression": W1_COMPRESSION, "rows": ROWS, "card_row": CARD_ROW,
                     "hide_rows": [HIDE_ROWS, HIDE_NS_LEAF, HIDE_SHARE, MASK_SHARE, COMP_NS],
                     "points": {k: [v[1], v[2]] for k, v in POINTS.items()}, "allowance": ALLOW}
    return rep


def hiding_summary(hitting: dict) -> dict:
    """Each design's hiding on the card's layout, and the smallest leaf (words on it, a call's A packed up to it) at
    which honest-only hiding fits beside what no true option closes (inputs-natural is false), floor and alone,
    unforced and with the salt PRF forced (M_key·y left)."""
    out = {}
    for lab, h in hitting.items():
        terms = h["terms"]
        rest = {p: sum(v for k, v in t.items() if not k.startswith("_") and k not in
                       ("final_word", "hashing", "coins", "pre_noise", "hiding")) for p, t in terms.items()}
        e = {"card": {p: t["_hiding"] for p, t in terms.items()}, "rest": rest}
        for name in ("unforced", "my"):
            for reading in W1_COMPRESSION:
                key = f"{name}_{reading}"
                e[f"leaf_{key}"] = next((r for r in BIG_ROWS if all(
                    rest[p] + hiding_terms(t["_bytes_mac"], POINTS[p][0], max(1, round(POINTS[p][1] * POINTS[p][2])),
                                           row=r, a_pack=r)[key] <= ALLOW for p, t in terms.items())), None)
        out[lab] = e
    return out


def table(rep: dict) -> str:
    lines = ["# premise ledger", "| id | rank | today | status | owner | closes |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['rank']} | {r['rank_today']} | {r['status']} | {r['owner']} | {r['closes']} |"
              for r in rep["ledger"]]
    lines.append("\n# ShaSpeed's precision (ssn_1l, 256 KiB rows)")
    for p, e in rep["shaspeed"].items():
        lines.append(f"{p}: h {e['h_floor']:.1f} / {e['h_alone']:.1f} W1 a MAC; δ* {e['delta_floor']:.2e} / "
                     f"{e['delta_alone']:.2e}; {e['instr_compression_floor']:.3f} / {e['instr_compression_alone']:.3f} "
                     "instructions a compression")
    lines.append("\n# coin branch (share of stock)")
    for p, e in rep["coin_branch"].items():
        lines.append(f"{p}: " + ", ".join(f"{k} {v:.4%}" for k, v in e.items() if isinstance(v, float) and k != "alpha"))
    lines.append("\n# minimal premise sets")
    for lab, h in rep["hitting"].items():
        lines.append(f"## {lab}")
        for s in h["minimal_sets"][:6]:
            worst = max(s["residual"].values())
            lines.append(f"- rank {s['rank_settled']} (today {s['rank_today']}; matmul {s['matmul_rank_settled']}/"
                         f"{s['matmul_rank_today']}): {', '.join(s['options']) or '(base only)'}; worst T1 {worst:.3%}")
        if not h["minimal_sets"]:
            best = h["every_option_residual"]
            lines.append(f"- none passes; with every option: {', '.join(f'{p} {v:.3%}' for p, v in best.items())}")
    lines.append("\n# hiding (share of stock; card: words on 256 KiB leaves, A one activation row a leaf)")

    def mib(r):
        return "none" if r is None else f"{r / 2 ** 20:g} MiB"
    for lab, e in rep["hiding"].items():
        c = e["card"]["llama_T8192"]
        lines.append(f"{lab}: Llama T 8,192 floor {c['unforced_floor']:.2%} (words {c['unforced_words_floor']:.2%}, "
                     f"A {c['unforced_a_floor']:.2%}), alone {c['unforced_alone']:.2%}, PRF forced {c['my_floor']:.2%}; "
                     f"big rows fit at {mib(e['leaf_unforced_floor'])} floor / {mib(e['leaf_unforced_alone'])} alone, "
                     f"PRF forced {mib(e['leaf_my_floor'])} / {mib(e['leaf_my_alone'])}")
    r3 = rep["rank3"]
    lines.append(f"# rank ≤ 3 overall: SHA-512 bound {r3['rank3_sha_lb_share']:.2%}; bytes a MAC ≤ "
                 f"{r3['max_bytes_mac_rank3']:.2e} (one 4 B word a {r3['min_macs_per_4B_word_rank3']:,.0f} MACs)")
    lines.append("# outputs only (one 4 B word a dot product): " + "; ".join(
        f"{p} hashing {v['hash_share']:.1%} of stock, δ* {v['delta_star']:.2e} ({v['instr_compression']:.1f} "
        "instructions a compression)" for p, v in r3["outputs_only"].items()))
    en = rep["energy"]
    lines.append(f"# energy (excluded): r {en['r']:.2f}; forming {en['forming_w1']:.3%} → {en['forming_energy']:.3%}; "
                 f"δ* {en['delta_w1']:.2e} → {en['delta_energy']:.2e}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", help="noise_data.py's output (SW-NOISE's data input)")
    ap.add_argument("--json")
    ap.add_argument("--table")
    ap.add_argument("--mutations", action="store_true")
    a = ap.parse_args()
    rep = report(a.data)
    rep["checks"] = checks(rep)
    if a.mutations:
        rep["mutations"] = mutations()
    txt = table(rep)
    print(txt)
    for c in rep["checks"]:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"] + (f" ({c['detail']})" if c["detail"] else ""))
    for mu in rep.get("mutations", []):
        print(("CAUGHT " if mu["caught"] else "MISSED ") + mu["mutation"])
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=str))
    if a.table:
        Path(a.table).write_text(txt)
    bad = [c for c in rep["checks"] if not c["pass"]] + [mu for mu in rep.get("mutations", []) if not mu["caught"]]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
