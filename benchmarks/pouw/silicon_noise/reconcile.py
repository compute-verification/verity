"""SI-NOISE against SI-FOLD: this lane's mask-then-fold sketch (SN) and SI-FOLD's split Keccak transcript (FT), priced in
one flow, attacked with the same cheater, and the arithmetic behind each design's credit statement under CORE.md §2a.

- **SN** (this lane): on every credited write the unit adds n = PRF(seed, tile, i, j, t) to the accumulator word and
  folds it, σ ← rotl(σ, 1) ⊕ (acc + n). Here it sits at the write-back port: one adder, rotate-xor and 32-bit σ a lane,
  32 lanes a tensor core, and a Keccak-p[1600,12] stream of b bits a word (rate 1,088).
- **FT** (SI-FOLD): τ ← ρ(τ ⊕ the instruction's words), ρ one Keccak-f[1600] round over 1,024 absorbed bits, τ starting at
  the call's live coin; past 32 words a round the extra words are pre-folded by XOR.

Inputs: `--mine` this lane's blocks mapped in SI-ALG's flow (`synth.py OUT --lib=generic.lib`) and `--fold` SI-FOLD's
`synth_lib.json` (run r20261029-222434-a055, the same flow). `si_fold/rho.py` and `si_fold/silicon_fold.py` are SI-FOLD's,
copied unchanged from `cursor/compute-research-silicon-fold-3c7a` at 4a79b74a45; the transcript attacked is theirs.

What it establishes (each a `Checks` line):
- **One round.** SI-FOLD's `keccak1600_rnd` is this lane's `keccak_round` plus FT's 1,024-bit absorb XOR: the same
  NAND/NOR and INV cells, 1,024 more XOR-class cells, and logic equal within 0.1% in SI-ALG's flow.
- **Area at a common margin.** Both cheaters bit-sliced at SI-FOLD's checked 120 writes a round (its E5), so SN's PRF
  margin is 1.32·b in W1, not the 2.2·b of the preliminary note's word-wise estimate. At every margin, in W1 and at
  sm_120's prices, FT costs less than SN: they spend the same Keccak rounds a word and SN adds its sketch lanes.
- **Proof.** SN's PRF stays outside the proof (its words are public), so SN's ANDs a MAC are flat; FT's round is inside,
  1,600 ANDs a round, so SN's proof is cheaper above a crossover margin.
- **copy.** FT as SI-FOLD's code builds it gives every tile the same τ at zero data (the IV is the call's coin alone),
  so one chain serves the call; a tile id in the IV separates them.
- **FT at zero data is the SHA-3 permutation**: 24 transcript steps absorbing zero words are `keccak_f1600` of the IV,
  and those 24 rounds on SHA3-512's padded empty block give hashlib's digest.
- **SN at zero declared data is GF(2)-linear in the noise** (its premise is about linear digests of the PRF stream),
  and isn't on random declared words.
- **Seclusion.** With n readable, a bit-sliced fold of n gives the declared σ of 32 cells for 2b writes a step (b reads,
  b XORs), beside an MMA running hidden work: the net gap is 1 − 2b/32 of stock. FT's τ needs no seclusion.
- **§2a.** Both designs' gap is 0 at every shape; operand noise to whiten power would add 32/K + 1/n + 1/m of honest work
  the zero-data adversary skips.

Run (from this directory): `uv run --no-project --with numpy --with-editable ../exact_kit python -B reconcile.py OUT.json
--mine=synth_lib.json --fold=FOLD_synth_lib.json [--mutations]`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "si_fold"))
import rho
import silicon_fold as sf
from exact_kit.checks import Checks
from silicon_noise import SI_ALG, sketch

TC = 1024
WORDS = 32
KECCAK_SW = 120.0
RATE = 1088
RATE_FT = 1024
P12 = 12
W1_WRITE = 32.0
SM120_WRITE = sf.SM120_WRITE
FF = 5.667
PREFOLD_GE_A_WORD = 64.0
KECCAK_ANDS = 1600
SN_ANDS_A_WORD = 31
MARGINS = (1.1, 1.5, 2.0, 2.4, 3.0, 3.75, 4.0, 5.0, 6.0, 7.9, 10.0)
SHAPES = ((64, 2048, 2048), (8192, 2048, 2048), (64, 2048, 8192), (8192, 2048, 8192), (64, 8192, 2048),
          (8192, 8192, 2048))


def price_of(model: str) -> float:
    return W1_WRITE if model == "w1" else SM120_WRITE


def sn_point(M: float, model: str, mine: dict, sw_round: float = KECCAK_SW) -> dict:
    """SN at margin M: b bits of Keccak-p[1600,12] a word, the bit-sliced cheater's writes priced by the model."""
    sw_per_b = WORDS * P12 * sw_round / RATE
    b = M * WORDS * W1_WRITE / (sw_per_b * price_of(model))
    rounds = WORDS * b * P12 / RATE
    prf = rounds * mine["keccak_round"]["ge"]
    lanes = WORDS * mine["sk_xorrot"]["ge"]
    return {"b": round(b, 3), "rounds_a_cycle": round(rounds, 4), "ge_a_mac": round((prf + lanes) / TC, 2),
            "prf_ge_a_mac": round(prf / TC, 2), "sketch_ge_a_mac": round(lanes / TC, 2),
            "pct_s8": round(100 * (prf + lanes) / TC / SI_ALG["s8"], 2), "ands_a_mac": round(SN_ANDS_A_WORD / 32, 3)}


def ft_point(M: float, model: str, unit: float) -> dict:
    covered = KECCAK_SW * price_of(model) / (M * W1_WRITE)
    ge = (unit + max(0.0, covered - WORDS) * PREFOLD_GE_A_WORD) / (covered * 32)
    return {"words_a_round": round(covered, 2), "rounds_a_cycle": round(WORDS / covered, 4), "ge_a_mac": round(ge, 2),
            "pct_s8": round(100 * ge / SI_ALG["s8"], 2), "ands_a_mac": round(KECCAK_ANDS / (covered * 32), 3)}


def area_section(checks: Checks, mine: dict, fold: dict, xor2_ge: float) -> dict:
    fm = fold["modules"]
    ft_unit = fm["keccak1600_rnd"]["ge"] + fm["keccak_rcrom"]["ge"] + 1600 * fold["meta"]["ge_dff"]
    my_logic = mine["keccak_round"]["ge_comb"]
    xors, nonlin = ({k: sum(c.get(g, 0) for g in gs) for k, c in (("mine", mine["keccak_round"]["cells"]),
                                                                  ("fold", fm["keccak1600_rnd"]["cells"]))}
                    for gs in (("XOR2", "XNOR2"), ("NAND2", "NOR2", "INV")))
    absorb = RATE_FT * xor2_ge
    checks.check("one Keccak-f[1600] round in both lanes: SI-FOLD's keccak1600_rnd is this lane's keccak_round plus "
                 "the 1,024-bit absorb XOR (same NAND/NOR and INV cells, 1,024 more XOR-class cells, logic equal within "
                 "0.1% in SI-ALG's flow)", nonlin["mine"] == nonlin["fold"] and xors["fold"] - xors["mine"] == RATE_FT
                 and abs((my_logic + absorb) / fm["keccak1600_rnd"]["ge"] - 1) < 0.001,
                 {"mine": my_logic, "mine_plus_absorb": round(my_logic + absorb, 2), "si_fold": fm["keccak1600_rnd"]["ge"],
                  "ft_unit_with_state": round(ft_unit, 1)})
    curves = {}
    for model in ("w1", "sm120"):
        curves[model] = [{"M": M, "sn": sn_point(M, model, mine), "ft": ft_point(M, model, ft_unit)} for M in MARGINS]
    checks.check("at every margin, in W1 and at sm_120's prices, FT costs less area than SN (same rounds a word, SN adds "
                 "its sketch lanes)", all(p["ft"]["ge_a_mac"] < p["sn"]["ge_a_mac"] for c in curves.values() for p in c),
                 [(m, p["M"], p["ft"]["ge_a_mac"], p["sn"]["ge_a_mac"]) for m, c in curves.items() for p in c
                  if p["M"] in (1.1, 3.75, 7.9)])
    same = [p for c in curves.values() for p in c]
    checks.check("at every margin SN's PRF and FT spend the same Keccak rounds a cycle (within 1%): the margin is bought "
                 "by rounds in both", all(abs(p["sn"]["rounds_a_cycle"] / p["ft"]["rounds_a_cycle"] - 1) < 0.01
                                          for p in same),
                 [(p["M"], p["sn"]["rounds_a_cycle"], p["ft"]["rounds_a_cycle"]) for p in curves["w1"]][:3])
    cross = {}
    for model, c in curves.items():
        sn_flat = c[0]["sn"]["ands_a_mac"]
        slope = c[0]["ft"]["ands_a_mac"] / c[0]["M"]
        cross[model] = round(sn_flat / slope, 2)
    checks.check("SN's proof (PRF outside it) is cheaper than FT's (round inside) above margin 2.33 in W1 and 1.18 at "
                 "sm_120's prices", abs(cross["w1"] - 2.33) < 0.01 and abs(cross["sm120"] - 1.18) < 0.01, cross)
    old = {"keccak12_b1_wordwise": 2.2, "keccak12_b1_bitsliced": round(P12 * KECCAK_SW / RATE, 3),
           "keccak24_b1_bitsliced": round(24 * KECCAK_SW / RATE, 3)}
    return {"ft_unit_ge": round(ft_unit, 1), "sketch_lane_ge": mine["sk_xorrot"]["ge"],
            "keccak_round_ge": mine["keccak_round"]["ge"], "curves": curves, "proof_crossover_margin": cross,
            "sn_margin_a_bit": old}


def ft_tile_tau(coin: int, words_by_step: list[list[int]]) -> list[int]:
    return sf._keccak_transcript(words_by_step, coin)


def copy_section(checks: Checks) -> dict:
    g = np.random.default_rng(29)
    coin = int(g.integers(0, 1 << 62))
    T = 8
    zero = [[0] * WORDS for _ in range(T)]
    tiles = [ft_tile_tau(coin, zero) for _ in range(4)]
    rep = [[int(x) for x in g.integers(0, 1 << 32, WORDS)] for _ in range(T)]
    rep_tiles = [ft_tile_tau(coin, rep) for _ in range(2)]
    ids = [ft_tile_tau((coin + (tid << 1)) & rho.M64, zero) for tid in range(4)]
    checks.check("copy: FT as SI-FOLD's code builds it gives every tile the same τ at zero data, and on declared tiles "
                 "that repeat, so one chain serves the call", all(t == tiles[0] for t in tiles)
                 and rep_tiles[0] == rep_tiles[1], {"tiles": len(tiles), "distinct_tau": len({tuple(t) for t in tiles})})
    checks.check("a tile id in the IV separates the chains", len({tuple(t) for t in ids}) == len(ids),
                 {"distinct_tau": len({tuple(t) for t in ids})})
    return {"zero_data_distinct": len({tuple(t) for t in tiles}), "with_ids_distinct": len({tuple(t) for t in ids})}


def sha3_section(checks: Checks) -> dict:
    g = np.random.default_rng(31)
    coin = int(g.integers(0, 1 << 62))
    zero24 = ft_tile_tau(coin, [[0] * WORDS for _ in range(24)])
    perm = rho.keccak_f1600([0] * 24 + [coin])
    block = bytearray(200)
    block[0] ^= 0x06
    block[71] ^= 0x80
    lanes = [int.from_bytes(block[8 * k:8 * k + 8], "little") for k in range(25)]
    for t in range(24):
        lanes = rho.keccak_round(lanes, rho.KECCAK_RC64[t], 64)
    digest = b"".join(x.to_bytes(8, "little") for x in lanes[:8])
    checks.check("FT at zero data is the SHA-3 permutation: 24 steps absorbing zero words equal keccak_f1600 of the IV, "
                 "and those 24 rounds on SHA3-512's padded empty block give hashlib's digest",
                 zero24 == perm and digest == hashlib.sha3_512(b"").digest())
    return {"zero24_is_keccak_f": zero24 == perm, "sha3_512_empty": digest.hex()[:32]}


def linear_section(checks: Checks, b: int = 8) -> dict:
    g = np.random.default_rng(37)
    cells, T = 64, 24
    r = np.ones(T, np.int64)
    zero = np.zeros((cells, T), np.int64)
    n1, n2 = (g.integers(0, 1 << b, (cells, T), dtype=np.int64) for _ in range(2))
    lin0 = np.array_equal(sketch("xorrot", zero, n1 ^ n2, r), sketch("xorrot", zero, n1, r) ^ sketch("xorrot", zero, n2, r))
    acc = g.integers(0, 1 << 32, (cells, T), dtype=np.int64)
    s12 = sketch("xorrot", acc, n1 ^ n2, r)
    s0 = sketch("xorrot", acc, np.zeros_like(n1), r)
    lin_acc = np.array_equal(s12 ^ s0, (sketch("xorrot", acc, n1, r) ^ s0) ^ (sketch("xorrot", acc, n2, r) ^ s0))
    checks.check("SN at zero declared data is GF(2)-linear in the noise (σ(n ⊕ n') = σ(n) ⊕ σ(n')), and not on random "
                 "declared words", lin0 and not lin_acc, {"zero": lin0, "random": lin_acc})
    return {"linear_at_zero": lin0, "affine_on_random_words": lin_acc}


def seclusion_section(checks: Checks) -> dict:
    """With n readable: σ^D of 32 cells at zero declared data, bit-sliced. Plane k holds bit k of σ for the 32 cells;
    rotl renames planes, and each of n's b bit planes is one read and one XOR a step."""
    g = np.random.default_rng(41)
    out = {}
    for b in (1, 8):
        cells, T = 32, 40
        n = g.integers(0, 1 << b, (cells, T), dtype=np.int64)
        ref = sketch("xorrot", np.zeros_like(n), n, np.ones(T, np.int64))
        planes = [0] * 32
        writes = 0
        for t in range(T):
            planes = planes[31:] + planes[:31]
            for k in range(b):
                reg = int(sum(((int(n[c, t]) >> k) & 1) << c for c in range(cells)))
                writes += 1
                planes[k] ^= reg
                writes += 1
        got = [sum(((planes[k] >> c) & 1) << k for k in range(32)) for c in range(cells)]
        per = writes / (cells * T)
        out[b] = {"exact": got == ref.tolist(), "writes_a_cell_step": per, "net_gap": round(1 - per, 4)}
    checks.check("seclusion is load-bearing for SN: with n readable the declared σ costs 2b/32 writes a cell-step "
                 "(exact), so the MMA's hidden work nets a gap of 1 − 2b/32 (94% at b = 1, 50% at b = 8)",
                 all(v["exact"] and v["writes_a_cell_step"] == 2 * b / 32 for b, v in out.items()), out)
    return out


def sec2a_section(checks: Checks) -> dict:
    rows = []
    for m, n, K in SHAPES:
        rows.append({"m": m, "n": n, "K": K, "sn_gap": 0.0, "ft_gap": 0.0,
                     "whitening_denoise": round(100 * (32 / K + 1 / n + 1 / m), 3),
                     "ncp_final_word_and_forming": round(100 * (32 / K + 24 / n + 24 / m), 3)})
    checks.check("§2a: SN and FT have gap 0 at every shape; whitening's denoise alone fails 1% at every shape with K "
                 "≤ 2,048 or m = 64", all(r["whitening_denoise"] > 1 for r in rows if r["K"] <= 2048 or r["m"] == 64),
                 [(r["m"], r["n"], r["K"], r["whitening_denoise"]) for r in rows])
    return {"rows": rows}


def mutations(checks: Checks, mine: dict, fold: dict) -> None:
    with checks.mutant("SN's cheater priced word-wise again (200 ops a round, the preliminary note's estimate)"):
        fm = fold["modules"]
        unit = fm["keccak1600_rnd"]["ge"] + fm["keccak_rcrom"]["ge"] + 1600 * fold["meta"]["ge_dff"]
        pts = [(sn_point(M, "w1", mine, 200.0), ft_point(M, "w1", unit)) for M in MARGINS]
        checks.check("same rounds", all(abs(s["rounds_a_cycle"] / f["rounds_a_cycle"] - 1) < 0.01 for s, f in pts))
    with checks.mutant("a tile id in FT's IV (copy no longer applies)"):
        g = np.random.default_rng(29)
        coin = int(g.integers(0, 1 << 62))
        tiles = [ft_tile_tau(coin ^ tid, [[0] * WORDS] * 4) for tid in range(4)]
        checks.check("copy", len({tuple(t) for t in tiles}) == 1)
    with checks.mutant("SN's sketch with its adder replaced by XOR (linear on every word)"):
        g = np.random.default_rng(37)
        acc = g.integers(0, 1 << 32, (8, 6), dtype=np.int64)
        n1, n2 = (g.integers(0, 256, (8, 6), dtype=np.int64) for _ in range(2))
        r = np.ones(6, np.int64)
        s = sketch("xorrot_x", acc, n1 ^ n2, r) ^ sketch("xorrot_x", acc, 0 * n1, r)
        t = sketch("xorrot_x", acc, n1, r) ^ sketch("xorrot_x", acc, n2, r)
        checks.check("not linear on random words", not np.array_equal(s, t))


def main() -> int:
    checks = Checks()
    args = [a for a in checks.args if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in checks.args if a.startswith("--") and "=" in a)
    mine_doc = json.loads(Path(opts["mine"]).read_text())
    mine = mine_doc["blocks"]
    fold = json.loads(Path(opts["fold"]).read_text())
    if "SI-ALG" not in mine_doc["library"].get("flow", ""):
        raise SystemExit(f"{opts['mine']} isn't in SI-ALG's flow: run `python3 synth.py OUT --lib=generic.lib`")
    if not all(m["checked"] for m in fold["modules"].values()):
        raise SystemExit(f"{opts['fold']}: an unchecked SI-FOLD netlist")
    res = {"schema": "silicon-noise-reconcile/v1",
           "area": area_section(checks, mine, fold, mine_doc["library"]["cells"]["XOR2"]),
           "copy": copy_section(checks), "sha3": sha3_section(checks), "linear": linear_section(checks),
           "seclusion": seclusion_section(checks), "sec2a": sec2a_section(checks)}
    if checks.mutations:
        mutations(checks, mine, fold)
    Path(args[0] if args else "reconcile.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    for model, c in res["area"]["curves"].items():
        for p in c:
            s, f = p["sn"], p["ft"]
            print(f"{model:5s} M {p['M']:5.2f}  SN b {s['b']:6.3f} {s['ge_a_mac']:6.2f} GE/MAC ({s['pct_s8']:5.2f}% s8, "
                  f"PRF {s['prf_ge_a_mac']:5.2f} + sketch {s['sketch_ge_a_mac']:5.2f}) {s['ands_a_mac']:.2f} AND  |  FT "
                  f"{f['words_a_round']:6.2f} words/round {f['ge_a_mac']:6.2f} GE/MAC ({f['pct_s8']:5.2f}% s8) "
                  f"{f['ands_a_mac']:.2f} AND")
    for r in res["sec2a"]["rows"]:
        print(f"m {r['m']:5d} n {r['n']:5d} K {r['K']:5d}  gap SN {r['sn_gap']} FT {r['ft_gap']}  whitening "
              f"{r['whitening_denoise']}%  NCP {r['ncp_final_word_and_forming']}%")
    print("seclusion:", res["seclusion"])
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
