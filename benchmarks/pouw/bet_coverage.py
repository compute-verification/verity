"""ACCEL BET-COVERAGE (CMP-124): the SHA-512 compressions and the lane work an H1 unit spends per credited INT8 MAC at the
frozen H200 shape, for each tile and commitment design, the share of the shape's MACs each credits at γ ≤ 10, its drawn
unit against C-Flock's limits, and the period's proving.

    uv run python benchmarks/pouw/bet_coverage.py [--json OUT] [--md OUT]

**The shape.** One H200 serving the frozen ~140B int8 model, M = 20,480 rows a call: 52 layers of QKV (N 16,512, K
16,384), O (16,384, 16,384), up (65,536, 16,384) and down (16,384, 65,536), and the head (256, 16,384), 1.397 × 10¹¹
MACs a token (note:compute-accounting/20261025T2330Z-report-h200-pouw).

**MAC-times.** One H200 MAC-time is 1/ρ_8, ρ_8 = 4,096 s8 MACs an SM clock × 132 SMs × 1.83 GHz = 9.895 × 10¹⁴ MAC/s, the
data sheet's 1,979 dense TOPS (`machine_model/data/devices/sm_90/sheet.toml`). A SHA-512 compression is ρ_8 / 4.79 G =
206,576 MAC-times at commit_gpu's measured whole-card rate (note:hash/20261018T0636Z-draft-structure). The served plain
GEMM runs at `PLAIN_EFF` of ρ_8, so P = 1/0.85 MAC-times a MAC.

**Hashing (exact counts, then priced).** Noise and seeds are `h1tile`'s own counts (`unit_compressions`' blocks and
seeds; J10-R's call stream is `j10r_compressions` in closed form, held to it at small shapes by `check_closed_form`).
What a design commits before the draw, a unit's CP-W receipt (2 (KS + m) words), its z (4 bytes an output) and, for a
lane-chunk draw, each chunk's boundary (h1zy's `CARRY` less its words, 220 words, and the chunk's own per-set words), is
priced as a `sha512/row-seg/v1` row (1 KiB segments after a constant block, then the digests) plus `TREE` compressions
for its hm96 leaf and tree node, an estimate. The lanes' fold keys (one compression a lane a unit) are lane work.

**Lane work (model, anchored).** A unit-segment (32 × 32 × 32 MACs, one warp) is priced on the H200's pipes (census
DC-90: ALU and IMAD 64 lanes an SM clock, issue 4 warp instructions, IMMA16832 at 0.656 of the wgmma peak): its floor is
the largest of ALU/16, IMAD/16, instructions/32 and 1/0.656 MAC-times a MAC, plus the fold keys (2,885 ALU a compression).
The counts are krm2's (H1U_JM2, the H200's best J10-R units: 229 instructions a unit-segment;
note:kernels-dh200c/20261024T0108Z-report-kernels-dh200c 21:00Z), split by the JCS SASS census (fold 64 ALU and 64 IMAD
a segment, B unpack 48 ALU, which JM2's two row blocks share; note:kernels-dh200c/20261024T2353Z-finding-jcs-registers).
The measured row (h1umjc_sm 10.70 ms on 25/32 of the SMs for 64 rows of Llama-3.1-8B, nsys r20261024-221117-9d20 beside
the gated row r20261024-225443-dd1e) gives the units' utilization of that floor, `util()`, and a design's projected lane
work U is its floor over it: scaled from one decode row, not measured at this shape. Per-unit-noise designs are given
krm2's lane work, a lower bound on theirs. `crosscheck()` prices what the H200 measured: `h1jr_cols` (J10-R's column
blocks, 3.02 ms in the anchor's nsys) and the per-layer kernel γ of Llama-3.1-8B at M = 8,192 (note:hash/20261019T0427Z-
draft-dither-hb), the model above each.

**Coverage.** Credited MACs run their unit beside the served GEMM, so a step crediting a share f takes
1 + f (U + H)/P plain steps, and γ ≤ 10 gives f = min(1, 9P/(U + H)) (`f_side`); the one-stream form, whose served
product comes from the units' MMAs, gives min(1, 9/((U + H)/P − 1)) (`f_one`).

**Proving.** The period credit (#4363's `H1CheckedCreditPeriod`: K_d = 24 draws a window, Λ = 67/8) reaches ε < 1% at
j = ⌊Λ ln(1/δ)/1%⌋ + 1 windows whatever a unit's size, so a period proves K_d·j drawn units. A drawn unit is its lanes'
chunk statements and the combine (`h1zy`: 32 n + 1, n = `h1zy.chunks`(K); a lane chunk at K = 4,096 is 48.2 M ANDs, and
the most a 2^27 block takes is K ≈ 5,041, note:compute-accounting/20261023T0607Z-draft-fp8-h1 §2), each instance priced
at `period_cost.today_rates` with 256 instances a statement (m = k_log + nbl ≤ 35).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import period_cost as PC

from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import h1zy as HZ

M = 20_480
LINEARS = (("qkv", 16_512, 16_384, 52), ("o", 16_384, 16_384, 52), ("up", 65_536, 16_384, 52),
           ("down", 16_384, 65_536, 52), ("head", 256, 16_384, 1))
MACS_PER_TOKEN = sum(n * k * cnt for _, n, k, cnt in LINEARS)
RHO_8 = 4096 * 132 * 1.83e9
SHA_RATE = 4.79e9
MAC_TIMES_PER_COMPRESSION = RHO_8 / SHA_RATE
PLAIN_EFF = 0.85
P = 1 / PLAIN_EFF
GAMMA = 10
TREE = 4
SHA_ALU = 2885
IMMA_FRACTION = 0.656029
CARRY_FIXED_WORDS = 16 + 32 + 4 * 32 + 32 + 4 + 8

ANCHOR_MS, ANCHOR_SMQ, ANCHOR_ROWS = 10.70, 25 / 32, 64
LLAMA8B = ((6144, 4096), (4096, 4096), (28_672, 4096), (4096, 14_336))
LLAMA8B_LAYERS = 32

K_D, LAMBDA, EPS = 24, Fraction(67, 8), 0.01
DELTAS = (1e-2, 1e-6)
STATEMENT_LANES = 256
V1_DRAW_S = 180.0
STEP_S = (3.7, 7.5)
HBM_BPS = 4.8e12
PERIODS = (("week", 7 * 86_400), ("month", 30 * 86_400), ("year", 365 * 86_400))


@dataclass(frozen=True)
class Seg:
    """Warp instructions a unit-segment (32 × 32 outputs, 32 of K): ALU, IMAD and all."""
    alu: float
    imad: float
    instr: float


KRM2 = Seg(alu=64 + 24 + 15, imad=83, instr=229)
KRM = Seg(alu=64 + 48 + 15, imad=83, instr=303)

H1JR_COLS_MS = 3.02
MEASURED_GAMMA = (("J10-R pd2l2-j10r (unpeeled, measured only)", "call", 19.11, "r20261019-110819-ffc6"),
                  ("per-unit pd2l2-os", "unit", 36.38, "r20261019-110819-ffc6"),
                  ("per-unit h1u-osp (peeled)", "unit", 42.10, "r20261019-112241-2a0f"))


def seg(fold: str = "fx1", unpack: bool = True, l: int = 32) -> Seg:
    """krm2's unit-segment with the fold li (½ ALU and ½ IMAD a word) in fx1's place (2 and 2), its B unpack (24 ALU, JM2's
    share) formed once a call instead, and the fold at segment length l (its words a MAC 1/l)."""
    fa, fi = (64, 64) if fold == "fx1" else (16, 16)
    fa, fi = fa * 32 / l, fi * 32 / l
    un = 24 if unpack else 0
    return Seg(KRM2.alu - 64 - 24 + fa + un, KRM2.imad - 64 + fi, KRM2.instr - (64 + 64 + 24) + fa + fi + un)


def floor_mac_times(s: Seg) -> float:
    """The unit-segment's floor in MAC-times a MAC: its 32,768 MACs take 8 SM clocks at ρ_8."""
    return max(s.alu / 16, s.imad / 16, s.instr / 32, 1 / IMMA_FRACTION)


def key_mac_times(k: int) -> float:
    """The lanes' fold keys: 32 compressions a unit at 2,885 ALU each, over its 1,024 K MACs."""
    return SHA_ALU * 32 / 64 * 4096 / (1024 * k)


def anchor() -> dict:
    """The anchor's MAC-times a MAC, measured, and its floor."""
    macs = ANCHOR_ROWS * LLAMA8B_LAYERS * sum(n * k for n, k in LLAMA8B)
    measured = ANCHOR_MS * 1e-3 * ANCHOR_SMQ * RHO_8 / macs
    keys = sum(n * k * key_mac_times(k) for n, k in LLAMA8B) / sum(n * k for n, k in LLAMA8B)
    return {"macs": macs, "measured": measured, "floor": floor_mac_times(KRM2) + keys}


def util() -> float:
    a = anchor()
    return a["floor"] / a["measured"]


def row_seg_compressions(n_bytes: float) -> float:
    """`sha512/row-seg/v1` over n bytes (each prefix a constant block, its midstate free), and TREE for leaf and node."""
    n = math.ceil(n_bytes)
    if n <= 0:
        return 0
    segs = [min(1024, n - i) for i in range(0, n, 1024)]
    return sum(-(-(s + 17) // 128) for s in segs) + -(-(64 * len(segs) + 17) // 128) + TREE


def check_closed_form() -> bool:
    for m, n, k in ((64, 64, 512), (128, 96, 1024), (256, 128, 4096)):
        lay = HT.Layout(32, 32, k)
        if Fraction(m * lay.rb + n * lay.cb + 2, (m // 32) * (n // 32)) + 2 + lay.lanes != HT.j10r_compressions(m, n, k):
            return False
    return True


@dataclass(frozen=True)
class Design:
    name: str
    title: str
    tile: tuple[int, int] = (32, 32)
    chunk: int | None = None
    noise: str = "call"
    seg: Seg | None = KRM2
    draw: str = "unit"
    checkpoint: str = "set"
    measured: str = ""
    period: str = ""


def unit_parts(d: Design, n: int, k: int) -> tuple[int, dict]:
    """(units, compressions a unit by part) for linear (n, k) under d, outside the units' lane work."""
    r, c = d.tile
    kk = d.chunk or k
    lay = HT.Layout(r, c, kk)
    units = (M // r) * (n // c) * (k // kk)
    if d.noise == "unit":
        parts = {"noise": lay.blocks + 4 + (lay.lanes if d.seg is None else 0)}
    else:
        call = HT.Layout(M, c, kk)
        parts = {"noise": (M * call.rb + n * call.cb + 2) * (k // kk) / units + 2}
    parts["receipt and z"] = row_seg_compressions(8 * lay.sets + 4 * r * c)
    if d.draw == "lane-chunk":
        ch = HZ.chunks(kk)
        words = 2 * lay.sets / ch if d.checkpoint == "set" else 2
        parts["chunk boundaries"] = lay.lanes * ch * row_seg_compressions(4 * (CARRY_FIXED_WORDS + words))
    return units, parts


def instances(d: Design, kk: int) -> int:
    lanes = (d.tile[0] // 4) * (d.tile[1] // 8)
    if d.draw == "lane-chunk":
        return 1
    return lanes * HZ.chunks(kk) + (1 if lanes > 1 else 0)


def lane_chunk_s() -> float:
    """One lane chunk's prove and Lean seconds at K = 4,096, at STATEMENT_LANES instances a statement."""
    r = PC.today_rates()
    prove = (r.prove_load_s + r.prove_session_s) / STATEMENT_LANES + r.prove_lane_s
    lean = (r.lean_process_s + r.lean_session_s) / STATEMENT_LANES + r.lean_lane_s
    return prove + lean


def evaluate(d: Design, u: float) -> dict:
    macs = M * MACS_PER_TOKEN
    comp: dict[str, float] = {}
    keys = draw_s = 0.0
    pop = 0
    inst: dict[int, int] = {}
    for _, n, k, cnt in LINEARS:
        units, parts = unit_parts(d, n, k)
        for p, v in parts.items():
            comp[p] = comp.get(p, 0.0) + cnt * units * v
        kk = d.chunk or k
        keys += cnt * M * n * k * key_mac_times(kk)
        drawn = cnt * units * ((d.tile[0] // 4) * (d.tile[1] // 8) * HZ.chunks(kk) if d.draw == "lane-chunk" else 1)
        inst[kk] = instances(d, kk)
        pop += drawn
        draw_s += drawn * (V1_DRAW_S if d.seg is None else instances(d, kk) * lane_chunk_s())
    per_mac = {p: v / macs for p, v in comp.items()}
    h = sum(per_mac.values()) * MAC_TIMES_PER_COMPRESSION
    floor = None if d.seg is None else floor_mac_times(d.seg) + keys / macs
    proj = None if floor is None else floor / u
    total = h + (proj or 0.0)
    kmax = max(k // HZ.chunks(k) for k in inst)
    return {"design": d.name, "title": d.title, "compressions_per_mac": per_mac,
            "compressions_per_mac_total": sum(per_mac.values()), "hash_mac_times": h, "lane_floor_mac_times": floor,
            "lane_projected_mac_times": proj, "f_side": min(1.0, (GAMMA - 1) * P / total),
            "f_one": min(1.0, (GAMMA - 1) / max(total / P - 1, 1e-9)),
            "f_side_floor": None if floor is None else min(1.0, (GAMMA - 1) * P / (h + floor)),
            "instances_per_draw": inst, "lane_chunk_k": kmax, "lane_chunk_ands": 1_912_609 + 11_297 * kmax,
            "fits_2_27": kmax <= 5041, "draw_s": draw_s / pop, "measured": d.measured, "period": d.period}


def pearl_c_h3() -> dict:
    """Pearl-C -h3's hashing a MAC at a 64 × 64 tile over the full k: the leaf (`sha512/row-seg/v1` over the tile's le32
    C̃ and U, 32 KiB) and 16 tickets (s512 of seed_A ‖ z, two compressions each); the rows are integrity's."""
    out = {}
    for k in (16_384, 65_536):
        c = row_seg_compressions(2 * 64 * 64 * 4) + 16 * 2
        out[k] = {"compressions_per_mac": c / (64 * 64 * k), "hash_mac_times": c / (64 * 64 * k) * MAC_TIMES_PER_COMPRESSION}
    return out


def operand_rows() -> dict:
    """Committing a unit's operand rows with it (its R A rows and C B rows of K bytes, as rows) against once a call (the
    call's A rows, integrity's activations; B is registered with the weights), a MAC at the shape's mix."""
    macs = M * MACS_PER_TOKEN
    out = {}
    for name, (r, c, kk) in (("per tile, 4×8 at K = 64", (4, 8, 64)), ("per tile, 32×32 at full K", (32, 32, None))):
        comp = sum(cnt * (M // r) * (n // c) * (k // (kk or k)) * (r + c) * row_seg_compressions(kk or k)
                   for _, n, k, cnt in LINEARS)
        out[name] = comp / macs
    out["once a call (A rows)"] = sum(cnt * M * row_seg_compressions(k) for _, n, k, cnt in LINEARS) / macs
    return {k: {"compressions_per_mac": v, "hash_mac_times": v * MAC_TIMES_PER_COMPRESSION} for k, v in out.items()}


def crosscheck() -> dict:
    """The model against what was measured on the H200: `h1jr_cols`' time at the anchor (J10-R's column blocks, once a
    call, 32 layers), and the per-layer kernel γ of Llama-3.1-8B at M = 8,192 (note:hash/20261019T0427Z-draft-dither-hb),
    priced with krm's lane work (no JM2, 303 instructions a unit-segment; those kernels predate krm, so a sanity check)."""
    cols = LLAMA8B_LAYERS * sum(n * HT.Layout(32, 32, k).cb for n, k in LLAMA8B)
    m = 8192
    macs = m * sum(n * k for n, k in LLAMA8B)
    keys = sum(n * k * key_mac_times(k) for n, k in LLAMA8B) / sum(n * k for n, k in LLAMA8B)
    u = (floor_mac_times(KRM) + keys) / util()
    unit_count = sum((m // 32) * (n // 32) for n, k in LLAMA8B)
    noise = {"call": (sum(m * HT.Layout(32, 32, k).rb + n * HT.Layout(32, 32, k).cb + 2 for n, k in LLAMA8B)
                      + 2 * unit_count) / macs,
             "unit": sum((m // 32) * (n // 32) * (HT.Layout(32, 32, k).blocks + 4) for n, k in LLAMA8B) / macs}
    rows = []
    for name, form, g, run in MEASURED_GAMMA:
        pred = (u + noise[form] * MAC_TIMES_PER_COMPRESSION) / P
        rows.append({"arm": name, "measured_gamma": g, "run": run, "model_gamma": pred,
                     "f_one_measured": min(1.0, (GAMMA - 1) / (g - 1)), "f_side_measured": min(1.0, (GAMMA - 1) / g)})
    return {"h1jr_cols_ms_model": cols / SHA_RATE * 1e3, "h1jr_cols_ms_measured": H1JR_COLS_MS, "m8192": rows}


def v5v6_row(row: dict) -> dict:
    """CMP-126's row for J10-R as built, a sampling block being one step (M rows through every static INT8 matmul, all
    its units registered): commit bytes, the range, SHA-512 seconds, the unit against C-Flock, and γ and coverage against
    the linears alone at P and against the 3.7–7.5 s step, from the projection and from the measured kernel γ."""
    macs = M * MACS_PER_TOKEN
    units = sum(cnt * (M // 32) * (n // 32) for _, n, k, cnt in LINEARS)
    z = sum(cnt * M * n * 4 for _, n, k, cnt in LINEARS)
    receipts = sum(cnt * (M // 32) * (n // 32) * 8 * HT.Layout(32, 32, k).sets for _, n, k, cnt in LINEARS)
    t_rho = macs / RHO_8
    t_lin = P * t_rho
    noise = row["compressions_per_mac"]["noise"]
    out = {"units": units, "range_log2": math.log2(units), "commit_bytes": {"z": z, "receipts": receipts},
           "a_row_bytes": sum(cnt * M * k for _, n, k, cnt in LINEARS), "sha_s": row["compressions_per_mac_total"]
           * macs / SHA_RATE, "noise_sha_s": noise * macs / SHA_RATE, "noise_over_plain": noise
           * MAC_TIMES_PER_COMPRESSION / P, "hbm_commit_s": 2 * (z + receipts) / HBM_BPS, "t_linears_s": t_lin,
           "unit_ands": {k: 32 * (1_912_609 + 11_297 * k) for k in (16_384, 65_536)}, "chunk_ands": 1_912_609
           + 11_297 * 4096}
    for src, total in (("projected", row["hash_mac_times"] + row["lane_projected_mac_times"]),
                       ("measured kernel γ", MEASURED_GAMMA[0][2] * P)):
        t_units = total * t_rho
        out[src] = {"gamma_full_linears": {"side": 1 + total / P, "one": total / P},
                    "f_linears": {"side": min(1.0, (GAMMA - 1) * P / total),
                                  "one": min(1.0, (GAMMA - 1) / (total / P - 1))},
                    "gamma_full_step": {s: {"side": 1 + t_units / s, "one": 1 + (t_units - t_lin) / s} for s in STEP_S},
                    "f_step": {s: {"side": min(1.0, (GAMMA - 1) * s / t_units),
                                   "one": min(1.0, (GAMMA - 1) * s / (t_units - t_lin))} for s in STEP_S}}
    return out


def windows(delta: float) -> int:
    return math.floor(float(LAMBDA) * math.log(1 / delta) / EPS) + 1


DESIGNS = (
    Design("v1-4x8-k64", "pinned PouwH1Tile_v1, 4×8 at K = 64, per-unit noise", tile=(4, 8), chunk=64, noise="unit",
           seg=None, measured="H200-POUW: hash-bound, 17 compressions a 2,048-MAC tile",
           period="none needed: as #4363"),
    Design("v4-32x32-k4096", "32×32 units at K = 4,096 chunks, per-unit noise (v3/v4)", chunk=4096, noise="unit",
           measured="per-unit one-stream at M = 8,192: kernel γ 36.4 (r20261019-110819-ffc6)",
           period="Scheme v3/v4 at K = 4,096; N_p, K_d, ρ as parameters"),
    Design("v4-32x32-fullk", "32×32 units at full K, per-unit noise", noise="unit",
           period="as above; h1zy refuses a chunked K over v3/v4, so v1's chunk path or a new one"),
    Design("j10r-asbuilt", "J10-R v5/v6, 32×32 at full K, krm2 units (baseline)",
           measured="H200 decode γ 4.338 vs exact-fused (r20261024-225443-dd1e), units 18.5 MAC-times a MAC; "
                    "older pd2l2-j10r kernel γ 19.11 at M = 8,192 (r20261019-110819-ffc6)",
           period="Scheme v5/v6 at call scope (J10 red team fix 2, three claim records: Daniel); J10-R lanes "
                  "(h1zy: \"a J10-R leaf has no lanes\"); N_p, K_d, ρ as parameters"),
    Design("j10r-k4096", "J10-R, 32×32 units at K = 4,096 chunks (each chunk its own unit, receipt and partial z)",
           chunk=4096, period="as j10r-asbuilt, and a check that a cell's chunk z sum to its served z"),
    Design("j10r-wcall", "J10-R + W′ = B + c + U formed once a call (or a column panel a CTA)", seg=seg(unpack=False),
           period="as j10r-asbuilt; W′ is the same bytes, so no claim changes"),
    Design("j10r-li", "J10-R + the li fold (½ ALU + ½ IMAD a word)", seg=seg(fold="li"),
           period="a new receipt Definition; ReceiptLuckAt for li (KLI₂₄, PIN): a claim, Daniel"),
    Design("j10r-wcall-li", "J10-R + W′ once a call + li", seg=seg(fold="li", unpack=False), period="as j10r-li"),
    Design("j10r-wcall-l64", "J10-R + W′ once a call + fx1 at segment length 64", seg=seg(unpack=False, l=64),
           period="Layout at L = 64; ReceiptLuckAt and Theorem K's count at L = 64: a claim, Daniel"),
    Design("j10r-lanedraw", "J10-R drawn as lane chunks, each chunk's boundary committed (per-set words)",
           draw="lane-chunk", period="unit = lane chunk; carries become outputs"),
    Design("j10r-lanedraw-cp", "J10-R drawn as lane chunks, one checkpoint a chunk", draw="lane-chunk",
           checkpoint="chunk", period="unit = lane chunk; CP-W's per-set checkpoint dropped (proj-r4's choice)"),
)


def period_rows(f: float, draw_s: float) -> list[dict]:
    rows = []
    for name, secs in PERIODS:
        macs = [secs / s * M * MACS_PER_TOKEN for s in STEP_S]
        for delta in DELTAS:
            draws = K_D * windows(delta)
            rows.append({"period": name, "delta": delta, "served_macs": macs,
                         "credited_macs": [f * (1 - EPS) * x for x in macs], "draws": draws,
                         "prove_process_days": draws * draw_s / 86_400, "processes": draws * draw_s / secs})
    return rows


def report() -> dict:
    u = util()
    rows = [evaluate(d, u) for d in DESIGNS]
    by = {r["design"]: r for r in rows}
    return {"schema": "bet-coverage/v1", "closed_form_ok": check_closed_form(), "anchor": anchor(), "util": u,
            "mac_times_per_compression": MAC_TIMES_PER_COMPRESSION, "P": P, "budget_mac_times": (GAMMA - 1) * P,
            "macs_per_step": M * MACS_PER_TOKEN, "windows": {str(d): windows(d) for d in DELTAS},
            "lane_chunk_s": lane_chunk_s(), "rows": rows, "pearl_c_h3": pearl_c_h3(), "operand_rows": operand_rows(),
            "crosscheck": crosscheck(), "v5v6": v5v6_row(by["j10r-asbuilt"]),
            "periods": {name: period_rows(by[name]["f_side"], by[name]["draw_s"])
                        for name in ("v1-4x8-k64", "j10r-asbuilt", "j10r-k4096", "j10r-wcall-li")}}


def fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float) and (abs(x) >= 1e5 or (abs(x) < 0.1 and x != 0)):
        return f"{x:.2e}"
    return f"{x:,.{nd}f}"


def markdown(rep: dict) -> str:
    a = rep["anchor"]
    out = [f"MAC-times a compression {rep['mac_times_per_compression']:,.0f}; P {rep['P']:.3f}; γ ≤ 10 budget "
           f"{rep['budget_mac_times']:.2f} MAC-times a credited MAC; anchor measured {a['measured']:.2f}, floor "
           f"{a['floor']:.2f}, utilization {rep['util']:.3f}; J10-R closed form "
           f"{'ok' if rep['closed_form_ok'] else 'WRONG'}; windows for ε < 1% {rep['windows']}; lane chunk "
           f"{rep['lane_chunk_s']:.1f} s", "",
           "| design | compressions a MAC | hash MAC-times | lane floor | lane projected | f side (at floor) | f one | "
           "instances a draw (K) | draw s |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        inst = ", ".join(f"{v} ({k})" for k, v in sorted(r["instances_per_draw"].items()))
        fl = "—" if r["f_side_floor"] is None else format(r["f_side_floor"], ".1%")
        out.append(f"| {r['design']} | {fmt(r['compressions_per_mac_total'])} | {fmt(r['hash_mac_times'])} | "
                   f"{fmt(r['lane_floor_mac_times'])} | {fmt(r['lane_projected_mac_times'])} | {r['f_side']:.1%} ({fl}) | "
                   f"{r['f_one']:.1%} | {inst} | {fmt(r['draw_s'], 0)} |")
    out += ["", "| design | noise | receipt and z | chunk boundaries |", "|---|---|---|---|"]
    for r in rep["rows"]:
        c = r["compressions_per_mac"]
        out.append(f"| {r['design']} | " + " | ".join(fmt(c.get(p)) for p in ("noise", "receipt and z", "chunk boundaries"))
                   + " |")
    out += ["", "Pearl-C -h3: " + "; ".join(f"k = {k}: {fmt(v['compressions_per_mac'])} a MAC, {v['hash_mac_times']:.2f} "
                                             f"MAC-times" for k, v in rep["pearl_c_h3"].items()),
            "", "Operand rows: " + "; ".join(f"{k}: {fmt(v['compressions_per_mac'])} a MAC, {v['hash_mac_times']:.2f} "
                                             f"MAC-times" for k, v in rep["operand_rows"].items())]
    x = rep["crosscheck"]
    out += ["", f"Cross-check: h1jr_cols at the anchor, model {x['h1jr_cols_ms_model']:.2f} ms, measured "
            f"{x['h1jr_cols_ms_measured']:.2f} ms", "", "| Llama-3.1-8B, M = 8,192 | measured γ | model γ (krm) | f one "
            "at measured γ | f side at measured γ | run |", "|---|---|---|---|---|---|"]
    for r in x["m8192"]:
        out.append(f"| {r['arm']} | {r['measured_gamma']:.2f} | {r['model_gamma']:.2f} | {r['f_one_measured']:.1%} | "
                   f"{r['f_side_measured']:.1%} | {r['run']} |")
    v = rep["v5v6"]
    out += ["", f"v5/v6 a step (one sampling block): {v['units']:,} units (range 2^{v['range_log2']:.2f}); commit z "
            f"{v['commit_bytes']['z']:.3e} B + receipts {v['commit_bytes']['receipts']:.3e} B (A rows, integrity's, "
            f"{v['a_row_bytes']:.3e} B); SHA-512 {v['sha_s']:.2f} s a step at full coverage, of it noise "
            f"{v['noise_sha_s']:.3f} s ({v['noise_over_plain']:.1%} of plain); HBM for the commit {v['hbm_commit_s']:.2f} s "
            f"(not in γ); linears {v['t_linears_s']:.2f} s; a unit {v['unit_ands'][16_384]:.3e} ANDs at K = 16,384, "
            f"{v['unit_ands'][65_536]:.3e} at 65,536; a lane chunk {v['chunk_ands']:.3e}", "",
            "| source | γ full, linears (side / one) | f at γ ≤ 10, linears | γ full, 3.7 s step | γ full, 7.5 s step | "
            "f at γ ≤ 10, 3.7 s step | f, 7.5 s step |", "|---|---|---|---|---|---|---|"]
    for src in ("projected", "measured kernel γ"):
        s = v[src]
        g, f = s["gamma_full_step"], s["f_step"]
        out.append(f"| {src} | {s['gamma_full_linears']['side']:.1f} / {s['gamma_full_linears']['one']:.1f} | "
                   f"{s['f_linears']['side']:.1%} / {s['f_linears']['one']:.1%} | "
                   f"{g[3.7]['side']:.1f} / {g[3.7]['one']:.1f} | {g[7.5]['side']:.1f} / {g[7.5]['one']:.1f} | "
                   f"{f[3.7]['side']:.1%} / {f[3.7]['one']:.1%} | {f[7.5]['side']:.1%} / {f[7.5]['one']:.1%} |")
    for name, rows in rep["periods"].items():
        out += ["", f"{name}:", "", "| period | δ | served MACs | credited MACs | draws | prove + Lean process-days | "
                "processes |", "|---|---|---|---|---|---|---|"]
        for p in rows:
            out.append(f"| {p['period']} | {p['delta']:g} | {fmt(p['served_macs'][1])}–{fmt(p['served_macs'][0])} | "
                       f"{fmt(p['credited_macs'][1])}–{fmt(p['credited_macs'][0])} | {p['draws']:,} | "
                       f"{p['prove_process_days']:,.1f} | {p['processes']:,.2f} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    rep = report()
    md = markdown(rep)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=str))
    if a.md:
        Path(a.md).write_text(md + "\n")
    print(md)
    return 0 if rep["closed_form_ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
