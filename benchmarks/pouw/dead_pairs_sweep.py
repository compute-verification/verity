#!/usr/bin/env python3
"""A reviewer's sweep of two classes against TT_OUT rev1 (CPU, bit-exact): dead pairs and the raised adder, on the H100 or,
with `--device sm120`, on the RTX PRO 6000's record.

`tests/test_tt_out_tile_h100_rev1.py` holds each class: its rows, its W1 program and the program's price. This runs them
over depths and seeds on whole 64 x 64 audit tiles (`--cases` for dead pairs, `--raised` for the raised adder), and
checks each mechanism on the scheme's own code (`--scheme`, `--raised-scheme`): `pearl_c.PearlC` at forming v1 with -h3's
SHA-512 lines forms a small tile, admits it, and computes its ticket words; the program's words are computed with the
device record's own atom and compared. `--raised-pearlc` does that on a whole 64 x 64 tile and adds the scheme's tile rule
(`clear_work.check_opened`: the per-row rules and the inert-work bound) on the served verifier's batch forms; its fill
`dead` is the dead-pair tile, and `gauss` with 0 groups an honest Gaussian tile.

`--device` picks the scheme's device record, the census's atom and the price records a row is printed at (`DEVICES`): each
row of `--raised-pearlc` names its record. The classes' rows and programs are the H100's as built; on another device
the count of the program's words that equal the scheme's says whether the class exists there.

    python benchmarks/pouw/dead_pairs_sweep.py --cases 8192:8192:1,16384:8192:2 --runs 1,2,3 --out "$OUT/cases.json"
    python benchmarks/pouw/dead_pairs_sweep.py --scheme 8192:1 --salts 3 --tile 4x4 --out "$OUT/scheme.json"
    python benchmarks/pouw/dead_pairs_sweep.py --raised 8192:8192:20:zero,8192:8192:20:gauss --units 1 --out "$OUT/raised.json"
    python benchmarks/pouw/dead_pairs_sweep.py --raised-scheme 8192:20:zero --salts 3 --tile 4x4 --out "$OUT/scheme.json"
    python benchmarks/pouw/dead_pairs_sweep.py --raised-pearlc 65536:8192+65536:170:gauss --salts 3 --units 1 --out "$OUT/pearlc.json"
    python benchmarks/pouw/dead_pairs_sweep.py --raised-pearlc 65536:8192+65536:32:ends,65536:8192+65536:170:edge --salts 3 --units 1 --out "$OUT/two.json"
    python benchmarks/pouw/dead_pairs_sweep.py --device sm120 --raised-pearlc 8192:8192:0:gauss,8192:8192:1:dead --salts 3 --units 1 --out "$OUT/sm120.json"
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ttout_tile_h100_rev1", HERE / "tests" / "test_tt_out_tile_h100_rev1.py")
T = importlib.util.module_from_spec(spec)
sys.modules["ttout_tile_h100_rev1"] = T
spec.loader.exec_module(T)
PC, H100, C = T.PC, T.H100, T.C

from verity.circuits.silicon import fp32  # noqa: E402
from verity.protocols.accounting.pouw.prover.schemes.pearl_c_device import SM120  # noqa: E402
from verity.protocols.accounting.pouw.verifier.protocol import Context, Matmul, Shape, Tile  # noqa: E402

#: the H100's tensor clock over its SM clock (the H200 card file's clock.tensor and clock.sm; Lean's
#: `h100TensorMHz / h100SmMHz`)
CLOCK = Fraction(1830, 1980)


@dataclass(frozen=True)
class W1:
    """The W1 prices a program is counted at, an E4M3 MAC being 1: an FP32 add, a BF16 MAC, one code of a cast, an INT32
    instruction, and an upper bound on one noise line (`T.LINE`, counted at the H100's prices: every price of sm_120's
    is at or under the H100's, so it bounds a line there too). SHA-512 and fetches are free, the adversary's best case."""

    fadd: Fraction
    bf: Fraction
    cast: Fraction
    int32: Fraction
    line: int = T.LINE


@dataclass(frozen=True)
class Record:
    """A price record a row is printed at: Lean's name for it, the scheme's prices (None: its device record's own), W1's
    prices for the programs, and whether a program's forming also pays ρ's FFMA on every 8th element."""

    name: str
    prices: object
    w1: W1
    stats: bool = False


def tensor_prices():
    """Lean's `Prices.h100Tensor`: the H100 record with its SM pipes priced per second at the tensor clock, the FP32 add
    at 32·CLOCK and the forming fs = ⌈32 + 64·CLOCK/2⌉ = 62, qa = ⌈32·CLOCK/8⌉ = 4."""
    return replace(H100.prices, device="sm_90-tensor-clock", fadd=32 * CLOCK, forming_credited=62, forming_a_only=4)


#: W1's prices on `W1Device.s8H100Best` (`T.CAST`, `T.INT32`), the ones `T._raised_cost` and `T._dead_pair_cost` count at
H100_W1 = W1(Fraction(H100.prices.fadd), Fraction(H100.prices.bf16_mac), Fraction(T.CAST), Fraction(T.INT32))
#: `W1Prices.sm120`: an FP32 add 8, a BF16 MAC 2, a cast 16 an instruction (8 a code), INT32 16
SM120_W1 = W1(Fraction(8), Fraction(2), Fraction(8), Fraction(16))
#: per device: the scheme's device record, the census's atom, and the price records a row is printed at. sm_120's are
#: the record's own (`Prices.sm120`, at `W1Prices.sm120`) and the one its Lean guarantee is stated at
#: (`Prices.sm120Loop`, at `W1Prices.sm120Loop`: the FP32 add at the measured loop's 1047/125, qa = 2)
DEVICES = {
    "h100": (H100, "hopper", (Record("devH100", None, H100_W1),)),
    "h100-tensor": (H100, "hopper", (Record("devH100Tensor", tensor_prices(), W1(32 * CLOCK, H100_W1.bf, T.CAST * CLOCK,
                                                                                T.INT32 * CLOCK), stats=True),)),
    "sm120": (SM120, "sm120", (
        Record("devSm120v1 Prices.sm120", None, SM120_W1),
        Record("devSm120v1 Prices.sm120Loop",
               replace(SM120.prices, device="sm_120-loop", fadd=Fraction(1047, 125), forming_a_only=2),
               replace(SM120_W1, fadd=Fraction(1047, 125))))),
}


def fires(credit: Fraction, cost: Fraction, cap: Fraction) -> bool:
    """`T._fires` at the scheme's cap: TT_OUT's event, the credited work (1 − cap)·credit beats the cost by more than the
    gap `T.G0`."""
    return (1 - cap) * credit > cost / (1 - T.G0)


def scheme_check(k: int, groups: int, salt: bytes, rows: int, cols: int, device: str = "h100") -> dict:
    """The mechanism on the scheme's own code: form `rows` x `cols` of the family with `PearlC.encode` (SHA-512 row seeds
    and lines), admit it, take its ticket words from `PearlC.checked`, and compute the program's with the device
    record's atom."""
    dev = DEVICES[device][0]
    sch = PC.PearlC(forming="v1", device=dev)
    s = Shape(8192, k, 8192)
    ctx = Context(salt, Matmul(0, s, 0), hashlib.sha512(b"root-a" + salt).digest(), hashlib.sha512(b"root-b").digest())
    XA = T._dead_pair_rows("A", rows, k, groups, C._rng("scheme", k, "A", salt.hex()))
    XB = T._dead_pair_rows("B", cols, k, groups, C._rng("scheme", k, "B"))
    a_rows = [bytes(r) for r in XA.astype("<f4").view("u1").reshape(rows, 4 * k)]
    b_rows = [bytes(r) for r in XB.astype("<f4").view("u1").reshape(cols, 4 * k)]
    ops = sch.encode(ctx, sch.noise(ctx), a_rows, b_rows, Tile(tuple(range(rows)), tuple(range(cols))))
    assert ops is not None
    admitted = sch.admissible(ctx, ops)
    checked = sch.checked(ctx, ops)
    top = PC.P.f32_to_fp8(448.0)
    live = [128 * g + l for g in range(groups) for l in range(0, 128, 8)]
    dead = [l for l in range(128 * groups) if l % 8]
    want_a = {448.0, 320.0, 288.0}
    dead_ok = all(abs(PC.P.fp8_to_f32(ops.a.codes[i][l])) in want_a for i in range(rows) for l in dead) and \
        all(abs(PC.P.fp8_to_f32(ops.b.codes[j][l])) in {448.0, 320.0, 352.0} for j in range(cols) for l in dead)
    equal, atoms = 0, 0
    for i in range(rows):
        for j in range(cols):
            a, b = ops.a.codes[i], ops.b.codes[j]
            acc, atoms = PC.P.f32_bits(4096.0 * groups), 0
            for lo in range(0, len(live), 30):
                part = live[lo:lo + 30]
                xa = [top, top] + [a[l] for l in part] + [0] * (30 - len(part))
                xb = [top, top | 0x80] + [b[l] for l in part] + [0] * (30 - len(part))
                acc, atoms = dev.atom.step(acc, xa, xb), atoms + 1
            total = acc
            for ws in range(128 * groups, k, 128):
                local = 0
                for t in range(ws, ws + 128, 32):
                    local = dev.atom.step(local, a[t:t + 32], b[t:t + 32])
                total = fp32.add(total, local)
                total = 0 if total == 0x80000000 else total
            equal += total == checked.c[i][j]
    return {"class": "dead-pairs", "scheme": sch.name, "k": k, "groups": groups, "salt": salt.hex()[:16],
            "tile": (rows, cols), "admitted": bool(admitted),
            "dead_codes_as_designed": bool(dead_ok), "words": rows * cols, "equal": int(equal),
            "atoms": atoms + (k - 128 * groups) // 32, "reference_atoms": k // 32}


def raised_scheme_check(k: int, groups: int, fill: str, salt: bytes, rows: int, cols: int, device: str = "h100") -> dict:
    """The raised adder on the scheme's own code, as `scheme_check`: the planted codes as designed, every other code of the
    planted groups' later atoms under 8, and against `PearlC.checked`'s words the words of the program that runs one atom a
    planted group (its accumulator the planted sum) and of the one that only skips each planted group's second and third
    atoms."""
    dev = DEVICES[device][0]
    sch = PC.PearlC(forming="v1", device=dev)
    s = Shape(8192, k, 8192)
    ctx = Context(salt, Matmul(0, s, 0), hashlib.sha512(b"root-a" + salt).digest(), hashlib.sha512(b"root-b").digest())
    XA = T._raised_rows("A", rows, k, groups, C._rng("scheme", "raised", k, "A", salt.hex()), fill)
    XB = T._raised_rows("B", cols, k, groups, C._rng("scheme", "raised", k, "B"), fill)
    a_rows = [bytes(r) for r in XA.astype("<f4").view("u1").reshape(rows, 4 * k)]
    b_rows = [bytes(r) for r in XB.astype("<f4").view("u1").reshape(cols, 4 * k)]
    ops = sch.encode(ctx, sch.noise(ctx), a_rows, b_rows, Tile(tuple(range(rows)), tuple(range(cols))))
    assert ops is not None
    admitted = sch.admissible(ctx, ops)
    checked = sch.checked(ctx, ops)
    val = PC.P.fp8_to_f32
    planted_ok, largest = True, 0.0
    for side, codes, plant in (("A", ops.a.codes, T.PLANT_A), ("B", ops.b.codes, T.PLANT_B)):
        for row in codes:
            for g in range(groups):
                top = -448.0 if side == "B" and g % 2 else 448.0
                planted_ok &= all(val(row[128 * g + l]) == top for l in (1, 2, 3))
                planted_ok &= all(val(row[128 * g + 32 * t + 1]) == v for t, v in zip((1, 2, 3), plant[g % 2]))
                largest = max([largest] + [abs(val(row[l])) for l in range(128 * g + 32, 128 * g + 128) if l % 32 != 1])
    equal = equal_skip = 0
    for i in range(rows):
        for j in range(cols):
            a, b = ops.a.codes[i], ops.b.codes[j]
            total = total_skip = 0
            for g in range(k // 128):
                local = 0
                for t in range(128 * g, 128 * g + 128, 32):
                    local = dev.atom.step(local, a[t:t + 32], b[t:t + 32])
                skipped = local
                if g < groups:
                    local = dev.atom.step(PC.P.f32_bits(T.PLANT_SUM[g % 2]), a[128 * g:128 * g + 32], b[128 * g:128 * g + 32])
                    skipped = 0
                    for t in (128 * g, 128 * g + 96):
                        skipped = dev.atom.step(skipped, a[t:t + 32], b[t:t + 32])
                total, total_skip = fp32.add(total, local), fp32.add(total_skip, skipped)
                total = 0 if total == 0x80000000 else total
                total_skip = 0 if total_skip == 0x80000000 else total_skip
            equal += total == checked.c[i][j]
            equal_skip += total_skip == checked.c[i][j]
    return {"class": "raised-adder", "scheme": sch.name, "fill": fill, "k": k, "groups": groups, "salt": salt.hex()[:16],
            "tile": (rows, cols),
            "admitted": bool(admitted), "planted_codes_as_designed": bool(planted_ok),
            "largest_other_code_in_later_atoms": largest, "words": rows * cols, "equal": int(equal),
            "atoms": k // 32 - 3 * groups, "equal_skip_only": int(equal_skip), "atoms_skip_only": k // 32 - 2 * groups,
            "reference_atoms": k // 32}


#: the two-raiser variants of the raised adder (`_two_raiser_rows`), named as `--raised-pearlc`'s fill
TWO_RAISER = ("ends", "edge")
#: how far "edge" moves B's planted entries off their cells' centres: past the cell's edge at one end of the noise
#: support (about 6 codes off), inside the cell at the salts drawn
EDGE_OFF, EDGE_RAISER = 3.0, 436.0


def _two_raiser_rows(side: str, rows: int, k: int, groups: int, rng: np.random.Generator, fill: str) -> np.ndarray:
    """`rows` rows of one operand of a raised-adder variant aimed at the debit's forced-zero term, which charges a product
    only where the adder cuts it at both ends of the noise support. Each of the first `groups` promotion groups holds two
    top-code entries (positions 1 and 2 of its first atom, −448 on B in an odd group), the raised rows' plants at position
    1 of each later atom, and zero at its other 123 positions, whose codes are noise alone. Two top codes make the
    group's accumulator 2·448² = 401,408, under 2^19, so the adder cuts a later product under 32, not under 64: at
    either end of the support two zero entries' codes are both 6 or both −6 and their product 36 is kept, while at a
    salt drawn their product is far under 32 and is cut.

    With `fill` "edge" B's planted entries are moreover not salt-dead: its raisers sit at α·x = ±436 (448 at a salt
    drawn, 416 at one end) and its plants 3 off their cells' centres (the centre's code at a salt drawn, the next cell's
    at one end), with one spike a row at k − 2 holding α. No planted pair is then dead in both rows."""
    assert fill in TWO_RAISER and groups % 2 == 0 and 5 * groups * 64 + 64 <= k, "liveness admits k/64 large entries a row"
    X = rng.standard_normal((rows, k))
    X[:, :128 * groups] = 0.0
    rho = np.sqrt((X[:, ::8] ** 2).mean(1))
    alpha = C.Q_MAX / ((T.SPIKE + 4.0) * rho)
    edge = fill == "edge" and side == "B"
    if edge:
        X[:, k - 2] = T.SPIKE * rho
    for g in range(groups):
        sign = -1.0 if side == "B" and g % 2 else 1.0
        for l in (1, 2):
            X[:, 128 * g + l] = sign * EDGE_RAISER / alpha if edge else sign * T.SPIKE * rho
        for t, v in zip((1, 2, 3), (T.PLANT_A if side == "A" else T.PLANT_B)[g % 2]):
            X[:, 128 * g + 32 * t + 1] = (v + (np.sign(v) * EDGE_OFF if edge else 0.0)) / alpha
    return X.astype(np.float32).astype(np.float64)


def raised_cost(s: Shape, atoms: int, units: int, w1: W1, stats: bool = False) -> Fraction:
    """`T._raised_cost` at the W1 prices `w1`, term for term: at `H100_W1` it is that function. With `stats` the forming
    also counts ρ's FFMA on every 8th element (an FP32 add over 8, an element), W1's exact forming."""
    m, k, n = s.m, s.k, s.n
    fadd, bf, cast = w1.fadd, w1.bf, w1.cast
    forming = 32 + cast + (fadd / 8 if stats else 0)
    chain = m * n * (32 * atoms + fadd * (k // 128 - 1))
    fused = m * PC.R * (k + fadd * (k // 128 - 1))
    peel = m * n * bf * PC.PEEL_K + m * PC.PEEL_K * cast
    job = 2 * (n * k * forming + (n + PC.R) * PC.R * (k + fadd * (k // 128)) + n * PC.R * PC.R * fadd
               + n * PC.PEEL_K * cast + (n + 2 * k) * w1.line)
    return Fraction(units * (chain + fused + peel + m * k * forming + m * w1.line) + job)


def dead_pair_cost(s: Shape, groups: int, atoms: int, units: int, w1: W1) -> Fraction:
    """`T._dead_pair_cost` at the W1 prices `w1`, term for term: at `H100_W1` it is that function."""
    m, k, n = s.m, s.k, s.n
    fadd, bf, cast, int32 = w1.fadd, w1.bf, w1.cast, w1.int32
    chain = m * n * (32 * atoms + fadd * (k // 128 - groups))
    fused = m * PC.R * (k + fadd * (k // 128 - 1)) + m * T.G * groups * 32 * 32
    peel = m * n * bf * PC.PEEL_K + m * PC.PEEL_K * cast
    forming = m * (k - 112 * groups) * (32 + cast)
    rows = m * (w1.line + 16 * groups * int32)
    job = 2 * (n * k * (32 + cast) + (n + PC.R) * PC.R * (k + fadd * (k // 128)) + n * PC.R * PC.R * fadd
               + n * PC.PEEL_K * cast + (n + 2 * k) * w1.line + n * 16 * groups * int32)
    return Fraction(units * (chain + fused + peel + forming + rows) + job)


def raised_pearlc_tile(k: int, ns: list[int], groups: int, fill: str, salt: bytes, rows: int = 64, cols: int = 64,
                       units: int = 1, cross: int = 64, scalar: bool = False, codes_only: bool = False,
                       device: str = "h100") -> list[dict]:
    """A class's whole audit tile that the scheme's own code forms, admits, checks and holds to its tile rule:
    `PearlC.encode` at forming v1 with -h3's SHA-512 row seeds and lines, `admissible`, `checked`'s ticket words, and
    `clear_work.check_opened`, the per-row rules and the inert-work bound on the tile and on each credited row, all on
    `fast_tile`'s batch forms (the served verifier's `--fast`). `fill` names the tile: the raised adder's `gauss` and
    `zero`, the two-raiser `ends` and `edge`, and `dead`, the dead-pair tile; `gauss` with 0 groups is an honest
    Gaussian tile, whose one program is the reference chain. The programs' words are computed on the scheme's codes at
    the census's atom and compared with `checked`'s on every word, and `cross` of `checked`'s words are recomputed with
    the scalar chain. `device` names the entry of `DEVICES`: its device record forms and checks the tile, and the
    result has one row per n of `ns`, per price record of the entry and per program: the forming reads neither n nor
    the prices; the rule, the bound, the credit and the cost do. `scalar` leaves the batch forms out: the forming, the
    check and the tile rule then run on the scalar reference, and only the count of flagged atoms in planted groups is
    the batch form's, held to the reference's total. `codes_only` stops after the forming and lists the planted codes
    that are not the designed ones at the salt drawn. A row's `fires` is TT_OUT's event for its program, and None where
    the program's words are not all the scheme's: a program that outputs other words is no program of the class."""
    sys.path.insert(0, str(HERE / "pearl_c_vllm"))
    sys.path.insert(0, str(HERE.parents[1] / "verity" / "protocols" / "accounting" / "pouw" / "clear"))
    import clear_work as CW
    import fast_tile as FT
    from verity.protocols.accounting.pouw.prover.schemes import pearl_c_debit as D
    from verity.protocols.accounting.pouw.prover.schemes.pearl_c_work import UnitRecord

    dev, atom, records = DEVICES[device]
    assert dev.g == T.G, "the programs promote every T.G atoms"
    C.set_atom(atom)
    schemes = [PC.PearlC(forming="v1", device=dev, **({} if r.prices is None else {"prices": r.prices})) for r in records]
    sch = schemes[0]
    scalar_chain = PC.chain
    undo = (lambda: None) if scalar else FT.install(sch)
    try:
        clock = {"start": time.perf_counter()}

        def lap(name: str) -> None:
            now = time.perf_counter()
            clock[name] = round(now - clock["start"], 1)
            clock["start"] = now

        tile = Tile(tuple(range(rows)), tuple(range(cols)))
        u = Matmul(0, Shape(8192, k, ns[0]), 0)
        ctx = Context(salt, u, hashlib.sha512(b"root-a" + salt).digest(), hashlib.sha512(b"root-b").digest())
        if fill == "dead":
            XA = T._dead_pair_rows("A", rows, k, groups, C._rng("scheme", k, "A", salt.hex()))
            XB = T._dead_pair_rows("B", cols, k, groups, C._rng("scheme", k, "B"))
        else:
            make = _two_raiser_rows if fill in TWO_RAISER else T._raised_rows
            XA = make("A", rows, k, groups, C._rng("scheme", "raised", k, "A", salt.hex()), fill)
            XB = make("B", cols, k, groups, C._rng("scheme", "raised", k, "B"), fill)
        a_rows = [bytes(r) for r in XA.astype("<f4").view("u1").reshape(rows, 4 * k)]
        b_rows = [bytes(r) for r in XB.astype("<f4").view("u1").reshape(cols, 4 * k)]
        ops = sch.encode(ctx, sch.noise(ctx), a_rows, b_rows, tile)
        assert ops is not None
        lap("encode")
        admitted = bool(sch.admissible(ctx, ops))
        checked = sch.checked(ctx, ops)
        lap("checked")
        a, b = np.array(ops.a.codes, dtype=np.uint8), np.array(ops.b.codes, dtype=np.uint8)
        value = np.array([np.nan if c & 0x7F == 0x7F else PC.P.fp8_to_f32(c) for c in range(256)])
        largest, product, off = 0.0, 0.0, []
        if fill == "dead":
            # every position off the every-8th grid of a planted group is dead, at one of the family's three codes
            dead = np.array([l for l in range(128 * groups) if l % 8], dtype=np.int64)
            for side, codes, allowed in (("A", a, (448.0, 320.0, 288.0)), ("B", b, (448.0, 320.0, 352.0))):
                vals = np.abs(value[codes][:, dead])
                off += [(side, int(i), int(dead[j]) // 128, int(dead[j]) % 128, allowed, float(vals[i, j]))
                        for i, j in zip(*np.nonzero(~np.isin(vals, allowed)))]
            planted = (rows + cols) * 112 * groups
        else:
            peak = []
            later = np.array([128 * g + l for g in range(groups) for l in range(32, 128) if l % 32 != 1], dtype=np.int64)
            for side, codes, plant in (("A", a, T.PLANT_A), ("B", b, T.PLANT_B)):
                vals = value[codes]
                for g in range(groups):
                    top = -448.0 if side == "B" and g % 2 else 448.0
                    designed = [(128 * g + l, top) for l in range(1, 3 if fill in TWO_RAISER else 4)]
                    designed += [(128 * g + 32 * t + 1, v) for t, v in zip((1, 2, 3), plant[g % 2])]
                    for pos, v in designed:
                        off += [(side, int(i), g, pos - 128 * g, v, float(vals[i, pos])) for i in np.flatnonzero(vals[:, pos] != v)]
                if groups:
                    largest = max(largest, float(np.abs(vals[:, later]).max()))
                    peak.append(np.abs(vals[:, later]).max(0))
            planted = (rows + cols) * groups * (5 if fill in TWO_RAISER else 6)
            product = float((peak[0] * peak[1]).max()) if groups else 0.0
        planted_ok = not off
        want = np.array(checked.c, dtype=np.uint32).reshape(-1)
        sample = [(i * rows // cross, (5 * i) % cols) for i in range(min(cross, rows * cols))]
        scalar_equal = sum(scalar_chain(ops.a.codes[i], ops.b.codes[j], dev) == int(want[i * cols + j]) for i, j in sample)
        lap("codes")
        cls = ("dead-pairs" if fill == "dead" else "honest" if groups == 0 else "two-raiser" if fill in TWO_RAISER
               else "raised-adder")
        if codes_only:
            return [{"class": cls, "scheme": sch.name, "fill": fill, "shape": (8192, k, ns[0]), "groups": groups,
                     "salt": salt.hex()[:16], "admitted": admitted,
                     "planted_codes_as_designed": planted_ok, "planted_entries": planted,
                     "off_design": len(off), "off_design_first": off[:40],
                     "scalar_chain_agrees": f"{scalar_equal}/{len(sample)}"}]
        ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        A, B = a[ii.ravel()], b[jj.ravel()]
        if fill == "dead":
            fns = (("dead-pair", T._dead_pair_program),)
        elif groups == 0:
            fns = (("reference", T._raised_program),)
        else:
            fns = (("one-atom", T._raised_program), ("skip", T._raised_skip_program))
        programs = {}
        for name, fn in fns:
            words, atoms = fn(A, B, groups)
            bits = words.astype(np.float32).view(np.uint32).copy()
            bits[bits == 0x80000000] = 0
            apart = np.abs(words.astype(np.float64) - want.view(np.float32).astype(np.float64))
            programs[name] = (int((bits == want).sum()), atoms, float(apart.max()))
        del A, B
        lap("programs")
        flags, _ = FT._flags(FT._rows(ops.a.codes), FT._rows(ops.b.codes), dev)
        in_planted = int(flags[:, :T.G * groups].sum())
        formed_a = [(ops.a.alpha[p], ops.a.x[p], ops.a.beta[p]) for p in range(rows)]
        if "formed_b" in inspect.signature(D.tile_debit).parameters:
            # a debit that reads B's forming too and counts the words' mends: the scheme's own `tile_debit` at this tree
            formed_b = [(ops.b.alpha[p], ops.b.x[p], ops.b.beta[p]) for p in range(cols)]
            atoms_f, zeros, elements, mends = D.tile_debit(ops.a.codes, ops.b.codes, formed_a, formed_b, dev)
        else:
            (atoms_f, zeros, elements), mends = D.tile_debit(ops.a.codes, ops.b.codes, formed_a, dev), None
        assert atoms_f == int(flags.sum())
        lap("debit")
        out = []
        for n in ns:
            s = Shape(8192, k, n)
            record = UnitRecord(Matmul(0, s, 0), real_rows=s.m, k_kept=k)
            for r, priced in zip(records, schemes):
                priced.check(Matmul(0, s, 0))
                rule = CW.check_opened(priced, record, tile, a_rows, b_rows, ops)
                lap(f"rule n={n}" + (f" {r.name}" if len(records) > 1 else ""))
                debit = D.debit_units(atoms_f, Fraction(cols * elements, n), priced.prices, dev, zeros=zeros,
                                      **({} if mends is None else {"mends": mends}))
                cap = priced.cap * CW.cell_credit(priced, record) * rows * cols
                per_word = Fraction(priced.credit_of(s), s.m * s.n)
                credit = units * Fraction(priced.credit_of(s))
                for name, (equal, atoms, apart) in programs.items():
                    if fill == "dead":
                        assert dead_pair_cost(s, groups, atoms, units, H100_W1) == T._dead_pair_cost(s, groups, atoms, units)
                        cost = dead_pair_cost(s, groups, atoms, units, r.w1)
                    else:
                        assert raised_cost(s, atoms, units, H100_W1) == T._raised_cost(s, atoms, units)
                        cost = raised_cost(s, atoms, units, r.w1, r.stats)
                    assert priced.cap != T.RHO or fires(credit, cost, priced.cap) == T._fires(credit, cost)
                    out.append({"class": cls, "scheme": priced.name, "record": r.name, "cap_share": str(priced.cap),
                                "forming": "PearlC v1, -h3 SHA-512 lines", "prices": priced.prices.device,
                                "forms": "scalar reference" if scalar else "fast_tile batch", "program": name, "fill": fill,
                                "shape": (s.m, k, n), "groups": groups, "units": units, "salt": salt.hex()[:16],
                                "admitted": admitted, "tile_rule": rule, "planted_codes_as_designed": planted_ok,
                                "off_design": len(off), "largest_other_code_in_later_atoms": largest,
                                "largest_other_product_in_later_atoms": product,
                                "words": rows * cols, "equal": equal, "largest_word_difference": apart,
                                "scalar_chain_agrees": f"{scalar_equal}/{len(sample)}", "atoms": atoms,
                                "reference_atoms": k // 32, "flagged": atoms_f, "flagged_in_planted_groups": in_planted,
                                "later_zero": int(zeros), "elements": int(elements),
                                "mends": None if mends is None else int(mends), "debit": float(debit), "cap": float(cap),
                                "debit_over_cap": bool(debit > cap),
                                "cost": float(cost / (units * s.m * s.n)), "credit": float(per_word),
                                "bound": float((1 - T.G0) * (1 - priced.cap) * per_word),
                                "fires": fires(credit, cost, priced.cap) if equal == rows * cols else None})
        for o in out:
            o["seconds"] = {key: v for key, v in clock.items() if key != "start"}
        return out
    finally:
        undo()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="", help="k:n:groups, comma separated")
    ap.add_argument("--runs", default="1")
    ap.add_argument("--units", type=int, default=256)
    ap.add_argument("--scheme", default="", help="k:groups, comma separated")
    ap.add_argument("--raised", default="", help="k:n:groups:fill[:program] (fill: zero or gauss; program: one-atom or skip), comma separated")
    ap.add_argument("--raised-scheme", default="", help="k:groups:fill, comma separated")
    ap.add_argument("--salt-from", type=int, default=0, help="the first salt's index, for --raised-pearlc")
    ap.add_argument("--scalar", action="store_true", help="--raised-pearlc on the scalar reference, without fast_tile's batch forms")
    ap.add_argument("--raised-pearlc", default="", help="k:n[+n...]:groups:fill (fill: gauss, zero, the two-raiser variants ends and edge, or dead, the "
                                                    "dead-pair tile; gauss with 0 groups is an honest tile), comma separated: whole tiles on the "
                                                    "scheme's code")
    ap.add_argument("--device", default="h100", choices=("h100", "sm120"), help="the scheme's device record, the census's atom and the price records "
                                                                                 "(--scheme, --raised-scheme, --raised-pearlc)")
    ap.add_argument("--codes-only", action="store_true", help="--raised-pearlc: form the tile and list the planted codes off their design")
    ap.add_argument("--tensor-clock", action="store_true", help="--raised-pearlc at the H100 record per second at the tensor clock")
    ap.add_argument("--salts", type=int, default=1)
    ap.add_argument("--tile", default="4x4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.device != "h100" and (args.cases or args.raised or args.tensor_clock):
        ap.error("--cases, --raised and --tensor-clock are the H100's: the test module forms and prices them there")
    device = "h100-tensor" if args.tensor_clock else args.device
    out = {"cases": [], "scheme": []}
    for spec_ in filter(None, args.cases.split(",")):
        k, n, groups = map(int, spec_.split(":"))
        for run in map(int, args.runs.split(",")):
            t0 = time.perf_counter()
            case = T._dead_pair_case(k=k, n=n, groups=groups, units=args.units, run=run)
            case["seconds"] = round(time.perf_counter() - t0, 1)
            out["cases"].append(case)
            print(json.dumps(case), flush=True)
    for spec_ in filter(None, args.raised.split(",")):
        k, n, groups, fill, program = (spec_.split(":") + ["one-atom"])[:5]
        for run in map(int, args.runs.split(",")):
            t0 = time.perf_counter()
            case = T._raised_case(k=int(k), n=int(n), groups=int(groups), units=args.units, run=run, fill=fill,
                                  program=program)
            case["seconds"] = round(time.perf_counter() - t0, 1)
            out["cases"].append(case)
            print(json.dumps(case), flush=True)
    for spec_ in filter(None, args.raised_pearlc.split(",")):
        k, ns, groups, fill = spec_.split(":")
        for i in range(args.salt_from, args.salt_from + args.salts):
            for r in raised_pearlc_tile(int(k), [int(n) for n in ns.split("+")], int(groups), fill,
                                        hashlib.sha512(b"raised/salt/%d" % i).digest()[:32], units=args.units,
                                        scalar=args.scalar, codes_only=args.codes_only, device=device):
                out["scheme"].append(r)
                print(json.dumps(r), flush=True)
    rows, cols = map(int, args.tile.split("x"))
    for spec_ in filter(None, args.raised_scheme.split(",")):
        k, groups, fill = spec_.split(":")
        for i in range(args.salts):
            t0 = time.perf_counter()
            r = raised_scheme_check(int(k), int(groups), fill, hashlib.sha512(b"raised/salt/%d" % i).digest()[:32], rows, cols,
                                    args.device)
            r["seconds"] = round(time.perf_counter() - t0, 1)
            out["scheme"].append(r)
            print(json.dumps(r), flush=True)
    for spec_ in filter(None, args.scheme.split(",")):
        k, groups = map(int, spec_.split(":"))
        for i in range(args.salts):
            t0 = time.perf_counter()
            r = scheme_check(k, groups, hashlib.sha512(b"dead-pairs/salt/%d" % i).digest()[:32], rows, cols, args.device)
            r["seconds"] = round(time.perf_counter() - t0, 1)
            out["scheme"].append(r)
            print(json.dumps(r), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
