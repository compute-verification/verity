"""The class-of-record window's commit against CMP-149's target (VLLM-128): a `class_rows.py` census priced in SHA-512
compressions, seconds on the H200 and AND gates, beside the served window's ANDs and seconds, each lever alone and all of them
stacked; the pad pass (positions whose values the engine never computed, recomputed) at three traffics; the registration's
size; and the totals over monitoring periods of one week, one month and one year.

Costs. A fresh output row is its digest x (`class_rows.blocks(128 + b)`: P(8 nbytes) and its b bytes) and `Costs.fresh` more
(hm96's c and h, two each, the frame leaf, and a port-tree node); an input row a wire or a registered read fills is re-leafed in
its reader's port tree for `Costs.reused` (h, the leaf, the node). `CENSUS` is class_rows' model, a node one compression;
`ENGINE` is the HASH-20 engine's, a node two (`commit_slots`: nine compressions a one-block row-slot). Compressions are priced
in ANDs at `SHA512_ANDS` each; the served window is the class circuit's ANDs (`unit_ands`).

Levers (`LEVERS`), each a change to the circuit's layout or the commitment's format, not to what the model computes:

- ``L1``, one FA3 tile. `fa3_tile` launches 64 × 192 when max_seqlen_q · (query heads a KV head) <= 64 and 192 × 128
  otherwise, so at Llama-3.2-1B a pure decode launch has key blocks of 192 and a prefill or mixed launch blocks of 128, and the
  class circuit, cut at the decode's 192, is inexact past 128 keys for every prompt position and every decode row of a mixed
  launch (r20261026-230245-33cf). One tile at every launch (the pinned FA3's ``use_one_mma_wg`` off) makes every position the
  circuit's at blocks of 128 (`Heads.units` at ``BN=128``), so the engine's own values are the circuit's: no pad pass for a
  position the engine ran.
- ``L2``, constants. An empty slot computes token 0 and then decodes greedily, a function of the weights and the class, and
  under a length-masked class Program (V3b's) a position past its request's end outputs zero rows: their rows are computed
  once and kept, and no pad pass runs them. Their hiding leaves stay fresh every window, since which slots are filled is
  private (their digests x could be kept too, which this doesn't price).
- ``L3a``, key groups. A first block of w keys reads its keys and values as one `AllGather_v1{w, D}` row re-gathered at every
  position (O(BN²) rows a block); read as ⌈w / 16⌉ rows of 16-key groups, each gathered once when it fills, only the last
  partial group is re-gathered.
- ``L3b``, one state row. A key block's running state (max, lane sums, O) is three rows, each hashed and re-leafed by its
  reader; as one row it is one.
- ``W`` and ``R``, ports by reference: a reader's wired or registered port is the writer's or the pinned weight's leaves
  (a drawn member's port tree is derived at the audit), so no input row is re-leafed.
- ``P``, shared output rows (CIR-113, `commit_slots --row-bytes`): units' outputs packed into rows of `ROW_BYTES`, so a
  fresh row's fixed cost is paid a KiB, not a unit's few bytes (its readers re-leaf as many rows as before, and a unit reads
  its segment for a few thousand ANDs, which this doesn't price).
- ``L4``, one member root: the registration is O(members) today (`Registration`), every member's public body (every row's
  b || c, 128 bytes) SHA-512'd on the CPU for `rows.leaf_layer` and written out, and every port root a JSON entry; under one
  Merkle root over members, each member's leaf over its port roots, it is a root, and an audit opens K members in O(K log M).

    python window_cost.py CENSUS.json [--target h200|sm120] [--served-s 1.7] [--step-ms 23] [--project S,PROMPT,OUTPUTS]
        [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from verity.core.protocols.sampled_proofs.optimize import SHA512_ANDS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import class_rows as CR  # noqa: E402
import class_size as CS  # noqa: E402

FIRST, END, GATHER = "AttnBlockFirst_v1", "AttnHeadEnd_v1", "AllGather_v1"
#: a full key block by target: FA3's (sm_90) and FA2's (sm_120)
BLOCKS = {"h200": "AttnBlock_v4", "sm120": "AttnBlock_v5"}
#: prefill is exact against the decode-shaped Definitions at this many keys or fewer (r20261026-230245-33cf)
EXACT_KEYS = 128
#: the key group `L3a` reads a first block's keys in
GROUP = 16
#: a shared output row's bytes (`P`): CIR-113's 1 KiB rows
ROW_BYTES = 1024
#: a member's units: V3c's cap (note:20261026T2036Z-report-v3c-window-commit Q3)
CAP = 64
#: one port root's entry in `registration.json` (``indent=1``): port, 128 hex of root and of domain, leaves
ROOT_JSON_BYTES = 330
#: the H200's rates, compressions a second: #4666's measured step and commit_gpu's peak (`commit_slots`)
RATES = {"3.1 G/s": 3.1e9, "4.79 G/s": 4.79e9}
#: the reference datacenter (`benchmarks/census/deployments.json`, dc-12500-machines-600s)
GPUS = 12_500 * 8
PERIODS = {"week": 7 * 86_400, "month": 30 * 86_400, "year": 365 * 86_400}


@dataclass(frozen=True)
class Costs:
    name: str
    fresh: int
    reused: int


CENSUS = Costs("census", CR.FRESH_EXTRA, CR.REUSED)
ENGINE = Costs("engine", CR.FRESH_EXTRA + 1, CR.REUSED + 1)

#: ANDs a unit on the H200 (int8 linears, FA3 attention), each `BS.compile(...).counts["and"]` of its Boolean twin at
#: GEMM_SM90_S8 / HopperBF16WgmmaDot16 (`tests/test_window_cost.py` holds them to it): `ScaledMmS8Coordinate_v2{K}` and
#: `Int8RowQuant_v2{K}` by K, `GemmCoordinate_v3{K}` by K, `AttnBlock_v9{D=64, NVIS=BN, FIRST=False, BN, CHECK}` by (BN, CHECK)
S8_COORD = {2048: 271_342, 8192: 1_069_678}
ROW_QUANT = {2048: 7_727_887, 8192: 30_903_055}
BF16_COORD = {2048: 1_007_682}
FA3_FULL = {(128, False): 8_131_559, (128, True): 8_131_622, (192, False): 12_107_687, (192, True): 12_107_750}
#: `AttnBlock_v9{D=64, NVIS, FIRST=True, BN, CHECK=True}`'s ANDs at NVIS = 1 ... 192, the same at every BN >= NVIS (FA3's first
#: block, `AttnBlockFirst_v1`, is FA2's word for word)
FA3_FIRST = (
    125_257, 186_040, 241_007, 296_926, 355_285, 411_140, 466_171, 525_418, 581_273, 637_192, 694_775, 750_630, 806_549,
    864_196, 920_051, 975_970, 1_117_073, 1_177_664, 1_233_519, 1_289_438, 1_348_621, 1_404_476, 1_460_395, 1_519_578,
    1_575_433, 1_631_352, 1_688_935, 1_744_790, 1_800_709, 1_858_356, 1_914_211, 1_970_130, 2_114_049, 2_174_640, 2_230_495,
    2_286_414, 2_345_597, 2_401_452, 2_457_371, 2_516_554, 2_572_409, 2_628_328, 2_685_911, 2_741_766, 2_797_685, 2_855_332,
    2_911_187, 2_967_106, 3_108_209, 3_168_800, 3_224_655, 3_280_574, 3_339_757, 3_395_612, 3_451_531, 3_510_714, 3_566_569,
    3_622_488, 3_680_071, 3_735_926, 3_791_845, 3_849_492, 3_905_347, 3_961_266, 4_107_185, 4_167_776, 4_223_631, 4_279_550,
    4_338_733, 4_394_588, 4_450_507, 4_509_690, 4_565_545, 4_621_464, 4_679_047, 4_734_902, 4_790_821, 4_848_468, 4_904_323,
    4_960_242, 5_101_345, 5_161_936, 5_217_791, 5_273_710, 5_332_893, 5_388_748, 5_444_667, 5_503_850, 5_559_705, 5_615_624,
    5_673_207, 5_729_062, 5_784_981, 5_842_628, 5_898_483, 5_954_402, 6_102_321, 6_162_912, 6_218_767, 6_274_686, 6_333_869,
    6_389_724, 6_445_643, 6_504_826, 6_560_681, 6_616_600, 6_674_183, 6_730_038, 6_785_957, 6_843_604, 6_899_459, 6_955_378,
    7_096_481, 7_157_072, 7_212_927, 7_268_846, 7_328_029, 7_383_884, 7_439_803, 7_498_986, 7_554_841, 7_610_760, 7_668_343,
    7_724_198, 7_780_117, 7_837_764, 7_893_619, 7_949_538, 8_093_457, 8_154_048, 8_209_903, 8_265_822, 8_325_005, 8_380_860,
    8_436_779, 8_495_962, 8_551_817, 8_607_736, 8_665_319, 8_721_174, 8_777_093, 8_834_740, 8_890_595, 8_946_514, 9_087_617,
    9_148_208, 9_204_063, 9_259_982, 9_319_165, 9_375_020, 9_430_939, 9_490_122, 9_545_977, 9_601_896, 9_659_479, 9_715_334,
    9_771_253, 9_828_900, 9_884_755, 9_940_674, 10_090_593, 10_151_184, 10_207_039, 10_262_958, 10_322_141, 10_377_996,
    10_433_915, 10_493_098, 10_548_953, 10_604_872, 10_662_455, 10_718_310, 10_774_229, 10_831_876, 10_887_731, 10_943_650,
    11_084_753, 11_145_344, 11_201_199, 11_257_118, 11_316_301, 11_372_156, 11_428_075, 11_487_258, 11_543_113, 11_599_032,
    11_656_615, 11_712_470, 11_768_389, 11_826_036, 11_881_891, 11_937_810)


def unit_ands(lab: str, target: str = "h200") -> int:
    """ANDs of one unit ``lab`` (class_size's label): on the H200 its int8 linears, bf16 GEMMs and FA3 blocks from the tables
    above, the rest (and every unit on sm_120) `class_size.measured`'s, refused where neither has it."""
    fam, s = CS.statics(lab)
    if target == "h200":
        if fam == "ScaledMmS8_v1":
            return s["N"] * S8_COORD[s["K"]]
        if fam == "Int8RowQuant_v1":
            return ROW_QUANT[s["K"]]
        if fam == "Gemm_v2":
            return s["N"] * BF16_COORD[s["K"]]
        if fam == FIRST and s["D"] == 64:
            return FA3_FIRST[s["NVIS"] - 1]
        if fam == "AttnBlock_v4" and s["D"] == 64 and s["NVIS"] == s["BN"]:
            return FA3_FULL[(s["BN"], s["CHECK"])]
    return CS.measured(lab)[0]


@dataclass(frozen=True)
class Unit:
    """A census template: ``n`` units, each with these input rows by what fills them, these output rows and their bytes, and
    ``x``, its output rows' digests (Σ blocks(128 + b))."""
    label: str
    n: int
    wired: int = 0
    registered: int = 0
    other: int = 0
    out: int = 0
    out_bytes: int = 0
    x: int = 0

    @property
    def ins(self) -> int:
        return self.wired + self.registered + self.other

    def compressions(self, c: Costs) -> int:
        return self.n * (self.x + c.fresh * self.out + c.reused * self.ins)


def canonical(lab: str) -> str:
    """class_rows' label as class_size writes it: JSON bools, without class_rows' marks on a repeated label."""
    return re.sub(r"=(True|False)\b", lambda m: "=" + m.group(1).lower(), lab.rstrip("'"))


def units_of(census: dict) -> list[Unit]:
    """A census's templates as `Unit`s (a wired+registered port counted wired)."""
    out = []
    for lab, t in census["templates"].items():
        ins = {k: v for k, v in t["rows"].items() if k != "out"}
        o = t["rows"].get("out", 0)
        fresh = t["compressions_a_unit"] - CR.REUSED * sum(ins.values())
        out.append(Unit(canonical(lab), t["units"], wired=sum(v for k, v in ins.items() if "wired" in k),
                        registered=sum(v for k, v in ins.items() if "wired" not in k and "registered" in k),
                        other=sum(v for k, v in ins.items() if "wired" not in k and "registered" not in k),
                        out=o, out_bytes=t["bytes"].get("out", 0), x=fresh - CR.FRESH_EXTRA * o))
    return out


def gather(w: int, D: int, n: int) -> Unit:
    return Unit(f"{GATHER}{{WORLD={w},N={D}}}", n, wired=w, out=1, out_bytes=2 * D * w, x=CR.blocks(128 + 2 * D * w))


@dataclass(frozen=True)
class Heads:
    """The class circuit's attention as `window_circuit` cuts it: S slots of P positions, H heads a position (over the layers),
    G key and value gathers a position, head dim D, key blocks of BN (`FA3_FULL`'s family ``block``), and the rows of a first
    block, a full block and a head's end (``first``, ``full``, ``end``: `Unit`s of n = 1)."""
    S: int
    P: int
    H: int
    G: int
    D: int
    BN: int
    block: str
    first: Unit
    full: Unit
    end: Unit

    def blocks_at(self, T: int, BN: int) -> tuple[int, int]:
        """(key blocks, the first visited block's keys) of a head over T keys."""
        nb = -(-T // BN)
        return nb, T - (nb - 1) * BN

    def first_label(self, nvis: int, BN: int) -> str:
        return f"{FIRST}{{D={self.D},NVIS={nvis},BN={BN}}}"

    def full_label(self, BN: int) -> str:
        """A full block's label, priced at CHECK=false's ANDs wherever the census has CHECK=true (63 ANDs more on the H200)."""
        return f"{self.block}{{D={self.D},NVIS={BN},FIRST=false,BN={BN},CHECK=false}}"

    def ands_at(self, T: int, BN: int, target: str) -> int:
        """The ANDs of a position's heads over T keys at key blocks of BN."""
        nb, nvis = self.blocks_at(T, BN)
        return self.H * (unit_ands(self.first_label(nvis, BN), target) + (nb - 1) * unit_ands(self.full_label(BN), target)
                         + unit_ands(self.end.label, target))

    def units(self, BN: int | None = None, group: int | None = None, packed: bool = False) -> list[Unit]:
        """The attention's units at key blocks of ``BN`` (the census's by default), its first blocks' keys in ``group``-key rows
        (`L3a`), its blocks' state ``packed`` in one row (`L3b`)."""
        BN = BN or self.BN
        first, gathers, full = Counter(), Counter(), 0
        for T in range(1, self.P + 1):
            nb, nvis = self.blocks_at(T, BN)
            first[nvis] += 1
            full += nb - 1
            if 1 < nvis < BN:
                gathers[nvis] += 1
        S, H, G, D = self.S, self.H, self.G, self.D
        fu, bu, eu = self.first, self.full, self.end
        if packed:
            fu = replace(fu, out=1, x=CR.blocks(128 + fu.out_bytes))
            bu = replace(bu, out=1, x=CR.blocks(128 + bu.out_bytes), wired=bu.wired - (bu.out - 1))
            eu = replace(eu, wired=1)
        out = [replace(eu, n=S * H * self.P)]
        for nvis, k in sorted(first.items()):
            reads = 0 if group is None or nvis in (1, BN) else 2 * (-(-nvis // group) - 1)
            out.append(replace(fu, label=self.first_label(nvis, BN), n=S * H * k, wired=fu.wired + reads))
        if full:
            out.append(replace(bu, label=self.full_label(BN), n=S * H * full))
        for w, k in sorted(gathers.items()):
            r = w if group is None else w % group
            if r > 1:
                out.append(gather(r, D, S * G * k))
        blocks_filled = self.P // BN
        if group is None:
            out.append(gather(BN, D, S * G * blocks_filled))
        else:
            out.append(gather(group, D, S * G * (self.P // group)))
            out.append(replace(gather(BN, D, S * G * blocks_filled), wired=BN // group))
        return [u for u in out if u.n]


def _one(xs: list, what: str):
    if len(set(xs)) != 1:
        raise ValueError(f"the census's attention differs in {what}: {sorted(set(xs))}")
    return xs[0]


def split(units: list[Unit], cls: dict, target: str = "h200") -> tuple[list[Unit], Heads]:
    """The census's units outside attention, and its attention (`Heads`, its full blocks ``target``'s family), refused unless
    `Heads.units` at the census's BN is exactly the census's attention: its first blocks, full blocks and head ends, and as
    many of its key gathers as it has."""
    S, P, block = cls["slots"], cls["positions"], BLOCKS[target]
    firsts = [u for u in units if u.label.startswith(FIRST + "{")]
    ss = [CS.statics(u.label)[1] for u in firsts]
    D, BN = _one([s["D"] for s in ss], "D"), _one([s["BN"] for s in ss], "BN")
    (end,) = [u for u in units if u.label.startswith(END + "{")]
    fulls = [u for u in units if u.label.split("{")[0] in BLOCKS.values()]
    if any(u.label.split("{")[0] != block for u in fulls):
        raise ValueError(f"the census's full key blocks are not {target}'s {block}: {sorted({u.label for u in fulls})}")
    H = end.n // (S * P)
    if P >= BN:
        G = sum(u.n for u in units if u.label == f"{GATHER}{{WORLD={BN},N={D}}}") // (S * (P // BN))
    else:
        G = sum(u.n for u in units if u.label == f"{GATHER}{{WORLD=2,N={D}}}") // S
    f0 = replace(firsts[0], label="", n=1)
    full = replace(fulls[0], label="", n=1) if fulls else replace(f0, wired=f0.wired + f0.out)
    heads = Heads(S, P, H, G, D, BN, block, f0, full, replace(end, n=1))
    have = Counter()
    for u in units:
        have[u.label] += u.n
    want = Counter()
    for u in heads.units():
        want[u.label] += u.n
    att = {u.label for u in firsts + fulls + [end]}
    got = Counter({lab: n for lab, n in have.items() if lab in att})
    want_att = Counter({lab: n for lab, n in want.items() if not lab.startswith(GATHER + "{")})
    if Counter({lab.replace(",CHECK=true", ",CHECK=false"): n for lab, n in got.items()}) != want_att:
        raise ValueError(f"the census's attention units are not the cut at D {D}, BN {BN}, {H} heads and {S} × {P} positions")
    short = {lab: n for lab, n in want.items() if lab.startswith(GATHER + "{") and have[lab] < n}
    if short:
        raise ValueError(f"the census has fewer key gathers than {G} a position at BN {BN}: {short}")
    rest = []
    for u in units:
        if u.label in att:
            continue
        take = min(want.get(u.label, 0), u.n) if u.label.startswith(GATHER + "{") else 0
        want[u.label] -= take
        if u.n - take:
            rest.append(replace(u, n=u.n - take))
    return rest, heads


LEVERS = {
    "L1": "one FA3 tile (blocks of 128 at every launch)",
    "L2": "empty slots and past-end positions as cached constants",
    "L3a": "first blocks' keys in 16-key group rows",
    "L3b": "a key block's state in one row",
    "W": "wired ports by reference",
    "R": "registered ports by reference",
    "L4": "one Merkle root over members",
    "P": "outputs in shared 1 KiB rows (CIR-113)",
}


def window_units(rest: list[Unit], heads: Heads, levers: frozenset[str]) -> list[Unit]:
    """The window's units under ``levers`` (a shared row's share of a unit is fractional under ``P``)."""
    us = rest + heads.units(BN=EXACT_KEYS if "L1" in levers else None, group=GROUP if "L3a" in levers else None,
                            packed="L3b" in levers)
    if "W" in levers:
        us = [replace(u, wired=0) for u in us]
    if "R" in levers:
        us = [replace(u, registered=0) for u in us]
    if "P" in levers:
        us = [replace(u, out=u.out_bytes / ROW_BYTES, x=u.out_bytes / ROW_BYTES * CR.blocks(128 + ROW_BYTES)) for u in us]
    return us


#: b || c, a row's public commit string (hm96 at SHA-512)
BC_BYTES = 128
#: a member's entry in the leaf layer's canonical JSON: its name and 128 hex
LEAF_JSON_BYTES = 150


@dataclass(frozen=True)
class Registration:
    """The window's registration as `verity.ml.windows.rows.write` makes it, members of `CAP` units: ``members``, port
    ``roots`` (a unit's rows each a port, the cut rows ``p#j`` included), the public bodies' bytes (b || c a row a unit), the
    leaf layer's compressions (each public body's SHA-512, then one over their canonical JSON), the JSON's bytes and the files
    (a member's public, private and units files); under a member root (``L4``), each member's leaf over its port roots and a
    tree over the members, two compressions a node, and nothing written but the root."""
    members: int
    roots: int
    public_bytes: int
    compressions: int
    json_bytes: int
    files: int

    @classmethod
    def of(cls, units: list[Unit], member_root: bool) -> Registration:
        members = sum(-(-u.n // CAP) for u in units)
        roots = round(sum(-(-u.n // CAP) * (u.ins + u.out) for u in units))
        if member_root:
            leaves = round(sum(-(-u.n // CAP) * CR.blocks(128 + 64 * (u.ins + u.out)) for u in units))
            return cls(members, roots, 0, leaves + 2 * members, 0, 0)
        rows = round(sum(u.n * (u.ins + u.out) for u in units))
        hashed = rows + members + CR.blocks(members * LEAF_JSON_BYTES)
        return cls(members, roots, rows * BC_BYTES, hashed, roots * ROOT_JSON_BYTES, 3 * members)


@dataclass(frozen=True)
class Traffic:
    """A window's requests: (prompt ids, outputs) of each filled slot."""
    name: str
    requests: tuple[tuple[int, int], ...]


TRAFFIC = (Traffic("full: 8 × (1,024 ids, 64 outputs)", ((1024, 64),) * 8),
           Traffic("mid: 8 × (256 ids, 64 outputs)", ((256, 64),) * 8),
           Traffic("dry run: 4 × (21 ids, 4 outputs)", ((21, 4),) * 4))


def served_ands(rest: list[Unit], heads: Heads, levers: frozenset[str], target: str) -> int:
    """The class circuit's ANDs under ``levers``: its attention at blocks of 128 under ``L1``, and under ``L2`` the length mask,
    one AND an output bit."""
    us = rest + heads.units(BN=EXACT_KEYS if "L1" in levers else None)
    ands = sum(u.n * unit_ands(u.label, target) for u in us)
    if "L2" in levers:
        ands += sum(u.n * 8 * u.out_bytes for u in us)
    return ands


def pad_pass(heads: Heads, traffic: Traffic, levers: frozenset[str], base_ands: float, step_s: float, target: str) -> dict:
    """The positions whose values the engine never computed, recomputed as batched decode steps over every slot at once, a
    slot's run from its first such position (V3c's model: its continuation): an empty slot's every position and a request's
    positions past its end, unless ``L2``; without ``L1``, from `EXACT_KEYS` on in a slot whose prompt passes it (its later
    positions read its inexact keys). Its positions, steps, seconds and ANDs (``base_ands`` a position and its heads')."""
    BN = EXACT_KEYS if "L1" in levers else heads.BN
    spans = []
    for s in range(heads.S):
        if s >= len(traffic.requests):
            if "L2" not in levers:
                spans.append((0, heads.P))
            continue
        lp, o = traffic.requests[s]
        ran = min(lp + o - 1, heads.P)
        start, stop = ran, ran if "L2" in levers else heads.P
        if "L1" not in levers and lp > EXACT_KEYS:
            start = EXACT_KEYS
        if start < stop:
            spans.append((start, stop))
    steps = max((b - a for a, b in spans), default=0)
    ands = sum(base_ands + heads.ands_at(p + 1, BN, target) for a, b in spans for p in range(a, b))
    return {"positions": sum(b - a for a, b in spans), "steps": steps, "seconds": steps * step_s, "ands": ands}


def project(census: dict, slots: int, prompt: int, outputs: int, target: str = "h200") -> tuple[list[Unit], Heads, dict]:
    """A census of one Program at a probe class within one key block (`class_rows.py --probe`), projected to ``slots``
    Programs of ``prompt`` ids and ``outputs`` outputs as `class_size.project` projects one: outside attention a fixed count
    of each template a position (`TokenFeed_v1` a prompt position and `SlotRow_v1` one a Program, each with its prompt's row
    of 4 bytes an id), refused unless each divides evenly; the heads rebuilt at the class's positions (`Heads.units`)."""
    c0 = census["class"]
    if c0["slots"] != 1:
        raise ValueError(f"a probe of {c0['slots']} Programs: project one")
    rest, heads = split(units_of(census), c0, target)
    P0, NP0, P = c0["positions"], c0["prompt"], prompt + outputs - 1
    out = []
    for u in rest:
        fam, s = CS.statics(u.label)
        lab = u.label.replace(f"NP={NP0}", f"NP={prompt}")
        if fam in (CS.FEED, CS.ROW):
            u = replace(u, out_bytes=u.out_bytes + 4 * (prompt - NP0),
                        x=u.x - CR.blocks(128 + 4 * NP0) + CR.blocks(128 + 4 * prompt))
        if fam == CS.FEED:
            n = 1 if s["FIRST"] else prompt - 1
        elif fam == CS.ROW:
            n = 1
        elif u.n % P0:
            raise ValueError(f"the probe has {u.n} {u.label} units, not a fixed count at each of its {P0} positions")
        else:
            n = u.n // P0 * P
        out.append(replace(u, label=lab, n=slots * n))
    cls = {"slots": slots, "prompt": prompt, "outputs": outputs, "positions": P}
    return out, replace(heads, S=slots, P=P), cls


def price(census: dict, target: str = "h200", served_s: float = 1.7, step_s: float = 0.023,
          to: tuple[int, int, int] | None = None) -> dict:
    """Every figure the report prints, as data, for ``census`` or its projection ``to`` a class (slots, prompt, outputs). The
    commit's seconds price its compressions, the registration's leaf layer's among them, at the engine's rates; today the leaf
    layer is `hashlib` on the committer's CPU, and its files are written there (``registration``'s bytes)."""
    if to is None:
        cls = census["class"]
        rest, heads = split(units_of(census), cls, target)
        total = census["total"]["compressions"]
    else:
        rest, heads, cls = project(census, *to, target)
        total = None
    out = {"class": cls, "target": target, "served_s": served_s, "step_s": step_s, "sha512_ands": SHA512_ANDS,
           "census_compressions": total, "projected_from": None if to is None else census["class"]}
    served0 = served_ands(rest, heads, frozenset(), target)
    att = sum(heads.ands_at(T, heads.BN, target) for T in range(1, heads.P + 1))
    base_ands = (served0 - heads.S * att) / (heads.S * heads.P)
    rows = []
    sets = [("baseline (census model)", frozenset(), CENSUS), ("baseline (engine)", frozenset(), ENGINE)]
    sets += [(f"{k}: {LEVERS[k]}", frozenset({k}), ENGINE) for k in LEVERS]
    sets += [("L1 + L3a + L3b", frozenset({"L1", "L3a", "L3b"}), ENGINE),
             ("every lever but P", frozenset(LEVERS) - {"P"}, ENGINE),
             ("every lever", frozenset(LEVERS), ENGINE)]
    for name, levers, costs in sets:
        us = window_units(rest, heads, levers)
        rows_c = round(sum(u.compressions(costs) for u in us))
        reg = Registration.of(us, "L4" in levers)
        served = served_ands(rest, heads, levers, target)
        pads = {t.name: pad_pass(heads, t, levers, base_ands, step_s, target) for t in TRAFFIC}
        full = pads[TRAFFIC[0].name]
        commit_c = rows_c + reg.compressions
        rows.append({"levers": name, "set": sorted(levers), "costs": costs.name, "row_compressions": rows_c,
                     "registration": reg.__dict__, "commit_compressions": commit_c, "commit_ands": commit_c * SHA512_ANDS,
                     "served_ands": served, "ratio_ands": commit_c * SHA512_ANDS / served,
                     "commit_s": {r: commit_c / v for r, v in RATES.items()},
                     "ratio_s": {r: commit_c / v / served_s for r, v in RATES.items()},
                     "pad_pass": pads,
                     "full_ratio_ands": (commit_c * SHA512_ANDS + full["ands"]) / served,
                     "full_ratio_s": (commit_c / RATES["3.1 G/s"] + full["seconds"]) / served_s})
    out["rows"] = rows
    windows = {p: GPUS * s / served_s for p, s in PERIODS.items()}
    out["periods"] = {p: {"windows": w, "served_ands": w * served0,
                          "commit_ands": {r["levers"]: w * r["commit_ands"] for r in rows}} for p, w in windows.items()}
    return out


def _e(v: float) -> str:
    return f"{v:.3g}"


def report(doc: dict) -> str:
    c, served_s, p0 = doc["class"], doc["served_s"], doc["projected_from"]
    source = (f"class_rows' total {_e(doc['census_compressions'])} compressions" if p0 is None else
              f"projected from a probe of {p0['slots']} × {p0['positions']} ({p0['prompt']} ids, {p0['outputs']} outputs)")
    lines = [f"class {c['slots']} × {c['positions']} positions ({c['prompt']} ids, {c['outputs']} outputs), target "
             f"{doc['target']}; served window {served_s} s; a decode step {doc['step_s'] * 1e3:.0f} ms; "
             f"{doc['sha512_ands']:,} ANDs a compression; target 0.20–0.25×; {source}",
             "", "| levers | rows compr. | leaf layer compr. | commit at 3.1 (4.79) G/s | commit ANDs | served ANDs | "
             "ratio, ANDs | ratio, time | with the full window's pad pass: ANDs, time |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in doc["rows"]:
        lines.append(f"| {r['levers']} | {_e(r['row_compressions'])} | {_e(r['registration']['compressions'])} | "
                     f"{r['commit_s']['3.1 G/s']:.2f} s ({r['commit_s']['4.79 G/s']:.2f} s) | {_e(r['commit_ands'])} | "
                     f"{_e(r['served_ands'])} | {r['ratio_ands']:.3f}× | {r['ratio_s']['3.1 G/s']:.2f}× | "
                     f"{r['full_ratio_ands']:.2f}×, {r['full_ratio_s']:.1f}× |")
    lines += ["", "| levers | " + " | ".join(t.name for t in TRAFFIC) + " |", "| --- |" + " --- |" * len(TRAFFIC)]
    for r in doc["rows"]:
        if r["costs"] == "engine":
            lines.append(f"| {r['levers']} | " + " | ".join(
                f"{p['steps']} steps, {p['seconds']:.1f} s, {_e(p['ands'])} ANDs" for p in r["pad_pass"].values()) + " |")
    lines += ["", "| levers | members | port roots | public bytes | leaf-layer compr. | registration JSON | files |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in doc["rows"]:
        g = r["registration"]
        lines.append(f"| {r['levers']} | {g['members']:,} | {g['roots']:,} | {_e(g['public_bytes'])} | {_e(g['compressions'])} | "
                     f"{_e(g['json_bytes'])} | {g['files']:,} |")
    lines += ["", f"over the reference datacenter ({GPUS:,} GPUs) serving class-of-record windows back to back:",
              "| period | windows | served ANDs | commit ANDs, baseline (engine) | commit ANDs, every lever |",
              "| --- | --- | --- | --- | --- |"]
    for p, v in doc["periods"].items():
        ca = v["commit_ands"]
        lines.append(f"| {p} | {_e(v['windows'])} | {_e(v['served_ands'])} | {_e(ca['baseline (engine)'])} | "
                     f"{_e(ca['every lever'])} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("census", help="class_rows.py --out's JSON")
    ap.add_argument("--target", choices=("h200", "sm120"), default="h200")
    ap.add_argument("--served-s", type=float, default=1.7, help="the served window's seconds (V3c's estimate: 4 prefill and 63 "
                    "decode steps)")
    ap.add_argument("--step-ms", type=float, default=23.0, help="a batch decode step's ms (the dry run's, r20261026-195741-11d7)")
    ap.add_argument("--project", default=None, metavar="SLOTS,PROMPT,OUTPUTS",
                    help="price a one-Program probe's projection to this class (`project`)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    to = tuple(int(v) for v in a.project.split(",")) if a.project else None
    doc = price(json.loads(Path(a.census).read_text()), a.target, a.served_s, a.step_ms / 1e3, to)
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(report(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
