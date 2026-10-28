"""The class-of-record window's commit against its target, 0.20 to 0.25 of the served window: a `class_rows.py` census priced in
SHA-512 compressions, seconds on the H200 and AND gates, beside the served window's ANDs and seconds, each lever alone and
all of them stacked; the pad pass (positions whose values the engine never computed, recomputed) at three traffics; the
registration's size; and the totals over monitoring periods of one week, one month and one year.

Costs. A fresh output row is its digest x (`class_rows.blocks(128 + b)`: P(8 nbytes) and its b bytes) and `Costs.fresh` more
(hm96's c and h, two each, the frame leaf, and a port-tree node); an input row a wire or a registered read fills is re-leafed in
its reader's port tree for `Costs.reused` (h, the leaf, the node). `CENSUS` is class_rows' model, a node one compression;
`ENGINE` is the commit engine's, a node two (`commit_slots`: nine compressions a one-block row-slot). Compressions are priced
in ANDs at `SHA512_ANDS` each; the served window is the class circuit's ANDs (`unit_ands`).

Levers (`LEVERS`), each a change to the circuit's layout or the commitment's format, not to what the model computes:

- ``L1``, one FA3 tile. `fa3_tile` launches 64 × 192 when max_seqlen_q · (query heads a KV head) <= 64 and 192 × 128
  otherwise, so at Llama-3.2-1B a pure decode launch has key blocks of 192 and a prefill or mixed launch blocks of 128, and the
  class circuit, cut at the decode's 192, is inexact past 128 keys for every prompt position and every decode row of a mixed
  launch. One tile at every launch (the pinned FA3's ``use_one_mma_wg`` off) makes every position the
  circuit's at blocks of 128 (`Heads.units` at ``BN=128``), so the engine's own values are the circuit's: no pad pass for a
  position the engine ran.
- ``L2``, constants. An empty slot computes token 0 and then decodes greedily, a function of the weights and the class, and
  under a length-masked class Program (V3b's) a position past its request's end outputs zero rows: their rows are computed
  once and kept, and no pad pass runs them. Their hiding leaves stay fresh every window, since which slots are filled is
  private (their digests x could be kept too, which this doesn't price).
- ``L3b``, one state row. A key block's running state (max, lane sums, O) is three rows, each hashed and re-leafed by its
  reader; as one row it is one.
- ``W`` and ``R``, ports by reference: a reader's wired or registered port is the writer's or the pinned weight's leaves
  (a drawn member's port tree is derived at the audit), so no input row is re-leafed.
- ``P``, shared output rows (`commit_slots --row-bytes`): units' outputs packed into rows of `ROW_BYTES`, so a
  fresh row's fixed cost is paid a KiB, not a unit's few bytes (its readers re-leaf as many rows as before, and a unit reads
  its segment for a few thousand ANDs, which this doesn't price).
- ``L4``, one member root: the registration is O(members) (`RegistrationSize`), every member's public body (every row's
  b || c, 128 bytes) SHA-512'd on the CPU for `rows.leaf_layer` and written out, and every port root a JSON entry; under one
  Merkle root over members, each member's leaf over its port roots, it is a root, and an audit opens K members in O(K log M).

Serving commits its own stream as well, while it serves and before the window's commit (`stream`, rows-sha512), and seals
it. ``F`` leaves out of it the attention interiors (the FA2 and FA3 M1 taps, `fa2.m1/` and `fa3.m1/`: each query row's
scores and probabilities at every key and each key block's running state), which nothing reads after the seal: the window
circuit's units read wired, registered (weights) and ingress rows only, its key blocks' running states are its own units'
outputs (`window_rows` evaluates them), and a served window samples no openings.

    python window_cost.py CENSUS.json [--target h200|sm120] [--served-s 1.7] [--step-ms 23]
        [--project S,PROMPT,OUTPUTS[,LANES,EVERY]] [--measure-cpu SCRATCH | --cpu-rates FILE] [--stream SIZING] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict.window_circuit import WindowClass

from verity.core.protocols.pous import wipe_monitoring
from verity.core.protocols.sampled_proofs.optimize import SHA512_ANDS
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import io_shell as IO
from verity.ml.windows import interface as IF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import class_rows as CR  # noqa: E402
import class_size as CS  # noqa: E402

FIRST, END = "AttnBlockFirst_v1", "AttnHeadEnd_v1"
#: a full key block by target: FA3's (sm_90) and FA2's (sm_120)
BLOCKS = {"h200": "AttnBlock_v4", "sm120": "AttnBlock_v5"}
#: prefill is exact against the decode-shaped Definitions at this many keys or fewer
EXACT_KEYS = 128
#: a shared output row's bytes (`P`): 1 KiB rows
ROW_BYTES = 1024
#: a member's units: V3c's cap
CAP = 64
#: one port root's entry in `registration.json` (``indent=1``): a port name of about 40 characters, 128 hex of root and of
#: domain, leaves
ROOT_JSON_BYTES = 360
#: the H200's rates, compressions a second: a measured step and commit_gpu's peak (`commit_slots`)
RATES = {"3.1 G/s": 3.1e9, "4.79 G/s": 4.79e9}
#: the reference datacenter (`benchmarks/census/deployments.json`, dc-12500-machines-600s)
GPUS = 12_500 * 8
#: core's monitoring periods (`wipe_monitoring`), seconds by name
PERIODS = dict(wipe_monitoring.PERIODS)


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
    1_575_433, 1_631_352, 1_688_871, 1_744_726, 1_800_645, 1_858_228, 1_914_083, 1_970_002, 2_111_105, 2_171_696, 2_227_551,
    2_283_470, 2_342_653, 2_398_508, 2_454_427, 2_513_610, 2_569_465, 2_625_384, 2_682_903, 2_738_758, 2_794_677, 2_852_260,
    2_908_115, 2_964_034, 3_105_137, 3_165_728, 3_221_583, 3_277_502, 3_336_685, 3_392_540, 3_448_459, 3_507_642, 3_563_497,
    3_619_416, 3_676_935, 3_732_790, 3_788_709, 3_846_292, 3_902_147, 3_958_066, 4_099_169, 4_159_760, 4_215_615, 4_271_534,
    4_330_717, 4_386_572, 4_442_491, 4_501_674, 4_557_529, 4_613_448, 4_670_967, 4_726_822, 4_782_741, 4_840_324, 4_896_179,
    4_952_098, 5_093_201, 5_153_792, 5_209_647, 5_265_566, 5_324_749, 5_380_604, 5_436_523, 5_495_706, 5_551_561, 5_607_480,
    5_664_999, 5_720_854, 5_776_773, 5_834_356, 5_890_211, 5_946_130, 6_087_233, 6_147_824, 6_203_679, 6_259_598, 6_318_781,
    6_374_636, 6_430_555, 6_489_738, 6_545_593, 6_601_512, 6_659_031, 6_714_886, 6_770_805, 6_828_388, 6_884_243, 6_940_162,
    7_081_265, 7_141_856, 7_197_711, 7_253_630, 7_312_813, 7_368_668, 7_424_587, 7_483_770, 7_539_625, 7_595_544, 7_653_063,
    7_708_918, 7_764_837, 7_822_420, 7_878_275, 7_934_194, 8_075_297, 8_135_888, 8_191_743, 8_247_662, 8_306_845, 8_362_700,
    8_418_619, 8_477_802, 8_533_657, 8_589_576, 8_647_095, 8_702_950, 8_758_869, 8_816_452, 8_872_307, 8_928_226, 9_069_329,
    9_129_920, 9_185_775, 9_241_694, 9_300_877, 9_356_732, 9_412_651, 9_471_834, 9_527_689, 9_583_608, 9_641_127, 9_696_982,
    9_752_901, 9_810_484, 9_866_339, 9_922_258, 10_063_361, 10_123_952, 10_179_807, 10_235_726, 10_294_909, 10_350_764,
    10_406_683, 10_465_866, 10_521_721, 10_577_640, 10_635_159, 10_691_014, 10_746_933, 10_804_516, 10_860_371, 10_916_290,
    11_057_393, 11_117_984, 11_173_839, 11_229_758, 11_288_941, 11_344_796, 11_400_715, 11_459_898, 11_515_753, 11_571_672,
    11_629_191, 11_685_046, 11_740_965, 11_798_548, 11_854_403, 11_910_322)


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
    """A census template: ``n`` units, each with these input rows and their bytes by what fills them, these output rows and
    their bytes, and ``x``, its output rows' digests (Σ blocks(128 + b))."""
    label: str
    n: int
    wired: int = 0
    registered: int = 0
    other: int = 0
    out: int = 0
    out_bytes: int = 0
    x: int = 0
    wired_bytes: int = 0
    registered_bytes: int = 0
    other_bytes: int = 0

    @property
    def ins(self) -> int:
        return self.wired + self.registered + self.other

    @property
    def row_bytes(self) -> int:
        """A unit's rows' words, in and out: what its member's private body holds of it beside its salts."""
        return self.wired_bytes + self.registered_bytes + self.other_bytes + self.out_bytes

    def compressions(self, c: Costs) -> int:
        return self.n * (self.x + c.fresh * self.out + c.reused * self.ins)


def canonical(lab: str) -> str:
    """class_rows' label as class_size writes it: JSON bools, without class_rows' marks on a repeated label."""
    return re.sub(r"=(True|False)\b", lambda m: "=" + m.group(1).lower(), lab.rstrip("'"))


def units_of(census: dict) -> list[Unit]:
    """A census's templates as `Unit`s (a wired+registered port counted wired)."""
    def by_kind(d: dict) -> dict:
        return {"wired": sum(v for k, v in d.items() if "wired" in k),
                "registered": sum(v for k, v in d.items() if "wired" not in k and "registered" in k),
                "other": sum(v for k, v in d.items() if k != "out" and "wired" not in k and "registered" not in k)}

    out = []
    for lab, t in census["templates"].items():
        rows, b = by_kind(t["rows"]), by_kind(t["bytes"])
        o = t["rows"].get("out", 0)
        fresh = t["compressions_a_unit"] - CR.REUSED * sum(rows.values())
        out.append(Unit(canonical(lab), t["units"], **rows, out=o, out_bytes=t["bytes"].get("out", 0),
                        x=fresh - CR.FRESH_EXTRA * o, **{f"{k}_bytes": v for k, v in b.items()}))
    return out


def kv_bytes(D: int) -> int:
    """A key's bytes in a key block's reads: its key's row and its value's, D bf16 each."""
    return 4 * D


@dataclass(frozen=True)
class Heads:
    """The class circuit's attention as `window_circuit` cuts it: S slots of P positions, H heads a position (over the layers),
    R wired rows a key (its key's and its value's rows as their writers hold them), head dim D, key blocks of BN (`BLOCKS`'
    family ``block``), and the rows of a first block, a full block and a head's end (``first``, ``full``, ``end``: `Unit`s of
    n = 1, a block's ``wired`` and ``wired_bytes`` those beside its keys' and values', R rows and `kv_bytes` a key)."""
    S: int
    P: int
    H: int
    R: int
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

    def units(self, BN: int | None = None, packed: bool = False) -> list[Unit]:
        """The attention's units at key blocks of ``BN`` (the census's by default), each block reading its keys' and values'
        rows, its blocks' state ``packed`` in one row (`L3b`)."""
        BN = BN or self.BN
        first, full = Counter(), 0
        for T in range(1, self.P + 1):
            nb, nvis = self.blocks_at(T, BN)
            first[nvis] += 1
            full += nb - 1
        S, H, R, kv = self.S, self.H, self.R, kv_bytes(self.D)
        fu, bu, eu = self.first, self.full, self.end
        if packed:
            fu = replace(fu, out=1, x=CR.blocks(128 + fu.out_bytes))
            bu = replace(bu, out=1, x=CR.blocks(128 + bu.out_bytes), wired=bu.wired - (bu.out - 1))
            eu = replace(eu, wired=1)
        out = [replace(eu, n=S * H * self.P)]
        for nvis, k in sorted(first.items()):
            out.append(replace(fu, label=self.first_label(nvis, BN), n=S * H * k, wired=fu.wired + R * nvis,
                               wired_bytes=fu.wired_bytes + kv * nvis))
        if full:
            out.append(replace(bu, label=self.full_label(BN), n=S * H * full, wired=bu.wired + R * BN,
                               wired_bytes=bu.wired_bytes + kv * BN))
        return [u for u in out if u.n]


def _one(xs: list, what: str):
    if len(set(xs)) != 1:
        raise ValueError(f"the census's attention differs in {what}: {sorted(set(xs))}")
    return xs[0]


def split(units: list[Unit], cls: dict, target: str = "h200") -> tuple[list[Unit], Heads]:
    """The census's units outside attention, and its attention (`Heads`, its full blocks ``target``'s family), refused unless
    `Heads.units` at the census's BN is exactly the census's attention, each unit's rows and bytes with its count: its first
    blocks, full blocks and head ends, its rows a key read off its first blocks of the two fewest keys."""
    S, P, block = cls["slots"], cls["positions"], BLOCKS[target]
    firsts = [u for u in units if u.label.startswith(FIRST + "{")]
    ss = [CS.statics(u.label)[1] for u in firsts]
    D, BN = _one([s["D"] for s in ss], "D"), _one([s["BN"] for s in ss], "BN")
    (end,) = [u for u in units if u.label.startswith(END + "{")]
    fulls = [u for u in units if u.label.split("{")[0] in BLOCKS.values()]
    if any(u.label.split("{")[0] != block for u in fulls):
        raise ValueError(f"the census's full key blocks are not {target}'s {block}: {sorted({u.label for u in fulls})}")
    H = end.n // (S * P)
    by_keys = {s["NVIS"]: u for u, s in zip(firsts, ss)}
    if len(by_keys) < 2:
        raise ValueError(f"the census's first blocks are at {sorted(by_keys)} keys: its rows a key need two key counts")
    a, b = sorted(by_keys)[:2]
    R = (by_keys[b].wired - by_keys[a].wired) // (b - a)
    kv = kv_bytes(D)
    f0 = replace(by_keys[a], label="", n=1, wired=by_keys[a].wired - R * a, wired_bytes=by_keys[a].wired_bytes - kv * a)
    full = (replace(fulls[0], label="", n=1, wired=fulls[0].wired - R * BN, wired_bytes=fulls[0].wired_bytes - kv * BN)
            if fulls else replace(f0, wired=f0.wired + f0.out, wired_bytes=f0.wired_bytes + f0.out_bytes))
    heads = Heads(S, P, H, R, D, BN, block, f0, full, replace(end, n=1))

    def shape(u: Unit) -> tuple:
        return (u.wired, u.registered, u.other, u.out, u.out_bytes, u.x, u.wired_bytes, u.registered_bytes, u.other_bytes)

    have, shapes = Counter(), {}
    for u in units:
        lab = u.label.replace(",CHECK=true", ",CHECK=false")
        have[lab] += u.n
        shapes.setdefault(lab, set()).add(shape(u))
    att = {u.label.replace(",CHECK=true", ",CHECK=false") for u in firsts + fulls + [end]}
    if ({lab: (have[lab], shapes[lab]) for lab in att} !=
            {u.label: (u.n, {shape(u)}) for u in heads.units()}):
        raise ValueError(f"the census's attention units are not the cut at D {D}, BN {BN}, {H} heads, {R} rows a key and "
                         f"{S} × {P} positions")
    att = {u.label for u in firsts + fulls + [end]}
    return [u for u in units if u.label not in att], heads


LEVERS = {
    "L1": "one FA3 tile (blocks of 128 at every launch)",
    "L2": "empty slots and past-end positions as cached constants",
    "L3b": "a key block's state in one row",
    "W": "wired ports by reference",
    "R": "registered ports by reference",
    "L4": "one Merkle root over members",
    "P": CS.SOURCES["levers"]["P"],
}


def window_units(rest: list[Unit], heads: Heads, levers: frozenset[str]) -> list[Unit]:
    """The window's units under ``levers`` (a shared row's share of a unit is fractional under ``P``)."""
    us = rest + heads.units(BN=EXACT_KEYS if "L1" in levers else None, packed="L3b" in levers)
    if "W" in levers:
        us = [replace(u, wired=0, wired_bytes=0) for u in us]
    if "R" in levers:
        us = [replace(u, registered=0, registered_bytes=0) for u in us]
    if "P" in levers:
        us = [replace(u, out=u.out_bytes / ROW_BYTES, x=u.out_bytes / ROW_BYTES * CR.blocks(128 + ROW_BYTES)) for u in us]
    return us


#: b || c, a row's public commit string (hm96 at SHA-512), and a row's salt (`verity.ml.windows.rows`)
BC_BYTES, SALT_BYTES = 128, 192
#: a member's entry in the leaf layer's canonical JSON: its name and 128 hex
LEAF_JSON_BYTES = 150


@dataclass(frozen=True)
class RegistrationSize:
    """The window's registration as `verity.ml.windows.rows.write` makes it, members of `CAP` units: ``members``, port
    ``roots`` (a unit's rows each a port, the cut rows ``p#j`` included), the public bodies' bytes (b || c a row a unit) and
    the private bodies' (every row's words and salt), the compressions (each public body's SHA-512 twice, for the leaf layer
    and for its member's entry, and one over the leaf layer's canonical JSON), the JSON's bytes and the files (a member's
    public, private and units files). Under a member root (``L4``), each member's leaf over its port roots and a tree over the
    members, two compressions a node, and nothing written but the root; the bodies are what the prover keeps until the sample
    either way."""
    members: int
    roots: int
    public_bytes: int
    private_bytes: int
    compressions: int
    json_bytes: int
    files: int
    member_root: bool

    @classmethod
    def of(cls, units: list[Unit], member_root: bool) -> RegistrationSize:
        members = sum(-(-u.n // CAP) for u in units)
        roots = round(sum(-(-u.n // CAP) * (u.ins + u.out) for u in units))
        rows = round(sum(u.n * (u.ins + u.out) for u in units))
        public, private = rows * BC_BYTES, round(sum(u.n * u.row_bytes for u in units)) + rows * SALT_BYTES
        if member_root:
            leaves = round(sum(-(-u.n // CAP) * CR.blocks(128 + 64 * (u.ins + u.out)) for u in units))
            return cls(members, roots, public, private, leaves + 2 * members, 0, 0, True)
        hashed = 2 * (rows + members) + CR.blocks(members * LEAF_JSON_BYTES)
        return cls(members, roots, public, private, hashed, roots * ROOT_JSON_BYTES, 3 * members, False)

    def cpu_s(self, rates: dict) -> dict:
        """Seconds of the registration's work on the committer's CPU at ``rates`` (`measure_cpu`'s), one thread as
        `rows.write` runs: the public bodies' SHA-512s, the JSON, and the files' bytes and creation; none under ``L4``."""
        if self.member_root:
            return {"sha512": 0.0, "json": 0.0, "write": 0.0, "files": 0.0, "total": 0.0}
        out = {"sha512": 2 * self.public_bytes / rates["sha512_bytes_s"], "json": self.json_bytes / rates["json_bytes_s"],
               "write": (self.public_bytes + self.private_bytes + self.json_bytes) / rates["write_bytes_s"],
               "files": self.files / rates["files_s"]}
        return out | {"total": sum(out.values())}


def measure_cpu(scratch: Path, mib: int = 256) -> dict:
    """This host's rates as `RegistrationSize.cpu_s` prices them, one thread, each over ``mib`` MiB: SHA-512 (`hashlib`)
    of member-sized public bodies (`CAP` units of 6 rows), `json.dumps` of port-root entries as `rows.registration` writes
    them (``indent=1, sort_keys``), 64 MiB files written and fsynced into ``scratch``, and 1 KiB files created there (each
    removed after)."""
    import hashlib
    import os
    import platform
    import time

    def rate(work, amount):
        t0 = time.perf_counter()
        work()
        return amount / (time.perf_counter() - t0)

    body = os.urandom(CAP * 6 * BC_BYTES)
    reps = mib * 2**20 // len(body)
    sha = rate(lambda: [hashlib.sha512(body).digest() for _ in range(reps)], reps * len(body))
    n = mib * 2**20 // ROOT_JSON_BYTES
    roots = [{"port": f"m{k // 6}/p{k % 6}", "root": os.urandom(64).hex(), "leaves": CAP, "domain": os.urandom(64).hex()}
             for k in range(n)]
    size = len(json.dumps({"roots": roots}, indent=1, sort_keys=True))
    js = rate(lambda: json.dumps({"roots": roots}, indent=1, sort_keys=True), size)
    scratch.mkdir(parents=True, exist_ok=True)
    chunk, big = os.urandom(64 * 2**20), max(1, mib // 64)

    def write_big():
        for i in range(big):
            with open(scratch / f"big{i}.bin", "wb") as f:
                f.write(chunk)
                f.flush()
                os.fsync(f.fileno())

    wr = rate(write_big, big * len(chunk))
    small, k = os.urandom(1024), mib * 16
    fs = rate(lambda: [(scratch / f"small{i}.json").write_bytes(small) for i in range(k)], k)
    for p in [*(scratch / f"big{i}.bin" for i in range(big)), *(scratch / f"small{i}.json" for i in range(k))]:
        p.unlink()
    return {"host": platform.node(), "mib": mib, "sha512_bytes_s": sha, "json_bytes_s": js, "write_bytes_s": wr,
            "files_s": fs}


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


#: serving's commit of its own stream (rows-sha512, `step_rows`): a 4 KiB row's digest, its salt derived from the step's key
#: (three compressions) and the engine's fresh row
STREAM_ROW_BYTES = 4096
STREAM_ROW = CR.blocks(128 + STREAM_ROW_BYTES) + 3 + ENGINE.fresh
#: the seal's store, bytes a second: a measured seal stored 29,327,490,072 committed bytes in 90.581 s
SEAL_BYTES_S = 29_327_490_072 / 90.581
STREAM_LEVER = "F: the attention interiors (fa*.m1) not committed"


def stream(sizing: dict, cls: dict, target: str, served_ands: float, served_s: float, lanes: int = 4,
           every: int = 16) -> dict:
    """Serving's own commit of the class's full window, made while it serves, before the window's and beside it
    (`step_rows`: every 4 KiB of a step's stream a hiding row, `STREAM_ROW` compressions), and the seal's store of it
    (`SEAL_BYTES_S`), whole and under ``F``. The stream is `native_host.schedule_stream_bound`'s, serving's stream of record
    under a declared schedule (every request to the class's outputs, a mixed step's FA3 launch padded to its longest query),
    at ``sizing``'s widths (an arena sizing record), under two schedules: every prompt in one prefill step, and the class's
    anchors (``lanes`` requests every ``every`` steps, `window_circuit.CLASS_OF_RECORD`'s, a bucket a step)."""
    from verity_vllm.commit.committer.native_host import schedule_stream_bound

    asz = dict(sizing, model=dict(sizing["model"], attn_backend="fa3" if target == "h200" else "fa2"))
    S, prompt, outputs = cls["slots"], cls["prompt"], cls["outputs"]
    schedules = {"one prefill step": [0] * S,
                 f"the class's anchors ({lanes} every {every} steps)": [s // lanes * every for s in range(S)]}
    out = {}
    for name, arrive in schedules.items():
        r = schedule_stream_bound(asz, [{"prompt_len": prompt, "max_tokens": outputs, "arrive_step": a} for a in arrive])
        row = {"bytes": r["run_bytes"], "hidden_bytes": r["hidden_bytes"], "steps": r["steps"]}
        for key, b in (("all", r["run_bytes"]), ("F", r["run_bytes"] - r["hidden_bytes"])):
            c = -(-b // STREAM_ROW_BYTES) * STREAM_ROW
            s = {k: c / v for k, v in RATES.items()}
            row[key] = {"compressions": c, "ands": c * SHA512_ANDS, "ratio_ands": c * SHA512_ANDS / served_ands, "s": s,
                        "ratio_s": {k: v / served_s for k, v in s.items()}, "seal_s": b / SEAL_BYTES_S,
                        "seal_ratio": b / SEAL_BYTES_S / served_s}
        out[name] = row
    return out


def _slot_definition(lab: str, cls: WindowClass) -> object:
    """A slot's template (`class_size.placed`) at class ``cls``: its Definition, ``lab``'s statics bound (the entry's stop ids
    none, which its rows don't read)."""
    fam, s = CS.statics(lab)
    if fam == CS.SCAN:
        return SS.bridge(cls.scan, cls.prompt)
    make = {CS.FEED: IO.TokenFeed, CS.COLLECT: IO.OutCollect, CS.ENTRY: IO.SlotEntry}[fam]
    return bind(make, **s, **({"STOP": ()} if fam == CS.ENTRY else {}))


def _slot_unit(u: Unit, lab: str, at: WindowClass, cls: WindowClass) -> Unit:
    """The probe's slot template ``u`` (at class ``at``) as ``lab`` at class ``cls``: its Definition's rows there (as
    class_rows counts them, uncut), read through ``u``'s one kind of row; refused unless ``u`` is its Definition's at ``at``."""
    kind = [k for k in ("wired", "registered", "other") if getattr(u, k)]
    if len(kind) != 1:
        raise ValueError(f"the probe's {u.label} reads its rows through {len(kind)} kinds of row, not one")

    def rows(lab: str, c: WindowClass) -> Unit:
        fn = _slot_definition(lab, c)
        ins, outs = IF.unit_rows(fn, None)
        ob = [2 * w for _, w in IF.layout(fn).outs]
        return replace(u, label=lab, out=len(outs), out_bytes=sum(ob), x=sum(CR.blocks(128 + b) for b in ob),
                       **{kind[0]: len(ins), f"{kind[0]}_bytes": sum(bits // 8 for _, _, bits, _ in ins)})

    if rows(u.label, at) != u:
        raise ValueError(f"the probe's {u.label} is not its Definition's rows at the probe's class, uncut")
    return rows(lab, cls)


def project(census: dict, slots: int, prompt: int, outputs: int, target: str = "h200", lanes: int = 0,
            every: int = 16) -> tuple[list[Unit], Heads, dict]:
    """A census of one Program at a probe class within one key block (`class_rows.py --probe`), projected to the class of
    ``slots`` Programs of ``prompt`` ids and ``outputs`` outputs (``lanes`` and ``every`` `window_circuit.WindowClass`'s), each
    template outside attention at `class_size.projected`'s label and count, a slot's at its Definition's rows there; the
    heads rebuilt at the class's positions (`Heads.units`)."""
    c0 = census["class"]
    if c0["slots"] != 1:
        raise ValueError(f"a probe of {c0['slots']} Programs: project one")
    rest, heads = split(units_of(census), c0, target)
    at, to = WindowClass(1, c0["prompt"], c0["outputs"]), WindowClass(slots, prompt, outputs, lanes, every)
    out = []
    for u in rest:
        lab, n = CS.projected(u.label, u.n, at, to)
        if CS.placed(u.label, 1, at.positions) is not None:
            u = _slot_unit(u, lab, at, to)
        out.append(replace(u, label=lab, n=slots * n))
    cls = {"slots": slots, "prompt": prompt, "outputs": outputs, "positions": to.positions}
    return out, replace(heads, S=slots, P=to.positions), cls


def position_layout(rest: list[Unit], heads: Heads, levers: frozenset[str], T: int) -> dict:
    """One slot's output rows at position T under ``levers`` as a `commit_slots` step layout (`slots._layout_json`'s form,
    for ``step.py time`` at ``--slots S``): a group a template, one tensor of its units' bytes, each unit its rows, a unit's
    bytes split evenly over its rows (the census keeps a unit's total, not each row's). A template outside attention is at
    every position as often, a slot's where it places it (`class_size.placed`). No lever but ``P``, whose shared rows are
    another layout (`e2e_h200.window.shared_layout`), changes a template's output rows outside attention."""
    if "P" in levers:
        raise ValueError("P's shared rows are FAMILIES' shared-row layout (e2e_h200.window.shared_layout), not this one")
    S, P, BN = heads.S, heads.P, EXACT_KEYS if "L1" in levers else heads.BN
    here = []
    for u in rest:
        k = CS.placed(u.label, T, P)
        if k is None:
            k, r = divmod(u.n, S * P)
            if r:
                raise ValueError(f"{u.n} {u.label} units are not a fixed count at each of {S} × {P} positions")
        here.append((u, k))
    nb, nvis = heads.blocks_at(T, BN)
    one = replace(heads, S=1, P=T, BN=BN)
    for u in one.units(BN=BN, packed="L3b" in levers):
        fam, s = CS.statics(u.label)
        k = heads.H * (nb - 1) if fam == heads.block else heads.H if fam == END or s["NVIS"] == nvis else 0
        here.append((u, k))
    groups = []
    for u, k in here:
        if not k or not u.out:
            continue
        sizes = [u.out_bytes // u.out + (j < u.out_bytes % u.out) for j in range(u.out)]
        units, at = [], 0
        for _ in range(k):
            unit = []
            for b in sizes:
                unit.append([[0, at, b]])
                at += b
            units.append(unit)
        groups.append({"sources": [at], "units": units})
    return {"name": f"window_cost {S} x {P} at T {T}, levers {','.join(sorted(levers)) or 'none'}", "groups": groups}


#: the plan's order, each step stacked on those before it: exactness (no pad pass), the registration, the rows' layout
PLAN = ("L1", "L2", "L4", "W", "R", "L3b", "P")


def price(census: dict, target: str = "h200", served_s: float = 1.7, step_s: float = 0.023,
          to: tuple[int, ...] | None = None, cpu: dict | None = None, sizing: dict | None = None) -> dict:
    """Every figure the report prints, as data, for ``census`` or its projection ``to`` a class (slots, prompt, outputs[,
    lanes, every]): each lever alone and the `PLAN` stacked. Every compression counts in ANDs wherever it runs. In seconds,
    the window's rows (and under ``L4`` the member tree) run on the GPU at `RATES`; the registration without ``L4`` runs on
    the committer's CPU, at ``cpu``'s measured rates (`measure_cpu`), else at the GPU's (``registration_cpu`` None: a lower
    bound). With an arena ``sizing`` record, serving's own stream commit too (`stream`)."""
    if to is None:
        cls = census["class"]
        rest, heads = split(units_of(census), cls, target)
        total = census["total"]["compressions"]
    else:
        rest, heads, cls = project(census, *to[:3], target, *to[3:])
        total = None
    out = {"class": cls, "target": target, "served_s": served_s, "step_s": step_s, "sha512_ands": SHA512_ANDS,
           "census_compressions": total, "projected_from": None if to is None else census["class"], "cpu": cpu}
    served0 = served_ands(rest, heads, frozenset(), target)
    att = sum(heads.ands_at(T, heads.BN, target) for T in range(1, heads.P + 1))
    base_ands = (served0 - heads.S * att) / (heads.S * heads.P)
    rows = []
    sets = [("baseline (census model)", frozenset(), CENSUS), ("baseline (engine)", frozenset(), ENGINE)]
    sets += [(f"{k}: {LEVERS[k]}", frozenset({k}), ENGINE) for k in LEVERS]
    sets += [("plan: " + " + ".join(PLAN[:i]), frozenset(PLAN[:i]), ENGINE) for i in range(2, len(PLAN) + 1)]
    for name, levers, costs in sets:
        us = window_units(rest, heads, levers)
        rows_c = round(sum(u.compressions(costs) for u in us))
        reg = RegistrationSize.of(us, "L4" in levers)
        served = served_ands(rest, heads, levers, target)
        pads = {t.name: pad_pass(heads, t, levers, base_ands, step_s, target) for t in TRAFFIC}
        full = pads[TRAFFIC[0].name]
        commit_c = rows_c + reg.compressions
        on_cpu = None if cpu is None or reg.member_root else reg.cpu_s(cpu)
        gpu_c = rows_c if on_cpu is not None else commit_c
        commit_s = {r: gpu_c / v + (on_cpu["total"] if on_cpu else 0.0) for r, v in RATES.items()}
        rows.append({"levers": name, "set": sorted(levers), "costs": costs.name, "row_compressions": rows_c,
                     "registration": reg.__dict__, "registration_cpu": on_cpu, "commit_compressions": commit_c,
                     "commit_ands": commit_c * SHA512_ANDS, "served_ands": served,
                     "ratio_ands": commit_c * SHA512_ANDS / served, "commit_s": commit_s,
                     "ratio_s": {r: s / served_s for r, s in commit_s.items()}, "pad_pass": pads,
                     "full_ratio_ands": (commit_c * SHA512_ANDS + full["ands"]) / served,
                     "full_ratio_s": (commit_s["3.1 G/s"] + full["seconds"]) / served_s})
    out["rows"] = rows
    out["stream"] = None if sizing is None else stream(sizing, cls, target, served0, served_s)
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
    cpu = doc["cpu"]
    rates = ("the registration on the GPU's rates (no CPU rates given: a lower bound)" if cpu is None else
             f"the registration on {cpu['host']}'s CPU, one thread: SHA-512 {cpu['sha512_bytes_s'] / 1e9:.2f} GB/s, JSON "
             f"{cpu['json_bytes_s'] / 1e6:.0f} MB/s, writes {cpu['write_bytes_s'] / 1e9:.2f} GB/s, {cpu['files_s']:,.0f} files/s")
    lines = [f"class {c['slots']} × {c['positions']} positions ({c['prompt']} ids, {c['outputs']} outputs), target "
             f"{doc['target']}; served window {served_s} s; a decode step {doc['step_s'] * 1e3:.0f} ms; "
             f"{doc['sha512_ands']:,} ANDs a compression; target 0.20–0.25×; {source}; {rates}",
             "", "| levers | rows compr. | registration compr. | commit ANDs | served ANDs | ratio, ANDs | commit s at 3.1 "
             "(4.79) G/s | ratio, time | with the full window's pad pass: ANDs, time |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in doc["rows"]:
        lines.append(f"| {r['levers']} | {_e(r['row_compressions'])} | {_e(r['registration']['compressions'])} | "
                     f"{_e(r['commit_ands'])} | {_e(r['served_ands'])} | {r['ratio_ands']:.3f}× | "
                     f"{r['commit_s']['3.1 G/s']:.3g} s ({r['commit_s']['4.79 G/s']:.3g} s) | {r['ratio_s']['3.1 G/s']:.3g}× | "
                     f"{r['full_ratio_ands']:.3f}×, {r['full_ratio_s']:.3g}× |")
    lines += ["", "the pad pass:", "", "| levers | " + " | ".join(t.name for t in TRAFFIC) + " |",
              "| --- |" + " --- |" * len(TRAFFIC)]
    for r in doc["rows"]:
        if r["costs"] == "engine":
            lines.append(f"| {r['levers']} | " + " | ".join(
                f"{p['steps']} steps, {p['seconds']:.1f} s, {_e(p['ands'])} ANDs" for p in r["pad_pass"].values()) + " |")
    lines += ["", "the registration (public and private bytes: what the prover keeps until the draw):", "",
              "| levers | members | port roots | public bytes | private bytes | compr. | JSON bytes | files | CPU s |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in doc["rows"]:
        g, on = r["registration"], r["registration_cpu"]
        lines.append(f"| {r['levers']} | {g['members']:,} | {g['roots']:,} | {_e(g['public_bytes'])} | {_e(g['private_bytes'])} | "
                     f"{_e(g['compressions'])} | {_e(g['json_bytes'])} | {g['files']:,} | "
                     f"{'-' if on is None else f'{on['total']:.3g}'} |")
    st = doc.get("stream")
    if st:
        lines += ["", f"serving's own stream commit, made while it serves, before the window's commit and beside it (rows-sha512: "
                  f"{STREAM_ROW} compressions a 4 KiB row; the seal's store at {SEAL_BYTES_S / 1e9:.3f} GB/s, "
                  f"{CS.SOURCES['seal_store']}), the full window; {STREAM_LEVER}:", "",
                  "| schedule | stream bytes | fa*.m1 bytes (share) | commit ANDs (ratio) | commit s at 3.1 (4.79) G/s (ratio) | "
                  "seal s (ratio) | under F: commit ANDs (ratio) | commit s (ratio) | seal s (ratio) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, r in st.items():
            cells = [f"{_e(r[k]['ands'])} ({r[k]['ratio_ands']:.4f}×) | {r[k]['s']['3.1 G/s']:.3g} s "
                     f"({r[k]['s']['4.79 G/s']:.3g} s) ({r[k]['ratio_s']['3.1 G/s']:.3f}×) | {r[k]['seal_s']:.3g} s "
                     f"({r[k]['seal_ratio']:.3g}×)" for k in ("all", "F")]
            lines.append(f"| {name} | {_e(r['bytes'])} | {_e(r['hidden_bytes'])} ({r['hidden_bytes'] / r['bytes']:.2f}) | "
                         f"{cells[0]} | {cells[1]} |")
    base, plan = "baseline (engine)", "plan: " + " + ".join(PLAN)
    by = {r["levers"]: r for r in doc["rows"]}
    pad = {k: by[k]["pad_pass"][TRAFFIC[0].name]["ands"] for k in (base, plan)}
    worst = max(st.values(), key=lambda r: r["bytes"]) if st else None
    head = ("| period | windows | served ANDs | commit ANDs, baseline (engine) | share | commit ANDs, the plan | share |"
            + (" with serving's stream: baseline | share | the plan + F | share |" if worst else ""))
    lines += ["", f"over the reference datacenter ({GPUS:,} GPUs) serving full class-of-record windows back to back (commit "
              "with its pad pass; share: of the served ANDs" + ("; serving's stream at its larger schedule" if worst else "")
              + "):", "", head, "|" + " --- |" * (head.count("|") - 1)]
    for p, v in doc["periods"].items():
        w, sv = v["windows"], v["served_ands"]
        b, a = v["commit_ands"][base] + w * pad[base], v["commit_ands"][plan] + w * pad[plan]
        line = f"| {p} | {_e(w)} | {_e(sv)} | {_e(b)} | {b / sv:.3f} | {_e(a)} | {a / sv:.3f} |"
        if worst:
            b2, a2 = b + w * worst["all"]["ands"], a + w * worst["F"]["ands"]
            line += f" {_e(b2)} | {b2 / sv:.3f} | {_e(a2)} | {a2 / sv:.3f} |"
        lines.append(line)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("census", help="class_rows.py --out's JSON")
    ap.add_argument("--target", choices=("h200", "sm120"), default="h200")
    ap.add_argument("--served-s", type=float, default=1.7, help="the served window's seconds (V3c's estimate: 4 prefill and 63 "
                    "decode steps)")
    ap.add_argument("--step-ms", type=float, default=23.0, help="a batch decode step's ms (a dry run's)")
    ap.add_argument("--project", default=None, metavar="SLOTS,PROMPT,OUTPUTS[,LANES,EVERY]",
                    help="price a one-Program probe's projection to this class (`project`)")
    cpu = ap.add_mutually_exclusive_group()
    cpu.add_argument("--measure-cpu", default=None, metavar="SCRATCH", help="measure this host's CPU rates (`measure_cpu`), "
                     "writing its test files into SCRATCH, and price the registration at them")
    cpu.add_argument("--cpu-rates", default=None, metavar="FILE", help="price the registration at an earlier --out's rates")
    ap.add_argument("--measure-mib", type=int, default=256, help="MiB each of --measure-cpu's rates is measured over")
    ap.add_argument("--layout", nargs=2, default=None, metavar=("T,T,...", "DIR"), help="write one slot's output rows at "
                    "each position T as a commit_slots layout, DIR/pos<T>.json (`position_layout`), for step.py time")
    ap.add_argument("--layout-levers", default="L1", help="the levers the layouts are under, comma-separated (default L1)")
    ap.add_argument("--stream", default=None, metavar="SIZING", help="price serving's own stream commit (`stream`) at this arena "
                    "sizing record's widths, or a fixture's under \"sizing\" (Llama-3.2-1B's: "
                    "integrations/vllm/tests/pipeline/fixtures/admission/b16.json)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    to = tuple(int(v) for v in a.project.split(",")) if a.project else None
    if a.layout:
        census = json.loads(Path(a.census).read_text())
        rest, heads = (project(census, *to[:3], a.target, *to[3:])[:2] if to else
                       split(units_of(census), census["class"], a.target))
        levers, d = frozenset(v for v in a.layout_levers.split(",") if v), Path(a.layout[1])
        d.mkdir(parents=True, exist_ok=True)
        for T in (int(v) for v in a.layout[0].split(",")):
            (d / f"pos{T}.json").write_text(json.dumps(position_layout(rest, heads, levers, T)) + "\n")
        return 0
    rates = (measure_cpu(Path(a.measure_cpu), a.measure_mib) if a.measure_cpu else
             json.loads(Path(a.cpu_rates).read_text())["cpu"] if a.cpu_rates else None)
    sizing = json.loads(Path(a.stream).read_text()) if a.stream else None
    sizing = sizing.get("sizing", sizing) if sizing else None
    doc = price(json.loads(Path(a.census).read_text()), a.target, a.served_s, a.step_ms / 1e3, to, rates, sizing)
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(report(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
