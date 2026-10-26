"""The padded size of a window class's circuit (V3b), at a class too large to lower: its units, ANDs and m.

`window_circuit.class_circuit` lowers the class Program once a slot, exactly. At the class of record
(`window_circuit.CLASS_OF_RECORD`: 8 slots, 1,024 prompt ids, 64 outputs, 1,087 positions a Program) a Program at
Llama-3.2-1B is 46,028,317 units, and the class's 8 take a pod some 22 minutes and 42 GB to lower. So the size is also a
projection (`project`) from an exact lowering of one Program of the same registration at a small probe class within one
key block, which the tests hold to the exact lowering past one block and two, and at the class of record itself:

* every template but the heads and their key gathers is a fixed number of units a position (`TokenFeed_v1` a prompt
  position; `ScanSlots_v1`, at the class's scan state row, `SlotRow_v1` and `SlotEntry_v1` one a Program; `OutCollect_v1`
  one a position, its first the first's), read off the probe, which is refused unless each divides evenly;
* a head over T keys is `AttentionHead_v19`'s Calls: `AttnBlockFirst_v1{NVIS}` (its last NVIS = T - (nb - 1)·BN keys),
  nb - 1 `AttnBlock_v5` and `AttnHeadEnd_v1`, nb = ceil(T / BN), a full block's CHECK whether FA2 guards it (its
  ``MASKED_FROM``, `attention_vocab.fa2_masked_from` of the Program's one-row decode: its first visited block only). Its
  keys and values reach it as an `AllGather_v1{BN, D}` row a full block, written once (when the block fills) and read by
  every later position, and an `AllGather_v1{NVIS, D}` of its first block, none where that block is one key (RoPE's row)
  or a full one.

`size` prices a census with the ANDs and k_log a unit at Llama-3.2-1B (`measured`, each with its source, scaled or
interpolated where only another binding was measured, and saying so), and m = ceil(log2 Σ units · 2^k_log). A gather's
k_log is its rows' (`rows_k`: C-Flock's composer gives each row a unit reads or writes an hm96 slot), so a unit past 256
rows in and out, past a 2^27-bit block, is refused as C-Flock refuses it.

    prove_path.py class-size --serve-run DIR --window K --row-dir DIR --advice FILE \\
        [--class SLOTS,PROMPT,OUTPUTS[,LANES,EVERY]] [--probe 1,2,2]
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from verity_vllm.program.predict import scan_state as SS

from verity.ml.definitions.registry.attention_vocab import fa2_masked_from

FIRST = "AttnBlockFirst_v1"
END = "AttnHeadEnd_v1"
BLOCK = "AttnBlock_v5"
FEED = "TokenFeed_v1"
ROW = "SlotRow_v1"
SCAN = "ScanSlots_v1"
COLLECT = "OutCollect_v1"
ENTRY = "SlotEntry_v1"
GATHER = "AllGather_v1"


def label(fn: Any) -> str:
    """A template's family and its integer and bool statics, the key a census counts it under."""
    return fn.id.split("{")[0] + "{" + ",".join(f"{k}={json.dumps(v)}" for k, v in fn.bindings.items() if type(v) in (int, bool)) + "}"


def statics(lab: str) -> tuple[str, dict[str, Any]]:
    fam, _, body = lab.partition("{")
    return fam, {k: json.loads(v) for k, v in (kv.split("=") for kv in body.rstrip("}").split(",") if kv)}


def _relabel(lab: str, **at: Any) -> str:
    """``lab`` with the statics ``at`` names set to theirs, where it has them."""
    fam, s = statics(lab)
    return fam + "{" + ",".join(f"{k}={json.dumps(at.get(k, v))}" for k, v in s.items()) + "}"


def census(circuit: Any) -> dict[str, int]:
    """``circuit``'s units by `label`."""
    out: Counter[str] = Counter()
    for fn, n in circuit.templates:
        out[label(fn)] += n
    return dict(out)


def _one(xs: list[Any], what: str) -> Any:
    if len(set(xs)) != 1:
        raise ValueError(f"the probe's heads differ in {what}: {sorted(set(xs))}")
    return xs[0]


def project(probe: Mapping[str, int], at: Any, cls: Any) -> dict[str, int]:
    """One Program's census at class ``cls`` from one Program's ``probe`` census at class ``at`` (`census` of
    ``class_circuit(registered, at, 1)``), the probe's heads within one key block."""
    P0, P, NP0 = at.positions, cls.positions, at.prompt
    if P0 < 3 or NP0 < 2:
        raise ValueError(f"a probe of {P0} positions and {NP0} prompt ids: it needs 3 and 2, to read a key gather twice and a "
                         "feed past the first")
    hs = {lab: n for lab, n in probe.items() if lab.startswith(FIRST + "{")}
    ss = [statics(lab)[1] for lab in hs]
    D, BN = (_one([s[k] for s in ss], k) for k in ("D", "BN"))
    by_t = {s["NVIS"]: hs[lab] for lab, s in zip(hs, ss)}
    if P0 > BN or sorted(by_t) != list(range(1, P0 + 1)) or len(set(by_t.values())) != 1:
        raise ValueError(f"the probe's heads are not one count of first blocks at each T of 1..{P0}, within one key block of "
                         f"{BN}: {by_t}")
    H, G, end = by_t[1], probe.get(f"{GATHER}{{WORLD=2,N={D}}}", 0), f"{END}{{D={D}}}"
    if probe.get(end) != H * P0 or any(lab.startswith(BLOCK + "{") for lab in probe):
        raise ValueError(f"the probe has {probe.get(end, 0)} {end} units, not one a head at each of its {P0} positions, or a "
                         f"{BLOCK} unit within one key block")
    rest = Counter({lab: n for lab, n in probe.items() if lab not in hs and lab != end})
    for t in range(2, P0 + 1):
        key = f"{GATHER}{{WORLD={t},N={D}}}"
        if rest[key] < G:
            raise ValueError(f"the probe has {rest[key]} {key} units, fewer than its {G} key gathers a position")
        rest[key] -= G
    out: Counter[str] = Counter()
    for lab, n in rest.items():
        fam, s = statics(lab)
        if fam == FEED:
            out[lab.replace(f"NP={NP0}", f"NP={cls.prompt}")] += 1 if s["FIRST"] else cls.prompt - 1
            if n != (1 if s["FIRST"] else NP0 - 1):
                raise ValueError(f"the probe has {n} {lab} units, not one a prompt position")
        elif fam in (SCAN, ROW, ENTRY):
            out[label(SS.bridge(cls.scan, cls.prompt)) if fam == SCAN else _relabel(lab, NP=cls.prompt, KO=cls.outputs)] += 1
            if n != 1:
                raise ValueError(f"the probe has {n} {lab} units, not one a Program")
        elif fam == COLLECT:
            out[_relabel(lab, KO=cls.outputs)] += 1 if s["FIRST"] else P - 1
            if n != (1 if s["FIRST"] else P0 - 1):
                raise ValueError(f"the probe has {n} {lab} units, not one a position")
        elif n % P0:
            raise ValueError(f"the probe has {n} {lab} units, not a fixed count at each of its {P0} positions")
        elif n:
            out[lab] += n // P0 * P
    for T in range(1, P + 1):
        nb, mf = -(-T // BN), fa2_masked_from(T, 1, 0, BN, False)
        nvis = T - (nb - 1) * BN
        out[f"{FIRST}{{D={D},NVIS={nvis},BN={BN}}}"] += H
        for b in range(nb - 1):
            out[f"{BLOCK}{{D={D},NVIS={BN},FIRST=false,BN={BN},CHECK={json.dumps(b >= mf)}}}"] += H
        out[end] += H
        out[f"{GATHER}{{WORLD={nvis},N={D}}}"] += G if 1 < nvis < BN else 0
    out[f"{GATHER}{{WORLD={BN},N={D}}}"] += G * (P // BN)
    return {lab: n for lab, n in sorted(out.items()) if n}


#: circuits' compositions of the blocks this estimate scaled or interpolated, each staged by C-Flock's `UnitStager` at V3b's
#: cuts (thread 1792894842.280569, 25 Oct)
COMPOSED = ("circuits' compositions r20261025-183214-951f, r20261025-183237-6217 and r20261025-183911-2915 "
            "(art:65bc8cddf39ec0076f17c19b4e13a74ad26554736b531b43fe6063f778cdcaf3)")
#: ANDs and k_log a unit at Llama-3.2-1B, each with its source: k_log is CIR-95's table at its cut reads
#: (note:circuits/20261024T0443Z-report-windows-advice-windows, r20261025-093758-baf3) unless said otherwise
MEASURED: dict[str, tuple[int, int, str]] = {
    "EmbeddingRead_v1{V=128256,H=2048}": (5_392_976, 25, "#4227's table (ANDs and block)"),
    "RMSNormTriton_v5{N=2048}": (9_416_444, 25, "r20261025-072237-b8e3"),
    "RMSNormFusedCuda_v2{N=2048}": (10_474_114, 27, "r20261025-072834-bd85; k_log the post-attention norm's (x cut 128), the "
                                    "input and final norms' (x cut 4) 25"),
    "Gemm_v2{K=2048,N=16}": (16 * 1_053_595, 27, "scaled: 16 × Gemm_v2{K=2048,N=1}'s, r20261025-072834-bd85"),
    "Gemm_v2{K=8192,N=4}": (4 * 4_215_067, 27, "scaled: 4 × Gemm_v2{K=8192,N=1}'s, r20261025-072834-bd85"),
    "Gemm_v2{K=2048,N=1}": (1_053_595, 24, "r20261025-072834-bd85"),
    "RoPEHead_v1{D=64}": (191_872, 22, COMPOSED + ", x cut 4"),
    "SiluMul_v2{I=8192}": (8192 * 1380, 26, "8,192 × SiluMulBf16_v4's 1,380 (circuits-bool-silu)"),
    "TokenSelectChunk_v2{C=768}": (91_823, 22, COMPOSED),
    "TokenSelectCombine_v2{V=128256,C=768}": (27_355, 27, COMPOSED + ", idx cut 167"),
    "AttnHeadEnd_v1{D=64}": (169_842, 22, "#4263, r20261025-123346-091b"),
    "AttnBlock_v5{D=64,NVIS=128,FIRST=false,BN=128,CHECK=false}": (9_281_000, 26, "#4263, r20261025-123346-091b"),
    "AttnBlock_v5{D=64,NVIS=128,FIRST=false,BN=128,CHECK=true}": (9_281_000, 26, "CHECK=false's (#4263, r20261025-123346-091b), "
                                                                   "which circuits found it equal to"),
}
#: `AttnBlockFirst_v1{D=64,NVIS,BN=128}`'s ANDs where #4263 (r20261025-123346-091b: NVIS 1, 17, 128) and circuits (`COMPOSED`)
#: composed it, linear between
FIRST_BLOCKS = {1: 96_886, 2: 165_175, 4: 300_567, 5: 367_670, 6: 435_639, 8: 571_031, 12: 843_099, 14: 979_101, 15: 1_047_070,
                16: 1_115_295, 17: 1_231_456, 24: 1_708_199, 32: 2_252_527, 33: 2_368_688, 34: 2_436_657, 36: 2_573_043,
                40: 2_845_431, 48: 3_389_759, 64: 4_526_991, 80: 5_664_223, 88: 6_257_127, 90: 6_393_065, 92: 6_529_131,
                96: 6_801_455, 112: 7_938_687, 128: 9_075_919}
#: its k_log: (the last NVIS at it, k_log), the steps circuits bisected to adjacent NVIS
FIRST_K = ((4, 22), (15, 23), (33, 24), (90, 25), (128, 26))
#: `TokenFeed_v1{NP}`'s k_log as circuits composed it: (the most NP, k_log), an upper bound below a composed NP
FEED_K = ((1024, 23), (2048, 24))
#: C-Flock's largest block, 2^27 bits (`verity_flock.circuit.K_MAX`)
K_MAX = 27


def rows_k(rows: int) -> int:
    """The least k_log of a unit of ``rows`` rows in and out: C-Flock's composer gives each an hm96 slot, the slots rounded up
    to a power of two, at 2^19 bits a slot of block, and no block is under 2^22. Every `AllGather_v1{W, 64}` circuits composed
    (W 2 to 128) and `{64, 16}` take exactly this at their W + 1 rows (`COMPOSED`), so a gather of 256 parts takes 2^28."""
    return max(22, 19 + (rows - 1).bit_length())


def measured(lab: str) -> tuple[int, int, str]:
    """(ANDs, k_log, source) of a unit ``lab`` at Llama-3.2-1B, refused where nothing measured it or a binding it scales from."""
    if lab in MEASURED:
        return MEASURED[lab]
    fam, s = statics(lab)
    if fam == GATHER and s["WORLD"] >= 2:
        return 0, rows_k(s["WORLD"] + 1), f"a copy (b1_tp2); k_log its {s['WORLD'] + 1} rows' (rows_k)"
    if fam == "AttnBlockFirst_v1" and (s["D"], s["BN"]) == (64, 128) and 1 <= s["NVIS"] <= 128:
        v, (lo, hi) = s["NVIS"], next((a, b) for a, b in zip(FIRST_BLOCKS, list(FIRST_BLOCKS)[1:]) if s["NVIS"] <= b)
        ands = round(FIRST_BLOCKS[lo] + (FIRST_BLOCKS[hi] - FIRST_BLOCKS[lo]) * max(v - lo, 0) / (hi - lo))
        k = next(k for last, k in FIRST_K if v <= last)
        return ands, k, "#4263 (r20261025-123346-091b) and " + COMPOSED + ("" if v in FIRST_BLOCKS else ": ANDs interpolated")
    if fam == FEED and (k := next((k for most, k in FEED_K if s["NP"] <= most), None)) is not None:
        return 30 if s["FIRST"] else 62, k, "circuit-check r20261025-174007-faec; k_log " + COMPOSED + (
            "" if s["NP"] in dict(FEED_K) else ", an upper bound")
    if fam == ROW and (k := next((k for most, k in FEED_K if s["NP"] <= most), None)) is not None:
        return 0, k, ("a copy (circuit-check r20261026-172423-b1a4); k_log an estimate, not composed: TokenFeed_v1{NP, "
                      "FIRST=true}'s, whose rows it holds but for one i32 more")
    if lab == "SlotMask_v1{}":
        return 32, 22, "circuit-check r20261026-172423-b1a4; k_log the least block, its 3 rows of one i32 each"
    raise ValueError(f"no measured ANDs or k_log for {lab} at Llama-3.2-1B: add its measurement to MEASURED")


def size(units: Mapping[str, int], times: int = 1) -> dict[str, Any]:
    """``units`` (a census) ``times`` over, priced by `measured`: units, ANDs, m and each template's row; refused naming every
    template with no measurement."""
    missing = []
    for lab in units:
        try:
            measured(lab)
        except ValueError:
            missing.append(lab)
    if missing:
        raise ValueError(f"no measured ANDs or k_log at Llama-3.2-1B for {', '.join(missing)}: add each to MEASURED")
    rows = [(lab, n * times, *measured(lab)) for lab, n in units.items()]
    past = [f"{lab} (2^{k})" for lab, _, _, k, _ in rows if k > K_MAX]
    if past:
        raise ValueError(f"{', '.join(past)}: an instance needs a block past 2^{K_MAX} bits, C-Flock's most; cut it finer")
    cells = sum(n << k for _, n, _, k, _ in rows)
    return {"units": sum(n for _, n, *_ in rows), "ands": sum(n * a for _, n, a, _, _ in rows), "m": math.ceil(math.log2(cells)),
            "templates": [{"template": lab, "units": n, "ands": a, "k_log": k, "source": src}
                          for lab, n, a, k, src in sorted(rows, key=lambda r: -r[1] * r[2])]}
