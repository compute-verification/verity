"""The padded size of a window class's circuit (V3b), at a class too large to lower: its units, ANDs and m.

`window_circuit.class_circuit` lowers the class Program once a slot, exactly. At the class of record
(`window_circuit.CLASS_OF_RECORD`: 8 slots, 1,024 prompt ids, 64 outputs, 1,087 positions a Program) a Program at
Llama-3.2-1B is 46,028,317 units, and the class's 8 take a pod some 22 minutes and 42 GB to lower. So the size is also a
projection (`project`) from an exact lowering of one Program of the same registration at a small probe class within one
key block, which the tests hold to the exact lowering at larger classes:

* every template but the heads is a fixed number of units a position (`projected`), a slot's where it places them
  (`placed`: `TokenFeed_v1` a prompt position past the first; `ScanSlots_v2`, at the class's scan state row, and
  `SelectEntry_v1` one a Program; `SelectCollect_v1` one at every other position, its first the first's), read off the
  probe, which is refused unless each divides evenly;
* a head over T keys is `AttentionHead_v19`'s Calls: `AttnBlockFirst_v1{NVIS}` (its last NVIS = T - (nb - 1)·BN keys),
  nb - 1 `AttnBlock_v5` and `AttnHeadEnd_v1`, nb = ceil(T / BN), a full block's CHECK whether FA2 guards it (its
  ``MASKED_FROM``, `attention_vocab.fa2_masked_from` of the Program's one-row decode: its first visited block only). A
  block reads its keys and values a row a key, so the exact lowering refuses a block past 126 keys, more than a unit's 256
  rows; the projection counts it all the same.

`size` prices a census with the ANDs and k_log a unit at Llama-3.2-1B (`measured`, each with its source, scaled or
interpolated where only another binding was measured, and saying so), and m = ceil(log2 Σ units · 2^k_log), refusing a
unit past a 2^27-bit block as C-Flock refuses it. A unit composed of few ANDs takes its rows' k_log (`rows_k`: C-Flock's
composer gives each row a unit reads or writes an hm96 slot).

    prove_path.py class-size --serve-run DIR --window K --row-dir DIR --advice FILE \\
        [--class SLOTS,PROMPT,OUTPUTS[,LANES,EVERY]] [--probe 1,2,2]
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity_vllm.program.predict import scan_state as SS

from verity.ml.definitions.registry.attention_vocab import fa2_masked_from

FIRST = "AttnBlockFirst_v1"
#: a first block's families: `AttnBlockFirst_v2{..., VS}` is v1 over a key's value as slices of VS columns
FIRSTS = (FIRST, "AttnBlockFirst_v2")
END = "AttnHeadEnd_v1"
BLOCK = "AttnBlock_v5"
FEED = "TokenFeed_v1"
SCAN = "ScanSlots_v2"
#: a position's select with the slot's collector, at every position but the last, and the last's with the slot's entry
COLLECT = "SelectCollect_v1"
ENTRY = "SelectEntry_v1"


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


def placed(lab: str, T: int, P: int) -> int | None:
    """A Program's units of template ``lab`` at position T of its P, for a slot's template: `TokenFeed_v1` at each of its NP
    prompt positions past the first, `ScanSlots_v2` at the first, `SelectEntry_v1` at the last and `SelectCollect_v1` at
    each other (its first the first's); None for any other, a fixed count at every position."""
    fam, s = statics(lab)
    if fam == FEED:
        return int(1 < T <= s["NP"])
    if fam == SCAN:
        return int(T == 1)
    if fam == ENTRY:
        return int(T == P)
    if fam == COLLECT:
        return int(T == 1 < P) if s["FIRST"] else int(1 < T < P)
    return None


def projected(lab: str, n: int, at: Any, cls: Any) -> tuple[str, int]:
    """A template outside the heads, ``n`` units of one Program at the probe's class ``at``, as one Program at class ``cls``
    has it: a slot's template (`placed`) at the class's prompt and outputs (`ScanSlots_v2` at its scan state row) and
    `placed`'s units there, any other the probe's fixed count a position at each of the class's, refused unless ``n`` is
    the probe's by that rule."""
    if placed(lab, 1, at.positions) is None:
        if n % at.positions:
            raise ValueError(f"the probe has {n} {lab} units, not a fixed count at each of its {at.positions} positions")
        return lab, n // at.positions * cls.positions
    if n != (want := sum(placed(lab, T, at.positions) or 0 for T in range(1, at.positions + 1))):
        raise ValueError(f"the probe has {n} {lab} units, not the {want} its slot places")
    to = label(SS.bridge(cls.scan, cls.prompt)) if lab.startswith(SCAN + "{") else _relabel(lab, NP=cls.prompt, KO=cls.outputs)
    return to, sum(placed(to, T, cls.positions) or 0 for T in range(1, cls.positions + 1))


def project(probe: Mapping[str, int], at: Any, cls: Any) -> dict[str, int]:
    """One Program's census at class ``cls`` from one Program's ``probe`` census at class ``at`` (`census` of
    ``class_circuit(registered, at, 1)``), the probe's heads within one key block."""
    P0, P, NP0 = at.positions, cls.positions, at.prompt
    if P0 < 2 or NP0 < 2:
        raise ValueError(f"a probe of {P0} positions and {NP0} prompt ids: it needs 2 of each, for a head and a feed past the "
                         "first")
    hs = {lab: n for lab, n in probe.items() if lab.split("{")[0] in FIRSTS}
    ss = [statics(lab)[1] for lab in hs]
    D, BN = (_one([s[k] for s in ss], k) for k in ("D", "BN"))
    fam, VS = _one([lab.split("{")[0] for lab in hs], "family"), _one([s.get("VS") for s in ss], "VS")
    vs = "" if VS is None else f",VS={VS}"
    by_t = {s["NVIS"]: hs[lab] for lab, s in zip(hs, ss)}
    if P0 > BN or sorted(by_t) != list(range(1, P0 + 1)) or len(set(by_t.values())) != 1:
        raise ValueError(f"the probe's heads are not one count of first blocks at each T of 1..{P0}, within one key block of "
                         f"{BN}: {by_t}")
    H, end = by_t[1], f"{END}{{D={D}}}"
    if probe.get(end) != H * P0 or any(lab.startswith(BLOCK + "{") for lab in probe):
        raise ValueError(f"the probe has {probe.get(end, 0)} {end} units, not one a head at each of its {P0} positions, or a "
                         f"{BLOCK} unit within one key block")
    out: Counter[str] = Counter()
    for lab, n in probe.items():
        if lab not in hs and lab != end:
            to, k = projected(lab, n, at, cls)
            out[to] += k
    for T in range(1, P + 1):
        nb, mf = -(-T // BN), fa2_masked_from(T, 1, 0, BN, False)
        nvis = T - (nb - 1) * BN
        out[f"{fam}{{D={D},NVIS={nvis},BN={BN}{vs}}}"] += H
        for b in range(nb - 1):
            out[f"{BLOCK}{{D={D},NVIS={BN},FIRST=false,BN={BN},CHECK={json.dumps(b >= mf)}}}"] += H
        out[end] += H
    return {lab: n for lab, n in sorted(out.items()) if n}


#: the sources that cite a run, an artifact or a change, kept as data (``sources.json``): ``composed`` and ``rows_3``, a
#: unit's by its label (``measured``) and a family's that `measured` prices past `MEASURED` (``families``)
SOURCES: dict[str, Any] = json.loads(Path(__file__).resolve().with_name("sources.json").read_text())
#: the compositions of the blocks this estimate scaled or interpolated, each staged by C-Flock's `UnitStager` at V3b's
#: cuts
COMPOSED: str = SOURCES["composed"]
#: the source of a scan slot unit circuit-check lowered whole, of 3 rows in and out
ROWS_3: str = SOURCES["rows_3"]
#: ANDs and k_log a unit at Llama-3.2-1B, each with its source (None: its label's in `SOURCES`): k_log is the cut reads'
#: table unless said otherwise
MEASURED: dict[str, tuple[int, int, str]] = {lab: (ands, k, SOURCES["measured"][lab] if src is None else src)
                                             for lab, (ands, k, src) in {
    "EmbeddingRead_v1{V=128256,H=2048}": (5_392_976, 25, None),
    "RMSNormTriton_v5{N=2048}": (9_416_444, 25, None),
    "RMSNormFusedCuda_v2{N=2048}": (10_474_114, 27, None),
    "Gemm_v2{K=2048,N=16}": (16 * 1_053_595, 27, None),
    "Gemm_v2{K=8192,N=4}": (4 * 4_215_067, 27, None),
    "Gemm_v2{K=2048,N=1}": (1_053_595, 24, None),
    "RoPEHead_v1{D=64}": (191_872, 22, COMPOSED + ", x cut 4"),
    "SiluMul_v2{I=8192}": (8192 * 1380, 26, None),
    "TokenSelectGroups_v1{C=2672,G=16}": (292_083, 27, None),
    "TokenSelectCombine_v2{V=128256,C=16}": (1_346_657, 26, None),
    "AttnHeadEnd_v1{D=64}": (169_842, 22, None),
    "AttnBlock_v5{D=64,NVIS=128,FIRST=false,BN=128,CHECK=false}": (9_281_000, 26, None),
    "AttnBlock_v5{D=64,NVIS=128,FIRST=false,BN=128,CHECK=true}": (9_281_000, 26, None),
    "ScanSlots_v2{NS=9353,NP=1024,IDB=17,K0=12,K=45}": (120, 24, None),
    "OutCollect_v1{KO=64,FIRST=true}": (155, 22, ROWS_3),
    "OutCollect_v1{KO=64,FIRST=false}": (2_141, 22, ROWS_3),
    "SlotEntry_v1{KO=64,IDB=17,NP=1024}": (5_676, 22, None),
}.items()}


def _calls(*labs: str) -> tuple[int, int, str]:
    """A unit of Calls of ``labs``, the first its largest: their `MEASURED` ANDs summed and the first's k_log (its rows'
    too), an estimate, not composed."""
    return (sum(MEASURED[lab][0] for lab in labs), MEASURED[labs[0]][1],
            f"its Calls' ANDs summed ({', '.join(labs)}), k_log the first's: not composed")


#: the select with its collector, and the last position's with the slot's entry (`io_shell.SelectCollect_v1`,
#: `SelectEntry_v1`)
_COMBINE = "TokenSelectCombine_v2{V=128256,C=16}"
MEASURED |= {
    "SelectCollect_v1{V=128256,C=16,KO=64,FIRST=true}": _calls(_COMBINE, "OutCollect_v1{KO=64,FIRST=true}"),
    "SelectCollect_v1{V=128256,C=16,KO=64,FIRST=false}": _calls(_COMBINE, "OutCollect_v1{KO=64,FIRST=false}"),
    "SelectEntry_v1{V=128256,C=16,KO=64,FIRST=false,IDB=17,NP=1024}": _calls(
        _COMBINE, "OutCollect_v1{KO=64,FIRST=false}", "SlotEntry_v1{KO=64,IDB=17,NP=1024}"),
}
#: `AttnBlockFirst_v1{D=64,NVIS,BN=128}`'s ANDs where the heads' run (NVIS 1, 17, 128) and the compositions (`COMPOSED`)
#: composed it, linear between
FIRST_BLOCKS = {1: 96_886, 2: 165_175, 4: 300_567, 5: 367_670, 6: 435_639, 8: 571_031, 12: 843_099, 14: 979_101, 15: 1_047_070,
                16: 1_115_295, 17: 1_231_456, 24: 1_708_199, 32: 2_252_527, 33: 2_368_688, 34: 2_436_657, 36: 2_573_043,
                40: 2_845_431, 48: 3_389_759, 64: 4_526_991, 80: 5_664_223, 88: 6_257_127, 90: 6_393_065, 92: 6_529_131,
                96: 6_801_455, 112: 7_938_687, 128: 9_075_919}
#: its k_log: (the last NVIS at it, k_log), its steps bisected to adjacent NVIS
FIRST_K = ((4, 22), (15, 23), (33, 24), (90, 25), (128, 26))
#: `TokenFeed_v1{NP}`'s k_log as composed: (the most NP, k_log), an upper bound below a composed NP
FEED_K = ((1024, 23), (2048, 24))
#: C-Flock's largest block, 2^27 bits (`verity_flock.circuit.K_MAX`)
K_MAX = 27


def rows_k(rows: int) -> int:
    """The least k_log of a unit of ``rows`` rows in and out: C-Flock's composer gives each an hm96 slot, the slots rounded up
    to a power of two, at 2^19 bits a slot of block, and no block is under 2^22. Every copy unit of W + 1 rows composed (W 2
    to 128, `COMPOSED`) takes exactly this, so a unit of 257 rows takes 2^28."""
    return max(22, 19 + (rows - 1).bit_length())


def measured(lab: str) -> tuple[int, int, str]:
    """(ANDs, k_log, source) of a unit ``lab`` at Llama-3.2-1B, refused where nothing measured it or a binding it scales from."""
    if lab in MEASURED:
        return MEASURED[lab]
    fam, s = statics(lab)
    by = SOURCES["families"]
    if fam == "AttnBlockFirst_v1" and (s["D"], s["BN"]) == (64, 128) and 1 <= s["NVIS"] <= 128:
        v, (lo, hi) = s["NVIS"], next((a, b) for a, b in zip(FIRST_BLOCKS, list(FIRST_BLOCKS)[1:]) if s["NVIS"] <= b)
        ands = round(FIRST_BLOCKS[lo] + (FIRST_BLOCKS[hi] - FIRST_BLOCKS[lo]) * max(v - lo, 0) / (hi - lo))
        k = next(k for last, k in FIRST_K if v <= last)
        return ands, k, by[fam] + " and " + COMPOSED + ("" if v in FIRST_BLOCKS else ": ANDs interpolated")
    if fam == FEED and (k := next((k for most, k in FEED_K if s["NP"] <= most), None)) is not None:
        return 30 if s["FIRST"] else 62, k, by[fam] + "; k_log " + COMPOSED + (
            "" if s["NP"] in dict(FEED_K) else ", an upper bound")
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
