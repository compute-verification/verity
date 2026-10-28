"""The padded size of a window class's circuit (V3b), at a class too large to lower: its units, ANDs and m.

`window_circuit.class_circuit` lowers the class Program once a slot, exactly. At the class of record
(`window_circuit.CLASS_OF_RECORD`: 8 slots, 1,024 prompt ids, 64 outputs, 1,087 positions a Program) a Program at
Llama-3.2-1B is 46,028,317 units, and the class's 8 take a pod some 22 minutes and 42 GB to lower. So the size is also a
projection (`project`) from an exact lowering of one Program of the same registration at a small probe class, which the
tests hold to the exact lowering at larger classes:

* every template but the heads is a fixed number of units a position (`projected`), a slot's where it places them
  (`placed`: `TokenFeed_v1` a prompt position past the first; `ScanSlots_v2`, at the class's scan state row, and
  `SlotEntry_v1` one a Program; `OutCollect_v1` one a position, its first the first's), read off the probe, which is
  refused unless each divides evenly;
* a head over T keys is its cut's Calls (`head`, `window_circuit.CUTS`: `AttentionHead_v21` and FA3's `_v22`), nb =
  ceil(T / BN) key blocks each its softmax step and its P·V groups of PK keys: the first visited block's
  `AttnBlockSoftmax_v1{NVIS, FIRST=true}` (its NVIS = T - (nb - 1)·BN keys), each block below it the cut's softmax step
  (`key_blocks`' ``later``) at BN keys, its CHECK whether the kernel guards it (``MASKED_FROM``,
  `attention_vocab.fa2_masked_from` of the Program's one-row decode: its first visited block only), each block's
  `AttnBlockPv_v1{N, VS}` groups (`fa2_check_inf.pv_groups`), and `AttnHeadEnd_v1`. The probe's heads are held to it.

`size` prices a census with the ANDs and k_log a unit at Llama-3.2-1B (`measured`, each with its source, scaled or
bounded where only another binding was measured, and saying so), and m = ceil(log2 Σ units · 2^k_log), refusing a unit
past a 2^27-bit block as C-Flock refuses it, and refusing every unit nothing measured, naming each: the cut's softmax
steps and P·V groups have no measurement at Llama-3.2-1B yet. A unit composed of few ANDs takes its rows' k_log
(`rows_k`: C-Flock's composer gives each row a unit reads or writes an hm96 slot).

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
from verity_vllm.program.predict import window_circuit as WC

from verity.ml.definitions.registry.attention_vocab import fa2_masked_from
from verity.ml.definitions.registry.fa2_check_inf import pv_groups

SOFTMAX = "AttnBlockSoftmax_v1"
PV = "AttnBlockPv_v1"
END = "AttnHeadEnd_v1"
#: the softmax step of a head's blocks below its first, by its Attention's cut's version (`window_circuit.CUTS`): FA2's
#: v21 `AttnBlockSoftmax_v1`, FA3's v22 `_v2` (FA3's guard)
LATER = {21: SOFTMAX, 22: "AttnBlockSoftmax_v2"}
FEED = "TokenFeed_v1"
SCAN = "ScanSlots_v2"
COLLECT = "OutCollect_v1"
ENTRY = "SlotEntry_v1"


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
    prompt positions past the first, `ScanSlots_v2` at the first, `SlotEntry_v1` at the last and `OutCollect_v1` at each
    (its first the first's); None for any other, a fixed count at every position."""
    fam, s = statics(lab)
    if fam == FEED:
        return int(1 < T <= s["NP"])
    if fam == SCAN:
        return int(T == 1)
    if fam == ENTRY:
        return int(T == P)
    if fam == COLLECT:
        return int(T == 1) if s["FIRST"] else int(T > 1)
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


def key_blocks(registered: Any, at: Any) -> dict[str, Any]:
    """The class Program's attentions at class ``at`` as `window_circuit.CUTS` cuts them: ``BN``, a key block's keys, and
    ``later``, the softmax step of a head's blocks below its first (`head`'s), refused unless one of each."""
    att = {(n.fn.bindings["BN"], n.fn.version) for n in WC.class_program(registered, at).program.fn.body.nodes
           if n.fn.name == "Attention"}
    if len(att) != 1 or next(iter(att))[1] not in WC.CUTS:
        raise ValueError(f"the class Program's attentions are not one cut version at one key block (BN, version): {sorted(att)}")
    (BN, version), = att
    return {"BN": BN, "later": LATER[WC.CUTS[version].version]}


def head(T: int, D: int, BN: int, PK: int, VS: int, later: str = SOFTMAX) -> Counter[str]:
    """A head's units over T keys by label, its cut's Calls (`dead_free.cut_head`): each key block, its first visited first,
    its softmax step and its P·V groups of PK keys, values as rows of VS words, and its end."""
    nb, mf, out = -(-T // BN), fa2_masked_from(T, 1, 0, BN, False), Counter()
    for b in range(nb - 1, -1, -1):
        first = b == nb - 1
        nvis = T - b * BN if first else BN
        out[f"{SOFTMAX if first else later}{{D={D},NVIS={nvis},FIRST={json.dumps(first)},"
            f"CHECK={json.dumps(first or b >= mf)},PK={PK}}}"] += 1
        for g, (lo, hi) in enumerate(pv_groups(nvis, PK)):
            out[f"{PV}{{D={D},N={hi - lo},VS={VS},FIRST={json.dumps(first and g == 0)}}}"] += 1
    out[f"{END}{{D={D}}}"] += 1
    return out


def heads_of(units: Mapping[str, Any]) -> dict[str, Any]:
    """The labels of ``units`` (a census's) of an attention's family, and their D, PK and VS, refused unless one of each."""
    att = {lab: n for lab, n in units.items() if lab.startswith(("AttnBlock", "AttnHead"))}
    ss = [statics(lab)[1] for lab in att]
    return {"labels": att, **{k: _one([s[k] for s in ss if k in s], k) for k in ("D", "PK", "VS")}}


def project(probe: Mapping[str, int], at: Any, cls: Any, BN: int, later: str = SOFTMAX) -> dict[str, int]:
    """One Program's census at class ``cls`` from one Program's ``probe`` census at class ``at`` (`census` of
    ``class_circuit(registered, at, 1)``), its heads at key blocks of ``BN`` and each block below a head's first the
    softmax step ``later`` (`key_blocks`), refused unless the probe's are `head`'s at each of its positions."""
    P0, P, NP0 = at.positions, cls.positions, at.prompt
    if P0 < 2 or NP0 < 2:
        raise ValueError(f"a probe of {P0} positions and {NP0} prompt ids: it needs 2 of each, for a head and a feed past the "
                         "first")
    hs = heads_of(probe)
    D, PK, VS = hs["D"], hs["PK"], hs["VS"]
    end = f"{END}{{D={D}}}"
    H = probe.get(end, 0) // P0
    want: Counter[str] = Counter()
    for T in range(1, P0 + 1):
        want.update({lab: H * k for lab, k in head(T, D, BN, PK, VS, later).items()})
    if not H or dict(want) != hs["labels"]:
        raise ValueError(f"the probe's heads are not {H} a position of the cut at D {D}, BN {BN}, PK {PK} and VS {VS} at each "
                         f"of its {P0} positions ({probe.get(end, 0)} {end} units): {hs['labels']}")
    out: Counter[str] = Counter()
    for lab, n in probe.items():
        if lab not in want:
            to, k = projected(lab, n, at, cls)
            out[to] += k
    for T in range(1, P + 1):
        out.update({lab: H * k for lab, k in head(T, D, BN, PK, VS, later).items()})
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
    "ScanSlots_v2{NS=9353,NP=1024,IDB=17,K0=12,K=45}": (120, 24, None),
    "OutCollect_v1{KO=64,FIRST=true}": (155, 22, ROWS_3),
    "OutCollect_v1{KO=64,FIRST=false}": (2_141, 22, ROWS_3),
    "SlotEntry_v1{KO=64,IDB=17,NP=1024}": (5_676, 22, None),
}.items()}
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
