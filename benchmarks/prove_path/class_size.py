"""The padded size of a window class's circuit (V3b), at a class too large to lower: its units, ANDs and m.

`window_circuit.class_circuit` lowers the class Program once a request. At 2a's class (4 slots, 2,048 prompt ids, 64 outputs:
2,111 positions a Program) a head reaches 2,111 keys, past the `window_circuit.KEYS` one gather reads until
`AttentionHead_v19` (#4263) binds, and a Program is some 94M units. So the size is a projection (`project`) from an exact
lowering of one Program of the same registration at a small probe class:

* every template but the heads and their key gathers is a fixed number of units a position (`TokenFeed_v1` a prompt
  position), read off the probe, which is refused unless each divides evenly;
* the heads are v19's: a head over T keys is `AttnBlockFirst_v1{NVIS}` (its last NVIS = T - (nb - 1)·BN keys), nb - 1
  `AttnBlock_v5` and `AttnHeadEnd_v1`, nb = ceil(T / BN). Its keys and values reach it as an `AllGather_v1{BN, D}` row a
  full block, written once and read by every later position, and an `AllGather_v1{NVIS, D}` of its first block, none where
  that block is one key (RoPE's row) or a full one. ``heads="v18"`` projects today's lowering instead (a head a unit over
  an `AllGather_v1{T, D}` of its keys), which the tests hold to an exact lowering at another class.

`size` prices a census with the ANDs and k_log a unit at Llama-3.2-1B (`measured`, each with its source, scaled or
interpolated where only another binding was measured, and saying so), and m = ceil(log2 Σ units · 2^k_log).

    prove_path.py class-size --serve-run DIR --window K --row-dir DIR --advice FILE [--class 4,2048,64] [--probe 1,2,2]
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

HEAD = "AttentionHead_v18"
FEED = "TokenFeed_v1"
GATHER = "AllGather_v1"


def label(fn: Any) -> str:
    """A template's family and its integer and bool statics, the key a census counts it under."""
    return fn.id.split("{")[0] + "{" + ",".join(f"{k}={json.dumps(v)}" for k, v in fn.bindings.items() if type(v) in (int, bool)) + "}"


def statics(lab: str) -> tuple[str, dict[str, Any]]:
    fam, _, body = lab.partition("{")
    return fam, {k: json.loads(v) for k, v in (kv.split("=") for kv in body.rstrip("}").split(",") if kv)}


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


def project(probe: Mapping[str, int], at: Any, cls: Any, heads: str = "v19") -> dict[str, int]:
    """One Program's census at class ``cls`` from one Program's ``probe`` census at class ``at`` (`census` of
    ``class_circuit(registered, at, 1)``), its heads ``heads``'s."""
    if heads not in ("v18", "v19"):
        raise ValueError(f"heads are v18's or v19's, not {heads!r}")
    P0, P, NP0 = at.positions, cls.positions, at.prompt
    if P0 < 3 or NP0 < 2:
        raise ValueError(f"a probe of {P0} positions and {NP0} prompt ids: it needs 3 and 2, to read a key gather twice and a "
                         "feed past the first")
    hs = {lab: n for lab, n in probe.items() if lab.startswith(HEAD + "{")}
    ss = [statics(lab)[1] for lab in hs]
    D, BN, MF = (_one([s[k] for s in ss], k) for k in ("D", "BN", "MASKED_FROM"))
    by_t = {s["T"]: hs[lab] for lab, s in zip(hs, ss)}
    if sorted(by_t) != list(range(1, P0 + 1)) or len(set(by_t.values())) != 1:
        raise ValueError(f"the probe's heads are not one count at each T of 1..{P0}: {by_t}")
    H, G = by_t[1], probe.get(f"{GATHER}{{WORLD=2,N={D}}}", 0)
    rest = Counter({lab: n for lab, n in probe.items() if lab not in hs})
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
        elif n % P0:
            raise ValueError(f"the probe has {n} {lab} units, not a fixed count at each of its {P0} positions")
        elif n:
            out[lab] += n // P0 * P
    for T in range(1, P + 1):
        if heads == "v18":
            out[f"{HEAD}{{T={T},D={D},BN={BN},MASKED_FROM={MF}}}"] += H
            out[f"{GATHER}{{WORLD={T},N={D}}}"] += G if T > 1 else 0
            continue
        nb = -(-T // BN)
        nvis = T - (nb - 1) * BN
        out[f"AttnBlockFirst_v1{{D={D},NVIS={nvis},BN={BN}}}"] += H
        for b in range(nb - 1):
            out[f"AttnBlock_v5{{D={D},NVIS={BN},FIRST=false,BN={BN},CHECK={json.dumps(b >= MF)}}}"] += H
        out[f"AttnHeadEnd_v1{{D={D}}}"] += H
        out[f"{GATHER}{{WORLD={nvis},N={D}}}"] += G if 1 < nvis < BN else 0
    if heads == "v19":
        out[f"{GATHER}{{WORLD={BN},N={D}}}"] += G * (P // BN)
    return {lab: n for lab, n in sorted(out.items()) if n}


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
    "RoPEHead_v1{D=64}": (1_534_976 // 8, 22, "RoPE_v1{NHEADS=8,D=64}'s / 8, r20261025-072237-b8e3"),
    "SiluMul_v2{I=8192}": (8192 * 1380, 26, "8,192 × SiluMulBf16_v4's 1,380 (circuits-bool-silu)"),
    "TokenSelectChunk_v2{C=768}": (12 * 7_347, 22, "scaled: 12 × TokenSelectChunk_v2{C=64}'s, r20261025-082521-1daa"),
    "TokenSelectCombine_v2{V=128256,C=768}": (round(332_633 * 167 / 2004), 27, "scaled: TokenSelectCombine_v2{V=128256,C=64}'s "
                                              "over 2,004 chunks to 167, r20261025-082521-1daa"),
    "AttnHeadEnd_v1{D=64}": (169_842, 22, "#4263, r20261025-123346-091b"),
    "AttnBlock_v5{D=64,NVIS=128,FIRST=false,BN=128,CHECK=false}": (9_281_000, 26, "#4263, r20261025-123346-091b"),
    "AttnBlock_v5{D=64,NVIS=128,FIRST=false,BN=128,CHECK=true}": (9_281_000, 26, "CHECK=false's (#4263, r20261025-123346-091b); "
                                                                   "the mask's compares unmeasured"),
}
#: `AttnBlockFirst_v1{D=64,NVIS,BN=128}` as #4263 measured it (r20261025-123346-091b): NVIS -> (ANDs, k_log)
FIRST_BLOCKS = {1: (96_886, 22), 17: (1_231_456, 24), 128: (9_075_919, 26)}
#: `AllGather_v1{WORLD,N}`'s k_log where CIR-95 measured it; a copy, no ANDs. Key gathers `{T,64}` (T <= 136) take 27, the
#: block CIR-95 measured at T = 136, an upper bound below it
GATHERS = {(4, 16): 22, (48, 16): 25, (128, 16): 27, (128, 4): 27, (167, 1): 27}


def measured(lab: str) -> tuple[int, int, str]:
    """(ANDs, k_log, source) of a unit ``lab`` at Llama-3.2-1B, refused where nothing measured it or a binding it scales from."""
    if lab in MEASURED:
        return MEASURED[lab]
    fam, s = statics(lab)
    if fam == GATHER and ((s["WORLD"], s["N"]) in GATHERS or (s["N"] == 64 and 2 <= s["WORLD"] <= 136)):
        return 0, GATHERS.get((s["WORLD"], s["N"]), 27), "a copy (b1_tp2); k_log CIR-95's"
    if fam == "AttnBlockFirst_v1" and (s["D"], s["BN"]) == (64, 128) and 1 <= s["NVIS"] <= 128:
        v, (lo, hi) = s["NVIS"], next((a, b) for a, b in zip(FIRST_BLOCKS, list(FIRST_BLOCKS)[1:]) if s["NVIS"] <= b)
        ands = round(FIRST_BLOCKS[lo][0] + (FIRST_BLOCKS[hi][0] - FIRST_BLOCKS[lo][0]) * max(v - lo, 0) / (hi - lo))
        k = FIRST_BLOCKS[1][1] if v == 1 else FIRST_BLOCKS[hi][1]
        return ands, k, "#4263 (r20261025-123346-091b)" + ("" if v in FIRST_BLOCKS else ": interpolated, k_log the next measured")
    if fam == FEED:
        return 0, 22, "unmeasured (a 32-bit compare, select and add): k_log the floor the smallest measured units take"
    raise ValueError(f"no measured ANDs or k_log for {lab} at Llama-3.2-1B: add its measurement to MEASURED")


def size(units: Mapping[str, int], times: int = 1) -> dict[str, Any]:
    """``units`` (a census) ``times`` over, priced by `measured`: units, ANDs, m and each template's row."""
    rows = [(lab, n * times, *measured(lab)) for lab, n in units.items()]
    cells = sum(n << k for _, n, _, k, _ in rows)
    return {"units": sum(n for _, n, *_ in rows), "ands": sum(n * a for _, n, a, _, _ in rows), "m": math.ceil(math.log2(cells)),
            "templates": [{"template": lab, "units": n, "ands": a, "k_log": k, "source": src}
                          for lab, n, a, k, src in sorted(rows, key=lambda r: -r[1] * r[2])]}
