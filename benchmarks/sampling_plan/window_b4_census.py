"""The served window's program as the planner reads it: the remote demo's B4 window, class 4,32,4 (`served_verdicts.sh`'s
`V3C_CLASS`), lowered wired at the served grain on a tree that serves it (`TREE`, the V3 tree: main's wired lowering at
4,32,4 refuses its readers), and written as a census, one entry a template of the class circuit's population
(`population.mixed_population_program`, a unit a template instance, as `one_stage/a5.py` and `serving_rows.py` register it).

Each entry: its units, its input rows (bytes, under the circuit's cuts) and output rows, its output bits, its H200 ANDs
(`window_cost.unit_ands`; an estimate, so marked, where it prices none) and its serve-time commit in SHA-512 compressions
(`class_rows`: an output row b bytes costs blocks(128 + b) + 6, an input row 4). The three linear families (the int8
coordinates at K 2048 and 8192, the bf16 GEMM at K 2048) also carry their alternative grains: the same Definition rebound
at N output channels a unit, its rows from its own layout, the coordinates conserved. The wired lowering refuses every
`Grain.gemm_words` but the served 32,768 at 4,32,4 (`refusals`), so the alternatives are the template's, its readers taken
as re-cut to match.

    TREE=/path/to/v3tree PYTHONPATH=$TREE:$TREE/integrations/vllm python benchmarks/sampling_plan/window_b4_census.py OUT.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

T = os.environ.get("TREE", str(Path(__file__).resolve().parents[2]))
sys.path[:0] = [T, T + "/integrations/vllm", T + "/benchmarks/prove_path"]

import class_rows as CR  # noqa: E402
import class_size as CS  # noqa: E402
import window_cost as WCO  # noqa: E402
from tests.program.predict import test_steps as S  # noqa: E402
from verity_vllm.program.model import INT8_EXACT_QUANT  # noqa: E402
from verity_vllm.program.predict import window_circuit as WC  # noqa: E402
from verity_vllm.program.predict.window_builder import Grain  # noqa: E402
from verity_vllm.program.predict.window_circuit import WindowClass  # noqa: E402

from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.windows import interface as IF  # noqa: E402

CLASS = (4, 32, 4)
SERVED_GRAIN = 32_768
GRAINS = (2048, 4096, 8192, 16_384, 65_536, 131_072)
LINEARS = ("ScaledMmS8_v1", "Gemm_v2")
#: N a unit, the alternatives a linear family is offered
WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128)
H200 = {"compute_capability": [9, 0], "num_sms": 132}
_COMBINE = CS.MEASURED["TokenSelectCombine_v2{V=128256,C=16}"][0]
#: ANDs for the templates `window_cost` prices none of, each an estimate from `class_size`'s measured pieces
ESTIMATES = {
    "SiluMul_v2{I=2048}": (2048 * 1380, "SiluMul_v2{I=8192}'s 1,380 ANDs an element, at I = 2048"),
    "RMSNormFusedCudaAdd_v1": (193_024, "one of RMSNormFusedCuda_v2{N=2048}'s eight 256-element add-and-square slices"),
    "RMSNormFusedCudaTail_v1{N=2048}": (WCO.CS.measured("RMSNormFusedCuda_v2{N=2048}")[0] - 8 * 193_024,
                                        "RMSNormFusedCuda_v2{N=2048} less its eight slices"),
    "SelectCollect_v1": (_COMBINE + CS.MEASURED["OutCollect_v1{KO=64,FIRST=false}"][0],
                         "TokenSelectCombine_v2 and OutCollect_v1 at KO 64 (an upper bound at KO 4)"),
    "SelectEntry_v1": (_COMBINE + CS.MEASURED["OutCollect_v1{KO=64,FIRST=false}"][0]
                       + CS.MEASURED["SlotEntry_v1{KO=64,IDB=17,NP=1024}"][0],
                       "TokenSelectCombine_v2, OutCollect_v1 and SlotEntry_v1 at the class of record's KO and NP"),
    "ScanSlots_v2": (120, "its Boolean compile's AND count (`bits.compile`)"),
}


def registration():
    config = {**json.loads(Path(T + "/integrations/vllm/tests/program/predict/fixtures/llama32-1b.config.json")
                           .read_text()), "quantization_config": dict(INT8_EXACT_QUANT)}
    return S.Registered.of(S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": config},
                                 target={**H200, "int8_exact": "linears"}), default_seed=7)


def ands_of(lab: str) -> tuple[int, str]:
    try:
        return WCO.unit_ands(lab, "h200"), ""
    except Exception:  # noqa: BLE001 - window_cost refuses what it doesn't price
        pass
    for key, (a, why) in ESTIMATES.items():
        if lab == key or lab.split("{")[0] == key:
            return a, f"estimate: {why}"
    raise KeyError(f"{lab}: neither window_cost nor ESTIMATES prices it")


def entry(fn, n: int, cut) -> dict:
    lab = CS.label(fn)
    lay = IF.layout(fn)
    ins = [2 * w for _, w, _ in lay.cut(cut)]
    outs = [2 * w for _, w in lay.outs]
    ands, why = ands_of(lab)
    _, ret = fn.signature()
    fresh = sum(CR.blocks(128 + b) + CR.FRESH_EXTRA for b in outs)
    return {"label": lab, "id": fn.id, "units": n, "in": ins, "out": outs, "out_bits": sum(ret.leaf_widths()),
            "ands": ands, "ands_source": why or "window_cost.unit_ands", "commit_compressions": fresh + CR.REUSED * len(ins)}


def main(out: str) -> None:
    t0 = time.time()
    reg = registration()
    refusals = {}
    for g in GRAINS:
        try:
            WC.lower(reg, WindowClass(*CLASS), grain=Grain(gemm_words=g), wired=True)
            refusals[g] = None
        except Exception as e:  # noqa: BLE001 - the lowering's refusal is the finding
            refusals[g] = f"{type(e).__name__}: {e}"[:600]
    c = WC.lower(reg, WindowClass(*CLASS), grain=Grain(gemm_words=SERVED_GRAIN), wired=True).circuit
    rows = []
    for fn, n in c.templates:
        e = entry(fn, n, c.cuts.get(IF.descriptor_id(fn)))
        if fn.id.split("{")[0] in LINEARS:
            N = fn.bindings["N"]
            alts = []
            for w in WIDTHS:
                if (n * N) % w:
                    continue
                try:
                    alt = bind(fn, N=w)
                    a = entry(alt, n * N // w, None)
                except Exception as err:  # noqa: BLE001 - a width the Definition refuses
                    a = {"N": w, "refused": f"{type(err).__name__}: {err}"[:300]}
                alts.append({"N": w, **a})
            e["N"], e["alternatives"] = N, alts
        rows.append(e)
    doc = {"class": CLASS, "grain": SERVED_GRAIN, "tree": T, "units": sum(r["units"] for r in rows), "templates": rows,
           "refusals": refusals, "seconds": round(time.time() - t0, 1)}
    Path(out).write_text(json.dumps(doc, indent=1))
    print("units", doc["units"], "templates", len(rows), "estimated",
          sorted({r["label"] for r in rows if r["ands_source"].startswith("estimate")}))


if __name__ == "__main__":
    main(sys.argv[1])
