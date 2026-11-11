"""GLM-5.3's per-expert fused cut at TP8 as a serving-rows partition (`SERVING_ROWS`, `verity-vllm tp-serving-rows --partition`):
each of a MoE layer's expert GEMM pair rows (w13 at K = 6144, N = 512; w2 at K = 256, N = 6144; each on the two kernels a
TP8 H200 rank serves them with) cut into fused blocks that read their rows from the registered tables
(`serving_templates.fused_block`: w13 a row a block, w2 32), and the reads of each projection's scale rows, each template's
members at most the units its statement holds, over one layer's rows (``scope.op_path_prefix``).  The router's
slots stay out of the window, so each fused block's key and routed weight are free ports (`expert_rows.wire`).

    uv run benchmarks/one_stage/glm53_fused_tp8.py --out glm53_fused_tp8.json [--layer 40] [--steps LO HI] [--max-units C]

``--max-units C`` holds each template's members to at most C units where its statement holds more: a window of more, smaller
members, as a prover's timing reads members of several sizes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

E, G, KB = 256, 128, 128
PAIRS = {"w13": ("MoeExpertFp8Gemm_v1", 6144, 512, 1), "w2": ("MoeExpertFp8GemmW_v1", 256, 6144, 32)}
FUSED = {"w13": "MoeExpertFp8ReadGemm_v1", "w2": "MoeExpertFp8ReadGemmW_v1"}
KERNELS = ((True, "HopperE4m3QgmmaDot32_v1"), (False, "HopperE4m3MmaV2Dot32_v1"))


def row(proj: str, swap: bool, dot: str) -> str:
    """Pair row ``proj``'s spec on one kernel, as the Build's Programs name it."""
    fam, K, N, _n = PAIRS[proj]
    return f'{fam}{{E={E},K={K},N={N},G={G},KB={KB},SWAP={str(swap).lower()},DOT={{"fn":"{dot}"}}}}'


def templates() -> list[str]:
    """Each projection's scale-row read, then its fused blocks on each kernel: of the pair row's unit templates
    (`serving_rows.unit_templates`), its fused block of n rows."""
    from verity.circuits.codec import descriptor_id
    from verity.commitments.gates.embedding import read
    from verity.definitions.call_scope import spec_base, spec_statics
    from verity_vllm.pipeline import serving_rows as SRW

    out = []
    for proj, (_fam, K, _N, n) in PAIRS.items():
        out.append(descriptor_id(read(E, 2 * K // G)))
        for swap, dot in KERNELS:
            fused = [t for t, _k in SRW.unit_templates(row(proj, swap, dot)) or ()
                     if spec_base(t) == FUSED[proj] and int(spec_statics(t)["N"]) == n]
            if len(fused) != 1:
                raise SystemExit(f"{row(proj, swap, dot)}: {len(fused)} fused blocks of {n} rows bind, not one")
            out.append(fused[0])
    return out


def main() -> None:
    from verity_vllm.pipeline import serving_rows as SRW

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--steps", type=int, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--max-units", type=int, default=None)
    a = ap.parse_args()
    tmpls = templates()
    doc = {"templates": tmpls, "law": "stratified:1",
           "scope": {"op_path_prefix": f"moe/L{a.layer}", **({"steps": a.steps} if a.steps else {})}}
    doc["chunk_cap"] = {t: min(c, a.max_units or c) for t, c in SRW.chunk_caps(tmpls).items()}
    Path(a.out).write_text(json.dumps(doc) + "\n")
    print(json.dumps(doc["chunk_cap"], indent=1))


if __name__ == "__main__":
    main()
