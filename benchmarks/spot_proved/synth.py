"""Synthetic a11 saved values for TP8-shaped Definitions (random benign inputs, outputs by the Definition): a timing and
smoke run of a11 --load-values before served values exist. Not served values; never reported as such.

    python synth.py --out DIR [--n 2] [--labels L1,L2]
"""
import argparse
import json
import sys
from pathlib import Path

TREE = Path.cwd()
for _p in ("benchmarks/one_stage", "tools/verity/circuit_check/src", "benchmarks/numerical/python", "integrations/vllm",
           "verity/ml/flock/python", "."):
    sys.path.insert(0, str(TREE / _p))

import numpy as np  # noqa: E402

TREES = None


def blocks():
    from verity.ml.definitions.registry import tp_allreduce as AR
    trees = json.dumps([AR.GLM53_H200_TREE] * 8, separators=(",", ":"))
    return {
        "SparseMlaSplitPart_T64": 'SparseMlaSplitPart_v1{T=64,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},'
                                  'INV={"fn":"Fa3InvSum_v1"},SCALE={"f64":"0x1.0000000000000p-4"}}',
        "AllReduceMultimem_W8": f"AllReduceMultimem_v1{{WORLD=8,SIZE=3072,TREES={trees},FTZ=false}}",
        "RMSNormFusedCudaVector8_N6144_B256": 'RMSNormFusedCudaVector8_v3{N=6144,EPS={"f64":"0x1.4f8b588e368f1p-17"},BLOCK=256}',
        "Fp8GroupQuantUe8m0_K1536": "Fp8GroupQuantUe8m0_v1{K=1536,G=128}",
        "MoeExpertFp8GemmW_K256_SWAP": 'MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v1"}}',
        "MoeExpertFp8GemmW_K256_MMA": 'MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v1"}}',
    }


def e4m3(rng, n):
    b = rng.integers(0, 0x7F, n, dtype=np.uint64)
    return b | (rng.integers(0, 2, n, dtype=np.uint64) << 7)


def f32_words(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)


def bf16(rng, n, scale):
    f = (rng.standard_normal(n) * scale).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint64)


def args_for(label, rng):
    if label.startswith("SparseMlaSplitPart"):
        return [bf16(rng, 512, 0.5), bf16(rng, 64, 0.5), bf16(rng, 64 * 576, 0.5)]
    if label.startswith("AllReduceMultimem"):
        return [bf16(rng, 8 * 3072, 1.0)]
    if label.startswith("RMSNormFused"):
        return [bf16(rng, 6144, 1.0), bf16(rng, 6144, 1.0), bf16(rng, 6144, 0.2)]
    if label.startswith("Fp8GroupQuant"):
        return [bf16(rng, 1536, 1.0)]
    if label.startswith("MoeExpertFp8GemmW"):
        E, K, G = 256, 256, 128
        return [e4m3(rng, K), f32_words(2.0 ** rng.integers(-9, -5, K // G)), np.array([rng.integers(0, E)], dtype=np.uint64),
                f32_words([0.25]), e4m3(rng, E * K), f32_words(2.0 ** rng.integers(-12, -8, E * (K // G)))]
    raise KeyError(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    import a11
    import a11_kinds as K
    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity.core.protocols.spot.one_stage import partition as P
    from verity_vllm.pipeline import serving_rows as SRW

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    want = set(a.labels.split(",")) if a.labels else None
    staged, kinds, units = {}, {}, {}
    for label, block in blocks().items():
        if want and label not in want:
            continue
        spec = SRW.template(block).fn
        rows_spec = K.rows_of(spec)
        ins, outs = [], []
        for _ in range(a.n):
            ev = a11.evaluate_units(spec, spec, args_for(label, rng), 1, 4_000_000)
            ins += ev["in"]
            outs += ev["out"]
        rows, routs = a11.unit_rows(spec, rows_spec, ins, outs)
        np.savez(out / f"{label}.npz", **{f"i.{r}": np.asarray(v, dtype=np.uint16) for r, v in rows.items()},
                 **{f"o.{o}": np.asarray(v, dtype=np.uint16) for o, v in routs.items()})
        units[spec.id] = spec
        staged[label] = {"staged": True, "unit": spec.id, "n": a.n, "name": label, "step": 0, "event": [0, 0, 0],
                         "calls": [], "twin": {"agree": True, "calls": a.n, "compared": 0}}
        kinds[label] = {"gates": spec.gates, "outcome": "lowers"}
        print(f"synth {label}: {a.n} units of {spec.id[:90]}", flush=True)
    (out / "units.json").write_bytes(canonical_json(encode_program(P.mixed_population_program([(u, 1) for u in units.values()]))))
    (out / "values.json").write_text(json.dumps({"staged": staged, "weights": {"from": "synthetic"}, "values": "synthetic",
                                                 "steps": "0", "layers": "", "per_kind": a.n}, indent=1, sort_keys=True))
    (out / "kinds.json").write_text(json.dumps({"kinds": kinds}, indent=1, sort_keys=True))
    (out / "window.json").write_text(json.dumps({"of": "synthetic random inputs (a timing run, not served values)",
                                                 "link": "none", "row": "synthetic", "root": None}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
