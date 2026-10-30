"""The conformance record of the registered library, as data.

    verity-vllm conformance [--out record.json]

One entry per registered definition (`Name_vN`): its status (`exact-model-tested` | `model-untested` |
`not-yet-established` | `superseded`), the free-text qualification attached to the status, and — for the
definitions with recorded production divergences or derived Inputs — a structured `qualifications` list. The
Input-gate qualifications (the `cos_sin_cache` table) are recorded here too, so that no
summary of the B1/B0 programs can be produced from this record without them.

`tests/program/test_conformance_record.py` holds the record to the registry, and `tests/program/test_nan_conversion.py`
holds its NaN entry to the L8 NaN store scan (the data directory that test names) and the measured conversion words.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition
from verity_vllm.config import option
from verity.ml.definitions.registry import b1, prims  # noqa: F401  (registration)

STATUSES = ("exact-model-tested", "model-untested", "not-yet-established", "superseded",
            # kinds of `ref_prims.py` whose semantics IS the evaluator's arithmetic;
            # "exact" against torch CPU only over the tested domain recorded in `docs/data/ref-prims/`.
            "reference-defined")

# Structured qualifications on Input gates: which committed leaves are *not* checkpoint tensors and what their
# registered source is. Counts are the correspondence records' (`corr-*.json`, `input_gate_records.weights`).
INPUT_QUALIFICATIONS = [
    {"input": "weights.cos_sin", "leaf_id": "w=layers.0.self_attn.rotary_emb.cos_sin_cache", "kind": "derived-table",
     "rebuilt_from": "vllm._compute_cos_sin_cache (CUDA cosf/sinf), GPU only",
     "cpu_rebuild": {"B1": {"differing_words": 17261, "total_words": 16777216}, "B0": {"differing_words": 6, "total_words": 524288}},
     "consumers": ["RoPEHead_v1", "RoPE_v1"],
     "statement": "the Input gates of this table are 'rebuilt from the registered source' only if that source is the CUDA computation"},
]

# Recorded production divergences, keyed by definition id.  The B0 store sweep is `verity-vllm rmsnorm-fused-sweep` over
# the B0 bundle (`rmsnorm_fused_sweep.BUNDLE`), which neither the tree nor the evidence store keeps.
DIVERGENCES = {
    "RMSNormFusedCuda_v1": [
        {"where": "B1 t=1/layers.27/post_attention_layernorm.rmsnorm/port=0, word 629 of 1536", "ir": "0x3f04", "captured": "0x3f03",
         "words_compared": 218112, "mismatching_words": 1,
         "cause": "models fused_add_rms_norm_kernel<bf16, 8>; production (VLLM_BATCH_INVARIANT=1) launches the generic kernel "
                  "(different per-thread partials and BlockReduce over 1024 threads) — see RMSNormFusedCuda_v2",
         "b0_store_sweep": {"rows": 2340, "rows_exact": 2327, "record": "not kept (the B0 bundle is not in the evidence store)"}},
    ],
    "RMSNormFusedCuda_v2": [
        {"where": "B1 t=1/layers.27/post_attention_layernorm.rmsnorm/port=0, word 629 of 1536",
         "status": "not re-checked locally (the B1 store is on the pod); the B0 store sweep reproduces all 2,340 rows incl. the 13 v1 misses",
         "b0_store_sweep": {"rows": 2340, "rows_exact": 2340, "record": "not kept (the B0 bundle is not in the evidence store)"}},
    ],
}

# The f32 -> bf16 conversion the production programs bind and its NaN rule. `F32ToBf16Rn_v1` (NaN -> 0x7FC0,
# the software canonicalisation and `veritor.tc-ampere-bf16@1`'s gate) stays registered and unchanged; B0/B1 bind `F2fpBf16_v1` (NaN -> 0x7FFF, the
# measured `cvt.rn.bf16.f32` word; `veritor.tc-ampere-bf16@2`'s gate). The two are equal on every non-NaN word, so a
# recorded trace without NaN words at conversion outputs is evidence about either binding.
NAN_CANONICALISATION = {
    "production_conversion": "F2fpBf16_v1",
    "nan_word": "0x7FFF",
    "nan_rule": "every FP32 NaN (either sign, quiet or signalling, any payload) -> 0x7FFF; sign and payload dropped",
    "source": "measured: verity/ml/circuits/ops/tests/fixtures/tc-total-2026-09-07/{a100,4090}/cvt_special.json -- 84 NaN inputs per device "
              "(12 +qNaN, 12 -qNaN, 34 +sNaN, 26 -sNaN), all 0x7FFF on both devices; not an assumption",
    "not_bound": {"F32ToBf16Rn_v1": "NaN -> 0x7FC0 (software canonical); registered, unchanged, not bound by B0/B1"},
    "interface_family": {"F2fpBf16_v1": "veritor.tc-ampere-bf16@2 f32_to_bf16", "F32ToBf16Rn_v1": "veritor.tc-ampere-bf16@1 f32_to_bf16"},
    "narrow_delta": {"python": "tests/program/test_nan_conversion.py test_the_two_registered_conversions_are_the_two_family_versions "
                               "(20,024 words: 12 NaN, 12 finite edge words, 20,000 random)",
                     "not_in_this_repository": "the exhaustive 2^32 sweeps that first showed it (a Lean statement, a Rust check "
                                               "and the predecessor implementation's Python sweep)"},
    "program_digests_unchanged": {"B0": "03405bbdc7ec2825abd79aad723a2c75a417e8b3084567f89cf3341e31a5ab2d",
                                  "B1": "11f3b7b6092d8020821f2388d91b029b2ad033a925bf2682bd918720d6e61097"},
    "recorded_traces_finite": {"B0": {"bf16_leaves": 14126, "bf16_words": 12568704, "nan_words": 0, "inf_words": 0},
                               "B1": {"bf16_leaves": 271278, "bf16_words": 916168704, "nan_words": 0, "inf_words": 0},
                               "record": "integrations/vllm/data/l8-nan-scan-2026-09-07/{B0,B1}.json (python -m verity_vllm.check.store_nan_scan)"},
}


def status_of(conformance: str) -> str:
    head = conformance.split(";")[0].split(":")[0].strip()
    for s in STATUSES:
        if head.startswith(s):
            return s
    return "not-yet-established"


def record() -> dict:
    defs = {}
    for fid, d in sorted(REGISTRY.defs.items()):
        if getattr(d, "family", None) == "Const":
            continue
        entry = {"kind": "primitive" if isinstance(d, PrimitiveDefinition) else "composite",
                 "status": status_of(d.conformance), "conformance": d.conformance}
        if isinstance(d, CompositeDefinition):
            entry["statics"] = list(d.statics)
        if fid in DIVERGENCES:
            entry["divergences"] = DIVERGENCES[fid]
        quals = [q for q in INPUT_QUALIFICATIONS if fid in q["consumers"]]
        if quals:
            entry["input_qualifications"] = [q["input"] for q in quals]
        defs[fid] = entry
    by_status: dict[str, int] = {}
    for e in defs.values():
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    return {"schema": "verity-ir/conformance/v0", "definitions": defs, "input_qualifications": INPUT_QUALIFICATIONS,
            "nan_canonicalisation": NAN_CANONICALISATION,
            "counts": by_status,
            "summary": ("not universally exact: " + "; ".join(
                [f"{k}: {v}" for k, v in sorted(by_status.items())]
                + [f"Input qualification on {q['input']} ({q['rebuilt_from']})" for q in INPUT_QUALIFICATIONS]
                + [f"divergence recorded on {fid}" for fid in DIVERGENCES if any('captured' in x for x in DIVERGENCES[fid])]))}


@dataclass(frozen=True, kw_only=True)
class Options:
    out: str | None = option("--out", default=None)


def main(a: Options) -> int:
    rec = record()
    text = json.dumps(rec, indent=1)
    if a.out:
        open(a.out, "w").write(text)
    else:
        print(text)
    return 0
