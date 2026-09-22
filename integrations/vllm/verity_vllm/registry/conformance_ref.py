"""Reference-vocabulary conformance record (lane NUMERIC): the `ref_prims.py` manifest + digest joined with the tested-domain
records written by `tests/ir/test_ref_prims.py` under `docs/data/ref-prims/`.

    PYTHONPATH=. python -m verity_vllm.registry.conformance_ref [--records docs/data/ref-prims] [--out FILE]

Every entry is labelled "conformance over tested domain, not universal correctness" (brief boundary 5).  A kind with no
record is listed with `"tested": false` — its semantics is still fully defined (the body is the definition); only the
torch-agreement evidence is missing.  This module adds nothing to `conformance.py`; it reads the registry and the files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verity_vllm.registry import ref_prims as R

DEFAULT_RECORDS = Path(__file__).resolve().parents[2] / "docs" / "data" / "ref-prims"

# Kinds without a record of their own: exercised gate-by-gate inside the listed tested kind (test_ref_prims.py checks
# twin == gates on the parent, so the child's evaluator/body is on the tested path).
COVERED_BY = {
    "DotRef_v1": "LinearRef_v1 (every output is one DotRef call)",
    "GeluRefElem_v1": "GeluRef_v1 (per element)",
    "SiluRefElem_v1": "SiluRef_v1 (per element)",
    "ArgmaxStepF32_v1": "ArgmaxRef_v1 (the fold step)",
    "RoPEHeadsRef_v1": "RoPERef_v1 (batched over heads; same body per head)",
    "F32Sub_v1": "LayerNormRef_v1 / SoftmaxRef_v1 / SiluRef_v1 bodies; specials in test_transcendental_specials",
    "F32Sqrt_v1": "RMSNormRef_v1 / LayerNormRef_v1 bodies (rsqrt = 1/sqrt); specials in test_transcendental_specials",
    "F32MaxRef_v1": "SoftmaxRef_v1 / EagerAttentionHeadRef_v1 bodies (row max); specials in test_transcendental_specials",
    "F32GtStrict_v1": "ArgmaxRef_v1 body (strict > : first index wins ties, as torch.argmax)",
}


def record(records_dir: Path = DEFAULT_RECORDS) -> dict:
    manifest = R.ref_vocab_manifest()
    digest = R.ref_vocab_digest()
    files = sorted(records_dir.glob("*.json")) if records_dir.is_dir() else []
    tested: dict[str, list[dict]] = {}
    stale = []
    for f in files:
        if f.name == "ref-vocab-manifest.json":
            continue
        rec = json.loads(f.read_text())
        base = rec["kind"].split("[")[0]
        for kid in base.split("+"):
            tested.setdefault(kid, []).append({"file": f.name, "DT": rec["DT"], "result": rec["result"], "domain": rec["domain"],
                                              "oracle": rec["oracle"], "label": rec["label"]})
        if rec.get("definition", {}).get("ref_vocab_digest") not in (None, digest):
            stale.append(f.name)
    kinds = {}
    for kid, entry in manifest["kinds"].items():
        cov = COVERED_BY.get(kid)
        kinds[kid] = {**entry, "tested": kid in tested, "records": tested.get(kid, []),
                      "covered_by": cov if kid not in tested else None,
                      "proof_support": "none (no Rust twin)"}
    profile_additions = {d.id: {"conformance": d.conformance, "statics": list(d.statics), "records": tested.get(d.id, []),
                                "proof_support": "yes via RopeOut_v1/RopeOutAdd_v1 twins (composite of twinned primitives)"}
                         for d in R.PROFILE_ADDITIONS}
    return {"schema": "verity-ir/conformance-ref/v0", "version": manifest["version"], "digest": digest,
            "label": "conformance over tested domain, not universal correctness",
            "kinds": kinds, "families": manifest["families"], "shared_dependencies": manifest["shared"],
            "profile_mode_additions_not_in_digest": profile_additions,
            "records_dir": str(records_dir), "records_with_other_digest": stale,
            "counts": {"kinds": len(kinds), "tested": sum(1 for k in kinds.values() if k["tested"]),
                       "covered_by_a_tested_kind": sum(1 for k in kinds.values() if not k["tested"] and k["covered_by"]),
                       "untested": sum(1 for k in kinds.values() if not k["tested"] and not k["covered_by"])}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", default=str(DEFAULT_RECORDS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rec = record(Path(a.records))
    text = json.dumps(rec, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
