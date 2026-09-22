"""Lane WALK: read-only extraction of the B0 output manifest facts from the pinned record.

Reads ONLY committed artefacts (the record `fixtures/e2e-composed/INT-B0-20260908T0725Z/`,
the workload `e2e/workload-B0-small.json`, `manifests/checkpoints.json`) plus the
registry types, and prints the facts `docs/vllm-poc/output-manifest-b0.md` quotes: the STEPS/LP
convention as computed by `e2e/run_composed.py`, the typed program's designated output type, the
boundary-tree leaf schema and counts, the pinned digests, and the W_out arithmetic under the three
width conventions (wire 32, typed vocab domain log2 V, pinned b = ceil(log2 V)).

Optionally (`--tokenizer-dir DIR`), verifies locally downloaded SmolLM2-135M tokenizer files against
the sha256 pins of `manifests/checkpoints.json` and reports EOS_ID and the token-id range.

Run from the repository root:

    PYTHONPATH=. .venv/bin/python scripts/walk_b0_manifest.py
    PYTHONPATH=. .venv/bin/python scripts/walk_b0_manifest.py --tokenizer-dir /tmp/walk-b0-tokenizer

Writes nothing.  Lane WALK, 2026-09-08.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

POC = Path(__file__).resolve().parents[1]
RECORD = POC / "fixtures" / "e2e-composed" / "INT-B0-20260908T0725Z"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=str(RECORD))
    ap.add_argument("--tokenizer-dir", default=None,
                    help="directory holding config.json / generation_config.json / tokenizer.json / "
                         "tokenizer_config.json downloaded at the pinned revision")
    args = ap.parse_args()
    record = Path(args.record)

    manifest = json.loads((record / "manifest.json").read_text())
    program = json.loads((record / "program" / "program.json").read_text())
    statement = json.loads((record / "verify" / "statement.json").read_text())
    workload = json.loads(Path(manifest["inputs"]["workload"].replace("/workspace/veritor-int/vllm-poc", str(POC))).read_text()) \
        if manifest["inputs"]["workload"].startswith("/workspace") else json.loads(Path(manifest["inputs"]["workload"]).read_text())

    # -- the STEPS/LP convention, exactly as e2e/run_composed.py stage_program computes it --------
    lp = max(int(x) for x in workload["prompt_lengths"])
    steps = max(int(x) for x in workload["max_tokens"]) - 1
    assert program["program"]["LP"] == lp and program["program"]["STEPS"] == steps, "record disagrees with the convention"
    max_tokens = max(int(x) for x in workload["max_tokens"])

    # -- boundary leaf schema (verity_vllm.capture.Boundary.leaves): counts must match the record --
    # per request: LP prompt_token(int32) + max_tokens output_presence(uint8) + n_present
    # output_token(int32) + sampling_params(json) + seed(json) + finish_reason(utf8)
    expected_full = lp + max_tokens + max_tokens + 3  # all slots present (finish_reason=length)
    boundary_counts = {k: v["leaf_count"] for k, v in statement["boundary_roots"].items()}
    V = program["program"]["static_configuration"]["V"]
    b = math.ceil(math.log2(V))

    out = {
        "record": str(record.relative_to(POC.parent)) if str(record).startswith(str(POC.parent)) else str(record),
        "program": {
            "function": program["program"]["function"],
            "digest": program["program"]["digest"],
            "gates": program["program"]["gates"],
            "input_gates": program["program"]["input_gates"],
            "LP": lp, "STEPS": steps,
            "steps_convention": "LP = max(workload prompt_lengths); STEPS = max(workload max_tokens) - 1 "
                                "(e2e/run_composed.py stage_program); the program's STEPS+1 = max_tokens "
                                "sampled tokens are its whole return, Array(STEPS+1, I32)",
        },
        "workload": {
            "requests": len(workload["requests"]),
            "prompt_len": lp, "max_tokens": max_tokens,
            "sha256": manifest["inputs"]["workload_sha256"],
        },
        "checkpoint": manifest["inputs"]["checkpoints"] | {"note": "B0 row of manifests/checkpoints.json"},
        "boundary_trees": {
            "expected_leaves_per_request_when_all_slots_present": expected_full,
            "leaf_counts_in_statement": boundary_counts,
            "schema": "LP x prompt_token(int32) + max_tokens x output_presence(uint8) + "
                      "n_present x output_token(int32) + sampling_params(json) + seed(json) + finish_reason(utf8); "
                      "tree order = code-point order of canonical id (verity_vllm/commit.py)",
        },
        "w_out_arithmetic_per_request": {
            "V": V, "b_ceil_log2_V": b, "log2_V": round(math.log2(V), 4),
            "designated_token_slots_v1_record": steps + 1,
            "wire_bits_v1 (Array(STEPS+1, I32), 32-bit wire)": (steps + 1) * 32,
            "typed_vocab_bits_v1 ((STEPS+1) * log2 V)": round((steps + 1) * math.log2(V), 2),
            "pinned_b16_bits_v1 ((STEPS+1) * 16, GUIDANCE §31.2)": (steps + 1) * 16,
            "serve3_projection": {
                "D_MAX (= max_tokens, serve3-spec §2)": max_tokens,
                "token_slots (D_MAX+1; slot D_MAX committed, never semantic)": max_tokens + 1,
                "pinned_b16_bits ((D_MAX+1) * 16)": (max_tokens + 1) * 16,
                "liveness_flag_bits_if_designated (D_MAX+1, PROPOSED)": max_tokens + 1,
            },
        },
        "plan": {
            "validated_unit": program["layer_program"]["function"],
            "layer_digest": program["layer_program"]["digest"],
            "plan_digest": manifest["inputs"]["plan_digest"],
            "bound_out_bits_layer": 18432,
            "note": "out_bits = wire-width sum of escaping boundary gates "
                    "(src/veritor/plans/boundary.py Boundary.out_bits); 18432 = 2 outputs x 576 x 16-bit bf16 wire",
        },
        "challenge": {
            "digest": manifest["inputs"]["challenge_digest"],
            "policy": statement["policy"],
            "selector": statement["selective"]["selection"],
        },
    }

    if args.tokenizer_dir:
        tdir = Path(args.tokenizer_dir)
        pins = {}
        cps = json.loads((POC / "manifests" / "checkpoints.json").read_text())
        b0 = next(c for c in cps["checkpoints"] if c.get("role") == "B0")
        for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
            got = hashlib.sha256((tdir / name).read_bytes()).hexdigest()
            want = b0["files"][name]["sha256"]
            pins[name] = {"sha256": got, "matches_pin": got == want}
        gen = json.loads((tdir / "generation_config.json").read_text())
        cfg = json.loads((tdir / "config.json").read_text())
        tok = json.loads((tdir / "tokenizer.json").read_text())
        ids = set(tok["model"]["vocab"].values()) | {a["id"] for a in tok.get("added_tokens", [])}
        out["tokenizer_pin"] = {
            "revision": b0["revision"],
            "files": pins,
            "eos_token_id": gen["eos_token_id"], "bos_token_id": gen["bos_token_id"],
            "config_vocab_size": cfg["vocab_size"],
            "distinct_token_ids": len(ids), "min_id": min(ids), "max_id": max(ids),
            "contiguous_0_to_max": ids == set(range(max(ids) + 1)),
            "special_tokens": [(a["id"], a["content"]) for a in tok.get("added_tokens", [])],
            "all_ids_fit_16_bits": max(ids) < 2 ** 16,
        }

    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
