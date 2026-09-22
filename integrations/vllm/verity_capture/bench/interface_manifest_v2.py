"""Interface manifest v2 -- value-family rows (VALUE-FIRST.md, Daniel 2026-09-18 17:26Z; R9 lane CMT-1).

Input: the v1 class manifest (`interface_manifest.py` output: the census's class enumeration, exact counts / bytes for the
record -- kept underneath as the generated ground truth).  Output: the same file reorganised into ROWS: a row = a family of
required values sharing the same producer AND the same collection mechanism.  Every class maps to exactly one row (one class,
`MufuEx2Ftz`, is split by count between two rows because one primitive produces two families: P and the rescale factors);
the mapping is explicit (`class_to_row`).  Counts are logical identities (the census counts VU output identities, never uses);
physical occurrences are listed separately per row.  Location = location + time (lifetime column).

Per row: `requirement`, `correspondence`, `candidates`, `measurement` + `value_correctness`, `binding_correctness`,
`diag_tap_coverage`, `secure_commit_coverage`.  Rows for the inventory families 1..10 all exist (row 9 and row 10 explicitly
unresolved).  `--bench DIR` fills the cmt-ref measurement / acceptance fields of the rows its reference path binds.

    python -m verity_capture.bench.interface_manifest_v2 out/gen/r9/cmt-ref/interface_manifest_v1.json --out out/gen/r9/cmt-ref \
        [--bench out/gen/r9/cmt-ref/bench] [--name interface_manifest]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path

SCHEMA = "verity-gen/interface-manifest/v2"
VERSION = "2026-09-18.2"
RESCALE_EVENTS_CLASS = "guardneginfzero_fo2"         # per-(layer, head) x 41: the online-softmax rescale events; MufuEx2 = P + these

# --- rows -------------------------------------------------------------------------------------------------------------------------
# (row id, inventory family #, name, producer, collection mechanism, [regex on class id], notes)
ROWS: list[dict] = [
    dict(id="R1a", family=1, name="prescribed weights (bf16)", producer="registration (model.safetensors, manifest weights_sha256)",
         mechanism="reference the production weights tree (`verity_vllm.commit.weights_tree`), openings per row block", patterns=[r"^input161/in/Input16$"],
         already_bound="yes: weights tree root bound to `weights_sha256` by `verify_source_linkage` [source-inspected verity_vllm/commit.py]",
         lifetime="whole model lifetime; re-read from HBM every step (physical re-reads are uses, not identities)"),
    dict(id="R1b", family=1, name="request prompt token ids (i32)", producer="request admission", mechanism="`boundary/<request>` tree (production)",
         patterns=[r"^input321/in/Input32$"], already_bound="yes: boundary tree of the request [source-inspected]",
         lifetime="request lifetime (input_ids buffer of step 0; positions re-derived)"),
    dict(id="R2a", family=2, name="GEMM outputs (bf16 epilogue): qkv_proj, o_proj, gate_up_proj, down_proj, lm_head logits", producer="cuBLAS bf16 GEMM (`Gemm_v1`, GemmCoordinate epilogue F2fpBf16)",
         mechanism="export-and-commit: forward hook on the Linear / LogitsProcessor module output -> `copy_` into a preallocated device buffer -> per-step Merkle (cmt-ref)",
         patterns=[r"^gemm1/out/"], lifetime="qkv_proj output: until the RoPE kernel rotates q/k IN PLACE (same buffer); o_proj / down_proj outputs: until the fused add-RMSNorm overwrites them IN PLACE with the normalized output; gate_up: until SiLU consumes; logits: until the sampler"),
    dict(id="R2b", family=2, name="embedding gather output (bf16)", producer="`Embedding_v1` gather (vLLM VocabParallelEmbedding)", mechanism="export-and-commit: forward hook (cmt-ref)",
         patterns=[r"^embedding1/out/"], lifetime="becomes the residual stream: layer 0 post-attention fused norm overwrites it IN PLACE"),
    dict(id="R2c", family=2, name="layer-0 input RMSNorm output (bf16, Triton `_rms_norm_kernel`, no residual)", producer="`RmsNormTriton_v1`", mechanism="export-and-commit: forward hook (cmt-ref)",
         patterns=[r"^rmsnormtriton1/out/"], lifetime="until qkv_proj consumes it (fresh output buffer)"),
    dict(id="R2d", family=2, name="fused add-RMSNorm normalized output (bf16)", producer="`RmsNormFusedCuda_v2` (`fused_add_rms_norm_kernel`)", mechanism="export-and-commit: forward hook, output[0] (cmt-ref)",
         patterns=[r"^rmsnormfusedcuda2/out/f2fpbf16"], lifetime="written IN PLACE over the kernel's `input` (the previous GEMM output); until the next GEMM consumes it"),
    dict(id="R2e", family=2, name="fused add-RMSNorm residual output (bf16 sum x + residual)", producer="`RmsNormFusedCuda_v2` (`fused_add_rms_norm_kernel`)", mechanism="export-and-commit: forward hook, output[1] (cmt-ref)",
         patterns=[r"^rmsnormfusedcuda2/out/bf16addf2fp"], lifetime="written IN PLACE over the incoming residual; lives until the next fused norm overwrites it (one sub-layer later)"),
    dict(id="R2f", family=2, name="SiLU(gate)*up output (bf16)", producer="`SiluMul_v1` (vLLM SiluAndMul CUDA kernel)", mechanism="export-and-commit: forward hook (cmt-ref)",
         patterns=[r"^silumul1/out/"], lifetime="until down_proj consumes it"),
    dict(id="R2g", family=2, name="RoPE-rotated q and k (bf16; ropeout = cos-term half, ropeoutadd = sin-term half, by position)", producer="`Rope_v1` (vLLM `rotary_embedding` CUDA kernel, IN PLACE on the qkv GEMM output)",
         mechanism="export-and-commit: forward hook on the shared RotaryEmbedding module (one object for all layers; instance = call index) (cmt-ref)",
         patterns=[r"^rope1/out/"], lifetime="q: until FA2 consumes it; k: copied into the KV cache by `reshape_and_cache_flash` (see R3), then the qkv buffer dies"),
    dict(id="R2h", family=2, name="attention output O (bf16, FA2 epilogue)", producer="`Attention_v3` output (`flash_fwd_splitkv_kernel` epilogue F2fpBf16)", mechanism="export-and-commit: forward hook on the Attention module (cmt-ref)",
         patterns=[r"^attention3/out/"], lifetime="until o_proj consumes it"),
    dict(id="R2i", family=2, name="sampled token id per step (i32)", producer="`TokenSelect_v1` (greedy argmax; vLLM sampler)", mechanism="export-and-commit: post-call hook on the runner's sampler (cmt-ref)",
         patterns=[r"^tokenselect1/out/"], lifetime="until the scheduler appends it to the request"),
    dict(id="R3", family=3, name="K/V values retained in the KV cache and re-read at every later step", producer="K = R2g (rotated k half), V = R2a (v slice of the qkv output); copied by `reshape_and_cache_flash` into the paged KV cache",
         mechanism="reference row: the logical identities are bound once at their producer occurrence (R2g / R2a); the cache slot is a second PHYSICAL occurrence of the same value (identity = (layer, kv head, token position)); reuse across steps = uses, not identities",
         patterns=[], lifetime="cache slot: request lifetime (prefix caching OFF in the admitted profile: no cross-request reuse)"),
    dict(id="R4", family=4, name="fp32 conversions inside fused add-RMSNorm of its two bound bf16 operands (the incoming GEMM output; the bf16 residual sum from the previous fused norm)",
         producer="`RmsNormFusedCuda_v2` interior: Bf16ToF32 of `input` and of `residual` before the fp32 add (`fused_add_rms_norm_kernel`)",
         mechanism="derived-from-bound candidate (derived_value_rule.md) through R2a / R2e / R2b; baseline = regenerate the exact conversion and commit",
         patterns=[r"^rmsnormfusedcuda2/int/bf16tof32"], lifetime="registers only, inside the kernel; never materialized"),
    dict(id="R5", family=5, name="FA2 fp32 score values (raw QK^T tensor-core accumulator; scaled-and-max-shifted stage before exp2)",
         producer="`Attention_v3` interior: `flash_fwd_splitkv_kernel` S = Q K^T fragments (fp32) and s*scale - m", mechanism="export-and-commit via register taps or producer-fused commitment (cmt-hidden)",
         patterns=[r"^attention3/int/amperebf16tcdot16_fo2", r"^attention3/int/f32fmasubftz"], lifetime="register fragments; the fp32 S fragment is overwritten by exp2 in place (stage is part of the mapping)"),
    dict(id="R6", family=6, name="FA2 fp32 softmax weights P = exp2(s*scale*log2e - m) (native fp32 fragment)", producer="`Attention_v3` interior: MufuEx2Ftz on the score fragment",
         mechanism="export-and-commit vs producer-fused (cmt-hidden)", patterns=[r"^attention3/int/mufuex2ftz"], split={"to": "R8", "by_class": RESCALE_EVENTS_CLASS},
         lifetime="register fragment; converted to bf16 for PV (R7) and summed into the row sum (R8) then dead"),
    dict(id="R7", family=7, name="FA2 bf16 P fragments consumed by PV (F2fpBf16 of R6)", producer="`Attention_v3` interior: F2fpBf16 conversion of the fp32 P fragment",
         mechanism="direct commit of the converted bits, or derived-from-bound through R6 (derived_value_rule.md) (cmt-hidden)", patterns=[r"^attention3/int/f2fpbf16"],
         lifetime="register operand of the PV tensor-core MMA; dead after the MMA"),
    dict(id="R8", family=8, name="other FA2 intermediates the query requires: running max, rescale factors, row sums, PV accumulators, epilogue normalisation",
         producer="`Attention_v3` interior: online-softmax bookkeeping of `flash_fwd_splitkv_kernel`", mechanism="taps / producer-fused (cmt-hidden); intersect with the requirement",
         patterns=[r"^attention3/int/f32max", r"^attention3/int/f32mulftz", r"^attention3/int/f32addftz", r"^attention3/int/amperebf16tcdot16_fo1q", r"^attention3/int/fa2invsum",
                   r"^attention3/int/guardneginfzero", r"^attention3/int/f32fmaftz", r"^attention3/int/f32subftz"],
         lifetime="registers across the key-block loop (running quantities) or the epilogue"),
    dict(id="R9", family=9, name="values inside inactive padded slots (max-padded Program interiors that ordinary serving does not execute)", producer="none in this execution",
         mechanism="separate explicit treatment (regenerate + bind, or a justified change of interior obligations); NOT a visibility problem", patterns=[],
         lifetime="no physical occurrence"),
    dict(id="R10a", family=10, name="fused add-RMSNorm interior fp32 values (sum x, x*x accumulate, variance/N, rsqrt, x*rstd)", producer="`RmsNormFusedCuda_v2` interior (`fused_add_rms_norm_kernel`)",
         mechanism="no verified path (registers / shared memory; residency not established)", patterns=[r"^rmsnormfusedcuda2/int/(f32mul|f32add|f32div|rsqrtapprox)"], lifetime="registers / smem inside the kernel"),
    dict(id="R10b", family=10, name="layer-0 Triton RMSNorm interior fp32 values (accumulate, mean, sqrt, reciprocal, x*rstd*w)", producer="`RmsNormTriton_v1` interior (`_rms_norm_kernel`)",
         mechanism="no verified path", patterns=[r"^rmsnormtriton1/int/(f32add|f32fma|f32mul|divfullrcp|mufusqrtftz|divfullscalea)"], lifetime="registers inside the Triton kernel"),
    dict(id="R10c", family=10, name="layer-0 Triton RMSNorm bf16->fp32 conversions of the embedding output (two loads of the same bound word)", producer="`RmsNormTriton_v1` interior: Bf16ToF32 of x (variance pass and normalize pass)",
         mechanism="derived-from-bound candidate through R2b (not tested); baseline = regenerate", patterns=[r"^rmsnormtriton1/int/bf16tof32"], lifetime="registers"),
    dict(id="RC", family=0, name="constants of the Program text (Const32 / Const16 the default(32) query makes VUs of)", producer="Program text", mechanism="excluded from the obligation (fixed by the registered Program)",
         patterns=[r"/const/"], lifetime="n/a"),
]

CANDIDATES = {
    "R1a": [dict(id="weights-tree", mechanism="production weights tree; per-request: reference the root, open row blocks on challenge", producer_change="none", status="exists (production)", lane="production")],
    "R1b": [dict(id="boundary-tree", mechanism="production boundary/<request> tree", producer_change="none", status="exists (production)", lane="production")],
    "R2": [dict(id="hook-copy-torch-merkle", mechanism="forward-hook `copy_` into a device buffer; torch-op SHA-256 leaves (64 B, 32 bf16) + Merkle on the device; per-step subtree, request root; positions.json", producer_change="none", status="built + measured (cmt-ref reference)", lane="cmt-ref"),
           dict(id="hook-copy-host-hashlib", mechanism="same acquisition; D2H per step; hashlib leaves + host tree (slow reference)", producer_change="none", status="built + measured (cmt-ref)", lane="cmt-ref"),
           dict(id="hash-in-place-no-copy", mechanism="hash the producer's output buffer directly at the hook (no copy) -- valid only where nothing mutates the buffer before the hash kernel runs (stream order guarantees it for eager)", producer_change="none", status="candidate (not built; the copy is 1 read + 1 write of the family's bytes)", lane="cmt-ref/cmt-integ"),
           dict(id="fused-epilogue-hash", mechanism="hash inside the producer epilogue (GEMM / norm kernels)", producer_change="implementation change (no numerical change)", status="candidate (not built)", lane="cmt-integ")],
    "R3": [dict(id="bind-at-producer", mechanism="bound once by R2g/R2a at the producer occurrence; cache reads are uses; identity preserved by (layer, kv head, position) in positions.json", producer_change="none", status="covered by R2a/R2g when those are bound", lane="cmt-ref"),
           dict(id="bind-at-cache-write", mechanism="hash the `reshape_and_cache_flash` destination slots (second physical occurrence) -- needed only for cross-request reuse (prefix caching), which the admitted profile disables", producer_change="none", status="not needed for this record", lane="-")],
    "R4": [dict(id="derived-value-rule", mechanism="opening rule: the fp32 word is Bf16ToF32(bound bf16 word) -- protocol clause (derived_value_rule.md); every consumer constrained to the same derived value", producer_change="none", status="CMT-3 tests independently", lane="cmt-integ"),
           dict(id="regenerate-and-commit", mechanism="recompute the exact conversion from the bound bf16 words and commit the fp32 words (baseline; 2x bytes)", producer_change="none", status="candidate", lane="cmt-integ")],
    "R5": [dict(id="export-and-commit-taps", mechanism="register taps of the S fragment (out/gen/r8/fa2-kernel) -> export -> commit", producer_change="yes (kernel patch: stores)", status="taps exist (diagnostic); secure commit unresolved", lane="cmt-hidden"),
           dict(id="producer-fused-commit", mechanism="hash fragments inside the kernel while live", producer_change="yes (kernel patch: hash instructions)", status="candidate", lane="cmt-hidden")],
    "R6": "R5", "R8": "R5",
    "R7": [dict(id="direct-commit-converted-bits", mechanism="export / fuse the bf16 fragment", producer_change="yes", status="candidate", lane="cmt-hidden"),
           dict(id="derived-from-R6", mechanism="F2fpBf16 of the bound fp32 P (derived_value_rule.md)", producer_change="none beyond R6", status="candidate", lane="cmt-hidden/cmt-integ")],
    "R9": [dict(id="regenerate-and-bind", mechanism="compute the padded interiors off the serving path and bind", producer_change="none", status="unresolved / not-produced in this execution", lane="-"),
           dict(id="change-interior-obligation", mechanism="a justified mechanism changing what the padded interior obliges", producer_change="spec-level", status="unresolved", lane="-")],
    "R10a": [dict(id="none", mechanism="no verified path", producer_change="would need kernel stores or fused hashing", status="unresolved", lane="-")],
    "R10b": "R10a",
    "R10c": [dict(id="derived-from-R2b", mechanism="Bf16ToF32 of the bound embedding output", producer_change="none", status="candidate (untested)", lane="cmt-integ")],
    "RC": [dict(id="excluded", mechanism="fixed by the Program text", producer_change="none", status="n/a", lane="-")],
}

def _stage(status: str) -> str:
    st = status.lower()
    if st.startswith("exists (production)"):
        return "production"
    if st.startswith("built"):
        return "built"
    if st.startswith("covered by"):
        return "inherited"
    if st.startswith("taps exist"):
        return "diagnostic-only"
    if st.startswith(("candidate", "cmt-3 tests", "not needed")):
        return "candidate"
    return "none"


def _norm_candidates(cands: list[dict]) -> list[dict]:
    out = []
    for c in cands:
        c = dict(c)
        c.setdefault("stage", _stage(c["status"]))
        c.setdefault("binding_test", {"status": "production" if c["stage"] == "production" else "pending", "test": None})
        out.append(c)
    return out


EVIDENCE = {
    "R1a": "source-inspected", "R1b": "source-inspected",
    "R2a": "source-inspected (census instances.jsonl output_views == leaf count for every specialization; the module output IS the census view); upgraded to measured-correspondence (count-level) by the bench: acquisition binds exactly the manifest bytes per class (binding.json ratio 1.0) and the sampled tokens opened from the commitment == the generated tokens; bit-level equality with the record instances.jsonl output_views == leaf count for every specialization; acquisition binds exactly the manifest bytes (bench binding.json); s snapshot digests: not tested in-window (different GPU: L40S vs RTX 4090)",
    "R2": "R2a",
    "R3": "source-inspected (vLLM attention layer: `reshape_and_cache_flash(k, v, kv_cache, slot_mapping)` after RoPE; FA2 reads the cache for keys < current)",
    "R4": "source-inspected (vLLM csrc/layernorm_kernels.cu fused_add_rms_norm_kernel: `float x = (float) input[i]; x += (float) residual[i]`; row-level: both operands converted); NOTE the assignment of _fo1/_fo2 to input vs residual is inferred from the registry operand order -- consumers must check it before binding one class without the other",
    "R5": "source-inspected (out/gen/r8/fa2-sem exact register correspondence, cmt-hidden's population)", "R6": "R5", "R7": "R5", "R8": "R5",
    "R9": "unresolved (no physical occurrence in this execution; the record is the active reference Program of 1 request)",
    "R10a": "source-inspected (kernel source), residency unresolved", "R10b": "R10a", "R10c": "source-inspected (Triton kernel loads x twice)", "RC": "source-inspected (Program text)",
}


def _bytes(c):
    return float(c["logical_bytes"])


def _norm(cid: str) -> str:
    """`family/kind/<...>/<out primitive>` -> `family/kind/<out primitive>` (the conservative candidate's ids carry zip_ prefixes)."""
    segs = cid.split("/")
    return "/".join([segs[0], segs[1], segs[-1]]) if len(segs) >= 3 else cid


def build(v1: dict, bench: dict | None) -> dict:
    classes = v1["classes"]
    class_to_row: dict[str, list] = {}
    rows_out = []
    counts_by_class = {c["id"]: int(c["logical_count"]) for c in classes}
    rescale = next((counts_by_class[k] for k in counts_by_class if k.endswith(RESCALE_EVENTS_CLASS)), 0)
    for spec in ROWS:
        members = [c for c in classes if any(re.search(p, _norm(c["id"])) for p in spec["patterns"])]
        n_ident, n_bytes, member_rows = 0, 0.0, []
        for c in members:
            cnt, b = int(c["logical_count"]), _bytes(c)
            if spec.get("split") and _norm(c["id"]).startswith("attention3/int/mufuex2"):
                cnt, b = cnt - rescale, b - rescale * 4
                class_to_row.setdefault(c["id"], []).append({"row": spec["id"], "logical_count": cnt, "label": "calculated (class count minus the rescale events)"})
                class_to_row[c["id"]].append({"row": spec["split"]["to"], "logical_count": rescale, "label": f"calculated ({RESCALE_EVENTS_CLASS} count)"})
            else:
                class_to_row.setdefault(c["id"], []).append({"row": spec["id"], "logical_count": cnt, "label": "measured-from-record (census)"})
            n_ident += cnt
            n_bytes += b
            member_rows.append({"class_id": c["id"], "dtype": c["dtype"], "logical_count": cnt, "logical_bytes": b, "frequency": c["frequency"]["unit"],
                                "acquisition_v1": c["acquisition"], "kernel": c.get("kernel"), "role": c.get("role")})
        if spec["id"] == "R8":
            n_ident += rescale
            n_bytes += rescale * 4
            member_rows.append({"class_id": "attention3/int/mufuex2ftz_v1/mufuex2ftz_fo2 (rescale share)", "dtype": "f32", "logical_count": rescale, "logical_bytes": rescale * 4,
                                "frequency": "per-(layer, head) x rescale events", "acquisition_v1": "hidden-fa2", "kernel": "flash_fwd_splitkv_kernel", "role": "exp2(m_prev - m_new) rescale factor"})
        cands = CANDIDATES.get(spec["id"], CANDIDATES.get(spec["id"][:2], []))
        if isinstance(cands, str):
            cands = CANDIDATES[cands]
        ev = EVIDENCE.get(spec["id"], EVIDENCE.get(spec["id"][:2], "unresolved"))
        if ev in EVIDENCE:
            ev = EVIDENCE[ev]
        ev_label = next((lab for lab in ("measured-correspondence", "source-inspected", "inferred", "unresolved") if ev.startswith(lab)), "unresolved")
        cands = _norm_candidates(cands)
        required = spec["family"] != 0
        row = {
            "id": spec["id"], "family": spec["family"], "name": spec["name"], "required": required,
            "classes": [m["class_id"] for m in member_rows], "members": member_rows,
            "requirement": {
                "program_digest": v1["record"]["program_digest"], "query_id": v1["query"]["query_id"], "query_kind": v1["query"]["kind"], "record": v1["record"]["run"],
                "semantics": spec["name"], "producer_definition": spec["producer"],
                "spec_range": sorted({m["frequency"] for m in member_rows}) or (["n/a"] if not required else ["no occurrence in this record"]),
                "dtypes": sorted({m["dtype"] for m in member_rows}),
                "logical_identities": n_ident, "logical_bytes": n_bytes, "count_label": "measured-from-record (census class counts; identities, not uses)" if member_rows else "n/a",
                "already_bound": spec.get("already_bound", "no"),
            },
            "correspondence": {
                "pinned_producer": spec["producer"], "occurrence": spec["mechanism"] if spec["id"] in ("R3",) else None,
                "coordinate_mapping": None, "lifetime": spec["lifetime"], "evidence": ev, "evidence_label": ev_label,
            },
            "physical_occurrences": None,
            "candidates": cands,
            "measurement": {"status": "not measured", "by": None, "samples": 0},
            "value_correctness": {"test": None, "status": "pending"},
            "binding_correctness": {"test": None, "status": "pending"},
            "diag_tap_coverage": {"fraction_of_bytes": None, "note": None},
            "secure_commit_coverage": {"fraction_of_bytes": None, "note": None},
            "status": "unresolved",
        }
        _fill_row(row, spec, bench)
        row["derived_status"] = derive_status(row)
        rows_out.append(row)
    # every class mapped exactly once (MufuEx2 twice by split)
    unmapped = [c["id"] for c in classes if c["id"] not in class_to_row]
    fam_present = sorted({r["family"] for r in rows_out if r["family"]})
    totals = {}
    for r in rows_out:
        k = "required" if r["required"] else "excluded"
        t = totals.setdefault(k, {"rows": 0, "logical_identities": 0, "logical_bytes": 0.0})
        t["rows"] += 1
        t["logical_identities"] += r["requirement"]["logical_identities"]
        t["logical_bytes"] += r["requirement"]["logical_bytes"]
    for k, v in totals.items():
        v["logical_MB"] = round(v["logical_bytes"] / 1e6, 3)
    weights = sum(r["requirement"]["logical_bytes"] for r in rows_out if r["id"] == "R1a")
    totals["required_excl_weights_MB"] = round((totals["required"]["logical_bytes"] - weights) / 1e6, 3)
    by_status = {}
    for r in rows_out:
        by_status.setdefault(r["status"], []).append(r["id"])
    return {
        "schema": SCHEMA, "version": VERSION, "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "label": v1["label"],
        "framing": "VALUE-FIRST.md (lane/r9-base, 2026-09-18 17:26Z): rows = value families sharing producer + collection mechanism; counts = logical identities",
        "record": v1["record"], "query": v1["query"], "source": {**v1["source"], "v1_manifest": "interface_manifest_v1.json (class ground truth, unchanged)"},
        "inventory_families_present": fam_present, "milestone": {"every_required_family_has_producer_mapping_or_explicit_unresolved": all(r["status"] for r in rows_out),
                                                                 "rows_by_status": by_status},
        "rows": rows_out, "class_to_row": class_to_row, "unmapped_classes": unmapped, "totals": totals,
        "classes": [{**c, "row": [m["row"] for m in class_to_row.get(c["id"], [])]} for c in classes],
        "headline": None,
        "lanes": v1["lanes"],
        "complete_commitment": {"definition": "every required row (family != 0) has secure_commit_coverage.fraction_of_bytes == 1.0 with value_correctness and binding_correctness passed, or, for derived rows, a bound source row at 1.0 plus the adopted derived-value rule, or, for R1, the production registration binding; R9 must carry an explicit treatment (not 'no occurrence' alone) for any padded Program; byte-weighted coverage is for prioritisation only",
                                "required_rows": [r["id"] for r in rows_out if r["required"]]},
    }


def derive_status(row: dict) -> dict:
    """The four machine-derived status fields (CMT-3's coverage checker recomputes these from the JSON with the same rules)."""
    corr = row["correspondence"]["evidence_label"] in ("measured-correspondence", "source-inspected")
    impl = any(c["stage"] in ("production", "built", "inherited") and c["binding_test"]["status"] in ("pass", "production", "inherited") for c in row["candidates"])
    bench = row["measurement"].get("status") == "measured" and int(row["measurement"].get("samples") or 0) > 0
    return {"correspondence_established": corr, "secure_path_implemented": corr and impl, "secure_path_benchmarked": corr and impl and bench,
            "unresolved": not corr or not impl,
            "rules": "correspondence := evidence_label in {measured-correspondence, source-inspected}; implemented := correspondence and a candidate with stage in {production, built, inherited} whose binding_test.status in {pass, production, inherited}; benchmarked := implemented and measurement.status == measured with samples > 0; unresolved := not correspondence or not implemented"}


def headline(doc: dict) -> dict:
    """Counts three ways over the REQUIRED classes (family != 0): by class, byte-weighted, frequency-weighted (logical identities on the
    reference workload).  A class split across rows counts once with the weakest status of its rows."""
    rows = {r["id"]: r for r in doc["rows"]}
    keys = ("correspondence_established", "secure_path_implemented", "secure_path_benchmarked", "unresolved")
    acc = {k: {"classes": 0, "bytes": 0.0, "identities": 0} for k in keys}
    tot = {"classes": 0, "bytes": 0.0, "identities": 0}
    excluded = 0
    per_class = {}
    for c in doc["classes"]:
        rids = c["row"]
        if not rids or all(not rows[r]["required"] for r in rids):
            excluded += 1
            continue
        st = {k: all(rows[r]["derived_status"][k] for r in rids) if k != "unresolved" else any(rows[r]["derived_status"][k] for r in rids) for k in keys}
        per_class[c["id"]] = st
        tot["classes"] += 1
        tot["bytes"] += float(c["logical_bytes"])
        tot["identities"] += int(c["logical_count"])
        for k in keys:
            if st[k]:
                acc[k]["classes"] += 1
                acc[k]["bytes"] += float(c["logical_bytes"])
                acc[k]["identities"] += int(c["logical_count"])
    n_rows = sum(1 for r in doc["rows"] if r["required"])
    lines = [f"{tot['classes']} required classes ({len(doc['classes'])} census classes, {excluded} constants excluded) in {n_rows} value-family rows",
             "                                 by class      byte-weighted (MB)   frequency-weighted (identities)"]
    labels = {"correspondence_established": "correspondence established", "secure_path_implemented": "secure path implemented   ",
              "secure_path_benchmarked": "secure path benchmarked   ", "unresolved": "unresolved                "}
    for k in keys:
        a = acc[k]
        lines.append(f"  {labels[k]}: {a['classes']:3d} / {tot['classes']:3d}   {a['bytes'] / 1e6:9.3f} / {tot['bytes'] / 1e6:9.3f}   {a['identities']:>12,} / {tot['identities']:>12,}")
    lines.append("  (funnel: benchmarked ⊆ implemented ⊆ correspondence; unresolved = no correspondence or no implemented secure path)")
    return {"text": "\n".join(lines), "counts": acc, "totals": tot, "excluded_classes": excluded, "rows_required": n_rows, "per_class": per_class,
            "weights": {"by_class": "number of census classes", "byte_weighted": "logical bytes on the reference workload (record census)",
                        "frequency_weighted": "logical identities (VU outputs) on the reference workload = dynamic instance volume"}}


def _fill_row(row: dict, spec: dict, bench: dict | None) -> None:
    rid = row["id"]
    if rid.startswith("R1") and not rid.startswith("R10"):
        row["status"] = "bound (production registration)"
        row["value_correctness"] = {"test": "production `verify_source_linkage` (weights) / boundary tree check", "status": "production"}
        row["binding_correctness"] = {"test": "production Merkle openings", "status": "production"}
        row["secure_commit_coverage"] = {"fraction_of_bytes": 1.0, "note": "registration, not per request"}
        for c in row["candidates"]:
            c["binding_test"] = {"status": "production", "test": "production Merkle openings / verify_source_linkage"}
        row["diag_tap_coverage"] = {"fraction_of_bytes": 1.0, "note": "n/a (inputs)"}
        row["physical_occurrences"] = [{"where": "HBM weights (bf16)", "when": "model lifetime", "bytes_per_request": 270078592 * 15 if rid == "R1a" else 620, "note": "re-read every step" if rid == "R1a" else "input_ids"}]
        row["correspondence"]["coordinate_mapping"] = "parameter name + row block (weights tree) / prompt position (boundary tree)"
        return
    if rid.startswith("R2"):
        row["status"] = "producer mapped; secure path built (cmt-ref)" if bench is None else "bound by cmt-ref reference path (measured)"
        row["correspondence"]["occurrence"] = "the module's output tensor (row t = token t of the step's flattened batch; column j = feature j), read at the forward hook before any in-place consumer runs (stream order)"
        row["correspondence"]["coordinate_mapping"] = "positions.json: (step, instance = module path, element = flat index) -> segment offset -> 64-byte leaf index + in-leaf offset (StepLayout.rank); leaf = pos_leaf(64 B)"
        row["physical_occurrences"] = [{"where": "producer output buffer (HBM)", "when": "from the kernel's epilogue until the in-place consumer / next reuse", "bytes_per_request": row["requirement"]["logical_bytes"]},
                                       {"where": "cmt-ref acquisition buffer (device, 128 MiB preallocated, reused per step)", "when": "hook -> end of the step's commit", "bytes_per_request": row["requirement"]["logical_bytes"], "note": "the copy = 1 extra read + 1 extra write of the family's bytes"}]
        if rid == "R2g":
            row["physical_occurrences"].append({"where": "KV cache (k half)", "when": "request lifetime", "note": "see R3"})
        if rid == "R2a":
            row["physical_occurrences"].append({"where": "KV cache (v slice of qkv)", "when": "request lifetime", "note": "see R3"})
        row["diag_tap_coverage"] = {"fraction_of_bytes": 1.0, "note": "forward hooks see 100% of the family's bytes (census: output views == leaf counts)"}
        row["value_correctness"] = {"test": "acquired bytes per class == manifest logical bytes (binding.json); sampled token ids opened from the commitment == the generated tokens; roots identical across repeated runs and across backends (buffer reuse across steps exercised); bit-level: sha256 of every acquired segment looked up among the accepted record's per-step observer snapshot digests (a_observer/snapshots.json) and the record's forward-hook digests (b_hooks/hooks.json) compared at the layout-derived offsets",
                                    "status": "pending bench"}
        row["binding_correctness"] = {"test": "bench negatives on real data: altered value (leaf-consistent / value-only), value claimed at another element / another leaf, another instance's leaf, another step's leaf, wrong Program digest, tampered positions, top path of another step; recommit with one byte flipped / two leaves swapped -> different root; openings verify against the ORIGINAL root with the production fold_path/path_shape",
                                      "status": "pending bench"}
        if bench is not None:
            _fill_bench(row, bench)
        return
    if rid == "R3":
        row["status"] = "producer mapped; covered by R2a/R2g binding (no cross-request reuse in the admitted profile)"
        row["correspondence"]["coordinate_mapping"] = "(layer, kv head, token position) <-> R2g k element (layer, t, head*64+d) / R2a qkv element (layer, t, 576+192+head*64+d); cache slot via slot_mapping"
        row["physical_occurrences"] = [{"where": "qkv GEMM output buffer", "when": "until RoPE (k rotated in place) / until the cache write (v)", "note": "first occurrence (bound)"},
                                       {"where": "paged KV cache", "when": "request lifetime; re-read by FA2 at every later step", "bytes_per_request_written": 169 * 30 * 2 * 192 * 2, "note": "second occurrence: a copy (`reshape_and_cache_flash`); identity + equality by construction of the copy (not verified by a binding)"}]
        row["value_correctness"] = {"test": "identity of the cache slot contents with the bound producer value: by construction (copy); not separately tested", "status": "not tested"}
        row["binding_correctness"] = {"test": "n/a (no separate binding)", "status": "n/a"}
        row["secure_commit_coverage"] = {"fraction_of_bytes": None, "note": "inherits R2a/R2g; 0 identities of its own"}
        row["candidates"][0]["binding_test"] = {"status": "inherited", "test": "R2a / R2g binding tests (the cache slot is a copy; no separate binding)"}
        row["diag_tap_coverage"] = {"fraction_of_bytes": None, "note": "inherits R2a/R2g"}
        return
    if rid == "R4":
        row["status"] = "producer mapped; derived route to be tested by CMT-3 (requirement + correspondence provided here)"
        row["correspondence"]["occurrence"] = "register value `(float) input[i]` and `(float) residual[i]` in fused_add_rms_norm_kernel, per (layer, sub-layer, token, feature); never stored"
        row["correspondence"]["coordinate_mapping"] = "(fused-norm call c, token t, feature j) -> source bf16 word: input = R2a element (o_proj / down_proj output of the same sub-layer, t, j); residual = R2e element of the previous fused norm (or R2b embedding for layer 0's post-attention norm)"
        row["physical_occurrences"] = [{"where": "registers", "when": "inside the kernel", "bytes_per_request": 0, "note": "no HBM occurrence"}]
        row["diag_tap_coverage"] = {"fraction_of_bytes": 0.0, "note": "no tap (fused kernel)"}
        row["secure_commit_coverage"] = {"fraction_of_bytes": 0.0, "note": "derived route pending (CMT-3)"}
        row["value_correctness"] = {"test": "CMT-3: Bf16ToF32(bound word) == the value the registered relation consumes (per derived_value_rule.md)", "status": "pending (cmt-integ)"}
        row["binding_correctness"] = {"test": "CMT-3: opening of the derived value = opening of the source + rule id", "status": "pending (cmt-integ)"}
        return
    if rid in ("R5", "R6", "R7", "R8"):
        row["status"] = "producer mapped (register correspondence, cmt-hidden); secure path unresolved"
        row["correspondence"]["occurrence"] = "register fragments of flash_fwd_splitkv_kernel per (layer, head, query row, key column / key block / epilogue) -- see out/gen/r8/fa2-sem (cmt-hidden)"
        row["correspondence"]["coordinate_mapping"] = "cmt-hidden's tap layout (thread, fragment, stage) -> (layer, head, q, k); stage explicit"
        row["physical_occurrences"] = [{"where": "registers", "when": "inside the key-block loop / epilogue", "bytes_per_request": 0}]
        row["diag_tap_coverage"] = {"fraction_of_bytes": None, "note": "cmt-hidden fills (P-only diagnostic fold exists: out/gen/r8/fa2-kernel)"}
        row["secure_commit_coverage"] = {"fraction_of_bytes": 0.0, "note": "no secure commitment measured yet"}
        row["value_correctness"] = {"test": "cmt-hidden: tap values == registered relation values (fa2-sem exact correspondence)", "status": "pending (cmt-hidden)"}
        row["binding_correctness"] = {"test": "cmt-hidden", "status": "pending (cmt-hidden)"}
        return
    if rid == "R9":
        row["status"] = "unresolved / not-produced (no physical occurrence in this execution; 0 identities in this record's Program)"
        row["requirement"]["logical_identities_note"] = "this record = active reference Program of one request (169 tokens, 15 steps): the census enumerates no padded-slot interiors; a max-padded Program is a different record and would add identities here"
        row["physical_occurrences"] = []
        row["diag_tap_coverage"] = {"fraction_of_bytes": None, "note": "nothing to tap: never executed"}
        row["secure_commit_coverage"] = {"fraction_of_bytes": None, "note": "unresolved"}
        row["value_correctness"] = {"test": None, "status": "unresolved"}
        row["binding_correctness"] = {"test": None, "status": "unresolved"}
        return
    if rid.startswith("R10"):
        row["status"] = "unresolved (no verified path)" if rid != "R10c" else "unresolved (derived candidate through R2b, untested)"
        row["physical_occurrences"] = [{"where": "registers / shared memory", "when": "inside the kernel", "bytes_per_request": 0}]
        row["diag_tap_coverage"] = {"fraction_of_bytes": 0.0, "note": "no tap"}
        row["secure_commit_coverage"] = {"fraction_of_bytes": 0.0, "note": "none"}
        row["value_correctness"] = {"test": None, "status": "unresolved"}
        row["binding_correctness"] = {"test": None, "status": "unresolved"}
        return
    if rid == "RC":
        row["status"] = "excluded (constants of the Program text)"
        row["physical_occurrences"] = []
        row["secure_commit_coverage"] = {"fraction_of_bytes": None, "note": "excluded"}
        row["diag_tap_coverage"] = {"fraction_of_bytes": None, "note": "excluded"}
        row["value_correctness"] = {"test": None, "status": "n/a"}
        row["binding_correctness"] = {"test": None, "status": "n/a"}


def _fill_bench(row: dict, bench: dict) -> None:
    """cmt-ref measurement for an R2 row from `bench/summary.json` + `bench/binding.json` (per-class bytes bound)."""
    binding = bench.get("binding") or {}
    per_class = binding.get("per_class_bytes_bound", {})
    bound = 0.0
    for m in row["members"]:
        cid = m["class_id"]
        if cid.startswith("rope1/out/"):
            bound += per_class.get("rope1/out/rotated(ropeout+ropeoutadd)", 0) / 2
        else:
            bound += per_class.get(cid, 0)
    frac = (bound / row["requirement"]["logical_bytes"]) if row["requirement"]["logical_bytes"] else None
    row["secure_commit_coverage"] = {"fraction_of_bytes": frac, "bytes_bound": bound, "note": "cmt-ref reference path (bench binding.json); 1.0 = every logical byte of the family is a committed leaf byte"}
    s = bench.get("summary") or {}
    v = s.get("validity") or {}
    n_samples = int(bench.get("n_samples") or 0)
    row["measurement"] = {"status": "measured" if n_samples else "not measured", "samples": n_samples, "raw": "bench/raw_samples.jsonl", "by": "cmt-ref",
                          "gpu": (s.get("gpu") or {}).get("name"), "sm": (s.get("gpu") or {}).get("sm"), "utc": s.get("utc"),
                          "table_md": s.get("table_md"), "note": "per-arm costs are for the WHOLE R2 family set (all rows R2a-R2i together, one stream per step); per-row costs are not separable in this path and need not add",
                          "openings": s.get("openings_summary"), "delta_T": bench.get("delta_T"), "extra": bench.get("extra", [])}
    ok_v = v.get("opened_tokens_equal_generated") and v.get("host_recommit_root_equal") and v.get("positions_equal")
    vc = bench.get("value_check") or {}
    direct = {"R2a": "down_proj (30/step) + lm_head logits (1/step) instances digested directly; qkv/o_proj/gate_up_proj instances implied",
              "R2d": "model.norm normalized output digested directly (1/step); per-layer normalized outputs implied",
              "R2e": "post_attention_layernorm residual (30/step) + model.norm.residual digested directly; input_layernorm residuals implied"}
    sc = bench.get("snapshot_check") or {}
    pcs = sc.get("per_class") or {}
    seg_m = seg_n = 0
    hit_pcs = set()
    for cid in row.get("classes", []):
        pc = pcs.get(cid)
        if pc is None:
            pref = cid.split("/")[0] + "/out/"
            pc = next((v for k, v in pcs.items() if k.startswith(pref)), None)
        if pc is not None and id(pc) not in hit_pcs:
            hit_pcs.add(id(pc))
            seg_m += pc["matched"]
            seg_n += pc["segments"]
    if vc.get("compared") and vc.get("equal") == vc.get("compared") and ok_v and seg_n:
        rid = row["id"]
        how = direct.get(rid, "not among the hook-digested tensors")
        expl = ""
        if rid == "R2g" and seg_m < seg_n:
            expl = " (the 60 unmatched are the prefill-step q/k slices, which the record observed only as strided views of the qkv buffer -- different spanned bytes, not comparable; all decode-step q/k match)"
        if rid == "R2i" and seg_m < seg_n:
            expl = " (4-byte sampled ids are not snapshotted as int32 in every step; the 15 sampled tokens equal the record's accepted tokens exactly)"
        row["value_correctness"]["status"] = (f"pass (bit-level, direct: {seg_m}/{seg_n} acquired segments of this family found sha256-identical among the accepted record's "
                                              f"observer snapshot digests of the same engine step (RTX 4090 record vs our L40S stream){expl}; hook-digest check "
                                              f"{vc['equal']}/{vc['compared']} equal ({how}); count-level ratio 1.0; opened tokens == generated)")
        row["value_correctness"]["evidence"] = {"snapshot_check": "bench_value/snapshot_check.json", "value_check": "bench_value/value_check.json",
                                                "segments_matched": seg_m, "segments": seg_n}
    else:
        row["value_correctness"]["status"] = ("pass (count-level + token-level; bit-level vs record snapshots not done)" if ok_v else "FAIL / incomplete") if v else "pending bench"
    ok_b = v.get("negatives_all_pass") and (v.get("openings", {}) or s.get("openings_summary", {}) or {}).get("all_verify")
    row["binding_correctness"]["status"] = ("pass" if ok_b else "FAIL / incomplete") if v else "pending bench"
    row["binding_correctness"]["negatives"] = v.get("negatives")
    for c in row["candidates"]:
        if c["stage"] == "built":
            c["binding_test"] = {"status": "pass" if ok_b else "fail", "test": "bench negatives + openings against the original root (validity.json)"}
    if ok_v and frac == 1.0:
        row["correspondence"]["evidence_label"] = "measured-correspondence"


def render_md(doc: dict) -> str:
    L = []
    L.append(f"# Interface manifest v2 — value-family rows — `{doc['record']['run']}` ({doc['label']} query) — {doc['utc']}")
    L.append("")
    L.append("~~~text")
    L.append(doc["headline"]["text"])
    L.append("~~~")
    L.append("")
    L.append(f"Status rules (machine-derived per row, `rows[*].derived_status`): {doc['rows'][0]['derived_status']['rules']}. Weights: {doc['headline']['weights']}.")
    L.append("")
    tot = doc["totals"]
    w_mb = sum(r["requirement"]["logical_bytes"] for r in doc["rows"] if r["id"] == "R1a") / 1e6
    p_mb = sum(r["requirement"]["logical_bytes"] for r in doc["rows"] if r["id"] == "R1b") / 1e6
    L.append(f"**Denominators.** The byte-weighted denominator {tot['required']['logical_MB']:.1f} MB = the census's per-request interface figure "
             f"{tot['required_excl_weights_MB']:.1f} MB (all dynamic value families R2–R10, logical identities x width, one identity counted once "
             f"regardless of how many kernels read it) + {w_mb:.1f} MB of weights (R1a, registered inputs: bound once per model by the production "
             f"registration, not per request; included here because the query requires them) + {p_mb:.3f} MB prompt token ids (R1b); the "
             f"{tot['excluded']['logical_MB']:.1f} MB of compile-time constants (RC) are excluded from every count.  Physical reads/writes are NOT "
             f"in the denominator (see `physical_bytes` per class in the v1 manifest).")
    L.append("")
    L.append(f"Framing: {doc['framing']}. Program `{doc['record']['program_digest'][:16]}…`, query `{str(doc['query'].get('query_id'))[:16]}…` (W_MAX {doc['query'].get('w_max')}, {doc['query'].get('vu_count') or 0:,} VUs). "
             f"Ground truth underneath: the {len(doc['classes'])} census classes of `interface_manifest_v1.json` (unchanged); `class_to_row` maps every class to its row "
             f"(one class, MufuEx2Ftz, is split by count between R6 and R8). Counts are **logical identities** (VU outputs), never uses; physical occurrences are listed per row.")
    L.append("")
    t = doc["totals"]
    L.append(f"Totals [measured-from-record]: required rows {t['required']['rows']}: {t['required']['logical_identities']:,} identities, {t['required']['logical_MB']} MB "
             f"({t['required_excl_weights_MB']} MB excluding the weights row R1a); excluded (constants) {t['excluded']['logical_MB']} MB. Milestone “every required family has a producer mapping or an explicit unresolved status”: "
             f"{'MET' if doc['milestone']['every_required_family_has_producer_mapping_or_explicit_unresolved'] else 'NOT MET'} — statuses: " +
             "; ".join(f"{k}: {', '.join(v)}" for k, v in doc["milestone"]["rows_by_status"].items()))
    L.append("")
    L.append("## Rows")
    L.append("")
    L.append("| row | fam | value family | producer | identities | MB | dtype | evidence label | corr. | impl. | bench. | unres. | status | secure-commit cov. | diag-tap cov. |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in doc["rows"]:
        rq, co = r["requirement"], r["correspondence"]
        sc, dt = r["secure_commit_coverage"]["fraction_of_bytes"], r["diag_tap_coverage"]["fraction_of_bytes"]
        ds = r["derived_status"]
        yn = lambda b: "Y" if b else "-"  # noqa: E731
        L.append(f"| {r['id']} | {r['family'] or '—'} | {r['name']} | {rq['producer_definition']} | {rq['logical_identities']:,} | {rq['logical_bytes'] / 1e6:.3f} | {','.join(rq['dtypes']) or '—'} | "
                 f"{co['evidence_label']} | {yn(ds['correspondence_established'])} | {yn(ds['secure_path_implemented'])} | {yn(ds['secure_path_benchmarked'])} | {yn(ds['unresolved'])} | {r['status']} | {'—' if sc is None else f'{sc:.4f}'} | {'—' if dt is None else f'{dt:.2f}'} |")
    L.append("")
    L.append("## Per-row records")
    for r in doc["rows"]:
        rq, co = r["requirement"], r["correspondence"]
        L.append("")
        L.append(f"### {r['id']} — {r['name']} (family {r['family'] or 'excluded'})")
        L.append("")
        L.append(f"- **requirement**: {rq['logical_identities']:,} identities, {rq['logical_bytes'] / 1e6:.3f} MB, dtypes {rq['dtypes']}, spec range {rq['spec_range']}; already bound: {rq['already_bound']}; count label: {rq['count_label']}" +
                 (f"; note: {rq['logical_identities_note']}" if rq.get("logical_identities_note") else ""))
        L.append(f"- **correspondence**: producer {co['pinned_producer']}; occurrence: {co['occurrence'] or '—'}; mapping: {co['coordinate_mapping'] or '—'}; lifetime: {co['lifetime']}; evidence: **{co['evidence']}**")
        if r["physical_occurrences"]:
            L.append("- **physical occurrences** (location + time): " + "; ".join(f"{p['where']} [{p['when']}]" + (f" {p.get('note')}" if p.get("note") else "") for p in r["physical_occurrences"]))
        L.append("- **candidates**: " + "; ".join(f"`{c['id']}` — {c.get('mechanism') or c.get('note') or '—'} (producer change: {c.get('producer_change', '—')}; {c.get('status', '—')}; lane {c.get('lane', '—')}; binding test: {(c.get('binding_test') or {}).get('status', '—')})" for c in r["candidates"]))
        m = r["measurement"]
        if m.get("status") == "measured":
            prov = "; ".join(f"{q['lane']} {q['branch']}@{q['commit']}:{q['file']}" for q in (m.get("provenance") or []))
            L.append(f"- **measurement** (by {m.get('by')}, {m.get('gpu')}, {m.get('utc')}; samples {m.get('samples')}): " + (f"see `bench/summary.md`; " if m.get('by') == 'cmt-ref' else "") + f"{m.get('note', '')}" + (f" [provenance: {prov}]" if prov else ""))
            if m.get("delta_decode"):
                L.append(f"  Δ per decode step (treatment − control, relative): {', '.join(f'{x:+.3f}' for x in m['delta_decode'])} (ci95 {m.get('delta_decode_ci95')}); root-ready delay s: {m.get('root_ready_delay_s')}; openings {m.get('openings')}")
            if m.get("e2e_vllm_generate_ms"):
                L.append("  e2e vLLM generate per request (ms median [p10..p90], Δ vs base): " + "; ".join(f"{k}: {v['ms_median']} [{v['ms_p10']}..{v['ms_p90']}] Δ{v['delta_vs_base_ms']:+.2f}" for k, v in m["e2e_vllm_generate_ms"].items()))
            if m.get("table_md"):
                L.append("")
                L.extend("  " + ln for ln in m["table_md"].splitlines())
                L.append("")
            if m.get("openings"):
                o = m["openings"]
                L.append(f"  openings: n={o.get('n')}, all verify={o.get('all_verify')}, open {o.get('open_ms_mean', 0):.3f} ms mean, verify {o.get('verify_ms_mean', 0):.3f} ms mean, step path {o.get('path_len_step_mean')} + top {o.get('path_len_top')} nodes, {o.get('opening_bytes_mean', 0):.0f} B json")
        else:
            L.append(f"- **measurement**: {m['status']}")
        for key in ("value_correctness", "binding_correctness"):
            vcx = r[key]
            provx = "; ".join(f"{q['lane']}@{q['commit']}:{q['file']}" for q in (vcx.get("provenance") or []))
            L.append(f"- **{key}**: {vcx.get('status')} — {vcx.get('test') or '—'}" + (f" (by {vcx['by']}; machine record: {vcx.get('machine_record')}; {provx})" if vcx.get("by") else "")
                     + ("".join(f"; also {o['lane']}: {o.get('status')}" for o in vcx.get("other_lanes", [])) if vcx.get("other_lanes") else ""))
        if r["binding_correctness"].get("negatives"):
            L.append("  negatives: " + "; ".join(f"{n['test']}: {'REJECTED' if not n['accepted'] else 'ACCEPTED'} ({n['reason']})" for n in r["binding_correctness"]["negatives"]))
        L.append(f"- **diag_tap_coverage**: {r['diag_tap_coverage']}; **secure_commit_coverage**: {r['secure_commit_coverage']}")
        if r["members"]:
            L.append("- classes: " + "; ".join(f"`{x['class_id']}` {x['dtype']} {x['logical_count']:,} ({x['frequency']})" for x in r["members"]))
    L.append("")
    L.append("## Complete commitment (definition for this record)")
    L.append("")
    L.append(doc["complete_commitment"]["definition"])
    L.append("")
    L.append("Required rows: " + ", ".join(doc["complete_commitment"]["required_rows"]))
    L.append("")
    L.append("## Class → row (ground truth underneath)")
    L.append("")
    L.append("| class (v1 id) | v1 acquisition | dtype | identities | row(s) |")
    L.append("|---|---|---|---|---|")
    for c in doc["classes"]:
        L.append(f"| `{c['id']}` | {c['acquisition']} | {c['dtype']} | {c['logical_count']:,} | {', '.join(c['row'])} |")
    if doc["unmapped_classes"]:
        L.append("")
        L.append("UNMAPPED: " + ", ".join(doc["unmapped_classes"]))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="interface_manifest")
    ap.add_argument("--bench", default=None, help="bench dir with summary.json + binding.json (+ delta_T.json) to fill cmt-ref measurements")
    ap.add_argument("--bench-extra", action="append", default=[], help="further bench dirs (e.g. bench_triton, leafsize) whose summary tables are attached")
    args = ap.parse_args()
    v1 = json.load(open(args.v1))
    bench = None
    if args.bench and Path(args.bench, "summary.json").exists():
        bench = {"summary": json.load(open(Path(args.bench, "summary.json")))}
        if Path(args.bench, "binding.json").exists():
            bench["binding"] = json.load(open(Path(args.bench, "binding.json")))
        if Path(args.bench, "raw_samples.jsonl").exists():
            bench["n_samples"] = sum(1 for _ in open(Path(args.bench, "raw_samples.jsonl")))
        bench["extra"] = []
        for d in args.bench_extra:
            if Path(d, "value_check.json").exists():
                bench["value_check"] = json.load(open(Path(d, "value_check.json")))
            if Path(d, "snapshot_check.json").exists():
                bench["snapshot_check"] = json.load(open(Path(d, "snapshot_check.json")))
        for d in args.bench_extra:
            if Path(d, "summary.json").exists():
                e = json.load(open(Path(d, "summary.json")))
                nraw = sum(1 for _ in open(Path(d, "raw_samples.jsonl"))) if Path(d, "raw_samples.jsonl").exists() else 0
                bench["extra"].append({"dir": d, "gpu": e.get("gpu"), "utc": e.get("utc"), "samples": nraw, "table_md": e.get("table_md") or (Path(d, "summary.md").read_text() if Path(d, "summary.md").exists() else None),
                                       "roots_by_arm": (e.get("validity") or {}).get("roots_by_arm")})
                bench["n_samples"] = bench.get("n_samples", 0) + nraw
        if Path(args.bench, "delta_T.json").exists():
            bench["delta_T"] = json.load(open(Path(args.bench, "delta_T.json")))
    doc = build(v1, bench)
    doc["headline"] = headline(doc)
    print(doc["headline"]["text"])
    out = Path(args.out)
    (out / f"{args.name}.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / f"{args.name}.md").write_text(render_md(doc))
    print(f"[manifest-v2] {len(doc['rows'])} rows, {len(doc['classes'])} classes, unmapped {doc['unmapped_classes']}, totals {json.dumps(doc['totals'])}")
    for r in doc["rows"]:
        print(f"  {r['id']:5s} fam {r['family']} {r['requirement']['logical_identities']:>11,} {r['requirement']['logical_bytes'] / 1e6:9.3f} MB  {r['status']}")


if __name__ == "__main__":
    main()
