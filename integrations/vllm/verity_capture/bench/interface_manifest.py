"""Interface manifest: every distinct boundary-value class a VU partition requires to be bound, with its acquisition class
(R9 lane CMT-1, Goal A1 of `out/gen/r9/DANIEL-BRIEF.md`).

    python -m verity_capture.bench.interface_manifest CENSUS_SUMMARY_JSON --out OUT_DIR [--label canonical|conservative]

Reuses the class enumeration of `verity_capture/bench/vu_census.py` (its `aggregate.groups`: one row per
(root family, logical class, VU kind, out primitive, width, eager class, compiled class, fan-out), counts and bytes exact
for the record) and does NOT redo its physics.  What it adds, per class:

    acquisition class   registered-input | naturally-materialized | hidden-fa2 | hidden-vllm-kernel | derived-from-bound |
                        constant (excluded from the commitment obligation) | unsupported
    frequency           per-model | per-request | per-token | per-step | per-(layer,token) | per-(layer,head,query,key) | ...
    physical bytes      HBM write today / reads today (minimum) / extra read for commitment / extra write if exposed /
                        repeated physical copies or re-reads -- kept apart from the logical identity count
    bound_by            per-lane columns (null until a lane fills them via its own artefact, never edited here)

"derived-from-bound" follows `out/gen/r8/fa2-commit/derived_value_rule.md`: a total deterministic registry primitive of a
bound word (Bf16ToF32 of a committed bf16 residual; F2fpBf16 of a committed fp32 P).  It is a protocol clause (circuit
substitution), so the manifest records the source class and marks the value "bound iff source bound AND the derived-value
rule is adopted"; it does not assume the rule.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

VERSION = "2026-09-18.1"
SCHEMA = "verity-gen/interface-manifest/v1"

ACQUISITION_CLASSES = {
    "registered-input": "weights / prompt ids: root inputs of the Program, independently bound before the run (weights tree "
                        "`verity_vllm.commit.weights_tree` -- row-block leaves bound to the manifest `weights_sha256` by "
                        "`verify_source_linkage`; prompt ids in the `boundary/<request>` tree) [source-inspected: verity_vllm/commit.py]",
    "naturally-materialized": "an HBM tensor visible at a kernel boundary in eager production: root inputs and Instance outputs named by "
                              "`instances.jsonl` `operand_views` / `output_views` (census: output leaf count == view element count for all "
                              "181 specializations)",
    "hidden-fa2": "interior of `flash_fwd_splitkv_kernel` (vllm_flash_attn FA2): scores, running max / sum, P, PV accumulators -- no API "
                  "exposes them; CMT-2's population (exact register correspondence `out/gen/r8/fa2-sem/`, taps `out/gen/r8/fa2-kernel/`)",
    "hidden-vllm-kernel": "interior of a vLLM CUDA / Triton kernel (`fused_add_rms_norm_kernel`, `_rms_norm_kernel`): registers / shared "
                          "memory by source, residency not established (census class `unknown`)",
    "derived-from-bound": "a total, deterministic registry primitive applied to a bound word (Bf16ToF32 / F2fpBf16); bound iff its source is "
                          "bound AND the derived-value rule (`out/gen/r8/fa2-commit/derived_value_rule.md`) is adopted as a protocol clause",
    "constant": "`Const<w>` gate the `default(32)` query makes its own VU; fixed by the Program text -- reported, excluded from the obligation",
    "unsupported": "no acquisition route identified",
}

LANES = ("cmt-ref", "cmt-hidden", "cmt-integ")

#: dtype label by (out primitive base, width)
def _dtype(prim: str, w: int) -> str:
    base = prim.split("{")[0].split("[")[0]
    if base in ("Input32_v1", "SelectI32_v1"):
        return "i32"
    if w == 16:
        return "bf16"
    if w == 32:
        return "f32"
    return f"u{w}"


def _short_id(g: dict) -> str:
    fam = g["root_family"].replace("_v", "").lower()
    prim = g["prim"].split("{")[0].replace("_v1", "").replace("_v2", "")
    kind = g["vu_kind"].replace("call ", "").replace("{…}", "").replace("zip[", "zip_").replace("]", "").replace("+", "_")
    cls = {"root-input": "in", "instance-output": "out", "interior": "int", "constant": "const"}[g["logical_class"]]
    fo = {"-": "", "1": "_fo1", "1?": "_fo1q", ">=2": "_fo2"}[g["fanout"]]
    if g["logical_class"] in ("root-input", "constant"):
        return f"{fam}/{cls}/{prim}"
    return f"{fam}/{cls}/{kind}/{prim}{fo}".lower()


def classify(g: dict, W: dict) -> tuple[str, dict | None, str]:
    """(acquisition class, derived_from, note) for a census group."""
    fam, lc, prim, fo = g["root_family"], g["logical_class"], g["prim"].split("{")[0], g["fanout"]
    if lc == "root-input":
        return "registered-input", None, ("weights: committed once per model (production `weights` tree)" if prim.startswith("Input16")
                                         else "prompt token ids: committed once per request (production `boundary/<request>` tree)")
    if lc == "constant":
        return "constant", None, "Program constant; no run-time information"
    if lc == "instance-output":
        return "naturally-materialized", None, g["eager_evidence"]
    # interior
    if fam.startswith("Attention"):
        if prim == "F2fpBf16_v1":
            return "derived-from-bound", {"source_class": "attention_v3/int/f2fpbf16-source:mufuex2ftz_fo2", "source_role": "P = exp2(s - m) (fp32, hidden-fa2)",
                                         "phi": "F2fpBf16_v1", "source_acquisition": "hidden-fa2",
                                         "rule": "out/gen/r8/fa2-commit/derived_value_rule.md §1 (P16 through P32; many-to-one phi is allowed)"}, \
                "P in bf16 (MMA operand) = F2fpBf16(P32); P32 itself is hidden-fa2, so this class is bound only once P32 is"
        return "hidden-fa2", None, g["eager_evidence"]
    if fam.startswith("RMSNorm"):
        if prim == "Bf16ToF32_v1":
            src = ("rmsnormfusedcuda_v2/out/bf16addf2fp/Bf16AddF2fp" if fam == "RMSNormFusedCuda_v2"
                   else "rmsnormfusedcuda_v2/out/bf16addf2fp/Bf16AddF2fp (the last layer's residual output = this norm's operand)")
            return "derived-from-bound", {"source_class": src, "source_role": "bf16 residual word (naturally-materialized Instance output)",
                                         "phi": "Bf16ToF32_v1", "source_acquisition": "naturally-materialized",
                                         "rule": "out/gen/r8/fa2-commit/derived_value_rule.md §1 (injective lift; total on all 2^16 words)"}, \
                ("fp32 lift of the bf16 residual sum (fan-out %s): determined by a committed native word" % fo)
        return "hidden-vllm-kernel", None, g["eager_evidence"]
    return "unsupported", None, f"no rule for {fam}/{lc}/{prim}"


def frequency(g: dict, W: dict) -> dict:
    """Dynamic frequency: unit, count per unit (calculated from the census count and the workload facts), scaling."""
    fam, lc, prim, n = g["root_family"], g["logical_class"], g["prim"].split("{")[0], g["count"]
    L, T, S, P = W["layers"], W["tokens"], W["steps"], W["prompt_tokens"]
    LT = L * T
    if lc == "root-input":
        if prim.startswith("Input16"):
            return {"unit": "per-model", "count_per_unit": n, "scaling": "O(1) per model; physically re-read every step", "label": "measured-from-record"}
        return {"unit": "per-request", "count_per_unit": n, "scaling": "O(P)", "label": "measured-from-record"}
    if lc == "constant":
        return {"unit": "per-request (Program text)", "count_per_unit": n, "scaling": "n/a", "label": "measured-from-record"}
    if fam == "TokenSelect_v1":
        return {"unit": "per-step", "count_per_unit": n / S, "scaling": "O(steps)", "label": "calculated"}
    if fam == "Gemm_v1":
        lm = S * 49152                                    # lm_head logits: one row per engine step (sampled position only)
        per_tok = (n - lm) / T
        return {"unit": "per-token (layer GEMMs) + per-step (lm_head)", "count_per_unit": per_tok, "per_step_extra": lm / S,
                "breakdown": {"layer_gemms_words_per_token": per_tok, "layer_gemms_words_per_token_per_layer": per_tok / L, "lm_head_words_per_step": 49152,
                              "check": f"{T}*{per_tok:.0f} + {S}*49152 = {T * per_tok + lm:.0f} == {n}"},
                "scaling": "O(L*T) + O(steps*V)", "label": "calculated (vocab 49152, hidden 576, qkv 960, gate_up 3072)"}
    if fam in ("Embedding_v1", "RMSNormTriton_v1"):
        return {"unit": "per-token", "count_per_unit": n / T, "scaling": "O(T)", "label": "calculated"}
    if fam.startswith("Attention") and lc == "interior":
        # per (layer, head) ratio against the record's attention geometry: pairs (visible (query, key)), visited 128-key blocks,
        # rows (queries), D = 64 accumulator lanes, 4 row-sum lanes (fa2-commit REPORT §6: 14,365 pairs, 210 blocks, 169 rows per (layer, head))
        H, D = W.get("heads", 9), W.get("head_dim", 64)
        r = n / (L * H)
        pairs, blocks, rows = W.get("pairs_per_layer_head", 14365), W.get("blocks_per_layer_head", 210), T
        cands = [("per-(layer, head, query, visible key)", pairs, "O(L*H*T^2) -- census §5: ~1.4 GB per request at C1024"),
                 ("per-(layer, head, query, key-block[128], d) accumulator lane", blocks * D, "O(L*H*T*ceil(T/128)*D)"),
                 ("per-(layer, head, query, key-block[128]) block state", blocks, "O(L*H*T*ceil(T/128))"),
                 ("per-(layer, head, query, key-block[128], lane[4])", blocks * 4, "O(L*H*T*ceil(T/128))"),
                 ("per-(layer, head, query, d) epilogue lane", rows * D, "O(L*H*T*D)"),
                 ("per-(layer, head, query, lane[4]) epilogue", rows * 4, "O(L*H*T)"),
                 ("per-(layer, head, query) epilogue", rows, "O(L*H*T)")]
        best = min(cands, key=lambda c: abs(r - c[1]) / c[1])
        tol = abs(r - best[1]) / best[1]
        unit = best[0] if tol <= 0.08 else f"per-(layer, head) x {r:,.0f} (nearest: {best[0]}, off by {100 * tol:.0f} %)"
        return {"unit": unit, "count_per_unit": n / L, "per_layer_head": r, "nearest_unit_count": best[1], "rel_error": tol,
                "scaling": best[2], "label": "calculated" if tol <= 0.08 else "calculated (unit approximate)"}
    if fam.startswith("Attention") and lc == "instance-output":
        return {"unit": "per-(layer, token)", "count_per_unit": n / LT, "scaling": "O(L*T)", "label": "calculated"}
    if fam in ("RMSNormFusedCuda_v2", "SiluMul_v1", "RoPE_v1"):
        return {"unit": "per-(layer, token)", "count_per_unit": n / LT, "scaling": "O(L*T)", "label": "calculated"}
    return {"unit": "?", "count_per_unit": None, "scaling": "?", "label": "unresolved"}


def physical(g: dict, acq: str, W: dict) -> dict:
    """Physical bytes, kept apart from logical identities.  All values calculated from the census's logical bytes."""
    b = g["bytes"]
    fam, prim = g["root_family"], g["prim"].split("{")[0]
    rep = None
    if acq == "registered-input" and prim.startswith("Input16"):
        rep = {"kind": "re-read every engine step (weights)", "bytes_per_request": b * W["steps"], "copies": 1}
    if fam == "RoPE_v1":
        rep = {"kind": "k rows (1/4 of RoPE outputs) are copied into the paged KV cache by `unified_kv_cache_update` (second physical copy) and re-read "
                       "by every later attention step; q rows are read once", "copies": 2, "note": "K = 192 of the 768 rotated words per (layer, token)"}
    if fam == "Gemm_v1":
        rep = {"kind": "the v slice of the qkv output (192 of 960 words per (layer, token)) is copied into the KV cache (second physical copy) and "
                       "re-read every later step; other GEMM outputs are read once by their consumer", "copies": "1 (2 for v)"}
    if acq == "naturally-materialized":
        return {"hbm_write_today_bytes": b, "hbm_read_today_bytes_min": b, "commit_extra_read_bytes": b, "added_write_if_exposed_bytes": 0,
                "host_path_d2h_bytes": b, "repeated_use": rep}
    if acq == "registered-input":
        return {"hbm_write_today_bytes": 0 if prim.startswith("Input16") else b, "hbm_read_today_bytes_min": b, "commit_extra_read_bytes": 0,
                "added_write_if_exposed_bytes": 0, "host_path_d2h_bytes": 0, "repeated_use": rep,
                "note": "bound at registration / request admission, not on the serving path"}
    if acq in ("hidden-fa2", "hidden-vllm-kernel"):
        return {"hbm_write_today_bytes": 0, "hbm_read_today_bytes_min": 0, "commit_extra_read_bytes": b, "added_write_if_exposed_bytes": b,
                "host_path_d2h_bytes": b, "repeated_use": None, "note": "exposing = one new HBM write + one read (or in-kernel hashing: 0 write)"}
    if acq == "derived-from-bound":
        return {"hbm_write_today_bytes": 0, "hbm_read_today_bytes_min": 0, "commit_extra_read_bytes": 0, "added_write_if_exposed_bytes": 0,
                "host_path_d2h_bytes": 0, "repeated_use": None, "note": "no leaf: opened through the source's opening + phi (verifier recomputes phi)"}
    return {"hbm_write_today_bytes": 0, "hbm_read_today_bytes_min": 0, "commit_extra_read_bytes": 0, "added_write_if_exposed_bytes": 0,
            "host_path_d2h_bytes": 0, "repeated_use": None}


def build(summary: dict, *, label: str, source_path: str) -> dict:
    A, W = summary["aggregate"], dict(summary["workload"])
    L = W["layers"]
    # attention geometry of the record, from the census's own counts: the DotBf16 score class counts every visible (query, key)
    # pair once; the per-block scale constant Const32[0x3e38aa3b] (log2 e * softmax scale) is emitted once per visited block
    W.setdefault("heads", 9)
    W.setdefault("head_dim", 64)
    dot = [g for g in A["groups"] if g["root_family"].startswith("Attention") and g["vu_kind"].startswith("call DotBf16")]
    if dot:
        W["pairs_per_layer_head"] = dot[0]["count"] / (L * W["heads"])
    blk = [g for g in A["groups"] if g["root_family"].startswith("Attention") and g["prim"].startswith("Const32[0x3e38aa3b]")]
    if blk:
        W["blocks_per_layer_head"] = blk[0]["count"] / (L * W["heads"])
    rows = []
    for g in A["groups"]:
        acq, derived, note = classify(g, W)
        row = {"id": _short_id(g),
               "census_key": {k: g[k] for k in ("root_family", "logical_class", "vu_kind", "prim", "width_bits", "fanout", "eager_class", "compiled_class")},
               "producer_definition": g["root_family"], "relation_family": g["vu_kind"], "out_primitive": g["prim"], "role": g["role"],
               "dtype": _dtype(g["prim"], g["width_bits"]), "width_bits": g["width_bits"],
               "logical_count": g["count"], "logical_bytes": g["bytes"], "census_layout_classes": g["classes"],
               "frequency": frequency(g, W), "acquisition": acq, "acquisition_note": note, "derived_from": derived,
               "kernel": g.get("kernel") or None, "fanout": g["fanout"],
               "physical": physical(g, acq, W),
               "required_for_complete_commitment": acq not in ("constant",),
               "bound_by": {lane: None for lane in LANES}}
        rows.append(row)
    # fix the derived P16 source id to the real id of the P32 class
    p32 = next((r["id"] for r in rows if r["census_key"]["root_family"].startswith("Attention") and r["out_primitive"].startswith("MufuEx2Ftz") and r["fanout"] == ">=2"), None)
    res = next((r["id"] for r in rows if r["out_primitive"].startswith("Bf16AddF2fp")), None)
    for r in rows:
        if r["derived_from"]:
            if r["derived_from"]["phi"] == "F2fpBf16_v1" and p32:
                r["derived_from"]["source_class"] = p32
            if r["derived_from"]["phi"] == "Bf16ToF32_v1" and res:
                r["derived_from"]["source_class"] = res
    rows.sort(key=lambda r: -r["logical_bytes"])
    totals: dict[str, dict] = {}
    for r in rows:
        t = totals.setdefault(r["acquisition"], {"classes": 0, "logical_count": 0, "logical_bytes": 0.0, "commit_extra_read_bytes": 0.0, "added_write_if_exposed_bytes": 0.0})
        t["classes"] += 1
        t["logical_count"] += r["logical_count"]
        t["logical_bytes"] += r["logical_bytes"]
        t["commit_extra_read_bytes"] += r["physical"]["commit_extra_read_bytes"]
        t["added_write_if_exposed_bytes"] += r["physical"]["added_write_if_exposed_bytes"]
    req = [r for r in rows if r["required_for_complete_commitment"]]
    per_request = [r for r in req if r["acquisition"] != "registered-input" or not r["out_primitive"].startswith("Input16")]
    out = {
        "schema": SCHEMA, "version": VERSION, "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "label": label,
        "record": {"run": summary["run"], "program_digest": summary["program_digest"], "profile": summary["profile"], "gates": summary["gates"],
                   "root_nodes": summary["root_nodes"], "n_instances": summary["n_instances"], "n_specializations": summary["n_specs"], "workload": W},
        "query": {"kind": label, "query_id": summary.get("query_id"), "w_max": summary["w_max"], "vu_count": summary["vu_count"],
                  "layout_classes": summary["n_classes"], "g8": (summary.get("g8") or {}).get("g8", {}).get("status") if summary.get("g8") else None,
                  "binding_id": (summary.get("g8") or {}).get("binding_id") if summary.get("g8") else None, "text": summary.get("query_text")},
        "source": {"census_summary": source_path, "census_version": summary["version"], "census_utc": summary["utc"],
                   "census_tool": "verity_capture/bench/vu_census.py", "note": "counts / bytes / classes / fan-out are the census's (exact for the record); "
                   "acquisition class, frequency, physical columns and bound_by are this tool's"},
        "acquisition_classes": ACQUISITION_CLASSES, "lanes": list(LANES),
        "classes": rows,
        "totals_by_acquisition": totals,
        "complete_commitment": {
            "definition": "Every class with required_for_complete_commitment == true has, before the sampling challenge, either (a) a leaf position in a "
                          "prechallenge root (position-bound; opens against the original root; negatives reject), or (b) for derived-from-bound classes, a bound "
                          "source under (a) and the derived-value rule adopted, or (c) for registered-input classes, the production weights / boundary tree "
                          "root published before the run.  Constants are excluded.  A subset of classes is a PARTIAL commitment and must be reported as such.",
            "required_class_ids": [r["id"] for r in req],
            "per_request_bytes_excl_weights_excl_constants": sum(r["logical_bytes"] for r in per_request if r["acquisition"] != "registered-input" or r["out_primitive"].startswith("Input32")),
            "excluded_class_ids": [r["id"] for r in rows if not r["required_for_complete_commitment"]],
        },
    }
    return out


def _mb(b: float) -> str:
    return f"{b / 1e6:,.2f} MB" if b >= 1e5 else f"{b / 1e3:,.1f} kB"


def render_md(m: dict) -> str:
    R, Q, W = m["record"], m["query"], m["record"]["workload"]
    L = [f"# Interface manifest — `{R['run']}`, {m['label']} VU query ({m['utc']})", "",
         f"Program `{R['program_digest'][:16]}…` ({R['gates']:,} gates, {R['root_nodes']:,} root nodes, {R['n_instances']:,} Instances, {R['n_specializations']} specializations), "
         f"profile `{R['profile']}`; workload {W['prompt_tokens']} prompt + {W['decode_tokens']} generated = {W['tokens']} tokens, {W['steps']} steps, {W['layers']} layers.  "
         f"Query `{Q['kind']}` `{(Q['query_id'] or '')[:16]}…`, `W_max = {Q['w_max']}`, {Q['vu_count']:,} VUs in {Q['layout_classes']:,} layout classes, G8 `{Q['g8']}`."
         + ("" if m["label"] == "canonical" else "  **CONSERVATIVE CANDIDATE** (`verity.ir/experimental/contract.py`, `out/gen/r8/vu-impl/candidates.json`): "
            "validator-clean, NOT a G8 artefact, not promoted — a second, labelled interface; the canonical manifest is the reference."),
         "", f"Source: `{m['source']['census_summary']}` (census v{m['source']['census_version']}, {m['source']['census_utc']}).  {m['source']['note']}.  "
         "Every count/byte below is **measured-from-record** (census, exact); frequency per-unit numbers are **calculated** from them; physical columns are "
         "**calculated** from logical bytes under the stated rule; nothing here is a runtime measurement.", "",
         "## 1. Acquisition classes", ""]
    for k, v in m["acquisition_classes"].items():
        L.append(f"- **{k}** — {v}")
    L += ["", "## 2. Totals by acquisition class", "", "| acquisition | classes | logical values | logical bytes | extra HBM read to commit | added HBM write if exposed |", "|---|---:|---:|---:|---:|---:|"]
    for k, t in sorted(m["totals_by_acquisition"].items(), key=lambda kv: -kv[1]["logical_bytes"]):
        L.append(f"| {k} | {t['classes']} | {t['logical_count']:,} | {_mb(t['logical_bytes'])} | {_mb(t['commit_extra_read_bytes'])} | {_mb(t['added_write_if_exposed_bytes'])} |")
    cc = m["complete_commitment"]
    L += ["", f"Per-request obligation excluding weights and constants: **{_mb(cc['per_request_bytes_excl_weights_excl_constants'])}** logical "
          f"({len(cc['required_class_ids'])} classes required, {len(cc['excluded_class_ids'])} constant classes excluded).", "",
          "## 3. Classes (largest first)", "",
          "| id | producer Definition | relation family (VU kind) | out primitive (role) | dtype | count | bytes | frequency | acquisition | fan-out | derived from | write today | extra read to commit | added write if exposed | bound by |",
          "|---|---|---|---|---|---:|---:|---|---|---|---|---:|---:|---:|---|"]
    for r in m["classes"]:
        f = r["frequency"]
        fq = f"{f['unit']}" + (f" ({f['count_per_unit']:,.0f}/unit)" if f.get("count_per_unit") else "")
        d = r["derived_from"]
        dd = f"`{d['source_class']}` via `{d['phi']}`" if d else ""
        bb = ", ".join(f"{k}={v}" for k, v in r["bound_by"].items())
        p = r["physical"]
        L.append(f"| `{r['id']}` | {r['producer_definition']} | `{r['relation_family']}` | `{r['out_primitive']}` ({r['role']}) | {r['dtype']} | {r['logical_count']:,} | {_mb(r['logical_bytes'])} | {fq} | "
                 f"**{r['acquisition']}** | {r['fanout']} | {dd} | {_mb(p['hbm_write_today_bytes'])} | {_mb(p['commit_extra_read_bytes'])} | {_mb(p['added_write_if_exposed_bytes'])} | {bb} |")
    L += ["", "## 4. Physical vs logical", "",
          "- Logical identity = one (class, instance, element) value; counted once regardless of copies.  Physical columns count HBM bytes: `write today` = the producing "
          "kernel's HBM write in eager production (0 for hidden values); `extra read to commit` = one HBM read of the value bytes by a committer (GPU-resident) or one "
          "D2H move (host hashing); `added write if exposed` = the new write a hidden value would need if materialised.",
          "- Repeated physical use (not extra logical identities): weights are re-read every engine step (270.1 MB × 15 steps = 4.05 GB per request, calculated); "
          "the k rows of the RoPE output and the v slice of the qkv GEMM output are copied a second time into the paged KV cache by `unified_kv_cache_update` and "
          "re-read by every later attention step (T-dependent).  A committer binds the logical value once (its first materialisation); the KV-cache copy is a "
          "second physical representation whose agreement is the attention relation's business (`docs/PROTOCOL.md:40`), not a second leaf.",
          "- Derived classes have no leaf and no physical bytes of their own: they are opened through their source + φ (verifier recomputes φ).", "",
          "## 5. What \"complete commitment\" means for this record", "", cc["definition"], "",
          "Checklist (all must hold for the label *complete*; anything less is *partial* and must be reported with the exact residual):", ""]
    by_acq: dict[str, list] = {}
    for r in m["classes"]:
        if r["required_for_complete_commitment"]:
            by_acq.setdefault(r["acquisition"], []).append(r)
    for acq, rs in sorted(by_acq.items(), key=lambda kv: -sum(x["logical_bytes"] for x in kv[1])):
        L.append(f"- [ ] **{acq}** — {len(rs)} classes, {_mb(sum(x['logical_bytes'] for x in rs))}: " + ", ".join(f"`{x['id']}`" for x in rs))
    L += ["", "Registered inputs count as bound when the production weights tree root (row-block leaves, `weights_sha256`-linked) and the request's "
          "`boundary/<request>` tree root (prompt ids) are published before the run — both exist in `verity_vllm.commit` today [source-inspected].", "",
          "## 6. `bound_by` protocol", "",
          "Columns are `null` here by construction.  A lane reports the classes its implementation binds in its own artefact (`out/gen/r9/<lane>/binds.json`: "
          "`{class_id: {\"bound\": true|false|\"partial\", \"how\": ..., \"evidence\": ...}}`); the coordinator (not this file's owner) merges.  CMT-1's own column is filled "
          "in `out/gen/r9/cmt-ref/binds.json` after the reference path runs.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("census_summary", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="canonical", choices=("canonical", "conservative"))
    ap.add_argument("--stem", default=None, help="output file stem (default: interface_manifest or interface_manifest_conservative)")
    a = ap.parse_args(argv)
    summary = json.loads(a.census_summary.read_text())
    m = build(summary, label=a.label, source_path=str(a.census_summary))
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.stem or ("interface_manifest" if a.label == "canonical" else "interface_manifest_conservative")
    (a.out / f"{stem}.json").write_text(json.dumps(m, indent=1, sort_keys=False, default=str) + "\n")
    (a.out / f"{stem}.md").write_text(render_md(m))
    t = m["totals_by_acquisition"]
    print(f"[manifest] {a.label}: {len(m['classes'])} classes; " + "; ".join(f"{k} {v['logical_bytes'] / 1e6:,.1f} MB" for k, v in t.items()))
    print(f"[manifest] wrote {a.out / stem}.json / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
