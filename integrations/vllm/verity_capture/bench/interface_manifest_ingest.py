"""Manifest v2.1: ingest the OTHER lanes' measurement / acceptance records into the value-family rows and re-derive the funnel.

    python -m verity_capture.bench.interface_manifest_ingest out/gen/r9/cmt-ref/interface_manifest.json --out out/gen/r9/cmt-ref

Sources (read with `git show <branch>:<file>`, provenance = lane, branch, commit, file kept on every ingested record):
  CMT-3  lane/r9-cmt-integ  coverage_complete_reference.json (row verdicts), raw/g_b0_fast_hidden_vc/{summary.json (5 interleaved
         control/treatment pairs -> delta samples), runs.jsonl (per-run commit record: openings verified, bytes bound, root-ready),
         value_check.json (tap-equal + re-exec by module class)}
  CMT-2  lane/r9-cmt-hidden  rows_5_8.json (rows R5-R8: four records + acceptance), cost_table.json (validity: oracle bit-exact
         dumps, CPU negatives list, GPU-tree openings; e2e vLLM generate medians/p10/p90 of 20 reps), rows_10ab.json (R10a/R10b
         derived-value route, exact-model checks + forged-word negatives; results/l40s_cmt2c/rows10_derived.json)
Rules are the manifest's own (`derive_status`): correspondence <= label in {measured-correspondence, source-inspected}; implemented <=
a candidate with a PASSING binding test that has a MACHINE-READABLE record (a JSON in the lane's tree with the verdict/counts; prose in
a .md is NOT enough); benchmarked <= implemented and a measurement record with real samples.  Rows whose lane record lacks a field are
left at the lower status and listed in `ingest.not_upgraded`.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from verity_capture.bench import interface_manifest_v2 as V2  # noqa: E402

LANES = {
    "cmt-integ": {"branch": "lane/r9-cmt-integ", "dir": "out/gen/r9/cmt-integ"},
    "cmt-hidden": {"branch": "lane/r9-cmt-hidden", "dir": "out/gen/r9/cmt-hidden"},
}


def _git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def _fetch(lane: str, rel: str, commit: str):
    branch = LANES[lane]["branch"]
    path = f"{LANES[lane]['dir']}/{rel}"
    text = _git(["show", f"{branch}:{path}"])
    prov = {"lane": lane, "branch": branch, "commit": commit, "file": path}
    if rel.endswith(".jsonl"):
        return [json.loads(l) for l in text.splitlines() if l.strip()], prov
    return json.loads(text), prov


def ingest(doc: dict) -> dict:
    rows = {r["id"]: r for r in doc["rows"]}
    commits = {lane: _git(["rev-parse", "--short", LANES[lane]["branch"]]).strip() for lane in LANES}
    log = {"lanes": {lane: {"branch": LANES[lane]["branch"], "commit": commits[lane]} for lane in LANES}, "ingested": [], "not_upgraded": []}

    # ---------------- CMT-3: complete reference (native_host fast + hidden M1 stream), run g_b0_fast_hidden_vc ---------------------
    ccr, p_ccr = _fetch("cmt-integ", "coverage_complete_reference.json", commits["cmt-integ"])
    run_dir = "raw/g_b0_fast_hidden_vc"
    summ, p_sum = _fetch("cmt-integ", f"{run_dir}/summary.json", commits["cmt-integ"])
    runs, p_runs = _fetch("cmt-integ", f"{run_dir}/runs.jsonl", commits["cmt-integ"])
    vc, p_vc = _fetch("cmt-integ", f"{run_dir}/value_check.json", commits["cmt-integ"])
    treat = [r for r in runs if r["arm"] != "control"]
    openings = [r["commit"]["openings"] for r in treat if r.get("commit")]
    op_n, op_ok = sum(o["n"] for o in openings), sum(o["verified"] for o in openings)
    integ_meas = {
        "status": "measured", "by": "cmt-integ", "samples": len(summ["delta_decode"]), "gpu": summ["versions"]["gpu"]["name"], "sm": "sm89", "utc": summ["utc"],
        "committer": summ["declaration"]["committer"], "pairs": summ["pairs"], "delta_wall": summ["delta_wall"], "delta_decode": summ["delta_decode"],
        "delta_decode_ci95": summ["delta_decode_ci95"], "delta_prefill": summ["delta_prefill"], "root_ready_delay_s": summ["root_ready_delay_s"],
        "spans_mean_s": summ["spans_mean_s"], "bytes_bound_per_run": treat[0]["commit"]["bytes_bound"], "leaves_per_run": treat[0]["commit"]["leaves"],
        "openings": {"n": op_n, "verified": op_ok, "bytes_mean": openings[0]["bytes_mean"], "generate_s_per_run": openings[0]["generate_s"]},
        "weight_registration": treat[0]["commit"]["weight_registration"], "provenance": [p_sum, p_runs],
        "note": "whole-committer costs (all rows bound by native_host fast + the hidden M1 stream together); per-row costs are not separable",
    }
    integ_bind = {"status": "pass" if op_n and op_ok == op_n else "FAIL / incomplete", "machine_record": True,
                  "test": f"openings generated in-run and verified against the ORIGINAL run root: {op_ok}/{op_n} over {len(openings)} runs (runs.jsonl commit.openings); "
                          "negatives: pytest verity_capture/experimental/commit_integ/tests (test_security, test_native_host_security) — result reported in delta_table.md as prose (67 passed), not ingested as evidence",
                  "provenance": [p_runs]}
    integ_val = {"status": "pass" if vc["verdict"] == "PASS" else vc["verdict"], "machine_record": True,
                 "test": f"tap-equal {vc['tap_equal']}/{vc['tensors']} tensors, re-executed {vc['reexec_equal']}/{vc['reexec_tensors']} (value_check.json by module class)",
                 "by_class": vc["by_class"], "provenance": [p_vc]}
    integ_status = {r["row"]: r for r in ccr["rows"]}
    for rid in ("R1a", "R1b", "R2a", "R2b", "R2c", "R2d", "R2e", "R2f", "R2g", "R2h", "R2i"):
        row = rows[rid]
        cand = {"id": "cmt-integ native_host (fast) — complete reference", "lane": "cmt-integ", "stage": "built",
                "mechanism": "forward hooks -> pinned staging arena -> host SHA-256 pos_leaf (64 KB chunks) per step, bind_root per step, run root; weights registered once per process",
                "producer_change": "none (hooks only)", "status": integ_status[rid]["status"], "binding_test": dict(integ_bind), "value_test": dict(integ_val),
                "coverage_row": {k: integ_status[rid][k] for k in ("diag_tap_coverage", "secure_commit_coverage", "residual_bytes", "status")}, "provenance": [p_ccr]}
        row["candidates"].append(cand)
        if row["measurement"].get("status") != "measured":
            row["measurement"] = dict(integ_meas)
        else:
            row["measurement"].setdefault("other_lanes", []).append(dict(integ_meas))
        _merge_check(row, "value_correctness", integ_val, "cmt-integ")
        _merge_check(row, "binding_correctness", integ_bind, "cmt-integ")
        log["ingested"].append({"row": rid, "lane": "cmt-integ", "records": ["candidate", "measurement", "value_correctness", "binding_correctness"]})

    # R4 / R10c: derived-value rule adopted by CMT-3 (coverage row secure-derived). Machine test record: row4_derived_verdict.json
    # (committed 19:19Z, 45c706d79+). Without it the rows stay at correspondence-only; with a PASS/PASS record and n_pass == n_tests
    # they become implemented (binding test) and benchmarked (the derived class commits 0 extra bytes; it is inside the measured
    # complete-reference run g_b0_fast_hidden_vc, so the measurement record is that run's).
    try:
        verd, p_verd = _fetch("cmt-integ", "row4_derived_verdict.json", commits["cmt-integ"])
        verd_rows = {v["row"]: v for v in verd.get("rows", [])}
    except Exception as exc:  # noqa: BLE001
        verd_rows, p_verd = {}, None
        log["not_upgraded"].append({"row": "R4/R10c", "lane": "cmt-integ", "reason": f"row4_derived_verdict.json not readable: {exc}"})
    for rid, src in (("R4", "R2e"), ("R10c", "R2b")):
        row = rows[rid]
        cr = integ_status[rid]
        v = verd_rows.get(rid)
        ok = bool(v) and v.get("value_correctness") == "PASS" and v.get("binding_correctness") == "PASS" and v.get("n_pass", 0) == v.get("n_tests", -1) > 0
        if ok:
            bt = {"status": "pass", "machine_record": True, "n_tests": v["n_tests"], "n_pass": v["n_pass"], "test": v["test_file"],
                  "negatives": [{"test": n, "accepted": False, "reason": "rejected (pod run, row4_derived_verdict.json)"} for n in v.get("negatives", [])],
                  "evidence": v.get("binding_correctness_evidence"), "pod": v.get("pod"), "gpu": v.get("gpu"),
                  "utc": v.get("timestamp"), "provenance": [p_verd]}
            vt = {"status": "pass", "machine_record": True, "test": v["test_file"], "positives": v.get("positives", []),
                  "evidence": v.get("value_correctness_evidence"), "provenance": [p_verd]}
            row["candidates"].append({"id": "cmt-integ derived-value rule (Bf16ToF32 injective lift)", "lane": "cmt-integ", "stage": "built",
                                      "mechanism": v.get("rule") or cr["note"], "occurrence": v.get("occurrence"), "source_row": src,
                                      "producer_change": "none", "status": v.get("status", cr["status"]), "opening": v.get("opening"),
                                      "binding_test": bt, "value_test": vt,
                                      "coverage_row": {k: cr[k] for k in ("diag_tap_coverage", "secure_commit_coverage", "residual_bytes", "status")},
                                      "provenance": [p_ccr, p_verd]})
            row["measurement"] = {"status": "measured", "by": "cmt-integ", "samples": summ.get("n_pairs", len(runs)), "gpu": "NVIDIA L40S", "sm": "sm89",
                                  "run": run_dir, "provenance": [p_sum, p_runs, p_verd],
                                  "note": "derived-value route: nothing extra committed (0 B), so the class's ΔT ≡ 0 by construction; it is bound through its "
                                          f"source row {src} inside the measured complete-reference run {run_dir}; opening = source chunk + path + identity"}
            _merge_check(row, "value_correctness", vt, "cmt-integ")
            _merge_check(row, "binding_correctness", bt, "cmt-integ")
            row["status"] = f"{v.get('status', 'secure-derived')} [cmt-integ verdict {v['n_pass']}/{v['n_tests']}]"
            log["ingested"].append({"row": rid, "lane": "cmt-integ", "records": ["candidates", "measurement", "value_correctness", "binding_correctness"],
                                    "machine_record": "row4_derived_verdict.json"})
        else:
            row["candidates"].append({"id": "cmt-integ derived-value rule (Bf16ToF32 injective lift)", "lane": "cmt-integ", "stage": "built",
                                      "mechanism": cr["note"], "producer_change": "none", "status": cr["status"],
                                      "binding_test": {"status": "no machine record", "machine_record": False,
                                                       "test": "verity_capture/experimental/commit_integ/tests/test_row4_derived.py; verdict JSON missing or not PASS"},
                                      "coverage_row": {k: cr[k] for k in ("diag_tap_coverage", "secure_commit_coverage", "residual_bytes", "status")}, "provenance": [p_ccr]})
            log["not_upgraded"].append({"row": rid, "lane": "cmt-integ", "lane_status": cr["status"],
                                        "reason": f"derived route adopted (source row {src} secure) but no PASS machine-readable verdict record; stays at correspondence-only under the manifest rule"})

    # ---------------- CMT-2: hidden FA2 rows R5-R8 ------------------------------------------------------------------------------
    r58, p_r58 = _fetch("cmt-hidden", "rows_5_8.json", commits["cmt-hidden"])
    cost, p_cost = _fetch("cmt-hidden", "cost_table.json", commits["cmt-hidden"])
    val = cost["validity"]
    dumps = val["oracle_bit_exact_M1_dumps"]
    negs = val["binding_negatives_cpu"]
    gtc = cost["gpu_tree_check"]["cases"]
    e2e = cost["e2e_vllm_generate"]["configs"]
    hid_val = {"status": "pass" if all(d["verdict"] == "BIT-EXACT" for d in dumps) and val["bit_identity_vs_installed_so_L40S"]["all_ok"] else "FAIL / incomplete", "machine_record": True,
               "test": f"M1 dump words BIT-EXACT vs the oracle on {len(dumps)} shapes ({sum(d['words'] for d in dumps):,} words); out/lse bit-identical with taps active on {val['bit_identity_vs_installed_so_L40S']['shapes']} shapes; in-kernel SHA-256 == hashlib over exported words",
               "provenance": [p_cost]}
    hid_bind = {"status": "pass" if negs and all(c["openings_ok"] and c["root_equal"] for c in gtc) else "FAIL / incomplete", "machine_record": True,
                "test": f"{len(negs)} negative classes rejected on the CPU engine over GPU-produced streams ({'; '.join(negs)}); GPU-built tree == host tree and openings verify on {len(gtc)}/{len(gtc)} cases (gpu_tree_check)",
                "caveat": "negatives executed on the CPU engine (no on-GPU adversarial run)", "provenance": [p_cost]}
    hid_meas = {"status": "measured", "by": "cmt-hidden", "samples": 20, "gpu": "NVIDIA L40S", "sm": "sm89", "utc": cost["utc"],
                "e2e_vllm_generate_ms": {k: {kk: v[kk] for kk in ("ms_median", "ms_p10", "ms_p90", "delta_vs_base_ms")} for k, v in e2e.items()},
                "e2e_setup": cost["e2e_vllm_generate"]["setup"][:300], "per_request_gpu_commit_ms": cost["commit"]["per_request_gpu_commit_ms"],
                "query_required_subset_bytes_per_request": cost["query_required_subset_matReq"]["per_request_bytes"],
                "also_in_run": "CMT-3 run g_b0_fast_hidden_vc acquires the M1 stream into its packed step stream (5 pairs, see R2 rows' measurement.other_lanes)",
                "provenance": [p_cost, p_sum], "note": "M1 export taps + GPU-resident tree; 20 reps after 3 warm-ups, medians/p10/p90 (samples are per-request wall times)"}
    for lr in r58["rows"]:
        rid = lr["id"]
        row = rows[rid]
        lab = lr["correspondence"].get("evidence_label", "")
        if lab.startswith("measured-correspondence") and row["correspondence"]["evidence_label"] != "measured-correspondence":
            row["correspondence"]["evidence_label_cmtref"] = row["correspondence"]["evidence_label"]
            row["correspondence"]["evidence_label"] = "measured-correspondence"
            row["correspondence"]["evidence_cmthidden"] = lab
            row["correspondence"]["provenance"] = [p_r58]
            row["correspondence"]["thread"] = lr["correspondence"].get("thread")
            row["correspondence"]["loop_point"] = lr["correspondence"].get("loop_point")
        for c in lr["candidates"]:
            row["candidates"].append({"id": f"cmt-hidden {c['id']}", "lane": "cmt-hidden", "stage": "built" if "VIABLE" in c.get("stage", "") or "validated" in c.get("stage", "") else "measured-negative",
                                      "mechanism": c.get("mechanism"), "producer_change": c.get("producer_change"), "status": c.get("stage"),
                                      "binds": c.get("binds"), "binding_test": dict(hid_bind), "value_test": dict(hid_val), "provenance": [p_r58]})
        row["measurement"] = dict(hid_meas)
        _merge_check(row, "value_correctness", {**hid_val, "lane_statement": lr["value_correctness"]}, "cmt-hidden")
        _merge_check(row, "binding_correctness", {**hid_bind, "lane_statement": lr["binding_correctness"]}, "cmt-hidden")
        row["diag_tap_coverage"]["cmt-hidden"] = lr["diag_tap_coverage"]
        row["secure_commit_coverage"]["cmt-hidden"] = lr["secure_commit_coverage"]
        log["ingested"].append({"row": rid, "lane": "cmt-hidden", "records": ["correspondence", "candidates", "measurement", "value_correctness", "binding_correctness"]})

    # ---------------- CMT-2: R10a / R10b derived-value route ----------------------------------------------------------------------
    r10, p_r10 = _fetch("cmt-hidden", "rows_10ab.json", commits["cmt-hidden"])
    der, p_der = _fetch("cmt-hidden", "results/l40s_cmt2c/rows10_derived.json", commits["cmt-hidden"])
    for lr in r10["rows"]:
        rid = lr["id"]
        row = rows[rid]
        for c in lr["candidates"]:
            c2 = dict(c)
            c2["id"] = f"cmt-hidden {c['id']}"
            c2.setdefault("binding_test", {})
            c2["binding_test"] = {**c2["binding_test"], "machine_record": True, "provenance": [p_r10, p_der]}
            c2["provenance"] = [p_r10]
            row["candidates"].append(c2)
        m = dict(lr["measurement"])
        m["provenance"] = [p_r10, p_der]
        m["note"] = "derived-value route: no producer change and nothing committed, so ΔT ≡ 0 by construction; the measured quantity is the verifier-side derivation cost and the exact-model check on real activations (samples = rows checked)"
        row["measurement"] = m
        _merge_check(row, "value_correctness", {**lr["value_correctness"], "machine_record": True, "provenance": [p_r10, p_der]}, "cmt-hidden")
        _merge_check(row, "binding_correctness", {**lr["binding_correctness"], "machine_record": True, "provenance": [p_r10, p_der]}, "cmt-hidden")
        row["diag_tap_coverage"]["cmt-hidden"] = lr["diag_tap_coverage"]
        row["secure_commit_coverage"]["cmt-hidden"] = lr["secure_commit_coverage"]
        row["status"] = f"{lr['status']} [cmt-hidden]"
        log["ingested"].append({"row": rid, "lane": "cmt-hidden", "records": ["candidates", "measurement", "value_correctness", "binding_correctness"]})

    # R3 / R9 stay unresolved (no lane record); note CMT-3's verdicts for completeness
    for rid in ("R3", "R9"):
        rows[rid]["candidates"].append({"id": "cmt-integ (no treatment)", "lane": "cmt-integ", "stage": "none", "status": integ_status[rid]["status"], "note": integ_status[rid]["note"],
                                        "binding_test": {"status": "n/a"}, "provenance": [p_ccr]})

    # ---------------- re-derive ----------------------------------------------------------------------------------------------------
    for row in doc["rows"]:
        row["derived_status"] = V2.derive_status(row)
        cr = integ_status.get(row["id"])
        if cr and row["derived_status"]["secure_path_implemented"] and cr.get("secure_commit_coverage") is not None and cr["status"] == "secure":
            for key in ("diag_tap_coverage", "secure_commit_coverage"):
                row[key]["fraction_of_bytes_v2_0"] = row[key].get("fraction_of_bytes")
                row[key]["fraction_of_bytes"] = cr[key]
                row[key]["note_v2_1"] = f"from cmt-integ coverage_complete_reference.json row {row['id']} (status {cr['status']}, residual {cr['residual_bytes']:.0f} B)"
        row.setdefault("status_v2_0", row["status"])
        by = row["measurement"].get("by")
        if row["derived_status"]["secure_path_benchmarked"] and by != "cmt-ref":
            row["status"] = f"secure path implemented + benchmarked by {by} [ingested v2.1]"
        elif row["derived_status"]["secure_path_benchmarked"]:
            row["status"] = "bound by cmt-ref reference path (measured); also cmt-integ complete reference [ingested v2.1]" if row["id"].startswith("R2") else row["status"]
        elif row["derived_status"]["secure_path_implemented"]:
            row["status"] = row["status_v2_0"] + " — implemented, not benchmarked [v2.1]"
        elif row["derived_status"]["unresolved"] and row["id"] in ("R4", "R10c"):
            row["status"] = f"unresolved under the manifest rule: derived route adopted by cmt-integ (coverage row '{integ_status[row['id']]['status']}') but no machine-readable test record [v2.1]"
    doc["headline"] = V2.headline(doc)
    doc["version"] = doc.get("version", "") + " -> v2.1 (all lanes ingested)"
    import datetime as _dt
    doc["utc_v2_0"] = doc.get("utc")
    doc["utc"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + " (v2.1)"
    doc["ingest"] = log
    return doc


def _merge_check(row: dict, key: str, rec: dict, lane: str) -> None:
    cur = row.get(key) or {}
    cur_status = str(cur.get("status", ""))
    if cur_status.startswith("pass") and cur.get("by", "cmt-ref") == "cmt-ref" and "pending" not in cur_status:
        cur.setdefault("other_lanes", []).append({"lane": lane, **rec})
    else:
        row[key] = {"test": rec.get("test"), "status": rec.get("status"), "by": lane, "machine_record": rec.get("machine_record"),
                    **{k: v for k, v in rec.items() if k not in ("test", "status")}, "cmtref_previous": cur}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="interface_manifest")
    args = ap.parse_args()
    doc = json.load(open(args.manifest))
    out = Path(args.out)
    (out / f"{args.name}_v2.0_cmtref_only.json").write_text(json.dumps(doc, indent=1) + "\n")
    doc = ingest(doc)
    (out / f"{args.name}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    md = V2.render_md(doc)
    ing = doc["ingest"]
    extra = ["", "## Ingested lane records (v2.1)", "",
             "Sources: " + "; ".join(f"{l}: `{v['branch']}` @ {v['commit']}" for l, v in ing["lanes"].items()) + ".",
             "Rows updated: " + ", ".join(f"{e['row']} ({e['lane']}: {', '.join(e['records'])})" for e in ing["ingested"]) + ".", "",
             "Ingest rule (in addition to the status rules above): a lane's binding / value test counts only if a MACHINE-READABLE record with the verdict or counts exists in the lane's committed tree "
             "(runs.jsonl `commit.openings.verified`, value_check.json, cost_table.json `validity`, rows10_derived.json); a pytest count quoted in a .md does not count. "
             "A measurement counts as a real sample when it is a per-run/per-request timing from a treatment run (CMT-3: 5 interleaved pairs; CMT-2: 20 reps) — for the derived-value rows R10a/b ΔT ≡ 0 by construction "
             "and the ingested sample is the verifier-side exact-model check + derivation cost. Every ingested record carries `provenance` = (lane, branch, commit, file).", ""]
    if ing["not_upgraded"]:
        extra.append("**Not upgraded (lane record lacks a field the rule needs):**")
        for e in ing["not_upgraded"]:
            extra.append(f"- {e['row']} — {e['lane']} status `{e['lane_status']}`: {e['reason']}")
    extra.append("")
    md = md.replace("\n## Rows\n", "\n" + "\n".join(extra) + "\n## Rows\n", 1) if "\n## Rows\n" in md else md + "\n".join(extra)
    (out / f"{args.name}.md").write_text(md)
    print(doc["headline"]["text"])
    for e in ing["not_upgraded"]:
        print("NOT UPGRADED:", e["row"], "-", e["reason"])


if __name__ == "__main__":
    main()
