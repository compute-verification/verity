#!/usr/bin/env python
"""S4 of `scripts/derive_e2e.sh` (lane CAPCORR, GUIDANCE §43.5): correspondence of a PROGRAM GIVEN AS A PARAMETER
against a certified record — binding (refuse on version mismatch), L2 (every committed interface of the replayed
requests re-evaluated gate by gate), L3 (every VU / proved obligation / embed row addressed to a gate/call of the
program).

    python scripts/derive_corr.py --program <spec> --record <dir> --out <dir> \
        --checkpoint model.safetensors --cos-sin-npy <fixture> [--stage bind,l2,l3] [--jobs 12]

`--program` spec (`verity_vllm.correspondence.program.load_program`):
    registry:vllm-poc/profiles/b0-a100-eager        hand-built registry program  -> label "harness validation"
    python:<module>:<callable>                       DERIVE's importer entry point  -> label "derived"
    descriptor:<path>                                a serialised descriptor (binding + L3 only; no typed params)

Exit codes: 0 all requested stages passed; 2 binding refused (program/vocabulary/profile mismatch); 3 L2 or L3
FAILED (a mismatch or an unlocated obligation); 4 usage/IO error.  Nothing heavy runs before the binding passes.

Artefacts (all under --out): binding.json, l2/corr-<req>.json + l2/summary.json, l3/address_map.json, s4.json
(stage summary), refusals.json (--self-test-refusals).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
for p in (REPO, REPO / "src", REPO / "vllm-poc", HERE.parent):  # HERE.parent: `python:capcorr_derived_program:program`
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verity_vllm.correspondence.core import CaptureIdentity, CommitStore, NotInProgram, run as corr_run  # noqa: E402
from verity_vllm.correspondence.program import (LoweringError, StructuralCircuitMap, binding_check, canonical_reference,  # noqa: E402
                                              load_program, locate_coordinate, lower_call_to_subcircuit, program_identity,
                                              record_pins, TC_LOWERING_RULES)

SCHEMA_S4 = "veritor/derive-corr/v1"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(p: Path, doc: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=False))


def host_info() -> dict[str, Any]:
    info = {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count()}
    try:
        info["mem_total_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        pass
    return info


# ---------------------------------------------------------------------------------------------------------
# binding
# ---------------------------------------------------------------------------------------------------------


def stage_bind(a: argparse.Namespace, out: Path) -> tuple[dict[str, Any], Any]:
    pins = record_pins(Path(a.record))
    t0 = time.time()
    lp = load_program(a.program, LP=a.LP or pins.get("LP"), STEPS=a.STEPS or pins.get("STEPS"), repo_root=REPO)
    ident = program_identity(lp)
    problems = binding_check(ident, pins, expect_vocabulary=a.expect_vocabulary, expect_profile_sha256=a.expect_profile_sha256)
    label = a.label or ("harness validation (hand-built registry program; NOT the derived-program deliverable, mandate boundary 6)"
                        if lp.source_kind == "registry" else "derived program")
    doc = {"schema": SCHEMA_S4 + "/binding", "utc": utc(), "host": host_info(), "label": label, "program": ident, "record": pins,
           "expected": {"vocabulary": a.expect_vocabulary, "semantic_profile_sha256": a.expect_profile_sha256},
           "problems": problems, "bound": not problems, "seconds": round(time.time() - t0, 2),
           "capture_configuration_note": ("the record was captured under semantic profile "
                                          f"{pins.get('semantic_profile')} (sha {str(pins.get('semantic_profile_sha256'))[:12]}): enforce_eager, "
                                          "cudagraph NONE, VLLM_BATCH_INVARIANT=1, dispatch-mode interception + recorder patches — NOT unchanged stock "
                                          "vLLM execution (REPLAY measured 8.3x); every claim below is about that configuration")}
    write_json(out / "binding.json", doc)
    if problems:
        print("[bind] REFUSED:", file=sys.stderr)
        for p in problems:
            print(f"[bind]   - {p}", file=sys.stderr)
    else:
        print(f"[bind] bound: {ident['function']} digest {ident['digest'][:16]} vocab {ident['vocabulary']['digest'][:16]} "
              f"== record {pins['record']} pins; label: {label}")
    return doc, lp


# ---------------------------------------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------------------------------------


def ensure_bundle(a: argparse.Namespace, out: Path) -> Path:
    """The extracted bundle root (the directory holding commitments.json)."""
    if a.bundle:
        b = Path(a.bundle)
        if (b / "commitments.json").exists():
            return b
        cands = list(b.glob("*/bundle/commitments.json")) + list(b.glob("bundle/commitments.json"))
        if cands:
            return cands[0].parent
        raise FileNotFoundError(f"--bundle {b}: no commitments.json under it")
    rd = Path(a.record)
    tars = sorted((rd / "bundle").glob("*.tar.gz"))
    if not tars:
        raise FileNotFoundError(f"{rd}/bundle has no tarball")
    dst = out / "bundle"
    cands = list(dst.glob("*/bundle/commitments.json"))
    if cands:
        return cands[0].parent
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[bundle] extracting {tars[0].name} -> {dst}")
    with tarfile.open(tars[0]) as tf:
        tf.extractall(dst)
    cands = list(dst.glob("*/bundle/commitments.json"))
    if not cands:
        raise FileNotFoundError("tarball did not contain bundle/commitments.json")
    return cands[0].parent


def replayed_requests(store: CommitStore) -> list[str]:
    """Requests with committed interiors (RU trees), i.e. the r_q-selected J."""
    reqs = sorted({v["request_id"] for v in store.commitments["trees"].values() if v.get("kind") == "ru"})
    return reqs


# ---------------------------------------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------------------------------------


def worker_l2(a: argparse.Namespace) -> int:
    """One L2 worker: positions `a.positions` of request `a.request` against the program `a.program`."""
    pins = record_pins(Path(a.record))
    lp = load_program(a.program, LP=a.LP or pins.get("LP"), STEPS=a.STEPS or pins.get("STEPS"), repo_root=REPO)
    ident = program_identity(lp)
    if binding_check(ident, pins, expect_vocabulary=a.expect_vocabulary, expect_profile_sha256=a.expect_profile_sha256):
        print("[l2-worker] binding refused; not running", file=sys.stderr)
        return 2
    C = dict(lp.config) if lp.config else dict(pins["static_configuration"])
    if "profile" not in C and pins.get("program_profile"):
        C["profile"] = pins["program_profile"]
    store = CommitStore(Path(a.bundle))
    program = lp.program
    layers = list(range(C["NL"])) if a.layers == ["all"] else [int(x) for x in a.layers]
    corr_run(store, [Path(p) for p in a.checkpoint], a.request, [int(x) for x in a.positions], layers, a.coord_samples, set(),
             Path(a.out), a.seed, LP=lp.LP, STEPS=lp.STEPS, cos_sin_npy=Path(a.cos_sin_npy) if a.cos_sin_npy else None, config="B0",
             program_factory=lambda LP, STEPS: program, circuit_map=lambda p: StructuralCircuitMap(p), static_config=C)
    return 0


def _blocks(positions: list[int], jobs: int) -> list[list[int]]:
    """Interleaved blocks so every worker gets early (cheap) and late (attention-heavy) positions."""
    jobs = max(1, min(jobs, len(positions)))
    return [positions[i::jobs] for i in range(jobs)]


def _merge_corr(parts: list[dict], request: str) -> dict:
    base = dict(parts[0])
    base["positions"] = sorted({t for p in parts for t in p["positions"]})
    base["records"] = sorted((r for p in parts for r in p["records"]), key=lambda r: r["value_identity"]["region"][0])
    seen: set[str] = set()
    nip = []
    for p in parts:
        for r in p["not_in_program"]:
            if r["capture_id"] not in seen:
                seen.add(r["capture_id"])
                nip.append(r)
    base["not_in_program"] = sorted(nip, key=lambda r: r["capture_id"])
    summ: Counter = Counter()
    for p in parts:
        for k, v in p["summary"].items():
            if isinstance(v, int):
                summ[k] += v
    base["summary"] = dict(summ)
    from verity_vllm.correspondence.core import summary_qualifications
    base["summary"]["qualifications"] = summary_qualifications(base["input_gate_records"]["weights"], base["records"], base["summary"])
    base["merged_from"] = len(parts)
    return base


def _interface_counts(rec: dict) -> dict[str, Any]:
    """Per committed interface (layer-op suffix + port, or embed/norm/lm_head/sampler): exact / mismatch / other."""
    table: dict[str, Counter] = {}
    for r in rec["records"]:
        ident = CaptureIdentity.parse(r["capture_id"])
        key = (re.sub(r"^layers\.\d+\.", "layers.*.", ident.op) + f"/port={ident.output}") if ident.layer.isdigit() else f"{ident.op}/port={ident.output}"
        table.setdefault(key, Counter())[r["check"]["status"]] += 1
    for r in rec["not_in_program"]:
        ident = CaptureIdentity.parse(r["capture_id"])
        table.setdefault(f"{ident.op}/port={ident.output}", Counter())["not_in_program"] += 1
    return {k: dict(v) for k, v in sorted(table.items())}


def stage_l2(a: argparse.Namespace, out: Path, lp, bundle: Path) -> dict[str, Any]:
    store = CommitStore(bundle)
    reqs = replayed_requests(store) if a.requests == ["all"] else a.requests
    l2dir = out / "l2"
    l2dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: dict[str, Any] = {"schema": SCHEMA_S4 + "/l2", "utc": utc(), "program": a.program, "requests": {}, "jobs": a.jobs, "coord_samples": a.coord_samples,
                               "seed": a.seed, "host": host_info()}
    total_leaves = 0
    all_ok = True
    for req in reqs:
        steps = store.steps(req)
        positions = sorted({t for v in store.commitments["trees"].values() if v.get("kind") == "ru" and v["request_id"] == req for t in v["positions"]})
        n_leaves = sum(len(store.ru(req, s)[0]["entries"]) for s in steps)
        total_leaves += n_leaves
        procs = []
        parts_paths = []
        for k, block in enumerate(_blocks(positions, a.jobs)):
            part = l2dir / f"part-{req[:8]}-{k:02d}.json"
            parts_paths.append(part)
            if part.exists() and part.with_suffix(".ok").exists() and a.resume:
                continue  # `.ok` is written by THIS process on a clean worker exit; a part without it is partial (killed worker) and is redone
            cmd = [sys.executable, str(HERE), "--worker-l2", "--program", a.program, "--record", a.record, "--bundle", str(bundle), "--out", str(part),
                   "--request", req, "--positions", *map(str, block), "--layers", *a.layers, "--coord-samples", str(a.coord_samples), "--seed", str(a.seed + k),
                   "--checkpoint", *a.checkpoint]
            if a.cos_sin_npy:
                cmd += ["--cos-sin-npy", a.cos_sin_npy]
            if a.expect_vocabulary:
                cmd += ["--expect-vocabulary", a.expect_vocabulary]
            if a.expect_profile_sha256:
                cmd += ["--expect-profile-sha256", a.expect_profile_sha256]
            if a.LP:
                cmd += ["--LP", str(a.LP)]
            if a.STEPS:
                cmd += ["--STEPS", str(a.STEPS)]
            log = open(l2dir / f"part-{req[:8]}-{k:02d}.log", "w")
            procs.append((k, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO)), log))
        print(f"[l2] {req[:8]}: {len(positions)} positions, {n_leaves:,} committed leaves, {len(procs)} workers")
        rcs = []
        for k, p, log in procs:
            rc = p.wait()
            rcs.append(rc)
            log.close()
            if rc == 0:
                (l2dir / f"part-{req[:8]}-{k:02d}.ok").write_text(f"{utc()} exit 0\n")
        if any(rc != 0 for rc in rcs):
            print(f"[l2] {req[:8]}: worker exit codes {rcs}", file=sys.stderr)
        parts = [json.loads(p.read_text()) for p in parts_paths if p.exists()]
        if not parts:
            summary["requests"][req] = {"status": "FAIL", "reason": "no worker output"}
            all_ok = False
            continue
        merged = _merge_corr(parts, req)
        merged["worker_exit_codes"] = rcs
        merged["committed_leaves"] = n_leaves
        merged["interfaces"] = _interface_counts(merged)
        covered = len(merged["records"]) + len(merged["not_in_program"])
        merged["coverage"] = {"records": len(merged["records"]), "not_in_program": len(merged["not_in_program"]), "committed_leaves": n_leaves,
                              "complete": covered == n_leaves}
        out_path = l2dir / f"corr-{req[:8]}.json"
        write_json(out_path, merged)
        s = merged["summary"]
        ok = s.get("mismatch", 0) == 0 and s.get("missing-operand", 0) == 0 and s.get("shape_mismatch", 0) == 0 and covered == n_leaves and all(rc == 0 for rc in rcs)
        all_ok &= ok
        summary["requests"][req] = {"status": "PASS" if ok else "FAIL", "file": out_path.name, "exact": s.get("exact", 0), "mismatch": s.get("mismatch", 0),
                                    "missing_operand": s.get("missing-operand", 0), "shape_mismatch": s.get("shape_mismatch", 0),
                                    "coordinate_sampled_gemm_leaves": s.get("sampled", 0), "not_in_program": len(merged["not_in_program"]),
                                    "committed_leaves": n_leaves, "coverage_complete": covered == n_leaves, "interfaces": merged["interfaces"],
                                    "qualifications": s.get("qualifications", [])}
        print(f"[l2] {req[:8]}: exact {s.get('exact', 0)} mismatch {s.get('mismatch', 0)} missing {s.get('missing-operand', 0)} "
              f"not_in_program {len(merged['not_in_program'])} / {n_leaves} leaves -> {'PASS' if ok else 'FAIL'}")
    summary["status"] = "PASS" if all_ok and reqs else "FAIL"
    summary["seconds"] = round(time.time() - t0, 1)
    summary["committed_leaves_total"] = total_leaves
    write_json(l2dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------------------------------------
# L3
# ---------------------------------------------------------------------------------------------------------


def _load_l2_status(out: Path) -> dict[str, str]:
    st: dict[str, str] = {}
    for f in (out / "l2").glob("corr-*.json"):
        d = json.loads(f.read_text())
        for r in d["records"]:
            st[r["capture_id"]] = r["check"]["status"]
        for r in d["not_in_program"]:
            st[r["capture_id"]] = "not_in_program"
    return st


def stage_l3(a: argparse.Namespace, out: Path, lp, bundle: Path | None) -> dict[str, Any]:
    from veritor.interface.programs import gemm_coordinate

    rd = Path(a.record)
    program = lp.program
    t0 = time.time()
    cmap = StructuralCircuitMap(program)
    l2status = _load_l2_status(out)
    entries: list[dict[str, Any]] = []
    unlocated: list[dict[str, Any]] = []
    out_of_program: list[dict[str, Any]] = []
    counts: Counter = Counter()

    def add(kind: str, key: str, call, gates, extra: dict[str, Any]) -> None:
        e = {"class": kind, "id": key, "address": canonical_reference(call, gates), **extra}
        entries.append(e)
        counts[kind] += 1

    # (1) every verifier VU
    for line in (rd / "verify" / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cid = r["canonical_id"]
        try:
            call, gates = cmap.locate(CaptureIdentity.parse(cid))
        except (NotInProgram, KeyError) as ex:
            unlocated.append({"class": "vu", "id": cid, "reason": f"{type(ex).__name__}: {ex}",
                              "category": ("value-not-in-pinned-program" if isinstance(ex, NotInProgram) else "locator-error"),
                              "relation_class": r.get("class"), "relation_id": r.get("relation_id"), "verifier_status": r.get("status")})
            continue
        add("vu", cid, call, gates, {"relation_class": r.get("class"), "relation_id": r.get("relation_id"), "verifier_status": r.get("status"),
                                     "l2_status": l2status.get(cid, "l2-not-run")})
    # (2) the proved GEMM-coordinate obligations (+ the Input equations of the same object)
    comp = json.loads((rd / "obligations" / "composed.json").read_text())
    sub_by_digest = {s["digest"]: s for s in comp["object"]["subcircuits"]}
    for ob in comp["object"]["obligations"]:
        tid = ob["target"]["id"]
        m = re.match(r"^(?P<cid>p=.*?/port=\d+)/Gemm\[(?P<j>\d+)\]$", tid)
        if m:
            cid, j = m.group("cid"), int(m.group("j"))
            try:
                gemm, coord = locate_coordinate(cmap, cid, j)
                sc = lower_call_to_subcircuit(coord)
            except (NotInProgram, KeyError, LoweringError) as ex:
                unlocated.append({"class": "gemm-proof", "id": tid, "reason": f"{type(ex).__name__}: {ex}"})
                continue
            stmt_digest = ob["subcircuit"]
            K = int(coord.fn.bindings["K"])
            # x operand identity: the coordinate's first K args must be the located producer call's returns (the value the
            # obligation's boundary.in names by capture id)
            x_ids = sorted({re.sub(r"(\[\d+\])+$", "", s) for s in ob["boundary"]["in"] if s.startswith("p=")})
            x_ok = None
            x_detail = None
            if len(x_ids) == 1:
                try:
                    xcall, xgates = cmap.locate(CaptureIdentity.parse(x_ids[0]))
                    from verity.ir.layout import param_leaf
                    first = tuple(param_leaf(coord.scope(), 0, k) for k in range(K))
                    x_ok = first == tuple(xgates[:K]) if len(xgates) >= K else False
                    x_detail = {"x_capture_id": x_ids[0], "x_call": xcall.id, "x_returns": [xgates[0], xgates[-1] + 1] if xgates else None}
                except (NotInProgram, KeyError) as ex:
                    x_ok, x_detail = False, {"x_capture_id": x_ids[0], "error": str(ex)}
            add("gemm-proof", tid, coord, coord.returns,
                {"gemm_call": canonical_reference(gemm), "coordinate": j, "K": K,
                 "statement_subcircuit": stmt_digest, "program_lowered_subcircuit": sc.digest,
                 "interface_pinned_subcircuit": gemm_coordinate(K).digest,
                 "kind_program_equal": sc.digest == stmt_digest == gemm_coordinate(K).digest,
                 "lowering_rule": TC_LOWERING_RULES["id"] + f"@{TC_LOWERING_RULES['version']}",
                 "x_operand_is_program_value": x_ok, "x_operand": x_detail,
                 "l2_status_of_row": l2status.get(cid, "l2-not-run"), "family": comp["object"]["family"]["id"]})
            continue
        m = re.match(r"^(?P<ref>.+)/InputEq$", tid)
        if m:
            ref = m.group("ref")
            try:
                ent = _locate_input_eq(cmap, program, ref, lp)
            except (KeyError, ValueError, NotImplementedError) as ex:
                unlocated.append({"class": "input-eq", "id": tid, "reason": f"{type(ex).__name__}: {ex}"})
                continue
            entries.append({"class": "input-eq", "id": tid, **ent})
            counts["input-eq"] += 1
            continue
        unlocated.append({"class": "obligation", "id": tid, "reason": "unrecognised target form"})
    # (3) the embed-gather coverage rows
    cov = rd / "proofs" / "proof-cov"
    census_files = sorted(cov.glob("embed-*/census.json"))
    for cf in census_files:
        cen = json.loads(cf.read_text())
        for vu in cen.get("vus", []):
            t = int(vu["t"])
            name = f"t={t}/embed"
            key = f"{cf.parent.name}:r={vu['request'][:8]}/t={t}"
            call = cmap.named.get(name)
            if call is None:
                unlocated.append({"class": "embed-row", "id": key, "reason": f"no Embedding call at position {t}"})
                continue
            add("embed-row", key, call, call.returns,
                {"request": vu["request"], "t": t, "phase": vu.get("phase"), "token": vu.get("token"),
                 "relation_id": cen.get("relation_id"), "kind_id_equal": cen.get("relation_id") == call.id,
                 "kind_program_equal": None,
                 "kind_program_note": ("the embed-gather@1 statement body (gather_word x576 + i32 offset arithmetic) is a different encoding "
                                       "of Embedding_v1's 576 gather gates; no lowering rule is pinned, so body equality is NOT asserted here "
                                       "— only the call identity and address"),
                 "l2_status_of_row": next((s for c, s in l2status.items() if f"/r={vu['request']}/t={t}/" in c and "/l=embed/" in c), "l2-not-run")})
    # (4) pad-serve statements: relations of Serve@3, which the certificate does not name (CORR-PREP G6)
    recs = cov / "records.jsonl"
    if recs.exists():
        for line in recs.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("family") == "pad-serve":
                out_of_program.append({"class": "pad-serve", "batch": Path(r.get("batch_dir", "")).name, "composition": r.get("composition"),
                                       "statement_digest": r.get("statement_digest"), "obligations": r.get("obligations"),
                                       "reason": "not a relation of the pinned program: Serve_v1 has no PadGuard/PadRowMux gate; these statements "
                                                 "belong to Serve@3, which the certificate does not pin (correspondence-gap.md G6)"})
    n_located = sum(counts[k] for k in ("vu", "gemm-proof", "embed-row"))
    n_unlocated = len([u for u in unlocated if u["class"] in ("vu", "gemm-proof", "embed-row")])
    gemm_entries = [e for e in entries if e["class"] == "gemm-proof"]
    kind_eq_fail = [e["id"] for e in gemm_entries if not e["kind_program_equal"]]
    x_fail = [e["id"] for e in gemm_entries if e["x_operand_is_program_value"] is False]
    l2_join = Counter(e.get("l2_status") or e.get("l2_status_of_row") for e in entries if e["class"] in ("vu", "gemm-proof", "embed-row"))
    ok = n_unlocated == 0 and not kind_eq_fail and not x_fail and not [u for u in unlocated if u["class"] == "input-eq"]
    unloc_cat = Counter(u.get("category", u["class"]) for u in unlocated)
    doc = {"schema": SCHEMA_S4 + "/address-map", "utc": utc(), "program": program_identity(lp), "record": rd.name,
           "verdict_note": ("PASS requires every VU, proved obligation and embed row of the record to be addressed to a gate/call of THIS program. "
                            "An unlocated item is a FAIL: either the locator is incomplete or the record's verifier checked a production value the "
                            "pinned program does not compute (category value-not-in-pinned-program) — the certificate then over-claims relative to "
                            "the pinned program and the program (or the VU population) must change, not the verdict."),
           "locator": {"kind": "StructuralCircuitMap", "annotation_agreement": cmap.annotation_agreement, "named_calls": len(cmap.named),
                       "note": "addresses are the program's canonical root-scoped gate intervals / function identities, reconstructed from the "
                               "canonical call order + dataflow checks — never from capture labels or hand-built call-name annotations"},
           "counts": {"located": dict(counts), "located_total": n_located, "unlocated_total": n_unlocated, "unlocated_by_category": dict(unloc_cat),
                      "proved_obligations_total": counts["gemm-proof"] + counts["input-eq"],
                      "out_of_program_pad_serve": len(out_of_program), "kind_program_equal_failures": kind_eq_fail, "x_operand_failures": x_fail,
                      "l2_status_join": dict(l2_join)},
           "status": "PASS" if ok else "FAIL", "seconds": round(time.time() - t0, 1),
           "entries": entries, "unlocated": unlocated, "out_of_program": out_of_program}
    write_json(out / "l3" / "address_map.json", doc)
    print(f"[l3] located {n_located} (vu {counts['vu']}, gemm-proof {counts['gemm-proof']}, embed-row {counts['embed-row']}; input-eq {counts['input-eq']}), "
          f"unlocated {n_unlocated}, pad-serve out-of-program {len(out_of_program)}; kind-program equal on {len(gemm_entries) - len(kind_eq_fail)}/{len(gemm_entries)} "
          f"GEMM proofs; l2 join {dict(l2_join)} -> {doc['status']}")
    return doc


def _locate_input_eq(cmap: StructuralCircuitMap, program, ref: str, lp) -> dict[str, Any]:
    """`w=<param>/c=<j>[0][k]` -> the weight Input gate; `p=…/b=prompt_token/t=<t>[h]` -> prompt Input gate (half h);
    `p=…/b=output_token/t=<t>[h]` -> the Final(t-1) TokenSelect return (the program value the committed token is)."""
    from verity_vllm.correspondence.core import _coll_range
    from verity.ir.layout import resolve
    m = re.match(r"^w=(?P<param>[^/]+)/c=(?P<j>\d+)\[0\]\[(?P<k>\d+)\]$", ref)
    C = dict(lp.config) if lp.config else None
    if m:
        param, j, k = m.group("param"), int(m.group("j")), int(m.group("k"))
        W = program.input_coll(1)
        lm = re.match(r"^layers\.(\d+)\.(.+)$", param)
        if not lm:
            raise KeyError(f"weight {param}: not a layer parameter")
        L, tail = int(lm.group(1)), lm.group(2)
        field = {"self_attn.qkv_proj.weight": "qkv_w", "self_attn.o_proj.weight": "o_w", "mlp.gate_up_proj.weight": "gate_up_w", "mlp.down_proj.weight": "down_w"}[tail]
        coll = W["layers"][L][field]
        start, end = _coll_range(program.circuit, coll)
        K = coll.type.elem.n if hasattr(coll.type, "elem") else None
        g = start + j * K + k
        return {"address": {"gate_class": "Input/weights", "gate": g, "input_gates": [start, end], "field": f"weights.layers[{L}].{field}", "row": j, "k": k},
                "relation": "InputEq: the committed weight word equals the checkpoint's word = the value of this Input gate"}
    m = re.match(r"^p=[^/]+/r=(?P<r>[^/]+)/b=(?P<f>prompt_token|output_token)/t=(?P<t>\d+)\[(?P<h>[01])\]$", ref)
    if not m:
        raise ValueError(f"unrecognised InputEq reference {ref}")
    t, h, f = int(m.group("t")), int(m.group("h")), m.group("f")
    if f == "prompt_token":
        pc = program.input_coll(0)
        g = resolve(program.circuit.root_scope, pc.refs[t])
        return {"address": {"gate_class": "Input/prompt", "gate": g, "position": t, "half": h},
                "relation": "InputEq: the committed prompt token half-word equals the served prompt token = this Input gate's value (half)"}
    fin = cmap.named.get(f"t={t - 1}/final")
    if fin is None:
        raise KeyError(f"no Final call at position {t - 1} for output token t={t}")
    from verity_vllm.correspondence.program import _select_child
    ts = _select_child(fin, "TokenSelect")
    return {"address": {**canonical_reference(ts, ts.returns), "position": t, "half": h, "call": f"t={t - 1}/final/TokenSelect"},
            "relation": "InputEq: the committed output token half-word equals the served token; the program value is TokenSelect's return at Final(t-1)"}


# ---------------------------------------------------------------------------------------------------------
# refusal self-test
# ---------------------------------------------------------------------------------------------------------


def self_test_refusals(a: argparse.Namespace, out: Path) -> int:
    """Negative tests of the binding: each case must exit 2 with the intended reason; the control must bind."""
    import tempfile
    rd = Path(a.record)
    small = ["manifest.json", "certificate.json", "summary.json", "program/program.json", "bundle/bundle.json"]

    def copy_record(tmp: Path) -> Path:
        dst = tmp / rd.name
        for rel in small:
            src = rd / rel
            if src.exists():
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst / rel)
        return dst

    def edit_json(p: Path, fn) -> None:
        d = json.loads(p.read_text())
        fn(d)
        p.write_text(json.dumps(d, indent=1))

    def run_bind(record: Path, extra: list[str], program: str | None = None) -> tuple[int, str]:
        cmd = [sys.executable, str(HERE), "--program", program or a.program, "--record", str(record), "--out", str(out / "refusal-tmp"), "--stage", "bind", *extra]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        return r.returncode, (r.stderr + r.stdout)

    cases = []
    with tempfile.TemporaryDirectory(prefix="capcorr-refusal-") as tmp_s:
        tmp = Path(tmp_s)
        # control
        rc, log = run_bind(rd, [])
        cases.append({"case": "control (unmodified record, matching program)", "expect_exit": 0, "exit": rc, "ok": rc == 0, "log_tail": log[-300:]})
        # 1. program digest differs (record pin edited) -> refused
        r1 = copy_record(tmp / "c1")
        edit_json(r1 / "program" / "program.json", lambda d: d["program"].__setitem__("digest", "00" * 32))
        rc, log = run_bind(r1, [])
        cases.append({"case": "record pins a different program digest", "expect_exit": 2, "exit": rc, "ok": rc == 2 and "program digest mismatch" in log, "log_tail": log[-300:]})
        # 2. a different program supplied (same record) -> refused
        rc, log = run_bind(rd, ["--STEPS", "7"], program="registry:B0")
        cases.append({"case": "different program supplied (Serve_v1 STEPS=7) against the certified record", "expect_exit": 2, "exit": rc,
                      "ok": rc == 2 and "program digest mismatch" in log, "log_tail": log[-300:]})
        # 3. kind-vocabulary version differs -> refused
        rc, log = run_bind(rd, ["--expect-vocabulary", "11" * 32])
        cases.append({"case": "kind-vocabulary version differs from the expected pin", "expect_exit": 2, "exit": rc, "ok": rc == 2 and "kind-vocabulary version mismatch" in log,
                      "log_tail": log[-300:]})
        # 4. profile differs (record program.json profile edited) -> refused
        r4 = copy_record(tmp / "c4")
        edit_json(r4 / "program" / "program.json", lambda d: d["program"]["profile"].__setitem__("linear", "cuBLAS gemm (not batch-invariant)"))
        rc, log = run_bind(r4, [])
        cases.append({"case": "record's program profile differs (linear kernel)", "expect_exit": 2, "exit": rc, "ok": rc == 2 and "profile mismatch" in log, "log_tail": log[-300:]})
        # 5. semantic profile sha differs -> refused
        rc, log = run_bind(rd, ["--expect-profile-sha256", "22" * 32])
        cases.append({"case": "semantic profile sha differs from the expected pin", "expect_exit": 2, "exit": rc, "ok": rc == 2 and "semantic profile mismatch" in log,
                      "log_tail": log[-300:]})
        # 6. static configuration differs (record edited: EPS) -> refused
        r6 = copy_record(tmp / "c6")
        edit_json(r6 / "program" / "program.json", lambda d: d["program"]["static_configuration"].__setitem__("EPS", 1e-6))
        rc, log = run_bind(r6, [])
        cases.append({"case": "record's static configuration differs (EPS)", "expect_exit": 2, "exit": rc, "ok": rc == 2 and "static configuration mismatch" in log,
                      "log_tail": log[-300:]})
    shutil.rmtree(out / "refusal-tmp", ignore_errors=True)
    ok = all(c["ok"] for c in cases)
    doc = {"schema": SCHEMA_S4 + "/refusals", "utc": utc(), "program": a.program, "record": rd.name, "cases": cases, "all_refused_as_intended": ok}
    write_json(out / "refusals.json", doc)
    for c in cases:
        print(f"[refusals] {'ok ' if c['ok'] else 'BAD'} exit {c['exit']} (expect {c['expect_exit']}): {c['case']}")
    print(f"[refusals] {'PASS' if ok else 'FAIL'}: {sum(c['ok'] for c in cases)}/{len(cases)}")
    return 0 if ok else 3


# ---------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", required=True, help="registry:<port>|registry:B0|python:<module>:<callable>|descriptor:<path>")
    ap.add_argument("--record", required=True, help="certified record directory (e.g. fixtures/results/INT-PAD-2/INTPAD2-B0-20260909T013519Z)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="bind,l2,l3", help="comma list of bind,l2,l3 (bind always runs first)")
    ap.add_argument("--bundle", default=None, help="extracted bundle root (else the record's tarball is extracted under --out)")
    ap.add_argument("--checkpoint", nargs="+", default=[], help="safetensors shard(s) of the pinned checkpoint (L2)")
    ap.add_argument("--cos-sin-npy", default=None, help="the committed GPU cos_sin_cache words (fixture, sha-equal to the weights leaf)")
    ap.add_argument("--requests", nargs="+", default=["all"])
    ap.add_argument("--layers", nargs="+", default=["all"])
    ap.add_argument("--coord-samples", type=int, default=32)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="reuse existing l2 part files")
    ap.add_argument("--LP", type=int, default=None)
    ap.add_argument("--STEPS", type=int, default=None)
    ap.add_argument("--expect-vocabulary", default=None, help="registry-version digest the program must be evaluated under (refuse otherwise)")
    ap.add_argument("--expect-profile-sha256", default=None, help="semantic profile sha256 the record must have been captured under (refuse otherwise)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--self-test-refusals", action="store_true")
    # worker mode
    ap.add_argument("--worker-l2", action="store_true")
    ap.add_argument("--request", default=None)
    ap.add_argument("--positions", nargs="+", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.worker_l2:  # --out is the part FILE here
        out.parent.mkdir(parents=True, exist_ok=True)
        return worker_l2(a)
    out.mkdir(parents=True, exist_ok=True)
    if a.self_test_refusals:
        return self_test_refusals(a, out)
    stages = [s.strip() for s in a.stage.split(",") if s.strip()]
    t0 = time.time()
    bind_doc, lp = stage_bind(a, out)
    s4: dict[str, Any] = {"schema": SCHEMA_S4, "utc": utc(), "record": Path(a.record).name, "program": bind_doc["program"], "label": bind_doc["label"],
                          "binding": {"bound": bind_doc["bound"], "problems": bind_doc["problems"]}, "stages": {}, "host": host_info()}
    if not bind_doc["bound"]:
        s4["status"] = "REFUSED"
        write_json(out / "s4.json", s4)
        return 2
    rc = 0
    if "l2" in stages:
        if not a.checkpoint:
            print("[l2] --checkpoint is required", file=sys.stderr)
            return 4
        bundle = ensure_bundle(a, out)
        l2 = stage_l2(a, out, lp, bundle)
        s4["stages"]["l2"] = {k: v for k, v in l2.items() if k != "requests"} | {"requests": {r[:8]: {k2: v2 for k2, v2 in v.items() if k2 != "interfaces"} for r, v in l2["requests"].items()}}
        rc = rc or (0 if l2["status"] == "PASS" else 3)
    if "l3" in stages:
        bundle = None
        l3 = stage_l3(a, out, lp, bundle)
        s4["stages"]["l3"] = {"status": l3["status"], "counts": l3["counts"], "seconds": l3["seconds"], "file": "l3/address_map.json"}
        rc = rc or (0 if l3["status"] == "PASS" else 3)
    s4["status"] = "PASS" if rc == 0 else "FAIL"
    s4["seconds"] = round(time.time() - t0, 1)
    write_json(out / "s4.json", s4)
    print(f"[s4] {s4['status']} ({s4['label']}) in {s4['seconds']}s -> {out / 's4.json'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
