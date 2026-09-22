#!/usr/bin/env python
# ============================================================================================================================
# LEGACY (DIRECTIVE v4) — NOT THE CONTRACT DRIVER.  Kept only to reproduce the REC-P1 historical record.
#   * Calls e2e.partition_sample.stage1 with legacy_seed_file=True (explicit since A10's WR-2): the challenge seed is written to seed.hex in the
#     record directory on the prover's host (docs/PROTOCOL-CONFORMANCE.md R7 — seed custody).
#   * No exemption request / epoch barrier / step-4 checks (R8, R16): the v4 chronology (legacy phase_s = interior digest).
# Refuses to run unless --legacy-directive-v4 is passed (R14).  With the flag, behaviour is unchanged.  The contract driver is
# e2e/verifier_v3/driver.py (seed held in the verifier process; F, |F| <= B_ex (RU count), r_s over J \ F); the
# contract runner is record_v5/ (A9).  [A10 2026-09-10]
# ============================================================================================================================
"""[L1 RECORD 2026-09-10] The verifier's side of the record protocol over the VALIDATED P1 partition (DIRECTIVE v3 §1.1 steps 2-5).

LEGACY (DIRECTIVE v4): see the banner above; requires ``--legacy-directive-v4``.  Contract path: ``e2e.verifier_v3.driver`` / ``record_v5``.

The prover is ``verity_vllm_adapter.selective_prover`` unchanged (vLLM venv, GPU): phase 0 commits every request's boundary (prompt tokens, output
tokens, params, seed -> the request commitments) and the weights tree, then waits for ``replay_challenge.json``; phase 2 replays ONLY the
selected requests J with instrumented kernels and commits their interiors (operator boundary rows), never recommitting weights or request inputs.

This process plays the verifier with ``e2e.partition_sample`` over the replicated population/v2 (RU = request; VU = unit of P1/cu=whole):

  READY_PHASE0 -> boundary digest = sha256(commitments_phase0.json) -> stage1 (seed committed, r_q; logged redraw if empty) -> replay_challenge.json
  READY_PHASE2 -> interior digest = sha256(commitments.json)        -> stage2 (r_s over the VUs of J) -> selected.json -> reveal -> verify

RU index r <-> the r-th request id in the prover's canonical order (sorted request ids); written to ``ru_map.json``.  The registered weight
commitment (``weights_authorized.json#expected_root``, ROT-INPUT) is compared with the prover's weights root at phase 0 and the outcome recorded
(a verifier check; a mismatch is recorded, never bridged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from e2e import partition_sample as ps  # noqa: E402

READY0, READY2, CHALLENGE_FILE, ABORT = "READY_PHASE0", "READY_PHASE2", "replay_challenge.json", "ABORT"

# [A10 2026-09-10] R14 legacy guard: this script is DIRECTIVE-v4 legacy and refuses to run unless the flag is passed explicitly.
LEGACY_FLAG = "--legacy-directive-v4"
LEGACY_POINTER = (
    f"{Path(__file__).name}: LEGACY (DIRECTIVE v4) — not the contract driver; refusing to run without {LEGACY_FLAG}.\n"
    "  Writes the challenge seed to seed.hex on the prover's host (R7); no exemption request, epoch barrier or step-4 checks (R8, R16).\n"
    "  Contract driver: e2e/verifier_v3/driver.py; contract runner: record_v5/ (A9).\n"
    f"  Historical reproduction of the REC-P1 record: re-run with {LEGACY_FLAG} (behaviour is otherwise unchanged)."
)


def legacy_gate(argv: list[str] | None) -> list[str] | None:
    """Return ``argv`` without the legacy flag, or ``None`` (after printing the pointer) when it is absent.  ``-h/--help`` pass."""
    args = list(sys.argv[1:] if argv is None else argv)
    if LEGACY_FLAG in args:
        return [x for x in args if x != LEGACY_FLAG]
    if any(x in ("-h", "--help") for x in args):
        return args
    print(LEGACY_POINTER, file=sys.stderr)
    return None


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wait_for(path: Path, abort: Path, proc: subprocess.Popen, timeout: float, what: str) -> None:
    t0 = time.time()
    while not path.exists():
        if abort.exists():
            raise SystemExit(f"prover ABORT while waiting for {what}: {abort.read_text()[:300]}")
        if proc.poll() is not None:
            raise SystemExit(f"prover exited {proc.returncode} before {what}")
        if time.time() - t0 > timeout:
            raise SystemExit(f"timeout waiting for {what}")
        time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    argv = legacy_gate(argv)
    if argv is None:
        return 2
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(LEGACY_FLAG, action="store_true", help="REQUIRED: acknowledge this is the DIRECTIVE-v4 legacy driver (see banner)")
    ap.add_argument("--record", required=True, help="record root (adapter/production must exist)")
    ap.add_argument("--instance-dir", required=True)
    ap.add_argument("--population", required=True, help="population.compact.json (per-instance population/v2)")
    ap.add_argument("--instances", type=int, required=True)
    ap.add_argument("--py-vllm", required=True)
    ap.add_argument("--semantic-profile", required=True)
    ap.add_argument("--case", default="B0")
    ap.add_argument("--checkpoints", default=str(POC / "manifests" / "checkpoints.json"))
    ap.add_argument("--arena-gb", type=float, default=8.0)
    ap.add_argument("--condition-nonempty", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600.0)
    a = ap.parse_args(argv)
    rec = Path(a.record)
    prod = rec / "adapter" / "production"
    if not (prod / "production.jsonl").exists():
        raise SystemExit(f"{prod}: no production.jsonl (serve first)")
    idir = Path(a.instance_dir)
    inst_path = idir / "protocol_instance.json"
    chron = rec / "chronology"
    sel = rec / "adapter" / "selective"
    hs = rec / "handshake"
    logs = rec / "logs"
    for d in (chron, sel, hs):
        if d.exists():
            raise SystemExit(f"{d} exists: a protocol run starts from nothing (an aborted audit's chronology is evidence, not scratch)")
    logs.mkdir(parents=True, exist_ok=True)
    hs.mkdir(parents=True)
    T: dict[str, float] = {}
    events: list[dict] = []

    def ev(name: str, **kw) -> None:
        events.append({"utc": utc(), "event": name, **kw})
        print(f"[rec_p1_protocol] {utc()} {name} " + " ".join(f"{k}={str(v)[:80]}" for k, v in kw.items()), flush=True)

    # pin the instance and its evidence in the record BEFORE anything is committed or drawn
    pinned = {}
    for name in ("protocol_instance.json", "compact_validation.json", "maps.json", "index.json", "weights_authorized.json", "expected_work.json"):
        if (idir / name).exists():
            shutil.copyfile(idir / name, rec / name)
            pinned[name] = sha256_file(rec / name)
    pop_sha = sha256_file(Path(a.population))
    inst = ps.load_instance(inst_path)
    idig = ps.instance_digest(inst)
    ev("instance_pinned", instance_digest=idig, files=list(pinned), population_file_sha256=pop_sha)
    registered = json.loads((idir / "weights_authorized.json").read_text())

    # --- the prover (phase 0 runs immediately) ---
    prover_cmd = [a.py_vllm, "-m", "verity_vllm_adapter.selective_prover", "--case", a.case, "--checkpoints", a.checkpoints, "--production", str(prod), "--out", str(sel),
                  "--handshake", str(hs), "--adapter-revision", "lane/record", "--semantic-profile", a.semantic_profile, "--arena-gb", str(a.arena_gb),
                  "--weights-format", "2", "--rows-per-leaf", "1"]
    env = {**os.environ, "PYTHONPATH": str(POC), "VLLM_BATCH_INVARIANT": "1"}
    t0 = time.perf_counter()
    plog = (logs / "selective-prover.log").open("w")
    prover = subprocess.Popen(prover_cmd, stdout=plog, stderr=subprocess.STDOUT, env=env, cwd=str(POC))
    ev("prover_started", cmd=" ".join(prover_cmd))
    try:
        wait_for(hs / READY0, hs / ABORT, prover, a.timeout, READY0)
        T["phase0_wall_s"] = round(time.perf_counter() - t0, 1)
        ready0 = json.loads((hs / READY0).read_text())
        phase0 = json.loads((sel / "phase0.json").read_text())
        c0 = json.loads((sel / "commitments_phase0.json").read_text())
        boundary_digest = sha256_file(sel / "commitments_phase0.json")
        assert boundary_digest == ready0["commitments_phase0_sha256"] == phase0["commitments_phase0_sha256"]
        prover_weights_root = c0["trees"]["weights"]["root"]
        weights_check = {"registered_root": registered["expected_root"], "prover_root": prover_weights_root, "equal": prover_weights_root == registered["expected_root"],
                         "registered_leaf_count": registered["leaf_count"], "prover_leaf_count": c0["trees"]["weights"]["leaf_count"],
                         "source": "weights_authorized.json (ROT-INPUT: the registered program's weight commitment, computed from the pinned checkpoint before any prover artefact) "
                                   "vs commitments_phase0.json#trees.weights.root", "label": "MEASURED"}
        ev("phase0", boundary_digest=boundary_digest, requests=phase0["requests"], boundary_trees=phase0["boundary_trees"], weights_root_equal=weights_check["equal"])
        # RU <-> request: the prover's canonical order (sorted request ids)
        prod_rows = [json.loads(l) for l in (prod / "production.jsonl").read_text().splitlines() if l.strip()]
        request_ids = sorted(r["request_id"] for r in prod_rows)
        if len(request_ids) != a.instances:
            raise SystemExit(f"{len(request_ids)} requests served but the instance has {a.instances} program instances")
        ru_map = {"schema": "veritor/rec/ru-map/v1", "rule": "RU index r = position of the request id in sorted(request_ids) (the prover's canonical order); "
                  "one RU per request (P1/cu=whole); global VU = r * N_V + i", "rus": [{"ru": r, "request_id": rid} for r, rid in enumerate(request_ids)]}
        (rec / "ru_map.json").write_text(json.dumps(ru_map, indent=1) + "\n")
        # --- RU challenge ---
        t1 = time.perf_counter()
        # [A10 WR-2] the library default is now in-process custody; the legacy reproduction asks for the file explicitly (prints a LEGACY line)
        doc = ps.stage1(inst_path, Path(a.population), boundary_digest, chron, instances=a.instances, condition_nonempty=a.condition_nonempty, legacy_seed_file=True)
        T["stage1_s"] = round(time.perf_counter() - t1, 2)
        rq = next(e for e in doc["events"] if e["name"] == "r_q")
        J = list(rq["selection"])
        selected_rids = [request_ids[r] for r in J]
        ev("r_q", J=J, selected=selected_rids, redraws=rq.get("redraws", 0), digest=rq["digest"])
        challenge = {"selected": selected_rids, "T_0": next(e["digest"] for e in doc["events"] if e["name"] == "seed_commitment"), "r_q": rq["digest"],
                     "commitments_phase0_sha256": boundary_digest, "utc": utc(), "sampler": ps.SAMPLER, "instance_digest": idig, "J": J}
        tmp = hs / (CHALLENGE_FILE + ".tmp")
        tmp.write_text(json.dumps(challenge, indent=1) + "\n")
        os.replace(tmp, hs / CHALLENGE_FILE)
        # --- replay of J, interiors committed ---
        t2 = time.perf_counter()
        wait_for(hs / READY2, hs / ABORT, prover, a.timeout, READY2)
        prover.wait(timeout=600)
        T["phase2_wall_s"] = round(time.perf_counter() - t2, 1)
        interior_digest = sha256_file(sel / "commitments.json")
        ready2 = json.loads((hs / READY2).read_text())
        assert interior_digest == ready2["commitments_sha256"]
        pj = json.loads((sel / "selective_prover.json").read_text())
        ev("phase2", interior_digest=interior_digest, k=pj["k"], ru_leaves=pj["phase2"]["ru_leaves"], replay_units=pj["phase2"]["replay_units_committed"])
        # weights / request roots must be unchanged by the replay (nothing recommitted)
        c2 = json.loads((sel / "commitments.json").read_text())
        unchanged = {k: (c2["trees"].get(k, {}).get("root") == v.get("root")) for k, v in c0["trees"].items()}
        ev("no_recommit_check", all_unchanged=all(unchanged.values()), trees_phase0=len(c0["trees"]), trees_phase2=len(c2["trees"]))
        # --- VU challenge ---
        t3 = time.perf_counter()
        doc = ps.stage2(inst_path, Path(a.population), interior_digest, chron, instances=a.instances)
        T["stage2_s"] = round(time.perf_counter() - t3, 2)
        ps.reveal(chron)
        problems = ps.verify(inst_path, Path(a.population), chron, instances=a.instances)
        rs = next(e for e in doc["events"] if e["name"] == "r_s")
        selected = json.loads((chron / "selected.json").read_text())
        ev("r_s", Q=len(rs["selection"]), candidates=rs.get("candidates"), kinds=len(selected["kinds"]), chronology_rederived=(not problems), problems=problems)
    finally:
        if prover.poll() is None:
            prover.kill()
        plog.close()
    T["total_s"] = round(time.perf_counter() - t0, 1)
    out = {"schema": "veritor/rec/p1-protocol/v1", "utc": utc(), "instance_digest": idig, "pinned_sha256": pinned, "population_file_sha256": pop_sha,
           "instances": a.instances, "ru_map": ru_map, "boundary_digest": boundary_digest, "interior_digest": interior_digest, "J": J, "selected_requests": selected_rids,
           "weights_check": weights_check, "no_recommit": {"per_tree_unchanged": unchanged, "all": all(unchanged.values())},
           "selected_vus": len(rs["selection"]), "selected_kinds": selected["kinds"], "chronology_problems": problems,
           "prover": {"phase0": phase0, "phase2_timing": pj.get("timing"), "phase2": {k: pj["phase2"].get(k) for k in ("replay_units_committed", "ru_leaves", "replayed_tokens", "capture", "trajectory_followed")},
                      "bytes": pj.get("bytes"), "log": str(logs / "selective-prover.log")},
           "timing": T, "events": events,
           "provenance": {"label": "MEASURED", "hardware": "record-gpu (1x RTX 4090)", "script": "scripts/rec_p1_protocol.py", "sampler": ps.SAMPLER,
                          "note": "phase 0 = request commitments (prompt/output tokens, params, seed) + weights tree; phase 2 = replay of J only, interiors = operator boundary rows; "
                                  "weights and request inputs are referenced, never recommitted"}}
    (rec / "protocol.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    with (rec / "costlog.jsonl").open("a") as f:
        f.write(json.dumps({"stage": "commit_inputs_outputs_seed_weights (phase 0)", "class": "serving", "seconds": T["phase0_wall_s"], "hardware": "RTX 4090 (record-gpu)",
                            "batch": f"{a.instances} requests", "label": "MEASURED", "note": "includes engine build + warm-up (selective_prover timing has the split)"}) + "\n")
        f.write(json.dumps({"stage": "replay_and_commit_interiors (phase 2)", "class": "replay", "seconds": T["phase2_wall_s"], "hardware": "RTX 4090 (record-gpu)",
                            "batch": f"{len(J)} selected request(s) in one batch", "label": "MEASURED"}) + "\n")
        f.write(json.dumps({"stage": "ru_challenge + vu_challenge (partition_sample stage1+stage2+verify)", "class": "verifier", "seconds": round(T["stage1_s"] + T["stage2_s"], 2),
                            "hardware": "record-gpu CPU", "batch": f"{a.instances} RUs, {len(rs['selection'])} VUs", "label": "MEASURED"}) + "\n")
    print(f"[rec_p1_protocol] DONE: J={J} ({selected_rids}), |Q|={len(rs['selection'])}, weights root equal={weights_check['equal']}, no recommit={all(unchanged.values())}, "
          f"chronology {'REDERIVED' if not problems else 'MISMATCH ' + str(problems)}; {T}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
