"""The CORE prover PROCESS (generic protocol demonstration; NOT vLLM inference) — the peer of ``verity_vllm_adapter.selective_prover`` in the
``verifier_v3/driver.py`` handshake, for ``DemoCore_v1``.  Mirrors its files and events exactly:

    phase 0  values (verity.ir reference interpreter) -> boundary trees (step 1) -> <out>/commitments_phase0.json, phase0.json
             -> <handshake>/READY_PHASE0 {commitments_phase0_sha256}
    wait     <handshake>/replay_challenge.json  (the verifier's J, r_q, instance digest; written only after its step-2 checks and r_q)
    phase 2  interior trees of J minus F (step 3) -> <out>/commitments.json, interior/<rid>/0/, exemption_request.json (response_record.build),
             selective_prover.json -> <handshake>/READY_PHASE2 {commitments_sha256, exemption_request_sha256}
    exit 0   (the verifier's step-4 checks, r_s and step 5 follow; obligations are built after Q by ``runner.py``'s prover side).

``--exempt`` = a PREDECLARED controlled exemption test (the RUs the prover will put in F if selected); labelled as such in the request.
Invoked either as ``python -m e2e.demo_core.prover --out … --handshake …`` (the runner) or through ``shim.py`` when the shipped driver's
argv (``-m verity_vllm_adapter.selective_prover --out … --handshake …``) is translated.  Never reads the verifier's seed; ABORT on any error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from verity_vllm_adapter import response_record as rr

from . import circuit as C, producer as PR, record as RC

READY0, READY2, CHALLENGE, ABORT = "READY_PHASE0", "READY_PHASE2", "replay_challenge.json", "ABORT"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_fixture(fx: Path) -> tuple[C.Partition, PR.Values, Any]:
    f = json.loads((fx / "fixture.json").read_text())
    prog, d = C.build(int(f["N"]), int(f["R"]))
    part = C.partition(int(f["N"]), int(f["R"]), prog, d)
    vals = PR.produce(part, {k: list(v) for k, v in f["inputs"].items()}, prog)
    return part, vals, f


def forge(part: C.Partition, vals: PR.Values, gate: int) -> PR.Values:
    """NEGATIVE TEST: a copy of ``vals`` whose word at ``gate`` (an interior gate: a VU output that is not an RU output) is ``value xor 1``.
    Downstream gates are NOT recomputed: the dishonest prover claims the stage's output tuple as evaluated, so exactly one VU relation
    (the one producing ``gate``, and the consumer reading it) is false in the committed record."""
    if not any(ru.lo <= gate < ru.hi and gate not in set(ru.returns) for ru in part.rus):
        raise ValueError(f"--forge-gate {gate}: not an interior gate of any RU")
    t = dict(vals.transcript)
    t[int(gate)] = int(t[int(gate)]) ^ 1
    return PR.Values(vals.partition, vals.inputs, t)


def build_record(fx: Path, sel: Path, *, interior: str = "ru") -> tuple[C.Partition, PR.Values, RC.CoreRecord]:
    part, vals, _ = load_fixture(fx)
    rec = RC.CoreRecord(part, vals, sel, interior=interior)
    rec.commit_phase0()
    return part, vals, rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DemoCore_v1 prover process (driver handshake peer)")
    ap.add_argument("--fixture", default=None, help="dir with fixture.json {N, R, inputs:{a,b,c}} (default: <record>/fixture, record = --out's grandparent)")
    ap.add_argument("--out", required=True, help="adapter/selective output dir")
    ap.add_argument("--handshake", required=True)
    ap.add_argument("--exempt", default="", help="comma-separated RU indices of the PREDECLARED controlled exemption test (F = these ∩ J)")
    ap.add_argument("--exemption-policy", default="gamble")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--instance-dir", default=None, help="[D1] the pinned instance dir (E9's --prover-cmd substitutes {instance_dir}); the RU-id base is its epoch's one request id (steps.ru_ids_of: <request>-ru<r>)")
    ap.add_argument("--request", default="per-ru", choices=list(RC.REQUEST_MODES),
                    help="request spelling: 'per-ru' (core-ru<r>, one request per RU; every archived run) | 'single' (E6's frozen CORE instance: one request core-0, RU r = step r)")
    ap.add_argument("--interior", default="ru", choices=list(RC.INTERIOR_MODES),
                    help="interior tree spelling: 'ru' (ru/<req>/0, today's shipped ck_elem) | 'e2' (interior/<req>/<step>, E2's interior-layout/chunks-4096/v1 through e2e.interior; the D1 path)")
    ap.add_argument("--forge-gate", type=int, default=None,
                    help="NEGATIVE TEST (labelled): the prover commits a FORGED word for this interior gate (committed value xor 1) -- the record is dishonest by "
                         "construction; the verifier must REJECT at the proof / opening step (the relation of that VU is false)")
    ap.add_argument("--early-rs", action="store_true",
                    help="NEGATIVE TEST (E9's a8 standin convention 'early-rs'): after r_q and BEFORE READY_PHASE2 the prover writes handshake/REQUEST_r_s and waits for "
                         "the verifier's named REFUSED_r_s; the outcome is recorded in selective_prover.json#early_challenge; then the honest run continues")
    # [D1] the shipped driver's spawn site passes Serve_v1-only arguments (--case, --checkpoints, --production, --adapter-revision,
    # --semantic-profile, --arena-gb, --expressibility, --unsupported-kinds): accepted and ignored, recorded in phase0.json#ignored_argv
    a, _unknown = ap.parse_known_args(argv)
    RC.set_request_mode(a.request)
    if a.instance_dir:  # the RU ids follow the pinned epoch's request id (E9's ru_ids_of), so the record and the verifier agree by construction
        epoch = json.loads((Path(a.instance_dir) / "protocol_instance.json").read_text()).get("epoch") or {}
        rids = list(epoch.get("request_ids") or [])
        if len(rids) == 1:
            RC.set_ru_base(str(rids[0]))
    sel, hs = Path(a.out), Path(a.handshake)
    fx = Path(a.fixture) if a.fixture else sel.parents[1] / "fixture"
    hs.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        part, vals, _f = load_fixture(fx)
        if a.forge_gate is not None:
            vals = forge(part, vals, int(a.forge_gate))
        rec = RC.CoreRecord(part, vals, sel, interior=a.interior)
        rec.commit_phase0()
        c0_sha = sha256_file(sel / "commitments_phase0.json")
        PR.write_value_records(vals, sel.parent / "values")
        (sel / "phase0.json").write_text(json.dumps({"schema": "veritor/demo-core/phase0/v1", "utc": utc(), "commitments_phase0_sha256": c0_sha, "profile_id": rec.profile_id,
                                                     "program_digest": part.program_digest, "plan_digest": part.digest, "N_R": part.n_r, "N_V": part.n_v,
                                                     "boundary_trees": sorted(t for t in rec.trees if t.startswith("boundary/")), "request_ids": [RC.request_id(r) for r in range(part.n_r)],
                                                     "evaluator": "verity.ir.evaluate.evaluate_region (reference interpreter)", "seconds": round(time.perf_counter() - t0, 3),
                                                     "fixture": str(fx), "ignored_argv": list(_unknown),
                                                     "label": "generic protocol demonstration (DemoCore_v1); NOT vLLM inference"}, indent=1) + "\n")
        _atomic(hs / READY0, {"utc": utc(), "commitments_phase0_sha256": c0_sha, "prover": "e2e.demo_core.prover"})
        # ---- wait for the verifier's replay challenge (J) ----
        t1 = time.time()
        while not (hs / CHALLENGE).exists():
            if (hs / ABORT).exists():
                print(f"[demo_core.prover] verifier ABORT: {(hs / ABORT).read_text()[:200]}", file=sys.stderr)
                return 3
            if time.time() - t1 > a.timeout:
                (hs / ABORT).write_text("prover: timeout waiting for replay_challenge.json\n")
                return 4
            time.sleep(0.2)
        ch = json.loads((hs / CHALLENGE).read_text())
        J = sorted(int(r) for r in ch["J"])
        if ch.get("commitments_phase0_sha256") != c0_sha:
            raise RuntimeError("challenge names another phase-0 commitment")
        # ---- step 3: interiors of J minus F; F = the predeclared test set ∩ J (never chosen from anything the verifier drew after r_q) ----
        early: dict[str, Any] | None = None
        if a.early_rs:  # NEGATIVE TEST: ask for r_s before the step-3 commitment; the verifier must refuse by name and draw nothing
            (hs / "REQUEST_r_s").write_text(json.dumps({"utc": utc(), "requested_by": "e2e.demo_core.prover --early-rs (NEGATIVE TEST)", "state": "after r_q, before READY_PHASE2"}) + "\n")
            t_e = time.time()
            while not (hs / "REFUSED_r_s").exists() and time.time() - t_e < min(60.0, a.timeout):
                if (hs / ABORT).exists():
                    break
                time.sleep(0.1)
            early = {"test": "early-rs", "refused": (hs / "REFUSED_r_s").exists(), "wait_s": round(time.time() - t_e, 3),
                     "refusal": (json.loads((hs / "REFUSED_r_s").read_text()) if (hs / "REFUSED_r_s").exists() else None),
                     "r_s_delivered_early": (hs / "selected.json").exists() or any(p.name.startswith("r_s") for p in hs.iterdir())}
        declared_F = sorted({int(x) for x in a.exempt.split(",") if x.strip()})
        F = [r for r in J if r in set(declared_F)]
        retained = [r for r in J if r not in set(F)]
        t2 = time.perf_counter()
        rec.commit_interior(retained)
        commit_s = round(time.perf_counter() - t2, 4)
        per_ru: dict[int, dict[str, Any]] = {}
        for r in J:
            ru = part.rus[r]
            # (a) consistency: the prover re-evaluates f_r (the stage) on the committed inputs and compares with the committed output tuple
            h_in = vals.words(list(ru.args[:part.N])) if r > 0 else vals.words(list(part.inputs["a"]))
            recomputed = PR.stage_eval(part, r, h_in, vals.words(list(part.inputs["b"])), vals.words(list(part.inputs["c"])))
            committed_out = vals.words(list(ru.returns))
            consistent = recomputed == committed_out
            cons = {"status": rr.CONSISTENT if consistent else rr.INCONSISTENT,
                    "method": "verity.ir reference interpreter: f_r(committed h_{r-1}, b, c) recomputed word-for-word == the committed output tuple h_r (RU r's boundary leaf)",
                    "evidence": {"words": part.N, "equal": consistent, "output_leaf": rec.gate_pos[ru.returns[0]][1]}, "established_by": "prover"}
            exempt = r in set(F)
            resp = rr.response(step3_committed=not exempt, exemption_requested=exempt,
                               exemption_reason=("PREDECLARED controlled exemption test (run config --exempt): the RU's step-3 values are withheld to exercise F ⊆ J, |F| <= B_ex, Q over J minus F" if exempt else None),
                               failure_stage=("policy" if exempt else None), failure_cause=("controlled test, not a replay failure" if exempt else None),
                               replay_s=0.0, commit_s=commit_s if not exempt else None)
            per_ru[r] = {"consistency": cons, "response": resp,
                         "unsupported": rr.unsupported(_kind_counts(ru), [], source="e2e.demo_core: every kind has a registered ck_elem builder (approved guest 9fd00e60); unsupported set = {}")}
        exreq = rr.build(instance_digest=str(ch["instance_digest"]), r_q=str(ch["r_q"]), J=J, ru_request={r: RC.request_id(r) for r in J}, per_ru=per_ru, B_ex=int(ch["B_ex"]),
                         policy=a.exemption_policy)
        exreq["label"] = "generic protocol demonstration (DemoCore_v1); F = predeclared controlled test set ∩ J" if F else "generic protocol demonstration (DemoCore_v1); F = ∅"
        (sel / "exemption_request.json").write_text(json.dumps(exreq, indent=1, sort_keys=True) + "\n")
        c2_sha, ex_sha = sha256_file(sel / "commitments.json"), sha256_file(sel / "exemption_request.json")
        pj = {"schema": "veritor/demo-core/selective-prover/v1", "utc": utc(), "k": len(J), "J": J, "F": F, "retained": retained,
              "phase2": {"replay_units_committed": len(retained), "ru_leaves": sum(rec.trees[rec.interior_tree(r)].leaf_count for r in retained),
                         "interior_words": sum(len(lf.words) for lf in rec.leaves.values() if lf.origin == "interior"),
                         "replayed_tokens": None, "capture": "verity.ir reference interpreter transcript (no capture hook: the values are the interpreter's)",
                         "trajectory_followed": None, "exemption_request": F, "interior_layout": rec.interior_layout, "interior_mode": a.interior, "request_mode": a.request, "ru_id_base": RC.ru_base(),
                         "early_challenge": early, "forged_gate": a.forge_gate},
              "timing": {"phase0_s": round(time.perf_counter() - t0, 3), "commit_interior_s": commit_s}, "bytes": {"commitments.json": (sel / "commitments.json").stat().st_size},
              "label": "generic protocol demonstration (DemoCore_v1); NOT vLLM inference"}
        (sel / "selective_prover.json").write_text(json.dumps(pj, indent=1, sort_keys=True) + "\n")
        _atomic(hs / READY2, {"utc": utc(), "commitments_sha256": c2_sha, "exemption_request_sha256": ex_sha, "F": F, "retained": retained, "prover": "e2e.demo_core.prover"})
        print(f"[demo_core.prover] done: J={J} F={F} retained={retained}; interior trees {len(retained)}; {round(time.perf_counter() - t0, 2)} s", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001 -- a prover failure is an ABORT the verifier records, never a silent exit
        (hs / ABORT).write_text(f"prover: {type(e).__name__}: {e}\n")
        raise


def _kind_counts(ru: C.RU) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in ru.vus:
        out[v.kind] = out.get(v.kind, 0) + 1
    return out


def _atomic(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    sys.exit(main())
