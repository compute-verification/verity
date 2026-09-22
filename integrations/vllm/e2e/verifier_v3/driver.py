#!/usr/bin/env python
"""[A3 contract 2026-09-10] The contract-ordered live driver: ``scripts/rec_p1_protocol.py`` (L1's verifier-side driver, DIRECTIVE v3
§1.1 steps 2-5) rebuilt on docs/PROTOCOL.md's five steps.  The prover is ``verity_vllm_adapter.selective_prover`` (this branch: it writes
``exemption_request.json`` in phase 2).  Differences from ``rec_p1_protocol.py``, each a conformance row:

  step 2 (R5/R13/R16)  BEFORE r_q: ``verifier_v3.steps.step2_check`` -- every declared request has a boundary tree, weights root == registered
                       (REFUSED, no longer merely recorded), y well-typed, served outputs == the committed boundary record (A2's reader); the
                       instance must declare the epoch (its request set is compared with the served set) and ``stage1`` receives the covered ids;
  R7                   the seed is held in this process (``stage1`` default since WR-2; ``legacy_seed_file`` never set here): no ``seed.hex`` in the record; the deployment
                       requirement stays: run THIS process on a host the prover cannot read (the legacy driver ran on the prover's host);
  step 4 (R6/R8/R16)   BEFORE r_s: ``steps.step4_check`` -- F ⊆ J, |F| <= B_ex (RU count), every retained RU's interior committed, phase-0 trees
                       unchanged (REFUSED, no longer merely recorded), both R17 fields present; then ``stage2(exemptions=F, committed_rus=...)``
                       draws Q over J minus F and records the ``exemption_request`` event;
  after Q (R17/R18)    ``steps.ru_fields`` -> ``<record>/verify/ru_fields.json`` (selected_checks_establishable from the actual selection, the
                       completion risk and the sampled unsupported identities).

The RU <-> request map is the epoch's declared request order (``epoch.request_ids`` sorted == the prover's canonical order); a served set that
differs from the declaration is a step-2 refusal.  Nothing else about the phases changed.  This driver cannot run on a CPU dev machine (vLLM
prover); its step functions are unit-tested in ``tests/contract/a3_sequence``.

[E9 EOD 2026-09-11] (brief interfaces 4 and 7):
  transcript          ``steps.Transcript`` is the explicit state machine: ``release("r_q")`` right before ``ps.stage1`` requires ``step2_complete``,
                      ``release("r_s")`` right before ``ps.stage2`` requires ``step4_complete`` (else ``PREMATURE_CHALLENGE``, exit 41, by name).  A
                      prover may write ``handshake/REQUEST_r_q`` / ``REQUEST_r_s`` at any time: ``wait_for`` answers a premature one with
                      ``handshake/REFUSED_<challenge>`` (the named refusal), records the event, and the run continues -- a request never draws.
                      ``verify/transcript.json`` + ``verify/seed_custody.json`` are written on every exit path.
  step 4 (+)          ``V3.interior_layout`` BEFORE r_s: with interior values in play (E6's kind table declares interior-producer kinds, or the
                      instance pins ``commitments.interior_layout``, or ``commitments.json`` carries ``interior/`` trees) the pin must be present,
                      equal E2's ``layout_digest()``, and every REQUIRED interior region of every retained RU must be supplied (E2's
                      ``validate_supplied``); else ``STEP4_REFUSED`` by name.  Fail-closed when the verifier cannot validate.
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
from typing import Any

HERE = Path(__file__).resolve().parent  # e2e/verifier_v3
POC = HERE.parent.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from e2e import partition_sample as ps  # noqa: E402
from e2e.verifier_v3 import steps  # noqa: E402

READY0, READY2, CHALLENGE_FILE, ABORT = "READY_PHASE0", "READY_PHASE2", "replay_challenge.json", "ABORT"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# [W-A9-4] machine-readable exit: every exit path writes <record>/verifier_exit.json {reason_code, step, problems, exit_code, utc} and exits
# with the code of its class, so A9 classifies by code, never by message text.
EXIT_CODES = {
    "OK": 0,                      # steps 1-4 done, chronology re-derived
    "CHRONOLOGY_MISMATCH": 1,     # steps done but partition_sample.verify re-derivation disagrees
    "USAGE": 2,                   # bad arguments / preconditions (no production.jsonl, record dir exists, legacy instance)
    "STEP2_REFUSED": 20,          # verifier refused at step 2 (before r_q): v_in(C) != x, v_out(C) != y, layout, request set, ...
    "STEP4_REFUSED": 40,          # verifier refused at step 4 (before r_s): F, interiors, output copies, ...
    "PROVER_ABORT": 30,           # the prover wrote ABORT (a prover abort, not a verifier rejection)
    "PROVER_EXIT": 31,            # the prover process ended before the awaited handshake file
    "TIMEOUT": 32,                # the awaited handshake file never appeared
    "INSTANCE_REFUSED": 21,       # [A8 / integrator 23:18Z] protocol_instance.json fails validate_instance (e.g. prompt_len != LP) or the named
                                  # --instance-format does not match the instance: refused before anything is committed or drawn
    "PREMATURE_CHALLENGE": 41,    # [E9] the verifier's own path asked the transcript for r_q / r_s before its prerequisite step was complete
                                  # (a driver defect, named; a PROVER's premature request is refused in the handshake and never ends the run)
}
_EXIT_REC: dict[str, Path] = {}
REQUEST_PREFIX, REFUSED_PREFIX = "REQUEST_", "REFUSED_"  # handshake files: the prover's request for a challenge / the verifier's named refusal


def exit_with(reason_code: str, message: str, *, step: str | None = None, problems: list[str] | None = None) -> "SystemExit":
    """Write verifier_exit.json (when the record dir is known) and return the SystemExit to raise."""
    code = EXIT_CODES[reason_code]
    doc = {"schema": "veritor/verifier-exit/v1", "utc": utc(), "reason_code": reason_code, "exit_code": code, "step": step, "problems": list(problems or []),
           "message": message[:1000], "prover_abort_text": None}
    rec = _EXIT_REC.get("rec")
    # [W-A9-8] the stage timers measured so far (phase0_wall_s, stage1_s, phase2_wall_s ...) travel with every exit, and the phase-0 serving
    # cost is appended to costlog.jsonl even when the run is refused after it -- the record's protocol.json is never written on a refusal
    T = dict(_EXIT_REC.get("timing") or {})
    t0 = _EXIT_REC.get("t0")
    if t0 is not None:
        T["total_s"] = round(time.perf_counter() - float(t0), 1)
    doc["timing"] = T
    tr = _EXIT_REC.get("tr")
    if tr is not None:  # [E9] the transcript state at exit (which challenges were released) travels with every exit
        doc["transcript_state"] = tr.state
        doc["challenge_requests_refused"] = len(tr.refusals)
    if rec is not None:
        try:
            ab = rec / "handshake" / ABORT
            if reason_code == "PROVER_ABORT" and ab.exists():
                doc["prover_abort_text"] = ab.read_text()[:1000]
            (rec / "verifier_exit.json").write_text(json.dumps(doc, indent=1) + "\n")
            if reason_code != "OK" and "phase0_wall_s" in T:
                with (rec / "costlog.jsonl").open("a") as f:
                    f.write(json.dumps({"stage": "commit_inputs_outputs_seed_weights (phase 0)", "class": "serving", "seconds": T["phase0_wall_s"],
                                        "batch": _EXIT_REC.get("batch"), "label": "MEASURED", "note": f"run exited {reason_code} after phase 0 (includes engine build + warm-up)"}) + "\n")
        except OSError:
            pass
    return _SystemExitWithCode(code, f"[verifier_v3.driver] {reason_code}: {message}")


class _SystemExitWithCode(SystemExit):
    def __init__(self, code: int, message: str) -> None:
        print(message, file=sys.stderr, flush=True)
        super().__init__(code)


def answer_challenge_requests(hs: Path, tr: "steps.Transcript", events: list[dict], ev=None) -> int:
    """[E9, brief interface 7] A prover's ``handshake/REQUEST_r_q`` / ``REQUEST_r_s`` is answered from the transcript state: premature -> the
    named refusal ``handshake/REFUSED_<challenge>`` (``PREMATURE_CHALLENGE``, state, required) + transcript event; the request file is kept as
    ``REQUEST_<c>.refused-<n>`` (evidence).  A request that is NOT premature is left in place: the challenge is delivered through the normal
    path (``replay_challenge.json`` after r_q; nothing is drawn because of a request).  Returns the number of refusals written."""

    n = 0
    for c in steps.CHALLENGE_REQUIRES:
        req = hs / f"{REQUEST_PREFIX}{c}"
        if not req.exists():
            continue
        e = tr.refuse_request(c, requested_by="prover (handshake REQUEST_" + c + ")")
        if e is None:
            continue
        k = len([p for p in hs.iterdir() if p.name.startswith(f"{REQUEST_PREFIX}{c}.refused-")])
        doc = {**e.to_json(), "utc": utc(), "refusal_index": k, "note": "refused by name; the run continues; nothing was drawn"}
        tmp = hs / f"{REFUSED_PREFIX}{c}.tmp"
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, hs / f"{REFUSED_PREFIX}{c}")
        os.replace(req, hs / f"{REQUEST_PREFIX}{c}.refused-{k}")
        events.append({"utc": doc["utc"], "event": f"{c}_REQUEST_REFUSED", **{k2: doc[k2] for k2 in ("reason_code", "state", "required", "requested_by")}})
        n += 1
    return n


def wait_for(path: Path, abort: Path, proc: subprocess.Popen, timeout: float, what: str, *, hs: Path | None = None, tr: "steps.Transcript | None" = None,
             events: list[dict] | None = None, ev=None) -> None:
    t0 = time.time()
    while not path.exists():
        if hs is not None and tr is not None and events is not None:
            answer_challenge_requests(hs, tr, events, ev)  # [E9] a premature REQUEST_r_q / REQUEST_r_s is refused by name while we wait
        if abort.exists():
            raise exit_with("PROVER_ABORT", f"prover ABORT while waiting for {what}: {abort.read_text()[:300]}", step=what)
        if proc.poll() is not None:
            raise exit_with("PROVER_EXIT", f"prover exited {proc.returncode} before {what}", step=what)
        if time.time() - t0 > timeout:
            raise exit_with("TIMEOUT", f"timeout waiting for {what}", step=what)
        time.sleep(0.5)
    if hs is not None and tr is not None and events is not None:
        answer_challenge_requests(hs, tr, events, ev)  # a request written together with the READY file is still premature: the checks have not run


def boundary_outputs(sel: Path) -> dict[str, list[int]] | None:
    """The committed boundary records' output ids (A2's reader); None when the reader is unavailable (the check is then reported NOT DONE)."""

    try:
        from verity_vllm.capture import CaptureReader

        reader = CaptureReader(sel)
        return {rid: list(reader.boundary(rid).output_token_ids) for rid in reader.request_ids()}
    except Exception:  # noqa: BLE001 -- reported, never silently passed
        return None


def boundary_inputs(sel: Path, c0: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """[A10 WR-5 / user directive 20:10Z, R1] The committed INPUT side of every request, read from the AUTHENTICATED LEAVES.

    For each request the verifier (1) builds the boundary tree from the record's leaf list (A2's ``Boundary.leaves``: one ``prompt_token/t``
    int32 leaf per prompt position, ``sampling_params`` and ``seed`` canonical-JSON leaves, ...) and checks its root and leaf count against
    the committed phase-0 tree (``root_ok``; the committed ``ids_digest`` is checked by V3/V5 over the same id list); (2) takes the prompt,
    seed and sampling params FROM THAT LEAF LIST -- the very objects hashed into the root -- not from ``boundary.json``'s fields: the
    ``prompt_token/t`` leaves are decoded as int32, required to be exactly ``t = 0..n-1`` (``prompt_contiguous``: length and ordering are
    facts of the committed leaf set) and returned in t order.  ``step2_check`` then compares ``sha256(json([ids]))`` (values, count, order)
    with the instance's declared ``prompt_sha256`` and ``prompt_len``.  None when A2's reader is unavailable (a refusal in step2_check)."""

    try:
        import numpy as np

        from verity_vllm import identity as vid
        from verity_vllm.capture import CaptureReader, boundary_tree_id
        from verity_vllm.commit import TreeCache

        reader = CaptureReader(sel)
        cache = TreeCache(reader, recompute_leaves=True)
        out = {}
        for rid in reader.request_ids():
            b = reader.boundary(rid)
            tid = boundary_tree_id(rid)
            t0 = (c0.get("trees") or {}).get(tid) or {}
            leaves = b.leaves(reader.profile_id)  # the leaf list the tree is built from (same call boundary_tree uses)
            try:
                tree = cache.tree(tid)
                root_ok = tree.root.hex() == t0.get("root") and int(t0.get("leaf_count", -1)) == tree.leaf_count == len(leaves)
            except Exception:  # noqa: BLE001
                root_ok = False
            prompt: dict[int, int] = {}
            prompt_leaves: dict[int, dict[str, Any]] = {}  # t -> {dtype, shape, hex}: the authenticated leaf as A2's check_input_layout reads it
            seed = None
            params: dict[str, Any] | None = None
            for v in leaves:
                cid = vid.parse(v.canonical_id)
                if cid.kind != "boundary" or cid.request_id != rid:
                    root_ok = False  # a foreign leaf in this request's tree: never authenticated as this request's input
                    continue
                if cid.field == "prompt_token":
                    prompt[int(cid.t)] = int(np.frombuffer(v.data, dtype="<i4")[0]) if v.dtype in ("int32", "<i4") else int(np.frombuffer(v.data, dtype=v.dtype)[0])
                    prompt_leaves[int(cid.t)] = {"dtype": str(v.dtype), "shape": [int(d) for d in v.shape], "hex": bytes(v.data).hex()}
                elif cid.field == "seed":
                    seed = json.loads(v.data.decode("utf-8"))
                elif cid.field == "sampling_params":
                    params = json.loads(v.data.decode("utf-8"))
            contiguous = sorted(prompt) == list(range(len(prompt))) and len(prompt) > 0
            out[rid] = {"root_ok": bool(root_ok), "prompt_contiguous": contiguous, "prompt_token_ids": [prompt[t] for t in sorted(prompt)], "prompt_leaves": prompt_leaves,
                        "output_token_ids": list(b.output_token_ids), "seed": seed, "sampling_params": dict(params or {}),
                        "source": "authenticated boundary leaves (prompt_token/t int32, seed, sampling_params); root == committed phase-0 root"}
        return out
    except Exception:  # noqa: BLE001 -- reported, never silently passed
        return None


def refuse(hs: Path, events: list[dict], name: str, problems: list[str]) -> None:
    (hs / ABORT).write_text(f"{name}: " + "; ".join(problems) + "\n")
    events.append({"utc": utc(), "event": f"{name}_REFUSED", "problems": problems})
    raise exit_with({"step2": "STEP2_REFUSED", "step4": "STEP4_REFUSED"}[name], f"{name} REFUSED: {problems}", step=name, problems=problems)


def population_barrier(pd: dict, *, hs: Path, events: list[dict], vdir: Path, step2: dict, ev=None) -> list[str]:
    """[user directive 16:04 PT; W-A9-13] The step-2 population check's consequence, BEFORE r_q.  ``pd`` = ``steps.population_derived(...)``.
    Records the result (``step2["population_derived"]``, ``<record>/verify/population_derived.json``); a deriver early exit (``UNAVAILABLE`` /
    ``REFUSED_BY_DERIVER`` / ``ERROR``: the verifier host cannot derive the population -- deriver missing, its pinned census inputs absent, no rules
    for the case) is refused by name at once (``STEP2_REFUSED``, ``verifier_exit.json`` with the deriver's status + problems; nothing drawn);
    a disagreement is returned as a step-2 problem (refused with the other step-2 problems, field named).  Never indexes a missing key."""

    rec = {k: pd.get(k) for k in steps.POPULATION_DERIVED_KEYS}
    step2["population_derived"] = rec
    status, problems = str(rec.get("status")), list(rec.get("problems") or [])
    if ev is not None:
        ev("V2.population_derived", ok=bool(rec.get("ok")), status=status, problems=problems[:3])
    try:
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "population_derived.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    except OSError:  # pragma: no cover
        pass
    if status in steps.POPULATION_DERIVER_EARLY:
        refuse(hs, events, "step2", [f"V2.population_derived {status}: the verifier cannot derive the expected population from the pinned instance -> not accepted "
                                     f"(deployment / configuration fault on the verifier host, not a prover disagreement)"] + problems[:4])
    if not rec.get("ok"):
        return [f"V2.population_derived ({status}): " + "; ".join(problems[:4])]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True, help="record root (adapter/production must exist)")
    ap.add_argument("--instance-dir", required=True, help="protocol_instance.json (with the epoch declared) and its evidence")
    ap.add_argument("--population", required=True, help="population.compact.json (per-instance population/v2)")
    ap.add_argument("--instances", type=int, required=True)
    ap.add_argument("--py-vllm", default=None, help="Serve_v1: the vLLM venv's python for verity_vllm_adapter.selective_prover (required for a serve program)")
    ap.add_argument("--semantic-profile", default=None, help="Serve_v1: passed to the selective prover (required for a serve program)")
    ap.add_argument("--program-kind", default="auto", choices=["auto", "serve", "registered"],
                    help="[E9, D1] 'serve' = Serve_v1 / vLLM (production.jsonl, capture reader, output copies); 'registered' = a generic registered verity.ir program "
                         "(E8's CORE: protocol_instance.json#program.kind == 'registered'); 'auto' reads the instance pin (absent -> serve). The vLLM-only steps are "
                         "NOT_APPLICABLE by name in the chronology on a registered program, never silently skipped")
    ap.add_argument("--prover-cmd", default=None,
                    help="[E9, D1] registered program: the prover's argv as ONE string (shlex), with {out} {handshake} {exemption_policy} {record} {instance_dir} "
                         "{population} substituted (e.g. \"<python> -m e2e.demo_core.prover --fixture <dir> --out {out} --handshake {handshake} --exemption-policy {exemption_policy}\"); "
                         "the prover must speak the handshake of docs/PROTOCOL.md (READY_PHASE0 / replay_challenge.json / READY_PHASE2 / ABORT)")
    ap.add_argument("--case", default="B0")
    ap.add_argument("--checkpoints", default=str(POC / "manifests" / "checkpoints.json"))
    ap.add_argument("--arena-gb", type=float, default=8.0)
    ap.add_argument("--condition-nonempty", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--expressibility", default=None, help="passed to the prover for the R18 count (kinds of class 'exempt' = no checker)")
    ap.add_argument("--unsupported-kinds", default=None, help="JSON list of 'fn|form' kinds without a checker (A5/A6's set); passed to the prover and used for ru_fields")
    ap.add_argument("--exemption-policy", default="gamble", choices=["gamble", "exempt-unsupported"],
                    help="the prover's PRE-Q exemption policy for unsupported checks (passed to the prover; the request must name it): 'gamble' exempts nothing, "
                         "'exempt-unsupported' exempts every selected RU with >= 1 unsupported VU. F is fixed at step 3 (READY_PHASE2) before r_s; stage2 refuses a later F")
    ap.add_argument("--instance-format", default="contract", choices=["contract", "record_v5"],
                    help="[integrator 23:18Z item 3] the instance/population loader pair, selected by name (recorded in the chronology; validated against the "
                         "instance's partition.partition_digest_kind). 'record_v5' imports A9's record_v5.instance_v5 / population_v5 explicitly -- no process-wide monkeypatch")
    ap.add_argument("--allow-legacy-seed-file", action="store_true", help="write seed.hex as the legacy driver did (labelled record_dir_file; NOT for a run whose bound is claimed)")
    a = ap.parse_args(argv)
    rec = Path(a.record)
    _EXIT_REC["rec"] = rec  # [W-A9-4] every later exit writes <record>/verifier_exit.json
    prod = rec / "adapter" / "production"
    idir = Path(a.instance_dir)
    inst_path = idir / "protocol_instance.json"
    # [E9, D1] the program kind decides which steps apply; the decision is pinned (instance) or explicit (flag), and recorded
    try:
        pkind = a.program_kind if a.program_kind != "auto" else steps.program_kind(json.loads(inst_path.read_text()))
    except (OSError, ValueError) as e:
        raise exit_with("USAGE", f"{inst_path}: unreadable ({type(e).__name__}: {str(e)[:120]})")
    if pkind == "serve":
        if not (prod / "production.jsonl").exists():
            raise exit_with("USAGE", f"{prod}: no production.jsonl (serve first)")
        if not a.py_vllm or not a.semantic_profile:
            raise exit_with("USAGE", "a Serve_v1 program needs --py-vllm and --semantic-profile (the selective prover's python and profile)")
    elif not a.prover_cmd:
        raise exit_with("USAGE", "a registered program needs --prover-cmd (the prover's argv template; see --help)")
    chron, sel, hs, logs, vdir = rec / "chronology", rec / "adapter" / "selective", rec / "handshake", rec / "logs", rec / "verify"
    for d in (chron, sel, hs):
        if d.exists():
            raise exit_with("USAGE", f"{d} exists: a protocol run starts from nothing (an aborted audit's chronology is evidence, not scratch)")
    logs.mkdir(parents=True, exist_ok=True)
    vdir.mkdir(parents=True, exist_ok=True)
    hs.mkdir(parents=True)
    T: dict[str, float] = {}
    _EXIT_REC["timing"], _EXIT_REC["batch"] = T, f"{a.instances} requests"  # [W-A9-8] timers reach verifier_exit.json / costlog on any exit
    events: list[dict] = []
    tr = steps.Transcript()  # [E9] the explicit state machine; r_q / r_s are released only through it
    _EXIT_REC["tr"] = tr
    (vdir / "seed_custody.json").write_text(json.dumps(steps.seed_custody_statement(), indent=1) + "\n")

    def ev(name: str, **kw) -> None:
        events.append({"utc": utc(), "event": name, **kw})
        print(f"[verifier_v3.driver] {utc()} {name} " + " ".join(f"{k}={str(v)[:80]}" for k, v in kw.items()), flush=True)

    def write_transcript() -> None:
        try:
            (vdir / "transcript.json").write_text(json.dumps({**tr.to_json(), "utc": utc()}, indent=1, default=str) + "\n")
        except OSError:  # pragma: no cover
            pass

    # pin the instance and its evidence in the record BEFORE anything is committed or drawn
    pinned = {}
    for name in ("protocol_instance.json", "compact_validation.json", "maps.json", "index.json", "weights_authorized.json", "expected_work.json"):
        if (idir / name).exists():
            shutil.copyfile(idir / name, rec / name)
            pinned[name] = sha256_file(rec / name)
    pop_sha = sha256_file(Path(a.population))
    # [item 3] loaders by name; [A8] an InstanceError is a machine-readable refusal (INSTANCE_REFUSED, 21), not an uncaught traceback
    from e2e.verifier_v3.reverify import install_instance_format

    load_inst, _load_pop = install_instance_format(ps, a.instance_format)
    fmt_problems = ps.check_instance_format(a.instance_format, json.loads(inst_path.read_text()))
    if fmt_problems:
        raise exit_with("INSTANCE_REFUSED", "; ".join(fmt_problems), step="instance", problems=fmt_problems)
    try:
        inst = load_inst(inst_path)
    except ps.InstanceError as e:
        raise exit_with("INSTANCE_REFUSED", f"protocol_instance.json refused by validate_instance ({a.instance_format}): {e}", step="instance", problems=[str(e)[:400]]) from None
    if not inst.is_contract_instance:
        raise exit_with("USAGE", "the instance declares no epoch / B_ex not an RU count: this driver runs contract instances only (legacy: scripts/rec_p1_protocol.py, labelled)")
    idig = ps.instance_digest(inst)
    ev("instance_pinned", instance_digest=idig, files=list(pinned), population_file_sha256=pop_sha, epoch=inst.epoch.epoch_id, B_ex=inst.epoch.B_ex)
    tr.advance("instance_pinned", instance_digest=idig)
    inst_doc = json.loads(inst_path.read_text())
    if pkind == "serve" or (idir / "weights_authorized.json").exists():
        registered = json.loads((idir / "weights_authorized.json").read_text())
    else:  # a registered program without prescribed weight inputs: no registry to compare against (named NOT_APPLICABLE at step 2)
        registered = {"expected_root": None}
    ev("program_kind", kind=pkind, source=("--program-kind" if a.program_kind != "auto" else "protocol_instance.json#program.kind"),
       weights_registry=(idir / "weights_authorized.json").exists())
    # the epoch IS the RU map: one RU per request (RU r = position of the request id in sorted(epoch.request_ids) == the prover's canonical order),
    # or -- a registered program declaring epoch.ru_indices beyond its requests -- one RU per stage call (<request>-ru<r>; steps.ru_ids_of).
    # Every RU-keyed structure below (boundary trees, ru/<id> trees, J / F / retained, coverage) uses the RU id; on a serve program it IS the request id.
    try:
        request_ids, ru_request_of, ru_rule = steps.ru_ids_of(inst, pkind)
    except ValueError as e:
        raise exit_with("INSTANCE_REFUSED", f"RU map: {e}", step="instance", problems=[str(e)])
    ru_of_request = {rid: r for r, rid in enumerate(request_ids)}
    ru_map = {"schema": "veritor/rec/ru-map/v1", "rule": ru_rule, "program_kind": pkind,
              "rus": [{"ru": r, "ru_id": rid, "request_id": ru_request_of[rid]} for r, rid in enumerate(request_ids)], "epoch_id": inst.epoch.epoch_id}
    (rec / "ru_map.json").write_text(json.dumps(ru_map, indent=1) + "\n")
    ev("ru_map", rus=len(request_ids), rule=ru_rule)
    unsupported_kinds = set(json.loads(Path(a.unsupported_kinds).read_text())) if a.unsupported_kinds else None
    if unsupported_kinds is None and a.expressibility:
        unsupported_kinds = {row["kind"] for row in json.loads(Path(a.expressibility).read_text())["rows"] if str(row["class"]).startswith("exempt")}

    # --- the prover (phase 0 runs immediately) ---
    if pkind == "serve":
        prover_cmd = [a.py_vllm, "-m", "verity_vllm_adapter.selective_prover", "--case", a.case, "--checkpoints", a.checkpoints, "--production", str(prod), "--out", str(sel),
                      "--handshake", str(hs), "--adapter-revision", "contract/a3-sequence", "--semantic-profile", a.semantic_profile, "--arena-gb", str(a.arena_gb),
                      "--weights-format", "2", "--rows-per-leaf", "1"]
        if a.expressibility:
            prover_cmd += ["--expressibility", a.expressibility]
        if a.unsupported_kinds:
            prover_cmd += ["--unsupported-kinds", a.unsupported_kinds]
        prover_cmd += ["--exemption-policy", a.exemption_policy]
        env = {**os.environ, "PYTHONPATH": str(POC), "VLLM_BATCH_INVARIANT": "1"}
    else:
        import shlex
        prover_cmd = shlex.split(a.prover_cmd.format(out=str(sel), handshake=str(hs), exemption_policy=a.exemption_policy, record=str(rec), instance_dir=str(idir),
                                                     population=str(a.population)))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(POC), str(POC.parent)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
    t0 = time.perf_counter()
    _EXIT_REC["t0"] = t0
    plog = (logs / "selective-prover.log").open("w")
    prover = subprocess.Popen(prover_cmd, stdout=plog, stderr=subprocess.STDOUT, env=env, cwd=str(POC))
    ev("prover_started", cmd=" ".join(prover_cmd))
    step2: dict = {}
    step4: dict = {}
    try:
        # ================= step 1 -> 2: the prover's first commitment; the verifier CHECKS before drawing J =================
        wait_for(hs / READY0, hs / ABORT, prover, a.timeout, READY0, hs=hs, tr=tr, events=events, ev=ev)
        T["phase0_wall_s"] = round(time.perf_counter() - t0, 1)
        tr.advance("ready_phase0")
        ready0 = json.loads((hs / READY0).read_text())
        phase0 = json.loads((sel / "phase0.json").read_text())
        c0 = json.loads((sel / "commitments_phase0.json").read_text())
        boundary_digest = sha256_file(sel / "commitments_phase0.json")
        if not (boundary_digest == ready0["commitments_phase0_sha256"] == phase0["commitments_phase0_sha256"]):
            refuse(hs, events, "step2", ["phase-0 commitment file digest != the prover's READY_PHASE0 digest"])
        t_s2 = time.perf_counter()
        if pkind == "serve":
            prod_rows = [json.loads(l) for l in (prod / "production.jsonl").read_text().splitlines() if l.strip()]
            b_out = boundary_outputs(sel)
            b_in = boundary_inputs(sel, c0)  # WR-5: committed prompt / seed / sampling params vs the instance's declared x, authenticated by the committed root
            problems = steps.step2_check(inst, c0, prod_rows, registered_weights_root=registered["expected_root"], boundary_outputs=b_out, boundary_inputs=b_in)
            step2 = {"problems": problems, "served_equals_committed_checked": b_out is not None, "weights_root": c0["trees"]["weights"]["root"],
                     "registered_root": registered["expected_root"], "seconds": round(time.perf_counter() - t_s2, 3), "label": "MEASURED", "program_kind": pkind}
            if b_out is None:
                problems = problems + ["served == committed boundary comparison NOT DONE (capture reader unavailable): step 2 cannot pass"]
            boundary_rids = sorted(r["request_id"] for r in prod_rows)
        else:
            # [E9, D1] a registered program: no served rows, no capture reader; the program-agnostic step-2 checks run, the vLLM-only ones are named
            prod_rows = []
            input_check, ic_note = steps.registered_input_check(inst_doc, rec)  # [D20] the builder module's v_in(C) = x predicate, by name (E8: make_input_check)
            problems, na = steps.step2_check_registered(inst, c0, registered_weights_root=registered.get("expected_root"), ru_ids=request_ids, input_check=input_check)
            ev("STEP2_INPUT_CHECK", source=ic_note, supplied=input_check is not None)
            for x in na:
                ev(x["event"], check=x["check"], reason=x["reason"], replaced_by=x.get("replaced_by"))
            step2 = {"problems": problems, "served_equals_committed_checked": False, "weights_root": ((c0.get("trees") or {}).get("weights") or {}).get("root"),
                     "registered_root": registered.get("expected_root"), "seconds": round(time.perf_counter() - t_s2, 3), "label": "MEASURED", "program_kind": pkind,
                     "not_applicable": na}
            boundary_rids = sorted(set(ru_request_of.values()))  # the REQUESTS the boundary covers (stage1's R16 barrier input); every RU's tree was checked above
        # [user directive 16:04 PT / integrator 23:18Z item 1] V2.population_derived at the step-2 barrier: the population fixes the stage-2 candidate
        # set, so the verifier derives the expected population from the pinned instance BEFORE r_q; a disagreement refuses and names the field
        pd = steps.population_derived(json.loads(inst_path.read_text()), Path(a.population), instances=a.instances,
                                      rows_self_checked_by="partition_sample.stage1: load_population(check='full') re-derives the partition digest from the rows before r_q")
        problems = problems + population_barrier(pd, hs=hs, events=events, vdir=vdir, step2=step2, ev=ev)
        # [E9, D24] V2.partition_digest BY NAME (both FULL digests) before r_q: the population file must carry the instance's pins exactly
        try:
            pins_problems = steps.population_pins_check(json.loads(inst_path.read_text()), json.loads(Path(a.population).read_text()))
        except (OSError, ValueError) as e:
            pins_problems = [f"V2.partition_digest: population file unreadable: {type(e).__name__}: {str(e)[:200]}"]
        ev("V2.partition_digest", ok=not pins_problems, problems=pins_problems)
        step2["population_pins"] = {"ok": not pins_problems, "problems": pins_problems}
        problems = problems + pins_problems
        ev("step2_check", ok=not problems, problems=problems, requests=len(boundary_rids), boundary_digest=boundary_digest, program_kind=pkind)
        if problems:
            refuse(hs, events, "step2", problems)
        tr.advance("step2_complete", problems=0)
        answer_challenge_requests(hs, tr, events, ev)  # a REQUEST_r_q written during the step-2 checks is now releasable: left for the normal path
        # --- r_q: the seed stays in this process; the boundary covers exactly the epoch's requests ---
        try:
            tr.release("r_q")  # [E9] refused by name if step 2 is not the current state (a driver defect; the prover's requests are answered above)
        except steps.PrematureChallenge as e:
            raise exit_with("PREMATURE_CHALLENGE", str(e), step="step2", problems=[str(e)])
        t1 = time.perf_counter()
        try:
            doc = ps.stage1(inst_path, Path(a.population), boundary_digest, chron, instances=a.instances, condition_nonempty=a.condition_nonempty, instance_format=a.instance_format,
                            legacy_seed_file=a.allow_legacy_seed_file, boundary_request_ids=boundary_rids)
        except (ps.ChronologyError, ps.InstanceError) as e:
            # [E9, D24] the driver boundary: partition_sample's refusal (population <-> instance binding, seed, R16 barrier) is a NAMED step-2
            # refusal with verifier_exit.json, never an uncaught traceback (E8's CORE-live-neg-mispin-01: exit 1, no verifier_exit.json)
            refuse(hs, events, "step2", [f"{steps.chronology_refusal_name(e)} ({type(e).__name__}): {str(e)[:600]}"])
        T["stage1_s"] = round(time.perf_counter() - t1, 2)
        rq = next(e for e in doc["events"] if e["name"] == "r_q")
        J = list(rq["selection"])
        selected_rids = [request_ids[r] for r in J]
        ev("r_q", J=J, selected=selected_rids, redraws=rq.get("redraws", 0), digest=rq["digest"], seed_custody=doc["seed_custody"])
        challenge = {"selected": selected_rids, "T_0": next(e["digest"] for e in doc["events"] if e["name"] == "seed_commitment"), "r_q": rq["digest"],
                     "commitments_phase0_sha256": boundary_digest, "utc": utc(), "sampler": ps.SAMPLER, "instance_digest": idig, "J": J,
                     "B_ex": inst.exemption_budget, "B_ex_unit": "replay_units", "epoch_id": inst.epoch.epoch_id}
        tmp = hs / (CHALLENGE_FILE + ".tmp")
        tmp.write_text(json.dumps(challenge, indent=1) + "\n")
        os.replace(tmp, hs / CHALLENGE_FILE)
        # ================= step 3 -> 4: the prover's response (interiors + F); the verifier CHECKS before drawing Q =================
        t2 = time.perf_counter()
        wait_for(hs / READY2, hs / ABORT, prover, a.timeout, READY2, hs=hs, tr=tr, events=events, ev=ev)
        prover.wait(timeout=600)
        T["phase2_wall_s"] = round(time.perf_counter() - t2, 1)
        tr.advance("ready_phase2")
        interior_digest = sha256_file(sel / "commitments.json")
        ready2 = json.loads((hs / READY2).read_text())
        if interior_digest != ready2["commitments_sha256"]:
            refuse(hs, events, "step4", ["interior commitment file digest != the prover's READY_PHASE2 digest"])
        exreq = json.loads((sel / "exemption_request.json").read_text())
        if sha256_file(sel / "exemption_request.json") != ready2.get("exemption_request_sha256"):
            refuse(hs, events, "step4", ["exemption_request.json digest != the prover's READY_PHASE2 digest"])
        c2 = json.loads((sel / "commitments.json").read_text())
        pj = json.loads((sel / "selective_prover.json").read_text())
        pj2 = pj.get("phase2") or {}
        ev("phase2", interior_digest=interior_digest, k=pj.get("k"), ru_leaves=pj2.get("ru_leaves"), replay_units=pj2.get("replay_units_committed"), F=exreq.get("F"))
        t_s4 = time.perf_counter()
        problems, F, committed = steps.step4_check(inst, J=J, r_q=rq["digest"], instance_digest=idig, c0=c0, c2=c2, exreq=exreq, ru_of_request=ru_of_request, policy=a.exemption_policy)
        step4 = {"problems": problems, "F": F, "committed_rus": committed, "B_ex": inst.exemption_budget, "seconds": round(time.perf_counter() - t_s4, 3), "label": "MEASURED",
                 "exemption_request_sha256": ready2.get("exemption_request_sha256"),
                 "exemption_policy": {**(exreq.get("exemption_policy") or {}), "configured": a.exemption_policy,
                                      "ordering": "F read from READY_PHASE2 / exemption_request.json and checked here BEFORE ps.stage2 derives r_s; stage2 refuses an F after r_s"}}
        ev("step4_check", ok=not problems, problems=problems, F=F, committed=committed)
        if problems:
            refuse(hs, events, "step4", problems)
        # [E9, NIGHT D35 -- T1's F1] V3.trees_rehashed at step 4, BEFORE r_s and BEFORE any check that opens a leaf (output copies, interior
        # regions): every tree the prover shipped so far (step-1 boundary trees, step-3 interiors) has its root RECOMPUTED from the leaf files and compared with the committed root; a mismatch is STEP4_REFUSED by name.
        t_rh = time.perf_counter()
        rh = steps.trees_rehashed(sel, c2 or {})
        T["trees_rehashed_s"] = round(time.perf_counter() - t_rh, 3)
        ev("V3.trees_rehashed", ok=rh["ok"], summary=rh["summary"], problems=rh["problems"][:4], not_applicable=rh["not_applicable"][:4])
        (vdir / "trees_rehashed.json").write_text(json.dumps(rh, indent=1, sort_keys=True, default=str) + "\n")
        if not rh["ok"]:
            refuse(hs, events, "step4", ["V3.trees_rehashed: " + p for p in rh["problems"][:5]])
        # R4 (A2's M2) at step 4: the two committed copies of every served token of every RETAINED RU agree (boundary output_token/t vs the
        # interior sampler.select/port=0 copy), both authenticated against the roots in commitments.json -- BEFORE r_s.  A disagreement
        # is refused (the prover's remedy is to exempt the RU in F before Q; it cannot pick one of two values after Q).
        t_oc = time.perf_counter()
        retained_rids = [request_ids[r] for r in J if r not in set(F)]
        if pkind == "serve":
            oc = steps.output_copy_step4(sel, c2, retained_rids)
            step4["output_copies"] = {**{k: oc[k] for k in ("ok", "rows", "problems", "per_request", "label")}, "seconds": round(time.perf_counter() - t_oc, 3), "placement": "step 4, before r_s"}
            ev("step4_output_copies", ok=oc["ok"], rows=oc["rows"], problems=oc["problems"][:5])
            if not oc["ok"]:
                refuse(hs, events, "step4", ["output copies: " + p for p in oc["problems"][:5]])
        else:
            na4 = steps.not_applicable("step4.output_copies", "no served token copy on a registered program: A2's M2 compares Serve_v1's boundary output_token/t with the "
                                       "sampler.select/port=0 interior copy; a registered program's output words are the boundary leaves themselves (one copy)")
            step4["output_copies"] = {"ok": None, "label": "NOT_APPLICABLE", "reason": na4["reason"], "placement": "step 4, before r_s"}
            ev(na4["event"], check=na4["check"], reason=na4["reason"])
        # [E9, brief interface 4] V3.interior_layout at step 4, BEFORE r_s: with interior values in play (E6's kind table declares interior-producer
        # kinds / the instance pins an interior layout / commitments.json carries interior trees) the pin must be present and equal E2's
        # layout_digest(), and every REQUIRED interior region of every retained RU must be supplied (E2's validate_supplied).  Fail closed.
        t_il = time.perf_counter()
        try:
            pop_doc_il = json.loads(Path(a.population).read_text())
        except (OSError, ValueError):
            pop_doc_il = None
        pop_kinds = steps.interior_kinds_in_population(pop_doc_il)
        il_pin = steps.interior_layout_pin(inst_doc)
        il_trees = steps.interior_trees_committed(c2)  # trees['interior/<req>/<step>'] and/or E2's interior_roots['<req>/<step>']
        in_play = bool(pop_kinds) or il_pin is not None or bool(il_trees)
        reason = (f"population declares {len(pop_kinds)} interior-producer kind(s)" if pop_kinds else "instance pins commitments.interior_layout" if il_pin is not None
                  else f"commitments.json carries {len(il_trees)} interior tree(s)" if il_trees else "population kind table declares no interior-producer kind (legacy table: producer field absent), no pin, no interior tree")
        # the REQUIRED regions (E3's derivation, E2's framing) are built only when there is something to check them against: a pin + retained RUs
        req_src, req_label, req_facts = (None, "not built: no interior_layout pin / no retained RU", {})
        if il_pin is not None and retained_rids:
            req_src, req_label, req_facts = steps.required_source_live(inst_doc, pop_doc_il, sel, c2, retained_rids)
        il = steps.interior_layout_check(inst_doc, sel=sel, c2=c2, retained_request_ids=retained_rids, interior_in_play=in_play, in_play_reason=reason,
                                         required_source=req_src, required_label=req_label)
        step4["interior_layout"] = {**{k: il[k] for k in ("ok", "status", "problems", "digest", "layout_digest", "trees", "interior_in_play", "reason", "validator", "required_source")},
                                    "required_facts": req_facts, "rus": {ru: f for ru, f in ((il.get("regions") or {}).get("rus") or {}).items()} if isinstance(il.get("regions"), dict) else None,
                                    "seconds": round(time.perf_counter() - t_il, 3), "placement": "step 4, before r_s", "retained_requests": [r[:8] for r in retained_rids]}
        ev("V3.interior_layout", ok=il["ok"], status=il["status"], in_play=in_play, trees=len(il["trees"]), required_source=il.get("required_source"), problems=il["problems"][:3])
        if not il["ok"]:
            refuse(hs, events, "step4", ["V3.interior_layout: " + p for p in il["problems"][:5]])
        # [E9, NIGHT G1 `prescribed-literal/v1`] V3.literal_table at step 4, BEFORE r_s: when the instance pins commitments.literals it must equal the
        # table the verifier derives from the PINNED PROGRAM alone (rule, sorted Const ids, per-width tree root / leaf_count / ids_digest); when it
        # does not and the population's program has literals at operand positions, Q may select a literal-reading VU -> that VU refuses
        # LITERAL_UNPINNED at step 5 (the pin is instance data, so the step-4 verdict here is the pin-vs-derivation equality; fail closed on mismatch)
        t_lt = time.perf_counter()
        lt_prog = req_facts.pop("_program", None) if isinstance(req_facts, dict) else None  # the step-4 build of required_source_live, when it ran
        if lt_prog is None and steps.literal_pin(inst_doc) is not None:
            try:
                from e2e.verifier_v3 import locator as _vloc

                lt_prog, _ltC, _lt_probs = _vloc.build_pinned_program(inst_doc["program"]["identity"], expected_digest=(pop_doc_il or {}).get("program_digest"),
                                                                     expected_gates=(inst_doc["program"].get("gates") or {}).get("value"), builder=inst_doc["program"].get("builder"))
            except Exception as e:  # noqa: BLE001
                lt_prog = None
                ev("V3.literal_table.program", error=f"{type(e).__name__}: {str(e)[:200]}")
        lt = steps.literal_table_check(inst_doc, program=lt_prog, in_play_reason="step 4: pin vs the verifier's derivation from the pinned program")
        step4["literal_table"] = {**{k: lt[k] for k in ("ok", "status", "problems", "literals", "trees", "rule")}, "seconds": round(time.perf_counter() - t_lt, 3),
                                  "placement": "step 4, before r_s"}
        ev("V3.literal_table", ok=lt["ok"], status=lt["status"], literals=lt["literals"], trees=lt["trees"], problems=lt["problems"][:3])
        if not lt["ok"]:
            refuse(hs, events, "step4", ["V3.literal_table: " + p for p in lt["problems"][:5]])
        tr.advance("step4_complete", F=F, retained=[r for r in J if r not in set(F)])
        answer_challenge_requests(hs, tr, events, ev)
        # --- r_s over J minus F ---
        try:
            tr.release("r_s")  # [E9] refused by name unless step 4 (F, interiors, regions, output copies) is the current state
        except steps.PrematureChallenge as e:
            raise exit_with("PREMATURE_CHALLENGE", str(e), step="step4", problems=[str(e)])
        t3 = time.perf_counter()
        try:
            doc = ps.stage2(inst_path, Path(a.population), interior_digest, chron, instances=a.instances, exemptions=F, committed_rus=committed, instance_format=a.instance_format)
        except (ps.ChronologyError, ps.InstanceError) as e:  # [E9, D24] named step-4 refusal at the driver boundary (never an uncaught traceback)
            refuse(hs, events, "step4", [f"{steps.chronology_refusal_name(e)} ({type(e).__name__}): {str(e)[:600]}"])
        T["stage2_s"] = round(time.perf_counter() - t3, 2)
        ps.reveal(chron)
        tr.advance("seed_revealed")
        problems = ps.verify(inst_path, Path(a.population), chron, instances=a.instances, instance_format=a.instance_format)
        rs = next(e for e in doc["events"] if e["name"] == "r_s")
        selected = json.loads((chron / "selected.json").read_text())
        ev("r_s", Q=len(rs["selection"]), candidates=rs.get("candidates"), kinds=len(selected["kinds"]), chronology_rederived=(not problems), problems=problems)
        # ================= after Q: the per-RU record (R17 + R18) =================
        rf = steps.ru_fields(inst, exreq=exreq, selected_rows=selected["selected"], unsupported_kinds=unsupported_kinds, s=inst.sampler.law.s)
        (vdir / "ru_fields.json").write_text(json.dumps(rf, indent=1, sort_keys=True) + "\n")
        ev("ru_fields", F=rf["F"], sampled_unsupported=rf["sampled_unsupported_count"], aggregate_completion_risk=rf["completion_risk"]["aggregate_p_select_at_least_one"])
    finally:
        if prover.poll() is None:
            prover.kill()
        plog.close()
        write_transcript()
        (rec / "protocol_events.json").write_text(json.dumps(events, indent=1, default=str) + "\n")
    T["total_s"] = round(time.perf_counter() - t0, 1)
    out = {"schema": "veritor/rec/p1-protocol/v2-contract", "utc": utc(), "instance_digest": idig, "pinned_sha256": pinned, "population_file_sha256": pop_sha,
           "instances": a.instances, "epoch": {"epoch_id": inst.epoch.epoch_id, "request_ids": request_ids, "B_ex": inst.epoch.B_ex, "B_ex_unit": "replay_units"},
           "ru_map": ru_map, "boundary_digest": boundary_digest, "interior_digest": interior_digest, "J": J, "F": F, "retained": [r for r in J if r not in set(F)],
           "selected_requests": selected_rids, "step2": step2, "step4": step4, "seed_custody": doc["seed_custody"],
           "transcript": {"state": tr.state, "events": [e["event"] for e in tr.events], "challenge_requests_refused": tr.refusals, "file": str(vdir / "transcript.json"),
                          "seed_custody_statement": str(vdir / "seed_custody.json")},
           "selected_vus": len(rs["selection"]), "selected_kinds": selected["kinds"], "chronology_problems": problems,
           "program_kind": pkind, "prover_cmd": prover_cmd,
           "prover": {"phase0": phase0, "phase2_timing": pj.get("timing"), "phase2": {k: pj2.get(k) for k in ("replay_units_committed", "ru_leaves", "replayed_tokens", "capture", "trajectory_followed", "exemption_request")},
                      "bytes": pj.get("bytes"), "log": str(logs / "selective-prover.log")},
           "timing": T, "events": events,
           "provenance": {"label": "MEASURED", "script": "e2e/verifier_v3/driver.py", "sampler": ps.SAMPLER,
                          "note": "docs/PROTOCOL.md steps 1-4 in order: step-2 checks before r_q, step-4 checks (F ⊆ J, |F| <= B_ex, retained interiors committed) before r_s; "
                                  "Q over J minus F; the seed never touches the record directory; weights and request inputs are referenced, never recommitted"}}
    (rec / "protocol.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    with (rec / "costlog.jsonl").open("a") as f:
        f.write(json.dumps({"stage": "commit_inputs_outputs_seed_weights (phase 0)", "class": "serving", "seconds": T["phase0_wall_s"], "batch": f"{a.instances} requests", "label": "MEASURED",
                            "note": "includes engine build + warm-up (selective_prover timing has the split)"}) + "\n")
        f.write(json.dumps({"stage": "replay_and_commit_interiors (phase 2)", "class": "replay", "seconds": T["phase2_wall_s"], "batch": f"{len(J)} selected request(s) in one batch", "label": "MEASURED"}) + "\n")
        f.write(json.dumps({"stage": "ru_challenge + vu_challenge (partition_sample stage1+stage2+verify) + step2/step4 checks", "class": "verifier",
                            "seconds": round(T["stage1_s"] + T["stage2_s"] + step2.get("seconds", 0) + step4.get("seconds", 0), 2), "batch": f"{a.instances} RUs, {len(rs['selection'])} VUs", "label": "MEASURED"}) + "\n")
    print(f"[verifier_v3.driver] DONE: J={J} F={F} ({selected_rids}), |Q|={len(rs['selection'])}, chronology {'REDERIVED' if not problems else 'MISMATCH ' + str(problems)}; {T}")
    code = "OK" if not problems else "CHRONOLOGY_MISMATCH"
    (rec / "verifier_exit.json").write_text(json.dumps({"schema": "veritor/verifier-exit/v1", "utc": utc(), "reason_code": code, "exit_code": EXIT_CODES[code], "step": "done",
                                                        "problems": list(problems), "message": f"J={J} F={F} |Q|={len(rs['selection'])}", "prover_abort_text": None}, indent=1) + "\n")
    return EXIT_CODES[code]


if __name__ == "__main__":
    sys.exit(main())
