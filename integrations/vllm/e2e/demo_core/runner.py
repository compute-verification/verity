"""The CORE run: docs/PROTOCOL.md's five steps over ``DemoCore_v1`` through the REAL engine, in ``verifier_v3/driver.py``'s event order
(generic protocol demonstration; NOT vLLM inference).

What is SHIPPED code called unchanged (named per check below): ``e2e.protocol_instance.load_instance`` (instance), ``e2e.population_v2``
(population self-check), ``e2e.partition_sample.stage1/stage2/reveal/verify`` (r_q, r_s, chronology re-derivation, seed custody),
``e2e.verifier_v3.steps.step4_check`` (F ⊆ J, |F| <= B_ex, retained interiors committed, no recommit), ``steps.ru_fields`` (R17/R18),
``steps.guest_acceptance`` (approved guest + epoch pins), ``coverage_v1.ck_elem.build_many`` (statements), ``scripts/rec_p1_prove.py`` on the
GPU pod (SP1 prove + host verify + check_binding), ``e2e.verifier_v3.expected.RecordCircuit`` + ``committed_trees`` and
``coverage_v1.coverage.expected_statements/check_coverage`` with ``veritor.interface.expected.check_expected_positions`` (step 5).

What is E8's (this circuit's adapter, labelled ``E8-CHECK``): the step-2 checks over the committed boundary trees (roots recomputed from the leaf
files; v_in(C) == declared x by the instance's digest; y well-typed; every RU's output tuple committed; weights tree == registration = empty),
the population derivation (rebuild program from identity -> population/v2 -> byte-equal to the pinned file), and the single-position check
replacing the vLLM-only ``output_copy_step4`` (no gate has two committed copies).  NOT executed here: ``driver.py``'s own process (its step-2
reader is Serve_v1's ``CaptureReader``; see the report) — every check it would run is run here by name.
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
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
POC = HERE.parents[1]
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from e2e import partition_sample as ps  # noqa: E402
from e2e import protocol_instance as pi  # noqa: E402
from e2e.verifier_v3 import steps  # noqa: E402
from verity_vllm import hashing  # noqa: E402
from verity_vllm.commit import MerkleTree  # noqa: E402

from . import circuit as C, instance as I, obligations as OB, producer as PR, prove_io as PIO, record as RC  # noqa: E402

READY0, READY2, CHALLENGE, ABORT = "READY_PHASE0", "READY_PHASE2", "replay_challenge.json", "ABORT"
LABEL = "generic protocol demonstration (DemoCore_v1); NOT vLLM inference"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.rows, self.failed = path, [], []
        path.parent.mkdir(parents=True, exist_ok=True)

    def check(self, code: str, ok: bool, what: str, *, who: str, **kw: Any) -> bool:
        row = {"utc": utc(), "check": code, "ok": bool(ok), "what": what, "code_owner": who, **kw}
        self.rows.append(row)
        if not ok:
            self.failed.append(code)
        print(f"[demo_core.runner] {code} {'PASS' if ok else 'FAIL'} [{who}] {what} " + " ".join(f"{k}={str(v)[:90]}" for k, v in kw.items()), flush=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return bool(ok)

    def note(self, text: str) -> None:
        row = {"utc": utc(), "note": text}
        self.rows.append(row)
        print(f"[demo_core.runner] {text}", flush=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def wait_for(path: Path, abort: Path, proc: subprocess.Popen, timeout: float, what: str) -> None:
    t0 = time.time()
    while not path.exists():
        if abort.exists():
            raise SystemExit(f"PROVER_ABORT while waiting for {what}: {abort.read_text()[:300]}")
        if proc.poll() is not None:
            raise SystemExit(f"PROVER_EXIT {proc.returncode} before {what}")
        if time.time() - t0 > timeout:
            raise SystemExit(f"TIMEOUT waiting for {what}")
        time.sleep(0.2)


# -- step-2 checks (E8-CHECK: this circuit's boundary) ---------------------------------------------------------------------------------------


def recompute_boundary_trees(sel: Path, c0: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Rebuild every boundary tree from the record's leaf files (index.json ids + leaf-i.bin) with the engine's MerkleTree; compare with c0."""
    out, problems = {}, []
    for tid, rec in c0["trees"].items():
        if not tid.startswith("boundary/"):
            continue
        d = sel / tid
        idx = json.loads((d / "index.json").read_text())
        leaves = {}
        words = {}
        for e in idx["entries"]:
            data = (d / f"leaf-{e['index']}.bin").read_bytes()
            leaves[e["canonical_id"]] = hashing.leaf_hash(e["canonical_id"], e["dtype"], tuple(e["shape"]), data)
            words[e["canonical_id"]] = {"words": [int(w) for w in np.frombuffer(data, dtype="<u2")], "gates": e["gates"], "origin": e["origin"], "dtype": e["dtype"], "shape": e["shape"]}
        t = MerkleTree(leaves, tid)
        ok = t.root.hex() == rec["root"] and t.leaf_count == int(rec["leaf_count"]) and t.ids_digest.hex() == rec["ids_digest"]
        if not ok:
            problems.append(f"{tid}: recomputed root/leaf_count/ids_digest != committed")
        out[tid] = {"root_ok": ok, "leaves": words, "leaf_count": t.leaf_count}
    return out, problems


def step2_checks(log: Log, inst, inst_doc: dict[str, Any], part: C.Partition, sel: Path, c0: dict[str, Any], registered: dict[str, Any], pop_path: Path) -> tuple[list[str], list[int]]:
    problems: list[str] = []
    trees, tp = recompute_boundary_trees(sel, c0)
    log.check("S2.boundary_trees", not tp, "every boundary tree recomputed from the record's leaf files (verity-vllm/leaf/v1, MerkleTree) == committed root, leaf_count, ids_digest", who="E8-CHECK", trees=len(trees), problems=tp)
    problems += tp
    # layout: one boundary tree per RU, holding the RU's output tuple (N words, gates == returns): "outgoing boundary runs of every CU"
    lay: list[str] = []
    for ru in part.rus:
        tid = f"boundary/{RC.request_id(ru.ru_index)}"
        t = trees.get(tid)
        outs = [lf for lf in (t or {}).get("leaves", {}).values() if lf["origin"] == "boundary" and lf["gates"] == [ru.returns[0], ru.returns[-1] + 1]]
        if t is None or len(outs) != 1 or len(outs[0]["words"]) != part.N:
            lay.append(f"RU {ru.ru_index}: output tuple not committed as one N-word leaf in {tid}")
    n_btrees = len({f"boundary/{RC.request_id(ru.ru_index)}" for ru in part.rus})
    log.check("S2.layout", not lay, "first boundary = the outgoing boundary run (output tuple, N x 16-bit) of EVERY RU, exactly one leaf per RU in its request's boundary tree (instance.partition.first_boundary.rule)",
              who="E8-CHECK", problems=lay, rus=part.n_r, boundary_trees=n_btrees, request_mode=RC.request_mode())
    problems += lay
    # v_in(C) == declared x: the input leaves' words in gate order vs epoch.inputs digest (the shipped declared_input_digest)
    rid = inst.epoch.request_ids[0]
    ins = [lf for lf in trees.get(f"boundary/{RC.request_id(0)}", {}).get("leaves", {}).values() if lf["origin"] == "input"]
    in_words: dict[int, int] = {}
    for lf in ins:
        for g, w in zip(range(lf["gates"][0], lf["gates"][1]), lf["words"]):
            in_words[g] = w
    declared_gates = [g for name in ("a", "b", "c") for g in part.inputs[name]]
    x_committed = [in_words.get(g) for g in declared_gates]
    x_ok = all(w is not None for w in x_committed) and pi.declared_input_digest([int(w) for w in x_committed]) == inst.epoch.inputs.per_request[rid].prompt_sha256 \
        and len(declared_gates) == inst.epoch.inputs.per_request[rid].prompt_len == int(inst.program.input_gates.value)  # (identity.LP is null for a registered program: E6's rule)
    log.check("S2.inputs_declared", x_ok, "v_in(C) == x: the committed input leaves (a, b, c; 3N words in Input-gate order, authenticated by the recomputed root) digest to the instance's declared input (epoch.inputs.per_request.prompt_sha256; count == LP)",
              who="E8-CHECK", words=len(declared_gates), digest=inst.epoch.inputs.per_request[rid].prompt_sha256[:16])
    if not x_ok:
        problems.append("v_in(C) != declared x")
    # y well-typed: the last RU's output leaf is N x uint16 (16-bit words) == outputs.designated
    last = part.rus[part.n_r - 1]
    y_leaf = [lf for lf in trees.get(f"boundary/{RC.request_id(part.n_r - 1)}", {}).get("leaves", {}).values()
              if lf["origin"] == "boundary" and lf["gates"] == [last.returns[0], last.returns[-1] + 1]]
    des = inst.outputs.designated[0]
    y_ok = len(y_leaf) == 1 and y_leaf[0]["dtype"] == "bfloat16" and y_leaf[0]["shape"] == [des.multiplicity] and len(y_leaf[0]["words"]) == des.multiplicity \
        and all(0 <= w < 2 ** des.domain_bits_declared for w in y_leaf[0]["words"]) and inst.outputs.W_out_bits.value == des.multiplicity * des.domain_bits_declared
    y = y_leaf[0]["words"] if y_leaf else None
    log.check("S2.output_typed", y_ok, "v_out(C) = y is well-typed: the designated output leaf (RU R-1's output tuple) is N x 16-bit words == outputs.designated (multiplicity, domain == wire bits); y published",
              who="E8-CHECK", y_words=y, W_out_bits=inst.outputs.W_out_bits.value)
    if not y_ok:
        problems.append("y not well-typed")
    # weights: this circuit has none; the committed weights tree must equal the registration (the empty tree)
    w_ok = c0["trees"]["weights"]["root"] == registered["expected_root"] and int(c0["trees"]["weights"]["leaf_count"]) == 0 == int(registered["leaf_count"])
    log.check("S2.weights_root", w_ok, "committed weights tree == the registered weight commitment (DemoCore_v1: no weights -> the empty tree, registered as such before the run)", who="E8-CHECK", root=c0["trees"]["weights"]["root"][:16])
    if not w_ok:
        problems.append("weights root != registration")
    # population derived from the pinned instance (E8's deriver for DemoCore_v1: rebuild the program from identity, re-derive population/v2, compare bytes)
    ident = inst_doc["program"]["identity"]
    prog2, d2 = C.build(part.N, part.R)
    part2 = C.partition(part.N, part.R, prog2, d2)
    pop2 = I.population_doc(part2)
    pinned = json.loads(pop_path.read_text())
    pd_ok = d2 == inst.program.digest and pop2["content_sha256"] == pinned["content_sha256"] and pop2["partition_digest"] == inst.partition.partition_digest \
        and pop2["vu_count"] == inst.partition.vu_count.value and pop2["ru_count"] == inst.partition.ru_count.value and ident["function"].startswith(C.identity(part.N, part.R)["function"])
    log.check("S2.population_derived", pd_ok, "DERIVED (E8 deriver for DemoCore_v1; E6's population_v3 has no rule for this program): program rebuilt from program.identity -> program_digest == pin; population/v2 re-derived -> content_sha256 == the pinned population file; partition_digest == instance",
              who="E8-CHECK", program_digest=d2[:16], content_sha256=pop2["content_sha256"][:16])
    if not pd_ok:
        problems.append("population not derivable from the pinned instance")
    # the epoch barrier input: every RU of the epoch has its boundary tree -> request 'core' is covered
    covered = all(f"boundary/{RC.request_id(r)}" in trees and trees[f"boundary/{RC.request_id(r)}"]["root_ok"] for r in inst.epoch.ru_indices)
    log.check("S2.request_set", covered, "the boundary commitment covers every RU of the declared epoch (R16 barrier input for partition_sample.stage1: boundary_request_ids)", who="E8-CHECK", rus=len(inst.epoch.ru_indices))
    if not covered:
        problems.append("epoch not covered by the boundary commitment")
    return problems, (y or [])


# -- step 5: proofs + coverage --------------------------------------------------------------------------------------------------------------


def prove_on_pod(rec: Path, *, ssh_wrapper: str, remote_dir: str, host: str, remote_repo: str, log_dir: Path, python: str | None = None) -> None:
    """Ship obligations/ to the pod, run the SHIPPED scripts/rec_p1_prove.py there (SP1 cuda prove + host verify + check_binding), pull proofs/ back."""
    ssh_cmd = Path(ssh_wrapper).read_text().splitlines()[-1] if ssh_wrapper.endswith(".ssh") else None
    wrapper = ssh_wrapper if ssh_wrapper.endswith(".sh") else None
    if wrapper is None:
        raise SystemExit("--pod-ssh must be the .ssh.sh wrapper written by E10")
    # derive the rsync -e command from the wrapper's ssh line
    line = [ln for ln in Path(wrapper).read_text().splitlines() if ln.strip().startswith("exec ssh")][0]
    parts = line.split()
    ssh_opts = " ".join(parts[1:-2])  # drop 'exec' ... 'root@host' '"$@"'
    target = parts[-2]
    rs = ["rsync", "-az", "-e", ssh_opts]
    subprocess.run([wrapper, f"mkdir -p {remote_dir}"], check=True)
    subprocess.run(rs + [str(rec / "obligations") + "/", f"{target}:{remote_dir}/obligations/"], check=True)
    subprocess.run(rs + [str(HERE) + "/", f"{target}:{remote_repo}/e2e/demo_core/"], check=True)
    cmd = (f"cd {remote_repo} && PYTHONPATH={remote_repo}:{remote_repo}/vllm-poc:{remote_repo}/src SP1_PROVER=cuda {python or remote_repo + '/.venv/bin/python'} scripts/rec_p1_prove.py "
           f"--record {remote_dir} --host {host} --prover cuda")
    with (log_dir / "rec_p1_prove.pod.log").open("w") as f:
        r = subprocess.run([wrapper, cmd], stdout=f, stderr=subprocess.STDOUT)
    subprocess.run(rs + [f"{target}:{remote_dir}/proofs/", str(rec / "proofs") + "/"], check=True)
    subprocess.run(rs + [f"{target}:{remote_dir}/costlog.jsonl", str(rec / "costlog.pod.jsonl")], check=False)
    if r.returncode != 0:
        print(f"[demo_core.runner] rec_p1_prove.py exited {r.returncode} (see logs/rec_p1_prove.pod.log)", flush=True)


def host_verify_on_pod(rec: Path, *, ssh_wrapper: str, remote_dir: str, host: str) -> list[dict[str, Any]]:
    """The VERIFIER's own SP1 verify of every proof (the same approved host binary; independent of the prover script's verify)."""
    rows = []
    for pj in sorted((rec / "proofs").glob("chunk*.json")):
        name = pj.stem
        cmd = f"SP1_PROVER=cpu {host} verify --proof {remote_dir}/proofs/{name}-compressed.bin --statement {remote_dir}/obligations/{name}/statement.bin"
        out = subprocess.run([ssh_wrapper, cmd], capture_output=True, text=True)
        line = [ln for ln in out.stdout.strip().splitlines() if ln.strip().startswith("{")]
        doc = json.loads(line[-1]) if line else {"ok": False, "error": out.stdout[-300:] + out.stderr[-300:]}
        rows.append({"chunk": name, **doc})
    return rows


def step5(log: Log, rec: Path, part: C.Partition, core: RC.CoreRecord, inst, selected: dict[str, Any], *, pod: dict[str, str] | None, approved_path: Path) -> dict[str, Any]:
    from e2e.coverage_v1.coverage import AcceptedProof, check_coverage, expected_statements, obligations_of_typed_object
    from e2e.verifier_v3.expected import RecordCircuit, committed_trees
    from veritor.interface import TypedObligationSet, check_binding

    sel = rec / "adapter" / "selective"
    c2 = json.loads((sel / "commitments.json").read_text())
    Q = [int(r["vu_index"]) for r in selected["selected"]]
    # ---- prover side: statements for exactly the selected VUs through the shipped builder ----
    vus = [part.vus[i] for i in Q]
    chunks = OB.chunk_units(vus, core, max_units=64)
    builts = [OB.build_chunk(c, core, name=f"{rec.name}-chunk{k:02d}") for k, c in enumerate(chunks)]
    ob = PIO.write_obligations(rec, builts, core, selected_vus=Q, instance_digest=selected["instance_digest"], note=f"selected.json ({len(Q)} VUs) -> ck_elem.build_many; one statement per <= 64 units")
    log.check("S5.statements_built", ob["covered_vus"] == len(Q) and not ob["uncovered_selected"], "every selected VU is in exactly one typed statement built by coverage_v1.ck_elem.build_many (family veritor.elem-bf16@1)",
              who="SHIPPED coverage_v1.ck_elem", selected=len(Q), statements=len(builts), transparent=[c["transparent_backend"] for c in ob["chunks"]])
    if not Q:
        log.check("S5.q_empty", True, "Q = ∅: nothing to prove; the run's verdict is the step-2/4 checks + chronology (a valid draw, not the milestone)", who="E8-CHECK")
        line = "sampled VUs = 0; covered by a verified expected statement = 0 (identities: none); proof objects = 0; uncovered = 0 (identities: none)"
        log.note(line)
        return {"Q": 0, "proof_objects": 0, "covered": 0, "covered_ids": [], "uncovered_ids": [], "coverage_line": line}
    # ---- proofs on the pod (shipped rec_p1_prove.py) ----
    if pod:
        prove_on_pod(rec, ssh_wrapper=pod["ssh"], remote_dir=pod["dir"], host=pod["host"], remote_repo=pod["repo"], log_dir=rec / "logs", python=pod.get("python"))
    proofs_path = rec / "proofs" / "proofs.json"
    proofs = json.loads(proofs_path.read_text()) if proofs_path.exists() else {}
    log.check("V4.prover_summary", bool(proofs), "proofs.json written by the prover (all chunks attempted)", who="SHIPPED rec_p1_prove.py", present=bool(proofs))
    approved = json.loads(approved_path.read_text())
    info = {"elf_sha256": (proofs.get("guest") or {}).get("elf_sha256"), "vk_hash": (proofs.get("guest") or {}).get("vk_hash")}
    v4g = steps.guest_acceptance(approved, info, pinned_elfs=inst.epoch.approved_elf_sha256, pinned_family=inst.epoch.family) if proofs else ["no proofs"]
    log.check("V4.guest", not v4g, "the host's guest (elf_sha256, vk_hash) is in approved_guests.json AND in the instance's epoch.approved_elf_sha256 pin; the pinned family is one the guest supports", who="SHIPPED steps.guest_acceptance",
              elf=str(info["elf_sha256"])[:12], problems=v4g)
    # ---- the verifier's own host verify (same approved binary, on the pod) + local check_binding of the public values ----
    hv = host_verify_on_pod(rec, ssh_wrapper=pod["ssh"], remote_dir=pod["dir"], host=pod["host"]) if pod else []
    (rec / "verify" / "host_verify.jsonl").write_text("".join(json.dumps(r) + "\n" for r in hv))
    accepted, per_chunk = [], []
    for k, b in enumerate(builts):
        name = f"chunk{k:02d}"
        pr = next((r for r in proofs.get("results", []) if r["chunk"] == name), None)
        v = next((r for r in hv if r["chunk"] == name), None)
        pv_hex = (v or {}).get("public_values") or ((pr or {}).get("verify") or {}).get("public_values")
        ok_host = bool(v and v.get("ok") and v.get("statement_match") and v.get("verdict") and not v.get("unsound") and str(v.get("vk_hash")).lower() == str(info["vk_hash"]).lower())
        bnd = check_binding(TypedObligationSet.from_json(b.record["object"]), bytes.fromhex(pv_hex), statement_format=3) if pv_hex else None
        st_ok = pr is not None and pr.get("statement_sha256") == sha256_file(rec / "obligations" / name / "statement.bin") and pr.get("h_O") == b.record["digest"]
        # [E9, D25] the record-side conjuncts BY NAME (sha256(statement.bin) == composed.json#statement_digest; h_L; h_O; verdict byte) -- the ONE shipped
        # function reverify.py V4 calls, so the live step 5 and the post-hoc re-verifier agree by name
        prc = steps.proof_record_consistency(rec / "obligations" / name, pv_hex) if pv_hex else ["no public values"]
        ok = ok_host and bnd is not None and bnd.ok and st_ok and not prc and str(pr.get("status", "")).startswith("PROVEN")
        per_chunk.append({"chunk": name, "host_verify_ok": ok_host, "binding_ok": bool(bnd and bnd.ok), "statement_bound": st_ok, "record_consistency_problems": prc, "prover_status": (pr or {}).get("status"), "accepted": ok,
                          "cycles": ((pr or {}).get("execute") or {}).get("total_cycles"), "prove_s": ((pr or {}).get("prove") or {}).get("prove_seconds"), "units": len(b.units)})
        if ok:
            accepted.append(AcceptedProof(name, obligations_of_typed_object(TypedObligationSet.from_json(b.record["object"])), typed_object=TypedObligationSet.from_json(b.record["object"])))
    log.check("V4.proofs_binding", bool(per_chunk) and all(c["accepted"] for c in per_chunk), "every proof object: SP1 compressed proof verifies under the approved vk (verifier's own `verify`), verdict true, not unsound, statement_match; public values h_O || h_L || 1 bind the typed object (check_binding) and the statement bytes",
              who="SHIPPED veritor-zk-host verify + veritor.interface.check_binding + steps.proof_record_consistency", chunks=per_chunk)
    # ---- coverage: expected statements of every selected VU (this circuit's positions) vs the accepted proofs' obligations (positions checked) ----
    class _Part:
        def locate(self, vu_id: str):
            return OB.expectation_of(part.vus[int(vu_id)], core)

    class _Cache:  # boundary trees' id lists from the record's index.json (committed_trees reads ru/ trees itself)
        def tree(self, tid: str):
            if tid == "weights":  # DemoCore_v1 has no weights: the committed weights tree is the empty tree (0 ids)
                return type("T", (), {"ids": []})()
            return type("T", (), {"ids": _index_ids(sel / tid)})()

    retained = list(selected.get("retained_rus") or range(part.n_r))
    retained_trees = {core.interior_tree(r) for r in retained}  # (request, step) of every retained RU under the record's spelling
    tree_ids = ["weights"] + [t for t in c2["trees"] if t.startswith("boundary/")] + [t for t in c2["trees"] if t in retained_trees]
    ct = committed_trees(sel, c2, tree_ids, _Cache())
    log.check("V5.trees", not ct.problems, "every committed tree a statement may reference: recomputed ids_digest == committed, id count == leaf_count", who="SHIPPED verifier_v3.expected.committed_trees", trees=len(ct.trees), problems=ct.problems)
    circuit = RecordCircuit(inst.epoch.family, {v.vu_index: v.kind for v in part.vus})
    try:
        expected = expected_statements(circuit, _Part(), [str(i) for i in Q])
        exp_err = None
    except Exception as e:  # noqa: BLE001
        expected, exp_err = None, f"{type(e).__name__}: {e}"
    log.check("V5.expected", expected is not None and len(expected) == len(Q), "the verifier derives the expected statement of every selected VU (positions from the circuit + committed id lists; kind body digest via the registered builder's dispatch; digest-unique)",
              who="SHIPPED coverage_v1.coverage.expected_statements + verifier_v3.expected.RecordCircuit (locator: E8's circuit map)", expected=len(expected or {}), error=exp_err)
    report = check_coverage(expected or {}, accepted, trees=ct.trees, rank_of=ct.rank_of()) if expected else None
    (rec / "verify" / "coverage.json").write_text(json.dumps(report.to_json() if report else {"error": exp_err}, indent=1) + "\n")
    log.check("V5.positions", report is not None and report.positions_checked and not report.position_problems, "every accepted proof binds exactly the positions the verifier derives (check_expected_positions: committed root, leaf count, framing, rank == position, subcircuit digest, no unused element)",
              who="SHIPPED veritor.interface.expected.check_expected_positions via coverage_v1.check_coverage", problems=(report.position_problems if report else None))
    log.check("V5.coverage", report is not None and report.ok, "every expected statement is covered exactly once by an accepted proof; no duplicate, no unexpected obligation",
              who="SHIPPED coverage_v1.coverage.check_coverage", expected=len(expected or {}), covered=len(report.covered) if report else 0, uncovered=list(report.uncovered) if report else None, proofs=len(accepted))
    covered_ids = sorted(report.covered, key=int) if report else []
    uncovered_ids = sorted(report.uncovered, key=int) if report else [str(i) for i in Q]
    # [INTEGRATOR D6] the coverage line of every record, verbatim form; identities listed, never |Q| = proofs + input equations
    coverage_line = (f"sampled VUs = {len(Q)}; covered by a verified expected statement = {len(covered_ids)} (identities: {_ids(covered_ids)}); "
                     f"proof objects = {len(accepted)}; uncovered = {len(uncovered_ids)} (identities: {_ids(uncovered_ids)})")
    log.note(coverage_line)
    return {"Q": len(Q), "proof_objects": len(accepted), "statements_built": len(per_chunk), "accepted_proofs": len(accepted), "covered": len(covered_ids),
            "covered_ids": covered_ids, "uncovered_ids": uncovered_ids, "coverage_line": coverage_line,
            "coverage_ok": bool(report and report.ok), "per_chunk": per_chunk, "host_verify": hv}


def _index_rows(tree_dir: Path) -> list[dict[str, Any]]:
    """The leaf rows of a tree's index.json: E8's boundary/ru index (``entries``) or E2's interior index (``leaves``, rank order)."""
    idx = json.loads((tree_dir / "index.json").read_text())
    return list(idx.get("entries") or idx.get("leaves") or [])


def _index_ids(tree_dir: Path) -> list[str]:
    return [e["canonical_id"] for e in _index_rows(tree_dir)]


def _index_gates(tree_dir: Path) -> list[int]:
    """Every gate a tree's index claims: E8 rows carry ``gates: [lo, hi)``; E2 rows carry ``g_first/g_last`` + the tree's ``gates.bin``."""
    rows = _index_rows(tree_dir)
    if rows and "gates" in rows[0]:
        return [g for e in rows for g in range(e["gates"][0], e["gates"][1])]
    gb = tree_dir / "gates.bin"
    if gb.exists():
        return [int(g) for g in np.frombuffer(gb.read_bytes(), dtype="<i8")]
    return [g for e in rows for g in range(int(e["g_first"]), int(e["g_last"]) + 1)]


def _certificate_or_error(part: C.Partition, q: str, s: str, B_ex: int, delta: str) -> dict[str, Any]:
    try:
        from . import certificate as CERT
        return CERT.certificate(part, q=q, s=s, B_ex=int(B_ex), delta=delta)
    except Exception as e:  # noqa: BLE001 -- the certificate is reported, never blocks the verdict
        return {"error": f"{type(e).__name__}: {e}", "label": "NOT DONE"}


def _ids(ids: list[str], keep: int = 12) -> str:
    if not ids:
        return "none"
    return ", ".join(ids) if len(ids) <= keep else ", ".join(ids[:keep]) + f", … (+{len(ids) - keep}; full list in verify/coverage.json)"


# -- the run --------------------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True, help="record root (created; must not exist)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--input-seed", type=int, default=1, help="the declared input x (deterministic words from this seed; NOT the verifier's seed)")
    ap.add_argument("--q", default="1")
    ap.add_argument("--s", default="1")
    ap.add_argument("--B-ex", type=int, default=0)
    ap.add_argument("--exempt", default="", help="PREDECLARED controlled exemption test: RU indices the prover withholds if selected (label: test)")
    ap.add_argument("--exemption-policy", default="gamble")
    ap.add_argument("--condition-nonempty", action="store_true")
    ap.add_argument("--pod-ssh", default=None, help="E10's <pod>.ssh.sh wrapper; with --pod-dir/--pod-host/--pod-repo proofs are made there")
    ap.add_argument("--pod-dir", default=None)
    ap.add_argument("--pod-host", default="/workspace/bin/veritor-zk-host-cuda")
    ap.add_argument("--pod-repo", default="/workspace/veritor")
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--request", default="per-ru", choices=list(RC.REQUEST_MODES),
                    help="request spelling the record uses: 'per-ru' (core-ru<r>; archived runs) | 'single' (E6's CORE pin: one request core-0, RU r = step r; the SHIPPED steps.committed_rus maps request->RU only, so S4.step4_check refuses by name until E9 maps (request, step))")
    ap.add_argument("--interior", default="ru", choices=list(RC.INTERIOR_MODES), help="interior tree spelling the prover writes: 'ru' (today's shipped ck_elem) | 'e2' (E2's interior/<req>/<step> through e2e.interior)")
    a = ap.parse_args(argv)
    RC.set_request_mode(a.request)
    rec = Path(a.record)
    if rec.exists():
        raise SystemExit(f"{rec} exists: a protocol run starts from nothing")
    for d in ("fixture", "instance", "adapter/selective", "handshake", "logs", "verify"):
        (rec / d).mkdir(parents=True)
    chron, sel, hs, logs, vdir, idir, fx = rec / "chronology", rec / "adapter" / "selective", rec / "handshake", rec / "logs", rec / "verify", rec / "instance", rec / "fixture"
    log = Log(vdir / "verifier.log")
    T: dict[str, float] = {}
    t0 = time.perf_counter()
    events: list[dict[str, Any]] = []

    def ev(name: str, **kw: Any) -> None:
        events.append({"utc": utc(), "event": name, **kw})
        print(f"[demo_core.runner] {utc()} {name} " + " ".join(f"{k}={str(v)[:80]}" for k, v in kw.items()), flush=True)

    # ---- the auditor's declarations, BEFORE anything is committed: fixture (x), instance, population, weights registration (empty) ----
    prog, d = C.build(a.N, a.R)
    part = C.partition(a.N, a.R, prog, d)
    inputs = PR.random_inputs(a.N, a.input_seed)
    (fx / "fixture.json").write_text(json.dumps({"schema": "veritor/demo-core/fixture/v1", "N": a.N, "R": a.R, "input_seed": a.input_seed, "inputs": inputs, "label": LABEL}, indent=1) + "\n")
    pop = I.population_doc(part)
    (idir / "population.compact.json").write_text(json.dumps(pop, indent=1, sort_keys=True) + "\n")
    decl = I.declared_inputs_doc(part, inputs)
    (idir / "declared_inputs.json").write_text(json.dumps(decl, indent=1) + "\n")
    epoch_id = f"{a.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_label = f"run config {a.name}: q={a.q} s={a.s} B_ex={a.B_ex} exempt=[{a.exempt}] policy={a.exemption_policy} (predeclared)"
    # the epoch's request id: E6's frozen spelling `core-0` under --request single (one program instance whose steps are the RUs); the archived
    # per-ru runs declared the program instance as request 'core' (their RU ids core-ru<r> are the record's request ids)
    epoch_request = RC.SINGLE_REQUEST_ID if a.request == "single" else "core"
    inst_doc = I.instance_doc(part, pop, instance_id=epoch_id, epoch_id=epoch_id, request_id=epoch_request, q=a.q, s=a.s, B_ex=a.B_ex,
                              B_ex_provenance=f"{run_label}; declared before r_q", declared=decl, profile_id=RC.profile_of(part), run_label=run_label,
                              interior_layout=(RC.E2_LAYOUT_PIN if a.interior == "e2" else None))
    (idir / "protocol_instance.json").write_text(json.dumps(inst_doc, indent=1, sort_keys=True) + "\n")
    empty_root = MerkleTree({}, "weights").root.hex()
    registered = {"schema": "veritor/demo-core/weights-authorized/v1", "expected_root": empty_root, "leaf_count": 0, "ids_digest": hashing.ids_digest([]).hex(),
                  "note": "DemoCore_v1 has no weights: the registered weight commitment is the empty tree", "utc": utc()}
    (idir / "weights_authorized.json").write_text(json.dumps(registered, indent=1) + "\n")
    (rec / "run_config.json").write_text(json.dumps({"schema": "veritor/demo-core/run-config/v1", "utc": utc(), "name": a.name, "N": a.N, "R": a.R, "input_seed": a.input_seed, "q": a.q, "s": a.s,
                                                     "B_ex": a.B_ex, "exempt_predeclared": [int(x) for x in a.exempt.split(",") if x.strip()], "exemption_policy": a.exemption_policy, "interior": a.interior, "request": a.request,
                                                     "condition_nonempty": a.condition_nonempty, "label": LABEL, "verifier_randomness": "fresh secrets.token_bytes(32) in partition_sample.stage1; never chosen, never re-drawn"}, indent=1) + "\n")
    pinned = {n: sha256_file(idir / n) for n in ("protocol_instance.json", "population.compact.json", "declared_inputs.json", "weights_authorized.json")}
    for n in ("protocol_instance.json", "weights_authorized.json"):
        shutil.copyfile(idir / n, rec / n)
    inst_path, pop_path = idir / "protocol_instance.json", idir / "population.compact.json"
    load_inst, _ = ps.loaders("contract")
    fmt_problems = ps.check_instance_format("contract", inst_doc)
    log.check("V8.instance_format", not fmt_problems, "instance format 'contract' matches partition.partition_digest_kind", who="SHIPPED partition_sample.check_instance_format", problems=fmt_problems)
    inst = load_inst(inst_path)
    log.check("V8.instance_load", True, "protocol_instance.json loads under the contract format (every validate_instance rule) and declares the epoch", who="SHIPPED protocol_instance.load_instance", contract=inst.is_contract_instance)
    idig = ps.instance_digest(inst)
    (idir / "instance_digest.txt").write_text(idig + "\n")
    ev("instance_pinned", instance_digest=idig, files=pinned, epoch=inst.epoch.epoch_id, B_ex=inst.epoch.B_ex, q=a.q, s=a.s)
    # RU <-> request map handed to the SHIPPED steps.step4_check (committed_rus maps a tree's request_id -> RU): injective under per-ru;
    # under single every RU shares request core-0 and only the tree's STEP names the RU -> the shipped map collapses (E9 flag, 21:18Z report)
    # the driver's form (driver.py: {rid: i for i, rid in enumerate(request_ids)} = request ORDINAL): per-ru -> {core-ru<r>: r}; single -> {core-0: 0}
    ru_of_request = {rid: i for i, rid in enumerate(dict.fromkeys(RC.request_id(r) for r in range(part.n_r)))}
    rule = ("RU r = stage call r of the one program instance (request 'core'); RU id core-ru<r>; VU index = r*2N + gate offset" if a.request == "per-ru"
            else f"RU r = step r of the one program instance (request {RC.SINGLE_REQUEST_ID}); its trees are boundary/{RC.SINGLE_REQUEST_ID} (leaf per RU) and <interior>/{RC.SINGLE_REQUEST_ID}/<r>; VU index = r*2N + gate offset")
    (rec / "ru_map.json").write_text(json.dumps({"schema": "veritor/demo-core/ru-map/v1", "rule": rule, "request_mode": a.request,
                                                 "rus": [{"ru": r, "ru_id": RC.request_id(r), "step": RC.step_of(r), "request_id": epoch_request, "gate_interval": [part.rus[r].lo, part.rus[r].hi]} for r in range(part.n_r)]}, indent=1) + "\n")

    # ---- the prover process (phase 0 runs immediately) ----
    prover_cmd = [a.py, "-m", "e2e.demo_core.prover", "--fixture", str(fx), "--out", str(sel), "--handshake", str(hs), "--exempt", a.exempt, "--exemption-policy", a.exemption_policy, "--interior", a.interior, "--request", a.request]
    env = {**os.environ, "PYTHONPATH": f"{POC}:{ROOT}:{ROOT / 'src'}"}
    plog = (logs / "prover.log").open("w")
    prover = subprocess.Popen(prover_cmd, stdout=plog, stderr=subprocess.STDOUT, env=env, cwd=str(POC))
    ev("prover_started", cmd=" ".join(prover_cmd))
    verdict = "NOT DONE"
    step2: dict[str, Any] = {}
    step4: dict[str, Any] = {}
    s5: dict[str, Any] = {}
    J: list[int] = []
    F: list[int] = []
    try:
        # ================= step 1 -> 2 =================
        wait_for(hs / READY0, hs / ABORT, prover, a.timeout, READY0)
        T["phase0_wall_s"] = round(time.perf_counter() - t0, 2)
        ready0 = json.loads((hs / READY0).read_text())
        c0 = json.loads((sel / "commitments_phase0.json").read_text())
        boundary_digest = sha256_file(sel / "commitments_phase0.json")
        log.check("V3.boundary", boundary_digest == ready0["commitments_phase0_sha256"], "sha256(commitments_phase0.json) == the prover's READY_PHASE0 digest", who="SHIPPED (driver.py rule)", digest=boundary_digest[:16])
        problems, y = step2_checks(log, inst, inst_doc, part, sel, c0, registered, pop_path)
        step2 = {"problems": problems, "y": y, "boundary_digest": boundary_digest, "label": "MEASURED"}
        ev("step2_check", ok=not problems, problems=problems)
        if problems:
            (hs / ABORT).write_text("step2: " + "; ".join(problems) + "\n")
            verdict = "STEP2_REFUSED"
            raise SystemExit(f"step2 REFUSED: {problems}")
        # r_q (shipped sampler; the seed is fresh verifier randomness held in this process)
        t1 = time.perf_counter()
        doc = ps.stage1(inst_path, pop_path, boundary_digest, chron, instances=1, condition_nonempty=a.condition_nonempty, instance_format="contract",
                        boundary_request_ids=list(inst.epoch.request_ids))
        T["stage1_s"] = round(time.perf_counter() - t1, 3)
        rq = next(e for e in doc["events"] if e["name"] == "r_q")
        J = [int(r) for r in rq["selection"]]
        ev("r_q", J=J, redraws=rq.get("redraws", 0), digest=rq["digest"], seed_custody=doc["seed_custody"])
        challenge = {"selected": [RC.request_id(r) for r in J], "T_0": next(e["digest"] for e in doc["events"] if e["name"] == "seed_commitment"), "r_q": rq["digest"],
                     "commitments_phase0_sha256": boundary_digest, "utc": utc(), "sampler": ps.SAMPLER, "instance_digest": idig, "J": J, "B_ex": inst.exemption_budget,
                     "B_ex_unit": "replay_units", "epoch_id": inst.epoch.epoch_id}
        tmp = hs / (CHALLENGE + ".tmp")
        tmp.write_text(json.dumps(challenge, indent=1) + "\n")
        os.replace(tmp, hs / CHALLENGE)
        # ================= step 3 -> 4 =================
        t2 = time.perf_counter()
        wait_for(hs / READY2, hs / ABORT, prover, a.timeout, READY2)
        prover.wait(timeout=120)
        T["phase2_wall_s"] = round(time.perf_counter() - t2, 2)
        interior_digest = sha256_file(sel / "commitments.json")
        ready2 = json.loads((hs / READY2).read_text())
        log.check("V3.interior", interior_digest == ready2["commitments_sha256"], "sha256(commitments.json) == the prover's READY_PHASE2 digest", who="SHIPPED (driver.py rule)", digest=interior_digest[:16])
        exreq = json.loads((sel / "exemption_request.json").read_text())
        log.check("V8.exemption_request", sha256_file(sel / "exemption_request.json") == ready2.get("exemption_request_sha256"), "exemption_request.json digest == READY_PHASE2 (F fixed with the interior commitment, before r_s)", who="SHIPPED (driver.py rule)", F=exreq.get("F"))
        c2 = json.loads((sel / "commitments.json").read_text())
        pj = json.loads((sel / "selective_prover.json").read_text())
        ev("phase2", interior_digest=interior_digest, k=pj["k"], ru_leaves=pj["phase2"]["ru_leaves"], F=exreq.get("F"))
        problems, F, committed = steps.step4_check(inst, J=J, r_q=rq["digest"], instance_digest=idig, c0=c0, c2=c2, exreq=exreq, ru_of_request=ru_of_request, policy=a.exemption_policy)
        log.check("S4.step4_check", not problems, "F ⊆ J, |F| <= B_ex, request bound to this instance/r_q, every retained RU's step-3 values committed, no phase-0 tree recommitted, per-RU fields well-formed",
                  who="SHIPPED verifier_v3.steps.step4_check + response_record.validate", J=J, F=F, committed=committed, problems=problems)
        # single position: no gate has two committed copies (the vLLM-only output_copy_step4 is NOT APPLICABLE: this circuit commits each value once)
        idx_gates: dict[int, list[str]] = {}
        for tid in c2["trees"]:
            if tid == "weights":
                continue
            for g in _index_gates(sel / tid):
                idx_gates.setdefault(g, []).append(tid)
        dup = {g: t for g, t in idx_gates.items() if len(t) > 1}
        retained = [r for r in J if r not in set(F)]
        want = set(g for name in ("a", "b", "c") for g in part.inputs[name]) | set(g for ru in part.rus for g in ru.returns) | set(v.gate for r in retained for v in part.rus[r].vus)
        log.check("S4.single_position", not dup and set(idx_gates) == want, "every committed gate has exactly ONE committed position (inputs + every RU output tuple at step 1, retained RUs' interior gates at step 3); output_copy_step4 (two token copies) NOT APPLICABLE to this circuit",
                  who="E8-CHECK", committed_gates=len(idx_gates), duplicates=len(dup), retained=retained)
        step4 = {"problems": problems, "F": F, "committed_rus": committed, "retained": retained, "B_ex": inst.exemption_budget, "single_position_ok": not dup and set(idx_gates) == want, "label": "MEASURED"}
        if problems or dup:
            (hs / ABORT).write_text("step4: " + "; ".join(problems) + "\n")
            verdict = "STEP4_REFUSED"
            raise SystemExit(f"step4 REFUSED: {problems}")
        # r_s over J minus F (shipped)
        t3 = time.perf_counter()
        doc = ps.stage2(inst_path, pop_path, interior_digest, chron, instances=1, exemptions=F, committed_rus=committed, instance_format="contract")
        T["stage2_s"] = round(time.perf_counter() - t3, 3)
        ps.reveal(chron)
        cproblems = ps.verify(inst_path, pop_path, chron, instances=1, instance_format="contract")
        log.check("V2.chronology", not cproblems, "chronology re-derived (seed opens its commitment, r_q incl. redraws, exemption request digest, r_s, selected.json)", who="SHIPPED partition_sample.verify", problems=cproblems)
        rs = next(e for e in doc["events"] if e["name"] == "r_s")
        selected = json.loads((chron / "selected.json").read_text())
        selected["retained_rus"] = retained
        ev("r_s", Q=len(rs["selection"]), candidates=rs.get("candidates"), kinds=len(selected["kinds"]))
        rf = steps.ru_fields(inst, exreq=exreq, selected_rows=selected["selected"], unsupported_kinds=set(), s=inst.sampler.law.s)
        (vdir / "ru_fields.json").write_text(json.dumps(rf, indent=1, sort_keys=True) + "\n")
        ev("ru_fields", F=rf["F"], sampled_unsupported=rf["sampled_unsupported_count"])
        # ================= step 5 =================
        core = RC.CoreRecord(part, PR.produce(part, inputs, prog), sel / "rebuilt", interior=a.interior)  # the prover side re-derives its trees (deterministic) to build statements
        core.commit_phase0()
        core.commit_interior(retained)
        same = all(core.commitments["trees"][t]["root"] == c2["trees"][t]["root"] for t in c2["trees"])
        log.check("S5.prover_trees_rebuilt", same, "the prover's statement-building record reproduces the committed roots byte-for-byte (deterministic writer)", who="E8-CHECK", trees=len(c2["trees"]))
        shutil.rmtree(sel / "rebuilt", ignore_errors=True)
        pod = {"ssh": a.pod_ssh, "dir": a.pod_dir, "host": a.pod_host, "repo": a.pod_repo} if a.pod_ssh and a.pod_dir else None
        t5 = time.perf_counter()
        s5 = step5(log, rec, part, core, inst, selected, pod=pod, approved_path=POC / "e2e" / "approved_guests.json")
        T["step5_s"] = round(time.perf_counter() - t5, 2)
        ok = not log.failed
        verdict = "ACCEPTED" if ok else "REJECTED"
    except SystemExit as e:
        if verdict == "NOT DONE":
            verdict = "ERROR"
        print(f"[demo_core.runner] {verdict}: {e}", file=sys.stderr, flush=True)
    finally:
        if prover.poll() is None:
            prover.kill()
        plog.close()
        T["total_s"] = round(time.perf_counter() - t0, 2)
        (rec / "protocol_events.json").write_text(json.dumps(events, indent=1, default=str) + "\n")
        summary = {"schema": "veritor/demo-core/protocol/v1", "utc": utc(), "name": a.name, "verdict": verdict, "checks_failed": log.failed, "checks": len(log.rows),
                   "instance_digest": idig, "pinned_sha256": pinned, "program_digest": part.program_digest, "partition_digest": pop["partition_digest"], "plan_digest": pop["plan_digest"],
                   "N_V": part.n_v, "N_R": part.n_r, "kinds": part.kind_counts(), "law": {"q": a.q, "s": a.s, "B_ex": a.B_ex}, "J": J, "F": F, "retained": [r for r in J if r not in set(F)],
                   "step2": step2, "step4": step4, "step5": s5, "timing": T, "label": LABEL,
                   # the output-freedom certificate U of THIS configuration through the shipped calculator (published even when U = W)
                   "U": _certificate_or_error(part, a.q, a.s, a.B_ex, inst_doc["security"]["delta"])}
        (rec / "protocol.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
        (rec / "verifier_exit.json").write_text(json.dumps({"schema": "veritor/verifier-exit/v1", "utc": utc(), "reason_code": verdict, "checks_failed": log.failed, "J": J, "F": F,
                                                            "Q": s5.get("Q"), "proof_objects": s5.get("proof_objects"), "covered": s5.get("covered"),
                                                            "coverage_line": s5.get("coverage_line")}, indent=1) + "\n")
        print(f"[demo_core.runner] {verdict}: J={J} F={F} failed={log.failed} {T}", flush=True)
        if s5.get("coverage_line"):
            print(f"[demo_core.runner] {s5['coverage_line']}", flush=True)
    return 0 if verdict == "ACCEPTED" else 1


if __name__ == "__main__":
    sys.exit(main())
