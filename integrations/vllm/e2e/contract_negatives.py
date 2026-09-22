#!/usr/bin/env python
"""[A8 CONFORMANCE 2026-09-10] Contract negatives of the archived REC-P1-B0 record through the mainline record verifier
(``verifier_v2/reverify.py`` @ 139e080f), each on an APFS clone (``cp -c``, copy-on-write) of the record under ``--work``; the archive is
never written.  Two groups:

* **L1 baseline** (``G1``-``G9``, ``T1``-``T3``): L1's twelve negatives (``scripts/rec_p1_negatives.py``) re-run here.
* **Contract cases** (``R-B`` baseline, ``R-N1``..``R-N6``): edits that express a ``docs/PROTOCOL.md`` case on the artefacts, then the whole
  verifier is run on the clone and every check's PASS/FAIL is recorded (``record/<case>/reverify.log``).

The laptop has no SP1 host, so ``--as-host`` makes this module a **stand-in for ``veritor-zk-host``** (``info`` / ``verify`` / ``verify-many``):
it answers from the archive's recorded verdicts (``verify/host_verify.jsonl``): a proof file whose sha256 is an archived proof's gets that proof's
recorded public values (``ok`` = recorded); any other proof is refused (``ok`` false — the stand-in cannot run SP1); ``statement_match`` is
recomputed exactly as the host does (sha256(statement bytes) == public h_L).  A proof file with a ``<proof>.reproved.json`` beside it stands for
an adversary's *re-made* proof of an edited statement whose relations are all true: the stand-in returns the public values the guest would
commit for it (h_O' || h_L' || 1).  Every row says which of its outcomes is SP1's and which is the stand-in's.  ``execute`` is not available:
T1-T3 report the Python relation verdict only.

Usage:
    PYTHONPATH=. python -m e2e.contract_negatives --archive ~/veritor-archive/REC-P1-B0-2026-09-10 --work /tmp/a8-rec \
        --out data/contract/a8-conformance/record [--only R-B,R-N1] [--skip-l1]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent  # e2e
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dump(p: Path, doc: Any) -> None:
    p.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")


# ------------------------------------------------------------------------------------------------------------------------------
# the host stand-in
# ------------------------------------------------------------------------------------------------------------------------------


class StandInHost:
    """Recorded-verdict stand-in for ``veritor-zk-host`` (see module docstring).  Never runs SP1."""

    def __init__(self, archive_rec: Path) -> None:
        # REC-P1-B0 layout: <archive>/setup/host-info.json + verify/host_verify.jsonl; A9 record_v5 layout: proofs/logs/host-info.log +
        # verify/reverify_v3/host_verify.jsonl (the pod's SP1 verdicts of the record's own proofs)
        info = archive_rec.parent / "setup" / "host-info.json"
        if not info.exists():
            info = archive_rec / "proofs" / "logs" / "host-info.log"
        self.info_doc = next(json.loads(ln) for ln in info.read_text().splitlines() if ln.strip().startswith("{"))
        self.by_proof_sha: dict[str, dict] = {}
        hv_path = archive_rec / "verify" / "host_verify.jsonl"
        if not hv_path.exists():
            hv_path = archive_rec / "verify" / "reverify_v3" / "host_verify.jsonl"
        rows = [json.loads(ln) for ln in hv_path.read_text().splitlines() if ln.strip()]
        for row in rows:
            hv = row.get("host_verify")
            pf = archive_rec / "proofs" / f"{row['chunk']}-compressed.bin"
            if hv and pf.exists():
                self.by_proof_sha[sha256_file(pf)] = hv

    def info(self) -> dict:
        return {**self.info_doc, "backend": "stand-in(recorded verdicts; not SP1)", "guest_features": []}

    def verify(self, proof: Path, statement: Path | None) -> dict:
        t0 = time.perf_counter()
        marker = Path(str(proof) + ".reproved.json")
        if marker.exists():
            m = json.loads(marker.read_text())
            # [segment 3] a marker may carry `verdict: false` (the guest's public verdict byte of a FALSE statement) and its own label
            pv, ok, verdict, backend = m["public_values"], True, bool(m.get("verdict", True)), m.get("backend", "stand-in:reproved (adversary's re-made proof of a true statement; not SP1)")
        else:
            hv = self.by_proof_sha.get(sha256_file(proof))
            if hv is None:
                return {"ok": False, "error": "stand-in: proof sha256 is not an archived proof's; SP1 verification is not available on this host", "backend": "stand-in"}
            pv, ok, verdict, backend = hv["public_values"], bool(hv["ok"]), bool(hv["verdict"]), "stand-in(recorded verdict)"
        h_l = bytes.fromhex(pv)[32:64].hex()
        t1 = time.perf_counter()
        sm = (hashlib.sha256(statement.read_bytes()).hexdigest() == h_l) if statement is not None else None
        t2 = time.perf_counter()
        return {"backend": backend, "ok": ok, "public_values": pv, "statement_digest": h_l, "typed_object_digest": bytes.fromhex(pv)[:32].hex(), "verdict": verdict,
                "unsound": False, "vk_hash": self.info_doc["vk_hash"], "statement_match": sm, "verify_seconds": 0.0, "load_seconds": t1 - t0, "statement_hash_seconds": t2 - t1,
                "job_wall_seconds": time.perf_counter() - t0}

    def main(self, argv: list[str]) -> int:
        cmd = argv[0] if argv else ""
        if cmd == "info":
            print(json.dumps(self.info()))
            return 0
        if cmd == "verify":
            ap = argparse.ArgumentParser()
            ap.add_argument("--proof", required=True)
            ap.add_argument("--statement", default=None)
            a = ap.parse_args(argv[1:])
            out = self.verify(Path(a.proof), Path(a.statement) if a.statement else None)
            print(json.dumps(out))
            return 0 if out.get("ok") else 1
        if cmd == "verify-many":
            print(json.dumps({"event": "ready", **{k: self.info_doc[k] for k in ("elf_sha256", "vk_hash", "sp1_version")}, "backend": "stand-in(recorded verdicts; not SP1)",
                              "client_seconds": 0.0, "setup_seconds": 0.0}), flush=True)
            for line in sys.stdin:
                if not line.strip():
                    continue
                job = json.loads(line)
                out = self.verify(Path(job["proof"]), Path(job["statement"]) if job.get("statement") else None)
                print(json.dumps({"id": job.get("id"), **out}), flush=True)
            return 0
        print(json.dumps({"ok": False, "error": f"stand-in host: subcommand {cmd!r} not available (no SP1 on this machine)"}))
        return 2


def write_stub(work: Path, archive_rec: Path) -> Path:
    stub = work / "host-stub.sh"
    stub.write_text("#!/bin/sh\n" f'export PYTHONPATH="{ROOT / "src"}:{POC}:{ROOT}"\n' f'exec "{sys.executable}" "{Path(__file__).resolve()}" --as-host "{archive_rec}" -- "$@"\n')
    stub.chmod(0o755)
    return stub


# ------------------------------------------------------------------------------------------------------------------------------
# clones and the verifier run
# ------------------------------------------------------------------------------------------------------------------------------


def clone(src: Path, dst: Path) -> None:
    """APFS copy-on-write clone (``cp -c``); only the files a case edits take new blocks."""
    if dst.exists():
        shutil.rmtree(dst)
    res = subprocess.run(["cp", "-Rc", str(src), str(dst)], capture_output=True, text=True)
    if res.returncode != 0 and sys.platform != "darwin":  # Linux (pod): reflink where the filesystem supports it, plain copy otherwise
        shutil.rmtree(dst, ignore_errors=True)
        res = subprocess.run(["cp", "-R", "--reflink=auto", str(src), str(dst)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"clonefile copy failed (same APFS volume needed): {res.stderr[-300:]}")


def parse_log(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if p.exists():
        for ln in p.read_text().splitlines():
            parts = ln.split(" ", 3)
            if len(parts) >= 3 and parts[2] in ("PASS", "FAIL"):
                out[parts[1]] = parts[2]
    return out


def slice_record(rec: Path, keep_chunks: int) -> str:
    """Keep only the first ``keep_chunks`` obligation chunks (and their proofs) of a clone: a laptop-sized slice (launch control: archive-verify is
    pod-only unless a small slice).  Every check that is per chunk is unchanged; coverage over the other chunks' VUs is reported as uncovered."""
    dirs = sorted(p for p in (rec / "obligations").iterdir() if p.is_dir() and p.name.startswith("chunk"))
    for d in dirs[keep_chunks:]:
        shutil.rmtree(d)
        for f in (rec / "proofs").glob(f"{d.name}*"):
            f.unlink()
    return f"slice: {keep_chunks} of {len(dirs)} chunks kept"


def run_reverify(rec: Path, pop: Path, expr: Path, stub: Path, outdir: Path, timeout_s: int, verifier: str = "v2") -> dict[str, Any]:
    script = ROOT / "verifier_v2" / "reverify.py" if verifier == "v2" else POC / "e2e" / "verifier_v3" / "reverify.py"
    head = [sys.executable, str(script)]
    if verifier == "v3" and (rec / "population.v5.json").exists():  # a record_v5 record: reverify under A9's two loader substitutions (see --as-reverify-v5)
        head = [sys.executable, str(Path(__file__).resolve()), "--as-reverify-v5"]
    cmd = [*head, "--record", str(rec), "--population", str(pop), "--expressibility", str(expr), "--host", str(stub),
           "--out", str(outdir), "--info-from-ready"]
    if (rec / "verify" / "verifier.log").exists():
        cmd += ["--compare", str(rec / "verify" / "verifier.log")]
    if (rec / "a8_extra_args.json").exists():  # an edit may add verifier-side CLI arguments (e.g. --approved <verifier-side copy>)
        cmd += json.loads((rec / "a8_extra_args.json").read_text())
        (rec / "a8_extra_args.json").unlink()
    outdir.mkdir(parents=True, exist_ok=True)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t = time.perf_counter()
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{POC}:{ROOT}"}
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
        rc, tail = res.returncode, (res.stdout.strip().splitlines() or [""])[-3:]
        err_tail = (res.stderr.strip().splitlines() or [])[-3:]  # a refusal before any check (load_instance) is a traceback on stderr, not a log line
    except subprocess.TimeoutExpired:
        rc, tail, err_tail = -9, [f"watchdog: killed after {timeout_s} s"], []
    wall = time.perf_counter() - t
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    checks = parse_log(outdir / "reverify.log")
    (outdir / "stdout.tail.txt").write_text("\n".join(tail) + "\n")
    if err_tail:
        (outdir / "stderr.tail.txt").write_text("\n".join(err_tail) + "\n")
    rec3 = outdir / "record_v3.json"
    return {"command": " ".join(cmd), "rc": rc, "seconds": round(wall, 1), "maxrss_mb_children": round(r1.ru_maxrss / (1 << 20), 1), "checks": len(checks),
            "failed": sorted(k for k, v in checks.items() if v == "FAIL"), "status_line": next((ln for ln in tail if "checks," in ln), tail[-1] if tail else ""),
            "stderr_tail": err_tail,
            "certificate": (json.loads((outdir / "certificate.json").read_text()) if (outdir / "certificate.json").exists() else None),
            "record_v3": ({k: json.loads(rec3.read_text()).get(k) for k in ("status", "status_line", "coverage", "sampled")} if rec3.exists() else None),
            "log_lines": [ln for ln in (outdir / "reverify.log").read_text().splitlines() if " FAIL " in ln or "V5.prover_mapping" in ln or "V8.step2" in ln] if (outdir / "reverify.log").exists() else []}


# ------------------------------------------------------------------------------------------------------------------------------
# the record edits (contract cases)
# ------------------------------------------------------------------------------------------------------------------------------


def rederive(chunk_dir: Path, record: dict) -> dict[str, str]:
    """Recompute canonical / h_O / statement (format 3) / h_L of an edited typed object, as a re-proving adversary would; write statement.bin
    and the stand-in's ``reproved`` marker (public values h_O' || h_L' || 1)."""
    from veritor.interface import TypedObligationSet, canonical_bytes, object_digest, validate
    from veritor.interface.lowering import statement_bytes_v3
    from veritor.protocol.proofs.wire import statement_digest

    assert int(record.get("statement_format", 2)) == 3, record.get("statement_format")
    obj = TypedObligationSet.from_json(record["object"])
    validate(obj)  # the interface's own well-formedness rules (E_* codes) must accept the edited object, or the run is meaningless
    record["canonical"] = canonical_bytes(record["object"]).decode()
    record["digest"] = object_digest(record["object"])
    st = statement_bytes_v3(obj)
    record["statement"] = record["statement_v3"] = st.hex()
    record["statement_digest"] = record["statement_v3_digest"] = statement_digest(st).hex()
    (chunk_dir / "statement.bin").write_bytes(st)
    dump(chunk_dir / "composed.json", record)
    proof = chunk_dir.parent.parent / "proofs" / f"{chunk_dir.name}-compressed.bin"
    dump(Path(str(proof) + ".reproved.json"), {"public_values": record["digest"] + record["statement_digest"] + "01", "note": "stand-in: adversary's re-made proof of the edited (true) statement"})
    return {"h_O": record["digest"], "h_L": record["statement_digest"]}


def load_chunk(rec: Path, name: str) -> tuple[Path, dict]:
    d = rec / "obligations" / name
    if not d.exists() and (rec / "obligations" / "chunk00").exists():  # single-chunk records (A9's TINY): every edit lands in chunk00
        d = rec / "obligations" / "chunk00"
    return d, json.loads((d / "composed.json").read_text())


def edit_n1_served_token(rec: Path) -> str:
    """C8 / R4: one served output token of a selected request changed in production.jsonl; every commitment untouched."""
    proto = json.loads((rec / "protocol.json").read_text())
    rid = proto["ru_map"]["rus"][proto["J"][0]]["request_id"]
    p = rec / "adapter" / "production" / "production.jsonl"
    lines = p.read_text().splitlines()
    for i, ln in enumerate(lines):
        d = json.loads(ln)
        if d["request_id"] == rid:
            old = d["output_token_ids"][0]
            d["output_token_ids"][0] = old ^ 1
            lines[i] = json.dumps(d)
            p.write_text("\n".join(lines) + "\n")
            return f"production.jsonl: request {rid[:8]} (RU {proto['J'][0]}) output_token_ids[0] {old} -> {old ^ 1}; boundary/interior commitments untouched"
    raise RuntimeError("selected request not in production.jsonl")


def edit_n2_labels(rec: Path) -> str:
    """C7 (labels only): chunk00 claims VU[1] twice and VU[0] never; the typed object is untouched (h_O, h_L, proof unchanged)."""
    d, r = load_chunk(rec, "chunk00")
    sv, pu = r["selected_vus"], r["provenance"]["units"]
    v0, v1 = sv["coordinates"][0], sv["coordinates"][1]
    sv["coordinates"][0] = v1
    pu[0] = copy.deepcopy(pu[1])
    dump(d / "composed.json", r)
    return f"chunk00/composed.json: selected_vus.coordinates[0] {v0} -> {v1}; provenance.units[0] := units[1]; object untouched"


def find_same_layer_pair(r: dict) -> tuple[int, int] | None:
    """Two coordinate units of one statement with the same (request, layer, op) and different t: B's weight row is already in the object."""
    pu = r["provenance"]["units"]
    seen: dict[tuple, int] = {}
    for i, u in enumerate(pu):
        key = (u["request_id"], u["layer"], u["op"])
        t = u["canonical_id"].split("/t=")[1].split("/")[0]
        if key in seen and pu[seen[key]]["canonical_id"].split("/t=")[1].split("/")[0] != t:
            return seen[key], i
        seen.setdefault(key, i)
    return None


def edit_n3_graft(rec: Path, chunk: str = "chunk01") -> str:
    """C7 on the object: in one statement, the obligation for selected coordinate A = (call_A, j_A) is rewritten to prove (call_A, j_B), the column
    of another selected coordinate B of the same layer (its weight row is already an authenticated leaf of the object); the relation
    out_A[j_B] = dot(x_A, W[j_B]) is TRUE (the honest GEMM computed the whole row).  Labels (selected_vus, provenance) still claim j_A.
    Result: a proof for a VU that was not selected, none for one that was; h_O / h_L re-derived (a re-proving adversary)."""
    d, r = load_chunk(rec, chunk)
    pair = find_same_layer_pair(r)
    assert pair is not None, "no same-layer pair in this chunk"
    A, B = r["provenance"]["units"][pair[0]], r["provenance"]["units"][pair[1]]
    ja, jb = int(A["j"]), int(B["j"])
    obs = r["object"]["obligations"]
    a = next(ob for ob in obs if ob["target"]["id"] == f"{A['canonical_id']}/Gemm[{ja}]")
    w_old, w_new = f"w={A['param']}/c={ja}[", f"w={B['param']}/c={jb}["
    assert A["param"] == B["param"]
    a["boundary"]["in"] = [x.replace(w_old, w_new) for x in a["boundary"]["in"]]
    a["declared_interface"]["args"] = [x.replace(w_old, w_new) for x in a["declared_interface"]["args"]]
    old_out = a["boundary"]["out"][0]
    assert old_out.endswith(f"[{ja}]")
    new_out = old_out[: -len(f"[{ja}]")] + f"[{jb}]"
    a["boundary"]["out"] = a["declared_interface"]["returns"] = [new_out]
    a["target"]["id"] = a["target"]["id"][: -len(f"[{ja}]")] + f"[{jb}]"
    for ref in r["object"]["references"]:
        if ref["ref"] == old_out:
            ref["ref"], ref["offset"] = new_out, [jb]
    dig = rederive(d, r)
    return (f"{d.name}/composed.json: obligation for VU {r['selected_vus']['coordinates'][pair[0]]} = ({A['canonical_id'].split('/r=')[1][:8]}… t={A['canonical_id'].split('/t=')[1].split('/')[0]} "
            f"{A['op']} j={ja}) rewritten to column j={jb} (not selected; its W row is B's, already in the object); labels untouched; object re-derived h_O={dig['h_O'][:12]} h_L={dig['h_L'][:12]}; "
            f"proof stands in as the adversary's re-proof")


def edit_n4_declared_budget(rec: Path) -> str:
    """C4 / R8: the instance's declared B_ex set to the VU count the verifier actually uses."""
    p = rec / "protocol_instance.json"
    d = json.loads(p.read_text())
    n_ex = sum(1 for ln in (rec / "mapping.units.jsonl").read_text().splitlines() if ln.strip() and str(json.loads(ln)["class"]).startswith("exempt"))
    old = d["security"]["B_ex"]
    d["security"]["B_ex"] = n_ex
    p.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    return f"protocol_instance.json: security.B_ex {old} -> {n_ex} (= selected VUs of class exempt, the number reverify.py:438 uses as allowance)"


def edit_n5_input_check_trust(rec: Path) -> str:
    """C3 / R2: one selected input unit's committed word changed in obligations/input_checks.jsonl with its `ok` left true (a prover-written file)."""
    p = rec / "obligations" / "input_checks.jsonl"
    lines = p.read_text().splitlines()
    for i, ln in enumerate(lines):
        c = json.loads(ln)
        if c["kind"] == "weight" and c.get("expected") is not None:
            old = c["committed"]
            c["committed"] = old ^ 1
            lines[i] = json.dumps(c, sort_keys=True)
            p.write_text("\n".join(lines) + "\n")
            return f"input_checks.jsonl: VU {c['vu_index']} ({c['leaf']}[{c['word']}]) committed {old} -> {old ^ 1} while expected={c['expected']} and ok=true kept"
    raise RuntimeError("no weight input check")


def edit_n6_relabel_exempt(rec: Path, chunk: str = "chunk00") -> str:
    """C4 / R8 (post hoc exemption): one selected *provable* VU relabelled class 'exempt (charged)' in mapping.units.jsonl; its obligation and
    labels removed from the statement (a re-proving adversary); the declared B_ex stays 0."""
    units_p = rec / "mapping.units.jsonl"
    lines = units_p.read_text().splitlines()
    d, r = load_chunk(rec, chunk)
    vu = r["selected_vus"]["coordinates"][0]
    for i, ln in enumerate(lines):
        u = json.loads(ln)
        if u["vu_index"] == vu:
            old = u["class"]
            u["class"] = "exempt (charged)"
            lines[i] = json.dumps(u, sort_keys=True)
            break
    else:
        raise RuntimeError("VU not in mapping")
    units_p.write_text("\n".join(lines) + "\n")
    A = r["provenance"]["units"][0]
    obs = r["object"]["obligations"]
    idx = next(i for i, ob in enumerate(obs) if ob["target"]["id"] == f"{A['canonical_id']}/Gemm[{int(A['j'])}]")
    del obs[idx]
    del r["selected_vus"]["coordinates"][0]
    del r["provenance"]["units"][0]
    dig = rederive(d, r)
    return f"mapping.units.jsonl: VU {vu} class {old!r} -> 'exempt (charged)'; {chunk}: its obligation + labels removed, h_O={dig['h_O'][:12]} re-derived; declared B_ex still 0"


CASES: dict[str, tuple[Any, str, str]] = {
    # id: (edit, contract case, what a PASS/FAIL pattern shows)
    "R-B": (None, "baseline", "the mainline verifier's verdict on an unedited clone (recorded SP1 verdicts via the stand-in)"),
    "R-N1": (edit_n1_served_token, "C8/R4 served y vs the computation", "which check ties the served token to anything: only D6.request_commitment (served == boundary record); no relation"),
    "R-N2": (edit_n2_labels, "C7 duplicate coverage (labels)", "V5 counts prover-written labels: a duplicated label is caught (dup + missing)"),
    "R-N3": (edit_n3_graft, "C7 duplicate coverage (object)", "V5 never reads the object's obligations: a proof for an unselected VU + none for a selected one passes when the labels are right"),
    "R-N4": (edit_n4_declared_budget, "C4/R8 declared B_ex", "the declared budget is pinned by the instance digest but never read; the Bound uses the VU count either way"),
    "R-N5": (edit_n5_input_check_trust, "C3/R2 input agreement", "V6 trusts input_checks.jsonl's `ok` (prover pipeline); no recomputation from the checkpoint"),
    "R-N6": (edit_n6_relabel_exempt, "C4/R8 post hoc exemption", "a provable VU relabelled exempt after the fact is accepted and charged; no F ⊆ J, |F| <= declared B_ex check"),
}


# ------------------------------------------------------------------------------------------------------------------------------
# L1's twelve negatives (baseline), through the stand-in where the host is needed
# ------------------------------------------------------------------------------------------------------------------------------


def edit_r9c_mapping(rec: Path) -> str:
    """[R9 regression (c)] the prover's ``mapping.units.jsonl`` reinterprets one SELECTED provable unit: its located column ``j`` is moved to
    another column of the same Gemm (a consistent-looking decoy).  verifier_v3 derives the unit itself and must name the disagreement."""
    p = rec / "mapping.units.jsonl"
    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    i, u = next((i, u) for i, u in enumerate(rows) if str(u.get("class", "")).startswith("provable") and isinstance(u.get("locate"), dict) and "j" in u["locate"])
    old = dict(u["locate"])
    u["locate"]["j"] = int(old["j"]) + 1
    rows[i] = u
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f"mapping.units.jsonl: VU {u['vu_index']} ({u.get('kind')}) locate.j {old['j']} -> {u['locate']['j']} (all other fields and every other unit untouched; the record's proofs are the archive's)"


def install_contract_instance(rec: Path, inst_path: Path) -> str:
    """Replace the clone's legacy instance with a contract instance (epoch, B_ex as an RU count, epoch.inputs).  Side effect, stated: the archived
    chronology / protocol.json pin the LEGACY instance digest, so V1.instance and V2.chronology fail on every clone that carries it."""
    shutil.copyfile(inst_path, rec / "protocol_instance.json")
    return f"protocol_instance.json := contract instance {inst_path.name} (epoch.inputs = declared prompts; V1.instance / V2.chronology fail on the pinned legacy digest by construction)"


def edit_r1a_recommit_prompt(rec: Path) -> str:
    """[R1 regression (a), post hoc form] the committed prompt of one request differs from the declared x by one token and the boundary tree is
    rebuilt so the record authenticates against the re-committed root (commitments_phase0.json / commitments.json updated).  The served prompt
    (production.jsonl) is untouched.  Side effects, stated: V3.boundary / V3.interior (file digests vs the archived chronology) fail too."""
    from verity_vllm.capture import CaptureReader, boundary_tree_id
    from verity_vllm.commit import TreeCache

    sel = rec / "adapter" / "selective"
    c0 = json.loads((sel / "commitments_phase0.json").read_text())
    c2 = json.loads((sel / "commitments.json").read_text())
    rid = sorted(k.split("/", 1)[1] for k in c0["trees"] if k.startswith("boundary/"))[0]
    bpath = sel / "ru" / rid / "boundary.json"
    b = json.loads(bpath.read_text())
    b["prompt_token_ids"][0] = int(b["prompt_token_ids"][0]) ^ 1
    bpath.write_text(json.dumps(b, indent=1) + "\n")
    tree = TreeCache(CaptureReader(sel), recompute_leaves=True).tree(boundary_tree_id(rid))
    old = c0["trees"][f"boundary/{rid}"]["root"]
    for c in (c0, c2):
        c["trees"][f"boundary/{rid}"]["root"] = tree.root.hex()
        c["trees"][f"boundary/{rid}"]["leaf_count"] = tree.leaf_count
    (sel / "commitments_phase0.json").write_text(json.dumps(c0, indent=1, sort_keys=True) + "\n")
    (sel / "commitments.json").write_text(json.dumps(c2, indent=1, sort_keys=True) + "\n")
    return f"boundary record of {rid[:8]}: prompt_token_ids[0] changed by one bit and the boundary tree RE-COMMITTED (root {old[:12]} -> {tree.root.hex()[:12]}); served prompt untouched"


def _edit_instance(rec: Path, mutate) -> str:
    """[R13] mutate the (already installed, contract) protocol_instance.json in place; the verifier reads its shape pins from this file."""
    p = rec / "protocol_instance.json"
    doc = json.loads(p.read_text())
    what = mutate(doc)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return f"protocol_instance.json: {what}"


def edit_r13_multiplicity_9(rec: Path) -> str:
    def m(doc):
        d = doc["outputs"]["designated"][0]
        d["multiplicity"] = 9
        doc["outputs"]["W_out_bits"]["value"] = 9 * int(d["domain_bits_declared"])
        return "outputs.designated[token_slot].multiplicity 16 -> 9, W_out_bits 256 -> 144 (program identity still LP=32, STEPS=15: the circuit has 16 output slots)"
    return _edit_instance(rec, m)


def edit_r13_multiplicity_off_by_one(rec: Path) -> str:
    """[R13, any shape] outputs.designated[token_slot].multiplicity := STEPS (one short of the circuit's STEPS + 1 root output slots); W_out_bits follows."""
    def m(doc):
        d = doc["outputs"]["designated"][0]
        steps, old = int(doc["program"]["identity"]["STEPS"]), int(d["multiplicity"])
        d["multiplicity"] = steps
        doc["outputs"]["W_out_bits"]["value"] = steps * int(d["domain_bits_declared"])
        return f"outputs.designated[token_slot].multiplicity {old} -> {steps} (= STEPS; the circuit has STEPS + 1 = {steps + 1} slots), W_out_bits follows"
    return _edit_instance(rec, m)


def _append_prompt_token(b: dict, token: int = 1) -> list[int]:
    """One more prompt token in a ``verity-vllm/boundary/v1`` record, kept structurally consistent: output records shift one position, the
    prefill step covers the LP+1 prompt positions and each decode step its shifted position (what an honest prover would have committed)."""
    ids = list(b["prompt_token_ids"]) + [int(token)]
    b["prompt_token_ids"] = ids
    for o in b.get("output_records") or []:
        o["t"] = int(o["t"]) + 1
    for st in b.get("steps") or []:
        st["positions"] = list(range(len(ids))) if st.get("phase") == "prefill" else [int(x) + 1 for x in st["positions"]]
    return ids


def edit_r1_lp_plus_one(rec: Path) -> str:
    """[R1, integrator 22:30Z item 3, post hoc form] the declared prompt of one request has LP+1 tokens (prompt_len, prompt_sha256 of the LP+1 ids)
    and the prover's boundary record / boundary tree commit the same LP+1 tokens (consistent).  The layout pin still says LP.  The verifier's
    load_instance must refuse (prompt_len != LP) before any check runs."""
    from verity_vllm.capture import CaptureReader, boundary_tree_id
    from verity_vllm.commit import TreeCache

    import e2e.protocol_instance as pi

    sel = rec / "adapter" / "selective"
    c0 = json.loads((sel / "commitments_phase0.json").read_text())
    c2 = json.loads((sel / "commitments.json").read_text())
    rid = sorted(k.split("/", 1)[1] for k in c0["trees"] if k.startswith("boundary/"))[0]
    bpath = sel / "ru" / rid / "boundary.json"
    b = json.loads(bpath.read_text())
    ids = _append_prompt_token(b)
    bpath.write_text(json.dumps(b, indent=1) + "\n")
    tree = TreeCache(CaptureReader(sel), recompute_leaves=True).tree(boundary_tree_id(rid))
    for c in (c0, c2):
        c["trees"][f"boundary/{rid}"]["root"] = tree.root.hex()
        c["trees"][f"boundary/{rid}"]["leaf_count"] = tree.leaf_count
    (sel / "commitments_phase0.json").write_text(json.dumps(c0, indent=1, sort_keys=True) + "\n")
    (sel / "commitments.json").write_text(json.dumps(c2, indent=1, sort_keys=True) + "\n")

    def m(doc):
        pr = doc["epoch"]["inputs"]["per_request"][rid]
        pr["prompt_len"], pr["prompt_sha256"] = len(ids), pi.declared_input_digest(ids)
        return f"epoch.inputs.per_request[{rid[:8]}] prompt_len {len(ids) - 1} -> {len(ids)} (LP+1), prompt_sha256 of the LP+1 ids; layout pin still count={doc['epoch']['inputs']['layout']['prompt']['count']}"
    what = _edit_instance(rec, m)
    return f"{what}; boundary record of {rid[:8]} has the same LP+1 tokens and its tree is RE-COMMITTED (root {tree.root.hex()[:12]})"


def edit_r13_tiny_identity(rec: Path) -> str:
    def m(doc):
        ident = doc["program"]["identity"]
        ident["LP"], ident["STEPS"] = 4, 8
        ident["function"] = f"Serve_v1{{C={{{ident['config_sha12']}}},LP=4,STEPS=8}}"
        return "program.identity LP=32/STEPS=15 -> LP=4/STEPS=8 (TINY shape; outputs still declare 16 slots; program digest / gates pins untouched)"
    return _edit_instance(rec, m)


def edit_r13_foreign_elf(rec: Path) -> str:
    def m(doc):
        doc["epoch"]["approved_elf_sha256"] = ["00" * 32]
        return "epoch.approved_elf_sha256 := [0000…00] (an ELF outside e2e/approved_guests.json)"
    return _edit_instance(rec, m)


def edit_r13_foreign_family(rec: Path) -> str:
    def m(doc):
        doc["epoch"]["family"] = "veritor.tc-hopper-fp8@1"
        return "epoch.family := veritor.tc-hopper-fp8@1 (a family the approved ELF does not implement)"
    return _edit_instance(rec, m)


def edit_instance_partition_digest_kind(rec: Path) -> str:
    """[Integrator 23:41Z] the instance's partition.partition_digest_kind edited (record_v5's a9/legacy-minus-inputs/v1 -> the contract kind): the
    named loader (--instance-format record_v5) no longer matches the instance -> V8.instance_format / V8.instance_load (reverify), INSTANCE_REFUSED
    (driver), before anything is committed or drawn."""
    def m(doc):
        old = doc["partition"]["partition_digest_kind"]
        doc["partition"]["partition_digest_kind"] = "partition-intervals/v1" if old != "partition-intervals/v1" else "a9/legacy-minus-inputs/v1"
        return f"partition.partition_digest_kind {old!r} -> {doc['partition']['partition_digest_kind']!r} (the CLI's --instance-format no longer matches)"
    return _edit_instance(rec, m)


def edit_r13_real_family_not_listed(rec: Path) -> str:
    """[R13, integrator 22:30Z] a family that DOES resolve in the interface (veritor.elem-bf16@1, digest 0617…) but which the verifier's own
    approved_guests.json entry for the record's host ELF does not list: a verifier-side copy of approved_guests.json with that key removed from
    the entry (the verifier's file is read-only; `--approved` names the copy).  Distinguishes `check_family_approved`'s "does not implement
    family (lists [...])" from the width-32 resolution failure V3-R13-FAM exercises."""
    fam = "veritor.elem-bf16@1"
    src = POC / "e2e" / "approved_guests.json"
    doc = json.loads(src.read_text())
    hv = rec / "verify" / "host_verify.jsonl"
    if not hv.exists():
        hv = rec / "verify" / "reverify_v3" / "host_verify.jsonl"
    first = json.loads(hv.read_text().splitlines()[0])
    host_elf = str(first.get("elf_sha256") or (first.get("host_verify") or {}).get("elf_sha256") or "")
    removed = []
    for g in doc["guests"]:
        if fam in (g.get("families") or {}):
            del g["families"][fam]
            removed.append(g["elf_sha256"][:12])
    ap = rec.parent / f"{rec.name}.approved_guests.json"
    ap.write_text(json.dumps(doc, indent=1) + "\n")
    (rec / "a8_extra_args.json").write_text(json.dumps(["--approved", str(ap)]))

    def m(d):
        d["epoch"]["family"] = fam
        return f"epoch.family := {fam} (resolves at width 32); verifier-side approved_guests copy {ap.name}: {fam} removed from entries {removed} (host ELF {host_elf[:12] or '?'})"
    return _edit_instance(rec, m)


def _sorted_object(o: dict) -> None:
    """Re-sort the typed object's lists as `veritor.interface.validate` requires (leaves by id, references by ref, subcircuits by digest)."""
    o["leaves"].sort(key=lambda L: L["id"])
    o["references"].sort(key=lambda r: r["ref"])
    o["subcircuits"].sort(key=lambda s: s["digest"])


def _rank(rec: Path, c2: dict, tid: str, leaf_id: str) -> int:
    """The committed rank of ``leaf_id`` in tree ``tid``: its position in the committed id list sorted by ``identity.sort_key``
    (``verity-vllm/leaf/v1`` framing; the verifier's `expected.committed_trees` + A2's rank oracle)."""
    from verity_vllm import identity
    from verity_vllm.capture import CaptureReader
    from verity_vllm.commit import TreeCache

    from e2e.verifier_v3.expected import committed_trees

    sel = rec / "adapter" / "selective"
    ct = committed_trees(sel, c2, [tid], TreeCache(CaptureReader(sel), recompute_leaves=True))
    assert tid in ct.ids, ct.problems
    return sorted(ct.ids[tid], key=identity.sort_key).index(leaf_id)


def _swap_leaf(o: dict, old_id: str, new_id: str, new_rank: int) -> None:
    """Every mention of leaf ``old_id`` in the object (leaves, references, obligation boundaries / declared interfaces) now names ``new_id``."""
    for L in o["leaves"]:
        if L["id"] == old_id:
            L["id"], L["leaf"], L["position"] = new_id, new_rank, new_rank
    for r in o["references"]:
        if r["leaf"] == old_id:
            r["leaf"] = new_id
            r["ref"] = new_id + r["ref"][len(old_id):]
    for ob in o["obligations"]:
        for key in ("in", "out"):
            ob["boundary"][key] = [new_id + x[len(old_id):] if x.startswith(old_id + "[") else x for x in ob["boundary"][key]]
        di = ob.get("declared_interface")
        if di:
            di["args"] = [new_id + x[len(old_id):] if x.startswith(old_id + "[") else x for x in di["args"]]
            di["returns"] = [new_id + x[len(old_id):] if x.startswith(old_id + "[") else x for x in di["returns"]]


def _description_node(rec: Path, op: str) -> tuple[Path, dict, dict]:
    p = rec / "adapter" / "selective" / "description.json"
    doc = json.loads(p.read_text())
    return p, doc, next(n for n in doc["nodes"] if n["op"] == op)


R9_3B_OP, R9_3B_DECOY_LAYER = "layers.17.mlp.down_proj.gemm", 16  # REC-P1-B0 chunk00 covers two selected coordinates of this op (K = 1536)


def _pick_3b(rec: Path, chunk: str) -> tuple[str, int, int]:
    """(op, layer, decoy layer) for the 3b edits: the P1 constants when that op is in the chunk, else the first down_proj GemmCoordinate unit of
    the chunk (every layer's down_proj has the same K, so any other layer is a same-K decoy)."""
    _, r = load_chunk(rec, chunk)
    ops = {u["op"] for u in r["provenance"]["units"]}
    if R9_3B_OP in ops:
        return R9_3B_OP, 17, R9_3B_DECOY_LAYER
    u = next(u for u in r["provenance"]["units"] if u["op"].endswith(".mlp.down_proj.gemm"))
    layer = int(u["layer"])
    layers = sorted({int(v["layer"]) for v in r["provenance"]["units"] if v["op"].endswith(".mlp.down_proj.gemm")} | {layer + 1, layer - 1 if layer else layer + 2})
    decoy = next(l for l in layers if l != layer and l >= 0)
    return u["op"], layer, decoy


def edit_r9_3b_x_decoy(rec: Path, chunk: str = "chunk00") -> str:
    """[Integrator 21:05Z, 3b-i] The prover's description.json relabels the x operand of one selected GemmCoordinate op to a decoy row of the same K
    (layer 16's act_mul instead of layer 17's), and its proof is over the decoy row (a consistent, re-proving adversary: object leaves /
    references / boundaries renamed, digests re-derived, stand-in re-proof).  The verifier that derives x from the description follows the
    relabelling; one that derives x from its own program does not.  Every commitment is untouched."""
    OP, LAYER, DECOY = _pick_3b(rec, chunk)
    p, doc, node = _description_node(rec, OP)
    xarg = node["arg"] if "arg" in node else next(a for a in node["args"] if a["name"] == "x")
    old_op = xarg["op"]
    xarg["op"] = f"layers.{DECOY}.mlp.act_fn.act_mul"
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    d, r = load_chunk(rec, chunk)
    c2 = json.loads((rec / "adapter" / "selective" / "commitments.json").read_text())
    o = r["object"]
    n = 0
    for ob in list(o["obligations"]):
        if not ob["boundary"]["out"] or f"/op={OP}/" not in ob["boundary"]["out"][0]:
            continue
        old_x = ob["boundary"]["in"][0].rsplit("[", 1)[0]
        assert f"/op={old_op}/" in old_x, old_x
        new_x = old_x.replace(f"/l={LAYER}/op={old_op}/", f"/l={DECOY}/op={xarg['op']}/")
        assert new_x != old_x, old_x
        tid = next(t for t in ob["roots"] if t.startswith("ru/"))
        _swap_leaf(o, old_x, new_x, _rank(rec, c2, tid, new_x))
        n += 1
    assert n, "no obligation of the op in this chunk"
    _sorted_object(o)
    dig = rederive(d, r)
    return (f"description.json: {OP}.args[x].op {old_op} -> {xarg['op']} (decoy row, same K); {chunk}: {n} obligation(s) of that op re-pointed at the decoy "
            f"x rows (leaves/references/boundaries), h_O={dig['h_O'][:12]} h_L={dig['h_L'][:12]}; proof stands in as the adversary's re-proof; commitments untouched")


def edit_r9_3b_w_decoy(rec: Path, chunk: str = "chunk00") -> str:
    """[Integrator 21:05Z, 3b-ii] The description names another registered weight tensor of the same K for the selected op
    (layers.16.mlp.down_proj.weight for layers.17's), and the proof opens that tensor's rows.  Same protocol as 3b-i."""
    from verity_vllm import identity
    from verity_vllm.capture import CaptureReader

    OP, LAYER, DECOY = _pick_3b(rec, chunk)
    p, doc, node = _description_node(rec, OP)
    warg = next(a for a in node["args"] if a["name"] == "w")
    old_param, new_param = warg["param"], f"layers.{DECOY}.mlp.down_proj.weight"
    warg["param"] = new_param
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    d, r = load_chunk(rec, chunk)
    c2 = json.loads((rec / "adapter" / "selective" / "commitments.json").read_text())
    reader = CaptureReader(rec / "adapter" / "selective")
    o = r["object"]
    n = 0
    for ob in list(o["obligations"]):
        if not ob["boundary"]["out"] or f"/op={OP}/" not in ob["boundary"]["out"][0]:
            continue
        j = int(ob["boundary"]["out"][0].rsplit("[", 1)[1][:-1])
        old_w = ob["boundary"]["in"][-1].rsplit("[", 2)[0]
        assert old_w == f"w={old_param}/c={j}", (old_w, j)
        new_w = reader.weights.leaf_id_for_row(str(identity.weight_id(new_param, None)), j)
        assert new_w != old_w and new_w.endswith(f"/c={j}"), new_w
        _swap_leaf(o, old_w, new_w, _rank(rec, c2, "weights", new_w))
        n += 1
    assert n
    _sorted_object(o)
    dig = rederive(d, r)
    return (f"description.json: {OP}.args[w].param {old_param} -> {new_param} (another registered tensor, same K); {chunk}: {n} obligation(s) re-pointed at "
            f"that tensor's rows, h_O={dig['h_O'][:12]} h_L={dig['h_L'][:12]}; proof stands in as the adversary's re-proof; commitments untouched")


def edit_r9_3c_input_check_rewrite(rec: Path, chunk: str = "chunk00") -> str:
    """[Integrator 21:05Z, 3c] One selected Input-equation row of obligations/input_checks.jsonl: `committed` and `expected` both set to a wrong
    word (the prover 'committed' a wrong value and wrote expected := committed, ok: true); the InputEq obligation in the chunk carries the same
    wrong value (re-proving adversary).  The weights tree is untouched, so a verifier that recomputes the word from the pinned checkpoint or opens
    the committed leaf itself refuses; one that reads `ok` / `expected` from the file does not."""
    from veritor.interface import input_equation

    d, r = load_chunk(rec, chunk)
    o = r["object"]
    ob = next(x for x in o["obligations"] if not x["boundary"]["in"] and x["target"]["id"].endswith("/InputEq"))
    leaf, rest = ob["boundary"]["out"][0].rsplit("[", 2)[0], ob["boundary"]["out"][0]
    word = int(rest.rsplit("[", 1)[1][:-1])
    p = rec / "obligations" / "input_checks.jsonl"
    lines = p.read_text().splitlines()
    hit = None
    for i, ln in enumerate(lines):
        c = json.loads(ln)
        if c["leaf"] == leaf and int(c["word"]) == word:
            assert c["committed"] == c["expected"] and c["ok"]
            old = int(c["committed"])
            c["committed"] = c["expected"] = old ^ 1
            lines[i] = json.dumps(c, sort_keys=True)
            hit = c
            break
    assert hit is not None, (leaf, word)
    p.write_text("\n".join(lines) + "\n")
    old_sc = next(s for s in o["subcircuits"] if s["digest"] == ob["subcircuit"])
    assert int(old_sc["gates"][0]["value"], 16) == old
    new_sc = input_equation(old ^ 1, 16)
    new_doc = {"definition": old_sc["definition"], "digest": new_sc.digest, "gates": [{"op": "Input", "value": f"{old ^ 1:04x}", "width": 16}], "in": [], "out": [0]}
    ob["subcircuit"] = new_sc.digest
    if not any(x["subcircuit"] == old_sc["digest"] for x in o["obligations"]):
        o["subcircuits"] = [s for s in o["subcircuits"] if s["digest"] != old_sc["digest"]]
    if not any(s["digest"] == new_sc.digest for s in o["subcircuits"]):
        o["subcircuits"].append(new_doc)
    _sorted_object(o)
    dig = rederive(d, r)
    return (f"input_checks.jsonl: VU {hit['vu_index']} ({leaf}[{word}]) committed = expected := {old} ^ 1 = {old ^ 1}, ok: true; {chunk}: its InputEq obligation "
            f"re-pointed at value {old ^ 1:#06x} (subcircuit {new_sc.digest[:12]}), h_O={dig['h_O'][:12]} h_L={dig['h_L'][:12]}; weights tree untouched (the real word is {old})")


V3_CASES: dict[str, tuple[Any, str, str]] = {
    # [Integrator 21:05Z] two more R9 regressions from A2's mapping-sources audit: prover-written files still USED by V5 / V6
    "V3-R9-3bx": (edit_r9_3b_x_decoy, "R9 (3b-i): description.json relabels a selected GemmCoordinate unit's x row to a decoy row of the same K; proof over the decoy",
                  "PASS = EA (the verifier derives x from the prover's description); owner A2 (tentative check V5.description)"),
    "V3-R9-3bw": (edit_r9_3b_w_decoy, "R9 (3b-ii): description.json points the selected op at another registered weight tensor of the same K; proof opens that tensor",
                  "PASS = EA; owner A2"),
    "V3-R9-3c": (edit_r9_3c_input_check_rewrite, "R9 (3c): input_checks.jsonl row's expected := (wrong) committed word, ok: true; InputEq proof carries the same value",
                 "PASS = EA (V6 trusts ok; V5 derives the Input expectation from the file's expected); owner A2"),
    # [R13] shape / guest pins, on the SMALL (32x15 = 16 out) archive with the contract instance installed
    "V3-R13-M9": (edit_r13_multiplicity_9, "R13: declared multiplicity 9 for a 32x15 circuit (16 slots); record serves 16", "does the verifier compare the declared slot count with the served outputs (V7) and with the circuit's output count?"),
    "V3-R13-TINYID": (edit_r13_tiny_identity, "R13: program identity LP=4/STEPS=8 (TINY) on a 32x15 record", "the verifier builds the declared identity in its own process: digest != the pinned program (V1.program_rebuilt) -- STEPS comes from the circuit, not the label"),
    "V3-R13-ELF": (edit_r13_foreign_elf, "R13: instance pins an ELF outside approved_guests.json", "V4.guest refuses: the host's guest is not in the instance's approved set"),
    "V3-R13-FAM2": (edit_r13_real_family_not_listed, "R13: instance pins a real family (elem-bf16@1) the verifier's approved entry for the host ELF does not list", "V4.guest must name 'does not implement family ... (lists [...])' from approved_guests.json#families, not a resolution error"),
    "V3-R13-FAM": (edit_r13_foreign_family, "R13: instance pins a family the approved ELF does not implement", "which check names the family disagreement"),
    # id: (edit, contract case, what the outcome shows) -- run with --verifier v3 (e2e.verifier_v3.reverify) on a SLICED clone
    "V3-B": (None, "baseline (legacy archive under the contract verifier)", "the failing set S0 every other row is compared against"),
    "V3-BI": (None, "baseline with the contract instance installed (epoch.inputs = the served prompts)", "the failing set S0' for V3-R1a: V1.instance / V2.chronology fail on the legacy pin, V8.step2 passes"),
    "V3-R1a": (edit_r1a_recommit_prompt, "R1 (a): committed prompt != declared x, agreed request fixed", "with a contract instance carrying epoch.inputs, V8.step2 names the request and 'committed prompt != declared x'"),
    "V3-R9b": (edit_n3_graft, "R9 (b): valid proof for a different unit than the one selected", "V5.coverage: the selected VU is uncovered and the grafted obligation is unexpected -- labels no longer decide"),
    "V3-R9c": (edit_r9c_mapping, "R9 (c): prover's mapping.units.jsonl reinterprets the selected unit", "V5.prover_mapping names the disagreement (derivation used regardless)"),
}


# [Integrator 21:58Z] the same regressions on A9's first fresh TINY record (V5-TINY-bex0-inc: live driver + vLLM, one chunk, baseline REFUSED
# S = {V5.uncovered}); every tamper must add its named FAIL to S.  Run with --record-name V5-TINY-bex0-inc --population <rec>/population.v5.json
# --expressibility <rec>/expressibility_v5.json (no --instance: the record's own contract instance is the pinned one).
T_CASES: dict[str, tuple[Any, str, str]] = {
    "T-B": (None, "TINY baseline (untouched clone)", "S = {V5.uncovered} (653 selected VUs of kinds without a checker; B_ex = 0)"),
    "T-R1a": (edit_r1a_recommit_prompt, "R1 (a): committed prompt != declared x (epoch.inputs fixed), tree re-committed", "V8.step2 names the request; V3.boundary (chronology digest)"),
    "T-R9b": (edit_n3_graft, "R9 (b): valid proof for a different unit than the one selected (same-layer graft in chunk00)", "V5.coverage / V5.positions"),
    "T-R9c": (edit_r9c_mapping, "R9 (c): mapping.units.jsonl reinterprets a selected unit (locate.j + 1)", "V5.prover_mapping names the disagreement"),
    "T-R9-3bx": (edit_r9_3b_x_decoy, "R9 (3b-i): description.json x -> decoy row of another layer, proof over the decoy", "V5.expected (V5.description)"),
    "T-R9-3bw": (edit_r9_3b_w_decoy, "R9 (3b-ii): description.json w -> another layer's down_proj tensor, proof opens it", "V5.expected (V5.description)"),
    "T-R13-M8": (edit_r13_multiplicity_off_by_one, "R13: declared multiplicity 9 -> 8 (= STEPS) on the 4x8 circuit (9 slots)", "V7.steps_vs_outputs (+ V7.output_domain / V8.step2 / D6 on the served 9)"),
    "T-R13-FAM": (edit_r13_foreign_family, "R13: epoch.family := a family not in the interface", "V4.guest (resolution branch)"),
    "T-R13-FAM2": (edit_r13_real_family_not_listed, "R13: epoch.family := elem-bf16@1, removed from the host ELF's approved entry (verifier-side copy)", "V4.guest 'does not implement family ... (lists ...)'"),
    "T-R1-LP": (edit_r1_lp_plus_one, "R1: declared prompt LP+1 tokens, committed consistently", "refused at load_instance (prompt_len != LP) before any check"),
}

# ------------------------------------------------------------------------------------------------------------------------------
# [Integrator 22:52Z, segment 3] CK-ELEM dispatch negatives on the TINY record: elementwise statements built by CK-ELEM's LOCAL path
# (e2e.coverage_v1.ck_elem: resolve / build_many / verify_transparent) for the selected units the ONE registry can build, staged once
# (--stage-elem, ≈ 2.8 GB) and installed into clones as extra obligation chunks (one per kind).  LABEL: no SP1 proof exists for them; the
# stand-in host's marker carries the TRANSPARENT verdict (the guest's Python mirror at the pinned gate-set digest) as the public verdict byte.
# reverify itself accepts a chunk for coverage only through a host-verified compressed proof (reverify.py: V4.proof.<chunk> -> proven ->
# accepted); there is no transparent path in the verifier -- that is what "covered" means, and every E-* row says so.
# ------------------------------------------------------------------------------------------------------------------------------

ELEM_STAGE: Path | None = None  # set from --elem-stage; the E-* edits read the staged chunks from here
ELEM_SLUG = {"SiluMulBf16_v1|batch": "silu", "Bf16AddF2fp_v1|batch": "add", "RopeOut_v1|batch": "rope", "RopeOutAdd_v1|batch": "ropeadd"}


def _locate_all(rec: Path, expr_path: Path):
    """The verifier's derivation as reverify makes it: program build (V1.program_rebuilt), ProgramLocator, locate_selected over selected.json."""
    from e2e.coverage_v1 import ck_elem as CE
    from e2e.verifier_v3 import locator as vloc

    inst = json.loads((rec / "protocol_instance.json").read_text())
    program, C, problems = vloc.build_pinned_program(inst["program"]["identity"], expected_digest=inst["program"]["digest"], expected_gates=inst["program"]["gates"]["value"])
    assert program is not None and not problems, problems
    loc = vloc.ProgramLocator(program, C)
    handle = CE.RecordHandle.open(rec / "adapter" / "selective", program, C, loc)
    sel = json.loads((rec / "chronology" / "selected.json").read_text())
    expr = json.loads(expr_path.read_text())
    klass = {r["kind"]: r["class"] for r in expr["rows"]}
    ru_req = {int(r["ru"]): r["request_id"] for r in json.loads((rec / "ru_map.json").read_text())["rus"]}
    units = vloc.locate_selected(loc, sel["selected"], klass=klass, ru_request=ru_req, n_v=int(expr["vus_per_request"]))
    return handle, units


def _stage_chunk(d: Path, built, verdict: str, units: list[dict], note: str, *, witness_hex: str | None = None) -> dict:
    """One staged chunk: composed.json (the builder's record; statement_digest := sha256 of the format-3 statement bytes, as rec_p1_prove writes it),
    statement.bin, a placeholder proof.bin, and the stand-in marker whose verdict byte is the transparent verdict."""
    import hashlib as _h

    d.mkdir(parents=True, exist_ok=True)
    record = dict(built.record)
    record["statement_v2_digest"] = record["statement_digest"]
    record["statement_digest"] = record["statement_v3_digest"] = _h.sha256(built.statement_bytes).hexdigest()
    if witness_hex is not None:
        record["witness"] = witness_hex
    record["verdict"] = "accept" if verdict == "accept" else "reject"
    record["a8_note"] = note
    dump(d / "composed.json", record)
    (d / "statement.bin").write_bytes(built.statement_bytes)
    (d / "proof.bin").write_bytes(b"A8 stand-in: no SP1 proof exists for this statement (transparent verdict only)\n" + bytes.fromhex(record["statement_digest"]))
    ok = verdict == "accept"
    dump(d / "marker.json", {"public_values": record["digest"] + record["statement_digest"] + ("01" if ok else "00"), "verdict": ok,
                             "backend": f"stand-in:transparent (CK-ELEM statement; verdict = the guest's Python mirror at the pinned gate-set digest: {verdict}; not SP1)",
                             "note": note})
    dump(d / "units.json", {"units": units, "transparent_verdict": verdict, "obligations": len(built.obj.obligations), "family": built.family, "note": note,
                            "h_O": record["digest"], "h_L": record["statement_digest"]})
    return {"dir": d.name, "units": len(units), "obligations": len(built.obj.obligations), "family": built.family, "transparent": verdict, "h_L": record["statement_digest"][:12]}


def stage_elem_statements(rec: Path, expr_path: Path, stage: Path) -> dict[str, Any]:
    """Build the elementwise statements for every selected unit with a registered builder (one chunk per kind) plus two variants of the SiluMul
    chunk: `silu-neighbour` (its first unit replaced by the NON-selected next word of the same batch: a true statement about an unselected unit)
    and `silu-flipped` (the honest statement with one output word of the witness flipped: the transparent verdict is a rejection)."""
    import dataclasses

    import numpy as np

    from e2e.coverage_v1 import ck_elem as CE
    from veritor.protocol.proofs.packed import encode_packed_witness

    handle, units = _locate_all(rec, expr_path)
    buildable = [u for u in units if isinstance(u.get("locate"), dict) and "builder" in u["locate"]]
    selected_gates = {(u["request_id"], int(u["gate_interval"][0])) for u in units}
    by_kind: dict[str, list[dict]] = {}
    for u in buildable:
        by_kind.setdefault(u["kind"], []).append(u)
    stage.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"utc": utc(), "record": str(rec), "selected": len(units), "buildable": len(buildable), "by_kind": {k: len(v) for k, v in by_kind.items()}, "chunks": [], "refusals": []}
    silu_built = silu_units = None
    for kind, us in sorted(by_kind.items()):
        resolved = []
        for u in us:
            try:
                resolved.append(CE.resolve(u, handle))
            except CE.Refusal as e:
                summary["refusals"].append({"vu": u["vu_index"], "kind": kind, "reason": str(e)[:200]})
        built = CE.build_many(resolved, handle, name=f"ck-elem/{kind}")
        verdict = CE.verify_transparent(built)
        summary["chunks"].append(_stage_chunk(stage / ELEM_SLUG[kind], built, verdict, us, f"CK-ELEM {kind}: {len(resolved)} selected units, transparent verdict {verdict}"))
        if kind == "SiluMulBf16_v1|batch":
            silu_built, silu_units = built, us
    if silu_built is None or not silu_units:
        # [E9, NIGHT D31.5 -- skip BY NAME, never an assertion] a record whose Q holds no SiluMul unit cannot express the neighbour / flipped
        # SiluMul negatives; the other CK-ELEM chunks are staged, the two are named as not expressible (the gate row reads NOT DONE by this name)
        reason = f"NOT_EXPRESSIBLE: no SiluMulBf16_v1|batch unit selected in this record (kinds in Q: {sorted(by_kind)})"
        summary["skipped"] = {"silu-neighbour": reason, "silu-flipped": reason}
        dump(stage / "summary.json", summary)
        return summary
    # (a) neighbour: the first SiluMul unit's statement replaced by the next word of the same batch, which is NOT a selected VU
    u0 = silu_units[0]
    lo, hi = u0["gate_interval"]
    step = next(s for s in (1, -1, 2, -2) if (u0["request_id"], lo + s) not in selected_gates)
    nb = dict(u0, gate_interval=[lo + step, hi + step])
    resolved = [CE.resolve(nb, handle)] + [CE.resolve(u, handle) for u in silu_units[1:]]
    built_n = CE.build_many(resolved, handle, name="ck-elem/SiluMul+neighbour")
    verdict_n = CE.verify_transparent(built_n)
    summary["chunks"].append(_stage_chunk(stage / "silu-neighbour", built_n, verdict_n, [nb] + silu_units[1:],
                                          f"VU {u0['vu_index']} (gate {lo}) replaced by its non-selected neighbour gate {lo + step} ({resolved[0].output.ref}); transparent verdict {verdict_n} (a true statement about an unselected unit)"))
    # (c) flipped: the honest SiluMul statement, one output word of the witness ^= 1 (the committed row is untouched: the opening no longer authenticates)
    u = silu_built.units[0]
    o = silu_built.openings[u.output.leaf]
    words = np.frombuffer(o.value, dtype="<u2").copy()
    words[u.output.flat] ^= 0x0001
    forged = dict(silu_built.openings)
    forged[u.output.leaf] = o.replace(value=words.tobytes())
    w_bytes = encode_packed_witness(CE._packed_witness(forged.values(), silu_built.statement))
    built_f = dataclasses.replace(silu_built, witness_bytes=w_bytes)
    verdict_f = CE.verify_transparent(built_f)
    summary["chunks"].append(_stage_chunk(stage / "silu-flipped", built_f, verdict_f, silu_units, f"output word {u.output.ref} of VU {u.vu_index} flipped in the witness (statement and commitments untouched); transparent verdict {verdict_f}",
                                          witness_hex=w_bytes.hex()))
    summary["neighbour"] = {"vu": u0["vu_index"], "gate": lo, "neighbour_gate": lo + step, "neighbour_selected": False}
    summary["flipped"] = {"vu": u.vu_index, "word": u.output.ref, "transparent": verdict_f}
    dump(stage / "summary.json", summary)
    return summary


def install_elem_chunks(rec: Path, names: list[str], *, proofs: bool = True) -> str:
    """Install staged chunks as obligations/chunk01.. with proofs/<chunk>-compressed.bin (+ .json, + the stand-in marker), in the given order.
    ``proofs=False``: the statements only (what a prover holding transparent verdicts and no SP1 proof can put in the record)."""
    assert ELEM_STAGE is not None and (ELEM_STAGE / "summary.json").exists(), "run --stage-elem first (or pass --elem-stage)"
    existing = sorted(p.name for p in (rec / "obligations").iterdir() if p.is_dir() and p.name.startswith("chunk"))
    n0 = max(int(n[5:]) for n in existing) + 1
    out = []
    for i, name in enumerate(names):
        src = ELEM_STAGE / name
        cname = f"chunk{n0 + i:02d}"
        d = rec / "obligations" / cname
        d.mkdir()
        shutil.copyfile(src / "composed.json", d / "composed.json")
        shutil.copyfile(src / "statement.bin", d / "statement.bin")
        m = json.loads((src / "marker.json").read_text())
        u = json.loads((src / "units.json").read_text())
        if not proofs:
            out.append(f"{cname}={name}({u['obligations']} obl, transparent {u['transparent_verdict']}, NO proof)")
            continue
        pf = rec / "proofs" / f"{cname}-compressed.bin"
        shutil.copyfile(src / "proof.bin", pf)
        dump(rec / "proofs" / f"{cname}.json", {"chunk": cname, "h_O": u["h_O"], "h_L": u["h_L"], "obligations": u["obligations"], "statement_format": 3,
                                                 "status": "A8 STAND-IN: no SP1 proof; the host stand-in returns the transparent verdict"})
        dump(Path(str(pf) + ".reproved.json"), m)
        out.append(f"{cname}={name}({u['obligations']} obl, transparent {u['transparent_verdict']})")
    return "; ".join(out)


ELEM_ALL = ["silu", "add", "rope", "ropeadd"]


def edit_e_baseline(rec: Path) -> str:
    return "all four elementwise chunks installed: " + install_elem_chunks(rec, ELEM_ALL)


def edit_e_statements_only(rec: Path) -> str:
    """(e, literal) the transparent-verified statements in the record WITHOUT any proof object: what "covered" requires is then visible.
    Found on the way: when the LAST chunk directory has no proof file, reverify.py:448 reads a stale `pf` (the proof loop's variable) and dies
    with FileNotFoundError instead of a named V4.proof FAIL (E-e0, kept as the finding); E-e1 puts a proof-carrying chunk last so the named
    outcome for the proof-less chunks is visible."""
    return "all four elementwise chunks installed as statements only (no proofs/<chunk>-compressed.bin): " + install_elem_chunks(rec, ELEM_ALL, proofs=False)


def edit_e_statements_only_named(rec: Path) -> str:
    a = install_elem_chunks(rec, ["silu", "add", "rope"], proofs=False)
    b = install_elem_chunks(rec, ["ropeadd"], proofs=True)
    return "three chunks as statements only, the fourth with its (stand-in) proof last: " + a + "; " + b


class NotExpressible(RuntimeError):
    """[E9, D31.5] an alteration the staged record cannot express, named (the gate reports NOT DONE with this name and reason)."""


def _staged(key: str) -> dict:
    doc = json.loads((ELEM_STAGE / "summary.json").read_text())
    if key not in doc:
        raise NotExpressible((doc.get("skipped") or {}).get({"neighbour": "silu-neighbour", "flipped": "silu-flipped"}.get(key, key), f"stage has no {key!r}"))
    return doc[key]


def edit_e_neighbour(rec: Path) -> str:
    s = _staged("neighbour")
    return f"(a) SiluMul chunk := statement for the non-selected neighbour of VU {s['vu']} (gate {s['gate']} -> {s['neighbour_gate']}): " + install_elem_chunks(rec, ["silu-neighbour", "add", "rope", "ropeadd"])


def edit_e_omit_family(rec: Path) -> str:
    s = json.loads((ELEM_STAGE / "summary.json").read_text())["by_kind"]
    return f"(b) RopeOut chunk omitted ({s.get('RopeOut_v1|batch')} selected units of that kind), the other three present: " + install_elem_chunks(rec, ["silu", "add", "ropeadd"])


def edit_e_flipped(rec: Path) -> str:
    s = _staged("flipped")
    return f"(c) SiluMul chunk := witness with output word {s['word']} of VU {s['vu']} flipped (transparent verdict {s['transparent']}): " + install_elem_chunks(rec, ["silu-flipped", "add", "rope", "ropeadd"])


def edit_e_family_unlisted(rec: Path) -> str:
    """(d) the statements' family (veritor.elem-bf16@1) removed from the host ELF's entry in a verifier-side approved_guests.json copy; the
    instance's epoch.family (tc-ampere-bf16@2) is untouched, so V4.guest is unaffected and the object-family check must name the chunk."""
    fam = "veritor.elem-bf16@1"
    doc = json.loads((POC / "e2e" / "approved_guests.json").read_text())
    removed = [g["elf_sha256"][:12] for g in doc["guests"] if (g.get("families") or {}).pop(fam, None) is not None]
    ap = rec.parent / f"{rec.name}.approved_guests.json"
    ap.write_text(json.dumps(doc, indent=1) + "\n")
    (rec / "a8_extra_args.json").write_text(json.dumps(["--approved", str(ap)]))
    return f"(d) verifier-side approved_guests copy: {fam} removed from entries {removed}; " + install_elem_chunks(rec, ELEM_ALL)


def edit_e_mapping_nonnull_wrong(rec: Path) -> str:
    """V5.prover_mapping rule change (prover `locate: null` on a table-unsupported unit tolerated): a NON-null wrong locate on such a unit must
    still refuse.  The prover's row for the first SiluMul unit gets the verifier's own shape with the neighbour's gate."""
    s = _staged("neighbour")
    p = rec / "mapping.units.jsonl"
    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    i, u = next((i, u) for i, u in enumerate(rows) if int(u["vu_index"]) == int(s["vu"]))
    old = u.get("locate")
    u["locate"] = {"builder": u["kind"], "gate": int(s["neighbour_gate"]), "family": "veritor.elem-bf16@1"}
    rows[i] = u
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f"mapping.units.jsonl: VU {u['vu_index']} ({u['kind']}, table class {u.get('class')!r}) locate {old!r} -> {u['locate']!r} (non-null, wrong gate); " + install_elem_chunks(rec, ELEM_ALL)


E_CASES: dict[str, tuple[Any, str, str]] = {
    # ids differ in more than case: the checkout sits on a case-insensitive APFS volume, so E-B / E-b would share one output dir
    "E-e": (edit_e_baseline, "(e) all buildable selected elementwise units present as transparent-verified statements (4 chunks, stand-in verdicts)", "does V5.coverage reach covered = 418, uncovered = 0 (V5.uncovered = 517 stays)?"),
    "E-e0": (edit_e_statements_only, "(e, literal) the transparent-verified statements present, no SP1 proof object for them", "does reverify count a transparent-verified statement as covered? (expected: no -- V4.proof.<chunk> 'no compressed proof', covered stays 282)"),
    "E-e1": (edit_e_statements_only_named, "(e, literal) three kinds as statements only, one with a proof object (last)", "V4.proof.chunk01-03 FAIL 'no compressed proof'; V5.coverage covered = 282 + 12: a transparent verdict is not coverage"),
    "E-a": (edit_e_neighbour, "(a) a statement for a neighbouring, non-selected SiluMul unit in place of the selected one", "V5.positions / V5.coverage name the selected VU uncovered and the neighbour unexpected"),
    "E-b": (edit_e_omit_family, "(b) one kind's chunk omitted, the other three present", "V5.coverage uncovered == that kind's unit count; covered == the other kinds' units (no duplicate hides it)"),
    "E-c": (edit_e_flipped, "(c) a SiluMul statement whose committed output word is flipped in the witness", "transparent verdict reject -> the stand-in's verdict byte is 0 -> V4.proof.<chunk> FAIL; its units uncovered"),
    "E-d": (edit_e_family_unlisted, "(d) statements of a family the host ELF's approved entry does not list (verifier-side copy)", "V4.family.<chunk> FAIL for the two elem-bf16 chunks; V5.coverage uncovered == their units; V4.guest (instance family) unaffected"),
    "E-R9c2": (edit_e_mapping_nonnull_wrong, "R9 (c) under the prover_mapping rule change: NON-null wrong locate on a table-unsupported (builder) unit", "V5.prover_mapping must still FAIL (only a null locate is tolerated)"),
}

NEEDS_INSTANCE = {"V3-BI", "V3-R1a", "V3-R9-3bx", "V3-R9-3bw", "V3-R9-3c", "V3-R13-M9", "V3-R13-TINYID", "V3-R13-ELF", "V3-R13-FAM", "V3-R13-FAM2"}  # the rows whose check needs epoch.inputs (the legacy archive instance declares none)


def l1_negatives(rec: Path, pop: Path, stub: Path, scratch: Path, run_tampers: bool) -> list[dict[str, Any]]:
    """L1's G1-G9 / T1-T3 (rec_p1_negatives.py), same tampers and gates; G2 exact (statement_match is recomputed), G3 and the guest verdicts of
    T1-T3 are `not checked` (no SP1 here)."""
    from veritor.interface import TypedObligationSet, check_binding

    from e2e import partition_sample as ps

    rows: list[dict[str, Any]] = []

    def row(nid, tamper, gate, refused, observed, seconds, sp1: str) -> None:
        rows.append({"id": nid, "tamper": tamper, "gate": gate, "refused": bool(refused), "observed": observed[:400], "seconds": round(seconds, 2), "sp1": sp1})
        print(f"[contract_negatives] {nid} {'REFUSED' if refused else 'NOT REFUSED (!!)'}: {gate}: {observed[:140]}", flush=True)

    def host(args: list[str]) -> tuple[int, dict | None]:
        res = subprocess.run([str(stub), *args], capture_output=True, text=True)
        doc = None
        for line in reversed(res.stdout.strip().splitlines()):
            if line.strip().startswith("{"):
                doc = json.loads(line)
                break
        return res.returncode, doc

    chunks = sorted(p for p in (rec / "obligations").iterdir() if p.is_dir() and p.name.startswith("chunk"))
    chunk = chunks[0]
    proof = rec / "proofs" / f"{chunk.name}-compressed.bin"
    record = json.loads((chunk / "composed.json").read_text())
    obj = TypedObligationSet.from_json(record["object"])
    fmt = int(record.get("statement_format", 2))
    _, info = host(["info"])
    _, good = host(["verify", "--proof", str(proof), "--statement", str(chunk / "statement.bin")])
    pv_good = bytes.fromhex(good["public_values"])

    t = time.perf_counter()
    approved = json.loads((POC / "e2e" / "approved_guests.json").read_text())
    gmap = {g["elf_sha256"]: dict(g) for g in approved["guests"]}
    gmap[info["elf_sha256"]]["vk_hash"] = "0x" + "00" * 32
    row("G1", "approved_guests.json copy: approved vk_hash altered", "V4.guest", str(gmap[info["elf_sha256"]]["vk_hash"]).lower() != str(info["vk_hash"]).lower(),
        f"host vk {info['vk_hash'][:18]} vs table 0x0000… -> refuse (reverify.py:258)", time.perf_counter() - t, "n/a (table comparison)")

    def flip(src: Path, dst: Path, off: int) -> None:
        b = bytearray(src.read_bytes())
        b[off] ^= 1
        dst.write_bytes(bytes(b))

    t = time.perf_counter()
    st_bad = scratch / "statement-flipped.bin"
    flip(chunk / "statement.bin", st_bad, 100)
    rc, vr = host(["verify", "--proof", str(proof), "--statement", str(st_bad)])
    row("G2", "statement.bin byte 100 flipped", "host verify statement_match (sha256(statement) == public h_L)", rc != 0 or not (vr or {}).get("statement_match"),
        f"rc={rc} ok={(vr or {}).get('ok')} statement_match={(vr or {}).get('statement_match')}", time.perf_counter() - t, "exact: statement_match is recomputed as the host does")

    t = time.perf_counter()
    pf_bad = scratch / "proof-flipped.bin"
    flip(proof, pf_bad, 4096)
    rc, vr = host(["verify", "--proof", str(pf_bad), "--statement", str(chunk / "statement.bin")])
    row("G3", "proof byte 4096 flipped", "host verify: SP1 proof invalid under the approved vk", rc != 0 or not (vr or {}).get("ok"),
        f"rc={rc} ok={(vr or {}).get('ok')} ({(vr or {}).get('error', '')[:80]})", time.perf_counter() - t, "NOT CHECKED: refused by the stand-in (unknown proof sha), not by SP1")

    t = time.perf_counter()
    b = check_binding(obj, pv_good[:64] + bytes([0]), statement_format=fmt)
    row("G4", "public verdict byte forced to 0", "check_binding ok AND verdict == 1 (reverify.py:310)", not (b.ok and b.verdict == 1), f"binds={b.ok} verdict={b.verdict}", time.perf_counter() - t, "n/a (binding is Python)")

    t = time.perf_counter()
    other = json.loads((chunks[1] / "composed.json").read_text())
    b = check_binding(TypedObligationSet.from_json(other["object"]), pv_good, statement_format=fmt)
    row("G5", "chunk00 proof for chunk01's object", "check_binding h_O || h_L", not b.ok, f"ok={b.ok} code={getattr(b, 'code', '')}", time.perf_counter() - t, "n/a")

    N = json.loads((rec / "protocol.json").read_text())["instances"]

    def chron_copy(name: str) -> Path:
        d = scratch / name
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(rec / "chronology", d)
        return d

    t = time.perf_counter()
    d = chron_copy("chron-seed")
    seed = bytearray(bytes.fromhex((d / "seed.hex").read_text().strip()))
    seed[0] ^= 1
    (d / "seed.hex").write_text(seed.hex() + "\n")
    c = json.loads((d / "chronology.json").read_text())
    for e in c["events"]:
        if e["name"] == "seed_reveal":
            e["seed"] = seed.hex()
    (d / "chronology.json").write_text(json.dumps(c, indent=1, sort_keys=True) + "\n")
    problems = ps.verify(rec / "protocol_instance.json", pop, d, instances=N)
    row("G6", "revealed seed one bit off", "partition_sample.verify: H(seed) != commitment", bool(problems), "; ".join(problems)[:300], time.perf_counter() - t, "n/a")

    t = time.perf_counter()
    d = chron_copy("chron-drop")
    s = json.loads((d / "selected.json").read_text())
    s["selected"] = s["selected"][1:]
    s["count"] -= 1
    (d / "selected.json").write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
    problems = ps.verify(rec / "protocol_instance.json", pop, d, instances=N)
    row("G7", "selected.json one VU dropped", "partition_sample.verify: selected.json == r_s selection", bool(problems), "; ".join(problems)[:300], time.perf_counter() - t, "n/a")

    t = time.perf_counter()
    pdoc = json.loads(pop.read_text())
    pdoc["rows"][0][1] += 1
    pbad = scratch / "population-tampered.json"
    pbad.write_text(json.dumps(pdoc, separators=(",", ":")))
    try:
        problems = ps.verify(rec / "protocol_instance.json", pbad, rec / "chronology", instances=N)
        obs, refused = "; ".join(problems)[:300], bool(problems)
    except Exception as e:  # noqa: BLE001
        obs, refused = f"{type(e).__name__}: {str(e)[:250]}", True
    row("G8", "population row count +1", "partition_sample.verify: population pin / partition digest", refused, obs, time.perf_counter() - t, "n/a")

    t = time.perf_counter()
    reg = json.loads((rec / "weights_authorized.json").read_text())
    c2 = json.loads((rec / "adapter" / "selective" / "commitments.json").read_text())
    row("G9", "registered weights root altered", "V3.weights_root (reverify.py:230)", c2["trees"]["weights"]["root"] != "00" + reg["expected_root"][2:], "prover root != altered registered root -> refuse", time.perf_counter() - t, "n/a")

    if not run_tampers:
        return rows
    try:
        import numpy as np
        from veritor.interface.lowering import statement_bytes_v3
        from veritor.protocol.proofs.wire import statement_digest

        from verity_vllm.capture import CaptureReader
        from verity_vllm.commit import Opening, TreeCache, load_commitments
        from verity_vllm.description import Description
        from verity_vllm.negatives import PatchedReader
        from verity_vllm.typed_obligation_export import InputEquation
        from verity_vllm_adapter import obligations as O
        from verity_vllm_adapter.coordinate_check import coordinate_openings, patched_commitments

        sel = rec / "adapter" / "selective"
        reader = CaptureReader(sel)
        desc = Description(reader.description)
        comm = load_commitments(reader)
        cache = TreeCache(reader)
        u0 = record["provenance"]["units"][0]
        cid, j = u0["canonical_id"], int(u0["j"])
        base = coordinate_openings(desc, reader, comm, cache, cid, j)
        checks = [json.loads(ln) for ln in (rec / "obligations" / "input_checks.jsonl").read_text().splitlines() if ln.strip()]
        wchk = next(c for c in checks if c["kind"] == "weight" and c["expected"] is not None)

        def mini(name: str, rd, cm, ch, cid_=cid, j_=j) -> dict:
            co = coordinate_openings(desc, rd, cm, ch, cid_, j_)
            o = ch.open(wchk["leaf"])
            assert isinstance(o, Opening)
            prefix = (0,) if len(o.shape) == 2 and int(o.shape[0]) == 1 else ()
            eqs = [InputEquation(o, (*prefix, wchk["word"]), wchk["expected"], wchk["source"])]
            _, recd = O.build(cm, [co], eqs, profile_id=rd.profile_id, name=f"A8-{name}", description=f"negative {name}", source="contract_negatives")
            ob = TypedObligationSet.from_json(recd["object"])
            return {"relations": recd["relations"], "witness": recd["witness_verdict"], "verdict": recd["verdict"], "h_L": statement_digest(statement_bytes_v3(ob)).hex()[:12]}

        t = time.perf_counter()
        wid = wchk["leaf"]
        words = np.frombuffer(reader.value(wid).data, dtype="<u2").copy()
        words[wchk["word"]] ^= 0x0080
        patched = PatchedReader(reader, weights={wid: words.tobytes()})
        cm1, ch1 = patched_commitments(reader, comm, patched, cache)
        v1 = mini("T1-weight", patched, cm1, ch1)
        root_changed = cm1["trees"]["weights"]["root"] != comm["trees"]["weights"]["root"]
        # `relations` is the Python evaluation of the relations over the (patched) committed values; `verdict` is the statement's claimed verdict.
        row("T1", f"weight word {wid}[{wchk['word']}] one bit; tree rebuilt", "Input equation rejects AND V3.weights_root", (v1["relations"] != "accept") or root_changed,
            f"python relations={v1['relations']} claimed verdict={v1['verdict']}; weights root changed={root_changed}", time.perf_counter() - t, "NOT CHECKED (guest execute unavailable; Python relation verdict shown)")

        t = time.perf_counter()
        xw = np.frombuffer(reader.value(base.x_id).data, dtype="<u2").copy()
        k = (base.x_cols[0] if base.x_cols else 0) + 3
        xw[k] = 0x5380 | ((int(xw[k]) & 0x8000) ^ 0x8000)
        patched = PatchedReader(reader, values={base.x_id: xw.tobytes()})
        cm2, ch2 = patched_commitments(reader, comm, patched, cache)
        v2 = mini("T2-x", patched, cm2, ch2)
        row("T2", f"operand x word {k} replaced; RU tree rebuilt", "GemmCoordinate relation rejects", v2["relations"] != "accept", f"python relations={v2['relations']} claimed verdict={v2['verdict']}", time.perf_counter() - t, "NOT CHECKED (Python relation verdict)")

        t = time.perf_counter()
        ow = np.frombuffer(reader.value(cid).data, dtype="<u2").copy()
        ow[j] ^= 1
        patched = PatchedReader(reader, values={cid: ow.tobytes()})
        cm3, ch3 = patched_commitments(reader, comm, patched, cache)
        v3 = mini("T3-out", patched, cm3, ch3)
        row("T3", f"output word out[{j}] one ulp; RU tree rebuilt", "GemmCoordinate relation rejects", v3["relations"] != "accept", f"python relations={v3['relations']} claimed verdict={v3['verdict']}", time.perf_counter() - t, "NOT CHECKED (Python relation verdict)")

        # A8 addition for R-N3: the grafted coordinate (call_A, j_B) is a TRUE relation over the honest committed values.
        t = time.perf_counter()
        r1 = json.loads((chunks[1] / "composed.json").read_text())
        pair = find_same_layer_pair(r1)
        if pair is not None:
            A, B = r1["provenance"]["units"][pair[0]], r1["provenance"]["units"][pair[1]]
            vg = mini("N3-graft-truth", reader, comm, cache, cid_=A["canonical_id"], j_=int(B["j"]))
            row("N3-truth", f"honest values: coordinate ({A['canonical_id'].split('/r=')[1][:8]}… t={A['canonical_id'].split('/t=')[1].split('/')[0]} {A['op']}, j={B['j']}) — the column R-N3 grafts",
                "relation over honest values must ACCEPT (so a re-proof carries verdict 1)", vg["relations"] == "accept", f"python relations={vg['relations']} claimed verdict={vg['verdict']}", time.perf_counter() - t, "Python relation verdict (the guest computes the same relation)")
    except Exception as e:  # noqa: BLE001
        rows.append({"id": "T1-T3", "tamper": "not run", "gate": "", "refused": None, "observed": f"{type(e).__name__}: {str(e)[:300]}", "seconds": 0, "sp1": "not run"})
    return rows


# ------------------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--as-host":
        rest = argv[3:] if len(argv) > 2 and argv[2] == "--" else argv[2:]
        return StandInHost(Path(argv[1])).main(rest)
    if argv and argv[0] == "--as-reverify-v5":
        # A9's record_v5 records: e2e.verifier_v3.reverify by import with exactly the two substitutions record_v5.verify_v5 makes (stated in the
        # rows): the population reader for the v5 schema and load_instance := load_instance_v5 (the enumerated allowlist of the two known
        # validate_instance violations of the legacy-minus-inputs instance).  Nothing else of the runner (freeze check, refused_outcome,
        # certificate) runs: the verdict is the verifier's.
        sys.path.insert(0, str(ROOT))
        from record_v5 import deps, instance_v5 as i5, population_v5 as p5

        deps.install()
        from e2e import partition_sample as ps
        from e2e.verifier_v3 import reverify as rv

        p5.install()
        ps.load_instance = i5.load_instance_v5
        return int(rv.main(argv[1:]) or 0)
    if argv and argv[0] == "--stage-elem":
        # [segment 3] build the CK-ELEM statements of a record ONCE (≈ 2.8 GB: verifier program build + capture reader): --stage-elem <record> <expressibility> <stage-dir>
        sys.path.insert(0, str(ROOT))
        s = stage_elem_statements(Path(argv[1]).expanduser(), Path(argv[2]).expanduser(), Path(argv[3]))
        print(json.dumps({k: v for k, v in s.items() if k != "chunks"}, indent=1))
        for c in s["chunks"]:
            print(json.dumps(c))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", required=True, help="the archive dir holding REC-P1-B0/, pop/, setup/ (read-only)")
    ap.add_argument("--elem-stage", default=None, help="[segment 3] the --stage-elem output dir the E-* cases install chunks from")
    ap.add_argument("--record-name", default="REC-P1-B0")
    ap.add_argument("--population", default=None, help="default <archive>/pop/population.compact.json")
    ap.add_argument("--expressibility", default=str(ROOT / "out" / "record" / "expressibility.json"))
    ap.add_argument("--work", default="/tmp/a8-rec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated case ids (R-B,R-N1,...)")
    ap.add_argument("--skip-l1", action="store_true")
    ap.add_argument("--skip-tampers", action="store_true", help="skip T1-T3 (needs numpy + the adapter's weights)")
    ap.add_argument("--timeout", type=int, default=1500, help="watchdog per reverify run (s)")
    ap.add_argument("--verifier", default="v2", choices=["v2", "v3"], help="v2 = verifier_v2/reverify.py (CASES); v3 = e2e.verifier_v3.reverify (V3_CASES)")
    ap.add_argument("--slice", type=int, default=None, help="keep only the first N obligation chunks of each clone (launch control: laptop-sized slice)")
    ap.add_argument("--instance", default=None, help="a contract protocol_instance.json to install in each clone (V3-R1a needs epoch.inputs; V1.instance/V2.chronology then fail by construction)")
    a = ap.parse_args(argv)
    global ELEM_STAGE
    ELEM_STAGE = Path(a.elem_stage) if a.elem_stage else None
    archive = Path(a.archive).expanduser()
    src = archive / a.record_name
    pop = Path(a.population) if a.population else archive / "pop" / "population.compact.json"
    expr = Path(a.expressibility)
    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    stub = write_stub(work, src)
    T0 = time.perf_counter()
    doc: dict[str, Any] = {"schema": "veritor/a8-contract-negatives/v1", "utc": utc(), "archive": str(src), "verifier": str(ROOT / "verifier_v2" / "reverify.py"), "host": "stand-in (recorded verdicts; not SP1)",
                           "population": str(pop), "cases": [], "l1_negatives": []}
    prev_p = out / "contract_negatives.json"
    if prev_p.exists():  # partial re-runs replace rows by id and keep the others
        prev = json.loads(prev_p.read_text())
        doc["cases"], doc["l1_negatives"] = prev.get("cases", []), prev.get("l1_negatives", [])
    only = set(a.only.split(",")) if a.only else None

    order = list(CASES) + list(V3_CASES) + list(T_CASES) + list(E_CASES)

    def put(rows: list[dict], row: dict) -> None:
        rows[:] = [r for r in rows if r["id"] != row["id"]] + [row]
        rows.sort(key=lambda r: order.index(r["id"]) if r["id"] in order else 99)
    if not a.skip_l1 and (only is None or "L1" in only):
        scratch = work / "l1-scratch"
        scratch.mkdir(exist_ok=True)
        doc["l1_negatives"] = l1_negatives(src, pop, stub, scratch, run_tampers=not a.skip_tampers)
        dump(out / "l1_negatives.json", {"utc": utc(), "rows": doc["l1_negatives"]})
    table = CASES if a.verifier == "v2" else ({**V3_CASES, **T_CASES, **E_CASES} if a.record_name != "REC-P1-B0" else V3_CASES)
    if a.verifier == "v3":
        doc["verifier"] = str(POC / "e2e" / "verifier_v3" / "reverify.py")
        doc["slice"] = a.slice
        doc["instance"] = a.instance
    for cid, (edit, case, shows) in table.items():
        if only is not None and cid not in only:
            continue
        cdir = work / cid
        print(f"[contract_negatives] {cid}: cloning …", flush=True)
        clone(src, cdir)
        prep = []
        if a.slice:
            prep.append(slice_record(cdir, a.slice))
        if a.instance and cid in NEEDS_INSTANCE:
            prep.append(install_contract_instance(cdir, Path(a.instance)))
        edit_txt = "none" if edit is None else edit(cdir)
        if prep:
            edit_txt = "; ".join(prep) + "; edit: " + edit_txt
        print(f"[contract_negatives] {cid}: {edit_txt[:160]}", flush=True)
        res = run_reverify(cdir, pop, expr, stub, out / cid, a.timeout, verifier=a.verifier)
        row = {"id": cid, "contract_case": case, "edit": edit_txt, "shows": shows, **res}
        if res["certificate"]:
            c = res["certificate"]
            row["bound"] = {"status_line": c["status_line"], "B_ex": c["selection"]["exempt_B_ex"], "bits": c["bound"]["bits"], "checks_failed": c["checks"]["failed"]}
        row.pop("certificate", None)
        put(doc["cases"], row)
        print(f"[contract_negatives] {cid}: rc={res['rc']} {res['seconds']} s maxrss(children)={res['maxrss_mb_children']} MB failed={res['failed']}", flush=True)
        dump(out / "contract_negatives.json", doc)
        shutil.rmtree(cdir, ignore_errors=True)  # the clone's edited blocks are the only cost; the log/certificate are under --out
    doc["seconds"] = round(time.perf_counter() - T0, 1)
    dump(out / "contract_negatives.json", doc)
    md = ["# A8 record-verifier runs (verifier_v2/reverify.py @ 139e080f on APFS clones of REC-P1-B0; host = recorded-verdict stand-in, not SP1)", "",
          f"utc {doc['utc']}; {doc['seconds']} s total. Commands: see contract_negatives.json `command` fields.", "", "| run | contract case | edit | failed checks | Bound | shows | s | maxrss MB |", "|---|---|---|---|---|---|---|---|"]
    for r in doc["cases"]:
        b = r.get("bound") or {}
        md.append(f"| {r['id']} | {r['contract_case']} | {r['edit'].replace('|', '/')} | {', '.join(r['failed']) or '—'} | {b.get('status_line', '?')} B_ex={b.get('B_ex')} | {r['shows']} | {r['seconds']} | {r['maxrss_mb_children']} |")
    if doc["l1_negatives"]:
        md += ["", "## L1 baseline (G1-G9, T1-T3)", "", "| id | tamper | gate | refused | observed | SP1 part |", "|---|---|---|---|---|---|"]
        for r in doc["l1_negatives"]:
            md.append(f"| {r['id']} | {r['tamper'].replace('|', '/')} | {r['gate']} | {r['refused']} | {r['observed'].replace('|', '/')} | {r['sp1']} |")
    (out / "contract_negatives.md").write_text("\n".join(md) + "\n")
    print(f"[contract_negatives] done in {doc['seconds']} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
