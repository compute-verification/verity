"""scan-period: `verity/certifier/v0`'s rows form (`NetworkCheck.Period`) on a network certifier run's execution phase
and the scan's answer for it (`scan_zk.py prove --group 1`'s ``answers.json``), checked by the period's Python mirror
and its Lean.

    python benchmarks/network_traces/scan_period.py --mirror PERIOD_CASE_PY --lean NETWORK_PERIOD --out DIR \
        [--v3c PERIOD_JSON --circuit CIRCUIT_RECORD] NAME=WARDEN_RUN,SCAN_OUT,RECORD[,tampered|chain|forged] ...

* **The case** (`rows_case`) of link-window W: the parameters the deployment RECORD names, the auditor's record of the
  warden run (`warden_chain.RECORD`, ``CERTIFIER_KEY``'s form: the gate's own ``out/deployment.json`` of the run as the
  auditor read it off the gate, which every ``deployment.json`` the run holds must be, `warden_chain.deployment_of`;
  ``own`` for a run the job made itself, scan_period.sh's, the job being its own auditor): its key, session
  and one schedule (whose caps over the one link-window are the cells, `WardenParams.caps`: `egress_cells` for each of
  the `lanes · window` connections it can carry), the link `warden_serve.LINK`, the Program the scan's registration's,
  the compute audits
  ``answers.json`` carries (the block round's, whose o0 is the scan's `prog`: its descriptor and steps, off which
  descriptor the period check counts its free inputs, and V3c's circuit record, wire order and pins where ``--v3c``'s
  ``period.json`` carries them, on which it counts them instead, `AUDIT_KEYS`),
  the jitter bound `leak_bounds.JITTER_BOUND` (K's jitter term T bits a link-window, `leak_bounds.JITTER`), and the
  scan that Program's descriptor (``joint/program.json``), off which the period check reads the scan's terms; the
  commitment the audits' registrations and the scan's; the evidence the audits' answers, then the scan's answer, the
  certificates of W's three records (``certified.jsonl``: requests, responses, fails) and W's two row registrations
  (``cells.jsonl``, as ``answers.json`` carries them). The rows form reads a registration of one link-window's 2T
  units, so a group of more is refused, the scan's `prog` must be an audit's `SlotEntry_v1` template's entries
  and its ``scan/state`` that audit's, and that audit must run the class circuit, the parameters' ``circuit``
  (`heldChecks`): the stand-in block audit's own Program here.
* **tampered** and **chain**: the same rows and certificates with that round's registration
  (``<round>/registration.json``) and its refusal as the answer: tampered's an egress-only Program's, a prover's answer
  for another statement than the link's Program; chain's the link's Program's, refused.
* **forged**: the honest case with the responses record's certificate's signature changed: a certificate the network
  certifier didn't sign.
* **The check**: each case through the mirror (``--mirror``, `period_case.py` loaded by path: `reference`) and
  ``network-period check`` (``--lean``), whose verdicts must agree and be the case's: reject, and where the scan's
  answer accepts, at the wiring (the mirror's ``UNWIRED`` in its reason and in Lean's). The block round's o0 is a
  `SlotEntry_v1` stand-in's entries, whose closure digest is none the period check pins (`entryPins`: the scan's
  `prog` is a ``scan/prog<j>`` a lane, and lane j's must be chunk j, ``<descriptor id>/c<j>/o0``, of a pinned
  `SlotEntry_v1`'s or `SelectEntry_v1`'s own entries, as the block round's lane-j o0 is of the stand-in's), so the
  honest case passes every check before the wiring and is refused there; its registration names the scan's state tree
  as ``scan/state``, the state the stand-in's `SCAN_STATE` is tied to (`stateTied`). ``DIR`` gets each case's files
  and ``period.json``, each case's answers.
* **V3c's audit** (``--v3c``, ``records_audit.py period``'s ``period.json``, for scans proved on V3c's trees,
  ``scan_zk.py --v3c``): the one compute audit, its Program's `SelectEntry_v1` chunks' o0 the scan's ``scan/prog<j>``
  and its `SCAN_STATE` the scan's ``scan/state``, its registration the one the scan's ``summary.json`` names (a foreign
  audit is refused before any check), held to the class circuit of the verifier's own window-circuit record
  (``--circuit``, `circuit_of_record`): a phase whose audit's registration names another Program is refused by name,
  in Lean and in the mirror. The honest case is then accepted at any sample of V3c's (``subset:1``
  included): the period check is the consistency check, the committed I/O against the certificates, and what a
  sample of one or two proof blocks leaves unproved is covert capacity, which K charges.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import leak_bounds as LB  # noqa: E402
import scan_zk as SZ  # noqa: E402
import warden_chain as WC  # noqa: E402
import warden_serve as WS  # noqa: E402
from verity.core.protocols.certifier.warden import WardenParams  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registration as RG  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity.ml.windows.interface import CIRCUIT  # noqa: E402

CERTIFICATE_KEYS = ("digest", "direction", "key_id", "link", "signature", "window")
#: a compute audit's parameters: its Program and steps, and where ``records_audit.py period`` gives them its circuit
#: record, its wires' order and the pins its registration holds, on which the period check counts its free inputs
AUDIT_KEYS = ("circuit", "order", "pins", "program", "steps")


def certificates(run: Path, w: int) -> list[dict]:
    """Execution phase w's three records' certificates in ``certified.jsonl``, in the rows form's order (requests,
    responses, fails: `warden_serve.CERTIFIED`), each the certificate's own fields."""
    docs = [json.loads(x)["certificate"] for x in SZ._found(run, "certified.jsonl").read_text().splitlines()
            if "certificate" in json.loads(x)]
    out = []
    for name in (WS.INGRESS, WS.EGRESS, WS.FAILS):
        link, d = WS.CERTIFIED[name]
        got = [c for c in docs if (c["link"], c["direction"], c["window"]) == (link, d.value, w)]
        if len(got) != 1:
            raise SystemExit(f"{run}: link-window {w}'s {name} record has {len(got)} certificates, not 1")
        out.append({k: got[0][k] for k in CERTIFICATE_KEYS})
    return out


def rows_case(run: Path, scan: Path, round_: str = "honest", v3c: Path | None = None,
              circuit: str | None = None, *, record: dict) -> dict:
    """The rows form's case (module docstring) of the warden run ``run``'s link-window, as scan_zk.py proved it into
    ``scan``: honest, ``answers.json``'s; forged, the honest one with a certificate's signature changed; a refused round
    (tampered, chain), that round's registration and its refusal. With ``v3c`` (V3c's ``period.json``,
    ``records_audit.py period``'s, for a scan proved on its trees, ``scan_zk.py --v3c``) the one compute audit is V3c's,
    in place of ``answers.json``'s, and its registration must be the one whose trees the scan read (``summary.json``'s
    ``v3c``): the period's audit, its scan's trees and the run's record are one registration, and an audit of another
    is refused by name, never put in its place. With ``v3c``, ``circuit`` is required: the class circuit, the Program
    digest of the verifier's own window-circuit record (`circuit_of_record`), to which the check holds V3c's
    registration. Without ``v3c`` the class circuit is the stand-in block audit's own Program, whose wiring the check
    refuses first (with no block audit, none, whose parameters the check refuses). The network certifier's key,
    session and schedule are ``record``'s, the auditor's record of the run's deployment (`warden_chain.RECORD`), and a
    run whose deployment.json isn't the record's is refused by name (`warden_chain.deployment_of`)."""
    if v3c is not None:
        if circuit is None:
            raise SystemExit(f"{v3c}: V3c's audit is held to the verifier's own window-circuit record: give its circuit")
        audit = json.loads(v3c.read_text())
        read = (json.loads((scan / "summary.json").read_text()).get("v3c") or {}).get("registration")
        if read != (got := RG.digest(audit["registration"])):
            raise SystemExit(f"{v3c}: a foreign audit: it registers {got[:16]}…, and the scan read the trees of "
                             f"{f'{read[:16]}…' if read else 'no V3c registration'}")
    dep, _ = WC.deployment_of(run, record)
    doc = json.loads((scan / "answers.json").read_text())
    if v3c is not None:
        doc["audits"] = [audit]
    elif circuit is None and doc.get("audits"):
        circuit = doc["audits"][0]["registration"]["program"]
    if len(doc["windows"]) != 1:
        raise SystemExit(f"{scan}: a group of {len(doc['windows'])} link-windows is one registration over all their "
                         "units, and the rows form reads one link-window's 2T: prove each with scan_zk.py --group 1")
    (x,) = doc["windows"]
    if round_ in ("honest", "forged"):
        reg, answer = doc["registration"], x["answer"]
    else:
        reg = json.loads((scan / round_ / "registration.json").read_text())
        why = json.loads((scan / "summary.json").read_text())["rounds"][round_]["audit"]["why"]
        answer = SZ.window_answer(reg, None, None, f"the scan's {round_} round refused: {why}")
    certs = certificates(run, x["window"])
    if round_ == "forged":
        sig = certs[1]["signature"]
        certs[1]["signature"] = sig[:-1] + ("1" if sig[-1] == "0" else "0")
    N, W, H = dep["key"]["params"]
    audits = doc.get("audits", [])
    caps = WardenParams(**dep["params"]).caps(1)
    params = {"audits": [{k: a[k] for k in AUDIT_KEYS if k in a} for a in audits], "cells": caps,
              "circuit": circuit, "first": x["window"], "jitter": LB.JITTER_BOUND, "key": dep["key"]["public"],
              "link": WS.LINK,
              "lms": {"H": H, "N": N, "W": W, "toy": False}, "program": doc["registration"]["program"],
              "scan": json.loads((scan / "joint" / "program.json").read_text()), "schedule": dict(dep["params"]),
              "session": dep["session"], "windows": 1}
    commitment = canonical_json({"audits": [a["registration"] for a in audits], "windows": [reg]})
    evidence = canonical_json({"audits": [a["answer"] for a in audits], "windows": [
        {"answer": answer, "certificates": certs, "registrations": x["registrations"]}]})
    return {"commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}


def circuit_of_record(path: Path) -> str:
    """The class circuit a window-circuit record names (``verity/window-circuit/v1``, the verifier's own, which
    ``served_verdicts.sh`` keeps as ``window-circuit-w<K>-<class>.json`` beside V3c's ``v3c-w<K>-<class>``): its
    Program's digest, which a window's registration under that class names too."""
    doc = json.loads(path.read_text())
    if doc.get("format") != CIRCUIT or not isinstance(doc.get("program"), str):
        raise SystemExit(f"{path}: not a {CIRCUIT} record naming its program")
    return doc["program"]


def lean_check(exe: str, case: dict, d: Path,
               run: Callable[[list[str]], subprocess.CompletedProcess] | None = None) -> dict:
    """``EXE check`` (``network-period``'s, or ``network-rows``') on the case's three files in ``d``: its answer (its
    last line). ``run`` runs it with its output captured as text (`subprocess.run`'s ``capture_output=True, text=True``
    by default)."""
    d.mkdir(parents=True, exist_ok=True)
    files = [d / "params.json", d / "commitment.bin", d / "evidence.json"]
    files[0].write_bytes(canonical_json(case["params"]))
    files[1].write_bytes(bytes.fromhex(case["commitment"]))
    files[2].write_bytes(bytes.fromhex(case["evidence"]))
    argv = [exe, "check", *map(str, files)]
    p = run(argv) if run else subprocess.run(argv, capture_output=True, text=True)
    if p.returncode not in (0, 1, 2) or not p.stdout.strip():
        raise SystemExit(f"{Path(exe).name} check exited {p.returncode}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def mirror(path: Path):
    spec = importlib.util.spec_from_file_location("period_case", path)
    mod = sys.modules["period_case"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case(ap: argparse.ArgumentParser, spec: str) -> tuple[str, Path, Path, dict, list[str]]:
    """The command line's case ``spec`` (NAME=WARDEN_RUN,SCAN_OUT,RECORD[,ROUND]) with its record read (``own``: the
    run's own, `warden_chain.own_record`), else a usage error naming the case and what it lacks or why its record is
    unread."""
    name, _, rest = spec.partition("=")
    fields = rest.split(",")
    if not name or not 3 <= len(fields) <= 4 or not all(fields[:3]):
        ap.error(f"case {spec!r}: give NAME=WARDEN_RUN,SCAN_OUT,RECORD[,tampered|chain|forged], RECORD a {WC.RECORD} "
                 "file or own")
    run, scan, record, *round_ = fields
    try:
        read = WC.own_record(Path(run)) if record == "own" else json.loads(Path(record).read_text())
    except (OSError, ValueError, KeyError) as e:
        ap.error(f"case {name}: its RECORD {record}: {type(e).__name__}: {e}")
    return name, Path(run), Path(scan), read, round_


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scan_period.py", description=__doc__.splitlines()[0])
    ap.add_argument("--mirror", type=Path, required=True, help="the period's mirror (period_case.py, its tests')")
    ap.add_argument("--lean", required=True, help="network-period, built")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--v3c", type=Path, metavar="PERIOD",
                    help="V3c's period.json (records_audit.py period): the one compute audit, for scans proved on its "
                         "trees (scan_zk.py --v3c); without it each case's audit is held to its own Program, which "
                         "Lean refuses at the wiring")
    ap.add_argument("--circuit", type=Path, metavar="CIRCUIT_RECORD",
                    help="the verifier's own window-circuit record of V3c's class (prove_path.py window-circuit): its "
                         "program is the class circuit V3c's audit is held to; required with --v3c")
    ap.add_argument("cases", nargs="+", help="NAME=WARDEN_RUN,SCAN_OUT,RECORD[,tampered|chain|forged], RECORD a "
                                             f"{WC.RECORD} file, or own for a run the job made itself")
    a = ap.parse_args(argv)
    if a.v3c and not a.circuit:
        ap.error("--v3c needs --circuit: V3c's audit is held to the verifier's own window-circuit record")
    circuit = circuit_of_record(a.circuit) if a.circuit else None
    cases = [_case(ap, spec) for spec in a.cases]
    m = mirror(a.mirror)
    a.out.mkdir(parents=True, exist_ok=True)
    got, ok = {}, True
    for name, run, scan, record, round_ in cases:
        case = rows_case(run, scan, *round_, v3c=a.v3c, circuit=circuit, record=record)
        (a.out / name).mkdir(exist_ok=True)
        (a.out / name / "case.json").write_text(json.dumps(case, indent=1, sort_keys=True))
        py, lean = m.reference(case), lean_check(a.lean, case, a.out / name)
        scan_said = {x["answer"]["verdict"] for x in json.loads((scan / "answers.json").read_text())["windows"]}
        honest = not round_ and scan_said == {"accept"}
        if honest and a.v3c:
            want = "accept"
            good = (py.get("verdict") == lean.get("verdict") == "accept"
                    and py.get("public") == lean.get("public"))
        else:
            why = m.UNWIRED if honest else None
            want = "reject" + (" at the wiring" if why else "")
            good = (py.get("verdict") == lean.get("verdict") == "reject" and py.get("public") == lean.get("public")
                    and (why is None or why in py.get("reason", "") and why in lean.get("reason", "")))
        ok &= good
        reg = json.loads(bytes.fromhex(case["commitment"]))["windows"][0]
        got[name] = {"want": want, "mirror": py, "lean": lean, "as_expected": good, "registration": RG.digest(reg)}
        print(f"SCPD {name}: want {want}; mirror {py.get('verdict')} {py.get('reason') or py.get('public')}; Lean "
              f"{lean.get('verdict')} {lean.get('reason') or lean.get('public')}; "
              f"{'as expected' if good else 'UNEXPECTED'}", flush=True)
    (a.out / "period.json").write_text(json.dumps({"cases": got, "ok": ok}, indent=1, sort_keys=True, default=str))
    print(f"SCPD done: {'ok' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
