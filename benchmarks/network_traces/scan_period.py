"""scan-period: `verity/certifier/v0`'s rows form (`NetworkCheck.Period`) on a warden run's group and the scan's answers
for it (`scan_zk.py prove`'s ``answers.json``), checked by the period's Python mirror and its Lean.

    python benchmarks/network_traces/scan_period.py --mirror PERIOD_CASE_PY --lean NETWORK_PERIOD --out DIR \
        NAME=WARDEN_RUN,SCAN_OUT[,tampered] ...

* **The case** (`rows_case`) of a group's link-windows W: the parameters the warden run's ``deployment.json`` (its key,
  session and one schedule), the link `warden_serve.LINK`, the Program the scan's registration's, no compute audit, and
  the scan's layout terms its free inputs over a link-window's buckets (`scan_free`); the commitment the scan's
  registration for each W; the evidence, for each W, the scan's answer, the certificates of W's three records
  (``certified.jsonl``: requests, responses, fails) and W's two row registrations (``cells.jsonl``, as ``answers.json``
  carries them).
* **tampered**: the same warden run's rows and certificates with the scan's tampered round's registration
  (``tampered/registration.json``, an egress-only Program) and its refusal as each W's answer: a prover's answer for
  another statement than the link's Program.
* **The check**: each case through the mirror (``--mirror``, `period_case.py` loaded by path: `reference`) and
  ``network-period check`` (``--lean``), whose verdicts must agree and be the case's (`EXPECT`): accept where the
  scan's answers accept, else reject. ``DIR`` gets each case's files and ``period.json``, each case's answers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scan_zk as SZ  # noqa: E402
import warden_serve as WS  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registration as RG  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402

CERTIFICATE_KEYS = ("digest", "direction", "key_id", "link", "signature", "window")


def scan_free(summary: dict) -> int:
    """The scan's free inputs a link-window, an overcount: each of its T buckets' units' rows (``summary.json``'s
    ``bits``) but the warden's own cells (``row``), which its row registrations bind."""
    return summary["T"] * sum(b for k, b in summary["bits"].items() if k.rsplit("/", 1)[-1] != "row")


def certificates(run: Path, w: int) -> list[dict]:
    """Link-window w's three records' certificates in ``certified.jsonl``, in the rows form's order (requests,
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


def rows_case(run: Path, scan: Path, round_: str = "honest") -> dict:
    """The rows form's case (module docstring) of the warden run ``run``'s group, as scan_zk.py proved it into
    ``scan``: honest, ``answers.json``'s; ``tampered``, the tampered round's registration and its refusal."""
    dep = json.loads(SZ._found(run, "deployment.json").read_text())
    doc = json.loads((scan / "answers.json").read_text())
    summary = json.loads((scan / "summary.json").read_text())
    windows = doc["windows"]
    if round_ == "honest":
        reg, answers = doc["registration"], [x["answer"] for x in windows]
    else:
        reg = json.loads((scan / round_ / "registration.json").read_text())
        why = summary["rounds"][round_]["audit"]["why"]
        answers = [SZ.window_answer(reg, None, None, f"the scan's {round_} round refused: {why}")] * len(windows)
    N, W, H = dep["key"]["params"]
    params = {"audits": [], "cells": [], "first": windows[0]["window"], "jitter": 1, "key": dep["key"]["public"],
              "link": WS.LINK, "lms": {"H": H, "N": N, "W": W, "toy": False}, "program": doc["registration"]["program"],
              "scan": {"free": scan_free(summary), "steps": summary["T"]}, "schedule": dict(dep["params"]),
              "session": dep["session"], "windows": len(windows)}
    commitment = canonical_json({"audits": [], "windows": [reg] * len(windows)})
    evidence = canonical_json({"audits": [], "windows": [
        {"answer": a, "certificates": certificates(run, x["window"]), "registrations": x["registrations"]}
        for x, a in zip(windows, answers)]})
    return {"commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}


def lean_check(exe: str, case: dict, d: Path) -> dict:
    """``network-period check`` on the case's three files in ``d``: its answer (its last line)."""
    d.mkdir(parents=True, exist_ok=True)
    files = [d / "params.json", d / "commitment.bin", d / "evidence.json"]
    files[0].write_bytes(canonical_json(case["params"]))
    files[1].write_bytes(bytes.fromhex(case["commitment"]))
    files[2].write_bytes(bytes.fromhex(case["evidence"]))
    p = subprocess.run([exe, "check", *map(str, files)], capture_output=True, text=True)
    if p.returncode not in (0, 1, 2) or not p.stdout.strip():
        raise SystemExit(f"network-period check exited {p.returncode}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def mirror(path: Path):
    spec = importlib.util.spec_from_file_location("period_case", path)
    mod = sys.modules["period_case"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scan_period.py", description=__doc__.splitlines()[0])
    ap.add_argument("--mirror", type=Path, required=True, help="the period's mirror (period_case.py, its tests')")
    ap.add_argument("--lean", required=True, help="network-period, built")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("cases", nargs="+", help="NAME=WARDEN_RUN,SCAN_OUT[,tampered]")
    a = ap.parse_args(argv)
    m = mirror(a.mirror)
    a.out.mkdir(parents=True, exist_ok=True)
    got, ok = {}, True
    for spec in a.cases:
        name, _, rest = spec.partition("=")
        run, scan, *round_ = rest.split(",")
        case = rows_case(Path(run), Path(scan), *round_)
        (a.out / name).mkdir(exist_ok=True)
        (a.out / name / "case.json").write_text(json.dumps(case, indent=1, sort_keys=True))
        py, lean = m.reference(case), lean_check(a.lean, case, a.out / name)
        scan_said = {x["answer"]["verdict"] for x in json.loads((Path(scan) / "answers.json").read_text())["windows"]}
        want = "accept" if not round_ and scan_said == {"accept"} else "reject"
        good = py.get("verdict") == lean.get("verdict") == want and py.get("public") == lean.get("public")
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
