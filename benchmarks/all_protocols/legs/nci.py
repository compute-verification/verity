"""The registration check: NCI's check of record (`NciCheck.Protocol`, ``verity/nci/v1``) on the window's own
declaration, the case onsite's registration check reads in its commit phase, before composing
(`Core.Composition.registered`; `verity_all_protocols.table.REGISTRATION`, with the legs' interface: ``leg(period_dir,
node, out) -> answer``). NCI is no component of onsite: a registration it refuses refuses onsite.

**The declaration** (`declaration`) is the window's program as the verifier holds it: the population program of the
circuit record it registered before the loop ran (``period.json``'s ``circuit``), rebuilt from that record
(`interface.of_records`, which holds it to the window's registration), and refused unless its digest is the
registration's ``program``.

**The case** (`case_of`, in the formats of `NciCheck/Protocol.lean`'s docstring):

* **params**: the policy (`GAMMA`, `ALPHA`); the log before the window's own record, empty, since each window's record
  opens its log (``previous`` null); the program function's parameters; and the program's descriptor;
* **commitment**: the window's registration record, as canonical JSON;
* **evidence**: the division, every input node ``input`` and every computed root node in IU 0 (the population's one
  batch of units), with θ empty and so no provenance. The loop's weights reach its units through the circuit record's
  registered reads, not as parameters bound at registration, so no input node is θ.

The answer is `nci_reference.check_baseline`'s on the case, the reference `NciCheck.Baseline` is held to. With
``nci`` among the period's tools (``run.sh`` builds it), Lean's ``nci serve`` answers too and must agree. The leg
writes ``out/nci.case.json`` (onsite's registration case), ``out/policy.json`` (`policy`, which the runner's row
carries) and ``out/nci.json``. NCI reads no coins, so a tampered window's declaration is its honest one's, and the
integrity leg is what refuses it.

**A tampered registration** (the runner's ``--registration-tamper``, ``period.json``'s ``registration_tamper``): the
case is on `tampered`'s record, not the window's, as a prover would show the registration check while its proofs open the
window's own roots. NCI reads no root (θ empty), so it accepts that record, and onsite refuses its digest, which is not
the one sampled proofs commit to (`Core.Composition.mismatch`).

**The policy.** A protocol carries no policy parameter, so the check reports what it measured: a unit of the loop is one
``ScaledMmS8Coordinate_v1{K=128}``, 9 gates of the canonical layout reading 2,112 bits, every one of them a run input,
so the window's one IU reads 2112/9 ≈ 235 novel bits a gate whatever the window's size, and uses its gates' share of
Γ. The verdict onsite composes is at α = 256 bits a gate (`ALPHA`) and Γ = 2^44 gates an IU (`GAMMA`, the GLM-5.3
demo's), a placeholder for the toy (`LABEL`). `policy` also states the verdict at the GLM-5.3 demo's α = 2^-25
(`GLM_ALPHA`, policy A in ``benchmarks/verifier_cost/nci_glm.py``), where the toy doesn't comply.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs.one_stage import draw as D
from verity.ml.circuits import codec
from verity.ml.windows import interface as IF

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "verity" / "core" / "protocols" / "network" / "nci" / "lean" / "scripts" / "nci_reference.py"
PROTOCOL = "verity/nci/v1"
#: the policy the composed verdict is at: Γ gates an IU (the GLM-5.3 demo's) and α novel bits a gate (module docstring)
GAMMA = 2 ** 44
ALPHA = (256, 1)
LABEL = "placeholder (toy)"
#: the GLM-5.3 demo's α (policy A), at which `policy` states the verdict too
GLM_ALPHA = (1, 2 ** 25)


def reference():
    """``nci_reference``, the NCI check's Python reference (a script of NCI's Lean package)."""
    spec = importlib.util.spec_from_file_location("nci_reference", REFERENCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def declaration(period_dir: Path) -> tuple[dict[str, Any], Any]:
    """The window's registration record and the verifier's copy of its program (module docstring)."""
    p = json.loads((period_dir / "period.json").read_text())
    if hashlib.sha512((period_dir / "circuit.json").read_bytes()).hexdigest() != p["circuit"]:
        raise ValueError("the window's circuit.json is not the circuit record the verifier registered (period.json's "
                         "circuit)")
    window, _ = IF.of_records(period_dir, D.parse_law(p["law"]), IF.resolve)
    rec = json.loads((period_dir / "registration.json").read_text())
    if codec.program_digest(window.program) != rec["program"]:
        raise ValueError("the verifier's program is not the one the window's registration names")
    return rec, window.program


def tampered(rec: dict[str, Any]) -> dict[str, Any]:
    """A registration other than ``rec`` under the same program (module docstring): ``rec`` with its first root's last hex
    digit changed."""
    roots = [dict(r) for r in rec["roots"]]
    h = roots[0]["root"]
    roots[0]["root"] = h[:-1] + ("1" if h[-1] == "0" else "0")
    return {**rec, "roots": roots}


def case_of(rec: dict[str, Any], program: Any) -> dict[str, Any]:
    """NCI's case on the window's record and program (module docstring): ``{"commitment", "evidence", "params"}``."""
    nr = reference()
    text = codec.canonical_json(codec.encode_program(program)).decode()
    inputs, _, _ = nr.root_nodes(text)
    params = {"alpha": list(ALPHA), "gamma": GAMMA, "log": [],
              "parameters": [[n, t.to_json()] for n, t in program.root.body.params], "program": text}
    evidence = {"owner": ["input" if i else 0 for i in inputs], "provenance": [], "theta": []}
    return {"commitment": codec.canonical_json(rec).hex(), "evidence": codec.canonical_json(evidence).hex(),
            "params": params}


def baseline(case: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The registration record's digest and the reference's verdict on ``case``: ``{"gates", "ius", "complies",
    "failure"}`` (each IU ``[size, novelReads]``), or ``{"error": code}``."""
    nr = reference()
    p, rec = case["params"], json.loads(bytes.fromhex(case["commitment"]))
    e = json.loads(bytes.fromhex(case["evidence"]))
    run = nr.record_digest(rec)
    instance = {"alpha": p["alpha"], "gamma": p["gamma"], "owner": e["owner"], "program": p["program"],
                "provenance": e["provenance"], "theta": e["theta"]}
    return run, nr.check_baseline(instance, {"log": [*p["log"], rec], "params": p["parameters"], "run": run})


def answer(case: dict[str, Any]) -> dict[str, Any]:
    """The reference's answer on ``case``, in the core interface's canonical form, as `NciCheck.Protocol.check` gives it
    on a case whose parts decode."""
    run, out = baseline(case)
    base = {"commitment": run, "profile": None, "protocol": PROTOCOL, "public": None}
    if "error" in out:
        return base | {"reason": out["error"], "verdict": "reject"}
    if out["complies"] is not True:
        u, rule = out["failure"]
        return base | {"reason": f"the division does not comply: isolation unit {u} fails its {rule} bound",
                       "verdict": "reject"}
    return base | {"profile": {"gates": out["gates"], "ius": out["ius"]}, "public": {}, "reason": None,
                   "verdict": "accept"}


def at_alpha(case: dict[str, Any], alpha: tuple[int, int]) -> dict[str, Any]:
    """``case`` under the policy's α ``alpha``."""
    return {**case, "params": {**case["params"], "alpha": list(alpha)}}


def policy(case: dict[str, Any]) -> dict[str, Any]:
    """What the check measured on ``case`` and its verdict at each policy (module docstring): each IU's gates, novel
    bits read, bits a gate and share of Γ; the verdict at `ALPHA` and `GAMMA`, the one onsite composes, labelled
    `LABEL`; and the verdict at `GLM_ALPHA`."""
    def verdict(out: dict[str, Any]) -> str:
        return f"error: {out['error']}" if "error" in out else "complies" if out["complies"] is True else "doesn't comply"

    _, out = baseline(case)
    _, glm = baseline(at_alpha(case, GLM_ALPHA))
    measured = None if "error" in out else {"gates": out["gates"], "ius": [
        {"gates": size, "novel_bits": novel, "bits_a_gate": round(novel / size, 2), "gamma_use": size / GAMMA}
        for size, novel in out["ius"]]}
    return {"label": LABEL, "alpha": list(ALPHA), "gamma": GAMMA, "measured": measured, "verdict": verdict(out),
            "at_glm": {"alpha": list(GLM_ALPHA), "gamma": GAMMA, "verdict": verdict(glm)}}


def serve(exe: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``nci serve`` on ``cases``, one canonical JSON line each: Lean's answers (`Core.Protocol`'s CLI)."""
    lines = "".join(codec.canonical_json(c).decode() + "\n" for c in cases)
    p = subprocess.run([exe, "serve"], input=lines, capture_output=True, text=True, check=True)
    return [json.loads(x) for x in p.stdout.splitlines()]


def leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """The registration check of ``benchmarks/all_protocols`` (module docstring)."""
    period_dir = Path(period_dir)
    p = json.loads((period_dir / "period.json").read_text())
    tools, tamper = p.get("tools") or {}, bool(p.get("registration_tamper"))
    t = time.monotonic()
    rec, program = declaration(period_dir)
    case = case_of(tampered(rec) if tamper else rec, program)
    got, lean = answer(case), None
    if "nci" in tools:
        lean = serve(tools["nci"], [case])[0]
        if {k: v for k, v in lean.items() if k != "reason"} != {k: v for k, v in got.items() if k != "reason"}:
            raise RuntimeError(f"nci answered {lean}, the reference {got}")
        got = lean
    at = policy(case)
    out.mkdir(parents=True, exist_ok=True)
    (out / "nci.case.json").write_text(json.dumps(case, sort_keys=True))
    (out / "policy.json").write_text(json.dumps(at, sort_keys=True, indent=1))
    (out / "nci.json").write_text(json.dumps({"answer": got, "lean": lean, "policy": at, "tampered": tamper,
                                              "seconds": round(time.monotonic() - t, 2)}, sort_keys=True, indent=1))
    return got


__all__ = ["ALPHA", "GAMMA", "GLM_ALPHA", "LABEL", "PROTOCOL", "answer", "at_alpha", "baseline", "case_of",
           "declaration", "leg", "policy", "reference", "serve", "tampered"]
