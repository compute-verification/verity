"""The door's Lean under mutation, one edit at a time in the run's clone: each mutant rebuilt (Flock's Lean-check roots
and flock-lower-check, then Spot) and the Lean-backed Python tests run, slow ones included; KILLED if a build or a test
fails, SURVIVED if both pass. Results as JSON lines in $OUT/lean_results.jsonl.

  python lean_mutants.py MUTANTS.json [ID ...]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
OUT = Path(os.environ["OUT"])
F = ROOT / "verity/protocols/proofs/flock/verifier"
S = ROOT / "verity/protocols/proofs/spot/verifier/lean"
C = ROOT / "verity/circuits/lean"
FLOCK = json.loads((F / "lean-audit.json").read_text())["roots"] + ["Flock.Test.LowerBad", "flock-lower-check"]
TESTS = [
    "tests/verity/protocols/proofs/flock/prover/test_lean_lower.py",
    "tests/verity/protocols/proofs/spot/verifier/test_one_circuit_chain.py",
    "tests/verity/protocols/proofs/spot/verifier/lean/test_restrict_difftest.py",
    "tests/verity/protocols/proofs/spot/verifier/lean/test_block_front_difftest.py",
    "tests/verity/protocols/proofs/spot/verifier/lean/test_restrict_block_circuit.py",
]
PY = str(ROOT / ".venv/bin/python")


def sh(cmd: list[str], cwd: Path, timeout: int = 3600) -> tuple[int, float, str]:
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, time.monotonic() - t0, "timeout"
    return p.returncode, time.monotonic() - t0, (p.stdout + p.stderr)


def builds(paths: set[str]) -> list[tuple[str, int, float, str]]:
    out = []
    if any("circuits/lean" in p for p in paths):
        out.append(("circuits",) + sh(["lake", "build", "Circuits"], C))
    out.append(("flock",) + sh(["lake", "build", *FLOCK], F))
    out.append(("spot",) + sh(["lake", "build", "Spot"], S))
    return out


def tests(deselect: list[str]) -> tuple[int, float, str]:
    des = [x for d in deselect for x in ("--deselect", d)]
    rc_all, t_all, tails = 0, 0.0, []
    for t in TESTS:
        rc, dt, out = sh([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-rfE", "--no-header", *des, t], ROOT)
        t_all += dt
        tails.append(f"{t}: rc={rc} {dt:.0f}s\n" + out[-3000:])
        if rc not in (0, 5):
            return rc, t_all, "\n".join(tails)
    return rc_all, t_all, "\n".join(tails)


def failed_ids(out: str) -> list[str]:
    return [ln.split(" ", 1)[1].split(" - ", 1)[0].strip() for ln in out.splitlines() if ln.startswith(("FAILED ", "ERROR "))]


def main(argv: list[str]) -> int:
    muts = json.loads(Path(argv[0]).read_text())
    if argv[1:]:
        muts = [m for m in muts if m["id"] in argv[1:]]
    for m in muts:
        src = (ROOT / m["file"]).read_text()
        assert src.count(m["old"]) == 1, f"{m['id']}: its text occurs {src.count(m['old'])} times in {m['file']}"
    res = OUT / "lean_results.jsonl"
    b = builds({"verity/circuits/lean"})
    print("baseline builds:", [(n, rc, round(dt)) for n, rc, dt, _ in b], flush=True)
    assert all(rc == 0 for _n, rc, _dt, _ in b), "\n".join(o[-3000:] for *_x, o in b)
    deselect: list[str] = []
    for t in TESTS:
        rc, dt, out = sh([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", "--no-header", t], ROOT)
        bad = failed_ids(out) if rc not in (0, 5) else []
        deselect += bad
        print(f"baseline {t}: rc={rc} {dt:.0f}s deselect={bad}", flush=True)
        if rc not in (0, 5) and not bad:
            print(out[-3000:])
    (OUT / "lean_baseline.json").write_text(json.dumps({"deselect": deselect}))
    for m in muts:
        path = ROOT / m["file"]
        src = path.read_text()
        path.write_text(src.replace(m["old"], m["new"]))
        rec = dict(m)
        try:
            b = builds({m["file"]})
            rec["builds"] = [{"pkg": n, "rc": rc, "secs": round(dt), "tail": o[-2500:] if rc else ""} for n, rc, dt, o in b]
            if any(rc != 0 for _n, rc, _dt, _o in b):
                rec["status"], rec["how"] = "KILLED", "build"
            else:
                rc, dt, out = tests(deselect)
                rec["tests_secs"] = round(dt)
                rec["tests_tail"] = out[-4000:]
                rec["status"], rec["how"] = ("SURVIVED", "") if rc in (0, 5) else ("KILLED", "test")
        finally:
            path.write_text(src)
        with res.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{m['id']} {rec['status']} {rec.get('how')} {m['desc']}", flush=True)
    b = builds({"verity/circuits/lean"})
    print("restored builds:", [(n, rc, round(dt)) for n, rc, dt, _ in b], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
