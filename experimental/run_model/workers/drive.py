"""The toy audit with Python driving: the auditor's side is a Python driver that runs the stateless Lean executables
`run-model draw` and `run-model verify` as subprocesses, as `benchmarks/one_stage` runs `flock-verify` today.

    python drive.py --exe PATH --depth D --units K --corrupt F [--grind] [--trials T]

* honest driver: one draw, the developer's openings of the drawn units, one verify;
* ``--grind``: the driver is part of the adversarial environment (ruling 2: Python is the environment). It reruns the
  draw until no drawn unit is a corrupted row, then verifies that draw. Each draw is a fresh OS draw, and `verify`
  can't tell which draw was first.

``--lean-drives`` runs the same corrupted table under the auditor's own Lean program instead (`run-model auditor`), where
the environment can only answer requests: the program draws once, after it holds the root.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from developer import Table, write_frame

HERE = Path(__file__).resolve().parent


def lean(exe: str, *args: str) -> tuple[int, str]:
    p = subprocess.run([exe, *args], capture_output=True, text=True)
    return p.returncode, p.stdout


def draw(exe: str, depth: int, units: int) -> dict:
    _, out = lean(exe, "draw", "--depth", str(depth), "--units", str(units))
    return json.loads(out.split(" ", 1)[1])


def verify(exe: str, table: Table, depth: int, units: list[int]) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".frames", delete=False) as f:
        for u in units:
            write_frame(f, table.opening(u))
    _, out = lean(exe, "verify", "--depth", str(depth), "--root", table.root.hex(), "--units",
                  ",".join(map(str, units)), "--openings", f.name)
    Path(f.name).unlink()
    return json.loads(out.split(" ", 1)[1])


def python_drives(exe: str, table: Table, depth: int, units: int, grind: bool) -> dict:
    tries = 0
    while True:
        tries += 1
        d = draw(exe, depth, units)
        if not grind or not table.bad.intersection(d["units"]) or tries >= 100_000:
            break
    v = verify(exe, table, depth, d["units"])
    return {"accepted": v["accepted"], "draws": tries, "why": v["why"]}


def lean_drives(exe: str, depth: int, units: int, corrupt: float, seed: int) -> dict:
    p = subprocess.run([exe, "auditor", "--depth", str(depth), "--units", str(units), "--", sys.executable,
                        str(HERE / "developer.py"), "--depth", str(depth), "--corrupt", str(corrupt), "--seed", str(seed)],
                       capture_output=True, text=True)
    lines = {ln.split(" ", 1)[0]: ln.split(" ", 1)[1] for ln in p.stdout.splitlines() if " " in ln}
    v = json.loads(lines["VERDICT"])
    return {"accepted": v["accepted"], "why": v["why"], "replay": lines.get("REPLAY")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--units", type=int, default=16)
    ap.add_argument("--corrupt", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--grind", action="store_true")
    ap.add_argument("--lean-drives", action="store_true")
    a = ap.parse_args()
    table = Table(a.depth, a.corrupt, a.seed)
    runs = []
    for t in range(a.trials):
        if a.lean_drives:
            runs.append(lean_drives(a.exe, a.depth, a.units, a.corrupt, a.seed))
        else:
            runs.append(python_drives(a.exe, table, a.depth, a.units, a.grind))
    mode = "lean-drives" if a.lean_drives else ("python-drives, grinding" if a.grind else "python-drives, honest driver")
    accepted = sum(r["accepted"] for r in runs)
    print(json.dumps({"mode": mode, "depth": a.depth, "units": a.units, "corrupt": a.corrupt,
                      "bad_rows": len(table.bad), "trials": a.trials, "accepted": accepted,
                      "accept_bound": round((1 - a.corrupt) ** a.units, 6),
                      "draws_per_run": [r.get("draws") for r in runs][:5], "example": runs[0]}))


if __name__ == "__main__":
    main()
