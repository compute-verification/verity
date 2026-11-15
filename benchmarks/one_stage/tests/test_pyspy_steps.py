"""`pyspy_steps.py` on a toy py-spy Chrome trace and step trace: the window is the main load's run of full steps, a
thread's time between its samples goes to every frame on its stack and to the innermost one, and each thread is named by
its process and thread frames."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "pyspy_steps.py"
T0 = 1_000.0
S = 1_000_000


def _ev(ph: str, ts: int, name: str, fn: str = "", tid: int = 7) -> dict:
    return {"args": {"filename": fn, "line": None}, "cat": "py-spy", "name": name, "ph": ph, "pid": 1, "tid": tid, "ts": ts}


def test_the_full_steps_split_by_frame(tmp_path: Path) -> None:
    ev = [_ev("B", 0, 'process 1:"python serve"'), _ev("B", 0, "thread (0x7): MainThread"), _ev("B", 0, "loop", "/x/a.py"),
          _ev("B", 1 * S, "step", "/x/b.py"), _ev("B", 1 * S, "seal_wait", "/x/c.py"),
          _ev("E", 3 * S, "seal_wait", "/x/c.py"), _ev("B", 3 * S, "hash", "/x/d.py"),
          _ev("E", 5 * S, "hash", "/x/d.py"), _ev("E", 5 * S, "step", "/x/b.py"), _ev("B", 6 * S, "idle", "/x/e.py")]
    (tmp_path / "p.json").write_text(json.dumps(ev))
    (tmp_path / "t0").write_text(f"{T0}\n")
    steps = [{"kind": "step", "step": 0, "t": T0 + 0.5, "T": 1024}, {"kind": "step", "step": 1, "t": T0 + 1.0, "T": 4096},
             {"kind": "step", "step": 2, "t": T0 + 3.0, "T": 8192}, {"kind": "step", "step": 3, "t": T0 + 5.0, "T": 8192}]
    (tmp_path / "tr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in steps))
    (tmp_path / "tm.json").write_text(json.dumps({"start_unix": T0}))
    out = tmp_path / "o.json"
    subprocess.run([sys.executable, str(SCRIPT), "--trace-json", str(tmp_path / "p.json"), "--t0", str(tmp_path / "t0"),
                    "--steps", str(tmp_path / "tr.jsonl"), "--timings", str(tmp_path / "tm.json"), "--T", "8192",
                    "--out", str(out)], check=True, capture_output=True)
    doc = json.loads(out.read_text())
    assert (doc["window_s"], doc["full_steps"], doc["step_s"], doc["first_step"], doc["last_step"]) == (4.0, 2, 2.0, 2, 3)
    th = doc["threads"][0]
    assert th["thread"] == 'process 1:"python serve" / MainThread'
    incl = {f: s for f, s, _ in th["inclusive_s_per_step"]}
    assert (incl["step (b.py)"], incl["seal_wait (c.py)"], incl["hash (d.py)"]) == (2.0, 1.0, 1.0)
    assert {f: s for f, s, _ in th["self_s_per_step"]} == {"seal_wait (c.py)": 1.0, "hash (d.py)": 1.0}
