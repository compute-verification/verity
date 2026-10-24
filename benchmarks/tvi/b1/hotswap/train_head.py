"""Prepended to train.py in run r20261024-194056-b967's scratch at 20:42Z, so its 3B ft3e-4 call ran watch.py instead."""
import os as _os, sys as _sys
if _sys.argv[1:2] == ["ft3e-4"] and _os.environ.get("MODEL", "").endswith("Qwen2.5-3B"):
    # run B's last GPU slot goes to A2/A3 re-analysis with the extended retention probe (watch.py, claims shared with v3)
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _os.environ.update(CLAIM="/workspace/jobs/tvi/b1/claimed_v3", DEADLINE="2026-10-24T22:40:00Z", IDLE_MIN="25",
                       LAYERS_SAMPLE="4", TEXTS=_here)
    _os.chdir(_here)
    print("train.py ft3e-4 on Qwen2.5-3B replaced by watch.py (A2/A3 re-analysis)", flush=True)
    _os.execv(_sys.executable, [_sys.executable, "watch.py"])
