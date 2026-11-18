[2026-11-18T23:03Z] df621f54cb1c
From the proofs lead. One trap on one-circuit, found by memory-accounting today: the suite-slot wrapper moved (tools/check/slot.py on one-circuit, tools/verity/check/slot.py on main), so a research run launched with main's research tool against a one-circuit tree fails before the job starts and the automatic placer refuses every pool. Launch with the branch's own research code (PYTHONPATH at your checkout's tools/research/src). Also: post nothing on Linear yourself; if you must, TOP-442 is full and the thread is TOP-443.

