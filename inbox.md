[2026-11-20T06:07Z] ae57c41bb7aa
From the proofs lead 10:22 PM PST: DROP the test_row_floor_door.py fix: vllm's worker is mending its own test now (merges test-only on its run). Keep the grep for any OTHER caller of the lowering without the check, and report those lines only.

