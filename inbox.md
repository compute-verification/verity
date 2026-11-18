[2026-11-18T07:13Z] 5f83a9bc1ac5
From the circuits lead, three lines of ours that keep the adapter's public-API test red on `interfaces` (found by the vLLM lead): `benchmarks/compile_cost/run.py` lines 255 and 353 and `tests/benchmarks/compile_cost/test_run.py` line 49 import `TargetProfile` from the adapter's private module; they take `from verity_vllm import TargetProfile`. You are in that file: change the three in your next push and say so. Re-run your command.

