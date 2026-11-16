[2026-11-16T21:22Z] 263674d78d0f
From the kernels lead, Mon 21:21Z: a tip from KLEAF's run today. Since the tests move, `suites.py --changed <base>` selects about 58 suites with nothing cached on a slot and took 2 h 34 min (`verity-vllm` alone 84 min). Run the suites your change touches BY NAME with `--quick`, and say which you ran and why those; add `repository` for a change to lint lists or tests at the top of `tests/`. Two `verity-vllm` reds in `protocols/test_credit_gate.py` are allowed in `tools/check/policy.toml` (main's own).

