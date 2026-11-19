[2026-11-19T17:50Z] 744309d50300
From the circuits lead: your run r20261119-154143-1cdc was cancelled at its 2 h limit. If it was a suites run: the adapter whole suite (verity-vllm, tests/adapters/vllm/tests) does NOT end inside any limit on this branch and takes every other suite verdict down with it; leave it out of --changed runs and run instead the adapter test FILES that import what you touched, split so each run ends in under an hour, head beside base by test name. Write each suite log under $OUT so a killed run still leaves the finished suites results.

