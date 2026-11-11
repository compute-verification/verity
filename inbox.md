[2026-11-11T02:30Z] 03747542ea45
REMOTEPY to coordinator, Tue 10 Nov 6:30 PM PST. **The run at the head is `r20261111-021908-9be5`. Its first two parts are in and pass; its opt-in tests are still running, about 25 minutes more. I end my turn on it.**

- **Branch and head:** `cursor/net-remote-nonlean-9f54` at `d38d1641cd03`, pushed 6:17 PM PST; the commit is `d38d1641cd` on vllm's `c1c9244e02b9`. Commented on NET-314 at 6:20 PM PST; my note is synced.
- **In so far, at `d38d1641cd03`** [run: `r20261111-021908-9be5`, its own lines and the suite's record]:
  - the `benchmarks/prove_path` suite: "1 suites: 1 passed (0 cached), 0 failed; 1.6 min of test time", 168 passed and 9 skipped;
  - `tests/test_vllm_boundary.py` and `tests/test_vllm_public_api.py`: "6 passed in 16.29s".
- **Still running in it:** `VERITY_PROVE_PATH_REMOTETIE=1` on `tests/test_pearltie.py` and `tests/test_remotetie.py`, then `VERITY_CERTIFIER_HEAVY=1`. Its last line will read "ALL rc suite=… root=… same=… optin_prove_path=… heavy_certifier=…". I expect "2 failed, 10 passed" (the two stale `test_remotetie.py` tests, as at `43c3ed558f31`) and "16 passed"; anything else is news.
- **A slip of mine in that run, mended:** its "SAME rc=1" line is not the code's. I sent the first version of my equality script, which needs a GPU device, so that part stopped at "Torch not compiled with CUDA enabled". The right version, sent alone at the head: "device_target equals the replaced statements in 4 of 4 cases" [run: `r20261111-022615-201b`]. So `same=1` in the last line of `9be5` is to be read with this.
- **Placement:** `--on auto` put `9be5` in the research pool on vy-nebius-cpu-7; it took a slot within two minutes.
- **Least sure of:** the new body on a GPU with vLLM, which nothing here runs. No test of ours calls it (`test_records_audit.py` stubs `device_target`).
- **Not run at this head:** our other nine suites and the root suite. They last ran at `48c8f46e7193`; this head adds the base's later commits, vllm's parts and the one line.

