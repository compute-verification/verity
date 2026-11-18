[2026-11-18T08:04Z] fc8f43e9c417
From the kernels lead, a narrowing of your job, no reply needed:
- compute-accounting said at 12:03 AM PST on TOP-441: the 29 failing tests from the salts change are THEIRS and a worker of theirs is mending them now (tests to the code on the tip, salts handed in by the caller), run on a node before their push. So mend ONLY your two salts tests in tests/benchmarks/pouw/test_pearl_c_hash_host.py (~426-464), and the smoke script's expected xfail count if it changes. Do not touch the other 29; do not sort them. Merge the tip before you push, since their worker is in the same test files, and if their push has already rewritten your two, keep theirs and report that.

