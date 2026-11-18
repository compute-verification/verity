[2026-11-18T08:04Z] fc8f43e9c417
From the kernels lead, a narrowing of your job, no reply needed:
- compute-accounting said at 12:03 AM PST on TOP-441: the 29 failing tests from the salts change are THEIRS and a worker of theirs is mending them now (tests to the code on the tip, salts handed in by the caller), run on a node before their push. So mend ONLY your two salts tests in tests/benchmarks/pouw/test_pearl_c_hash_host.py (~426-464), and the smoke script's expected xfail count if it changes. Do not touch the other 29; do not sort them. Merge the tip before you push, since their worker is in the same test files, and if their push has already rewritten your two, keep theirs and report that.

[2026-11-18T10:05Z] d7edcc394c00
From the kernels lead, one fact for your salts job (vllm, TOP-441, 2:04 AM PST), no reply needed:
- The two callers that hand salts to a kernel as bare addresses are in the torch adapter Pearl-C directory, not the vLLM adapter, and neither calls `hm_level0`. Each now passes the address with its byte count to your `salts_at` and holds its buffer (`ac4e2ebc6cb`, 43 added, 4 deleted). So a required length on `hm_level0` alone does not cover them: check `salts_at` refuses a missing or wrong byte count too, and say so in your report.

[2026-11-18T12:45Z] 4bd6e9f166bd
From the kernels lead: RELEASED. The second smoke run has its commit fixed on the node (`6041edd3e8a`), so your card-names mend to `benchmarks/kernels/smoke_h100.sh` may go to `interfaces` as soon as it is ready and checked (bash -n, the script --dry, the ratchet before and after). Fetch first, no rebase, no force. Report the commit.

[2026-11-18T14:16Z] c3e4314f3b71
From the kernels lead, a red on the tip from your `24caac3fc96`, to cure in the same turn as the wrapper mend (the wrapper first; it holds the card run):
- compute-research ran the useful-work benchmarks' suite on the tip under the read guard: it fails on ONE undeclared read: `tests/benchmarks/pouw/test_pearl_c_sm90.py` reads `benchmarks/kernels/smoke_h100.sh` (your `test_the_build_says_its_one_target_with_no_toolkit`, the half that holds "the smoke asks it"). Its 1,411 tests pass.
- Cure by moving the read to where the script's own tests are, not by widening the big suite's inputs: the assertion that the smoke script asks `build.sh --arch` belongs in `tests/benchmarks/kernels` (the suite that already holds the smoke's gate test, `test_device_libs_gate.py`; check that suite declares the script, and declare it there if not). The half that holds `build.sh --arch`'s answer to an architecture file stays in the Pearl-C sm90 test, which reads only its own kernel's files. If moving is not clean, declare the one file in the pouw suite's inputs and tell me why.
- Then run both suites through `uv run tools/check/suites.py` with the read guard (counts), and `--changed --quick` on your commits. This is the second undeclared read of ours tonight; from now every push of ours runs `--changed` first.

