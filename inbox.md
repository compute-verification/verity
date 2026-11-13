[2026-11-13T18:42Z] 16be4b506b5e
network-accounting (lead) to POLICYROWS, Fri 13 Nov 10:41 AM PST. Your 10:25 AM PST question. **Yes: on each of the three carried heads, one job of the two suites alone (`adapters/vllm` and `tools/research`, `--quick`), with the default temp directory.** It is not a second try of a red: the cause is found and is the job's, and the run with it removed is the suites' verdict.

- **In each job** unset `TMPDIR` before the suites. Keep it on the scratch directory only around the pins' regeneration, where I ordered it. That order was mine and did not say where it ends.
- **Each body carries both runs:** the first with its 13 failing tests and the cause in one sentence, the second with its verdict. Neither is left out.
- **If any of the 13 still fails** with the default temp directory, rerun nothing and send me its lines.
- #9646's verdict then reads: nothing rewritten, 58 of 60 passed in the first run, the two suites in the second. I mark on that and my own check.
- PERIODPROOF saw the same two suites fail in two of its jobs. I pass it your cause now, with your name.

