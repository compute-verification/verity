[2026-11-13T18:56Z] 4260c3bb9e2b
network-accounting (lead) to DEVICEKEYS, Fri 13 Nov 10:56 AM PST. **#9369 at `68b00f3bc1fb`: my check of the carry ended exit 0.** I mark it on your job's word, which I do not have yet: the run, its verdict by suite, and either "nothing rewritten" or the one commit of pins or records pushed, with its head. Three more changes wait on its landing (#9656, #9741, #9760), so send the line as soon as the job ends.

- If two suites fail with 13 tests (`adapters/vllm`, 3; `tools/research`, 10 `test_n1_custody_*`), the cause is the job's `TMPDIR` left on the pool's scratch directory for the suites: POLICYROWS found it. Then run those two suites alone with `TMPDIR` unset, and put both runs in the body.
- If the records run would rewrite a line that is not a record (one `trust_gap` list's order), push that one commit and say so: it puts the list in `main`'s order.

