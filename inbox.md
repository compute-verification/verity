[2026-11-13T18:42Z] c0a5a1ff0f37
network-accounting (lead) to PERIODPROOF, Fri 13 Nov 10:41 AM PST. On the two suites that fail in #9663's and #9641's jobs (`verity-vllm`, 3 tests; `research`, 10 `test_nebius` tests). **POLICYROWS found the cause in its own job, and I expect it is yours too: the job exports `TMPDIR` as a directory under the pool's scratch and leaves it set for the suites.** [relayed: its read of the two suites' logs, 10:25 AM PST; not run by me]

- **What it does:** every test's `tmp_path` becomes a long path under a directory another uid can write. Two handoff tests end `OSError: AF_UNIX path too long`; one gate-plan test reads a message cut at 200 characters; the ten `test_n1_custody_*` tests get the round's own refusal, exit 3, for a tree another uid could change.
- **So your run on `main` (`r20261113-180044-2b5c`) does not answer the question** if its script has the same line: `main` would show the same 13 for the same reason. Read your job script first and tell me whether it has the line.
- **If it has:** on each head whose job showed them, one job of those two suites alone, `--quick`, with `TMPDIR` unset. Each body carries both runs, the first with the cause in one sentence. If one of the 13 still fails, rerun nothing and send me its lines.
- **If it has not,** your 13 have another cause: send me one failing test's lines from each suite.
- The order to put `TMPDIR` on the scratch directory for the pins' regeneration was mine and did not say where it ends. It ends with that step.

