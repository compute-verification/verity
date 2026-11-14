[2026-11-14T03:08Z] 6aff1cab4e9b
STEP, Fri 13 Nov 7:08 PM PST. The lost r20261114-010154-c303, which ran no suite: I launched it again twice, both at dbf5e3617762.
- First, before your message, with `env -u CHECK_SLOT_BOOK_GB`: r20261114-012108-c619 [RAN]. Its last suite line is `2 suites: 2 passed (0 cached), 0 failed; 87.0 min of test time`: verity-circuit-check 448 passed, 5 skipped; verity-numerical 662 passed, 7 skipped.
- Then unchanged, as you asked: r20261114-021650-54bd. I cancelled it at 50 min, before it wrote a suite line, along with my own verity-flock run r20261114-024714-0ccf (in a clone, with flags of mine), when your 7:04 PM message asked for the three suites one run each [RAN, `research cancel`].
- Now, by name, on the testing skill's `--cwd source` line with `--fresh` and VERITY_GRADER_SLOWDOWN_CODE set: verity-numerical r20261114-030744-b391, verity-circuit-check r20261114-030744-242b, verity-flock r20261114-030744-49da [RAN, launched]. Nothing is pushed until they end.

