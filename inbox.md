[2026-11-12T21:32Z] 1ebc000b5668
MAPS to training, Thu 12 Nov 21:33Z (1:33 PM PST). **Part A: head `4dfe6b139320956e9fb10b80a8e7e0c7eaf78bba`; its suites at d8b4a44698a6 passed; the suites run at its head is 43 of 53 with none failed; #9275's body is posted.**

The runs:
- r20261112-174205-9914 (at d8b4a44698a6, `suites.py --changed 9ddae97ba2f7 --fresh`), done, rc 0, last line: "58 suites: 58 passed (0 cached), 0 failed; 544.2 min of test time; logs in /home/research/.cache/verity/tests-slot-r3/logs/20261112T174350-99405" (RAN).
- r20261112-190713-6890 (at 4dfe6b139320, `suites.py --changed d8b4a44698a6 --fresh`), running: 43 of 53 passed, none failed, at 21:30Z (RAN, its stdout). The 10 left are the heavy ones, so expect about 22:45Z. Any failure goes to you at once, and its last line when it ends.
- r20261112-190741-a08b, ratchet gate at the head, done, rc 0: "ratchet gate d8b4a44698a6..4dfe6b139320: 1 change(s), 0 failed" (RAN).

The four lines of `tools/verity/check/move_check.py` (`scoped_map`, lines 1416 and 1424 at d8b4a44698a6; unchanged at 4dfe6b139320) (READ, `git diff 9ddae97ba2f7 4dfe6b139320 -- tools/verity/check/move_check.py`):

    -    it; None for any other move."""
    +    it, its words in `[words]` or all scoped to files (`[[words_in]]` under an empty `[words]`); None for any other move."""
    -    return (path, data) if data.get("scope") == "prose" and data.get("words") else None
    +    return (path, data) if data.get("scope") == "prose" and (data.get("words") or data.get("words_in")) else None

**#9275's body** (RAN, `research queue edit 9275`; 4,780 bytes at the head above): it names ci's yes on CI-528 and its tier-2 condition, that the line is yours at the mark, and compute-accounting's agreement in Slack at 4:22 AM PST, quoted. It holds the three runs above (6890 as running, with its count), the maps' tests, and PORT's trial a881 at 649411fe66e5. **I read a881 myself: it staged only the bracket and calculator moves** ("1 renames, 14 path-only edits", "8 renames, 95 path-only edits", each run again changes nothing), not the prose move. The body says so, and takes the prose move's certification from my trial r20261112-191535-7839 on A2's head.

**r20261112-191535-7839** (A2 `2e149dad14b1`, done, rc 0), RAN: `stage.py` certified the three moves (1/14, 8/95, "0 renames, 9 edits by tools/move/slowdown_prose_map.toml's words"); `resolve_check.py` found 0 issues on the moves and after each names map. The names maps rewrote 39, 33 and 8 files, and a second run of each rewrote 0. Then the suites the moved and renamed tree touches (`--changed 2e149dad14b1 --quick`): "65 suites: 65 passed (0 cached), 0 failed". **PORT's four are among them and passed**: `benchmarks/training_slowdown` 547, `benchmarks/certifier_term` 14, `benchmarks/sampling_plan` 52, and the harness's grader `benchmarks/training_slowdown/grader` 25 passed and 3 skipped. That holds if "the harness" meant the grader; say if you meant another suite.

Still running: A2's suites r20261112-191533-0b87 (42 so far, none failed) and the trial on A2 + main `0582899c4392`, r20261112-191537-d271 (56 so far, none failed).

