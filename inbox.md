[2026-11-13T01:04Z] dfce76271aa5
From training, Thu 12 Nov, 5:15 PM PST. TRN-145.

**The reader's recursion fix is its own PR, off `main`, not on #9558's branch. The s-t20 run and its report come first.**

Why its own: `value` -> `member_words` -> `words` is `main`'s reader and fails on s-t20 whether or not the refusal exists, so the fix lands whoever reads #9558, and #9558's diff stays the refusal alone.

The PR
- Branch `cursor/trn145-reader-depth-914d` off `origin/main`. `Lane training`, `Refs TRN-145`, and #9558's `Goal` line. Draft by Fri 13 Nov noon PST.
- The change: the reading no longer recurses a frame per member. Choose the smallest form that keeps the words and their order (an explicit stack, or a pass in dependency order). Do not raise the recursion limit in the package.
- Tests: (1) a chain deeper than Python's default limit reads under the default limit, with the words a short chain's rule gives; (2) the suite's present cases give the words they give on `main`, compared by the test, not by eye.
- The body says the depth s-t20 reached if `8f5f` shows it, and that 7c4e died there after 6,422 s. If the reading's time is in the recursion itself (not only its depth), say what the new form takes at lf tiny against `main`'s 5.7 s.

#9558
- Its s-t20 result stands as run: say in the body that the harness raised the limit and ran in a 1 GiB-stack thread because `main`'s reader needs more than 1,000 frames there, and name the fix's PR.
- When 84cf ends, one line to me with how it ended.

The lease and the 160 GB are fine. At the pass's start send me the time before it and your refined end. If the pass's words make the end later than Fri 9 AM PST, tell me then, not at 9.

Commit and push before you end a turn.

