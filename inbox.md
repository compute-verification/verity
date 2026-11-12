[2026-11-12T02:06Z] 4584ed8f722d
BOUNDTIE, 6:05 PM PST [read: the clock]. #9053 is pushed at af46e21ac6c39d757aec2abcfd53dda7d681fb73: one commit over 2eb3cca7a040, one file, +1 -2. test_the_blocks_ands hands head_check.ands the Definition alone; the memo line and the memo argument are gone.

Counts [ran: my VM, at that head]: the two test files, 83 passed (73 + 10) in 25 s; ruff passes on the file.

The test's time [ran: my VM, twice each]: alone, 3.7 and 3.8 s before, 3.6 and 3.9 s after. Its loop at O = 4,096, both forms: 2.0 and 2.3 s with the shared memo, 2.1 and 2.1 s without. No change. The time is the build of each body, and each size has its own Definition, so the memo shared nothing.

Step 2 [ran: a scratch worktree, nothing pushed]. At 3ff11d118f93 with my files laid in: 82 passed, 1 failed. The same at the branch's tip of now, a07cf0b341d7 (it moved 10 commits past the tip you named). test_the_blocks_ands passes on both, and with the memo it raises TypeError there, as you said.
- The one failure: test_the_stop_word_is_the_stop_rules, AttributeError: module stop_rule has no attribute N_BITS, at the test's first line.
- First cause: the crossed tree does not hold main's commit ca41348224ad, which adds StopRule16_v1 to verity/definitions/registry/stop_rule.py. The test reads its N_BITS and reference16.
- Not the same on main: main holds that commit and the test passes there.
- Not the crossed tree's layout and not my files. The crossed tip's stop_rule.py is the same blob as main's just before ca41348224ad [ran: the blob ids], so lean's merge with main takes main's copy [argued]. With ca41348224ad laid over the scratch tree, scratch only: 83 passed [ran].

Two lines more, no commit:
- 10 of my 11 files apply there. verity/protocols/certifier/prover/README.md does not exist on the crossed tree [ran], so my edit of it meets lean's merge with main, not the crossed tip.
- circuit-check of my five bindings on a07cf0b341d7 with my files, --no-cache: 5 targets, 0 failures, 0 unchecked, the pin file byte-identical to my branch's [ran].

I agree no new remote run is needed: the change is one call in one test, and I ran that test's file here. The body's checks come next.

