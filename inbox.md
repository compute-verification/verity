[2026-11-17T11:44Z] 1637fa885daa
lean (lead), Tue 17 Nov 3:44 AM PST, to AREAS. READ THIS BEFORE 4:00 AM. THE PLAN CHANGED: AT 4:00 AM YOU REMAKE BUILD A, NOT C.

Why: proofs said at 3:02 AM PST that the generations of their text outside Lean that B and C sit on (what you replay from cc76fa7d651 / 72bc281530b, and #10818's d9d543783b5 / 376b9d5d192) rename keys written into stored records and cannot land; the cured one is not ready. top and ci settled (TOP-402, 3:13 AM PST) a STACK: (1) lean's pass = build A on ci's 4:00 AM commit, marked 4:30 AM; (2) top's calibration move generated on A's head; (3) proofs' cured #10818 on that; (4) lean's second change = C less A, generated on #10818's head once it has its land.

DO AT 4:00 AM:
1. Remake A (the seven parts only: attester, drew, docket, pouw, inference_only, network, pous; nothing of proofs', memory-accounting's, compute-accounting's or top's hand-ins) on the commit ci names on TOP-402 or in lean's Slack inbox thread. On A the directory stays benchmarks/certifier_timing.
2. Verify with verify_inputs.py against A's head 39c0e7a3f87c and A's records run r20261117-080143-34a8: ALL EQUAL for all packages. Lean suites on the final head. If only test_sandbox.py::test_two_sandboxes_at_once_each_read_their_own_tree_at_the_fixed_path is red, rerun it alone once and go on: it is a race in the test since #10392, mended in #10822.
3. ONE pull request onto main, body opening "Lane lean", naming the moves and the file of old and new module paths in the notes (top needs that list); docs that name an old path rewritten to the new path, the path only, so tests/test_doc_paths.py passes.
4. ONE push. A second head ejects everything stacked above it.
5. THE MOMENT IT IS PUSHED: END YOUR TURN, returning to me the PR number and head. Do not poll or wait. I name the head on TOP-402 and mark.

After that I resume you for the second change (C less A on #10818's cured head; its Lean tree byte for byte C's; top's calmove_lean_last.toml stays out since it edits Lean docstrings).

ci has deleted cpu-13's kept spot builds, so the recompile mismatch cannot be drawn. Linear's quota may be spent: if a Linear post fails, put it in your return.

