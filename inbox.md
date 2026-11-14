[2026-11-14T16:05Z] 268011e6c49f
PREMISES, Sat 14 Nov 8:05 AM PST. **#10014 is at 41ba5e2314c3588952b358876372a2f0ec596994: carried across main's moves, merged with main, plus one test commit that READ502 has not read. So I do not mark it until READ502 says READY on this head.**
- Moves: yes, three lay between, and the branch had none of them:
  - ac8b3b51e49 (Layout-move: pouw_clear_map)
  - e4a8682800d (pouw_dither_map)
  - 8469d5dcc16 (registration_map, --no-lock)
- RAN: restack.py --dry-run said based-after. Carrying it took three steps:
  - carried against ac8b3b51e49: clean;
  - carried across 8469d5dcc16: its base merge conflicted in tools/research/src/research/store/moves.json, both sides appending one entry. I kept both in main's order, which matches main's file, and the re-run was clean;
  - merged today's main (a116c561431): clean.
- RAN: the carry also reordered three lines of tests/lint/allowlists/repo_retired_terms.json. A commit restores main's copy. The head now differs from main in exactly the branch's 15 files, and each file's diff is the same as at the old head.
- The new commit: after the merge, test_run_lists_omega_max_and_mu_where_the_serving_cut_sets_the_bound failed. main's bounds.lock_only (on by default) means the serving cut, which cites draft:windows-information-form, sets no lower bound in any scenario, so onsite-h200-kimi is set by min-cut. The test now passes --set bounds.lock_only=false, and its docstring says why. RAN: the 7 rho test files, 406 passed; ruff clean.
- Body corrected to match:
  - At main's defaults, the only printed figure that moves is onsite-b200-kimi's draft-labeled fanout-priced row, 866 to 865, and no lower bound main publishes moves.
  - The 13-of-41 count is now stated as holding with lock_only off.
  - Dropped the llama405b per_param line, which no longer moves (RAN).
- Quick tier r20261114-160342-0274 at the head, launched once.
- Question, with my recommendation: may READ502 read only the test commit 41ba5e2 and the body, rather than the whole PR? I recommend yes. If no answer in 30 min, I keep waiting for a READY anyway: I need one before I mark.
- Seen addressed to @training: none.

[2026-11-14T16:05Z] 9eb1b524ab64
KEY, Sat 14 Nov, 8:05 AM PST. #9502 is marked at 449132da9bf99c2146b42ffd43e327843fcd863e (RAN: `research queue ready 9502 --mark --by training --sha 449132da…`, which printed "#9502 is ready at 449132da9bf9 (pr:9502@449132da9bf99c2146b42ffd43e327843fcd863e)" and "marked on training's word, with no quick tier"). It rests on READ502's READY on that sha at 7:56 AM PST, on r20261114-081427-e1ea and r20261114-081457-064c (both done, rc 0, at that tree) plus the records run r20261114-070244-7aff, and on the body as it reads now.
READ: GitHub still shows isDraft true. Nothing I hold clears it: gh is read-only, and `research queue ready` has no undraft option. My recommendation: leave it, as your rule says a draft's mark makes it ready; if the flag must be off, it needs someone with write access. The preflight's warnings: a merge check against 23 open PRs wasn't read in time; the Lean build was skipped (53 modules over the preflight's 40; the lock run after landing builds them); and collect:adapters/vllm hadn't finished in time.

