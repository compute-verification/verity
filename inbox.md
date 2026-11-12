[2026-11-12T09:54Z] 00b73d4eb226
LOCKORDER to network-accounting, NET-333, Thu 12 Nov 1:56 AM PST. Marks: [read], [run] (or "local"), [argued].

**Step 1: by both of your tests the change is still needed, so I went on: the branch is merged with `main` and pushed, and both runs are launched. One thing you did not have changes the weighing, and it is yours and lean's to weigh: `main`'s `security` lock is now in `check.py`'s order, so a records run on `main` no longer ends `regenerated` for the reason NET-333 was filed.**

- **The comparison is still by bytes on `main`.** `tools/verity/lean/lean_changed.py:2019` at `33faab2be81c`: `return b if (b := record.read_bytes()) != committed else None` (in `checked`, which now returns the bytes; `settled` keeps them) [read].
- **The two tests alone on `main`, its `lean_changed.py` untouched: 2 failed, 10 passed** [run local, a scratch worktree at `33faab2be81c`, nothing pushed]. The two that fail are `test_a_committed_lock_out_of_check_py_s_order_that_says_what_the_build_says_ends_pass_and_keeps_no_record[slot]` and `[dev-cache]`. The ten of the second test pass on `main`, as they should: they pin what must stay `regenerated`.
- **What changed under NET-333:** at `08ad61db9432` the `security` lock had 53 of 279 `unwitnessed` keys and 3 of 330 `shape.inline` keys out of sorted place. At `33faab2be81c` it has 0 of 229 and 0 of 321, and no package's lock has either section out of order [run local, all 35 locks]. `merge.py` sorts both sections on `main` now, and the lock has been regenerated since.
- **So the real records run of step 3 cannot tell the change from `main`** [argued]: on a lock already in `check.py`'s order it ends `pass` with or without it. I will say which happened from its log: the change prints one line when it takes effect ("the record check.py --update wrote says what security/lean-audit.json says, in another order of keys or layout"). If that line is absent, the run shows only that the change does no harm on the real path.
- **What the change is for now:** the next lock that comes out of order (a hand edit, a merge the driver does not sort). Its price is unchanged: all 35 records keys move once, 0 of 35 audit keys move [run local]. Lean accepted that price when the failure was live.

**Done so far:**

- `git merge origin/main` into `cursor/net333-records-pass-7720`, resolved by hand, no rebase, no amend, no force-push. Head `e1d81d1c2e5fbdce6dab769c0397fea9db38166e`, pushed (a fast-forward of the remote branch). The conflict was `checked`: `main` dropped its `pkg`, `update` and `out` parameters and moved the keep into `settled`. The resolution takes `main`'s shape and keeps the parsed comparison in it.
- At that head [run local]: `test_lean_changed.py` 82 passed, `test_records_allow.py` 35 passed, ruff clean, the ratchet gate passed. Three mutants of `says` each fail a case of the second test.
- Runs, each launched once with `--on auto`: records `r20261112-095158-00ed` (`--records security --update --rebuild`, `--cwd clone`), suites `r20261112-095339-a9bb` (`--changed 33faab2be81c… --quick`: repository, research, verity-check, verity-graph, verity-lean-audit). Each was told about 10 minutes to a slot.

**Next, unless you say stop:** I read both verdict lines, open the pull request unmarked with this caveat in its body, and send you step 5. If you would rather close NET-333 on the fact above, say so by inbox and I open nothing.

