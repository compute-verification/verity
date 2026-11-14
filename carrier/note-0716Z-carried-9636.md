# CARRIER, @NOW@ Sat 14 Nov: #9636 is carried onto #9458's head and `main`, at `4f51d8ff9b16` on my scratch branch; three check runs wait in line; not yet on #9636's branch

Clock read with `date -u` at @NOW@. **The carry is made and pushed to my scratch branch only. #9636's own branch is untouched (`a287d4bf1247`) until the runs are green. All three runs are waiting for a slot, so 10:30Z becomes about 11:30Z [guess]; 20:00Z stands.**

**The carried head** [run]

- `cursor/proofs-carrier-9636-carry-9257` at `4f51d8ff9b161a922570980d44c5c73d965b9ec6`, pushed. It descends from #9636's head `a287d4bf1247`, so the push to #9636's branch is a fast-forward.
- It holds #9458's head `1b4b9acc84f3` (the one you marked) and `main` at `512644d14fbc`.
- `restack.py` crossed all 30 moves, never by hand: 4, then 25 up to Dither's, then the clear audit's, each pass ending `clean` (runs `r20261114-055226-0840` and `r20261114-061855-bc20`, vy-nebius-cpu-5, both done, rc 0). No stale pin (`pins.py --stale`, exit 0).

**The settlements: three commits of mine, seven hunks, every one both sides** [run]

| Commit | What it merges | Hunks | Where |
|---|---|---|---|
| `b443ddb01f1f` | `a275f10724a9`, #9458's line at the terms move's base | 4 | `run`'s signature, its two calls in `service.py`, a docstring in `Spot/Pieces.lean`: `certified` beside `family` |
| `cee0a04f27c7` | `1b4b9acc84f3`, #9458's head | 3 | `run`'s signature and its two calls: `certified` beside #9555's `key` and `main` |
| `4f51d8ff9b16` | no merge: one file to `main`'s side | 0 | `tests/lint/allowlists/repo_retired_terms.json` |

- `test_stop_audit.py` at the terms move's base is #9458's own settlement, taken with `a275f10724a9`. I settled none of #9458's five files.
- The merge of `main` (`e3eb62b196db`) had no conflict.
- The last commit: the reruns of Dither's move and the clear audit's left three entries of the retired-terms list in another order than `main`'s, same counts. #9636 changes no entry, so I took `main`'s file whole.
- [derived] None of the seven hunks asks what the check should do: each side adds a keyword argument or a clause the other does not touch.

**What the carried head is, against `main`** [run]

- 48 files: the 38 that #9458's head changes (with #9810 and #9717 under it) and #9636's own 20, 10 in both. No file outside those.
- **#9636's own difference is what it was.** Against #9458's head merged with `main`, it is the same 20 files with the same added and removed line counts as at the old head, but one: `service.py` adds 19 lines where it added 18, the wrapped `certified=e.certified)`.
- **The README is under its cap**: 47,656 bytes against 49,152. The carried #9458 is 47,542 and #9636 adds its one line of 114. My 06:00Z figure of 1,459 was wrong: it compared two heads that held different `main`s.

**The locks, as committed (merged by the lock's driver, no run yet)** [run, my compare of each record's `signature`, `type_hash`, `unfolded_hash`]

- **No guarantee's own record differs from `main`'s**, in `security/lean-audit.json` (1,349 records) or `security/Properties/lean-audit.json` (715). The old head said the same against the `main` it held.
- Against #9636's old head, 145 and 65 own records differ, and every one of them is `main`'s own, from landings since `4a7291116e11`.
- Whether a records run writes these same locks is the run below.

**Runs, all at `4f51d8ff9b16`, all waiting for a slot at 07:15Z** [run]

| Run | Node | What | In line |
|---|---|---|---|
| `r20261114-070915-cc7b` | vy-nebius-cpu-10 | `lean_changed.py --records` from the SPOT verifier's Lean with `--update` (it audits that package, the certifier's verifier, `security` and `security/Properties`, each with `security/Proofs`), 120 GB; then the nine Lean files' tries with `--siblings` | 8 ahead |
| `r20261114-071129-b00e` | vy-nebius-cpu-5 | Markdown caps gate, ratchet gate, retired-terms lint, `suites.py --changed 512644d14fbc --quick` | first, waiting for memory |
| `r20261114-071306-9328` | vy-nebius-cpu-4 | the SPOT verifier's two suites whole, slow tests included | 4 ahead |

- [derived] Why the slow tier: `one_stage/verifier.py`, which I settled by hand, is what the slow `test_a_history_source_is_the_root_the_verifier_computes` drives, and the Lean executable the slow `lean/tests` build is of the merged sources.

**Others** [run, 07:15Z]

- #9513, #9810, #9717 and #9549 are not on `main`. No `ChargeOnce.lean` pull request yet. TWOLEG's pull request is not open.

**Not established**

- Any run's verdict: none has started.
- That the committed locks are what a records run writes.
- When a slot frees on vy-nebius-cpu-10: six slots, eight runs ahead, the oldest holder 96 minutes in.
