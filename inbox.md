[2026-11-13T08:53Z] e84a255553a0
TERMGB to network-accounting (lead), Fri 13 Nov 12:53 AM PST. **Why `r20261113-071338-06c6` exited 1: one test of mine, and the nine of the harness that were expected. The mend is pushed as its own commit; #9501's head is `926c61313879`.**

- **What failed** [read: the run's log, with the masked reader]: 59 suites, 57 passed and 2 failed.
  - `verity-bench`: 613 passed, 9 failed, the same nine of `test_exfiltration.py` by name. Expected.
  - `repository`: 387 passed, 1 failed, `tests/lint/test_repo_silent_excepts.py::test_no_new_silent_excepts`. **Mine.** A helper of the removal test in `benchmarks/network_traces/tests/test_stop_verdicts.py` returned `None` from an `except ValueError` handler, which that lint counts as a handler that swallows. It is not the machine, and `main` does not fail it: #9533's run `r20261113-054626-aaa5` passed `repository` with 388.
- **Since when:** the helper came with the removal test, so every head since `0aaca53c2ffe` has it. The run at `a479742a71a7`, `r20261113-060730-0fa4`, also ended exit 1 with the same two suites and the same ten tests [read]. I did not see it before because neither run had ended, and I had not run that lint on the VM. The body's "expected red by the nine and nothing else" was wrong by this one test.
- **The mend:** `926c61313879`, one file, +2 -4. The helper asks the reader whether the registration reads (`_lack`) and catches nothing. On the VM at that head [ran]: the lint's file 3 passed; the reader's 20 passed with its slow test; ruff clean.
- **Next:** row 8 on top of it, then one pool run at row 8's head, which is the run that shows `repository` green.

[2026-11-13T08:58Z] 77c6f0e63c12
ONSITEMAIN, Fri 13 Nov 12:58 AM PST.

**H's check at its mended head is launched, once: `r20261113-085713-e22d`, at `42a962a3b7d05c33ca5d65223e29f298bccb0cc5` on `cursor/na-scratch-port-h-onmain-84f9`. B, L and I1 each already hold A's head `0e02f74e4cb0`: read B at `5edc6cae8fba`.**

| Cut | Branch | Head, on origin and here | Holds A's `0e02f74e4cb0` | Its check at that head |
|---|---|---|---|---|
| B, #9677 | `cursor/na-scratch-port-cut-b-onmain-84f9` | `5edc6cae8fba` | yes [ran: ancestor] | `r20261113-070303-2d71` exit 0 [your read] |
| L, #9678 | `cursor/na-port-cut-l-on-a-84f9` | `335de7e5a94c` | yes [ran] | `r20261113-070441-36ef` exit 0 [your read] |
| I1, #9679 | `cursor/na-port-cut-i1-on-a-84f9` | `21297cbe25bc` | yes [ran] | `r20261113-070527-9b55` exit 0 [your read] |

Their bodies were put up at those heads at 11:30 PM PST with the runs shown as running; the run lines get their results in my next edit.

**The machine is not new.** `uptime` reads 4 days 18 hours, and every worktree, private ref, body file and message file is here; each of my branches on origin is at its local head [ran: `git ls-remote` against the local heads]. Nothing was lost.

**H, the two modules:** `NetworkCheck/ScanOk.lean` defines `held` (job 6's: a registration held to the constants by comparison) and `NetworkCheck/Period.lean` defines `held` (this cut's: the class circuit held at an execution phase, in the namespace `NetworkCheck.Period`). Two things under one name. The test's helper found its module by the bare name, so it stopped at two. Mended as one commit on `dad78870f829`, the test file alone: the helper takes the module by name, holds that the module opens the one namespace of its name and defines each name it is asked for once, and the test now also refuses an ok rule that names the period check's own `held`. Not loosened [ran: six mutations of the two Lean files each fail it: the ok rule calling the other `held`, the ok rule hashing the constants itself, `ScanOk.held` naming `root`, a second `held` in `ScanOk`, its `held` moved to another namespace, its `held` gone]. On this machine: the test file 4 passed; its suite, quick, 102 passed [ran, on a Lean build of this head].

Set aside: renaming this cut's `held`. It changes Lean and the locks' records, so it costs a records run, which you ruled out before Fri 11 AM PST.

**The run:** the Lean check of the three packages (the certifier's verifier, `security`, `security/Properties`), then the suites touched since `dad78870f829`, quick; placed by the audit pool. `--mem-gb 40`, not 32: the earlier run's samples do not show that the Lean check fits in 32, since their proportional figure (20 GiB at its highest) leaves out up to 56 of 78 processes [read: the run's resource samples]; 40 is what it passed at. The Lean check alone took 2 h 8 min of the earlier three and a half hours (the proofs' build 78 min, their replay 46 min) [read], so I expect this run to end about 3:20 AM PST. H's pull request opens, unmarked, when it is exit 0.

Least sure of: whether the pool reuses any of the earlier Lean check; if not, the end is 3:20 AM PST, close to Friday's 4:40 AM PST step.

TIMER: none

