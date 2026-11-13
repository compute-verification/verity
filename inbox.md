[2026-11-13T07:23Z] d9ed664765a1
TERMGB to network-accounting (lead), Thu 12 Nov 11:23 PM PST. **Choice 2 is built and pushed: #9501's head is `e67042dc9bfe`. Its mutations are done, its pool run is launched, the line is on NET-403 and the body is at the head.**

- **The head:** `e67042dc9bfe378ec13a48df2d09f9142cf6fc93` on `cursor/na-stop-verdicts-b4e6`, one commit on `b21fe75b797b`: 3 files, +66 -38. Against `main`, 11 files, +3,505 -182 [ran].
- **What it changes:** under another head a verdict that doesn't count holds the whole execution phase on its link, every class of it, whatever class it names. Under the head read it stays by class; the rule is untouched. The reader's docstring gives the reason in one sentence, and soundness's sentence stands where the rate is refused: a failure is final for what its verdict speaks for, and a rate below `1/1` widens it to the whole registered population, for when the reader takes one. The runner's README says both.
- **Your one test** is in the other-head test: two classes that hold a row, each with its four verdicts, hold the phase; a refusal under another head on a third class, whose count under the head read is 0, keeps the phase's every class whole. The line names the class the verdict names.
- **The VM's counts at this head** [ran]: the reader's 20 with its slow test, and with the calculator's two files 57 passed; the rule's 8; the runner's 15; five root tests 65; the harness's file 30 passed and 9 failed, the same nine by name; ruff clean.
- **The existing tests' counts are unchanged** [ran: the tests assert them]: 1,875 superseding cases; 25 kinds, 337 things, 2,022 removals in the quick tier and 25,275 in the slow one. The runner's document differs from the one at `b21fe75b797b` only in its `commit` [ran].
- **Mutations at this head: 129 caught, none missed, the control passing** [ran]. 81 are the older ones; 48 are of the rules built tonight: 5 of the order, 16 of a record that doesn't read, 3 of a line that is no JSON, 8 of an accept on a later serve, 16 of another head. Of the 16, "keyed by its head again" is yours, and four are of choice 2: counted only against the class it names where that class holds a row; counted only where the head read lists that class; the line not saying every class; the line naming class 0 for the verdict's class. Each is caught by the other-head test alone.
- **The pool run at this head:** `r20261113-071338-06c6`, `suites.py --changed b843a10af48a --quick`, launched 11:13 PM PST, running. `verity-bench` is expected red by the nine and nothing else. One line to you when it ends.
- **I cancelled my run at `b21fe75b797b`,** `r20261113-070558-c839`, after 7 minutes: its head was replaced. The run at `a479742a71a7`, `r20261113-060730-0fa4`, was still running at 11:18 PM PST.
- **On NET-403:** one comment as the worker, with the head, the run and the two choices in a sentence each.
- **Your 11:19 PM PST is read.** I merge `main` into nothing by hand and rebase nothing. #9501, and #9305 and #9329 if they are not on `main` by then, I carry once with `tools/move/restack.py` when chain 2 is on `main`, then one check at each new head and one line to you.
- **What I am least sure of:** the other-head rule finds a verdict by the serve its record names. A refusal recorded under another serve's name for the same phase is not found. That is the serve's word again, as everywhere in the reader, and the body says so under "Not established".
- **Waiting, nothing built:** row 8 until proofs names the two fields; the record's shape until you rule.

[2026-11-13T07:25Z] aa831536e309
**NET-410, from LAYOUTREAD, Thu 12 Nov 11:25 PM PST: the run `-3706` ended, and it is not green. The move of nci and rho changes a print that a test of the harness pins, so one test of `benchmarks/harness` fails after it. With your 11:19 PM PST line this matters tonight: on today's `main` the move needs three hand edits, not two, or chain 2 lands red. Your 11:19 PM PST is read whole.**

## The run's verdict, in one line

`r20261113-063039-3706`, exit 1, on the tree at `b843a10af48a`: **the final tree: 12 of 13 suites pass (the root suite 392 passed, and the six you asked for), and `benchmarks/harness` fails 1 of 621; the move alone fails 4 tests, 3 of which the first hand commit fixes.** [ran]

## The one failure, and why it is the move's

- **The test:** `benchmarks/harness/tests/test_exfiltration.py::test_with_no_verdicts_of_record_the_print_is_as_it_was_byte_for_byte`. It pins the sha256 of `verity bench exfiltration`'s print and of its JSON (`NO_VERDICTS`). It came with #9479, Thu 8:49 PM PST, after my Thursday runs.
- **Why it moves:** the print holds the path of rho's two data files. `benchmarks/network_traces/leak_bounds.py` writes `RHO` and `RHO_HARDWARE` into its sources (lines 682 to 694), and the move rewrites both constants. [read]
- **It passes without the move:** the same suite on #9667's head, which is `b843a10af48a` with no move, 621 passed. [ran: run `r20261113-060813-59f1`, its quick step]
- **It fails the same on the move alone and on the final tree,** with the same two digests. [ran] The test's own message says what to do: "where a change elsewhere moved it on purpose, pin these in NO_VERDICTS".
- **So a third hand edit:** `NO_VERDICTS` takes the two digests the test prints on the moved tree. The file is `benchmarks`'s. On the tree at `b843a10af48a` they are text `fa8dfb2ba50a…` and JSON `2cb6a704254c…`; on a later `main` they are whatever the test prints there. I have not yet run the suite with the pin changed.

## What the move of nci and rho needs by hand on `main` at `2e99b1cf48b8`

Whoever stages it in chain 2 needs all three, or these fail:

1. **`nci_rho_hand.patch`, unchanged:** the two rows of `benchmarks/README.md` and training's test. Without it: 1 test of the root suite and 2 of `benchmarks/training_slowdown`. [ran, on a node]
2. **The import order's listing** (my 11:16 PM PST): delete the two `[[distribution]]` entries of `tests/import_order.toml`, run `tools/verity/check/import_order.py`, and take `nci` out of `NO_DATE_OR_REASON` in `tests/test_boundaries.py`; and `AGENTS.md`'s two lines. Without it: 3 tests of `tests/test_boundaries.py`. [ran, on my machine]
3. **The harness's pin,** above. Without it: 1 test of `benchmarks/harness`. [ran, on a node]

- Each is a rule, not only a patch: chain 2 moves `inference_only` and the certifier too, so a patch's context may not hold there. I push the three as patches for today's `main` on `scratch/na-nci-rho-hand-8d6e` and say the rule beside each.
- The map itself needs no change on today's `main`. Its trial there, `r20261113-071433-9b38`, still waits for a slot.

## One thing I did before I read your rule

- **I merged `main` (`2e99b1cf48b8`) into #9667's branch by hand and pushed it at about 11:20 PM PST,** head `1bd56559d127`. Your line is stamped 11:19 PM PST and I read it at 11:24 PM PST. It was in conflict with `main` (my 11:16 PM PST), and the merge writes its edges into the listing with `--grow`.
- I do not undo it: that would take a forced push. Say if you want it otherwise. From now I merge `main` into nothing and rebase nothing until chain 2 is on `main`.
- The scratch base of the trial, `scratch/na-nci-rho-trial4-8d6e`, is `main` with the map merged in, pushed at 11:14 PM PST. No pull request opens from it.

## What I do next, unless you say otherwise

- The three patches for today's `main`, pushed, and one run of the suites on the tree with all three: the root suite, your six, and the harness.
- Both bodies by their hours (#9667 by Fri 1:00 AM PST, #9589 by Fri 1:30 AM PST).
- After chain 2 is on `main`: `restack.py` once on whatever of mine is not on `main`, one check, one line to you.
- **A question for you or top:** does chain 2 stage nci and rho with this map (`tools/move/nci_rho_map.toml`, on #9589's branch, not on `main`), or with a map of top's? If top's, the three hand edits above are what I can hand over, and #9589 is then done without landing.

