[2026-11-11T17:44Z] 91721fc0b884
TERMGB to the network-accounting coordinator. NET-326, #8777. Wed 11 Nov, 9:43 AM PST.

**Head `7d82236c6b50`, pushed. The merge of `main` was clean: no conflict, no file resolved by hand. The row is its own commit on top. One run is under way, `r20261111-173407-45a4`. No mark.**

## 1. The merge

- [run] `git merge origin/main` at `219763135681`: clean. Merge commit `30f8f20accda`, parents `663befe492a6` and `219763135681`. The driver was already set in this clone to the line you gave.
- [run] `08c62377327f`, #8579's `e525c7d1734b` and #8568's `8fb2b94ee4b4` are all ancestors of it.
- [run] `git diff --stat origin/main...30f8f20accda`, the merge alone: the four files you expected, and nothing under `benchmarks/spot_figures/`.

  ```
   benchmarks/certifier_term/README.md                |  23 +++-
   benchmarks/network_traces/leak_bounds.py           | 127 +++++++++++++++------
   benchmarks/network_traces/pyproject.toml           |  10 +-
   .../network_traces/tests/test_certifier_term.py    |  76 +++++++++++-
   4 files changed, 189 insertions(+), 47 deletions(-)
  ```

  Its patch-id without context is `67e03c7b1b80`, the one it had before the move, through it and last night.
- [run] `git diff --stat origin/main...HEAD` at the head, both changes:

  ```
   benchmarks/README.md                               |   2 +-
   benchmarks/certifier_term/README.md                |  45 ++++--
   benchmarks/network_traces/leak_bounds.py           | 173 +++++++++++++++------
   benchmarks/network_traces/pyproject.toml           |  10 +-
   .../network_traces/tests/test_certifier_term.py    | 160 +++++++++++++++++--
   5 files changed, 309 insertions(+), 81 deletions(-)
  ```
- [run] Pushed as a fast-forward of `663befe492a6`. `main` has since moved to `cc6ee7446278`; I merged once, as asked.

## 2. The row `design, no stop block`

Commit `7d82236c6b50`: four files, 125 lines added, 39 removed.

**All four identities hold as written.** None needed building around.

- [run] schedules = the `scan` row's; failed connections, jitter and entries = the `design` row's. Every scope, monitoring period and D_j, at the default, at `--world-requests 1.778e16 --response-ids 1024`, and at each with `--failure-rate 1e-6`.
- [run] **Your numbers match, none tuned.** At the table's inputs the row's schedules, as the command prints them:

  | scope | week, GB | month, GB | year, GB |
  | --- | ---: | ---: | ---: |
  | world | 468,856 | 2,009,384 | 24,447,500 |
  | datacenter | 125 | 535.9 | 6,520 |

  With `--failure-rate 1e-6` the failed connections are 69.74 GB a world year (69.7356).
- [run] At the default (the plan's count, at most 4,096 ids):

  | scope | week, GB | month, GB | year, GB |
  | --- | ---: | ---: | ---: |
  | world | 554,228 | 2,375,261 | 28,899,014 |
  | datacenter | 147.8 | 633.5 | 7,707 |
  | node | 0.01182 | 0.05068 | 0.6166 |

  The world's year as printed: schedules 28,899,014 GB, cells 0, jitter 0 at D_j = 1 and 1,847,615 GB at D_j = 2, term 28,899,014 GB at D_j = 1 (96,330 times the target) and 30,746,629 GB at D_j = 2, entries "the compute side's". It is half of today's scan (57,798,028 GB): the scan's schedules without its fail cells.

**How.** The smallest way that keeps `CHANGES` three:

- `terms(..., stop=True)`: only the lengths' change reads it. With the lengths held and no stop block the schedules stay each connection's.
- `Kind` gains `stop` (default true). The new row is `("lengths", "entries")` with `stop=False`, placed just before `design`, standing "a design with no run behind it".
- `all three` is still `tuple(CHANGES)`.

**Every existing row prints what it printed.**

- [run] The `--json` documents at `30f8f20accda` and at `7d82236c6b50`, at the four sets of inputs above: all 63 existing rows equal field for field in each, 9 new rows, and the only legend text that differs is `design`'s.
- [run] The printed tables' `row` column is wider, so padding moves. No number does.
- That is a comparison I ran, not a test in the tree: a tree test cannot read the code before the commit. What the tree holds is that the new row is the only one with `stop` false and that `stop` moves nothing but the schedules of a row whose lengths are held.

**The stop block's definition.** [read] `GLOSSARY.md` does not define it, and the words were nowhere in the tree (0 hits at `219763135681`). The text says it in one clause at first use: "a stop block, a proof block that checks where a response's ids stop". The tree says "sampled on a served audit", not "drawn": `draw` is a retired term.

**Where the count is now eight.** The module's docstring, `Kind`'s and `ROWS`' comments, the command's help (`leak_bounds.main`'s description, which `run.py` prints), `benchmarks/certifier_term/README.md` and `benchmarks/README.md`'s line for the command, and the tests. `run.py` itself lists no row and is unchanged.

**The tests** (`benchmarks/network_traces/tests/test_certifier_term.py`, 17 now):

- New: `test_the_design_with_no_stop_block_is_charged_the_scans_schedules_and_is_the_designs_otherwise`. The four identities on six sets of inputs, and the row's place, standing and flag.
- New: `test_only_the_lengths_change_reads_the_stop_block`. Over all eight subsets of `CHANGES`, `stop` moves only the schedules, and only where the lengths are held.
- Changed to eight rows: `test_the_table_on_a_hand_worked_world`, `test_the_failed_connections_are_charged_pooled_only_where_the_lengths_are_held`, `test_every_row_names_its_request_count`, `test_the_document_names_its_value_unit_commit_and_run`, `test_the_command_prints_the_table_and_its_json`.
- [run] Local at the head: that file 17 passed, `test_leak_bounds.py` 9 passed, `test_chain_record.py -k leak` 2 passed, ruff clean, the nine root files 112 passed.

**The default class today.** [run] At most 4,096 ids a response, 18 bits an id, an execution phase of 1,280 buckets of 0.1 s (128 s), so 13 bits a connection. #Σ_sync = 128, no lanes, failure rate 0. Request count `plan`, 1.7784008649596718e16 a world year. No default changed.

## 3. Against #8779's head `9b5d80511bdf`

Scratch only. Nothing pushed, both worktrees removed.

- [run] **Merged directly into my head, git stops in one file**, `benchmarks/network_traces/pyproject.toml`, the suite's `inputs` list.
  - Mine: the planner's files (`benchmarks/spot_figures/cards`, `cards.py`, `figure1.py`, `total_cost.py`, `breakeven`).
  - Theirs: `benchmarks/spot_figures/cards/workload.glm-5.3.json` alone, the line `main` has from #8579.
  - Why: `9b5d80511bdf` holds #8579's head from before the layout move (`fdb3b995260e`) and not the landed `e525c7d1734b`, so the merge has two bases. It is last night's conflict again. I did not close it.
- [run] **Merged the way it will arrive, both merges are clean**: `9b5d80511bdf` into `main` at `219763135681`, then that into my head (one base).
- [run] On that tree `test_exfiltration.py` is **10 passed, 1 failed: the one known failure and nothing new from the row.** It is `test_the_network_certifiers_part_is_beside_it_at_the_planners_requests`, now at **line 369** of #8779's head (it was line 185 at its old head): the planner's count asserted below itself. With that line reading `lb.inputs(requests="census")`, in the scratch only: 11 passed.

**What of #8779's reads every row** [read, and run on the scratch tree]:

- `benchmarks/harness/verity_bench/exfiltration.py` lines 455 to 456 put the whole `legend` and every row of the world's year in its document, and `render` (lines 580 to 585) prints one legend line and one table line for each. So its report already prints the new row, eight lines where it printed seven, with no change of benchmarks'.
- Its sums are over `TOGETHER` (lines 82 to 84), four names pinned: `scan`, `N2`, `all three`, `design`. The test pins that list at line 379. The new row is in no sum until benchmarks adds it there.
- The test at lines 370 to 372 compares its rows with `lb.document`'s, both read from `ROWS`: it passes with eight.
- No count of rows (7 or 63) anywhere in its files. `benchmarks/exfiltration/README.md` lines 30 and 35 list the rows in prose, without the new one.

## 4. The run

- [run] **`r20261111-173407-45a4`**, launched 9:34 AM PST at `7d82236c6b50` with `--on auto`, no node named. State `running` at 9:42 AM PST.
- The same command as `4bf6`: the six suites by name (`research`, `verity-graph`, `verity-network-traces`, `verity-network-e2e`, `verity-network-sim`, `benchmarks/one_stage`), then the nine root files.
- [run] The merge did not change the selection: `suites.py --changed origin/main --list` names seven, those six and `repository`, which the nine root files stand in for as before.

## 5. The PR

- [run] Title: `verity bench certifier_term counts the sampling plan's requests by default, and adds the row "design, no stop block"`.
- [run] Body replaced (21,448 bytes): both changes, the merge and the stat, the row's print at the default and at the table's inputs, what the row does not show, the scratch result, and the run as under way. **Its verdict lines are not in the body yet**: the run has not ended.
- [read] Unmarked, no label, head `7d82236c6b50`.

## Least sure of, and not looked at

- **The `lengths` and `all three` rows still print 0 schedules**, and `lengths`' text says a held length makes them 0 without naming the stop block. By the row's own reasoning they presume one too. You said only the `design` row's text changes, so I left them. Your call.
- The row's charge is the scan's rule, taken from your brief. I did not argue that a server picking the count can move a full schedule's bits, or that the charge is tight.
- Six of the nine root files were run before and after the row and print the same warnings. The other three (repository, boundaries, pythonpath entries) I ran only at the head: 73 passed.
- I did not read #8779's diff beyond its `exfiltration.py`, its test and its README.

WAITS ON: `r20261111-173407-45a4`, then #8779 on `main` for line 369.

[2026-11-11T17:46Z] 55f4444d1d5b
LEANMERGE to the network-accounting coordinator, Wed 11 Nov, 9:45 AM PST. The words for the period check's and the scan's hunks.

**Branch `cursor/na-cross-closes-period-4d1a`, one commit `e3b0f6c6e8af3c4d87bed6b03b6df5c3a4a5b25f` on `c094f1ef8a0f`. All 11 of my hunks have a word, and both hunks of `Period.lean` are closed in the same commit. Both hunks of `Witnesses.lean` have a word (not edited).** The two sides can both hold: no honest scan of the branch is refused by `main`'s checks once they are called with the lanes' nodes. [ran: the closed Lean builds, and Lean and the mirror agree on the period protocol's 29 tests, runs below]

My first push, `cursor/na-cross-closes-period-7c3e` at `b30915af6b62`, is superseded: ignore it. It is the same closes without the fix of point 1 below (13 lines of `period_case.py`), which a run found after I pushed. I did not amend or force-push; the new branch is one commit on `c094f1ef8a0f`.

**One thing the text merge broke that no hunk shows: a test of the branch's fails on the merged `period_case.py`, and I fixed it (point 1 below). Say if you would rather it stay red for the crossed head.**

## What meets

The branch (`bc3fe08931`) gave each lane its own input node for its slot entries: `prog<j>` reads row `⌊t / every⌋` of lane j's node, and `readsEntries` takes the lanes' nodes `pks`. `main` (#8547) added three checks to `scanOf` and its mirror: `ingressReturn` (`c60e4c13ef`), `argsApart` (`3d305cd8b5`) and `inputsRead` (`998f25c0a2`). `main`'s `e7487fc1ca` already made the last two take a list of the slot entries' nodes, "a scan whose lanes read slot entries' nodes of their own passes each lane's node; this scan has one", and calls them with `[pk]`. So the close calls them with `pks` and one lane's entries. [read]

The name: the merged Lean declares `NetTiming.BoundaryGates.Fits` (`security/Definitions/NetworkCertifier/SampledAudit.lean:130`) and no `NetTiming.Layout`. `main`'s strings `NetTiming.Layout.Fits` are stale on `main` itself. Every closed line says `BoundaryGates`; the closed tree holds no `NetTiming.Layout`. [read, ran `rg`]

**lean's rule test, for every hunk: no.** A piece of the branch's side of each hunk, searched with `git log -S` over all of `main`'s history on every path of that file's name, is in 0 commits (13 pieces; the control, the base's text, finds its 3 commits). So no hunk is an older version of a change `main` made again. [ran]

## The words

Line = where `main`'s side begins on `main` (MAINMERGE's number); in brackets the marker's line at `c094f1ef8a0f`.

| File | `main` line | Word | Why |
|---|---|---|---|
| `period_case.py` | 31 [33] | `main`, with the merged name | `main` `c60e4c13ef` widened the sentence to the three kinds of check; the branch `1b68820d08` only renamed `Layout` to `BoundaryGates` |
| `period_case.py` | 497 [540] | `main`, with the merged name | the same two commits: `main` added `_ingress_return` to `_scan`'s docstring, the branch only renamed |
| `period_case.py` | 502 [551] | closed lines | the branch's node a lane (`bc3fe08931`), then `main`'s `argsApart` (`3d305cd8b5`) and `inputsRead` (`998f25c0a2`) said of a lane's node |
| `period_case.py` | 578 [638] | closed lines (code) | the branch's `_reads_entries(…, pks, j)` (`bc3fe08931`), then `main`'s two checks (`3d305cd8b5`, `998f25c0a2`) with `pks` for `r0[1]` and `nb * E` a node for `L * nb * E`. `main`'s side whole raises `IndexError` at one lane: `r0` is a list of the lanes' refs on the branch [ran] |
| `period_case.py` | 1268 [1403] | `both`: `main`'s two lines, then the branch's three | `main` `c60e4c13ef` added the note on the view's width; the branch `bc3fe08931` reworded the lanes' lines after it |
| `period_case.py` | 1816 [2107] | closed lines | the branch's lane negatives (`bc3fe08931`), then `main`'s ending (`c60e4c13ef`, `3d305cd8b5`) with the merged name |
| `period_case.py` | 1859 [2160] | `main`, with the merged name | `main` (`c60e4c13ef`, `3d305cd8b5`) lists its new negatives `view` and `state` there; the branch `1b68820d08` only renamed |
| `test_network_period_protocol.py` | 314 [410] | closed lines | the branch's sentence (the entries' bits `0f5486f22a`, the lane negatives `bc3fe08931`), then `main`'s ending (`c60e4c13ef`) with the merged name |
| `test_network_period_protocol.py` | 334 [447] | `main`, with the merged name | `main` `c60e4c13ef` renamed the test to `test_each_scan_check_refuses_its_negative` and widened its sentence; the branch only renamed `Layout`. Nothing else names the old test [ran `git grep`] |
| `scan_zk.py` | 948 [1145] | closed lines | `main`'s docstring (`c2a8ea8689`, the tied value) said of a value a lane, over the branch's code line `lanes=cls.NR` (`bc3fe08931`) |
| `security/Proofs/NetworkCertifier/README.md` | 79 [79] | `both`: `main`'s four entries, then the branch's entry for `Witness.lean` | `main` `c071d8c9b4` added the four entries above the base's two-line `Witness.lean` entry; the branch `691ed547f7` rewrote that entry. The five paths exist and the merged `Witness.lean` is the branch's file whole, holding what its entry says [read] |
| `Period.lean` | [759] | closed lines | the branch's paragraph (`bc3fe08931`), `main`'s paragraph on the two ties word for word (`c60e4c13ef`), then `main`'s two paragraphs (`3d305cd8b5`, `998f25c0a2`) said of a lane's node |
| `Period.lean` | [852] | closed lines | the branch's lanes' check and `readsEntries … pks`, then `main`'s `own`, `argsApart prog own pks` and `inputsRead prog own pks (nb * ety.leaves)`, the refusals reworded. `nb` is defined above (`let nb := T / s.every`), `pks : List Nat` [read] |
| `Witnesses.lean` (not edited) | [82] | `both`: `main`'s `PooledLeakAccepted` bullet, then the branch's three lines | `main` `ad269a84f3` added the bullet; `main`'s last three lines there are the base's, which the branch reworded ("block", "incorrect"; `06fcce21f9`) |
| `Witnesses.lean` (not edited) | [517] | `both`: `main`'s `PooledLeakAccepted_nontrivial`, then the branch's two-line docstring | `main` `96e5550a4b` added the theorem above the base's docstring, which the branch reworded |

`main`'s new theorem in `Witnesses.lean` names nothing the branch declares otherwise: `PooledGood` is `main`'s (`PooledLeak.lean` is unchanged on the branch since the base, and the merged file is `main`'s), and `notLay`, `notStmt`, `notPart`, `Pr` and `Claims.prUniform` are declared the same on both sides and in the merged tree. [read]

In `_scan` at the code hunk the merged function defines `L` (`s.lanes`), `nb` (`T // s.every`), `E` (`leaves_of(ety)`), `own_args`, `pks` (a node index a lane) and `r0`, a list of refs, one a lane. [read]

## Lines beyond the hunks, in my files (lines at `c094f1ef8a0f`)

1. **`period_case.py:1701` against `period_case.py:1665` and `test_network_period_protocol.py:563`: fixed in `_strata_resize`.** `main`'s `build` (`c60e4c13ef`) draws one random value fewer for the scan's widths (the view is now 16 bits a state word, not a draw). Every rows-form case of a seed is therefore another case than on the branch. The branch's test `test_a_profile_that_is_no_sample_of_its_level_is_refused` (`8d5b3bbdca`) builds `rows-tied-strata-sizes` at seeds 0 to 3 and asserts that some hold strata above their level's size and some below; which, `_strata_resize` drew. On the branch seeds 1 and 2 are below; on the merged file none of 0 to 3 is, and the test fails on its last line (`r20261111-172617-180f`: 28 passed, this one failed). [ran] The fix: `_strata_resize` takes its way from the seed (`("more", "twice", "fewer")[seed % 3]`), so any three seeds in a row hold both; over 240 seeds the mirror refuses each for the strata's sizes, seed mod 3 = 2 below, the others above. [ran, in the mirror] `period_case.build` has other readers I could not run (`benchmarks/network_traces`, `benchmarks/one_stage`, `benchmarks/prove_path`, PoUW's period difftest, the remote composition's tests): a test of theirs that counts on what a seed holds can fail the same way.
2. **`scan_zk.py:1187`, `tied=("prog",)`: fixed to `tied=lane_kinds(cls)`.** `main` (`c2a8ea8689`) names the scan's input `prog` tied. The branch's `scan_root` (`bc3fe08931`) has no `prog`: its inputs are `prog0…`, which `lane_kinds` lists. One-stage's `_tied` takes a tied name as a parameter of the Program, so `prog` would be refused as a bad tied name. [read; argued, not run: `registration.py` holds markers]
3. `scan_zk.py:89-92`, the module docstring (`main`'s, the same commit): "names ``prog`` tied … `rests_on_period_check: ["prog"]`". Reworded for the lanes' `prog<j>`.
4. `period_case.py:2180`, `scan_negatives`' docstring (`main` `3d305cd8b5`): "`entries`, the egress state the slot entries' input node". Its code reads bucket 0's first entry port, lane 0's node on the branch. Reworded "lane 0's"; the code is unchanged and is refused by `argsApart`. [ran, in the mirror]
5. `test_network_period_protocol.py:464` (`main`'s): the same words, reworded the same way.
6. Read and left: `main`'s other lines in `Period.lean` (the three definitions at 508-543, the `ingressReturn` step) name nothing the branch renamed, and their docstrings are true of a node a lane. `scan_zk.py:1545` calls `WZ.firewall`, which the merged `warden_zk.py:321` declares (`zk_server_env` is `n2_zk.py`'s). `bad_scans` still holds the branch's lane negatives and `scan_negatives` both sides' (`view`, `state`, `shared`, `entries`, `unread` beside the older seven). [read]

## Lines outside my files (not edited)

- **`benchmarks/network_traces/tests/test_scan_zk_challenger.py:88`** (`main` `ab09a419f3`): `res["rests_on_period_check"] == ["prog"]` on a real round. On the merged scan it is the lanes' names, `list(sz.lane_kinds(setup.cls))`, with the fix of point 2; without it the registration is refused. The docstring at 44-45 says the same. This file has no hunk, so nobody closes it unless told.
- `tests/test_scan_zk.py:469,471` and `tests/test_chain_record.py:450,454,455` pass `["prog"]` in as data and read it back: they hold either way. `chain_record.py:37` is prose.
- `verity/protocols/certifier/prover/ingress_scan.py:63` and `verity/protocols/certifier/verifier/README.md:372` say "row j·nb + ⌊t / EVERY⌋ of one slot-entries input". That text is the base's and both sides': the branch's own older prose, not the merge's.
- The proofs that walk `scanOf` step by step (`security/Proofs/Properties/NetworkScan/Accept.lean`, `Causal.lean`, `Fits.lean`) merged with no marker. `main`'s `998f25c0a2` changed `Accept.lean` by one line for its own steps. Whether they go through over the closed `scanOf` I did not check: that needs a build of `security`. The locks' records of `scanOf` change.

## What I ran

- On each closed file: no marker, `py_compile`, `ruff check --isolated --select E9,F63,F7,F82` (and all of `F`): clean. No closed line is over 120 columns. [ran]
- **The closed `_scan` in the mirror, in a scratch tree** (the branch's side `c08d034460b0` with the closed `period_case.py`; not the merged tree, whose `warden.py` and `registration.py` hold markers). Over 180 generator-form scans of 1 to 3 lanes: every honest scan accepted; each of `scan_negatives`' twelve refused with its own fragment (2124 refusals); the branch's lane negatives refused (`skew` by `readsEntries`, `shared` by the lanes' check, `lanes + 1` by the parameters). Control: with `main`'s side whole in the code hunk the same script dies on `IndexError`. [ran]
- **`r20261111-172327-a08c`: `lake build NetworkCheck` in the closed merged tree** (at `b30915af6b62`, whose `Period.lean` is byte for byte the one of `e3b0f6c6e8af`, sha256 `8a6fd6583bcc…`): "Build completed successfully (67 jobs)", rc 0, no warning in `Period.lean`. The package requires no Lean file that holds markers (its closure: Flock's verifier, the NCI check, the interface, the certifier program, LMS, the identity hash, the difftest library). [ran]
- **`r20261111-173610-3878`: the period protocol's difftest, Lean against the mirror: 32 passed, 0 failed, 293 s.** The network check's suite, fresh, the tests of `test_network_period_protocol.py` (29) and `test_slot_entry_pins.py` (3), at a scratch commit: the branch's side `c08d034460b0` with the three closed files (`Period.lean`, `period_case.py`, the test file). It is not the merged tree, whose Python holds markers; but the Lean packages the network check requires are the same files in both, `Period.lean` aside, which is the closed one in both (`git diff --stat` over the closure: empty). So the closed `scanOf` and the closed `_scan` answered alike on the honest scans, on `bad_scans` and on each of `scan_negatives`'. What it does not cover: `main`'s Python outside the three files. [ran]
- `r20261111-172617-180f`: the same suite before the fix of point 1 (the period protocol's file alone): 28 passed, 1 failed, the coverage assertion of point 1. [ran]
- No test of the merged tree can run: `period_case.py` imports `warden.py` and `one_stage/registration.py`, which hold markers. [read]

## What I did not do, and what I am least sure of

- No build of `security`, so nothing on the proofs over the closed `scanOf` (above).
- The `tied=lane_kinds(cls)` fix (point 2) is read, not run: a scan round needs one-stage's registration, which holds markers.
- I read `main`'s whole change to each of my files since the base and every place where one side's line lies within about ten lines of the other's. I did not read the branch's 400 lines of its own in each file for errors that are the branch's alone.
- The scratch commits are on my VM only: `7441e9fb58d0` is `c08d034460b0` with `Period.lean`, `period_case.py` and `test_network_period_protocol.py` of `e3b0f6c6e8af`; `69e6498448`, its parent, the same of `b30915af6b62`.

