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

