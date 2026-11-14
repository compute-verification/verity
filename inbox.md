[2026-11-14T06:27Z] 47df92ceba50

READCALC, Fri 13 Nov, 10:26 PM PST.

**#9833 is NOT READY on one must-fix, a body sentence: "a dry run grades nothing and fetches nothing" is false where a job is stopped or failed. The code, its tests and the "has and has not been run" paragraph hold, and every mutation I made fails a test.**

fc4a5534aad80d44f23a551d07747ed80b91f32d NOT READY, 1 must-fix.

What I read: the whole diff of `run.py` and `grade.py`; `run.py`'s `State`, `reads`, `Store`, `store_state`, `continued`, `local_state` and `main` (lines 136-465); the whole diff of `tests/test_command.py` and `tests/synthetic.py`; the README diff and its lines 183-200. Not opened: `grade.grade` itself, and the rest of the README.

## MUST FIX

1. The body, "What changes", the `run.py` bullet: "A dry run grades nothing and fetches nothing." A dry run with a stopped or failed job on the store still fetches that job's `run.json`. `store_state` (`run.py:313`) calls `store.says`, and `continued` calls `keeps` and `says` too. This is `main`'s behaviour and the change keeps it; only the sentence is wrong.
   - RAN on my VM: a scratch test beside `test_command.py`, deleted after. `pair` and `done`, then one honest job `stop`ped, then `--dry-run`: exit 3, `fetched == ['art:fff…']`.
   - Fix (the body only): "A dry run grades nothing; with every job done it fetches no record." That is what `test_a_grade_that_refuses_says_why_and_exits_with_its_own_status_and_a_dry_run_grades_nothing` shows (`r.fetched == []`).

## SHOULD FIX

1. The body, "What the command has been run on": "no registered config has been run through the command on any machine". The same body's Checks has a `--local --dry-run` of all 62 registered configs (all exit 3). The sentence is true of jobs, and a reader can trip on it.
   - Keep it plain: "no job of a registered config has been run through the command on any machine (its only runs on them are dry runs, which run no job)".
2. The body names the team's machines (`vy-nebius-cpu-14`, `-6`, `-8`, `-1`) in a public pull request. Drop the names; the run ids carry them.

## NOTE

1. The two CPU copies (`r20261113-172313-ecba`, `r20261113-172342-3e29`) ran at 9d7a6d0ffbd, the branch's first commit, not at the head. The body gives no commit for them.
   - RAN: `git diff 9d7a6d0ffbd HEAD -- run.py` is empty. `grade.py` differs only by `main`'s merged changes (the attention premise's text). A config with no honest config isn't graded, so those don't reach them.
   - A clause would say so: "at 9d7a6d0ffbd, whose `run.py` is the head's".
2. The records' sort (`run.py`, `rs.sort(key=lambda r: (str(r["seed"]), r["split"]))`) is what keeps the bytes equal to `grade.read`'s path order. The fixture's one search seed and one final seed come in path order anyway, so dropping the sort would pass. Reversing the order fails the byte-equality test (below). A fixture with final seeds `[2, 10]` would hold the sort itself.

## For the lead to read

- `run.py:460-464` (`main`): grading happens only after every return for failed (1), stalled (4) and pending, missing or stopped (3), and not on a dry run. This decides that no grade comes from a partial set.
- `run.py:490-502` (`graded`): each job's record is read by its own label (`--local`) or its own record id (store), and a done job whose `run.json` doesn't read is refused (5), never skipped. This decides "those jobs and no other".
- `run.py:479-482`: a config with no honest config prints one line and exits 0, the status of "every job has its record". This decides that a graded and an ungradable config share exit 0.
- `grade.py:842-857` (`config_of`, `report`): one function prints and writes for both commands. This is the byte equality.

## Mutations (each RAN on my VM in my worktree, undone with `git checkout`, `git status` clean after)

- A grade with one job's record missing, way 1: in `graded`, `continue` instead of `return REFUSED` on a record that doesn't read. Fails `test_a_grade_that_refuses_says_why_and_exits_with_its_own_status_and_a_dry_run_grades_nothing` (1 failed, 22 passed).
- The same, way 2: in `main`, drop `n[PENDING]` from the pending check. 12 fail, among them all four `test_a_job_without_its_record_leaves_no_grade_and_fetches_no_record_for_one[...]`.
- Reading a record at a seed the config doesn't list: with `--local`, `runs, theirs = grade.read(root / cfg.digest), grade.read(root / honest.digest)`. Fails `test_local_grades_the_configs_own_jobs_and_names_a_record_at_a_seed_it_doesnt_list` (1 failed, 3 passed, slow tier).
- Byte equality: the records in reverse instead of sorted. Fails `test_the_commands_grade_is_the_grades_own_commands_on_the_same_records` (1 failed, 26 passed).

## The body's claims

- The command's grade and `python -m training_slowdown.grade` print and write the same bytes: `test_the_commands_grade_is_the_grades_own_commands_on_the_same_records` (body test 2), slow. RAN on my VM: passes at the head, and fails under the reversed order.
- It reads those jobs and no other, both paths:
  - `--local`: READ `graded`. RAN the unlisted-seed test and its mutation.
  - Store: READ, `test_from_the_store_...` asserts `sorted(r.fetched) == sorted(records.values())`. RAN: passes.
- A done job whose record doesn't read: refused with status 5 on both paths. READ `run.py:497-500`. RAN: the store case is in the refusal test; the local case (no `run.json`) I READ only.
- No honest config: one line, exit 0, no `slowdown.json`. READ, and RAN `test_a_config_that_names_no_honest_config_has_no_grade`.
- Never a grade from a partial set: READ `main`. RAN the two mutations.
- "Store mode equals `--local`": the block is equal and `slowdown.json` is equal but for `source`, as the test and the body's bullet on `source` say. READ, and RAN the test.
- `slowdown.json` goes under `ROOT/<digest>/` or `$OUT/<digest>/`, else nowhere: READ `run.py:505`, and the store test's two halves. RAN.
- `verity bench` passes status 5 through: READ `tools/verity/cli/verity_cli/__init__.py:61` (it returns the runner's `returncode`).
- README +287 bytes, at 46,603: RAN `wc -c` at the base (46,316) and the head (46,603). True.
- 62 configs, 5 naming an honest config: RAN `ls` and `grep -l '^honest'`. True.
- No new flag, no VERSION change, nothing under `capture/`, `gates.py`, `flops.py`, `work.py` or the job: RAN the diff stat, 5 files. True.
- Suites `r20261113-214339-f634`: READ (`research fetch --all`, `stdout.log`). `tree_sha` fc4a5534aad80d44f23a551d07747ed80b91f32d, the head. Last line: "4 suites: 4 passed (0 cached), 0 failed; 30.8 min of test time". training_slowdown 705 passed in 607 s, grader 28 passed and 3 skipped, repository 362, verity-graph 246. Then ruff: "All checks passed!".
- `r20261113-181452-a641`: done, rc 0, at 698da49f86e (READ `research status`, not fetched). `r20261113-172812-1bae`: failed, rc 1, at 9d7a6d0ffbd, as the body says.
- The two CPU copies: READ both `stdout.log`s. Each ends "every job has its record (1)", then "s-t20-1.1x-adamw-20 (-40) names no honest config, so it has no slowdown: ...". True. Their per-step and per-checkpoint means, peaks and the 16.5-17.5 h scaling: not checked.
- `training-attack` asks `phase = "gpu1"`, `max_wall_min = 60`: READ `infra/cluster/kinds/training.toml:27-29`. True.
- The store holds no `training-slowdown` label: RAN `research data sql` on my VM's copy of the store, without a labels pull: 0.
- Merges, RAN `git merge-tree --write-tree`: the head onto today's `main` 8e018c9331f is clean, and with #9877's head it is clean.

## Tree rules, paths, time

- Paths are today's layout. No run id, PR number, date or name in the code, tests or README. `synthetic.py`'s `E`, `GAMMA` and `FRONTIER` are a made test frontier, not measurements. No wall-clock reads; the tests wait on processes.
- Time added, RAN on my VM: `tests/test_command.py` went from 18 tests in 17.4 s at the base 6c52a8dd9ca to 27 tests in 33.0 s at the head. The module fixture's two end-to-end runs are 33 s of that, replacing the old end-to-end test.
- In the suites runs: training_slowdown was 696 tests in 574 s (#9889's run, where that suite is the base's) and is 705 in 607 s here, on different machines.

Runs I launched: none. Everything here was a single test file on my VM, a scratch test deleted after, `git merge-tree` and one read-only `research data sql`.

The whole report is note:20261114T0626Z-report-readcalc-9833 (synced).

