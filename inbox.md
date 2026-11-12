[2026-11-12T01:25Z] e1b2546d6130
TERMGB (the execution that holds job 8) to coordinator, Wed 11 Nov, 5:26 PM PST. Part A: the merge is pushed.

**#8777's head: `6b8edbf5de81f278348219ac167b722a3a173493`**, pushed 5:23:50 PM PST, a fast-forward from `f28f4e4acc2d`; origin equals local [ran].
- `f4adac94be4cfc56f8f5d0dcb6a8846476cf858f`: the merge commit, parents `f28f4e4acc2d` and `main` at `be9a3d566905`.
- `6b8edbf5de81`: the one line, `benchmarks/harness/tests/test_exfiltration.py:369`, `lb.inputs()` to `lb.inputs(requests="census")`. The line was where you said and read as you said [read]. Without it 1 of the file's 11 tests fails at the merge commit (`1.7784…e+16 < 1.7784…e+16`); with it 11 pass [ran].

**Files closed by hand: none.** Git stopped on nothing [ran]. Two files both sides changed and git merged by hunks, both sides kept [read]: `benchmarks/README.md`, `benchmarks/network_traces/pyproject.toml`. `main` changed one more file of ours, `benchmarks/network_traces/scan_period.py` (7 added, 2 removed, all `main`'s). `leak_bounds.py`, `test_certifier_term.py` and `benchmarks/certifier_term` are byte for byte the old head's [ran].

**The risk, `benchmarks/network_traces/leak_bounds.py`: the merged head against `main` is 198 added, 76 removed, in these lines of `main`'s file** [ran: `git diff -U0`]:
- 42-45, 49, 57-61: the module docstring (which request a row counts; the term of record at D_j = 2 and the label; the design's rows).
- after 102: `PLANNER`, `PLAN_CASE`. 134-137: `CHANGES`' two texts.
- 177, 183, 193: `terms` gains `stop: bool = True` (signature's second line, docstring, the `Terms(0 if held and stop …` line). Its `dj: int = 1` on line 176 is not touched by #8777.
- after 425 to 848: the certifier term's table, everything below `JITTERS` (`OF_RECORD`, `SET_TIME`, `Kind`, `ROWS`, `Input`, `inputs`, `table`, `document`, `_n`, `render`, `main`).
- Nothing in 194-424: `LinkLoad`, `seconds`, `per_link`, `datacenter`, `at_link_rate` are `main`'s.

**Against CROSSTICK's close of that file** (`cursor/na-cross-main-closes-84ef`, the same file at `a4e1d7e04617` and at its head `33360da0a088`; 20 added, 4 removed against `main`, at `main`'s lines 25, after 94, after 258, 283-284, 286, after 309):
- No hunk of #8777's touches or abuts one of the close's. The nearest pair is the close's insert after 94 (`JITTER_BOUND`, `JITTER`) and #8777's after 102.
- A three-way merge of the one file (#8777's merged head, `main` as base, the close) is clean, 994 lines [ran: `git merge-file`, scratch only; I ran no test on its result].
- So if #8777 lands first, neither side moves in lines. What the merged file then holds is two names for one bound, `OF_RECORD = 2` (#8777, for the table) and `JITTER_BOUND = 2` (the close, for `LinkLoad`), beside `jitter_bits` and `_jitter`. That is a naming question for the table, not a conflict.
- Not checked: lean's crossed head itself (`3ff11d118f93`), whose own copy of the file is 44 added, 609 removed against `main` (it predates the certifier term's table); I compared only the close.

**Next:** one run at `6b8edbf5de81` through `--on auto` (the five suites, `benchmarks/harness`, the nine root files); I name it by inbox when it is launched. Then the body, fetched from GitHub first. No mark.

**VM state** [ran]: this VM is new (up since 5:20 PM PST): no other execution's process, no timer, no run of mine in flight. Nothing to cancel.

