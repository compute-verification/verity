[2026-11-12T23:31Z] c2bfbefd707f
TERMGB to network-accounting (lead), Thu 12 Nov 3:31 PM PST. Your 3:27 PM PST order is read whole. The two lines are made and the suites are launched.

**#9533 is at `bd70324920f4`**, two commits on `8751f749b45f`, unmarked:
- **`d080a055d858`, the harness's two lines, alone.** One file, `benchmarks/harness/tests/test_exfiltration.py`, 3 lines in, 3 out; nothing else under `benchmarks/harness/`. The file's 18 tests pass [ran on the VM]. The lines, for your note to proofs on NET-403:

| | before | after |
| --- | --- | --- |
| the comment | `# the amounts are the calculator's at its own default δ, where every one of these rows' ε is 1 or more` | `# the amounts are the calculator's at its own default δ, where the world's year's ε is 2^-20 and every other row's is less` |
| each row | `assert r["eps"] >= 1 and t["certifier_GB"] == r["term"]["2"]["GB"]` | `assert r["eps"] <= 2 ** -20 and t["certifier_GB"] == r["term"]["2"]["GB"]` |
| the world's year | `assert default["world", "year", "scan"]["eps"] == pytest.approx(11_012_646, abs=0.5)` | `assert default["world", "year", "scan"]["eps"] == 2 ** -20` |

- **`bd70324920f4`, the words for the row that pools the most.** The README, the calculator's docstring and `--scan-delta`'s help now say "the largest scope's longest period", found from the inputs and not fixed by name; with today's inputs it is the world's year. The code is as it was. No figure moves.
- **The body** names the commit and says it is the one change in proofs' folder, with the same three lines.

**The suites, once at `bd70324920f4`: `r20261112-233013-c80e`, launched 3:30 PM PST.** I send its verdict line when it has ended. #9501's branch takes the merge only after it exits 0, and I say its id then.

**W: the calculator's part starts now** on `cursor/na-stop-verdicts-w-b4e6`, by your four rulings, pushed as I go. The reader's part waits for BOUNDTIE's branch name. I saw `cursor/na-head-tick-6a2d` on origin by its name only and have read nothing of it; I build on nothing of BOUNDTIE's until it names its branch to me.

**The mark's string**: `run, not proved`, taken.

