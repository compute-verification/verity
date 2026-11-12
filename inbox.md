[2026-11-12T19:42Z] 3c8b5bd32231
TERMGB to the network-accounting coordinator, Thu 12 Nov 11:42 AM PST, to your 11:35 AM. **`main` at `0e8d16c3b8be` is merged into NET-394's branch: head `efe2fcdb33e81ff05842cff0a2e27138ccd1fb7d`, pushed. Nothing stopped. The harness suite still fails at it: 16 of the 18 tests of `test_exfiltration.py`, every one at the reader's refusal of the fourth mark. Nothing #9404 prints contradicts the ruling.**

**The merge** [ran]
- A merge commit, no rewrite. `benchmarks/harness/` and `benchmarks/exfiltration/` are `main`'s, whole (no difference from `main` under either).
- It holds the 11:19 AM rulings, commit `3785695f01`: `--lanes N` refuses an N whose R is under the load with both numbers (`--lanes 16`: 688 and 1,540, "give at least 36 lanes"); the count's line names its schedule and gives the cost runner's as that runner's own; `lengths` and `all three` each say why they are charged no count. The three figures of 11:22 AM are unchanged.
- At the head: the calculator's tests 34 passed, the runner's 15 passed [ran on the VM].

**The harness suite at the head** [ran on the VM: `benchmarks/harness/tests/test_exfiltration.py`]
- 16 of 18 fail, each through `verity_bench/exfiltration.py` line 499.
- With the fourth mark given in memory and nothing written, 5 of 18 fail, at the test's lines 410, 489, 562, 668 and 768.

**The places proofs' side has to change, at `efe2fcdb33e8`**
- `benchmarks/harness/verity_bench/exfiltration.py`
  - line 151: `MARKS` holds three marks; line 499 refuses any other ("an input is marked one of measured, modeled, assumed, with its source: set_time_card") [ran].
  - lines 2 and 923: "every input labelled measured, modeled or assumed" [read].
  - lines 34, 54 and 788 give the certifier's part as the schedules, the fail cells and the jitter. The count is a fourth term on the two design rows. It reaches the command inside each row's `term`; `AMOUNTS` (line 94) reads no `count` [read].
- `benchmarks/harness/tests/test_exfiltration.py`
  - the old label "with a set-time card (not yet tested)": lines 562, 563, 568, 570 and 668 [ran for 562 and 668; read for the rest].
  - the design rows' figures before the count: line 410 (a row of the sums), line 489 (1,847,615, now 1,939,394), line 768 (a datacenter's year: 7,014 and 6,521, now 7,039 and 6,546) [ran]; and by a search for the old figures, lines 499, 514, 520, 524-525, 565, 571, 674-675, 756 and 769-770 [read, not run: each fails only once the ones above it pass].
  - lines 795, 798 and 833: the test named for three marks, and its reads of `E.MARKS`.
- `benchmarks/exfiltration/README.md`
  - line 5: the three marks. Lines 34-35 and 49-50: the certifier's part as three terms. Line 41: "no timing test has shown such a card".

**#9404 and the ruling, one line** [read: its README, `render` and the label's test; I could not run the command here, it waited on the store past 2 minutes]: nothing contradicts it. The headline's certifier's part is the scan's term of record at D_j = 2, and every figure at D_j = 1 or of a row whose times are taken once prints only under the label the calculator's document gives, which the command reads and holds no copy of. What it lacks is the mark: it refuses `simulated` and prints no mark beside the card's figure.

**Two things to know**
- The command's "output count" is another thing than the count of this change: it is the `schedules` of the row `design, no stop block` (line 606), 24,453,012 GB, and it does not move. A reader may take one for the other; I name mine "the count" and say what the head states.
- With the card, the command's headline under the label is `all three` at 0 GB. That row carries no count, by your ruling. `design` beside it is 91,778 GB.

**Runs**
- `r20261112-194109-40df`: the 8 suites the change touches, at the merged head, launched once at 11:41 AM PST; about 21 minutes to a slot.
- **A slip:** I launched `r20261112-193044-5b7e` at `3785695f01` at 11:30 AM, before your message, and cancelled it at 11:40 AM when the head moved. I took it to be waiting; it had been checking for 7 minutes. Its verdict is not used.
- `r20261112-181342-dd6c` (the scratch's suites) is still running.

**Next:** the trial merges and the body are written; the pull request opens when the run's verdicts are read. The 1:30 PM report keeps its hour, and lean's branch is taken at its ready line (#9375 carries no mark at 11:34 AM PST, head `64ee10f17900`).

Least sure of: the list of the test's lines that hold old figures. I ran only as far as the first failure in each test.

TIMER: none

