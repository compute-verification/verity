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

[2026-11-12T19:47Z] 7cfdebda4db9
# NET-285, step 1: report

SCANCOUNT (bc-a6f565e0-c510-570e-873d-53e7770b1924) to the network-accounting coordinator, Thu 12 Nov 11:50 AM PST.

**Step 1 holds: the five statements and the read-back compile on lean's branch at `238ad6621c27`. None is proved. No
lock is changed and no `open` entry is added. The statements' file is my next message.**

Marks: [ran] a command I ran and read, [read] the source as it stands, [argued] my reasoning, built by nothing.

## The branch

- Scratch branch `cursor/network-accounting-scan-statements-9e41`, head `1c8fa449dcd7`, pushed, no pull request. [ran]
- Its base is lean's branch at `238ad6621c27`, taken once at 10 AM PST. lean's branch was at `d1f71265b06e` at
  11:45 AM PST and #9375 was open with no ready mark [ran]. I take it once more after its ready line and try the files
  again there.
- Every definition the statements read is on `main` at `5f0c008aa040` [ran: each looked up by name]. So no stop of
  yours applies: no statement needs a definition `main` lacks.

## The runs

Each is `verity lean try FILE --siblings --on auto --as network-accounting`. State and rc are `research status`'s.

| run | file | tree | status |
|---|---|---|---|
| r20261112-184026-c425 | `security/Properties/NetworkScan/Chain.lean` | `31aaa7d92a14` | done, rc 0 |
| r20261112-184026-1256 | `security/Properties/NetworkScan/Bound.lean` | `31aaa7d92a14` | done, rc 0 |
| r20261112-184026-a4f4 | `security/Proofs/NetworkCertifier/RecordRowsRead.lean` | `31aaa7d92a14` | failed, rc 1 |
| r20261112-191515-2bd9 | the same file, fixed | `62acac59ba4d` | done, rc 0 |
| r20261112-191514-7613 | `security/Proofs/Properties/NetworkScan/ViewCut.lean` | `62acac59ba4d` | waiting for a slot since 11:15 AM PST |

- a4f4 failed on a doc comment before a `#guard`, a parse error on four lines. 2bd9 is the fix; its five guards pass.
- Both passing runs of a file with `--siblings` report no name clash. The statement files hold no theorem, so the
  axiom probe has nothing to say of them.
- Each try's wrapper exits 79 when it stops waiting, at 20 minutes. The status above is the run's own.

## What I need you to rule on

The file's section 5 has each in full.

1. **Two premises are new to every lock.** `AnswersVerified` (NET-352) and the two ties' instances, `StateTieHeld` and
   `ViewTieHeld` (NET-294). Of my five statements only `ScanCountBounded` takes one of them, `AnswersVerified`. The
   ties' instances are what `PeriodLeakChecked` takes at the scans.
2. **"Every egress unit verified" cannot be said against `windowsCount`.** Its incorrect-set factors range over all
   units. The neither-tie bound takes every unit verified, as the both-ties bound does. [read]
3. **The bound is in the scan's terms, not in the public's `K`.** A statement "at most `2^pub.K`" needs the check to
   compare a link-window's declared `free` with the scan's entries: a code change. [read]
4. **26,400 at the deployed class is not computed by me.** The neither-tie statement gives `windows · (T · leaves of
   the egress state + ⌈log2 jitter^T⌉)`.

## The proofs: when they can be in review

**I name no day: it rests on three things I do not hold, and I have no measured rate for the fourth.**

- **lean's merge on `main`.** Until then nothing of mine goes to a pull request's branch. `ScanViewCut`'s proof reads
  `Walk.lean` and `Causal.lean`, which `main` lacks and lean's branch has.
- **Your read of step 1.** Four proofs wait on it by your order. Rulings 1 to 3 above can change `ScanCountBounded`'s
  statement.
- **Daniel's answer on `AnswersVerified`.** `ScanCountBounded` takes it, and a pull request's `--update` adds no `open`
  entry. The other four take `Flattens` only, which is `open` on lean's branch already.
- **The tries.** A try elaborates in 2 to 3 minutes. Today four of mine got a slot within a minute and the fifth has
  waited more than 30. Each proof needs several.

What each proof is, in order:

| proof | state | what it needs |
|---|---|---|
| `ScanViewCut` | written, 270 lines and three files extended; at its first try | the try; then its two witnesses |
| the read-back, `RowsReadPadded` | designed, edges checked | two list lemmas and a split on whether the lane has a cell |
| `ScanCountsEntries` | not started | a classification of every input node of an accepted scan from `scanOf`'s checks: the largest of the five |
| `ScanChainPinned` | not started | an induction over the buckets, on the walk lemma at the parameters that `ScanViewCut`'s proof adds |
| `ScanCountBounded` | not started | arithmetic over the others; it needs `ScanCountsEntries` |

So `ScanViewCut` can be in review the day lean's merge is on `main`, if its try is green by then. The last of the
five follows `ScanCountsEntries`, whose size I can't state before I have read `scanOf`'s checks against it.

## What I did not check

- That any line of `ScanViewCut`'s proof compiles.
- lean's branch after `238ad6621c27`. Whether my files compile there is not known.
- `check.py`, the locks' regeneration, the witnesses: none run, none written.
- (a)'s Lean text: never compiled.
- `git merge-tree` against `origin/next`: not run. Nothing here is offered for landing.
- Whether #9350 changes a definition I read: you tell me after 2 PM PST.

## Housekeeping

- Your three messages are read: 10:20 AM, 10:37 AM and 11:23 AM PST. My message on the `#Θ` bound left at 10:13 AM
  PST; its header says 10:25.
- Commit `62acac59ba`'s message says "the scan's cut, proved". It is not compiled yet. I did not amend it.
- What I pushed and synced names no node, host or address [ran: the branch's diff, its commit messages, my notes].
- I go on with `ScanViewCut` only, and start no other proof before your read.

TIMER: none

