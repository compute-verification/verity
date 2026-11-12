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

[2026-11-12T19:47Z] 5ee202aa3609
# NET-285, step 1: the scan's count, restated

SCANCOUNT (bc-a6f565e0-c510-570e-873d-53e7770b1924) to the network-accounting coordinator.

**The five statements and the read-back compile on lean's branch. None is proved. `ScanViewCut`'s proof is written and
at its first try. No lock is changed and no `open` entry is added.**

- Scratch branch `cursor/network-accounting-scan-statements-9e41`, head `1c8fa449dcd7`, no pull request, from lean's
  merge branch at `238ad6621c27` (taken once, Thu 12 Nov 10 AM PST; not taken again: its ready line is not out).
- The statements were tried at trees `31aaa7d92a14` and `62acac59ba4d` of that branch; the commits after them change
  the proof's files only.
- Marks: [ran] a command I ran and read, [read] the source as it stands on that branch, [argued] my reasoning, built
  by nothing.

## 1. The statements

All are in `main`'s vocabulary: `NetTiming.CheckedUnique`, `CheckedWitness`, `freeInputs`, `BoundaryGates`,
`certifierRun`, `windowsCount`, `RecordFormat.ReadsBy`. None reads the onsite branch's wires.

| statement | file | compiles |
|---|---|---|
| `ScanViewCut`, `ScanChainPinned`, `ScanCountsEntries`, `StateTieHeld`, `ViewTieHeld` | `security/Properties/NetworkScan/Chain.lean` | [ran] r20261112-184026-c425, exit 0 |
| `AnswersVerified`, `ScanCountBounded` | `security/Properties/NetworkScan/Bound.lean` | [ran] r20261112-184026-1256, exit 0 |
| `RowsReadPadded` (the `#Θ` step) | `security/Proofs/NetworkCertifier/RecordRowsRead.lean` | [ran] r20261112-191515-2bd9, exit 0 |

The first two files are in the properties tree: each statement is about the code's `scanOf` and `check`. The read-back
is beside `RowsFormat.recordFormat`, which is in the proofs tree; it is a `Prop` there until it is a theorem.

### 1.1 `ScanViewCut`: with neither tie, the egress views are the cut

~~~lean
def ScanViewCut : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog) (Pt : Claims.Sampling.Partition) (d : Claims.Sampling.Design),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P → Flattens P C →
      NetTiming.CheckedUnique C Pt d (scanLayout P s) s.window
        (NetTiming.freeInputs C (scanLayout P s) s.window \ viewSet P s) (fun _ => True)
~~~

In plain words: if `scanOf` accepts the scan, then in every flattening of it, two runs that agree on the egress units'
`state` gates (`viewSet`) and on the held gates have the same egress rows at every step. So every other free input is
a checked input under the check that holds of every run.

- Premise: `Properties.NetworkScan.Flattens` (the statement's gates are the Program flattened).
- No tie, no check, no `ε`. `viewSet` is the leaves of parameter 0 of each of the `T` egress members.

### 1.2 `ScanChainPinned`: the two ties pin the `sin` and `state` gates

~~~lean
def ScanChainPinned : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog) (Pt : Claims.Sampling.Partition) (d : Claims.Sampling.Design),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P → Flattens P C →
      NetTiming.CheckedUnique C Pt d (scanLayout P s) s.window (sinSet P s ∪ viewSet P s)
        (fun val => StateTied P s val ∧ ViewTied P s val)
~~~

In plain words: at an accepted scan, two runs on which both ties hold, with one ingress through step `t`, the same
inputs off the `sin` and `state` gates and the same held gates, have the same egress rows at step `t`.

- Premise: `Flattens`.
- `StateTied`: each later bucket's `sin` gates hold what the unit before returned as its state. `ViewTied`: each egress
  `state` holds what its bucket's ingress unit returned as its view. Each compares lists of values, so lengths too.

### 1.3 The ties' premises, at `certifierRun … tops` (soundness's point 1)

~~~lean
def StateTieHeld (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (K : ℕ) (s : Schedule)
    (tops : Claims.Sampling.AuditStatement → List (Claims.Sampling.Opening Bool)) (ε : Claims.Msg → ℝ)
    (x : Claims.Msg) : Prop :=
  NetTiming.CheckedWitness H D
    (NetTiming.certifierRun H D.commit 1 K s.allocation (layOf s) (fun _ => s.window) tops) (stateTiedAt s) ε x
~~~

`ViewTieHeld` is the same with `viewTiedAt s`.

In plain words: at every prover, an accepted run whose extracted witness has no fully sampled proof block incorrect has
the tie on its run **at the network certifier's rows**, except with probability `ε x`.

- These are the two `CheckedWitness` hypotheses `PeriodLeakChecked` takes, with `ε₁` and `ε₂` in its error. They are
  definitions, not statements to prove: no file of mine proves either.
- The trace's own run (`traceRun`) is read nowhere. My branch of Mon 9 Nov stated the ties there; that form is gone.
- `stateTiedAt s x val` is `StateTied P s val` where `x`'s descriptor decodes to `P`, and no condition where it
  doesn't. That is how `layOf` reads a statement's layout too.

### 1.4 `ScanCountsEntries`: what is left free is slot entries

~~~lean
def ScanCountsEntries : Prop :=
  ∀ (sj j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog),
    scheduleOf sj = .ok s → scanOf j program s = .ok sc → Flock.canon j = .ok c →
    Flock.Program.decode c.toUTF8 = .ok P → Flattens P C →
      NetTiming.freeInputs C (scanLayout P s) s.window \ (sinSet P s ∪ viewSet P s) ⊆ entryGates P s ∧
        (entryGates P s).card ≤ sc.entries
~~~

In plain words: at an accepted scan, a free input that is neither tie's gate is a slot entry gate, and the slot entry
gates number at most the bits `scanOf` counts (`Scan.entries = lanes · (T / every) · bits of an entry`).

- Premise: `Flattens`.
- `entryBits ≤ t.free` is dropped (soundness's point 3). It cannot be derived: `scanOf` returns `free := 0`, and an
  audit's `free` is read from the parameters' JSON and compared with nothing [read].
- It is `⊆`, not `=`, against Monday's form: a slot entry gate no bucket reads would break equality and changes no
  count.

### 1.5 `AnswersVerified`: the premise for "every scan unit proved" (soundness's point 3)

~~~lean
def AnswersVerified (pub : Public) {n : ℕ} (sw : Fin n → Claims.Sampling.AuditStatement)
    (Pw : Fin n → Claims.Sampling.Partition) : Prop :=
  ∀ (i : Fin n) (t : AuditTerms), pub.units[i.val]? = some t → t.facts.drawn = t.facts.units →
    (sw i).design.miss (Pw i) 1 ≤ 0
~~~

In plain words: where link-window `i`'s published facts name as many units sampled as its level holds, the statement's
own design on its partition accepts with no unit incorrect only.

- It is a new premise, on no lock. It names what NET-352's switch gives: the period check reads each answer from the
  evidence and does not run the sampled verifier.
- "Every unit sampled" itself is read from the public the check accepts with (`t.facts.drawn = t.facts.units`, a
  hypothesis of `ScanCountBounded` about `pub`), never from a bound on the incorrect blocks.

### 1.6 `RowsReadPadded`: the `#Θ` step, in form (b)

~~~lean
def lanesOfRowsPadded [DecidableEq κ] (pads nb : ℕ) (rows : ℕ → List (κ × CellView Cell)) (c : κ) :
    List (ℕ × CellView Cell) :=
  ((List.range nb).flatMap fun b =>
    ((rows b).filterMap fun f => if f.1 = c then some f.2 else none).map fun x => (b, x)).drop pads

def RowsFormat.readPadded [DecidableEq κ] (F : RowsFormat κ Cell L K) {S n : ℕ} (pads : ℕ)
    (o : Fin n → List (Obs L K S)) : κ → List (ℕ × CellView Cell) :=
  lanesOfRowsPadded pads (n * F.T) (F.frames o)

def RowsReadPadded : Prop :=
  ∀ (κ Cell : Type) [DecidableEq κ] (L K : ℕ) (F : RowsFormat κ Cell L K) (hF : F.Wf) (S n : ℕ),
    (F.recordFormat hF S n).ReadsBy (fun θ : ℕ × ℕ => F.readPadded (θ.1 * θ.2)) fun θ s => s.r = θ.1 ∧ s.D = θ.2
~~~

In plain words: a connection's frames in bucket order, less the first `r · D`, are its lane's cells, each in its
bucket. So the rows record reads back given the pair `(r, D)` alone, on every schedule with that pair that fits the
format. The registered parameters are then one pair and `#Θ = 1`.

- No premise. `ReadsBy` is `main`'s definition; `Par` is `ℕ × ℕ`.
- It replaces `#Θ ≤ 2^(schedules + cells)`, which is false as written (my message of 10:13 AM PST; section 3).

### 1.7 `ScanCountBounded`, in its new form (soundness's point 2)

~~~lean
def paddings (s : Schedule) : Set (ℕ × ℕ) := {(s.slots, s.delay)}

noncomputable def faultyMost (δ : ℝ) {n : ℕ} (sw : Fin n → Claims.Sampling.AuditStatement)
    (Pw : Fin n → Claims.Sampling.Partition) (i : Fin n) : ℕ :=
  (NetTiming.faultySets (sw i).design (Pw i) δ).sup (NetTiming.faultyBits (sw i).prog (Pw i))

def ScanCountBounded : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (sj j : Lean.Json) (cs : String) (P : Flock.Program.Prog),
      scheduleOf sj = .ok s → scanOf j p.program s = .ok scan → Flock.canon j = .ok cs →
      Flock.Program.decode cs.toUTF8 = .ok P →
    ∀ (sw : Fin p.windows → Claims.Sampling.AuditStatement) (Pw : Fin p.windows → Claims.Sampling.Partition)
      (reg : Claims.Sampling.AuditStatement → Finset ℕ) (δ : ℝ), 0 ≤ δ →
      (∀ i, (⟨(sw i).desc.toArray⟩ : ByteArray) = cs.toUTF8 ∧ Flattens P (sw i).prog) →
      (∀ t ∈ pub.units, t.facts.drawn = t.facts.units) → AnswersVerified pub sw Pw →
        (paddings s).encard *
            ((NetTiming.windowsCount (layOf s) (fun _ => s.window) δ (fun x => reg x ∪ tiesOf s x) p.jitter sw Pw
              (faultyMost δ sw Pw) : ℕ) : ℕ∞) ≤
          ((2 ^ (p.windows * (scan.entries + clog2 (p.jitter ^ s.window))) : ℕ) : ℕ∞) ∧
        (paddings s).encard *
            ((NetTiming.windowsCount (layOf s) (fun _ => s.window) δ (fun x => reg x ∪ cutOf s x) p.jitter sw Pw
              (faultyMost δ sw Pw) : ℕ) : ℕ∞) ≤
          ((2 ^ (p.windows * (viewCount P s + clog2 (p.jitter ^ s.window))) : ℕ) : ℕ∞)
~~~

In plain words: at a period the check accepts in the rows form, where every link-window's published facts name every
unit sampled and each answer read is verified, `PeriodLeakChecked`'s right-hand side at the scans, `#Θ · windowsCount
(reg ∪ chk)` with `Θ` the one registered pair, is

- at most `2^(windows · (entries + ⌈log2 jitter^T⌉))` with both ties' gates checked, and
- at most `2^(windows · (viewCount + ⌈log2 jitter^T⌉))` with every free input off the views checked.

Its premises: `Flattens` at each window's statement; `AnswersVerified`. The left side is the right-hand side of
`PeriodLeakChecked` letter for letter at `lay := layOf s`, `k := fun _ => s.window`, `Dj := p.jitter`,
`β := faultyMost δ sw Pw` (the least `β` that `WindowsPlaced` takes) and `chk := tiesOf s` or `cutOf s`.

## 2. Every premise, by name

Locks read at `origin/main` `4a90174415b7` and at lean's branch `e209a56847d4` [ran: each name looked up in the three
locks under `security/`].

| premise | what takes it | on `main`'s lock today | what discharges it |
|---|---|---|---|
| `Properties.NetworkScan.Flattens` | all five statements | No: defined on `main`, in no lock there. On lean's branch: `open`, LEAN-186, `@network-accounting`, due 30 Nov | LEAN-186 |
| `StateTieHeld` (= `NetTiming.CheckedWitness` at `certifierRun`, `stateTiedAt`) | `PeriodLeakChecked` at the scans, both ties | `NetTiming.CheckedWitness` is `open` on `main`: CIR-190, `@proofs`, due 30 Nov. This instance is in no lock | a refusal of the period check, NET-294 (NET-286) |
| `ViewTieHeld` (the same at `viewTiedAt`) | the same | the same | the same; tie 2 is read nowhere on `main` |
| `AnswersVerified` | `ScanCountBounded` | No. New, in no lock | NET-352's switch |
| `NetTiming.CheckedUnique` | what `ScanViewCut` and `ScanChainPinned` conclude | `open` on `main`: CIR-190 | these two statements discharge it at the scans |
| `NetTiming.Guarantees.PeriodLeakChecked` | no statement of mine takes it; `ScanCountBounded`'s left side is its right-hand side | stated on `main`; in `shape.frozen`, whose note says its theorem is not proved yet | PERIODPROOF's proof |

- **Two premises are new to every lock: `AnswersVerified`, and the two ties' instances.** I added neither. They go to
  Daniel through you.
- `RowsReadPadded` takes no premise.

## 3. Your five items (message of 10:20 AM PST), and the 10:37 AM line

**1. `ScanCountBounded` at the read-back with `Par = ℕ × ℕ`, and the read-back beside it.** Sections 1.7 and 1.6. Both
compile.

**2. The instance in numbers: one lane, cap 1, `T = 32`, `S = 1`, one window, `#Θ = 1`.**

- The right-hand side is `1 · 2^#(free \ (reg ∪ chk)) · #faultySets · 2^β · jitter^32`.
- With every unit verified, `#faultySets = 1` and `β = 0`. With both ties and nothing registered, it is
  `2^E' · jitter^32`, where `E'` counts the free inputs off the ties, slot entry gates all, and
  `E' ≤ entries = (32 / every) · bits of an entry`.
- The 31 schedules are 31 records, so 31 accepted sequences of one registered scan with one ingress. With both ties
  and every unit verified, a scan's egress rows are a function of its ingress and those `E'` gates, so 31 of them need
  `2^E' ≥ 31`. A scan with fewer than 5 such gates accepts fewer than 31 of the records, and the rest are not counted
  [argued].
- At the Python builder's entry row the bound is far above 31: an entry is `33 + KO · IDB` bits rounded up to a
  multiple of 16, at least 48 [read: `entry_bits`, `verity/protocols/certifier/prover/ingress_scan.py`], and 112 at
  the class `Period.lean`'s docstring names [read]. So the statement's bound there is at least `2^48 · jitter^32`.
- **So (b) is not false at the instance.** If the registered scan accepts all 31 records, the sender's `log2 31` bits
  are 31 values of its free inputs: they are counted in `2^E'`, not in `#Θ`.

**3. The edges of the read-back.** None breaks it.

- [ran] A brute force in Python of the read against the code's cells, on every schedule that fits: 1 connection, 18,432
  schedules of which 6,870 fit; 2 connections, 315,000 of which 93,826 fit; 3 connections, 78,732 of which 26,840 fit.
  It covers `r = 0`, `D = 0`, a lane of no cells, two lanes admitted in one bucket, `S > 1`, and padding the period's
  end cuts. Every fitting schedule reads back. The script is `readback_edges.py` in my store.
- [ran] Five `#guard`s in `RecordRowsRead.lean`, on Lean's own `certifierCode` and `bucketFrames`: the queue fixture;
  `r = 0`; `D = 0` with two lanes admitted in one bucket; a lane of no cells whose padding the period's end cuts; one
  lane of one cell admitted at each of 31 buckets (the instance). They pass in r20261112-191515-2bd9.
- [argued] Why a lane of no cells is safe: it has at most `r · D` frames in the period, all padding, so dropping
  `r · D` leaves none, whatever its admission. Why a lane with a cell is safe: `Fits` puts its first cell in the
  period, and all `r · D` padding frames are in buckets before it.

**4. (a), as a remark. In no lock, not compiled.**

~~~lean
def SchedulesUnder {Conn : Type} (r D : ℕ) (cap : Conn → ℕ) (Θ : Set (NetTiming.CertifierSchedule Conn)) : Prop :=
  (∀ s ∈ Θ, s.r = r ∧ s.D = D ∧ ∀ c, s.cells c ≤ cap c) ∧ ∀ s ∈ Θ, ∀ s' ∈ Θ, s.cells = s'.cells → s.adm = s'.adm
~~~

- With it `Θ.encard ≤ ∏ c, (cap c + 1) ≤ 2^schedules`, since `s ↦ s.cells` is injective on `Θ` [argued].
- Without it the bound is false: one connection, cap 1, `r = D = 1`, `S = 1`, `T = 32`, and `Θ` the 31 schedules that
  admit at buckets 0 to 30. Each fits and gives another client view; `2^(schedules + cells) = 4 < 31`. At cap 0 every
  admission fits and `Θ` is infinite.
- What would discharge it: the device's admission rule as a Lean function of the arrivals and the lanes' lengths, with
  the device's run computing its schedule by it. No Lean defines it today [ran: `adm` is only ever a field read].

**5.** With the read-back at `Par = ℕ × ℕ` the count has `#Θ = 1`, and K is unchanged at 26,400 because no Lean
defines the device's admission rule. No statement of mine reads the check's `schedules` or `cells` [read].

**The file.** No open pull request touches `security/Proofs/NetworkCertifier/RecordRows.lean` [ran: 208 open, by
file]. I did not write in it: the read-back is a new file beside it, `RecordRowsRead.lean`. `security/Proofs/
NetworkCertifier.lean`, which imports it, is also changed by #9350 and #8829: one import line each, a merge by hand
at worst.

## 4. Where each statement goes in `PeriodLeakChecked`

`L := 1`, `S := s.allocation`, `lay := layOf s`, `k := fun _ => s.window`, `Dj := p.jitter`.

| binder of `PeriodLeakChecked` | both ties | neither tie |
|---|---|---|
| `chk` | `tiesOf s` | `cutOf s` |
| `valid₁`, `valid₂` | `stateTiedAt s`, `viewTiedAt s` | `fun _ _ => True`, twice |
| `hchk` | `ScanChainPinned` | `ScanViewCut` |
| the two `CheckedWitness` | `StateTieHeld`, `ViewTieHeld`, `ε₁ + ε₂` | a check that holds of every run is never missed: `ε₁ = ε₂ = 0` [argued] |
| `rd`, `allows`, `hread`, `Θ` | `RowsReadPadded`, `paddings s` | the same |
| the count | `ScanCountBounded`, first part | `ScanCountBounded`, second part |

Two gaps between the statements and the binders, both small and both for step 2 [argued]:

- `hchk` wants the check `fun r => valid₁ x r ∧ valid₂ x r`. `ScanViewCut` concludes at `fun _ => True`, and the
  binder gives `True ∧ True`: one rewriting lemma.
- `hchk` is at `(sw i).prog` and `layOf s (sw i)`; the statements are at `C` and `scanLayout P s`. They meet where the
  statement's descriptor decodes to `P`, which `ScanCountBounded` assumes of each `sw i`.

## 5. Four findings for you to rule on

1. **"Every egress unit verified" cannot be said against `windowsCount`.** Its incorrect-set factors range over all
   the partition's units, not the egress units. So the neither-tie bound takes every unit verified, as the both-ties
   bound does. Your point 4 says "every egress unit". [read: `faultySets`, `windowsCount`]
2. **The bound is in the scan's terms, not in the public's `K`.** `ScanCountBounded`'s right-hand side is
   `windows · (entries + ⌈log2 jitter^T⌉)` or `windows · (viewCount + …)`. The check's term for a link-window's free
   inputs is its tied audits' declared `free`, which the check compares with neither. A statement "the count is at most
   `2^pub.K`" would need that comparison in the check: a code change, not a proof. [read]
3. **`AnswersVerified` is two things in substance.** That the answer read is the sampled verifier's (NET-352), and
   that the statement's design is the profile's published rule. I stated the one consequence the count reads (`miss(1) ≤
   0` where every unit is sampled). If you want them apart, it is two named premises. [argued]
4. **I did not compute the neither-tie exponent at the deployed parameters.** The statement gives `windows · (T ·
   leaves of the egress template's state + ⌈log2 jitter^T⌉)`. That this is 26,400 at the deployed class is not checked
   by me.

## 6. What a listing would need in the locks (none of it is done)

- Roots: `Properties.NetworkScan.Chain`, `Properties.NetworkScan.Bound` are added to `security/Properties/
  lakefile.toml`; `Proofs.Properties.NetworkScan.ViewCut` to `security/Proofs/lakefile.toml`. Two open pull requests
  touch the first file (#9373, #9071) and four the second (#9373, #9091, #9071, #8139).
- The properties lock's `layers["Properties.NetworkScan"].imports` would gain `Definitions.NetworkCertifier.
  PeriodChecked` and `Definitions.NetworkCertifier.PeriodConditions`, with its why.
- A `shape.properties` entry and `_honest` and `_nontrivial` witnesses for each listed guarantee, as `scanFits` has.
- `open` entries for the premises of section 2. **I add none.**

## 7. The proofs: where they stand

- **`ScanViewCut`: written end to end, not compiled.** First try r20261112-191514-7613, at tree `62acac59ba4d`, has
  waited for a slot of the pool since 11:15 AM PST.
  - `Walk.lean`: a template's body reads, outside its own gates, only what its parameters resolve to
    (`run_param`, `memberReads_params`).
  - `Accept.lean`, `Fits.lean`: `scanOf`'s acceptance gives the egress template's one parameter, its argument an input
    node's leaves, and the egress gates' place (`egress_gates`).
  - `ViewCut.lean`: no egress gate depends on a free input off the views (`view_closure`), so `CheckedUnique` holds
    under every check (`checkedUnique_of_reads`, with `scanCausal`), and `scanViewCut : ScanViewCut`.
  - It takes `Flattens` and nothing else, and holds for every `T ≥ 1`.
  - No open pull request touches `Walk.lean`, `Accept.lean`, `Fits.lean` or `Causal.lean` [ran].
- **The other four: not started.** In your order: the read-back, `ScanCountsEntries`, `ScanChainPinned`,
  `ScanCountBounded`.

## 8. What I did not check

- That any line of `ScanViewCut`'s proof compiles.
- (a)'s Lean text: never compiled.
- `check.py`, the locks' regeneration, the witnesses: none run, none written.
- lean's branch after `238ad6621c27`: it was at `d1f71265b06e` at 11:45 AM PST, with `main` merged in and no ready
  mark; I take it once, after its ready line. Whether my files still compile there is not known.
- Whether #9350 changes a definition I read: you tell me after 2 PM PST.
- `git merge-tree` against `origin/next`: not run; nothing here is offered for landing.
- The neither-tie exponent at the deployed parameters (finding 4).

