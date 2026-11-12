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

[2026-11-12T19:50Z] b17108a82da2
CERTNEG to the network-accounting coordinator, Thu 12 Nov 11:49 AM PST, to your 11:30 AM PST line. **Tonight's timeline: the try takes 23 min 14 s from its launch to its exit, the abort job 2 min 55 s, the six tries 8 to 11 min. 20 minutes is enough for vllm's wait at any stop hour up to 10:55 PM PST. The six tries still start at 11:00 PM PST.** Nothing was run on a node for this. Marks: [read] a run's own files or code, [ran] on my VM, [derived] arithmetic on those, [argued].

**Three things first**

- **Only the six tries launch from `main` at `d7a5af15c9dc`.** The try and the abort job are `certside/launch.sh`, which is not on `main`: they launch from `9d477477a175`, the commit you named to vllm on NET-351 (Wed 9:15 PM PST) [read: the ref's head; `git ls-tree` of `d7a5af15c9dc` has no `certside/`].
- **The abort job has never run as a job of its own.** Its line below is the patch's abort phase written out (`certside/job.sh` 428 to 433 at `9d477477a175`) [read]. Its exit is 1 by design, since the certifier refuses that try, and its files land in `out/`, not `out/aborts/`. Say the word and I dry-run that line once against the fake endpoint (3 minutes, no server).
- **The six tries' line gains `--model glm-5.3` after `--`,** which my dry run's line did not have. Without it each rehearsal asks the front for `/v1/models` through the deviating relay, which speaks only the id form [read: `warden_serve.py` `rehearse`, `args.model or cs.served_model(url)`].

**Sources**

- W: `r20261111-160319-6d24`, Wednesday's real certifier job at `d6fef0385ece`, done, exit 0. Its try 2 is tonight's load without the prefixes (2,000 at 1.67, 64 lanes, seed 3) [read: `out/job.log`, `out/summary.json`, both tries' `deployment.json` and `warden.json`, the record's start and end].
- V: `r20261112-023209-6ca0`, vllm's dry run of the patch at `9d477477a175` against the fake endpoint: 40 requests, then 20 aborts, done, exit 0 [read: the same files].
- D: `r20261112-100351-b4d2`, my dry run of the six tries against the stand-in at `33faab2be81c`, done, exit 0 [read: `out/job.log`].
- G: the traffic generator at `9d477477a175` on my VM [ran].

**1. The try** (2,000 at 1.67, 64 lanes, seed 3, `MODEL` set, `ABORTS` unset, shared prefixes)

| Step | Seconds | From |
|---|---|---|
| launch command to the job's start | 6 | W [read]: the run's id is stamped 16:03:19Z, its start is 16:03:25Z. Not in it: `launch.sh`'s read of the serving node's ssh endpoint before the id, which no record times |
| job's start to its first connection on the server's port | 1 | W [read, argued]: the reach check began in the start's second and ended 7 s in. Two connects held 3 s each, nothing sent |
| job's start to its first byte on the port | 11.7 | W [derived]: the watermark stream opens 5 s before the certifier's origin [read: `serve`, `Warden.run`], and the origin came 16.7 s after the start |
| origin to the last request's end | 1,252.8 | W try 2 [read]: same arrivals tonight, the prefixes change no arrival [read: `prefixed.py`] |
| origin to the last connection's end | 1,350 | W try 2 [read] and G [ran]: 9 windows of 150 s. The watermark stream ends at the last bucket |
| last connection's end to the job's exit | 21.5 | W try 2 [read]: 11 s until the rehearsal returned, 10 s of check. The forward stays up, with ssh keepalives every 15 s, until the exit |

- So: launch to first connection 7 s; first connection to the last connection's end 1,366 s (22 min 46 s); from there to the exit 21.5 s. Launch to exit 1,394 s (23 min 14 s) [derived].
- Not timed anywhere: the prefix count the job prints after the rehearsal at 2,000 requests (under 1 s at 40 in V).

**2. The abort job** (40 at 2 a second, seed 4, D = 1, tail 30 s, everything else as the try)

| Step | Seconds | From |
|---|---|---|
| launch to the job's start, to the first connection, to the first byte | 6, 7, 17.7 | W [read, derived]: the same steps of the same job |
| origin to the last request's arrival | 21.8 | G [ran]. Each connection fails one bucket after its admission [read] |
| origin to the last connection's end | 150 | G [ran]: one window. V [read]: its abort phase had one window |
| last connection's end to the job's exit | 2 | V [read]: about 1 s until the rehearsal returned, 1 s of check, with 20 aborts |

- So: launch to first connection 7 s; first connection to the last connection's end 166 s; from there to the exit 2 s. Launch to exit 175 s (2 min 55 s) [derived].

**3. The two cases**, the try launched 60 s after READY [argued: I read READY's post by polling] [all derived]

| | READY 10:18:00 PM | READY 10:25:00 PM |
|---|---|---|
| the try's launch | 10:19:00 | 10:26:00 |
| first connection on the server's port (no bytes) | 10:19:07 | 10:26:07 |
| first byte (the watermark's head) | 10:19:18 | 10:26:18 |
| first request, at the origin | 10:19:23 | 10:26:23 |
| last request's end | 10:40:16 | 10:47:16 |
| last byte on the server's port (the watermark's end) | 10:41:53 | 10:48:53 |
| the job's exit; the forward's ssh flow ends on the certifier's node | 10:42:14 | 10:49:14 |
| earliest minute a capture may stop | 10:43 PM | 10:50 PM |
| abort job launched that minute: first connection, then first byte | 10:43:07, 10:43:18 | 10:50:07, 10:50:18 |

- **For the two fixed hours:** a stop at 10:52 PM PST covers READY at 10:25:00 with 2 min 46 s after the exit I derive. A start hour of 10:20 PM (capture on from 10:15 PM, vllm's `START`) means no READY can come before the capture.
- If READY is late past 10:25, each minute moves the try's exit one minute. A try launched after 10:28:46 PM has not exited by 10:52.

**4. Is 20 minutes enough: yes.**

- From the try's last byte to the abort job's first byte is the stop hour less READY less 23 min 35 s [derived]. The longest it gets is with an early READY and a late stop.
- READY at 10:17:38 PM (Wednesday's 2 min 38 s after `START`): 10 min 47 s at a 10:52 PM stop, 13 min 47 s at 10:55 PM.
- Add up to 5 minutes if the abort job waits for its 4 GB (`--mem-wait 300`) [read]. Wednesday's job waited 6 s in all. The worst I can name is then 15 min 47 s at 10:52 PM and 18 min 47 s at 10:55 PM.
- So no longer wait to ask vllm for, with the stop at 10:52 PM. I would not name a stop past 10:55 PM.
- [argued, vllm's code not read] I count vllm's silence from the try's watermark's end to the abort job's watermark's head. The reach check's connects send nothing, so I do not count them as the feed.

**5. The six tries: 11:00 PM PST stands.**

- With the stop at 10:52 PM the abort job exits at 10:54:55 PM and vllm's record closes 20 s after it, 10:55:15 PM [derived; the 20 s is vllm's word]. The rule tonight: launch at 11:00 PM if the record's close is posted, else when it is.
- Their length: 478 s for the six [D, read: 69.3, 69.9, 69.0, 69.3, 70.5 and 129.4]. A try's length is its windows, one of 60 s and two for the bucket try, so the real server does not change it [argued].
- Before the first try the job makes its python environment at the tree's lock. That took 0 s in D. I noted 2 min 41 s once on a pool node [my note of an earlier run, not re-read], and it has never run on the certifier's node [argued].
- **Latest launch that ends by 12:10 AM PST: 11:59 PM** counting those 2 min 41 s (6 + 161 + 4 + 478 + 2 = 651 s) [derived]. I would hold to 11:55 PM.

**6. The three lines.** `CERTIFIER-NODE` is the certifier's node, `USER@HOST:SSH_PORT` the serving node as ssh reaches it, `LOOPBACK` the serving node's loopback address, `KEYFILE` the tunnel key's file. None is a value.

The try, from a clean checkout of `9d477477a175`:

    ON=CERTIFIER-NODE PORT=18746 MODEL=glm-5.3 TRIES=1 PREFIXED=1 PREFIX_BLOCK=64 LANES=64 RATE=1.67 REQUESTS=2000 SEED=3 bash certside/launch.sh

The abort job, same checkout, once the capture has stopped:

    ON=CERTIFIER-NODE PORT=18746 MODEL=glm-5.3 TRIES=1 PREFIXED=1 PREFIX_BLOCK=64 LANES=64 RATE=2 REQUESTS=40 SEED=4 DELAY=1 TAIL=30 TIMEOUT=10m bash certside/launch.sh

The six tries, from a clean checkout of `main` at `d7a5af15c9dc`:

    research run --on CERTIFIER-NODE --project verity --source "$PWD" --cwd source \
      --mem-gb 4 --mem-wait 300 --timeout 40m --custody-ttl 6h \
      --env PEER_AUTHORITY=USER@HOST:SSH_PORT --env FRONT_AUTHORITY=LOOPBACK:18746 --env 'TUNNEL_KEY=~/.ssh/KEYFILE' \
      -- bash benchmarks/network_traces/deviations.sh \
         --requests 8 --bucket-requests 64 --rate 1.67 --max-tokens 16 --tail 30 --cell-bytes 4096 --hold-ms 500 \
         -- --model glm-5.3 --watermark --slots 1 --bucket-ms 250 --delay 60 --lanes 64 --queue 64 --window 240 \
            --ingress-cells 4 --ingress-buckets 16 --egress-cells 4096 --max-tokens-cap 256 --prompt-max 1024

- `launch.sh` reads the serving node's ssh endpoint itself and prints no address, but its one line names both machines by their short names, so its output goes through the mask [read]. Its `ON` and `MACHINE` default to Wednesday's two machines, so `ON` is needed only if the certifier's node is another.
- `DELAY`, `TAIL`, `RATE`, `REQUESTS` and `SEED` pass through `launch.sh` to the job [read: its list of names].

**For the capture's reader**

- Each certifier job opens connections on the server's port that send nothing: the reach check's, two a `certside` job on Wednesday and one for the six tries' job [read]. They are a SYN and a FIN each, in no record of the try.
- Each job's forward is one ssh flow from the certifier's node, up from the job's start to its exit, with keepalives.

**A slip of mine:** I printed one run's telemetry events in my own shell without the mask, and a line named an instance. Nothing was sent or written anywhere.

**Not looked at:** vllm's server code (what it counts as a run's end or as a connection); infra's capture; the certifier's node as it is tonight (its memory, its uv cache); Wednesday's server record; how long `launch.sh`'s read of the ssh endpoint takes.

NET-396 beside this: the both-paths test and the suites have waited in the research pool's line since 11:18 AM PST; your yes on the run's tree is read, and the G = 1 run launches when the both-paths test has passed.

TIMER: none

[2026-11-12T19:51Z] 3ea47196f67b
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 11:51 AM PST. **Your 11:19 AM order is in the script. The toy by line B was launched at 11:16 AM, three minutes before it, with the script as it then was, and still waits for a slot. One question: a second toy with the final script, yes or no. I launch none without your word.**

My last message's header said 11:33 AM; it was sent at 11:08 AM PST. I wrote the hour from a guess. Every hour here is from `date`.

**The toy in flight** [ran]: `r20261112-191615-b258`, by main's launcher from a clean checkout of `origin/main` `0582899c4392`, `--on auto`, on a research pool and not on the pool's node (compared by the run's record, not printed). Launched once. It has waited 34 minutes for a slot. Its script holds the commit, the tree id and an unchanged tree, and that a commit of `main.txt` is in the history; it has no pin asked by name and no `on_main` line.

**The final script** [ran on the VM only]: SHA-256 `9276fb1a8c019efe6b906640ab8f5fff7a2e571de7a74d9671794ab967e95a09`; the reader is unchanged from the toy's, `1b01b724ef571f80818ff2ec4f3a4db81b5e82c1f19942aa537f4296315a7973`.
- Before anything runs: the newest commit of `main.txt` in the clone's history must be `TREE_PIN`, else exit 66. That is git's count, not `on_main`: the function needs the tree's Python environment, which only the build makes.
- Right after the build, before the warm and the audit: the tree's own `on_main`, held to `TREE_PIN`, else exit 66. In a clone of the real input with its environment the script's lines print `15d964e7e829023c475bbdb55e3e9dd456223be3` in under a second.
- The toy on the VM by line B, three steps: the real input, exit 64 at the empty kept dir after the tree's line; the toy's tree, exit 0 with the pin's line, the `sampled:` line and the workers' line; the real input with another pin asked, exit 66. Line A's toy with the same script prints the same lines as before, numbers aside.

**What a second toy would add:** the pin's two lines through main's launcher on a machine. What it would not: main's launcher on the `network` pool with five hours, which no toy shows.

**GitHub** [read]: neither line reaches GitHub for our repository, at the launch or in the job. The source goes to the machine as git objects over the launcher's own ssh; the job's clone is of the sent file; `uv.lock` names no git source. The job reaches a third party's public repository only where the node's shared prover checkout is missing. From the VM, my inbox reaches `origin` with the credential in the checkout's URL, and the notes their own repository with `RESEARCH_NOTES_TOKEN`. So with the repository private both lines launch if the VM still reads `origin`. The table is in `LAUNCH.md`.

The rest at 2 PM. I go on with the port.

TIMER: none

[2026-11-12T19:51Z] 50af8deb22dc
CERTNEG, Thu 12 Nov 11:51 AM PST, NET-396. **The G = 1 run is launched, once: `r20261112-195057-ab2e`, 11:51 AM PST, on `main` at `4a90174415b7` plus my two commits (head `0212b89c7cdc`). Lean's tip is not run. It should end about 12:45 PM PST.**

- **One change from my 11:24 AM line: I launched before the both-paths test passed.** That test (`r20261112-191806-4bd2`) and the suites (`r20261112-191856-299f`) have waited in the research pool's line since 11:18 AM PST and have not started [read: `research status`, 11:47 AM PST]. If the test then fails on the fields, this run's head is stale and I say so before I report a split.
- [ran] It took the baseline's slot at once (`n0`, 21 of its 23 cores pinned, 128 GB reserved) on the baseline's machine [read: the two records' machine equals this launch's, compared without printing it].
- [read: `job.sh`] The prover's build cache is keyed by source, so this run rebuilds the prover first. Its start is longer than the baseline's 70 s; its two numbers are not touched by that.
- Your 11:41 AM line is read: the label and the body say which `main` it was and that lean's tip was not run; the stopped session's short record goes in the body as not covered; the GPU feature set's build under "not looked at".
- Tonight's timeline went to your inbox at 11:50 AM PST.

TIMER: none

[2026-11-12T19:52Z] 7e93064fa5af
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 11:52 AM PST. Your 11:32 AM PST question (PRF-600's mismatch). Read-only: nothing built, nothing run.

**The period check and its writers agree on one id at both heads, so no period reads `undecided` or rejects because of the id. The mismatch is one layer up: the id is one-stage v1's, the answer is read out of the period's own evidence, and SPOT, the composites' integrity subprotocol, neither writes nor takes a v1 answer. No Lean computed the answers our period accepts.**

Heads: M = `origin/main` `0e8d16c3b8be`; L = lean's merge head `origin/cursor/lean-remote-merge-onmain-fc8d` `64ee10f17900` (my fetch, 11:40 AM PST). Lines are M / L. Every line below is [read] unless marked.

**1. The id our period check reads, and its readers on our side**

- **The id:** `verity/sampled-proofs/one-stage/v1`, stated as a literal of the period check's own: `NetworkCheck/Period.lean` `oneStage`, 250 / 270. It names neither Spot's constant (`Spot/Registration.lean` 101 / 108) nor the registry's (`Identity/Domains/Shared.lean` 51 at both).
- **The Lean reader is one function**, `answered`, 861-870 / 921-931: another id is a **reject**, not an undecided (862 / 922); then the answer's digest must be the registration's (863 / 923); then accept gives the profile, `undecided` leaves the finding undecided (869 / 929), anything else rejects.
- **Its three callers:** the compute audit's answer (`auditCheck`, 1096 / 1189), the scan's answer in the rows form (`rowsWindowCheck`, 1085 / 1178) and in the clock form (`windowCheck`, 893 / 953). The scan's own answer is read under the same v1 id.
- **Where the answers come from:** the period's evidence (`Evidence.audits`, `Window.answer`, 400-402 and 391 / 422-424 and 413). The module's docstring says so: "that the audits' answers are the sampled-proofs component's own, which the composition's theorem is to bind" (228-229 / 248-249).
- **The Python mirror:** `verity/protocols/certifier/verifier/network/tests/period_case.py`, `ONE_STAGE = R.PROTOCOL` (83 / 97) and its check (746 / 785).
- **The writers:**
  - the compute audit's answer: `benchmarks/prove_path/records_audit.py period` (698 / 796) through `benchmarks/one_stage/served_integrity.py` `answer` (`PROTOCOL = R.PROTOCOL` 49; 53 and 198; the file is the same at both heads);
  - the scan's answer: `benchmarks/network_traces/scan_zk.py` `window_answer` (1460, 1468 / 1657, 1665);
  - the case: `benchmarks/network_traces/scan_period.py` `rows_case` copies both into the evidence (102 / 114). At L only, `certifier_piece.py period` builds the case from the compute audit's `period.json` (511-521).
- **The difftest's cases:** `period_case.py` writes its answers under `ONE_STAGE` (1135 / 1236) and its wrong-protocol cases under `verity/sampled-proofs/two-stage/v1` (1607, 1669 / 1839, 1901).
- **Tests that name the id:** `benchmarks/network_traces/tests/test_scan_zk.py` 413 / 466, `test_certifier_piece.py` 158 / 162, and `test_block_certifier.py` 81, which uses it only as a wrong id for the period's own answer.
- **Kept fixtures:** none holds a compute audit's answer. `certifier-period-answers.json` holds the period's own answers, every `protocol` in it `verity/certifier/v0`. The four JSON fixtures that name v1's id are one-stage's, under `benchmarks/one_stage/interfaces/tests/fixtures/`.
- **`security/`:** no file names `oneStage` or either id, at either head. The witness of a refused answer uses the id `other` (`security/Proofs/Properties/NetworkPeriod.lean` 211-213 / 537-539).

**2. What the other side takes and refuses, and what a period reads today**

- **SPOT's id is `verity/sampled-proofs/v1`** (`VStarRecursive/Sampled.lean` 33 at both). It refuses a one-stage statement that names no proofs, which is v1's (`V1_REFUSED` 38-41; the parameters at 60, the evidence at 67). The file differs between the heads only by `main` added at L (97-101).
- **Both composites name SPOT, not v1:** `CoreOnsite.lean` 63-65 (components) and 119 ("`verity/sampled-proofs/one-stage/v1` is not a component of this composition"), the same at both heads.
- **A compute audit run on `main` today writes v1's id:** `served_integrity.answer`, 198. So does one at L. At L only, `benchmarks/one_stage/served_cases.py` also writes SPOT's case and answer beside it, for the composites, and says no composite takes v1's case and a served window of several members has no SPOT case (its docstring, 23-24 and 32-34). The file is not on `main`.
- **So a period that holds a real compute audit from `main` today reads the audit's own verdict:**
  - **accept**, by `answered` 865-868 / 925-928, when the audit accepted and its record names the key and session its coins verified under and says whether it wrote the case (`served_integrity.py` 176-198);
  - **undecided**, by 869 / 929, when the audit accepted without them (`served_integrity.py` 176-191);
  - never either one because of the id. I ran none.
- **proofs' second sentence holds at both heads:** one undecided compute audit leaves the period undecided unless a finding rejects (`firstFinding` 1144 / 1236; `check` 1256-1257 / 1377-1378).
- **What does not hold is the docstring's claim.** An answer SPOT computed carries SPOT's id (`Core.Protocol.answer` sets `protocol := P.id`, `CoreProtocol/Protocol.lean` 109), and `answered` would reject it at 862 / 922. The answer `answered` accepts is the one Python wrote. Nothing in the composition binds it: onsite's ties are the record tie and the store (`CoreOnsite.lean` 82, the same at both heads), and neither names the period check.

**3. Which side is right, and the change that settles it**

- **SPOT's side is right** [argued]: v1's verdicts are the evidence's, and an answer in our period's evidence is the prover's word until a Lean verifier computed it. lean's merge changes none of these Lean lines but their numbers.
- **The literal alone cannot be flipped today** [read]:
  - SPOT's one-stage form takes one `--zk` member; a served window of several members has no SPOT case (`served_cases.py` at L, 32-34). So a real period's compute audit would have no accepting answer.
  - The scan's own audit has four members and SPOT's per-member form accepts nothing (`Spot/OneStageZk.lean` `checkMembersAt_refuses` 1262, and `ROW_TABLES` 944-946, the same at both heads). So no scan would have one either.
  - A flip without the composition's binding only renames whose word the period takes.
- **The change that settles it, in two parts** [argued]:
  1. The composition hands the period check SPOT's own answer for each compute audit's record, or a tie holds the evidence's copy to it (item 4). Then `answered` reads SPOT's id for a compute audit. Files: `NetworkCheck/Period.lean` (`oneStage`, `answered`, the docstring), `CoreOnsite.lean` (`ties`), `period_case.py` (the mirror and its cases), `records_audit.py period` or `served_cases.py` as the answer's source, `scan_period.py`, and the three test files of item 1.
  2. The scan's answer stays Python's under v1's id until SPOT's per-member form accepts. It is named as open where the period check's guarantee is stated, as job 6's body already says of the member's tie.
- **Statements and lock records of ours:** no statement names the id. `answered` is read by a statement (`security/Properties/NetworkPeriod.lean` 76 / 135) and by `check`. So a change of `oneStage` or `answered` moves the records that read them, with no statement's text changed [argued, from the records rule my run hit today]: the same records job 6 moves, soundness's 23 and proofs' 1 among them, a second time.
- **Where it belongs** [argued]: in an issue of its own under NET-352, whose goal is that the period check reads verified answers.
  - Not job 6's pull request: job 6 changes what a scan's registration names, not whose answers a period takes, and part 1 needs SPOT to answer for a served window first.
  - Not NET-383: that is the onsite record's refusal of a local session's coins, in Python, on the answer's `public.coins`.
  - If you want job 6's body to carry it, one line under "Not established": "The period check reads each audit's and each scan's answer out of its own evidence under one-stage v1's id; no Lean computed it, and the composition does not bind it." Say the word and I add it.

**4. The tie proofs describes**

- **What it would read on our side:** the period's commitment's `audits`, the compute audits' registration records (`Commitment.audits`, 352 / 374; `commitmentOf`, 796 / 856), each by `Flock.WindowRecord.digest`, the digest `answered` already holds each answer to (863 / 923). The other side is SPOT's commitment's digest (`Spot/OneStage.lean` 406-409 at both; `Sampled.lean` `digest c := c.digest`).
- **Job 6's rows form adds nothing it needs and takes nothing from it:** the audits' registrations are in the period's commitment at `main` already, and job 6's constants are about the scan's registration, not the audits'.
- **Two limits of a tie in today's form** (`CoreComposition/Tie.lean` 19-31, the same at both heads):
  - A tie holds that every value its subprotocols name is one value. That gives "every compute audit of the period is SPOT's one record". "Exactly the registrations the period accepted, each once" over several records needs one SPOT case a record, which a composite of one sampled-proofs case does not hold.
  - A tie reads cases, not answers. It binds the record. The profile and the bound the period reads off the evidence's answer (`factsOf`, 1057 / 1143) stay the prover's unless the composition passes SPOT's answer in.

**Not read:** whether a5 sets `coins.verified` in a local development session (I read where it is written, `a5.py` 479 / 490, not when); that `Spot.Registration.digest` is `Flock.WindowRecord.digest` (`answered`'s docstring names both); PRF-600's branch; the remote composite beyond its ties' names.

TIMER: none

[2026-11-12T19:58Z] 98561a106d75
TERMGB to the network-accounting coordinator, Thu 12 Nov 11:58 AM PST. **NET-394 is open and unmarked: [#9454](https://github.com/compute-verification/verity/pull/9454), head `efe2fcdb33e81ff05842cff0a2e27138ccd1fb7d` on `cursor/na-certifier-term-simulated-b4e6`, 5 files, +667 -153 against `main` `0e8d16c3b8be`, all under `benchmarks/`. Its own suites pass on a node; the harness suite fails in the 16 tests I named, and nowhere else.**

**The run** [ran, launched once at the head; read: its status and its log]
- `r20261112-194109-40df`: `suites.py --changed 0e8d16c3b8be --quick`, 8 suites. Failed, exit 1: 7 passed, 1 failed, none cached.
  - passed: `benchmarks/certifier_term` 15; `verity-network-traces` 652; `verity-network-sim` 21; `verity-network-e2e` 70; `benchmarks/one_stage` 637; `repository` 352; `verity-graph` 220.
  - failed: `verity-bench`, 547 passed and 16 failed. All 16 are `tests/test_exfiltration.py`'s, the same 16 as on the VM.

**The readers** [read]
- `benchmarks/certifier_term/run.py`: changed (the card's measurement).
- `benchmarks/harness/verity_bench/exfiltration.py` and its test: not changed; the places are in my 11:42 AM message.
- `benchmarks/network_traces/chain_record.py` and `deviations.py`: read `per_link`, `at_link_rate`, `terms` and constants, not the document, the label or the marks. Unchanged; their tests are in the network traces' suite, which passed.
- No file outside `benchmarks/` reads the document or the result [ran: a search].

**The meetings** [ran: `git merge-tree` at the head]
- `main` `0e8d16c3b8be`: merged in, no stop.
- lean's branch at `d1f71265b06e`: clean by text. Both change `leak_bounds.py`: lean's sets `OF_RECORD = JITTER_BOUND`, with `JITTER_BOUND = 2`, one line above the label's comment I rewrite. On a trial merge the calculator's 34 and the runner's 15 pass, and the harness test fails as at the head [ran on the VM, nothing pushed].
- #9305 and #9329: no file in common. The trial merge stops in `benchmarks/one_stage/tests/test_audit_two_members.py`, which is `main`'s own stop with each of them since #9362 landed (11:13 AM PST) [ran: `main` alone stops there].
- #9378's head `1d6ebf70ff8c`: clean, no file in common.

**R at the benchmark's phase, and how** [ran]
- T = 1,280 buckets, D = 30, ingress buckets 30, six classes. A lane releases ⌈(1,280 + 30 - 1) / 31⌉ = 43 a phase.
- A link serves 1,540 requests a phase. The least lanes that hold them are 36, so R = 36 x 43 = 1,548, and the count is 6 x log2(1,549) = 63.58 bits a link a phase.
- Over the world's year 91,778 GB, 305.9 times the target; a datacenter's year 24.48 GB. The body holds the three rows for a week, a month and a year at both figures, with the count's share: 0.348% and 0.374% of `design, no stop block`, 4.73% of `design` at D_j = 2 and all of it at D_j = 1.

**Not established**
- The count's rule is not on `main`; it is read from #9378.
- The schedule is assumed: no run served this load at 36 lanes.
- `lengths` and `all three` without the count is argued from the forms.
- ε is over 1 for the datacenter and the world at every period, as on `main`; the command says it bounds nothing there.
- The printed tables gained a `count` column after `cells`. The other columns keep their order, but this is a change to the columns: say if you read "column order" as none added.

**Slips**
- The run I launched at 11:30 AM and cancelled at 11:40 AM, already named.
- My 11:42 AM message said "about 21 minutes to a slot"; the run started sooner and had ended when I read it at 11:56 AM.

**NET-393, a plan line; not started** [read: the issue, and where the tree names `PeriodLeakAccepted`]: one pull request under `benchmarks/certifier_term`. (1) The window count from the calculator's own link-phases: 4,725 a week, 20,250 a month, 246,375 a year for a link, times the links a scope pools. (2) The δ a window's term is at, from `NetTiming.Guarantees.PeriodLeakAccepted`'s statement and the instance the lock names, with its source. (3) The period total at three sizes: the card's floor, one request of 3,320.5 tokens, and the class audit's link window. (4) A data file for that window's 1,843 requests, with where the number comes from. First step when you say go: I read the statement and row 60's arithmetic and send you the formula before any code. Open now: the source of 1,843, and that the total at the link window is vacuous until proofs changes row 3, which the table would say.

**Also done:** #9329's body holds the weights-tie paragraph as you worded it (head unchanged, `c77b01b8b79a`, unmarked).

**Next:** lean's branch at its ready line (#9375 carried no mark at 11:56 AM PST; the retake stops in one file, the two-member test, where #9362's close goes), then the 1:30 PM report.

TIMER: none

[2026-11-12T20:03Z] 01121929f769
BOUNDTIE to the network-accounting coordinator, Thu 12 Nov 12:03 PM PST [clock read]. **NET-388's pull request is open and unmarked, and my turn ends with this line.** Each line is marked.

**The pull request**
- #9457, `cursor/na-turn-away-on-tick-9d4f` at `259aadb07b66c3767d49b3950e29933c20f7e6d3`, eleven commits on `1d6ebf70ff8c`, 10 files, none outside the certifier's package, its tests and `benchmarks/network_traces/` [ran]. Opened by `research queue open`, linked to NET-388, no mark. `gh` could not read it back (its API rate limit), so its state is as the command printed it.
- The body has the rule, the three conditions each with its test, the issue's five other tests, the stream's end as its own section, the share of a link's time, what the cell charges, what I decided that no one ruled, and the limits.
- **One line of the body is one status behind:** it has `r20261112-193414-73a6` as running, and it is done (below). `research queue edit` was refused by the same rate limit at 12:03 PM PST. I put it right with the other runs' statuses in my next turn.

**The runs at this head are launched and not read** [ran: each `research status`, last at 12:02 PM PST]. The pool gave them no slot between 11:46 AM and 12:02 PM PST. I read them first thing in my next turn and put their statuses in the body.

| run | head | what | status |
|---|---|---|---|
| `r20261112-193050-3071` | `23f3446303b0`, one commit before the head | certifier's verifier suite; its `network` suite; `verity-network-traces` | done, rc 0: 1425 passed 1 skipped; 85 passed; 617 passed; 0 cached |
| `r20261112-193153-34b1` | `23f3446303b0` | ratchet gate against `origin/main`, then against `1d6ebf70ff8c` | failed, rc 1: retired-term sites alone, 72 over the three changes, 9 of this one's, all the code's own names |
| `r20261112-193127-2728` | `23f3446303b0` | `repository`, whole | waiting: not read |
| `r20261112-194600-eb11` | `259aadb07b66` | the three suites | waiting: not read |
| `r20261112-194636-9134` | `259aadb07b66` | `repository`, whole | waiting: not read |
| `r20261112-194733-aba8` | `259aadb07b66` | ratchet gate, as above | waiting: not read |
| `r20261112-194838-6e3f` | scratch C `34509dd7346c` | the three suites and `repository` | waiting: not read |
| `r20261112-193414-73a6` | scratch C one commit earlier, `804ffaebd19d` | the same | done, rc 0: 1480 passed 1 skipped; 91 passed; 682 passed; `repository` 358 passed; 0 cached |

- At the head, locally, one file at a time: the four test files the change touches pass (15, 8, 19, 23), and the gate against `1d6ebf70ff8c` names the same 9 sites and nothing else [ran].

**What changed after my 11:27 AM PST line**
- **The gate's two silent handlers** are gone: the pacer reports a bucket unwritten where a link takes its last part and not the stream's end, and the device counts a turned-away link its client resets (`turned_away_reset`, in `warden.json`) [ran].
- **The read of a turned-away link is bounded**: at most a byte past the cells a request may hold (`ingress_cells`), as an admitted connection's ingress is. A link that sends more is read no further and is closed at the run's end [ran]. This is the one commit after `23f3446303b0`.
- **Why the bound, and one thing for soundness** [argued, not measured]: the device reads a link on its loop. When the loop reads, the kernel may open that link's TCP window to its client, so a client that sends enough sees when the loop read, and a server's writes move the loop. That holds today for every link the device accepts, up to what the stream's reader buffers. Without the bound, one turned-away link would have given its client that view without limit for `span` buckets. With it a turned-away link gives what an admitted one does. Whether that signal is charged anywhere I did not find in `PROTOCOL.md`; it is soundness's to say.
- **Mutations:** 18 changes to the device and the pacer, each failing at least one test, none hanging [ran].

**Scratch C**
- `cursor/na-scratch-c-388-lean-4f8a` at `34509dd7346c1c205ef3a858b5d4bc5c29d1529c`: NET-388's head merged onto scratch B, in two merge commits (the head before the bound, then the head). Nothing stopped. Its tree is scratch B's merged with the head directly [ran: `git merge-tree`].
- NET-388 shares eight files with #9375's branch: `warden.py`, `pacer.py`, `warden_serve.py`, `PROTOCOL.md`, the README (49,046 bytes merged), `test_network_certifier_pacer.py`, `test_network_certifier_warden.py`, `test_warden_serve.py`.
- **The one place they meet in behaviour** [read: that branch's `pacer.py` `main` and `_drain`, `Warden._report`; argued from the order]: there a part whose link has not reached the pacer by its release is lost and the window is faulted. A turn-away's link is sent in its cell's tick, before that bucket is handed, and the pacer takes every link its channel holds before a release that misses one. The four test files pass on scratch C locally, the paced turn-away among them [ran].

**The share of a link's time** [arithmetic on the cost runner's schedule]: `span` is 542 buckets, 54.2 s a run. A run of a week 0.0090%, of a month of 30 days 0.0021%, of a year of 365 days 0.00017%. A device that runs a day at a time: 0.063% at every period.

**Decided here and ruled by no one:** the stream's end with the cell, the open link, the read's bound and the reset's count; the pacer's keys for turn-aways (2³² − n); a link already closing at its tick gets no cell; that the pacer now ends the stream with every link's last part, an admitted connection's too.

**Not looked at:** a served run at form 2 under the pacer; anything on a network that is not loopback; a turn-away in the run's last bucket over TCP; the ruling's own text (I have it from your line).

**Your 11:45 AM PST line** reached me through the shell hook. I did not start #9378's change or the read for soundness. I end the turn here, at the pull request, so that your turn reaches me.

[2026-11-12T20:07Z] 4f08aa64c2a8
# NET-285: `ScanViewCut`'s theorem compiles

SCANCOUNT (bc-a6f565e0-c510-570e-873d-53e7770b1924) to the network-accounting coordinator, Thu 12 Nov 12:08 PM PST.

**`scanViewCut : ScanViewCut` elaborates, at its first try, with the three standard axioms only. It takes `Flattens`
and nothing else, so I do not call it proved: `Flattens` is `open` on lean's branch.** No answer is owed.

- [ran] r20261112-191514-7613, done, rc 0, at tree `62acac59ba4d`: "lean exit 0 in 378 s"; the axiom probe: "none of
  its 6 theorems uses an axiom beyond propext, Classical.choice and Quot.sound".
- [ran] Its `lake build` rebuilt `Walk.lean`, `Accept.lean`, `Fits.lean` and `Causal.lean` with my changes: all
  built. No warning is on a line of mine.
- [ran] `--siblings`: no name clash among the 1,268 built modules of the package. 2,707 modules had no build to check
  against.
- **Not checked yet:**
  - The head `1c8fa449dcd7` rewrites two steps of one lemma after that tree. Its try r20261112-195003-6300 is in line.
  - `security/Proofs/Properties/NetworkScan.lean`, which reads the two files I extended: its try is launched.
  - lean's branch after `238ad6621c27`.
- **What the proof shows**, as soundness's point 4 asks: a template's body reads, outside its own gates, only what its
  parameters resolve to (`WalkProof.memberReads_params`). The egress template has one parameter, `state`. So no
  egress gate depends on a free input off the views, for every `T ≥ 1`.
- **Left for a listing:** its two witnesses. The nontrivial one needs a Program built by hand with one input gate and
  its flattening; the tree has none today. No lock is changed.
- I start no other proof before your read.

TIMER: none

