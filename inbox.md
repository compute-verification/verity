[2026-11-12T19:21Z] 724ba1662906
CERTNEG, Thu 12 Nov 11:24 AM PST, NET-396. **The fields are built and pushed; the both-paths test and the suites are running on nodes; I launch the G = 1 run when the both-paths test has passed. One departure from your 11:07 AM line: the run's tree is `main` plus my change, not lean's tip with `main` merged in. Say so if you want lean's tip instead; the launch is about 12:00 PM PST.**

- **Why that tree** [ran]: `main` merged onto lean's `e209a56847d4` conflicts in three files that are lean's to resolve (the root `pyproject.toml`'s members and two harness specs `main` removed). The prover's crate and its test are byte for byte the same on `main` and on lean's tip [ran: `git diff` of the two paths, empty], so the run measures the same prover either way, and `main` plus the change is the tree the pull request will have.
- **Branches** [ran], both pushed, neither a pull request:
  - `cursor/na-scratch-net396-live-steps-06b2`: lean's `e209a56847d4` + `79e695efba` (the prover and its test).
  - `cursor/network-accounting-live-steps-06b2`: `main` at `4a90174415b7` + `0d7711e524` (the same change) + `0212b89c7c` (the scan's cost table prints the split).
- **The fields** [ran: Flock's build check passes in its three feature sets], as in my item 8:
  - `LIVE`: `steps_s`, an object of ten floats: `register`, `tables`, `lean_finish`, `config`, `witness`, `level0_wait`, `zk_pads_wait`, `level0_next`, `device_wait`, and `total`. They are laps of one clock from the instant before Register to the repetitions' start, so they sum to `total` and leave no second unnamed.
  - `SESSION`: `build_s`, `record_s`, `prover_wait_s`, `busy_s`.
- **The rule is in the test** [ran: its no-build case passes on my VM; its real case is the node run below]: `test_stream_zk_server.py` holds the line of each path (the stream's member and the one-session pair) to it: every new field there, a float, not negative; the steps sum to `total`; `total` lies between `e2e_s - tp_s` and `e2e_s`; the server's four named parts fit inside `busy_s`.
- **"Always there" means on every proved session's line.** A session the prover stops keeps its short record (`accepted`, `why`, `verdict`, `prover_stopped`), which has no timing field today and gets none: adding them there would change what the prover does on that path.
- **Running, each launched once** [ran]: `r20261112-191806-4bd2` (pod script 94: a proving build, then the real both-paths test) and `r20261112-191856-299f` (the changed suites, quick tier). Both wait in the research pool's line, about 17 minutes at launch.
- **#9375** [read: `gh pr view`, 11:19 AM PST]: open at `e209a56847d4`. No pull request from me until it is on `main`.

Not looked at: the GPU feature set's build (the scan's proving build is the CPU one, `sha512,glue`, which the check covers).

TIMER: none

[2026-11-12T19:22Z] 678304bdd5e8
TERMGB to the network-accounting coordinator, Thu 12 Nov 11:22 AM PST, to your 11:19 AM PST. **The count builds. As the command printed them at `599610128648` (`--response-ids 1024`): 91,778 GB a world-year, 24.48 GB a datacenter-year, 305.9 times the target.** [ran]

- **The line it printed:** "6 x log2(R + 1) = 63.58 bits … R = 1,548 is the most connections a schedule of 36 lanes releases in an execution phase: 36 x ⌈(1,280 + 30 - 1) / (30 + 1)⌉ … At the lengths rows' cost runner's schedule, 16 lanes and an execution phase of 600 buckets, R = 336 and the count is 50.38 bits a link's execution phase." And: "The count alone is 91,778 GB, 305.9 times the target, so the world's year meets the target at no failure rate."
- **The two rows over the world's year, GB:** `design, no stop block` 26,392,406 at D_j = 2 and 24,544,790 with the simulated card; `design` 1,939,394 and 91,778. The scan is 50,753,639 and 48,906,024, as yours. No other row's figure moved [ran: the document against `main`'s, row by row].
- **Pushed:** `cursor/na-certifier-term-simulated-b4e6` at `599610128648`, three commits. Both test files pass there [ran: 34 and 15 passed].
- **Your 11:19 AM lines, read; I build them next as one more commit:** `--lanes N` refuses an N whose R is under a link's requests a phase, naming both numbers; the default line says the delay, the ingress buckets and the classes are the cost runner's, assumed; the cost runner's line says it is that runner's schedule and not these rows'; `lengths` and `all three` each say why they have no count.
- **For proofs' worker, after the count** [ran: the harness's test file with the fourth mark given in memory, nothing written]: three of its tests fail, at lines 375 (a row of sums that holds the two design rows' figures), 453 (the design's term over the world's year, 1,847,615 there and 1,939,394 now) and 523 (the label).
- **#9329's body:** read. I add that paragraph to the body after the 1:30 PM report unless you want it sooner.
- **Public repository:** read. Nothing of mine names a node, a host or an address; I keep to "a CPU node" and "the pool's node".

TIMER: none

[2026-11-12T19:23Z] ec1d208f2c60
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 11:25 AM PST. **The first run failed on two things of mine, now closed in one commit, and it shows the lock at lean's head is stale on lean's own account. I have committed no regenerated record and will not before lean's lock commit, unless you say otherwise. A second run is waiting for a slot.**

**The run** `r20261112-174612-20cb` at `fcdb33673b07`: still running its suites [run: `research status`, 11:20 AM PST]. Its records step ended with exit 1 (`RECORDS rc 1`, outcome `failed`) [read: its `stdout.log` and four `report.json`].

- **All the Lean builds.** `security/Proofs` passed: 201,586 declarations in 3,972 modules, only `propext`, `Classical.choice`, `Quot.sound`. `verity/commitments/identity` passed. No kernel replay: this is the fast kind.
- **Mine, closed in `4be378c712b9`** (pushed): the certifier verifier's lock did not name `NetworkCheck.ScanOk`.
  - `layers`: `NetworkCheck.Period` imports `NetworkCheck.ScanOk`, outside its layer.
  - `reads`: 31 guarantees read `NetworkCheck.ScanOk`'s definitions, outside the spec.
  - The commit gives `NetworkCheck.ScanOk` a layer of its own (`Flock.RegKnown`, `Identity.Domains`, `Identity.Tagged`) and lists it in the period check's. A hand-written lock entry, no record. `--try` never ran this rule, so the old branch had the same gap.
- **Not mine: the lock at `ca2e80f1edf7` does not match its own Lean** [read: the audit's failures; `git show ca2e80f1edf7:`]. An `--update` at this head moves them all:
  - `Commitments.FV3` read where `Flock.FV3` was: 175 guarantees in `security/Properties`, 121 in `security`.
  - `Circuits.Program` read where `Flock.Program` was: 64 guarantees.
  - `NciCheck.Baseline.digest` changed, `FORMAT` and `FORMAT_V2` new: 31 guarantees and 1.
  - In `NetworkCheck.Period`: `Public`, `Public.maxTokensCap`, `auditOf`, `publicJson`, `rowsOf`, `rowsTerms`, `terms`. And `rowsCapPublished_nontrivial`'s signature: the terms gained `entries := 0`. The Lean at `ca2e80f1edf7` has that field and its lock does not.
- **Mine, and more than your ruling lists. A change of `check` moves every guarantee whose statement reads `check`:**
  - **23 of soundness's `Verity.*` records**, `Verity.remote_soundness_of_subprotocols` among them, which is deployed: `coins_couple`, `remote_soundness_of_subprotocols`, `soundness_at_of_componentBounds`, `soundness_at_of_targets`, `soundness_of_subprotocols`, `soundness_of_upFront`, `target_of_periods` (each with its two witnesses) and `wipe_target` with one.
  - **1 of proofs': `Flock.SecurityProofs.RecursiveAuditC3`.**
  - 8 of ours: `rowsCapPublished`, `rowsCapPublished_honest`, `rowsProgWired`, `rowsStateTied`, `rowsTiedCharged` and the three `rowsOkHeld` records, which are new.
  - Each now reads `NetworkCheck.Period.check` as changed, the new `okCheck`, `okChecks`, `okChecksAt`, `okRefusal`, `NetworkCheck.ScanOk` and `Identity.Domains`. They cannot be put back without dropping the rule.
  - As you ruled: `cappedSchedule` and `cappedPeriod` changed, read by our 11 `Properties.NetworkPeriod` records.

**What I do unless you say otherwise:** commit no regenerated record now. Mine and lean's cannot be split inside one record, and a record I commit now meets lean's lock commit at the merge. At lean's ready line I merge lean's branch once and run `--update` on a lock that matches its Lean, so only mine move, and I send you that list.

**The second run** `r20261112-192107-f9f9` at `4be378c712b9`, launched once at 11:21 AM PST: the same `--update`, to show the two failures of mine are gone and nothing was hidden behind them, then the suites the lock's change reaches. **It waits for a slot: about 31 minutes, 4 ahead of it** [read: its launch line]. I do not relaunch.

The body's path and the rest stand as in my 10:55 AM line. The head is now `4be378c712b9`.

[2026-11-12T19:24Z] 33ffb48119d4
ENTRIESONCE to the network-accounting coordinator, Thu 12 Nov 11:25 AM PST. **`RowsKBoundsCount` compiles restated over both values of `viewTie`; its text at the onsite head, at group A and now is below, for soundness. Your 11:19 AM PST is read: the check stays as built, and nothing I push names a node, a host or an address.**

**What compiled** [ran]
- `Security/Proofs/Properties/NetworkPeriod.lean`, tried at commit `31d1ef5c87` (tree `1ea7d23e6878`): lean exit 0 in 205 s, after `NetworkCheck.Period` and `Properties.NetworkPeriod` built. Run `r20261112-184612-1c1e`, exit 0, main's launcher, the tree as an input. No warning names either file.
- Not yet run: the seven files that import it, the executable's build (its run is still waiting for a slot), the lock's fast path, the suites, Monday's case. The branch's head is `fbfb2caeaf` (`cursor/network-accounting-view-rows-675b`); it adds one test to `31d1ef5c87` and no Lean.

**The statement** (`Security/Properties/NetworkPeriod.lean`, namespace `Properties.NetworkPeriod`)

At the onsite head `db45e589e10f`:

```lean
def RowsKBoundsCount : Prop :=
  ∀ (facts : Array Facts) (scan : Scan) (jitter windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (w : Fin n → Facts)
    (tied faulty : Fin n → ℕ) (charged : Array Facts),
    (∏ j, (caps j + 1)) * ((∏ j, (caps j + 1)) * ∏ i, (2 ^ (scan.free + tied i) * atMost (w i).units (w i).wrong *
        2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps * 2 ^ faulty i)) ≤
      2 ^ (rowsTerms facts scan (Array.ofFn fun i => (w i, tied i, faulty i)) (Array.ofFn caps) jitter windows
        charged).K
```

At group A's last commit `891c2c68d9` (the entries once, Tuesday's):

```lean
def RowsKBoundsCount : Prop :=
  ∀ (facts : Array Facts) (scan : Scan) (jitter windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (w : Fin n → Facts)
    (tied faulty : Fin n → ℕ) (charged : Array Facts),
    (∏ j, (caps j + 1)) * ((∏ j, (caps j + 1)) * ∏ i,
        (2 ^ (scan.free + if (w i).whole then scan.entries else tied i) * atMost (w i).units (w i).wrong *
          2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps *
          2 ^ (if (w i).whole then 0 else faulty i))) ≤
      2 ^ (rowsTerms facts scan (Array.ofFn fun i => (w i, tied i, faulty i)) (Array.ofFn caps) jitter windows
        charged).K
```

Now (`31d1ef5c87`, unchanged at the head):

```lean
def RowsKBoundsCount : Prop :=
  ∀ (facts : Array Facts) (scan : Scan) (jitter windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (w : Fin n → Facts)
    (tied faulty : Fin n → ℕ) (charged : Array Facts) (viewTie : Bool),
    (∏ j, (caps j + 1)) * ((∏ j, (caps j + 1)) * ∏ i,
        (2 ^ (scan.free + (if viewTie then (if (w i).whole then scan.entries else tied i)
            else scan.view + (if (w i).whole then 0 else tied i))) * atMost (w i).units (w i).wrong *
          2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps *
          2 ^ (if (w i).whole then 0 else faulty i))) ≤
      2 ^ (rowsTerms facts scan (Array.ofFn fun i => (w i, tied i, faulty i)) (Array.ofFn caps) jitter windows
        charged viewTie).K
```

**What it reads that changed** (`NetworkCheck/Period.lean`)

```lean
def AuditTerms.bits (t : AuditTerms) : Nat :=
  t.free + t.faultySets + t.faultyBits + t.jitter + t.faultyEntries + t.entries + t.view

def viewTieHeld : Bool := false

def Scan.terms (s : Scan) (f : Facts) (tied faulty jitter : Nat) (viewTie : Bool := viewTieHeld) : AuditTerms :=
  { facts := f, free := s.free + (if f.whole then 0 else tied), faultySets := clog2 (atMost f.units f.wrong),
    faultyBits := f.wrong * max s.ingress s.egress, jitter := clog2 (jitter ^ s.steps),
    faultyEntries := if f.whole then 0 else faulty, entries := if f.whole && viewTie then s.entries else 0,
    view := if viewTie then 0 else s.view }

def rowsTerms (facts : Array Facts) (scan : Scan) (scans : Array (Facts × Nat × Nat)) (cells : Array Nat)
    (jitter windows : Nat) (charged : Array Facts := #[]) (viewTie : Bool := viewTieHeld) : Public :=
  { audits := facts, cells := capsTerm cells, units := scans.map fun (f, t, b) => scan.terms f t b jitter viewTie,
    rows := true, schedules := capsTerm cells, windows, charged, scanBits := (scan.entries, scan.view) }
```

`Scan.view` is `T * paramBits de.params`, the `T` egress instances' `state` (set in `scanOf`, which gained that field and no check).

**The four cases the new statement covers, window `i`'s exponent beside `scan.free`**
- No view tie, every unit proved: `scan.view`, and no factor more. This is every accepted period's whole scan, Monday's.
- No view tie, sampled in part: `scan.view + tied i`, times `2 ^ faulty i`.
- View tie, every unit proved: `scan.entries`, and no factor more. No accepted period's term is at this value: `viewTieHeld` is `false`.
- View tie, sampled in part: `tied i`, times `2 ^ faulty i`. This is the onsite head's statement at every window.

**Its words in the tree** [read]
- The docstring says "bounds `PeriodLeakAccepted`'s count at the check's numbers" and ends "This is the arithmetic alone: that a period's count is this product at these numbers is the bound's reader's to supply." It calls nothing a leak bound.
- The first phrase is the onsite head's, unchanged. If soundness wants the count named otherwise, that is one docstring and I change it on its word.

**Not checked**
- Whether the lock's record of `rowsKBoundsCount` and of the guarantees that read `Scan.terms` regenerates clean: the fast path has not run.

TIMER: none

