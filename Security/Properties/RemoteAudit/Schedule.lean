import Definitions.Accounting.Schedule
import RemoteAudit.Schedule
import Properties.RemoteAudit

/-!
# The schedule check fixes each credited pass's plan (`RemoteAudit.Schedule`)

`RemoteAudit.Schedule.check` (`verity/core/protocols/remote_bandwidth/lean/RemoteAudit/Schedule.lean`, which
`schedule.py` is held to by the difftest, and without whose acceptance `audit.ledger` credits nothing after the seed)
holds a session's passes and appends to the schedule the prover publishes, at the cap on the schedule's bits the
prover committed before the session's first coin, and returns the schedule's bits.

* `ScheduleChecked`: whenever the check accepts, the schedule is well formed, the passes are at its steps, each pass's
  plan is the schedule's plan at the pass's step (`Schedule.plan`, computed by the verifier), the appends are the
  schedule's (`Schedule.appendRuns`), and the bits returned are its encoding's (`Schedule.bits`), at most the cap.
* `ScheduleFixesLayout`: so the schedule fixes them: two sessions the check accepts on one schedule have the same
  passes and the same appends. A layout that follows a coin at a schedule the verifier counted is refused.

They take no premise. The schedule's bits (`scheduleBits`) are one-to-one on well-formed schedules
(`ScheduleCode.scheduleBits_injective` in `Proofs.Properties.RemoteAudit.Schedule`: LEB128 is prefix-free, and each
list comes after its length).

The model's layout (`Accounting.ScheduleDeterministic`, `Accounting.Assumptions.SteeredLayout`) is the verifier's:
`sessionLayout`, `codeLayout` of the session's appends and the runs the pass sent, tile `i` at `tileAt`, the
`position` at which `checkOpening` checks the block at position `i` of the pass's plan (`runBlocks`). Three properties
of the code tie it there: `PlanBlocks` (`Plan.block` is the runs' blocks in order), `CodeLayoutTiles` (a plan of `N`
blocks with positions in range is laid out at them) and `OpensAtPosition` (an accepted opening is checked at its
block's position, in that tree). `ScheduleLayoutGap`, that the layout is a map of the appends and the runs, holds of it
(`scheduleLayoutGap_code`), so the check's acceptance at the cap `a` at every coin gives `ScheduleDeterministic` at
`a` and `SteeredLayout` at `a + 1` (`scheduleDeterministic_of_checked`, `steered_of_checked`). The cap is fixed
before the coins, so it is one `a` at every coin, where the bits of the schedule published after them can follow the
coins.

Each property's premises hold at the deployed shape (`planBlocks_honest`, `codeLayoutTiles_honest`,
`opensAtPosition_honest`, `steered_of_checked_honest`), and each conclusion fails for an input an accept-all verifier
would take (`planBlocks_nontrivial`, `codeLayoutTiles_nontrivial`, `opensAtPosition_nontrivial`,
`steered_of_checked_nontrivial`), in `Proofs.Properties.RemoteAudit.Schedule`.
-/

open RemoteAudit

namespace Properties.RemoteAudit

/-- Whenever the schedule check accepts schedule `sc` over `blocks` registered blocks at cap `cap` for passes
`(step, runs)` and appends `(first, count)`, returning `b`: `sc` is well formed, the passes are at its steps in order,
each pass's runs are the schedule's plan at its step, the appends are the schedule's, and `b` is the schedule's bits,
at most `cap`. -/
def ScheduleChecked : Prop :=
  ∀ (sc : Schedule.Schedule) (blocks cap : Nat) (passes : Array (Nat × Array (Nat × Nat)))
    (appends : Array (Nat × Nat)) (b : Nat),
    Schedule.check sc blocks cap passes appends = .ok b →
      Schedule.malformed sc blocks = none ∧ passes.map Prod.fst = sc.steps ∧
        (∀ p ∈ passes, p.2 = Schedule.plan sc blocks p.1) ∧ appends = Schedule.appendRuns sc blocks ∧
        b = Schedule.bits sc ∧ b ≤ cap

/-- Whenever the schedule check accepts two sessions on one schedule, at any caps, they have the same passes, plans
included, and the same appends. -/
def ScheduleFixesLayout : Prop :=
  ∀ (sc : Schedule.Schedule) (blocks cap cap' : Nat) (passes passes' : Array (Nat × Array (Nat × Nat)))
    (appends appends' : Array (Nat × Nat)) (b b' : Nat),
    Schedule.check sc blocks cap passes appends = .ok b → Schedule.check sc blocks cap' passes' appends' = .ok b' →
      passes = passes' ∧ appends = appends'

/-- The schedule's encoding as bits, each byte low bit first: `Schedule.bits sc` of them (`scheduleBits_length`). -/
def scheduleBits (sc : Schedule.Schedule) : List Bool :=
  (Schedule.encode sc).flatMap fun b => List.ofFn fun i : Fin 8 => b.toNat.testBit i

/-- **The model's layout of a pass is a map of the session's appends and the pass's plan.** At each prefix `c` of the
session's coins and each codec and append coins `y`, the model's layout of the pass is `pos c` of the session's appends
`appended c y` and the runs `sent c y` the pass's plan sent. At the verifier's own layout (`sessionLayout`, `pos c`
its `codeLayout`) it holds of the code (`scheduleLayoutGap_code`). -/
def ScheduleLayoutGap {Pfx Y : Type} {N T M : ℕ} (pos : Pfx → Array (ℕ × ℕ) → Array (ℕ × ℕ) → Fin N ↪ Fin T × Fin M)
    (appended sent : Pfx → Y → Array (ℕ × ℕ)) (layout : Pfx → Y → Fin N ↪ Fin T × Fin M) : Prop :=
  ∀ c y, layout c y = pos c (appended c y) (sent c y)

/-- The store indices a pass's plan holds, in its order: its runs `(first, count)` concatenated. Position `p` of the
pass is entry `p` (`PlanBlocks`: `Plan.block` computes it). -/
def runBlocks (runs : Array (ℕ × ℕ)) : List ℕ := runs.toList.flatMap fun r => List.range' r.1 r.2

/-- **Where the verifier checks store index `j`** (`checkOpening`), over `blocks` registered blocks and the session's
appends `(first, count)`: `(0, j)`, in the registration's payload tree, for a registered block (`j < blocks`); `(k, j)`,
in append `k − 1`'s tree under its coin, `k − 1` the last append whose first index is at most `j` (`appendOf`); none
where no append starts at or below `j`. The model's position `(t, m)` is block `m` of append `t`, append `0` the
registration. -/
def position (blocks : ℕ) (appends : Array (ℕ × ℕ)) (j : ℕ) : Option (ℕ × ℕ) :=
  if j < blocks then some (0, j)
  else
    let k := bisectRight (appends.map Prod.fst) j
    if k = 0 then none else some (k, j)

/-- The verifier's position of tile `i` of a pass: `position` of the block at position `i` of its plan's runs. -/
def tileAt (blocks : ℕ) (appends runs : Array (ℕ × ℕ)) (i : ℕ) : Option (ℕ × ℕ) :=
  (runBlocks runs)[i]?.bind (position blocks appends)

open Classical in
/-- **The verifier's layout of a pass**, of the session's appends `appends` and the runs `runs` its plan sent: the
embedding of the pass's `N` tiles that puts tile `i` at `tileAt` of it, where the plan holds `N` blocks, each at a
position inside `Fin T × Fin M`, at distinct positions; `dflt` where there is none. -/
noncomputable def codeLayout {N T M : ℕ} (blocks : ℕ) (dflt : Fin N ↪ Fin T × Fin M) (appends runs : Array (ℕ × ℕ)) :
    Fin N ↪ Fin T × Fin M :=
  if h : ∃ e : Fin N ↪ Fin T × Fin M, (runBlocks runs).length = N ∧
      ∀ i : Fin N, tileAt blocks appends runs i = some (((e i).1 : ℕ), ((e i).2 : ℕ)) then h.choose
  else dflt

/-- **The model's layout, the verifier's**: at prefix `c` and coins `y`, `codeLayout` of the session's appends
`appended c y` and the runs `sent c y` the pass's plan sent, `dflt c` where they give no layout of `N` tiles. -/
noncomputable def sessionLayout {Pfx Y : Type} {N T M : ℕ} (blocks : ℕ) (dflt : Pfx → Fin N ↪ Fin T × Fin M)
    (appended sent : Pfx → Y → Array (ℕ × ℕ)) : Pfx → Y → Fin N ↪ Fin T × Fin M :=
  fun c y => codeLayout blocks (dflt c) (appended c y) (sent c y)

/-- **A plan's positions are its runs' blocks in order.** Of every plan the verifier makes (`Plan.make?`, which
`Plan.changed` ends in, and `Plan.whole`), its length is its runs' blocks' (`runBlocks`), and the block at position
`p` below it (`Plan.block`, a bisection of its running ends) is entry `p`. -/
def PlanBlocks : Prop :=
  (∀ (runs : Array (Int × Int)) (blocks : ℕ) (pl : Plan), Plan.make? runs blocks = some pl →
    pl.length = (runBlocks pl.runs).length ∧ ∀ p < pl.length, (runBlocks pl.runs)[p]? = some (pl.block p)) ∧
  ∀ blocks, (Plan.whole blocks).length = (runBlocks (Plan.whole blocks).runs).length ∧
    ∀ p < (Plan.whole blocks).length, (runBlocks (Plan.whole blocks).runs)[p]? = some ((Plan.whole blocks).block p)

/-- **Where a plan of `N` blocks has positions in `Fin T × Fin M`, the verifier's layout is them.** Of a plan the
verifier makes (`Plan.make?`) whose runs hold `N` blocks, each with a position below `(T, M)`, `codeLayout` puts each
tile `i` at its block's position (`tileAt`): the plan's runs are disjoint, so its blocks, the positions' second
coordinates, are distinct. -/
def CodeLayoutTiles : Prop :=
  ∀ (N T M blocks extent : ℕ) (appends : Array (ℕ × ℕ)) (runs : Array (Int × Int)) (pl : Plan),
    Plan.make? runs extent = some pl → ∀ dflt : Fin N ↪ Fin T × Fin M, (runBlocks pl.runs).length = N →
    (∀ i < N, ∃ t m, tileAt blocks appends pl.runs i = some (t, m) ∧ t < T ∧ m < M) →
    ∀ i : Fin N, tileAt blocks appends pl.runs i =
      some (((codeLayout blocks dflt appends pl.runs i).1 : ℕ), ((codeLayout blocks dflt appends pl.runs i).2 : ℕ))

/-- **The verifier checks each opened block at its position.** Whenever `checkOpening` accepts draw `(i, p)`'s opening
`o` (under the registration's digest and the receipter of pass `i`'s seed, as `Session` checks it), the block opened is
store index `j`, the block at position `p` of pass `i`'s plan, and `position` puts it at `(t, j)`: `t = 0` and its
decoding under the registration's key at index `j` is a payload at leaf `j` of the registration's payload tree, or
`t ≥ 1` and its decoding under append `t − 1`'s coin is a payload at leaf `j` of that append's tree. -/
def OpensAtPosition : Prop :=
  ∀ (reg : Registration) (appends : Array Append) (w : WindowRecord) (i p : ℕ) (o : Opening),
    checkOpening reg reg.digest appends w (receipterOf reg w i) i p o = none →
    let j := w.passes[i]!.plan.block p
    ∃ t, position reg.blocks (appends.map fun a => (a.first, a.count)) j = some (t, j) ∧ ∃ payload,
      (t = 0 ∧ Codec.decode o.block j reg.key none = .ok (some payload) ∧
        reg.payloadDomain.verifyOpening reg.payloadRoot reg.payloadDomain.count j (hidden payload o.payloadSalt) SCHEMA
          o.payloadPath = true) ∨
      (0 < t ∧ ∃ a, appends[t - 1]? = some a ∧
        Codec.decode o.block j reg.key (some (o.digest, some a.coin)) = .ok (some payload) ∧
        (appendDomain reg.digest a.first a.count).verifyOpening a.root (appendDomain reg.digest a.first a.count).count
          j (hidden payload o.payloadSalt) SCHEMA o.payloadPath = true)

end Properties.RemoteAudit
