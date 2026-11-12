module

public import Proofs.Flock.SpotFlock.Code

/-!
# Round 2's check gives the proved verifier's statement checks: the statements

The two-stage guarantees are stated of a modeled verifier, `SpotFlock.V h`, over the auditor's copies `h`. Its round
`stage + 1` checks the session statements (`stmtOk`). The verifier that runs checks them in round 2
(`Spot.SecondStage.check`, called by `Spot.Run.run` with what it holds of the run, an `Own`). This module states, clause
by clause, what a passing round 2 gives the model's statement check; `RunsAs.Round2` proves each.

**What the model reads of round 2, and what it does not.** The model's `readsOk` is round 2's own interior order and
names (`interiorOrderOk`, `namesOk`) and per-statement checks (`codesOk`, which is `SecondStage.stmtCodes`), at what
the model holds of the run (`ownOf`). So they follow from round 2's, where the running verifier holds what the model holds (`OwnAgrees`) and the
statements round 2 reads are the session statements' (`StmtsAre`). The rest of `stmtOk` round 2 does not check, and each
is a premise here, named for what it says:

* `TwoHolds`: the design evaluates (`two`: the two-stage scope, the layout fitting the members' files, each reference
  statement reading as a member's). It reads only the auditor's copies, never the run.
* `RecordsHeld`: the run's records name their registrations' names (`recordsOk`), which `Run.run`'s registration
  step checks, not round 2.
* `NamesDistinct`: the templates' names are distinct. Round 2 takes the statements by position and finds a statement's
  part by its template's name, so with two parts of one name it passes a run with fewer statements than parts.
* `RoundsHeld`, `ScopeHeld`, `PinHeld`: each session statement has its template's reference statement's rounds, is in
  scope (`sessionOk`) and is the reference statement but for its sample, registered reads and public file (`pinOk`).
  Round 2 reads a session statement only as the JSON `Stmt` (its template, its public file's header, its sample and its
  registered reads) and the digests a verdict of record names of it; no check of round 2 reads its circuit, its tables,
  its flags or its rounds.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Claims
open Lean (Json)

/-! ## What the running verifier holds, against the model -/

/-- **The running verifier's `Own` on what the model reads**: the fields the model's (`ownOf`) leaves at their defaults
set to them: the reads the verifiers of record report (`reported`), what they name of each session (`verified`), the
names of what the replay block returns (`returns`), the values the verifier supplies or fixes (`supplied`,
`fixed`) and the templates the auditor's params record as reads (`reads`). Round 2's per-statement checks read none
of these but `verified`. -/
def core (own : Spot.SecondStage.Own) : Spot.SecondStage.Own :=
  { own with reported := none, verified := #[], returns := #[], supplied := #[], fixed := #[], reads := #[] }

/-- **The running verifier holds what the model holds of the run**, at the run's statement `st`, round 1's record `rec`,
the second stage's draw and round 0's coin: the replay sample, the parts, the two records, the second draw's units, the
design's links, wires and rows, and the templates' reference headers and registered reads (`ownOf`). -/
def OwnAgrees (h : Held) (own : Spot.SecondStage.Own) (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin) : Prop :=
  core own = ownOf h st rec draw c₀

/-- **The templates' names are distinct**, as the members' order names them. -/
def NamesDistinct (h : Held) : Prop :=
  ((List.range h.ms.length).map fun t => h.names.getD t "").Nodup

/-- **The statements round 2 reads are the session statements'**: the message `m₂` states as many session statements as
`ss` has statements, and the `t`-th decodes as a statement whose reading as round 2 reads one (`stmtJsonOf`) is the
`t`-th of `ss`'s. -/
def StmtsAre (h : Held) (m₂ : Msg) (ss : Array Json) : Prop :=
  ∃ xs, stmtsOf m₂ = some xs ∧ xs.length = ss.size ∧ ∀ t < xs.length, ∃ w,
    FlockDeploy.decode (xs.getD t (h.xS₀ t)) = some w ∧
      stmtJsonOf h t w = Spot.SecondStage.Stmt.ofJson? (ss.toList.getD t .null)

/-- **The design evaluates** (`two`), at every statement: it reads only the auditor's copies. -/
def TwoHolds (h : Held) : Prop := ∀ s, (two h s).isSome = true

/-- **The run's records name their registrations' names** (`recordsOk`). Round 2 does not make it: `Run.run`'s
registration step (`Registration.check`) does. -/
def RecordsHeld (h : Held) (x m₁ : Msg) : Prop := recordsOk h x m₁ = true

/-- **Each session statement has its template's reference statement's rounds.** No check of round 2 makes it. -/
def RoundsHeld (h : Held) (m₂ : Msg) : Prop :=
  ∀ t < h.ms.length, (FlockDeploy.V true).k (stmtAt h m₂ t) = k₀ h t

/-- **Each session statement is in scope** (`sessionOk`). No check of round 2 makes it; `second-tables` holds only
that a verdict of record set the session up as one table. -/
def ScopeHeld (h : Held) (m₂ : Msg) : Prop :=
  ∀ t < h.ms.length, sessionOk h t (stmtAt h m₂ t) = true

/-- **Each session statement is its template's reference statement** but for its sample, registered reads and public
file (`pinOk`). No check of round 2 makes it: round 2 holds the public file's header and the registered reads to the
reference's, and reads nothing else of the statement. -/
def PinHeld (h : Held) (m₂ : Msg) : Prop :=
  ∀ t < h.ms.length, ∀ w, FlockDeploy.decode (stmtAt h m₂ t) = some w → pinOk h t w = true

/-! ## The statements, clause by clause -/

/-- **A passing round 2 holds the interior record to the replay sample as the model does** (`interiorOrderOk`). -/
def InteriorOrderOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin),
    OwnAgrees h own st rec draw c₀ → (Spot.SecondStage.check true own (some ss)).codes = #[] →
      Spot.SecondStage.interiorOrderOk (ownOf h st rec draw c₀) = true

/-- **A passing round 2 holds the interior registration's names to the boundary record's as the model does**
(`namesOk`, the model's once its `readsOk` reads it). -/
def NamesOkOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin),
    OwnAgrees h own st rec draw c₀ → (Spot.SecondStage.check true own (some ss)).codes = #[] →
      Spot.SecondStage.namesOk (ownOf h st rec draw c₀) = true

/-- **A passing round 2 passes the model's code check of each statement** (`codesOk`) at its position. -/
def CodesOkOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin),
    OwnAgrees h own st rec draw c₀ → (Spot.SecondStage.check true own (some ss)).codes = #[] →
      ∀ t < ss.size, ∀ S, Spot.SecondStage.Stmt.ofJson? (ss.toList.getD t .null) = some S →
        codesOk h (ownOf h st rec draw c₀) t S = true

/-- **A passing round 2 reads one statement a template**, where the templates' names are distinct. -/
def CountOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin),
    OwnAgrees h own st rec draw c₀ → NamesDistinct h → (Spot.SecondStage.check true own (some ss)).codes = #[] →
      ss.size = h.ms.length

/-- **A passing round 2 gives the model's `readsOk` at each template**, where the session statements are their
templates' reference statements but for the run's fields (`PinHeld`). -/
def ReadsOkOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (x m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h)
    (st : Roots) (rec draw : Json),
    decode x = some st → jsonMsg m₁ = some rec → secondDraw h c₀ c₁ = some draw → OwnAgrees h own st rec draw c₀ →
      StmtsAre h m₂ ss → PinHeld h m₂ → (Spot.SecondStage.check true own (some ss)).codes = #[] →
        ∀ t < ss.size, readsOk h t x m₁ (stmtAt h m₂ t) c₀ c₁ = true

/-- **A passing round 2 gives the model's `stmtOkAt` at each template**, with the premises round 2 does not check. -/
def StmtOkAtOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (x m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h)
    (st : Roots) (rec draw : Json),
    decode x = some st → jsonMsg m₁ = some rec → secondDraw h c₀ c₁ = some draw → OwnAgrees h own st rec draw c₀ →
      StmtsAre h m₂ ss → TwoHolds h → RoundsHeld h m₂ → ScopeHeld h m₂ → PinHeld h m₂ →
        (Spot.SecondStage.check true own (some ss)).codes = #[] →
          ∀ t < ss.size, stmtOkAt h x m₁ (stmtAt h m₂ t) c₀ c₁ t = true

/-- **A passing round 2 gives the model's statement check** (`stmtOk`), with the premises round 2 does not check. -/
def StmtOkOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (x m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h)
    (st : Roots) (rec draw : Json),
    decode x = some st → jsonMsg m₁ = some rec → secondDraw h c₀ c₁ = some draw → OwnAgrees h own st rec draw c₀ →
      StmtsAre h m₂ ss → TwoHolds h → NamesDistinct h → RecordsHeld h x m₁ → RoundsHeld h m₂ → ScopeHeld h m₂ →
        PinHeld h m₂ → (Spot.SecondStage.check true own (some ss)).codes = #[] →
          stmtOk h x m₁ m₂ c₀ c₁ = true

end SpotFlock.RunsAs
