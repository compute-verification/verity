module

public import Proofs.Flock.SpotFlock.Code

/-! What a passing round 2 gives the model's statement check, clause by clause; the clauses round 2 does not check
(`TwoHolds`, `RecordsHeld`, `InputsNone`, `NamesDistinct`, `RoundsHeld`, `ScopeHeld`, `PinHeld`) are premises. -/

@[expose] public section

namespace SpotFlock.RunsAs.Round2

open Claims
open Lean (Json)

/-- The model's `ownOf` with the fields it leaves at their defaults set to the running verifier's; round 2's
per-statement checks read none of these but `verified`. -/
def core (own : Spot.SecondStage.Own) : Spot.SecondStage.Own :=
  { own with reported := none, verified := #[], returns := #[], supplied := #[], fixed := #[], reads := #[],
             tables := #[], widths := #[] }

def OwnAgrees (h : Held) (own : Spot.SecondStage.Own) (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin) : Prop :=
  core own = ownOf h st rec draw c₀

/-- Round 2 finds a statement's part by its template's name, so with two parts of one name it passes a run with fewer
statements than parts. -/
def NamesDistinct (h : Held) : Prop :=
  ((List.range h.ms.length).map fun t => h.names.getD t "").Nodup

def StmtsAre (h : Held) (m₂ : Msg) (ss : Array Json) : Prop :=
  ∃ xs, stmtsOf m₂ = some xs ∧ xs.length = ss.size ∧ ∀ t < xs.length, ∃ w,
    FlockDeploy.decode (xs.getD t (h.xS₀ t)) = some w ∧
      stmtJsonOf h t w = Spot.SecondStage.Stmt.ofJson? (ss.toList.getD t .null)

def TwoHolds (h : Held) : Prop := ∀ s, (two h s).isSome = true

/-- Round 2 does not make it: `Run.run`'s registration step (`Docket.check`) does. -/
def RecordsHeld (h : Held) (x m₁ : Msg) : Prop := recordsOk h x m₁ = true

/-- Round 2 passes a value the verifier pinned, fills or opens, where the model's `readsOk` refuses every registered
value, so the model's clause follows from round 2's only where there is no such value. -/
def RegistersNone (rec : Json) : Prop :=
  ∀ es, Spot.getD rec "registered" = .arr es → ∀ e ∈ es, ∀ v, Spot.getD e "value" ≠ .str v

/-- At a design whose interior registration registers a value, such as the replay block's `logits`, it is false and
the statements that take it say nothing. -/
def InputsNone (own : Spot.SecondStage.Own) : Prop := Spot.SecondStage.inputsFailure (core own) = none

/-- No check of round 2 makes it. -/
def RoundsHeld (h : Held) (m₂ : Msg) : Prop :=
  ∀ t < h.ms.length, (FlockDeploy.V true).k (stmtAt h m₂ t) = k₀ h t

/-- No check of round 2 makes it; `second-tables` holds only that a verdict of record set the session up as one
table. -/
def ScopeHeld (h : Held) (m₂ : Msg) : Prop :=
  ∀ t < h.ms.length, sessionOk h t (stmtAt h m₂ t) = true

/-- No check of round 2 makes it: round 2 holds the public file's header and the registered reads to the reference's,
and reads nothing else of the statement. -/
def PinHeld (h : Held) (m₂ : Msg) : Prop :=
  ∀ t < h.ms.length, ∀ w, FlockDeploy.decode (stmtAt h m₂ t) = some w → pinOk h t w = true

def InteriorOrderOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin),
    OwnAgrees h own st rec draw c₀ → (Spot.SecondStage.check true own (some ss)).codes = #[] →
      Spot.SecondStage.interiorOrderOk (ownOf h st rec draw c₀) = true

def NamesOkOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin),
    OwnAgrees h own st rec draw c₀ → (Spot.SecondStage.check true own (some ss)).codes = #[] →
      Spot.SecondStage.namesOk (ownOf h st rec draw c₀) = true

def CodesOkOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin),
    OwnAgrees h own st rec draw c₀ → (Spot.SecondStage.check true own (some ss)).codes = #[] →
      ∀ t < ss.size, ∀ S, Spot.SecondStage.Stmt.ofJson? (ss.toList.getD t .null) = some S →
        codesOk h (ownOf h st rec draw c₀) t S = true

def CountOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin),
    OwnAgrees h own st rec draw c₀ → NamesDistinct h → (Spot.SecondStage.check true own (some ss)).codes = #[] →
      ss.size = h.ms.length

def ReadsOkOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (x m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h)
    (st : Roots) (rec draw : Json),
    decode x = some st → jsonMsg m₁ = some rec → secondDraw h c₀ c₁ = some draw → OwnAgrees h own st rec draw c₀ →
      StmtsAre h m₂ ss → InputsNone own → PinHeld h m₂ →
        (Spot.SecondStage.check true own (some ss)).codes = #[] →
        ∀ t < ss.size, readsOk h t x m₁ (stmtAt h m₂ t) c₀ c₁ = true

def StmtOkAtOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (x m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h)
    (st : Roots) (rec draw : Json),
    decode x = some st → jsonMsg m₁ = some rec → secondDraw h c₀ c₁ = some draw → OwnAgrees h own st rec draw c₀ →
      StmtsAre h m₂ ss → InputsNone own → TwoHolds h → RoundsHeld h m₂ → ScopeHeld h m₂ → PinHeld h m₂ →
        (Spot.SecondStage.check true own (some ss)).codes = #[] →
          ∀ t < ss.size, stmtOkAt h x m₁ (stmtAt h m₂ t) c₀ c₁ t = true

def StmtOkOfCheck : Prop :=
  ∀ (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (x m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h)
    (st : Roots) (rec draw : Json),
    decode x = some st → jsonMsg m₁ = some rec → secondDraw h c₀ c₁ = some draw → OwnAgrees h own st rec draw c₀ →
      StmtsAre h m₂ ss → InputsNone own → TwoHolds h → NamesDistinct h → RecordsHeld h x m₁ →
        RoundsHeld h m₂ → ScopeHeld h m₂ → PinHeld h m₂ → (Spot.SecondStage.check true own (some ss)).codes = #[] →
          stmtOk h x m₁ m₂ c₀ c₁ = true

end SpotFlock.RunsAs.Round2
