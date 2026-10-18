import Models.Accounting.Assumptions.TS
import Definitions.Accounting.TS.Receipt

/-!
# (I7) from the receipt (trusted)

The luck premise (I7) `LuckAt` split at the receipt (LEDGER §1.3–§1.5, note:20261017T1723Z-draft-ledger), each
`def G : Prop` proved as `Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS.Receipt`:

* `LuckOfReceipt`: with the openings pinned, the operand luck at `p` and the receipt luck at `pr ≤ 1` give (I7) at
  `p + (1 − p)·pr`.
* `ReceiptLuckCaseA`: Lemma B in case (a), the receipt luck is 0 when every round's committed value is the full state
  of duo or fx1 units, one deficient word a unit (`Pouw.Guarantees.Receipt.DuoRunWordInjective`,
  `Fx1RunWordInjective`).
* `LuckCaseA`: the two together, (I7) at the operand luck `p` in case (a).
* `OperandLuckCube`: the operand luck at `luck_quad`'s ¾ for rounds that draw uniform cube bits against a degree-2
  operand error.
* `LuckCaseACube`: case (a) with such rounds, (I7) at ¾ outright.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS

/-- **(I7) from the operand and receipt lucks.** For every ideal game split at the receipt whose openings are pinned,
the operand luck at `p` and the receipt luck at `pr ≤ 1` give (L) at `p + (1 − p)·pr`: a deficient round passes only
if its committed value is the honest one, which happens with every deficient word right (probability at most `p`) or
with one wrong, kept by the receipt with probability at most `pr`. -/
def LuckOfReceipt : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S : ℕ} (G : ReceiptGame Ω S) (p pr : ℝ),
    G.Pinned → G.OperandLuck p → G.ReceiptLuck pr → pr ≤ 1 → LuckAt (p + (1 - p) * pr) G.game

/-- **Lemma B in case (a): the receipt luck is 0.** For every ideal game split at the receipt whose rounds commit the
full state of duo or fx1 units from pinned starts, one deficient word a unit, a round whose deficient words are not
all right never keeps the honest committed value: each unit's final state determines its deficient word. -/
def ReceiptLuckCaseA : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S : ℕ} (G : ReceiptGame Ω S), G.CaseA → G.ReceiptLuck 0

/-- **(I7) in case (a) at the operand luck.** For every ideal game split at the receipt whose openings are pinned and
whose rounds commit the full state of duo or fx1 units, one deficient word a unit, the operand luck at `p` gives (L) at
`p`. -/
def LuckCaseA : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S : ℕ} (G : ReceiptGame Ω S) (p : ℝ),
    G.Pinned → G.CaseA → G.OperandLuck p → LuckAt p G.game

/-- **The operand luck at ¾ for degree-2 operands** (`luck_quad`). For every ideal game split at the receipt whose
rounds draw uniform cube bits after fixing their share and a degree-2 operand error, nonzero when the round is
deficient, the operand luck holds at ¾. -/
def OperandLuckCube : Prop :=
  ∀ (R : Type) [CommRing R] {α V : Type} [Fintype α] [Fintype V] [DecidableEq V] {S : ℕ}
    (G : ReceiptGame (α × Finset V) S), G.CubeOperands R → G.OperandLuck (3 / 4)

/-- **(I7) at ¾ in case (a) for degree-2 operands.** For every ideal game split at the receipt whose openings are
pinned, whose rounds commit the full state of duo or fx1 units, one deficient word a unit, and draw uniform cube bits
after fixing their share and a degree-2 operand error, (L) holds at ¾, with no luck hypothesis. -/
def LuckCaseACube : Prop :=
  ∀ (R : Type) [CommRing R] {α V : Type} [Fintype α] [Fintype V] [DecidableEq V] {S : ℕ}
    (G : ReceiptGame (α × Finset V) S), G.Pinned → G.CaseA → G.CubeOperands R → LuckAt (3 / 4) G.game

end Accounting.Guarantees.TS
