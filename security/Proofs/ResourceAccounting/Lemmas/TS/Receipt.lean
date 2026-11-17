module

public import Definitions.Pouw.Assumptions.TS
public import Definitions.Pouw.TS.Receipt

/-! Steps of proofs of the luck premise (I7) `LuckAt` split at the receipt, pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS ResourceAccounting.Assumptions.TS

def LuckOfReceipt : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S : ℕ} (G : ReceiptGame Ω S) (p pr : ℝ),
    G.Pinned → G.OperandLuck p → G.ReceiptLuck pr → pr ≤ 1 → LuckAt (p + (1 - p) * pr) G.game

def ReceiptLuckCaseA : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S : ℕ} (G : ReceiptGame Ω S), G.CaseA → G.ReceiptLuck 0

def LuckCaseA : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S : ℕ} (G : ReceiptGame Ω S) (p : ℝ),
    G.Pinned → G.CaseA → G.OperandLuck p → LuckAt p G.game

def OperandLuckCube : Prop :=
  ∀ (R : Type) [CommRing R] {α V : Type} [Fintype α] [Fintype V] [DecidableEq V] {S : ℕ}
    (G : ReceiptGame (α × Finset V) S), G.CubeOperands R → G.OperandLuck (3 / 4)

def LuckCaseACube : Prop :=
  ∀ (R : Type) [CommRing R] {α V : Type} [Fintype α] [Fintype V] [DecidableEq V] {S : ℕ}
    (G : ReceiptGame (α × Finset V) S), G.Pinned → G.CaseA → G.CubeOperands R → LuckAt (3 / 4) G.game

def LiSegmentsBind : Prop :=
  ∀ {Ω U : Type} {n S W C H : ℕ} (D : LiUnits Ω U n S W C H) (ω : Ω), D.same ω → D.right ω

def LiExecutionPhaseUnbound : Prop :=
  ∀ {Ω U : Type} (n : ℕ) (D : LiUnits Ω U (n + 1) 2 1 1 1) (ω : Ω),
    (∀ u, ∀ s < 2, D.pos ω u s = 0 ∧ D.value ω u s = D.honest ω u s 0 ^^^ BitVec.twoPow (n + 1) n) →
      D.sameExecutionPhase ω ∧ ∀ u, (∀ s < 2, D.value ω u s ≠ D.honest ω u s (D.pos ω u s)) ∧
        D.checkpoint ω u (D.word ω u) 0 ≠ D.checkpoint ω u (D.honest ω u) 0

def DuoSegmentsBind : Prop :=
  ∀ {Ω U : Type} {n S W C H : ℕ} (D : DuoCPUnits Ω U n S W C H) (ω : Ω), D.same ω → D.right ω

def Fx1SegmentsBind : Prop :=
  ∀ {Ω U : Type} {n S W C H : ℕ} (D : Fx1CPUnits Ω U n S W C H) (ω : Ω), D.same ω → D.right ω

def LuckCaseAKeyed : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S : ℕ} (G : ReceiptGame Ω S), G.Pinned → G.CaseA → G.IncorrectWords → LuckAt 0 G.game

end ResourceAccounting.Guarantees.TS
