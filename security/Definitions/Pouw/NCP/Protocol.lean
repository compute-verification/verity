module

public import Definitions.Pouw.D1
public import Definitions.Pouw.NCP.Defs

/-! NCP as a protocol, with the lead's parameters (`IsLead`) and domain (`leadDomain`). -/

@[expose] public section

namespace Pouw.NCP

open Finset

def InSmall (x : ℤ) : Prop := -31 ≤ x ∧ x ≤ 31

/-- 6 bits of `H` output per entry; not `Pouw.InNoise`, the problem statement's `[−63, 63]`. -/
def InNoise6 (x : ℤ) : Prop := -31 ≤ x ∧ x ≤ 32

/-- The first clause excludes N1's structures; the second, step-level involutions such as a step reversal; the third,
`R² = ±I`, which makes the zero-input checked value at depth `2k` zero or twice the one at depth `k`. -/
def Admissible (k d : ℕ) (R : Mat) : Prop :=
  SignedPerm k R ∧
  (∀ t < k / d,
    (∀ j, t * d ≤ j → j < (t + 1) * d → ∀ i, t * d ≤ i → i < (t + 1) * d → R i j = 0) ∧
    ∀ t' < k / d, ¬ ∀ j, t * d ≤ j → j < (t + 1) * d → ∃ i, t' * d ≤ i ∧ i < (t' + 1) * d ∧ R i j ≠ 0) ∧
  (¬ ∀ i < k, ∀ j < k, mm k R R i j = if i = j then 1 else 0) ∧
  ¬ ∀ i < k, ∀ j < k, mm k R R i j = if i = j then -1 else 0

namespace GameProofs

def shiftBy (c k : ℕ) : Mat := fun i j => if j < k ∧ i = (if j + c < k then j + c else j + c - k) then 1 else 0

end GameProofs

/-- NCP's parameters. -/
structure Params (Q R S : Type) where
  d : ℕ
  qAct : S → ℕ → Mat → ℕ → List Q
  qWt : S → ℕ → List Q
  expAct : Shape → List R → Mat
  expWt : Shape → List R → Mat × Mat
  Wref : Layout → ℕ → ℕ

namespace Params

variable {Q R S : Type} (P : Params Q R S)

/-- Unit `u`'s `E₁`. -/
def e1 (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  P.expAct (U.layout.shape u) ((P.qAct s u A (U.layout.fixedId u)).map H)

/-- Unit `u`'s weight noise `(F₁, R)`, shared by every unit on the same (epoch, weight). -/
def wtNoise (H : Q → R) (s : S) (U : Workload) (u : ℕ) : Mat × Mat :=
  P.expWt (U.layout.shape u) ((P.qWt s (U.layout.fixedId u)).map H)

def opX (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  stackX (U.layout.shape u).k A (P.e1 H s U u A) (P.wtNoise H s U u).2

def opY (H : Q → R) (s : S) (U : Workload) (u : ℕ) : Mat :=
  stackY (U.layout.shape u).k (U.fixed (U.layout.fixedId u)) (P.wtNoise H s U u).2 (P.wtNoise H s U u).1

/-- Mod `2^32`, the int32 accumulator's two's-complement value, so no int32 range bound is needed. -/
def Y (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Checked :=
  fun t => mmW (D1.blockEnd P.d (3 * (U.layout.shape u).k) t) (P.opX H s U u A) (P.opY H s U u)

/-- The compared entries: blocks `t < ⌈3k/d⌉`, rows `i < m`, columns `j < n`. -/
def checkIdx (sh : Shape) : Finset (ℕ × ℕ × ℕ) :=
  range (D1.nSegments P.d (3 * sh.k)) ×ˢ range sh.m ×ˢ range sh.n

def refChecked (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Checked :=
  fun t => mmW (D1.blockEnd P.d (3 * (U.layout.shape u).k) t) (P.opX H s U u A) (P.opY H s U u)

def refOut (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  mmW (3 * (U.layout.shape u).k) (P.opX H s U u A) (P.opY H s U u)

def toProtocol : Protocol Q R S where
  Y := P.Y
  checkIdx := P.checkIdx
  refChecked := P.refChecked
  refOut := P.refOut
  Wref := P.Wref

def NoiseRange : Prop :=
  (∀ (sh : Shape) (a : List R), ∀ i < sh.m, ∀ l < sh.k, InNoise6 (P.expAct sh a i l)) ∧
  (∀ (sh : Shape) (b : List R), ∀ l < sh.k, ∀ j < sh.n, InNoise6 ((P.expWt sh b).1 l j))

def RSigned : Prop := ∀ (sh : Shape) (b : List R), Shape.InDefaultDomain sh → SignedPerm sh.k (P.expWt sh b).2

def RAdmissible : Prop :=
  ∀ (sh : Shape) (b : List R), Shape.InDefaultDomain sh → Admissible sh.k P.d (P.expWt sh b).2

def QueryBound (c : ℕ) : Prop := ∀ (s : S) (u : ℕ) (A : Mat) (w : ℕ), (P.qAct s u A w).length + (P.qWt s w).length ≤ c

/-- `W_ref` is not fixed here. -/
def IsLead : Prop := P.d = 16 ∧ P.NoiseRange ∧ P.RAdmissible

def nz (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) (i : ℕ) : Mat :=
  if i = 0 then stackX (U.layout.shape u).k 0 (P.e1 H s U u A) (P.wtNoise H s U u).2
  else if i = 1 then stackY (U.layout.shape u).k 0 (P.wtNoise H s U u).2 (P.wtNoise H s U u).1 else 0

end Params

def leadDomain (L : Layout) : Prop := L.ShapesIn Shape.InDefaultDomain ∧ ∀ w, L.rowsOf w ≤ 2 ^ 40

end Pouw.NCP
