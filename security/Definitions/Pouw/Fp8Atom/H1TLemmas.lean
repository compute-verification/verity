module

public import Definitions.Pouw.Fp8Atom.H1TVec
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Data.Fintype.Pi

/-! H-1T's M3 statements: the lemmas the one-flip clauses of `DistinctLiveH1T` rest on, and the clauses. -/

@[expose] public section

namespace Pouw.Fp8Atom.H1T

open Pouw.Fp8Atom

def truncTo (g y : ℚ) : ℚ := (if y < 0 then -1 else 1) * (⌊|y| / g⌋ : ℚ) * g

def trunc14 (y : ℚ) : ℚ := if y = 0 then 0 else truncTo (2 ^ (binade |y| - 13)) y

def gridExp (a b : Fin 32 → Code) : ℤ := maxExp (-139) (List.ofFn fun l => product (a l) (b l))

def preSum (a b : Fin 32 → Code) : ℚ := ∑ l, truncTo (2 ^ (gridExp a b - 13)) (val (a l) * val (b l))

def H1TAtomValue : Prop :=
  ∀ a b : Fin 32 → Code, (∀ l, (a l).isNaN = false) → (∀ l, (b l).isNaN = false) →
    wordQ (hopperStep 0 a b) = some (trunc14 (preSum a b))

def Trunc14Mono : Prop := Monotone trunc14

def Trunc14Cell : Prop :=
  ∀ y y' : ℚ, y ≠ 0 → |trunc14 y - y| < 2 ^ (binade |y| - 13) ∧
    (trunc14 y = trunc14 y' → |y - y'| < 2 ^ (binade |y| - 13))

def negFlip (i : Fin 3) (σ : SliceSalt) : SliceSalt := { σ with neg := Function.update σ.neg i (!σ.neg i) }

def H1TSlice : Prop :=
  ∀ (xs : Fin 29 → Code) (σ : SliceSalt) (b : ℕ → Code) (j τ : ℕ), (∀ l, (xs l).isNaN = false) →
    (∀ r, (b r).isNaN = false) →
    (|preSum (formSlice xs σ).1 (colLanes b j τ)| ≤ 6142976 ∧
      |preSum (formSlice xs σ).1 (colLanes b j τ)| < 2 ^ 21 * floorF (sliceMax xs)) ∧
    (|preSum (formSlice xs σ).2 (colLanes b j τ)| ≤ 738304 ∧
      |preSum (formSlice xs σ).2 (colLanes b j τ)| < 2 ^ 21 * floorF (sliceMax xs)) ∧
    ∀ i : Fin 3,
      gridExp (formSlice xs (negFlip i σ)).1 (colLanes b j τ) = gridExp (formSlice xs σ).1 (colLanes b j τ) ∧
      gridExp (formSlice xs (negFlip i σ)).2 (colLanes b j τ) = gridExp (formSlice xs σ).2 (colLanes b j τ) ∧
      preSum (formSlice xs (negFlip i σ)).1 (colLanes b j τ) = preSum (formSlice xs σ).1 (colLanes b j τ) -
        2 * tagVal (floorF (sliceMax xs)) (σ.neg i) (σ.man i) * val (tuple j i) ∧
      preSum (formSlice xs (negFlip i σ)).2 (colLanes b j τ) = preSum (formSlice xs σ).2 (colLanes b j τ) +
        2 * tagVal (floorF (sliceMax xs)) (σ.neg i) (σ.man i) * val (tuple j i)

def H1TChainMono : Prop :=
  ∀ (T : ℕ) (x b : ℕ → Code) (j : ℕ) (σ σ' : ℕ → SliceSalt) (p : ℕ),
    (∀ t ≤ p, freshVal T x b j σ t ≤ freshVal T x b j σ' t) → runVal T x b j σ p ≤ runVal T x b j σ' p

def H1TRange : Prop :=
  ∀ U : Unit, U.Legal → ∀ (i : Fin U.m) (j : Fin U.n) (g : U.Salt) (p : ℕ), p < 2 * U.T →
    (wordQ (freshWord U.T (U.row i) (U.col j) j.val (U.rowSalt g i) p)).isSome = true ∧
    |freshVal U.T (U.row i) (U.col j) j.val (U.rowSalt g i) p| ≤ (if p < U.T then 6142976 else 738304) ∧
    |runVal U.T (U.row i) (U.col j) j.val (U.rowSalt g i) p| < 2 ^ 33


def H1TFormCover : Prop :=
  ∀ xs : Fin 29 → Code, (∀ l, (xs l).isNaN = false) →
    ∃ fi : ℕ, fi ≤ 4 ∧ floorF (sliceMax xs) = fOf fi ∧ ∀ l, |val (xs l)| < 2 ^ (fi + 5)

def posIdx {T : ℕ} : Pos T → ℕ := Sum.elim (fun p => p.val) (fun p => p.val.val)

def Run2 {T : ℕ} (a : Pos T) : Prop := a.isRight = true ∧ T ≤ posIdx a

def FarTags (j j' : ℕ) : Prop := ∃ k : Fin 3, 72 ≤ |val (tuple j k) - val (tuple j' k)|

def OppLater {T : ℕ} (j j' : ℕ) (a a' : Pos T) : Prop :=
  posIdx a' < T ∧ posIdx a < T + posIdx a' ∧
    ∃ k : Fin 3, val (tuple j k) * val (tuple j' k) < 0 ∧ (a'.isLeft = true ∨ 36 ≤ |val (tuple j' k)|)

def Early {T : ℕ} (j : ℕ) (a : Pos T) : Prop :=
  ∃ k : Fin 3, 32 * ((posIdx a + 2 - T : ℕ) : ℚ) + 4 ≤ |val (tuple j k)|

def SkipsLast {T : ℕ} (a : Pos T) : Prop :=
  posIdx a + 1 < T ∨ (a.isLeft = true ∧ T ≤ posIdx a ∧ posIdx a + 1 < 2 * T)

/-- The words `H1TOneFlip` does not show to move. -/
def Stuck {T : ℕ} (j : ℕ) (a : Pos T) : Prop := Run2 a ∧ ¬ Early j a

/-- The pairs one flip does not separate, which `H1TOneFlip` leaves out. -/
def Hard {T : ℕ} (j j' : ℕ) (a a' : Pos T) : Prop :=
  (a.isRight = true ∧ a'.isRight = true ∧ posIdx a < T ∧ posIdx a = posIdx a' ∧ ¬ FarTags j j' ∧
      ¬ OppLater j j' a a' ∧ ¬ OppLater j' j a' a) ∨
    (Run2 a ∧ ¬(a'.isLeft = true ∧ posIdx a < posIdx a') ∧ ¬ OppLater j j' a a' ∧ ¬(Early j a ∧ SkipsLast a')) ∨
    (Run2 a' ∧ ¬(a.isLeft = true ∧ posIdx a' < posIdx a) ∧ ¬ OppLater j' j a' a ∧ ¬(Early j' a' ∧ SkipsLast a))

def H1TOneFlip : Prop :=
  ∀ U : Unit, U.Legal → ∀ i : Fin U.m,
    (∀ a a' : Fin U.n × Pos U.T, a ≠ a' → ¬ Hard a.1.val a'.1.val a.2 a'.2 →
      2 * (Finset.univ.filter fun s : Fin U.T → RawSlice =>
        U.word (i, a) (fun _ => s) = U.word (i, a') (fun _ => s)).card ≤ Fintype.card (Fin U.T → RawSlice)) ∧
    ∀ (a : Fin U.n × Pos U.T), ¬ Stuck a.1.val a.2 → ∀ q : ℚ,
      2 * (Finset.univ.filter fun s : Fin U.T → RawSlice => U.word (i, a) (fun _ => s) = q).card ≤
        Fintype.card (Fin U.T → RawSlice)

end Pouw.Fp8Atom.H1T
