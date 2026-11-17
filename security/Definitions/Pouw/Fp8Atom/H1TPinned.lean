module

public import Definitions.Pouw.Fp8Atom.H1TVectors
public import Mathlib.Data.Fintype.Prod
public import Mathlib.Data.Fintype.Sum

/-! H-1T's guarantees: the facts about the map `DistinctLiveH1T` reads that don't need the Target A proof, and the FP32 lemmas that proof uses. -/

@[expose] public section

namespace Pouw.Fp8Atom.H1T

open Pouw.Fp8Atom

def H1TFormTable : Prop :=
  formTable.map (fun v => (v.x, v.pos, v.fi)) = formKeys ∧ ∀ v ∈ formTable,
    (formReal (fOf v.fi) (codeOf v.x) v.pos).1.val = v.x1 ∧ (formReal (fOf v.fi) (codeOf v.x) v.pos).2.val = v.x2

def H1TTagTable : Prop :=
  tagTable.map (fun v => (v.fi, v.neg, v.m)) = tagKeys ∧ ∀ v ∈ tagTable,
    (enc (tagVal (fOf v.fi) v.neg (fin8 v.m))).val = v.c1 ∧ (enc (-tagVal (fOf v.fi) v.neg (fin8 v.m))).val = v.c2 ∧
    val (codeOf v.c1) = tagVal (fOf v.fi) v.neg (fin8 v.m) ∧ val (codeOf v.c2) = -tagVal (fOf v.fi) v.neg (fin8 v.m)

def H1TSaltConformance : Prop :=
  74 ≤ saltVectors.length ∧ ∀ v ∈ saltVectors, packSalt (extract (rawOf v.r)) = (v.pos, v.neg, v.man)

def H1TChainConformance : Prop :=
  31 ≤ rowVectors.length ∧ ∀ v ∈ rowVectors, ∀ p < 2 * v.T,
    freshWord v.T v.xs v.bs v.j v.σ p = v.I.getD p 0 ∧ wordQ (v.R.getD p 0) = some (runVal v.T v.xs v.bs v.j v.σ p)

def H1TReadsFields : Prop :=
  ∀ (U : Unit) (g g' : U.Salt), (∀ i τ, extract (g i τ) = extract (g' i τ)) → ∀ c, U.word c g = U.word c g'

def H1TNotRawWord : Prop :=
  ∀ (U : Unit) (c : Fin U.m × Fin U.n × Pos U.T) (i : Fin U.m) (τ : Fin U.T) (ℓ : Fin 18),
    U.word c ≠ fun g => ((g i τ ℓ : ℕ) : ℚ)

def H1TTupleInjective : Prop := ∀ j j' : ℕ, j < 88412 → j' < 88412 → tuple j = tuple j' → j = j'

def H1TTupleWrap : Prop := ∀ j : ℕ, tuple (j + 88412) = tuple j

def H1TWrapDuplicate : Prop :=
  ∀ (U : Unit) (j j' : Fin U.n), j'.val = j.val + 88412 → (∀ r, U.b r j = U.b r j') →
    ∀ (i : Fin U.m) (a : Pos U.T), U.word (i, j, a) = U.word (i, j', a)

def H1TPosCard : Prop := ∀ T : ℕ, 2 ≤ T → Fintype.card (Pos T) = 4 * T - 3

/-- The witness is at `k = 32,768` and `n = 88,412`, with `T = 1,130`. -/
def H1TNonVacuous : Prop :=
  ∃ U : Unit, U.Legal ∧ U.m = 1 ∧ U.k = 32768 ∧ U.n = 88412 ∧ U.T = 1130 ∧ Fintype.card (Pos U.T) = 4517

def RnErr : Prop := ∀ y : ℚ, |rn y - y| ≤ ulp y / 2

def RnAddMovesHalf : Prop := ∀ r i : ℚ, Representable r → r ≠ 0 → ulp r < 2 * |i| → rnAdd r i ≠ r

def HalfOfInvolution : Prop :=
  ∀ (α : Type) [Fintype α] (ψ : α → α) (P : α → Bool), Function.Involutive ψ → (∀ a, P a = true → P (ψ a) = false) →
    2 * (Finset.univ.filter fun a => P a = true).card ≤ Fintype.card α

end Pouw.Fp8Atom.H1T
