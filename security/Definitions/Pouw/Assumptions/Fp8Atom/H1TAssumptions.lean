module

public import Mathlib.Data.Fintype.Pi
public import Definitions.Pouw.Fp8Atom.H1T

/-! H-1T's named assumptions: `DistinctLiveH1T`, Target A one unit at a time, and `HalfApartH1T`, the half-salt clause. -/

@[expose] public section

namespace Pouw.Fp8Atom.H1T

/-- **`DistinctLiveH1T`** (Target A per unit): on every legal unit, the credited words are pairwise distinct functions
of the unit's salt, and none of them is free. -/
def DistinctLiveH1T : Prop :=
  ∀ U : Unit, U.Legal → Function.Injective U.word ∧ ∀ c, U.word c ∉ U.free

/-- **The half-salt clause**, stronger than Target A and not the target. Row `i`'s words read only row `i`'s salt, so
every row is given the same salt `s`. -/
def HalfApartH1T : Prop :=
  ∀ U : Unit, U.Legal → ∀ i : Fin U.m,
    (∀ a a' : Fin U.n × Pos U.T, a ≠ a' →
      2 * (Finset.univ.filter fun s : Fin U.T → RawSlice =>
        U.word (i, a) (fun _ => s) = U.word (i, a') (fun _ => s)).card ≤ Fintype.card (Fin U.T → RawSlice)) ∧
    ∀ (a : Fin U.n × Pos U.T) (q : ℚ),
      2 * (Finset.univ.filter fun s : Fin U.T → RawSlice => U.word (i, a) (fun _ => s) = q).card ≤
        Fintype.card (Fin U.T → RawSlice)

end Pouw.Fp8Atom.H1T
