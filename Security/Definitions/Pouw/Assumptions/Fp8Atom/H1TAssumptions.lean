import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.Sum
import Mathlib.Data.Fintype.Prod
import Definitions.Pouw.Fp8Atom.H1T

/-!
# H-1T's named assumptions

* **`DistinctLiveH1T`** is Target A for H-1T, one unit at a time. It is §5.3 of `docs/pouw/hardness-shaped-matrices.md`
  with 9 changes: the salt is the fields of the raw XOF words that the kernel reads
  (`extract`); the credited words are the `4T − 3` of `Pos`; B′ is the real codes plus the fixed tag table; `n ≤ 88,412`
  per unit, whose salt is its own; `X_q` and B′ are any finite codes; `T = ⌈k/29⌉` with zero padding. On a legal
  unit, no two credited words are equal as functions of the salt, and no credited word is free (a constant or a raw
  XOF word). This is the statement the Lean is to prove. Until then
  it is Assumed, and it is a hypothesis of the pins that use it.
* **`HalfApartH1T`** is the half-salt clause. It is stronger than Target A, and it is not the target: it is
  the input to the lifting and branching arguments. Two distinct credited words of one row are equal on at most half of
  that row's salts, and a credited word takes any one value on at most half of them. It is Derived for `I` words (one
  flip separates them) and Simulated for `R` words (pair frequencies at most 0.375). `census` does not measure
  it.
-/

namespace Pouw.Fp8Atom.H1T

/-- **`DistinctLiveH1T`** (Target A per unit): on every legal unit, the credited words are pairwise distinct functions
of the unit's salt, and none of them is free. -/
def DistinctLiveH1T : Prop :=
  ∀ U : Unit, U.Legal → Function.Injective U.word ∧ ∀ c, U.word c ∉ U.free

/-- **The half-salt clause** (not the target): on every legal unit, for each row `i`, two distinct credited words of
row `i` agree on at most half of row `i`'s salts, and a credited word of row `i` equals a given value on at most half of
them. Row `i`'s words read only row `i`'s salt, so every row is given the same salt `s`. -/
def HalfApartH1T : Prop :=
  ∀ U : Unit, U.Legal → ∀ i : Fin U.m,
    (∀ a a' : Fin U.n × Pos U.T, a ≠ a' →
      2 * (Finset.univ.filter fun s : Fin U.T → RawSlice =>
        U.word (i, a) (fun _ => s) = U.word (i, a') (fun _ => s)).card ≤ Fintype.card (Fin U.T → RawSlice)) ∧
    ∀ (a : Fin U.n × Pos U.T) (q : ℚ),
      2 * (Finset.univ.filter fun s : Fin U.T → RawSlice => U.word (i, a) (fun _ => s) = q).card ≤
        Fintype.card (Fin U.T → RawSlice)

end Pouw.Fp8Atom.H1T
