import Pous.Model.Labelling
import Pous.Model.M1

/-!
# The dense single-layer scheme (S1)

The definitions of the dense draft (`lean/submissions/dense/DenseMeetsDraft.lean`), verbatim: the complete DAG, the
label-shaped random oracle, and the scheme `C_i = H(i, C_{<i}) ⊕ W_i` at rate 1, with no `pp` and no coins.
-/

namespace Pous.Dense

open Pous.Labelling

/-- The complete DAG on `Fin B`: every earlier node is a parent. -/
def completeDAG (B : ℕ) : TopoDAG B where
  par i := Finset.univ.filter fun j => j < i
  par_lt _ _ hj := (Finset.mem_filter.1 hj).2

/-- The dense scheme's model: B5's random oracle on label-shaped queries. -/
abbrev denseModel (B ℓ : ℕ) : IdealModel := IdealModel.randomOracle (LabelQry B ℓ) (Bits ℓ)

/-- The `B` blocks of `W`. -/
def blocks (B ℓ : ℕ) (W : Bits (B * ℓ)) : Fin B → Bits ℓ := (codeEquiv B ℓ).symm W

/-- **S1, the dense single-layer data-carrying labeling** (`docs/p3-scheme.md` §4), on one segment of `B` blocks
of `ℓ` bits. `C_i = ℓ_i = H(i, ℓ_{<i}) ⊕ W_i` over the complete DAG, with no `pp` and no coins, at rate 1.
Decoding recomputes `W_i = C_i ⊕ H(i, C_{<i})` with one query per block. -/
def denseScheme (B ℓ : ℕ) [NeZero B] : Scheme (denseModel B ℓ) where
  n := B * ℓ
  t := 0
  B := B
  ℓ := ℓ
  B_pos := Nat.pos_of_ne_zero (NeZero.ne B)
  Coins := Unit
  enc W H _ := (fun i => i.elim0, fun i => label (completeDAG B) (blocks B ℓ W) H i)
  dec _ C f := codeEquiv B ℓ fun i =>
    bxor (C i) (f (i, fun j => if j ∈ (completeDAG B).par i then some (C j) else none))

end Pous.Dense
