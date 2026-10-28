import Definitions.Pous.Model.Labelling
import Definitions.Pous.Model.M1

/-!
# The dense single-layer scheme (S1)

The definitions of the dense scheme (`Security/Proofs/Pous/Dense/DenseMeets.lean` proves `DenseMeets`): the complete
DAG, the label-shaped random oracle, and the scheme `C_i = H(i, C_{<i}) ⊕ W_i` at rate 1, with no `pp` and no coins.
Then the segmented scheme of `Security/Proofs/Pous/Dense/DenseReCert.lean`: `N` such segments of `B` blocks on global
block indices `Fin (N*B)`, a node's parents being the earlier blocks of its own segment. Last, the same store keyed as
the implementation keys it (`segTagScheme`): segment `s` is the dense scheme on `B` blocks with oracle `H(tag_s, ·)`,
`tag_s = salt ‖ u64 s`, and local queries.
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

/-- The disjoint union of `N` complete DAGs on `B` blocks, on global indices `Fin (N*B)`: node `k`'s parents are the
earlier blocks of its own segment (`j/B = k/B` and `j < k`). -/
def segDAG (N B : ℕ) : Labelling.TopoDAG (N * B) where
  par k := Finset.univ.filter fun j => j.val / B = k.val / B ∧ j.val < k.val
  par_lt _ _ hj := (Finset.mem_filter.1 hj).2.2

/-- The dense scheme on the disjoint union of `N` complete DAGs of `B` blocks: `segDAG N B` in place of `completeDAG`. -/
def segScheme (N B ℓ : ℕ) [NeZero (N * B)] : Scheme (denseModel (N * B) ℓ) where
  n := (N * B) * ℓ
  t := 0
  B := N * B
  ℓ := ℓ
  B_pos := Nat.pos_of_ne_zero (NeZero.ne (N * B))
  Coins := Unit
  enc W H _ := (fun i => i.elim0, fun i => label (segDAG N B) (blocks (N * B) ℓ W) H i)
  dec _ C f := codeEquiv (N * B) ℓ fun i =>
    bxor (C i) (f (i, fun j => if j ∈ (segDAG N B).par i then some (C j) else none))

/-- The global block index is exactly the `(segment, block)` pair. -/
def segIndexEquiv (N B : ℕ) : Fin N × Fin B ≃ Fin (N * B) := finProdFinEquiv

/-- The low 64 bits of `s` (`u64 s`), as in `tag_s = salt ‖ u64 s`. -/
def u64 (s : ℕ) : Bits 64 := fun j => s.testBit j.val

/-- The implementation's segment tag `tag_s = salt ‖ u64 s`: a 192-bit salt then `u64 s`, filling `chainDense`'s 256-bit
salt slot exactly. -/
def tag (salt : Bits 192) (s : ℕ) : Bits (192 + 64) := Fin.append salt (u64 s)

/-- The implementation's oracle: one random oracle keyed by a 256-bit segment tag and a local label query. -/
abbrev tagModel (B ℓ : ℕ) : IdealModel := IdealModel.randomOracle (Bits (192 + 64) × LabelQry B ℓ) (Bits ℓ)

/-- Block `i` of segment `s` of `W`, at global position `segIndexEquiv N B (s, i)`. -/
def segBlocks (N B ℓ : ℕ) (W : Bits ((N * B) * ℓ)) (s : Fin N) : Fin B → Bits ℓ :=
  fun i => blocks (N * B) ℓ W (segIndexEquiv N B (s, i))

/-- **The segmented dense scheme as the implementation keys it.** Segment `s` is the dense scheme on `B` blocks with
oracle `H(tag_s, ·)`, `tag_s = salt ‖ u64 s`, local block indices and local parents. Block `(s, i)` sits at global
position `segIndexEquiv N B (s, i)` of the codeword. -/
def segTagScheme (N B ℓ : ℕ) [NeZero (N * B)] (salt : Bits 192) : Scheme (tagModel B ℓ) where
  n := (N * B) * ℓ
  t := 0
  B := N * B
  ℓ := ℓ
  B_pos := Nat.pos_of_ne_zero (NeZero.ne (N * B))
  Coins := Unit
  enc W H _ := (fun i => i.elim0, fun g =>
    label (completeDAG B) (segBlocks N B ℓ W ((segIndexEquiv N B).symm g).1)
      (fun q => H (tag salt ((segIndexEquiv N B).symm g).1.val, q)) ((segIndexEquiv N B).symm g).2)
  dec _ C f := codeEquiv (N * B) ℓ fun g =>
    bxor (C g) (f (tag salt ((segIndexEquiv N B).symm g).1.val, (((segIndexEquiv N B).symm g).2,
      fun j => if j ∈ (completeDAG B).par ((segIndexEquiv N B).symm g).2
        then some (C (segIndexEquiv N B (((segIndexEquiv N B).symm g).1, j))) else none)))

end Pous.Dense
