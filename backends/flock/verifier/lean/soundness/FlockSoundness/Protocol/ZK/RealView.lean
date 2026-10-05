import FlockSoundness.Protocol.ZK.Table

/-!
# Zero knowledge: the real prover's view and `S_shvzk`'s, with real hm96 leaves

`Table.view` and `Table.sim` have every leaf the final message doesn't open ideal (world `W₁`). Here those hidden leaves
are real: hm96 leaves of what they commit, each under a fresh salt. For the prover, that is level 0's columns at the other
positions and each rep's pads columns at its other positions; for `S_shvzk`, its dummy run's level-0 columns and zero pads
columns (§3.2 step 6).

* `Table.Hid`: the hidden leaves. `Table.viewWith`, `Table.simWith`: the view with its hidden leaves given.
* `Table.Hm96`: hm96's structure (a leaf of content `col` under salt `y` is `H(dig(col) + M·y, c(y))`, an ideal one
  `H(U, c(y))`) and the two codes at the positions nobody opens.
* `Table.Hm96.viewR`, `Table.Hm96.simR`: the two views with real leaves. `RealLeaves.table_shvzk_hm96` is about them.
-/

namespace FlockSoundness.ZK

open Finset

namespace Table

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B]
  (T : Table F K W Salt (B × Salt) D Hi VL)

/-- The leaves nobody opens: level 0's other positions, and each rep's other pads positions. Its size is at most the
paper's `N_hid`, which counts every level-0 and pads-tree leaf. -/
abbrev Hid := Fin T.nU ⊕ (Fin 2 × Fin T.nPU)

/-- The view with its hidden leaves `L`. -/
noncomputable def viewWith [CharP F 2] (w : W) (ω : T.Rand) (L : T.Hid → D) : T.View :=
  let v := T.view w ω
  { v with
    leaves0 := (v.leaves0.1, fun q => L (.inl q))
    leavesP := fun r => ((v.leavesP r).1, fun k => L (.inr (r, k))) }

/-- `S_shvzk`'s output with its hidden leaves `L`. -/
noncomputable def simWith [CharP F 2] (w0 : W) (σ : T.SimRand) (L : T.Hid → D) : T.View :=
  let v := T.sim w0 σ
  { v with
    leaves0 := (v.leaves0.1, fun q => L (.inl q))
    leavesP := fun r => ((v.leavesP r).1, fun k => L (.inr (r, k))) }

variable {T} {C : Type}

end Table

end FlockSoundness.ZK
