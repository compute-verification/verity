import Definitions.Flock.ZK.Table

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

/-- **hm96 leaves** (`verity/commitments/hm96/PROTOCOL.md`): a leaf of content `col` under salt `y` is
`H(dig(col) + M·y, c(y))`, with `dig(col)` the content's digest, `M·y` the pinned key's hash of the salt and
`c(y) = H(sp ‖ y)`; an ideal leaf of seed `(U, y)` is `H(U, c(y))`. Also the two codes at the positions nobody opens,
whose columns the hidden leaves commit. -/
structure Hm96 (C : Type) where
  My : Salt → B
  cs : Salt → C
  H : B × C → D
  dig0 : (Fin T.nL → F) × (Fin 2 → Fin 2 → F) → B
  digP : F × F → B
  leaf0_eq : ∀ col y, T.leaf0 col y = H (dig0 col + My y, cs y)
  leafP_eq : ∀ col y, T.leafP col y = H (digP col + My y, cs y)
  ideal_eq : ∀ u y, T.ideal (u, y) = H (u, cs y)
  /-- The level-0 code at the other positions, and each rep's pads code at its other positions. -/
  EU : Fin T.nU → Fin T.L → F
  GpU : Fin T.nU → Fin T.tpad → F
  ChU : Fin 2 → Fin T.nPU → PadIdx T.nE → F
  CpU : Fin 2 → Fin T.nPU → Fin T.nPad → F

variable {T} {C : Type}

/-- The digests the prover's hidden leaves commit: level 0's columns at the other positions (all 68 lanes), and each rep's
pads columns `(C_h, C_μ)` at its other positions. -/
def Hm96.dig (M : T.Hm96 C) (w : W) (ω : T.Rand) : T.Hid → B
  | .inl q => M.dig0 (fun l => ∑ c, T.lanes w ω.1 l c * M.EU q c + ∑ j, ω.2.1.1 l j * M.GpU q j,
      fun r i => ∑ c, ω.2.1.2.1 r i c * M.EU q c + ∑ j, ω.2.1.2.2 r i j * M.GpU q j)
  | .inr ⟨r, k⟩ => M.digP (∑ i, M.ChU r k i * T.hFull (ω.2.2.1 r) i + ∑ j, M.CpU r k j * ω.2.2.2.1 r j,
      ∑ i, M.ChU r k i * ω.2.2.2.2.2.1 r i + ∑ j, M.CpU r k j * ω.2.2.2.2.1 r j)

/-- The digests `S_shvzk`'s hidden leaves commit: its dummy run's level-0 columns, and zero pads columns (§3.2 step 6). -/
def Hm96.digS (M : T.Hm96 C) (w0 : W) (σ : T.SimRand) : T.Hid → B
  | .inl q => M.dig w0 (T.dummyOf σ) (.inl q)
  | .inr _ => M.digP (0, 0)

/-- **The real prover's view**: `Table.view` with its hidden leaves real hm96 leaves, each under its own fresh salt. -/
noncomputable def Hm96.viewR [CharP F 2] (M : T.Hm96 C) (w : W) (ω : T.Rand) (ys : T.Hid → Salt) : T.View :=
  T.viewWith w ω fun i => M.H (M.dig w ω i + M.My (ys i), M.cs (ys i))

/-- **`S_shvzk` with real leaves**: its hidden leaves are hm96 leaves of its dummy run's level-0 columns and of zero pads
columns, each under a fresh salt. -/
noncomputable def Hm96.simR [CharP F 2] (M : T.Hm96 C) (w0 : W) (σ : T.SimRand) (ys : T.Hid → Salt) : T.View :=
  T.simWith w0 σ fun i => M.H (M.digS w0 σ i + M.My (ys i), M.cs (ys i))

end Table

end FlockSoundness.ZK
