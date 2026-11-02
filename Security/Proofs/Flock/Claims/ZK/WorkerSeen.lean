import Proofs.Flock.Claims.ZK.Worker

/-!
# Lemma B at values read from what the workers see

`Worker` proves Lemma B at values that read a table's mask slot and level 0's hiding (`Table.Vals`). A worker sees more
before its algebraic phase: level 0's salts, at the opened positions and at the others, and every table's cells, not
only its own. Its values are then a function of what all the tables' workers see (`Table.Seen`), and a masked value
may read randomness sampled after its pad, so Lemma B's sequential masking no longer applies to the switched view.

The chain is reordered instead. With the pads trees' hidden leaves ideal first, a bijection of each table's
randomness that keeps what the workers see moves the switched view at constant values `c` to the honest view
(`fixE`): where `c`'s rows hold it moves the free pads by `padShift`, `μ`, `pad_h` and `pad_μ` so that every message
stays (`swView_fix`), elsewhere it is `shift`. Then level 0's hidden leaves go ideal and Lemma B is the honest prover's.

* `constVals`, `fixAmt`, `fixE`: the constant values, the move of the inner proof's pads and the bijection.
* `realCore_fix`, `swView_fix`, `swViewWith_fix`, `swDig_inl_fix`, `swPre_fix`: the switched view, with its hidden
  leaves given, its level-0 digests and its pre-final view at constant values, at the moved randomness, are the honest
  prover's.
-/

namespace FlockSoundness.ZK.Table

open Finset

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K]

/-- **Constant clear values**: `c` at every witness, mask slot and level 0's hiding. -/
def constVals (T : Table F K W Salt Seed D Hi VL) (c : Fin 2 → Fin T.nE → F) : T.Vals := fun r _ _ _ => c r

/-- `padShift` from the honest values to `c`'s, as the table's own type. -/
def padShiftC (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (u : T.Mask) (V : T.Vis) :
    Fin 2 → (Fin T.nE → F) × F × F :=
  T.onVals.padShift (w, T.constVals c) (w, T.vals) u V

theorem padShiftC_apply (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (u : T.Mask)
    (V : T.Vis) (r : Fin 2) :
    T.padShiftC w c u V r = (fun i => c r i - T.vals r w u V i, -(T.ζ r * (c r T.ia - T.vals r w u V T.ia)),
      -(c r T.ib - T.vals r w u V T.ib)) :=
  rfl

variable [CharP F 2]

open Classical in
/-- **The move of the inner proof's pads** at `(u, V, h)`: where `c`'s rows hold, with `h' := h + padShift` and
`d := h − h'` on the full pad vectors, `pad_h` by a solution `δ` of `C_p·δ = C_h·d`, `pad_μ` by `−δ/β` and `μ` by
`d/β`; nothing elsewhere. -/
noncomputable def fixAmt (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (u : T.Mask)
    (V : T.Vis) (h : Fin 2 → (Fin T.nE → F) × F × F) :
    (Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) :=
  if T.onVals.RowsAt (w, T.constVals c) u V then
    (fun r => Function.invFun (fun p : Fin T.nPad → F => fun k => ∑ j, T.Cp r k j * p j) fun k =>
        ∑ i, T.Ch r k i * (T.hFull (h r) i - T.hFull (h r + T.padShiftC w c u V r) i),
      fun r j => -(Function.invFun (fun p : Fin T.nPad → F => fun k => ∑ j, T.Cp r k j * p j) (fun k =>
        ∑ i, T.Ch r k i * (T.hFull (h r) i -
          T.hFull (h r + T.padShiftC w c u V r) i)) j / T.β r),
      fun r i => (T.hFull (h r) i - T.hFull (h r + T.padShiftC w c u V r) i) / T.β r)
  else 0

/-- **The fix map** at constant values `c`: the free pads moved by `padShift`, the inner proof's pads by `fixAmt`, the
mask slot, level 0's hiding and every salt and seed kept. -/
noncomputable def fixE (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) : T.Rand ≃ T.Rand where
  toFun ω := (ω.1, ω.2.1, ω.2.2.1 + T.padShiftC w c ω.1 ω.2.1,
    ω.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).1, ω.2.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).2.1,
    ω.2.2.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).2.2, ω.2.2.2.2.2.2)
  invFun ω :=
    (ω.1, ω.2.1, ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1,
      ω.2.2.2.1 - (T.fixAmt w c ω.1 ω.2.1 (ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1)).1,
      ω.2.2.2.2.1 -
        (T.fixAmt w c ω.1 ω.2.1 (ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1)).2.1,
      ω.2.2.2.2.2.1 -
        (T.fixAmt w c ω.1 ω.2.1 (ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1)).2.2,
      ω.2.2.2.2.2.2)
  left_inv ω := by
    obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
    simp only [add_sub_cancel_right]
  right_inv ω := by
    obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
    simp only [sub_add_cancel]

theorem fixE_apply (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (ω : T.Rand) :
    T.fixE w c ω = (ω.1, ω.2.1, ω.2.2.1 + T.padShiftC w c ω.1 ω.2.1,
      ω.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).1, ω.2.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).2.1,
      ω.2.2.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).2.2, ω.2.2.2.2.2.2) :=
  rfl

/-- Where `c`'s rows fail, the fix map is `shift`. -/
theorem fixE_of_not_rows (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (ω : T.Rand)
    (hr : ¬ T.onVals.RowsAt (w, T.constVals c) ω.1 ω.2.1) :
    T.fixE w c ω = T.onVals.shift (w, T.constVals c) (w, T.vals) ω := by
  have h0 : T.fixAmt w c ω.1 ω.2.1 ω.2.2.1 = 0 := by
    unfold fixAmt
    exact ite_eq_right hr
  rw [fixE_apply, h0]
  simp only [Prod.fst_zero, Prod.snd_zero, add_zero]
  rfl

omit [CharP F 2] in
/-- The `δ` of `fixAmt` solves `C_p·δ = C_h·d`. -/
theorem fixAmt_spec (T : Table F K W Salt Seed D Hi VL) (hpads : T.PadsOnto) (r : Fin 2) (d : PadIdx T.nE → F) :
    (fun k => ∑ j, T.Cp r k j *
        Function.invFun (fun p : Fin T.nPad → F => fun k => ∑ j, T.Cp r k j * p j) (fun k => ∑ i, T.Ch r k i * d i) j)
      = fun k => ∑ i, T.Ch r k i * d i :=
  Function.invFun_eq (hpads r _)

variable [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi]

set_option smartUnfolding false in
/-- The view's components at a table's witness paired with any clear values: the table's, with `v`'s values masked. -/
theorem onVals_realCore (T : Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand) :
    T.onVals.realCore (w, v) ω = ((T.nu w + T.maskHom ω.1,
      (fun q l => ∑ c, T.lanes w ω.1 l c * T.E q c, fun r => T.y1 w ω.1 r, 0) + T.mask1 ω.2.1,
      (fun r => (v r w ω.1 ω.2.1, 0, 0)) + T.mask2 ω.2.2.1,
      (fun r k => ∑ i, T.Ch r k i * T.hFull (ω.2.2.1 r) i, 0, fun r => T.hFull (ω.2.2.1 r), 0) +
        T.mask3 ω.2.2.2) : T.Core) :=
  rfl

/-- **Where `c`'s rows hold, the fix map keeps every component of the view** (Lemma B's sampling order): the masked
values and `ρ_in`, `σ_in` by `padShift`, `C_h` by `C_p·δ = C_h·d`, `Y₂` and `Y₁` by `−δ/β` and `d/β`. -/
theorem realCore_fix (T : Table F K W Salt Seed D Hi VL) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0) (w : W)
    (c : Fin 2 → Fin T.nE → F) (ω : T.Rand) (hr : T.onVals.RowsAt (w, T.constVals c) ω.1 ω.2.1) :
    T.onVals.realCore (w, T.constVals c) ω = T.realCore w (T.fixE w c ω) := by
  obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
  dsimp only at hr
  rw [onVals_realCore, fixE_apply]
  simp only [realCore, fixAmt, ite_eq_left hr]
  refine Prod.ext rfl (Prod.ext rfl (Prod.ext ?_ ?_))
  · funext r
    simp only [Pi.add_apply, mask2_apply, padShiftC_apply, constVals, Prod.mk_add_mk, Prod.fst_add, Prod.snd_add]
    refine Prod.ext ?_ (Prod.ext ?_ ?_)
    · funext i
      simp only [Pi.add_apply]
      ring
    · ring
    · ring
  · simp only [mask3_apply, Prod.mk_add_mk]
    refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl))
    · funext r k
      have hs := congrFun (T.fixAmt_spec hpads r fun i =>
        T.hFull (h r) i - T.hFull (h r + T.padShiftC w c u V r) i) k
      simp only [Pi.add_apply, mul_add, mul_sub, Finset.sum_add_distrib, Finset.sum_sub_distrib] at hs ⊢
      linear_combination -hs
    · funext r j
      have := hβ r
      simp only [Pi.add_apply]
      field_simp
      ring
    · funext r i
      have := hβ r
      simp only [Pi.add_apply]
      field_simp
      ring

omit [CharP F 2] [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] in
set_option smartUnfolding false in
theorem toBase_psi (T : Table F K W Salt Seed D Hi VL) (κ : T.Core) : T.toBase (T.onVals.psi κ) = T.psi κ :=
  rfl

/-- **The switched view at constant values is the honest view at the moved randomness**: where `c`'s rows hold by
`realCore_fix` (both are `psi` of their components), elsewhere because the fix map is `shift`. -/
theorem swView_fix (T : Table F K W Salt Seed D Hi VL) (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2)
    (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0) (w : W) (hreg : T.sReg w = T.pubReg) (hinner : T.InnerHolds w)
    (c : Fin 2 → Fin T.nE → F) (ω : T.Rand) :
    T.toBase (T.onVals.swView (w, T.constVals c) (w, T.vals) ω) = T.view w (T.fixE w c ω) := by
  by_cases hr : T.onVals.RowsAt (w, T.constVals c) ω.1 ω.2.1
  · simp only [swView, hr, ↓reduceIte]
    rw [← T.onVals.psi_realCore_at hW hβ (w, T.constVals c) hreg ω (hr ω rfl rfl), T.realCore_fix hpads hβ w c ω hr,
      toBase_psi, T.psi_realCore hW hβ w hreg hinner]
  · simp only [swView, hr, ↓reduceIte]
    rw [T.fixE_of_not_rows w c ω hr]
    exact T.toBase_view w _

/-- **What a worker sees of a table's randomness** before its algebraic phase: the mask slot, level 0's hiding, and
level 0's salts at the opened positions and at the others. -/
abbrev Seen (T : Table F K W Salt Seed D Hi VL) : Type :=
  T.Mask × T.Vis × (Fin T.nS → Salt) × (Fin T.nU → Salt)

/-- What a worker sees of the randomness `ω` and level 0's hidden leaves' salts `ys`. -/
def seenOf (T : Table F K W Salt Seed D Hi VL) (ω : T.Rand) (ys : Fin T.nU → Salt) : T.Seen :=
  (ω.1, ω.2.1, ω.2.2.2.2.2.2.1, ys)

omit [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] in
/-- The fix map keeps what the workers see. -/
theorem seenOf_fixE (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (ω : T.Rand)
    (ys : Fin T.nU → Salt) : T.seenOf (T.fixE w c ω) ys = T.seenOf ω ys :=
  rfl

section Hm96

variable {B C : Type} [AddCommGroup B]


/-- **The switched view with its hidden leaves given, at constant values**, is the honest one at the moved
randomness. -/
theorem swViewWith_fix (T : Table F K W Salt (B × Salt) D Hi VL)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0) (w : W)
    (hreg : T.sReg w = T.pubReg) (hinner : T.InnerHolds w) (c : Fin 2 → Fin T.nE → F) (ω : T.Rand) (L : T.Hid → D) :
    T.toBase (T.onVals.swViewWith (w, T.constVals c) (w, T.vals) ω L) = T.viewWith w (T.fixE w c ω) L := by
  have h := T.swView_fix hW hpads hβ w hreg hinner c ω
  by_cases hr : T.onVals.RowsAt (w, T.constVals c) ω.1 ω.2.1
  · simp only [swView, hr, ↓reduceIte] at h
    simp only [swViewWith, hr, ↓reduceIte, viewWith]
    rw [← h]
    rfl
  · simp only [swView, hr, ↓reduceIte] at h
    simp only [swViewWith, hr, ↓reduceIte, viewWith]
    rw [← h]
    rfl

omit [AddCommGroup Salt] [AddCommGroup Hi] in
/-- **The switched view's level-0 digests are the honest ones at the moved randomness**: they read the mask slot and
level 0's hiding, which the fix map and `shift` keep. -/
theorem swDig_inl_fix (T : Table F K W Salt (B × Salt) D Hi VL) (M : T.Hm96 C) (w : W) (c : Fin 2 → Fin T.nE → F)
    (ω : T.Rand) (q : Fin T.nU) :
    M.onVals.swDig (w, T.constVals c) (w, T.vals) ω (.inl q) = M.dig w (T.fixE w c ω) (.inl q) := by
  unfold Hm96.swDig
  split_ifs <;> rfl

/-- **The honest view with level 0's hidden leaves real and the pads trees' ideal**: real hm96 leaves of level 0's
columns under the salts `ys`, ideal leaves of the seeds `ωu.2` in the pads trees. -/
noncomputable def Hm96.viewMix {T : Table F K W Salt (B × Salt) D Hi VL} (M : T.Hm96 C) (w : W) (ωu : T.Rand × (Fin 2 × Fin T.nPU → B × Salt))
    (ys : Fin T.nU → Salt) : T.View :=
  T.viewWith w ωu.1 (Sum.elim (fun q => M.H (M.dig w ωu.1 (.inl q) + M.My (ys q), M.cs (ys q))) fun i => T.ideal (ωu.2 i))

/-- The hidden leaves' seeds split into level 0's and the pads trees'. -/
def hidSplit (T : Table F K W Salt (B × Salt) D Hi VL) (X : Type) :
    (T.Rand × (Fin 2 × Fin T.nPU → X)) × (Fin T.nU → X) ≃ T.Rand × (T.Hid → X) where
  toFun p := (p.1.1, Sum.elim p.2 p.1.2)
  invFun q := ((q.1, fun i => q.2 (.inr i)), fun q' => q.2 (.inl q'))
  left_inv _ := rfl
  right_inv q := Prod.ext rfl (funext fun i => by rcases i with q' | i <;> rfl)

end Hm96

section Mix

open Game

variable {B C : Type} [AddCommGroup B] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt]
  [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]

/-- **Lemma B with level 0's hidden leaves real and the pads trees' ideal**, for one table: every event has
probabilities within `(n_U + |Hid|)·δ₁` of each other under the honest prover with those leaves and under `S_shvzk`
with real leaves (`ideal_leaves_swap_fintype` on level 0's leaves, then `table_shvzk_hm96`'s steps). -/
theorem mix_shvzk_hm96 (T : Table F K W Salt (B × Salt) D Hi VL) (M : T.Hm96 C) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding M.My M.cs δ₁) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (w : W) (hreg : T.sReg w = T.pubReg) (hinner : T.InnerHolds w) (w0 : W) (E : T.View → Prop) :
    prCoin (fun p : (T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) × (Fin T.nU → Salt) => E (M.viewMix w p.1 p.2))
        ≤ prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2))
          + ((T.nU : ENNReal) + Fintype.card T.Hid) * δ₁
      ∧ prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2))
        ≤ prCoin (fun p : (T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) × (Fin T.nU → Salt) => E (M.viewMix w p.1 p.2))
          + ((T.nU : ENNReal) + Fintype.card T.Hid) * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have : Nonempty T.SimRand := ⟨0⟩
  have hS := (T.table_shvzk hrank hpad hW hpads hβ w hreg hinner w0).prCoin_eq E
  have hv : prCoin (fun p : (T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) × (Fin T.nU → B × Salt) =>
        E (T.viewWith w p.1.1 (Sum.elim (fun q => M.H ((p.2 q).1, M.cs (p.2 q).2)) fun i => T.ideal (p.1.2 i))))
      = prCoin (fun ω => E (T.view w ω)) := by
    rw [← T.prCoin_viewWith_ideal w E, ← prCoin_congr_equiv (T.hidSplit (B × Salt))]
    congr 1
    funext p
    refine congrArg (fun L => E (T.viewWith w p.1.1 L)) (funext fun i => ?_)
    rcases i with q | i
    · exact (M.ideal_eq (p.2 q).1 (p.2 q).2).symm
    · rfl
  have hs : prCoin (fun p : T.SimRand × (T.Hid → B × Salt) =>
        E (T.simWith w0 p.1 fun i => M.H ((p.2 i).1, M.cs (p.2 i).2)))
      = prCoin (fun σ => E (T.sim w0 σ)) := by
    rw [← T.prCoin_simWith_ideal w0 E]
    congr 1
    funext p
    congr 2
    funext i
    exact (M.ideal_eq (p.2 i).1 (p.2 i).2).symm
  have v1 := ideal_leaves_swap_fintype M.My M.cs hT1 M.H
    (fun (ωu : T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) q => M.dig w ωu.1 (.inl q))
    (fun ωu L => E (T.viewWith w ωu.1 (Sum.elim L fun i => T.ideal (ωu.2 i))))
  have v2 := real_leaves_swap_fintype M.My M.cs hT1 M.H
    (fun (ωu : T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) q => M.dig w ωu.1 (.inl q))
    (fun ωu L => E (T.viewWith w ωu.1 (Sum.elim L fun i => T.ideal (ωu.2 i))))
  have s1 := ideal_leaves_swap_fintype M.My M.cs hT1 M.H (M.digS w0) (fun σ L => E (T.simWith w0 σ L))
  have s2 := real_leaves_swap_fintype M.My M.cs hT1 M.H (M.digS w0) (fun σ L => E (T.simWith w0 σ L))
  beta_reduce at v1 v2 s1 s2
  rw [hv] at v1 v2
  rw [hs] at s1 s2
  simp only [Fintype.card_fin] at v1 v2
  constructor
  · calc prCoin (fun p : (T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) × (Fin T.nU → Salt) => E (M.viewMix w p.1 p.2))
        ≤ prCoin (fun ω => E (T.view w ω)) + T.nU * δ₁ := v1
      _ = prCoin (fun σ => E (T.sim w0 σ)) + T.nU * δ₁ := by rw [hS]
      _ ≤ (prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2)) + Fintype.card T.Hid * δ₁)
          + T.nU * δ₁ := add_le_add s2 le_rfl
      _ = _ := by ring
  · calc prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2))
        ≤ prCoin (fun σ => E (T.sim w0 σ)) + Fintype.card T.Hid * δ₁ := s1
      _ = prCoin (fun ω => E (T.view w ω)) + Fintype.card T.Hid * δ₁ := by rw [hS]
      _ ≤ (prCoin (fun p : (T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) × (Fin T.nU → Salt) =>
            E (M.viewMix w p.1 p.2)) + T.nU * δ₁) + Fintype.card T.Hid * δ₁ := add_le_add v2 le_rfl
      _ = _ := by ring

end Mix

end FlockSoundness.ZK.Table
