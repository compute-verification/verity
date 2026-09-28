import Proofs.Flock.Soundness.ZK.Refusal
import Proofs.Flock.Soundness.ZK.RealLeaves

/-!
# Zero knowledge at a refused coin: the prover's prefix, with real leaves

`Refusal.table_prefinal_indep_cut` is Lemma A at every coin vector, for the pre-final view cut to what the prover sent,
in world `W₀` (every pre-final leaf ideal). Here the same at a refused coin with the real prover's leaves:

* `Table.View.toPre`, `Table.abort`: what a prover that stops sends, read off its own view: the pre-final messages (every
  leaf as the real view has it, the opened ones too: their salts are never revealed), cut to the guarded messages it sent.
  `Table.pubAbort`: the same with the public region values in place of the region claims' `ŝ`, which is what the
  simulator outputs there.
* `Table.table_prefinal_indep_pub`: Lemma A with the public region values, at any two witnesses (the dummy one need not
  carry the region data).
* `Table.table_abort_hm96`: at messages the coins admit, the real prover's prefix on a witness carrying the public region
  data and the honest prefix on any dummy witness `w₀` (run on `S_shvzk`'s dummy randomness, with the public region values)
  are within `2·|Pre|·δ₁`, both ways. `Pre` is every pre-final leaf: level 0's tree, both pads trees and both commitments
  to `τ`. Every one of them is unopened at a refusal (the final message never comes), so T1 is paid on each, not only on
  `Hid`.
-/

namespace FlockSoundness.ZK

open Finset Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Table

section Defs

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)

/-- **A view's pre-final messages**: everything but the final message (`τ`, `y_τ`, the opened rows, columns and salts). -/
def View.toPre (v : T.View) : T.PreView where
  sHat := v.sHat
  sReg := v.sReg
  masked := v.masked
  y1 := v.y1
  ybar := v.ybar
  lig := v.lig
  inner := v.inner
  Y1 := v.Y1
  Y2 := v.Y2
  leaves0 := v.leaves0
  tauCom := v.tauCom
  leavesP := v.leavesP

/-- **What a prover that stops sends**, read off its view: the pre-final messages, cut to the guarded messages `S`. -/
noncomputable def abort (S : T.Sent) (v : T.View) : T.Cut := T.cut S (View.toPre T v)

/-- **The simulator's output at a refusal**: `abort` with the public region values in place of the region claims' `ŝ`. -/
noncomputable def pubAbort (S : T.Sent) (v : T.View) : T.Cut := { T.abort S v with sReg := T.pubReg }

/-- The pre-final view cut to `S`, with the public region values. -/
noncomputable def pubCut (S : T.Sent) (v : T.PreView) : T.Cut := { T.cut S v with sReg := T.pubReg }

/-- `pubAbort` is `pubCut` of the view's pre-final messages. -/
theorem pubAbort_eq (S : T.Sent) (v : T.View) : T.pubAbort S v = T.pubCut S (View.toPre T v) := rfl

end Defs

section Real

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K]
  (T : Table F K W Salt (B × Salt) D Hi VL)

/-- **Every pre-final leaf**: level 0's opened positions, each rep's opened pads positions and each rep's commitment to `τ`;
then the hidden leaves `Hid`. -/
abbrev Pre := (Fin T.nS ⊕ (Fin 2 × Fin T.nQ) ⊕ Fin 2) ⊕ T.Hid

/-- The ideal seeds of every pre-final leaf in the prover's randomness. -/
def idealOf (s : T.Seeds) : T.Pre → B × Salt :=
  Sum.elim (Sum.elim s.2.1 (Sum.elim (fun rk => s.2.2.2.2.1 rk.1 rk.2) s.2.2.2.2.2.2.2.1))
    (Sum.elim s.2.2.1 fun rk => s.2.2.2.2.2.1 rk.1 rk.2)

/-- Replace the ideal seeds of every pre-final leaf. -/
def setIdeal (s : T.Seeds) (u : T.Pre → B × Salt) : T.Seeds :=
  (s.1, fun q => u (.inl (.inl q)), fun q => u (.inr (.inl q)), s.2.2.2.1, fun r k => u (.inl (.inr (.inl (r, k)))),
    fun r k => u (.inr (.inr (r, k))), s.2.2.2.2.2.2.1, fun r => u (.inl (.inr (.inr r))), s.2.2.2.2.2.2.2.2)

/-- The salts of the leaves the final message would open: level 0's, each rep's pads', each rep's `y_τ`. -/
def openOf (s : T.Seeds) : Fin T.nS ⊕ (Fin 2 × Fin T.nQ) ⊕ Fin 2 → Salt :=
  Sum.elim s.1 (Sum.elim (fun rk => s.2.2.2.1 rk.1 rk.2) s.2.2.2.2.2.2.1)

/-- Replace the salts of the leaves the final message would open. -/
def setOpen (s : T.Seeds) (y : Fin T.nS ⊕ (Fin 2 × Fin T.nQ) ⊕ Fin 2 → Salt) : T.Seeds :=
  (fun q => y (.inl q), s.2.1, s.2.2.1, fun r k => y (.inr (.inl (r, k))), s.2.2.2.2.1, s.2.2.2.2.2.1,
    fun r => y (.inr (.inr r)), s.2.2.2.2.2.2.2)

/-- The prover's randomness with its seeds replaced by `f`. -/
def mapSeeds (ω : T.Rand) (f : T.Seeds → T.Seeds) : T.Rand :=
  (ω.1, ω.2.1, ω.2.2.1, ω.2.2.2.1, ω.2.2.2.2.1, ω.2.2.2.2.2.1, f ω.2.2.2.2.2.2)

theorem idealOf_setIdeal (s : T.Seeds) (u : T.Pre → B × Salt) : T.idealOf (T.setIdeal s u) = u := by
  funext i
  rcases i with (q | ⟨r, k⟩ | r) | (q | ⟨r, k⟩) <;> rfl

theorem openOf_setOpen (s : T.Seeds) (y : Fin T.nS ⊕ (Fin 2 × Fin T.nQ) ⊕ Fin 2 → Salt) :
    T.openOf (T.setOpen s y) = y := by
  funext i
  rcases i with q | ⟨r, k⟩ | r <;> rfl

variable [AddCommGroup B] [CharP F 2]

variable [AddCommGroup Salt] [AddCommGroup Hi] [Fintype F] [DecidableEq F]
  [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]

/-- **Lemma A with the public region values** (§4.2), for one table: at messages the coins admit, the pre-final views in
world `W₀` on any two witnesses, cut to those messages with the public region values in place of the region claims'
`ŝ`, are equally distributed. -/
theorem table_prefinal_indep_pub (S : T.Sent) (hS : T.Admits S) (w w' : W) :
    SameDist (fun ω => T.pubCut S (T.preView w ω)) (fun ω => T.pubCut S (T.preView w' ω)) := by
  classical
  -- the table whose region claims' messages are the public ones at every witness
  let T' : Table F K W Salt (B × Salt) D Hi VL := { T with sReg := fun _ => T.pubReg }
  have h := T'.table_prefinal_indep_cut ⟨S.cl, S.y1, S.fin⟩ hS w w' rfl rfl
  have h2 := h.comp fun v : T'.Cut =>
    (⟨v.sHat, v.sReg, v.masked, v.y1, v.ybar, v.lig, v.inner, v.Y1, v.Y2, v.leaves0, v.tauCom, v.leavesP⟩ : T.Cut)
  convert h2 using 2 <;> apply Cut.ext <;> first | rfl | (funext r; show @ite _ _ _ _ _ = @ite _ _ _ _ _; congr 2; funext i; cases i <;> rfl)

/-- The pre-final view with its leaves `L`, every pre-final leaf by `Pre`. -/
noncomputable def preLeaves (w : W) (ω : T.Rand) (L : T.Pre → D) : T.PreView :=
  { T.preView w ω with
    leaves0 := (fun q => L (.inl (.inl q)), fun q => L (.inr (.inl q)))
    tauCom := fun r => L (.inl (.inr (.inr r)))
    leavesP := fun r => (fun k => L (.inl (.inr (.inl (r, k)))), fun k => L (.inr (.inr (r, k)))) }

variable {T} in
/-- The digest each pre-final leaf of the prover's view commits. -/
noncomputable def Hm96.digAll {C : Type} (M : T.Hm96 C) (digT : F → B) (w : W) (ω : T.Rand) : T.Pre → B
  | .inl (.inl q) => M.dig0 ((T.view w ω).rows q, (T.view w ω).xrows q)
  | .inl (.inr (.inl (r, k))) => M.digP ((T.view w ω).ch r k, (T.view w ω).cm r k)
  | .inl (.inr (.inr r)) => digT ((T.view w ω).tau r)
  | .inr i => M.dig w ω i

omit [AddCommGroup Salt] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt]
  [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi] in
/-- The real prover's pre-final messages: every leaf the hm96 leaf of its digest under its own salt, the opened ones' from
the randomness, the hidden ones' from `ys`. -/
theorem toPre_viewR {C : Type} (M : T.Hm96 C) (digT : F → B) (hT : ∀ t y, T.leafT t y = M.H (digT t + M.My y, M.cs y))
    (w : W) (ω : T.Rand) (ys : T.Hid → Salt) :
    View.toPre T (M.viewR w ω ys) = T.preLeaves w ω fun i =>
      M.H (M.digAll digT w ω i + M.My (Sum.elim (T.openOf ω.2.2.2.2.2.2) ys i),
        M.cs (Sum.elim (T.openOf ω.2.2.2.2.2.2) ys i)) := by
  apply PreView.ext
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact Prod.ext (funext fun q => M.leaf0_eq _ _) rfl
  · exact funext fun r => hT _ _
  · exact funext fun r => Prod.ext (funext fun k => M.leafP_eq _ _) rfl

omit [AddCommGroup Hi] [DecidableEq F] [DecidableEq Salt] [DecidableEq B] [DecidableEq Hi] in
/-- **Every pre-final leaf's salt on its own**: the real prover's prefix, on its randomness and hidden salts, has the
distribution of the same messages with a fresh salt per pre-final leaf (its opened leaves' salts are read nowhere else
before the final message). -/
theorem prCoin_salts {C : Type} (M : T.Hm96 C) (digT : F → B) (w : W) (P : T.PreView → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) => P (T.preLeaves w p.1 fun i =>
        M.H (M.digAll digT w p.1 i + M.My (Sum.elim (T.openOf p.1.2.2.2.2.2.2) p.2 i),
          M.cs (Sum.elim (T.openOf p.1.2.2.2.2.2.2) p.2 i))))
      = prCoin (fun p : T.Rand × (T.Pre → Salt) => P (T.preLeaves w p.1 fun i =>
          M.H (M.digAll digT w p.1 i + M.My (p.2 i), M.cs (p.2 i)))) := by
  have : Nonempty Salt := ⟨0⟩
  let Ψ : (T.Rand × (T.Hid → Salt)) × (Fin T.nS ⊕ (Fin 2 × Fin T.nQ) ⊕ Fin 2 → Salt) ≃ T.Rand × (T.Pre → Salt) :=
    { toFun := fun x => (T.mapSeeds x.1.1 fun s => T.setOpen s x.2, Sum.elim (T.openOf x.1.1.2.2.2.2.2.2) x.1.2)
      invFun := fun x => ((T.mapSeeds x.1 fun s => T.setOpen s fun i => x.2 (.inl i), fun i => x.2 (.inr i)),
        T.openOf x.1.2.2.2.2.2.2)
      left_inv := fun x => Prod.ext (Prod.ext rfl rfl) (T.openOf_setOpen _ _)
      right_inv := fun x => Prod.ext (by
          change T.mapSeeds _ (fun s => T.setOpen s (T.openOf _)) = _
          rfl) (by
          funext i
          rcases i with i | i
          · exact congrFun (T.openOf_setOpen x.1.2.2.2.2.2.2 fun i => x.2 (.inl i)) i
          · rfl) }
  have h := prCoin_congr_equiv Ψ (fun p : T.Rand × (T.Pre → Salt) => P (T.preLeaves w p.1 fun i =>
    M.H (M.digAll digT w p.1 i + M.My (p.2 i), M.cs (p.2 i))))
  rw [← h, ← prCoin_fst (E := Fin T.nS ⊕ (Fin 2 × Fin T.nQ) ⊕ Fin 2 → Salt)]
  rfl

omit [AddCommGroup Hi] [DecidableEq F] [DecidableEq Salt] [DecidableEq B] [DecidableEq Hi] in
/-- **Fresh ideal seeds for every pre-final leaf give `preView`**: it reads those seeds only in its leaves. -/
theorem prCoin_preLeaves_ideal (w : W) (P : T.PreView → Prop) :
    prCoin (fun p : T.Rand × (T.Pre → B × Salt) => P (T.preLeaves w p.1 fun i => T.ideal (p.2 i)))
      = prCoin (fun ω => P (T.preView w ω)) := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty B := ⟨0⟩
  let Φ : T.Rand × (T.Pre → B × Salt) ≃ T.Rand × (T.Pre → B × Salt) :=
    { toFun := fun p => (T.mapSeeds p.1 fun s => T.setIdeal s p.2, T.idealOf p.1.2.2.2.2.2.2)
      invFun := fun p => (T.mapSeeds p.1 fun s => T.setIdeal s p.2, T.idealOf p.1.2.2.2.2.2.2)
      left_inv := fun p => Prod.ext rfl (T.idealOf_setIdeal _ _)
      right_inv := fun p => Prod.ext rfl (T.idealOf_setIdeal _ _) }
  have h := prCoin_congr_equiv Φ (fun p => P (T.preLeaves w p.1 fun i => T.ideal (p.2 i)))
  rw [← h, ← prCoin_fst (E := T.Pre → B × Salt) (fun ω => P (T.preView w ω))]
  rfl

omit [DecidableEq F] [DecidableEq Salt] [DecidableEq B] [DecidableEq Hi] in
/-- **The real prefix against world `W₀`**, both ways, at `|Pre|·δ₁` (T1 on every pre-final leaf). -/
theorem prCoin_pubAbort_close {C : Type} (M : T.Hm96 C) (digT : F → B)
    (hT : ∀ t y, T.leafT t y = M.H (digT t + M.My y, M.cs y)) {δ₁ : ℝ≥0∞}
    (hT1 : Assumptions.Hm96Hiding M.My M.cs δ₁) (S : T.Sent) (w : W) (G : T.Cut → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) => G (T.pubAbort S (M.viewR w p.1 p.2)))
        ≤ prCoin (fun ω => G (T.pubCut S (T.preView w ω))) + Fintype.card T.Pre * δ₁
      ∧ prCoin (fun ω => G (T.pubCut S (T.preView w ω)))
        ≤ prCoin (fun p : T.Rand × (T.Hid → Salt) => G (T.pubAbort S (M.viewR w p.1 p.2))) + Fintype.card T.Pre * δ₁ := by
  classical
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have e1 : prCoin (fun p : T.Rand × (T.Hid → Salt) => G (T.pubAbort S (M.viewR w p.1 p.2)))
      = prCoin (fun p : T.Rand × (T.Pre → Salt) => G (T.pubCut S (T.preLeaves w p.1 fun i =>
          M.H (M.digAll digT w p.1 i + M.My (p.2 i), M.cs (p.2 i))))) := by
    rw [← T.prCoin_salts M digT w (fun v => G (T.pubCut S v))]
    congr 1
    funext p
    rw [pubAbort_eq, toPre_viewR T M digT hT]
  have e2 : prCoin (fun p : T.Rand × (T.Pre → B × Salt) => G (T.pubCut S (T.preLeaves w p.1 fun i =>
        M.H ((p.2 i).1, M.cs (p.2 i).2))))
      = prCoin (fun ω => G (T.pubCut S (T.preView w ω))) := by
    rw [← T.prCoin_preLeaves_ideal w (fun v => G (T.pubCut S v))]
    congr 1
    funext p
    congr 4
    funext i
    exact (M.ideal_eq (p.2 i).1 (p.2 i).2).symm
  have v1 := ideal_leaves_swap_fintype M.My M.cs hT1 M.H (M.digAll digT w)
    (fun ω L => G (T.pubCut S (T.preLeaves w ω L)))
  have v2 := real_leaves_swap_fintype M.My M.cs hT1 M.H (M.digAll digT w)
    (fun ω L => G (T.pubCut S (T.preLeaves w ω L)))
  beta_reduce at v1 v2
  rw [e2] at v1 v2
  rw [e1]
  exact ⟨v1, v2⟩

omit [AddCommGroup B] [CharP F 2] [AddCommGroup Salt] [AddCommGroup Hi] [DecidableEq F]
  [DecidableEq Salt] [DecidableEq B] [DecidableEq Hi] in
/-- `S_shvzk`'s randomness is its dummy run's and fresh values the dummy run doesn't read. -/
theorem prCoin_dummyOf (P : T.Rand → (T.Hid → Salt) → Prop) :
    prCoin (fun p : T.SimRand × (T.Hid → Salt) => P (T.dummyOf p.1) p.2)
      = prCoin (fun p : T.Rand × (T.Hid → Salt) => P p.1 p.2) := by
  have : Nonempty F := ⟨0⟩
  let e : T.SimRand × (T.Hid → Salt) ≃ (T.Rand × (T.Hid → Salt)) × ((Fin T.nC → Fin 128 → F)
      × (Fin 2 → (Fin T.nE → F) × F × F) × ((Fin 2 → Fin T.nQ → F) × (Fin 2 → Fin T.nPad → F)
        × (Fin 2 → PadIdx T.nE → F))) :=
    { toFun := fun p => ((T.dummyOf p.1, p.2), (p.1.1.2, p.1.2.2.1.2, p.1.2.2.2.2))
      invFun := fun p => (((p.1.1.1, p.2.1), p.1.1.2.1, (p.1.1.2.2.1, p.2.2.1), (p.1.1.2.2.2, p.2.2.2)), p.1.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  rw [← prCoin_congr_equiv e.symm]
  exact prCoin_fst (fun p : T.Rand × (T.Hid → Salt) => P p.1 p.2)

/-- **A refusal, with real leaves** (Lemma A and T1, §4.1–§4.2): at messages `S` the coins admit, for a witness `w`
carrying the public region data and any dummy witness `w₀`, every event on what the real prover sends before it stops
(`abort` of its view `viewR`, every leaf the real hm96 leaf of what it commits) and on the simulator's output (`pubAbort`
of the honest view on `w₀`, at `S_shvzk`'s dummy randomness) has probabilities within `2·|Pre|·δ₁` of each other, both
ways. `τ`'s commitment is an hm96 leaf of its own digest `digT` (`hT`). -/
theorem table_abort_hm96 {C : Type} (M : T.Hm96 C) (digT : F → B)
    (hT : ∀ t y, T.leafT t y = M.H (digT t + M.My y, M.cs y)) {δ₁ : ℝ≥0∞}
    (hT1 : Assumptions.Hm96Hiding M.My M.cs δ₁) (S : T.Sent) (hS : T.Admits S) (w : W) (hreg : T.sReg w = T.pubReg)
    (w0 : W) (G : T.Cut → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) => G (T.abort S (M.viewR w p.1 p.2)))
        ≤ prCoin (fun p : T.SimRand × (T.Hid → Salt) => G (T.pubAbort S (M.viewR w0 (T.dummyOf p.1) p.2)))
          + 2 * Fintype.card T.Pre * δ₁
      ∧ prCoin (fun p : T.SimRand × (T.Hid → Salt) => G (T.pubAbort S (M.viewR w0 (T.dummyOf p.1) p.2)))
        ≤ prCoin (fun p : T.Rand × (T.Hid → Salt) => G (T.abort S (M.viewR w p.1 p.2)))
          + 2 * Fintype.card T.Pre * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have hr : prCoin (fun p : T.Rand × (T.Hid → Salt) => G (T.abort S (M.viewR w p.1 p.2)))
      = prCoin (fun p : T.Rand × (T.Hid → Salt) => G (T.pubAbort S (M.viewR w p.1 p.2))) := by
    congr 1
    funext p
    congr 1
    change T.cut S _ = { T.cut S _ with sReg := T.pubReg }
    exact Cut.ext rfl hreg rfl rfl rfl rfl rfl rfl rfl rfl rfl rfl
  have hs := T.prCoin_dummyOf (fun ω ys => G (T.pubAbort S (M.viewR w0 ω ys)))
  have hA := (T.table_prefinal_indep_pub S hS w w0).prCoin_eq G
  obtain ⟨a1, a2⟩ := T.prCoin_pubAbort_close M digT hT hT1 S w G
  obtain ⟨b1, b2⟩ := T.prCoin_pubAbort_close M digT hT hT1 S w0 G
  rw [hr, hs]
  constructor
  · calc _ ≤ _ := a1
      _ = prCoin (fun ω => G (T.pubCut S (T.preView w0 ω))) + Fintype.card T.Pre * δ₁ := by rw [hA]
      _ ≤ _ + Fintype.card T.Pre * δ₁ := add_le_add b2 le_rfl
      _ = _ := by ring
  · calc _ ≤ _ := b1
      _ = prCoin (fun ω => G (T.pubCut S (T.preView w ω))) + Fintype.card T.Pre * δ₁ := by rw [hA]
      _ ≤ _ + Fintype.card T.Pre * δ₁ := add_le_add a2 le_rfl
      _ = _ := by ring

end Real

end Table

end FlockSoundness.ZK
