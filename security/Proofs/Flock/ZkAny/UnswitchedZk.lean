module

public import Proofs.Flock.ZkAny.Deferred

/-! The unswitched prover is zero knowledge at any values, at a simulator that reads no worker's values (`RefSession.unP_zk`). -/

@[expose] public section

namespace FlockSoundness.ZK.Table

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

noncomputable def muMove {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (u : T.Mask) (V : T.Vis)
    (h : Fin 2 → (Fin T.nE → F) × F × F) : Fin 2 → PadIdx T.nE → F :=
  fun r i => (T.hFull (h r) i - T.hFull (h r + T.padShiftC w c u V r) i) / T.β r

noncomputable def fixU {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) : T.Rand ≃ T.Rand where
  toFun ω := (ω.1, ω.2.1, ω.2.2.1 + T.padShiftC w c ω.1 ω.2.1, ω.2.2.2.1, ω.2.2.2.2.1,
    ω.2.2.2.2.2.1 + T.muMove w c ω.1 ω.2.1 ω.2.2.1, ω.2.2.2.2.2.2)
  invFun ω := (ω.1, ω.2.1, ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1, ω.2.2.2.1, ω.2.2.2.2.1,
    ω.2.2.2.2.2.1 - T.muMove w c ω.1 ω.2.1 (ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1), ω.2.2.2.2.2.2)
  left_inv ω := by
    obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
    simp only [add_sub_cancel_right]
  right_inv ω := by
    obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
    simp only [sub_add_cancel]

open Classical in
set_option smartUnfolding false in
theorem cut_unPre_fixU {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (S : T.Sent) (hS : T.Admits S) (w : W) (c : Fin 2 → Fin T.nE → F)
    (ω : T.Rand) : T.cut S (T.unPre w (T.constVals c) ω) = T.cut S (T.preView w (T.fixU w c ω)) := by
  obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
  refine Cut.ext rfl rfl ?_ rfl rfl rfl ?_ ?_ rfl rfl rfl rfl
  · funext r i
    show c r i + (h r).1 i = T.vals r w u V i + (h r + T.padShiftC w c u V r).1 i
    rw [padShiftC_apply, Prod.fst_add, Pi.add_apply]
    ring
  · funext r
    show T.innerMsg r (h r) = T.innerMsg r (h r + T.padShiftC w c u V r)
    rw [padShiftC_apply]
    simp only [innerMsg, Prod.fst_add, Prod.snd_add, Pi.add_apply]
    refine Prod.ext ?_ ?_ <;> ring
  · funext r
    show (if S.fin r then some (fun i => T.hFull (h r) i + T.β r * μ r i) else none) =
      if S.fin r then some (fun i => T.hFull (h r + T.padShiftC w c u V r) i +
        T.β r * (μ r i + T.muMove w c u V h r i)) else none
    by_cases hf : S.fin r
    · have hβ := hS.2.2 r hf
      rw [ite_eq_left hf, ite_eq_left hf]
      refine congrArg some (funext fun i => ?_)
      simp only [muMove]
      field_simp
      ring
    · rw [ite_eq_right hf, ite_eq_right hf]

theorem unPre_congr {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (w : W) (v v' : T.Vals) (ω : T.Rand)
    (h : ∀ r, v r w ω.1 ω.2.1 = v' r w ω.1 ω.2.1) : T.unPre w v ω = T.unPre w v' ω := by
  unfold unPre
  rw [T.onVals_preView_congr w v v' ω h]

end FlockSoundness.ZK.Table

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.fixUB {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (c : Fin 2 → Fin z.nE → FlockLevel3.GF128) :
    ZkView.Base z Hi ≃ ZkView.Base z Hi where
  toFun b :=
    let s : Fin 2 → (Fin z.nE → FlockLevel3.GF128) × FlockLevel3.GF128 × FlockLevel3.GF128 :=
      T.tab.padShiftC w c b.1 b.2.1
    let m : Fin 2 → FlockSoundness.ZK.PadIdx z.nE → FlockLevel3.GF128 := T.tab.muMove w c b.1 b.2.1 b.2.2.1
    (b.1, b.2.1, b.2.2.1 + s, b.2.2.2.1, b.2.2.2.2.1, b.2.2.2.2.2.1 + m, b.2.2.2.2.2.2)
  invFun b :=
    let s : Fin 2 → (Fin z.nE → FlockLevel3.GF128) × FlockLevel3.GF128 × FlockLevel3.GF128 :=
      T.tab.padShiftC w c b.1 b.2.1
    let m : Fin 2 → FlockSoundness.ZK.PadIdx z.nE → FlockLevel3.GF128 :=
      T.tab.muMove w c b.1 b.2.1 (b.2.2.1 - s)
    (b.1, b.2.1, b.2.2.1 - s, b.2.2.2.1, b.2.2.2.2.1, b.2.2.2.2.2.1 - m, b.2.2.2.2.2.2)
  left_inv b := by
    have e := (T.tab.fixU w c).symm_apply_apply (T.randOf b 0)
    exact Prod.ext rfl (Prod.ext rfl (Prod.ext (congrArg (fun ω => ω.2.2.1) e)
      (Prod.ext rfl (Prod.ext rfl (Prod.ext (congrArg (fun ω => ω.2.2.2.2.2.1) e) rfl)))))
  right_inv b := by
    have e := (T.tab.fixU w c).apply_symm_apply (T.randOf b 0)
    exact Prod.ext rfl (Prod.ext rfl (Prod.ext (congrArg (fun ω => ω.2.2.1) e)
      (Prod.ext rfl (Prod.ext rfl (Prod.ext (congrArg (fun ω => ω.2.2.2.2.2.1) e) rfl)))))

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.randOf_fixUB {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (c : Fin 2 → Fin z.nE → FlockLevel3.GF128)
    (b : ZkView.Base z Hi) (u : ZkView.LeafPos z → ZkView.Bits × ZkView.Salt) :
    T.randOf (T.fixUB w c b) u = T.tab.fixU w c (T.randOf b u) :=
  rfl

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.unPreWith_congr {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v v' : T.tab.Vals) (b : ZkView.Base z Hi)
    (lv : ZkView.LeafPos z → ByteArray) (h : ∀ r, v r w b.1 b.2.1 = v' r w b.1 b.2.1) :
    T.unPreWith w v b lv = T.unPreWith w v' b lv := by
  unfold ZkView.Coined.unPreWith
  rw [T.tab.unPre_congr w v v' (T.randOf b 0) h]

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.cut_unPreWith_fixUB {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (S : T.tab.Sent) (hS : T.tab.Admits S) (w : W)
    (v : T.tab.Vals) (b : ZkView.Base z Hi) (lv lv' : ZkView.LeafPos z → ByteArray) (hl : ∀ p, lv p = lv' p) :
    T.tab.cut S (T.unPreWith w v b lv) = T.tab.cut S (T.preWith w (T.fixUB w (fun r => v r w b.1 b.2.1) b) lv') := by
  obtain rfl : lv = lv' := funext hl
  rw [T.unPreWith_congr w v (T.tab.constVals fun r => v r w b.1 b.2.1) b lv fun _ => rfl]
  have h : T.tab.cut S (T.tab.unPre w (T.tab.constVals fun r => v r w b.1 b.2.1) (T.randOf b 0)) =
      T.tab.cut S (T.tab.preView w (T.randOf (T.fixUB w (fun r => v r w b.1 b.2.1) b) 0)) :=
    T.tab.cut_unPre_fixU S hS w (fun r => v r w b.1 b.2.1) (T.randOf b 0)
  exact FlockSoundness.ZK.Table.cut_withLeaves T.tab S _ _ h _ _ _

set_option smartUnfolding false in
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.unContent_inl {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (p : Fin (z.nS + z.nU)) : T.unContent w v b (.inl p) = T.content w b (.inl p) :=
  rfl

set_option smartUnfolding false in
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.content_fixUB_inl {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (c : Fin 2 → Fin z.nE → FlockLevel3.GF128)
    (b : ZkView.Base z Hi) (p : Fin (z.nS + z.nU)) : T.content w (T.fixUB w c b) (.inl p) = T.content w b (.inl p) :=
  rfl

open Finset in
theorem card_unCutAt_real0 {J : ℕ} {V : Type} [DecidableEq V] (C : Fin J → Composed.CoinedSetup)
    (cz : ∀ j, Composed.CoinsZK (C j).Rest) (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m)
    (vs : (∀ j, ZkView.Seen (C j).z) → ∀ j, ((C j).T (cz j)).tab.Vals) (out : (∀ j, ((C j).T (cz j)).tab.Cut) → V)
    (σ : V) :
    #{u : ((∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) → ZkView.Salt)) ×
          (Leaves1 (fun j => (C j).z) → ZkView.Bits × ZkView.Salt) |
        out (unCutAt C cz wt (vs (seenAt C u.1.1 u.1.2)) u.1.1 fun l => Sum.elim
          (fun l0 => ZkView.leafH (unDigOf C cz wt (vs (seenAt C u.1.1 u.1.2)) u.1.1 ((leaves01 _).symm (.inl l0)) +
            ZkView.My (u.1.2 l0), ZkView.saltHash (u.1.2 l0)))
          (fun l1 => ZkView.leafH ((u.2 l1).1, ZkView.saltHash (u.2 l1).2)) (leaves01 _ l)) = σ}
      = #{u : ((∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) → ZkView.Salt)) ×
          (Leaves1 (fun j => (C j).z) → ZkView.Bits × ZkView.Salt) |
        out (cutAt C cz wt u.1.1 fun l => Sum.elim
          (fun l0 => ZkView.leafH (digOf C cz wt u.1.1 ((leaves01 _).symm (.inl l0)) + ZkView.My (u.1.2 l0),
            ZkView.saltHash (u.1.2 l0)))
          (fun l1 => ZkView.leafH ((u.2 l1).1, ZkView.saltHash (u.2 l1).2)) (leaves01 _ l)) = σ} := by
  let cv : (∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) → ZkView.Salt) →
      ∀ j, Fin 2 → Fin (C j).z.nE → FlockLevel3.GF128 :=
    fun q j r => vs (seenAt C q.1 q.2) j r (wt j) (q.1 j).1 (q.1 j).2.1
  let fixQ : (∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) → ZkView.Salt) ≃
      (∀ j, ZkView.Base (C j).z (C j).Hi) × (Leaves0 (fun j => (C j).z) → ZkView.Salt) :=
    { toFun := fun q => (fun j => ((C j).T (cz j)).fixUB (wt j) (cv q j) (q.1 j), q.2)
      invFun := fun q => (fun j => (((C j).T (cz j)).fixUB (wt j) (cv q j)).symm (q.1 j), q.2)
      left_inv := fun q =>
        Prod.ext (funext fun j => (((C j).T (cz j)).fixUB (wt j) (cv q j)).symm_apply_apply _) rfl
      right_inv := fun q =>
        Prod.ext (funext fun j => (((C j).T (cz j)).fixUB (wt j) (cv q j)).apply_symm_apply _) rfl }
  refine Finset.card_equiv (fixQ.prodCongr (Equiv.refl _)) fun u => ?_
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  refine Iff.of_eq (congrArg (fun x => out x = σ) (funext fun j => ?_))
  refine ((C j).T (cz j)).cut_unPreWith_fixUB _
    (famSent_admits (fun j => ((C j).T (cz j)).tab) (fun j => ((C j).T (cz j)).rep) j) (wt j)
    (vs (seenAt C u.1.1 u.1.2) j) (u.1.1 j) _ _ fun p => ?_
  rcases p with p | p
  · exact congrArg (fun d => ZkView.leafH (ZkView.bitsOf d + ZkView.My (u.1.2 ⟨j, p⟩), ZkView.saltHash (u.1.2 ⟨j, p⟩)))
      ((((C j).T (cz j)).unContent_inl _ _ _ p).trans (((C j).T (cz j)).content_fixUB_inl _ _ _ p).symm)
  · exact congrArg (fun y : ZkView.Bits × ZkView.Salt => ZkView.leafH (y.1, ZkView.saltHash y.2))
      (rfl : u.2 ⟨j, p⟩ = u.2 ⟨j, p⟩)

namespace Proto

noncomputable def TPreModel.unMsgs {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x)
    (wt : ∀ j, FlockSoundness.Model.Witness (M.C j).S.m)
    (vs : ∀ a, (∀ j, ZkView.Seen (M.C j).z) → ∀ j, ((M.C j).T (M.cz a j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (M.C j).z (M.C j).Hi) (s : ∀ j, ZkView.Seen (M.C j).z) (a : Fin (G.K x + 1) → Out)
    (lv : ZkView.Leaves (fun j => (M.C j).z) → ByteArray) : Fin (G.K x + 1) → Msg :=
  M.out a (unCutAt M.C (M.cz a) wt (vs a s) ω lv)

def TPreModel.UnCausal {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x)
    (wt : ∀ j, FlockSoundness.Model.Witness (M.C j).S.m)
    (vs : ∀ a, (∀ j, ZkView.Seen (M.C j).z) → ∀ j, ((M.C j).T (M.cz a j)).tab.Vals) : Prop :=
  ∀ (ω : ∀ j, ZkView.Base (M.C j).z (M.C j).Hi) (s : ∀ j, ZkView.Seen (M.C j).z)
    (p : ZkView.Leaves (fun j => (M.C j).z) → ZkView.Bits × ByteArray) (a a' : Fin (G.K x + 1) → Out)
    (i : Fin (G.K x + 1)), (∀ j, j < i → a j = a' j) →
    M.unMsgs wt vs ω s a (fun l => ZkView.leafH (unDigOf M.C (M.cz a) wt (vs a s) ω l + (p l).1, (p l).2)) i =
      M.unMsgs wt vs ω s a' (fun l => ZkView.leafH (unDigOf M.C (M.cz a') wt (vs a' s) ω l + (p l).1, (p l).2)) i

def TPreModel.UnLink {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x) (H : Msg → Msg) (ν : G.Nonce) (w : W) :
    Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧
    (∀ a j, ((M.C j).T (M.cz a j)).tab.sReg (M.wt₀ j) = ((M.C j).T (M.cz a j)).tab.pubReg) ∧ M.Causal ∧
    (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    (∃ vs : ∀ a, (∀ j, ZkView.Seen (M.C j).z) → ∀ j, ((M.C j).T (M.cz a j)).tab.Vals, M.UnCausal (M.wt w) vs ∧
      ∀ r a, (G.preStrat H x ν (G.msg x w ν)).msg r a =
        M.unMsgs (M.wt w) vs (M.eR r).1 (seenOfP M.C (M.eR r)) a
          (unExeOf M.C (M.cz a) (M.wt w) (vs a (seenOfP M.C (M.eR r))) (M.eR r).1 (M.eR r).2)) ∧
    ∀ r a, (G.preStrat H x ν (G.dmsg x ν)).msg r a =
      M.msgs M.wt₀ (M.eR r).1 a (exeOf M.C (M.cz a) M.wt₀ (M.eR r).1 (M.eR r).2)

set_option maxRecDepth 20000 in
theorem TPreModel.runs_close_un {W : Type} {G : Proto W} {x : Msg} (M : G.TPreModel x) (H : Msg → Msg)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (w : W)
    (hL : M.UnLink H ν w) (hOK : M.OK) (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1))
    (E : (Fin (G.K x + 1) → Msg) → Prop) :
    |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
      Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))|
      ≤ 2 * M.size * δ₁.toReal := by
  obtain ⟨hv, hreg₀, hc, heR, ⟨vs, hsc, hlink⟩, hlink₀⟩ := hL
  have : Nonempty (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) := ⟨fun _ => 0⟩
  let e := leaves01 fun j => (M.C j).z
  let Φ : (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (ZkView.Leaves (fun j => (M.C j).z) → ZkView.Salt) ≃
      (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (Leaves0 (fun j => (M.C j).z) ⊕ Leaves1 (fun j => (M.C j).z) →
        ZkView.Salt) :=
    (Equiv.refl _).prodCongr (e.arrowCongr (Equiv.refl _))
  have heR' : ∀ p, Nat.card {r // Φ (M.eR r) = p} = M.kR := fun p => by
    rw [← heR (Φ.symm p)]
    exact Nat.card_congr (Equiv.subtypeEquivRight fun r => (Φ.eq_symm_apply (x := p) (y := M.eR r)).symm)
  let P : (∀ j, FlockSoundness.Model.Witness (M.C j).S.m) → FlockSoundness.ZK.Strategy
      ((∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (Leaves0 (fun j => (M.C j).z) ⊕ Leaves1 (fun j => (M.C j).z) →
        ZkView.Bits × ByteArray)) Msg Out (G.K x + 1) := fun wt =>
    { msg := fun q a => M.msgs wt q.1 a fun l =>
        ZkView.leafH (digOf M.C (M.cz a) wt q.1 (e.symm (e l)) + (q.2 (e l)).1, (q.2 (e l)).2)
      causal := fun q a a' i h => by
        simp only [Equiv.symm_apply_apply]
        exact hc wt q.1 (fun l => q.2 (e l)) a a' i h }
  let msgsS : (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (Leaves0 (fun j => (M.C j).z) → ZkView.Salt) →
      (Fin (G.K x + 1) → Out) → (Leaves1 (fun j => (M.C j).z) → ByteArray) → Fin (G.K x + 1) → Msg :=
    fun q a lv1 => M.unMsgs (M.wt w) vs q.1 (seenAt M.C q.1 q.2) a fun l => Sum.elim
      (fun l0 => ZkView.leafH (unDigOf M.C (M.cz a) (M.wt w) (vs a (seenAt M.C q.1 q.2)) q.1 (e.symm (.inl l0)) +
        ZkView.My (q.2 l0), ZkView.saltHash (q.2 l0))) lv1 (e l)
  let digS : (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (Leaves0 (fun j => (M.C j).z) → ZkView.Salt) →
      (Fin (G.K x + 1) → Out) → Leaves1 (fun j => (M.C j).z) → ZkView.Bits :=
    fun q a l1 => unDigOf M.C (M.cz a) (M.wt w) (vs a (seenAt M.C q.1 q.2)) q.1 (e.symm (.inr l1))
  have hsS : ∀ q (p : Leaves1 (fun j => (M.C j).z) → ZkView.Bits × ByteArray) (a a' : Fin (G.K x + 1) → Out)
      (i : Fin (G.K x + 1)), (∀ j, j < i → a j = a' j) →
      msgsS q a (fun l1 => ZkView.leafH (digS q a l1 + (p l1).1, (p l1).2)) i =
        msgsS q a' (fun l1 => ZkView.leafH (digS q a' l1 + (p l1).1, (p l1).2)) i := by
    intro q p a a' i h
    have k := fun a : Fin (G.K x + 1) → Out => leaves01_split_congr
      (M.unMsgs (M.wt w) vs q.1 (seenAt M.C q.1 q.2) a)
      (fun l y => ZkView.leafH (unDigOf M.C (M.cz a) (M.wt w) (vs a (seenAt M.C q.1 q.2)) q.1 l + y.1, y.2))
      (fun l0 => (ZkView.My (q.2 l0), ZkView.saltHash (q.2 l0))) p
    exact (congrFun (k a) i).trans ((hsc q.1 (seenAt M.C q.1 q.2)
      (fun l => Sum.elim (fun l0 => (ZkView.My (q.2 l0), ZkView.saltHash (q.2 l0))) p (e l)) a a' i h).trans
      (congrFun (k a') i).symm)
  let S : FlockSoundness.ZK.Strategy (((∀ j, ZkView.Base (M.C j).z (M.C j).Hi) ×
      (Leaves0 (fun j => (M.C j).z) → ZkView.Salt)) × (Leaves1 (fun j => (M.C j).z) → ZkView.Bits × ByteArray))
      Msg Out (G.K x + 1) :=
    { msg := fun qp a => msgsS qp.1 a fun l1 => ZkView.leafH (digS qp.1 a l1 + (qp.2 l1).1, (qp.2 l1).2)
      causal := fun qp a a' i h => hsS qp.1 qp.2 a a' i h }
  have hl : ∀ r, (G.preStrat H x ν (G.msg x w ν)).msg r =
      S.msg (((Φ (M.eR r)).1, fun l => (Φ (M.eR r)).2 (.inl l)),
        fun l => (ZkView.My ((Φ (M.eR r)).2 (.inr l)), ZkView.saltHash ((Φ (M.eR r)).2 (.inr l)))) :=
    fun r => funext fun a => by
      rw [hlink r a, unExeOf_eq]
      have k := leaves01_elim_congr (M.unMsgs (M.wt w) vs (M.eR r).1 (seenOfP M.C (M.eR r)) a) fun l =>
        ZkView.leafH (unDigOf M.C (M.cz a) (M.wt w) (vs a (seenOfP M.C (M.eR r))) (M.eR r).1 l +
          ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))
      exact k.symm
  have hl₀ : ∀ r, (G.preStrat H x ν (G.dmsg x ν)).msg r =
      (P M.wt₀).msg ((Φ (M.eR r)).1, fun l => (ZkView.My ((Φ (M.eR r)).2 l), ZkView.saltHash ((Φ (M.eR r)).2 l))) :=
    fun r => funext fun a => by
      rw [hlink₀ r a, exeOf_eq]
      have k := leaves01_symm_congr (M.msgs M.wt₀ (M.eR r).1 a) fun l =>
        ZkView.leafH (digOf M.C (M.cz a) M.wt₀ (M.eR r).1 l + ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))
      exact k
  rw [TPreModel.size, Fintype.card_congr e]
  exact G.runs_close_sw_of_ideal H x ν w ZkView.My ZkView.saltHash hT1 hδ₁ ZkView.leafH
    (fun wt ω a lv => M.msgs wt ω a fun l => lv (e l)) (fun wt ω a l => digOf M.C (M.cz a) wt ω (e.symm l))
    P (fun _ _ _ _ => rfl) msgsS digS S (fun _ _ _ => rfl) (M.wt w) M.wt₀
    (fun a σ => card_unCutAt_real0 M.C (M.cz a) (M.wt w) (vs a) (M.out a) σ)
    (fun a σ => card_cutAt_ideal01 M.C (M.cz a) (M.wt w) M.wt₀ (fun j => (hOK j).ok.2 _ _ (hv j)) (hreg₀ a)
      (M.out a) σ)
    (fun r => Φ (M.eR r)) M.kR heR' hl hl₀ V E

def TOffModel.UnLink {W : Type} {G : Proto W} {x : Msg} (M : G.TOffModel x) (H : Msg → Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) : Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧
    (∀ j, ((M.C j).T (M.cz j)).tab.sReg (M.wt₀ j) = ((M.C j).T (M.cz j)).tab.pubReg) ∧
    (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    (∃ vs : (∀ j, ZkView.Seen (M.C j).z) → ∀ j, ((M.C j).T (M.cz j)).tab.Vals, ∀ r, G.realVec H x w ν r c =
      M.out (unCutAt M.C M.cz (M.wt w) (vs (seenOfP M.C (M.eR r))) (M.eR r).1
        (unExeOf M.C M.cz (M.wt w) (vs (seenOfP M.C (M.eR r))) (M.eR r).1 (M.eR r).2))) ∧
    ∀ r, G.dVec H x ν r c = M.out (cutAt M.C M.cz M.wt₀ (M.eR r).1 (exeOf M.C M.cz M.wt₀ (M.eR r).1 (M.eR r).2))

theorem TOffModel.close_un {W : Type} {G : Proto W} {x : Msg} (M : G.TOffModel x) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) (hL : M.UnLink H ν c w) (hOK : M.OK) (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun r : G.Rand => E (G.dVec H x ν r c))|
      ≤ 2 * M.size * δ₁.toReal := by
  obtain ⟨hv, hreg₀, heR, ⟨vs, hR⟩, hD⟩ := hL
  have : Nonempty (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) := ⟨fun _ => 0⟩
  let e := leaves01 fun j => (M.C j).z
  let Φ : (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (ZkView.Leaves (fun j => (M.C j).z) → ZkView.Salt) ≃
      (∀ j, ZkView.Base (M.C j).z (M.C j).Hi) × (Leaves0 (fun j => (M.C j).z) ⊕ Leaves1 (fun j => (M.C j).z) →
        ZkView.Salt) :=
    (Equiv.refl _).prodCongr (e.arrowCongr (Equiv.refl _))
  have heR' : ∀ p, Nat.card {r // Φ (M.eR r) = p} = M.kR := fun p => by
    rw [← heR (Φ.symm p)]
    exact Nat.card_congr (Equiv.subtypeEquivRight fun r => (Φ.eq_symm_apply (x := p) (y := M.eR r)).symm)
  rw [TOffModel.size, Fintype.card_congr e]
  exact G.refused_close_sw_of_ideal H x w ν c ZkView.My ZkView.saltHash hT1 hδ₁ ZkView.leafH
    (fun wt ω lv => M.out (cutAt M.C M.cz wt ω fun l => lv (e l))) (fun wt ω l => digOf M.C M.cz wt ω (e.symm l))
    (fun q lv1 => M.out (unCutAt M.C M.cz (M.wt w) (vs (seenAt M.C q.1 q.2)) q.1 fun l => Sum.elim
      (fun l0 => ZkView.leafH (unDigOf M.C M.cz (M.wt w) (vs (seenAt M.C q.1 q.2)) q.1 (e.symm (.inl l0)) +
        ZkView.My (q.2 l0), ZkView.saltHash (q.2 l0))) lv1 (e l)))
    (fun q l1 => unDigOf M.C M.cz (M.wt w) (vs (seenAt M.C q.1 q.2)) q.1 (e.symm (.inr l1)))
    (M.wt w) M.wt₀
    (fun σ => card_unCutAt_real0 M.C M.cz (M.wt w) vs M.out σ)
    (fun σ => card_cutAt_ideal01 M.C M.cz (M.wt w) M.wt₀ (fun j => (hOK j).ok.2 _ _ (hv j)) hreg₀ M.out σ)
    (fun r => Φ (M.eR r)) M.kR heR'
    (fun r => by
      rw [hR r, unExeOf_eq]
      have k := leaves01_elim_congr
        (fun lv => M.out (unCutAt M.C M.cz (M.wt w) (vs (seenOfP M.C (M.eR r))) (M.eR r).1 lv)) fun l =>
          ZkView.leafH (unDigOf M.C M.cz (M.wt w) (vs (seenOfP M.C (M.eR r))) (M.eR r).1 l +
            ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))
      exact k.symm)
    (fun r => by
      rw [hD r, exeOf_eq]
      have k := leaves01_symm_congr (fun lv => M.out (cutAt M.C M.cz M.wt₀ (M.eR r).1 lv)) fun l =>
        ZkView.leafH (digOf M.C M.cz M.wt₀ (M.eR r).1 l + ZkView.My ((M.eR r).2 l), ZkView.saltHash ((M.eR r).2 l))
      exact k)
    E

end Proto

namespace RefSession

/-- **The refusing prover's message after the coins `cs`**, on table witnesses `wt` and tape `t`: before the last coin
`msgAt`'s round encoding of the cut view, after it `off` of the cut view, at every coin vector. -/
noncomputable def offMsgAt (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (t : S.Tape) (cs : List S.Coin) : Msg :=
  if cs.length < refK x then S.enc x (S.pad cs) cs.length (S.cutOf x (S.pad cs) wt t)
  else S.off x (S.pad cs) (S.cutOf x (S.pad cs) wt t)

/-- **The refusing reference protocol**: `refP` whose dummy refuses at every coin vector (`offMsgAt` on the dummy's
table witnesses) and whose SHVZK simulator at a coin vector is that dummy's messages there, on a tape of its own. -/
noncomputable def offP (S : RefSession) : Proto Flock.Wit :=
  { S.refP with
    dmsg := fun x _ t cs => .ret (S.offMsgAt x (S.wt₀ x) t cs)
    SimRand := fun _ => S.Tape
    finSim := fun _ => S.finTape
    neSim := fun _ => S.neTape
    sim := fun x _ c t => .ret ((List.range (refK x + 1)).map fun j => S.offMsgAt x (S.wt₀ x) t ((List.ofFn c).take j)) }

noncomputable instance instFintypeOffPCoin (S : RefSession) : Fintype S.offP.Coin := S.finCoin

noncomputable def unPS (S : RefSession) (vs : S.SwVals) : Proto Flock.Wit :=
  { S.offP with msg := (S.unP vs).msg }

noncomputable instance instFintypeUnPSCoin (S : RefSession) (vs : S.SwVals) : Fintype (S.unPS vs).Coin := S.finCoin

theorem unPS_prover (S : RefSession) (vs : S.SwVals) : (S.unPS vs).prover = (S.unP vs).prover := rfl

theorem unPS_simulatorE (S : RefSession) (vs : S.SwVals) (t M : ℕ) :
    (S.unPS vs).simulatorE t M = S.offP.simulatorE t M :=
  Proto.simulatorE_msg _ _ t M

theorem unPre_msg (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (hc : S.UnCausal vs) (t : S.Tape) (a : Fin (refK x + 1) → Out) (i : Fin (refK x + 1)) :
    S.refP.pmG H x ν (fun cs => .ret (S.unMsgAt x (S.wt x w) (vs x w ν) t cs)) (ansHist a i) =
      S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1)
        (S.unCutOf x (S.coinsOf a) (S.wt x w) (vs x w ν (S.coinsOf a)) t))) (ansHist a i) := by
  unfold Proto.pmG
  congr 2
  refine strat_congr_gen _ _ _ _ _ _ _ fun hne => ?_
  have hi : 0 < i.val := by
    rcases Nat.eq_zero_or_pos i.val with h0 | h0
    · exact absurd (by simp [ansHist, h0]) hne
    · exact h0
  have hl : ((ansHist a i).tail.map fun p => S.decOf p.2).length = i.val - 1 := by
    simp only [List.length_map, List.length_tail, ansHist, List.length_ofFn]
  show OC.ret (S.unMsgAt x (S.wt x w) (vs x w ν) t ((ansHist a i).tail.map fun p => S.decOf p.2)) = OC.ret _
  unfold unMsgAt
  rw [hl, ite_eq_left (by omega),
    S.enc_unCutOf vs x w ν hc t _ (S.coinsOf a) (i.val - 1) fun k hk => S.pad_ansHist a i k hk]

theorem offPre_msg (S : RefSession) (H : Msg → Msg) (x : Msg) (ν : S.Nonce) (hc : S.Causal)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (t : S.Tape) (a : Fin (refK x + 1) → Out)
    (i : Fin (refK x + 1)) :
    S.refP.pmG H x ν (fun cs => .ret (S.offMsgAt x wt t cs)) (ansHist a i) =
      S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1) (S.cutOf x (S.coinsOf a) wt t)))
        (ansHist a i) := by
  unfold Proto.pmG
  congr 2
  refine strat_congr_gen _ _ _ _ _ _ _ fun hne => ?_
  have hi : 0 < i.val := by
    rcases Nat.eq_zero_or_pos i.val with h0 | h0
    · exact absurd (by simp [ansHist, h0]) hne
    · exact h0
  have hl : ((ansHist a i).tail.map fun p => S.decOf p.2).length = i.val - 1 := by
    simp only [List.length_map, List.length_tail, ansHist, List.length_ofFn]
  show OC.ret (S.offMsgAt x wt t ((ansHist a i).tail.map fun p => S.decOf p.2)) = OC.ret _
  unfold offMsgAt
  rw [hl, ite_eq_left (by omega), S.enc_cutOf x hc wt t _ (S.coinsOf a) (i.val - 1) fun k hk => S.pad_ansHist a i k hk]

theorem unMsgAt_take (S : RefSession) (vs : S.SwVals) (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (hc : S.UnCausal vs)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) (j : ℕ) (hj : j < refK x + 1) :
    S.unMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j) =
      if j < refK x then S.enc x c j (S.unCutOf x c (S.wt x w) (vs x w ν c) t)
      else S.off x c (S.unCutOf x c (S.wt x w) (vs x w ν c) t) := by
  have hl : ((List.ofFn c).take j).length = j := by simp only [List.length_take, List.length_ofFn]; omega
  unfold unMsgAt
  rw [hl]
  by_cases h : j < refK x
  · rw [ite_eq_left h, ite_eq_left h]
    exact S.enc_unCutOf vs x w ν hc t _ c j fun k hk => S.pad_take c j k hk
  · rw [ite_eq_right h, ite_eq_right h]
    obtain rfl : j = refK x := by omega
    rw [S.pad_ofFn c]

theorem offMsgAt_take (S : RefSession) (x : Msg) (hc : S.Causal) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) (j : ℕ) (hj : j < refK x + 1) :
    S.offMsgAt x wt t ((List.ofFn c).take j) =
      if j < refK x then S.enc x c j (S.cutOf x c wt t) else S.off x c (S.cutOf x c wt t) := by
  have hl : ((List.ofFn c).take j).length = j := by simp only [List.length_take, List.length_ofFn]; omega
  unfold offMsgAt
  rw [hl]
  by_cases h : j < refK x
  · rw [ite_eq_left h, ite_eq_left h]
    exact S.enc_cutOf x hc wt t _ c j fun k hk => S.pad_take c j k hk
  · rw [ite_eq_right h, ite_eq_right h]
    obtain rfl : j = refK x := by omega
    rw [S.pad_ofFn c]

noncomputable def unPreModel (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (ν : S.Nonce) :
    (S.unPS vs).TPreModel x where
  J := S.J x
  C := S.C x
  cz a := S.cz x (S.coinsOf a)
  out a σ i := S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1) σ)) (ansHist a i)
  wt := S.wt x
  wt₀ := S.wt₀ x
  eR := S.eP x
  kR := S.kP x

theorem unPreModel_link (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (x : Msg)
    (hx : A x) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce) : (S.unPreModel vs H x ν).UnLink H ν w := by
  refine ⟨hV x w hR, fun a j => hD x (S.coinsOf a) j, ?_, fun p => (hT x hx).1 p _,
    ⟨fun a => vs x w ν (S.coinsOf a), ?_, fun t a => funext fun i => S.unPre_msg vs H x w ν huc t a i⟩,
    fun t a => funext fun i => S.offPre_msg H x ν hc (S.wt₀ x) t a i⟩
  · intro wt ω p a a' i h
    show S.refP.pmG H x ν _ (ansHist a i) = S.refP.pmG H x ν _ (ansHist a' i)
    rw [ansHist_congr a a' i h]
    refine congrArg (fun g => S.refP.pmG H x ν g (ansHist a' i)) (funext fun _ => congrArg OC.ret ?_)
    exact hc x wt ω p (S.coinsOf a) (S.coinsOf a') (i.val - 1) fun k hk =>
      congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))
  · intro ω s p a a' i h
    show S.refP.pmG H x ν _ (ansHist a i) = S.refP.pmG H x ν _ (ansHist a' i)
    rw [ansHist_congr a a' i h]
    refine congrArg (fun g => S.refP.pmG H x ν g (ansHist a' i)) (funext fun _ => congrArg OC.ret ?_)
    exact huc x w ν ω s p (S.coinsOf a) (S.coinsOf a') (i.val - 1) fun k hk =>
      congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))

noncomputable def unOffModel (S : RefSession) (vs : S.SwVals) (x : Msg) (c : Fin (refK x) → S.Coin) :
    (S.unPS vs).TOffModel x where
  J := S.J x
  C := S.C x
  cz := S.cz x c
  out σ := (List.range (refK x + 1)).map fun j => if j < refK x then S.enc x c j σ else S.off x c σ
  wt := S.wt x
  wt₀ := S.wt₀ x
  eR := S.eP x
  kR := S.kP x

theorem unOffModel_link (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (x : Msg)
    (hx : A x) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce) (c : Fin (refK x) → S.Coin) :
    (S.unOffModel vs x c).UnLink H ν c w := by
  refine ⟨hV x w hR, fun j => hD x c j, fun p => (hT x hx).1 p _, ⟨vs x w ν c, fun (t : S.Tape) => ?_⟩,
    fun (t : S.Tape) => ?_⟩
  · show (List.range (refK x + 1)).map (fun j => S.unMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j)) =
      (List.range (refK x + 1)).map fun j =>
        if j < refK x then S.enc x c j (S.unCutOf x c (S.wt x w) (vs x w ν c) t)
        else S.off x c (S.unCutOf x c (S.wt x w) (vs x w ν c) t)
    exact List.map_congr_left fun j hj => S.unMsgAt_take vs x w ν huc t c j (List.mem_range.1 hj)
  · show (List.range (refK x + 1)).map (fun j => S.offMsgAt x (S.wt₀ x) t ((List.ofFn c).take j)) =
      (List.range (refK x + 1)).map fun j =>
        if j < refK x then S.enc x c j (S.cutOf x c (S.wt₀ x) t) else S.off x c (S.cutOf x c (S.wt₀ x) t)
    exact List.map_congr_left fun j hj => S.offMsgAt_take x hc _ t c j (List.mem_range.1 hj)

theorem unPS_prefixHides (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x)
    (hS : S.Tables) : (S.unPS vs).PrefixHides H R fun x => 2 * S.nLeaves x * δ₁.toReal :=
  (S.unPS vs).prefixHides_of_runs H R _ fun x w hR ν V E =>
    (S.unPreModel vs H x ν).runs_close_un H hT1 hδ₁ ν w
      (S.unPreModel_link vs H R A hV hD hc huc hT x (hRA x w hR) w hR ν) (hS x) V E

theorem unPS_shvzk (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs) (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x)
    (hS : S.Tables) : (S.unPS vs).SHVZK H R fun x => 2 * S.nLeaves x * δ₁.toReal := fun x w hR ν c E =>
  (S.unOffModel vs x c).close_un H hT1 hδ₁ ν c w
    (S.unOffModel_link vs H R A hV hD hc huc hT x (hRA x w hR) w hR ν c) (hS x) E

theorem unP_zk (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop) (t : ℕ)
    (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (hS : S.Tables) (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (huc : S.UnCausal vs)
    (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x) :
    ZKmalR H (fun x => refK x + 2) (S.unP vs).prover R (S.offP.simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
  have h : ZKmalR H (fun x => refK x + 2) (S.unP vs).prover R (S.offP.simulatorE t (t * 3 ^ t))
      (fun x => 4 * (2 * S.nLeaves x * δ₁.toReal) + 4 * t * (2 * S.nLeaves x * δ₁.toReal) + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
    have h0 := (S.unPS vs).gkE_zk_pow H R t ht _ _
      (S.unPS_prefixHides vs H R A hT1 hδ₁ hV hD hc huc hT hRA hS)
      (S.unPS_shvzk vs H R A hT1 hδ₁ hV hD hc huc hT hRA hS) (fun x => by positivity) (S.binding H)
    rw [S.unPS_simulatorE vs] at h0
    exact h0
  intro x w hR Θ Vs θ hk Ev
  refine (h x w hR Θ Vs θ hk Ev).trans (le_of_eq ?_)
  beta_reduce
  ring

theorem withStops_unP_zk (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (hRf : S.StopsCausal Rf) (huc : S.UnCausal vs) (hT : S.TapeOKOn A)
    (hRA : ∀ x w, R x w → A x) :
    ZKmalR H (fun x => refK x + 2) ((S.withStops Rf stop).unP vs).prover R
      ((S.withStops Rf stop).offP.simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  (S.withStops Rf stop).unP_zk vs H R A t ht hT1 hδ₁ (S.withStops_tables Rf stop hS) (S.withStops_complete Rf stop hV)
    (S.withStops_dummyReg Rf stop hD) (S.withStops_causal Rf stop hRf hc) (S.withStops_unCausal Rf stop vs hRf huc) hT
    hRA

end RefSession

end Claims.ZKMal
