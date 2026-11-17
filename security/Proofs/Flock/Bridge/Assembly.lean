module

public import Proofs.Flock.Bridge.Canon

/-! `Bridge.AssemblyTargetC`, proved: the assembly with canonical openings that concludes `SampledKSLiveV2`. -/

@[expose] public section

namespace Flock.Bridge

theorem choose_min_count {n kd K₀ : ℕ} (hkd1 : 1 ≤ kd) (hkd : kd ≤ n) :
    (n - min K₀ (n - kd + 1)).choose kd = (n - K₀).choose kd := by
  rcases le_total K₀ (n - kd + 1) with h | h
  · rw [min_eq_left h]
  · rw [min_eq_right h, Nat.choose_eq_zero_of_lt (by omega), Nat.choose_eq_zero_of_lt (by omega)]

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

theorem _root_.Flock.Bridge.constsOf_isInput {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}
    (outs : Finset (Fin (Discharge.Layout.keyN c hU n))) {g : Fin (Discharge.Layout.keyN c hU n)} {b : Bool}
    (h : Bridge.constsOf c hU n g = some b) : (((keyProg c hU n).circuit outs).op g).isInput = true := by
  have hg : g.val < 2 + n * Discharge.Layout.keyB c hU := by
    simp only [Bridge.constsOf] at h
    split_ifs at h with h1 h2
    · subst h1; show 0 < 2 + n * Discharge.Layout.keyB c hU; omega
    · rcases h2 with h2 | h2
      · subst h2; show 1 < 2 + n * Discharge.Layout.keyB c hU; omega
      · exact h2.1
  exact (((keyProg c hU n).partition outs).input_iff g).1
    (Discharge.Layout.unit_uniform_input (k := Discharge.Placed.inGates n (Discharge.Layout.keyB c hU))
      (R := Discharge.Placed.circuitRows c hU)
      (ins := Discharge.Placed.insOf n (Discharge.Layout.keyB c hU)
        (Discharge.Aliased.keyIdx c (Discharge.Layout.keyB c hU)))
      (one := Discharge.Placed.oneGate n (Discharge.Layout.keyB c hU))
      (hone := Discharge.Placed.insOf_ne_one n (Discharge.Layout.keyB c hU)
        (Discharge.Aliased.keyIdx c (Discharge.Layout.keyB c hU)))
      n g hg)

theorem _root_.Flock.Bridge.assemblyTargetC : Bridge.AssemblyTargetC := by
  intro hT hF _ hC hE n Reg I ht D hD hU pub hpub hpt hscope hv2 hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg
    N Ω _ _ Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ tv htv q L P vb σ Xr δ Cc rows roots Cd κ
    hTab hLink
  by_cases hk0 : kd = 0
  · refine (FlockSoundness.Game.prob_le_one _ _ τ).trans (le_trans (le_of_eq ?_) (le_self_add.trans le_self_add))
    rw [hk0]
    simp
  have hkd1 : 1 ≤ kd := Nat.pos_of_ne_zero hk0
  have hTu := hT n Reg I ht D hD hU pub hpub hpt hscope hv2 hdup (Bridge.PortRoots.fixes_of_none hroots) kd hkd hdiv
    dj S₀ d₀ y₀ wr dg N Ω Lb src hTags
    hCust τ
  have hκin : ∀ g b, κ g = some b → (Cc.op g).isInput = true := fun g b h =>
    Bridge.constsOf_isInput (Discharge.Layout.outsOf D.unroll hU n) h
  have hns := Bridge.nsOf_sum hpop
  have hsub : (Claims.Sampling.faulty Claims.Sampling.sha512Msg Cd (Bridge.stmtOf P κ (.subset kd) desc q roots)
      (Bridge.partOf P κ (Bridge.nsOf desc qr)) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr)).filter
        (· < n) ⊆
      (incorrectRegZc execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb
        (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2) (reg σ) Xr).map
        Fin.valEmbedding := by
    classical
    intro v hv
    obtain ⟨hv, hlt⟩ := Finset.mem_filter.1 hv
    refine Finset.mem_map.2 ⟨⟨v, hlt⟩, ?_, rfl⟩
    by_contra hu
    obtain ⟨-, op, hop⟩ := not_not.1 (mt (mem_incorrectRegZc _ _ _ _ _ _).2 hu)
    refine hF P vb.pos (fun o g => vb.bit (vb.val o) g) id vb.com (vb.cs (reg σ))
      (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2)
      Claims.Sampling.sha512Msg Cd roots Xr κ ⟨v, hlt⟩ hκin ⟨op, hop⟩
      (fun g hg hn o ho hc => hTu ⟨v, hlt⟩ g hg hn o ho (hc.2 hn)) ?_ _ (.subset kd) desc q hns hv
    intro g hg b hb
    obtain ⟨-, hbit, hcan⟩ := hop g hg
    exact hbit.symm.trans (hcan.1 b hb)
  have hcard : ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ (.subset kd) desc q roots)
      (Bridge.partOf P κ (Bridge.nsOf desc qr)) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr) →
      min K₀ (n - kd + 1) ≤ (incorrectRegZc execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb
        (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2) (reg σ) Xr).card := fun hrel =>
    (hC P _ hns kd K₀ hkd1 hkd _ _ κ desc q roots _ hrel).trans
      ((Finset.card_le_card hsub).trans (Finset.card_map _).le)
  have hE' := hE n Reg I ht D hD hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust
    k Rw M hk hRw hM ρ hρ hr (t' + tv) (add_nonneg ht' htv) (min K₀ (n - kd + 1)) ((min_le_left _ _).trans hK₀) τ
    (2 * (t' + tv)) (t' + tv) hTab hLink
  rw [Bridge.choose_min_count hkd1 hkd] at hE'
  refine le_trans (FlockSoundness.Game.prob_mono ?_ _ τ) hE'
  exact fun o ho => ⟨ho.1, hcard ho.2⟩

end Flock.Guarantees
