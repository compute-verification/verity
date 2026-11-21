module

public import Proofs.Flock.Bridge.V5.KSLive
public import Proofs.Flock.Bridge.V5.SubsetStatement
public import Proofs.Flock.Bridge.Count

/-!
# `SampledKSLiveV5Subset`, proved

`Flock.Bridge.sampledKSLiveV5On : Flock.Bridge.SampledKSLiveV5On`: V5's assembly (`assemblyTargetV5`'s argument) at the
registration's strata summing to `n` and the sampling design on them, with canonical transport at public inputs
(`transportsLiveV5`), M1's core (`faultyRightTargetC`) and `EndToEndHiddenV5` (`endToEndHiddenV5_of`), none of which reads
the population. `Flock.SecurityProofs.SampledKSLiveV5Subset`: that bound at `PopCovers` under a uniform-subset `designAt`,
whose strata sum to `n` (`nsOf_sum_of_popCovers`) and which is on them (`samplingDesignOn_samplesUnder_cover`), with
`ecr/sha-512`'s term at the roots finder in place of the guard, as `SampledKSLiveV5`'s proof has it.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **V5's bound under the roots guard, at its strata's facts** (`Flock.Bridge.SampledKSLiveV5On`). -/
theorem _root_.Flock.Bridge.sampledKSLiveV5On : Bridge.SampledKSLiveV5On := by
  have hT : Bridge.TransportsLiveV5 := Bridge.transportsLiveV5
  have hF : Bridge.FaultyRightTargetC := Bridge.faultyRightTargetC
  have hE : Bridge.EndToEndHiddenV5 := Bridge.endToEndHiddenV5_of Bridge.openingsCanonV5
  intro n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup hroots desc qr lj tb samplingDesign kd hlaw hb hns₀ hon₀
    hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' B τ qF qS q L P vb σ Xr Cc Lm
    rows roots Cd κ r ns δ hTab hLink
  classical
  have hns : ns.sum = n := hns₀
  have hon : Bridge.SamplingDesignOn samplingDesign ns kd := hon₀
  by_cases hrel : Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ r desc q roots)
      (Bridge.partOf P κ ns) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr)
  · exact (FlockSoundness.Game.prob_mono (q := fun _ => False) (fun o ho => ho.2 hrel) _ τ).trans
      ((FlockSoundness.Game.prob_false _ τ).le.trans zero_le)
  obtain ⟨F, hFe⟩ : ∃ F, Claims.Sampling.faulty Claims.Sampling.sha512Msg Cd (Bridge.stmtOf P κ r desc q roots)
      (Bridge.partOf P κ ns) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr) = F := ⟨_, rfl⟩
  have hle : (Bridge.designOf r).Pa (Bridge.partOf P κ ns) F ≤ δ := by
    rw [← hFe]
    exact not_lt.1 hrel
  have hW : (Finset.univ.filter fun u : Fin n => u.val ∈ F).map Fin.valEmbedding = F.filter (· < n) := by
    ext v
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Fin.valEmbedding_apply]
    constructor
    · rintro ⟨u, hu, rfl⟩; exact ⟨hu, u.isLt⟩
    · rintro ⟨hv, hlt⟩; exact ⟨⟨v, hlt⟩, hv, rfl⟩
  rw [Bridge.pa_partOf_window P hns, Bridge.window_zero_filter, ← hW] at hle
  have hTu := hT n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup hroots samplingDesign kd hdiv dj S₀ d₀ y₀ wr dg N L
    hTags hCust (Bridge.samplingDesignSized_size (Bridge.samplingDesignSized_samplesUnder hlaw hb) hCust.2) τ
  have hκin : ∀ g b, κ g = some b → (Cc.op g).isInput = true := fun g b h =>
    Bridge.constsOfP_isInput (Discharge.Layout.outsOfP D.unroll hU n pi) h
  have hsub : (Finset.univ.filter fun u : Fin n => u.val ∈ F) ⊆ incorrectRegZc execArith P (planZHJP dj y₀)
      (tabZHJP dj y₀) vb (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2) (reg σ) Xr := by
    intro u hu
    have hv : u.val ∈ Claims.Sampling.faulty Claims.Sampling.sha512Msg Cd (Bridge.stmtOf P κ r desc q roots)
        (Bridge.partOf P κ ns) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr) := by
      rw [hFe]
      exact (Finset.mem_filter.1 hu).2
    by_contra hu'
    obtain ⟨-, op, hop⟩ := not_not.1 (mt (mem_incorrectRegZc _ _ _ _ _ _).2 hu')
    refine hF P vb.pos (fun o g => vb.bit (vb.val o) g) id vb.com (vb.cs (reg σ))
      (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2)
      Claims.Sampling.sha512Msg Cd roots Xr κ u hκin ⟨op, hop⟩
      (fun g hg hn o ho hc => hTu u g hg hn o ho (hc.2 hn)) ?_ ns r desc q hns hv
    intro g hg b hb
    obtain ⟨-, hbit, hcan⟩ := hop g hg
    exact hbit.symm.trans (hcan.1 b hb)
  have hE' := hE n Reg I own ht D hD hU pi hpi pub hpub hscope samplingDesign kd ns hns hon hdiv dj S₀ d₀ y₀ wr dg N Ω Lb
    src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' (Finset.univ.filter fun u : Fin n => u.val ∈ F) τ qF qS hTab hLink
  refine le_trans (FlockSoundness.Game.prob_mono ?_ _ τ) (hE'.trans ?_)
  · exact fun o ho => ⟨ho.1, hsub⟩
  · exact add_le_add (add_le_add (ENNReal.ofReal_le_ofReal hle) le_rfl) le_rfl

end Flock.Guarantees

/-- **The sampled audit's knowledge soundness, live, at the verifier's public inputs, under the uniform subset at a
population the strata cover** (`Flock.Guarantees.SampledKSLiveV5Subset`). -/
theorem Flock.SecurityProofs.SampledKSLiveV5Subset : Flock.Guarantees.SampledKSLiveV5Subset := by
  intro n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup desc qr hpop lj tb hsub samplingDesign kd hdiv dj S₀ d₀ y₀
    wr dg N Ω _ _ Lb src hA3 hCust k Rw M hk hRw hM ρ hρ hr t' ht' B τ qF qS q
  split
  swap
  · next hTags =>
    refine nonpos_iff_eq_zero.1 ((FlockSoundness.Game.prob_mono (q := fun _ => False) (fun o ho => ?_) _ τ).trans
      (FlockSoundness.Game.prob_false _ τ).le)
    obtain ⟨-, -, _, _, hk⟩ := ho
    exact hTags (Flock.Bridge.tagsOkZHJ_of_ksScope hk)
  rename_i hTags
  intro L P vb σ Xr Cc Lm rows roots Cd κ r ns δ hTab hLink
  have hns := Flock.Bridge.nsOf_sum_of_popCovers hpop
  rcases Flock.Bridge.designAt_cases desc qr lj tb n with hlaw | h0
  · exact Flock.Bridge.le_add_ecrBoundL (FlockSoundness.Game.prob_le_one _ _ τ) fun hroots =>
      Flock.Bridge.sampledKSLiveV5On n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup hroots desc qr lj tb
        samplingDesign kd hlaw (Flock.Bridge.notBernoulli_samplesUnder hlaw) hns
        (Flock.Bridge.samplingDesignOn_samplesUnder_cover hpop hlaw hsub) hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags
        hCust k Rw M hk hRw hM ρ hρ hr t' ht' B τ qF qS q hTab hLink
  · have hδ : δ = 1 := by
      change (Flock.Bridge.designOf (Properties.Spot.oneStageOf (Flock.Bridge.designAt desc qr lj tb n))).Pa
        (Flock.Bridge.partOf P κ ns) B = 1
      rw [h0]
      change (Flock.Bridge.designOf (.subset 0)).Pa (Flock.Bridge.partOf P κ ns) B = 1
      rw [Flock.Bridge.pa_partOf_subset _ hns]
      simp
    refine (FlockSoundness.Game.prob_le_one _ _ τ).trans ?_
    rw [hδ, ENNReal.ofReal_one]
    exact le_add_right (le_add_right (le_add_right le_rfl))
