module

public import Proofs.Flock.Bridge.KSLive
public import Proofs.Flock.Bridge.V5.Assembly
public import Proofs.Flock.Bridge.V5.EndToEndHidden
public import Proofs.Flock.Bridge.V5.OpeningsCanon
public import Proofs.Flock.Bridge.V5.Statement
public import Proofs.Flock.Bridge.V5.Transports
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Bridge.Count

/-!
# `SampledKSLiveV5`, proved

`Flock.Bridge.sampledKSLiveV5R : Flock.Bridge.SampledKSLiveV5R`: `AssemblyTargetV5` (`assemblyTargetV5`) with
canonical transport at public inputs (`transportsLiveV5`), M1's core (`faultyRightTargetC`) and `EndToEndHiddenV5`
(`endToEndHiddenV5_of`, from `openingsCanonV5`) plugged in. `Flock.SecurityProofs.SampledKSLiveV5 :
Flock.Guarantees.SampledKSLiveV5`: that bound under the roots guard, with `ecr/sha-512`'s term at the roots finder in place
of the guard (`Bridge.le_add_ecrBoundL`), at the sampling design `Bridge.designAt` names, which the verifier samples under
(`Bridge.designAt_cases`) or is `subset 0`, whose `δ` is 1 (`Bridge.pa_partOf_subset`); at tags outside `TagsOkZHJ`,
probability 0, since passing `Flock.KsScope.checkOf` gives them (`Bridge.tagsOkZHJ_of_ksScope`).
-/

@[expose] public section

/-- **V5's bound under the roots guard** (`Flock.Bridge.SampledKSLiveV5R`). -/
theorem Flock.Bridge.sampledKSLiveV5R : Flock.Bridge.SampledKSLiveV5R :=
  Flock.Bridge.assemblyTargetV5 Flock.Bridge.transportsLiveV5 Flock.Bridge.faultyRightTargetC
    (Flock.Bridge.endToEndHiddenV5_of Flock.Bridge.openingsCanonV5)

/-- **`verify --zk`'s form check passes only at hidden-output tags**: `Flock.KsScope.checkOf` refuses first at tags that
`tagsWhy` names, whatever circuit it is given, so passing it gives `TagsOkZHJ`. -/
theorem Flock.Bridge.tagsOkZHJ_of_ksScope {I : FlockSoundness.Discharge.Exec.Inputs} {parsed : Flock.V Flock.Circuit}
    {count : ℕ} {draw : Option Lean.Json}
    (h : Flock.KsScope.checkOf parsed (FlockSoundness.Discharge.ZkExec.zkTags I) I.publicFile count draw = .ok ()) :
    FlockSoundness.Discharge.ZkHidden.TagsOkZHJ I := by
  have ht : Flock.KsScope.tagsWhy (FlockSoundness.Discharge.ZkExec.zkTags I) = none := by
    unfold Flock.KsScope.checkOf at h
    cases hw : Flock.KsScope.tagsWhy (FlockSoundness.Discharge.ZkExec.zkTags I) with
    | some w => simp [hw] at h
    | none => rfl
  obtain ⟨-, h2, h3, h4⟩ := Flock.KsScope.tagsWhy_none ht
  exact ⟨(FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs I).symm.trans h2,
    (FlockSoundness.Discharge.ZkExec.zkTags_retainRounds I).symm.trans h3,
    (FlockSoundness.Discharge.ZkExec.zkTags_merkleLeaf I).symm.trans h4⟩

/-- **The sampled audit's knowledge soundness, live, at the verifier's public inputs, with or without shared rows**
(`Flock.Guarantees.SampledKSLiveV5`). -/
theorem Flock.SecurityProofs.SampledKSLiveV5 : Flock.Guarantees.SampledKSLiveV5 := by
  intro n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup desc qr hpop lj tb law kd hdiv dj S₀ d₀ y₀ wr dg
    N Ω _ _ Lb src hA3 hCust k Rw M hk hRw hM ρ hρ hr t' ht' B τ qF qS q
  split
  swap
  · next hTags =>
    refine nonpos_iff_eq_zero.1 ((FlockSoundness.Game.prob_mono (q := fun _ => False) (fun o ho => ?_) _ τ).trans
      (FlockSoundness.Game.prob_false _ τ).le)
    obtain ⟨-, -, _, _, hk⟩ := ho
    exact hTags (Flock.Bridge.tagsOkZHJ_of_ksScope hk)
  rename_i hTags
  intro L P vb σ Xr Cc Lm rows roots Cd κ r ns δ hTab hLink
  rcases Flock.Bridge.designAt_cases desc qr lj tb n with hlaw | h0
  · exact Flock.Bridge.le_add_ecrBoundL (FlockSoundness.Game.prob_le_one _ _ τ) fun hroots =>
      Flock.Bridge.sampledKSLiveV5R n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup hroots desc qr hpop lj tb law
        kd hlaw (Flock.Bridge.notBernoulli_drawsUnder hlaw) hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust k Rw M hk
        hRw hM ρ hρ hr t' ht' B τ qF qS q hTab hLink
  · have hδ : δ = 1 := by
      change (Flock.Bridge.designOf (Properties.Spot.oneStageOf (Flock.Bridge.designAt desc qr lj tb n))).Pa
        (Flock.Bridge.partOf P κ ns) B = 1
      rw [h0]
      change (Flock.Bridge.designOf (.subset 0)).Pa (Flock.Bridge.partOf P κ ns) B = 1
      rw [Flock.Bridge.pa_partOf_subset _ (Flock.Bridge.nsOf_sum hpop)]
      simp
    refine (FlockSoundness.Game.prob_le_one _ _ τ).trans ?_
    rw [hδ, ENNReal.ofReal_one]
    exact le_add_right (le_add_right (le_add_right le_rfl))
