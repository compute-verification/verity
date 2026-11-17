module

public import Proofs.Flock.EndToEndRegistered.Statement

/-! The proofs of `EndToEndRegistered` and `EndToEndRegisteredSampled`, Flock's end-to-end guarantee with registered
reads. -/

@[expose] public section

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- **Flock's one-stage audit, end to end, with registered reads.** -/
theorem EndToEndRegistered : Flock.Guarantees.EndToEndRegistered := by
  intro n Reg I ht own D hD hU pub hpub hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk
    hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P hcS hv1 vb σ hCR hA2
  have h := Discharge.ZkReg.zk_session_soundHJR_custody_regSt I ht own hcS hv1 hU hpub hscope kd hkd hdiv dj y₀ wr dg
    N Lb src hA3 hTags hCust hk hRw hM hρ hr ht' K₀ τ (qF := qF) (qS := qS) hCR
  rw [SamplingDesign.subset_miss hkd hK₀, Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

/-- **Flock's one-stage audit, end to end, with registered reads, at the sampled proof blocks.** -/
theorem EndToEndRegisteredSampled : Flock.Guarantees.EndToEndRegisteredSampled := by
  intro n Reg I ht own D hD hU pub hpub hscope kd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags hCust k Rw M hk hRw hM ρ
    hρ hr t' ht' τ qF qS L P hcS hv1 vb σ hCR hA2
  have h := Discharge.ZkReg.zk_session_soundHJR_custody_sampled_regSt I ht own hcS hv1 hU hpub hscope kd hdiv dj y₀ wr
    dg N Lb src hTags hCust hk hRw hM hρ hr ht' τ (qF := qF) (qS := qS) hCR
  rw [Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

/-- **`EndToEndRegistered` at a flat circuit file.** -/
theorem EndToEndRegisteredFlat : Flock.Guarantees.EndToEndRegisteredFlat := by
  intro n Reg I ht own c hc hU pub hpub hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk
    hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P vb σ hCR hA2
  have h := Discharge.ZkReg.zk_session_soundHJR_custody_reg I ht own hc hU hpub hscope kd hkd hdiv dj y₀ wr dg N Lb
    src hA3 hTags hCust hk hRw hM hρ hr ht' K₀ τ (qF := qF) (qS := qS) hCR
  rw [SamplingDesign.subset_miss hkd hK₀, Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

/-- **`EndToEndRegisteredSampled` at a flat circuit file.** -/
theorem EndToEndRegisteredSampledFlat : Flock.Guarantees.EndToEndRegisteredSampledFlat := by
  intro n Reg I ht own c hc hU pub hpub hscope kd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags hCust k Rw M hk hRw hM ρ
    hρ hr t' ht' τ qF qS L P vb σ hCR hA2
  have h := Discharge.ZkReg.zk_session_soundHJR_custody_sampled_reg I ht own hc hU hpub hscope kd hdiv dj y₀ wr dg N Lb
    src hTags hCust hk hRw hM hρ hr ht' τ (qF := qF) (qS := qS) hCR
  rw [Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

end Flock.SecurityProofs
