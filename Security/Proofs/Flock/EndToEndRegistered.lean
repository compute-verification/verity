module

public import Proofs.Flock.EndToEndRegistered.Statement

/-!
# Proof of Flock's end-to-end guarantee with registered reads

`Flock.SecurityProofs.EndToEndRegistered : Flock.Guarantees.EndToEndRegistered` is `zk_session_soundHJR_custody_regSt`
with its first conjunct's first and last terms in closed form, as `EndToEnd` is `zk_session_soundJ_custody_reg`'s
(`Audit.Law.subset_miss`, and `linkBoundZC` at A2); its second conjunct is unchanged.
`Flock.SecurityProofs.EndToEndRegisteredDrawn : Flock.Guarantees.EndToEndRegisteredDrawn` is
`zk_session_soundHJR_custody_sampled_regSt` with its first conjunct's last term in closed form. Both read the circuit the
descriptor unrolls to, a flat file's or a templated statement's (`Flock.Desc.parseStmt_stmt`), and each flat form is
`zk_session_soundHJR_custody_reg` or `zk_session_soundHJR_custody_drawn_reg` the same way, at the circuit
`HmOut.parse` reads.
-/

@[expose] public section

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- **Flock's one-stage audit, end to end, with registered reads.** -/
theorem EndToEndRegistered : Flock.Guarantees.EndToEndRegistered := by
  intro n Reg I ht own D hD hU pub hpub hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk
    hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P hcS hv1 vb σ hCR hA2
  have h := Discharge.ZkReg.zk_session_soundHJR_custody_regSt I ht own hcS hv1 hU hpub hscope kd hkd hdiv dj y₀ wr dg
    N Lb src hA3 hTags hCust hk hRw hM hρ hr ht' K₀ τ (qF := qF) (qS := qS) hCR
  rw [Law.subset_miss hkd hK₀, Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

/-- **Flock's one-stage audit, end to end, with registered reads, at the sampled proof blocks.** -/
theorem EndToEndRegisteredDrawn : Flock.Guarantees.EndToEndRegisteredDrawn := by
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
  rw [Law.subset_miss hkd hK₀, Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

/-- **`EndToEndRegisteredDrawn` at a flat circuit file.** -/
theorem EndToEndRegisteredDrawnFlat : Flock.Guarantees.EndToEndRegisteredDrawnFlat := by
  intro n Reg I ht own c hc hU pub hpub hscope kd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags hCust k Rw M hk hRw hM ρ
    hρ hr t' ht' τ qF qS L P vb σ hCR hA2
  have h := Discharge.ZkReg.zk_session_soundHJR_custody_drawn_reg I ht own hc hU hpub hscope kd hdiv dj y₀ wr dg N Lb
    src hTags hCust hk hRw hM hρ hr ht' τ (qF := qF) (qS := qS) hCR
  rw [Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

end Flock.SecurityProofs
