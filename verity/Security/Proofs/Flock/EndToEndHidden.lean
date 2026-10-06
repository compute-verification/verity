import Specs.Flock.Guarantees.EndToEndHidden

/-!
# Proof of C-Flock's end-to-end guarantee under hidden outputs

`Flock.SecurityProofs.EndToEndHidden : Flock.Guarantees.EndToEndHidden` is `zk_session_soundHJ_custody_reg` with its
first and last terms in closed form, as `EndToEnd` is `zk_session_soundJ_custody_reg`'s (`Audit.Law.subset_miss`, and
`linkBoundZC` at A2). `Flock.SecurityProofs.EndToEndHiddenDrawn : Flock.Guarantees.EndToEndHiddenDrawn` is
`zk_session_soundHJ_custody_drawn_reg` with its last term in closed form.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- **C-Flock's one-stage audit, end to end, under hidden outputs.** -/
theorem EndToEndHidden : Flock.Guarantees.EndToEndHidden := by
  intro n Reg I ht c hc hU pub hpub hpt hscope kd hkd hdiv dj S₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk hRw
    hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P vb σ hCR hA2
  have h := Discharge.ZkHidden.zk_session_soundHJ_custody_reg I ht hc hU hpub hpt hscope kd hkd hdiv dj y₀ wr dg N Lb
    src hA3 hTags hCust hk hRw hM hρ hr ht' K₀ τ (qF := qF) (qS := qS) hCR
  rw [Law.subset_miss hkd hK₀, Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

/-- **C-Flock's one-stage audit, end to end, under hidden outputs, at the drawn units.** -/
theorem EndToEndHiddenDrawn : Flock.Guarantees.EndToEndHiddenDrawn := by
  intro n Reg I ht c hc hU pub hpub hpt hscope kd hdiv dj S₀ y₀ wr dg N Ω _ _ Lb src hTags hCust k Rw M hk hRw hM ρ hρ
    hr t' ht' τ qF qS L P vb σ hCR hA2
  have h := Discharge.ZkHidden.zk_session_soundHJ_custody_drawn_reg I ht hc hU hpub hpt hscope kd hdiv dj y₀ wr dg N
    Lb src hTags hCust hk hRw hM hρ hr ht' τ (qF := qF) (qS := qS) hCR
  rw [Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

end Flock.SecurityProofs
