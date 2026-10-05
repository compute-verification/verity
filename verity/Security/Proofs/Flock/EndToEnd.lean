import Specs.Flock.Guarantees.EndToEnd

/-!
# Proof of C-Flock's end-to-end guarantee

`Flock.SecurityProofs.EndToEnd : Flock.Guarantees.EndToEnd` is `zk_session_soundJ_custody` with its first and last terms
in closed form:
* `zk_session_soundJ_custody`: one `--zk` session of `J` tables at the record's custody, at the live subset draw, is
  sound, `Pr[flock-verify accepts ∧ at least K₀ wrong units] ≤ miss(K₀) + ε_ks + δ_link`;
* `Audit.Law.subset_miss`: the uniform `kd`-subset's `miss(K₀)` is `C(n − K₀, kd) / C(n, kd)` at `K₀ ≤ n`;
* `linkBoundZC`: at a prover whose link finders satisfy A2 (`LinkCRZC`), `δ_link` is its closed form.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- **C-Flock's one-stage audit, end to end.** -/
theorem EndToEnd : Flock.Guarantees.EndToEnd := by
  intro n Reg I ht c hc hU pub hpub hpt hscope kd hkd hdiv dj S₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk hRw
    hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P vb σ hCR hA2
  have h := Discharge.Composed.zk_session_soundJ_custody I ht hc hU hpub hpt hscope kd hkd hdiv dj y₀ wr dg N Lb src hA3
    hTags hCust hk hRw hM hρ hr ht' K₀ τ (qF := qF) (qS := qS) hCR
  rw [Law.subset_miss hkd hK₀, Discharge.ZkLink.linkBoundZC, if_pos hA2] at h
  exact h

end Flock.SecurityProofs
