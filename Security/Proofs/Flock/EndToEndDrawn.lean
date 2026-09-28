import Proofs.Flock.EndToEndDrawn.Statement

/-!
# Proof of C-Flock's end-to-end guarantee at the drawn units

`Flock.SecurityProofs.EndToEndDrawn : Flock.Guarantees.EndToEndDrawn` is `zk_session_soundJ_custody_drawn_reg` with its
last term in closed form:
* `zk_session_soundJ_custody_drawn_reg`: one `--zk` session of `J` tables at the record's custody is sound at the drawn
  units, `Pr[flock-verify accepts ∧ some drawn unit wrong on the public file] ≤ ε_ks + δ_link`;
* `linkBoundZC`: at a prover whose link finders satisfy A2 (`LinkCRZC`), `δ_link` is its closed form.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- **C-Flock's one-stage audit, end to end, at the drawn units.** -/
theorem EndToEndDrawn : Flock.Guarantees.EndToEndDrawn := by
  intro n Reg I ht c hc hU pub hpub hpt hscope kd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags hCust k Rw M hk hRw hM ρ
    hρ hr t' ht' τ qF qS L P vb σ hCR hA2
  have h := Discharge.Composed.zk_session_soundJ_custody_drawn_reg I ht hc hU hpub hpt hscope kd hdiv dj y₀ wr dg N Lb
    src hTags hCust hk hRw hM hρ hr ht' τ (qF := qF) (qS := qS) hCR
  rw [Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

end Flock.SecurityProofs
