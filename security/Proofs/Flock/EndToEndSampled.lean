module

public import Proofs.Flock.EndToEndSampled.Statement
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
import Proofs.Flock.Soundness.Discharge.Composed.SampledJ

/-!
# Proof of Flock's end-to-end guarantee at the sampled proof blocks

`Flock.SecurityProofs.EndToEndSampled : Flock.Guarantees.EndToEndSampled` is `zk_session_soundJ_custody_sampled_reg` with its
last term in closed form:
* `zk_session_soundJ_custody_sampled_reg`: one `--zk` session of `J` tables at the record's custody is sound at the sampled
  proof blocks, `Pr[verity-verify-proof accepts ∧ some drawn unit wrong on the public file] ≤ ε_ks + ε_bind`;
* `linkBoundZC`: at a prover whose link finders satisfy A2 (`LinkCRZC`), `ε_bind` is its closed form.
-/

@[expose] public section

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- **Flock's one-stage audit, end to end, at the sampled proof blocks.** -/
theorem EndToEndSampled : Flock.Guarantees.EndToEndSampled := by
  intro n Reg I ht c hc hU pub hpub hpt hscope kd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags hCust k Rw M hk hRw hM ρ
    hρ hr t' ht' τ qF qS L P vb σ hCR hA2
  have h := Discharge.Composed.zk_session_soundJ_custody_sampled_reg I ht hc hU hpub hpt hscope kd hdiv dj y₀ wr dg N Lb
    src hTags hCust hk hRw hM hρ hr ht' τ (qF := qF) (qS := qS) hCR
  rw [Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

end Flock.SecurityProofs
