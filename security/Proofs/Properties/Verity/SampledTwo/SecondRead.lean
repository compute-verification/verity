module

public import Proofs.Properties.Verity.SampledTwo.Steps
public import Proofs.Properties.Spot

/-!
# An accepted two-stage check's second-stage bytes are the coins its draw read

`Steps` reads the first stage's bytes against what the run read (`sourceRead_le`). The second stage's bytes `ξ₂` are
read by the interior audit's check (`Spot.OneStage.checkAt` at `some ξ₂`), which refuses a draw that leaves a coin
opened and not read (`Spot.OneStage.coinsReason`).

* `coinsReason_read`: a draw the check finds no fault in read `read` bytes of its coins, and the coins hold as many
  whole challenger coins as those bytes reach: `⌈read / COIN_BYTES⌉ = src.size / COIN_BYTES`.
* `checkAt_read`: the one-stage check at the coins `ξ` accepts only at such a draw, under the verifier's law as derived
  and the record's population.
* `second_read`: where the two-stage check accepts, the interior audit's draw is such a draw of `ξ₂`, the bytes
  `OneStageZk.coinsOf` read.

Nothing imports this module.
-/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)
open Spot

private theorem ite_ite_none {c₁ c₂ : Bool} {s₁ s₂ : String}
    (h : (if c₁ = true then some s₁ else if c₂ = true then some s₂ else none) = none) : c₂ = false := by
  cases c₁ <;> cases c₂ <;> simp at h ⊢

/-- **A draw the check finds no fault in reads every coin opened**: it read `read` bytes of `src`, and `src` holds as
many whole challenger coins as those bytes reach. -/
theorem coinsReason_read {v : Registration.Verifier} {law rec_ draw : Json} {src : ByteArray}
    (h : OneStage.coinsReason v law rec_ draw src = none) :
    ∃ n units read, nat? (getD rec_ "population") = some n ∧ OneStage.drawnFrom v law n src = .ok (units, read) ∧
      (read + Challenger.COIN_BYTES - 1) / Challenger.COIN_BYTES = src.size / Challenger.COIN_BYTES := by
  unfold OneStage.coinsReason at h
  cases hn : nat? (getD rec_ "population") with
  | none => simp [hn] at h
  | some n =>
    cases hdf : OneStage.drawnFrom v law n src with
    | error why => simp [hn, hdf] at h
    | ok ur =>
      obtain ⟨units, read⟩ := ur
      simp only [hn, hdf] at h
      exact ⟨n, units, read, rfl, hdf, by simpa using ite_ite_none h⟩

/-- **The one-stage check at the coins `ξ` accepts only where its draw read every coin of `ξ`**, under the verifier's
law as derived and the record's population. -/
theorem checkAt_read {p : OneStage.Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : OneStage.Evidence}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : OneStage.checkAt p c (some ξ) e = .accept pub prof) :
    ∃ law n units read,
      Draw.derive p.verifier.program p.verifier.query p.verifier.law p.verifier.workTable = .ok law ∧
      nat? (getD c.record "population") = some n ∧ OneStage.drawnFrom p.verifier law n ξ = .ok (units, read) ∧
      (read + Challenger.COIN_BYTES - 1) / Challenger.COIN_BYTES = ξ.size / Challenger.COIN_BYTES := by
  unfold OneStage.checkAt at h
  cases hw : OneStage.coinsWhy p c e ξ with
  | some why =>
    have hb : (some ξ).bind (OneStage.coinsWhy p c e) = some why := hw
    rw [hb] at h
    simp at h
  | none =>
    have hb : (some ξ).bind (OneStage.coinsWhy p c e) = none := hw
    rw [hb] at h
    have hck : OneStage.check p c e = .accept pub prof := h
    obtain ⟨law, _, _, hreg, -⟩ := _root_.Properties.Spot.acceptMeans _ _ _ _ _ hck
    have hd := _root_.Properties.Spot.registered_law hreg
    have hr : OneStage.coinsReason p.verifier law c.record e.draw ξ = none := by
      have h' := hw
      unfold OneStage.coinsWhy at h'
      simp only [hd] at h'
      exact h'
    obtain ⟨n, units, read, hn, hdf, hsz⟩ := coinsReason_read hr
    exact ⟨law, n, units, read, hd, hn, hdf, hsz⟩

/-- **An accepted two-stage check's second-stage bytes are the coins its draw read**: the interior audit's draw, under
its verifier's law as derived and its record's population, read `read` bytes of the bytes `ξ₂` its sample read, and
`ξ₂` holds as many whole challenger coins as those bytes reach. -/
theorem second_read {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ law n units read, OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂ ∧
      Draw.derive p.audit.base.verifier.program p.audit.base.verifier.query p.audit.base.verifier.law
        p.audit.base.verifier.workTable = .ok law ∧
      nat? (getD c₂.record "population") = some n ∧
      OneStage.drawnFrom p.audit.base.verifier law n ξ₂ = .ok (units, read) ∧
      (read + Challenger.COIN_BYTES - 1) / Challenger.COIN_BYTES = ξ₂.size / Challenger.COIN_BYTES := by
  obtain ⟨-, c₂, ξ₂, r, i, run, hi, -, -, -, -, htwo⟩ := accepted_steps h
  obtain ⟨-, hξ, -⟩ := inputAt_ok hi
  obtain ⟨law, n, units, read, hd, hn, hdf, hsz⟩ := checkAt_read htwo
  exact ⟨c₂, ξ₂, law, n, units, read, hξ, hd, hn, hdf, hsz⟩

end VStar.Recursive.SampledTwo
