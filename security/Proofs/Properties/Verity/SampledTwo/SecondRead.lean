module

public import Proofs.Properties.Verity.SampledTwo.Steps

/-!
# An accepted two-stage check's second-stage bytes are the coins its sample read

`Steps` reads the first stage's bytes against what the run read (`sourceRead_le`). The second stage's bytes `ξ₂` are
read by the form's last step (`secondCheck` at `ξ₂`), which refuses a sample that leaves a coin opened and not read
(`Spot.OneStage.coinsReason`).

* `coinsReason_read`: a sample the check finds no fault in read `read` bytes of its coins, and the coins hold as many
  whole challenger coins as those bytes reach: `⌈read / COIN_BYTES⌉ = src.size / COIN_BYTES`.
* `secondCheck_read`: the last step at the coins `ξ` accepts only at such a sample, under the second stage's law and the
  record's population.
* `second_read`: where the two-stage check accepts, the interior audit's sample is such a sample of `ξ₂`, the bytes
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

/-- **A sample the check finds no fault in reads every coin opened**: it read `read` bytes of `src`, and `src` holds as
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

/-- **The form's last step at the coins `ξ` accepts only where its sample read every coin of `ξ`**, under the second
stage's law `law` at the verifier whose sampling design is a subset of all the record's population. -/
theorem secondCheck_read {v : Registration.Verifier} {delta : Core.Ratio} {c : OneStage.Commitment} {law : Json}
    {ξ : ByteArray} {e : OneStage.Evidence} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : secondCheck v delta c law ξ e = .accept pub prof) :
    ∃ n units read, nat? (getD c.record "population") = some n ∧
      OneStage.drawnFrom { v with law := subsetLaw n } law n ξ = .ok (units, read) ∧
      (read + Challenger.COIN_BYTES - 1) / Challenger.COIN_BYTES = ξ.size / Challenger.COIN_BYTES := by
  unfold secondCheck at h
  cases hm : nat? (getD c.record "population") with
  | none => simp [hm] at h
  | some m =>
    simp only [hm] at h
    cases hr : OneStage.coinsReason { v with law := subsetLaw m } law c.record e.draw ξ with
    | some why => simp [hr] at h
    | none =>
      obtain ⟨n, units, read, hn, hdf, hsz⟩ := coinsReason_read hr
      have hmn : m = n := by rw [hm] at hn; exact Option.some.inj hn
      subst hmn
      first
        | exact ⟨_, units, read, rfl, hdf, hsz⟩
        | exact ⟨_, units, read, hm, hdf, hsz⟩

/-- **An accepted two-stage check's second-stage bytes are the coins its sample read**: the interior audit's sample, under
the second stage's law `secondLawOf` computes and its record's population, read `read` bytes of the bytes `ξ₂` its
sample read, and `ξ₂` holds as many whole challenger coins as those bytes reach. -/
theorem second_read {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ run law n units read, OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂ ∧ secondLawOf p run = .ok law ∧
      nat? (getD c₂.record "population") = some n ∧
      OneStage.drawnFrom { p.audit.base.verifier with law := subsetLaw n } law n ξ₂ = .ok (units, read) ∧
      (read + Challenger.COIN_BYTES - 1) / Challenger.COIN_BYTES = ξ₂.size / Challenger.COIN_BYTES := by
  obtain ⟨-, c₂, ξ₂, r, i, run, hi, -, -, -, -, law, hl, htwo⟩ := accepted_steps h
  obtain ⟨-, hξ, -⟩ := inputAt_ok hi
  obtain ⟨n, units, read, hn, hdf, hsz⟩ := secondCheck_read htwo
  exact ⟨c₂, ξ₂, run, law, n, units, read, hξ, hl, hn, hdf, hsz⟩

end VStar.Recursive.SampledTwo
