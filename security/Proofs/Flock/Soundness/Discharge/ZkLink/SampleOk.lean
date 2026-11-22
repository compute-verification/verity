module

public import Proofs.Flock.Soundness.Discharge.Exec.SamplingDesignSample
public import Proofs.Flock.Soundness.Discharge.Exec.Pts
public import Proofs.Flock.Soundness.Discharge.ZkLink.TablesAt
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
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk

/-!
# The challenger's sample at every sample of the live sampling design, at `--zk`

`zk_session_sound` takes `SampleOkZK I dj mPts` at each of `SamplingDesign.execOS`'s samples: the sample file `dj S` proves `S`'s units,
and setupH, under the `--zk` tags, sets up at `mPts` link points. The first half is what the record carries, so it is
custody; the second is verity-verify-proof's own arithmetic. This file splits them.

* `drawOf samplingDesign pop S`: the verifier's own sample of `S`, as `UnitSample.ofJson` decodes it: its sampling design and population, `S`'s
  units ascending, no closure.
* `SampleFileZK dj samplingDesign pop`: every sample file `dj S` that `UnitSample.ofJson` decodes decodes to `drawOf samplingDesign pop S`. It is a
  property of the file the challenger records, not of verity-verify-proof (DrawWords' `hdj`, with the sampling design).
* `sampleOf_proved`, `sampleOf_mem`, `sampleOf_size`: the sample `drawOf` decodes proves `S`'s units, each of them, and
  has `S`'s size; `ofJson_count`: a sample file that `UnitSample.ofJson` decodes passes its per-stratum count.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink

open Flock Exec ZkExec Finset FlockSoundness.Refine

theorem forIn_throw_ok {α : Type} (P : α → Bool) (msg : α → String) :
    ∀ (l : List α) (r : PUnit.{1}),
      forIn (m := V) l PUnit.unit
          (fun s _ => if P s = true then throw (msg s) else pure (ForInStep.yield PUnit.unit)) = .ok r →
        ∀ s ∈ l, P s = false
  | [], _, _, s, hs => by simp at hs
  | a :: l, r, h, s, hs => by
      rw [List.forIn_cons] at h
      by_cases ha : P a = true
      · rw [ite_eq_left ha, throw_bind'] at h
        exact absurd h (throw_ne_ok _ _)
      · rw [ite_eq_right ha] at h
        rcases List.mem_cons.1 hs with rfl | hs
        · simpa using ha
        · exact forIn_throw_ok P msg l r h s hs

theorem foldl_count_filter (p : ℕ → Bool) : ∀ (l : List ℕ) (a : ℕ),
    l.foldl (fun acc u => if p u = true then acc + 1 else acc) a = a + (l.filter p).length
  | [], a => by simp
  | u :: l, a => by
      rw [List.foldl_cons, foldl_count_filter p l, List.filter_cons]
      split <;> (simp_all; try omega)

-- `walk_step` down the unfolded do-block of `ofJson`, the sample's decoder, a goal per branch
set_option maxHeartbeats 2000000 in
/-- **`UnitSample.ofJson`'s per-stratum count**: a sample it decodes under a stratified sampling design holds exactly `k_s` of its units in
each stratum `s`'s ranges. -/
theorem ofJson_count {j : Lean.Json} {D : Draw.UnitSample} (h : Draw.UnitSample.ofJson j = .ok D) {K : ℕ}
    {strata : Array Draw.Stratum} (hl : D.samplingDesign = .stratified K strata) (s : Draw.Stratum) (hs : s ∈ strata) :
    D.units.foldl (fun acc u => if (s.units.any fun x => decide (x.1 ≤ u) && decide (u < x.2)) = true then acc + 1
      else acc) 0 = s.k := by
  unfold Draw.UnitSample.ofJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
  all_goals
    have hp := ‹(pure _ : V Draw.SamplingDesign) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq] at hp
  all_goals
    simp only [reduceCtorEq, Draw.SamplingDesign.stratified.injEq] at hl
  all_goals
    obtain ⟨rfl, rfl⟩ := hl
    have hf := ‹forIn _ _ _ = Except.ok _›
    rw [← Array.forIn_toList] at hf
    simpa using forIn_throw_ok _ _ _ _ hf s (by simpa using hs)

variable {n : ℕ}

/-- **The verifier's own sample of `S`** under `samplingDesign` over `[0, pop)`, as `UnitSample.ofJson` decodes it: `S`'s units,
ascending, and no closure. -/
def drawOf (samplingDesign : Draw.SamplingDesign) (pop : ℕ) (S : Finset (Fin n)) : Draw.UnitSample :=
  { samplingDesign, population := pop, units := ((S.image Fin.val).sort (· ≤ ·)).toArray }

/-- **The record's sample file is the verifier's sample** (custody of the record's `unit_draw`): at every sample `S`, a sample
file `dj S` that `UnitSample.ofJson` decodes decodes to the verifier's own sample of `S` under `samplingDesign` over `[0, pop)`. -/
def SampleFileZK (dj : Finset (Fin n) → Lean.Json) (samplingDesign : Draw.SamplingDesign) (pop : ℕ) : Prop :=
  ∀ S D, Draw.UnitSample.ofJson (dj S) = .ok D → D = drawOf samplingDesign pop S

theorem sampleOf_proved (samplingDesign : Draw.SamplingDesign) (pop : ℕ) (S : Finset (Fin n)) :
    (drawOf samplingDesign pop S).proved = (drawOf samplingDesign pop S).units := by
  simp [Draw.UnitSample.proved, drawOf]

theorem sampleOf_mem (samplingDesign : Draw.SamplingDesign) (pop : ℕ) (S : Finset (Fin n)) (x : ℕ) :
    x ∈ (drawOf samplingDesign pop S).units ↔ ∃ u ∈ S, u.val = x := by
  simp [drawOf]

theorem sampleOf_size (samplingDesign : Draw.SamplingDesign) (pop : ℕ) (S : Finset (Fin n)) : (drawOf samplingDesign pop S).units.size = S.card := by
  simp [drawOf, card_image_of_injective _ Fin.val_injective]

end FlockSoundness.Discharge.ZkLink
