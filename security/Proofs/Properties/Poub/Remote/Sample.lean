module

public import Properties.Poub.Remote
import Proofs.Properties.Poub.Remote.Bisect
import Proofs.Flock.SpotFlock.SampleBelow
import Proofs.Flock.Claims.Sampling.Select.SecondUnits

/-!
# The window's samples (`SampleIsSubset`)

A window's sample is the proof system's draw on its sample's coins (`Spot.Draw.sampleUnits`'s subset law), which, when
it ends, is `k` distinct units below `U` (`sampleUnits_subset_spec`), and `receiptSample` maps each to its receipt by the
passes' running ends (`locate`).
-/

@[expose] public section

open Poub

namespace Properties.Poub

/-! ## The samples -/

open Properties.Spot.SecurityProofs (bind_ok bind_error onWith_take) in
/-- **A `subset` sample that ends is `k` distinct units below `n`**: `sampleUnits`' units are `Draw.subset`'s
(`Flock.Bridge.sampleUnits_subset_lt`'s reading), of size `k` (`subset_size`) and distinct (`subset_nodup`). -/
theorem sampleUnits_subset_spec {k n : ℕ} {src : ByteArray} {units : Array ℕ} {read : ℕ}
    (h : Spot.Draw.sampleUnits (.subset k) n src = .ok (units, read)) :
    units.size = k ∧ units.toList.Nodup ∧ ∀ x ∈ units.toList, x < n := by
  unfold Spot.Draw.sampleUnits at h
  cases hw : (Flock.Draw.drawWith (fun _ need => Spot.Draw.take src need) (.subset k) n).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨r, read'⟩ := y
    rw [hw, bind_ok] at h
    obtain ⟨d, rfl⟩ : ∃ d, r = some d := by
      cases r with
      | none => exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
      | some d => exact ⟨d, rfl⟩
    dsimp only at h
    obtain ⟨hd, -⟩ := Prod.mk.inj (Except.ok.inj h)
    subst hd
    unfold Flock.Draw.drawWith at hw
    dsimp only at hw
    rw [StateT.run_bind] at hw
    cases ho : (Flock.Draw.onWith (fun need => Spot.Draw.take src need)
        (2 * k * Flock.Draw.width n + 64) (Flock.Draw.subset k n)).run 0 with
    | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
    | ok z =>
      obtain ⟨v, p⟩ := z
      obtain ⟨j, D, rfl, -, hst⟩ := onWith_take src _ (by omega) _ 0 _ _ ho
      obtain ⟨-, -, hrun, -⟩ := hst
      rw [ho, bind_ok, StateT.run_pure] at hw
      obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
      obtain rfl := (Option.some.inj h1).symm
      exact ⟨Flock.Bridge.subset_size hrun, Properties.Spot.SecurityProofs.subset_nodup hrun⟩

theorem sampleIsSubset : SampleIsSubset := by
  intro samples coin lengths sampled h
  unfold receiptSample receiptSampleRead at h
  dsimp only at h
  have hP := prefixSums_fold (fun n => n) lengths
  simp only [List.map_id'] at hP
  generalize (lengths.foldl (fun (acc : Array Nat × Nat) n => (acc.1.push (acc.2 + n), acc.2 + n)) (#[], 0)).1 = ends
    at h hP
  have htotal := back_getD_of_prefix hP
  rw [htotal] at h
  by_cases hk0 : min samples lengths.toList.sum = 0
  · rw [if_pos hk0] at h
    replace h := (Option.some.inj h).symm
    subst h
    exact ⟨#[], Or.inl ⟨hk0, rfl⟩, rfl, by simpa using hk0.symm, fun k hk => absurd hk (by simp),
      fun k₁ _ h₁ => absurd h₁ (by simp)⟩
  rw [if_neg hk0] at h
  cases hu : Spot.Draw.sampleUnits (.subset (min samples lengths.toList.sum)) lengths.toList.sum coin with
  | error e =>
    rw [hu] at h
    cases h
  | ok x =>
    obtain ⟨us, read⟩ := x
    rw [hu] at h
    change some _ = some sampled at h
    replace h := (Option.some.inj h).symm
    obtain ⟨hsize, hnodup, hlt⟩ := sampleUnits_subset_spec hu
    have hdsize : sampled.size = us.size := by rw [h]; simp
    have hk : ∀ k (hk : k < sampled.size), sampled[k].1 < lengths.size ∧ sampled[k].2 < lengths[sampled[k].1]! ∧
        receiptIndex lengths sampled[k].1 sampled[k].2 = us[k]! := by
      intro k hk
      have huk : us[k]! < lengths.toList.sum := by
        rw [getElem!_pos us k (by omega)]
        exact hlt _ (Array.getElem_mem_toList _)
      obtain ⟨hi, hlo, hhi, hoff⟩ := locate hP _ huk
      have hd : sampled[k] = (bisectRight ends us[k]!,
          us[k]! - (if bisectRight ends us[k]! = 0 then 0 else ends[bisectRight ends us[k]! - 1]!)) := by
        subst h
        simp [getElem!_pos us k (by simpa using hk)]
      rw [hd]
      simp only [Array.length_toList] at hi
      refine ⟨hi, ?_, ?_⟩
      · rw [hoff]
        have : lengths[bisectRight ends us[k]!]! = lengths.toList[bisectRight ends us[k]!]! := by
          rw [getElem!_pos lengths _ hi, getElem!_pos lengths.toList _ (by simpa using hi)]
          simp
        rw [this]
        omega
      · simp only [receiptIndex]
        rw [hoff]
        omega
    refine ⟨us, Or.inr ⟨Nat.pos_of_ne_zero hk0, read, hu⟩, hdsize, hsize, hk, fun k₁ k₂ h₁ h₂ heq => ?_⟩
    have e₁ := (hk k₁ h₁).2.2
    have e₂ := (hk k₂ h₂).2.2
    rw [heq] at e₁
    rw [e₁, getElem!_pos us k₁ (by omega), getElem!_pos us k₂ (by omega)] at e₂
    have : us.toList[k₁]'(by simp; omega) = us.toList[k₂]'(by simp; omega) := by simpa using e₂
    exact (List.Nodup.getElem_inj hnodup).1 this

end Properties.Poub
