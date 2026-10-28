import Proofs.Nci.Fit

/-!
# The fit bound with counted multiplications left out

`FitsSlots` and `ReachSlots` are adopted for dense recipes with two kinds of work left out:
initialization windows (each server's first window, which builds its weights from a seed, so `FitsSlots`'s slot half
fails there) and threads that aren't the trained model's tokens (side computations inside the training circuit, such as
data filters). `keep T s` is `T` with its counted multiplications cut to the gates in `s`, and `keepSlots σ s` is `σ` on
it. `fitBoundSlots_keep` is `fitBoundSlots_start` at `keep T s`, with the counted multiplications outside `s` added to
the bound: a citation that leaves work out charges it.

* `fitBoundSlots_init` cuts by window: `s` is every gate outside the windows in `I`. It asks `FitsSlots` of the other
  windows only, and `ReachSlots`, counted outside `I`, of the threads started outside `I`. A thread an initialization
  window starts has every kept multiplication in a window it entered, whose entry fee pays for it, so it needs no
  `ReachSlots`. The charge is the counted multiplications inside `I` (`benchmarks/nci/fit.py`, `left_out`).
* `fitBoundSlots_threads` cuts by thread: `s` is the counted multiplications of the threads in `K`, under `ReachSlots`
  and `FitsSlots` at `keep T s`.
-/

namespace Nci.SecurityProofs.Fit

open Verity Finset

variable {c : RegisteredCircuit}

/-- `T` with its counted multiplications cut to the gates in `s`. -/
abbrev keep (T : Threading c) (s : Finset (Fin c.D.N)) : Threading c where
  m := T.m
  W := T.W
  thread := T.thread
  thread_W := T.thread_W
  M := T.M.filter (· ∈ s)
  M_computed := (filter_subset _ _).trans T.M_computed
  weight := T.weight
  weight_W g hg := T.weight_W g (mem_filter.1 hg).1
  weight_reads g hg := T.weight_reads g (mem_filter.1 hg).1
  act := T.act
  act_reads g hg := T.act_reads g (mem_filter.1 hg).1
  act_thread g hg := T.act_thread g (mem_filter.1 hg).1
  reuse := T.reuse
  reuse_le w t := le_trans (card_le_card (filter_subset_filter _ (filter_subset _ _))) (T.reuse_le w t)

/-- A slot map of `T` on `keep T s`. -/
def keepSlots {T : Threading c} (σ : T.Slots) (s : Finset (Fin c.D.N)) : (keep T s).Slots where
  p := σ.p
  slot := σ.slot
  reuse_le s' t := le_trans (card_le_card (filter_subset_filter _ (filter_subset _ _))) (σ.reuse_le s' t)

theorem mem_keep_threadMults (T : Threading c) (s R : Finset (Fin c.D.N)) (t : Fin T.m) (g : Fin c.D.N) :
    g ∈ (keep T s).threadMults R t ↔ g ∈ T.threadMults R t ∧ g ∈ s := by
  simp only [Threading.threadMults, Threading.mults, keep, mem_filter, mem_inter]
  tauto

theorem keep_mults_subset (T : Threading c) (s R : Finset (Fin c.D.N)) : (keep T s).mults R ⊆ T.mults R :=
  fun g hg => by
    simp only [Threading.mults, mem_inter, mem_filter] at hg ⊢
    exact ⟨hg.1.1, hg.2⟩

theorem keepSlots_slotsUsed_subset {T : Threading c} (σ : T.Slots) (s R : Finset (Fin c.D.N)) :
    (keepSlots σ s).slotsUsed R ⊆ σ.slotsUsed R :=
  image_subset_image (image_subset_image (keep_mults_subset T s R))

theorem keep_entered_subset (T : Threading c) (s : Finset (Fin c.D.N)) {k : ℕ} (P : Partition c.D k)
    (start : Fin T.m → Fin k) (j : Fin k) : (keep T s).entered P start j ⊆ T.entered P start j := by
  intro t ht
  simp only [Threading.entered, Threading.visits, mem_filter, mem_univ, true_and] at ht ⊢
  obtain ⟨⟨g, hg⟩, hj⟩ := ht
  exact ⟨⟨g, ((mem_keep_threadMults T s _ t g).1 hg).1⟩, hj⟩

/-- **The fit bound with work left out**: under `fitBoundSlots_start`'s premises at `keep T s`, `T`'s counted
multiplications number at most its bound plus those outside `s`, written multiplied out. -/
theorem fitBoundSlots_keep (T : Threading c) (σ : T.Slots) (s : Finset (Fin c.D.N)) {k : ℕ} (P : Partition c.D k)
    (start : Fin T.m → Fin k) (Nstar : ℕ) (S b_w w_e : ℝ) (hgap : S < Nstar * b_w)
    (hS : ∀ j, (novelReads c (P.gates j) : ℝ) ≤ S)
    (hR : ∀ t, ((keep T s).threadMults (P.gates (start t)) t).Nonempty →
      Nstar ≤ ((keepSlots σ s).threadSlots univ t).card)
    (hF : Assumptions.FitsSlots c (keep T s) (keepSlots σ s) P start b_w w_e) :
    4 * w_e * (Nstar * b_w - S) * T.M.card ≤
      k * T.reuse * Nstar * S ^ 2 + 4 * w_e * (Nstar * b_w - S) * (T.M.filter (· ∉ s)).card := by
  have hk := fitBoundSlots_start (keepSlots σ s) P start Nstar S b_w w_e hgap hS hR hF
  have hM : (keep T s).M = T.M.filter (· ∈ s) := rfl
  have hr : (keep T s).reuse = T.reuse := rfl
  rw [hM, hr] at hk
  have hsplit : (T.M.card : ℝ) = ((T.M.filter (· ∈ s)).card : ℝ) + ((T.M.filter (· ∉ s)).card : ℝ) := by
    exact_mod_cast (card_filter_add_card_filter_not (s := T.M) (· ∈ s)).symm
  rw [hsplit, mul_add]
  exact add_le_add hk le_rfl

/-- The gates outside the windows in `I`. -/
def outside {k : ℕ} (P : Partition c.D k) (I : Finset (Fin k)) : Finset (Fin c.D.N) :=
  univ.filter fun g => ∀ j ∈ I, g ∉ P.gates j

theorem keep_outside_mults {T : Threading c} {k : ℕ} (P : Partition c.D k) {I : Finset (Fin k)} {j : Fin k}
    (hj : j ∈ I) : (keep T (outside P I)).mults (P.gates j) = ∅ := by
  refine eq_empty_of_forall_notMem fun g hg => ?_
  simp only [Threading.mults, outside, mem_inter, mem_filter, mem_univ, true_and] at hg
  exact hg.1.2 j hj hg.2

/-- **The fit bound with initialization windows left out**: `FitsSlots` at every window outside `I`, and `ReachSlots`
outside `I` for the threads started outside `I` that do counted work in their start window, bound `T`'s counted
multiplications by `FitBoundSlots`'s bound plus those inside `I`. -/
theorem fitBoundSlots_init (T : Threading c) (σ : T.Slots) {k : ℕ} (P : Partition c.D k) (I : Finset (Fin k))
    (start : Fin T.m → Fin k) (Nstar : ℕ) (S b_w w_e : ℝ) (hbw : 0 ≤ b_w) (hwe : 0 ≤ w_e) (hgap : S < Nstar * b_w)
    (hS : ∀ j, (novelReads c (P.gates j) : ℝ) ≤ S)
    (hR : ∀ t, start t ∉ I → ((keep T (outside P I)).threadMults (P.gates (start t)) t).Nonempty →
      Nstar ≤ ((keepSlots σ (outside P I)).threadSlots univ t).card)
    (hF : ∀ j ∉ I, b_w * (σ.slotsUsed (P.gates j)).card + w_e * (T.entered P start j).card ≤
      novelReads c (P.gates j)) :
    4 * w_e * (Nstar * b_w - S) * T.M.card ≤
      k * T.reuse * Nstar * S ^ 2 +
        4 * w_e * (Nstar * b_w - S) * (T.M.filter fun g => ∃ j ∈ I, g ∈ P.gates j).card := by
  have hR' : ∀ t, ((keep T (outside P I)).threadMults (P.gates (start t)) t).Nonempty →
      Nstar ≤ ((keepSlots σ (outside P I)).threadSlots univ t).card := by
    intro t ⟨g, hg⟩
    by_cases hI : start t ∈ I
    · have hm : g ∈ (keep T (outside P I)).mults (P.gates (start t)) := (mem_filter.1 hg).1
      rw [keep_outside_mults P hI] at hm
      exact absurd hm (notMem_empty g)
    · exact hR t hI ⟨g, hg⟩
  have hF' : Assumptions.FitsSlots c (keep T (outside P I)) (keepSlots σ (outside P I)) P start b_w w_e := by
    intro j
    by_cases hI : j ∈ I
    · have hm := keep_outside_mults (T := T) P hI
      have hs : (keepSlots σ (outside P I)).slotsUsed (P.gates j) = ∅ := by
        rw [Threading.Slots.slotsUsed, Threading.weightsUsed, hm, image_empty, image_empty]
      have he : (keep T (outside P I)).entered P start j = ∅ := by
        refine eq_empty_of_forall_notMem fun t ht => ?_
        simp only [Threading.entered, Threading.visits, mem_filter, mem_univ, true_and] at ht
        obtain ⟨g, hg⟩ := ht.1
        have hg' : g ∈ (keep T (outside P I)).mults (P.gates j) := (mem_filter.1 hg).1
        rw [hm] at hg'
        exact notMem_empty g hg'
      rw [hs, he, card_empty, card_empty, Nat.cast_zero, mul_zero, mul_zero, add_zero]
      exact Nat.cast_nonneg _
    · have hp : ((keepSlots σ (outside P I)).slotsUsed (P.gates j)).card ≤ (σ.slotsUsed (P.gates j)).card :=
        card_le_card (keepSlots_slotsUsed_subset σ _ _)
      have he : ((keep T (outside P I)).entered P start j).card ≤ (T.entered P start j).card :=
        card_le_card (keep_entered_subset T _ P start j)
      have h1 := mul_le_mul_of_nonneg_left (Nat.cast_le.2 hp : (_ : ℝ) ≤ _) hbw
      have h2 := mul_le_mul_of_nonneg_left (Nat.cast_le.2 he : (_ : ℝ) ≤ _) hwe
      have h3 := hF j hI
      linarith
  have hk := fitBoundSlots_keep T σ (outside P I) P start Nstar S b_w w_e hgap hS hR' hF'
  have hout : T.M.filter (· ∉ outside P I) = T.M.filter fun g => ∃ j ∈ I, g ∈ P.gates j := by
    refine filter_congr fun g _ => ?_
    simp only [outside, mem_filter, mem_univ, true_and, not_forall, not_not, exists_prop]
  rwa [hout] at hk

/-- The counted multiplications of the threads in `K`. -/
def threadGates (T : Threading c) (K : Finset (Fin T.m)) : Finset (Fin c.D.N) :=
  T.M.filter fun g => ∃ t ∈ K, T.thread (T.act g) = some t

/-- **The fit bound with threads left out**: `ReachSlots` and `FitsSlots` at `T` cut to the threads in `K` bound `T`'s
counted multiplications by `FitBoundSlots`'s bound plus those of the other threads. -/
theorem fitBoundSlots_threads (T : Threading c) (σ : T.Slots) (K : Finset (Fin T.m)) {k : ℕ} (P : Partition c.D k)
    (start : Fin T.m → Fin k) (Nstar : ℕ) (S b_w w_e : ℝ) (hgap : S < Nstar * b_w)
    (hS : ∀ j, (novelReads c (P.gates j) : ℝ) ≤ S)
    (hR : Assumptions.ReachSlots c (keep T (threadGates T K)) (keepSlots σ (threadGates T K)) Nstar)
    (hF : Assumptions.FitsSlots c (keep T (threadGates T K)) (keepSlots σ (threadGates T K)) P start b_w w_e) :
    4 * w_e * (Nstar * b_w - S) * T.M.card ≤
      k * T.reuse * Nstar * S ^ 2 + 4 * w_e * (Nstar * b_w - S) * (T.M.filter (· ∉ threadGates T K)).card :=
  fitBoundSlots_keep T σ _ P start Nstar S b_w w_e hgap hS
    (fun t h => hR t (h.mono (threadMults_mono (keep T (threadGates T K)) (subset_univ _) t))) hF

end Nci.SecurityProofs.Fit
