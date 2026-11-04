import Definitions.Nci.Guarantees.Fit

/-!
# Proof of the fit bound

The fit bound, over a slot map `σ` (`Threading.Slots`). Write `p_j^t` for the slots thread `t` uses in window `j`, `A_t`
for `p_j^t` at its start window and `B_t` for the sum of `p_j^t` over the windows it entered. A thread's counted
multiplications in a window number at most `reuse` per slot it uses there (`card_threadMults_le_reuse_slots`).

* Token side (`token_bound`): `ReachSlots` gives `A_t + B_t ≥ N*` and `FitsSlots` gives `A_t b_w ≤ S`, so a thread's
  multiplications times `N* b_w − S` are at most `reuse · N* b_w B_t`. A thread with no counted multiplication in its
  start window has `A_t = 0` and needs no `ReachSlots`, so `fitBoundSlots_start` asks it only of the others.
* Window side (`window_ineq`): `Σ_t B_t ≤ Σ_j p_j e_j`, with `p_j` the slots window `j` uses and `e_j` the threads it
  entered, and `FitsSlots` gives `b_w p_j + w_e e_j ≤ S`, so `4 b_w w_e p_j e_j ≤ S²` by AM–GM.

`FitBound` is the case of one slot per weight gate (`idSlots`): for a slot map one-to-one on the weight gates,
`FitsSlots` is `Fits` and `ReachSlots` is `Reach` (`fitsSlots_iff`, `reachSlots_iff`).
-/

namespace Nci.SecurityProofs.Fit

open Verity Finset

/-- A thread's counted multiplications in `R` number at most `reuse` per slot it uses in `R`. -/
theorem card_threadMults_le_reuse_slots {c : RegisteredCircuit} {T : Threading c} (σ : T.Slots) (R : Finset (Fin c.D.N))
    (t : Fin T.m) : (T.threadMults R t).card ≤ T.reuse * (σ.threadSlots R t).card := by
  have himg : σ.threadSlots R t = (T.threadMults R t).image (σ.slot ∘ T.weight) := by
    rw [Threading.Slots.threadSlots, Threading.threadWeights, image_image]
  rw [himg]
  refine card_le_mul_card_image (f := σ.slot ∘ T.weight) _ _ fun s _ => ?_
  refine le_trans (card_le_card ?_) (σ.reuse_le s t)
  intro g hg
  rw [mem_filter] at hg ⊢
  rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter] at hg
  exact ⟨hg.1.1.1, hg.2, hg.1.2⟩

/-- The weights a thread uses in `R` are among those `R` uses. -/
theorem threadWeights_subset {c : RegisteredCircuit} (T : Threading c) (R : Finset (Fin c.D.N)) (t : Fin T.m) :
    T.threadWeights R t ⊆ T.weightsUsed R := by
  intro w hw
  obtain ⟨g, hg, rfl⟩ := mem_image.1 hw
  exact mem_image_of_mem T.weight (mem_filter.1 hg).1

/-- The slots a thread uses in `R` are among those `R` uses. -/
theorem threadSlots_subset {c : RegisteredCircuit} {T : Threading c} (σ : T.Slots) (R : Finset (Fin c.D.N)) (t : Fin T.m) :
    σ.threadSlots R t ⊆ σ.slotsUsed R := by
  intro s hs
  obtain ⟨w, hw, rfl⟩ := mem_image.1 hs
  exact mem_image_of_mem σ.slot (threadWeights_subset T R t hw)

/-- The weights `R` uses are weight gates. -/
theorem weightsUsed_subset_W {c : RegisteredCircuit} (T : Threading c) (R : Finset (Fin c.D.N)) : T.weightsUsed R ⊆ T.W := by
  intro w hw
  obtain ⟨g, hg, rfl⟩ := mem_image.1 hw
  exact T.weight_W g (mem_inter.1 hg).1

theorem threadMults_mono {c : RegisteredCircuit} (T : Threading c) {R R' : Finset (Fin c.D.N)} (h : R ⊆ R')
    (t : Fin T.m) : T.threadMults R t ⊆ T.threadMults R' t := by
  intro g hg
  rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter] at hg ⊢
  exact ⟨⟨hg.1.1, h hg.1.2⟩, hg.2⟩

/-- Every computed gate lies in a unit of the partition. -/
theorem exists_gates {c : RegisteredCircuit} {k : ℕ} (P : Partition c.D k) {g : Fin c.D.N} (hg : g ∈ c.D.computed) :
    ∃ j, g ∈ P.gates j := by
  have hin : c.D.isInput g = false := (mem_filter.1 hg).2
  have hne : P.unit g ≠ none := fun h => by
    have h' := (P.input_iff g).1 h
    rw [hin] at h'
    exact Bool.false_ne_true h'
  obtain ⟨j, hj⟩ := Option.ne_none_iff_exists'.1 hne
  exact ⟨j, mem_filter.2 ⟨mem_univ _, hj⟩⟩

/-- Every counted multiplication is one of some thread's in some window. -/
theorem card_M_le {c : RegisteredCircuit} (T : Threading c) {k : ℕ} (P : Partition c.D k) :
    T.M.card ≤ ∑ t : Fin T.m, ∑ j : Fin k, (T.threadMults (P.gates j) t).card := by
  have hsub : T.M ⊆ univ.biUnion fun t => univ.biUnion fun j => T.threadMults (P.gates j) t := by
    intro g hg
    obtain ⟨j, hj⟩ := exists_gates P (T.M_computed hg)
    obtain ⟨t, ht⟩ := Option.isSome_iff_exists.1 (T.act_thread g hg)
    have hgj : g ∈ T.threadMults (P.gates j) t := by
      rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter]
      exact ⟨⟨hg, hj⟩, ht⟩
    simp only [mem_biUnion, mem_univ, true_and]
    exact ⟨t, j, hgj⟩
  calc T.M.card ≤ (univ.biUnion fun t => univ.biUnion fun j => T.threadMults (P.gates j) t).card :=
        card_le_card hsub
    _ ≤ ∑ t, (univ.biUnion fun j => T.threadMults (P.gates j) t).card := card_biUnion_le
    _ ≤ ∑ t, ∑ j, (T.threadMults (P.gates j) t).card := sum_le_sum fun t _ => card_biUnion_le

/-- The weights a thread uses anywhere are used in some window. -/
theorem threadWeights_univ_subset {c : RegisteredCircuit} (T : Threading c) {k : ℕ} (P : Partition c.D k) (t : Fin T.m) :
    T.threadWeights univ t ⊆ univ.biUnion fun j => T.threadWeights (P.gates j) t := by
  intro w hw
  obtain ⟨g, hg, rfl⟩ := mem_image.1 hw
  rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter] at hg
  obtain ⟨j, hj⟩ := exists_gates P (T.M_computed hg.1.1)
  have hgj : g ∈ T.threadMults (P.gates j) t := by
    rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter]
    exact ⟨⟨hg.1.1, hj⟩, hg.2⟩
  exact mem_biUnion.2 ⟨j, mem_univ _, mem_image_of_mem T.weight hgj⟩

/-- The slots a thread uses anywhere are used in some window. -/
theorem card_threadSlots_univ_le {c : RegisteredCircuit} {T : Threading c} (σ : T.Slots) {k : ℕ} (P : Partition c.D k)
    (t : Fin T.m) : (σ.threadSlots univ t).card ≤ ∑ j : Fin k, (σ.threadSlots (P.gates j) t).card := by
  have hsub : σ.threadSlots univ t ⊆ univ.biUnion fun j => σ.threadSlots (P.gates j) t := by
    intro s hs
    obtain ⟨w, hw, rfl⟩ := mem_image.1 hs
    obtain ⟨j, -, hj⟩ := mem_biUnion.1 (threadWeights_univ_subset T P t hw)
    exact mem_biUnion.2 ⟨j, mem_univ _, mem_image_of_mem σ.slot hj⟩
  exact (card_le_card hsub).trans card_biUnion_le

/-- A thread's slots in a window count at its start window or at a window it entered. -/
theorem card_threadSlots_le_split {c : RegisteredCircuit} {T : Threading c} (σ : T.Slots) {k : ℕ} (P : Partition c.D k)
    (start : Fin T.m → Fin k) (t : Fin T.m) (j : Fin k) :
    ((σ.threadSlots (P.gates j) t).card : ℝ) ≤
      (if j = start t then ((σ.threadSlots (P.gates j) t).card : ℝ) else 0) +
        (if t ∈ T.entered P start j then ((σ.threadSlots (P.gates j) t).card : ℝ) else 0) := by
  by_cases hj : j = start t
  · rw [ite_eq_left hj]
    have h0 : (0 : ℝ) ≤ (if t ∈ T.entered P start j then ((σ.threadSlots (P.gates j) t).card : ℝ) else 0) := by
      split_ifs <;> positivity
    linarith
  · rw [ite_eq_right hj, zero_add]
    by_cases hv : (T.threadMults (P.gates j) t).Nonempty
    · have hmem : t ∈ T.entered P start j := by
        simp only [Threading.entered, Threading.visits, mem_filter, mem_univ, true_and]
        exact ⟨hv, fun h => hj h.symm⟩
      exact le_of_eq (ite_eq_left hmem).symm
    · have hw : T.threadWeights (P.gates j) t = ∅ := by
        rw [Threading.threadWeights, not_nonempty_iff_eq_empty.1 hv, image_empty]
      have hz : σ.threadSlots (P.gates j) t = ∅ := by
        rw [Threading.Slots.threadSlots, hw, image_empty]
      simp [hz]

/-- The token side's arithmetic: from `A + B ≥ N`, `A b_w ≤ S < N b_w` and `W ≤ r (A + B)`,
`W (N b_w − S) ≤ r N b_w B`. -/
theorem token_ineq {N bw S A B W r : ℝ} (hS : 0 ≤ S) (hN0 : 0 ≤ N) (hgap : S < N * bw) (hA : A * bw ≤ S)
    (hN : N ≤ A + B) (hW : W ≤ r * (A + B)) (hr : 0 ≤ r) :
    W * (N * bw - S) ≤ r * N * bw * B := by
  have h1 : A * bw * N ≤ S * N := mul_le_mul_of_nonneg_right hA hN0
  have h2 : S * N ≤ S * (A + B) := mul_le_mul_of_nonneg_left hN hS
  have h3 : (A + B) * (N * bw - S) ≤ N * bw * B := by linarith
  calc W * (N * bw - S) ≤ r * (A + B) * (N * bw - S) := mul_le_mul_of_nonneg_right hW (by linarith)
    _ = r * ((A + B) * (N * bw - S)) := by ring
    _ ≤ r * (N * bw * B) := mul_le_mul_of_nonneg_left h3 hr
    _ = r * N * bw * B := by ring

/-- **The token side**: a thread's counted multiplications, times `N* b_w − S`, are at most `reuse · N* b_w` times the
slots it uses in the windows it entered. It needs `ReachSlots` only for a thread with counted multiplications in its
start window: one without has every slot it uses in a window it entered. -/
theorem token_bound {c : RegisteredCircuit} {T : Threading c} (σ : T.Slots) {k : ℕ} (P : Partition c.D k)
    (start : Fin T.m → Fin k) (Nstar : ℕ) (S b_w w_e : ℝ) (hbw : 0 < b_w) (hwe : 0 ≤ w_e) (hgap : S < Nstar * b_w)
    (hS : ∀ j, (novelReads c (P.gates j) : ℝ) ≤ S) (t : Fin T.m)
    (hR : (T.threadMults (P.gates (start t)) t).Nonempty → Nstar ≤ (σ.threadSlots univ t).card)
    (hF : Assumptions.FitsSlots c T σ P start b_w w_e) :
    (∑ j, ((T.threadMults (P.gates j) t).card : ℝ)) * (Nstar * b_w - S) ≤
      T.reuse * Nstar * b_w *
        ∑ j, (if t ∈ T.entered P start j then ((σ.threadSlots (P.gates j) t).card : ℝ) else 0) := by
  have hS0 : 0 ≤ S := (Nat.cast_nonneg _).trans (hS (start t))
  have hN0 : (0 : ℝ) ≤ Nstar := Nat.cast_nonneg _
  have hr0 : (0 : ℝ) ≤ T.reuse := Nat.cast_nonneg _
  have hsplit : ∑ j, ((σ.threadSlots (P.gates j) t).card : ℝ) ≤
      ((σ.threadSlots (P.gates (start t)) t).card : ℝ) +
        ∑ j, (if t ∈ T.entered P start j then ((σ.threadSlots (P.gates j) t).card : ℝ) else 0) := by
    calc ∑ j, ((σ.threadSlots (P.gates j) t).card : ℝ)
        ≤ ∑ j, ((if j = start t then ((σ.threadSlots (P.gates j) t).card : ℝ) else 0) +
            (if t ∈ T.entered P start j then ((σ.threadSlots (P.gates j) t).card : ℝ) else 0)) :=
          sum_le_sum fun j _ => card_threadSlots_le_split σ P start t j
      _ = _ := by rw [sum_add_distrib, sum_ite_eq', ite_eq_left (mem_univ _)]
  have hW : ∑ j, ((T.threadMults (P.gates j) t).card : ℝ) ≤
      T.reuse * (((σ.threadSlots (P.gates (start t)) t).card : ℝ) +
        ∑ j, (if t ∈ T.entered P start j then ((σ.threadSlots (P.gates j) t).card : ℝ) else 0)) := by
    calc ∑ j, ((T.threadMults (P.gates j) t).card : ℝ)
        ≤ ∑ j, (T.reuse : ℝ) * (σ.threadSlots (P.gates j) t).card :=
          sum_le_sum fun j _ => by exact_mod_cast card_threadMults_le_reuse_slots σ _ t
      _ = T.reuse * ∑ j, ((σ.threadSlots (P.gates j) t).card : ℝ) := by rw [mul_sum]
      _ ≤ _ := mul_le_mul_of_nonneg_left hsplit hr0
  by_cases hwork : (T.threadMults (P.gates (start t)) t).Nonempty
  · have hN : (Nstar : ℝ) ≤ ∑ j, ((σ.threadSlots (P.gates j) t).card : ℝ) := by
      exact_mod_cast (hR hwork).trans (card_threadSlots_univ_le σ P t)
    have hA : ((σ.threadSlots (P.gates (start t)) t).card : ℝ) * b_w ≤ S := by
      have hp : ((σ.threadSlots (P.gates (start t)) t).card : ℝ) ≤ (σ.slotsUsed (P.gates (start t))).card :=
        Nat.cast_le.2 (card_le_card (threadSlots_subset σ _ t))
      have h1 := mul_le_mul_of_nonneg_right hp hbw.le
      have h2 := hF (start t)
      have h3 : 0 ≤ w_e * ((T.entered P start (start t)).card : ℝ) := mul_nonneg hwe (Nat.cast_nonneg _)
      have h4 := hS (start t)
      linarith
    exact token_ineq hS0 hN0 hgap hA (hN.trans hsplit) hW hr0
  · have hA : σ.threadSlots (P.gates (start t)) t = ∅ := by
      rw [Threading.Slots.threadSlots, Threading.threadWeights, not_nonempty_iff_eq_empty.1 hwork, image_empty,
        image_empty]
    rw [hA, card_empty, Nat.cast_zero, zero_add] at hW
    have hW0 : (0 : ℝ) ≤ ∑ j, ((T.threadMults (P.gates j) t).card : ℝ) := sum_nonneg fun _ _ => Nat.cast_nonneg _
    calc (∑ j, ((T.threadMults (P.gates j) t).card : ℝ)) * (Nstar * b_w - S)
        ≤ (∑ j, ((T.threadMults (P.gates j) t).card : ℝ)) * (Nstar * b_w) :=
          mul_le_mul_of_nonneg_left (by linarith) hW0
      _ ≤ (T.reuse * ∑ j, (if t ∈ T.entered P start j then ((σ.threadSlots (P.gates j) t).card : ℝ) else 0)) *
            (Nstar * b_w) := mul_le_mul_of_nonneg_right hW (mul_nonneg hN0 hbw.le)
      _ = _ := by ring

/-- **The window side**, by AM–GM: `b_w p + w_e e ≤ S` gives `4 b_w w_e p e ≤ S²`. -/
theorem window_ineq {bw we p e S : ℝ} (hbw : 0 ≤ bw) (hwe : 0 ≤ we) (hp : 0 ≤ p) (he : 0 ≤ e)
    (h : bw * p + we * e ≤ S) : 4 * (bw * we) * (p * e) ≤ S ^ 2 := by
  have h0 : 0 ≤ bw * p + we * e := add_nonneg (mul_nonneg hbw hp) (mul_nonneg hwe he)
  have h1 : (bw * p + we * e) * (bw * p + we * e) ≤ S * S := mul_le_mul h h h0 (h0.trans h)
  nlinarith [sq_nonneg (bw * p - we * e)]

/-- The two sides together. -/
theorem fit_ineq {M X Bs PE N r bw we S n : ℝ} (hwe : 0 ≤ we) (hbw : 0 ≤ bw) (hN : 0 ≤ N) (hr : 0 ≤ r)
    (hgap : 0 ≤ N * bw - S) (hM : M ≤ X) (htok : X * (N * bw - S) ≤ r * N * bw * Bs) (hB : Bs ≤ PE)
    (hwin : 4 * (bw * we) * PE ≤ n * S ^ 2) :
    4 * we * (N * bw - S) * M ≤ n * r * N * S ^ 2 := by
  have h4 : (0 : ℝ) ≤ 4 * we := by linarith
  have hrNb : 0 ≤ r * N * bw := mul_nonneg (mul_nonneg hr hN) hbw
  calc 4 * we * (N * bw - S) * M = 4 * we * (M * (N * bw - S)) := by ring
    _ ≤ 4 * we * (X * (N * bw - S)) := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hM hgap) h4
    _ ≤ 4 * we * (r * N * bw * Bs) := mul_le_mul_of_nonneg_left htok h4
    _ ≤ 4 * we * (r * N * bw * PE) := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hB hrNb) h4
    _ = r * N * (4 * (bw * we) * PE) := by ring
    _ ≤ r * N * (n * S ^ 2) := mul_le_mul_of_nonneg_left hwin (mul_nonneg hr hN)
    _ = n * r * N * S ^ 2 := by ring

/-- The slot map of one slot per weight gate. -/
def idSlots {c : RegisteredCircuit} (T : Threading c) : T.Slots where
  p := c.D.N
  slot := id
  reuse_le s t := T.reuse_le s t

theorem idSlots_injOn {c : RegisteredCircuit} (T : Threading c) : Set.InjOn (idSlots T).slot T.W := by
  intro a _ b _ h
  exact h

/-- A slot map under which a thread uses at most one version of each slot meets `Slots.reuse_le` at `T.reuse`. -/
def slotsOfOneVersion {c : RegisteredCircuit} (T : Threading c) (n : ℕ) (slot : Fin c.D.N → Fin n)
    (h : ∀ g ∈ T.M, ∀ g' ∈ T.M, T.thread (T.act g) = T.thread (T.act g') →
      slot (T.weight g) = slot (T.weight g') → T.weight g = T.weight g') : T.Slots where
  p := n
  slot := slot
  reuse_le s t := by
    obtain he | ⟨g₀, hg₀⟩ :=
      (T.M.filter fun g => slot (T.weight g) = s ∧ T.thread (T.act g) = some t).eq_empty_or_nonempty
    · rw [he, card_empty]
      exact Nat.zero_le _
    · rw [mem_filter] at hg₀
      refine le_trans (card_le_card ?_) (T.reuse_le (T.weight g₀) t)
      intro g hg
      rw [mem_filter] at hg ⊢
      exact ⟨hg.1, h g hg.1 g₀ hg₀.1 (hg.2.2.trans hg₀.2.2.symm) (hg.2.1.trans hg₀.2.1.symm), hg.2.2⟩

/-- With every slot one version (`σ.slot` one-to-one on the weight gates), `R` uses as many slots as weights. -/
theorem card_slotsUsed {c : RegisteredCircuit} {T : Threading c} (σ : T.Slots) (hσ : Set.InjOn σ.slot T.W)
    (R : Finset (Fin c.D.N)) : (σ.slotsUsed R).card = (T.weightsUsed R).card := by
  rw [Threading.Slots.slotsUsed]
  exact card_image_of_injOn (hσ.mono (Finset.coe_subset.2 (weightsUsed_subset_W T R)))

/-- With every slot one version, a thread uses as many slots as weights in `R`. -/
theorem card_threadSlots {c : RegisteredCircuit} {T : Threading c} (σ : T.Slots) (hσ : Set.InjOn σ.slot T.W)
    (R : Finset (Fin c.D.N)) (t : Fin T.m) : (σ.threadSlots R t).card = (T.threadWeights R t).card := by
  rw [Threading.Slots.threadSlots]
  exact card_image_of_injOn (hσ.mono (Finset.coe_subset.2
    (Finset.Subset.trans (threadWeights_subset T R t) (weightsUsed_subset_W T R))))

/-- **`FitsSlots` is `Fits` when every slot has one version.** -/
theorem fitsSlots_iff {c : RegisteredCircuit} {T : Threading c} (σ : T.Slots) (hσ : Set.InjOn σ.slot T.W) {k : ℕ}
    (P : Partition c.D k) (start : Fin T.m → Fin k) (b_w w_e : ℝ) :
    Assumptions.FitsSlots c T σ P start b_w w_e ↔ Assumptions.Fits c T P start b_w w_e := by
  simp only [Assumptions.FitsSlots, Assumptions.Fits, card_slotsUsed σ hσ]

/-- **`ReachSlots` is `Reach` when every slot has one version.** -/
theorem reachSlots_iff {c : RegisteredCircuit} {T : Threading c} (σ : T.Slots) (hσ : Set.InjOn σ.slot T.W) (Nstar : ℕ) :
    Assumptions.ReachSlots c T σ Nstar ↔ Assumptions.Reach c T Nstar := by
  simp only [Assumptions.ReachSlots, Assumptions.Reach, card_threadSlots σ hσ]

/-- **The fit bound over parameter slots**, with `ReachSlots` asked only of the threads that have counted
multiplications in their start window (`token_bound`). With no counted multiplication the bound is `0 ≤ …`; otherwise
some window exists, so `S ≥ 0` and `b_w > 0`, and at `w_e ≤ 0` the left side is not positive. -/
theorem fitBoundSlots_start {c : RegisteredCircuit} {T : Threading c} (σ : T.Slots) {k : ℕ} (P : Partition c.D k)
    (start : Fin T.m → Fin k) (Nstar : ℕ) (S b_w w_e : ℝ) (hgap : S < Nstar * b_w)
    (hS : ∀ j, (novelReads c (P.gates j) : ℝ) ≤ S)
    (hR : ∀ t, (T.threadMults (P.gates (start t)) t).Nonempty → Nstar ≤ (σ.threadSlots univ t).card)
    (hF : Assumptions.FitsSlots c T σ P start b_w w_e) :
    4 * w_e * (Nstar * b_w - S) * T.M.card ≤ k * T.reuse * Nstar * S ^ 2 := by
  have hN0 : (0 : ℝ) ≤ Nstar := Nat.cast_nonneg _
  have hr0 : (0 : ℝ) ≤ T.reuse := Nat.cast_nonneg _
  rcases T.M.eq_empty_or_nonempty with hM | ⟨g, hg⟩
  · rw [hM, card_empty, Nat.cast_zero, mul_zero]
    positivity
  obtain ⟨j₀, -⟩ := exists_gates P (T.M_computed hg)
  have hS0 : 0 ≤ S := (Nat.cast_nonneg _).trans (hS j₀)
  have hbw : 0 < b_w := by
    by_contra h
    nlinarith [mul_nonneg hN0 (neg_nonneg.2 (not_lt.1 h))]
  have hgap0 : 0 ≤ (Nstar : ℝ) * b_w - S := by linarith
  rcases le_or_gt w_e 0 with hwe | hwe
  · have h1 : 0 ≤ ((Nstar : ℝ) * b_w - S) * T.M.card := mul_nonneg hgap0 (Nat.cast_nonneg _)
    have h2 : 4 * w_e * (Nstar * b_w - S) * T.M.card ≤ 0 := by
      nlinarith [mul_nonneg (neg_nonneg.2 hwe) h1]
    exact h2.trans (by positivity)
  have hMr : (T.M.card : ℝ) ≤ ∑ t, ∑ j, ((T.threadMults (P.gates j) t).card : ℝ) := by
    exact_mod_cast card_M_le T P
  have htok : (∑ t, ∑ j, ((T.threadMults (P.gates j) t).card : ℝ)) * (Nstar * b_w - S) ≤
      T.reuse * Nstar * b_w *
        ∑ t, ∑ j, (if t ∈ T.entered P start j then ((σ.threadSlots (P.gates j) t).card : ℝ) else 0) := by
    rw [sum_mul, mul_sum]
    exact sum_le_sum fun t _ => token_bound σ P start Nstar S b_w w_e hbw hwe.le hgap hS t (hR t) hF
  have hB : ∑ t, ∑ j, (if t ∈ T.entered P start j then ((σ.threadSlots (P.gates j) t).card : ℝ) else 0) ≤
      ∑ j, ((σ.slotsUsed (P.gates j)).card : ℝ) * (T.entered P start j).card := by
    rw [sum_comm]
    refine sum_le_sum fun j _ => ?_
    rw [sum_ite_mem, univ_inter]
    calc ∑ t ∈ T.entered P start j, ((σ.threadSlots (P.gates j) t).card : ℝ)
        ≤ ∑ _t ∈ T.entered P start j, ((σ.slotsUsed (P.gates j)).card : ℝ) :=
          sum_le_sum fun t _ => Nat.cast_le.2 (card_le_card (threadSlots_subset σ _ t))
      _ = _ := by rw [sum_const, nsmul_eq_mul]; ring
  have hwin : 4 * (b_w * w_e) * ∑ j, ((σ.slotsUsed (P.gates j)).card : ℝ) * (T.entered P start j).card ≤
      k * S ^ 2 := by
    rw [mul_sum]
    calc ∑ j, 4 * (b_w * w_e) * (((σ.slotsUsed (P.gates j)).card : ℝ) * (T.entered P start j).card)
        ≤ ∑ _j : Fin k, S ^ 2 := sum_le_sum fun j _ =>
          window_ineq hbw.le hwe.le (Nat.cast_nonneg _) (Nat.cast_nonneg _) ((hF j).trans (hS j))
      _ = k * S ^ 2 := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  exact fit_ineq hwe.le hbw.le hN0 hr0 hgap0 hMr htok hB hwin

end Nci.SecurityProofs.Fit

namespace Nci.SecurityProofs

open Verity Finset Nci.SecurityProofs.Fit

/-- **The fit bound over parameter slots**: `fitBoundSlots_start`, whose premise `ReachSlots` gives. -/
theorem FitBoundSlots : Guarantees.FitBoundSlots := by
  intro c T σ k P start Nstar S b_w w_e hgap hS hR hF
  exact fitBoundSlots_start σ P start Nstar S b_w w_e hgap hS
    (fun t h => hR t (h.mono (threadMults_mono T (subset_univ _) t))) hF

/-- **The old form follows**: `FitBound` is `FitBoundSlots` at one slot per weight gate (`idSlots`), where `ReachSlots`
is `Reach` and `FitsSlots` is `Fits`. -/
theorem fitBound_of_slots (h : Guarantees.FitBoundSlots) : Guarantees.FitBound := by
  intro c T k P start Nstar S b_w w_e hgap hS hR hF
  exact h c T (idSlots T) k P start Nstar S b_w w_e hgap hS ((reachSlots_iff _ (idSlots_injOn T) Nstar).2 hR)
    ((fitsSlots_iff _ (idSlots_injOn T) P start b_w w_e).2 hF)

/-- **The fit bound**, as `FitBoundSlots` at one slot per weight gate. -/
theorem FitBound : Guarantees.FitBound := fitBound_of_slots FitBoundSlots

end Nci.SecurityProofs
