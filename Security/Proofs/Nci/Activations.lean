import Definitions.Nci.Guarantees.Activations
import Proofs.Nci.Threads

/-!
# Proofs of NCI's activations bound over all of training

For one small subcircuit `R` (`activations_reads`), call a thread **full** for `R` when `R` holds a whole activation of it
(`Links.Whole`), and let `ρ t` count the activations of thread `t` that `R` uses and does not hold whole (`openActs`).
* At a full thread, take its earliest whole activation (by gate index, which the links decrease): the activations it is
  computed from are of its thread, used by `R`, and not whole, so `ρ t ≥ dd` (`le_card_openActs`).
* At a thread that is not full, every activation `R` uses is not whole, and each enters at most `e` counted
  multiplications, so the thread's multiplications in `R` number at most `e ρ t` (`card_threadMults_le_open`).
* At every thread, its multiplications in `R` number at most `reuse` per weight `R` uses (`card_threadMults_le`).
The `ρ t` add up to at most the activations `ActivationsIncompressibleAll` charges (`sum_card_openActs_le`), so Theorem
1's counting (`threaded_count_of`) gives `Λ = max(e, reuse · K / (4 dd b_w))`. A thread with fewer than `dd` open
activations is not full, which is the threading inequality that counting takes.

Through a partial containment (`ActivationsBoundOn`), each IU of a compliant division of the declaration computes a piece
of `S` that is small and reads no more than the IU (`piece`, `novelReads_piece`), and the pieces divide `S`.
-/

namespace Nci.SecurityProofs

open Verity Finset

section Activations

variable {c : RegisteredCircuit} {T : Threading c} (L : T.Links)

theorem le_of_feeds {C : Circuit} {g t : Fin C.N} (hf : C.Feeds g t) : g ≤ t := by
  induction hf with
  | refl => exact le_rfl
  | step hgh _ ih => exact (C.reads_lt _ _ hgh).le.trans ih

/-- The activation of a counted multiplication that `a` is computed from precedes `a`. -/
theorem act_lt_of_mem_src {a g : Fin c.D.N} (hg : g ∈ L.src a) : T.act g < a :=
  (c.D.reads_lt g _ (T.act_reads g (L.src_M a hg))).trans_le (le_of_feeds (L.src_feeds a g hg))

/-- The activations of thread `t` that `R` uses and does not hold whole. -/
def openActs (R : Finset (Fin c.D.N)) (t : Fin T.m) : Finset (Fin c.D.N) :=
  ((T.actsUsed R).filter fun a => ¬ L.Whole R a).filter fun a => T.thread a = some t

variable {L}

/-- **A full thread charges `dd` activations**: if `R` holds an activation of thread `t` whole, it uses at least `dd`
activations of thread `t` that it does not hold whole, those its earliest whole activation of `t` is computed from. -/
theorem le_card_openActs {R : Finset (Fin c.D.N)} {dd : ℕ}
    (hd : ∀ a, (L.src a).Nonempty →
      dd ≤ (((L.src a).filter fun g => T.thread (T.act g) = T.thread a).image T.act).card)
    {t : Fin T.m} {a : Fin c.D.N} (hat : T.thread a = some t) (ha : L.Whole R a) : dd ≤ (openActs L R t).card := by
  obtain ⟨a₀, ⟨hat₀, ha₀⟩, hmin⟩ :=
    (wellFounded_lt (α := Fin c.D.N)).has_min {a | T.thread a = some t ∧ L.Whole R a} ⟨a, hat, ha⟩
  refine (hd a₀ ha₀.1).trans (card_le_card fun x hx => ?_)
  obtain ⟨g, hg, rfl⟩ := mem_image.1 hx
  obtain ⟨hgs, hgt⟩ := mem_filter.1 hg
  rw [hat₀] at hgt
  have hgm : g ∈ T.mults R := mem_inter.2 ⟨L.src_M a₀ hgs, ha₀.2 hgs⟩
  simp only [openActs, Threading.actsUsed, mem_filter]
  exact ⟨⟨mem_image_of_mem _ hgm, fun hw => hmin _ ⟨hgt, hw⟩ (act_lt_of_mem_src L hgs)⟩, hgt⟩

/-- **A thread that is not full costs an activation per `e` multiplications**: when `R` holds no activation of thread
`t` whole, every activation of `t` that `R` uses is open, and each enters at most `e` counted multiplications. -/
theorem card_threadMults_le_open {R : Finset (Fin c.D.N)} {e : ℕ}
    (he : ∀ a, (T.M.filter fun g => T.act g = a).card ≤ e) {t : Fin T.m}
    (ht : ¬ ∃ a, T.thread a = some t ∧ L.Whole R a) :
    (T.threadMults R t).card ≤ e * (openActs L R t).card := by
  have hsub : T.threadMults R t ⊆ (openActs L R t).biUnion fun a => T.M.filter fun g => T.act g = a := by
    intro g hg
    rw [Threading.threadMults, mem_filter] at hg
    obtain ⟨hgm, hgt⟩ := hg
    refine mem_biUnion.2 ⟨T.act g, ?_, mem_filter.2 ⟨(mem_inter.1 hgm).1, rfl⟩⟩
    simp only [openActs, Threading.actsUsed, mem_filter]
    exact ⟨⟨mem_image_of_mem _ hgm, fun hw => ht ⟨_, hgt, hw⟩⟩, hgt⟩
  calc (T.threadMults R t).card ≤ ∑ a ∈ openActs L R t, (T.M.filter fun g => T.act g = a).card :=
        (card_le_card hsub).trans card_biUnion_le
    _ ≤ ∑ _a ∈ openActs L R t, e := sum_le_sum fun a _ => he a
    _ = e * (openActs L R t).card := by rw [sum_const, smul_eq_mul, mul_comm]

variable (L) in
/-- The open activations of all threads number at most those the premise charges. -/
theorem sum_card_openActs_le (R : Finset (Fin c.D.N)) :
    ∑ t, (openActs L R t).card ≤ ((T.actsUsed R).filter fun a => ¬ L.Whole R a).card := by
  rw [card_eq_sum_card_fiberwise (f := T.thread) (s := (T.actsUsed R).filter fun a => ¬ L.Whole R a)
    (t := univ) fun _ _ => mem_univ _, Fintype.sum_option]
  exact Nat.le_add_left _ _

end Activations

/-- **The activations bound for one subcircuit.** -/
theorem activations_reads (c : RegisteredCircuit) (T : Threading c) (L : T.Links) (Γ : ℕ) (K b_w b_a : ℝ) (dd e : ℕ)
    (hbw : 0 < b_w) (hba : 0 < b_a) (hdd : 0 < dd) (he : ∀ a, (T.M.filter fun g => T.act g = a).card ≤ e)
    (hd : ∀ a, (L.src a).Nonempty →
      dd ≤ (((L.src a).filter fun g => T.thread (T.act g) = T.thread a).image T.act).card)
    (hA : Assumptions.ActivationsIncompressibleAll c T L Γ K b_w b_a) (R : Finset (Fin c.D.N)) (hR : Small c Γ K R) :
    b_a * (T.mults R).card ≤ novelReads c R * T.bound K b_w dd e := by
  refine threaded_count_of T R (fun t => ((openActs L R t).card : ℝ)) (T.weightsUsed R).card
    (card_threadMults_le T R) (fun _ => Nat.cast_nonneg _) hbw hba hdd (fun t ht => ?_) ?_ hR.2.2
  · have hlt : (openActs L R t).card < dd := by
      exact_mod_cast (show ((openActs L R t).card : ℝ) < dd from ht)
    have hnf : ¬ ∃ a, T.thread a = some t ∧ L.Whole R a := fun ⟨_, hat, ha⟩ =>
      absurd (le_card_openActs hd hat ha) (not_le.2 hlt)
    show ((T.threadMults R t).card : ℝ) ≤ e * ((openActs L R t).card : ℝ)
    exact_mod_cast card_threadMults_le_open he hnf
  · have h := hA R hR
    have hs : (∑ t, ((openActs L R t).card : ℝ)) ≤ ((T.actsUsed R).filter fun a => ¬ L.Whole R a).card := by
      exact_mod_cast sum_card_openActs_le L R
    have h2 := mul_le_mul_of_nonneg_left hs hba.le
    show b_w * ((T.weightsUsed R).card : ℝ) + b_a * ∑ t, ((openActs L R t).card : ℝ) ≤ novelReads c R
    linarith

/-- **The activations bound for one subcircuit** ([ACTIVATIONS-BOUND], per subcircuit). -/
theorem ActivationsReads : Guarantees.ActivationsReads := activations_reads

/-- **The activations bound over all of training, with dark gates** ([ACTIVATIONS-BOUND]). -/
theorem ActivationsBoundOn : Guarantees.ActivationsBoundOn := by
  intro Γ α K b_w b_a c d T L dd e M hα hαK hbw hba hdd ⟨n, P, hP⟩ hB he hd hA
  set Λ := T.bound K b_w dd e
  have hΛ : 0 ≤ Λ := le_max_of_le_left (Nat.cast_nonneg _)
  have hreads : ∀ u, (novelReads c (piece M P u) : ℝ) ≤ α * d.D.size (P.gates u) := fun u =>
    (Nat.cast_le.2 (novelReads_piece hB u)).trans (hP u).2
  have hpiece : ∀ u, Small c Γ K (piece M P u) := fun u =>
    ⟨piece_subset_computed u, (card_piece_le u).trans (hP u).1,
      ((hreads u).trans (mul_le_mul_of_nonneg_left (by exact_mod_cast (hP u).1) hα)).trans hαK⟩
  have hM : (T.M ∩ M.S).card = ∑ u, (T.mults (piece M P u)).card := by
    have h := sum_card_filter_piece (M := M) (P := P) T.M fun g => g
    simp only [filter_mem_eq_inter] at h
    exact h.symm
  calc b_a * ((T.M ∩ M.S).card : ℝ) = ∑ u, b_a * ((T.mults (piece M P u)).card : ℝ) := by
        rw [hM, Nat.cast_sum, mul_sum]
    _ ≤ ∑ u, (novelReads c (piece M P u) : ℝ) * Λ :=
        sum_le_sum fun u _ => activations_reads c T L Γ K b_w b_a dd e hbw hba hdd he hd hA _ (hpiece u)
    _ ≤ ∑ u, α * (d.D.size (P.gates u) : ℝ) * Λ := sum_le_sum fun u _ => mul_le_mul_of_nonneg_right (hreads u) hΛ
    _ = α * Λ * d.D.size d.D.computed := by
        rw [← sum_mul, ← mul_sum, show (d.D.size d.D.computed : ℝ) = ∑ u, (d.D.size (P.gates u) : ℝ) by
          exact_mod_cast (Partition.sum_card_gates P).symm]
        ring

/-- **Training is slowed, by the activations bound, with dark gates.** -/
theorem ActivationsTrainingOn : Guarantees.ActivationsTrainingOn := by
  intro Γ α K b_w b_a κ c d T L dd e M hα hαK hbw hba hdd hP hB he hd hA hκ
  have h := ActivationsBoundOn Γ α K b_w b_a c d T L dd e M hα hαK hbw hba hdd hP hB he hd hA
  have hs : (T.M.card : ℝ) ≤ M.dark.card + (T.M ∩ M.S).card := by
    exact_mod_cast card_le_dark_add_inter M T.M_computed
  have h1 := mul_le_mul_of_nonneg_left hκ hba.le
  have h2 := mul_le_mul_of_nonneg_left hs hba.le
  nlinarith

open Classical in
/-- **The premise over all counted multiplications gives the chain's**, where the chain is part of the threading. -/
theorem activationsIncompressible_of_all {c : RegisteredCircuit} {ch : Chain c.D} {T : Threading c} {L : T.Links}
    (h : ChainIn ch T L) {Γ : ℕ} {K b_w b_a : ℝ} (hw : 0 ≤ b_w) (ha : 0 ≤ b_a)
    (hA : Assumptions.ActivationsIncompressibleAll c T L Γ K b_w b_a) :
    Assumptions.ActivationsIncompressible c ch Γ K b_w b_a := by
  intro R hR hΓ hK
  refine le_trans (add_le_add (mul_le_mul_of_nonneg_left (Nat.cast_le.2 ?_) hw)
    (mul_le_mul_of_nonneg_left (Nat.cast_le.2 ?_) ha)) (hA R ⟨hR, hΓ, hK⟩)
  · rw [← card_image_of_injective (ch.weightsUsed R) h.wgate_inj]
    refine card_le_card fun y hy => ?_
    obtain ⟨w, hw', rfl⟩ := mem_image.1 hy
    obtain ⟨t, ht⟩ := (mem_filter.1 hw').2
    exact mem_image.2 ⟨ch.prod w.1 t w.2.1 w.2.2, mem_inter.2 ⟨h.prod_M _ _ _ _, ht⟩, h.weight_prod _ _ _ _⟩
  · rw [← card_image_of_injective _ h.xgate_inj]
    refine card_le_card fun y hy => ?_
    obtain ⟨x, hx, rfl⟩ := mem_image.1 hy
    obtain ⟨hxu, hnl⟩ := mem_filter.1 hx
    obtain ⟨i, hi⟩ := (mem_filter.1 hxu).2
    refine mem_filter.2 ⟨mem_image.2 ⟨ch.prod x.1 x.2.1 i x.2.2, mem_inter.2 ⟨h.prod_M _ _ _ _, hi⟩,
      h.act_prod _ _ _ _⟩, fun hW => hnl ?_⟩
    obtain ⟨⟨_ | k, hℓ⟩, t, j⟩ := x
    · obtain ⟨g, hg⟩ := hW.1
      rw [h.src_first hℓ t j] at hg
      exact absurd hg (notMem_empty g)
    · exact fun j' => hW.2 (h.src_link ⟨k, by omega⟩ hℓ t j j')

/-- **The premise over all counted multiplications gives the chain's.** -/
theorem ActivationsChain : Guarantees.ActivationsChain :=
  fun _ _ _ _ _ _ _ _ h hw ha hA => activationsIncompressible_of_all h hw ha hA

end Nci.SecurityProofs
