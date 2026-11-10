module

public import Definitions.Nci.Guarantees.Witnesses
public import Proofs.Nci.Rate
public import Proofs.Nci.Reads
public import Proofs.Nci.RemotePeriod
public import Proofs.Nci.Capability.Witnesses
public import Proofs.Core.Circuit
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Tactic.Linarith

/-!
# The inference-only policy's guarantees: proofs of their witnesses

Each witness of `Definitions.Nci.Guarantees.Witnesses`, at the toy (`Nci.Capability.Toy`) and the instances of
`Nci.Witness`. The facts they share come first: the toy's one proof block is a `(2, 1)`-partition of exactly two gates,
so the toy complies at `Γ = 2` and the rate `α = 1/2`, and with the token registered at `α = 0`; the threading `T1` is
`(1, 0)`-threaded and its weights incompressible at half a bit each, read off the toy's sixteen sets of gates; `T0` has
no counted multiplication and bound `e`; and `F` is positive.
-/

@[expose] public section

namespace Nci.Witness

open Verity Finset Nci.Capability.Toy

theorem isCK_D : P1.IsCK D 2 1 := { boolean := ⟨fun _ => rfl, by decide⟩, card := by decide, reads := by decide }

theorem isCK_Dθ : P1.IsCK Dθ 2 0 := { boolean := ⟨fun _ => rfl, by decide⟩, card := by decide, reads := by decide }

/-- The toy's one proof block is two gates reading one fresh bit, the token: it complies at `Γ = 2` and `α = 1/2`. -/
theorem compliesRate_D : CompliesRate 2 (1 / 2) D := by
  simpa only [Nat.cast_one, Nat.cast_ofNat] using SecurityProofs.compliesRate_of_complies isCK_D (by decide)

/-- With the token registered too, the toy's one proof block reads no fresh bit: it complies at `α = 0`. -/
theorem compliesRate_Dθ : CompliesRate 2 0 Dθ := by
  simpa only [Nat.cast_zero, zero_div] using SecurityProofs.compliesRate_of_complies isCK_Dθ (by decide)

theorem size_D : D.D.size D.D.computed = 2 := by decide

theorem card_T1 : T1.M.card = 1 := by decide

theorem theta_id (r : RegisteredCircuit) :
    Assumptions.ThetaRegistered r r (PartialContainment.ofContainment (idC r.D)) := fun _ _ h => h

theorem not_theta_Dθ : ¬ Assumptions.ThetaRegistered D Dθ Mθ :=
  fun h => absurd (h 1 (by decide) (by decide)) (by decide)

theorem dark_id (r : RegisteredCircuit) : (PartialContainment.ofContainment (c := r) (d := r) (idC r.D)).dark = ∅ :=
  Finset.sdiff_self _

theorem T1_threaded {k : ℝ} : T1.Threaded 2 k 1 0 := by
  have key : ∀ R : Finset (Fin 4), R ⊆ C.computed → ∀ t : Fin 1, (T1.threadReads R t).card < 1 →
      (T1.threadMults R t).card ≤ 0 * (T1.threadReads R t).card := by decide
  exact fun R hR t => key R hR.1 t

theorem T1_weights_nat : ∀ R : Finset (Fin 4), R ⊆ C.computed →
    (T1.weightsUsed R).card + (T1.threadReads R (0 : Fin 1)).card ≤ 2 * novelReads D R := by decide

theorem T1_weights {k : ℝ} : Assumptions.WeightsIncompressible D T1 2 k (1 / 2) (1 / 2) := by
  intro R hR
  have h := T1_weights_nat R hR.1
  have hs : ∑ t, (T1.threadReads R t).card = (T1.threadReads R (0 : Fin 1)).card := Fin.sum_univ_one _
  rw [Nat.cast_sum, ← Nat.cast_sum, hs]
  have h' : ((T1.weightsUsed R).card : ℝ) + (T1.threadReads R (0 : Fin 1)).card ≤ 2 * novelReads D R := by exact_mod_cast h
  linarith

theorem T1_weightsSlots {k : ℝ} : Assumptions.WeightsIncompressibleSlots D T1 σ1 2 k (1 / 2) (1 / 2) := by
  have key : ∀ R : Finset (Fin 4), R ⊆ C.computed →
      (σ1.slotsUsed R).card + (T1.threadReads R (0 : Fin 1)).card ≤ 2 * novelReads D R := by decide
  intro R hR
  have h := key R hR.1
  have hs : ∑ t, (T1.threadReads R t).card = (T1.threadReads R (0 : Fin 1)).card := Fin.sum_univ_one _
  rw [Nat.cast_sum, ← Nat.cast_sum, hs]
  have h' : ((σ1.slotsUsed R).card : ℝ) + (T1.threadReads R (0 : Fin 1)).card ≤ 2 * novelReads D R := by exact_mod_cast h
  linarith

theorem coverage_T1 : Assumptions.Coverage 1 D (2 * T1.M.card) := by
  unfold Assumptions.Coverage
  rw [size_D, card_T1]
  norm_num

theorem T0_threaded (r : RegisteredCircuit) (c : ℕ) (k : ℝ) (d e : ℕ) : (T0 r).Threaded c k d e :=
  fun _ _ t => t.elim0

theorem T0_weights (r : RegisteredCircuit) (c : ℕ) (k b_w b_a : ℝ) :
    Assumptions.WeightsIncompressible r (T0 r) c k b_w b_a := by
  intro R _
  have h0 : ∑ t, ((T0 r).threadReads R t).card = 0 := Finset.sum_eq_zero fun t _ => Fin.elim0 t
  rw [h0]
  simp [Threading.weightsUsed, Threading.mults, T0]

theorem T0_weightsSlots (r : RegisteredCircuit) (c : ℕ) (k b_w b_a : ℝ) :
    Assumptions.WeightsIncompressibleSlots r (T0 r) (σ0 r) c k b_w b_a := by
  intro R _
  have h0 : ∑ t, ((T0 r).threadReads R t).card = 0 := Finset.sum_eq_zero fun t _ => Fin.elim0 t
  rw [h0]
  simp [Threading.Slots.slotsUsed, Threading.weightsUsed, Threading.mults, T0]

theorem not_coverage_T0 : ¬ Assumptions.Coverage 1 D (2 * (T0 D).M.card) := by
  unfold Assumptions.Coverage
  rw [size_D, show (T0 D).M.card = 0 from rfl]
  norm_num

theorem bound_T0 (r : RegisteredCircuit) (k b_w : ℝ) (d e : ℕ) : (T0 r).bound k b_w d e = e := by
  simp [Threading.bound, T0]

theorem bound_T1 (k : ℝ) (hk : 0 ≤ k) : T1.bound k (1 / 2) 1 0 = k / 2 := by
  have : T1.reuse = 1 := rfl
  simp only [Threading.bound, this, Nat.cast_one, Nat.cast_zero]
  rw [max_eq_right (by positivity)]
  ring

theorem entersAtMost_T1 : Assumptions.EntersAtMost T1 1 := by
  unfold Assumptions.EntersAtMost
  decide

theorem drawsAtLeast_noLinks {r : RegisteredCircuit} (T : Threading r) (dd : ℕ) :
    Assumptions.DrawsAtLeast (noLinks T) dd :=
  fun _ h => absurd h (by simp [noLinks])

theorem activations_T1 : Assumptions.ActivationsIncompressibleAll D T1 (noLinks T1) 2 1 (1 / 2) (1 / 2) := by
  have key : ∀ R : Finset (Fin 4), R ⊆ C.computed →
      (T1.weightsUsed R).card + (T1.actsUsed R).card ≤ 2 * novelReads D R := by decide
  intro R hR
  rw [filter_true_of_mem fun a _ h => by simp [Threading.Links.Whole, noLinks] at h]
  have h' : ((T1.weightsUsed R).card : ℝ) + (T1.actsUsed R).card ≤ 2 * novelReads D R := by
    exact_mod_cast key R hR.1
  linarith

theorem entersAtMost_T0 (r : RegisteredCircuit) : Assumptions.EntersAtMost (T0 r) 0 :=
  fun _ => (card_le_card (filter_subset _ _)).trans (by simp [T0])

theorem activations_T0 (r : RegisteredCircuit) (c : ℕ) (k b_w b_a : ℝ) :
    Assumptions.ActivationsIncompressibleAll r (T0 r) (noLinks (T0 r)) c k b_w b_a := by
  intro R _
  simp [Threading.weightsUsed, Threading.actsUsed, Threading.mults, T0]

theorem readFactor_ge {K b_w b_a d : ℝ} (hK : 0 < K) (hw : 0 ≤ b_w) (ha : 0 ≤ b_a) (hd : 0 ≤ d) :
    min (9 / 4 * b_w * b_a * d / K) (3 / 4 * b_a * √(b_w / K)) ≤ readFactor K b_w b_a d := by
  unfold readFactor
  have hbdd : BddAbove (Set.range fun μ : Set.Ioo (0 : ℝ) 1 =>
      min ((1 + √(μ : ℝ)) ^ 2 * b_w * b_a * d / K) (√2 * (1 + √(μ : ℝ)) * √(1 - μ) * b_a * √(b_w / K))) := by
    refine ⟨4 * (b_w * b_a * d / K), ?_⟩
    rintro _ ⟨μ, rfl⟩
    refine (min_le_left _ _).trans ?_
    have h1 : √(μ : ℝ) ≤ 1 := (Real.sqrt_le_sqrt μ.2.2.le).trans_eq Real.sqrt_one
    have h0 : 0 ≤ √(μ : ℝ) := Real.sqrt_nonneg _
    have h4 : (1 + √(μ : ℝ)) ^ 2 ≤ 4 := by nlinarith
    have hbd : 0 ≤ b_w * b_a * d / K := by positivity
    calc (1 + √(μ : ℝ)) ^ 2 * b_w * b_a * d / K = (1 + √(μ : ℝ)) ^ 2 * (b_w * b_a * d / K) := by ring
      _ ≤ 4 * (b_w * b_a * d / K) := mul_le_mul_of_nonneg_right h4 hbd
  refine le_ciSup_of_le hbdd ⟨1 / 4, by norm_num, by norm_num⟩ ?_
  have hs : √((1 / 4 : ℝ)) = 1 / 2 := by
    rw [show (1 / 4 : ℝ) = (1 / 2) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  have h34 : (1 / 2 : ℝ) ≤ √(1 - 1 / 4) := by
    calc (1 / 2 : ℝ) = √((1 / 2) ^ 2) := (Real.sqrt_sq (by norm_num)).symm
      _ ≤ √(1 - 1 / 4) := Real.sqrt_le_sqrt (by norm_num)
  have h2 : (1 : ℝ) ≤ √2 := by
    calc (1 : ℝ) = √1 := Real.sqrt_one.symm
      _ ≤ √2 := Real.sqrt_le_sqrt (by norm_num)
  show _ ≤ min ((1 + √(1 / 4 : ℝ)) ^ 2 * b_w * b_a * d / K)
    (√2 * (1 + √(1 / 4 : ℝ)) * √(1 - 1 / 4) * b_a * √(b_w / K))
  rw [hs]
  refine min_le_min (le_of_eq (by ring)) ?_
  have hx : 0 ≤ b_a * √(b_w / K) := by positivity
  calc 3 / 4 * b_a * √(b_w / K) = (1 * (1 + 1 / 2) * (1 / 2)) * (b_a * √(b_w / K)) := by ring
    _ ≤ (√2 * (1 + 1 / 2) * √(1 - 1 / 4)) * (b_a * √(b_w / K)) := by gcongr
    _ = √2 * (1 + 1 / 2) * √(1 - 1 / 4) * b_a * √(b_w / K) := by ring

theorem F_ge {α K b_w b_a d : ℝ} (hα : 0 < α) (hK : 0 < K) (hw : 0 ≤ b_w) (ha : 0 ≤ b_a) (hd : 0 ≤ d) :
    min (9 / 4 * b_w * b_a * d / K) (3 / 4 * b_a * √(b_w / K)) / (2 * α) ≤ F α K b_w b_a d := by
  rw [SecurityProofs.F_eq_readFactor]
  exact div_le_div_of_nonneg_right (readFactor_ge hK hw ha hd) (by positivity)

theorem ch0_weightsUsed (D' : Circuit) (R : Finset (Fin D'.N)) : (ch0 D').weightsUsed R = ∅ :=
  eq_empty_of_forall_notMem fun x _ => x.1.elim0

theorem ch0_inputsUsed (D' : Circuit) (R : Finset (Fin D'.N)) : (ch0 D').inputsUsed R = ∅ :=
  eq_empty_of_forall_notMem fun x _ => x.1.elim0

theorem ch0_G (D' : Circuit) : (ch0 D').G = 0 := Finset.sum_eq_zero fun ℓ _ => ℓ.elim0

/-- `ch1` is linked to no whole entry: its one multiplication has no predecessor. -/
theorem ch1_notLinked (R : Finset (Fin 4)) (x : (ℓ : Fin ch1.L) × Fin ch1.N × Fin (ch1.d ℓ)) :
    ¬ ch1.LinkedWhole R x.1 x.2.1 x.2.2 := by
  rcases x with ⟨⟨ℓ, hℓ⟩, t, j⟩
  cases ℓ with
  | zero => exact id
  | succ ℓ => exact absurd hℓ (by change ¬ (ℓ + 1 < 1); omega)

theorem ch1_nat : ∀ R : Finset (Fin 4), R ⊆ C.computed →
    (ch1.weightsUsed R).card ≤ novelReads D R ∧ (ch1.inputsUsed R).card ≤ novelReads D R := by decide

theorem ch1_G : ch1.G = 1 := rfl

theorem ch1_spread {n : ℕ} (P : Partition C n) (x : (ℓ : Fin ch1.L) × Fin ch1.N × Fin (ch1.e ℓ)) :
    ch1.spread P x.1 x.2.1 x.2.2 ≤ 1 := by
  classical
  unfold Chain.spread
  refine card_le_one.2 fun u hu v hv => ?_
  simp only [mem_filter, mem_univ, true_and] at hu hv
  obtain ⟨_, hu⟩ := hu
  obtain ⟨_, hv⟩ := hv
  rw [Partition.mem_gates] at hu hv
  exact Option.some_injective _ (hu.symm.trans hv)

theorem dark_M1 : M1.dark = ∅ := Finset.sdiff_self _

theorem dark_Mθ : Mθ.dark = ∅ := Finset.sdiff_self _

theorem darkReads_M1 (K : ℝ) : Assumptions.DarkReads M1 K :=
  fun v hv => absurd hv (by rw [dark_M1]; exact notMem_empty v)

theorem darkReads_Mθ (K : ℝ) : Assumptions.DarkReads Mθ K :=
  fun v hv => absurd hv (by rw [dark_Mθ]; exact notMem_empty v)

theorem coverage_ch1 : Assumptions.Coverage 1 D (2 * ch1.G) := by
  unfold Assumptions.Coverage
  rw [size_D, ch1_G]
  norm_num

theorem not_coverage_ch0 : ¬ Assumptions.Coverage 1 D (2 * (ch0 C).G) := by
  unfold Assumptions.Coverage
  rw [size_D, ch0_G]
  norm_num

theorem ch1_T1_incompressible {k : ℝ} : Assumptions.TrainingIncompressible D ch1 T1 2 k (1 / 2) (1 / 2) := by
  intro n P
  refine ⟨fun _ => 0, ?_, fun u hu => ⟨?_, T1_weights _ hu⟩⟩
  · refine le_trans (Finset.sum_le_sum fun x _ => Nat.sub_le_of_le_add (ch1_spread P x)) ?_
    simp
  · simp only [ch1_notLinked, not_false_eq_true, filter_const, ↓reduceIte, Nat.max_zero]
    obtain ⟨hw, hi⟩ := ch1_nat _ hu.1
    have hw' : ((ch1.weightsUsed (P.gates u)).card : ℝ) ≤ novelReads D (P.gates u) := by exact_mod_cast hw
    have hi' : ((ch1.inputsUsed (P.gates u)).card : ℝ) ≤ novelReads D (P.gates u) := by exact_mod_cast hi
    linarith

theorem ch0_T0_incompressible : Assumptions.TrainingIncompressible D (ch0 C) (T0 D) 2 1 1 2 := by
  intro n P
  refine ⟨fun _ => 0, le_trans (le_of_eq (Finset.sum_eq_zero fun x _ => x.1.elim0)) (Nat.zero_le _),
    fun u hu => ⟨?_, T0_weights D 2 1 1 2 _ hu⟩⟩
  rw [ch0_weightsUsed, ch0_inputsUsed]
  simp

theorem pch1_notLinked (R : Finset (Fin 4))
    (x : (ℓ : Fin ch1.toPacked.L) × Fin ch1.toPacked.N × Fin (ch1.toPacked.d ℓ)) :
    ¬ ch1.toPacked.LinkedWhole R x.1 x.2.1 x.2.2 := by
  rcases x with ⟨⟨ℓ, hℓ⟩, t, j⟩
  cases ℓ with
  | zero => exact id
  | succ ℓ => exact absurd hℓ (by change ¬ (ℓ + 1 < 1); omega)

theorem pch1_nat : ∀ R : Finset (Fin 4), R ⊆ C.computed →
    (ch1.toPacked.weightsUsed R).card ≤ novelReads D R ∧ (ch1.toPacked.inputsUsed R).card ≤ novelReads D R := by
  decide

theorem pch1_spread {n : ℕ} (P : Partition C n)
    (x : (ℓ : Fin ch1.toPacked.L) × Fin ch1.toPacked.N × Fin (ch1.toPacked.e ℓ)) :
    ch1.toPacked.spread P x.1 x.2.1 x.2.2 ≤ 1 := by
  classical
  unfold PackedChain.spread
  refine card_le_one.2 fun u hu v hv => ?_
  simp only [mem_filter, mem_univ, true_and] at hu hv
  obtain ⟨_, hu⟩ := hu
  obtain ⟨_, hv⟩ := hv
  rw [Partition.mem_gates] at hu hv
  exact Option.some_injective _ (hu.symm.trans hv)

theorem pch1_T1_incompressible :
    Assumptions.PackedTrainingIncompressible D ch1.toPacked T1 2 1 (1 / 2) (1 / 2) := by
  intro n P
  refine ⟨fun _ => 0, ?_, fun u hu => ⟨?_, T1_weights _ hu⟩⟩
  · refine le_trans (Finset.sum_le_sum fun x _ => Nat.sub_le_of_le_add (pch1_spread P x)) ?_
    simp
  · simp only [pch1_notLinked, not_false_eq_true, filter_const, ↓reduceIte, Nat.max_zero]
    obtain ⟨hw, hi⟩ := pch1_nat _ hu.1
    have hw' : ((ch1.toPacked.weightsUsed (P.gates u)).card : ℝ) ≤ novelReads D (P.gates u) := by
      exact_mod_cast hw
    have hi' : ((ch1.toPacked.inputsUsed (P.gates u)).card : ℝ) ≤ novelReads D (P.gates u) := by
      exact_mod_cast hi
    linarith

/-- The factor the contained bounds charge a product at `μ = 1`, `K = 1`, `b_w = b_a = 1/2`, `dd = emax = 1`: its
second branch, `μ b_a / max(emax, K/(4 b_w dd)) = 1/2`. -/
theorem factor_half : (1 / 2 : ℝ) ≤
    max (min ((1 + √1) ^ 2 * (1 / 2 * (1 / 2) * 1 / 1)) (√2 * (1 + √1) * √(1 - 1) * (1 / 2 * √(1 / 2 / 1))))
      (1 * (1 / 2) / max (1 : ℝ) (1 / (4 * (1 / 2) * 1))) :=
  le_max_of_le_right (by rw [max_eq_left (by norm_num : (1 / (4 * (1 / 2) * 1) : ℝ) ≤ 1)]; norm_num)

theorem not_bound {m x y : ℝ} (hm : 1 / 2 ≤ m) (hx : 1 ≤ x) (hy : y ≤ 0) : ¬ (m * x ≤ y) :=
  fun h => by nlinarith

theorem mb0_updateHardness (r : RegisteredCircuit) (s : ℕ) (b : ℝ) : Cover.Assumptions.UpdateHardness (mb0 r) s b := by
  intro S _ B hB _
  rw [Finset.subset_empty.1 hB]
  simp

theorem mb0_upstream (r : RegisteredCircuit) (A : Empty → Finset (Fin (Cover.Circuit.ofDecl r).N)) (D' : ℕ) :
    (mb0 r).Upstream A D' where
  disjoint := by intro j; exact j.elim
  disjoint_θ j := j.elim
  upstream j := j.elim

theorem mb0_coverage_zero : Assumptions.Coverage 0 D (mb0 D).P.card := by
  unfold Assumptions.Coverage
  simp [mb0]

theorem mb0_not_coverage_two : ¬ Assumptions.Coverage 2 D (mb0 D).P.card := by
  unfold Assumptions.Coverage
  rw [size_D]
  norm_num [mb0]

theorem D_blockFits : Cover.Assumptions.BlockFits D 2 1 1 4 := by
  refine ⟨by decide, ?_⟩
  have h : (D.D.width D.θ : ℝ) ≤ 1 := by exact_mod_cast (show D.D.width D.θ ≤ 1 by decide)
  push_cast
  linarith

theorem capBelowDepth_one_two : Cover.Assumptions.CapBelowDepth 1 1 2 := by
  unfold Cover.Assumptions.CapBelowDepth
  norm_num

end Nci.Witness

namespace Nci.SecurityProofs

open Verity Finset Nci.Capability.Toy Nci.Witness Nci.Dampener

theorem ThreadedTraining_honest : Guarantees.ThreadedTraining_honest :=
  ⟨compliesRate_D, theta_id D, T1_threaded, T1_weights, coverage_T1⟩

theorem ThreadedTraining_nontrivial : Guarantees.ThreadedTraining_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_D, theta_id D, T0_threaded _ _ _ _ _,
    T0_weights _ _ _ _ _, not_coverage_T0, ?_⟩
  rw [bound_T0, size_D]
  norm_num

theorem ThreadedTrainingOn_nontrivial : Guarantees.ThreadedTrainingOn_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_D, theta_id D, T0_threaded _ _ _ _ _,
    T0_weights _ _ _ _ _, not_coverage_T0, ?_⟩
  rw [bound_T0, size_D, dark_id]
  norm_num

theorem ThreadedThrottled_honest : Guarantees.ThreadedThrottled_honest :=
  ⟨compliesRate_D, theta_id D, T1_threaded, T1_weights⟩

theorem ThreadedThrottled_nontrivial : Guarantees.ThreadedThrottled_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded, T1_weights,
    ?_⟩
  rw [card_T1]
  norm_num

theorem ThreadedThrottledOn_nontrivial : Guarantees.ThreadedThrottledOn_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded, T1_weights,
    ?_⟩
  rw [show (T1.M ∩ Mθ.S).card = 1 by decide]
  norm_num

theorem ThreadedThrottledDark_nontrivial : Guarantees.ThreadedThrottledDark_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded, T1_weights,
    ?_⟩
  rw [card_T1, dark_Mθ]
  norm_num

theorem ThreadedThrottledSlots_honest : Guarantees.ThreadedThrottledSlots_honest :=
  ⟨compliesRate_D, theta_id D, T1_threaded, T1_weightsSlots⟩

theorem ThreadedThrottledSlots_nontrivial : Guarantees.ThreadedThrottledSlots_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded,
    T1_weightsSlots, ?_⟩
  rw [card_T1]
  norm_num

theorem ThreadedThrottledOnSlots_nontrivial : Guarantees.ThreadedThrottledOnSlots_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded,
    T1_weightsSlots, ?_⟩
  rw [show (T1.M ∩ Mθ.S).card = 1 by decide]
  norm_num

theorem ThreadedThrottledDarkSlots_nontrivial : Guarantees.ThreadedThrottledDarkSlots_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded,
    T1_weightsSlots, ?_⟩
  rw [card_T1, dark_Mθ]
  norm_num

theorem ActivationsBoundOn_honest : Guarantees.ActivationsBoundOn_honest :=
  ⟨compliesRate_D, theta_id D, entersAtMost_T1, drawsAtLeast_noLinks _ _, activations_T1⟩

theorem ActivationsBoundOn_nontrivial : Guarantees.ActivationsBoundOn_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ,
    entersAtMost_T1, drawsAtLeast_noLinks _ _, activations_T1, ?_⟩
  rw [show (T1.M ∩ Mθ.S).card = 1 by decide]
  norm_num

theorem ActivationsTrainingOn_honest : Guarantees.ActivationsTrainingOn_honest :=
  ⟨compliesRate_D, theta_id D, entersAtMost_T1, drawsAtLeast_noLinks _ _, activations_T1, coverage_T1⟩

theorem ActivationsTrainingOn_nontrivial : Guarantees.ActivationsTrainingOn_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_D, theta_id D,
    entersAtMost_T0 _, drawsAtLeast_noLinks _ _, activations_T0 _ _ _ _ _, not_coverage_T0, ?_⟩
  rw [bound_T0, size_D, dark_id]
  norm_num

theorem LayeredTraining_honest : Guarantees.LayeredTraining_honest :=
  ⟨⟨layered1 1 2⟩, compliesRate_D, theta_id D, by norm_num [Assumptions.CapBelowColumns], T1_weights, coverage_T1⟩

theorem LayeredTraining_nontrivial : Guarantees.LayeredTraining_nontrivial := by
  refine ⟨⟨layered0 D 1 0 0 2⟩, by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_D, theta_id D,
    by norm_num [Assumptions.CapBelowColumns], T0_weights _ _ _ _ _, not_coverage_T0, ?_⟩
  rw [bound_T0, size_D]
  norm_num

theorem windowsHold_T1 : Assumptions.WindowsHold T1 0 1 (1 / 2) 2 1 1 1 (1 / 2) (1 / 2) 1 0 := by
  unfold Assumptions.WindowsHold
  rw [bound_T1 _ (by norm_num), card_T1]
  norm_num

theorem not_windowsHold_T0 : ¬ Assumptions.WindowsHold (T0 D) 1 1 (1 / 2) 0 1 1 0 (1 / 2) (1 / 2) 1 0 := by
  unfold Assumptions.WindowsHold
  rw [bound_T0, show (T0 D).M.card = 0 from rfl]
  norm_num

theorem dampener_T0_fails :
    ¬ (efficiency 1 1 2 0 ≤
        unregisteredShare 2 0 +
            min (residualShare 0 2) (memoryCap (T0 D).reuse 1 0 ((1 : ℕ) / (2 : ℕ) : ℝ) 1 (1 / 2) (1 / 2) * (1 + 1 / 1)) +
          declaredShare 1 (rhoT ((1 : ℕ) / (2 : ℕ) : ℝ) (1 / 2) ((T0 D).bound 1 (1 / 2) 1 0)) 2 0 + 0) := by
  rw [bound_T0, show (T0 D).reuse = 0 from rfl]
  norm_num [efficiency, unregisteredShare, residualShare, memoryCap, declaredShare, rhoT]

theorem DampenerThreaded_honest : Guarantees.DampenerThreaded_honest := by
  refine ⟨compliesRate_D, theta_id D, T1_threaded, T1_weights, by rw [size_D]; norm_num, by decide,
    by norm_num [Assumptions.WindowLength], ?_, windowsHold_T1⟩
  rw [show T1.reuse = 1 from rfl]
  norm_num

theorem DampenerThreaded_nontrivial : Guarantees.DampenerThreaded_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, compliesRate_D, theta_id D, T0_threaded _ _ _ _ _, T0_weights _ _ _ _ _,
    by rw [size_D]; norm_num, by decide, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num [Assumptions.WindowLength], by norm_num, not_windowsHold_T0, dampener_T0_fails⟩

theorem DampenerThreadedSlots_honest : Guarantees.DampenerThreadedSlots_honest := by
  refine ⟨compliesRate_D, theta_id D, T1_threaded, T1_weightsSlots, by rw [size_D]; norm_num, by decide,
    by norm_num [Assumptions.WindowLength], ?_, windowsHold_T1⟩
  rw [show T1.reuse = 1 from rfl]
  norm_num

theorem DampenerThreadedSlots_nontrivial : Guarantees.DampenerThreadedSlots_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, compliesRate_D, theta_id D, T0_threaded _ _ _ _ _, T0_weightsSlots _ _ _ _ _,
    by rw [size_D]; norm_num, by decide, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num [Assumptions.WindowLength], by norm_num, not_windowsHold_T0, dampener_T0_fails⟩

theorem IncompressibleContainedTraining_honest : Guarantees.IncompressibleContainedTraining_honest :=
  ⟨compliesRate_D, theta_id D, ch1_T1_incompressible, coverage_ch1⟩

theorem IncompressibleContainedTraining_nontrivial : Guarantees.IncompressibleContainedTraining_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, fun ℓ => ℓ.elim0, compliesRate_D, theta_id D,
    ch0_T0_incompressible, not_coverage_ch0, ?_⟩
  have hF := F_ge (α := 1 / 2) (K := 1) (b_w := 1) (b_a := 2) (d := 1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  rw [size_D]
  norm_num at hF ⊢
  linarith

theorem IncompressibleContainedMatMulOn_honest : Guarantees.IncompressibleContainedMatMulOn_honest :=
  ⟨compliesRate_D, theta_id D, darkReads_M1 _, ch1_T1_incompressible⟩

theorem IncompressibleContainedMatMulOn_nontrivial : Guarantees.IncompressibleContainedMatMulOn_nontrivial :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => le_rfl, fun _ => le_rfl, compliesRate_Dθ, not_theta_Dθ, darkReads_Mθ _, ch1_T1_incompressible,
    not_bound factor_half (Nat.one_le_cast.2 (by decide)) (by norm_num)⟩

theorem IncompressibleContainedTrainingOn_honest : Guarantees.IncompressibleContainedTrainingOn_honest :=
  ⟨compliesRate_D, theta_id D, darkReads_M1 _, ch1_T1_incompressible, coverage_ch1⟩

theorem IncompressibleContainedTrainingOn_nontrivial : Guarantees.IncompressibleContainedTrainingOn_nontrivial :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => le_rfl, fun _ => le_rfl, compliesRate_Dθ, not_theta_Dθ, darkReads_Mθ _, ch1_T1_incompressible, coverage_ch1,
    not_bound factor_half (by rw [size_D, dark_Mθ]; norm_num) (by norm_num)⟩

theorem IncompressibleThreadedThrottled_honest : Guarantees.IncompressibleThreadedThrottled_honest :=
  ⟨compliesRate_D, theta_id D, T1_threaded, ch1_T1_incompressible⟩

theorem IncompressibleThreadedThrottled_nontrivial : Guarantees.IncompressibleThreadedThrottled_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded,
    ch1_T1_incompressible, ?_⟩
  rw [card_T1]
  norm_num

theorem IncompressibleThreadedThrottledOn_nontrivial : Guarantees.IncompressibleThreadedThrottledOn_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded,
    ch1_T1_incompressible, ?_⟩
  rw [show (T1.M ∩ Mθ.S).card = 1 by decide]
  norm_num

theorem coverage_pch1 : Assumptions.Coverage 1 D (2 * ch1.toPacked.G) := by
  unfold Assumptions.Coverage
  rw [size_D, show ch1.toPacked.G = 1 from rfl]
  norm_num

theorem PackedIncompressibleContainedMatMulOn_honest : Guarantees.PackedIncompressibleContainedMatMulOn_honest :=
  ⟨compliesRate_D, theta_id D, darkReads_M1 _, pch1_T1_incompressible⟩

theorem PackedIncompressibleContainedMatMulOn_nontrivial :
    Guarantees.PackedIncompressibleContainedMatMulOn_nontrivial :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => le_rfl, fun _ => le_rfl, compliesRate_Dθ, not_theta_Dθ, darkReads_Mθ _, pch1_T1_incompressible,
    not_bound factor_half (Nat.one_le_cast.2 (by decide)) (by norm_num)⟩

theorem PackedIncompressibleContainedTrainingOn_honest :
    Guarantees.PackedIncompressibleContainedTrainingOn_honest :=
  ⟨compliesRate_D, theta_id D, darkReads_M1 _, pch1_T1_incompressible, coverage_pch1⟩

theorem PackedIncompressibleContainedTrainingOn_nontrivial :
    Guarantees.PackedIncompressibleContainedTrainingOn_nontrivial :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => le_rfl, fun _ => le_rfl, compliesRate_Dθ, not_theta_Dθ, darkReads_Mθ _, pch1_T1_incompressible,
    coverage_pch1, not_bound factor_half (by rw [size_D, dark_Mθ]; norm_num) (by norm_num)⟩

theorem oneWindow_withinCompute : Assumptions.WithinCompute OneWindow.cap 2 := by
  norm_num [Assumptions.WithinCompute]

theorem oneWindow_pouwSound : Assumptions.PouwSound (PMF.uniformOfFintype Bool) OneWindow.right 0 :=
  fun _ => (prob_of_forall_not _ fun _ h => h).le

theorem oneWindow_not_pouwSound : ¬ Assumptions.PouwSound (PMF.uniformOfFintype Bool) OneWindow.wrong 0 := by
  intro h
  have h1 := h 0
  have h2 := prob_add_not (PMF.uniformOfFintype Bool) (fun ω => OneWindow.wrong ω 0)
  rw [prob_of_forall_not (PMF.uniformOfFintype Bool) (E := fun ω => ¬ OneWindow.wrong ω 0) fun _ h => h trivial]
    at h2
  linarith

theorem DampenerRemote_honest : Guarantees.DampenerRemote_honest :=
  ⟨oneWindow_withinCompute, oneWindow_pouwSound, fun _ b => b.elim, fun b => b.elim, Cut.SecurityProofs.toy_complies,
    Cut.SecurityProofs.toy_values, Cut.SecurityProofs.toy_noSharedPair, Cut.SecurityProofs.toy_hardness 2,
    Cut.SecurityProofs.toy_exports 2, by show 2 ≤ 0 + 0 + Cut.Toy.W.N + 0; rw [Cut.SecurityProofs.toy_N],
    fun _ => by norm_num [Assumptions.ResidualAtMost, wrongCapacity]⟩

theorem DampenerRemote_nontrivial : Guarantees.DampenerRemote_nontrivial := by
  refine ⟨by norm_num, by norm_num, fun _ => by norm_num, by simp [Assumptions.WithinCompute], oneWindow_not_pouwSound,
    by norm_num, by norm_num, by norm_num, fun _ b => b.elim, fun _ b => b.elim,
    fun _ _ _ => by simp, by norm_num, fun b => b.elim, by norm_num, by norm_num, by norm_num,
    fun _ b => b.elim, fun b => b.elim, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, Cut.SecurityProofs.toy_complies, Cut.SecurityProofs.toy_values,
    Cut.SecurityProofs.toy_noSharedPair, Cut.SecurityProofs.toy_hardness 2, Cut.SecurityProofs.toy_exports 2,
    by norm_num, by show 100 ≤ 0 + 100 + Cut.Toy.W.N + 0; omega, by norm_num,
    fun _ => by norm_num [Assumptions.ResidualAtMost, wrongCapacity], by norm_num, ?_⟩
  intro h
  rw [prob_of_forall_not] at h
  · norm_num at h
  · rintro - ⟨h1, -⟩
    rw [Cut.SecurityProofs.toy_workHE] at h1
    norm_num [efficiency, unregisteredShare, residualShare, lagShare] at h1

end Nci.SecurityProofs

namespace Nci.Cover.SecurityProofs

open Verity Nci.Capability.Toy Nci.Witness

theorem CoverTraining_honest : Guarantees.CoverTraining_honest :=
  ⟨compliesRate_D, theta_id D, mb0_updateHardness D 4 1, mb0_upstream D _ 2, capBelowDepth_one_two, D_blockFits,
    mb0_coverage_zero⟩

theorem CoverTraining_nontrivial : Guarantees.CoverTraining_nontrivial :=
  ⟨by norm_num, by norm_num, compliesRate_D, theta_id D, mb0_updateHardness D 4 1, mb0_upstream D _ 2,
    capBelowDepth_one_two, D_blockFits, mb0_not_coverage_two, by rw [size_D]; norm_num⟩

end Nci.Cover.SecurityProofs
