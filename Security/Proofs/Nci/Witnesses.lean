import Definitions.Nci.Guarantees.Witnesses
import Proofs.Nci.Reads
import Proofs.Nci.Capability.Witnesses
import Proofs.Core.Circuit
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Linarith

/-!
# The inference-only policy's guarantees in `(c, k)`: proofs of their witnesses

Each witness of `Definitions.Nci.Guarantees.Witnesses`, at the toy (`Nci.Capability.Toy`) and the instances of
`Nci.Witness`. The facts they share come first: the toy complies at `(2, 1)`, and with the token registered at `(2, 0)`;
the threading `T1` is `(1, 0)`-threaded and its weights incompressible at half a bit each, read off the toy's sixteen
sets of gates; `T0` has no counted multiplication and bound `e`; and `F` is positive.
-/

namespace Nci.Witness

open Verity Finset Nci.Capability.Toy

theorem complies_D : Complies 2 1 D :=
  ⟨1, P1, { boolean := ⟨fun _ => rfl, by decide⟩, card := by decide, reads := by decide }⟩

theorem complies_Dθ : Complies 2 0 Dθ :=
  ⟨1, P1, { boolean := ⟨fun _ => rfl, by decide⟩, card := by decide, reads := by decide }⟩

theorem boolean_D : D.D.Boolean := ⟨fun _ => rfl, by decide⟩

theorem size_D : D.D.size D.D.computed = 2 := by decide

theorem card_T1 : T1.M.card = 1 := by decide

theorem theta_id (r : Declaration) :
    Assumptions.ThetaRegistered r r (PartialContainment.ofContainment (idC r.D)) := fun _ _ h => h

theorem not_theta_Dθ : ¬ Assumptions.ThetaRegistered D Dθ Mθ :=
  fun h => absurd (h 1 (by decide) (by decide)) (by decide)

theorem dark_id (r : Declaration) : (PartialContainment.ofContainment (c := r) (d := r) (idC r.D)).dark = ∅ :=
  Finset.sdiff_self _

/-- The toy complies at no read cap `0`: its gate `2` reads the token, an unregistered input, which no proof block
holds. -/
theorem not_complies_zero : ¬ Complies 2 0 D := by
  rintro ⟨B, P, hP⟩
  have h2 : P.unit 2 ≠ none := fun h => absurd ((P.input_iff 2).1 h) (by decide)
  obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 h2
  have h1 : P.unit 1 = none := (P.input_iff 1).2 rfl
  have hmem : (1 : Fin 4) ∈ D.unregistered (P.gates u) := by
    simp only [Declaration.unregistered, Circuit.into, Partition.gates, mem_sdiff, mem_filter, mem_univ, true_and]
    refine ⟨⟨by simp [h1], 2, hu, by decide⟩, by decide⟩
  have := hP.reads u
  rw [Nat.le_zero, card_eq_zero] at this
  simp [this] at hmem

theorem T1_threaded : T1.Threaded 2 1 1 0 := by
  have key : ∀ R : Finset (Fin 4), R ⊆ C.computed → ∀ t : Fin 1, (T1.threadReads R t).card < 1 →
      (T1.threadMults R t).card ≤ 0 * (T1.threadReads R t).card := by decide
  exact fun R hR t => key R hR.1 t

theorem T1_weights_nat : ∀ R : Finset (Fin 4), R ⊆ C.computed →
    (T1.weightsUsed R).card + (T1.threadReads R (0 : Fin 1)).card ≤ 2 * novelReads D R := by decide

theorem T1_weights : Assumptions.WeightsIncompressible D T1 2 1 (1 / 2) (1 / 2) := by
  intro R hR
  have h := T1_weights_nat R hR.1
  have hs : ∑ t, (T1.threadReads R t).card = (T1.threadReads R (0 : Fin 1)).card := Fin.sum_univ_one _
  rw [Nat.cast_sum, ← Nat.cast_sum, hs]
  have h' : ((T1.weightsUsed R).card : ℝ) + (T1.threadReads R (0 : Fin 1)).card ≤ 2 * novelReads D R := by exact_mod_cast h
  linarith

theorem T1_weightsSlots : Assumptions.WeightsIncompressibleSlots D T1 σ1 2 1 (1 / 2) (1 / 2) := by
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

theorem T0_threaded (r : Declaration) (c : ℕ) (k : ℝ) (d e : ℕ) : (T0 r).Threaded c k d e :=
  fun _ _ t => t.elim0

theorem T0_weights (r : Declaration) (c : ℕ) (k b_w b_a : ℝ) :
    Assumptions.WeightsIncompressible r (T0 r) c k b_w b_a := by
  intro R _
  have h0 : ∑ t, ((T0 r).threadReads R t).card = 0 := Finset.sum_eq_zero fun t _ => Fin.elim0 t
  rw [h0]
  simp [Threading.weightsUsed, Threading.mults, T0]

theorem T0_weightsSlots (r : Declaration) (c : ℕ) (k b_w b_a : ℝ) :
    Assumptions.WeightsIncompressibleSlots r (T0 r) (σ0 r) c k b_w b_a := by
  intro R _
  have h0 : ∑ t, ((T0 r).threadReads R t).card = 0 := Finset.sum_eq_zero fun t _ => Fin.elim0 t
  rw [h0]
  simp [Threading.Slots.slotsUsed, Threading.weightsUsed, Threading.mults, T0]

theorem not_coverage_T0 : ¬ Assumptions.Coverage 1 D (2 * (T0 D).M.card) := by
  unfold Assumptions.Coverage
  rw [size_D, show (T0 D).M.card = 0 from rfl]
  norm_num

theorem bound_T0 (r : Declaration) (k b_w : ℝ) (d e : ℕ) : (T0 r).bound k b_w d e = e := by
  simp [Threading.bound, T0]

theorem bound_T1 (k : ℝ) (hk : 0 ≤ k) : T1.bound k (1 / 2) 1 0 = k / 2 := by
  have : T1.reuse = 1 := rfl
  simp only [Threading.bound, this, Nat.cast_one, Nat.cast_zero]
  rw [max_eq_right (by positivity)]
  ring

theorem entersAtMost_T1 : Assumptions.EntersAtMost T1 1 := by
  unfold Assumptions.EntersAtMost
  decide

theorem drawsAtLeast_noLinks {r : Declaration} (T : Threading r) (dd : ℕ) :
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

theorem entersAtMost_T0 (r : Declaration) : Assumptions.EntersAtMost (T0 r) 0 :=
  fun _ => (card_le_card (filter_subset _ _)).trans (by simp [T0])

theorem activations_T0 (r : Declaration) (c : ℕ) (k b_w b_a : ℝ) :
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

/-- A declaration whose coverage at `κ` asks nothing is in the class at every parameter: every declaration runs the
chain of no multiplication. -/
theorem inClass_of_le_zero {T : Declaration} {c k : ℕ} {b_w b_a κ : ℝ} {d : ℕ}
    (hT : κ * T.D.size T.D.computed ≤ 0) : Assumptions.InClass T c k b_w b_a κ d := by
  intro D' _ _
  refine ⟨ch0 D'.D, fun ℓ => ℓ.elim0, ?_, ?_, ?_⟩
  · unfold Assumptions.Coverage
    rw [ch0_G]
    simpa using hT
  · intro R _ _ _
    rw [ch0_weightsUsed, ch0_inputsUsed]
    simp
  · intro n P _
    refine ⟨fun _ => 0, le_trans (le_of_eq (Finset.sum_eq_zero fun x _ => x.1.elim0)) (Nat.zero_le _), fun u => ?_⟩
    rw [ch0_weightsUsed]
    simp

theorem size_E0 : E0.D.size E0.D.computed = 0 := rfl

theorem implements_E0 : E0.Implements E0 := ⟨implE0, fun g => g.elim0⟩

theorem complies_E0 (c k : ℕ) : Complies c k E0 :=
  ⟨0, ⟨fun g => g.elim0, fun g => g.elim0⟩,
    { boolean := ⟨fun g => g.elim0, fun g => g.elim0⟩, card := fun u => u.elim0, reads := fun u => u.elim0 }⟩

/-- The slowdown factor of the declaration of no gates is `0`: it implements itself and complies, at `0 / 0 = 0`. -/
theorem slowdown_E0 (c k : ℕ) : slowdown (Complies c k) E0 = 0 :=
  nonpos_iff_eq_zero.1 ((iInf₂_le E0 ⟨implements_E0, complies_E0 c k⟩).trans_eq (by rw [size_E0]; simp))

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

theorem ch1_activations {b_w b_a : ℝ} (hw : 0 ≤ b_w) (ha : 0 ≤ b_a) (hs : b_w + b_a ≤ 1) :
    Assumptions.ActivationsIncompressible D ch1 2 1 b_w b_a := by
  intro R hR _ _
  simp only [ch1_notLinked, not_false_eq_true, filter_const, ↓reduceIte]
  obtain ⟨hw', hi'⟩ := ch1_nat R hR
  have hw'' : ((ch1.weightsUsed R).card : ℝ) ≤ novelReads D R := by exact_mod_cast hw'
  have hi'' : ((ch1.inputsUsed R).card : ℝ) ≤ novelReads D R := by exact_mod_cast hi'
  have h0 : (0 : ℝ) ≤ novelReads D R := Nat.cast_nonneg _
  nlinarith

theorem ch1_partialSums {b_w b_a : ℝ} (hw : b_w ≤ 1) :
    Assumptions.PartialSumsIncompressible D ch1 2 1 b_w b_a := by
  intro n P _
  refine ⟨fun _ => 0, ?_, fun u => ?_⟩
  · refine le_trans (Finset.sum_le_sum fun x _ => Nat.sub_le_of_le_add (ch1_spread P x)) ?_
    simp
  · have h := (ch1_nat _ (P.gates_subset_computed u)).1
    have h' : ((ch1.weightsUsed (P.gates u)).card : ℝ) ≤ novelReads D (P.gates u) := by exact_mod_cast h
    have h0 : (0 : ℝ) ≤ (ch1.weightsUsed (P.gates u)).card := Nat.cast_nonneg _
    simp only [CharP.cast_eq_zero, mul_zero, add_zero]
    nlinarith

theorem not_ch1_activations : ¬ Assumptions.ActivationsIncompressible D ch1 2 1 1 2 := by
  intro h
  have := h {2} (by decide) (by decide) (by norm_num [show novelReads D {2} = 1 by decide])
  simp only [ch1_notLinked, not_false_eq_true, filter_const, ↓reduceIte] at this
  rw [show (ch1.weightsUsed {2}).card = 1 by decide,
    show (ch1.inputsUsed {2}).card = 1 by decide, show novelReads D {2} = 1 by decide] at this
  norm_num at this

theorem smallBlocks_P1 : Assumptions.SmallBlocks D P1 2 1 := by
  have key : ∀ u : Fin 1, D.D.size (P1.gates u) ≤ 2 ∧ novelReads D (P1.gates u) ≤ 1 := by decide
  exact fun u => ⟨(key u).1, by exact_mod_cast (key u).2⟩

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

/-- The toy's run with the token `0`: admissible, the weight at its registered value. -/
def tokenZero : C.Transcript := fun g => if g = 1 then 0 else 1

theorem support_pos (T : Finset (Fin 4)) : 1 ≤ support D ones T :=
  card_pos.2 (Finset.Nonempty.image ⟨ones, mem_filter.2 ⟨mem_univ _, fun _ _ => rfl⟩⟩ _)

/-- A set holding gate `2` takes two values over the toy's admissible runs: `2` holds the token. -/
theorem support_two (T : Finset (Fin 4)) (h : 2 ∈ T) : 2 ≤ support D ones T := by
  have hones : ones ∈ admissible D ones := mem_filter.2 ⟨mem_univ _, fun _ _ => rfl⟩
  have hzero : tokenZero ∈ admissible D ones := mem_filter.2 ⟨mem_univ _, by decide⟩
  have hne : values (D := C) T ones ≠ values (D := C) T tokenZero := fun he => by
    have h2 := congrArg (fun v => BitVec.getLsbD v 0) (congrFun he ⟨2, h⟩)
    simp only [values] at h2
    rw [eval_two, eval_two] at h2
    exact absurd h2 (by decide)
  exact one_lt_card.2 ⟨_, mem_image_of_mem _ hones, _, mem_image_of_mem _ hzero, hne⟩

theorem rpow_le_support (T : Finset (Fin 4)) {x : ℝ} (hx : x ≤ 1) (h0 : 2 ∉ T → x ≤ 0) :
    (2 : ℝ) ^ x ≤ support D ones T := by
  by_cases h : 2 ∈ T
  · calc (2 : ℝ) ^ x ≤ 2 ^ (1 : ℝ) := Real.rpow_le_rpow_of_exponent_le (by norm_num) hx
      _ = 2 := Real.rpow_one 2
      _ ≤ support D ones T := by exact_mod_cast support_two T h
  · calc (2 : ℝ) ^ x ≤ 2 ^ (0 : ℝ) := Real.rpow_le_rpow_of_exponent_le (by norm_num) (h0 h)
      _ = 1 := Real.rpow_zero 2
      _ ≤ support D ones T := by exact_mod_cast support_pos T

theorem ch1_count : ∀ T : Finset (Fin 4), (ch1.weightsUsed T).card = (if 2 ∈ T then 1 else 0) ∧
    (ch1.inputsUsed T).card = (if 2 ∈ T then 1 else 0) := by decide

theorem ch1_charge (R T : Finset (Fin 4)) (b_w b_a : ℝ) :
    ch1.charge R b_w b_a T = (b_w + b_a) * ((if 2 ∈ T then 1 else 0 : ℕ) : ℝ) := by
  unfold Chain.charge
  simp only [ch1_notLinked, not_false_eq_true, filter_const, ↓reduceIte]
  rw [(ch1_count T).1, (ch1_count T).2]
  ring

theorem ch1_activationsSupport (k : ℝ) : Assumptions.ActivationsSupport D ones ch1 k (1 / 2) (1 / 2) := by
  intro R _ T _ _
  rw [ch1_charge]
  exact rpow_le_support T (by split_ifs <;> norm_num) fun h => by simp [h]

theorem ch1_partialSumsSupport (c : ℕ) (k : ℝ) : Assumptions.PartialSumsSupport D ones ch1 c k (1 / 2) (1 / 2) := by
  intro n P _
  refine ⟨fun _ => 0, ?_, fun u => ?_⟩
  · refine le_trans (Finset.sum_le_sum fun x _ => Nat.sub_le_of_le_add (ch1_spread P x)) ?_
    simp
  · simp only [Nat.cast_zero, mul_zero, add_zero]
    rw [(ch1_count _).1]
    exact rpow_le_support _ (by split_ifs <;> norm_num) fun h => by simp [h]

theorem ch0_activationsSupport (k b_w b_a : ℝ) : Assumptions.ActivationsSupport D ones (ch0 C) k b_w b_a := by
  intro R _ T _ _
  rw [show (ch0 C).charge R b_w b_a T = 0 by simp [Chain.charge, ch0_weightsUsed, ch0_inputsUsed], Real.rpow_zero]
  exact_mod_cast support_pos T

theorem ch0_partialSumsSupport (c : ℕ) (k b_w b_a : ℝ) :
    Assumptions.PartialSumsSupport D ones (ch0 C) c k b_w b_a := by
  intro n P _
  refine ⟨fun _ => 0, le_trans (le_of_eq (Finset.sum_eq_zero fun x _ => x.1.elim0)) (Nat.zero_le _), fun u => ?_⟩
  rw [ch0_weightsUsed]
  simp only [card_empty, Nat.cast_zero, mul_zero, add_zero, Real.rpow_zero]
  exact_mod_cast support_pos _

theorem ch1_T1_incompressible : Assumptions.TrainingIncompressible D ch1 T1 2 1 (1 / 2) (1 / 2) := by
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

end Nci.Witness

namespace Nci.SecurityProofs

open Verity Finset Nci.Capability.Toy Nci.Witness Nci.Dampener

theorem ThreadedTraining_honest : Guarantees.ThreadedTraining_honest :=
  ⟨complies_D, theta_id D, T1_threaded, T1_weights, coverage_T1⟩

theorem ThreadedTraining_nontrivial : Guarantees.ThreadedTraining_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, complies_D, theta_id D, T0_threaded _ _ _ _ _,
    T0_weights _ _ _ _ _, not_coverage_T0, ?_⟩
  rw [bound_T0, size_D]
  norm_num

theorem ThreadedTrainingOn_nontrivial : Guarantees.ThreadedTrainingOn_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, complies_D, theta_id D, T0_threaded _ _ _ _ _,
    T0_weights _ _ _ _ _, not_coverage_T0, ?_⟩
  rw [bound_T0, size_D, dark_id]
  norm_num

theorem ActivationsBoundOn_honest : Guarantees.ActivationsBoundOn_honest :=
  ⟨complies_D, theta_id D, entersAtMost_T1, drawsAtLeast_noLinks _ _, activations_T1⟩

theorem ActivationsBoundOn_nontrivial : Guarantees.ActivationsBoundOn_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, complies_Dθ, not_theta_Dθ,
    entersAtMost_T1, drawsAtLeast_noLinks _ _, activations_T1, ?_⟩
  rw [show (T1.M ∩ Mθ.S).card = 1 by decide]
  norm_num

theorem ActivationsTrainingOn_honest : Guarantees.ActivationsTrainingOn_honest :=
  ⟨complies_D, theta_id D, entersAtMost_T1, drawsAtLeast_noLinks _ _, activations_T1, coverage_T1⟩

theorem ActivationsTrainingOn_nontrivial : Guarantees.ActivationsTrainingOn_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, complies_D, theta_id D,
    entersAtMost_T0 _, drawsAtLeast_noLinks _ _, activations_T0 _ _ _ _ _, not_coverage_T0, ?_⟩
  rw [bound_T0, size_D, dark_id]
  norm_num

theorem LayeredTraining_honest : Guarantees.LayeredTraining_honest :=
  ⟨⟨layered1 1 2⟩, complies_D, theta_id D, by norm_num [Assumptions.CapBelowColumns], T1_weights, coverage_T1⟩

theorem LayeredTraining_nontrivial : Guarantees.LayeredTraining_nontrivial := by
  refine ⟨⟨layered0 D 1 0 0 2⟩, by norm_num, by norm_num, by norm_num, by norm_num, complies_D, theta_id D,
    by norm_num [Assumptions.CapBelowColumns], T0_weights _ _ _ _ _, not_coverage_T0, ?_⟩
  rw [bound_T0, size_D]
  norm_num

theorem windowsHold_T1 : Assumptions.WindowsHold T1 2 1 0 1 2 1 1 1 (1 / 2) (1 / 2) 1 0 := by
  unfold Assumptions.WindowsHold
  rw [bound_T1 _ (by norm_num), card_T1]
  norm_num

theorem not_windowsHold_T0 : ¬ Assumptions.WindowsHold (T0 D) 2 1 1 1 0 1 1 0 (1 / 2) (1 / 2) 1 0 := by
  unfold Assumptions.WindowsHold
  rw [bound_T0, show (T0 D).M.card = 0 from rfl]
  norm_num

theorem dampener_T0_fails :
    ¬ (efficiency 1 1 2 0 ≤
        unmonitoredShare 2 0 +
            min (residualShare 0 2) (memoryCap (T0 D).reuse 1 0 ((1 : ℕ) / (2 : ℕ) : ℝ) 1 (1 / 2) (1 / 2) * (1 + 1 / 1)) +
          declaredShare 1 (rhoT ((1 : ℕ) / (2 : ℕ) : ℝ) (1 / 2) ((T0 D).bound 1 (1 / 2) 1 0)) 2 0 + 0) := by
  rw [bound_T0, show (T0 D).reuse = 0 from rfl]
  norm_num [efficiency, unmonitoredShare, residualShare, memoryCap, declaredShare, rhoT]

theorem DampenerThreaded_honest : Guarantees.DampenerThreaded_honest := by
  refine ⟨complies_D, theta_id D, T1_threaded, T1_weights, by rw [size_D]; norm_num, by decide,
    by norm_num [Assumptions.WindowLength], ?_, windowsHold_T1⟩
  rw [show T1.reuse = 1 from rfl]
  norm_num

theorem DampenerThreaded_nontrivial : Guarantees.DampenerThreaded_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, complies_D, theta_id D, T0_threaded _ _ _ _ _, T0_weights _ _ _ _ _,
    by rw [size_D]; norm_num, by decide, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num [Assumptions.WindowLength], by norm_num, not_windowsHold_T0, dampener_T0_fails⟩

theorem DampenerThreadedSlots_honest : Guarantees.DampenerThreadedSlots_honest := by
  refine ⟨complies_D, theta_id D, T1_threaded, T1_weightsSlots, by rw [size_D]; norm_num, by decide,
    by norm_num [Assumptions.WindowLength], ?_, windowsHold_T1⟩
  rw [show T1.reuse = 1 from rfl]
  norm_num

theorem DampenerThreadedSlots_nontrivial : Guarantees.DampenerThreadedSlots_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, complies_D, theta_id D, T0_threaded _ _ _ _ _, T0_weightsSlots _ _ _ _ _,
    by rw [size_D]; norm_num, by decide, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num [Assumptions.WindowLength], by norm_num, not_windowsHold_T0, dampener_T0_fails⟩

theorem Inference_honest : Guarantees.Inference_honest :=
  ⟨boolean_D, by decide, by decide, by decide⟩

theorem Inference_nontrivial : Guarantees.Inference_nontrivial :=
  ⟨boolean_D, fun ⟨_, h, _⟩ => absurd (h 0) (by decide), not_complies_zero⟩

theorem Training_honest : Guarantees.Training_honest :=
  ⟨by decide, inClass_of_le_zero (by norm_num)⟩

theorem Training_nontrivial : Guarantees.Training_nontrivial := by
  refine ⟨by decide, by norm_num, by norm_num, by norm_num, by norm_num,
    inClass_of_le_zero (by rw [size_E0]; norm_num), ?_⟩
  rw [slowdown_E0, nonpos_iff_eq_zero, ENNReal.ofReal_eq_zero]
  have hF := F_ge (α := ((1 : ℕ) / (2 : ℕ) : ℝ)) (K := 1) (b_w := 1) (b_a := 1) (d := 1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  norm_num at hF ⊢
  linarith

theorem UpdatedMatMul_honest : Guarantees.UpdatedMatMul_honest :=
  ⟨complies_D, fun _ => le_rfl, ch1_activations (by norm_num) (by norm_num) (by norm_num),
    ch1_partialSums (by norm_num)⟩

theorem UpdatedMatMul_nontrivial : Guarantees.UpdatedMatMul_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, fun _ => le_rfl, complies_D, not_ch1_activations,
    ch1_partialSums le_rfl, ?_⟩
  have hF := F_ge (α := ((1 : ℕ) / (2 : ℕ) : ℝ)) (K := 1) (b_w := 1) (b_a := 2) (d := 1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  rw [ch1_G, size_D]
  norm_num at hF ⊢
  linarith

theorem UpdatedMatMulReads_honest : Guarantees.UpdatedMatMulReads_honest :=
  ⟨fun _ => le_rfl, smallBlocks_P1, ch1_activations (by norm_num) (by norm_num) (by norm_num),
    ch1_partialSums (by norm_num)⟩

theorem UpdatedMatMulReads_nontrivial : Guarantees.UpdatedMatMulReads_nontrivial := by
  refine ⟨by norm_num, by norm_num, fun _ => le_rfl, smallBlocks_P1, not_ch1_activations, ch1_partialSums le_rfl, ?_⟩
  have hF := readFactor_ge (K := 1) (b_w := 1) (b_a := 2) (d := 1) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)
  rw [ch1_G, Fin.sum_univ_one, show novelReads D (P1.gates 0) = 1 by decide]
  norm_num at hF ⊢
  linarith

theorem ContainedTraining_honest : Guarantees.ContainedTraining_honest :=
  ⟨complies_D, theta_id D, ch1_activationsSupport _, ch1_partialSumsSupport _ _, coverage_ch1⟩

theorem ContainedTraining_nontrivial : Guarantees.ContainedTraining_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, fun ℓ => ℓ.elim0, complies_D, theta_id D,
    ch0_activationsSupport _ _ _, ch0_partialSumsSupport _ _ _ _, not_coverage_ch0, ?_⟩
  have hF := F_ge (α := ((1 : ℕ) / (2 : ℕ) : ℝ)) (K := 1) (b_w := 1) (b_a := 2) (d := 1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  rw [size_D]
  norm_num at hF ⊢
  linarith

theorem ContainedMatMulOn_honest : Guarantees.ContainedMatMulOn_honest :=
  ⟨complies_D, theta_id D, darkReads_M1 _, ch1_activationsSupport _, ch1_partialSumsSupport _ _⟩

theorem ContainedMatMulOn_nontrivial : Guarantees.ContainedMatMulOn_nontrivial :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => le_rfl, fun _ => le_rfl, complies_Dθ, not_theta_Dθ, darkReads_Mθ _, ch1_activationsSupport _,
    ch1_partialSumsSupport _ _, not_bound factor_half (Nat.one_le_cast.2 (by decide)) (by norm_num)⟩

theorem ContainedTrainingOn_honest : Guarantees.ContainedTrainingOn_honest :=
  ⟨complies_D, theta_id D, darkReads_M1 _, ch1_activationsSupport _, ch1_partialSumsSupport _ _, coverage_ch1⟩

theorem ContainedTrainingOn_nontrivial : Guarantees.ContainedTrainingOn_nontrivial :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => le_rfl, fun _ => le_rfl, complies_Dθ, not_theta_Dθ, darkReads_Mθ _, ch1_activationsSupport _,
    ch1_partialSumsSupport _ _, coverage_ch1,
    not_bound factor_half (by rw [size_D, dark_Mθ]; norm_num) (by norm_num)⟩

theorem IncompressibleContainedTraining_honest : Guarantees.IncompressibleContainedTraining_honest :=
  ⟨complies_D, theta_id D, ch1_T1_incompressible, coverage_ch1⟩

theorem IncompressibleContainedTraining_nontrivial : Guarantees.IncompressibleContainedTraining_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, fun ℓ => ℓ.elim0, complies_D, theta_id D,
    ch0_T0_incompressible, not_coverage_ch0, ?_⟩
  have hF := F_ge (α := ((1 : ℕ) / (2 : ℕ) : ℝ)) (K := 1) (b_w := 1) (b_a := 2) (d := 1)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  rw [size_D]
  norm_num at hF ⊢
  linarith

theorem IncompressibleContainedMatMulOn_honest : Guarantees.IncompressibleContainedMatMulOn_honest :=
  ⟨complies_D, theta_id D, darkReads_M1 _, ch1_T1_incompressible⟩

theorem IncompressibleContainedMatMulOn_nontrivial : Guarantees.IncompressibleContainedMatMulOn_nontrivial :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => le_rfl, fun _ => le_rfl, complies_Dθ, not_theta_Dθ, darkReads_Mθ _, ch1_T1_incompressible,
    not_bound factor_half (Nat.one_le_cast.2 (by decide)) (by norm_num)⟩

theorem IncompressibleContainedTrainingOn_honest : Guarantees.IncompressibleContainedTrainingOn_honest :=
  ⟨complies_D, theta_id D, darkReads_M1 _, ch1_T1_incompressible, coverage_ch1⟩

theorem IncompressibleContainedTrainingOn_nontrivial : Guarantees.IncompressibleContainedTrainingOn_nontrivial :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => le_rfl, fun _ => le_rfl, complies_Dθ, not_theta_Dθ, darkReads_Mθ _, ch1_T1_incompressible, coverage_ch1,
    not_bound factor_half (by rw [size_D, dark_Mθ]; norm_num) (by norm_num)⟩

theorem coverage_pch1 : Assumptions.Coverage 1 D (2 * ch1.toPacked.G) := by
  unfold Assumptions.Coverage
  rw [size_D, show ch1.toPacked.G = 1 from rfl]
  norm_num

theorem PackedIncompressibleContainedMatMulOn_honest : Guarantees.PackedIncompressibleContainedMatMulOn_honest :=
  ⟨complies_D, theta_id D, darkReads_M1 _, pch1_T1_incompressible⟩

theorem PackedIncompressibleContainedMatMulOn_nontrivial :
    Guarantees.PackedIncompressibleContainedMatMulOn_nontrivial :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => le_rfl, fun _ => le_rfl, complies_Dθ, not_theta_Dθ, darkReads_Mθ _, pch1_T1_incompressible,
    not_bound factor_half (Nat.one_le_cast.2 (by decide)) (by norm_num)⟩

theorem PackedIncompressibleContainedTrainingOn_honest :
    Guarantees.PackedIncompressibleContainedTrainingOn_honest :=
  ⟨complies_D, theta_id D, darkReads_M1 _, pch1_T1_incompressible, coverage_pch1⟩

theorem PackedIncompressibleContainedTrainingOn_nontrivial :
    Guarantees.PackedIncompressibleContainedTrainingOn_nontrivial :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => le_rfl, fun _ => le_rfl, complies_Dθ, not_theta_Dθ, darkReads_Mθ _, pch1_T1_incompressible,
    coverage_pch1, not_bound factor_half (by rw [size_D, dark_Mθ]; norm_num) (by norm_num)⟩

end Nci.SecurityProofs
