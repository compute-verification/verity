import Proofs.Accounting.ServingBlocks
import Definitions.Accounting.Guarantees.ServingBlocks.Straddle

/-!
# The uncertified cap across the certified boundary: proofs

Two ways of paying for a family of counted operations, each with every gate in at most `μ` bodies:
* `charge_le`: each has a gate of its body in `T`, so they weigh at most `ω_max μ |T|`;
* `capacity_le`: each has at least `s` gates of its body in `U`, so they weigh at most `ω_max μ |U| / s`.

What the blocks leave of `R` splits by where an operation's representative is and whether its body is inside `R`'s
uncertified gates `U`. Inside `U`, `capacity_le` pays at `s` gates each. Straddling with its representative in `U`,
`exists_crossing` finds a gate of its body in `in(U) \ θ`; with its representative in a block `B i`, one in
`in(B i) \ θ` (`blocks_charge`).
-/

namespace Accounting.SecurityProofs.ServingBlocks

open Verity Finset Accounting.ServingBlocks

/-- **The charge**: counted operations `F`, each with a gate of its body in `T`, weigh at most `ω_max μ |T|`. -/
theorem charge_le {C : Circuit} (J : Counted C) (F : Finset (Fin C.N)) (hF : F ⊆ J.V) (T : Finset (Fin C.N))
    (ωmax : ℝ) (μ : ℕ) (hμ : Overlap J μ) (hω0 : 0 ≤ ωmax) (hω : ∀ v ∈ J.V, J.ω v ≤ ωmax)
    (hT : ∀ v ∈ F, ∃ a ∈ T, a ∈ J.body v) :
    ∑ v ∈ F, J.ω v ≤ ωmax * μ * T.card := by
  classical
  calc ∑ v ∈ F, J.ω v ≤ ∑ a ∈ T, ∑ v ∈ F with a ∈ J.body v, J.ω v :=
        sum_le_sum_sum_filter F T (fun a v => a ∈ J.body v) J.ω (fun v hv => J.ω_nonneg v (hF hv)) hT
    _ ≤ ∑ _a ∈ T, ωmax * μ := by
        refine Finset.sum_le_sum fun a _ => ?_
        calc _ ≤ ∑ v ∈ J.V with a ∈ J.body v, J.ω v :=
              Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset_filter _ hF)
                fun v hv _ => J.ω_nonneg v (Finset.mem_of_mem_filter v hv)
          _ ≤ ∑ v ∈ J.V with a ∈ J.body v, ωmax :=
              Finset.sum_le_sum fun v hv => hω v (Finset.mem_of_mem_filter v hv)
          _ = (J.V.filter fun v => a ∈ J.body v).card * ωmax := by rw [Finset.sum_const, nsmul_eq_mul]
          _ ≤ μ * ωmax := mul_le_mul_of_nonneg_right (by exact_mod_cast hμ a) hω0
          _ = ωmax * μ := mul_comm _ _
    _ = ωmax * μ * T.card := by rw [Finset.sum_const, nsmul_eq_mul, mul_comm]

/-- **The capacity**: counted operations `F`, each with at least `s > 0` gates of its body in `U`, weigh at most
`ω_max μ |U| / s`: counted with their gates in `U`, they hold at most `μ |U|` gates. -/
theorem capacity_le {C : Circuit} (J : Counted C) (F : Finset (Fin C.N)) (hF : F ⊆ J.V) (U : Finset (Fin C.N))
    (ωmax : ℝ) (μ s : ℕ) (hs : 0 < s) (hμ : Overlap J μ) (hω0 : 0 ≤ ωmax) (hω : ∀ v ∈ J.V, J.ω v ≤ ωmax)
    (hsize : ∀ v ∈ F, s ≤ (J.body v ∩ U).card) :
    ∑ v ∈ F, J.ω v ≤ ωmax * μ * U.card / s := by
  classical
  have hcount : ∑ v ∈ F, (J.body v ∩ U).card ≤ μ * U.card := by
    have hcard : ∀ v, (J.body v ∩ U).card = ∑ a ∈ U, if a ∈ J.body v then 1 else 0 := fun v => by
      rw [← Finset.card_filter, Finset.filter_mem_eq_inter, Finset.inter_comm]
    calc ∑ v ∈ F, (J.body v ∩ U).card = ∑ a ∈ U, (F.filter fun v => a ∈ J.body v).card := by
          simp only [hcard, Finset.card_filter]
          exact Finset.sum_comm
      _ ≤ ∑ _a ∈ U, μ := Finset.sum_le_sum fun a _ =>
          (Finset.card_le_card (Finset.filter_subset_filter _ hF)).trans (hμ a)
      _ = μ * U.card := by rw [Finset.sum_const, smul_eq_mul, mul_comm]
  rw [le_div_iff₀ (by exact_mod_cast hs : (0 : ℝ) < s), Finset.sum_mul]
  calc ∑ v ∈ F, J.ω v * s ≤ ∑ v ∈ F, ωmax * ((J.body v ∩ U).card : ℝ) := Finset.sum_le_sum fun v hv =>
        (mul_le_mul_of_nonneg_right (hω v (hF hv)) (Nat.cast_nonneg s)).trans
          (mul_le_mul_of_nonneg_left (by exact_mod_cast hsize v hv) hω0)
    _ = ωmax * ((∑ v ∈ F, (J.body v ∩ U).card : ℕ) : ℝ) := by rw [Nat.cast_sum, Finset.mul_sum]
    _ ≤ ωmax * ((μ * U.card : ℕ) : ℝ) := mul_le_mul_of_nonneg_left (by exact_mod_cast hcount) hω0
    _ = ωmax * μ * U.card := by push_cast; ring

theorem mem_uncertified {C : Circuit} {R : Finset (Fin C.N)} {ι : Type*} [Fintype ι] {B : ι → Finset (Fin C.N)}
    {g : Fin C.N} : g ∈ uncertified R B ↔ g ∈ R ∧ ∀ i, g ∉ B i := by
  simp [uncertified]

/-- **The blocks' charge**: counted operations `F`, each with its representative in a block `B i` and computed wholly
inside none, weigh at most `ω_max μ Σ_i |in(B i) \ θ|` when the bodies lead to their representatives. -/
theorem blocks_charge {d : Declaration} (J : Counted d.D) {ι : Type*} [Fintype ι] (B : ι → Finset (Fin d.D.N))
    (F : Finset (Fin d.D.N)) (hF : F ⊆ J.V) (ωmax : ℝ) (μ : ℕ) (hB : OperationBodies d J) (hμ : Overlap J μ)
    (hω0 : 0 ≤ ωmax) (hω : ∀ v ∈ J.V, J.ω v ≤ ωmax) (hin : ∀ v ∈ F, ∃ i, v ∈ B i)
    (hout : ∀ v ∈ F, ∀ i, ¬ J.body v ⊆ B i) :
    ∑ v ∈ F, J.ω v ≤ ωmax * μ * ∑ i, (d.unregistered (B i)).card := by
  classical
  calc ∑ v ∈ F, J.ω v ≤ ∑ i, ∑ v ∈ F with v ∈ B i, J.ω v :=
        sum_le_sum_sum_filter F univ (fun i v => v ∈ B i) J.ω (fun v hv => J.ω_nonneg v (hF hv))
          fun v hv => (hin v hv).imp fun i hi => ⟨Finset.mem_univ i, hi⟩
    _ ≤ ∑ i, ωmax * μ * ((d.unregistered (B i)).card : ℝ) := Finset.sum_le_sum fun i _ =>
        charge_le J _ ((Finset.filter_subset _ _).trans hF) _ ωmax μ hμ hω0 hω fun v hv => by
          obtain ⟨hvF, hvi⟩ := Finset.mem_filter.1 hv
          exact exists_crossing hB (hF hvF) hvi (hout v hvF i)
    _ = ωmax * μ * ∑ i, (d.unregistered (B i)).card := by rw [← Finset.mul_sum, Nat.cast_sum]

/-- The operations what the blocks leave of `R` sums over. -/
theorem outside_filter {C : Circuit} {J : Counted C} {R : Finset (Fin C.N)} {ι : Type*} [Fintype ι]
    {B : ι → Finset (Fin C.N)} {v : Fin C.N} :
    v ∈ J.V.filter (fun v => v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i) ↔ v ∈ J.V ∧ v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i :=
  Finset.mem_filter

theorem UncertifiedStructural : Guarantees.ServingBlocks.UncertifiedStructural := by
  intro C J R ι _ B ωmax μ s hs hω0 hω hμ hsize hno
  exact capacity_le J _ (Finset.filter_subset _ _) _ ωmax μ s hs hμ hω0 hω fun v hv => by
    obtain ⟨hvV, hvR, hout⟩ := outside_filter.1 hv
    rw [Finset.inter_eq_left.2 (hno v hvV hvR hout)]
    exact hsize v hvV

theorem StraddleBound : Guarantees.ServingBlocks.StraddleBound := by
  intro d J R ι _ B ωmax μ hB hμ hω0 hω
  classical
  set F := J.V.filter fun v => v ∈ R ∧ (∀ i, ¬ J.body v ⊆ B i) ∧ ¬ J.body v ⊆ uncertified R B
  have hF : F ⊆ J.V := Finset.filter_subset _ _
  unfold Counted.straddle crossing
  rw [← Finset.sum_filter_add_sum_filter_not F (· ∈ uncertified R B)]
  have hU : ∑ v ∈ F with v ∈ uncertified R B, J.ω v ≤
      ωmax * μ * (d.unregistered (uncertified R B)).card :=
    charge_le J _ ((Finset.filter_subset _ _).trans hF) _ ωmax μ hμ hω0 hω fun v hv => by
      obtain ⟨hvF, hvU⟩ := Finset.mem_filter.1 hv
      obtain ⟨hvV, -, -, hnot⟩ := Finset.mem_filter.1 hvF
      exact exists_crossing hB hvV hvU hnot
  have hBl : ∑ v ∈ F with v ∉ uncertified R B, J.ω v ≤ ωmax * μ * ∑ i, (d.unregistered (B i)).card :=
    blocks_charge J B _ ((Finset.filter_subset _ _).trans hF) ωmax μ hB hμ hω0 hω
      (fun v hv => by
        obtain ⟨hvF, hvU⟩ := Finset.mem_filter.1 hv
        obtain ⟨-, hvR, -⟩ := Finset.mem_filter.1 hvF
        by_contra h
        exact hvU (mem_uncertified.2 ⟨hvR, fun i hi => h ⟨i, hi⟩⟩))
      fun v hv => (Finset.mem_filter.1 (Finset.mem_filter.1 hv).1).2.2.1
  push_cast at hBl ⊢
  linarith

theorem UncertifiedCrossing : Guarantees.ServingBlocks.UncertifiedCrossing := by
  intro d J R ι _ B ωmax μ s hs hB hμ hsize hω0 hω
  classical
  unfold UncertifiedCap Counted.outside
  set F := J.V.filter fun v => v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i
  have hF : F ⊆ J.V := Finset.filter_subset _ _
  rw [← Finset.sum_filter_add_sum_filter_not F (fun v => J.body v ⊆ uncertified R B)]
  have hin : ∑ v ∈ F with J.body v ⊆ uncertified R B, J.ω v ≤ ωmax * μ * (uncertified R B).card / s :=
    capacity_le J _ ((Finset.filter_subset _ _).trans hF) _ ωmax μ s hs hμ hω0 hω fun v hv => by
      obtain ⟨hvF, hsub⟩ := Finset.mem_filter.1 hv
      rw [Finset.inter_eq_left.2 hsub]
      exact hsize v (hF hvF)
  have hst : ∑ v ∈ F with ¬ J.body v ⊆ uncertified R B, J.ω v = J.straddle R B := by
    unfold Counted.straddle
    rw [Finset.filter_filter]
    exact Finset.sum_congr (Finset.filter_congr fun v _ => by tauto) fun _ _ => rfl
  have := StraddleBound d J R B ωmax μ hB hμ hω0 hω
  rw [hst]
  calc _ ≤ ωmax * μ * (uncertified R B).card / s + ωmax * μ * crossing d R B := add_le_add hin this
    _ = _ := by ring

theorem UncertifiedLocal : Guarantees.ServingBlocks.UncertifiedLocal := by
  intro d J R ι _ B ωmax μ s hs hB hμ hsize hω0 hω
  classical
  unfold UncertifiedCap Counted.outside
  set F := J.V.filter fun v => v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i
  have hF : F ⊆ J.V := Finset.filter_subset _ _
  rw [← Finset.sum_filter_add_sum_filter_not F (· ∈ uncertified R B)]
  have hU : ∑ v ∈ F with v ∈ uncertified R B, J.ω v ≤ ωmax * μ * (uncertified R B).card / s :=
    capacity_le J _ ((Finset.filter_subset _ _).trans hF) _ ωmax μ s hs hμ hω0 hω fun v hv => by
      obtain ⟨hvF, hvU⟩ := Finset.mem_filter.1 hv
      exact hsize v (hF hvF) hvU
  have hBl : ∑ v ∈ F with v ∉ uncertified R B, J.ω v ≤ ωmax * μ * ∑ i, (d.unregistered (B i)).card :=
    blocks_charge J B _ ((Finset.filter_subset _ _).trans hF) ωmax μ hB hμ hω0 hω
      (fun v hv => by
        obtain ⟨hvF, hvU⟩ := Finset.mem_filter.1 hv
        obtain ⟨-, hvR, -⟩ := Finset.mem_filter.1 hvF
        by_contra h
        exact hvU (mem_uncertified.2 ⟨hvR, fun i hi => h ⟨i, hi⟩⟩))
      fun v hv => (Finset.mem_filter.1 (Finset.mem_filter.1 hv).1).2.2
  calc _ ≤ ωmax * μ * (uncertified R B).card / s + ωmax * μ * ∑ i, (d.unregistered (B i)).card :=
        add_le_add hU hBl
    _ = _ := by push_cast; ring

end Accounting.SecurityProofs.ServingBlocks
