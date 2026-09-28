import Proofs.Flock.Soundness.Game.Runs
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# Extraction by filling: the extractor's own table

The extractor reruns the prover `K` times and takes, at each position, the first opening one of its
runs made there, and a fixed row `z₀` where none did (`firstTable`). A fresh run opens a position
off that table only if it conflicts with the run that supplied the row, or opens a position no
extractor run opened (`off_firstTable_le`). Over the `K` independent runs, the first costs
`K·Pr₂[conflict]` (a union over the `K` fresh–extractor pairs, each distributed as two independent
runs) and the second costs `Σ_p q_p (1 − q_p)^K ≤ N/(eK)` (`expect_off_firstTable_le`,
`sum_missing_le`), where `q_p` is the probability that a run opens `p`.

If every table `T` with a failing property costs at least `ε − ε_c` of such off-table mass, one
Markov step bounds the probability that the extractor's table fails (`tail_of_fixed`), and the
product form `ε · Pr[fails] ≤ 2ε_c + 2·E[mass]` needs no condition on `ε` (`joint_of_fixed`).
-/

namespace FlockSoundness

namespace Extract

open Game Rewinding

variable {β P V : Type} [Fintype P] [DecidableEq V]

/-- **The extractor's table** from runs `bs`: at each position, the first opening a run made there;
`z₀` where none did. -/
def firstTable (op : β → P → Option V) (z₀ : P → V) (bs : List β) (p : P) : V :=
  (bs.findSome? fun b => op b p).getD (z₀ p)

theorem conflictAt_comm (op : β → P → Option V) (a b : β) (p : P) :
    ConflictAt op (a, b) p ↔ ConflictAt op (b, a) p :=
  ⟨fun ⟨h1, h2, h3⟩ => ⟨h2, h1, Ne.symm h3⟩, fun ⟨h1, h2, h3⟩ => ⟨h2, h1, Ne.symm h3⟩⟩

open Classical in
/-- **Off the extractor's table**: a conflict with the run that supplied the row, or a position no
extractor run opened. -/
theorem off_firstTable_le (op : β → P → Option V) (z₀ : P → V) (bs : List β) (ω : β) :
    ind (fun ω => ∃ p, Off op (firstTable op z₀ bs) ω p) ω ≤
      (bs.map fun b => ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b)).sum +
        ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
          ind (fun ω => (op ω p).isSome) ω := by
  have hsum0 : 0 ≤ (bs.map fun b => ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b)).sum :=
    List.sum_nonneg (by
      simp only [List.mem_map]
      rintro _ ⟨b, -, rfl⟩
      exact ind_nonneg _ _)
  have hprod0 : ∀ p, 0 ≤ (bs.map fun b => ind (fun b => op b p = none) b).prod := fun p =>
    List.prod_nonneg (by
      simp only [List.mem_map]
      rintro _ ⟨b, -, rfl⟩
      exact ind_nonneg _ _)
  have hsum1 : 0 ≤ ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
      ind (fun ω => (op ω p).isSome) ω :=
    Finset.sum_nonneg fun p _ => mul_nonneg (hprod0 p) (ind_nonneg _ _)
  by_cases h : ∃ p, Off op (firstTable op z₀ bs) ω p
  · rw [ind, if_pos h]
    obtain ⟨p, hsome, hne⟩ := h
    cases hf : bs.findSome? (fun b => op b p) with
    | some w =>
      obtain ⟨b, hb, hbw⟩ := List.exists_of_findSome?_eq_some hf
      have hconf : ConflictAt op (ω, b) p := by
        refine ⟨hsome, by simp [hbw], ?_⟩
        rw [hbw]
        simpa [firstTable, hf] using hne
      have h1 : ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b) = 1 := if_pos ⟨p, hconf⟩
      have h2 : ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b) ≤
          (bs.map fun b => ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b)).sum :=
        List.single_le_sum (fun x hx => by
          simp only [List.mem_map] at hx
          obtain ⟨b, -, rfl⟩ := hx
          exact ind_nonneg _ _) _ (List.mem_map_of_mem hb)
      linarith
    | none =>
      have hall : ∀ b ∈ bs, op b p = none := List.findSome?_eq_none_iff.1 hf
      have hprod : (bs.map fun b => ind (fun b => op b p = none) b).prod = 1 :=
        List.prod_eq_one (by
          simp only [List.mem_map]
          rintro _ ⟨b, hb, rfl⟩
          exact if_pos (hall b hb))
      have h1 : ind (fun ω => (op ω p).isSome) ω = 1 := if_pos hsome
      have h2 := Finset.single_le_sum (f := fun p => (bs.map fun b => ind (fun b => op b p = none) b).prod *
          ind (fun ω => (op ω p).isSome) ω) (fun p _ => mul_nonneg (hprod0 p) (ind_nonneg _ _))
        (Finset.mem_univ p)
      simp only [hprod, h1, one_mul] at h2
      linarith
  · rw [ind, if_neg h]
    linarith

omit [Fintype P] [DecidableEq V] in
theorem expect_list_sum (g : Game β) (s : Strategy g) (f : β → β → ℝ) :
    ∀ bs : List β, expect (fun ω => (bs.map fun b => f ω b).sum) g s =
      (bs.map fun b => expect (fun ω => f ω b) g s).sum
  | [] => by simp [expect_const]
  | b :: bs => by
      simp only [List.map_cons, List.sum_cons]
      rw [expect_add, expect_list_sum g s f bs]

open Classical in
/-- **The extractor's off-table mass**, over `K` independent runs: `K` fresh–extractor pairs, each
distributed as two independent runs, and the positions no extractor run opened. -/
theorem expect_off_firstTable_le (op : β → P → Option V) (z₀ : P → V) (g : Game β)
    (s : Strategy g) (K : ℕ) :
    expectN g s K (fun bs => expect (ind fun ω => ∃ p, Off op (firstTable op z₀ bs) ω p) g s) ≤
      K * expect₂ (fun a a' => ind (fun x : β × β => ∃ p, ConflictAt op x p) (a, a')) g s +
        ∑ p, expect (ind fun ω => (op ω p).isSome) g s *
          (1 - expect (ind fun ω => (op ω p).isSome) g s) ^ K := by
  refine (expectN_mono g s (fun bs => expect_mono (off_firstTable_le op z₀ bs) g s) K).trans
    (le_of_eq ?_)
  have hin : ∀ bs : List β, expect (fun ω =>
      (bs.map fun b => ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b)).sum +
        ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
          ind (fun ω => (op ω p).isSome) ω) g s =
      (bs.map fun b => expect (fun ω => ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b)) g s).sum +
        ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
          expect (ind fun ω => (op ω p).isSome) g s := by
    intro bs
    rw [expect_add, expect_list_sum, expect_sum]
    congr 1
    exact Finset.sum_congr rfl fun p _ => expect_const_mul _ _ g s
  simp only [hin]
  rw [expectN_add, expectN_sum_map, expectN_sum]
  congr 1
  · congr 1
    simp only [expect₂]
    refine congrArg (fun f => expect f g s) (funext fun b => ?_)
    refine congrArg (fun f => expect f g s) (funext fun ω => ?_)
    unfold ind
    by_cases h : ∃ p, ConflictAt op (ω, b) p
    · rw [if_pos h, if_pos (by obtain ⟨p, hp⟩ := h; exact ⟨p, (conflictAt_comm op ω b p).1 hp⟩)]
    · rw [if_neg h, if_neg (fun ⟨p, hp⟩ => h ⟨p, (conflictAt_comm op b ω p).1 hp⟩)]
  · refine Finset.sum_congr rfl fun p _ => ?_
    rw [show (fun bs : List β => (bs.map fun b => ind (fun b => op b p = none) b).prod *
        expect (ind fun ω => (op ω p).isSome) g s) =
      fun bs => expect (ind fun ω => (op ω p).isSome) g s *
        (bs.map fun b => ind (fun b => op b p = none) b).prod from funext fun _ => mul_comm _ _,
      expectN_const_mul, expectN_prod_map]
    have hN : ∀ b, ind (fun b => op b p = none) b = 1 - ind (fun ω => (op ω p).isSome) b :=
      fun b => by
        unfold ind
        by_cases h : op b p = none
        · rw [if_pos h, if_neg (by simp [h])]; ring
        · rw [if_neg h, if_pos (by simpa using Option.ne_none_iff_isSome.1 h)]; ring
    rw [show (fun b => ind (fun b => op b p = none) b) =
      fun b => 1 - ind (fun ω => (op ω p).isSome) b from funext hN, expect_sub, expect_const]

/-- `q (1 − q)^K ≤ 1/(eK)`. -/
theorem mul_one_sub_pow_le (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (K : ℕ) (hK : 1 ≤ K) :
    q * (1 - q) ^ K ≤ 1 / (Real.exp 1 * K) := by
  have hKpos : (0 : ℝ) < K := by exact_mod_cast hK
  have h1 : (1 - q) ^ K ≤ Real.exp (-q) ^ K :=
    pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp (-q)]) K
  have h2 : Real.exp (-q) ^ K = Real.exp (-(q * K)) := by
    rw [← Real.exp_nat_mul]; ring_nf
  have h3 : q * K ≤ Real.exp (q * K - 1) := by linarith [Real.add_one_le_exp (q * K - 1)]
  have h4 : q * Real.exp (-(q * K)) ≤ 1 / (Real.exp 1 * K) := by
    rw [le_div_iff₀ (by positivity)]
    calc q * Real.exp (-(q * K)) * (Real.exp 1 * K)
        = (q * K) * Real.exp (-(q * K)) * Real.exp 1 := by ring
      _ ≤ Real.exp (q * K - 1) * Real.exp (-(q * K)) * Real.exp 1 := by
          gcongr
      _ = 1 := by rw [← Real.exp_add, ← Real.exp_add]; ring_nf; exact Real.exp_zero
  calc q * (1 - q) ^ K ≤ q * Real.exp (-q) ^ K := mul_le_mul_of_nonneg_left h1 hq0
    _ = q * Real.exp (-(q * K)) := by rw [h2]
    _ ≤ 1 / (Real.exp 1 * K) := h4

/-- **The missing positions**: `Σ_p q_p (1 − q_p)^K ≤ N/(eK)`. -/
theorem sum_missing_le (q : P → ℝ) (hq0 : ∀ p, 0 ≤ q p) (hq1 : ∀ p, q p ≤ 1) (K : ℕ) (hK : 1 ≤ K) :
    ∑ p, q p * (1 - q p) ^ K ≤ Fintype.card P / (Real.exp 1 * K) := by
  calc ∑ p, q p * (1 - q p) ^ K ≤ ∑ _p : P, 1 / (Real.exp 1 * K) :=
        Finset.sum_le_sum fun p _ => mul_one_sub_pow_le (q p) (hq0 p) (hq1 p) K hK
    _ = Fintype.card P / (Real.exp 1 * K) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring

open Classical in
/-- **The extractor's off-table mass**: at most `K·Pr₂[conflict] + N/(eK)`. -/
theorem expect_off_firstTable_le' (op : β → P → Option V) (z₀ : P → V) (g : Game β)
    (s : Strategy g) (K : ℕ) (hK : 1 ≤ K) :
    expectN g s K (fun bs => expect (ind fun ω => ∃ p, Off op (firstTable op z₀ bs) ω p) g s) ≤
      K * expect₂ (fun a a' => ind (fun x : β × β => ∃ p, ConflictAt op x p) (a, a')) g s +
        Fintype.card P / (Real.exp 1 * K) :=
  (expect_off_firstTable_le op z₀ g s K).trans (add_le_add le_rfl (sum_missing_le _
    (fun p => expect_nonneg (ind_nonneg _) g s) (fun p => expect_ind_le_one _ g s) K hK))

omit [Fintype P] [DecidableEq V] in
open Classical in
/-- **One Markov step.** If every run list whose table fails has off-table mass at least `ε − ε_c`,
the extractor fails with probability at most `E[mass]/(ε − ε_c)`. -/
theorem tail_of_fixed (g : Game β) (s : Strategy g) (Fm : List β → ℝ) (X : List β → Prop)
    (hF : ∀ bs, 0 ≤ Fm bs) (ε εc : ℝ) (hε : εc < ε) (hX : ∀ bs, ¬ X bs → ε ≤ εc + Fm bs) (K : ℕ) :
    expectN g s K (ind fun bs => ¬ X bs) ≤ expectN g s K Fm / (ε - εc) := by
  have hpos : 0 < ε - εc := by linarith
  rw [le_div_iff₀ hpos, mul_comm, ← expectN_const_mul]
  refine expectN_mono g s (fun bs => ?_) K
  unfold ind
  split_ifs with h
  · linarith [hX bs h]
  · linarith [hF bs]

omit [Fintype P] [DecidableEq V] in
open Classical in
/-- **The product form**: `ε · Pr[fails] ≤ ε_c + E[mass]`, with no condition on `ε`. Run list by
run list, `ε · 1[fails] ≤ ε_c + mass`. -/
theorem joint_of_fixed (g : Game β) (s : Strategy g) (Fm : List β → ℝ) (X : List β → Prop)
    (hF : ∀ bs, 0 ≤ Fm bs) (ε εc : ℝ) (hεc : 0 ≤ εc)
    (hX : ∀ bs, ¬ X bs → ε ≤ εc + Fm bs) (K : ℕ) :
    ε * expectN g s K (ind fun bs => ¬ X bs) ≤ εc + expectN g s K Fm := by
  rw [← expectN_const_mul, ← expectN_const g s εc K, ← expectN_add]
  refine expectN_mono g s (fun bs => ?_) K
  unfold ind
  split_ifs with h
  · linarith [hX bs h]
  · linarith [hF bs]

end Extract

end FlockSoundness
