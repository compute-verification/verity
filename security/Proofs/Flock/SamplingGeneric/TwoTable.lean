module

public import Proofs.Flock.SamplingGeneric.TwoCount
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Topology.Sheaves.Init

/-!
# A two-stage count bound the kernel can check

`card_influenceTwo_le_uniform` takes `K`, a bound on every admissible set's proof blocks. This file gives one a kernel
can evaluate, for a two-stage design of one shape: a first stage that misses each of `R` replay blocks with
probability at most `q = (den − num)/den`, independently or less often, and one stratum of `N` proof blocks in each
replay block, `k` of them drawn.

* `pa_le_prod`: **the first stage, independent or less often**. When the first stage accepts a set `T` of replay blocks
  with probability at most `q^{|T|}`, the acceptance probability at a footprint is at most
  `∏_ρ (q + (1 − q)·e_ρ)`. Lean's `bernoulli:NUM/DEN` rule on one class meets it with equality (`bernoulli_rep`).
* `factor_eq`: with one stratum, replay block `ρ`'s factor at `c` incorrect proof blocks is
  `factor c / (den·C(N, k))`, `factor c = (den − num)·C(N, k) + num·C(N − c, k)`.
* `table`, `prod_le_table`: **the largest product**. Row `r`'s entry `m` is at least `∏_{i<r} factor(c_i)` for every
  `c_i ≤ N` with `Σ c_i = m`. Each row is the last one shifted by each `c ≤ N`, scaled by `factor c` and maxed
  pointwise, so a kernel evaluates it in `O(r·N²)` steps.
* `card_le_of_table`: **the count**. When row `R`'s entries past `K` are at most `(den·C(N, k))^R / 2^e`, every set
  admissible at `δ = 2^-e` has at most `K` proof blocks. `influence_le_of_table`: so the served outputs at any inputs
  number at most `2^{b·K}·Σ_{j≤K} C(R·N, j)`, every proof block committing at most `b` bits.
-/

@[expose] public section

namespace Claims.Sampling.TwoTable

open Claims.Sampling Finset

/-! ## The first stage, independent or less often -/

/-- **The first stage, independent or less often**: when it accepts each set `T` of the `R` replay blocks with
probability at most `q^{|T|}`, the acceptance probability at a footprint is at most `∏_ρ (q + (1 − q)·e_ρ)`, each
replay block's own. -/
theorem pa_le_prod (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) {R : ℕ} (hR : d.part.blocks.length = R)
    {q : ℝ} (hrep : ∀ T ⊆ range R, d.replay.Pa d.part T ≤ q ^ T.card) :
    d.Pa Pt fp ≤ ∏ ρ ∈ range R, (q + (1 - q) * d.esc Pt fp ρ) := by
  have hsplit : ∀ ρ ∈ range R, q + (1 - q) * d.esc Pt fp ρ = q * (1 - d.esc Pt fp ρ) + d.esc Pt fp ρ :=
    fun ρ _ => by ring
  rw [prod_congr rfl hsplit, prod_add]
  unfold Design.Two.Pa
  rw [hR]
  refine sum_le_sum fun T hT => ?_
  have hw : 0 ≤ (∏ ρ ∈ T, (1 - d.esc Pt fp ρ)) * ∏ ρ ∈ range R \ T, d.esc Pt fp ρ :=
    mul_nonneg (prod_nonneg fun ρ _ => sub_nonneg.2 (d.esc_le_one Pt fp ρ))
      (prod_nonneg fun ρ _ => d.esc_nonneg Pt fp ρ)
  rw [prod_mul_distrib, prod_const, mul_assoc (q ^ T.card)]
  exact mul_le_mul_of_nonneg_right (hrep T (mem_powerset.1 hT)) hw

/-- **Lean's `bernoulli:NUM/DEN` rule**, on a replay partition of one class, accepts a set `T` of its replay blocks
with probability `((den − num)/den)^{|T|}`: `pa_le_prod`'s hypothesis, with equality. -/
theorem bernoulli_rep {d : Design.Two} {num den : ℕ} (hd : d.replay = ⟨[.bernoulli num den]⟩)
    {c : List (List (List ℕ))} (hc : d.part.classes = [c]) :
    ∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤ (((den - num : ℕ) : ℝ) / den) ^ T.card := by
  intro T hT
  have hb : d.part.blocks.length = c.flatten.length := by simp [Partition.blocks, hc]
  have hw : window 0 c.flatten.length T = T := by
    unfold window
    rw [filter_true_of_mem fun v hv => ⟨Nat.zero_le v, by rw [Nat.zero_add, ← hb]; exact mem_range.1 (hT hv)⟩]
    simp
  rw [hd]
  show classesPa [OneStage.bernoulli num den] d.part.classes 0 T ≤ _
  rw [hc]
  show OneStage.Pa (strataSizes c) (.bernoulli num den) (window 0 c.flatten.length T) * 1 ≤ _
  rw [mul_one, hw]
  exact le_rfl

/-! ## One stratum -/

/-- With one stratum, replay block `ρ`'s escape is its one cell's. -/
theorem esc_one {d : Design.Two} {Pt : Partition} (fp : ℕ → ℕ → ℕ) (ρ : ℕ) (hs : Pt.strata.length = 1) :
    d.esc Pt fp ρ = escCell (cell Pt d.part ρ 0).card (d.k.getD 0 0) (fp ρ 0) := by
  rw [Design.Two.esc, hs, prod_range_one]

/-- **A replay block's factor**, over `den·C(N, k)`: `(den − num)·C(N, k) + num·C(N − c, k)` at `c` incorrect proof
blocks of its `N`, `k` drawn, the first stage at rate `num/den`. -/
def factor (N k num den c : ℕ) : ℕ := (den - num) * N.choose k + num * (N - c).choose k

theorem factor_eq {N k num den : ℕ} (hden : 0 < den) (hnum : num ≤ den) (hkN : k ≤ N) (m : ℕ) :
    ((den - num : ℕ) : ℝ) / den + (1 - ((den - num : ℕ) : ℝ) / den) * escCell N k m =
      ((factor N k num den m : ℕ) : ℝ) / ((den * N.choose k : ℕ) : ℝ) := by
  have hB : (0 : ℝ) < N.choose k := by exact_mod_cast Nat.choose_pos hkN
  have hd : (0 : ℝ) < den := by exact_mod_cast hden
  unfold escCell factor
  rw [Nat.min_eq_left hkN]
  push_cast [Nat.cast_sub hnum]
  field_simp
  ring

/-! ## The largest product -/

/-- Two lists of naturals maxed pointwise, the shorter one padded with `0`. -/
def pmax : List ℕ → List ℕ → List ℕ
  | [], l => l
  | a :: as, [] => a :: as
  | a :: as, b :: bs => max a b :: pmax as bs

theorem getD_pmax : ∀ (l l' : List ℕ) (i : ℕ), (pmax l l').getD i 0 = max (l.getD i 0) (l'.getD i 0)
  | [], l, i => by rw [pmax, List.getD_nil, Nat.zero_max]
  | a :: as, [], i => by rw [pmax, List.getD_nil, Nat.max_zero]
  | a :: as, b :: bs, 0 => by rw [pmax, List.getD_cons_zero, List.getD_cons_zero, List.getD_cons_zero]
  | a :: as, b :: bs, i + 1 => by
    rw [pmax, List.getD_cons_succ, List.getD_cons_succ, List.getD_cons_succ, getD_pmax as bs i]

/-- `l` shifted right by `c` places: `c` zeros, then `l`. -/
def shift : ℕ → List ℕ → List ℕ
  | 0, l => l
  | c + 1, l => 0 :: shift c l

theorem getD_shift : ∀ (c : ℕ) (l : List ℕ) (i : ℕ), (shift c l).getD (c + i) 0 = l.getD i 0
  | 0, l, i => by rw [Nat.zero_add, shift]
  | c + 1, l, i => by rw [show c + 1 + i = (c + i) + 1 by omega, shift, List.getD_cons_succ, getD_shift c l i]

theorem getD_map_mul (a : ℕ) : ∀ (l : List ℕ) (i : ℕ), (l.map (a * ·)).getD i 0 = a * l.getD i 0
  | [], i => by rw [List.map_nil, List.getD_nil, Nat.mul_zero]
  | x :: xs, 0 => by rw [List.map_cons, List.getD_cons_zero, List.getD_cons_zero]
  | x :: xs, i + 1 => by rw [List.map_cons, List.getD_cons_succ, List.getD_cons_succ, getD_map_mul a xs i]

/-- The row after `row`: entry `m` is the largest `F c · row[m − c]` over `c ≤ N`. -/
def nextRow (F : ℕ → ℕ) (N : ℕ) (row : List ℕ) : List ℕ :=
  (List.range (N + 1)).foldr (fun c acc => pmax (shift c (row.map (F c * ·))) acc) []

theorem le_nextRow (F : ℕ → ℕ) (N : ℕ) (row : List ℕ) {c : ℕ} (hc : c ≤ N) (i : ℕ) :
    F c * row.getD i 0 ≤ (nextRow F N row).getD (c + i) 0 := by
  have key : ∀ L : List ℕ, c ∈ L →
      F c * row.getD i 0 ≤ (L.foldr (fun c acc => pmax (shift c (row.map (F c * ·))) acc) []).getD (c + i) 0 := by
    intro L
    induction L with
    | nil => intro h; simp at h
    | cons x xs ih =>
      intro hL
      rw [List.foldr_cons, getD_pmax]
      rcases List.mem_cons.1 hL with h | h
      · subst h
        exact le_max_of_le_left (le_of_eq (by rw [getD_shift, getD_map_mul]))
      · exact le_max_of_le_right (ih h)
  exact key _ (List.mem_range.2 (by omega))

/-- **The table**: row `r`'s entry `m` bounds `∏_{i<r} F(c_i)` over `c_i ≤ N` with `Σ c_i = m` (`prod_le_table`). -/
def table (F : ℕ → ℕ) (N : ℕ) : ℕ → List ℕ
  | 0 => [1]
  | r + 1 => nextRow F N (table F N r)

theorem prod_le_table (F : ℕ → ℕ) (N : ℕ) : ∀ (r : ℕ) (c : ℕ → ℕ), (∀ i < r, c i ≤ N) →
    ∏ i ∈ range r, F (c i) ≤ (table F N r).getD (∑ i ∈ range r, c i) 0
  | 0, c, _ => by simp [table]
  | r + 1, c, hc => by
    rw [prod_range_succ, sum_range_succ, table, Nat.add_comm (∑ i ∈ range r, c i),
      Nat.mul_comm (∏ i ∈ range r, F (c i))]
    calc F (c r) * ∏ i ∈ range r, F (c i) ≤ F (c r) * (table F N r).getD (∑ i ∈ range r, c i) 0 :=
          Nat.mul_le_mul_left _ (prod_le_table F N r c fun i hi => hc i (by omega))
      _ ≤ _ := le_nextRow F N _ (hc r (by omega)) _

/-- Every entry of `l` from index `j` on meets `P`, when those of `l.drop j` and `0` do. -/
theorem getD_of_drop {P : ℕ → Prop} (h0 : P 0) :
    ∀ (l : List ℕ) (j : ℕ), (∀ x ∈ l.drop j, P x) → ∀ m, j ≤ m → P (l.getD m 0)
  | [], _, _, m, _ => by rw [List.getD_nil]; exact h0
  | x :: xs, 0, h, 0, _ => by rw [List.getD_cons_zero]; exact h x (by simp)
  | x :: xs, 0, h, m + 1, _ => by
    rw [List.getD_cons_succ]
    exact getD_of_drop h0 xs 0 (fun y hy => h y (by rw [List.drop_zero] at hy ⊢; exact List.mem_cons_of_mem _ hy))
      m (Nat.zero_le _)
  | x :: xs, j + 1, h, 0, hj => absurd hj (by omega)
  | x :: xs, j + 1, h, m + 1, hj => by
    rw [List.getD_cons_succ]
    exact getD_of_drop h0 xs j (by simpa using h) m (by omega)

/-! ## The count -/

/-- With every proof block in a cell of stratum `0` of one of `R` replay blocks, a set of proof blocks holds as many as
its counts add to. -/
theorem card_eq_sum_counts {Pt Pρ : Partition} {R : ℕ}
    (hcov : ∀ v < Pt.blocks.length, ∃ ρ < R, v ∈ cell Pt Pρ ρ 0) {T : Finset ℕ} (hT : T ⊆ range Pt.blocks.length) :
    T.card = ∑ ρ ∈ range R, counts Pt Pρ T ρ 0 := by
  classical
  have hdisj : ((range R : Finset ℕ) : Set ℕ).PairwiseDisjoint fun ρ => T ∩ cell Pt Pρ ρ 0 := by
    intro ρ _ ρ' _ hne
    show Disjoint (T ∩ cell Pt Pρ ρ 0) (T ∩ cell Pt Pρ ρ' 0)
    refine disjoint_left.2 fun v hv hv' => hne ?_
    have h1 := (mem_filter.1 (mem_inter.1 hv).2).2.1
    have h2 := (mem_filter.1 (mem_inter.1 hv').2).2.1
    exact Option.some_injective _ (h1.symm.trans h2)
  have hT' : T = (range R).biUnion fun ρ => T ∩ cell Pt Pρ ρ 0 := by
    ext v
    simp only [mem_biUnion, mem_inter, mem_range]
    constructor
    · intro hv
      obtain ⟨ρ, hρ, hc⟩ := hcov v (mem_range.1 (hT hv))
      exact ⟨ρ, hρ, hv, hc⟩
    · rintro ⟨_, _, hv, _⟩
      exact hv
  unfold counts
  conv_lhs => rw [hT']
  rw [card_biUnion hdisj]

/-- With `R` replay blocks of `N` proof blocks each, in one stratum, there are `R·N` proof blocks. -/
theorem blocks_length_eq {Pt Pρ : Partition} {R N : ℕ} (hN : ∀ ρ < R, (cell Pt Pρ ρ 0).card = N)
    (hcov : ∀ v < Pt.blocks.length, ∃ ρ < R, v ∈ cell Pt Pρ ρ 0) : Pt.blocks.length = R * N := by
  have h := card_eq_sum_counts hcov (subset_refl (range Pt.blocks.length))
  rw [card_range] at h
  rw [h, sum_congr rfl (g := fun _ => N), sum_const, card_range, smul_eq_mul]
  intro ρ hρ
  unfold counts
  rw [inter_eq_right.2 (by unfold cell; exact filter_subset _ _), hN ρ (mem_range.1 hρ)]

/-- **The count**: with the first stage missing each of `R` replay blocks with probability at most `(den − num)/den`,
independently or less often, and one stratum of `N` proof blocks in each, `k ≤ N` drawn, when row `R` of the table of
`factor` is at most `(den·C(N, k))^R / 2^e` past index `K`, every set admissible at `δ = 2^-e` has at most `K` proof
blocks. -/
theorem card_le_of_table {Pt : Partition} {d : Design.Two} {R N k num den e K : ℕ}
    (hden : 0 < den) (hnum : num ≤ den) (hR : d.part.blocks.length = R)
    (hrep : ∀ T ⊆ range R, d.replay.Pa d.part T ≤ (((den - num : ℕ) : ℝ) / den) ^ T.card)
    (hs : Pt.strata.length = 1) (hk : d.k.getD 0 0 = k) (hkN : k ≤ N)
    (hN : ∀ ρ < R, (cell Pt d.part ρ 0).card = N)
    (hcov : ∀ v < Pt.blocks.length, ∃ ρ < R, v ∈ cell Pt d.part ρ 0)
    (htab : ∀ m, K + 1 ≤ m → (table (factor N k num den) N R).getD m 0 * 2 ^ e ≤ (den * N.choose k) ^ R) :
    ∀ T ∈ admissibleTwo Pt d (((2 : ℝ) ^ e)⁻¹), T.card ≤ K := by
  classical
  intro T hT
  unfold admissibleTwo at hT
  obtain ⟨hTs, hM⟩ := mem_filter.1 hT
  have hM' : ((2 : ℝ) ^ e)⁻¹ < d.Pa Pt (counts Pt d.part T) := hM
  have hcN : ∀ ρ < R, counts Pt d.part T ρ 0 ≤ N := fun ρ hρ =>
    (card_le_card inter_subset_right).trans (hN ρ hρ).le
  have hPa : d.Pa Pt (counts Pt d.part T) ≤
      ∏ ρ ∈ range R, ((factor N k num den (counts Pt d.part T ρ 0) : ℕ) : ℝ) / ((den * N.choose k : ℕ) : ℝ) := by
    refine (pa_le_prod d Pt _ hR hrep).trans (le_of_eq (prod_congr rfl fun ρ hρ => ?_))
    rw [esc_one _ _ hs, hN ρ (mem_range.1 hρ), hk]
    exact factor_eq hden hnum hkN _
  have hD : (0 : ℝ) < ((den * N.choose k : ℕ) : ℝ) ^ R :=
    pow_pos (by exact_mod_cast Nat.mul_pos hden (Nat.choose_pos hkN)) R
  have h1 := lt_of_lt_of_le hM' hPa
  rw [prod_div_distrib, prod_const, card_range, lt_div_iff₀ hD] at h1
  have h2 := mul_lt_mul_of_pos_right h1 (by positivity : (0 : ℝ) < 2 ^ e)
  rw [inv_mul_eq_div, div_mul_cancel₀ _ (by positivity)] at h2
  have hlt : (den * N.choose k) ^ R < (∏ ρ ∈ range R, factor N k num den (counts Pt d.part T ρ 0)) * 2 ^ e := by
    exact_mod_cast h2
  by_contra hK
  have hle := prod_le_table (factor N k num den) N R (fun ρ => counts Pt d.part T ρ 0) hcN
  rw [← card_eq_sum_counts hcov (mem_powerset.1 hTs)] at hle
  exact lt_irrefl _ (lt_of_lt_of_le hlt ((Nat.mul_le_mul_right _ hle).trans (htab _ (by omega))))

/-- **The influence bound at a table's count**: on a statement in scope whose design has `card_le_of_table`'s shape and
whose proof blocks each commit at most `b` bits, the served outputs at any input values number at most
`2^{b·K}·Σ_{j≤K} C(R·N, j)` (`card_influenceTwo_le_uniform`). -/
theorem influence_le_of_table (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hsc : InScopeTwo Pt d s) {R N k num den e K b : ℕ}
    (hden : 0 < den) (hnum : num ≤ den) (hR : d.part.blocks.length = R)
    (hrep : ∀ T ⊆ range R, d.replay.Pa d.part T ≤ (((den - num : ℕ) : ℝ) / den) ^ T.card)
    (hs : Pt.strata.length = 1) (hk : d.k.getD 0 0 = k) (hkN : k ≤ N)
    (hN : ∀ ρ < R, (cell Pt d.part ρ 0).card = N)
    (hcov : ∀ v < Pt.blocks.length, ∃ ρ < R, v ∈ cell Pt d.part ρ 0)
    (hb : ∀ v < Pt.blocks.length, committed s.prog (Pt.blocks.getD v []) ≤ b)
    (htab : ∀ m, K + 1 ≤ m → (table (factor N k num den) N R).getD m 0 * 2 ^ e ≤ (den * N.choose k) ^ R)
    (a : ℕ → Option Bool) :
    (influenceTwo H Cm s Pt d (((2 : ℝ) ^ e)⁻¹) a).card ≤ 2 ^ (b * K) * ∑ j ∈ range (K + 1), (R * N).choose j := by
  classical
  have hK := card_le_of_table hden hnum hR hrep hs hk hkN hN hcov htab
  rw [← blocks_length_eq hN hcov]
  refine card_influenceTwo_le_uniform H Cm hsc _ a (fun T hT => ?_) hK
  have hTs : T ⊆ range Pt.blocks.length := by
    unfold admissibleTwo at hT
    exact mem_powerset.1 (mem_filter.1 hT).1
  calc bitsOf s Pt T ≤ ∑ _v ∈ T, b := sum_le_sum fun v hv => hb v (mem_range.1 (hTs hv))
    _ = b * T.card := by rw [sum_const, smul_eq_mul, Nat.mul_comm]
    _ ≤ b * K := Nat.mul_le_mul_left _ (hK T hT)

end Claims.Sampling.TwoTable
