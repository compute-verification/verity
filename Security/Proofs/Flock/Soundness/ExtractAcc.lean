import Proofs.Flock.Soundness.Extract

/-!
# Extraction from accepted reruns: the expected-time extractor's law

The expected-time extractor (DESIGN.md §3) runs the session once and, only if that run accepts,
reruns until `k` reruns have accepted: `1 + k` runs in expectation, whatever the acceptance
probability `ε`, and no estimate of `ε`. Its accepted reruns are `k` independent runs conditioned
on all accepting, and it fills its table from them (`firstTable`). `expectAcc` is that law, written
without the loop: `k` independent runs weighted by their acceptance, over `ε^k`.

Against it, an accepted fresh run opens a position off the table in one of two ways
(`off_firstTable_le`), with these costs (`expectAcc_off_le`):

* it conflicts with the accepted rerun that supplied the row: `(k/ε)·Pr₂[conflict]`;
* it opens a position no accepted rerun opened: `Σ_p q_p (1 − q_p/ε)^k ≤ N·ε/(ek)`, with `q_p` the
  probability that a run accepts and opens `p`. This term is relative to `ε`.

`joint_acc_le` is the product form: if every run list whose table fails leaves an accepted fresh
run at least `ε − ε_c` of that mass, then `ε · Pr[fails] ≤ ε_c + N·ε/(ek) + √(k·Pr₂)`. The conflict
term's `1/ε` becomes a square root because the product is also at most `ε`.
-/

namespace FlockSoundness

namespace Extract

open Game Rewinding

variable {β P V : Type}

/-- **The law of `k` independent accepted runs**: the expectation of `F` over the extractor's
accepted reruns, as `k` runs weighted by their acceptance, over `ε^k`. It is zero when `ε = 0`. -/
noncomputable def expectAcc (g : Game β) (s : Strategy g) (acc : β → Prop) (k : ℕ)
    (F : List β → ℝ) : ℝ :=
  expectN g s k (fun bs => (bs.map (ind acc)).prod * F bs) / expect (ind acc) g s ^ k

theorem ind_and (p q : β → Prop) (a : β) : ind (fun a => p a ∧ q a) a = ind p a * ind q a := by
  unfold ind
  by_cases hp : p a <;> by_cases hq : q a <;> simp [hp, hq]

theorem prod_ind_nonneg (acc : β → Prop) (bs : List β) : 0 ≤ (bs.map (ind acc)).prod :=
  List.prod_nonneg (by
    simp only [List.mem_map]
    rintro _ ⟨b, -, rfl⟩
    exact ind_nonneg _ _)

theorem expectAcc_mono (g : Game β) (s : Strategy g) (acc : β → Prop) (k : ℕ)
    {F F' : List β → ℝ} (h : ∀ bs, F bs ≤ F' bs) :
    expectAcc g s acc k F ≤ expectAcc g s acc k F' :=
  div_le_div_of_nonneg_right
    (expectN_mono g s (fun bs => mul_le_mul_of_nonneg_left (h bs) (prod_ind_nonneg acc bs)) k)
    (pow_nonneg (expect_nonneg (ind_nonneg acc) g s) k)

theorem expectAcc_nonneg (g : Game β) (s : Strategy g) (acc : β → Prop) (k : ℕ)
    {F : List β → ℝ} (h : ∀ bs, 0 ≤ F bs) : 0 ≤ expectAcc g s acc k F :=
  div_nonneg (expectN_nonneg g s (fun bs => mul_nonneg (prod_ind_nonneg acc bs) (h bs)) k)
    (pow_nonneg (expect_nonneg (ind_nonneg acc) g s) k)

theorem expectAcc_add (g : Game β) (s : Strategy g) (acc : β → Prop) (k : ℕ)
    (F F' : List β → ℝ) :
    expectAcc g s acc k (fun bs => F bs + F' bs) =
      expectAcc g s acc k F + expectAcc g s acc k F' := by
  unfold expectAcc
  rw [← add_div, ← expectN_add]
  congr 1
  exact congrArg (expectN g s k) (funext fun bs => mul_add _ _ _)

theorem expectAcc_const (g : Game β) (s : Strategy g) (acc : β → Prop) (k : ℕ) (c : ℝ)
    (hε : 0 < expect (ind acc) g s) : expectAcc g s acc k (fun _ => c) = c := by
  unfold expectAcc
  rw [show (fun bs : List β => (bs.map (ind acc)).prod * c) =
      fun bs => c * (bs.map (ind acc)).prod from funext fun _ => mul_comm _ _,
    expectN_const_mul, expectN_prod_map]
  exact mul_div_cancel_right₀ c (pow_ne_zero k hε.ne')

/-- **A per-run quantity summed over `k` weighted runs**:
`E[f] · E_k[Π f · Σ h] = k · E[f]^k · E[f·h]`. -/
theorem expectN_prod_mul_sum (g : Game β) (s : Strategy g) (f h : β → ℝ) :
    ∀ k : ℕ, expect f g s * expectN g s k (fun bs => (bs.map f).prod * (bs.map h).sum) =
      k * expect f g s ^ k * expect (fun b => f b * h b) g s
  | 0 => by simp [expectN]
  | k + 1 => by
      have hk := expectN_prod_mul_sum g s f h k
      have step : ∀ a, expectN g s k (fun as => ((a :: as).map f).prod * ((a :: as).map h).sum) =
          expect f g s ^ k * (f a * h a) +
            expectN g s k (fun as => (as.map f).prod * (as.map h).sum) * f a := by
        intro a
        have e : (fun as : List β => ((a :: as).map f).prod * ((a :: as).map h).sum) =
            fun as => (f a * h a) * (as.map f).prod + f a * ((as.map f).prod * (as.map h).sum) := by
          funext as
          simp only [List.map_cons, List.prod_cons, List.sum_cons]
          ring
        rw [e, expectN_add, expectN_const_mul, expectN_const_mul, expectN_prod_map]
        ring
      have e2 : expectN g s (k + 1) (fun bs => (bs.map f).prod * (bs.map h).sum) =
          expect f g s ^ k * expect (fun b => f b * h b) g s +
            expectN g s k (fun as => (as.map f).prod * (as.map h).sum) * expect f g s := by
        simp only [expectN]
        rw [show (fun a => expectN g s k fun as => ((a :: as).map f).prod * ((a :: as).map h).sum) =
          fun a => expect f g s ^ k * (f a * h a) +
            expectN g s k (fun as => (as.map f).prod * (as.map h).sum) * f a from funext step,
          expect_add, expect_const_mul]
        rw [show (fun a => expectN g s k (fun as => (as.map f).prod * (as.map h).sum) * f a) =
          fun a => expectN g s k (fun as => (as.map f).prod * (as.map h).sum) * f a from rfl]
        rw [show expect (fun a => expectN g s k (fun as => (as.map f).prod * (as.map h).sum) * f a) g s =
          expectN g s k (fun as => (as.map f).prod * (as.map h).sum) * expect f g s from
          expect_const_mul _ f g s]
      rw [e2, mul_add, show expect f g s * (expectN g s k (fun as => (as.map f).prod * (as.map h).sum) *
          expect f g s) = expect f g s * (expect f g s *
            expectN g s k (fun as => (as.map f).prod * (as.map h).sum)) by ring, hk]
      push_cast
      ring

/-- **A product of two per-run quantities over `k` runs** is the expectation of their product to
the `k`. -/
theorem expectN_prod_mul_prod (g : Game β) (s : Strategy g) (f u : β → ℝ) (k : ℕ) :
    expectN g s k (fun bs => (bs.map f).prod * (bs.map u).prod) =
      expect (fun b => f b * u b) g s ^ k := by
  rw [← expectN_prod_map g s (fun b => f b * u b) k]
  exact congrArg (expectN g s k) (funext fun bs => List.prod_map_mul.symm)

open Classical in
/-- Pointwise: an accepted run off the table conflicts with a rerun, or is accepted and opens a
position no rerun opened. -/
theorem ind_acc_off_le [Fintype P] [DecidableEq V] (op : β → P → Option V) (z₀ : P → V)
    (acc : β → Prop) (bs : List β) (ω : β) :
    ind (fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) ω ≤
      (bs.map fun b => ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b)).sum +
        ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
          ind (fun ω => acc ω ∧ (op ω p).isSome) ω := by
  have h1 := off_firstTable_le op z₀ bs ω
  have ha0 := ind_nonneg acc ω
  have ha1 := ind_le_one acc ω
  have hs0 : 0 ≤ (bs.map fun b => ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b)).sum :=
    List.sum_nonneg (by
      simp only [List.mem_map]
      rintro _ ⟨b, -, rfl⟩
      exact ind_nonneg _ _)
  have hm := mul_le_mul_of_nonneg_left h1 ha0
  rw [mul_add, Finset.mul_sum] at hm
  rw [ind_and]
  have hc : ind acc ω * (bs.map fun b => ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b)).sum ≤
      (bs.map fun b => ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b)).sum :=
    mul_le_of_le_one_left hs0 ha1
  have hp : ∑ p, ind acc ω * ((bs.map fun b => ind (fun b => op b p = none) b).prod *
      ind (fun ω => (op ω p).isSome) ω) = ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
        ind (fun ω => acc ω ∧ (op ω p).isSome) ω :=
    Finset.sum_congr rfl fun p _ => by rw [ind_and]; ring
  linarith

open Classical in
/-- **The accepted fresh run's off-table mass, against `k` accepted reruns**: `(k/ε)·Pr₂[conflict]`
for the rerun that supplied the row, and `N·ε/(ek)` for the positions no accepted rerun opened. -/
theorem expectAcc_off_le [Fintype P] [DecidableEq V] (op : β → P → Option V) (z₀ : P → V)
    (g : Game β) (s : Strategy g) (acc : β → Prop) (k : ℕ) (hk : 1 ≤ k) :
    expectAcc g s acc k (fun bs =>
        expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s) ≤
      k / expect (ind acc) g s *
          expect₂ (fun a a' => ind (fun x : β × β => ∃ p, ConflictAt op x p) (a, a')) g s +
        Fintype.card P * expect (ind acc) g s / (Real.exp 1 * k) := by
  set ε := expect (ind acc) g s with hεdef
  have hε0 : 0 ≤ ε := expect_nonneg (ind_nonneg acc) g s
  have hk0 : (0 : ℝ) < k := by exact_mod_cast hk
  have hE2 : 0 ≤ expect₂ (fun a a' => ind (fun x : β × β => ∃ p, ConflictAt op x p) (a, a')) g s :=
    expect_nonneg (fun _ => expect_nonneg (fun _ => ind_nonneg _ _) g s) g s
  rcases hε0.eq_or_lt with h0 | hpos
  · have hz : expectAcc g s acc k (fun bs =>
        expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s) = 0 := by
      unfold expectAcc
      rw [← hεdef, ← h0, zero_pow (by omega), div_zero]
    rw [hz, ← h0]
    simp
  -- the conflict mass of rerun `b`, and the accepted opening probability of position `p`
  set c : β → ℝ := fun b =>
    expect (fun ω => ind (fun x : β × β => ∃ p, ConflictAt op x p) (ω, b)) g s with hcdef
  set q : P → ℝ := fun p => expect (ind fun ω => acc ω ∧ (op ω p).isSome) g s with hqdef
  have hc0 : ∀ b, 0 ≤ c b := fun b => expect_nonneg (fun _ => ind_nonneg _ _) g s
  have hpt : ∀ bs : List β,
      expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s ≤
        (bs.map c).sum + ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod * q p := by
    intro bs
    refine (expect_mono (ind_acc_off_le op z₀ acc bs) g s).trans (le_of_eq ?_)
    rw [expect_add, expect_list_sum, expect_sum]
    congr 1
    exact Finset.sum_congr rfl fun p _ => expect_const_mul _ _ g s
  -- weighted by the reruns' acceptance
  have hW : expectN g s k (fun bs => (bs.map (ind acc)).prod *
        expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s) ≤
      expectN g s k (fun bs => (bs.map (ind acc)).prod * (bs.map c).sum) +
        ∑ p, q p * expectN g s k (fun bs => (bs.map (ind acc)).prod *
          (bs.map fun b => ind (fun b => op b p = none) b).prod) := by
    refine (expectN_mono g s (fun bs => mul_le_mul_of_nonneg_left (hpt bs)
      (prod_ind_nonneg acc bs)) k).trans (le_of_eq ?_)
    rw [show (fun bs : List β => (bs.map (ind acc)).prod * ((bs.map c).sum +
        ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod * q p)) =
      fun bs => (bs.map (ind acc)).prod * (bs.map c).sum +
        ∑ p, q p * ((bs.map (ind acc)).prod *
          (bs.map fun b => ind (fun b => op b p = none) b).prod) from funext fun bs => by
        rw [mul_add, Finset.mul_sum]
        congr 1
        exact Finset.sum_congr rfl fun p _ => by ring,
      expectN_add, expectN_sum]
    congr 1
    exact Finset.sum_congr rfl fun p _ => expectN_const_mul g s _ _ k
  -- the conflict part
  have hA := expectN_prod_mul_sum g s (ind acc) c k
  have hAc : expect (fun b => ind acc b * c b) g s ≤ expect c g s :=
    expect_mono (fun b => mul_le_of_le_one_left (hc0 b) (ind_le_one acc b)) g s
  have hc2 : expect c g s =
      expect₂ (fun a a' => ind (fun x : β × β => ∃ p, ConflictAt op x p) (a, a')) g s := by
    simp only [hcdef, expect₂]
    refine congrArg (fun f => expect f g s) (funext fun b => ?_)
    refine congrArg (fun f => expect f g s) (funext fun ω => ?_)
    have hiff : (∃ p, ConflictAt op (ω, b) p) ↔ ∃ p, ConflictAt op (b, ω) p :=
      exists_congr fun p => conflictAt_comm op ω b p
    simp only [ind, hiff]
    split_ifs <;> rfl
  -- the missing part
  have hB : ∀ p, expectN g s k (fun bs => (bs.map (ind acc)).prod *
      (bs.map fun b => ind (fun b => op b p = none) b).prod) = (ε - q p) ^ k := by
    intro p
    rw [expectN_prod_mul_prod]
    congr 1
    rw [hqdef, hεdef, ← expect_sub]
    refine congrArg (fun f => expect f g s) (funext fun b => ?_)
    rw [ind_and]
    unfold ind
    by_cases h1 : acc b <;> by_cases h2 : op b p = none <;> simp [h1, h2, Option.isSome_iff_ne_none]
  have hq0 : ∀ p, 0 ≤ q p := fun p => expect_nonneg (ind_nonneg _) g s
  have hqε : ∀ p, q p ≤ ε := fun p =>
    expect_mono (ind_mono fun b (h : acc b ∧ (op b p).isSome) => h.1) g s
  have hεk : 0 < ε ^ k := pow_pos hpos k
  have hmiss : ∀ p, q p * (ε - q p) ^ k / ε ^ k ≤ ε / (Real.exp 1 * k) := by
    intro p
    have hx0 : 0 ≤ q p / ε := div_nonneg (hq0 p) hε0
    have hx1 : q p / ε ≤ 1 := (div_le_one hpos).2 (hqε p)
    have key := mul_one_sub_pow_le (q p / ε) hx0 hx1 k hk
    have e : q p * (ε - q p) ^ k / ε ^ k = ε * (q p / ε * (1 - q p / ε) ^ k) := by
      rw [one_sub_div hpos.ne', div_pow]
      field_simp
    rw [e]
    calc ε * (q p / ε * (1 - q p / ε) ^ k) ≤ ε * (1 / (Real.exp 1 * k)) :=
          mul_le_mul_of_nonneg_left key hε0
      _ = ε / (Real.exp 1 * k) := by ring
  -- assemble
  unfold expectAcc
  rw [← hεdef]
  have hconf : expectN g s k (fun bs => (bs.map (ind acc)).prod * (bs.map c).sum) / ε ^ k ≤
      k / ε * expect c g s := by
    rw [div_le_iff₀ hεk]
    have : ε * expectN g s k (fun bs => (bs.map (ind acc)).prod * (bs.map c).sum) ≤
        ε * (k / ε * expect c g s * ε ^ k) := by
      rw [hA]
      have e : ε * (k / ε * expect c g s * ε ^ k) = k * ε ^ k * expect c g s := by
        field_simp
      rw [e]
      exact mul_le_mul_of_nonneg_left hAc (mul_nonneg hk0.le hεk.le)
    exact le_of_mul_le_mul_left this hpos
  have hsum : (∑ p, q p * expectN g s k (fun bs => (bs.map (ind acc)).prod *
      (bs.map fun b => ind (fun b => op b p = none) b).prod)) / ε ^ k ≤
      Fintype.card P * ε / (Real.exp 1 * k) := by
    rw [div_eq_mul_inv, Finset.sum_mul]
    simp only [← div_eq_mul_inv]
    calc ∑ p, q p * expectN g s k (fun bs => (bs.map (ind acc)).prod *
          (bs.map fun b => ind (fun b => op b p = none) b).prod) / ε ^ k
        = ∑ p, q p * (ε - q p) ^ k / ε ^ k := Finset.sum_congr rfl fun p _ => by rw [hB p]
      _ ≤ ∑ _p : P, ε / (Real.exp 1 * k) := Finset.sum_le_sum fun p _ => hmiss p
      _ = Fintype.card P * ε / (Real.exp 1 * k) := by
          rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring
  calc expectN g s k (fun bs => (bs.map (ind acc)).prod *
        expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s) / ε ^ k
      ≤ (expectN g s k (fun bs => (bs.map (ind acc)).prod * (bs.map c).sum) +
          ∑ p, q p * expectN g s k (fun bs => (bs.map (ind acc)).prod *
            (bs.map fun b => ind (fun b => op b p = none) b).prod)) / ε ^ k :=
        div_le_div_of_nonneg_right hW hεk.le
    _ = expectN g s k (fun bs => (bs.map (ind acc)).prod * (bs.map c).sum) / ε ^ k +
          (∑ p, q p * expectN g s k (fun bs => (bs.map (ind acc)).prod *
            (bs.map fun b => ind (fun b => op b p = none) b).prod)) / ε ^ k := add_div _ _ _
    _ ≤ k / ε * expect c g s + Fintype.card P * ε / (Real.exp 1 * k) := add_le_add hconf hsum
    _ = _ := by rw [hc2]

/-- `min(ε, a + x/ε) ≤ a + √x`: the product is at most `ε`, so a term `x/ε` costs at most `√x`. -/
theorem min_le_add_sqrt {ε a x : ℝ} (hε : 0 < ε) (ha : 0 ≤ a) (hx : 0 ≤ x) :
    min ε (a + x / ε) ≤ a + Real.sqrt x := by
  by_cases h : x / ε ≤ Real.sqrt x
  · exact (min_le_right _ _).trans (by linarith)
  · push Not at h
    have hs : Real.sqrt x < x / ε := h
    have hsx : 0 ≤ Real.sqrt x := Real.sqrt_nonneg x
    have hε' : ε < Real.sqrt x := by
      by_contra hc
      push Not at hc
      have : x / ε ≤ Real.sqrt x := by
        rw [div_le_iff₀ hε]
        calc x = Real.sqrt x * Real.sqrt x := (Real.mul_self_sqrt hx).symm
          _ ≤ Real.sqrt x * ε := mul_le_mul_of_nonneg_left hc hsx
      linarith
    exact (min_le_left _ _).trans (by linarith)

open Classical in
/-- **The product form, for accepted reruns.** If every run list whose table fails leaves an
accepted fresh run at least `ε − ε_c` of off-table mass (`hX`), then
`ε · Pr[fails] ≤ ε_c + N·ε/(ek) + √(k·Pr₂[conflict])` over the extractor's `k` accepted reruns. -/
theorem joint_acc_le [Fintype P] [DecidableEq V] (op : β → P → Option V) (z₀ : P → V)
    (g : Game β) (s : Strategy g) (acc : β → Prop) (X : List β → Prop) (εc : ℝ) (hεc : 0 ≤ εc)
    (hX : ∀ bs, ¬ X bs → expect (ind acc) g s ≤
      εc + expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s)
    (k : ℕ) (hk : 1 ≤ k) :
    expect (ind acc) g s * expectAcc g s acc k (ind fun bs => ¬ X bs) ≤
      εc + Fintype.card P * expect (ind acc) g s / (Real.exp 1 * k) +
        Real.sqrt (k * expect₂ (fun a a' =>
          ind (fun x : β × β => ∃ p, ConflictAt op x p) (a, a')) g s) := by
  set ε := expect (ind acc) g s with hεdef
  set E₂ := expect₂ (fun a a' => ind (fun x : β × β => ∃ p, ConflictAt op x p) (a, a')) g s
    with hE₂def
  have hε0 : 0 ≤ ε := expect_nonneg (ind_nonneg acc) g s
  have hk0 : (0 : ℝ) < k := by exact_mod_cast hk
  have hE2 : 0 ≤ E₂ := expect_nonneg (fun _ => expect_nonneg (fun _ => ind_nonneg _ _) g s) g s
  have hN : 0 ≤ Fintype.card P * ε / (Real.exp 1 * k) := by positivity
  rcases hε0.eq_or_lt with h0 | hpos
  · have hz : ε * expectAcc g s acc k (ind fun bs => ¬ X bs) = 0 := by rw [← h0, zero_mul]
    rw [hz]
    have : 0 ≤ Real.sqrt (k * E₂) := Real.sqrt_nonneg _
    linarith
  set M := fun bs : List β => expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s
  have hM0 : ∀ bs, 0 ≤ M bs := fun bs => expect_nonneg (ind_nonneg _) g s
  -- run list by run list, `ε · 1[fails] ≤ ε_c + mass`
  have hfix : ε * expectAcc g s acc k (ind fun bs => ¬ X bs) ≤ εc + expectAcc g s acc k M := by
    have h1 : ε * expectAcc g s acc k (ind fun bs => ¬ X bs) =
        expectAcc g s acc k (fun bs => ε * ind (fun bs => ¬ X bs) bs) := by
      unfold expectAcc
      rw [mul_div_assoc', ← expectN_const_mul]
      congr 1
      exact congrArg (expectN g s k) (funext fun bs => by ring)
    rw [h1, ← expectAcc_const g s acc k εc hpos, ← expectAcc_add]
    refine expectAcc_mono g s acc k fun bs => ?_
    unfold ind
    split_ifs with h
    · linarith [hX bs h]
    · linarith [hM0 bs]
  have hoff := expectAcc_off_le op z₀ g s acc k hk
  have hle1 : ε * expectAcc g s acc k (ind fun bs => ¬ X bs) ≤ ε := by
    have : expectAcc g s acc k (ind fun bs => ¬ X bs) ≤ 1 := by
      rw [← expectAcc_const g s acc k 1 hpos]
      exact expectAcc_mono g s acc k (ind_le_one _)
    nlinarith
  have hmin := min_le_add_sqrt hpos (add_nonneg hεc hN) (mul_nonneg hk0.le hE2)
  have hform : εc + expectAcc g s acc k M ≤ εc + Fintype.card P * ε / (Real.exp 1 * k) +
      k * E₂ / ε := by
    have : k / ε * E₂ = k * E₂ / ε := by ring
    linarith
  calc ε * expectAcc g s acc k (ind fun bs => ¬ X bs)
      ≤ min ε (εc + Fintype.card P * ε / (Real.exp 1 * k) + k * E₂ / ε) :=
        le_min hle1 (hfix.trans hform)
    _ ≤ _ := hmin

end Extract

end FlockSoundness
