import Proofs.Flock.Soundness.LinkFinder

/-!
# Extraction from accepted reruns, linearly: a conflict finder at level 0

`joint_acc_le` bounds `ε · Pr[the extractor's table fails]` by `ε_c + N·ε/(ek) + √(k·Pr₂[conflict])`: an accepted fresh
run off the table conflicts with the rerun that supplied the row, at `(k/ε)·Pr₂`, and the product with `ε` turns that
into a root. Here the fresh run's acceptance stays with the conflict, so the conflict is one among `k + 1` accepted runs,
and the bound is linear:

* `expectAcc_off_le_lin`: against `k` accepted reruns, an accepted fresh run's off-table mass is at most
  `ε · expectAcc_{k+1}[two of the runs conflict] + N·ε/(ek)` (`PairConflict`; the missing positions as in
  `expectAcc_off_le`, `expectAcc_missing_le`);
* `joint_acc_le_lin`: so `ε · Pr[fails] ≤ ε_c + N·ε/(ek) + ε · expectAcc_{k+1}[PairConflict]`;
* `conflictG`: the extractor's own runs as a finder: a run, then `k` waits of at most `R` runs. It succeeds
  (`ConflictSucc`) when the run accepts, every wait finds an accepted run (`firsts`), and two of those `k + 1` accepted
  runs conflict. Its runs are independent runs of one game, so nothing is rewound;
* `conflict_succ_ge`: `ε · expectAcc_{k+1}[PairConflict] ≤ Pr[ConflictSucc] + (k + 1)/(eR)`;
* `expect_conflictCost_le`: the finder plays at most `k + 1` runs in expectation, whatever `ε` (the waits are played
  only after an accepted run, and `ε · geom ε R ≤ 1`);
* `joint_acc_le_ecr`: under expected-time collision resistance at the finder, `ε · Pr[fails]` is at most
  `ε_c + N·ε/(ek) + (k + 1)·E[c]/2^256 + (k + 1)/(eR)`: weight 1, `k + 1` runs.
-/

namespace FlockSoundness

namespace Extract

open Game Rewinding

variable {β P V : Type}

/-- **Two runs of the list conflict** at some position. -/
def PairConflict (op : β → P → Option V) (bs : List β) : Prop :=
  ∃ a ∈ bs, ∃ b ∈ bs, ∃ p, ConflictAt op (a, b) p

/-- **`k + 1` runs, the last split off.** -/
theorem expectN_snoc (g : Game β) (s : Strategy g) : ∀ (k : ℕ) (F : List β → ℝ),
    expectN g s (k + 1) F = expectN g s k (fun bs => expect (fun ω => F (bs ++ [ω])) g s)
  | 0, _ => rfl
  | k + 1, F => by
      show expect (fun a => expectN g s (k + 1) fun as => F (a :: as)) g s =
        expect (fun a => expectN g s k fun bs => expect (fun ω => F ((a :: bs) ++ [ω])) g s) g s
      exact congrArg (fun f => expect f g s) (funext fun a => expectN_snoc g s k (fun as => F (a :: as)))

/-- **The last of `k + 1` accepted runs, split off**: `ε · expectAcc_{k+1} F` is `expectAcc_k` of the last run's
accepted expectation. -/
theorem mul_expectAcc_succ (g : Game β) (s : Strategy g) (acc : β → Prop) (k : ℕ) (F : List β → ℝ)
    (hε : 0 < expect (ind acc) g s) :
    expect (ind acc) g s * expectAcc g s acc (k + 1) F =
      expectAcc g s acc k (fun bs => expect (fun ω => ind acc ω * F (bs ++ [ω])) g s) := by
  have h1 : expectN g s (k + 1) (fun bs => (bs.map (ind acc)).prod * F bs) =
      expectN g s k (fun bs => (bs.map (ind acc)).prod * expect (fun ω => ind acc ω * F (bs ++ [ω])) g s) := by
    rw [expectN_snoc]
    refine congrArg (expectN g s k) (funext fun bs => ?_)
    rw [← expect_const_mul]
    refine congrArg (fun f => expect f g s) (funext fun ω => ?_)
    simp only [List.map_append, List.map_cons, List.map_nil, List.prod_append, List.prod_cons, List.prod_nil]
    ring
  unfold expectAcc
  rw [h1, pow_succ]
  have hk : expect (ind acc) g s ^ k ≠ 0 := pow_ne_zero k hε.ne'
  field_simp

open Classical in
/-- Pointwise: an accepted run off the table conflicts, accepted, with a rerun, or is accepted and opens a position no
rerun opened. -/
theorem ind_acc_off_le_lin [Fintype P] [DecidableEq V] (op : β → P → Option V) (z₀ : P → V)
    (acc : β → Prop) (bs : List β) (ω : β) :
    ind (fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) ω ≤
      ind acc ω * ind (PairConflict op) (bs ++ [ω]) +
        ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
          ind (fun ω => acc ω ∧ (op ω p).isSome) ω := by
  have hprod0 : ∀ p, 0 ≤ (bs.map fun b => ind (fun b => op b p = none) b).prod := fun p =>
    List.prod_nonneg (by
      simp only [List.mem_map]
      rintro _ ⟨b, -, rfl⟩
      exact ind_nonneg _ _)
  have hsum0 : 0 ≤ ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
      ind (fun ω => acc ω ∧ (op ω p).isSome) ω :=
    Finset.sum_nonneg fun p _ => mul_nonneg (hprod0 p) (ind_nonneg _ _)
  have hc0 : 0 ≤ ind acc ω * ind (PairConflict op) (bs ++ [ω]) :=
    mul_nonneg (ind_nonneg _ _) (ind_nonneg _ _)
  by_cases h : acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p
  · have hL : ind (fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) ω = 1 := ite_eq_left h
    rw [hL]
    obtain ⟨ha, p, hsome, hne⟩ := h
    cases hf : bs.findSome? (fun b => op b p) with
    | some w =>
      obtain ⟨b, hb, hbw⟩ := List.exists_of_findSome?_eq_some hf
      have hconf : ConflictAt op (ω, b) p := by
        refine ⟨hsome, by simp [hbw], ?_⟩
        rw [hbw]
        simpa [firstTable, hf] using hne
      have hpc : PairConflict op (bs ++ [ω]) := ⟨ω, by simp, b, by simp [hb], p, hconf⟩
      have h1 : ind acc ω = 1 := ite_eq_left ha
      have h2 : ind (PairConflict op) (bs ++ [ω]) = 1 := ite_eq_left hpc
      rw [h1, h2, one_mul]
      linarith
    | none =>
      have hall : ∀ b ∈ bs, op b p = none := List.findSome?_eq_none_iff.1 hf
      have hprod : (bs.map fun b => ind (fun b => op b p = none) b).prod = 1 :=
        List.prod_eq_one (by
          simp only [List.mem_map]
          rintro _ ⟨b, hb, rfl⟩
          exact ite_eq_left (hall b hb))
      have h1 : ind (fun ω => acc ω ∧ (op ω p).isSome) ω = 1 := ite_eq_left ⟨ha, hsome⟩
      have h2 := Finset.single_le_sum (f := fun p => (bs.map fun b => ind (fun b => op b p = none) b).prod *
          ind (fun ω => acc ω ∧ (op ω p).isSome) ω) (fun p _ => mul_nonneg (hprod0 p) (ind_nonneg _ _))
        (Finset.mem_univ p)
      simp only [hprod, h1, one_mul] at h2
      linarith
  · have hL : ind (fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) ω = 0 := ite_eq_right h
    rw [hL]
    linarith

open Classical in
/-- **The positions no accepted rerun opened**: `Σ_p q_p (1 − q_p/ε)^k ≤ N·ε/(ek)` over `k` accepted reruns, with `q_p`
the probability that a run accepts and opens `p`. -/
theorem expectAcc_missing_le [Fintype P] (op : β → P → Option V) (g : Game β) (s : Strategy g)
    (acc : β → Prop) (k : ℕ) (hk : 1 ≤ k) :
    expectAcc g s acc k (fun bs => ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
        expect (ind fun ω => acc ω ∧ (op ω p).isSome) g s) ≤
      Fintype.card P * expect (ind acc) g s / (Real.exp 1 * k) := by
  set ε := expect (ind acc) g s with hεdef
  have hε0 : 0 ≤ ε := expect_nonneg (ind_nonneg acc) g s
  rcases hε0.eq_or_lt with h0 | hpos
  · unfold expectAcc
    rw [← hεdef, ← h0, zero_pow (by omega), div_zero]
    simp
  set q : P → ℝ := fun p => expect (ind fun ω => acc ω ∧ (op ω p).isSome) g s with hqdef
  have hB : ∀ p, expect (fun b => ind acc b * ind (fun b => op b p = none) b) g s = ε - q p := by
    intro p
    rw [hqdef, hεdef, ← expect_sub]
    refine congrArg (fun f => expect f g s) (funext fun b => ?_)
    unfold ind
    by_cases h1 : acc b <;> by_cases h2 : op b p = none <;> simp [h1, h2, Option.isSome_iff_ne_none]
  have hterm : ∀ p, expectAcc g s acc k (fun bs => (bs.map fun b => ind (fun b => op b p = none) b).prod * q p) =
      q p * (ε - q p) ^ k / ε ^ k := by
    intro p
    unfold expectAcc
    rw [← hεdef]
    congr 1
    rw [show (fun bs : List β => (bs.map (ind acc)).prod *
        ((bs.map fun b => ind (fun b => op b p = none) b).prod * q p)) =
      fun bs => q p * ((bs.map (ind acc)).prod * (bs.map fun b => ind (fun b => op b p = none) b).prod) from
        funext fun bs => by ring,
      expectN_const_mul, expectN_prod_mul_prod, hB p]
  have hq0 : ∀ p, 0 ≤ q p := fun p => expect_nonneg (ind_nonneg _) g s
  have hqε : ∀ p, q p ≤ ε := fun p =>
    expect_mono (ind_mono fun b (h : acc b ∧ (op b p).isSome) => h.1) g s
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
  rw [expectAcc_sum g s acc Finset.univ k
    (fun p bs => (bs.map fun b => ind (fun b => op b p = none) b).prod * q p)]
  calc ∑ p, expectAcc g s acc k (fun bs => (bs.map fun b => ind (fun b => op b p = none) b).prod * q p)
      = ∑ p, q p * (ε - q p) ^ k / ε ^ k := Finset.sum_congr rfl fun p _ => hterm p
    _ ≤ ∑ _p : P, ε / (Real.exp 1 * k) := Finset.sum_le_sum fun p _ => hmiss p
    _ = Fintype.card P * ε / (Real.exp 1 * k) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring

open Classical in
/-- **The accepted fresh run's off-table mass, against `k` accepted reruns, linearly**: `ε · expectAcc_{k+1}` of a
conflicting pair among the `k + 1` accepted runs, and `N·ε/(ek)` for the positions no accepted rerun opened. Where
`expectAcc_off_le` has `(k/ε)·Pr₂[conflict]`, the fresh run's acceptance stays here with its conflict. -/
theorem expectAcc_off_le_lin [Fintype P] [DecidableEq V] (op : β → P → Option V) (z₀ : P → V)
    (g : Game β) (s : Strategy g) (acc : β → Prop) (k : ℕ) (hk : 1 ≤ k) :
    expectAcc g s acc k (fun bs =>
        expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s) ≤
      expect (ind acc) g s * expectAcc g s acc (k + 1) (ind (PairConflict op)) +
        Fintype.card P * expect (ind acc) g s / (Real.exp 1 * k) := by
  have hε0 : 0 ≤ expect (ind acc) g s := expect_nonneg (ind_nonneg acc) g s
  rcases hε0.eq_or_lt with h0 | hpos
  · have hz : expectAcc g s acc k (fun bs =>
        expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s) = 0 := by
      unfold expectAcc
      rw [← h0, zero_pow (by omega), div_zero]
    rw [hz, ← h0]
    simp
  have hpt : ∀ bs : List β,
      expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s ≤
        expect (fun ω => ind acc ω * ind (PairConflict op) (bs ++ [ω])) g s +
          ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
            expect (ind fun ω => acc ω ∧ (op ω p).isSome) g s := by
    intro bs
    refine (expect_mono (ind_acc_off_le_lin op z₀ acc bs) g s).trans (le_of_eq ?_)
    rw [expect_add, expect_sum]
    congr 1
    exact Finset.sum_congr rfl fun p _ => expect_const_mul _ _ g s
  calc _ ≤ expectAcc g s acc k (fun bs => expect (fun ω => ind acc ω * ind (PairConflict op) (bs ++ [ω])) g s +
          ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
            expect (ind fun ω => acc ω ∧ (op ω p).isSome) g s) := expectAcc_mono g s acc k hpt
    _ = expectAcc g s acc k (fun bs => expect (fun ω => ind acc ω * ind (PairConflict op) (bs ++ [ω])) g s) +
          expectAcc g s acc k (fun bs => ∑ p, (bs.map fun b => ind (fun b => op b p = none) b).prod *
            expect (ind fun ω => acc ω ∧ (op ω p).isSome) g s) := expectAcc_add g s acc k _ _
    _ ≤ _ := by
        rw [← mul_expectAcc_succ g s acc k (ind (PairConflict op)) hpos]
        linarith [expectAcc_missing_le op g s acc k hk]

open Classical in
/-- **The product form for accepted reruns, linearly.** If every run list whose table fails leaves an accepted fresh run
at least `ε − ε_c` of off-table mass (`hX`), then `ε · Pr[fails] ≤ ε_c + N·ε/(ek) + ε · expectAcc_{k+1}[PairConflict]`
over the extractor's `k` accepted reruns. -/
theorem joint_acc_le_lin [Fintype P] [DecidableEq V] (op : β → P → Option V) (z₀ : P → V)
    (g : Game β) (s : Strategy g) (acc : β → Prop) (X : List β → Prop) (εc : ℝ) (hεc : 0 ≤ εc)
    (hX : ∀ bs, ¬ X bs → expect (ind acc) g s ≤
      εc + expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s)
    (k : ℕ) (hk : 1 ≤ k) :
    expect (ind acc) g s * expectAcc g s acc k (ind fun bs => ¬ X bs) ≤
      εc + Fintype.card P * expect (ind acc) g s / (Real.exp 1 * k) +
        expect (ind acc) g s * expectAcc g s acc (k + 1) (ind (PairConflict op)) := by
  have hoff := expectAcc_off_le_lin op z₀ g s acc k hk
  set ε := expect (ind acc) g s with hεdef
  have hε0 : 0 ≤ ε := expect_nonneg (ind_nonneg acc) g s
  rcases hε0.eq_or_lt with h0 | hpos
  · rw [← h0]
    simpa using hεc
  set M := fun bs : List β => expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s
  have hM0 : ∀ bs, 0 ≤ M bs := fun bs => expect_nonneg (ind_nonneg _) g s
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
  linarith

/-- **The level-0 conflict finder**: a run, then `k` waits of at most `R` runs each. -/
def conflictG (g : Game β) (R k : ℕ) : Game (β × List (List β)) :=
  g.bind fun a => (runsG (runsG g R) k).map fun bl => (a, bl)

/-- `s` in every run. -/
def conflictStrat (g : Game β) (s : Strategy g) (R k : ℕ) : Strategy (conflictG g R k) :=
  Strategy.bind g _ s fun _ => Strategy.map _ _ (runsStrat (runsG g R) (runsStrat g s R) k)

theorem expect_conflictG (g : Game β) (s : Strategy g) (R k : ℕ) (F : β × List (List β) → ℝ) :
    expect F (conflictG g R k) (conflictStrat g s R k) =
      expect (fun a => expectN (runsG g R) (runsStrat g s R) k fun bl => F (a, bl)) g s := by
  unfold conflictG conflictStrat
  rw [expect_bind]
  refine congrArg (fun f => expect f g s) (funext fun a => ?_)
  rw [expect_map, expect_runsG]

/-- **The conflict finder succeeds**: its run accepts, every wait finds an accepted run, and two of those `k + 1`
accepted runs conflict. -/
def ConflictSucc (acc : β → Prop) (op : β → P → Option V) (x : β × List (List β)) : Prop :=
  acc x.1 ∧ ∃ bs, firsts acc x.2 = some bs ∧ PairConflict op (x.1 :: bs)

theorem ind_conflictSucc (acc : β → Prop) (op : β → P → Option V) (a : β) (bl : List (List β)) :
    ind (ConflictSucc acc op) (a, bl) =
      ind acc a * (firsts acc bl).elim 0 fun bs => ind (PairConflict op) (a :: bs) := by
  cases h : firsts acc bl with
  | none => simp [ConflictSucc, ind, h]
  | some bs =>
      by_cases ha : acc a <;> by_cases hc : PairConflict op (a :: bs) <;> simp [ConflictSucc, ind, h, ha, hc]

/-- **The conflict finder's success** is the law of `k + 1` runs, weighted by their acceptance, having a conflicting
pair, times the waits' `geom ε R ^ k`. -/
theorem expect_conflictSucc (g : Game β) (s : Strategy g) (acc : β → Prop) (op : β → P → Option V) (R k : ℕ) :
    expect (ind (ConflictSucc acc op)) (conflictG g R k) (conflictStrat g s R k) =
      geom (expect (ind acc) g s) R ^ k *
        expectN g s (k + 1) (fun bs => (bs.map (ind acc)).prod * ind (PairConflict op) bs) := by
  rw [expect_conflictG]
  have h : ∀ a, expectN (runsG g R) (runsStrat g s R) k (fun bl => ind (ConflictSucc acc op) (a, bl)) =
      geom (expect (ind acc) g s) R ^ k *
        expectN g s k (fun bs => ((a :: bs).map (ind acc)).prod * ind (PairConflict op) (a :: bs)) := by
    intro a
    rw [show (fun bl => ind (ConflictSucc acc op) (a, bl)) = fun bl =>
        ind acc a * (firsts acc bl).elim 0 (fun bs => ind (PairConflict op) (a :: bs)) from
        funext fun bl => ind_conflictSucc acc op a bl,
      expectN_const_mul, expectN_firsts]
    have e : (fun bs : List β => ((a :: bs).map (ind acc)).prod * ind (PairConflict op) (a :: bs)) =
        fun bs => ind acc a * ((bs.map (ind acc)).prod * ind (PairConflict op) (a :: bs)) :=
      funext fun bs => by simp only [List.map_cons, List.prod_cons]; ring
    rw [e, expectN_const_mul]
    ring
  rw [show (fun a => expectN (runsG g R) (runsStrat g s R) k fun bl => ind (ConflictSucc acc op) (a, bl)) =
      fun a => geom (expect (ind acc) g s) R ^ k *
        expectN g s k (fun bs => ((a :: bs).map (ind acc)).prod * ind (PairConflict op) (a :: bs)) from
      funext h, expect_const_mul]
  rfl

/-- **The finder against the accepted law**: `ε · expectAcc_{k+1}[PairConflict] ≤ Pr[ConflictSucc] + (k + 1)/(eR)`. -/
theorem conflict_succ_ge (g : Game β) (s : Strategy g) (acc : β → Prop) (op : β → P → Option V) (R k : ℕ)
    (hR : 1 ≤ R) :
    expect (ind acc) g s * expectAcc g s acc (k + 1) (ind (PairConflict op)) ≤
      expect (ind (ConflictSucc acc op)) (conflictG g R k) (conflictStrat g s R k) + (k + 1) / (Real.exp 1 * R) := by
  have h := expectAcc_le_waits g s acc (k + 1) R hR (ind (PairConflict op)) (ind_nonneg _) (ind_le_one _)
  rw [expect_conflictSucc]
  set ε := expect (ind acc) g s
  set X := expectN g s (k + 1) (fun bs => (bs.map (ind acc)).prod * ind (PairConflict op) bs)
  have hε1 : ε ≤ 1 := expect_ind_le_one acc g s
  have hg0 : 0 ≤ geom ε R := geom_nonneg hε1 R
  have hεg : ε * geom ε R ≤ 1 := LinkFinder.geom_mul_le_one hε1 R
  have hX0 : 0 ≤ X :=
    expectN_nonneg g s (fun bs => mul_nonneg (prod_ind_nonneg acc bs) (ind_nonneg _ _)) (k + 1)
  have hgk : 0 ≤ geom ε R ^ k := pow_nonneg hg0 k
  have hw : ε * geom ε R ^ (k + 1) * X ≤ geom ε R ^ k * X := by
    have e : ε * geom ε R ^ (k + 1) * X = geom ε R ^ k * (ε * geom ε R) * X := by ring
    rw [e]
    refine mul_le_mul_of_nonneg_right ?_ hX0
    calc geom ε R ^ k * (ε * geom ε R) ≤ geom ε R ^ k * 1 := mul_le_mul_of_nonneg_left hεg hgk
      _ = _ := mul_one _
  push_cast at h
  linarith

/-- **A wait's cost**: its runs' costs, up to and including the first accepted one, or all of them. -/
noncomputable def waitCostC (acc : β → Prop) (c : β → ℝ) : List β → ℝ
  | [] => 0
  | b :: bs => c b + (1 - ind acc b) * waitCostC acc c bs

theorem waitCostC_nonneg (acc : β → Prop) (c : β → ℝ) (hc : ∀ b, 0 ≤ c b) : ∀ bs, 0 ≤ waitCostC acc c bs
  | [] => le_rfl
  | b :: bs => add_nonneg (hc b) (mul_nonneg (by linarith [ind_le_one acc b]) (waitCostC_nonneg acc c hc bs))

/-- **A wait costs `geom ε R` runs** in expectation. -/
theorem expectN_waitCostC (g : Game β) (s : Strategy g) (acc : β → Prop) (c : β → ℝ) : ∀ R : ℕ,
    expectN g s R (waitCostC acc c) = geom (expect (ind acc) g s) R * expect c g s
  | 0 => by simp [expectN, waitCostC, geom]
  | R + 1 => by
      have ih := expectN_waitCostC g s acc c R
      have h : ∀ b, expectN g s R (fun bs => waitCostC acc c (b :: bs)) =
          c b + (1 - ind acc b) * (geom (expect (ind acc) g s) R * expect c g s) := by
        intro b
        simp only [waitCostC]
        rw [expectN_add, expectN_const, expectN_const_mul, ih]
      simp only [expectN]
      rw [show (fun b => expectN g s R fun bs => waitCostC acc c (b :: bs)) = fun b =>
          c b + (geom (expect (ind acc) g s) R * expect c g s) * (1 - ind acc b) from
          funext fun b => by rw [h b]; ring,
        expect_add, expect_const_mul, expect_sub, expect_const, geom_succ]
      ring

/-- **The conflict finder's cost**: its run's, and when that accepts, its waits'. -/
noncomputable def conflictCost (acc : β → Prop) (c : β → ℝ) (x : β × List (List β)) : ℝ :=
  c x.1 + ind acc x.1 * (x.2.map (waitCostC acc c)).sum

/-- **The conflict finder plays at most `k + 1` runs** in expectation, whatever `ε`: `1 + ε · k · geom ε R`. -/
theorem expect_conflictCost_le (g : Game β) (s : Strategy g) (acc : β → Prop) (c : β → ℝ) (hc : ∀ b, 0 ≤ c b)
    (R k : ℕ) :
    expect (conflictCost acc c) (conflictG g R k) (conflictStrat g s R k) ≤ (k + 1) * expect c g s := by
  rw [expect_conflictG]
  have hpt : ∀ a, expectN (runsG g R) (runsStrat g s R) k (fun bl => conflictCost acc c (a, bl)) =
      c a + ind acc a * (k * (geom (expect (ind acc) g s) R * expect c g s)) := by
    intro a
    simp only [conflictCost]
    rw [expectN_add, expectN_const, expectN_const_mul, expectN_sum_map, expect_runsG, expectN_waitCostC]
  rw [show (fun a => expectN (runsG g R) (runsStrat g s R) k fun bl => conflictCost acc c (a, bl)) =
      fun a => c a + (k * (geom (expect (ind acc) g s) R * expect c g s)) * ind acc a from
      funext fun a => by rw [hpt a]; ring, expect_add, expect_const_mul]
  set ε := expect (ind acc) g s
  have hε1 : ε ≤ 1 := expect_ind_le_one acc g s
  have hεg : ε * geom ε R ≤ 1 := LinkFinder.geom_mul_le_one hε1 R
  have hc0 : 0 ≤ expect c g s := expect_nonneg hc g s
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hw : k * (geom ε R * expect c g s) * ε ≤ k * expect c g s := by
    have e : k * (geom ε R * expect c g s) * ε = k * expect c g s * (ε * geom ε R) := by ring
    rw [e]
    exact mul_le_of_le_one_right (mul_nonneg hk0 hc0) hεg
  linarith

/-- **Level 0 under expected-time collision resistance, linearly**: where every success of the conflict finder outputs
a collision of `Hc` and the finder obeys the generic bound at its cost,
`ε · Pr[fails] ≤ ε_c + N·ε/(ek) + (k + 1)·E[c]/2^256 + (k + 1)/(eR)`: weight 1, at `k + 1` runs. -/
theorem joint_acc_le_ecr [Fintype P] [DecidableEq V] {Dh : Type} (Hc : List UInt8 → Dh)
    (op : β → P → Option V) (z₀ : P → V) (g : Game β) (s : Strategy g) (acc : β → Prop) (X : List β → Prop)
    (εc : ℝ) (hεc : 0 ≤ εc)
    (hX : ∀ bs, ¬ X bs → expect (ind acc) g s ≤
      εc + expect (ind fun ω => acc ω ∧ ∃ p, Off op (firstTable op z₀ bs) ω p) g s)
    (k : ℕ) (hk : 1 ≤ k) (R : ℕ) (hR : 1 ≤ R)
    (out : β × List (List β) → Option (List UInt8 × List UInt8)) (c : β → ℝ) (hc : ∀ b, 0 ≤ c b)
    (hout : ∀ x, ConflictSucc acc op x → ∃ a b, out x = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b)
    (hCR : Assumptions.SHA512CRExpected Hc (conflictG g R k) (conflictStrat g s R k) out (conflictCost acc c)) :
    expect (ind acc) g s * expectAcc g s acc k (ind fun bs => ¬ X bs) ≤
      εc + Fintype.card P * expect (ind acc) g s / (Real.exp 1 * k) +
        (k + 1) * expect c g s / 2 ^ (256 : ℝ) + (k + 1) / (Real.exp 1 * R) := by
  have h1 := joint_acc_le_lin op z₀ g s acc X εc hεc hX k hk
  have h2 := conflict_succ_ge g s acc op R k hR
  have h3 : expect (ind (ConflictSucc acc op)) (conflictG g R k) (conflictStrat g s R k) ≤
      (prob (fun x => ∃ a b, out x = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b) (conflictG g R k)
        (conflictStrat g s R k)).toReal := by
    rw [prob_toReal]
    exact expect_mono (ind_mono hout) _ _
  have h4 : expect (conflictCost acc c) (conflictG g R k) (conflictStrat g s R k) / 2 ^ (256 : ℝ) ≤
      (k + 1) * expect c g s / 2 ^ (256 : ℝ) :=
    div_le_div_of_nonneg_right (expect_conflictCost_le g s acc c hc R k) (by positivity)
  have h5 := h3.trans (hCR.trans h4)
  linarith

end Extract

end FlockSoundness
