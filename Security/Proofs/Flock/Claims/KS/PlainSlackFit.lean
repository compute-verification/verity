import Proofs.Flock.Claims.KS.ForkCoupling
import Proofs.Flock.Level3.ArithFacts

/-!
# The slack fits the plain weight at the coupling's chances

`slackEsc_fitsFork_plain` asks that the slack's escapes' weights total at most level 0's `2·2^{d₀}·Q₀`, the room
`plain_fork_room` leaves in the plain weight. At the chances `δ_t` of the coupling at `fnAt`
(`FnFork.δN`, `δ_t = |X|^{-min (n t) M}`, `FnFork.δAt_eq`) with every `n t ≥ 1`, each `δ_t` is at most `2^{-128}`, and:

* `slackW_le_geom`, `sum_slackW_le`: a fork's slack weights fall geometrically past its round, so they total at most
  `6ε` at chances at most `ε ≤ 1/2`, whatever the number of rounds;
* `slackC_le_mul`, `sum_slackC_le`: by Cauchy–Schwarz, the escapes' weights total at most the levels' total weight
  times `2·L·6ε`;
* `fast100_slackNums`, `sum_cOf_le`: at a `fast100` schedule the levels weigh under `2^33` in all, there are at most
  `7`, and level 0 makes a query;
* `slack_fits_fast100`, `plain_slack_fits`: so the slack fits, with room to spare, at the deployed plain verifier's
  coins (`GF128`, `M = 65536`).
-/

namespace Claims.Flock

/-- **`Δ_{j,i}` at chances at most `ε`** is at most `ε^{i-j}`. -/
theorem slackΔ_le_pow {δ : ℕ → ℝ} {ε : ℝ} (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, δ t ≤ ε) (j i : ℕ) :
    slackΔ δ j i ≤ ε ^ (i - j) := by
  unfold slackΔ
  calc ∏ t ∈ Finset.Ico j i, δ t ≤ ∏ _t ∈ Finset.Ico j i, ε :=
        Finset.prod_le_prod₀ (fun t _ => hδ0 t) fun t _ => hδ t
    _ = ε ^ (i - j) := by rw [Finset.prod_const, Nat.card_Ico]

/-- **The slack's weight at round `i` of a fork at round `j < K`, at chances at most `ε ≤ 1/2`**: at most
`3ε·2^{-(i-j)}` from `j` on, and `0` before. -/
theorem slackW_le_geom {K : ℕ} {δ : ℕ → ℝ} {ε : ℝ} (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, δ t ≤ ε) (hε0 : 0 ≤ ε)
    (hε : ε ≤ 1 / 2) {j : ℕ} (hj : j < K) (i : ℕ) :
    slackW K δ j i ≤ if j ≤ i then 3 * ε * (1 / 2) ^ (i - j) else 0 := by
  have hpow : ∀ e : ℕ, 1 ≤ e → ε ^ e ≤ 2 * ε * (1 / 2) ^ e := fun e he => by
    obtain ⟨e, rfl⟩ := Nat.exists_eq_add_of_le' he
    rw [pow_succ, pow_succ]
    have := pow_le_pow_left₀ hε0 hε e
    nlinarith [pow_nonneg hε0 e]
  unfold slackW
  split_ifs with h1 h2 h3 h2 h3 h3
  · have hK : K - j = i - j + 1 := by omega
    have a := (slackΔ_le_pow hδ0 hδ j i).trans (hpow _ (by omega))
    have b := (slackΔ_le_pow hδ0 hδ j K).trans (hpow _ (by omega))
    rw [hK, pow_succ] at b
    nlinarith [pow_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2) (i - j)]
  · omega
  · have a := (slackΔ_le_pow hδ0 hδ j i).trans (hpow _ (by omega))
    nlinarith [pow_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2) (i - j)]
  · omega
  · have hK : K - j = i - j + 1 := by omega
    have b := (slackΔ_le_pow hδ0 hδ j K).trans (hpow _ (by omega))
    rw [hK, pow_succ] at b
    nlinarith [pow_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2) (i - j)]
  · omega
  · rw [add_zero]; positivity
  · simp

/-- **The slack's weights of a fork at round `j < K`, at chances at most `ε ≤ 1/2`, total at most `6ε`**, whatever
`K`: they fall geometrically past `j`. -/
theorem sum_slackW_le {K : ℕ} {δ : ℕ → ℝ} {ε : ℝ} (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, δ t ≤ ε) (hε0 : 0 ≤ ε)
    (hε : ε ≤ 1 / 2) {j : ℕ} (hj : j < K) :
    ∑ i : Fin K, slackW K δ j i.val ≤ 6 * ε := by
  calc ∑ i : Fin K, slackW K δ j i.val
      ≤ ∑ i ∈ Finset.range K, if j ≤ i then 3 * ε * (1 / 2) ^ (i - j) else 0 := by
        rw [← Fin.sum_univ_eq_sum_range]
        exact Finset.sum_le_sum fun i _ => slackW_le_geom hδ0 hδ hε0 hε hj i.val
    _ = ∑ i ∈ Finset.Ico j K, 3 * ε * (1 / 2) ^ (i - j) := by
        rw [← Finset.sum_filter]
        congr 1
        ext i
        simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico]
        omega
    _ = 3 * ε * ∑ k ∈ Finset.range (K - j), (1 / 2 : ℝ) ^ k := by
        rw [Finset.sum_Ico_eq_sum_range, Finset.mul_sum]
        exact Finset.sum_congr rfl fun k _ => by rw [Nat.add_sub_cancel_left]
    _ ≤ 3 * ε * 2 := mul_le_mul_of_nonneg_left (sum_geometric_two_le _) (by positivity)
    _ = 6 * ε := by ring

/-- **Round `i`'s weight is at most the levels' total weight times the slack weights' total at `i`**
(Cauchy–Schwarz). -/
theorem slackC_le_mul {n : ℕ} (sch : FlockSoundness.Accounting.Schedule)
    (w : Fin 2 → Fin sch.levels.length → Fin n → ℝ) (hw : ∀ r ℓ i, 0 ≤ w r ℓ i) (i : Fin n) :
    slackC sch w i ≤ (∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1)) *
      ∑ a : Fin 2 × Fin sch.levels.length, w a.1 a.2 i := by
  unfold slackC
  rw [Finset.sum_congr rfl fun a _ => Real.sqrt_mul (cOf_nonneg sch (a.2.val + 1)) (w a.1 a.2 i)]
  refine (Finset.sum_mul_sq_le_sq_mul_sq _ _ _).trans (le_of_eq ?_)
  congr 1
  · exact Finset.sum_congr rfl fun a _ => Real.sq_sqrt (cOf_nonneg sch _)
  · exact Finset.sum_congr rfl fun a _ => Real.sq_sqrt (hw a.1 a.2 i)

/-- **The slack's escapes' weights at chances at most `ε ≤ 1/2`**: at most the levels' total weight times `2·L·6ε`,
`L` the schedule's levels, whatever the rounds `K`. -/
theorem sum_slackC_le {K : ℕ} (sch : FlockSoundness.Accounting.Schedule)
    (j : Fin 2 → Fin sch.levels.length → Fin K) {δ : ℕ → ℝ} {ε : ℝ}
    (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, δ t ≤ ε) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 2) :
    ∑ i : Fin K, slackC sch (fun r ℓ (i : Fin K) => slackW K δ (j r ℓ).val i.val) i ≤
      (∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1)) * (2 * sch.levels.length * (6 * ε)) := by
  have hw : ∀ (r : Fin 2) (ℓ : Fin sch.levels.length) (i : Fin K), 0 ≤ slackW K δ (j r ℓ).val i.val :=
    fun r ℓ i => slackW_nonneg hδ0 K _ _
  calc ∑ i : Fin K, slackC sch (fun r ℓ (i : Fin K) => slackW K δ (j r ℓ).val i.val) i
      ≤ ∑ i : Fin K, (∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1)) *
          ∑ a : Fin 2 × Fin sch.levels.length, slackW K δ (j a.1 a.2).val i.val :=
        Finset.sum_le_sum fun i _ => slackC_le_mul sch _ hw i
    _ = (∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1)) *
          ∑ a : Fin 2 × Fin sch.levels.length, ∑ i : Fin K, slackW K δ (j a.1 a.2).val i.val := by
        rw [← Finset.mul_sum, Finset.sum_comm]
    _ ≤ (∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1)) *
          ∑ _a : Fin 2 × Fin sch.levels.length, 6 * ε :=
        mul_le_mul_of_nonneg_left
          (Finset.sum_le_sum fun a _ => sum_slackW_le hδ0 hδ hε0 hε (j a.1 a.2).isLt)
          (Finset.sum_nonneg fun a _ => cOf_nonneg sch _)
    _ = (∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1)) * (2 * sch.levels.length * (6 * ε)) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, nsmul_eq_mul]
        push_cast
        ring

/-- **The `fast100` numbers the slack's fit reads**: the levels' `Σ 2^{d}·Q` is under `2^32`, there are at most `7`
levels, and level 0 makes a query. -/
theorem fast100_slackNums {m : ℕ} {sch : FlockSoundness.Accounting.Schedule}
    (h : FlockSoundness.Accounting.fast100 m = some sch) :
    (sch.levels.map fun l => 2 ^ l.logLen * l.queries).sum ≤ 2 ^ 32 ∧ sch.levels.length ≤ 7 ∧
      1 ≤ (sch.levels.headD default).queries := by
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : FlockSoundness.Accounting.fast100 m = none := by unfold FlockSoundness.Accounting.fast100; split <;> simp_all
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  interval_cases m <;> (simp only [FlockSoundness.Accounting.fast100, Option.some.injEq] at h; subst h; decide +kernel)

/-- **The levels past 0 weigh under `2^33` in all**, both reps, at a `fast100` schedule. -/
theorem sum_cOf_le {m : ℕ} {sch : FlockSoundness.Accounting.Schedule}
    (h : FlockSoundness.Accounting.fast100 m = some sch) :
    ∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1) ≤ 2 ^ 33 := by
  obtain ⟨hS, -, -⟩ := fast100_slackNums h
  set f : FlockSoundness.Accounting.Level → ℝ := fun l => 2 ^ l.logLen * l.queries with hf
  have hd : (default : FlockSoundness.Accounting.Level).queries = 0 := rfl
  have hhead := head_add_sum_shift f (by simp [hf, hd]) sch.levels
  have hfh : 0 ≤ f (sch.levels.headD default) := by simp only [hf]; positivity
  have hsum : ∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1) =
      2 * ∑ ℓ : Fin sch.levels.length, f (sch.levels.getD (ℓ.val + 1) default) := by
    simp only [Fintype.sum_prod_type, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      Nat.cast_ofNat]
    rfl
  have hcast : (sch.levels.map f).sum = (((sch.levels.map fun l => 2 ^ l.logLen * l.queries).sum : ℕ) : ℝ) := by
    simp only [hf, Nat.cast_list_sum, List.map_map, Function.comp_def, Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat]
  have hS' : (((sch.levels.map fun l => 2 ^ l.logLen * l.queries).sum : ℕ) : ℝ) ≤ 2 ^ 32 := by
    exact_mod_cast hS
  rw [hsum]
  linarith

/-- **The slack fits at a `fast100` schedule at chances at most `2^{-128}`**: its escapes' weights total at most
level 0's `2·2^{d₀}·Q₀`, the room `plain_fork_room` leaves, whatever the rounds `K`. -/
theorem slack_fits_fast100 {K m : ℕ} {sch : FlockSoundness.Accounting.Schedule}
    (hsch : FlockSoundness.Accounting.fast100 m = some sch)
    (j : Fin 2 → Fin sch.levels.length → Fin K) {δ : ℕ → ℝ} (hδ0 : ∀ t, 0 ≤ δ t)
    (hδ : ∀ t, δ t ≤ ((2 : ℝ) ^ 128)⁻¹) :
    ∑ i : Fin K, slackC sch (fun r ℓ (i : Fin K) => slackW K δ (j r ℓ).val i.val) i ≤
      2 * ((2 : ℝ) ^ (sch.levels.headD default).logLen * (sch.levels.headD default).queries) := by
  obtain ⟨-, hL, hq⟩ := fast100_slackNums hsch
  have hε0 : (0 : ℝ) ≤ ((2 : ℝ) ^ 128)⁻¹ := by positivity
  have hε : ((2 : ℝ) ^ 128)⁻¹ ≤ 1 / 2 := by norm_num
  refine (sum_slackC_le sch j hδ0 hδ hε0 hε).trans ?_
  have hc := sum_cOf_le hsch
  have hc0 : 0 ≤ ∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1) :=
    Finset.sum_nonneg fun a _ => cOf_nonneg sch _
  have hL' : (sch.levels.length : ℝ) ≤ 7 := by exact_mod_cast hL
  have hq' : (1 : ℝ) ≤ (sch.levels.headD default).queries := by exact_mod_cast hq
  have hp : (1 : ℝ) ≤ 2 ^ (sch.levels.headD default).logLen := one_le_pow₀ (by norm_num)
  have h1 : (∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1)) *
      (2 * sch.levels.length * (6 * ((2 : ℝ) ^ 128)⁻¹)) ≤ 2 ^ 33 * (2 * 7 * (6 * ((2 : ℝ) ^ 128)⁻¹)) :=
    mul_le_mul hc (by gcongr) (by positivity) (by positivity)
  have h2 : (2 : ℝ) ^ 33 * (2 * 7 * (6 * ((2 : ℝ) ^ 128)⁻¹)) ≤ 2 := by norm_num
  nlinarith

/-- **The coupling's chances with every revealed length positive** are at most `|X|⁻¹`. -/
theorem δN_le_inv {K M : ℕ} {X : Type} [Fintype X] [DecidableEq X] [Nonempty X] (hM : 1 ≤ M) {n : Fin K → ℕ}
    (hn : ∀ t, 1 ≤ n t) (t : ℕ) : FnFork.δN X M n t ≤ (Fintype.card X : ℝ)⁻¹ := by
  unfold FnFork.δN
  split_ifs with ht
  · rw [FnFork.δAt_eq]
    have h1 : (Fintype.card X : ℝ)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ (by exact_mod_cast Fintype.card_pos)
    have h0 : (0 : ℝ) ≤ (Fintype.card X : ℝ)⁻¹ := by positivity
    calc (Fintype.card X : ℝ)⁻¹ ^ min (n ⟨t, ht⟩) M ≤ (Fintype.card X : ℝ)⁻¹ ^ 1 :=
          pow_le_pow_of_le_one h0 h1 (le_min (hn _) hM)
      _ = (Fintype.card X : ℝ)⁻¹ := pow_one _
  · positivity

end Claims.Flock

namespace FlockDeploy.KS

open Flock

/-- **The slack fits the plain weight at the coupling's chances**: at the deployed plain verifier, with the chances
`FnFork.δN GF128 65536 n` of the coupling at `fnAt` (`forkCoinCoupling_plain`) and every revealed length `n t ≥ 1`,
the slack's escapes' weights at the plain fork rounds total at most level 0's `2·2^{d₀}·Q₀`, as
`slackEsc_fitsFork_plain` and `PlainSimWrPAt` ask. -/
theorem plain_slack_fits {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h0 : (tables false x).head? = some (spec, st)) {m k0 : ℕ} {l₀ l₁ : FlockSoundness.Accounting.Level}
    {rest : List FlockSoundness.Accounting.Level} {ext : ℕ}
    (hsch : FlockSoundness.Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (n : Fin ((V false).k x) → ℕ) (hn : ∀ t, 1 ≤ n t) :
    ∑ i, Claims.Flock.slackC ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        (fun r ℓ (i : Fin ((V false).k x)) =>
          Claims.Flock.slackW ((V false).k x) (Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 n)
            (plainJ h0 hsch r ℓ).val i.val) i ≤
      2 * ((2 : ℝ) ^ l₀.logLen * l₀.queries) := by
  have hδ : ∀ t, Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 n t ≤ ((2 : ℝ) ^ 128)⁻¹ := fun t => by
    have := Claims.Flock.δN_le_inv (X := FlockLevel3.GF128) (M := 65536) (by norm_num) hn t
    rwa [FlockLevel3.card_F, Nat.cast_pow, Nat.cast_ofNat] at this
  exact Claims.Flock.slack_fits_fast100 hsch (plainJ h0 hsch) (Claims.Flock.FnFork.δN_nonneg n) hδ

end FlockDeploy.KS
