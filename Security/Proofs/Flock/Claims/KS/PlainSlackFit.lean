import Proofs.Flock.Claims.KS.ForkCoupling
import Proofs.Flock.Claims.KS.PlainSimW
import Proofs.Flock.Level3.ArithFacts
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# The slack fits the plain weight at the coupling's chances

`slackEsc_fitsFork_plain` asks that the slack's escapes' weights total at most level 0's `2·2^{d₀}·Q₀`, the room
`plain_fork_room` leaves in the plain weight. At the chances `δ_t` of the coupling at `fnAt`
(`FnFork.δN`, `δ_t = |A|^{-min (n t) M}`, `FnFork.δAt_eq`), each `δ_t` is at most `1` (`δN_le_one`), and at most
`2^{-128}` where `n t ≥ 1` (`δN_le_inv`), which a simulator's lengths are at every round before the last, the last
message's coin being revealed to nobody. The plain fork rounds are before the last round (`plainJ_succ_lt`), and:

* `slackΔ_le_pow`, `slackΔ_last_le`, `slackW_le_geom`, `sum_slackW_le`: a fork's slack weights fall geometrically past
  its round, so they total at most `8ε` at chances at most `ε ≤ 1/2` before the last round and at most `1` there,
  whatever the number of rounds;
* `slackC_le_mul`, `sum_slackC_le`: by Cauchy–Schwarz, the escapes' weights total at most the levels' total weight
  times `2·L·8ε`;
* `fast100_slackNums`, `sum_cOf_le`: at a `fast100` schedule the levels weigh under `2^33` in all, there are at most
  `7`, and level 0 makes a query;
* `slack_fits_fast100`, `plain_slack_fits`: so the slack fits, with room to spare, at the deployed plain verifier's
  coins (`GF128`, `M = 65536`);
* `plain_fork_le_fnAt`: and a fork bound at most the simulated fork at `fnAt` at a plain fork round is at most the
  accepting fork with a collision there (`plainForkEv`) and the slack at those chances, as `PlainSimWrPAtD` asks
  (`forkCoinCoupling_plain`, `slack_of_coupling`).
-/

namespace Claims.Flock

/-- **`Δ_{j,i}` at chances at most `ε` before the last round**, `i < K`: at most `ε^{i-j}`. -/
theorem slackΔ_le_pow {K : ℕ} {δ : ℕ → ℝ} {ε : ℝ} (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, t + 1 < K → δ t ≤ ε) (j : ℕ)
    {i : ℕ} (hi : i < K) : slackΔ δ j i ≤ ε ^ (i - j) := by
  unfold slackΔ
  calc ∏ t ∈ Finset.Ico j i, δ t ≤ ∏ _t ∈ Finset.Ico j i, ε :=
        Finset.prod_le_prod₀ (fun t _ => hδ0 t) fun t ht => hδ t (by have := (Finset.mem_Ico.1 ht).2; omega)
    _ = ε ^ (i - j) := by rw [Finset.prod_const, Nat.card_Ico]

/-- **`Δ_{j,K}` with the last round's chance at most `1`** is at most `ε^{K-1-j}`. -/
theorem slackΔ_last_le {K : ℕ} {δ : ℕ → ℝ} {ε : ℝ} (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, t + 1 < K → δ t ≤ ε)
    (hδ1 : ∀ t, δ t ≤ 1) (hε0 : 0 ≤ ε) {j : ℕ} (hj : j < K) : slackΔ δ j K ≤ ε ^ (K - 1 - j) := by
  obtain ⟨K, rfl⟩ : ∃ K', K = K' + 1 := ⟨K - 1, by omega⟩
  have h := slackΔ_le_pow hδ0 hδ j (i := K) (by omega)
  unfold slackΔ at h ⊢
  rw [Finset.prod_Ico_succ_top (by omega), Nat.add_sub_cancel]
  calc (∏ t ∈ Finset.Ico j K, δ t) * δ K ≤ ε ^ (K - j) * 1 := mul_le_mul h (hδ1 K) (hδ0 K) (pow_nonneg hε0 _)
    _ = ε ^ (K - j) := mul_one _

/-- **The slack's weight at round `i < K` of a fork at round `j`, `j + 1 < K`**, at chances at most `ε ≤ 1/2` before
the last round and at most `1` there: at most `4ε·2^{-(i-j)}` from `j` on, and `0` before. -/
theorem slackW_le_geom {K : ℕ} {δ : ℕ → ℝ} {ε : ℝ} (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, t + 1 < K → δ t ≤ ε)
    (hδ1 : ∀ t, δ t ≤ 1) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 2) {j : ℕ} (hj : j + 1 < K) {i : ℕ} (hi : i < K) :
    slackW K δ j i ≤ if j ≤ i then 4 * ε * (1 / 2) ^ (i - j) else 0 := by
  have hpow : ∀ e : ℕ, 1 ≤ e → ε ^ e ≤ 2 * ε * (1 / 2) ^ e := fun e he => by
    obtain ⟨e, rfl⟩ := Nat.exists_eq_add_of_le' he
    rw [pow_succ, pow_succ]
    have := pow_le_pow_left₀ hε0 hε e
    nlinarith [pow_nonneg hε0 e]
  have hp : 0 ≤ ε * (1 / 2 : ℝ) ^ (i - j) := by positivity
  unfold slackW
  by_cases h1 : j < i
  · rw [ite_eq_left h1, ite_eq_left h1.le]
    have a := (slackΔ_le_pow hδ0 hδ j hi).trans (hpow _ (by omega))
    by_cases h2 : i + 1 = K
    · have b := slackΔ_last_le hδ0 hδ hδ1 hε0 (j := j) (by omega)
      rw [show K - 1 - j = i - j by omega] at b
      have b' := b.trans (hpow _ (by omega))
      rw [ite_eq_left h2]
      nlinarith
    · rw [ite_eq_right h2]
      nlinarith
  · rw [ite_eq_right h1, ite_eq_right (show ¬ i + 1 = K by omega), add_zero]
    split_ifs
    · positivity
    · exact le_rfl

/-- **The slack's weights of a fork at round `j`, `j + 1 < K`, total at most `8ε`**, at chances at most `ε ≤ 1/2`
before the last round and at most `1` there, whatever `K`: they fall geometrically past `j`. -/
theorem sum_slackW_le {K : ℕ} {δ : ℕ → ℝ} {ε : ℝ} (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, t + 1 < K → δ t ≤ ε)
    (hδ1 : ∀ t, δ t ≤ 1) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 2) {j : ℕ} (hj : j + 1 < K) :
    ∑ i : Fin K, slackW K δ j i.val ≤ 8 * ε := by
  calc ∑ i : Fin K, slackW K δ j i.val
      ≤ ∑ i : Fin K, if j ≤ i.val then 4 * ε * (1 / 2) ^ (i.val - j) else 0 :=
        Finset.sum_le_sum fun i _ => slackW_le_geom hδ0 hδ hδ1 hε0 hε hj i.isLt
    _ = ∑ i ∈ Finset.range K, if j ≤ i then 4 * ε * (1 / 2) ^ (i - j) else 0 :=
        Fin.sum_univ_eq_sum_range (fun i => if j ≤ i then 4 * ε * (1 / 2) ^ (i - j) else 0) K
    _ = ∑ i ∈ Finset.Ico j K, 4 * ε * (1 / 2) ^ (i - j) := by
        rw [← Finset.sum_filter]
        congr 1
        ext i
        simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico]
        omega
    _ = 4 * ε * ∑ k ∈ Finset.range (K - j), (1 / 2 : ℝ) ^ k := by
        rw [Finset.sum_Ico_eq_sum_range, Finset.mul_sum]
        exact Finset.sum_congr rfl fun k _ => by rw [Nat.add_sub_cancel_left]
    _ ≤ 4 * ε * 2 := mul_le_mul_of_nonneg_left (sum_geometric_two_le _) (by positivity)
    _ = 8 * ε := by ring

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

/-- **The slack's escapes' weights**, at forks before the last round and chances at most `ε ≤ 1/2` before it and
at most `1` there: at most the levels' total weight times `2·L·8ε`, `L` the schedule's levels, whatever the rounds
`K`. -/
theorem sum_slackC_le {K : ℕ} (sch : FlockSoundness.Accounting.Schedule) (j : Fin 2 → Fin sch.levels.length → Fin K)
    (hj : ∀ r ℓ, (j r ℓ).val + 1 < K) {δ : ℕ → ℝ} {ε : ℝ} (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, t + 1 < K → δ t ≤ ε)
    (hδ1 : ∀ t, δ t ≤ 1) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 2) :
    ∑ i : Fin K, slackC sch (fun r ℓ (i : Fin K) => slackW K δ (j r ℓ).val i.val) i ≤
      (∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1)) * (2 * sch.levels.length * (8 * ε)) := by
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
          ∑ _a : Fin 2 × Fin sch.levels.length, 8 * ε :=
        mul_le_mul_of_nonneg_left
          (Finset.sum_le_sum fun a _ => sum_slackW_le hδ0 hδ hδ1 hε0 hε (hj a.1 a.2))
          (Finset.sum_nonneg fun a _ => cOf_nonneg sch _)
    _ = (∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1)) * (2 * sch.levels.length * (8 * ε)) := by
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
    (j : Fin 2 → Fin sch.levels.length → Fin K) (hj : ∀ r ℓ, (j r ℓ).val + 1 < K) {δ : ℕ → ℝ}
    (hδ0 : ∀ t, 0 ≤ δ t) (hδ : ∀ t, t + 1 < K → δ t ≤ ((2 : ℝ) ^ 128)⁻¹) (hδ1 : ∀ t, δ t ≤ 1) :
    ∑ i : Fin K, slackC sch (fun r ℓ (i : Fin K) => slackW K δ (j r ℓ).val i.val) i ≤
      2 * ((2 : ℝ) ^ (sch.levels.headD default).logLen * (sch.levels.headD default).queries) := by
  obtain ⟨-, hL, hq⟩ := fast100_slackNums hsch
  have hε0 : (0 : ℝ) ≤ ((2 : ℝ) ^ 128)⁻¹ := by positivity
  have hε : ((2 : ℝ) ^ 128)⁻¹ ≤ 1 / 2 := by norm_num
  refine (sum_slackC_le sch j hj hδ0 hδ hδ1 hε0 hε).trans ?_
  have hc := sum_cOf_le hsch
  have hc0 : 0 ≤ ∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1) :=
    Finset.sum_nonneg fun a _ => cOf_nonneg sch _
  have hL' : (sch.levels.length : ℝ) ≤ 7 := by exact_mod_cast hL
  have hq' : (1 : ℝ) ≤ (sch.levels.headD default).queries := by exact_mod_cast hq
  have hp : (1 : ℝ) ≤ 2 ^ (sch.levels.headD default).logLen := one_le_pow₀ (by norm_num)
  have h1 : (∑ a : Fin 2 × Fin sch.levels.length, cOf sch (a.2.val + 1)) *
      (2 * sch.levels.length * (8 * ((2 : ℝ) ^ 128)⁻¹)) ≤ 2 ^ 33 * (2 * 7 * (8 * ((2 : ℝ) ^ 128)⁻¹)) :=
    mul_le_mul hc (by gcongr) (by positivity) (by positivity)
  have h2 : (2 : ℝ) ^ 33 * (2 * 7 * (8 * ((2 : ℝ) ^ 128)⁻¹)) ≤ 2 := by norm_num
  nlinarith

/-- **The coupling's chances are at most `1`.** -/
theorem δN_le_one {K M : ℕ} {A : Type} [Fintype A] [DecidableEq A] [Nonempty A] (n : Fin K → ℕ) (t : ℕ) :
    FnFork.δN A M n t ≤ 1 := by
  unfold FnFork.δN
  split_ifs with ht
  · rw [FnFork.δAt_eq]
    exact pow_le_one₀ (by positivity) (inv_le_one_of_one_le₀ (by exact_mod_cast Fintype.card_pos))
  · exact zero_le_one

/-- **The coupling's chance at a round whose revealed length is positive** is at most `|A|⁻¹`. -/
theorem δN_le_inv {K M : ℕ} {A : Type} [Fintype A] [DecidableEq A] [Nonempty A] (hM : 1 ≤ M) {n : Fin K → ℕ}
    {t : ℕ} (hn : ∀ ht : t < K, 1 ≤ n ⟨t, ht⟩) : FnFork.δN A M n t ≤ (Fintype.card A : ℝ)⁻¹ := by
  unfold FnFork.δN
  split_ifs with ht
  · rw [FnFork.δAt_eq]
    have h1 : (Fintype.card A : ℝ)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ (by exact_mod_cast Fintype.card_pos)
    have h0 : (0 : ℝ) ≤ (Fintype.card A : ℝ)⁻¹ := by positivity
    calc (Fintype.card A : ℝ)⁻¹ ^ min (n ⟨t, ht⟩) M ≤ (Fintype.card A : ℝ)⁻¹ ^ 1 :=
          pow_le_pow_of_le_one h0 h1 (le_min (hn ht) hM)
      _ = (Fintype.card A : ℝ)⁻¹ := pow_one _
  · positivity

end Claims.Flock

namespace FlockDeploy.KS

open Flock

/-- **The plain fork rounds are before the last round**: each is a recorded round, and the last message follows
them. -/
theorem plainJ_succ_lt {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h0 : (tables false x).head? = some (spec, st)) {sch : FlockSoundness.Accounting.Schedule}
    (hsch : FlockSoundness.Accounting.fast100 st.m = some sch) (r : Fin 2) (ℓ : Fin sch.levels.length) :
    (plainJ h0 hsch r ℓ).val + 1 < (V false).k x := by
  obtain ⟨r, hr⟩ := r
  obtain ⟨s, sts, hs, hst⟩ := setupOf_head h0
  obtain ⟨sF, hF, rfl⟩ := exec_sched hsch
  have h2 := (FlockSoundness.Refine.fast100_dims _ _ hF).1.2.1
  have hℓ := ℓ.isLt
  simp only [FlockSoundness.Refine.schedOf, List.length_map, Array.length_toList] at hℓ
  have hR := repRounds_plain (spec := spec) hF
  have hle := rep_le_rounds (zk := false) (spec := spec) hst
  have hlt := capRound_lt (k0 := sF.k0) h2 (show ℓ.val + 1 ≤ sF.levels.size by omega)
  have hk : (V false).k x = 2 + rounds false spec sts := by simp only [V, hs]
  show forkRound spec st (FlockSoundness.Refine.schedOf sF).k0 r ℓ.val + 1 < (V false).k x
  simp only [FlockSoundness.Refine.schedOf]
  unfold forkRound
  rw [hk]
  interval_cases r <;> omega

/-- **The slack fits the plain weight at the coupling's chances**: at the deployed plain verifier, with the chances
`FnFork.δN GF128 65536 n` of the coupling at `fnAt` (`forkCoinCoupling_plain`) and a positive revealed length
`n t ≥ 1` at every round before the last, the slack's escapes' weights at the plain fork rounds total at most level
0's `2·2^{d₀}·Q₀`, as `slackEsc_fitsFork_plain` and `PlainSimWrPAtD` ask. -/
theorem plain_slack_fits {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h0 : (tables false x).head? = some (spec, st)) {m k0 : ℕ} {l₀ l₁ : FlockSoundness.Accounting.Level}
    {rest : List FlockSoundness.Accounting.Level} {ext : ℕ}
    (hsch : FlockSoundness.Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (n : Fin ((V false).k x) → ℕ) (hn : ∀ t : Fin ((V false).k x), t.val + 1 < (V false).k x → 1 ≤ n t) :
    ∑ i, Claims.Flock.slackC ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        (fun r ℓ (i : Fin ((V false).k x)) =>
          Claims.Flock.slackW ((V false).k x) (Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 n)
            (plainJ h0 hsch r ℓ).val i.val) i ≤
      2 * ((2 : ℝ) ^ l₀.logLen * l₀.queries) := by
  have hδ : ∀ t, t + 1 < (V false).k x →
      Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 n t ≤ ((2 : ℝ) ^ 128)⁻¹ := fun t ht => by
    have := Claims.Flock.δN_le_inv (A := FlockLevel3.GF128) (M := 65536) (by norm_num) (t := t)
      fun h => hn ⟨t, h⟩ ht
    rwa [FlockLevel3.card_F, Nat.cast_pow, Nat.cast_ofNat] at this
  exact Claims.Flock.slack_fits_fast100 hsch (plainJ h0 hsch) (plainJ_succ_lt h0 hsch)
    (Claims.Flock.FnFork.δN_nonneg n) hδ (Claims.Flock.δN_le_one n)

/-- **The fork clause at `fnAt`**: at a plain fork round, a bound at most the simulated fork of a simulator whose
coins are `Claims.Bridge.Live.fnAt` at a prefix key is at most the accepting fork with a collision there
(`plainForkEv`) and the slack at the coupling's chances `FnFork.δN GF128 65536 n` (`slack_of_coupling` at
`forkCoinCoupling_plain`), the fork bound `PlainSimWrPAtD` asks for. -/
theorem plain_fork_le_fnAtK (keep : Bool → Bool) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)])
    {m k0 : ℕ} {l₀ l₁ : FlockSoundness.Accounting.Level} {rest : List FlockSoundness.Accounting.Level} {ext : ℕ}
    (hsch : FlockSoundness.Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) {D : Type} [Fintype D]
    [DecidableEq D] {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) {L : ℕ} (hL : L = (V false).k x)
    (n : Fin ((V false).k x) → ℕ)
    {key : Fin ((V false).k x) → (Fin ((V false).k x) → Claims.Bridge.Live.LCoin 65536) → D}
    (hk : Claims.Flock.FnFork.PrefixKey n key) (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) {a : ℝ}
    (ha : a ≤ Claims.Flock.simForkPr sha512 (VK keep) P x
      (τ := Fin ((V false).k x) → D → Claims.Bridge.Live.LCoin 65536) (LC := Claims.Bridge.Live.LCoin 65536)
      (fun q => Claims.Bridge.Live.fnAt hL n key q.1 q.2) (fun t => Claims.Flock.FnFork.forkN t) ω
      (plainJ (x := x) (tables_head_one h) hsch r ℓ).val) :
    a ≤ (Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) =>
        plainForkEvK keep h hsch P r ℓ ω s.1 s.2) +
      Claims.Flock.forkSlack sha512 (VK keep) P x (Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 n)
        (plainJ (x := x) (tables_head_one h) hsch r ℓ).val ω :=
  ha.trans (Claims.Flock.slack_of_coupling sha512 (VK keep) P x (fun t => Claims.Flock.FnFork.δN_nonneg n t)
    (forkCoinCoupling_plainK keep P x ω hL n hk) (plainJ (x := x) (tables_head_one h) hsch r ℓ))

/-- `plain_fork_le_fnAtK` at the deployed plain verifier (`keep := id`). -/
theorem plain_fork_le_fnAt {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)])
    {m k0 : ℕ} {l₀ l₁ : FlockSoundness.Accounting.Level} {rest : List FlockSoundness.Accounting.Level} {ext : ℕ}
    (hsch : FlockSoundness.Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) {D : Type} [Fintype D]
    [DecidableEq D] {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) {L : ℕ} (hL : L = (V false).k x)
    (n : Fin ((V false).k x) → ℕ)
    {key : Fin ((V false).k x) → (Fin ((V false).k x) → Claims.Bridge.Live.LCoin 65536) → D}
    (hk : Claims.Flock.FnFork.PrefixKey n key) (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) {a : ℝ}
    (ha : a ≤ Claims.Flock.simForkPr sha512 (V false) P x
      (τ := Fin ((V false).k x) → D → Claims.Bridge.Live.LCoin 65536) (LC := Claims.Bridge.Live.LCoin 65536)
      (fun q => Claims.Bridge.Live.fnAt hL n key q.1 q.2) (fun t => Claims.Flock.FnFork.forkN t) ω
      (plainJ (tables_head_one h) hsch r ℓ).val) :
    a ≤ (Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) =>
        plainForkEv h hsch P r ℓ ω s.1 s.2) +
      Claims.Flock.forkSlack sha512 (V false) P x (Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 n)
        (plainJ (tables_head_one h) hsch r ℓ).val ω :=
  plain_fork_le_fnAtK id (x := x) (spec := spec) (st := st) (h := h) (m := m) (k0 := k0) (l₀ := l₀) (l₁ := l₁) (rest := rest) (ext := ext) (hsch := hsch) (D := D) (Ω := Ω) (P := P) (ω := ω) (L := L) (hL := hL) (n := n) (key := key) (hk := hk) (r := r) (ℓ := ℓ) (a := a) (ha := ha)

end FlockDeploy.KS
