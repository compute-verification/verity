import Proofs.Flock.Claims.DeployCode

/-!
# Flock's numbers at the deployed instances

`Numbers`, `NumbersTable` and `NumbersGamma` bound `κ`, `ε_table`, `γ` and `λ` over any `Deploy`'s fields; here they
are read at the deployed records `FlockDeploy.deployPlain` and `deployZk`, for a statement that sets up
(`setupOf zk x = some (s, spec, sts)`) with `n = sts.size ≥ 1` tables, each at an `m` its verifier serves
(`22 ≤ m ≤ 35`, or `25 ≤ m ≤ 35` with `--zk`), each of a planned table's shape (`Stmt.InRangeWide` / `InRange`:
`k_log ≤ 27`, at most `1026` / `1024` regions, `m_pts ≤ 64`), at two link points (`spec.points ≤ 2`, the setup's
default); `PlainIn` and `ZkIn` collect these, with `n ≤ 200`:

* each table's error is at most `2^{−205}` (`tableErr_plain_le`, `tableErr_zk_le`), and `ε_table`, their sum, at most
  `n·2^{−205}` (`εtable_le`);
* `m′ = 210` and `K = 2^256` (`waits_deploy`, `cap_deploy`); `N₀`, the tables' level-0 lengths summed, each `2^d`
  with `10 ≤ d ≤ 23`, is `2^10` to `n·2^23` (`N₀_le`), so `k = ⌈4N₀/e⌉ ≤ n·2^24` (`k_le_mul`); at one table
  `N₀ = 2^{d₀}` and `k = kVal d₀` exactly (`N₀_one`, `k_plain`, `k_zk`);
* the verifier's rounds are at most `2 + 4400·n` (`Vk_le`: each rep at most `2200`, `repRounds_plain_le`,
  `repRounds_zk_le`) and the weight at most `n·2^33` (`weight_le`);
* `κ ≤ n·2^{−202.98}` (`κ_le_mul` at `ε_table ≤ n·2^{−205}` and `N₀ ≤ n·2^23`) and `λ ≥ 201.98 − log₂ n`
  (`lam_ge_of_rounds_mul`, and at the instances `κ_plain_le`, `lam_plain_ge`, `κ_zk_le`, `lam_zk_ge`), so
  `λ ≥ 195.98` at up to `64` tables, the verifier's cap (`lam_plain_ge_64`, `lam_zk_ge_64`); at one table
  `ε_table ≤ 2^{−205}`, `κ ≤ 2^{−202.98}`, `γ ≤ 2^51` and `λ ≥ 201.98` (`plain_one`, `zk_one`).

`ε_table` and `N₀` here are the session's sums over its tables, so the spec's per-table `2^{−205}` costs `log₂ n` bits
of `λ` at `n` tables.
-/

namespace Claims.Flock.Numbers

section Deploy

variable {Coin Col T : Type}

/-- `k` at `0 ≤ N₀ ≤ n·2^23`, `n ≥ 1`, is at most `n·2^24`. -/
theorem k_le_mul {N n : ℝ} (hn : 1 ≤ n) (h0 : 0 ≤ N) (hN : N ≤ n * 2 ^ 23) :
    (⌈4 * N / Real.exp 1⌉₊ : ℝ) ≤ n * 2 ^ 24 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  refine (Nat.ceil_lt_add_one (by positivity)).le.trans ?_
  have h1 : 4 * N / Real.exp 1 ≤ n * 2 ^ 24 - 1 := by
    rw [div_le_iff₀ he0]
    have hm : 0 ≤ n * 2 ^ 24 - 1 := by nlinarith
    have := mul_le_mul_of_nonneg_left hlo.le hm
    nlinarith
  linarith

/-- `2^{−203} + 2^{−210} + 2^{−231} + 2^{−250} ≤ 2^{−202.98}`. -/
theorem terms_le : (2 : ℝ) ^ (-(203 : ℝ)) + 2 ^ (-(210 : ℝ)) + 2 ^ (-(231 : ℝ)) + 2 ^ (-(250 : ℝ)) ≤
    2 ^ (-(202.98 : ℝ)) := by
  have e : ∀ n : ℕ, (2 : ℝ) ^ (-((203 + n : ℕ) : ℝ)) = 2 ^ (-(203 : ℝ)) * 2⁻¹ ^ n := by
    intro n
    rw [Nat.cast_add, neg_add, Real.rpow_add (by norm_num), Real.rpow_neg (by norm_num) (n : ℝ),
      Real.rpow_natCast, inv_pow]
    norm_num
  have e7 : (2 : ℝ) ^ (-(210 : ℝ)) = 2 ^ (-(203 : ℝ)) * 2⁻¹ ^ 7 := by rw [← e 7]; norm_num
  have e28 : (2 : ℝ) ^ (-(231 : ℝ)) = 2 ^ (-(203 : ℝ)) * 2⁻¹ ^ 28 := by rw [← e 28]; norm_num
  have e47 : (2 : ℝ) ^ (-(250 : ℝ)) = 2 ^ (-(203 : ℝ)) * 2⁻¹ ^ 47 := by rw [← e 47]; norm_num
  calc (2 : ℝ) ^ (-(203 : ℝ)) + 2 ^ (-(210 : ℝ)) + 2 ^ (-(231 : ℝ)) + 2 ^ (-(250 : ℝ))
      = 2 ^ (-(203 : ℝ)) * (1 + 2⁻¹ ^ 7 + 2⁻¹ ^ 28 + 2⁻¹ ^ 47) := by rw [e7, e28, e47]; ring
    _ ≤ 2 ^ (-(203 : ℝ)) * 2 ^ ((1 : ℝ) / 50) := mul_le_mul_of_nonneg_left slack_le (by positivity)
    _ = 2 ^ (-(202.98 : ℝ)) := by
        rw [← Real.rpow_add (by norm_num)]; norm_num

/-- **`κ ≤ n·2^{−202.98}`** when `ε_table ≤ n·2^{−205}` and `N₀ ≤ n·2^23`, `n ≥ 1`, at the deployed `m′` and `K`:
each of `κ`'s four terms is at most `n` times its one-table bound (`k ≤ n·2^24`, `k_le_mul`). -/
theorem κ_le_mul (D : Deploy Coin Col T) (x : Msg) {n : ℝ} (hn : 1 ≤ n) (hε : D.εtable x ≤ n * 2 ^ (-(205 : ℝ)))
    (hw : 210 ≤ D.waits x) (hK : D.cap x = cap) (hN : 2 ^ 10 ≤ D.N₀ x) (hN' : D.N₀ x ≤ n * 2 ^ 23) :
    κ D x ≤ n * 2 ^ (-(202.98 : ℝ)) := by
  have hn0 : (0 : ℝ) ≤ n := by linarith
  have hk1 : 1507 ≤ D.k x := k_ge hN
  have hk2 : (D.k x : ℝ) ≤ n * 2 ^ 24 := k_le_mul hn (le_trans (by norm_num) hN) hN'
  have h1 : 4 * D.εtable x ≤ n * 2 ^ (-(203 : ℝ)) := by
    have e : (2 : ℝ) ^ (-(203 : ℝ)) = 4 * 2 ^ (-(205 : ℝ)) := by
      rw [show (-(203 : ℝ)) = 2 + -(205 : ℝ) by norm_num, Real.rpow_add (by norm_num)]; norm_num
    calc 4 * D.εtable x ≤ 4 * (n * 2 ^ (-(205 : ℝ))) := by linarith
      _ = n * 2 ^ (-(203 : ℝ)) := by rw [e]; ring
  have h2 : (2 : ℝ) ^ (-(D.waits x : ℝ)) ≤ n * 2 ^ (-(210 : ℝ)) :=
    (Real.rpow_le_rpow_of_exponent_le (by norm_num) (neg_le_neg (by exact_mod_cast hw))).trans
      (le_mul_of_one_le_left (by positivity) hn)
  have h3 : 2 * (D.k x : ℝ) / D.cap x ≤ n * 2 ^ (-(231 : ℝ)) := by
    have hc : (D.cap x : ℝ) = 2 ^ 256 := by rw [hK, cap]; push_cast; ring
    have e : (2 : ℝ) ^ (-(231 : ℝ)) = (2 ^ 231)⁻¹ := by
      rw [Real.rpow_neg (by norm_num), show (231 : ℝ) = ((231 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    rw [hc, e, div_le_iff₀ (by positivity)]
    calc 2 * (D.k x : ℝ) ≤ 2 * (n * 2 ^ 24) := by linarith
      _ = n * (2 ^ 231)⁻¹ * 2 ^ 256 := by ring
  have h4 : Real.exp (-(D.k x : ℝ) / 8) ≤ n * 2 ^ (-(250 : ℝ)) :=
    (exp_neg_k_le hk1).trans (le_mul_of_one_le_left (by positivity) hn)
  unfold κ
  calc 4 * D.εtable x + 2 ^ (-(D.waits x : ℝ)) + 2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8)
      ≤ n * 2 ^ (-(203 : ℝ)) + n * 2 ^ (-(210 : ℝ)) + n * 2 ^ (-(231 : ℝ)) + n * 2 ^ (-(250 : ℝ)) :=
        add_le_add (add_le_add (add_le_add h1 h2) h3) h4
    _ = n * (2 ^ (-(203 : ℝ)) + 2 ^ (-(210 : ℝ)) + 2 ^ (-(231 : ℝ)) + 2 ^ (-(250 : ℝ))) := by ring
    _ ≤ n * 2 ^ (-(202.98 : ℝ)) := mul_le_mul_of_nonneg_left terms_le hn0

/-- **`λ ≥ 201.98 − log₂ n`** when `κ ≤ n·2^{−202.98}`, `n ≥ 1`, and `γ ≤ 2^105`. -/
theorem lam_ge_mul {κ γ n : ℝ} (hn : 1 ≤ n) (hκ0 : 0 < κ) (hκ : κ ≤ n * 2 ^ (-(202.98 : ℝ))) (hγ0 : 0 ≤ γ)
    (hγ : γ ≤ 2 ^ (105 : ℝ)) : 201.98 - Real.logb 2 n ≤ lamOf κ γ := by
  have hn0 : 0 < n := by linarith
  have hl : 0 ≤ Real.logb 2 n := Real.logb_nonneg (by norm_num) hn
  have hκ' : κ ≤ 2 ^ (-(202.98 - Real.logb 2 n)) := by
    rw [neg_sub, sub_eq_add_neg, Real.rpow_add (by norm_num), Real.rpow_logb (by norm_num) (by norm_num) hn0]
    exact hκ
  rcases hγ0.lt_or_eq with hγ0 | hγ0
  · have h := lamOf_ge hκ0 hκ' hγ0 hγ
    rw [min_eq_left (by linarith)] at h
    linarith
  · unfold lamOf
    rw [← hγ0, Real.logb_zero, zero_div, sub_zero]
    have h1 : 202.98 - Real.logb 2 n ≤ Real.logb 2 (1 / κ) := by
      rw [one_div, Real.logb_inv, le_neg, ← Real.rpow_le_rpow_left_iff (by norm_num : (1 : ℝ) < 2),
        Real.rpow_logb (by norm_num) (by norm_num) hκ0]
      exact hκ'
    have h2 : 202.98 - Real.logb 2 n ≤ min (Real.logb 2 (1 / κ)) 255.5 := le_min h1 (by linarith)
    linarith

/-- **`λ ≥ 201.98 − log₂ n`** at the deployed numbers over `n ≥ 1` tables, up to `2^20` rounds and a weight up to
`2^64`: `ε_table ≤ n·2^{−205}` and `N₀ ≤ n·2^23`, the tables' errors and level-0 lengths summed. At `n = 1` it is
`lam_ge_of_rounds`. -/
theorem lam_ge_of_rounds_mul (D : Deploy Coin Col T) (x : Msg) {n : ℝ} (hn : 1 ≤ n) (hε0 : 0 ≤ D.εtable x)
    (hε : D.εtable x ≤ n * 2 ^ (-(205 : ℝ))) (hw : 210 ≤ D.waits x) (hK : D.cap x = cap) (hN : 2 ^ 10 ≤ D.N₀ x)
    (hN' : D.N₀ x ≤ n * 2 ^ 23) (hr : D.V.k x ≤ 2 ^ 20) (hC0 : 0 ≤ D.weight x) (hC : D.weight x ≤ 2 ^ 64) :
    201.98 - Real.logb 2 n ≤ lamOf (κ D x) (γ D x) :=
  lam_ge_mul hn (κ_pos D x hε0) (κ_le_mul D x hn hε hw hK hN hN') (γ_nonneg D x) (γ_le_105 D x hr hC0 hC)

/-- **`γ ≤ 2^51`** at up to `4402` rounds and a weight up to `2^33`. -/
theorem γ_le_51 (D : Deploy Coin Col T) (x : Msg) (hr : D.V.k x ≤ 4402) (hC0 : 0 ≤ D.weight x)
    (hC : D.weight x ≤ 2 ^ 33) : γ D x ≤ 2 ^ (51 : ℝ) := by
  refine (γ_le_of D x (r := 4402) (by exact_mod_cast hr) hC0 hC).trans ?_
  have hs : Real.sqrt (2 * 4402 * 2 ^ 33) ≤ 8700000 := by
    rw [Real.sqrt_le_left (by norm_num)]
    norm_num
  rw [show (51 : ℝ) = ((51 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have h0 : 0 ≤ Real.sqrt (2 * 4402 * 2 ^ 33) := Real.sqrt_nonneg _
  nlinarith

end Deploy

end Claims.Flock.Numbers

namespace FlockDeploy

open Flock
open FlockSoundness
open Claims.Flock (PlainIdx ZkIdx plainSched plainLevel0 plainLen zkSched zkLevel0 zkLen)

/-! ## Each table's error -/

/-- **A plain table's error is at most `2^{−205}`** at every `m` the plain verifier serves and every planned shape. -/
theorem tableErr_plain_le (spec : SessionSpec) (st : Stmt) (h1 : 22 ≤ st.m) (h2 : st.m ≤ 35)
    (hst : (acct spec st).InRangeWide) : tableErr false spec st ≤ 2 ^ (-(205 : ℝ)) := by
  obtain ⟨s, hs⟩ : ∃ s, Accounting.fast100 st.m = some s := by
    generalize st.m = m at h1 h2
    interval_cases m <;> exact ⟨_, rfl⟩
  simp only [tableErr, Bool.false_eq_true, ite_false, hs, Option.getD_some]
  exact Claims.Flock.Numbers.εtable_plain hs _ hst

/-- **A masked table's error is at most `2^{−205}`** at every `m` the masked verifier serves and every shape in
`InRange`: `padLinkError + (padRepError + ε_inner)²` at `m1Schedule`, with `ε_inner ≤ innerUp`. -/
theorem tableErr_zk_le (spec : SessionSpec) (st : Stmt) (h1 : 25 ≤ st.m) (h2 : st.m ≤ 35)
    (hst : (acct spec st).InRange) : tableErr true spec st ≤ 2 ^ (-(205 : ℝ)) := by
  have hf := Accounting.m1Facts_spec (Accounting.m1Facts_all st.m h1 h2)
  have hl := Accounting.padLinkError_le (Accounting.m1Schedule st.m) (acct spec st) (Accounting.m1Pad st.m) hst
  have hr0 := Accounting.m1_rep_nonneg st.m h1 h2 (acct spec st)
  have hr := Claims.Flock.Numbers.padRepError_le_of (Accounting.m1Schedule st.m) (acct spec st) hst
    (Accounting.m1Pad st.m) 2 (Accounting.m1SqrtUp st.m) hf.1 hf.2.2.1 hf.2.2.2.2
  have hc := Rat.cast_le (K := ℝ) |>.2 (Claims.Flock.Numbers.m1_check_zk st.m h1 h2)
  push_cast at hc
  have hi := Claims.Flock.Numbers.innerUp_nonneg
  have hsq : (Accounting.padRepError (Accounting.m1Schedule st.m) (acct spec st) (Accounting.m1Pad st.m) 2 +
      (Claims.Flock.Numbers.innerUp : ℝ)) ^ 2 ≤
      ((Accounting.padRepUp (Accounting.m1Schedule st.m) (Accounting.m1Pad st.m) 2 (Accounting.m1SqrtUp st.m) : ℝ) +
        Claims.Flock.Numbers.innerUp) ^ 2 := by
    gcongr
  simp only [tableErr, ite_true]
  exact ((add_le_add hl hsq).trans hc).trans Accounting.half_pow_205_le

theorem tableErr_nonneg (zk : Bool) (spec : SessionSpec) (st : Stmt) : 0 ≤ tableErr zk spec st := by
  unfold tableErr
  split
  · have : 0 ≤ Accounting.padLinkError (Accounting.m1Schedule st.m) (acct spec st) (Accounting.m1Pad st.m) := by
      unfold Accounting.padLinkError; positivity
    positivity
  · unfold Accounting.tableError Accounting.linkError; positivity

/-! ## The session's numbers -/

theorem tables_eq {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupOf zk x = some (s, spec, sts)) : tables zk x = sts.toList.map fun st => (spec, st) := by
  simp only [tables, hs]

theorem εtable_nonneg (zk : Bool) (x : Claims.Msg) : 0 ≤ εtable zk x := by
  unfold εtable
  exact List.sum_nonneg fun a ha => by
    obtain ⟨p, -, rfl⟩ := List.mem_map.1 ha
    exact tableErr_nonneg zk p.1 p.2

/-- **`ε_table ≤ n·2^{−205}`**: the session's `n` tables' errors, summed, each at most `2^{−205}`. -/
theorem εtable_le {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupOf zk x = some (s, spec, sts)) (h : ∀ st ∈ sts.toList, tableErr zk spec st ≤ 2 ^ (-(205 : ℝ))) :
    εtable zk x ≤ sts.size * 2 ^ (-(205 : ℝ)) := by
  unfold εtable
  rw [tables_eq hs, List.map_map]
  refine (List.sum_le_length_nsmul _ ((2 : ℝ) ^ (-(205 : ℝ))) fun a ha => ?_).trans (le_of_eq (by simp))
  obtain ⟨st, hst, rfl⟩ := List.mem_map.1 ha
  exact h st hst

theorem waits_deploy (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    (deployPlain rel).waits x = 210 ∧ (deployZk rel).waits x = 210 := ⟨rfl, rfl⟩

theorem cap_deploy (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    (deployPlain rel).cap x = Claims.Flock.Numbers.cap ∧ (deployZk rel).cap x = Claims.Flock.Numbers.cap := ⟨rfl, rfl⟩

/-- The plain level 0 at each `m` served: `fast100`'s schedule has at most `6` lanes (log) and `7` levels (`7` at
`m = 34, 35`), and `10 ≤ d₀ ≤ 23`, `d₀` one of `fast100`'s level-0 lengths. -/
theorem plain_facts : ∀ i : PlainIdx, (plainSched i.m).k0 ≤ 6 ∧ (plainSched i.m).levels.size ≤ 7 ∧
    plainLen i ∈ [10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23] := by
  decide +kernel

/-- The masked level 0 at each `m` served, likewise. -/
theorem zk_facts : ∀ i : ZkIdx, (zkSched i.m).1.k0 ≤ 6 ∧ (zkSched i.m).1.levels.size ≤ 7 ∧
    zkLen i ∈ [10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23] := by
  decide +kernel

theorem plainIdx_of {m : ℕ} (h1 : 22 ≤ m) (h2 : m ≤ 35) : ∃ i : PlainIdx, i.m = m :=
  ⟨⟨m - 22, by omega⟩, by simp only [Claims.Flock.PlainIdx.m]; omega⟩

theorem zkIdx_of {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) : ∃ i : ZkIdx, i.m = m :=
  ⟨⟨m - 25, by omega⟩, by simp only [Claims.Flock.ZkIdx.m]; omega⟩

theorem d_mem_bounds {d : ℕ} (h : d ∈ [10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23]) : 10 ≤ d ∧ d ≤ 23 := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at h
  omega

/-- A plain table's level-0 length is `2^10` to `2^23` at every `m` the plain verifier serves. -/
theorem tableLen_plain_bounds (st : Stmt) (h1 : 22 ≤ st.m) (h2 : st.m ≤ 35) :
    2 ^ 10 ≤ tableLen false st ∧ tableLen false st ≤ 2 ^ 23 := by
  obtain ⟨i, hi⟩ := plainIdx_of h1 h2
  have hd := (plain_facts i).2.2
  rw [Claims.Flock.plainLen, hi] at hd
  obtain ⟨hlo, hhi⟩ := d_mem_bounds hd
  simp only [tableLen, Bool.false_eq_true, ite_false]
  exact ⟨pow_le_pow_right₀ (by norm_num) hlo, pow_le_pow_right₀ (by norm_num) hhi⟩

/-- A masked table's level-0 length is `2^10` to `2^23` at every `m` the masked verifier serves. -/
theorem tableLen_zk_bounds (st : Stmt) (h1 : 25 ≤ st.m) (h2 : st.m ≤ 35) :
    2 ^ 10 ≤ tableLen true st ∧ tableLen true st ≤ 2 ^ 23 := by
  obtain ⟨i, hi⟩ := zkIdx_of h1 h2
  have hd := (zk_facts i).2.2
  rw [Claims.Flock.zkLen, hi] at hd
  obtain ⟨hlo, hhi⟩ := d_mem_bounds hd
  simp only [tableLen, ite_true]
  exact ⟨pow_le_pow_right₀ (by norm_num) hlo, pow_le_pow_right₀ (by norm_num) hhi⟩

/-- **`2^10 ≤ N₀ ≤ n·2^23`**: `N₀` sums the session's `n ≥ 1` tables' level-0 lengths, each `2^10` to `2^23`. -/
theorem N₀_le {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupOf zk x = some (s, spec, sts)) (h0 : 0 < sts.size)
    (h : ∀ st ∈ sts.toList, 2 ^ 10 ≤ tableLen zk st ∧ tableLen zk st ≤ 2 ^ 23) :
    2 ^ 10 ≤ N₀ zk x ∧ N₀ zk x ≤ sts.size * 2 ^ 23 := by
  unfold N₀
  rw [tables_eq hs, List.map_map]
  constructor
  · have hm : tableLen zk sts[0] ∈ sts.toList.map ((fun p => tableLen zk p.2) ∘ fun st => (spec, st)) :=
      List.mem_map.2 ⟨sts[0], Array.getElem_mem_toList h0, rfl⟩
    refine (h _ (Array.getElem_mem_toList h0)).1.trans (List.single_le_sum (fun a ha => ?_) _ hm)
    obtain ⟨st, hst, rfl⟩ := List.mem_map.1 ha
    exact le_trans (by norm_num) (h st hst).1
  · refine (List.sum_le_length_nsmul _ ((2 : ℝ) ^ 23) fun a ha => ?_).trans (le_of_eq (by simp))
    obtain ⟨st, hst, rfl⟩ := List.mem_map.1 ha
    exact (h st hst).2

/-- **`N₀` at one table** is that table's level-0 length, `2^{d₀}`. -/
theorem N₀_one {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupOf zk x = some (s, spec, sts)) (h1 : sts.size = 1) :
    N₀ zk x = (2 : ℝ) ^ (if zk then zkLevel0 (sts[0]'(by omega)).m else plainLevel0 (sts[0]'(by omega)).m).d := by
  have hl : sts.toList = [sts[0]'(by omega)] := by
    apply List.ext_getElem (by simp [h1])
    intro i hi _
    have : i = 0 := by simp only [Array.length_toList] at hi; omega
    subst this
    simp
  simp only [N₀, tables_eq hs, hl, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, tableLen]

/-- **`N₀` and `k` without `--zk`, at one table**: `N₀ = 2^{d₀}`, the table's plain level-0 length,
`2^10 ≤ N₀ ≤ 2^23`, and `k = ⌈4N₀/e⌉ = kVal d₀`. -/
theorem k_plain (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hs : setupOf false x = some (s, spec, sts)) (h0 : sts.size = 1)
    (h1 : 22 ≤ (sts[0]'(by omega)).m) (h2 : (sts[0]'(by omega)).m ≤ 35) :
    (deployPlain rel).N₀ x = 2 ^ (plainLevel0 (sts[0]'(by omega)).m).d ∧ 2 ^ 10 ≤ (deployPlain rel).N₀ x ∧
      (deployPlain rel).N₀ x ≤ 2 ^ 23 ∧
      (deployPlain rel).k x = Claims.Flock.Numbers.kVal (plainLevel0 (sts[0]'(by omega)).m).d := by
  obtain ⟨i, hi⟩ := plainIdx_of h1 h2
  have hd := (plain_facts i).2.2
  rw [Claims.Flock.plainLen, hi] at hd
  have hN : (deployPlain rel).N₀ x = 2 ^ (plainLevel0 (sts[0]'(by omega)).m).d := N₀_one hs h0
  obtain ⟨hlo, hhi⟩ := d_mem_bounds hd
  refine ⟨hN, ?_, ?_, ?_⟩
  · rw [hN]; exact pow_le_pow_right₀ (by norm_num) hlo
  · rw [hN]; exact pow_le_pow_right₀ (by norm_num) hhi
  · unfold Claims.Flock.Deploy.k
    rw [hN]
    exact Claims.Flock.Numbers.kRep_eq hd

/-- **`N₀` and `k` with `--zk`, at one table**: `N₀ = 2^{d₀}`, the table's masked level-0 length,
`2^10 ≤ N₀ ≤ 2^23`, and `k = ⌈4N₀/e⌉ = kVal d₀`. -/
theorem k_zk (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hs : setupOf true x = some (s, spec, sts)) (h0 : sts.size = 1)
    (h1 : 25 ≤ (sts[0]'(by omega)).m) (h2 : (sts[0]'(by omega)).m ≤ 35) :
    (deployZk rel).N₀ x = 2 ^ (zkLevel0 (sts[0]'(by omega)).m).d ∧ 2 ^ 10 ≤ (deployZk rel).N₀ x ∧
      (deployZk rel).N₀ x ≤ 2 ^ 23 ∧
      (deployZk rel).k x = Claims.Flock.Numbers.kVal (zkLevel0 (sts[0]'(by omega)).m).d := by
  obtain ⟨i, hi⟩ := zkIdx_of h1 h2
  have hd := (zk_facts i).2.2
  rw [Claims.Flock.zkLen, hi] at hd
  have hN : (deployZk rel).N₀ x = 2 ^ (zkLevel0 (sts[0]'(by omega)).m).d := N₀_one hs h0
  obtain ⟨hlo, hhi⟩ := d_mem_bounds hd
  refine ⟨hN, ?_, ?_, ?_⟩
  · rw [hN]; exact pow_le_pow_right₀ (by norm_num) hlo
  · rw [hN]; exact pow_le_pow_right₀ (by norm_num) hhi
  · unfold Claims.Flock.Deploy.k
    rw [hN]
    exact Claims.Flock.Numbers.kRep_eq hd

/-! ## The verifier's rounds and the weight -/

theorem foldl_add_eq {α : Type} (f : α → ℕ) :
    ∀ (l : List α) (a : ℕ), l.foldl (fun acc b => acc + f b) a = a + (l.map f).sum
  | [], a => by simp
  | b :: l, a => by
    simp only [List.foldl_cons, foldl_add_eq f l, List.map_cons, List.sum_cons]
    omega

/-- **A plain rep records at most `2200` rounds** at a served `m`, `k_log ≤ 27`, at most `1026` regions and two link
points: `(3 + (m − 6)) + (3 + (k_log − 6)) + (2 + 2·regions + 1) + (k0 + 5 + 11·(levels − 1)) ≤ 2188`. -/
theorem repRounds_plain_le (spec : SessionSpec) (st : Stmt) (h1 : 22 ≤ st.m) (h2 : st.m ≤ 35)
    (hk : st.c.kLog ≤ 27) (hR : st.regions.size ≤ 1026) (hp : spec.points ≤ 2) : repRounds false spec st ≤ 2200 := by
  have hc : spec.points * st.regions.size ≤ 2052 := (Nat.mul_le_mul hp hR).trans (by norm_num)
  obtain ⟨i, hi⟩ := plainIdx_of h1 h2
  have hf := plain_facts i
  rw [hi] at hf
  unfold repRounds
  simp only [Bool.false_eq_true, ite_false]
  cases h : fast100 st.m with
  | none => simp
  | some sch =>
    have : plainSched st.m = sch := by simp [plainSched, h]
    rw [this] at hf
    dsimp only
    omega

/-- **A masked rep records at most `2200` rounds** at a served `m`, `k_log ≤ 27`, at most `1026` regions and two link
points: `(m + (k_log − 6)) + (2 + 2·regions + 1) + (k0 + 7 + 11·(levels − 1)) + 4 ≤ 2194`. -/
theorem repRounds_zk_le (spec : SessionSpec) (st : Stmt) (h1 : 25 ≤ st.m) (h2 : st.m ≤ 35)
    (hk : st.c.kLog ≤ 27) (hR : st.regions.size ≤ 1026) (hp : spec.points ≤ 2) : repRounds true spec st ≤ 2200 := by
  have hc : spec.points * st.regions.size ≤ 2052 := (Nat.mul_le_mul hp hR).trans (by norm_num)
  obtain ⟨i, hi⟩ := zkIdx_of h1 h2
  have hf := zk_facts i
  rw [hi] at hf
  unfold repRounds
  simp only [ite_true]
  cases h : Zk.schedule st.m with
  | error e => simp
  | ok p =>
    obtain ⟨sch, t⟩ := p
    have : (zkSched st.m).1 = sch := by simp [zkSched, h]
    rw [this] at hf
    dsimp only
    omega

theorem rounds_le {zk : Bool} {spec : SessionSpec} {sts : Array Stmt} {b : ℕ}
    (h : ∀ st ∈ sts.toList, repRounds zk spec st ≤ b) : rounds zk spec sts ≤ 2 * b * sts.size := by
  unfold rounds
  rw [← Array.foldl_toList, foldl_add_eq (fun st => 2 * repRounds zk spec st), zero_add]
  refine (List.sum_le_length_nsmul _ (2 * b) fun a ha => ?_).trans (le_of_eq (by simp [Nat.mul_comm]))
  obtain ⟨st, hst, rfl⟩ := List.mem_map.1 ha
  exact Nat.mul_le_mul_left 2 (h st hst)

/-- **The verifier's rounds**: at most `2 + 4400·n` at `n` tables of at most `2200` rounds per rep. -/
theorem Vk_le {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupOf zk x = some (s, spec, sts)) (h : ∀ st ∈ sts.toList, repRounds zk spec st ≤ 2200) :
    (V zk).k x ≤ 2 + 4400 * sts.size := by
  have := rounds_le h
  simp only [V, hs]
  omega

theorem fast100_some {m : ℕ} (h1 : 22 ≤ m) (h2 : m ≤ 35) : ∃ s, Accounting.fast100 m = some s := by
  interval_cases m <;> exact ⟨_, rfl⟩

theorem tableWeight_plain_le (st : Stmt) (h1 : 22 ≤ st.m) (h2 : st.m ≤ 35) : tableWeight false st ≤ 2 ^ 33 := by
  obtain ⟨s, hs⟩ := fast100_some h1 h2
  simp only [tableWeight, Bool.false_eq_true, ite_false, hs, Option.getD_some]
  exact Claims.Flock.Numbers.weightOf_le_fast100 hs (Nat.zero_le _)

theorem tableWeight_zk_le (st : Stmt) (h1 : 25 ≤ st.m) (h2 : st.m ≤ 35) (hk : st.c.kLog ≤ 27) :
    tableWeight true st ≤ 2 ^ 33 := by
  obtain ⟨i, hi⟩ := zkIdx_of h1 h2
  have hf := zk_facts i
  rw [hi] at hf
  simp only [tableWeight, ite_true]
  exact Claims.Flock.Numbers.weightOf_le_m1_pads h1 h2 hk hf.1

theorem weight_nonneg (zk : Bool) (x : Claims.Msg) : 0 ≤ weight zk x := by
  unfold weight
  exact List.sum_nonneg fun a ha => by
    obtain ⟨p, -, rfl⟩ := List.mem_map.1 ha
    unfold tableWeight
    split <;> exact Claims.Flock.Numbers.weightOf_nonneg _ _

/-- **The weight `C`**: at most `n·2^33` at `n` tables of weight at most `2^33`. -/
theorem weight_le {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupOf zk x = some (s, spec, sts)) (h : ∀ st ∈ sts.toList, tableWeight zk st ≤ 2 ^ 33) :
    weight zk x ≤ sts.size * 2 ^ 33 := by
  unfold weight
  rw [tables_eq hs, List.map_map]
  refine (List.sum_le_length_nsmul _ ((2 : ℝ) ^ 33) fun a ha => ?_).trans (le_of_eq (by simp))
  obtain ⟨st, hst, rfl⟩ := List.mem_map.1 ha
  exact h st hst

/-! ## `κ`, `γ` and `λ` at the instances -/

/-- **A plain session in scope**: it sets up with `1 ≤ n ≤ 200` tables at two link points, each at an `m` the plain
verifier serves and of a planned table's shape. -/
structure PlainIn (x : Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt) : Prop where
  setup : setupOf false x = some (s, spec, sts)
  size_pos : 0 < sts.size
  size_le : sts.size ≤ 200
  points : spec.points ≤ 2
  tab : ∀ st ∈ sts.toList, 22 ≤ st.m ∧ st.m ≤ 35 ∧ (acct spec st).InRangeWide

/-- **A masked session in scope**: likewise, at an `m` the masked verifier serves and a shape in `InRange`. -/
structure ZkIn (x : Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt) : Prop where
  setup : setupOf true x = some (s, spec, sts)
  size_pos : 0 < sts.size
  size_le : sts.size ≤ 200
  points : spec.points ≤ 2
  tab : ∀ st ∈ sts.toList, 25 ≤ st.m ∧ st.m ≤ 35 ∧ (acct spec st).InRange

theorem size_one_le {sts : Array Stmt} (h : 0 < sts.size) : (1 : ℝ) ≤ sts.size := by
  exact_mod_cast h

/-- **`κ ≤ n·2^{−202.98}`** at the deployed plain verifier, `n` its tables. -/
theorem κ_plain_le (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : PlainIn x s spec sts) :
    Claims.Flock.κ (deployPlain rel) x ≤ sts.size * 2 ^ (-(202.98 : ℝ)) := by
  obtain ⟨hlo, hhi⟩ := N₀_le h.setup h.size_pos fun st hst =>
    have := h.tab st hst
    tableLen_plain_bounds st this.1 this.2.1
  refine Claims.Flock.Numbers.κ_le_mul _ x (size_one_le h.size_pos) ?_ (le_of_eq rfl) rfl hlo hhi
  exact εtable_le h.setup fun st hst =>
    have := h.tab st hst
    tableErr_plain_le spec st this.1 this.2.1 this.2.2

/-- **`κ ≤ n·2^{−202.98}`** at the deployed masked verifier, `n` its tables. -/
theorem κ_zk_le (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : ZkIn x s spec sts) :
    Claims.Flock.κ (deployZk rel) x ≤ sts.size * 2 ^ (-(202.98 : ℝ)) := by
  obtain ⟨hlo, hhi⟩ := N₀_le h.setup h.size_pos fun st hst =>
    have := h.tab st hst
    tableLen_zk_bounds st this.1 this.2.1
  refine Claims.Flock.Numbers.κ_le_mul _ x (size_one_le h.size_pos) ?_ (le_of_eq rfl) rfl hlo hhi
  exact εtable_le h.setup fun st hst =>
    have := h.tab st hst
    tableErr_zk_le spec st this.1 this.2.1 this.2.2

theorem Vk_plain_le (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : PlainIn x s spec sts) : (deployPlain rel).V.k x ≤ 2 + 4400 * sts.size :=
  Vk_le h.setup fun st hst =>
    have := h.tab st hst
    repRounds_plain_le spec st this.1 this.2.1 this.2.2.1 this.2.2.2.1 h.points

theorem Vk_zk_le (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : ZkIn x s spec sts) : (deployZk rel).V.k x ≤ 2 + 4400 * sts.size :=
  Vk_le h.setup fun st hst =>
    have := h.tab st hst
    repRounds_zk_le spec st this.1 this.2.1 this.2.2.1 (this.2.2.2.1.trans (by norm_num)) h.points

theorem weight_plain_le (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : PlainIn x s spec sts) : (deployPlain rel).weight x ≤ sts.size * 2 ^ 33 :=
  weight_le h.setup fun st hst =>
    have := h.tab st hst
    tableWeight_plain_le st this.1 this.2.1

theorem weight_zk_le (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : ZkIn x s spec sts) : (deployZk rel).weight x ≤ sts.size * 2 ^ 33 :=
  weight_le h.setup fun st hst =>
    have := h.tab st hst
    tableWeight_zk_le st this.1 this.2.1 this.2.2.1

/-- `γ ≤ 2^105` at up to `200` tables: at most `2 + 4400·200` rounds and a weight at most `200·2^33`. -/
theorem γ_le_105_of {C Col T : Type} (D : Claims.Flock.Deploy C Col T) (x : Claims.Msg) {n : ℕ} (hn : n ≤ 200)
    (hr : D.V.k x ≤ 2 + 4400 * n) (hC0 : 0 ≤ D.weight x) (hC : D.weight x ≤ n * 2 ^ 33) :
    Claims.Flock.γ D x ≤ 2 ^ (105 : ℝ) := by
  refine Claims.Flock.Numbers.γ_le_105 D x (by omega) hC0 (hC.trans ?_)
  have : (n : ℝ) ≤ 200 := by exact_mod_cast hn
  nlinarith

/-- **`λ ≥ 201.98 − log₂ n`** at the deployed plain verifier, `n ≤ 200` its tables. -/
theorem lam_plain_ge (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : PlainIn x s spec sts) :
    201.98 - Real.logb 2 sts.size ≤ Claims.lamOf (Claims.Flock.κ (deployPlain rel) x) (Claims.Flock.γ (deployPlain rel) x) :=
  Claims.Flock.Numbers.lam_ge_mul (size_one_le h.size_pos) (Claims.Flock.κ_pos _ x (εtable_nonneg false x))
    (κ_plain_le rel h) (Claims.Flock.γ_nonneg _ x)
    (γ_le_105_of _ x h.size_le (Vk_plain_le rel h) (weight_nonneg false x) (weight_plain_le rel h))

/-- **`λ ≥ 201.98 − log₂ n`** at the deployed masked verifier, `n ≤ 200` its tables. -/
theorem lam_zk_ge (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : ZkIn x s spec sts) :
    201.98 - Real.logb 2 sts.size ≤ Claims.lamOf (Claims.Flock.κ (deployZk rel) x) (Claims.Flock.γ (deployZk rel) x) :=
  Claims.Flock.Numbers.lam_ge_mul (size_one_le h.size_pos) (Claims.Flock.κ_pos _ x (εtable_nonneg true x))
    (κ_zk_le rel h) (Claims.Flock.γ_nonneg _ x)
    (γ_le_105_of _ x h.size_le (Vk_zk_le rel h) (weight_nonneg true x) (weight_zk_le rel h))

theorem logb_size_le_6 {sts : Array Stmt} (h0 : 0 < sts.size) (h64 : sts.size ≤ 64) :
    Real.logb 2 sts.size ≤ 6 := by
  have h64' : (sts.size : ℝ) ≤ (2 : ℝ) ^ (6 : ℕ) := by norm_num; exact_mod_cast h64
  calc Real.logb 2 sts.size ≤ Real.logb 2 ((2 : ℝ) ^ (6 : ℕ)) :=
        Real.logb_le_logb_of_le (by norm_num) (by exact_mod_cast h0) h64'
    _ = 6 := by rw [Real.logb_pow, Real.logb_self_eq_one (by norm_num)]; norm_num

/-- **`λ ≥ 195.98`** at the deployed plain verifier at up to `64` tables, the verifier's cap (`Tags.MAX_TABLES`). -/
theorem lam_plain_ge_64 (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : PlainIn x s spec sts) (h64 : sts.size ≤ 64) :
    195.98 ≤ Claims.lamOf (Claims.Flock.κ (deployPlain rel) x) (Claims.Flock.γ (deployPlain rel) x) := by
  have := lam_plain_ge rel h
  have := logb_size_le_6 h.size_pos h64
  linarith

/-- **`λ ≥ 195.98`** at the deployed masked verifier at up to `64` tables, the verifier's cap (`Tags.MAX_TABLES`). -/
theorem lam_zk_ge_64 (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : ZkIn x s spec sts) (h64 : sts.size ≤ 64) :
    195.98 ≤ Claims.lamOf (Claims.Flock.κ (deployZk rel) x) (Claims.Flock.γ (deployZk rel) x) := by
  have := lam_zk_ge rel h
  have := logb_size_le_6 h.size_pos h64
  linarith

/-- **One plain table**: `ε_table ≤ 2^{−205}`, `κ ≤ 2^{−202.98}`, `γ ≤ 2^51` and `λ ≥ 201.98`. -/
theorem plain_one (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : PlainIn x s spec sts) (h1 : sts.size = 1) :
    (deployPlain rel).εtable x ≤ 2 ^ (-(205 : ℝ)) ∧ Claims.Flock.κ (deployPlain rel) x ≤ 2 ^ (-(202.98 : ℝ)) ∧
      Claims.Flock.γ (deployPlain rel) x ≤ 2 ^ (51 : ℝ) ∧
      201.98 ≤ Claims.lamOf (Claims.Flock.κ (deployPlain rel) x) (Claims.Flock.γ (deployPlain rel) x) := by
  have hε := εtable_le h.setup fun st hst =>
    have := h.tab st hst
    tableErr_plain_le spec st this.1 this.2.1 this.2.2
  have hκ := κ_plain_le rel h
  have hl := lam_plain_ge rel h
  have hr := Vk_plain_le rel h
  have hC := weight_plain_le rel h
  rw [h1] at hε hκ hl hr hC
  simp only [Nat.cast_one, one_mul, Real.logb_one, sub_zero] at hε hκ hl hC
  exact ⟨hε, hκ, Claims.Flock.Numbers.γ_le_51 _ x (by omega) (weight_nonneg false x) hC, hl⟩

/-- **One masked table**: `ε_table ≤ 2^{−205}`, `κ ≤ 2^{−202.98}`, `γ ≤ 2^51` and `λ ≥ 201.98`. -/
theorem zk_one (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : ZkIn x s spec sts) (h1 : sts.size = 1) :
    (deployZk rel).εtable x ≤ 2 ^ (-(205 : ℝ)) ∧ Claims.Flock.κ (deployZk rel) x ≤ 2 ^ (-(202.98 : ℝ)) ∧
      Claims.Flock.γ (deployZk rel) x ≤ 2 ^ (51 : ℝ) ∧
      201.98 ≤ Claims.lamOf (Claims.Flock.κ (deployZk rel) x) (Claims.Flock.γ (deployZk rel) x) := by
  have hε := εtable_le h.setup fun st hst =>
    have := h.tab st hst
    tableErr_zk_le spec st this.1 this.2.1 this.2.2
  have hκ := κ_zk_le rel h
  have hl := lam_zk_ge rel h
  have hr := Vk_zk_le rel h
  have hC := weight_zk_le rel h
  rw [h1] at hε hκ hl hr hC
  simp only [Nat.cast_one, one_mul, Real.logb_one, sub_zero] at hε hκ hl hC
  exact ⟨hε, hκ, Claims.Flock.Numbers.γ_le_51 _ x (by omega) (weight_nonneg true x) hC, hl⟩

end FlockDeploy
