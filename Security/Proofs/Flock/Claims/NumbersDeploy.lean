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
* `m′ = 210` and `K = 2^256` (`waits_deploy`, `cap_deploy`), `N₀ = 2^{d₀}` with `10 ≤ d₀ ≤ 23`, and
  `k = ⌈4N₀/e⌉ = kVal d₀` exactly (`k_plain`, `k_zk`);
* the verifier's rounds are at most `2 + 4400·n` (`Vk_le`: each rep at most `2200`, `repRounds_plain_le`,
  `repRounds_zk_le`) and the weight at most `n·2^33` (`weight_le`);
* `κ ≤ n·2^{−202.98}` and `λ ≥ 201.98 − log₂ n` (`κ_plain_le`, `lam_plain_ge`, `κ_zk_le`, `lam_zk_ge`); at one table
  `ε_table ≤ 2^{−205}`, `κ ≤ 2^{−202.98}`, `γ ≤ 2^51` and `λ ≥ 201.98` (`plain_one`, `zk_one`).

`ε_table` here is the session's sum over its tables, so the spec's per-table `2^{−205}` costs `log₂ n` bits of `λ` at
`n` tables.
-/

namespace Claims.Flock.Numbers

section Deploy

variable {Coin Col T : Type}

/-- **`κ ≤ n·2^{−202.98}`** when `ε_table ≤ n·2^{−205}`, `n ≥ 1`, at the deployed `m′`, `K` and `N₀`. -/
theorem κ_le_mul (D : Deploy Coin Col T) (x : Msg) {n : ℝ} (hn : 1 ≤ n) (hε : D.εtable x ≤ n * 2 ^ (-(205 : ℝ)))
    (hw : 210 ≤ D.waits x) (hK : D.cap x = cap) (hN : 2 ^ 10 ≤ D.N₀ x) (hN' : D.N₀ x ≤ 2 ^ 23) :
    κ D x ≤ n * 2 ^ (-(202.98 : ℝ)) := by
  have h1 := κ_le { D with εtable := fun _ => 2 ^ (-(205 : ℝ)) } x le_rfl hw hK hN hN'
  have e : κ { D with εtable := fun _ => (2 : ℝ) ^ (-(205 : ℝ)) } x =
      4 * 2 ^ (-(205 : ℝ)) + (κ D x - 4 * D.εtable x) := by
    unfold κ Deploy.k
    dsimp only
    ring
  have hR : 0 ≤ κ D x - 4 * D.εtable x := by
    have : κ D x - 4 * D.εtable x =
        2 ^ (-(D.waits x : ℝ)) + 2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8) := by
      unfold κ; ring
    rw [this]
    positivity
  rw [e] at h1
  calc κ D x = 4 * D.εtable x + (κ D x - 4 * D.εtable x) := by ring
    _ ≤ 4 * (n * 2 ^ (-(205 : ℝ))) + n * (κ D x - 4 * D.εtable x) :=
        add_le_add (by linarith) (le_mul_of_one_le_left hR hn)
    _ = n * (4 * 2 ^ (-(205 : ℝ)) + (κ D x - 4 * D.εtable x)) := by ring
    _ ≤ n * 2 ^ (-(202.98 : ℝ)) := mul_le_mul_of_nonneg_left h1 (by linarith)

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

theorem N₀_eq {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupOf zk x = some (s, spec, sts)) (h0 : 0 < sts.size) :
    N₀ zk x = 2 ^ (if zk then zkLevel0 sts[0].m else plainLevel0 sts[0].m).d := by
  have : (tables zk x).head? = some (spec, sts[0]) := by
    simp [tables_eq hs, List.head?_eq_getElem?, h0]
  simp [N₀, this]

/-- **`N₀` and `k` without `--zk`**: `N₀ = 2^{d₀}`, table 0's plain level-0 length, `2^10 ≤ N₀ ≤ 2^23`, and
`k = ⌈4N₀/e⌉ = kVal d₀`. -/
theorem k_plain (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hs : setupOf false x = some (s, spec, sts)) (h0 : 0 < sts.size) (h1 : 22 ≤ sts[0].m)
    (h2 : sts[0].m ≤ 35) :
    (deployPlain rel).N₀ x = 2 ^ (plainLevel0 sts[0].m).d ∧ 2 ^ 10 ≤ (deployPlain rel).N₀ x ∧
      (deployPlain rel).N₀ x ≤ 2 ^ 23 ∧ (deployPlain rel).k x = Claims.Flock.Numbers.kVal (plainLevel0 sts[0].m).d := by
  obtain ⟨i, hi⟩ := plainIdx_of h1 h2
  have hd := (plain_facts i).2.2
  rw [Claims.Flock.plainLen, hi] at hd
  have hN : (deployPlain rel).N₀ x = 2 ^ (plainLevel0 sts[0].m).d := N₀_eq hs h0
  obtain ⟨hlo, hhi⟩ := d_mem_bounds hd
  refine ⟨hN, ?_, ?_, ?_⟩
  · rw [hN]; exact pow_le_pow_right₀ (by norm_num) hlo
  · rw [hN]; exact pow_le_pow_right₀ (by norm_num) hhi
  · unfold Claims.Flock.Deploy.k
    rw [hN]
    exact Claims.Flock.Numbers.kRep_eq hd

/-- **`N₀` and `k` with `--zk`**: `N₀ = 2^{d₀}`, table 0's masked level-0 length, `2^10 ≤ N₀ ≤ 2^23`, and
`k = ⌈4N₀/e⌉ = kVal d₀`. -/
theorem k_zk (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hs : setupOf true x = some (s, spec, sts)) (h0 : 0 < sts.size) (h1 : 25 ≤ sts[0].m)
    (h2 : sts[0].m ≤ 35) :
    (deployZk rel).N₀ x = 2 ^ (zkLevel0 sts[0].m).d ∧ 2 ^ 10 ≤ (deployZk rel).N₀ x ∧
      (deployZk rel).N₀ x ≤ 2 ^ 23 ∧ (deployZk rel).k x = Claims.Flock.Numbers.kVal (zkLevel0 sts[0].m).d := by
  obtain ⟨i, hi⟩ := zkIdx_of h1 h2
  have hd := (zk_facts i).2.2
  rw [Claims.Flock.zkLen, hi] at hd
  have hN : (deployZk rel).N₀ x = 2 ^ (zkLevel0 sts[0].m).d := N₀_eq hs h0
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
points: `(m − 3) + (k_log − 3) + (2 + 2·regions + 1) + (k0 + 5 + 11·(levels − 1)) ≤ 2188`. -/
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
points: `(m + k_log − 6) + (2 + 2·regions + 1) + (k0 + 7 + 11·(levels − 1)) + 4 ≤ 2194`. -/
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

theorem PlainIn.tab₀ {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (h : PlainIn x s spec sts) :
    22 ≤ (sts[0]'h.size_pos).m ∧ (sts[0]'h.size_pos).m ≤ 35 :=
  have := h.tab _ (Array.getElem_mem_toList h.size_pos)
  ⟨this.1, this.2.1⟩

theorem ZkIn.tab₀ {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (h : ZkIn x s spec sts) :
    25 ≤ (sts[0]'h.size_pos).m ∧ (sts[0]'h.size_pos).m ≤ 35 :=
  have := h.tab _ (Array.getElem_mem_toList h.size_pos)
  ⟨this.1, this.2.1⟩

/-- **`κ ≤ n·2^{−202.98}`** at the deployed plain verifier, `n` its tables. -/
theorem κ_plain_le (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : PlainIn x s spec sts) :
    Claims.Flock.κ (deployPlain rel) x ≤ sts.size * 2 ^ (-(202.98 : ℝ)) := by
  obtain ⟨-, hlo, hhi, -⟩ := k_plain rel h.setup h.size_pos h.tab₀.1 h.tab₀.2
  refine Claims.Flock.Numbers.κ_le_mul _ x (size_one_le h.size_pos) ?_ (le_of_eq rfl) rfl hlo hhi
  exact εtable_le h.setup fun st hst =>
    have := h.tab st hst
    tableErr_plain_le spec st this.1 this.2.1 this.2.2

/-- **`κ ≤ n·2^{−202.98}`** at the deployed masked verifier, `n` its tables. -/
theorem κ_zk_le (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : ZkIn x s spec sts) :
    Claims.Flock.κ (deployZk rel) x ≤ sts.size * 2 ^ (-(202.98 : ℝ)) := by
  obtain ⟨-, hlo, hhi, -⟩ := k_zk rel h.setup h.size_pos h.tab₀.1 h.tab₀.2
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
