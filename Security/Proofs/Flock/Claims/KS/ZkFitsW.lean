import Proofs.Flock.Claims.KS.FitsR
import Proofs.Flock.Claims.KS.ZkAccJ

/-!
# The masked escapes with level 0's fork event fit the weight, counted per round

The witness-lane bound (`FlockSoundness.Discharge.ZkLink.session_sound_of_table_accZ_w`) adds level 0's term
`√(2^{d₀}·2Q₀·Adv₀)` to each table's, one more fork event per table (`IZW`, `cZW`). Joined by round with the old ones
(`joinEscW`, on accepted runs where every table's are: `joinEscW_onAcc`), the weights are every table's
(`joinEscW_c_sum`), and those at round `0` are at most every table's but `2` (`joinEscW_c_zero_le`): a table puts at
most one event there, and every event but one weighs at least `2` (`cZW_add_two_le`, at least one `τ` and level 0's
or both `τ`'s).

Each table's weight is `2·Σ_ℓ 2^{d_ℓ}·Q_ℓ + 2·N² + 2 + 4·2^{d₀}/e`, which pays exactly for its fork events with level
0's and its share of `k` (`cZW_sum_le_tableWeight`); `k < 4·N₀/e + 1` leaves `FitsFork` short (it charges every event
twice, and the tables' weights leave less than `1` for `k`'s ceiling). Counted per round (`Claims.Escapes.FitsForkR`)
only round `0`'s events are charged twice, and those weigh at least `2` less than the whole per table, so the joined
escapes fit the deployed weight with the fork event of weight `k` (`joinEscW_fitsForkR_zk`).
-/

namespace Claims.Flock

/-! ## Level 0's fork event, joined by round -/

section Join

/-- The masked fork events with level 0's: `IZ`'s, and one at level 0. -/
abbrev IZW (L : ℕ) : Type := IZ L ⊕ Unit

/-- Their weights: `cZ`'s, and level 0's `2^{d₀}·2Q₀`. -/
noncomputable def cZW (sch : FlockSoundness.Accounting.Schedule) (Np : ℕ) : IZW sch.levels.length → ℝ
  | .inl i => cZ sch Np i
  | .inr _ => 2 ^ (FlockSoundness.lvOf sch 0).logLen * (2 * (FlockSoundness.lvOf sch 0).queries)

theorem cZW_nonneg (sch : FlockSoundness.Accounting.Schedule) (Np : ℕ) (i : IZW sch.levels.length) :
    0 ≤ cZW sch Np i :=
  match i with
  | .inl i => cZ_nonneg sch Np i
  | .inr _ => by unfold cZW; positivity

theorem lvOf_zero (sch : FlockSoundness.Accounting.Schedule) :
    FlockSoundness.lvOf sch 0 = sch.levels.headD default := by
  unfold FlockSoundness.lvOf
  cases sch.levels <;> rfl

/-- **The masked weights sum to the levels', both pads roots', both `τ`'s and level 0's.** -/
theorem cZW_sum (sch : FlockSoundness.Accounting.Schedule) (Np : ℕ) :
    ∑ i, cZW sch Np i = (∑ i : Fin 2 × Fin sch.levels.length, cOf sch (i.2.val + 1)) + 2 * ((Np : ℝ) * Np) + 2 +
      2 ^ (FlockSoundness.lvOf sch 0).logLen * (2 * (FlockSoundness.lvOf sch 0).queries) := by
  show ∑ i : ((Fin 2 × Fin sch.levels.length) ⊕ (Fin 2 ⊕ Fin 2)) ⊕ Unit, cZW sch Np i = _
  rw [Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_sum_type]
  simp only [cZW, cZ, Finset.sum_const, Finset.card_univ, Fintype.card_fin, Fintype.card_unit, nsmul_eq_mul,
    Nat.cast_ofNat, mul_one, one_mul, Nat.cast_one]
  ring

/-- **Every masked event but one weighs at least `2`**, when level 0 has a query. -/
theorem cZW_add_two_le (sch : FlockSoundness.Accounting.Schedule) (Np : ℕ)
    (hq : 1 ≤ (FlockSoundness.lvOf sch 0).queries) (i : IZW sch.levels.length) :
    cZW sch Np i + 2 ≤ ∑ i, cZW sch Np i := by
  rw [cZW_sum]
  have hA : 0 ≤ ∑ i : Fin 2 × Fin sch.levels.length, cOf sch (i.2.val + 1) :=
    Finset.sum_nonneg fun i _ => cOf_nonneg sch _
  have hN : 0 ≤ (Np : ℝ) * Np := mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have h0 : (2 : ℝ) ≤ 2 ^ (FlockSoundness.lvOf sch 0).logLen * (2 * (FlockSoundness.lvOf sch 0).queries) := by
    have h1 : (1 : ℝ) ≤ 2 ^ (FlockSoundness.lvOf sch 0).logLen := one_le_pow₀ (by norm_num)
    have h2 : (1 : ℝ) ≤ (FlockSoundness.lvOf sch 0).queries := by exact_mod_cast hq
    nlinarith
  rcases i with (i | (r | r)) | u
  · have := Finset.single_le_sum (f := fun i : Fin 2 × Fin sch.levels.length => cOf sch (i.2.val + 1))
      (fun i _ => cOf_nonneg sch _) (Finset.mem_univ i)
    simp only [cZW, cZ] at this ⊢
    linarith
  · simp only [cZW, cZ]
    linarith
  · simp only [cZW, cZ]
    linarith
  · simp only [cZW]
    linarith

variable {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)

/-- **The masked escapes of `n` tables with level 0's, joined by round** (`joinEscZ` at `IZW`). -/
noncomputable def joinEscW (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZW (sch j).levels.length → Fin (V.k x))
    (ev : (j : ℕ) → IZW (sch j).levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → TiedColl H (run H P V x ω s).vLog) : Escapes H V P x where
  I := Fin (V.k x)
  c r := ∑ j ∈ Finset.range n, ∑ i ∈ Finset.univ.filter (fun i => rd j i = r), cZW (sch j) (Np j) i
  j r := r
  ev r ω s s' := ∃ j < n, ∃ i, rd j i = r ∧ ev j i ω s s'
  L := Unit
  β _ := 1
  ev₁ _ ω s := ∃ j < n, ev₁ j ω s
  c_nonneg _ := Finset.sum_nonneg fun j _ => Finset.sum_nonneg fun i _ => cZW_nonneg _ _ i
  β_nonneg _ := zero_le_one
  tied r ω s s' := by
    rintro ⟨j, -, i, rfl, h⟩
    exact tied j i ω s s' h
  tied₁ _ ω s := by
    rintro ⟨j, -, h⟩
    exact tied₁ j ω s h

variable {H V P x}

section Args

variable (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
  (rd : (j : ℕ) → IZW (sch j).levels.length → Fin (V.k x))
  (ev : (j : ℕ) → IZW (sch j).levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
  (ev₁ : ℕ → Ω → (Fin (V.k x) → Coin) → Prop)
  (tied : ∀ j i ω s s', ev j i ω s s' →
    TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (rd j i) s s')).vLog))
  (tied₁ : ∀ j ω s, ev₁ j ω s → TiedColl H (run H P V x ω s).vLog)

/-- **The joined weights are every table's.** -/
theorem joinEscW_c_sum :
    ∑ r, (joinEscW H V P x n sch Np rd ev ev₁ tied tied₁).c r = ∑ j ∈ Finset.range n, ∑ i, cZW (sch j) (Np j) i := by
  show ∑ r : Fin (V.k x), ∑ j ∈ Finset.range n, ∑ i ∈ Finset.univ.filter (fun i => rd j i = r),
    cZW (sch j) (Np j) i = _
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_fiberwise _ _ _

/-- **The joined weight at a round is at most every table's but `2`**, where each table puts at most one event there
and level 0 has a query at each table. -/
theorem joinEscW_c_le (hinj : ∀ j, Function.Injective (rd j))
    (hq : ∀ j < n, 1 ≤ (FlockSoundness.lvOf (sch j) 0).queries) (r : Fin (V.k x)) :
    (joinEscW H V P x n sch Np rd ev ev₁ tied tied₁).c r ≤
      ∑ j ∈ Finset.range n, (∑ i, cZW (sch j) (Np j) i - 2) := by
  show ∑ j ∈ Finset.range n, ∑ i ∈ Finset.univ.filter (fun i => rd j i = r), cZW (sch j) (Np j) i ≤ _
  refine Finset.sum_le_sum fun j hj => ?_
  have hj := Finset.mem_range.1 hj
  have hcard : (Finset.univ.filter fun i => rd j i = r).card ≤ 1 :=
    Finset.card_le_one.2 fun a ha b hb => by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
      exact hinj j (ha.trans hb.symm)
  rcases Nat.le_one_iff_eq_zero_or_eq_one.1 hcard with h0 | h1
  · rw [Finset.card_eq_zero.1 h0, Finset.sum_empty]
    have := cZW_add_two_le (sch j) (Np j) (hq j hj) (.inr ())
    linarith [cZW_nonneg (sch j) (Np j) (.inr ())]
  · obtain ⟨a, ha⟩ := Finset.card_eq_one.1 h1
    rw [ha, Finset.sum_singleton]
    linarith [cZW_add_two_le (sch j) (Np j) (hq j hj) a]

/-- **The joined escapes fit `C` with the fork event of weight `c`, counted per round**, when `2·c` and every table's
weight but `2` and every table's weight are at most `2C`. -/
theorem joinEscW_fitsForkR (hinj : ∀ j, Function.Injective (rd j))
    (hq : ∀ j < n, 1 ≤ (FlockSoundness.lvOf (sch j) 0).queries) (C c : ℝ) (hk : 0 < V.k x)
    (h : 2 * c + ∑ j ∈ Finset.range n, (∑ i, cZW (sch j) (Np j) i - 2) +
      ∑ j ∈ Finset.range n, ∑ i, cZW (sch j) (Np j) i ≤ 2 * C) :
    (joinEscW H V P x n sch Np rd ev ev₁ tied tied₁).FitsForkR C c := by
  refine Escapes.fitsForkR_of_round _ C c ⟨fun a b h => h, fun r => ⟨r, rfl⟩⟩ (⟨0, hk⟩ : Fin (V.k x)) rfl
    (le_trans ?_ h) ?_
  · rw [joinEscW_c_sum]
    linarith [joinEscW_c_le n sch Np rd ev ev₁ tied tied₁ hinj hq ⟨0, hk⟩]
  · show ∑ _l : Unit, (1 : ℝ) ≤ V.k x
    rw [Fintype.sum_unique]
    exact_mod_cast hk

/-- **The joined escapes are on accepted runs** where every table's events are. -/
theorem joinEscW_onAcc
    (hacc : ∀ j i ω s s', ev j i ω s s' →
      (run H P V x ω s).out = .acc ∧ (run H P V x ω (forkCoins (rd j i) s s')).out = .acc)
    (hacc₁ : ∀ j ω s, ev₁ j ω s → (run H P V x ω s).out = .acc) :
    (joinEscW H V P x n sch Np rd ev ev₁ tied tied₁).OnAcc := by
  refine ⟨?_, ?_⟩
  · rintro r ω s s' ⟨j, -, i, rfl, h⟩
    exact hacc j i ω s s' h
  · rintro - ω s ⟨j, -, h⟩
    exact hacc₁ j ω s h

end Args

end Join

end Claims.Flock

namespace FlockDeploy.KS

open Flock FlockSoundness

/-- **A table's weight pays for its masked fork events with level 0's and its share of `k`**: the fork events'
weights are `2·Σ_ℓ 2^{d_ℓ}·Q_ℓ + 2·N_p² + 2`, level 0's included, within the table's
`2·Σ_ℓ 2^{d_ℓ}·Q_ℓ + 2·N² + 2 + 4·2^{d₀}/e` when `N_p` is at most the deployed pads' domain. -/
theorem cZW_sum_le_tableWeight {st : Stmt} {sch : FlockSoundness.Accounting.Schedule} (hsch : sch = Accounting.m1Schedule st.m)
    (Np : ℕ)
    (hNp : Np ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0)) :
    4 * tableLen true st / Real.exp 1 + ∑ i, Claims.Flock.cZW sch Np i ≤ tableWeight true st := by
  set f : Accounting.Level → ℝ := fun l => 2 ^ l.logLen * l.queries with hf
  have hd : (default : Accounting.Level).queries = 0 := rfl
  have hhead := Claims.Flock.head_add_sum_shift f (by simp [hf, hd]) sch.levels
  have hsum : ∑ i, Claims.Flock.cZW sch Np i =
      2 * ∑ ℓ : Fin sch.levels.length, f (sch.levels.getD (ℓ.val + 1) default) + 2 * ((Np : ℝ) * Np) + 2 +
        2 * f (sch.levels.headD default) := by
    rw [Claims.Flock.cZW_sum, Claims.Flock.lvOf_zero, Fintype.sum_prod_type]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Nat.cast_ofNat, hf]
    unfold Claims.Flock.cOf
    ring
  set pd := Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0)
  have htw : tableWeight true st = Claims.Flock.Numbers.weightOf sch pd := by
    simp only [tableWeight, ↓reduceIte, hsch, pd]
  have hNp2 : (Np : ℝ) * Np ≤ (pd : ℝ) ^ 2 := by
    have := Nat.mul_le_mul hNp hNp
    rw [sq]
    exact_mod_cast this
  have hlen : tableLen true st = 2 ^ (Claims.Flock.zkLevel0 st.m).d := rfl
  have hfits := kFits_all true st.m
  simp only [level0, wSched, ↓reduceIte, ← hsch] at hfits
  have hpow : (2 : ℝ) ^ (Claims.Flock.zkLevel0 st.m).d ≤ 2 ^ (sch.levels.headD default).logLen :=
    pow_le_pow_right₀ (by norm_num) hfits
  have hdiv : 4 * (2 : ℝ) ^ (Claims.Flock.zkLevel0 st.m).d / Real.exp 1 ≤
      4 * 2 ^ (sch.levels.headD default).logLen / Real.exp 1 := by
    gcongr
  rw [hsum, hlen, htw, Claims.Flock.Numbers.weightOf, weightNat_cast, ← hhead]
  linarith

/-- **The masked escapes with level 0's fit the deployed weight with the fork event of weight `k`, counted per
round**, at a statement whose tables `st j` are C-Flock ZK's, when each table's pads' weight is at most its deployed
pads' domain and each table puts at most one event at a round: `2·k < 8·N₀/e + 2`, round `0`'s weights are at most
every table's but `2`, and every table's at most its weight but `4·len/e` (`cZW_sum_le_tableWeight`). -/
theorem joinEscW_fitsForkR_zk (rel : Readable.Parsed → Readable.Relation) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} (st : ℕ → Stmt)
    (hst : ∀ j < (tables true x).length, (tables true x)[j]? = some (spec, st j))
    (h25 : ∀ j < (tables true x).length, 25 ≤ (st j).m) (h35 : ∀ j < (tables true x).length, (st j).m ≤ 35)
    (sch : ℕ → FlockSoundness.Accounting.Schedule) (hsch : ∀ j < (tables true x).length, sch j = Accounting.m1Schedule (st j).m)
    (Np : ℕ → ℕ) (hNp : ∀ j < (tables true x).length,
      Np j ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new (st j).m (st j).c.kLog
        (Claims.Flock.zkSched (st j).m).1.k0))
    (rd : (j : ℕ) → Claims.Flock.IZW (sch j).levels.length → Fin ((deployZk rel).V.k x))
    (hinj : ∀ j, Function.Injective (rd j))
    (ev : (j : ℕ) → Claims.Flock.IZW (sch j).levels.length → Ω → (Fin ((deployZk rel).V.k x) → Coin) →
      (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployZk rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployZk rel).V x ω (Claims.forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → Claims.TiedColl sha512 (Claims.run sha512 P (deployZk rel).V x ω s).vLog) :
    (Claims.Flock.joinEscW sha512 (deployZk rel).V P x (tables true x).length sch Np rd ev ev₁ tied
      tied₁).FitsForkR ((deployZk rel).weight x) ((deployZk rel).k x) := by
  set n := (tables true x).length with hn
  have hq : ∀ j < n, 1 ≤ (FlockSoundness.lvOf (sch j) 0).queries := fun j hj => by
    rw [Claims.Flock.lvOf_zero, hsch j hj]
    have := m1_queries₀_pos ⟨(st j).m - 25, by have := h35 j hj; omega⟩
    simp only [Nat.sub_add_cancel (h25 j hj)] at this
    exact this
  refine Claims.Flock.joinEscW_fitsForkR n sch Np rd ev ev₁ tied tied₁ hinj hq _ _ (V_k_pos true x) ?_
  have hw : weight true x = ∑ j ∈ Finset.range n, tableWeight true (st j) :=
    sum_map_eq_range (tables true x) (fun j => (spec, st j)) hst _
  have hN : N₀ true x = ∑ j ∈ Finset.range n, tableLen true (st j) :=
    sum_map_eq_range (tables true x) (fun j => (spec, st j)) hst _
  have hk : ((deployZk rel).k x : ℝ) = (⌈4 * N₀ true x / Real.exp 1⌉₊ : ℝ) := rfl
  show 2 * ((deployZk rel).k x : ℝ) + _ + _ ≤ 2 * weight true x
  rcases Nat.eq_zero_or_pos n with h0 | hpos
  · have := k_le_weight true x
    rw [h0, Finset.sum_range_zero, Finset.sum_range_zero, add_zero, add_zero, hk]
    linarith
  have hper : ∀ j ∈ Finset.range n, 4 * tableLen true (st j) / Real.exp 1 +
      ∑ i, Claims.Flock.cZW (sch j) (Np j) i ≤ tableWeight true (st j) := fun j hj => by
    have hj := Finset.mem_range.1 hj
    exact cZW_sum_le_tableWeight (hsch j hj) (Np j) (hNp j hj)
  have hs := Finset.sum_le_sum hper
  rw [Finset.sum_add_distrib, ← Finset.sum_div, ← Finset.mul_sum, ← hN, ← hw] at hs
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have he := Real.exp_pos 1
  have hceil := Nat.ceil_lt_add_one (div_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 4) (N₀_nonneg true x)) he.le)
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast hpos
  rw [hk]
  linarith

end FlockDeploy.KS
