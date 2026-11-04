import Definitions.Pouw.Assumptions.GameStatements
import Proofs.Pouw.Prob

/-!
# Facts about the named instance `D1.m1`

Its parameters and sharing, noise range, query bound and unit-specific `E_L` queries, a 16-unit workload in its
domain, and the bound on its explicit `W_ref`.
-/

namespace Pouw.Proofs

open Finset

variable {Q R S : Type}

/-- Entry bounds on the factors give `NoiseRange` when `r·α·β ≤ 63` for `E = E_L·E_R` and `r·α'·β' ≤ 63` for
`F = F_L·F_R`. W6's distribution (outer `[−3, 3]`, inner `±1`, `r = 16`) has `16·3·1 = 16·1·3 = 48`. -/
theorem noiseRange_of_factor_bounds (P : D1.Params Q R S) (α β α' β' : ℤ) (hE : (P.r : ℤ) * (α * β) ≤ 63)
    (hF : (P.r : ℤ) * (α' * β') ≤ 63)
    (hEL : ∀ sh a b, ∀ i < sh.m, ∀ t < P.r, |(P.expAct sh a b).1 i t| ≤ α)
    (hER : ∀ sh a b, ∀ t < P.r, ∀ l < sh.k, |(P.expAct sh a b).2 t l| ≤ β)
    (hFL : ∀ sh b, ∀ l < sh.k, ∀ t < P.r, |(P.expWt sh b).1 l t| ≤ α')
    (hFR : ∀ sh b, ∀ t < P.r, ∀ j < sh.n, |(P.expWt sh b).2 t j| ≤ β') : P.NoiseRange := by
  have key : ∀ (a b : ℤ), (P.r : ℤ) * (a * b) ≤ 63 → ∀ (x y : ℕ → ℤ), (∀ t < P.r, |x t| ≤ a) →
      (∀ t < P.r, |y t| ≤ b) → InNoise (∑ t ∈ range P.r, x t * y t) := by
    intro a b hab x y hx hy
    have h1 : |∑ t ∈ range P.r, x t * y t| ≤ ∑ t ∈ range P.r, |x t * y t| := Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ t ∈ range P.r, |x t * y t| ≤ (P.r : ℤ) * (a * b) := by
      have := Finset.sum_le_card_nsmul (range P.r) (fun t => |x t * y t|) (a * b) fun t ht => by
        have ht' := Finset.mem_range.1 ht
        rw [abs_mul]
        exact mul_le_mul (hx t ht') (hy t ht') (abs_nonneg _) ((abs_nonneg _).trans (hx t ht'))
      simpa [nsmul_eq_mul] using this
    have := abs_le.1 (h1.trans (h2.trans hab))
    exact ⟨this.1, this.2⟩
  exact ⟨fun sh a b i hi l _ => key α β hE (fun t => (P.expAct sh a b).1 i t) (fun t => (P.expAct sh a b).2 t l)
      (fun t ht => hEL sh a b i hi t ht) (fun t ht => hER sh a b t ht l ‹_›),
    fun sh b l hl j _ => key α' β' hF (fun t => (P.expWt sh b).1 l t) (fun t => (P.expWt sh b).2 t j)
      (fun t ht => hFL sh b l hl t ht) (fun t ht => hFR sh b t ht j ‹_›)⟩

namespace M1

open D1

variable (Wref : Layout → ℕ → ℕ)

theorem entry_abs_le (f : M1R → ℤ) (c : ℤ) (hc : 0 ≤ c) (hf : ∀ x, |f x| ≤ c) (a : List M1R) (i : ℕ) :
    |m1Entry f a i| ≤ c := by
  unfold m1Entry
  cases a[i]? with
  | none => simpa using hc
  | some x => exact hf x

theorem outer_abs_le (x : M1R) : |m1Outer x| ≤ 3 := by
  unfold m1Outer
  have := x.1.isLt
  rw [abs_le]; omega

theorem sign_abs_le (x : M1R) : |m1Sign x| ≤ 1 := by
  unfold m1Sign; split <;> simp

theorem entry_append (f : M1R → ℤ) (a b : List M1R) (i : ℕ) : m1Entry f (a ++ b) (a.length + i) = m1Entry f b i := by
  unfold m1Entry
  rw [List.getElem?_append_right (by omega), Nat.add_sub_cancel_left]

/-- `E_R` reads only the answers to its group's queries. -/
theorem expAct_snd (sh : Shape) (H : M1Q → M1R) (s : Salt) (u : ℕ) (A : Mat) (w : ℕ) (b : List M1R) :
    ((m1 Wref).expAct sh (((m1 Wref).qAct s u A w).map H) b).2 =
      fun t l => m1Entry m1Sign ((m1ERQ s w (u / 16)).map H) (t * sh.k + l) := by
  funext t l
  have h : ((m1ELQ s u A w).map H).length = 2 ^ 21 := by simp [m1ELQ]
  simp only [m1]
  rw [List.map_append, ← h, entry_append]

theorem isMilestone : (m1 Wref).IsMilestone := by
  refine ⟨rfl, rfl, fun _ _ _ _ => rfl, fun H s sh u u' A A' w h => ?_⟩
  rw [expAct_snd, expAct_snd, h]

theorem noiseRange : (m1 Wref).NoiseRange :=
  noiseRange_of_factor_bounds (m1 Wref) 3 1 1 3 (by show ((16 : ℕ) : ℤ) * (3 * 1) ≤ 63; norm_num)
    (by show ((16 : ℕ) : ℤ) * (1 * 3) ≤ 63; norm_num)
    (fun _ a _ _ _ _ _ => entry_abs_le _ 3 (by norm_num) outer_abs_le a _)
    (fun _ a _ _ _ _ _ => entry_abs_le _ 1 (by norm_num) sign_abs_le a _)
    (fun _ b _ _ _ _ => entry_abs_le _ 1 (by norm_num) sign_abs_le b _)
    (fun _ b _ _ _ _ => entry_abs_le _ 3 (by norm_num) outer_abs_le b _)

theorem qAct_length (s : Salt) (u : ℕ) (A : Mat) (w : ℕ) : ((m1 Wref).qAct s u A w).length = 2 ^ 21 + 2 ^ 20 := by
  simp [m1, m1ELQ, m1ERQ]

theorem queryBound : (m1 Wref).QueryBound m1Queries := fun s u A w => by
  rw [qAct_length]
  simp [m1, m1FQ, m1Queries]

/-- A unit's `E_L` queries are asked for no other unit below `2^64`. -/
theorem elQ_unique (s s' : Salt) (u u' : ℕ) (A A' : Mat) (w w' : ℕ) (hu : u < 2 ^ 64) (hu' : u' < 2 ^ 64)
    (x : M1Q) (hx : x ∈ m1ELQ s u A w) (hx' : x ∈ (m1 Wref).qAct s' u' A' w' ++ (m1 Wref).qWt s' u' w') : u = u' := by
  simp only [m1, m1ELQ, m1ERQ, m1FQ, List.mem_append, List.mem_map] at hx hx'
  obtain ⟨c, -, rfl⟩ := hx
  rcases hx' with (⟨c', -, h⟩ | ⟨c', -, h⟩) | ⟨c', -, h⟩
  · simp only [Sum.inl.injEq, Prod.mk.injEq] at h
    have := congrArg Fin.val h.2.1
    simp only [field, Nat.mod_eq_of_lt hu, Nat.mod_eq_of_lt hu'] at this
    exact this.symm
  · simp at h
  · simp at h

/-- Sixteen milestone-shape units on one zero weight: one full group of `2^16` rows, the least the domain allows. -/
def sixteen : Workload := ⟨⟨16, fun _ => milestoneShape, fun _ => 0⟩, fun _ => 0⟩

theorem milestone_m1Shape : M1Shape milestoneShape := by
  unfold M1Shape milestoneShape; norm_num

theorem sixteen_groupRows (u : ℕ) (hu : u < 16) : m1GroupRows sixteen.layout u = 2 ^ 16 := by
  show ∑ u' ∈ (range 16).filter (fun u' => u' / 16 = u / 16 ∧ (0 : ℕ) = 0), milestoneShape.m = 2 ^ 16
  rw [Finset.filter_true_of_mem fun u' hu' => ⟨by have := Finset.mem_range.1 hu'; omega, rfl⟩,
    Finset.sum_const, Finset.card_range]
  unfold milestoneShape; norm_num

theorem sixteen_inDomain : sixteen.InDomain m1Domain := by
  refine ⟨by norm_num [sixteen], fun u hu => ⟨milestone_m1Shape, by norm_num [sixteen], ?_⟩, fun _ _ _ _ _ _ => ?_⟩
  · exact le_of_eq (sixteen_groupRows u hu).symm
  · unfold InInt7 sixteen; norm_num

theorem entry_const (f : M1R → ℤ) (l : List M1Q) (c : M1R) (i : ℕ) (hi : i < l.length) :
    m1Entry f (l.map fun _ => c) i = f c := by
  unfold m1Entry
  rw [List.getElem?_map, List.getElem?_eq_getElem hi]
  rfl

/-- The constant oracle `(4, +)` gives the noise entry `16` at `(0, 0)`. -/
theorem noise_ne_zero (s : Salt) (A : Mat) :
    mm (m1 Wref).r ((m1 Wref).actF (fun _ => ((4 : Fin 7), true)) s sixteen 0 A).1
      ((m1 Wref).actF (fun _ => ((4 : Fin 7), true)) s sixteen 0 A).2 0 0 ≠ 0 := by
  have h : ∀ t ∈ range 16,
      ((m1 Wref).actF (fun _ => ((4 : Fin 7), true)) s sixteen 0 A).1 0 t *
        ((m1 Wref).actF (fun _ => ((4 : Fin 7), true)) s sixteen 0 A).2 t 0 = 1 := by
    intro t ht
    have ht := Finset.mem_range.1 ht
    simp only [D1.Params.actF, m1]
    rw [entry_const _ _ _ _ (by simp [m1ELQ, m1ERQ]; omega),
      entry_const _ _ _ _ (by simp [m1ELQ, m1ERQ, sixteen, milestoneShape]; omega)]
    simp [m1Outer, m1Sign]
  unfold mm
  rw [show (m1 Wref).r = 16 from rfl, Finset.sum_congr rfl h]
  simp

theorem salt_zero_lt : 0 < 2 ^ 128 := Nat.two_pow_pos 128

/-! ## The schedule `m1Wref` -/

theorem rowShare_le (work mu rows c : ℕ) (hrows : 0 < rows) (h : work * mu ≤ rows * c) :
    rowShare work mu rows ≤ c := by
  unfold rowShare
  rw [Nat.div_le_iff_le_mul_add_pred hrows]
  generalize work * mu = a at *
  generalize rows * c = b at *
  omega

theorem groupRows_le_rowsOf (L : Layout) (u : ℕ) : m1GroupRows L u ≤ L.rowsOf (L.wid u) :=
  Finset.sum_le_sum_of_subset (Finset.monotone_filter_right _ fun _ _ h => h.2)

theorem m1Wref_le (L : Layout) (hD : m1Domain L) (u : ℕ) (hu : u < L.N) :
    m1Wref L u ≤ (L.shape u).m * 2 ^ 32 + 233 * 2 ^ 16 * (L.shape u).m := by
  obtain ⟨⟨-, -, -, hk, hn⟩, -, hg⟩ := hD u hu
  have hw := (groupRows_le_rowsOf L u)
  have h1 : rowShare (16 * (L.shape u).k * (L.shape u).n) (L.shape u).m (m1GroupRows L u) ≤
      16 * 2 ^ 16 * (L.shape u).m := by
    refine rowShare_le _ _ _ _ (lt_of_lt_of_le (by norm_num) hg) ?_
    rw [hk, hn]
    nlinarith
  have h3 : rowShare (576 * (L.shape u).n) (L.shape u).m (m1GroupRows L u) ≤ 576 * (L.shape u).m := by
    refine rowShare_le _ _ _ _ (lt_of_lt_of_le (by norm_num) hg) ?_
    rw [hn]
    nlinarith
  have h2 : rowShare ((16 + 20) * (L.shape u).k * (L.shape u).n) (L.shape u).m (L.rowsOf (L.wid u)) ≤
      36 * 2 ^ 16 * (L.shape u).m := by
    refine rowShare_le _ _ _ _ (lt_of_lt_of_le (by norm_num) (hg.trans hw)) ?_
    rw [hk, hn]
    nlinarith
  dsimp only [m1Wref]
  rw [hk, hn] at *
  linarith

theorem Wmm_le_m1Wref (L : Layout) (u : ℕ) : (L.shape u).Wmm ≤ m1Wref L u := by
  unfold Shape.Wmm
  dsimp only [m1Wref]
  omega

theorem m1Wref_real (L : Layout) (hD : m1Domain L) (u : ℕ) (hu : u < L.N) :
    (m1Wref L u : ℝ) ≤ m1Omega * ((L.shape u).Wmm : ℝ) := by
  have h := m1Wref_le L hD u hu
  obtain ⟨-, -, -, hk, hn⟩ := (hD u hu).1
  have hW : (L.shape u).Wmm = (L.shape u).m * 2 ^ 32 := by unfold Shape.Wmm; rw [hk, hn]; ring
  have h' : (m1Wref L u : ℝ) ≤ (((L.shape u).m * 2 ^ 32 + 233 * 2 ^ 16 * (L.shape u).m : ℕ) : ℝ) := by
    exact_mod_cast h
  rw [hW]
  unfold m1Omega
  push_cast at h' ⊢
  linarith

end M1

/-- **The named instance.** -/
theorem milestoneInstance : GameStatements.MilestoneInstance := fun Wref =>
  ⟨M1.isMilestone Wref, M1.noiseRange Wref, M1.queryBound Wref,
    fun s s' u u' A A' w w' hu hu' x hx hx' => M1.elQ_unique Wref s s' u u' A A' w w' hu hu' x hx hx',
    fun _ => ((4 : Fin 7), true), ⟨0, M1.salt_zero_lt⟩, M1.sixteen, 0, M1.sixteen_inDomain,
    by norm_num [M1.sixteen], M1.noise_ne_zero Wref _ 0⟩

/-- **The milestone's `W_ref`**: `W_mm ≤ m1Wref ≤ (65769/65536)·W_mm` on the domain, and `γ = 14017/1644225`. -/
theorem milestoneWref : GameStatements.MilestoneWref :=
  ⟨fun U hU u hu => ⟨M1.Wmm_le_m1Wref U.layout u, M1.m1Wref_real U.layout hU.2.1 u hu⟩,
    by unfold D1.m1Omega; norm_num,
    by unfold reductionGamma Params.γ0 D1.m1Omega; norm_num⟩

/-- **`E_R` is separated across (weight, group)**, beside `MilestoneInstance`'s `E_L` clause: the `E_R` queries of
two (weight, group) pairs with weight ids below `2^64` and groups below `2^60` meet only if the pairs are equal. -/
theorem milestoneERSeparated (s s' : Salt) (w w' g g' : ℕ) (hw : w < 2 ^ 64) (hw' : w' < 2 ^ 64)
    (hg : g < 2 ^ 60) (hg' : g' < 2 ^ 60) (x : D1.M1Q) (hx : x ∈ D1.m1ERQ s w g) (hx' : x ∈ D1.m1ERQ s' w' g') :
    w = w' ∧ g = g' := by
  simp only [D1.m1ERQ, List.mem_map] at hx hx'
  obtain ⟨c, -, rfl⟩ := hx
  obtain ⟨c', -, h⟩ := hx'
  simp only [Sum.inr.injEq, Sum.inl.injEq, Prod.mk.injEq] at h
  have h1 := congrArg Fin.val h.2.1
  have h2 := congrArg Fin.val h.2.2.1
  simp only [D1.field, Nat.mod_eq_of_lt hw, Nat.mod_eq_of_lt hw', Nat.mod_eq_of_lt hg, Nat.mod_eq_of_lt hg']
    at h1 h2
  exact ⟨h1.symm, h2.symm⟩

end Pouw.Proofs
