module

public import Proofs.Properties.Spot.FactorUpper.Bounds
public import Definitions.Claims.SamplingTwoRankin
public import Properties.Spot.FactorUpper

/-!
# Rankin's factor bounded above: the sets' form

`FactorUpperSound`. A cell's escape is at most `e^{−k m/n}` (`escCell_le_exp`), and `0` once `k ≥ n` and `m ≥ 1`, so
`passes(U) ≤ φ(y(U))` at `φ(y) = 1 − q(1 − e^{−y})` and `y(U) = Σ_{v∈U} r_v`. The sum over the sets of a replay
block's proof blocks groups them by stratum and width (`keysOf`). Below each chord of `h` it is at most the product
`∏_g (1 + 2^{b_g} e^{−θ s r_g})^{n_g}` (`light_sound`), and a set meeting a cell sampled whole is charged at `h(∞)`
(`whole_sound`).
-/

@[expose] public section

namespace Properties.Spot.SecurityProofs.FactorUpperProof

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

/-! ## A cell's escape -/

theorem sub_mul_le (a n k : ℕ) (ha : a ≤ n) : (a - k) * n ≤ a * (n - k) := by
  rcases le_or_gt a k with h | h
  · rw [Nat.sub_eq_zero_of_le h, zero_mul]; exact Nat.zero_le _
  · obtain ⟨c, rfl⟩ : ∃ c, a = k + c := ⟨a - k, by omega⟩
    obtain ⟨e, rfl⟩ : ∃ e, n = k + c + e := ⟨n - (k + c), by omega⟩
    rw [show k + c - k = c by omega, show k + c + e - k = c + e by omega]
    nlinarith

theorem descFactorial_mul_le (n a : ℕ) (ha : a ≤ n) :
    ∀ k, (a.descFactorial k : ℝ) * (n : ℝ) ^ k ≤ (a : ℝ) ^ k * (n.descFactorial k : ℝ)
  | 0 => by simp
  | k + 1 => by
    rw [Nat.descFactorial_succ, Nat.descFactorial_succ]
    have ih := descFactorial_mul_le n a ha k
    have key : ((a - k : ℕ) : ℝ) * n ≤ (a : ℝ) * ((n - k : ℕ) : ℝ) := by exact_mod_cast sub_mul_le a n k ha
    push_cast
    calc ((a - k : ℕ) : ℝ) * (a.descFactorial k : ℝ) * (n : ℝ) ^ (k + 1)
        = (((a - k : ℕ) : ℝ) * n) * ((a.descFactorial k : ℝ) * (n : ℝ) ^ k) := by ring
      _ ≤ ((a : ℝ) * ((n - k : ℕ) : ℝ)) * ((a : ℝ) ^ k * (n.descFactorial k : ℝ)) :=
        mul_le_mul key ih (by positivity) (by positivity)
      _ = _ := by ring

theorem escCell_nonneg (n k m : ℕ) : 0 ≤ escCell n k m := by unfold escCell; positivity

theorem escCell_zero (n k : ℕ) : escCell n k 0 = 1 := by
  unfold escCell
  rw [Nat.sub_zero, div_self]
  exact_mod_cast (Nat.choose_pos (min_le_right k n)).ne'

/-- A cell of `n` proof blocks with `m` incorrect, `k` of them sampled, escapes with probability at most
`(1 − m/n)^k`. -/
theorem escCell_le_pow {n k m : ℕ} (hk : k ≤ n) (hm : m ≤ n) (hn : 0 < n) :
    escCell n k m ≤ (1 - (m : ℝ) / n) ^ k := by
  unfold escCell
  rw [min_eq_left hk]
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hf : (0 : ℝ) < (k ! : ℝ) := by exact_mod_cast factorial_pos k
  have hD : (0 : ℝ) < (n.descFactorial k : ℝ) := by exact_mod_cast Nat.descFactorial_pos.2 hk
  have h1 : ((n - m).choose k : ℝ) = ((n - m).descFactorial k : ℝ) / (k ! : ℝ) := by
    rw [Nat.descFactorial_eq_factorial_mul_choose]; push_cast; field_simp
  have h2 : (n.choose k : ℝ) = (n.descFactorial k : ℝ) / (k ! : ℝ) := by
    rw [Nat.descFactorial_eq_factorial_mul_choose]; push_cast; field_simp
  have h3 : (1 - (m : ℝ) / n) = ((n - m : ℕ) : ℝ) / n := by
    rw [Nat.cast_sub hm]; field_simp
  rw [h1, h2, h3, div_pow, div_div_div_cancel_right₀ hf.ne', div_le_div_iff₀ hD (by positivity)]
  have := descFactorial_mul_le n (n - m) (Nat.sub_le n m) k
  linarith

theorem escCell_le_exp {n k m : ℕ} (hk : k ≤ n) (hm : m ≤ n) (hn : 0 < n) :
    escCell n k m ≤ Real.exp (-((k : ℝ) * m / n)) := by
  refine (escCell_le_pow hk hm hn).trans ?_
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have h0 : 0 ≤ 1 - (m : ℝ) / n := by
    rw [sub_nonneg, div_le_one hnR]; exact_mod_cast hm
  have h1 : 1 - (m : ℝ) / n ≤ Real.exp (-((m : ℝ) / n)) := by linarith [Real.add_one_le_exp (-((m : ℝ) / n))]
  calc (1 - (m : ℝ) / n) ^ k ≤ Real.exp (-((m : ℝ) / n)) ^ k := pow_le_pow_left₀ h0 h1 k
    _ = Real.exp (-((k : ℝ) * m / n)) := by rw [← Real.exp_nat_mul]; congr 1; ring

/-- A cell sampled whole (`k ≥ n`) escapes only with no incorrect proof block. -/
theorem escCell_whole {n k m : ℕ} (hk : n ≤ k) (hm : m ≤ n) (hm0 : 0 < m) : escCell n k m = 0 := by
  unfold escCell
  rw [min_eq_right hk, Nat.choose_eq_zero_of_lt (by omega : n - m < n)]
  simp

theorem escCell_le_one (n k m : ℕ) : escCell n k m ≤ 1 := by
  unfold escCell
  have hC : (0 : ℝ) < (n.choose (min k n) : ℝ) := by exact_mod_cast Nat.choose_pos (min_le_right k n)
  rw [div_le_one hC]
  exact_mod_cast Nat.choose_le_choose _ (Nat.sub_le n m)

/-- The rate a cell's bound samples at, `0` when sampled whole. -/
def rateOf (c : Cell) : ℚ := c.rate.getD 0

theorem rateOf_nonneg (c : Cell) : 0 ≤ rateOf c := by
  unfold rateOf Cell.rate; split_ifs <;> simp; positivity

/-- **A cell's escape at the bound's rate**: a cell of `n ≤ c.size` proof blocks, sampled at `k ≥ c.k`, with `m ≤ n`
incorrect, escapes with probability at most `e^{−r m}`, `r` the bound's rate (`0` when sampled whole). -/
theorem escCell_le_rate {n k m : ℕ} (hm : m ≤ n) {c : Cell} (hck : c.k ≤ k) (hcn : n ≤ c.size) :
    escCell n k m ≤ Real.exp (-((rateOf c : ℝ) * m)) := by
  rcases Nat.eq_zero_or_pos m with h0 | hm0
  · subst h0; rw [escCell_zero]; simp
  by_cases hkn : n ≤ k
  · rw [escCell_whole hkn hm hm0]; exact (Real.exp_pos _).le
  simp only [not_le] at hkn
  have hn : 0 < n := by omega
  refine (escCell_le_exp hkn.le hm hn).trans (Real.exp_le_exp.2 ?_)
  unfold rateOf Cell.rate
  split_ifs with hs
  · simp only [Option.getD_none, Rat.cast_zero, zero_mul, neg_zero, Left.neg_nonpos_iff]; positivity
  · simp only [Option.getD_some]
    have hsR : (0 : ℝ) < c.size := by exact_mod_cast (show 0 < c.size by omega)
    have hnR : (0 : ℝ) < n := by exact_mod_cast hn
    rw [neg_le_neg_iff]
    push_cast
    have hk' : (c.k : ℝ) ≤ k := by exact_mod_cast hck
    have hn' : (n : ℝ) ≤ c.size := by exact_mod_cast hcn
    have hmR : (0 : ℝ) ≤ m := by positivity
    rw [show (c.k : ℝ) / c.size * m = c.k * m / c.size by ring, div_le_div_iff₀ hsR hnR]
    nlinarith [mul_le_mul_of_nonneg_right (mul_le_mul hk' hn' hnR.le (by positivity : (0 : ℝ) ≤ k)) hmR]

/-- A cell the bound samples whole is sampled whole. -/
theorem escCell_rate_none {n k m : ℕ} (hm : m ≤ n) (hm0 : 0 < m) {c : Cell} (hck : c.k ≤ k) (hcn : n ≤ c.size)
    (hr : c.rate = none) : escCell n k m = 0 := by
  unfold Cell.rate at hr
  split_ifs at hr with hs
  exact escCell_whole (by omega) hm hm0

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

/-! ## Sums over proof blocks, by key -/

section Keys

variable {G : Type*} [DecidableEq G]

theorem prod_le_keys (K : List G) (hK : K.Nodup) (D : Finset ℕ) (κ : ℕ → G) (hD : ∀ v ∈ D, κ v ∈ K) (n : G → ℕ)
    (hn : ∀ k ∈ K, cntOf κ D k ≤ n k) (f : G → ℝ) (hf : ∀ k, 1 ≤ f k) :
    ∏ v ∈ D, f (κ v) ≤ (K.map fun k => f k ^ n k).prod := by
  rw [Finset.prod_comp, ← List.prod_toFinset _ hK]
  have hsub : D.image κ ⊆ K.toFinset := fun k hk => by
    obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hk; exact List.mem_toFinset.2 (hD v hv)
  calc ∏ k ∈ D.image κ, f k ^ (D.filter fun v => κ v = k).card
      ≤ ∏ k ∈ K.toFinset, f k ^ (D.filter fun v => κ v = k).card :=
        Finset.prod_le_prod_of_subset_of_one_le₀ hsub (fun k _ => pow_nonneg (by linarith [hf k]) _)
          fun k _ _ => one_le_pow₀ (hf k)
    _ ≤ ∏ k ∈ K.toFinset, f k ^ n k :=
        Finset.prod_le_prod₀ (fun k _ => pow_nonneg (by linarith [hf k]) _) fun k hk =>
          pow_le_pow_right₀ (hf k) (hn k (List.mem_toFinset.1 hk))

theorem sum_le_keys (K : List G) (hK : K.Nodup) (D : Finset ℕ) (κ : ℕ → G) (hD : ∀ v ∈ D, κ v ∈ K) (n : G → ℕ)
    (hn : ∀ k ∈ K, cntOf κ D k ≤ n k) (f : G → ℝ) (hf : ∀ k, 0 ≤ f k) :
    ∑ v ∈ D, f (κ v) ≤ (K.map fun k => (n k : ℝ) * f k).sum := by
  rw [Finset.sum_comp, ← List.sum_toFinset _ hK]
  have hsub : D.image κ ⊆ K.toFinset := fun k hk => by
    obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hk; exact List.mem_toFinset.2 (hD v hv)
  calc ∑ k ∈ D.image κ, (D.filter fun v => κ v = k).card • f k
      ≤ ∑ k ∈ K.toFinset, (D.filter fun v => κ v = k).card • f k :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub fun k _ _ => nsmul_nonneg (hf k) _
    _ ≤ ∑ k ∈ K.toFinset, (n k : ℝ) * f k :=
        Finset.sum_le_sum fun k hk => by
          rw [nsmul_eq_mul]
          exact mul_le_mul_of_nonneg_right (by exact_mod_cast hn k (List.mem_toFinset.1 hk)) (hf k)

theorem capped_le_keys (K : List G) (hK : K.Nodup) (D : Finset ℕ) (κ : ℕ → G) (hD : ∀ v ∈ D, κ v ∈ K) (n : G → ℕ)
    (hn : ∀ k ∈ K, cntOf κ D k ≤ n k) (x : G → ℝ) (hx : ∀ k, 0 ≤ x k) (m : G → ℕ) :
    ∑ A ∈ D.powerset with ∀ k ∈ K.toFinset, cntOf κ A k ≤ m k, ∏ v ∈ A, x (κ v) ≤
      (K.map fun k => capP (m k) (n k) (x k)).prod := by
  rw [capped_sum_eq K.toFinset κ x D (fun v hv => List.mem_toFinset.2 (hD v hv)) m, ← List.prod_toFinset _ hK]
  exact Finset.prod_le_prod₀ (fun k _ => capP_nonneg _ _ (hx k)) fun k hk =>
    capP_mono (hn k (List.mem_toFinset.1 hk)) (hx k)

end Keys

/-! ## The checker's keys -/

theorem mem_dedup {l : List ℕ} {x : ℕ} : x ∈ dedup l ↔ x ∈ l := by
  induction l with
  | nil => simp [dedup]
  | cons y l ih =>
    unfold dedup
    split_ifs with h
    · rw [ih]; simp only [List.mem_cons]
      constructor
      · exact Or.inr
      · rintro (rfl | h'); · simpa using h
        · exact h'
    · simp only [List.mem_cons, ih]

theorem nodup_dedup (l : List ℕ) : (dedup l).Nodup := by
  induction l with
  | nil => simp [dedup]
  | cons y l ih =>
    unfold dedup
    split_ifs with h
    · exact ih
    · refine List.nodup_cons.2 ⟨fun hy => h ?_, ih⟩
      simpa using mem_dedup.1 hy

theorem countIn_pos_mem {parts : List (ℕ × ℕ)} {b : ℕ} (h : 0 < countIn parts b) : b ∈ parts.map Prod.snd := by
  unfold countIn at h
  by_contra hb
  have : (parts.filter fun x => x.2 == b) = [] := by
    rw [List.filter_eq_nil_iff]
    intro x hx hxb
    exact hb (List.mem_map.2 ⟨x, hx, by simpa using hxb⟩)
  rw [this] at h; simp at h

theorem mem_keysOf {cs : List Cell} {reads : List (List (ℕ × ℕ))} {k : ℕ × ℕ} :
    k ∈ keysOf cs reads ↔ k.1 < cs.length ∧ 0 < countIn (reads.getD k.1 []) k.2 := by
  unfold keysOf
  simp only [List.mem_flatMap, List.mem_range, List.mem_map, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro ⟨t, ht, b, ⟨_, hb⟩, rfl⟩; exact ⟨ht, hb⟩
  · rintro ⟨ht, hb⟩
    exact ⟨k.1, ht, k.2, ⟨mem_dedup.2 (countIn_pos_mem hb), hb⟩, rfl⟩

theorem keysOf_nodup (cs : List Cell) (reads : List (List (ℕ × ℕ))) : (keysOf cs reads).Nodup := by
  unfold keysOf
  refine List.nodup_flatMap.2 ⟨fun t _ => ?_, ?_⟩
  · exact ((nodup_dedup _).filter _).map (fun a b h => by simpa using h)
  · refine (List.nodup_range).pairwise_of_forall_ne fun t _ t' _ htt' => ?_
    show List.Disjoint _ _
    rw [List.disjoint_left]
    intro k hk hk'
    simp only [List.mem_map] at hk hk'
    obtain ⟨b, _, rfl⟩ := hk
    obtain ⟨b', _, h⟩ := hk'
    exact htt' (congrArg Prod.fst h).symm

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

/-! ## A replay block's proof blocks, by stratum and width -/

/-- A proof block's width: the bits it commits. -/
def wid (s : AuditStatement) (Pt : Partition) (v : ℕ) : ℕ := committed s.prog (Pt.blocks.getD v [])

/-- A proof block's key: its stratum and width. -/
def keyOf (s : AuditStatement) (Pt : Partition) (v : ℕ) : ℕ × ℕ := (Pt.stratumOf v, wid s Pt v)

theorem mem_blocksIn {Pt Pρ : Partition} {ρ v : ℕ} (h : v ∈ blocksIn Pt Pρ ρ) :
    v ∈ cell Pt Pρ ρ (Pt.stratumOf v) ∧ Pt.stratumOf v < Pt.strata.length := by
  unfold blocksIn at h
  obtain ⟨t, ht, hv⟩ := Finset.mem_biUnion.1 h
  have hv' := hv
  unfold cell at hv'
  rw [Finset.mem_filter] at hv'
  rw [hv'.2.2]
  exact ⟨hv, Finset.mem_range.1 ht⟩

theorem cell_subset_blocksIn {Pt Pρ : Partition} {ρ t : ℕ} (ht : t < Pt.strata.length) :
    cell Pt Pρ ρ t ⊆ blocksIn Pt Pρ ρ := fun _ hv => Finset.mem_biUnion.2 ⟨t, Finset.mem_range.2 ht, hv⟩

theorem inter_cell {Pt Pρ : Partition} {ρ t : ℕ} {Y : Finset ℕ} (hY : Y ⊆ blocksIn Pt Pρ ρ) :
    Y ∩ cell Pt Pρ ρ t = Y.filter fun v => Pt.stratumOf v = t := by
  ext v
  simp only [Finset.mem_inter, Finset.mem_filter]
  constructor
  · rintro ⟨hv, hc⟩; unfold cell at hc; exact ⟨hv, (Finset.mem_filter.1 hc).2.2⟩
  · rintro ⟨hv, rfl⟩; exact ⟨hv, (mem_blocksIn (hY hv)).1⟩

section Block

variable {s : AuditStatement} {Pt : Partition} {d : Design.Two} {ρ : ℕ} {cs : List Cell}

theorem getD_eq {t : ℕ} (ht : t < cs.length) : cs[t]? = some (cs.getD t default) := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem ht]

theorem covers_get (hc : Properties.Spot.Covers s Pt d ρ cs) {t : ℕ} (ht : t < Pt.strata.length) :
    (cs.getD t default).k ≤ d.k.getD t 0 ∧
      ∀ b, ((cell Pt d.part ρ t).filter fun v => wid s Pt v = b).card ≤ (cs.getD t default).count b :=
  hc.2 t _ (getD_eq (hc.1 ▸ ht))

theorem sum_countIn_le (parts : List (ℕ × ℕ)) (W : Finset ℕ) :
    ∑ b ∈ W, countIn parts b ≤ (parts.map Prod.fst).sum := by
  induction parts with
  | nil => simp [countIn]
  | cons x parts ih =>
    have hx : ∀ b, countIn (x :: parts) b = (if x.2 = b then x.1 else 0) + countIn parts b := by
      intro b; unfold countIn; simp only [List.filter_cons, beq_iff_eq]; split_ifs <;> simp
    simp only [hx, Finset.sum_add_distrib, Finset.sum_ite_eq, List.map_cons, List.sum_cons]
    have h1 : (if x.2 ∈ W then x.1 else 0) ≤ x.1 := by split_ifs <;> omega
    omega

theorem card_cell_le_size (hc : Properties.Spot.Covers s Pt d ρ cs) {t : ℕ} (ht : t < Pt.strata.length) :
    (cell Pt d.part ρ t).card ≤ (cs.getD t default).size := by
  rw [Finset.card_eq_sum_card_image (wid s Pt) (cell Pt d.part ρ t)]
  exact (Finset.sum_le_sum fun b _ => (covers_get hc ht).2 b).trans (sum_countIn_le _ _)

/-- A set's `y`: the bound's rates summed over its proof blocks, `0` in the cells sampled whole. -/
noncomputable def yOf (Pt : Partition) (cs : List Cell) (U : Finset ℕ) : ℝ :=
  ∑ v ∈ U, (rateOf (cs.getD (Pt.stratumOf v) default) : ℝ)

theorem yOf_nonneg (U : Finset ℕ) : 0 ≤ yOf Pt cs U :=
  Finset.sum_nonneg fun _ _ => by exact_mod_cast rateOf_nonneg _

theorem esc_nonneg (fp : ℕ → ℕ → ℕ) : 0 ≤ d.esc Pt fp ρ :=
  Finset.prod_nonneg fun _ _ => escCell_nonneg _ _ _

theorem esc_le_exp (hc : Properties.Spot.Covers s Pt d ρ cs) {U : Finset ℕ}
    (hU : U ⊆ blocksIn Pt d.part ρ) : d.esc Pt (counts Pt d.part U) ρ ≤ Real.exp (-yOf Pt cs U) := by
  unfold Design.Two.esc
  have hy : yOf Pt cs U = ∑ t ∈ Finset.range Pt.strata.length,
      (rateOf (cs.getD t default) : ℝ) * (counts Pt d.part U ρ t : ℝ) := by
    unfold yOf counts
    rw [← Finset.sum_fiberwise_of_maps_to (g := Pt.stratumOf) (t := Finset.range Pt.strata.length)
      (fun v hv => Finset.mem_range.2 (mem_blocksIn (hU hv)).2)]
    refine Finset.sum_congr rfl fun t _ => ?_
    rw [inter_cell hU, Finset.sum_congr rfl (fun v hv => by rw [(Finset.mem_filter.1 hv).2]), Finset.sum_const,
      nsmul_eq_mul, mul_comm]
  rw [hy, ← Finset.sum_neg_distrib, Real.exp_sum]
  refine Finset.prod_le_prod₀ (fun t _ => escCell_nonneg _ _ _) fun t ht => ?_
  have ht' := Finset.mem_range.1 ht
  exact escCell_le_rate (Finset.card_le_card Finset.inter_subset_right) (covers_get hc ht').1
    (card_cell_le_size hc ht')

theorem esc_whole (hc : Properties.Spot.Covers s Pt d ρ cs) {U : Finset ℕ}
    (hU : U ⊆ blocksIn Pt d.part ρ) {v : ℕ} (hv : v ∈ U) (hr : (cs.getD (Pt.stratumOf v) default).rate = none) :
    d.esc Pt (counts Pt d.part U) ρ = 0 := by
  unfold Design.Two.esc
  obtain ⟨hcell, ht⟩ := mem_blocksIn (hU hv)
  refine Finset.prod_eq_zero (Finset.mem_range.2 ht) ?_
  exact escCell_rate_none (Finset.card_le_card Finset.inter_subset_right)
    (Finset.card_pos.2 ⟨v, Finset.mem_inter.2 ⟨hv, hcell⟩⟩) (covers_get hc ht).1 (card_cell_le_size hc ht) hr

theorem passes_le_phi (hc : Properties.Spot.Covers s Pt d ρ cs) {q : ℝ} (hq0 : 0 ≤ q) {U : Finset ℕ}
    (hU : U ⊆ blocksIn Pt d.part ρ) : d.passes Pt q (counts Pt d.part U) ρ ≤ phi q (yOf Pt cs U) := by
  unfold Design.Two.passes phi
  nlinarith [esc_le_exp hc hU]

theorem passes_nonneg {q : ℝ} (hq1 : q ≤ 1) (fp : ℕ → ℕ → ℕ) : 0 ≤ d.passes Pt q fp ρ := by
  unfold Design.Two.passes
  have := esc_nonneg (d := d) (Pt := Pt) (ρ := ρ) fp
  have he : d.esc Pt fp ρ ≤ 1 := Finset.prod_le_one₀ (fun _ _ => escCell_nonneg _ _ _)
    fun _ _ => escCell_le_one _ _ _
  have hq : q * (1 - d.esc Pt fp ρ) ≤ 1 := by
    rcases le_total 0 q with h | h
    · nlinarith
    · nlinarith
  linarith

theorem passes_whole (hc : Properties.Spot.Covers s Pt d ρ cs) (q : ℝ) {U : Finset ℕ}
    (hU : U ⊆ blocksIn Pt d.part ρ) {v : ℕ} (hv : v ∈ U) (hr : (cs.getD (Pt.stratumOf v) default).rate = none) :
    d.passes Pt q (counts Pt d.part U) ρ = 1 - q := by
  unfold Design.Two.passes; rw [esc_whole hc hU hv hr]; ring

theorem passes_empty (q : ℝ) : d.passes Pt q (counts Pt d.part ∅) ρ = 1 := by
  unfold Design.Two.passes Design.Two.esc counts
  simp [escCell_zero]

/-! ## The checker's groups of a replay block -/

/-- The group of key `k` (`readGroups`). -/
def grp (cs : List Cell) (reads : List (List (ℕ × ℕ))) (S p : ℕ) (k : ℕ × ℕ) : Group :=
  let n := countIn (reads.getD k.1 []) k.2
  ⟨k.1, n, k.2, (cs.getD k.1 default).rate, lnUpper n p, lnLower n p, lnUpper ((S / k.2 : ℕ) : ℚ) p,
    lnFactLower (S / k.2) p⟩

theorem readGroups_eq (cs : List Cell) (reads : List (List (ℕ × ℕ))) (S p : ℕ) :
    readGroups cs reads S p = (keysOf cs reads).map (grp cs reads S p) := rfl

theorem grp_ok {reads : List (List (ℕ × ℕ))} {S p : ℕ} {k : ℕ × ℕ} (hk : k ∈ keysOf cs reads) :
    groupOk (grp cs reads S p k) S := by
  have hn := (mem_keysOf.1 hk).2
  have hnR : (0 : ℝ) < (countIn (reads.getD k.1 []) k.2 : ℕ) := by exact_mod_cast hn
  refine ⟨hn, ?_, ?_, ?_, lnFactLower_le _ p⟩
  · have := log_le_lnUpper (y := ((countIn (reads.getD k.1 []) k.2 : ℕ) : ℚ)) (by exact_mod_cast hnR) p
    rw [Rat.cast_natCast] at this; exact this
  · have := lnLower_le_log (y := ((countIn (reads.getD k.1 []) k.2 : ℕ) : ℚ)) (by exact_mod_cast hnR) p
    rw [Rat.cast_natCast] at this; exact this
  · intro hm
    have := log_le_lnUpper (y := ((S / k.2 : ℕ) : ℚ)) (by exact_mod_cast hm) p
    rw [Rat.cast_natCast] at this; exact this

/-- The keys whose cells the bound samples at a rate, and those it samples whole. -/
def finKeys (cs : List Cell) (reads : List (List (ℕ × ℕ))) : List (ℕ × ℕ) :=
  (keysOf cs reads).filter fun k => (cs.getD k.1 default).rate.isSome

def infKeys (cs : List Cell) (reads : List (List (ℕ × ℕ))) : List (ℕ × ℕ) :=
  (keysOf cs reads).filter fun k => (cs.getD k.1 default).rate.isNone

theorem split_eq (cs : List Cell) (reads : List (List (ℕ × ℕ))) (S p : ℕ) :
    split (readGroups cs reads S p) =
      ((finKeys cs reads).map (grp cs reads S p), (infKeys cs reads).map (grp cs reads S p)) := by
  simp only [split, readGroups_eq, List.filter_map]
  rfl

theorem key_mem_fin {reads : List (List (ℕ × ℕ))} {k : ℕ × ℕ} (hk : k ∈ keysOf cs reads)
    (hr : (cs.getD k.1 default).rate.isSome) : k ∈ finKeys cs reads := List.mem_filter.2 ⟨hk, hr⟩

theorem key_mem_inf {reads : List (List (ℕ × ℕ))} {k : ℕ × ℕ} (hk : k ∈ keysOf cs reads)
    (hr : (cs.getD k.1 default).rate = none) : k ∈ infKeys cs reads :=
  List.mem_filter.2 ⟨hk, Option.isNone_iff_eq_none.2 hr⟩

theorem cntOf_keyOf (D : Finset ℕ) (k : ℕ × ℕ) :
    cntOf (keyOf s Pt) D k = ((D.filter fun v => Pt.stratumOf v = k.1).filter fun v => wid s Pt v = k.2).card := by
  unfold cntOf
  rw [Finset.filter_filter]
  congr 1
  ext v
  simp [keyOf, Prod.ext_iff]

/-- The sets' groups' counts: every proof block of `D ⊆ ρ` has a key of `cs`'s, and each key's count is at most its
group's. -/
theorem covers_keys (hc : Properties.Spot.Covers s Pt d ρ cs) {D : Finset ℕ}
    (hD : D ⊆ blocksIn Pt d.part ρ) :
    (∀ v ∈ D, keyOf s Pt v ∈ keysOf cs (cs.map Cell.parts)) ∧
      ∀ k, cntOf (keyOf s Pt) D k ≤ countIn ((cs.map Cell.parts).getD k.1 []) k.2 := by
  have hparts : ∀ t, t < Pt.strata.length → ∀ b,
      countIn ((cs.map Cell.parts).getD t []) b = (cs.getD t default).count b := by
    intro t ht b
    have ht' : t < cs.length := hc.1 ▸ ht
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem ht', Cell.count]
  have hsub : ∀ t b, ((D.filter fun v => Pt.stratumOf v = t).filter fun v => wid s Pt v = b) ⊆
      ((cell Pt d.part ρ t).filter fun v => wid s Pt v = b) := by
    intro t b v hv
    simp only [Finset.mem_filter] at hv ⊢
    obtain ⟨⟨hvD, rfl⟩, hb⟩ := hv
    exact ⟨(mem_blocksIn (hD hvD)).1, hb⟩
  refine ⟨fun v hv => ?_, fun k => ?_⟩
  · obtain ⟨hcell, ht⟩ := mem_blocksIn (hD hv)
    refine mem_keysOf.2 ⟨hc.1 ▸ ht, ?_⟩
    show 0 < countIn ((cs.map Cell.parts).getD (Pt.stratumOf v) []) (wid s Pt v)
    rw [hparts _ ht]
    refine lt_of_lt_of_le ?_ ((covers_get hc ht).2 (wid s Pt v))
    exact Finset.card_pos.2 ⟨v, Finset.mem_filter.2 ⟨hcell, rfl⟩⟩
  · rw [cntOf_keyOf]
    by_cases ht : k.1 < Pt.strata.length
    · rw [hparts _ ht]
      exact (Finset.card_le_card (hsub _ _)).trans ((covers_get hc ht).2 k.2)
    · have : (D.filter fun v => Pt.stratumOf v = k.1) = ∅ := by
        rw [Finset.filter_eq_empty_iff]
        intro v hv h
        exact ht (h ▸ (mem_blocksIn (hD hv)).2)
      simp [this]

end Block

end

section

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

theorem LogUB.oadd {a b : Option ℚ} {X Y : ℝ} (ha : LogUB a X) (hb : LogUB b Y) (hY : 0 ≤ Y) :
    LogUB (oadd a b) (X * Y) := by
  cases a with
  | none => show X * Y = 0; rw [show X = 0 from ha, zero_mul]
  | some a =>
    cases b with
    | none => show X * Y = 0; rw [show Y = 0 from hb, mul_zero]
    | some b =>
      show X * Y ≤ Real.exp ((a + b : ℚ) : ℝ)
      push_cast; rw [Real.exp_add]
      exact mul_le_mul ha hb hY (Real.exp_pos _).le

theorem lnPassWhole_sound {q θ : ℚ} (hq1 : (q : ℝ) ≤ 1) (hθ : (0 : ℝ) ≤ θ) (p : ℕ) :
    LogUB (lnPassWhole q θ p) ((1 - (q : ℝ)) ^ (θ : ℝ)) := by
  unfold lnPassWhole
  split_ifs with h0 h1
  · subst h0; show _ ≤ Real.exp ((0 : ℚ) : ℝ); simp
  · show (1 - (q : ℝ)) ^ (θ : ℝ) ≤ Real.exp ((θ * lnUpper (1 - q) p : ℚ) : ℝ)
    have hq : (q : ℝ) < 1 := by exact_mod_cast h1
    have hpos : (0 : ℝ) < 1 - q := by linarith
    rw [Real.rpow_def_of_pos hpos]
    push_cast
    apply Real.exp_le_exp.2
    have := log_le_lnUpper (y := 1 - q) (by push_cast; linarith) p
    push_cast at this
    nlinarith
  · have hq : (q : ℝ) = 1 := le_antisymm hq1 (by exact_mod_cast not_lt.1 h1)
    show (1 - (q : ℝ)) ^ (θ : ℝ) = 0
    rw [hq, sub_self, Real.zero_rpow (by exact_mod_cast h0)]

/-- `Σ_{U ⊆ D} 2^{b(U)} e^{−c y(U)} = ∏_{v∈D} (1 + 2^{b_v} e^{−c f_v})`. -/
theorem sum_bits_exp (s : AuditStatement) (Pt : Partition) (D : Finset ℕ) (c : ℝ) (f : ℕ → ℝ) :
    ∑ U ∈ D.powerset, (2 : ℝ) ^ bitsOf s Pt U * Real.exp (-(c * ∑ v ∈ U, f v)) =
      ∏ v ∈ D, (1 + 2 ^ wid s Pt v * Real.exp (-(c * f v))) := by
  rw [Finset.prod_one_add]
  refine Finset.sum_congr rfl fun U _ => ?_
  rw [Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum, ← Real.exp_sum, Finset.mul_sum,
    ← Finset.sum_neg_distrib]
  rfl

theorem sum_bits (s : AuditStatement) (Pt : Partition) (D : Finset ℕ) :
    ∑ U ∈ D.powerset, (2 : ℝ) ^ bitsOf s Pt U = ∏ v ∈ D, (1 + 2 ^ wid s Pt v) := by
  have := sum_bits_exp s Pt D 0 fun _ => 0
  simpa using this

theorem cntOf_mono {G : Type*} [DecidableEq G] {κ : ℕ → G} {D D' : Finset ℕ} (h : D' ⊆ D) (k : G) :
    cntOf κ D' k ≤ cntOf κ D k :=
  Finset.card_le_card (Finset.filter_subset_filter _ h)

section Sets

variable {s : AuditStatement} {Pt : Partition} {d : Design.Two} {ρ : ℕ} {cs : List Cell}

/-- The replay block's proof blocks in cells the bound samples at a rate. -/
noncomputable def finBlocks (Pt : Partition) (d : Design.Two) (ρ : ℕ) (cs : List Cell) : Finset ℕ :=
  (blocksIn Pt d.part ρ).filter fun v => (cs.getD (Pt.stratumOf v) default).rate.isSome

/-- The replay block's proof blocks in cells the bound samples whole. -/
noncomputable def infBlocks (Pt : Partition) (d : Design.Two) (ρ : ℕ) (cs : List Cell) : Finset ℕ :=
  (blocksIn Pt d.part ρ).filter fun v => ¬ (cs.getD (Pt.stratumOf v) default).rate.isSome

theorem one_le_prod_bits (D : Finset ℕ) (c : ℕ → ℝ) :
    1 ≤ ∏ w ∈ D, (1 + 2 ^ wid s Pt w * Real.exp (c w)) :=
  Finset.one_le_prod₀ fun w _ => by
    have := mul_nonneg (pow_nonneg (zero_le_two (α := ℝ)) (wid s Pt w)) (Real.exp_pos (c w)).le
    linarith

theorem one_le_prod_bits' (D : Finset ℕ) : 1 ≤ ∏ w ∈ D, (1 + (2 : ℝ) ^ wid s Pt w) :=
  Finset.one_le_prod₀ fun w _ => by have := pow_nonneg (zero_le_two (α := ℝ)) (wid s Pt w); linarith

/-- **Rankin's factor, split**: the empty set, the nonempty sets in cells sampled at a rate, and the sets meeting a cell
sampled whole, each passing with probability `1 − q`. -/
theorem rankinFactor_le (hc : Properties.Spot.Covers s Pt d ρ cs) (q θ : ℝ) :
    rankinFactor s Pt d q θ ρ ≤
      1 + ∑ U ∈ (finBlocks Pt d ρ cs).powerset.erase ∅,
          (2 : ℝ) ^ bitsOf s Pt U * d.passes Pt q (counts Pt d.part U) ρ ^ θ +
        (1 - q) ^ θ * ((∏ w ∈ finBlocks Pt d ρ cs, (1 + 2 ^ wid s Pt w)) *
          ((∏ w ∈ infBlocks Pt d ρ cs, (1 + 2 ^ wid s Pt w)) - 1)) := by
  set B := blocksIn Pt d.part ρ
  set Bf := finBlocks Pt d ρ cs
  set Bi := infBlocks Pt d ρ cs
  set T : Finset ℕ → ℝ := fun U => (2 : ℝ) ^ bitsOf s Pt U * d.passes Pt q (counts Pt d.part U) ρ ^ θ
  have hBf : Bf ⊆ B := Finset.filter_subset _ _
  have hpowB : B.powerset.filter (· ⊆ Bf) = Bf.powerset := by
    ext U; simp only [Finset.mem_filter, Finset.mem_powerset]
    exact ⟨fun h => h.2, fun h => ⟨h.trans hBf, h⟩⟩
  have hT0 : T ∅ = 1 := by simp [T, bitsOf, passes_empty]
  have hrw : ∀ U ∈ B.powerset.filter (fun U => ¬ U ⊆ Bf), T U = (1 - q) ^ θ * (2 : ℝ) ^ bitsOf s Pt U := by
    intro U hU
    obtain ⟨hUB, hUf⟩ := Finset.mem_filter.1 hU
    rw [Finset.mem_powerset] at hUB
    obtain ⟨w, hwU, hwf⟩ := Finset.not_subset.1 hUf
    have hr : (cs.getD (Pt.stratumOf w) default).rate = none := by
      by_contra hne
      exact hwf (Finset.mem_filter.2 ⟨hUB hwU, Option.isSome_iff_ne_none.2 hne⟩)
    simp only [T]
    rw [passes_whole hc q hUB hwU hr, mul_comm]
  have hall := sum_bits s Pt B
  rw [← Finset.sum_filter_add_sum_filter_not B.powerset (· ⊆ Bf), hpowB, sum_bits,
    ← Finset.prod_filter_mul_prod_filter_not B fun v => (cs.getD (Pt.stratumOf v) default).rate.isSome] at hall
  have hnot : ∑ U ∈ B.powerset.filter (fun U => ¬ U ⊆ Bf), T U =
      (1 - q) ^ θ * ((∏ w ∈ Bf, (1 + 2 ^ wid s Pt w)) * ((∏ w ∈ Bi, (1 + 2 ^ wid s Pt w)) - 1)) := by
    rw [Finset.sum_congr rfl hrw, ← Finset.mul_sum]
    congr 1
    change (∏ w ∈ Bf, (1 + (2 : ℝ) ^ wid s Pt w)) + _ =
      (∏ w ∈ Bf, (1 + (2 : ℝ) ^ wid s Pt w)) * ∏ w ∈ Bi, (1 + (2 : ℝ) ^ wid s Pt w) at hall
    rw [mul_sub, mul_one]
    linarith
  calc rankinFactor s Pt d q θ ρ = ∑ U ∈ B.powerset, T U := rfl
    _ = (T ∅ + ∑ U ∈ Bf.powerset.erase ∅, T U) + ∑ U ∈ B.powerset.filter (fun U => ¬ U ⊆ Bf), T U := by
      rw [← Finset.sum_filter_add_sum_filter_not B.powerset (· ⊆ Bf), hpowB,
        Finset.add_sum_erase _ _ (Finset.empty_mem_powerset Bf)]
    _ = _ := by rw [hT0, hnot]
    _ ≤ _ := le_rfl

/-- The keys and counts of the cells sampled at a rate, and of those sampled whole. -/
theorem fin_keys (hc : Properties.Spot.Covers s Pt d ρ cs) :
    (∀ w ∈ finBlocks Pt d ρ cs, keyOf s Pt w ∈ finKeys cs (cs.map Cell.parts)) ∧
      ∀ k, cntOf (keyOf s Pt) (finBlocks Pt d ρ cs) k ≤ countIn ((cs.map Cell.parts).getD k.1 []) k.2 := by
  have hkeys := covers_keys hc (D := blocksIn Pt d.part ρ) le_rfl
  exact ⟨fun w hw => key_mem_fin (hkeys.1 w (Finset.filter_subset _ _ hw)) (Finset.mem_filter.1 hw).2,
    fun k => (cntOf_mono (Finset.filter_subset _ _) k).trans (hkeys.2 k)⟩

theorem inf_keys (hc : Properties.Spot.Covers s Pt d ρ cs) :
    (∀ w ∈ infBlocks Pt d ρ cs, keyOf s Pt w ∈ infKeys cs (cs.map Cell.parts)) ∧
      ∀ k, cntOf (keyOf s Pt) (infBlocks Pt d ρ cs) k ≤ countIn ((cs.map Cell.parts).getD k.1 []) k.2 := by
  have hkeys := covers_keys hc (D := blocksIn Pt d.part ρ) le_rfl
  exact ⟨fun w hw => key_mem_inf (hkeys.1 w (Finset.filter_subset _ _ hw))
      (Option.not_isSome_iff_eq_none.1 (Finset.mem_filter.1 hw).2),
    fun k => (cntOf_mono (Finset.filter_subset _ _) k).trans (hkeys.2 k)⟩

/-- **A chord's sum**: at slope `σ`, the sets of the cells sampled at a rate, `Σ_U 2^{b(U)} e^{−θ σ y(U)}`. -/
theorem light_sound (hc : Properties.Spot.Covers s Pt d ρ cs) (θ σ : ℚ) (p : ℕ) :
    LogUB (lightSets (Ln2.of p) (((finKeys cs (cs.map Cell.parts)).map (grp cs (cs.map Cell.parts) 0 p)).map
        fun g => (g, -(θ * σ * g.r))) p)
      (∑ U ∈ (finBlocks Pt d ρ cs).powerset.erase ∅,
        (2 : ℝ) ^ bitsOf s Pt U * Real.exp (-((θ : ℝ) * σ * yOf Pt cs U))) := by
  set reads := cs.map Cell.parts
  set Kf := finKeys cs reads
  set G := grp cs reads 0 p
  set Bf := finBlocks Pt d ρ cs
  set c : ℕ → ℝ := fun w => -((θ : ℝ) * σ * (rateOf (cs.getD (Pt.stratumOf w) default) : ℝ))
  have hsum : ∑ U ∈ Bf.powerset.erase ∅, (2 : ℝ) ^ bitsOf s Pt U * Real.exp (-((θ : ℝ) * σ * yOf Pt cs U)) =
      ∏ w ∈ Bf, (1 + 2 ^ wid s Pt w * Real.exp (c w)) - 1 := by
    have := sum_bits_exp s Pt Bf ((θ : ℝ) * σ) fun w => (rateOf (cs.getD (Pt.stratumOf w) default) : ℝ)
    rw [← Finset.add_sum_erase _ _ (Finset.empty_mem_powerset Bf)] at this
    simp only [bitsOf, Finset.sum_empty, pow_zero, mul_zero, neg_zero, Real.exp_zero, one_mul] at this
    unfold yOf
    simp only [bitsOf, c]
    linarith
  rw [hsum]
  have hls := lightSets_sound (ln2Ok_of p) ((Kf.map G).map fun g => (g, -(θ * σ * g.r))) (S := 0)
    (fun x hx => by
      simp only [List.map_map, List.mem_map, Function.comp] at hx
      obtain ⟨k, hk, rfl⟩ := hx
      exact grp_ok (List.mem_filter.1 hk).1) p
  refine hls.mono (by linarith [one_le_prod_bits (s := s) (Pt := Pt) Bf c]) ?_
  have := prod_le_keys Kf ((keysOf_nodup cs reads).filter _) Bf (keyOf s Pt) (fin_keys hc).1 (fun k => (G k).n)
    (fun k _ => (fin_keys hc).2 k)
    (fun k => 1 + 2 ^ k.2 * Real.exp (-((θ : ℝ) * σ * (rateOf (cs.getD k.1 default) : ℝ))))
    (fun k => by
      have := mul_nonneg (pow_nonneg (zero_le_two (α := ℝ)) k.2)
        (Real.exp_pos (-((θ : ℝ) * σ * (rateOf (cs.getD k.1 default) : ℝ)))).le
      linarith)
  refine sub_le_sub_right (this.trans (le_of_eq ?_)) 1
  simp only [List.map_map, Function.comp_def]
  congr 1
  refine List.map_congr_left fun k _ => ?_
  simp only [G, grp, Group.r, rateOf]
  push_cast
  ring_nf

/-- **The sets meeting a cell sampled whole**: `(1 − q)^θ ∏_F (1 + 2^b)^n (∏_I (1 + 2^b)^n − 1)`. -/
theorem whole_sound (hc : Properties.Spot.Covers s Pt d ρ cs) {q θ : ℚ} (hq1 : (q : ℝ) ≤ 1)
    (hθ : (0 : ℝ) ≤ θ) (p : ℕ) :
    LogUB (if ((infKeys cs (cs.map Cell.parts)).map (grp cs (cs.map Cell.parts) 0 p)).isEmpty then none
      else oadd (lnPassWhole q θ p)
        ((Fac.prod (((infKeys cs (cs.map Cell.parts)).map (grp cs (cs.map Cell.parts) 0 p)).map
            (wholeFac (Ln2.of p) · 0 p)) p).l.map
          ((Fac.prod (((finKeys cs (cs.map Cell.parts)).map (grp cs (cs.map Cell.parts) 0 p)).map
            (wholeFac (Ln2.of p) · 0 p)) p).f + ·)))
      ((1 - (q : ℝ)) ^ (θ : ℝ) * ((∏ w ∈ finBlocks Pt d ρ cs, (1 + 2 ^ wid s Pt w)) *
          ((∏ w ∈ infBlocks Pt d ρ cs, (1 + 2 ^ wid s Pt w)) - 1))) := by
  set reads := cs.map Cell.parts
  set Kf := finKeys cs reads
  set Ki := infKeys cs reads
  set G := grp cs reads 0 p
  set L := Ln2.of p
  set Bf := finBlocks Pt d ρ cs
  set Bi := infBlocks Pt d ρ cs
  have hL : ln2Ok L := ln2Ok_of p
  have hpq : (0 : ℝ) ≤ (1 - (q : ℝ)) ^ (θ : ℝ) := Real.rpow_nonneg (by linarith) _
  have hPi1 := one_le_prod_bits' (s := s) (Pt := Pt) Bi
  have hPf1 := one_le_prod_bits' (s := s) (Pt := Pt) Bf
  by_cases hKe : Ki = []
  · have hBie : Bi = ∅ := Finset.eq_empty_of_forall_notMem fun w hw => by
      have := (inf_keys hc).1 w hw
      rw [show infKeys cs (cs.map Cell.parts) = Ki from rfl, hKe] at this
      simp at this
    have : (Ki.map G).isEmpty = true := by rw [hKe]; rfl
    rw [if_pos this]
    show _ = (0 : ℝ)
    rw [hBie, Finset.prod_empty, sub_self, mul_zero, mul_zero]
  have hne : (Ki.map G).isEmpty = false := by
    cases hK : Ki with
    | nil => exact absurd hK hKe
    | cons _ _ => rfl
  rw [if_neg (by rw [hne]; simp)]
  have hPf := prod_map_bounds' (Kf.map G) (wholeFac L · 0 p)
    (fun g => (1 + 2 ^ g.b * Real.exp ((0 : ℚ) : ℝ)) ^ g.n)
    (fun g hg => by
      obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hg
      exact wholeFac_bounds hL (grp_ok (List.mem_filter.1 hk).1) 0 p) p
  have hPi := prod_map_bounds' (Ki.map G) (wholeFac L · 0 p)
    (fun g => (1 + 2 ^ g.b * Real.exp ((0 : ℚ) : ℝ)) ^ g.n)
    (fun g hg => by
      obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hg
      exact wholeFac_bounds hL (grp_ok (List.mem_filter.1 hk).1) 0 p) p
  set Pf := ((Kf.map G).map fun g => (1 + 2 ^ g.b * Real.exp ((0 : ℚ) : ℝ)) ^ g.n).prod
  set Pi := ((Ki.map G).map fun g => (1 + 2 ^ g.b * Real.exp ((0 : ℚ) : ℝ)) ^ g.n).prod
  have hPfe : Pf ≤ Real.exp (Fac.prod ((Kf.map G).map (wholeFac L · 0 p)) p).f := by
    have := le_exp_of_bounds hPf; linarith
  have hb := LogUB.scale hPi.2.2 hPi.1 (c := (Fac.prod ((Kf.map G).map (wholeFac L · 0 p)) p).f) hPfe
  have hPf0 : 0 ≤ Pf := by linarith [hPf.1]
  have hwh := LogUB.oadd (lnPassWhole_sound hq1 hθ p) hb (mul_nonneg hPf0 hPi.1)
  refine hwh.mono (mul_nonneg hpq (mul_nonneg (by linarith) (by linarith))) ?_
  refine mul_le_mul_of_nonneg_left ?_ hpq
  have hkeysf := fin_keys hc
  have hkeysi := inf_keys hc
  have hf := prod_le_keys Kf ((keysOf_nodup cs reads).filter _) Bf (keyOf s Pt) hkeysf.1 (fun k => (G k).n)
    (fun k _ => hkeysf.2 k) (fun k => 1 + 2 ^ k.2) (fun k => by linarith [pow_nonneg (zero_le_two (α := ℝ)) k.2])
  have hi := prod_le_keys Ki ((keysOf_nodup cs reads).filter _) Bi (keyOf s Pt) hkeysi.1 (fun k => (G k).n)
    (fun k _ => hkeysi.2 k) (fun k => 1 + 2 ^ k.2) (fun k => by linarith [pow_nonneg (zero_le_two (α := ℝ)) k.2])
  have hPfeq : (Kf.map fun k => (1 + (2 : ℝ) ^ k.2) ^ (G k).n).prod = Pf := by
    simp only [Pf, List.map_map, Function.comp_def, Rat.cast_zero, Real.exp_zero, mul_one]; rfl
  have hPieq : (Ki.map fun k => (1 + (2 : ℝ) ^ k.2) ^ (G k).n).prod = Pi := by
    simp only [Pi, List.map_map, Function.comp_def, Rat.cast_zero, Real.exp_zero, mul_one]; rfl
  rw [hPfeq] at hf
  rw [hPieq] at hi
  exact mul_le_mul hf (sub_le_sub_right hi 1) (by linarith) hPf0

end Sets

/-- **The bound is above Rankin's factor.** -/
theorem factorUpper_sound : Properties.Spot.FactorUpperSound := by
  intro cs P v hv s Pt d ρ hc
  by_cases hP : P.valid = true
  swap
  · simp [lnFactorUpper, hP] at hv
  obtain ⟨hq0, hq1, hθ, -⟩ := Params.valid_parts hP
  unfold lnFactorUpper at hv
  rw [if_pos hP] at hv
  simp only [groupsOf, split_eq] at hv
  refine (rankinFactor_le hc _ _).trans ?_
  refine bound_sound hP hv ((finBlocks Pt d ρ cs).powerset.erase ∅) (fun U => (2 : ℝ) ^ bitsOf s Pt U) (yOf Pt cs)
    (fun U => d.passes Pt (P.q : ℝ) (counts Pt d.part U) ρ ^ (P.θ : ℝ)) (fun _ _ => by positivity)
    (fun U _ => yOf_nonneg U) (fun U hU => ?_) (fun σ => light_sound hc P.θ σ P.p) (whole_sound hc hq1 hθ P.p)
  have hUB : U ⊆ blocksIn Pt d.part ρ :=
    (Finset.mem_powerset.1 (Finset.mem_of_mem_erase hU)).trans (Finset.filter_subset _ _)
  exact Real.rpow_le_rpow (passes_nonneg hq1 _) (passes_le_phi hc hq0 hUB) hθ

end

end Properties.Spot.SecurityProofs.FactorUpperProof
