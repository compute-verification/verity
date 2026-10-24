import Models.Accounting.Guarantees.RemoteDraw
import Proofs.Accounting.RemoteHeld.OverTime
import Proofs.Accounting.RateAudit.Horizon
import Proofs.Accounting.Prob
import Proofs.Accounting.Actual.WipeDraw

/-!
# The openings' pooled bound: the proof

Window `t`'s size `U_t` and correct receipts `C_t` are fixed before its openings coin (`Live`), so given the other coins
and the earlier windows' coins the window's draw is `Key.subset` on the first blocks `prf t` gives a uniform coin.

* **On uniform first blocks** (`Draw.pr_test_le`). Step `i` of the partial Fisher–Yates reads its first word, uniform
  (`Draw.pr_word_le`: a first word is the top bits of a uniform block). The sampler rejects it with probability below
  `2^-64` (`Draw.pr_reject_le`, `2^w mod N < N ≤ 2^(w − 64)`), so some step is rejected with probability at most
  `s'_t/2^64` (`Draw.pr_not_accepted_le`). An accepted word gives each of the `U_t − i` offsets at most a `1/(U_t − i)`
  share (`Draw.pr_hit_le`, the wipe's `card_fiber_le`). Once the earlier steps took receipts of `C_t`, `g_t − i` of the
  positions left hold receipts of `C_t` (`Draw.card_arrange_mem`), so step `i` takes one with probability at most
  `(g_t − i)/(U_t − i) ≤ g_t/U_t`, and the draw lands in `C_t` with probability at most `(g_t/U_t)^(s'_t)`, `0` when
  `g_t < s'_t` (`Draw.pr_window_le`, the steps as rounds of `Draw.pr_rounds_le`).
* **On the coin** (`Draw.pooledDraw`). An accepted draw ignores the fallbacks (`Draw.openings_congr`), so the draw on
  `prf t` landing in `C_t` is the window test `test U_t s'_t C_t` on `prf t`, which outputs true with probability at
  most its probability on uniform blocks plus `advantage (prf t)` (`Actual.pr_le_add_prfAdvantage`).
* **Over the session.** Such per-round bounds, each read off the rounds before, multiply: every round's event holds
  while their product is at most `c` with probability at most `c` (`Draw.pr_live_le`, peeling round 0 as
  `Rate.Horizon.pr_causal_le` does), and with an additive error a round, at most `c` plus the errors' sum
  (`Draw.pr_rounds_add_le`, `Draw.prod_min_le`). When the wrong receipts pass the pooled bound, the wrong shares
  `f_t = 1 − g_t/U_t` are no point of its knapsack, so a window has fewer correct receipts than draws, or
  `Σ_t s'_t·f_t` passes the budget `S·(1 − δ^(1/S))` and weighted AM–GM puts the product at most `δ`
  (`Draw.prod_le_of_pooled_lt`). `PooledDraw` is the two at `c = δ`, the errors `s_t/2^64 + advantage (prf t)`;
  `LiveCoinsDrawn` and the memory line at the record's pooled bound (`HeldDrawn` and its time-held forms) follow from
  it as `Held` does from `LiveCoins`.
-/

namespace Accounting.SecurityProofs.Remote

open Finset Accounting.Rate Accounting.Rate.Horizon
open Pous (pr)
open scoped ENNReal
open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions

namespace Draw

/-! ## Adaptive rounds -/

/-- Rounds `0, …, n − 1`, each with coins `Y i` and then `Z i`. Round `i`'s event `A i` reads the coins of rounds up to
`i`, and `h i`, read off the earlier rounds and round `i`'s first coins, bounds the probability of `A i` over its second
coins. Then every event holds while the bounds multiply to at most `c` with probability at most `c`. -/
theorem pr_live_le : ∀ (n : ℕ) (Y Z : Fin n → Type) [∀ i, Fintype (Y i)] [∀ i, Fintype (Z i)]
    (A : (i : Fin n) → ((j : Fin n) → Y j × Z j) → Prop) (h : (i : Fin n) → ((j : Fin n) → Y j × Z j) → ℝ≥0∞),
    (∀ i ω ω', (∀ j, j ≤ i → ω j = ω' j) → (A i ω ↔ A i ω')) →
    (∀ i ω ω', (∀ j, j < i → ω j = ω' j) → (ω i).1 = (ω' i).1 → h i ω = h i ω') →
    (∀ i ω, pr (fun z : Z i => A i (Function.update ω i ((ω i).1, z))) ≤ h i ω) →
    ∀ c, pr (fun ω => (∀ i, A i ω) ∧ ∏ i, h i ω ≤ c) ≤ c := by
  intro n
  induction n with
  | zero =>
    intro Y Z _ _ A h _ _ _ c
    by_cases hc : 1 ≤ c
    · exact (pr_le_one _).trans hc
    · refine (Bandwidth.pr_mono (fun ω hω => (hc (by simpa using hω.2) : False))).trans ?_
      rw [pr_false]
      exact zero_le
  | succ n ih =>
    intro Y Z _ _ A h hA hh hb c
    classical
    let T : Type := (i : Fin n) → Y i.succ × Z i.succ
    let e : ((j : Fin (n + 1)) → Y j × Z j) ≃ (Z 0 × T) × Y 0 :=
      { toFun := fun ω => (((ω 0).2, Fin.tail ω), (ω 0).1)
        invFun := fun q => Fin.cons (q.2, q.1.1) q.1.2
        left_inv := fun ω => by simp
        right_inv := fun q => by simp }
    rw [pr_equiv e]
    refine pr_le_of_fiber _ _ (fun y₀ => ?_)
    simp only [e, Equiv.coe_fn_symm_mk]
    rcases isEmpty_or_nonempty (Z 0 × T) with hE | ⟨⟨z₀, t₀⟩⟩
    · exact (le_of_eq (Bandwidth.pr_eq_zero_of_card _ Fintype.card_eq_zero)).trans zero_le
    set H := h 0 (Fin.cons (y₀, z₀) t₀ : (j : Fin (n + 1)) → Y j × Z j) with hHdef
    have hH : ∀ z t, h 0 (Fin.cons (y₀, z) t : (j : Fin (n + 1)) → Y j × Z j) = H :=
      fun z t => hh 0 _ _ (fun j hj => absurd hj (Fin.not_lt_zero j)) rfl
    let A₀ : Z 0 → Prop := fun z => A 0 (Fin.cons (y₀, z) t₀ : (j : Fin (n + 1)) → Y j × Z j)
    have hA₀ : ∀ z t, A 0 (Fin.cons (y₀, z) t : (j : Fin (n + 1)) → Y j × Z j) ↔ A₀ z := fun z t =>
      hA 0 _ _ (fun j hj => by
        obtain rfl : j = 0 := Fin.le_zero_iff.1 hj
        rfl)
    have hpA : pr A₀ ≤ H := by
      have := hb 0 (Fin.cons (y₀, z₀) t₀)
      simpa [Fin.update_cons_zero, A₀] using this
    let A' : Z 0 → (i : Fin n) → T → Prop := fun z i t => A i.succ (Fin.cons (y₀, z) t)
    let h' : Z 0 → (i : Fin n) → T → ℝ≥0∞ := fun z i t => h i.succ (Fin.cons (y₀, z) t)
    have hrest : ∀ z c', pr (fun t : T => (∀ i, A' z i t) ∧ ∏ i, h' z i t ≤ c') ≤ c' := by
      intro z
      refine ih (fun i => Y i.succ) (fun i => Z i.succ) (A' z) (h' z) ?_ ?_ ?_
      · intro i ω ω' hω
        refine hA i.succ _ _ (fun j hj => ?_)
        cases j using Fin.cases with
        | zero => rfl
        | succ j => simpa using hω j (Fin.succ_le_succ_iff.1 hj)
      · intro i ω ω' hω h1
        refine hh i.succ _ _ (fun j hj => ?_) (by simpa using h1)
        cases j using Fin.cases with
        | zero => rfl
        | succ j => simpa using hω j (Fin.succ_lt_succ_iff.1 hj)
      · intro i ω
        have := hb i.succ (Fin.cons (y₀, z) ω)
        simpa [A', h', Fin.cons_update] using this
    let B : Z 0 × T → Prop := fun p => (∀ i, A' p.1 i p.2) ∧ (∏ i, h' p.1 i p.2) * H ≤ c
    refine (Bandwidth.pr_mono (fun p hp => (?_ : A₀ p.1 ∧ B p))).trans ?_
    · obtain ⟨hall, hprod⟩ := hp
      refine ⟨(hA₀ p.1 p.2).1 (hall 0), fun i => hall i.succ, ?_⟩
      rw [Fin.prod_univ_succ, hH, mul_comm] at hprod
      exact hprod
    by_cases hc : c = ⊤
    · rw [hc]; exact le_top
    by_cases hH0 : H = 0
    · have hA0 : pr A₀ = 0 := le_antisymm (hpA.trans hH0.le) zero_le
      refine (pr_and_le_mul A₀ B ⊤ (fun _ _ => le_top)).trans ?_
      rw [hA0, zero_mul]
      exact zero_le
    refine (pr_and_le_mul A₀ B (c / H) (fun z _ => ?_)).trans ?_
    · refine (Bandwidth.pr_mono (fun t ht => ⟨ht.1, ?_⟩)).trans (hrest z (c / H))
      exact (ENNReal.le_div_iff_mul_le (Or.inl hH0) (Or.inr hc)).2 ht.2
    calc pr A₀ * (c / H) ≤ H * (c / H) := by gcongr
      _ ≤ c := ENNReal.mul_div_le

/-! ## One coordinate -/

/-- An event that reads one coordinate of a product has at most that coordinate's probability. -/
theorem pr_coord_le {ι : Type} [Fintype ι] [DecidableEq ι] {β : ι → Type} [∀ i, Fintype (β i)] (i : ι)
    (P : β i → Prop) : pr (fun z : (j : ι) → β j => P (z i)) ≤ pr P := by
  rw [pr_equiv (Equiv.piSplitAt i β)]
  refine pr_le_of_fiber _ _ (fun y => ?_)
  simp [Equiv.piSplitAt_symm_apply]

/-! ## The first word -/

theorem width_le {N : ℕ} (h : N < 2 ^ 448) : width N ≤ 512 := by
  have : Nat.size N ≤ 448 := Nat.size_le.2 h
  unfold width
  omega

theorem mul_le_width (N : ℕ) : N * 2 ^ 64 ≤ 2 ^ width N := by
  have h1 : N < 2 ^ Nat.size N := Nat.lt_size_self N
  have h2 : Nat.size N + 64 ≤ width N := by unfold width; omega
  calc N * 2 ^ 64 ≤ 2 ^ Nat.size N * 2 ^ 64 := Nat.mul_le_mul_right _ h1.le
    _ = 2 ^ (Nat.size N + 64) := by rw [pow_add]
    _ ≤ 2 ^ width N := Nat.pow_le_pow_right (by norm_num) h2

theorem bitsVal_word (N : ℕ) {L : ℕ} (b : Pous.Bits L) :
    Pous.bitsVal (word N b) = Pous.bitsVal b / 2 ^ (L - width N) := by
  simp [word, Pous.bitsVal]

open Classical in
/-- **A first word has at most `2^(512 − w)` blocks.** -/
theorem card_word_fiber_le (N : ℕ) {L : ℕ} (u : Pous.Bits (width N)) :
    (univ.filter fun z : Pous.Bits L => word N z = u).card ≤ 2 ^ (L - width N) := by
  have hinj : Set.InjOn (fun z : Pous.Bits L => Pous.bitsVal z % 2 ^ (L - width N))
      ↑(univ.filter fun z : Pous.Bits L => word N z = u) := by
    intro z hz z' hz' h
    simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hz hz'
    have e : Pous.bitsVal z / 2 ^ (L - width N) = Pous.bitsVal z' / 2 ^ (L - width N) := by
      rw [← bitsVal_word, ← bitsVal_word, hz, hz']
    apply Actual.bitsVal_injective L
    rw [← Nat.div_add_mod (Pous.bitsVal z) (2 ^ (L - width N)),
      ← Nat.div_add_mod (Pous.bitsVal z') (2 ^ (L - width N)), e]
    exact congrArg _ h
  calc _ ≤ (range (2 ^ (L - width N))).card :=
        card_le_card_of_injOn _ (fun z _ => mem_range.2 (Nat.mod_lt _ (by positivity))) hinj
    _ = _ := card_range _

open Classical in
/-- **The first word is uniform**: an event of the first word has at most its probability over uniform words. -/
theorem pr_word_le {N L : ℕ} (hw : width N ≤ L) (P : Pous.Bits (width N) → Prop) :
    pr (fun z : Pous.Bits L => P (word N z)) ≤ pr P := by
  have hc : (univ.filter fun z : Pous.Bits L => P (word N z)).card ≤
      (univ.filter P).card * 2 ^ (L - width N) := by
    calc _ = ∑ u ∈ univ.filter P,
          ((univ.filter fun z : Pous.Bits L => P (word N z)).filter fun z => word N z = u).card :=
          card_eq_sum_card_fiberwise (fun z hz => mem_filter.2 ⟨mem_univ _, (mem_filter.1 hz).2⟩)
      _ ≤ ∑ _u ∈ univ.filter P, 2 ^ (L - width N) := sum_le_sum fun u _ =>
          (card_le_card fun z hz => by
            simp only [mem_filter, mem_univ, true_and] at hz ⊢
            exact hz.2).trans (card_word_fiber_le N u)
      _ = _ := by rw [sum_const, smul_eq_mul]
  refine (Actual.pr_le_ofReal _ (by positivity) ?_).trans (Actual.ofReal_le_pr P)
  have h512 : ((2 : ℝ) ^ L) = 2 ^ width N * 2 ^ (L - width N) := by
    rw [← pow_add, Nat.add_sub_cancel' hw]
  simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Nat.cast_pow, Nat.cast_ofNat]
  rw [h512, div_mul_eq_mul_div, mul_div_assoc, mul_div_cancel_left₀ _ (by positivity)]
  exact_mod_cast hc

/-- **The sampler rejects a first word with probability below `2^-64`.** -/
theorem pr_reject_le {N L : ℕ} (h0 : 0 < N) (hw : width N ≤ L) :
    pr (fun z : Pous.Bits L => Actual.wordIndex N (word N z) = none) ≤ ENNReal.ofReal (1 / 2 ^ 64) := by
  classical
  refine (pr_word_le hw (fun u => Actual.wordIndex N u = none)).trans ?_
  have hc : (univ.filter fun u : Pous.Bits (width N) => Actual.wordIndex N u = none).card = 2 ^ width N % N := by
    have h1 := card_filter_add_card_filter_not (s := (univ : Finset (Pous.Bits (width N))))
      (fun u : Pous.Bits (width N) => (Actual.wordIndex N u).isSome)
    rw [Actual.card_accepted, card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin] at h1
    have h2 : (univ.filter fun u : Pous.Bits (width N) => ¬ (Actual.wordIndex N u).isSome).card =
        (univ.filter fun u : Pous.Bits (width N) => Actual.wordIndex N u = none).card := by
      congr 1
      ext u
      simp
    have h3 : 2 ^ width N % N ≤ 2 ^ width N := Nat.mod_le _ _
    omega
  have key : ((univ.filter fun u : Pous.Bits (width N) => Actual.wordIndex N u = none).card : ℝ) ≤
      1 / 2 ^ 64 * Fintype.card (Pous.Bits (width N)) := by
    rw [hc]
    simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Nat.cast_pow, Nat.cast_ofNat]
    have hm : 2 ^ width N % N < N := Nat.mod_lt _ h0
    have : ((2 ^ width N % N : ℕ) : ℝ) * 2 ^ 64 ≤ 2 ^ width N := by
      have : (2 ^ width N % N) * 2 ^ 64 ≤ 2 ^ width N := (Nat.mul_le_mul_right _ hm.le).trans (mul_le_width N)
      exact_mod_cast this
    rw [div_mul_eq_mul_div, one_mul, le_div_iff₀ (by positivity)]
    exact this
  refine Actual.pr_le_ofReal _ (by positivity) ?_
  convert key using 3
  ext u
  simp

/-- **Each offset has at most a `1/N` share**: the first word gives an offset in `P` with probability at most
`|P ∩ [0, N)| / N`. -/
theorem pr_hit_le {N L : ℕ} (hw : width N ≤ L) (P : ℕ → Prop) [DecidablePred P] :
    pr (fun z : Pous.Bits L => ∃ o : Fin N, Actual.wordIndex N (word N z) = some o ∧ P o) ≤
      ENNReal.ofReal (((range N).filter P).card / N) := by
  classical
  refine (pr_word_le hw (fun u => ∃ o : Fin N, Actual.wordIndex N u = some o ∧ P o)).trans ?_
  have hsub : (univ.filter fun u : Pous.Bits (width N) => ∃ o : Fin N, Actual.wordIndex N u = some o ∧ P o) ⊆
      (univ.filter fun o : Fin N => P o).biUnion
        (fun o => univ.filter fun u : Pous.Bits (width N) => Actual.wordIndex N u = some o) := by
    intro u hu
    simp only [mem_filter, mem_univ, true_and, mem_biUnion] at hu ⊢
    obtain ⟨o, h1, h2⟩ := hu
    exact ⟨o, h2, h1⟩
  have hP : (univ.filter fun o : Fin N => P o).card = ((range N).filter P).card := by
    rw [card_filter, card_filter, Fin.sum_univ_eq_sum_range (fun o => if P o then 1 else 0) N]
  have hc : (univ.filter fun u : Pous.Bits (width N) => ∃ o : Fin N, Actual.wordIndex N u = some o ∧ P o).card ≤
      ((range N).filter P).card * (2 ^ width N / N) := by
    refine (card_le_card hsub).trans ((card_biUnion_le).trans ?_)
    rw [← hP, ← smul_eq_mul, ← sum_const]
    exact sum_le_sum fun o _ => Actual.card_fiber_le o
  have key : ((univ.filter fun u : Pous.Bits (width N) =>
      ∃ o : Fin N, Actual.wordIndex N u = some o ∧ P o).card : ℝ) ≤
      (((range N).filter P).card / N : ℝ) * Fintype.card (Pous.Bits (width N)) := by
    simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Nat.cast_pow, Nat.cast_ofNat]
    calc (((univ.filter fun u : Pous.Bits (width N) => ∃ o : Fin N,
            Actual.wordIndex N u = some o ∧ P o).card : ℕ) : ℝ)
        ≤ (((range N).filter P).card : ℝ) * ((2 ^ width N / N : ℕ) : ℝ) := by exact_mod_cast hc
      _ ≤ (((range N).filter P).card : ℝ) * ((2 : ℝ) ^ width N / N) := by
          gcongr
          exact_mod_cast Nat.cast_div_le
      _ = (((range N).filter P).card / N : ℝ) * 2 ^ width N := by ring
  refine Actual.pr_le_ofReal _ (by positivity) ?_
  convert key using 3
  ext u
  simp

/-! ## The arrangement -/

theorem arrange_succ (j : ℕ → ℕ) (i : ℕ) : arrange j (i + 1) = arrange j i * Equiv.swap i (i + j i) := rfl

/-- Later swaps leave the earlier positions alone. -/
theorem arrange_fix (j : ℕ → ℕ) {p i : ℕ} (hp : p < i) : ∀ i', i ≤ i' → arrange j i' p = arrange j i p := by
  intro i' hi
  induction i', hi using Nat.le_induction with
  | base => rfl
  | succ i' hi ih =>
    rw [arrange_succ, Equiv.Perm.mul_apply, Equiv.swap_apply_of_ne_of_ne (by omega) (by omega), ih]

theorem arrange_self (j : ℕ → ℕ) (i : ℕ) : arrange j (i + 1) i = arrange j i (i + j i) := by
  rw [arrange_succ, Equiv.Perm.mul_apply, Equiv.swap_apply_left]

theorem arrange_congr {j j' : ℕ → ℕ} : ∀ {k : ℕ}, (∀ l, l < k → j l = j' l) → arrange j k = arrange j' k
  | 0, _ => rfl
  | k + 1, h => by
    rw [arrange_succ, arrange_succ, arrange_congr (fun l hl => h l (by omega)), h k (by omega)]

/-- Swaps inside `[0, U)` keep `[0, U)`. -/
theorem arrange_lt {j : ℕ → ℕ} {U : ℕ} : ∀ {i : ℕ}, i ≤ U → (∀ l, l < i → l + j l < U) →
    ∀ p, p < U → arrange j i p < U
  | 0, _, _, p, hp => hp
  | i + 1, hi, hj, p, hp => by
    rw [arrange_succ, Equiv.Perm.mul_apply]
    refine arrange_lt (by omega) (fun l hl => hj l (by omega)) _ ?_
    rw [Equiv.swap_apply_def]
    have := hj i (by omega)
    split_ifs <;> omega

theorem image_arrange {j : ℕ → ℕ} {U i : ℕ} (hi : i ≤ U) (hj : ∀ l, l < i → l + j l < U) :
    (range U).image (arrange j i) = range U := by
  refine eq_of_subset_of_card_le (fun q hq => ?_) ?_
  · obtain ⟨p, hp, rfl⟩ := mem_image.1 hq
    exact mem_range.2 (arrange_lt hi hj p (mem_range.1 hp))
  · rw [card_image_of_injective _ (Equiv.injective _)]

/-- The positions of `[0, U)` holding receipts of `C` number `|C ∩ [0, U)|`. -/
theorem card_arrange_mem {j : ℕ → ℕ} {U i : ℕ} (hi : i ≤ U) (hj : ∀ l, l < i → l + j l < U) (C : Finset ℕ) :
    ((range U).filter fun p => arrange j i p ∈ C).card = ((range U).filter (· ∈ C)).card := by
  calc ((range U).filter fun p => arrange j i p ∈ C).card
      = (((range U).filter fun p => arrange j i p ∈ C).image (arrange j i)).card :=
        (card_image_of_injective _ (Equiv.injective _)).symm
    _ = (((range U).image (arrange j i)).filter (· ∈ C)).card := by rw [filter_image]
    _ = _ := by rw [image_arrange hi hj]

/-! ## One window's draw on uniform first blocks -/

theorem card_split (U i : ℕ) (hi : i ≤ U) (Q : ℕ → Prop) [DecidablePred Q] :
    ((range U).filter Q).card = ((range i).filter Q).card + ((range (U - i)).filter fun o => Q (i + o)).card := by
  rw [card_filter, card_filter, card_filter, ← sum_range_add_sum_Ico _ hi, sum_Ico_eq_sum_range]

/-- **Rounds of uniform coins**: the adaptive product bound `pr_live_le` with no first coins. -/
theorem pr_rounds_le (n : ℕ) (Z : Fin n → Type) [∀ i, Fintype (Z i)]
    (A : (i : Fin n) → ((j : Fin n) → Z j) → Prop) (h : (i : Fin n) → ((j : Fin n) → Z j) → ℝ≥0∞)
    (hA : ∀ i ω ω', (∀ j, j ≤ i → ω j = ω' j) → (A i ω ↔ A i ω'))
    (hh : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → h i ω = h i ω')
    (hb : ∀ i ω, pr (fun z : Z i => A i (Function.update ω i z)) ≤ h i ω) (c : ℝ≥0∞) :
    pr (fun ω => (∀ i, A i ω) ∧ ∏ i, h i ω ≤ c) ≤ c := by
  classical
  let e : ((j : Fin n) → Unit × Z j) ≃ ((j : Fin n) → Z j) :=
    Equiv.piCongrRight fun j => Equiv.punitProd (Z j)
  have he : ∀ ω j, e ω j = (ω j).2 := fun _ _ => rfl
  rw [pr_equiv e.symm]
  simp only [Equiv.symm_symm]
  refine pr_live_le n (fun _ => Unit) Z (fun i ω => A i (e ω)) (fun i ω => h i (e ω)) ?_ ?_ ?_ c
  · intro i ω ω' hω
    exact hA i _ _ (fun j hj => by rw [he, he, hω j hj])
  · intro i ω ω' hω _
    exact hh i _ _ (fun j hj => by rw [he, he, hω j hj])
  · intro i ω
    have hu : ∀ z : Z i, e (Function.update ω i ((ω i).1, z)) = Function.update (e ω) i z := by
      intro z
      funext j
      by_cases hj : j = i
      · subst hj
        simp [he]
      · simp [he, Function.update_of_ne hj]
    simp only [hu]
    exact hb i (e ω)

theorem offsets_lt {m U k : ℕ} {b : Blocks m} (r : ℕ → ℕ) (hk : k ≤ m) (ha : accepted U k b) :
    ∀ l, l < k → l + offsets U b r l < U := by
  intro l hl
  have hlm : l < m := by omega
  have h := ha ⟨l, hlm⟩ hl
  simp only at h
  obtain ⟨o, ho⟩ := Option.isSome_iff_exists.1 h
  have hU : l < U := by
    by_contra hU
    have : o.val < U - l := o.isLt
    omega
  simp only [offsets, hlm, ↓reduceDIte, offset, ho, Option.map_some, Option.getD_some]
  have : o.val < U - l := o.isLt
  omega

theorem offsets_congr {m U k : ℕ} {b : Blocks m} (r r' : ℕ → ℕ) (hk : k ≤ m) (ha : accepted U k b) :
    ∀ l, l < k → offsets U b r l = offsets U b r' l := by
  intro l hl
  have hlm : l < m := by omega
  have h := ha ⟨l, hlm⟩ hl
  simp only at h
  obtain ⟨o, ho⟩ := Option.isSome_iff_exists.1 h
  simp only [offsets, hlm, ↓reduceDIte, offset, ho, Option.map_some, Option.getD_some]

theorem openings_congr {m U k : ℕ} {b : Blocks m} (r r' : ℕ → ℕ) (hk : k ≤ m) (ha : accepted U k b) :
    openings U k b r = openings U k b r' := by
  unfold openings
  rw [arrange_congr (offsets_congr r r' hk ha)]

open Classical in
/-- **One window's draw lands in `C`** with probability at most `(g/U)^k`, `g = |C ∩ [0, U)|`, and never when
`g < k`: step `i` takes one of the `U − i` positions left, `g − i` of which hold receipts of `C` once the earlier steps
took receipts of `C`, each with at most a `1/(U − i)` share. -/
theorem pr_window_le {m U k : ℕ} (hm : m < 2 ^ 448) (hU : U ≤ m) (hk : k ≤ U) (C : Finset ℕ) (r : ℕ → ℕ) :
    pr (fun b : Blocks m => accepted U k b ∧ openings U k b r ⊆ C) ≤
      ENNReal.ofReal (if k ≤ ((range U).filter (· ∈ C)).card
        then ((((range U).filter (· ∈ C)).card : ℝ) / U) ^ k else 0) := by
  set g := ((range U).filter (· ∈ C)).card with hg
  let σ : Blocks m → ℕ → Equiv.Perm ℕ := fun b => arrange (offsets U b r)
  let cnt : Blocks m → ℕ → ℕ := fun b i => ((range (U - i)).filter fun o => σ b i (i + o) ∈ C).card
  let A : (i : Fin m) → Blocks m → Prop := fun i b =>
    (i : ℕ) < k → (Actual.wordIndex (U - i) (word (U - i) (b i))).isSome = true ∧ σ b (i + 1) i ∈ C
  let h : (i : Fin m) → Blocks m → ℝ≥0∞ := fun i b =>
    if (i : ℕ) < k then ENNReal.ofReal ((cnt b i : ℝ) / ((U - i : ℕ) : ℝ)) else 1
  have hoff : ∀ (b b' : Blocks m) (i : ℕ), (∀ l : Fin m, (l : ℕ) < i → b l = b' l) →
      ∀ l, l < i → offsets U b r l = offsets U b' r l := by
    intro b b' i hb l hl
    unfold offsets
    split_ifs with hlm
    · rw [hb ⟨l, hlm⟩ hl]
    · rfl
  have hσ : ∀ (b b' : Blocks m) (i : ℕ), (∀ l : Fin m, (l : ℕ) < i → b l = b' l) → σ b i = σ b' i :=
    fun b b' i hb => arrange_congr (hoff b b' i hb)
  have hkm : k ≤ m := hk.trans hU
  refine (Bandwidth.pr_mono (fun b hb => (?_ : (∀ i, A i b) ∧ ∏ i, h i b ≤ _))).trans
    (pr_rounds_le m (fun _ => Pous.Bits 512) A h ?_ ?_ ?_ _)
  · obtain ⟨ha, hsub⟩ := hb
    have hA : ∀ i, A i b := by
      intro i hi
      refine ⟨ha i hi, hsub ?_⟩
      refine mem_image.2 ⟨i, mem_range.2 hi, ?_⟩
      exact arrange_fix _ (Nat.lt_succ_self _) k hi
    refine ⟨hA, ?_⟩
    have hlt := offsets_lt (b := b) r hkm ha
    have hcnt : ∀ i, i ≤ k → cnt b i + i = g := by
      intro i hi
      have h1 := card_split U i (hi.trans hk) (fun p => σ b i p ∈ C)
      rw [show ((range U).filter fun p => σ b i p ∈ C).card = g from
        card_arrange_mem (hi.trans hk) (fun l hl => hlt l (by omega)) C] at h1
      have h2 : ((range i).filter fun p => σ b i p ∈ C) = range i := by
        refine filter_true_of_mem fun p hp => ?_
        have hp' := mem_range.1 hp
        have := (hA ⟨p, by omega⟩ (by simp only; omega)).2
        simp only at this
        rwa [show σ b i p = σ b (p + 1) p from arrange_fix _ (Nat.lt_succ_self _) i (by omega)]
      rw [h2, card_range] at h1
      simp only [cnt]
      omega
    have hkg : k ≤ g := by have := hcnt k le_rfl; omega
    simp only [hkg, ↓reduceIte]
    have hh : ∀ i : Fin m, h i b = ENNReal.ofReal (if (i : ℕ) < k then ((g - i : ℕ) : ℝ) / ((U - i : ℕ) : ℝ) else 1) := by
      intro i
      simp only [h]
      split_ifs with hi
      · have := hcnt i hi.le
        rw [show cnt b i = g - i by omega]
      · simp
    simp only [hh]
    rw [← ENNReal.ofReal_prod_of_nonneg (fun i _ => by split_ifs <;> positivity)]
    refine ENNReal.ofReal_le_ofReal ?_
    rw [Fin.prod_univ_eq_prod_range (fun i => if i < k then ((g - i : ℕ) : ℝ) / ((U - i : ℕ) : ℝ) else 1) m,
      ← prod_range_mul_prod_Ico _ hkm]
    have hIco : ∏ i ∈ Ico k m, (if i < k then ((g - i : ℕ) : ℝ) / ((U - i : ℕ) : ℝ) else 1) = 1 :=
      prod_eq_one fun i hi => by simp [not_lt.2 (mem_Ico.1 hi).1]
    rw [hIco, mul_one]
    refine (prod_le_prod₀ (g := fun _ => (g : ℝ) / U) (fun i _ => by split_ifs <;> positivity)
      (fun i hi => ?_)).trans (le_of_eq (by rw [prod_const, card_range]))
    have hik := mem_range.1 hi
    simp only [hik, ↓reduceIte]
    have hgU : g ≤ U := by
      have := card_le_card (filter_subset (· ∈ C) (range U))
      rwa [card_range] at this
    rw [Nat.cast_sub (by omega), Nat.cast_sub (by omega)]
    have hU0 : (0 : ℝ) < U - i := by
      have : (i : ℝ) < U := by exact_mod_cast (show i < U by omega)
      linarith
    have hU1 : (0 : ℝ) < U := by exact_mod_cast (show 0 < U by omega)
    rw [div_le_div_iff₀ hU0 hU1]
    have : (g : ℝ) ≤ U := by exact_mod_cast hgU
    nlinarith [(Nat.cast_nonneg i : (0 : ℝ) ≤ i)]
  · intro i b b' hb
    have hbi : b i = b' i := hb i le_rfl
    have hs : σ b (i + 1) = σ b' (i + 1) := hσ b b' (i + 1) (fun l hl => hb l (by
      rw [Fin.le_def]
      omega))
    simp only [A, hbi, hs]
  · intro i b b' hb
    have hs : σ b i = σ b' i := hσ b b' i (fun l hl => hb l (by
      rw [Fin.lt_def]
      exact hl))
    simp only [h, cnt, hs]
  · intro i b
    by_cases hi : (i : ℕ) < k
    swap
    · simp only [h, hi, ↓reduceIte]
      exact pr_le_one _
    simp only [h, hi, ↓reduceIte]
    have hiU : (i : ℕ) < U := by omega
    have hs : ∀ z, σ (Function.update b i z) i = σ b i := fun z =>
      hσ _ _ i (fun l hl => Function.update_of_ne (Fin.ne_of_lt (by rw [Fin.lt_def]; exact hl)) _ _)
    refine (Bandwidth.pr_mono (fun z hz => (?_ : ∃ o : Fin (U - i),
      Actual.wordIndex (U - i) (word (U - i) z) = some o ∧ σ b i (i + o) ∈ C))).trans
      (pr_hit_le (width_le ((Nat.sub_le U i).trans_lt (hU.trans_lt hm))) (fun o => σ b i (i + o) ∈ C))
    obtain ⟨h1, h2⟩ := hz hi
    simp only [Function.update_self] at h1
    obtain ⟨o, ho⟩ := Option.isSome_iff_exists.1 h1
    refine ⟨o, ho, ?_⟩
    have e1 : σ (Function.update b i z) (i + 1) i = σ b i (i + offset (U - i) (r i) z) := by
      simp only [σ]
      rw [arrange_self]
      have := hs z
      simp only [σ] at this
      rw [this]
      simp [offsets, i.isLt]
    rw [e1] at h2
    simpa [offset, ho] using h2


open Classical in
/-- A window's draw is rejected somewhere with probability at most `k/2^64`. -/
theorem pr_not_accepted_le {m U k : ℕ} (hm : m < 2 ^ 448) (hU : U ≤ m) (hk : k ≤ U) :
    pr (fun b : Blocks m => ¬ accepted U k b) ≤ ENNReal.ofReal (k / 2 ^ 64) := by
  refine (Bandwidth.pr_mono (fun b hb => (?_ : ∃ i : Fin m,
    (i : ℕ) < k ∧ Actual.wordIndex (U - i) (word (U - i) (b i)) = none))).trans
    ((Bandwidth.pr_exists_le _).trans ?_)
  · unfold accepted at hb
    push Not at hb
    obtain ⟨i, hi, h⟩ := hb
    exact ⟨i, hi, by simpa using h⟩
  calc ∑ i : Fin m, pr (fun b : Blocks m => (i : ℕ) < k ∧ Actual.wordIndex (U - i) (word (U - i) (b i)) = none)
      ≤ ∑ i : Fin m, (if (i : ℕ) < k then ENNReal.ofReal (1 / 2 ^ 64) else 0) := sum_le_sum fun i _ => by
        split_ifs with hi
        · refine (Bandwidth.pr_mono (fun b hb => hb.2)).trans ?_
          refine (pr_coord_le (β := fun _ : Fin m => Pous.Bits 512) i
            (fun z => Actual.wordIndex (U - i) (word (U - i) z) = none)).trans ?_
          exact pr_reject_le (by omega) (width_le (by omega))
        · exact (Bandwidth.pr_mono (Q := fun _ => False) (fun b hb => absurd hb.1 hi)).trans (le_of_eq pr_false)
    _ = ((univ.filter fun i : Fin m => (i : ℕ) < k).card : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 64) := by
        rw [sum_ite, sum_const_zero, add_zero, sum_const, nsmul_eq_mul]
    _ ≤ (k : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 64) := by
        gcongr
        exact_mod_cast (card_le_card_of_injOn (fun i : Fin m => (i : ℕ))
          (fun i hi => mem_range.2 (mem_filter.1 hi).2) (fun a _ b _ h => Fin.ext h)).trans_eq (card_range k)
    _ = ENNReal.ofReal (k / 2 ^ 64) := by
        rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (by positivity)]
        congr 1
        ring

open Classical in
/-- **A window test on uniform first blocks** outputs true with probability at most `k/2^64` (some first word
rejected) plus `(g/U)^k`, `g = |C ∩ [0, U)|`, the latter `0` when `g < k`. -/
theorem pr_test_le {m U k : ℕ} (hm : m < 2 ^ 448) (hU : U ≤ m) (hk : k ≤ U) (C : Finset ℕ) :
    pr (test (m := m) U k C) ≤ ENNReal.ofReal (k / 2 ^ 64) + ENNReal.ofReal (if k ≤ ((range U).filter (· ∈ C)).card
        then ((((range U).filter (· ∈ C)).card : ℝ) / U) ^ k else 0) := by
  rw [pr_equiv (Equiv.prodComm Unit (Blocks m))]
  refine pr_le_of_fiber _ _ (fun _ => ?_)
  refine (Bandwidth.pr_mono (fun b hb => (?_ : ¬ accepted U k b ∨
      (accepted U k b ∧ openings U k b (fun _ => 0) ⊆ C)))).trans
    ((Bandwidth.pr_or_le _ _).trans (add_le_add (pr_not_accepted_le hm hU hk) (pr_window_le hm hU hk C _)))
  by_cases ha : accepted U k b
  · exact Or.inr ⟨ha, hb ha⟩
  · exact Or.inl ha

/-! ## Rounds with an additive error -/

/-- Adding each round's error and capping at 1 costs at most the errors' sum: for `h i ≤ 1`,
`∏ min(1, h i + ε i) ≤ ∏ h i + Σ ε i`. -/
theorem prod_min_le {ι : Type} (u : Finset ι) (h ε : ι → ℝ≥0∞) (h1 : ∀ i ∈ u, h i ≤ 1) :
    ∏ i ∈ u, min 1 (h i + ε i) ≤ ∏ i ∈ u, h i + ∑ i ∈ u, ε i := by
  classical
  induction u using Finset.induction_on with
  | empty => simp
  | @insert a u ha ih =>
    rw [prod_insert ha, prod_insert ha, sum_insert ha]
    have hP : ∏ i ∈ u, min 1 (h i + ε i) ≤ 1 := prod_le_one fun i _ => min_le_left _ _
    have ha1 : h a ≤ 1 := h1 a (mem_insert_self a u)
    have ih' := ih fun i hi => h1 i (mem_insert_of_mem hi)
    calc min 1 (h a + ε a) * ∏ i ∈ u, min 1 (h i + ε i)
        ≤ (h a + ε a) * ∏ i ∈ u, min 1 (h i + ε i) := by gcongr; exact min_le_right _ _
      _ = h a * ∏ i ∈ u, min 1 (h i + ε i) + ε a * ∏ i ∈ u, min 1 (h i + ε i) := add_mul _ _ _
      _ ≤ h a * (∏ i ∈ u, h i + ∑ i ∈ u, ε i) + ε a * 1 := by gcongr
      _ = h a * ∏ i ∈ u, h i + h a * ∑ i ∈ u, ε i + ε a := by rw [mul_add, mul_one]
      _ ≤ h a * ∏ i ∈ u, h i + 1 * ∑ i ∈ u, ε i + ε a := by gcongr
      _ = h a * ∏ i ∈ u, h i + (ε a + ∑ i ∈ u, ε i) := by rw [one_mul, add_assoc, add_comm (∑ i ∈ u, ε i)]

/-- **Rounds of uniform coins with an additive error**: when round `i`'s event has probability at most `h i + ε i`
over its own coin, with `h i ≤ 1` read off the earlier rounds, every event holds while the `h i` multiply to at most
`c` with probability at most `c + Σ_i ε i`. -/
theorem pr_rounds_add_le (n : ℕ) (Z : Fin n → Type) [∀ i, Fintype (Z i)]
    (A : (i : Fin n) → ((j : Fin n) → Z j) → Prop) (h : (i : Fin n) → ((j : Fin n) → Z j) → ℝ≥0∞)
    (ε : Fin n → ℝ≥0∞) (hA : ∀ i ω ω', (∀ j, j ≤ i → ω j = ω' j) → (A i ω ↔ A i ω'))
    (hh : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → h i ω = h i ω') (h1 : ∀ i ω, h i ω ≤ 1)
    (hb : ∀ i ω, pr (fun z : Z i => A i (Function.update ω i z)) ≤ h i ω + ε i) (c : ℝ≥0∞) :
    pr (fun ω => (∀ i, A i ω) ∧ ∏ i, h i ω ≤ c) ≤ c + ∑ i, ε i := by
  refine (Bandwidth.pr_mono (fun ω hω => (?_ : (∀ i, A i ω) ∧
      ∏ i, min 1 (h i ω + ε i) ≤ c + ∑ i, ε i))).trans
    (pr_rounds_le n Z A (fun i ω => min 1 (h i ω + ε i)) hA (fun i ω ω' hω => by rw [hh i ω ω' hω])
      (fun i ω => le_min (pr_le_one _) (hb i ω)) _)
  exact ⟨hω.1, (prod_min_le univ (fun i => h i ω) ε (fun i _ => h1 i ω)).trans (add_le_add hω.2 le_rfl)⟩

/-! ## The pooled bound -/

/-- Wrong receipts beyond the pooled bound leave the windows' draws, all landing on correct receipts, a probability
of at most `δ`: `∏_t (g_t/U_t)^(d_t) ≤ δ`, and `0` when a window has fewer correct receipts than draws. -/
theorem prod_le_of_pooled_lt {n : ℕ} (U g d : Fin n → ℕ) {δ : ℝ} (hδ : 0 ≤ δ) (hg : ∀ t, g t ≤ U t)
    (h : pooled n U d δ < ∑ t, ((U t - g t : ℕ) : ℝ)) :
    ∏ t, (if d t ≤ g t then ((g t : ℝ) / U t) ^ d t else 0) ≤ δ := by
  by_cases hdg : ∀ t, d t ≤ g t
  swap
  · push Not at hdg
    obtain ⟨t, ht⟩ := hdg
    rw [Finset.prod_eq_zero (Finset.mem_univ t) (by simp [not_le.2 ht])]
    exact hδ
  simp only [hdg, ↓reduceIte]
  set f : Fin n → ℝ := fun t => ((U t - g t : ℕ) : ℝ) / U t with hf
  have hfU : ∀ t, f t * U t = ((U t - g t : ℕ) : ℝ) := by
    intro t
    rcases Nat.eq_zero_or_pos (U t) with h0 | hpos
    · have : g t = 0 := by have := hg t; omega
      simp [hf, h0, this]
    · have hU : (U t : ℝ) ≠ 0 := by positivity
      simp only [hf]
      field_simp
  have hbox : ∀ t, 0 ≤ f t ∧ f t ≤ 1 - (d t : ℝ) / U t := by
    intro t
    refine ⟨by positivity, ?_⟩
    rcases Nat.eq_zero_or_pos (U t) with h0 | hpos
    · simp [hf, h0]
    · have hU : (0 : ℝ) < U t := by exact_mod_cast hpos
      have hd : (d t : ℝ) ≤ g t := by exact_mod_cast hdg t
      simp only [hf]
      rw [Nat.cast_sub (hg t), sub_div, div_self hU.ne']
      gcongr
  have hbdd : BddAbove ((fun f : Fin n → ℝ => ∑ t, f t * U t) ''
      {f | (∀ t, 0 ≤ f t ∧ f t ≤ 1 - (d t : ℝ) / U t) ∧ ∑ t, (d t : ℝ) * f t ≤ budget (∑ t, d t) δ}) := by
    refine ⟨∑ t, (U t : ℝ), ?_⟩
    rintro _ ⟨f', ⟨hf', -⟩, rfl⟩
    refine Finset.sum_le_sum (fun t _ => ?_)
    have h1 : f' t ≤ 1 := (hf' t).2.trans (by have : 0 ≤ (d t : ℝ) / U t := by positivity
                                              linarith)
    nlinarith [(hf' t).1, (Nat.cast_nonneg (U t) : (0 : ℝ) ≤ U t)]
  have hover : budget (∑ t, d t) δ < ∑ t, (d t : ℝ) * f t := by
    by_contra hle
    push Not at hle
    have := le_csSup hbdd ⟨f, ⟨hbox, hle⟩, rfl⟩
    simp only [hfU] at this
    exact absurd h (not_lt.2 this)
  set S := ∑ t, d t with hS
  have hd0 : S = 0 → ∀ t, d t = 0 := fun h0 t => by
    have := Finset.single_le_sum (fun i _ => Nat.zero_le (d i)) (Finset.mem_univ t)
    omega
  have hSpos : 0 < S := by
    rcases Nat.eq_zero_or_pos S with h0 | h0
    · have hz : ∑ t, (d t : ℝ) * f t = 0 := Finset.sum_eq_zero (fun t _ => by simp [hd0 h0 t])
      rw [hz, budget, h0] at hover
      simp at hover
    · exact h0
  set x : Fin n → ℝ := fun t => (g t : ℝ) / U t with hx
  have hx0 : ∀ t, 0 ≤ x t := fun t => by positivity
  have hdx : ∀ t, (d t : ℝ) * x t = d t - d t * f t := by
    intro t
    rcases Nat.eq_zero_or_pos (U t) with h0 | hpos
    · have : d t = 0 := by have := hdg t; have := hg t; omega
      simp [this]
    · have hU : (U t : ℝ) ≠ 0 := by positivity
      simp only [hx, hf]
      rw [Nat.cast_sub (hg t)]
      field_simp
      ring
  have hSR0 : (0 : ℝ) < S := by exact_mod_cast hSpos
  have hsum : ∑ t, (d t : ℝ) * x t < S * δ ^ ((S : ℝ)⁻¹) := by
    rw [Finset.sum_congr rfl (fun t _ => hdx t), Finset.sum_sub_distrib, ← Nat.cast_sum, ← hS]
    unfold budget at hover
    linarith
  have hamgm := Real.geom_mean_le_arith_mean Finset.univ (fun t => (d t : ℝ)) x (fun t _ => Nat.cast_nonneg _)
    (by rw [← Nat.cast_sum, ← hS]; exact hSR0) (fun t _ => hx0 t)
  have hSR : (∑ t, (d t : ℝ)) = S := by rw [hS, Nat.cast_sum]
  have hP : ∏ t, x t ^ (d t : ℝ) = ∏ t, x t ^ d t :=
    Finset.prod_congr rfl (fun t _ => Real.rpow_natCast _ _)
  rw [hSR, hP] at hamgm
  have hP0 : 0 ≤ ∏ t, x t ^ d t := Finset.prod_nonneg (fun t _ => pow_nonneg (hx0 t) _)
  have hroot : (∏ t, x t ^ d t) ^ ((S : ℝ)⁻¹) ≤ δ ^ ((S : ℝ)⁻¹) := by
    refine hamgm.trans ?_
    rw [div_le_iff₀ hSR0]
    linarith
  calc ∏ t, x t ^ d t = ((∏ t, x t ^ d t) ^ ((S : ℝ)⁻¹)) ^ S :=
        (Real.rpow_inv_natCast_pow hP0 hSpos.ne').symm
    _ ≤ (δ ^ ((S : ℝ)⁻¹)) ^ S := pow_le_pow_left₀ (Real.rpow_nonneg hP0 _) hroot S
    _ = δ := Real.rpow_inv_natCast_pow hδ hSpos.ne'

/-! ## The session -/

section Session

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} (D : Session n m s κ X)

theorem draws_le (t : Fin n) (ω : Coins n κ X) : D.draws t ω ≤ m :=
  (min_le_right _ _).trans (Nat.lt_succ_iff.1 (D.size t ω).isLt)

theorem good_le (t : Fin n) (ω : Coins n κ X) : D.good t ω ≤ D.size t ω :=
  (card_le_card (filter_subset _ _)).trans_eq (card_range _)

end Session

/-- `PooledDraw` at one session. -/
theorem pooledDraw {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} [Fintype X]
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ) (D : Session n m s κ X)
    (T : Trace (Coins n κ X)) {δ : ℝ} (hδ : 0 ≤ δ) (hm : m < 2 ^ 448) (hL : Live D) (hR : Refused D prf rest T)
    (hC : Counted D T) :
    T.Holds (δ + ((∑ t, s t : ℕ) : ℝ) / 2 ^ 64 + (∑ t, advantage (prf t)).toReal)
      (fun ω => (T.wrong ω : ℝ) ≤ D.pooled δ ω) := by
  classical
  let ρ : Fin n → Coins n κ X → ℝ := fun t ω =>
    if D.draws t ω ≤ D.good t ω then ((D.good t ω : ℝ) / (D.size t ω : ℕ)) ^ D.draws t ω else 0
  have hρ0 : ∀ t ω, 0 ≤ ρ t ω := fun t ω => by
    simp only [ρ]
    split_ifs
    · positivity
    · exact le_rfl
  have hρ1 : ∀ t ω, ρ t ω ≤ 1 := fun t ω => by
    simp only [ρ]
    split_ifs
    · exact pow_le_one₀ (by positivity) (div_le_one_of_le₀ (by exact_mod_cast good_le D t ω) (by positivity))
    · exact zero_le_one
  let A : Fin n → Coins n κ X → Prop := fun t ω =>
    test (D.size t ω) (D.draws t ω) (D.correct t ω) ((), prf t (ω.2 t))
  let ε : Fin n → ℝ≥0∞ := fun t => ENNReal.ofReal (s t / 2 ^ 64) + advantage (prf t)
  have hadv : ∀ t, advantage (prf t) ≤ 1 := fun t =>
    iSup_le fun _ => iSup_le fun _ => iSup_le fun _ => tsub_le_self.trans (pr_le_one _)
  have hreal : ∀ ω, T.accepts ω ∧ ¬ (T.wrong ω : ℝ) ≤ D.pooled δ ω →
      (∀ t, A t ω) ∧ ∏ t, ENNReal.ofReal (ρ t ω) ≤ ENNReal.ofReal δ := by
    intro ω ⟨hacc, hw⟩
    refine ⟨fun t ha => ?_, ?_⟩
    · have h := hR ω hacc t
      simp only [Session.opened] at h
      rwa [openings_congr (fun _ => 0) (rest ω t) (draws_le D t ω) ha]
    · rw [← ENNReal.ofReal_prod_of_nonneg (fun t _ => hρ0 t ω)]
      refine ENNReal.ofReal_le_ofReal (prod_le_of_pooled_lt (fun t => D.size t ω) (fun t => D.good t ω)
        (fun t => D.draws t ω) hδ (fun t => good_le D t ω) ?_)
      have hw' : D.pooled δ ω < T.wrong ω := not_le.1 hw
      have hc : (T.wrong ω : ℝ) ≤ D.wrong ω := by exact_mod_cast hC ω hacc
      calc pooled n (fun t => D.size t ω) (fun t => D.draws t ω) δ = D.pooled δ ω := rfl
        _ < D.wrong ω := hw'.trans_le hc
        _ = ∑ t, ((D.size t ω - D.good t ω : ℕ) : ℝ) := by rw [Session.wrong, Nat.cast_sum]
  have hx : ∀ x : X, pr (fun c : Fin n → Pous.Bits κ => (∀ t, A t (x, c)) ∧
      ∏ t, ENNReal.ofReal (ρ t (x, c)) ≤ ENNReal.ofReal δ) ≤ ENNReal.ofReal δ + ∑ t, ε t := by
    intro x
    have hl : ∀ t (c c' : Fin n → Pous.Bits κ), (∀ j, j < t → c j = c' j) →
        D.size t (x, c) = D.size t (x, c') ∧ D.correct t (x, c) = D.correct t (x, c') :=
      fun t c c' h => hL t (x, c) (x, c') rfl h
    refine pr_rounds_add_le n (fun _ => Pous.Bits κ) (fun t c => A t (x, c))
      (fun t c => ENNReal.ofReal (ρ t (x, c))) ε ?_ ?_ (fun t c => ENNReal.ofReal_le_one.2 (hρ1 t _)) ?_ _
    · intro t c c' h
      have e := hl t c c' (fun j hj => h j hj.le)
      have hd : D.draws t (x, c) = D.draws t (x, c') := by simp only [Session.draws, e.1]
      simp only [A, hd, e.1, e.2, h t le_rfl]
    · intro t c c' h
      have e := hl t c c' h
      have e1 : D.draws t (x, c) = D.draws t (x, c') := by simp only [Session.draws, e.1]
      have e2 : D.good t (x, c) = D.good t (x, c') := by
        unfold Session.good
        rw [e.2, e.1]
      have e3 : ((D.size t (x, c) : ℕ) : ℝ) = (D.size t (x, c') : ℕ) := by rw [e.1]
      simp only [ρ, e1, e2, e3]
    · intro t c
      have e : ∀ z, D.size t (x, Function.update c t z) = D.size t (x, c) ∧
          D.correct t (x, Function.update c t z) = D.correct t (x, c) :=
        fun z => hl t _ _ (fun j hj => Function.update_of_ne hj.ne _ _)
      have hd : ∀ z, D.draws t (x, Function.update c t z) = D.draws t (x, c) := fun z => by
        simp only [Session.draws, (e z).1]
      have hU : (D.size t (x, c) : ℕ) ≤ m := Nat.lt_succ_iff.1 (D.size t (x, c)).isLt
      have hk : D.draws t (x, c) ≤ D.size t (x, c) := min_le_right _ _
      rw [pr_equiv (Equiv.punitProd (Pous.Bits κ)).symm]
      refine (Bandwidth.pr_mono (fun y hy => (?_ : test (D.size t (x, c)) (D.draws t (x, c)) (D.correct t (x, c))
          (y.1, prf t y.2)))).trans ((Actual.pr_le_add_prfAdvantage (prf t) _ (pr_test_le hm hU hk _)).trans ?_)
      · simp only [A, Equiv.symm_symm, Equiv.punitProd_apply, (e _).1, (e _).2, hd, Function.update_self] at hy
        exact hy
      have hk' : (D.draws t (x, c) : ℝ) / 2 ^ 64 ≤ s t / 2 ^ 64 := by
        gcongr
        exact_mod_cast min_le_left _ _
      have ha : Pous.StreamedWipe.prfAdvantage (prf t)
          (test (D.size t (x, c)) (D.draws t (x, c)) (D.correct t (x, c))) ≤ advantage (prf t) :=
        le_iSup_of_le _ (le_iSup_of_le _ (le_iSup_of_le _ le_rfl))
      calc _ = ENNReal.ofReal (ρ t (x, c)) + (ENNReal.ofReal ((D.draws t (x, c) : ℝ) / 2 ^ 64) +
            Pous.StreamedWipe.prfAdvantage (prf t)
              (test (D.size t (x, c)) (D.draws t (x, c)) (D.correct t (x, c)))) := by
            rw [add_comm (ENNReal.ofReal _) (ENNReal.ofReal _), add_assoc]
            rfl
        _ ≤ ENNReal.ofReal (ρ t (x, c)) + ε t := add_le_add le_rfl (add_le_add (ENNReal.ofReal_le_ofReal hk') ha)
  have hall : pr (fun ω : Coins n κ X => (∀ t, A t ω) ∧ ∏ t, ENNReal.ofReal (ρ t ω) ≤ ENNReal.ofReal δ) ≤
      ENNReal.ofReal δ + ∑ t, ε t := by
    rw [pr_equiv (Equiv.prodComm X (Fin n → Pous.Bits κ))]
    exact pr_le_of_fiber _ _ hx
  have hadv' : ∑ t, advantage (prf t) ≠ ⊤ :=
    ENNReal.sum_ne_top.2 fun t _ => ne_top_of_le_ne_top ENNReal.one_ne_top (hadv t)
  have hpos : 0 ≤ δ + ((∑ t, s t : ℕ) : ℝ) / 2 ^ 64 := by positivity
  have hε : ENNReal.ofReal δ + ∑ t, ε t =
      ENNReal.ofReal (δ + ((∑ t, s t : ℕ) : ℝ) / 2 ^ 64) + ∑ t, advantage (prf t) := by
    have h1 : ENNReal.ofReal (((∑ t, s t : ℕ) : ℝ) / 2 ^ 64) = ∑ t, ENNReal.ofReal ((s t : ℝ) / 2 ^ 64) := by
      rw [Nat.cast_sum, sum_div]
      exact ENNReal.ofReal_sum_of_nonneg fun t _ => by positivity
    rw [ENNReal.ofReal_add hδ (by positivity), h1, add_assoc, ← sum_add_distrib]
  unfold Trace.Holds
  rw [Accounting.pr_eq_toReal]
  have h := (Bandwidth.pr_mono hreal).trans hall
  rw [hε] at h
  calc _ ≤ (ENNReal.ofReal (δ + ((∑ t, s t : ℕ) : ℝ) / 2 ^ 64) + ∑ t, advantage (prf t)).toReal :=
        ENNReal.toReal_mono (ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top, hadv'⟩) h
    _ = _ := by rw [ENNReal.toReal_add ENNReal.ofReal_ne_top hadv', ENNReal.toReal_ofReal hpos]

end Draw

/-- **The openings' pooled bound** (`Accounting.Guarantees.Remote.PooledDraw`). -/
theorem PooledDraw : Accounting.Guarantees.Remote.PooledDraw := by
  intro n m k κ X _ prf rest D T δ hδ hm hL hR hC
  exact Draw.pooledDraw prf rest D T hδ hm hL hR hC

/-- **Live coins, derived** (`Accounting.Guarantees.Remote.LiveCoinsDrawn`). -/
theorem LiveCoinsDrawn : Accounting.Guarantees.Remote.LiveCoinsDrawn := by
  intro n m k κ X _ prf rest D T W δ hδ hm hL hR hC hW
  exact holdsMono (fun ω ha h => h.trans (hW ω ha)) (Draw.pooledDraw prf rest D T hδ hm hL hR hC)

/-- **The snapshot held at the record's pooled bound** (`Accounting.Guarantees.Remote.HeldDrawn`). -/
theorem HeldDrawn : Accounting.Guarantees.Remote.HeldDrawn := by
  intro n m k κ X _ prf rest D T S s₀ e s E β Br Bd δs δc hE hs hβ hδ hm hI hsq hd hS hr hc hk hL hR hC
  refine holdsMono (fun ω ha h => ?_) (holdsAnd hk (Draw.pooledDraw prf rest D T hδ hm hL hR hC))
  obtain ⟨hm', h0, hP⟩ := hc ω ha
  exact held_of ω hE hs hβ hI hsq hd hS hr hm' h0 hP h.1 h.2

/-- **Held over time with the links charged, at the record's pooled bound**
(`Accounting.Guarantees.Remote.HeldOverTimeLinksDrawn`). -/
theorem HeldOverTimeLinksDrawn : Accounting.Guarantees.Remote.HeldOverTimeLinksDrawn := by
  intro n m k κ X _ prf rest D T K S s E β βl Br Bd Bl δs δc hE hs hβ hβl hδ hm hl hsq hd hS hr hp hcov hk hL hR hC
  refine holdsMono (fun ω ha h => ?_) (holdsAnd hk (Draw.pooledDraw prf rest D T hδ hm hL hR hC))
  exact heldOverLinks_of K ω hE hs hβ hβl hl hsq hd hS hr hp hcov ha h.1 h.2

/-- **Held over time at the record's pooled bound, under Identity** (`Accounting.Guarantees.Remote.HeldOverTimeDrawn`):
the link-charged form at links of rate 0 and burst 0. -/
theorem HeldOverTimeDrawn : Accounting.Guarantees.Remote.HeldOverTimeDrawn := by
  intro n m k κ X _ prf rest D T K S s E β Br Bd δs δc hE hs hβ hδ hm hI hsq hd hS hr hp hcov hk hL hR hC
  simpa only [add_zero] using HeldOverTimeLinksDrawn n m k κ X prf rest D T K S s E β 0 Br Bd 0 δs δc hE hs hβ le_rfl
    hδ hm (links_of_identity hI) hsq hd hS hr hp hcov hk hL hR hC

end Accounting.SecurityProofs.Remote
