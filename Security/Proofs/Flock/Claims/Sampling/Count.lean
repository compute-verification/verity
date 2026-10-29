import Batteries.Tactic.OpenPrivate
import SampledProofs.Bound
import Proofs.Flock.Claims.Sampling.Escape

/-!
# The deployed count bound, at the claims' strata draws

`SampledProofs.Bound.countBound strata δ` is what the sampled verifier publishes as the most wrong units an accepted
transcript holds except with probability `δ`: it multiplies the strata's factors `(n − j − k)/(n − j)` greedily, the
largest available first, and stops before the product falls below `δ`. Here it is proved, as the code is, against the
claims' `Design.miss` (`Definitions.Claims.Sampling`):

* `countBound_miss`: **`miss(countBound + 1) < δ`** for every design that is a strata draw over the partition's proof
  blocks whose strata have the code's sizes and counts, every `0 < δ` (`countBound_strataPa`, through `strataPa_eq_V`
  and `sum_card_window`); `countBound_miss_design`: the same for every well-formed design whose rules are strata draws
  (`Design.Pa_eq_strataPa`; every one-stage design but Bernoulli's, `OneStage.strata`), `strataOf_subset` the uniform
  `k`-subset's one stratum.
* `choose_div_eq_gr`: a stratum's escape `C(n − m, k)/C(n, k)` is the product of its first `m` factors, each in
  `[0, 1]` and falling with `j` (`fac_anti`).
* `V_le_of_sep`: an exchange: when a threshold `τ` separates the factors a choice of counts takes from those it
  leaves, every choice of at least as many units escapes with probability at most the choice's own.
* `next_some`, `next_none`: the code's `next` picks a stratum with room whose next factor is largest, or reports that
  none has room; `countBound_good`: the loop keeps the greedy invariant, so whichever way it returns, every `K + 1`
  wrong units escape with probability at most a number below `δ`.
-/

namespace Claims.Sampling.Count

open Finset SampledProofs.Bound
open private next reduce atLeast from SampledProofs.Bound

/-! ## A stratum's factors -/

/-- The `j`-th factor of a stratum of `n` units drawing `k`: the probability that its draw misses a further wrong unit
given that it missed `j`, `(n − j − k)/(n − j)`. -/
noncomputable def fac (n k j : ℕ) : ℝ := ((n - j - k : ℕ) : ℝ) / ((n - j : ℕ) : ℝ)

theorem fac_nonneg (n k j : ℕ) : 0 ≤ fac n k j := by unfold fac; positivity

theorem fac_le_one (n k j : ℕ) : fac n k j ≤ 1 := by
  unfold fac
  rcases Nat.eq_zero_or_pos (n - j) with h | h
  · simp [h]
  · rw [div_le_one (by exact_mod_cast h)]
    exact_mod_cast Nat.sub_le _ _

theorem fac_anti {n k j j' : ℕ} (h : j ≤ j') : fac n k j' ≤ fac n k j := by
  rcases Nat.eq_zero_or_pos (n - j' - k) with h0 | hp
  · simp only [fac, h0, Nat.cast_zero, zero_div]
    exact fac_nonneg n k j
  · unfold fac
    rw [div_le_div_iff₀ (by exact_mod_cast (by omega : 0 < n - j')) (by exact_mod_cast (by omega : 0 < n - j))]
    rw [Nat.cast_sub (by omega : k ≤ n - j'), Nat.cast_sub (by omega : k ≤ n - j)]
    have : ((n - j' : ℕ) : ℝ) ≤ (n - j : ℕ) := by exact_mod_cast (by omega)
    nlinarith [(Nat.cast_nonneg k : (0 : ℝ) ≤ k)]

/-- Past its `finite` wrong units a stratum's draw cannot miss another. -/
theorem fac_of_finite_le {s : Core.Stratum} {m : ℕ} (h : Stratum.finite s ≤ m) : fac s.size s.k m = 0 := by
  unfold Stratum.finite at h
  unfold fac
  split_ifs at h with hk
  · have : s.size - m = 0 := by omega
    simp [this]
  · have : s.size - m - s.k = 0 := by omega
    simp [this]

theorem room_le {s : Core.Stratum} {m : ℕ} (h : m < Stratum.finite s) : s.k ≤ s.size - m ∧ 0 < s.size - m := by
  unfold Stratum.finite at h
  split_ifs at h with hk
  · simp only [beq_iff_eq] at hk
    omega
  · simp only [beq_iff_eq] at hk
    omega

theorem fac_eq {n k j : ℕ} (hk : k ≤ n - j) (hp : 0 < n - j) : fac n k j = 1 - (k : ℝ) / (n - j : ℕ) := by
  unfold fac
  have : ((n - j : ℕ) : ℝ) ≠ 0 := by exact_mod_cast hp.ne'
  rw [Nat.cast_sub hk]
  field_simp

/-- `next`'s comparison: between two strata with room, `k_s (n_t − m_t) < k_t (n_s − m_s)` exactly when `s`'s next
factor is the larger. -/
theorem lt_iff_fac {s t : Core.Stratum} {m mt : ℕ} (hs : m < Stratum.finite s) (ht : mt < Stratum.finite t) :
    s.k * (t.size - mt) < t.k * (s.size - m) ↔ fac t.size t.k mt < fac s.size s.k m := by
  obtain ⟨hks, hps⟩ := room_le hs
  obtain ⟨hkt, hpt⟩ := room_le ht
  rw [fac_eq hks hps, fac_eq hkt hpt, sub_lt_sub_iff_left,
    div_lt_div_iff₀ (by exact_mod_cast hps) (by exact_mod_cast hpt)]
  exact_mod_cast Iff.rfl

/-- The product of a stratum's first `m` factors. -/
noncomputable def gr (n k m : ℕ) : ℝ := ∏ j ∈ range m, fac n k j

theorem gr_nonneg (n k m : ℕ) : 0 ≤ gr n k m := prod_nonneg fun j _ => fac_nonneg n k j

/-- **A stratum's escape is the product of its factors**: `C(n − m, k)/C(n, k) = ∏_{j<m} (n − j − k)/(n − j)`. -/
theorem choose_div_eq_gr {n k : ℕ} (hk : k ≤ n) : ∀ {m : ℕ}, m ≤ n → ((n - m).choose k : ℝ) / n.choose k = gr n k m
  | 0, _ => by
    have : (n.choose k : ℝ) ≠ 0 := by exact_mod_cast (Nat.choose_pos hk).ne'
    simp [gr, this]
  | m + 1, hm => by
    have ih := choose_div_eq_gr hk (m := m) (by omega)
    have hn' : n - m = (n - (m + 1)) + 1 := by omega
    have key : ((n - (m + 1)).choose k : ℝ) * ((n - m : ℕ) : ℝ) = ((n - m).choose k : ℝ) * ((n - m - k : ℕ) : ℝ) := by
      have := Nat.choose_mul_succ_eq (n - (m + 1)) k
      rw [← hn'] at this
      exact_mod_cast this
    have hp : ((n - m : ℕ) : ℝ) ≠ 0 := by exact_mod_cast (by omega : n - m ≠ 0)
    have hc : (n.choose k : ℝ) ≠ 0 := by exact_mod_cast (Nat.choose_pos hk).ne'
    rw [gr, prod_range_succ, ← gr, ← ih, fac, div_mul_div_comm, div_eq_div_iff hc (mul_ne_zero hc hp)]
    linear_combination (n.choose k : ℝ) * key

/-! ## The exchange -/

section Exchange

variable (S : ℕ) (n k : ℕ → ℕ)

/-- The escape of the counts `m`, stratum `i` holding `m i` wrong units. -/
noncomputable def V (m : ℕ → ℕ) : ℝ := ∏ i ∈ range S, gr (n i) (k i) (m i)

/-- `τ` separates the factors the counts `m` take from those they leave: each stratum's later factors are at most
`τ`, its taken ones at least. -/
def Sep (m : ℕ → ℕ) (τ : ℝ) : Prop :=
  ∀ i < S, (∀ l, m i ≤ l → fac (n i) (k i) l ≤ τ) ∧ ∀ l < m i, τ ≤ fac (n i) (k i) l

variable {S n k}

theorem V_nonneg (m : ℕ → ℕ) : 0 ≤ V S n k m := prod_nonneg fun _ _ => gr_nonneg _ _ _

theorem gr_sep {n k a b : ℕ} {τ : ℝ} (hτ0 : 0 ≤ τ) (hup : ∀ l, b ≤ l → fac n k l ≤ τ)
    (hdn : ∀ l < b, τ ≤ fac n k l) : gr n k a * τ ^ (b - a) ≤ gr n k b * τ ^ (a - b) := by
  rcases le_total b a with hba | hab
  · rw [Nat.sub_eq_zero_of_le hba, pow_zero, mul_one, gr, ← prod_range_mul_prod_Ico _ hba, ← gr]
    refine mul_le_mul_of_nonneg_left ?_ (gr_nonneg _ _ _)
    calc ∏ j ∈ Ico b a, fac n k j ≤ ∏ _j ∈ Ico b a, τ :=
          prod_le_prod₀ (fun j _ => fac_nonneg _ _ _) fun j hj => hup j (mem_Ico.1 hj).1
      _ = τ ^ (a - b) := by rw [prod_const, Nat.card_Ico]
  · rw [Nat.sub_eq_zero_of_le hab, pow_zero, mul_one]
    conv_rhs => rw [gr, ← prod_range_mul_prod_Ico _ hab, ← gr]
    refine mul_le_mul_of_nonneg_left ?_ (gr_nonneg _ _ _)
    calc τ ^ (b - a) = ∏ _j ∈ Ico a b, τ := by rw [prod_const, Nat.card_Ico]
      _ ≤ ∏ j ∈ Ico a b, fac n k j := prod_le_prod₀ (fun _ _ => hτ0) fun j hj => hdn j (mem_Ico.1 hj).2

/-- **The exchange**: when `τ ∈ [0, 1]` separates the factors `mg` takes from those it leaves, every choice `m` of at
least as many wrong units escapes with probability at most `mg`'s. -/
theorem V_le_of_sep {mg : ℕ → ℕ} {τ : ℝ} (hsep : Sep S n k mg τ) (hτ0 : 0 ≤ τ) (hτ1 : τ ≤ 1) {m : ℕ → ℕ}
    (hsum : ∑ i ∈ range S, mg i ≤ ∑ i ∈ range S, m i) : V S n k m ≤ V S n k mg := by
  by_cases hup : ∃ i ∈ range S, mg i < m i
  · rcases eq_or_lt_of_le hτ0 with hτ | hτ
    · obtain ⟨i, hi, hlt⟩ := hup
      have : gr (n i) (k i) (m i) = 0 :=
        prod_eq_zero (mem_range.2 hlt) (le_antisymm (hτ ▸ (hsep i (mem_range.1 hi)).1 _ le_rfl) (fac_nonneg _ _ _))
      rw [V, prod_eq_zero hi this]
      exact V_nonneg mg
    · have hA : ∑ i ∈ range S, (mg i - m i) ≤ ∑ i ∈ range S, (m i - mg i) := by
        have : ∑ i ∈ range S, m i + ∑ i ∈ range S, (mg i - m i) =
            ∑ i ∈ range S, mg i + ∑ i ∈ range S, (m i - mg i) := by
          rw [← sum_add_distrib, ← sum_add_distrib]
          exact sum_congr rfl fun i _ => by omega
        omega
      have hprod : V S n k m * τ ^ (∑ i ∈ range S, (mg i - m i)) ≤
          V S n k mg * τ ^ (∑ i ∈ range S, (m i - mg i)) := by
        rw [V, V, ← prod_pow_eq_pow_sum, ← prod_pow_eq_pow_sum, ← prod_mul_distrib, ← prod_mul_distrib]
        exact prod_le_prod₀ (fun i _ => mul_nonneg (gr_nonneg _ _ _) (pow_nonneg hτ0 _))
          fun i hi => gr_sep hτ0 (hsep i (mem_range.1 hi)).1 (hsep i (mem_range.1 hi)).2
      have hpow : τ ^ (∑ i ∈ range S, (m i - mg i)) ≤ τ ^ (∑ i ∈ range S, (mg i - m i)) :=
        pow_le_pow_of_le_one hτ0 hτ1 hA
      exact le_of_mul_le_mul_right (hprod.trans (mul_le_mul_of_nonneg_left hpow (V_nonneg mg))) (pow_pos hτ _)
  · push Not at hup
    have : ∀ i ∈ range S, m i = mg i :=
      (sum_eq_sum_iff_of_le hup).1 (le_antisymm (sum_le_sum hup) hsum)
    exact le_of_eq (prod_congr rfl fun i hi => by rw [this i hi])

end Exchange

/-! ## The code's `next` and `reduce` -/

/-- The loop's state at the counts `m`: each stratum with its wrong units so far. -/
def state (strata : Array Core.Stratum) (m : ℕ → ℕ) : Array (Core.Stratum × ℕ) :=
  Array.ofFn fun i : Fin strata.size => (strata[i], m i)

theorem mem_state {strata : Array Core.Stratum} {m : ℕ → ℕ} {e : (Core.Stratum × ℕ) × ℕ} :
    e ∈ (state strata m).toList.zipIdx ↔ ∃ h : e.2 < strata.size, e.1 = (strata[e.2], m e.2) := by
  rw [List.mem_zipIdx_iff_getElem?, state, Array.toList_ofFn, List.getElem?_ofFn]
  constructor
  · intro h
    split at h
    · rename_i hlt
      exact ⟨hlt, (Option.some.inj h).symm⟩
    · simp at h
  · rintro ⟨hlt, he⟩
    simp [hlt, he]

theorem foldl_inv {α β : Type*} (F : β → α → β) (I : List α → β → Prop)
    (h : ∀ E b x, I E b → I (E ++ [x]) (F b x)) (l : List α) (b : β) (hb : I [] b) : I l (l.foldl F b) := by
  suffices ∀ (l E : List α) b, I E b → I (E ++ l) (l.foldl F b) by simpa using this l [] b hb
  intro l
  induction l with
  | nil => intro E b hb; simpa using hb
  | cons x l ih => intro E b hb; simpa using ih (E ++ [x]) (F b x) (h E b x hb)

/-- What `next`'s fold keeps over the strata it has seen: none has room, or the one it holds has room and the largest
next factor. -/
def NextInv (E : List ((Core.Stratum × ℕ) × ℕ)) : Option (ℕ × Core.Stratum × ℕ) → Prop
  | none => ∀ e ∈ E, Stratum.finite e.1.1 ≤ e.1.2
  | some (i, s, m) => ((s, m), i) ∈ E ∧ m < Stratum.finite s ∧ ∀ e ∈ E, fac e.1.1.size e.1.1.k e.1.2 ≤ fac s.size s.k m

theorem next_inv (st : Array (Core.Stratum × ℕ)) : NextInv st.toList.zipIdx (next st) := by
  unfold next
  rw [← Array.foldl_toList, Array.toList_zipIdx]
  refine foldl_inv _ NextInv ?_ _ _ (by simp [NextInv])
  rintro E b ⟨⟨s, m⟩, i⟩ hb
  dsimp only
  split_ifs with h1
  · match b, hb with
    | none, hb => exact fun e he => (List.mem_append.1 he).elim (hb e) fun he => by simp at he; subst he; exact h1
    | some (j, t, mt), ⟨hmem, hroom, hmax⟩ =>
      refine ⟨List.mem_append_left _ hmem, hroom, fun e he => (List.mem_append.1 he).elim (hmax e) fun he => ?_⟩
      simp only [List.mem_singleton] at he
      subst he
      rw [fac_of_finite_le h1]
      exact fac_nonneg _ _ _
  · have h1' : m < Stratum.finite s := by omega
    match b, hb with
    | none, hb =>
      refine ⟨List.mem_append_right _ (List.mem_singleton_self _), h1', fun e he => ?_⟩
      rcases List.mem_append.1 he with he | he
      · rw [fac_of_finite_le (hb e he)]
        exact fac_nonneg _ _ _
      · simp only [List.mem_singleton] at he
        subst he
        exact le_rfl
    | some (j, t, mt), ⟨hmem, hroom, hmax⟩ =>
      dsimp only
      split_ifs with h2
      · have hlt := (lt_iff_fac h1' hroom).1 h2
        refine ⟨List.mem_append_right _ (List.mem_singleton_self _), h1', fun e he => ?_⟩
        rcases List.mem_append.1 he with he | he
        · exact (hmax e he).trans hlt.le
        · simp only [List.mem_singleton] at he
          subst he
          exact le_rfl
      · have hle := not_lt.1 fun h => h2 ((lt_iff_fac h1' hroom).2 h)
        refine ⟨List.mem_append_left _ hmem, hroom, fun e he => (List.mem_append.1 he).elim (hmax e) fun he => ?_⟩
        simp only [List.mem_singleton] at he
        subst he
        exact hle

/-- **`next` reports no stratum with room** only when none has any. -/
theorem next_none {strata : Array Core.Stratum} {m : ℕ → ℕ} (h : next (state strata m) = none) :
    ∀ i (hi : i < strata.size), Stratum.finite strata[i] ≤ m i := by
  have := next_inv (state strata m)
  rw [h] at this
  intro i hi
  exact this ((strata[i], m i), i) (mem_state.2 ⟨hi, rfl⟩)

/-- **`next` picks a stratum with room whose next factor is the largest.** -/
theorem next_some {strata : Array Core.Stratum} {m : ℕ → ℕ} {i : ℕ} {s : Core.Stratum} {m₀ : ℕ}
    (h : next (state strata m) = some (i, s, m₀)) :
    ∃ hi : i < strata.size, strata[i] = s ∧ m i = m₀ ∧ m₀ < Stratum.finite s ∧
      ∀ j (hj : j < strata.size), fac strata[j].size strata[j].k (m j) ≤ fac s.size s.k m₀ := by
  have := next_inv (state strata m)
  rw [h] at this
  obtain ⟨hmem, hroom, hmax⟩ := this
  obtain ⟨hi, he⟩ := mem_state.1 hmem
  simp only [Prod.mk.injEq] at he
  exact ⟨hi, he.1.symm, he.2.symm, hroom, fun j hj => hmax ((strata[j], m j), j) (mem_state.2 ⟨hj, rfl⟩)⟩

/-- `reduce` keeps the ratio and a positive denominator. -/
theorem reduce_spec {a d : ℕ} (hd : 0 < d) :
    0 < (reduce a d).2 ∧ ((reduce a d).1 : ℝ) / (reduce a d).2 = (a : ℝ) / d := by
  unfold reduce
  dsimp only
  split_ifs with hg
  · exact ⟨hd, rfl⟩
  · have hg0 : 0 < Nat.gcd a d := Nat.gcd_pos_of_pos_right _ hd
    have hga : Nat.gcd a d ∣ a := Nat.gcd_dvd_left a d
    have hgd : Nat.gcd a d ∣ d := Nat.gcd_dvd_right a d
    refine ⟨Nat.div_pos (Nat.le_of_dvd hd hgd) hg0, ?_⟩
    rw [Nat.cast_div hga (by exact_mod_cast hg0.ne'), Nat.cast_div hgd (by exact_mod_cast hg0.ne')]
    have : ((Nat.gcd a d : ℕ) : ℝ) ≠ 0 := by exact_mod_cast hg0.ne'
    have : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
    field_simp

theorem atLeast_eq (a b : ℕ) (δ : Core.Ratio) : atLeast a b δ = decide (δ.num * b ≤ a * δ.den) := rfl

/-! ## The loop -/

theorem forIn_inv {β : Type} (f : ℕ → β → Id (ForInStep β)) (P : ℕ → β → Prop) (Q : β → Prop)
    (hf : ∀ K b, P K b → (∃ b', (f K b).run = .done b' ∧ Q b') ∨ ∃ b', (f K b).run = .yield b' ∧ P (K + 1) b') :
    ∀ r K b, P K b → Q (forIn (List.range' K r) b f).run ∨ P (K + r) (forIn (List.range' K r) b f).run := by
  intro r
  induction r with
  | zero => intro K b hb; right; simpa using hb
  | succ r ih =>
    intro K b hb
    rw [List.range'_succ, List.forIn_cons, Id.run_bind]
    rcases hf K b hb with ⟨b', he, hq⟩ | ⟨b', he, hp⟩
    · rw [he]
      exact Or.inl hq
    · rw [he]
      rcases ih (K + 1) b' hp with h | h
      · exact Or.inl h
      · right
        rwa [show K + (r + 1) = K + 1 + r by omega]

theorem foldl_size (l : List Core.Stratum) (a : ℕ) :
    l.foldl (fun n s => n + s.size) a = a + (l.map Core.Stratum.size).sum := by
  induction l generalizing a with
  | nil => simp
  | cons s l ih => rw [List.foldl_cons, ih, List.map_cons, List.sum_cons]; omega

theorem population_eq (strata : Array Core.Stratum) (n : ℕ → ℕ)
    (hn : ∀ i (h : i < strata.size), strata[i].size = n i) :
    strata.foldl (fun n s => n + s.size) 0 = ∑ i ∈ range strata.size, n i := by
  rw [← Array.foldl_toList, foldl_size, zero_add, ← List.ofFn_getElem_eq_map, List.sum_ofFn,
    ← Fin.sum_univ_eq_sum_range (fun i => n i)]
  have hl : strata.toList.length = strata.size := Array.length_toList
  refine Fintype.sum_equiv (finCongr hl) _ _ fun i => ?_
  simp only [finCongr_apply, Fin.val_cast, Array.getElem_toList]
  exact hn i (hl ▸ i.2)

theorem good_of_loop {β : Type} (G : ℕ → Prop) (r : ℕ) (init : Option ℕ × β)
    (f : ℕ → Option ℕ × β → Id (ForInStep (Option ℕ × β))) (g : Option ℕ × β → Id ℕ) (pop : ℕ)
    (P : ℕ → Option ℕ × β → Prop) (h0 : P 0 init) (hnone : ∀ K b, P K b → b.1 = none) (hpop : G pop)
    (hgs : ∀ b K, b.1 = some K → (g b).run = K) (hgn : ∀ b, b.1 = none → (g b).run = pop)
    (hf : ∀ K b, P K b → (∃ b', (f K b).run = .done b' ∧ b'.1 = some K ∧ G K) ∨
      ∃ b', (f K b).run = .yield b' ∧ P (K + 1) b') :
    G (Id.run (forIn (List.range' 0 r) init f >>= g)) := by
  rw [Id.run_bind]
  rcases forIn_inv f P (fun b => ∃ K, b.1 = some K ∧ G K)
      (fun K b hb => (hf K b hb).imp (fun ⟨b', he, h1, h2⟩ => ⟨b', he, K, h1, h2⟩) id) r 0 init h0 with
    ⟨K, hK, hG⟩ | h
  · rw [hgs _ _ hK]
    exact hG
  · rw [hgn _ (hnone _ _ h)]
    exact hpop

theorem gr_succ (n k m : ℕ) : gr n k (m + 1) = gr n k m * fac n k m := prod_range_succ _ _

theorem state_modify (strata : Array Core.Stratum) (m : ℕ → ℕ) (i : ℕ) :
    (state strata m).modify i (fun x => (x.1, x.2 + 1)) = state strata (Function.update m i (m i + 1)) := by
  apply Array.ext
  · simp [state]
  · intro j h1 h2
    rw [Array.getElem_modify]
    simp only [state, Array.getElem_ofFn]
    split_ifs with hij
    · subst hij
      simp
    · simp [Function.update_of_ne (Ne.symm hij)]

/-- **The count bound holds the greedy invariant to its return**: with the strata's sizes `n` and counts `k`, every
choice of `countBound + 1` or more wrong units escapes with probability at most some `c < δ`. -/
theorem countBound_good (strata : Array Core.Stratum) (n k : ℕ → ℕ)
    (hnk : ∀ i (h : i < strata.size), strata[i].size = n i ∧ strata[i].k = k i) (δ : Core.Ratio)
    (hδ : 0 < δ.num) (hδd : 0 < δ.den) :
    ∃ c : ℝ, 0 ≤ c ∧ c < (δ.num : ℝ) / δ.den ∧ ∀ m : ℕ → ℕ, (∀ i < strata.size, m i ≤ n i) →
      countBound strata δ + 1 ≤ ∑ i ∈ range strata.size, m i → V strata.size n k m ≤ c := by
  set S := strata.size
  have hδr : (0 : ℝ) < (δ.num : ℝ) / δ.den := by positivity
  let Good : ℕ → Prop := fun K => ∃ c : ℝ, 0 ≤ c ∧ c < (δ.num : ℝ) / δ.den ∧
    ∀ m : ℕ → ℕ, (∀ i < S, m i ≤ n i) → K + 1 ≤ ∑ i ∈ range S, m i → V S n k m ≤ c
  have hfac : ∀ j (hj : j < S) (l : ℕ), fac strata[j].size strata[j].k l = fac (n j) (k j) l := fun j hj l => by
    rw [(hnk j hj).1, (hnk j hj).2]
  show Good (countBound strata δ)
  have hpop : strata.foldl (fun n s => n + s.size) 0 = ∑ i ∈ range S, n i :=
    population_eq strata n fun i h => (hnk i h).1
  unfold countBound
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel,
    Nat.div_one, hpop]
  refine good_of_loop Good _ _ _ _ _ (fun K (b : Option ℕ × Array (Core.Stratum × ℕ) × ℕ × ℕ) =>
      b.1 = none ∧ ∃ m : ℕ → ℕ, b.2.1 = state strata m ∧
      ∑ i ∈ range S, m i = K ∧ 0 < b.2.2.2 ∧ (b.2.2.1 : ℝ) / b.2.2.2 = V S n k m ∧
      ∀ i < S, ∀ j < S, ∀ l < m i, fac (n j) (k j) (m j) ≤ fac (n i) (k i) l) ?_ (fun _ _ h => h.1) ?_
      (fun b K h => by simp only [h]; rfl) (fun b h => by simp only [h]; rfl) ?_
  · refine ⟨rfl, fun _ => 0, ?_, by simp, one_pos, by simp [V, gr], fun i _ j _ l hl => absurd hl (Nat.not_lt_zero _)⟩
    apply Array.ext
    · simp [state]
    · intro j h1 h2
      simp [state]
  · refine ⟨0, le_rfl, hδr, fun m hm hsum => absurd hsum ?_⟩
    have : ∑ i ∈ range S, m i ≤ ∑ i ∈ range S, n i := sum_le_sum fun i hi => hm i (mem_range.1 hi)
    simp only [Id.run_pure]
    omega
  rintro K ⟨o, st, num, den⟩ ⟨ho, m, hst, hsum, hden, hV, hgr⟩
  dsimp only at ho hst hden hV ⊢
  subst ho hst
  rcases hnx : next (state strata m) with _ | ⟨i, s, m₀⟩
  · left
    simp only [Id.run_pure]
    refine ⟨_, rfl, rfl, 0, le_rfl, hδr, fun mB _ hB => ?_⟩
    obtain ⟨i, hi, hlt⟩ := exists_lt_of_sum_lt (by omega : ∑ i ∈ range S, m i < ∑ i ∈ range S, mB i)
    have h0 : fac (n i) (k i) (m i) = 0 := by
      rw [← hfac i (mem_range.1 hi)]
      exact fac_of_finite_le (next_none hnx i (mem_range.1 hi))
    rw [V, prod_eq_zero hi (prod_eq_zero (mem_range.2 hlt) h0)]
  · obtain ⟨hi, hs, hm, hroom, hmax⟩ := next_some hnx
    subst hs hm
    obtain ⟨hkr, hpr⟩ := room_le hroom
    have hd : 0 < den * (strata[i].size - m i) := Nat.mul_pos hden hpr
    obtain ⟨hb, hab⟩ := reduce_spec (a := num * (strata[i].size - m i - strata[i].k)) hd
    set τ := fac (n i) (k i) (m i) with hτ
    set m' := Function.update m i (m i + 1) with hm'
    have hratio : ((reduce (num * (strata[i].size - m i - strata[i].k)) (den * (strata[i].size - m i))).1 : ℝ) /
        (reduce (num * (strata[i].size - m i - strata[i].k)) (den * (strata[i].size - m i))).2 = V S n k m' := by
      rw [hab]
      have hV' : V S n k m' = V S n k m * τ := by
        unfold V
        rw [← mul_prod_erase _ _ (mem_range.2 hi), ← mul_prod_erase _ _ (mem_range.2 hi), hm',
          Function.update_self, gr_succ,
          prod_congr rfl fun j hj => by rw [Function.update_of_ne (ne_of_mem_erase hj)]]
        ring
      rw [hV', ← hV, hτ, ← hfac i hi, fac, Nat.cast_mul, Nat.cast_mul, mul_div_mul_comm]
    have hsep : Sep S n k m' τ := by
      intro u hu
      by_cases hui : u = i
      · subst hui
        simp only [hm', Function.update_self]
        refine ⟨fun l hl => fac_anti (by omega), fun l hl => ?_⟩
        rcases Nat.lt_succ_iff_lt_or_eq.1 hl with hl | hl
        · exact hgr u hu u hu l hl
        · rw [hl]
      · simp only [hm', Function.update_of_ne hui]
        refine ⟨fun l hl => (fac_anti hl).trans ?_, fun l hl => hgr u hu i hi l hl⟩
        have := hmax u hu
        rwa [hfac u hu, hfac i hi] at this
    have hsum' : ∑ j ∈ range S, m' j = K + 1 := by
      rw [hm', sum_update_of_mem (mem_range.2 hi), ← hsum, sum_eq_add_sum_sdiff_singleton_of_mem (mem_range.2 hi) m]
      omega
    dsimp only
    split_ifs with hat
    · left
      simp only [Id.run_pure]
      refine ⟨_, rfl, rfl, V S n k m', V_nonneg _, ?_, fun mB _ hB =>
        V_le_of_sep hsep (fac_nonneg _ _ _) (fac_le_one _ _ _) (by omega)⟩
      rw [← hratio, div_lt_div_iff₀ (by exact_mod_cast hb) (by exact_mod_cast hδd)]
      simp only [Bool.not_eq_true', atLeast_eq, decide_eq_false_iff_not, not_le] at hat
      exact_mod_cast hat
    · right
      simp only [Id.run_pure]
      refine ⟨_, rfl, rfl, m', state_modify strata m i, hsum', hb, hratio, fun u hu j hj l hl =>
        ((hsep j hj).1 _ le_rfl).trans ((hsep u hu).2 l hl)⟩

/-! ## At the claims' strata draws -/

/-- The slot at which stratum `i` starts: the sizes of the strata before it. -/
def strataOff (ss : List (ℕ × ℕ)) (i : ℕ) : ℕ := ((ss.take i).map Prod.fst).sum

/-- `strataPa` as a product over its strata, stratum `i` reading `B` on its own window. -/
theorem strataPa_eq_prod : ∀ (ss : List (ℕ × ℕ)) (o : ℕ) (B : Finset ℕ),
    strataPa ss o B = ∏ i ∈ range ss.length, ((((ss.getD i (0, 0)).1 -
      (window (o + strataOff ss i) (ss.getD i (0, 0)).1 B).card).choose (ss.getD i (0, 0)).2 : ℝ) /
        ((ss.getD i (0, 0)).1.choose (ss.getD i (0, 0)).2 : ℝ))
  | [], _, _ => by simp [strataPa]
  | (n, k) :: ss, o, B => by
    rw [strataPa, strataPa_eq_prod ss (o + n) B, List.length_cons, prod_range_succ', mul_comm]
    congr 1
    refine prod_congr rfl fun i _ => ?_
    simp [strataOff, add_assoc]

theorem card_window (o n : ℕ) (B : Finset ℕ) :
    (window o n B).card = (B.filter fun v => o ≤ v ∧ v < o + n).card :=
  card_image_of_injOn fun a ha b hb h => by
    have ha' := (mem_filter.1 (mem_coe.1 ha)).2.1
    have hb' := (mem_filter.1 (mem_coe.1 hb)).2.1
    have h' : a - o = b - o := h
    omega

/-- The strata's windows count the members of `B` in `[o, o + Σ n_s)`. -/
theorem sum_card_window : ∀ (ss : List (ℕ × ℕ)) (o : ℕ) (B : Finset ℕ),
    ∑ i ∈ range ss.length, (window (o + strataOff ss i) (ss.getD i (0, 0)).1 B).card =
      (B.filter fun v => o ≤ v ∧ v < o + (ss.map Prod.fst).sum).card
  | [], o, B => by
    rw [List.length_nil, range_zero, sum_empty, eq_comm, card_eq_zero, filter_eq_empty_iff]
    intro v _
    simp
  | (n, k) :: ss, o, B => by
    have ih := sum_card_window ss (o + n) B
    have hs : ∀ i ∈ range ss.length, (window (o + strataOff ((n, k) :: ss) (i + 1))
        (((n, k) :: ss).getD (i + 1) (0, 0)).1 B).card =
        (window (o + n + strataOff ss i) (ss.getD i (0, 0)).1 B).card := fun i _ => by
      simp [strataOff, add_assoc]
    rw [List.length_cons, sum_range_succ', sum_congr rfl hs, ih]
    simp only [strataOff, List.take_zero, List.map_nil, List.sum_nil, add_zero, List.getD_cons_zero, List.map_cons,
      List.sum_cons]
    rw [card_window, ← card_union_of_disjoint (disjoint_filter.2 fun v _ h1 h2 => by omega)]
    congr 1
    ext v
    simp only [mem_union, mem_filter]
    by_cases hv : v ∈ B
    · simp only [hv, true_and]
      omega
    · simp [hv]

/-- **A strata draw's miss is the strata's product of factors**, stratum `i` holding the members of `B` in its
window. -/
theorem strataPa_eq_V (ss : List (ℕ × ℕ)) (hk : ∀ p ∈ ss, p.2 ≤ p.1) (o : ℕ) (B : Finset ℕ) :
    strataPa ss o B = V ss.length (fun i => (ss.getD i (0, 0)).1) (fun i => (ss.getD i (0, 0)).2)
      (fun i => (window (o + strataOff ss i) (ss.getD i (0, 0)).1 B).card) := by
  rw [strataPa_eq_prod, V]
  refine prod_congr rfl fun i hi => ?_
  have hmem : ss.getD i (0, 0) ∈ ss := by
    rw [List.getD_eq_getElem _ _ (mem_range.1 hi)]
    exact List.getElem_mem _
  exact choose_div_eq_gr (hk _ hmem) ((card_le_card (window_subset _ _ B)).trans_eq (card_range _))

/-- **The deployed count bound at a strata draw**: for strata with the code's sizes and counts, every set of
`countBound strata δ + 1` or more slots escapes with probability at most some `c < δ`. -/
theorem countBound_strataPa {ss : List (ℕ × ℕ)} (hk : ∀ p ∈ ss, p.2 ≤ p.1) (strata : Array Core.Stratum)
    (hlen : strata.size = ss.length)
    (hs : ∀ i (h₁ : i < strata.size) (h₂ : i < ss.length), strata[i].size = ss[i].1 ∧ strata[i].k = ss[i].2)
    (δ : Core.Ratio) (hδ : 0 < δ.num) (hδd : 0 < δ.den) :
    ∃ c : ℝ, 0 ≤ c ∧ c < (δ.num : ℝ) / δ.den ∧ ∀ B ⊆ range (ss.map Prod.fst).sum,
      countBound strata δ + 1 ≤ B.card → strataPa ss 0 B ≤ c := by
  have hnk : ∀ i (h : i < strata.size), strata[i].size = (ss.getD i (0, 0)).1 ∧
      strata[i].k = (ss.getD i (0, 0)).2 := fun i h => by
    have h₂ : i < ss.length := hlen ▸ h
    rw [List.getD_eq_getElem _ _ h₂]
    exact hs i h h₂
  obtain ⟨c, hc0, hcδ, hc⟩ := countBound_good strata _ _ hnk δ hδ hδd
  refine ⟨c, hc0, hcδ, fun B hB ht => ?_⟩
  rw [strataPa_eq_V ss hk 0 B, ← hlen]
  refine hc _ (fun i _ => (card_le_card (window_subset _ _ B)).trans_eq (card_range _)) ?_
  rw [hlen, sum_card_window ss 0 B,
    filter_true_of_mem fun v hv => ⟨Nat.zero_le v, by have := mem_range.1 (hB hv); omega⟩]
  exact ht

/-- **The deployed count bound is sound at the claims' design**: for a design that is a strata draw over the
partition's proof blocks (`Design.Pa_eq_strataPa`), with the code's strata, `miss(countBound strata δ + 1) < δ`: an
accepted transcript holds at most `countBound strata δ` faulty units except with probability `δ`. -/
theorem countBound_miss {d : Design} {P : Partition} {ss : List (ℕ × ℕ)}
    (hPa : ∀ B ⊆ range P.blocks.length, d.Pa P B = strataPa ss 0 B) (hN : (ss.map Prod.fst).sum = P.blocks.length)
    (hk : ∀ p ∈ ss, p.2 ≤ p.1) (strata : Array Core.Stratum) (hlen : strata.size = ss.length)
    (hs : ∀ i (h₁ : i < strata.size) (h₂ : i < ss.length), strata[i].size = ss[i].1 ∧ strata[i].k = ss[i].2)
    (δ : Core.Ratio) (hδ : 0 < δ.num) (hδd : 0 < δ.den) :
    d.miss P (countBound strata δ + 1) < (δ.num : ℝ) / δ.den := by
  obtain ⟨c, hc0, hcδ, hc⟩ := countBound_strataPa hk strata hlen hs δ hδ hδd
  refine lt_of_le_of_lt (Real.iSup_le (fun B => ?_) hc0) hcδ
  obtain ⟨B, hB, ht⟩ := B
  rw [hPa B hB]
  exact hc B (hN ▸ hB) ht

theorem length_flatten_eq_sum (c : List (List (List ℕ))) : c.flatten.length = (strataSizes c).sum := by
  rw [List.length_flatten, strataSizes]

/-- Each class's strata cover its proof blocks, under a well-formed design. -/
theorem strata_cover {d : Design} {P : Partition} (hw : d.WellFormed P) {sss : List (List (ℕ × ℕ))}
    (hss : ∀ p ∈ (d.rules.zip P.classes).zip sss, p.1.1.strata (strataSizes p.1.2) = some p.2) :
    ∀ p ∈ (d.rules.zip P.classes).zip sss, (p.2.map Prod.fst).sum = p.1.2.flatten.length := fun p hp => by
  rw [OneStage.strata_sum (hss p hp) (hw.2 p.1 (List.of_mem_zip hp).1), length_flatten_eq_sum]

/-- **A design whose rules are strata draws is one strata draw** over the partition's proof blocks: the classes'
strata in order. -/
theorem Design.Pa_eq_strataPa {d : Design} {P : Partition} (hw : d.WellFormed P)
    {sss : List (List (ℕ × ℕ))} (hs : sss.length = P.classes.length)
    (hss : ∀ p ∈ (d.rules.zip P.classes).zip sss, p.1.1.strata (strataSizes p.1.2) = some p.2)
    (B : Finset ℕ) : d.Pa P B = strataPa sss.flatten 0 B := by
  refine Design.Pa_strata d P sss hw.1 hs (fun p hp => ⟨strata_cover hw hss p hp, fun B' hB' => ?_⟩) B
  rw [OneStage.Pa_strata (hss p hp) (by rwa [← length_flatten_eq_sum])]

theorem sum_flatten_strata : ∀ (rs : List OneStage) (cs : List (List (List (List ℕ)))) (sss : List (List (ℕ × ℕ))),
    rs.length = cs.length → sss.length = cs.length →
    (∀ p ∈ (rs.zip cs).zip sss, (p.2.map Prod.fst).sum = p.1.2.flatten.length) →
    (sss.flatten.map Prod.fst).sum = cs.flatten.flatten.length
  | [], [], [], _, _, _ => rfl
  | [], [], _ :: _, _, h, _ => absurd h (by simp)
  | [], _ :: _, _, h, _, _ => absurd h (by simp)
  | _ :: _, [], _, h, _, _ => absurd h (by simp)
  | _ :: _, _ :: _, [], _, h, _ => absurd h (by simp)
  | r :: rs, c :: cs, ss :: sss, hl, hs, h => by
    have h0 := h ((r, c), ss) (by simp)
    dsimp only at h0
    rw [List.flatten_cons, List.map_append, List.sum_append, h0,
      sum_flatten_strata rs cs sss (by simpa using hl) (by simpa using hs) fun p hp => h p (by simp [hp]),
      List.flatten_cons, List.flatten_append, List.length_append]

/-- The strata of a design that is one strata draw cover the partition's proof blocks. -/
theorem sum_strata_eq_length {d : Design} {P : Partition} (hw : d.WellFormed P) {sss : List (List (ℕ × ℕ))}
    (hs : sss.length = P.classes.length)
    (hss : ∀ p ∈ (d.rules.zip P.classes).zip sss, p.1.1.strata (strataSizes p.1.2) = some p.2) :
    (sss.flatten.map Prod.fst).sum = P.blocks.length :=
  sum_flatten_strata d.rules P.classes sss hw.1 hs (strata_cover hw hss)

/-- **The deployed count bound is sound at every design that draws strata**: `miss(countBound strata δ + 1) < δ` for
the classes' strata in order, given as the code's. -/
theorem countBound_miss_design {d : Design} {P : Partition} (hw : d.WellFormed P)
    {sss : List (List (ℕ × ℕ))} (hs : sss.length = P.classes.length)
    (hss : ∀ p ∈ (d.rules.zip P.classes).zip sss, p.1.1.strata (strataSizes p.1.2) = some p.2)
    (strata : Array Core.Stratum) (hlen : strata.size = sss.flatten.length)
    (hst : ∀ i (h₁ : i < strata.size) (h₂ : i < sss.flatten.length),
      strata[i].size = sss.flatten[i].1 ∧ strata[i].k = sss.flatten[i].2)
    (δ : Core.Ratio) (hδ : 0 < δ.num) (hδd : 0 < δ.den) :
    d.miss P (countBound strata δ + 1) < (δ.num : ℝ) / δ.den := by
  refine countBound_miss (fun B _ => Design.Pa_eq_strataPa hw hs hss B) (sum_strata_eq_length hw hs hss)
    (fun p hp => ?_) strata hlen hst δ hδ hδd
  obtain ⟨ss, hss', hp'⟩ := List.mem_flatten.1 hp
  obtain ⟨q, hq, rfl⟩ : ∃ q ∈ (d.rules.zip P.classes).zip sss, q.2 = ss := by
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hss'
    have hi' : i < ((d.rules.zip P.classes).zip sss).length := by simp [hw.1, hs] at hi ⊢; omega
    exact ⟨_, List.getElem_mem hi', by simp⟩
  exact OneStage.strata_le (hss q hq) (hw.2 q.1 (List.of_mem_zip hq).1) p hp'

/-- `Bound.strataOf` reads the uniform `k`-subset as one stratum of every unit, the strata `OneStage.strata` gives
`subset`. -/
theorem strataOf_subset (N k : ℕ) :
    SampledProofs.Bound.strataOf N (.subset k) = some #[{ name := "unit", size := N, k }] := rfl

end Claims.Sampling.Count
