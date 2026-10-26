import Mathlib
import Definitions.Accounting.Guarantees.Actual.WipeDraw
import Proofs.Accounting.Actual.WipePrf

/-!
# The store held read by read, on the deployed labels and the deployed index draw

`Accounting.Guarantees.Actual.WipeReadHeldPrfDraw`. The sampler accepts `2^w − (2^w mod N)` words
(`card_accepted`), and gives each index at most `⌊2^w / N⌋` of them (`card_fiber_le`), a uniform share. So on uniform
words, the coins on which every read's word is accepted and the event holds at the indices they give are at most the
event's share at uniform indices, and the rest is the words some read rejects, at most `n (2^w mod N) / 2^w` of them by
Bernoulli's inequality (`pr_drawEvent_le`). The deployed event implies the distinguisher's on the coin words
(`drawEvent_of`), which costs the coin function's advantage (`pr_draw_le`), and `WipeReadHeldPrf` bounds the event at
uniform indices with the table drawn last (`pr_comp_equiv`). At the deployed wipe the sampler's term is below `2^-54`
(`drawRejection_deployed`).
-/

namespace Accounting.SecurityProofs.Actual

open Finset
open scoped ENNReal

/-- The event read off the words: when every read's word gives an index, `E` at those indices, and true otherwise.
`Actual.drawDistinguisher` is it at `heldEvent`. -/
def drawEvent {Y : Type*} {n N w : ℕ} (E : Y × (Fin n → Fin N) → Prop) (x : Y × (Fin n → Pous.Bits w)) : Prop :=
  ∀ c : Fin n → Fin N, (∀ j, Actual.wordIndex N (x.2 j) = some (c j)) → E (x.1, c)

theorem wordIndex_isSome {N w : ℕ} (u : Pous.Bits w) :
    (Actual.wordIndex N u).isSome ↔ Pous.bitsVal u < 2 ^ w - 2 ^ w % N := by
  unfold Actual.wordIndex
  split_ifs with h <;> simp [h]

theorem wordIndex_eq_some {N w : ℕ} {u : Pous.Bits w} {i : Fin N} (h : Actual.wordIndex N u = some i) :
    Pous.bitsVal u < 2 ^ w - 2 ^ w % N ∧ Pous.bitsVal u % N = i := by
  unfold Actual.wordIndex at h
  split_ifs at h with h'
  · exact ⟨h', congrArg Fin.val (Option.some_injective _ h)⟩

theorem limit_eq (N w : ℕ) : 2 ^ w - 2 ^ w % N = N * (2 ^ w / N) := by
  have h := Nat.div_add_mod (2 ^ w) N
  omega

open Classical in
/-- **The sampler accepts `2^w − (2^w mod N)` words.** -/
theorem card_accepted (N w : ℕ) :
    (univ.filter fun u : Pous.Bits w => (Actual.wordIndex N u).isSome).card = 2 ^ w - 2 ^ w % N := by
  have : (univ.filter fun u : Pous.Bits w => (Actual.wordIndex N u).isSome) =
      (univ.filter fun i : Fin (2 ^ w) => (i : ℕ) < 2 ^ w - 2 ^ w % N).map (Pous.bitsEquiv w).symm.toEmbedding := by
    ext u
    simp [Finset.mem_map_equiv, wordIndex_isSome, Pous.bitsVal]
  rw [this, card_map, Fin.card_filter_val_lt]
  exact min_eq_right (Nat.sub_le _ _)

open Classical in
/-- **Each index has at most `⌊2^w / N⌋` accepted words**: an accepted word is fixed by its quotient by `N`, which is
below `⌊2^w / N⌋`. -/
theorem card_fiber_le {N w : ℕ} (i : Fin N) :
    (univ.filter fun u : Pous.Bits w => Actual.wordIndex N u = some i).card ≤ 2 ^ w / N := by
  have hN : 0 < N := i.pos
  calc _ ≤ (range (2 ^ w / N)).card := by
        refine card_le_card_of_injOn (fun u => Pous.bitsVal u / N) (fun u hu => ?_) (fun u hu v hv huv => ?_)
        · obtain ⟨h1, -⟩ := wordIndex_eq_some (mem_filter.1 hu).2
          rw [limit_eq] at h1
          rw [coe_range, Set.mem_Iio, Nat.div_lt_iff_lt_mul hN]
          linarith [Nat.mul_comm N (2 ^ w / N)]
        · obtain ⟨-, hu2⟩ := wordIndex_eq_some (mem_filter.1 hu).2
          obtain ⟨-, hv2⟩ := wordIndex_eq_some (mem_filter.1 hv).2
          apply bitsVal_injective w
          rw [← Nat.div_add_mod (Pous.bitsVal u) N, ← Nat.div_add_mod (Pous.bitsVal v) N, hu2, hv2]
          exact congrArg (fun q => N * q + (i : ℕ)) huv
    _ = 2 ^ w / N := card_range _

open Classical in
theorem card_allAccepted (n N w : ℕ) :
    (univ.filter fun u : Fin n → Pous.Bits w => ∀ j, (Actual.wordIndex N (u j)).isSome).card =
      (2 ^ w - 2 ^ w % N) ^ n := by
  have : (univ.filter fun u : Fin n → Pous.Bits w => ∀ j, (Actual.wordIndex N (u j)).isSome) =
      Fintype.piFinset fun _ => univ.filter fun x : Pous.Bits w => (Actual.wordIndex N x).isSome := by
    ext u
    simp [Fintype.mem_piFinset]
  rw [this, Fintype.card_piFinset_const, card_accepted]

open Classical in
theorem card_draws_le {n N w : ℕ} (c : Fin n → Fin N) :
    (univ.filter fun u : Fin n → Pous.Bits w => ∀ j, Actual.wordIndex N (u j) = some (c j)).card ≤
      (2 ^ w / N) ^ n := by
  have : (univ.filter fun u : Fin n → Pous.Bits w => ∀ j, Actual.wordIndex N (u j) = some (c j)) =
      Fintype.piFinset fun j => univ.filter fun x : Pous.Bits w => Actual.wordIndex N x = some (c j) := by
    ext u
    simp [Fintype.mem_piFinset]
  rw [this, Fintype.card_piFinset]
  calc _ ≤ ∏ _j : Fin n, 2 ^ w / N := prod_le_prod fun j _ => card_fiber_le (c j)
    _ = _ := by simp

open Classical in
/-- **The count on uniform words**: the words some read rejects, for every `y`, and for each `(y, c)` in `E` the words
that give the indices `c`. -/
theorem card_drawEvent_le {Y : Type} [Fintype Y] {n N w : ℕ} (E : Y × (Fin n → Fin N) → Prop) :
    (univ.filter (drawEvent (w := w) E)).card ≤
      Fintype.card Y * ((2 ^ w) ^ n - (2 ^ w - 2 ^ w % N) ^ n) + (univ.filter E).card * (2 ^ w / N) ^ n := by
  set A := univ.filter fun u : Fin n → Pous.Bits w => ∀ j, (Actual.wordIndex N (u j)).isSome with hAdef
  set F := fun c : Fin n → Fin N =>
    univ.filter fun u : Fin n → Pous.Bits w => ∀ j, Actual.wordIndex N (u j) = some (c j) with hFdef
  have hsub : univ.filter (drawEvent (w := w) E) ⊆
      (univ ×ˢ Aᶜ) ∪ (univ.filter E).biUnion (fun yc => {yc.1} ×ˢ F yc.2) := by
    rintro ⟨y, u⟩ hyu
    have hyu' := (mem_filter.1 hyu).2
    by_cases hA : u ∈ A
    · apply mem_union_right
      have hA' : ∀ j, (Actual.wordIndex N (u j)).isSome := (mem_filter.1 hA).2
      let c : Fin n → Fin N := fun j => (Actual.wordIndex N (u j)).get (hA' j)
      have hc : ∀ j, Actual.wordIndex N (u j) = some (c j) := fun j => (Option.some_get _).symm
      refine mem_biUnion.2 ⟨(y, c), mem_filter.2 ⟨mem_univ _, hyu' c hc⟩, ?_⟩
      exact mem_product.2 ⟨mem_singleton_self _, mem_filter.2 ⟨mem_univ _, hc⟩⟩
    · exact mem_union_left _ (mem_product.2 ⟨mem_univ _, mem_compl.2 hA⟩)
  have hA : Aᶜ.card = (2 ^ w) ^ n - (2 ^ w - 2 ^ w % N) ^ n := by
    rw [card_compl, hAdef, card_allAccepted]
    simp [Fintype.card_bool]
  calc _ ≤ _ := card_le_card hsub
    _ ≤ (univ ×ˢ Aᶜ).card + ((univ.filter E).biUnion (fun yc => {yc.1} ×ˢ F yc.2)).card := card_union_le _ _
    _ ≤ Fintype.card Y * Aᶜ.card + ∑ yc ∈ univ.filter E, ({yc.1} ×ˢ F yc.2).card := by
        rw [card_product, card_univ]
        exact add_le_add le_rfl card_biUnion_le
    _ ≤ _ := by
      rw [hA]
      refine add_le_add le_rfl ?_
      calc ∑ yc ∈ univ.filter E, ({yc.1} ×ˢ F yc.2).card ≤ ∑ _yc ∈ univ.filter E, (2 ^ w / N) ^ n :=
            sum_le_sum fun yc _ => by rw [card_product, card_singleton, one_mul]; exact card_draws_le yc.2
        _ = _ := by rw [sum_const, smul_eq_mul]

open Classical in
theorem pr_le_ofReal {α : Type} [Fintype α] (P : α → Prop) {β : ℝ} (hβ : 0 ≤ β)
    (h : ((univ.filter P).card : ℝ) ≤ β * Fintype.card α) : Pous.pr P ≤ ENNReal.ofReal β := by
  unfold Pous.pr
  apply ENNReal.div_le_of_le_mul
  rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul hβ]
  exact ENNReal.ofReal_le_ofReal h

open Classical in
theorem ofReal_le_pr {α : Type} [Fintype α] (P : α → Prop) :
    ENNReal.ofReal (((univ.filter P).card : ℝ) / Fintype.card α) ≤ Pous.pr P := by
  unfold Pous.pr
  rcases Nat.eq_zero_or_pos (Fintype.card α) with h | h
  · simp [h]
  · rw [ENNReal.ofReal_div_of_pos (by exact_mod_cast h), ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]

/-- **Bernoulli's inequality** as the rejected share: `M^n − L^n ≤ n (M − L)/M · M^n`. -/
theorem pow_sub_pow_le {M L : ℝ} (hM : 0 < M) (hL : 0 ≤ L) (n : ℕ) :
    M ^ n - L ^ n ≤ n * (M - L) / M * M ^ n := by
  have hb := one_add_mul_le_pow (a := -((M - L) / M)) (by
    have : (M - L) / M ≤ 1 := (div_le_one hM).2 (by linarith)
    linarith) n
  have h1 : 1 + -((M - L) / M) = L / M := by field_simp; ring
  rw [h1, div_pow] at hb
  have hMn : 0 < M ^ n := pow_pos hM n
  rw [le_div_iff₀ hMn] at hb
  calc M ^ n - L ^ n ≤ M ^ n - (1 + n * -((M - L) / M)) * M ^ n := by linarith
    _ = n * (M - L) / M * M ^ n := by ring

open Classical in
/-- **The sampler on uniform words**: the event read off the words is at most the event at uniform indices plus
`n (2^w mod N) / 2^w`. -/
theorem pr_drawEvent_le {Y : Type} [Fintype Y] {n N w : ℕ} (E : Y × (Fin n → Fin N) → Prop) :
    Pous.pr (drawEvent (w := w) E) ≤ Pous.pr E + ENNReal.ofReal (n * (2 ^ w % N : ℕ) / 2 ^ w) := by
  set ρ : ℝ := n * (2 ^ w % N : ℕ) / 2 ^ w with hρdef
  set e : ℝ := ((univ.filter E).card : ℝ) / (Fintype.card (Y × (Fin n → Fin N))) with hedef
  have hρ : 0 ≤ ρ := by positivity
  have he : 0 ≤ e := by positivity
  refine (pr_le_ofReal _ (add_nonneg he hρ) ?_).trans
    ((ENNReal.ofReal_add he hρ).le.trans (add_le_add (ofReal_le_pr E) le_rfl))
  have hc := card_drawEvent_le (w := w) E
  have hmod : 2 ^ w % N ≤ 2 ^ w := Nat.mod_le _ _
  have hLM : (2 ^ w - 2 ^ w % N) ^ n ≤ (2 ^ w) ^ n := Nat.pow_le_pow_left (Nat.sub_le _ _) n
  have hQ : 2 ^ w / N * N ≤ 2 ^ w := Nat.div_mul_le_self _ _
  have hE : (univ.filter E).card ≤ Fintype.card Y * N ^ n := by
    have := card_le_univ (univ.filter E)
    simpa [Fintype.card_prod, Fintype.card_fun] using this
  have hcR : ((univ.filter (drawEvent (w := w) E)).card : ℝ) ≤
      (Fintype.card Y : ℝ) * (((2 : ℝ) ^ w) ^ n - ((2 ^ w - 2 ^ w % N : ℕ) : ℝ) ^ n) +
        (univ.filter E).card * ((2 ^ w / N : ℕ) : ℝ) ^ n := by
    have := (Nat.cast_le (α := ℝ)).2 hc
    push_cast [Nat.cast_sub hLM] at this
    exact this
  have hcard : Fintype.card (Y × (Fin n → Pous.Bits w)) = Fintype.card Y * (2 ^ w) ^ n := by simp
  have hcard' : Fintype.card (Y × (Fin n → Fin N)) = Fintype.card Y * N ^ n := by simp
  rw [hcard]
  rw [hcard'] at hedef
  push_cast
  set y : ℝ := (Fintype.card Y : ℝ)
  set k : ℝ := ((univ.filter E).card : ℝ)
  set M : ℝ := (2 : ℝ) ^ w
  set L : ℝ := ((2 ^ w - 2 ^ w % N : ℕ) : ℝ)
  set Q : ℝ := ((2 ^ w / N : ℕ) : ℝ)
  have hM : 0 < M := by positivity
  have hL0 : 0 ≤ L := by positivity
  have hR : ((2 ^ w % N : ℕ) : ℝ) = M - L := by
    simp only [L, M]; rw [Nat.cast_sub hmod]; push_cast; ring
  have hy : 0 ≤ y := by positivity
  have hk0 : 0 ≤ k := by positivity
  have hQN : Q * N ≤ M := by simp only [Q, M]; exact_mod_cast hQ
  have hii : y * (M ^ n - L ^ n) ≤ ρ * (y * M ^ n) := by
    rw [hρdef, hR]
    have := pow_sub_pow_le hM hL0 n
    nlinarith [this, hy]
  have hi : k * Q ^ n ≤ e * (y * M ^ n) := by
    rcases eq_or_lt_of_le (show (0 : ℝ) ≤ y * (N : ℝ) ^ n by positivity) with h0 | hpos
    · have : k = 0 := by
        have hk : k ≤ y * (N : ℝ) ^ n := by simp only [k, y]; exact_mod_cast hE
        linarith
      rw [this, zero_mul]; positivity
    · have hed : e = k / (y * (N : ℝ) ^ n) := by rw [hedef]; push_cast; rfl
      rw [hed, div_mul_eq_mul_div, le_div_iff₀ hpos]
      have hQNn : Q ^ n * (N : ℝ) ^ n ≤ M ^ n := by
        rw [← mul_pow]; exact pow_le_pow_left₀ (by positivity) hQN n
      nlinarith [hQNn, hk0, hy, mul_nonneg hk0 hy]
  calc _ ≤ y * (M ^ n - L ^ n) + k * Q ^ n := hcR
    _ ≤ ρ * (y * M ^ n) + e * (y * M ^ n) := add_le_add hii hi
    _ = (e + ρ) * (y * M ^ n) := by ring

open Classical in
theorem pr_mono {α : Type} [Fintype α] {P Q : α → Prop} (h : ∀ x, P x → Q x) : Pous.pr P ≤ Pous.pr Q := by
  unfold Pous.pr
  exact ENNReal.div_le_div_right (Nat.cast_le.2 (card_le_card (monotone_filter_right _ fun x _ => h x))) _

open Classical in
/-- A probability is the same over the coins in another order. -/
theorem pr_comp_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (P : β → Prop) :
    Pous.pr (fun x => P (e x)) = Pous.pr P := by
  unfold Pous.pr
  rw [Fintype.card_congr e, card_equiv e (t := univ.filter P) fun x => by simp]

/-- **The deployed event implies the distinguisher's on the coin words**: an accepted first word fixes the index. -/
theorem drawEvent_of {Y : Type} {n N w κ : ℕ} (coins : Pous.Bits κ → Fin n → Pous.Bits w)
    (rest : Pous.Bits κ → Fin n → Fin N) (E : Y × (Fin n → Fin N) → Prop) (x : Y × Pous.Bits κ)
    (h : E (x.1, fun j => Actual.drawIndex (rest x.2 j) (coins x.2 j))) : drawEvent E (x.1, coins x.2) := by
  intro c hc
  have : (fun j => Actual.drawIndex (rest x.2 j) (coins x.2 j)) = c := funext fun j => by
    simp only [Actual.drawIndex, hc j, Option.getD_some]
  rwa [this] at h

/-- **The index step**: at the drawn indices, any event is at most its chance at uniform indices, plus the sampler's
term, plus the coin function's advantage against the event read off the words. -/
theorem pr_draw_le {Y : Type} [Fintype Y] {n N w κ : ℕ} (coins : Pous.Bits κ → Fin n → Pous.Bits w)
    (rest : Pous.Bits κ → Fin n → Fin N) (E : Y × (Fin n → Fin N) → Prop) :
    Pous.pr (fun x : Y × Pous.Bits κ => E (x.1, fun j => Actual.drawIndex (rest x.2 j) (coins x.2 j))) ≤
      Pous.pr E + ENNReal.ofReal (n * (2 ^ w % N : ℕ) / 2 ^ w) +
        Pous.StreamedWipe.prfAdvantage coins (drawEvent E) :=
  (pr_mono fun x h => drawEvent_of coins rest E x h).trans (pr_le_add_prfAdvantage coins _ (pr_drawEvent_le E))

/-- The coins `((ω, c), s)` in the order `((ω, s), c)`. -/
def swapLast (A B C : Type*) : (A × B) × C ≃ (A × C) × B where
  toFun x := ((x.1.1, x.2), x.1.2)
  invFun x := ((x.1.1, x.2), x.1.2)
  left_inv _ := rfl
  right_inv _ := rfl

/-- **At the deployed wipe** (`N = 40,625,000,000` labels, 104-bit words, 74,000 reads) the sampler's term is below
`2^-54`. -/
theorem drawRejection_deployed :
    (74000 : ℝ) * ((2 ^ 104 % 40625000000 : ℕ) : ℝ) / 2 ^ 104 < 1 / 2 ^ 54 := by
  norm_num

theorem WipeReadHeldPrfDraw : Guarantees.Actual.WipeReadHeldPrfDraw := by
  intro Ω L Z _ _ n W N b κ κ' w prf coins rest wipe S ℓ X Y V Ch D ℓmax a m K arrive ζ quiet T hS hlat hlive hlv hq
    hlate
  have hlab := WipeReadHeldPrf prf wipe S ℓ X Y V Ch D ℓmax a m K arrive ζ quiet T hS hlat hlive hlv hq hlate
  have hE : Pous.pr (fun y : (Ω × Pous.Bits κ) × (Fin n → Fin N) =>
        Actual.heldEvent S X V Ch arrive T m K ((y.1.1, prf y.1.2), y.2)) ≤
      ENNReal.ofReal (Real.exp (-((K : ℝ) * (1 - ((a : ℝ) - 1) / N))) +
          n * T.card * 2 ^ m * (N.choose a : ℝ) / 2 ^ (a * b)) +
        Pous.StreamedWipe.prfAdvantage prf (Actual.heldDistinguisher S X V Ch arrive T m K) := by
    rw [← pr_comp_equiv (swapLast Ω (Fin n → Fin N) (Pous.Bits κ))]
    exact hlab
  calc _ ≤ _ := pr_draw_le coins rest
        (fun y : (Ω × Pous.Bits κ) × (Fin n → Fin N) =>
          Actual.heldEvent S X V Ch arrive T m K ((y.1.1, prf y.1.2), y.2))
    _ ≤ _ := add_le_add (add_le_add hE le_rfl) le_rfl
    _ = _ := add_right_comm _ _ _

end Accounting.SecurityProofs.Actual
