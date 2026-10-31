import Proofs.Pouw.Lead.PStar.Lanes

/-!
# The coin-lane walks under `-rot`: what the 127 weight opens (proofs)

An entry's word after some steps is a value fixed before the draw (its data's products, tags and code) plus its lane's
increments summed over the steps. Every test of a word is a test on such a sum: the word against a value fixed before
the draw, a revisit (the word at one step against the word at an earlier one, the sum over the steps between), and a
coincidence with another row's word of the same tile, whose lanes read the same coins.

* **A sum over independent steps is capped by any one step** (`steps_sum_fiber_le`): with each step's digits drawn
  independently from `S`, a sum over a set of steps containing step `u₀` takes a value on at most `P·|S|^(T−1)` of the
  `|S|^T` draws when no value has more than `P` preimages under step `u₀`'s term. Its values may be in any additive
  group, so `k` words summed as one vector are one sum.
* **An entry's own walk** (`rotWalk_share`): over any nonempty set of steps, so for the word and every revisit, the
  summed increments hit a value fixed before the draw on at most a `p* = 65/2,097,152` share of the draws.
* **Rows of opposite classes** (`crossStep_fiber_le`, `crossStep_fiber_attained`, `crossStep_share`,
  `crossWalk_share`): an even row and an odd row of one tile read the same coins, and their step increments differ by
  `(c − c') + (b − b') + 126·(s₀ − s₁)`. As `b ∈ {0, 1}`, `b + 126·s₀` fixes `b` and `s₀`, so no value is that
  difference for more than `2·128²` of the `4·128³` draws `(b, b', s)`, a `1/256` share, and `c − c'` is it for exactly
  that many. The same share caps the difference summed over any nonempty set of steps.
* **`k` words of one row pair** (`crossTile_fiber_le`, `crossTile_fiber_attained`, `crossTile_share`,
  `crossTileWalk_share`): the two rows' bits are shared by all their columns, so the columns' coincidences are not
  independent. Column 0's equation fixes `b` and its `s₀`, and each other column's equation then fixes its own `s₀`, so
  `k` targets are hit together on at most a `(1/2)·(1/128)^k` share of the draws, `2^−29` for a 4-word tile, at one
  step (where `c − c'` in every column attains it) or summed over any nonempty set of steps.
* **Over the tile** (`rot_sum_hits_le`): with row bits drawn per row and coins per column, every row reading its own
  class by parity, targets fixed before the draw are hit on at most a `65/2,097,152` share of the entries on average.
-/

namespace Pouw.SecurityProofs.Lead.PStar

open Finset Pouw.Dimension.Alphabet

section Steps

variable {ι γ G : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq γ] [AddCommGroup G] [DecidableEq G]

/-- **A sum over independent steps is capped by any one step**: if no value has more than `P` preimages under step
`u₀`'s term, a sum over steps `I ∋ u₀` takes any value on at most `P·|S|^(T−1)` of the draws `w : ι → S`. -/
theorem steps_sum_fiber_le (S : Finset γ) (f : ι → γ → G) (I : Finset ι) {u₀ : ι} (hu₀ : u₀ ∈ I) {P : ℕ}
    (hP : ∀ v, (S.filter fun x => f u₀ x = v).card ≤ P) (v : G) :
    ((Fintype.piFinset fun _ : ι => S).filter fun w => ∑ u ∈ I, f u (w u) = v).card
      ≤ P * S.card ^ (Fintype.card ι - 1) := by
  rcases S.eq_empty_or_nonempty with hS | ⟨z₀, hz₀⟩
  · rw [card_eq_zero.mpr]
    · exact Nat.zero_le _
    · refine eq_empty_of_forall_notMem fun w hw => ?_
      rw [mem_filter, Fintype.mem_piFinset] at hw
      have := hw.1 u₀
      rw [hS] at this
      exact notMem_empty _ this
  have hmaps : Set.MapsTo (fun w => Function.update w u₀ z₀)
      (((Fintype.piFinset fun _ : ι => S).filter fun w => ∑ u ∈ I, f u (w u) = v : Finset (ι → γ)) : Set (ι → γ))
      (((Fintype.piFinset fun _ : ι => S).filter fun y => y u₀ = z₀ : Finset (ι → γ)) : Set (ι → γ)) := by
    intro w hw
    simp only [mem_coe, mem_filter, Fintype.mem_piFinset] at hw ⊢
    refine ⟨fun u => ?_, Function.update_self _ _ _⟩
    by_cases hu : u = u₀
    · subst hu; rw [Function.update_self]; exact hz₀
    · rw [Function.update_of_ne hu]; exact hw.1 u
  rw [card_eq_sum_card_fiberwise hmaps]
  calc ∑ y ∈ (Fintype.piFinset fun _ : ι => S).filter (fun y => y u₀ = z₀),
        (((Fintype.piFinset fun _ : ι => S).filter fun w => ∑ u ∈ I, f u (w u) = v).filter
          fun w => Function.update w u₀ z₀ = y).card
      ≤ ∑ _y ∈ (Fintype.piFinset fun _ : ι => S).filter (fun y => y u₀ = z₀), P := by
        refine sum_le_sum fun y _ => ?_
        refine (card_le_card_of_injOn (fun w => w u₀) (fun w hw => ?_) ?_).trans
          (hP (v - ∑ u ∈ I.erase u₀, f u (y u)))
        · simp only [mem_coe, mem_filter, Fintype.mem_piFinset] at hw
          obtain ⟨⟨hwS, hwv⟩, hwy⟩ := hw
          simp only [mem_coe, mem_filter]
          refine ⟨hwS u₀, ?_⟩
          rw [← add_sum_erase I _ hu₀] at hwv
          have hrest : ∑ u ∈ I.erase u₀, f u (y u) = ∑ u ∈ I.erase u₀, f u (w u) := by
            refine sum_congr rfl fun u hu => ?_
            rw [← hwy, Function.update_of_ne (ne_of_mem_erase hu)]
          rw [hrest, ← hwv, add_sub_cancel_right]
        · intro w hw w' hw' he
          simp only [mem_coe, mem_filter] at hw hw'
          funext u
          by_cases hu : u = u₀
          · subst hu; exact he
          · have h := congrFun (hw.2.trans hw'.2.symm) u
            rwa [Function.update_of_ne hu, Function.update_of_ne hu] at h
    _ = P * S.card ^ (Fintype.card ι - 1) := by
        rw [sum_const, Fintype.card_filter_piFinset_const_eq_of_mem S u₀ hz₀, nsmul_eq_mul, Nat.cast_id, mul_comm]

omit [DecidableEq γ] in
theorem card_steps (S : Finset γ) : (Fintype.piFinset fun _ : ι => S).card = S.card ^ Fintype.card ι := by
  rw [Fintype.card_piFinset, prod_const, card_univ]

end Steps

theorem card_bits : ({0, 1} : Finset ℤ).card = 2 := by decide

/-- **An entry's own walk**: over any nonempty set of steps `I`, each step drawing the row's bit and the tile's coins
afresh, the summed increments hit a value fixed before the draw on at most a `65/2,097,152` share of the draws. -/
theorem rotWalk_share {T : ℕ} (c : ℤ) (i : ℕ) (I : Finset (Fin T)) (hI : I.Nonempty) (v : ℤ) :
    2097152 * ((Fintype.piFinset fun _ : Fin T => ({0, 1} : Finset ℤ) ×ˢ triples).filter
        fun w => ∑ u ∈ I, rotLane c i (w u).1 (w u).2 = v).card ≤
      65 * (Fintype.piFinset fun _ : Fin T => ({0, 1} : Finset ℤ) ×ˢ triples).card := by
  obtain ⟨u₀, hu₀⟩ := hI
  have h := steps_sum_fiber_le (({0, 1} : Finset ℤ) ×ˢ triples) (fun _ x => rotLane c i x.1 x.2) I hu₀
    (P := 130) (fun w => rotLane_fiber_le c i w) v
  have hS : (({0, 1} : Finset ℤ) ×ˢ triples).card = 4194304 := by
    rw [card_product, card_bits, card_triples]; norm_num
  rw [card_steps, hS, Fintype.card_fin]
  rw [hS, Fintype.card_fin] at h
  obtain ⟨T', rfl⟩ : ∃ T', T = T' + 1 := ⟨T - 1, by have := u₀.2; omega⟩
  rw [Nat.add_sub_cancel] at h
  calc 2097152 * _ ≤ 2097152 * (130 * 4194304 ^ T') := Nat.mul_le_mul_left _ h
    _ = 65 * 4194304 ^ (T' + 1) := by ring

/-- Two rows of opposite classes reading the same coins: their step increments differ by
`(c − c') + (b − b') + 126·(s₀ − s₁)`. -/
theorem rotLane_sub_cross {c c' : ℤ} {i i' : ℕ} (hi : i % 2 = 0) (hi' : i' % 2 = 1) (b b' : ℤ)
    (s : ℤ × ℤ × ℤ) :
    rotLane c i b s - rotLane c' i' b' s = (c - c') + (b - b') + 126 * (s.1 - s.2.1) := by
  have hi'' : ¬ i' % 2 = 0 := by omega
  simp only [rotLane, rotSum, hi, hi'', ite_true, ite_false, evenSum, oddSum]
  ring

/-- **The cross-class step difference's largest fiber**: no value is it for more than `2·128²` of the draws
`(b, b', s)`. -/
theorem crossStep_fiber_le (c c' : ℤ) {i i' : ℕ} (hi : i % 2 = 0) (hi' : i' % 2 = 1) (v : ℤ) :
    (((({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ triples).filter fun x =>
      rotLane c i x.1.1 x.2 - rotLane c' i' x.1.2 x.2 = v).card ≤ 32768 := by
  calc _ ≤ (({0, 1} : Finset ℤ) ×ˢ (coins ×ˢ coins)).card :=
        card_le_card_of_injOn (fun x => (x.1.2, x.2.2.1, x.2.2.2)) ?_ ?_
    _ = 32768 := by rw [card_product, card_product, card_coins, card_bits]
  · intro x hx
    simp only [mem_coe, mem_filter, mem_product, mem_triples] at hx
    simp only [mem_coe, mem_product]
    exact ⟨hx.1.1.2, hx.1.2.2.1, hx.1.2.2.2⟩
  · rintro ⟨⟨b, b'⟩, s0, s1, s2⟩ hx ⟨⟨β, β'⟩, t0, t1, t2⟩ hy he
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl, rfl⟩ := he
    simp only [mem_coe, mem_filter, mem_product, mem_insert, mem_singleton, rotLane_sub_cross hi hi'] at hx hy
    have : b = β ∧ s0 = t0 := by omega
    obtain ⟨rfl, rfl⟩ := this
    rfl

/-- **It is attained**: `c − c'` is the difference for `2·128²` draws, `b = b'` and `s₀ = s₁`. -/
theorem crossStep_fiber_attained (c c' : ℤ) {i i' : ℕ} (hi : i % 2 = 0) (hi' : i' % 2 = 1) :
    32768 ≤ (((({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ triples).filter fun x =>
      rotLane c i x.1.1 x.2 - rotLane c' i' x.1.2 x.2 = c - c').card := by
  calc 32768 = (({0, 1} : Finset ℤ) ×ˢ (coins ×ˢ coins)).card := by
        rw [card_product, card_product, card_coins, card_bits]
    _ ≤ _ := card_le_card_of_injOn (fun y => ((y.1, y.1), y.2.1, y.2.1, y.2.2)) ?_ ?_
  · intro y hy
    simp only [mem_coe, mem_product] at hy
    simp only [mem_coe, mem_filter, mem_product, mem_triples, rotLane_sub_cross hi hi']
    exact ⟨⟨⟨hy.1, hy.1⟩, hy.2.1, hy.2.1, hy.2.2⟩, by ring⟩
  · rintro ⟨b, x, y⟩ - ⟨b', x', y'⟩ - he
    simp only [Prod.mk.injEq] at he
    obtain ⟨⟨rfl, -⟩, rfl, -, rfl⟩ := he
    rfl

/-- **The cross-class share**: an even and an odd row of one tile have step words differing by a value fixed before
the draw on at most a `1/256` share of the `4·128³` draws, and on exactly that share for `c − c'`. -/
theorem crossStep_share (c c' : ℤ) {i i' : ℕ} (hi : i % 2 = 0) (hi' : i' % 2 = 1) :
    (∀ v, 256 * ((((({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ triples).filter fun x =>
        rotLane c i x.1.1 x.2 - rotLane c' i' x.1.2 x.2 = v).card) ≤
        ((({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ triples).card) ∧
      256 * ((((({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ triples).filter fun x =>
        rotLane c i x.1.1 x.2 - rotLane c' i' x.1.2 x.2 = c - c').card) ≥
        ((({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ triples).card := by
  have hS : ((({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ triples).card = 8388608 := by
    rw [card_product, card_product, card_bits, card_triples]; norm_num
  rw [hS]
  refine ⟨fun v => ?_, ?_⟩
  · have := crossStep_fiber_le c c' hi hi' v; omega
  · have := crossStep_fiber_attained c c' hi hi'; omega

/-- **The cross-class walk**: summed over any nonempty set of steps, the two rows' increments differ by a value fixed
before the draw on at most a `1/256` share of the draws. -/
theorem crossWalk_share {T : ℕ} (c c' : ℤ) {i i' : ℕ} (hi : i % 2 = 0) (hi' : i' % 2 = 1)
    (I : Finset (Fin T)) (hI : I.Nonempty) (v : ℤ) :
    256 * ((Fintype.piFinset fun _ : Fin T => (({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ triples).filter
        fun w => ∑ u ∈ I, (rotLane c i (w u).1.1 (w u).2 - rotLane c' i' (w u).1.2 (w u).2) = v).card ≤
      (Fintype.piFinset fun _ : Fin T => (({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ triples).card := by
  obtain ⟨u₀, hu₀⟩ := hI
  have h := steps_sum_fiber_le ((({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ triples)
    (fun _ x => rotLane c i x.1.1 x.2 - rotLane c' i' x.1.2 x.2) I hu₀ (P := 32768)
    (fun w => crossStep_fiber_le c c' hi hi' w) v
  have hS : ((({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ triples).card = 8388608 := by
    rw [card_product, card_product, card_bits, card_triples]; norm_num
  rw [card_steps, hS, Fintype.card_fin]
  rw [hS, Fintype.card_fin] at h
  obtain ⟨T', rfl⟩ : ∃ T', T = T' + 1 := ⟨T - 1, by have := u₀.2; omega⟩
  rw [Nat.add_sub_cancel] at h
  calc 256 * _ ≤ 256 * (32768 * 8388608 ^ T') := Nat.mul_le_mul_left _ h
    _ = 8388608 ^ (T' + 1) := by ring

/-- One step's digits for `k` words of an even and an odd row: the two rows' bits and the `k` columns' coins. -/
abbrev tileDigits (k : ℕ) : Finset ((ℤ × ℤ) × (Fin k → ℤ × ℤ × ℤ)) :=
  (({0, 1} : Finset ℤ) ×ˢ ({0, 1} : Finset ℤ)) ×ˢ Fintype.piFinset fun _ : Fin k => triples

/-- The `k` words' step differences between an even row `i` and an odd row `i'`. -/
def crossTile (k : ℕ) (c c' : ℤ) (i i' : ℕ) (x : (ℤ × ℤ) × (Fin k → ℤ × ℤ × ℤ)) : Fin k → ℤ :=
  fun j => rotLane c i x.1.1 (x.2 j) - rotLane c' i' x.1.2 (x.2 j)

theorem card_tileDigits (k : ℕ) : (tileDigits k).card = 4 * 2097152 ^ k := by
  rw [card_product, card_product, card_bits, card_steps, card_triples, Fintype.card_fin]; norm_num

/-- **`k` words of one row pair at one step**: column 0's equation fixes `b` and its `s₀`, and each other column's its
own `s₀`, so no `k` targets are hit together on more than `2·(128²)^k` of the draws. -/
theorem crossTile_fiber_le {k : ℕ} (hk : 0 < k) (c c' : ℤ) {i i' : ℕ} (hi : i % 2 = 0) (hi' : i' % 2 = 1)
    (v : Fin k → ℤ) : ((tileDigits k).filter fun x => crossTile k c c' i i' x = v).card ≤ 2 * 16384 ^ k := by
  calc _ ≤ (({0, 1} : Finset ℤ) ×ˢ Fintype.piFinset fun _ : Fin k => coins ×ˢ coins).card :=
        card_le_card_of_injOn (fun x => (x.1.2, fun j => ((x.2 j).2.1, (x.2 j).2.2))) ?_ ?_
    _ = 2 * 16384 ^ k := by
        rw [card_product, card_steps, card_product, card_coins, card_bits, Fintype.card_fin]
  · intro x hx
    simp only [tileDigits, mem_coe, mem_filter, mem_product, Fintype.mem_piFinset] at hx
    simp only [mem_coe, mem_product, Fintype.mem_piFinset]
    refine ⟨hx.1.1.2, fun j => ?_⟩
    have := mem_triples.mp (hx.1.2 j)
    exact ⟨this.2.1, this.2.2⟩
  · rintro ⟨⟨b, b'⟩, s⟩ hx ⟨⟨β, β'⟩, t⟩ hy he
    simp only [tileDigits, mem_coe, mem_filter, mem_product, mem_insert, mem_singleton] at hx hy
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, he⟩ := he
    have hs : ∀ j, (s j).2.1 = (t j).2.1 ∧ (s j).2.2 = (t j).2.2 := fun j => by
      have := congrFun he j
      simp only [Prod.mk.injEq] at this
      exact this
    have hxj : ∀ j, (c - c') + (b - b') + 126 * ((s j).1 - (s j).2.1) = v j := fun j => by
      have := congrFun hx.2 j
      simpa only [crossTile, rotLane_sub_cross hi hi'] using this
    have hyj : ∀ j, (c - c') + (β - b') + 126 * ((t j).1 - (t j).2.1) = v j := fun j => by
      have := congrFun hy.2 j
      simpa only [crossTile, rotLane_sub_cross hi hi'] using this
    have hb : b = β := by
      have h0 := hxj ⟨0, hk⟩
      have h0' := hyj ⟨0, hk⟩
      have := (hs ⟨0, hk⟩).1
      omega
    subst hb
    have hst : s = t := funext fun j => by
      have h1 := hxj j
      have h2 := hyj j
      have h3 := (hs j).1
      exact Prod.ext (by omega) (Prod.ext (hs j).1 (hs j).2)
    rw [hst]

/-- **It is attained**: `c − c'` in every column is hit by `2·(128²)^k` draws, `b = b'` and `s₀ = s₁` in each column. -/
theorem crossTile_fiber_attained (k : ℕ) (c c' : ℤ) {i i' : ℕ} (hi : i % 2 = 0) (hi' : i' % 2 = 1) :
    2 * 16384 ^ k ≤ ((tileDigits k).filter fun x => crossTile k c c' i i' x = fun _ => c - c').card := by
  calc 2 * 16384 ^ k = (({0, 1} : Finset ℤ) ×ˢ Fintype.piFinset fun _ : Fin k => coins ×ˢ coins).card := by
        rw [card_product, card_steps, card_product, card_coins, card_bits, Fintype.card_fin]
    _ ≤ _ := card_le_card_of_injOn (fun y => ((y.1, y.1), fun j => ((y.2 j).1, (y.2 j).1, (y.2 j).2))) ?_ ?_
  · intro y hy
    simp only [mem_coe, mem_product, Fintype.mem_piFinset] at hy
    simp only [tileDigits, mem_coe, mem_filter, mem_product, Fintype.mem_piFinset, mem_triples]
    refine ⟨⟨⟨hy.1, hy.1⟩, fun j => ⟨(hy.2 j).1, (hy.2 j).1, (hy.2 j).2⟩⟩, funext fun j => ?_⟩
    simp only [crossTile, rotLane_sub_cross hi hi']
    ring
  · rintro ⟨b, g⟩ - ⟨b', g'⟩ - he
    simp only [Prod.mk.injEq] at he
    obtain ⟨⟨rfl, -⟩, he⟩ := he
    refine Prod.ext rfl (funext fun j => ?_)
    have := congrFun he j
    simp only [Prod.mk.injEq] at this
    exact Prod.ext this.1 this.2.2

/-- **The `k`-word share at one step**: `k` words of an even and an odd row of one tile all hit targets fixed before
the draw on at most a `(1/2)·(1/128)^k` share of the draws (`2^−29` for `k = 4`), and on exactly that share for
`c − c'` in every column. -/
theorem crossTile_share {k : ℕ} (hk : 0 < k) (c c' : ℤ) {i i' : ℕ} (hi : i % 2 = 0) (hi' : i' % 2 = 1) :
    (∀ v, 2 * 128 ^ k * ((tileDigits k).filter fun x => crossTile k c c' i i' x = v).card ≤ (tileDigits k).card) ∧
      2 * 128 ^ k * ((tileDigits k).filter fun x => crossTile k c c' i i' x = fun _ => c - c').card ≥
        (tileDigits k).card := by
  rw [card_tileDigits]
  have hpow : 2 * 128 ^ k * (2 * 16384 ^ k) = 4 * 2097152 ^ k := by
    rw [show (16384 : ℕ) = 128 * 128 by norm_num, show (2097152 : ℕ) = 128 * 16384 by norm_num, mul_pow, mul_pow,
      show (16384 : ℕ) = 128 * 128 by norm_num, mul_pow]
    ring
  refine ⟨fun v => ?_, ?_⟩
  · calc 2 * 128 ^ k * _ ≤ 2 * 128 ^ k * (2 * 16384 ^ k) :=
          Nat.mul_le_mul_left _ (crossTile_fiber_le hk c c' hi hi' v)
      _ = 4 * 2097152 ^ k := hpow
  · calc 4 * 2097152 ^ k = 2 * 128 ^ k * (2 * 16384 ^ k) := hpow.symm
      _ ≤ 2 * 128 ^ k * _ := Nat.mul_le_mul_left _ (crossTile_fiber_attained k c c' hi hi')

/-- **The `k`-word walk**: summed over any nonempty set of steps, `k` words of an even and an odd row of one tile all
hit targets fixed before the draw on at most a `(1/2)·(1/128)^k` share of the draws. -/
theorem crossTileWalk_share {T k : ℕ} (hk : 0 < k) (c c' : ℤ) {i i' : ℕ} (hi : i % 2 = 0) (hi' : i' % 2 = 1)
    (I : Finset (Fin T)) (hI : I.Nonempty) (v : Fin k → ℤ) :
    2 * 128 ^ k * ((Fintype.piFinset fun _ : Fin T => tileDigits k).filter
        fun w => ∑ u ∈ I, crossTile k c c' i i' (w u) = v).card ≤
      (Fintype.piFinset fun _ : Fin T => tileDigits k).card := by
  obtain ⟨u₀, hu₀⟩ := hI
  have h := steps_sum_fiber_le (tileDigits k) (fun _ x => crossTile k c c' i i' x) I hu₀ (P := 2 * 16384 ^ k)
    (fun w => crossTile_fiber_le hk c c' hi hi' w) v
  rw [card_steps, card_tileDigits, Fintype.card_fin] at *
  obtain ⟨T', rfl⟩ : ∃ T', T = T' + 1 := ⟨T - 1, by have := u₀.2; omega⟩
  rw [Nat.add_sub_cancel] at h
  have hpow : 2 * 128 ^ k * (2 * 16384 ^ k) = 4 * 2097152 ^ k := by
    rw [show (2097152 : ℕ) = 128 * 16384 by norm_num, mul_pow]
    ring
  calc 2 * 128 ^ k * _ ≤ 2 * 128 ^ k * (2 * 16384 ^ k * (4 * 2097152 ^ k) ^ T') := Nat.mul_le_mul_left _ h
    _ = 2 * 128 ^ k * (2 * 16384 ^ k) * (4 * 2097152 ^ k) ^ T' := by ring
    _ = (4 * 2097152 ^ k) ^ (T' + 1) := by rw [hpow]; ring

variable {r s : ℕ}

/-- The entries of a tile whose increment equals its target, row `i` weighted by its class `i % 2`. -/
def rotHits (c : ℤ) (v : Fin r → Fin s → ℤ) (D : (Fin r → ℤ) × (Fin s → ℤ × ℤ × ℤ)) : ℕ :=
  (univ.filter fun e : Fin r × Fin s => rotLane c e.1 (D.1 e.1) (D.2 e.2) = v e.1 e.2).card

/-- **Over the tile**: with a bit drawn per row, a coin triple per column read by every row, and each row weighted by
its class, targets fixed before the draw are hit on at most a `65/2,097,152` share of the `r·s` entries on average. -/
theorem rot_sum_hits_le (c : ℤ) (hr : 0 < r) (hs : 0 < s) (v : Fin r → Fin s → ℤ) :
    2097152 * ∑ D ∈ draws ({0, 1} : Finset ℤ) triples r s, rotHits c v D ≤
      65 * (r * s) * (draws ({0, 1} : Finset ℤ) triples r s).card := by
  have hentry : ∀ e : Fin r × Fin s, ∑ D ∈ draws ({0, 1} : Finset ℤ) triples r s,
      (if rotLane c e.1 (D.1 e.1) (D.2 e.2) = v e.1 e.2 then 1 else 0) ≤
        130 * (2 ^ (r - 1) * 2097152 ^ (s - 1)) := fun e => by
    rw [← card_filter, entry_card (rotLane c e.1) {0, 1} triples e.1 e.2 (v e.1 e.2), card_bits, card_triples]
    exact Nat.mul_le_mul_right _ (rotLane_fiber_le c e.1 _)
  simp only [rotHits, card_filter]
  rw [sum_comm, card_draws, card_bits, card_triples]
  obtain ⟨r', rfl⟩ : ∃ r', r = r' + 1 := ⟨r - 1, by omega⟩
  obtain ⟨s', rfl⟩ : ∃ s', s = s' + 1 := ⟨s - 1, by omega⟩
  simp only [Nat.add_sub_cancel] at hentry
  calc 2097152 * ∑ e : Fin (r' + 1) × Fin (s' + 1), ∑ D ∈ draws ({0, 1} : Finset ℤ) triples (r' + 1) (s' + 1),
        (if rotLane c e.1 (D.1 e.1) (D.2 e.2) = v e.1 e.2 then 1 else 0)
      ≤ 2097152 * ∑ _e : Fin (r' + 1) × Fin (s' + 1), 130 * (2 ^ r' * 2097152 ^ s') :=
        Nat.mul_le_mul_left _ (sum_le_sum fun e _ => hentry e)
    _ = 65 * ((r' + 1) * (s' + 1)) * (2 ^ (r' + 1) * (128 ^ 3) ^ (s' + 1)) := by
        rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, nsmul_eq_mul, Nat.cast_id]
        ring

end Pouw.SecurityProofs.Lead.PStar
