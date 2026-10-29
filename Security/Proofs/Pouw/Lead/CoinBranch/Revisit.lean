import Proofs.Pouw.Lead.PStar.AlphabetProofs
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.FinCases

/-!
# Coin branch: revisits at tile grain (proofs)

A program that branches on the coins skips a tile only where it can predict the tile's value equal to one it holds.
At fixed coins the tiles' distinct values are at least their number less the equal pairs (`card_le_image_add_pairs`).
A revisit, a tile returning at step `t` to its value at an earlier step `t'` (chosen data walking a row back over values
it already holds), needs the tile's difference to vanish. That difference is the step-`t` coins' term plus
a term the step-`t` coins don't enter, which is fixed when those coins are drawn. So a revisit is bounded by the step-`t`
coins alone (`fresh_card_le`).

* **Fresh coins** (`fresh_card_le`): a condition `φ x = g y` on independent coordinates `x` and `y` holds on at most
  `N·|Y|` pairs when no value of `φ` has more than `N` preimages.
* **One tile, one step** (`tile_solutions_le`): a tile's rows each add their own bit `b_i` and every column its own lane
  triple `s_j`, weighted by the row's `A′_i`. If two of its rows' weights determine a triple, the column of `j₀` fixes
  every bit and then every other column's triple: at most `|L|` of the `2^|I|·|L|^|J|` fresh coins hit any target.
* **`-rot`** (`rot_pair_inj`, `rot_tile_solutions_le`, `rot_revisit_card_le`): `(127, 1, 1)` and `(1, 127, 1)`
  determine an odd 7-bit triple, so a
  16 × 8 tile holding both row classes returns to any value on at most `2^21` of its `2^184` fresh coins: `2^−163`,
  whatever the data and the other coins.
-/

namespace Pouw.SecurityProofs.Lead.CoinBranch

open Finset Pouw.Dimension.Alphabet

/-- **Distinct values and equal pairs**: a set holds at most its distinct values plus its ordered pairs with equal
values. -/
theorem card_le_image_add_pairs {α X : Type} [LinearOrder α] [DecidableEq X] (s : Finset α) (f : α → X) :
    s.card ≤ (s.image f).card + ((s ×ˢ s).filter fun p => p.1 < p.2 ∧ f p.1 = f p.2).card := by
  induction s using Finset.induction_on_max with
  | empty => simp
  | insert a s ha ih =>
    have hsub : ((s ×ˢ s).filter fun p => p.1 < p.2 ∧ f p.1 = f p.2) ⊆
        ((insert a s ×ˢ insert a s).filter fun p => p.1 < p.2 ∧ f p.1 = f p.2) := fun p hp => by
      rw [mem_filter, mem_product] at hp ⊢
      exact ⟨⟨mem_insert_of_mem hp.1.1, mem_insert_of_mem hp.1.2⟩, hp.2⟩
    rw [image_insert, card_insert_of_notMem fun h => lt_irrefl a (ha a h)]
    by_cases hf : f a ∈ s.image f
    · rw [insert_eq_of_mem hf]
      obtain ⟨x, hx, hfx⟩ := mem_image.1 hf
      have hnot : (x, a) ∉ (s ×ˢ s).filter fun p => p.1 < p.2 ∧ f p.1 = f p.2 := fun h =>
        lt_irrefl a (ha a (mem_product.1 (mem_filter.1 h).1).2)
      have hsub' : insert (x, a) ((s ×ˢ s).filter fun p => p.1 < p.2 ∧ f p.1 = f p.2) ⊆
          ((insert a s ×ˢ insert a s).filter fun p => p.1 < p.2 ∧ f p.1 = f p.2) := by
        refine insert_subset_iff.2 ⟨?_, hsub⟩
        rw [mem_filter, mem_product]
        exact ⟨⟨mem_insert_of_mem hx, mem_insert_self a s⟩, ha x hx, hfx⟩
      have := card_le_card hsub'
      rw [card_insert_of_notMem hnot] at this
      omega
    · rw [card_insert_of_notMem hf]
      have := card_le_card hsub
      omega

/-- **Fresh coins**: a condition `φ x = g y` on independent coordinates holds on at most `N·|Y|` pairs when no value of
`φ` has more than `N` preimages in `X`. -/
theorem fresh_card_le {X Y Z : Type} [DecidableEq Z] (sX : Finset X) (sY : Finset Y) (φ : X → Z) (g : Y → Z) {N : ℕ}
    (hN : ∀ z, (sX.filter fun x => φ x = z).card ≤ N) :
    ((sX ×ˢ sY).filter fun p => φ p.1 = g p.2).card ≤ N * sY.card := by
  classical
  rw [card_eq_sum_card_fiberwise (f := Prod.snd) (t := sY) fun p hp => (mem_product.1 (mem_filter.1 hp).1).2]
  calc ∑ y ∈ sY, (((sX ×ˢ sY).filter fun p => φ p.1 = g p.2).filter fun p => p.2 = y).card
      ≤ ∑ _y ∈ sY, N := sum_le_sum fun y _ => by
        refine (card_le_card_of_injOn Prod.fst (fun p hp => ?_) fun p hp q hq h => ?_).trans (hN (g y))
        · simp only [mem_coe, mem_filter, mem_product] at hp ⊢
          exact ⟨hp.1.1.1, hp.2 ▸ hp.1.2⟩
        · simp only [mem_coe, mem_filter] at hp hq
          exact Prod.ext h (hp.2.trans hq.2.symm)
    _ = N * sY.card := by rw [sum_const, nsmul_eq_mul, Nat.cast_id, mul_comm]

/-- **One tile, one step**: rows `I` each add their own bit and columns `J` their own triple from `L`, row `i` weighting
it by `W i`. If rows `ie` and `io` together determine a triple, the solutions of `b_i + W i · s_j = c i j` number at
most `|L|`: column `j₀`'s triple fixes every bit, and the bits fix every other triple. -/
theorem tile_solutions_le {I J : Type} [Fintype I] [Fintype J] [DecidableEq I] [DecidableEq J]
    (L : Finset (Fin 3 → ℤ)) (W : I → Fin 3 → ℤ) (ie io : I)
    (hinj : ∀ s ∈ L, ∀ s' ∈ L, ∑ k, W ie k * s k = ∑ k, W ie k * s' k →
      ∑ k, W io k * s k = ∑ k, W io k * s' k → s = s')
    (j₀ : J) (c : I → J → ℤ) :
    ((Fintype.piFinset (fun _ : I => ({0, 1} : Finset ℤ)) ×ˢ Fintype.piFinset (fun _ : J => L)).filter
      fun p => (fun i j => p.1 i + ∑ k, W i k * p.2 j k) = c).card ≤ L.card := by
  refine card_le_card_of_injOn (fun p => p.2 j₀) (fun p hp => ?_) fun p hp q hq he => ?_
  · simp only [mem_coe, mem_filter, mem_product, Fintype.mem_piFinset] at hp ⊢
    exact hp.1.2 j₀
  · simp only [mem_coe, mem_filter, mem_product, Fintype.mem_piFinset] at hp hq
    have he : p.2 j₀ = q.2 j₀ := he
    have hp' : ∀ i j, p.1 i + ∑ k, W i k * p.2 j k = c i j := fun i j => congrFun (congrFun hp.2 i) j
    have hq' : ∀ i j, q.1 i + ∑ k, W i k * q.2 j k = c i j := fun i j => congrFun (congrFun hq.2 i) j
    have hb : p.1 = q.1 := funext fun i => by
      have h1 := hp' i j₀
      have h2 := hq' i j₀
      rw [he] at h1
      linarith
    refine Prod.ext hb (funext fun j => hinj _ (hp.1.2 j) _ (hq.1.2 j) ?_ ?_)
    · have h1 := hp' ie j
      have h2 := hq' ie j
      rw [hb] at h1
      linarith
    · have h1 := hp' io j
      have h2 := hq' io j
      rw [hb] at h1
      linarith

theorem mem_oddLane_seven {x : ℤ} (hx : x ∈ oddLane 7) : ∃ k, -64 ≤ k ∧ k ≤ 63 ∧ x = 2 * k + 1 := by
  simp only [oddLane, mem_image, mem_Icc] at hx
  obtain ⟨k, ⟨h1, h2⟩, rfl⟩ := hx
  exact ⟨k, by norm_num at h1; omega, by norm_num at h2; omega, rfl⟩

/-- `-rot`'s row weights `A′`: `(127, 1, 1)` on even rows and `(1, 127, 1)` on odd rows. -/
def rotRow (i : ℕ) (k : Fin 3) : ℤ := if (k : ℕ) = i % 2 then 127 else 1

/-- **`-rot`'s two row classes determine the triple**: on odd lanes in `±127`, `(127, 1, 1) · s` and
`(1, 127, 1) · s` fix `s`. Their difference is `126 (s₀ − s₁)`, and then `128 s₀ + s₂` has one odd `s₀`. -/
theorem rot_pair_inj (s s' : Fin 3 → ℤ) (hs : ∀ k, s k ∈ oddLane 7) (hs' : ∀ k, s' k ∈ oddLane 7)
    (h1 : ∑ k, rotRow 0 k * s k = ∑ k, rotRow 0 k * s' k)
    (h2 : ∑ k, rotRow 1 k * s k = ∑ k, rotRow 1 k * s' k) : s = s' := by
  have w00 : rotRow 0 0 = 127 := by decide
  have w01 : rotRow 0 1 = 1 := by decide
  have w02 : rotRow 0 2 = 1 := by decide
  have w10 : rotRow 1 0 = 1 := by decide
  have w11 : rotRow 1 1 = 127 := by decide
  have w12 : rotRow 1 2 = 1 := by decide
  rw [Fin.sum_univ_three, Fin.sum_univ_three, w00, w01, w02] at h1
  rw [Fin.sum_univ_three, Fin.sum_univ_three, w10, w11, w12] at h2
  obtain ⟨a0, -, -, e0⟩ := mem_oddLane_seven (hs 0)
  obtain ⟨b0, -, -, f0⟩ := mem_oddLane_seven (hs' 0)
  obtain ⟨a2, a2l, a2u, e2⟩ := mem_oddLane_seven (hs 2)
  obtain ⟨b2, b2l, b2u, f2⟩ := mem_oddLane_seven (hs' 2)
  have q0 : s 0 = s' 0 := by omega
  have q1 : s 1 = s' 1 := by omega
  have q2 : s 2 = s' 2 := by omega
  funext k
  fin_cases k
  · exact q0
  · exact q1
  · exact q2

/-- The lane triples of `-rot`: three odd 7-bit lanes. Irreducible, since reducing membership in it enumerates all
`2^21` triples. -/
@[irreducible] def rotLanes : Finset (Fin 3 → ℤ) := Fintype.piFinset fun _ => oddLane 7

theorem mem_rotLanes {s : Fin 3 → ℤ} : s ∈ rotLanes ↔ ∀ k, s k ∈ oddLane 7 := by
  unfold rotLanes
  exact Fintype.mem_piFinset

theorem card_rotLanes : rotLanes.card = 2 ^ 21 := by
  have h7 : (oddLane 7).card = 2 ^ 7 := card_oddLane 6
  unfold rotLanes
  rw [Fintype.card_piFinset, prod_const, card_univ, Fintype.card_fin, h7]
  norm_num

/-- **`-rot`, one 16 × 8 tile, one step**: rows alternate the two classes, so of the `2^16·(2^21)^8 = 2^184` bits and
triples the step draws for the tile, at most `2^21` hit any target. -/
theorem rot_tile_solutions_le (c : Fin 16 → Fin 8 → ℤ) :
    ((Fintype.piFinset (fun _ : Fin 16 => ({0, 1} : Finset ℤ)) ×ˢ Fintype.piFinset (fun _ : Fin 8 => rotLanes)).filter
      fun p => (fun i j => p.1 i + ∑ k, rotRow i.val k * p.2 j k) = c).card ≤ 2 ^ 21 := by
  have hinj : ∀ s ∈ rotLanes, ∀ s' ∈ rotLanes, ∑ k, rotRow 0 k * s k = ∑ k, rotRow 0 k * s' k →
      ∑ k, rotRow 1 k * s k = ∑ k, rotRow 1 k * s' k → s = s' := by
    intro s hs s' hs' h1 h2
    rw [mem_rotLanes] at hs hs'
    exact rot_pair_inj s s' hs hs' h1 h2
  have h := tile_solutions_le rotLanes (fun i : Fin 16 => rotRow i.val) 0 1 hinj 0 c
  exact h.trans card_rotLanes.le

theorem card_rot_fresh :
    (Fintype.piFinset (fun _ : Fin 16 => ({0, 1} : Finset ℤ)) ×ˢ
      Fintype.piFinset (fun _ : Fin 8 => rotLanes)).card = 2 ^ 184 := by
  simp only [card_product, Fintype.card_piFinset, prod_const, card_univ, Fintype.card_fin, card_rotLanes]
  rw [show ({0, 1} : Finset ℤ).card = 2 by decide]
  norm_num

/-- **`-rot`, a revisit of one 16 × 8 tile**: whatever the data and the coins other than the step's own (`g`), the
tile's step-`t` difference `b_i + A′_i · s_j` meets `g` on at most a `2^−163` share of the coins. -/
theorem rot_revisit_card_le {Y : Type} (sY : Finset Y) (g : Y → Fin 16 → Fin 8 → ℤ) :
    2 ^ 163 * (((Fintype.piFinset (fun _ : Fin 16 => ({0, 1} : Finset ℤ)) ×ˢ
        Fintype.piFinset (fun _ : Fin 8 => rotLanes)) ×ˢ sY).filter fun p =>
        (fun i j => p.1.1 i + ∑ k, rotRow i.val k * p.1.2 j k) = g p.2).card ≤
      2 ^ 184 * sY.card := by
  have h := fresh_card_le (Fintype.piFinset (fun _ : Fin 16 => ({0, 1} : Finset ℤ)) ×ˢ
      Fintype.piFinset (fun _ : Fin 8 => rotLanes)) sY
    (fun x => fun (i : Fin 16) (j : Fin 8) => x.1 i + ∑ k, rotRow i.val k * x.2 j k) g rot_tile_solutions_le
  calc 2 ^ 163 * _ ≤ 2 ^ 163 * (2 ^ 21 * sY.card) := Nat.mul_le_mul_left _ h
    _ = 2 ^ 184 * sY.card := by ring

end Pouw.SecurityProofs.Lead.CoinBranch
