import Proofs.Pouw.Prob
import Proofs.Pouw.Lead.Tail.Across
import Proofs.Pouw.Lead.Tail.Columns
import Proofs.Pouw.Lead.Tail.Slots
import Proofs.Pouw.Lead.Tail.Route

/-!
# The coin-branch tail at the three points and decode

`coinTail_le`: at the three points and decode, for every call of `n, K ≥ 4,096` and at most `2^30` tiles, a call's
repeated tiles exceed the coin-branch mean by more than the slack on at most a `2^−42` share of the coins, whatever
its data.

The blocks of one step, coin tile and column block don't read disjoint coins. A word is a prefix sum over steps, so a
tile at step `t` reads every earlier step's coin lanes and tag bits `b`, and step 0's tag weights `G` (one set a
column, shared by every step and coin tile) and `a₁` bits (one set a row, shared by every step and column block). Two
tiles at one step, column block and coin tile have rows of one parity, so their coin lanes cancel, and only `G`,
`a₁` and `b` separate them. So the repeats split over the blocks (`repeats_le_blocks`): the blocks that repeat an
earlier tile (`across`), each through its own coin lanes, and the blocks with two equal tiles of their own
(`within`), which sit in the column blocks with one (`hitCols`, through each column block's own `G`), each with few
such slots (`hitSlots`, through each slot's fresh `a₁` or `b` bits). Each count is sequentially dominated in its own
order (`count_tail_fiber`).

`split_of` is the split: when the across blocks are fewer than `a`, the hit column blocks fewer than `k` and every
column block's hit slots fewer than `x`, the repeats are at most `ρ (a − 1) + (ρ − 1)(w − 1)`, below the tail's least
repeats. `pr_split_le` bounds its event by the three counts' tails, and `pr_tail_le` is both at a call. A route
(`Tail.route_le`) picks the four counts.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- A count that never exceeds `size` reaches `r` within `cut size λ r` when it does within `λ^r / r!`. -/
theorem pr_le_cut {α : Type} [Fintype α] {f : α → ℕ} {size r : ℕ} {l : ℝ} (hsize : ∀ ω, f ω ≤ size)
    (h : Pouw.pr (fun ω => r ≤ f ω) ≤ l ^ r / r.factorial) : Pouw.pr (fun ω => r ≤ f ω) ≤ cut size l r := by
  unfold cut
  split_ifs with hr
  · exact (Pouw.pr_eq_zero fun ω hω => absurd (hω.trans (hsize ω)) (not_le.2 hr)).le
  · exact h

/-- **The split**: on coins whose repeats pass the tail's least count, the across blocks reach `a`, the hit column
blocks reach `k`, or some column block's hit slots reach `x`, for counts meeting the route's two constraints. -/
theorem split_of (L : Pouw.Lead.Layout) (htile : L.tile = 64) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) {a k w x : ℕ}
    (h1 : rho L * (a - 1) + (rho L - 1) * (w - 1) < tailCount L) (h2 : (k - 1) * (x - 1) < w) (ω : Coins L)
    (hω : coinBranchMean L + coinSlack L < tileWords * (repeats L κ ω : ℚ)) :
    (a ≤ (across L κ ω).card ∨ k ≤ (hitCols L κ ω).card) ∨
      ∃ cb ∈ (Finset.univ : Finset (Fin L.colBlocks)), x ≤ (hitSlots L κ ω cb).card := by
  by_contra hc
  simp only [not_or, not_le, not_exists, Finset.mem_univ, true_and] at hc
  obtain ⟨⟨hX, hG⟩, hB⟩ := hc
  have hW : (within L κ ω).card ≤ w - 1 :=
    Nat.le_sub_one_of_lt (lt_of_le_of_lt ((within_le L κ ω fun cb => Nat.le_sub_one_of_lt (hB cb)).trans
      (Nat.mul_le_mul_right _ (Nat.le_sub_one_of_lt hG))) h2)
  exact absurd ((tailCount_le hω).trans ((repeats_le_blocks L htile κ ω).trans
    (Nat.add_le_add (Nat.mul_le_mul_left _ (Nat.le_sub_one_of_lt hX)) (Nat.mul_le_mul_left _ hW)))) (not_le.2 h1)

/-- **The split's union bound**: the three counts' tails, each cut at its index set's size, bound the split's event. -/
theorem pr_split_le (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) {a k x : ℕ}
    (hX : Pouw.pr (fun ω : Coins L => a ≤ (across L κ ω).card) ≤ lamX L ^ a / a.factorial)
    (hG : Pouw.pr (fun ω : Coins L => k ≤ (hitCols L κ ω).card) ≤ lamG L ^ k / k.factorial)
    (hB : ∀ cb, Pouw.pr (fun ω : Coins L => x ≤ (hitSlots L κ ω cb).card) ≤ lamB L ^ x / x.factorial) :
    Pouw.pr (fun ω : Coins L => (a ≤ (across L κ ω).card ∨ k ≤ (hitCols L κ ω).card) ∨
        ∃ cb ∈ (Finset.univ : Finset (Fin L.colBlocks)), x ≤ (hitSlots L κ ω cb).card) ≤
      cut (blkCount L) (lamX L) a + cut L.colBlocks (lamG L) k + L.colBlocks * cut (L.S * L.tiles) (lamB L) x := by
  have hX' := pr_le_cut (size := blkCount L)
    (fun ω => (Finset.card_le_univ _).trans_eq (by simp [blkCount, Blk, mul_assoc])) hX
  have hG' := pr_le_cut (size := L.colBlocks) (fun ω => (Finset.card_le_univ _).trans_eq (Fintype.card_fin _)) hG
  have hB' := fun cb => pr_le_cut (size := L.S * L.tiles) (fun ω => (Finset.card_le_univ _).trans_eq (by simp)) (hB cb)
  calc _ ≤ (Pouw.pr (fun ω : Coins L => a ≤ (across L κ ω).card) +
          Pouw.pr (fun ω : Coins L => k ≤ (hitCols L κ ω).card)) +
        ∑ cb, Pouw.pr (fun ω : Coins L => x ≤ (hitSlots L κ ω cb).card) :=
        (Pouw.pr_or_le _ _).trans (add_le_add (Pouw.pr_or_le _ _)
          (Pouw.Proofs.pr_exists_le_sum _ fun cb ω => x ≤ (hitSlots L κ ω cb).card))
    _ ≤ (cut (blkCount L) (lamX L) a + cut L.colBlocks (lamG L) k) +
        ∑ _cb : Fin L.colBlocks, cut (L.S * L.tiles) (lamB L) x :=
        add_le_add (add_le_add hX' hG') (Finset.sum_le_sum fun cb _ => hB' cb)
    _ = _ := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- **The coin-branch tail at a call, through the split**: at a call whose coin tiles are whole (or which has one row
block), counts meeting the route's two constraints bound the tail by the route's three terms. -/
theorem pr_tail_le (T m n K : ℕ) (hm : 64 ∣ m ∨ m ≤ 16)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ) {a k w x : ℕ}
    (h1 : rho (lead T m n K) * (a - 1) + (rho (lead T m n K) - 1) * (w - 1) < tailCount (lead T m n K))
    (h2 : (k - 1) * (x - 1) < w) :
    Pouw.pr (fun ω : Coins (lead T m n K) =>
        coinBranchMean (lead T m n K) + coinSlack (lead T m n K) < tileWords * (repeats (lead T m n K) κ ω : ℚ)) ≤
      cut (blkCount (lead T m n K)) (lamX (lead T m n K)) a + cut (lead T m n K).colBlocks (lamG (lead T m n K)) k +
        (lead T m n K).colBlocks * cut ((lead T m n K).S * (lead T m n K).tiles) (lamB (lead T m n K)) x :=
  (Pouw.pr_mono (split_of _ rfl κ h1 h2)).trans (pr_split_le _ κ (across_tail T m n K κ a)
    (columns_tail T m n K κ k) fun cb => slots_tail T m n K hm κ cb x)

/-- **The coin-branch tail at the three points and decode.** -/
theorem coinTail_le (T n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K) :
    ∀ L ∈ [pt512x64 n K, pt512x512 n K, pt8192 n K, lead T 1 n K],
      L.rowBlocks * L.colBlocks * L.S ≤ 2 ^ 30 → coinTail L ≤ 1 / 2 ^ 42 := by
  intro L hL htiles
  obtain ⟨a, k, w, x, h1, h2, h3⟩ := route_le T n K hn hK L hL htiles
  obtain ⟨T', m, rfl, hm⟩ : ∃ T' m, L = lead T' m n K ∧ (64 ∣ m ∨ m ≤ 16) := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hL
    rcases hL with rfl | rfl | rfl | rfl
    · exact ⟨512, 64, rfl, Or.inl (by norm_num)⟩
    · exact ⟨512, 512, rfl, Or.inl (by norm_num)⟩
    · exact ⟨8192, 8192, rfl, Or.inl (by norm_num)⟩
    · exact ⟨T, 1, rfl, Or.inr (by norm_num)⟩
  exact ciSup_le fun κ => (pr_tail_le T' m n K hm κ h1 h2).trans h3

end Pouw.SecurityProofs.Lead.Tail
