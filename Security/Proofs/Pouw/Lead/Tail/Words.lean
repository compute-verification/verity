import Proofs.Pouw.Lead.Tail.Blocks
import Proofs.Pouw.Lead.Tags.Repeats

/-!
# What the three fiber counts share

The lead commits whole MMA tiles (`lead_m`, `lead_n`) in 64-row coin tiles, so a tile's position `8 r + c` is the word
of its row `r` and column `c` (`tileOf_at`), and a coin tile's row blocks are `4 ct + a` for `a < 4`. A block that
repeats a tile of its own has two row blocks `a < b` of its coin tile whose tiles agree (`withinAt_pair`), and a coin
tile has at most `pairs` such pairs (`pairsAt_card`). A fiber count covers its event by pieces on each of which a map
is injective (`card_filter_le_cover`): for a slot's bits, their values off one row block (`offBlock`).

The three tails read a tile's coins through `tileOf_congr` and `coinTerm_congr`: a tile reads its rows' and columns'
coin terms at its step and before, and a coin term reads its row's tag bits at its step, its row's `a₁` and its
column's `G` at step 0, and its coin tile's lanes at its step and column. Blocks and slots are numbered in mixed
radix (`mul_add_lt`, `mul_add_inj`).
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Finset Pouw.Lead

theorem lead_m (T m n K : ℕ) : (lead T m n K).m = 16 * (lead T m n K).rowBlocks := by
  simp only [Layout.rowBlocks, lead, padRows]; omega

theorem lead_n (T m n K : ℕ) : (lead T m n K).n = 8 * (lead T m n K).colBlocks := by
  simp only [Layout.colBlocks, lead, padCols]; omega

theorem lead_tiles (T m n K : ℕ) : (lead T m n K).tiles = ((lead T m n K).rowBlocks + 3) / 4 := by
  simp only [Layout.tiles, Layout.rowBlocks, lead, padRows]; omega

/-- A tile's position `8 r + c` is the word of its row `r` and column `c`. -/
theorem tileOf_at {L : Pouw.Lead.Layout} (hLm : L.m = 16 * L.rowBlocks) (hLn : L.n = 8 * L.colBlocks)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) {rb cb : ℕ} (hrb : rb < L.rowBlocks) (hcb : cb < L.colBlocks)
    (t : Fin L.S) {r c : ℕ} (hr : r < 16) (hc : c < 8) :
    tileOf L κ ω rb cb t ⟨8 * r + c, by unfold tileWords; omega⟩ =
      word L κ ω ⟨16 * rb + r, by omega⟩ t ⟨8 * cb + c, by omega⟩ := by
  have h1 : (8 * r + c) / 8 = r := by omega
  have h2 : (8 * r + c) % 8 = c := by omega
  unfold tileOf
  rw [dite_eq_left ⟨by rw [h1]; omega, by rw [h2]; omega⟩]
  congr 1 <;> exact Fin.ext (by simp only [h1, h2])

/-- The pairs `a < b` of row blocks of coin tile `ct`, both in the call. -/
def pairsAt (L : Pouw.Lead.Layout) (ct : ℕ) : Finset (ℕ × ℕ) :=
  (range 4 ×ˢ range 4).filter fun p => p.1 < p.2 ∧ 4 * ct + p.2 < L.rowBlocks

theorem pairsAt_card (L : Pouw.Lead.Layout) (ct : ℕ) : #(pairsAt L ct) ≤ pairs L := by
  have key : ∀ k ≤ 4, #((range k ×ˢ range k).filter fun p : ℕ × ℕ => p.1 < p.2) = k * (k - 1) / 2 := by decide
  calc #(pairsAt L ct) ≤ #((range (rho L) ×ˢ range (rho L)).filter fun p : ℕ × ℕ => p.1 < p.2) := by
        refine card_le_card fun p hp => ?_
        simp only [pairsAt, mem_filter, mem_product, mem_range] at hp ⊢
        unfold rho
        omega
    _ = pairs L := key _ (min_le_left _ _)

/-- **A block's own repeat is a pair's**: two of its coin tile's row blocks `a < b` whose tiles agree. -/
theorem withinAt_pair {L : Pouw.Lead.Layout} {κ : Fin L.m → Fin L.S → Fin L.n → ℤ} {ω : Coins L} {β : Blk L}
    (h : withinAt L κ ω β) : ∃ p ∈ pairsAt L β.2.1,
      tileOf L κ ω (4 * β.2.1 + p.1) β.2.2 β.1 = tileOf L κ ω (4 * β.2.1 + p.2) β.2.2 β.1 := by
  obtain ⟨rb, hrb, rb', hrb', h1, h2, hne, heq⟩ := h
  have e1 : 4 * (β.2.1 : ℕ) + rb % 4 = rb := by omega
  have e2 : 4 * (β.2.1 : ℕ) + rb' % 4 = rb' := by omega
  rcases lt_or_gt_of_ne hne with hlt | hlt
  · refine ⟨(rb % 4, rb' % 4), ?_, ?_⟩
    · simp only [pairsAt, mem_filter, mem_product, mem_range]; omega
    · simp only [e1, e2]; exact heq
  · refine ⟨(rb' % 4, rb % 4), ?_, ?_⟩
    · simp only [pairsAt, mem_filter, mem_product, mem_range]; omega
    · simp only [e1, e2]; exact heq.symm

/-- **A cover by pinned pieces**: if `P` holds only where some piece `Q i` (`i ∈ s`) does, and on each piece some map
to `W` is injective, `P` holds on at most `|s| · |W|` points. Over a type as large as a slot's bits, the elaborator
unfolds `Finset.univ` when it checks a membership in it, so the fiber counts reason through this lemma, about their
pieces' predicates only. -/
theorem card_filter_le_cover {V W I : Type} [Fintype V] [Fintype W] (P : V → Prop) [DecidablePred P] (s : Finset I)
    (Q : I → V → Prop) (hP : ∀ v, P v → ∃ i ∈ s, Q i v)
    (hQ : ∀ i ∈ s, ∃ f : V → W, ∀ v w, Q i v → Q i w → f v = f w → v = w) :
    #(univ.filter P) ≤ #s * Fintype.card W := by
  classical
  calc #(univ.filter P) ≤ #(s.biUnion fun i => univ.filter (Q i)) := card_le_card fun v hv => by
        obtain ⟨i, hi, hq⟩ := hP v (mem_filter.1 hv).2
        exact mem_biUnion.2 ⟨i, hi, mem_filter.2 ⟨mem_univ _, hq⟩⟩
    _ ≤ ∑ i ∈ s, #(univ.filter (Q i)) := card_biUnion_le
    _ ≤ ∑ _i ∈ s, Fintype.card W := sum_le_sum fun i hi => by
        obtain ⟨f, hf⟩ := hQ i hi
        rw [← card_univ]
        exact card_le_card_of_injOn f (fun _ _ => mem_coe.2 (mem_univ _)) fun v hv w hw h =>
          hf v w (mem_filter.1 hv).2 (mem_filter.1 hw).2 h
    _ = #s * Fintype.card W := by rw [sum_const, smul_eq_mul]

/-- A function on `Fin 64` off block `b`'s 16 indices. -/
def offBlock {X : Type} (b : ℕ) (v : Fin 64 → X) : Fin 48 → X :=
  fun k => v ⟨if (k : ℕ) < 16 * b then k else k + 16, by split_ifs <;> omega⟩

theorem offBlock_eq {X : Type} {b : ℕ} (hb : b < 4) {v w : Fin 64 → X} (h : offBlock b v = offBlock b w)
    (k : Fin 64) (hk : (k : ℕ) / 16 ≠ b) : v k = w k := by
  have hk' : (if (k : ℕ) < 16 * b then (k : ℕ) else k - 16) < 48 := by split_ifs <;> omega
  have := congrFun h ⟨_, hk'⟩
  simp only [offBlock] at this
  convert this using 2 <;> exact Fin.ext (by simp only; split_ifs <;> omega)

/-- Column `8 cb + c`. -/
def colIdx (L : Pouw.Lead.Layout) (hLn : L.n = 8 * L.colBlocks) (cb : Fin L.colBlocks) (c : Fin 8) : Fin L.n :=
  ⟨8 * cb + c, by have := cb.isLt; have := c.isLt; omega⟩

/-- Column `j`'s column block. -/
def colBlk (L : Pouw.Lead.Layout) (hLn : L.n = 8 * L.colBlocks) (j : Fin L.n) : Fin L.colBlocks :=
  ⟨(j : ℕ) / 8, by have := j.isLt; omega⟩

theorem colIdx_colBlk (L : Pouw.Lead.Layout) (hLn : L.n = 8 * L.colBlocks) (j : Fin L.n) :
    colIdx L hLn (colBlk L hLn j) ⟨(j : ℕ) % 8, Nat.mod_lt _ (by norm_num)⟩ = j :=
  Fin.ext (by simp only [colIdx, colBlk]; omega)

/-- **A tile reads its rows' and columns' coin terms at its step and before.** -/
theorem tileOf_congr (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω ω' : Coins L) (rb cb : ℕ)
    (t : Fin L.S) (h : ∀ (i : Fin L.m) (τ : Fin L.S) (j : Fin L.n), (i : ℕ) / 16 = rb → (j : ℕ) / 8 = cb → τ ≤ t →
      coinTerm L ω i τ j = coinTerm L ω' i τ j) :
    tileOf L κ ω rb cb t = tileOf L κ ω' rb cb t := by
  funext p
  have hp : (p : ℕ) < 128 := p.isLt
  unfold tileOf
  split_ifs with hc
  · unfold word
    congr 1
    refine sum_congr rfl fun τ _ => ?_
    split_ifs with hτ
    · rw [h _ τ _ (by simp only; omega) (by simp only; omega) hτ]
    · rfl
  · rfl

/-- **A coin term reads its row's tag bits at its step, its row's `a₁` and its column's `G` at step 0, and its coin
tile's lanes at its step and column.** -/
theorem coinTerm_congr (L : Pouw.Lead.Layout) (htile : L.tile = 64) (ω ω' : Coins L) (i : Fin L.m) (τ : Fin L.S)
    (j : Fin L.n) (hb : (τ : ℕ) ≠ 0 → ω.1 i τ = ω'.1 i τ) (ha : (τ : ℕ) = 0 → ∀ q, ω.2.1 i q = ω'.2.1 i q)
    (hg : (τ : ℕ) = 0 → ∀ q, ω.2.2.1 q j = ω'.2.2.1 q j)
    (hs : ∀ (k : Fin L.tiles) ℓ, (k : ℕ) = (i : ℕ) / 64 → ω.2.2.2 τ k ℓ j = ω'.2.2.2 τ k ℓ j) :
    coinTerm L ω i τ j = coinTerm L ω' i τ j := by
  unfold coinTerm laneTerm Coins.s Coins.G
  congr 1
  · split_ifs with h0
    · simp only [hg h0, ha h0]
    · rw [hb h0]
  · refine sum_congr rfl fun ℓ _ => ?_
    split_ifs with hl
    · rw [hs _ ℓ (by simp only [htile])]
    · rfl

end Pouw.SecurityProofs.Lead.Tail
