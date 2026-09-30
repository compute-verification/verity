import FlockSoundness.Refine.StmtOf
import FlockLevel3.Placed

/-!
# `stmtOf`'s matrices are the placed ones

Two readings of one accepted statement's level-0 matrices meet in the end-to-end theorem's places:
- the refinement's model statement, `stmtOf st h regions`, whose `A₀` and `B₀` count the ranges' slot entries and Δ's
  pairs (`matA`, `matB`);
- flock-verifier's `placedA st` and `placedB st` (`FlockLevel3.Placed`), which the placement proofs read and on which a
  table class is built (`ExecTemplateClass.lean`, `ExecFlatCopies.lean`).

They agree entry by entry when three layout facts hold of the statement's nets and leaf circuit (`PlacedWF`):
- a sparse matrix's column indices are inside its width;
- a lookup's low minterms are inside its net's `B` width;
- a lookup's output word has at most 32 bits.

Then `stmtOf_A₀` and `stmtOf_B₀` give the matrices, and `stmtOf_eq_placed` the statement: the model statement with
`st`'s `m`, `k_log`, pin and regions and the placed matrices is `stmtOf st h regions`. The proofs:
- the two `lastCover`s pick the same range (`lastCover_eq`);
- a count mod 2 is a count cast (`spCnt_cast`);
- a lookup sends at most one product row to an entry, so its count is `lookupEntry`'s indicator (`lkCnt_cast`);
- Δ's count is the length of its filter (`dCnt_cast`).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3

/-- **The layout facts under which the two readings agree**: every sparse matrix a slot reads (each net's `A` and `B`,
the leaf circuit's) has its columns inside its width, and every net's lookup has its low minterms inside the net's `B`
width and at most 32 output bits. -/
structure PlacedWF (st : Stmt) : Prop where
  netA : ∀ n i t, i < (st.c.nets.getD n default).2.a.rows → t < (st.c.nets.getD n default).2.a.len[i]! →
    (st.c.nets.getD n default).2.a.col i t < (st.c.nets.getD n default).2.a.width
  netB : ∀ n i t, i < (st.c.nets.getD n default).2.b.rows → t < (st.c.nets.getD n default).2.b.len[i]! →
    (st.c.nets.getD n default).2.b.col i t < (st.c.nets.getD n default).2.b.width
  compA : ∀ i t, i < st.leaf.comp.a.rows → t < st.leaf.comp.a.len[i]! → st.leaf.comp.a.col i t < st.leaf.comp.a.width
  compB : ∀ i t, i < st.leaf.comp.b.rows → t < st.leaf.comp.b.len[i]! → st.leaf.comp.b.col i t < st.leaf.comp.b.width
  lookup : ∀ n lk, (st.c.nets.getD n default).2.lookup = some lk →
    lk.loTop + 2 ^ lk.lo ≤ (st.c.nets.getD n default).2.b.width ∧ lk.bits ≤ 32

/-! ## The covering range -/

theorem foldl_cover (c : ℕ) : ∀ (rs : List Range) (init : Option Range),
    rs.foldl (fun acc r => if covers r c then some r else acc) init = (Refine.lastCover rs c).or init
  | [], init => by simp [Refine.lastCover]
  | r :: rs, init => by
    rw [List.foldl_cons, foldl_cover c rs, Refine.lastCover, Option.or_assoc]
    congr 1
    by_cases h : covers r c <;> simp [h]

/-- **The two `lastCover`s pick the same range**: the last in the layout that covers `c`. -/
theorem lastCover_eq (rs : List Range) (c : ℕ) : Refine.lastCover rs c = FlockLevel3.lastCover rs c := by
  rw [FlockLevel3.lastCover, foldl_cover, Option.or_none]

/-! ## Slot entries -/

theorem card_range_filter (n : ℕ) (p : ℕ → Prop) [DecidablePred p] :
    ((Finset.range n).filter p).card = ((List.range n).filter fun t => decide (p t)).length := by
  rw [← Multiset.coe_card, ← Multiset.filter_coe, Multiset.coe_range]
  rfl

/-- **A sparse matrix's count, mod 2**, when its columns are inside its width. -/
theorem spCnt_cast (m : Sparse) (hw : ∀ i t, i < m.rows → t < m.len[i]! → m.col i t < m.width) (i c : ℕ) :
    ((spCnt m i c : ℕ) : ZMod 2) = sparseEntry m i c := by
  unfold spCnt sparseEntry
  by_cases hi : i < m.rows
  · by_cases hc : c < m.width
    · rw [if_pos ⟨hi, hc⟩, if_pos hi, card_range_filter]
    · rw [if_neg (fun h => hc h.2), if_pos hi]
      have : (List.range m.len[i]!).filter (fun t => decide (m.col i t = c)) = [] := by
        rw [List.filter_eq_nil_iff]
        intro t ht
        have := hw i t hi (List.mem_range.1 ht)
        simp only [decide_eq_true_eq]
        omega
      rw [this]
      rfl
  · rw [if_neg (fun h => hi h.1), if_neg hi]
    rfl

/-- **A lookup sends at most one product row to an entry.** With at most 32 output bits, the count `lkCnt` is
`lookupEntry`'s indicator: row `i` is product row `(j, h)` for `j = (i − prod0) / nh` and `h = (i − prod0) % nh`, and
column `c` is low minterm `c − loTop`. -/
theorem lkCnt_cast (lk : LookupTable) (hb : lk.bits ≤ 32) (i c : ℕ) :
    ((lkCnt lk i c : ℕ) : ZMod 2) = lookupEntry (some lk) i c := by
  have hnh : 0 < 2 ^ (lk.n - lk.lo) := Nat.two_pow_pos _
  simp only [lookupEntry, lkCnt]
  generalize 2 ^ (lk.n - lk.lo) = nh at hnh ⊢
  generalize 2 ^ lk.lo = nlo
  have fwd : ∀ h j, h < nh → j < lk.bits → lk.prod0 + j * nh + h = i →
      lk.prod0 ≤ i ∧ i < lk.prod0 + lk.bits * nh ∧ (i - lk.prod0) / nh = j ∧ (i - lk.prod0) % nh = h := by
    intro h j hh hjb hi
    have hle : lk.prod0 ≤ i := by omega
    have hdm : (i - lk.prod0) / nh = j ∧ (i - lk.prod0) % nh = h :=
      (Nat.div_mod_unique hnh).2 ⟨by rw [Nat.mul_comm]; omega, hh⟩
    have h1 : nh * (j + 1) ≤ nh * lk.bits := Nat.mul_le_mul_left _ (by omega)
    have h2 : nh * (j + 1) = j * nh + nh := by rw [Nat.mul_succ, Nat.mul_comm]
    refine ⟨hle, ?_, hdm⟩
    rw [Nat.mul_comm lk.bits nh]
    omega
  split_ifs with hP
  · obtain ⟨hle, hlt, hlo, hhi, hbit⟩ := hP
    have hq : (i - lk.prod0) / nh < lk.bits := (Nat.div_lt_iff_lt_mul hnh).2 (by omega)
    have hdm := Nat.div_add_mod (i - lk.prod0) nh
    rw [Finset.card_eq_one.2 ⟨((i - lk.prod0) % nh, c - lk.loTop, (i - lk.prod0) / nh), ?_⟩]
    · exact Nat.cast_one
    ext ⟨h, l, j⟩
    simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_range, Finset.mem_singleton, Prod.mk.injEq]
    dsimp only
    constructor
    · rintro ⟨⟨hh, -, -⟩, hlc, -, hjb, hi⟩
      obtain ⟨-, -, hj, hh'⟩ := fwd h j hh hjb hi
      exact ⟨hh'.symm, by omega, hj.symm⟩
    · rintro ⟨rfl, rfl, rfl⟩
      dsimp only
      refine ⟨⟨Nat.mod_lt _ hnh, by omega, by omega⟩, by omega, hbit, hq, ?_⟩
      rw [Nat.mul_comm] at hdm
      omega
  · rw [Finset.card_eq_zero.2 (Finset.filter_eq_empty_iff.2 ?_)]
    · exact Nat.cast_zero
    rintro ⟨h, l, j⟩ hx ⟨hlc, hbit, hjb, hi⟩
    simp only [Finset.mem_product, Finset.mem_range] at hx
    dsimp only at hx hlc hbit hjb hi
    obtain ⟨hle, hlt, hj, hh'⟩ := fwd h j hx.1 hjb hi
    apply hP
    refine ⟨hle, hlt, by omega, by omega, ?_⟩
    rw [hj, hh', show c - lk.loTop = l by omega]
    exact hbit

/-- **A lookup slot's `B` share, mod 2**: `lkSide`'s width guard is vacuous where the lookup writes. -/
theorem lkSide_cast (b : Sparse) (lko : Option LookupTable)
    (hl : ∀ lk, lko = some lk → lk.loTop + 2 ^ lk.lo ≤ b.width ∧ lk.bits ≤ 32) (i c : ℕ) :
    ((lkSide b lko i c : ℕ) : ZMod 2) = lookupEntry lko i c := by
  cases lko with
  | none => rfl
  | some lk =>
    obtain ⟨hw, hb⟩ := hl lk rfl
    unfold lkSide
    by_cases hc : c < b.width
    · rw [if_pos hc]
      exact lkCnt_cast lk hb i c
    · rw [if_neg hc]
      have : lookupEntry (some lk) i c = 0 := by
        simp only [lookupEntry]
        exact if_neg fun h' => hc (lt_of_lt_of_le h'.2.2.2.1 hw)
      rw [this]
      rfl

/-- `slotA`'s entries, mod 2, are `slotEntryA`'s. -/
theorem slotA_cast (st : Stmt) (hw : PlacedWF st) (k : Kind) (i c : ℕ) :
    ((slotA st k i c : ℕ) : ZMod 2) = slotEntryA st k i c := by
  cases k with
  | net n => exact spCnt_cast _ (hw.netA n) i c
  | mask =>
    show ((if i = c then 1 else 0 : ℕ) : ZMod 2) = if i = c then 1 else 0
    split_ifs <;> rfl
  | comp => exact spCnt_cast _ hw.compA i c

/-- `slotB`'s entries, mod 2, are `slotEntryB`'s. -/
theorem slotB_cast (st : Stmt) (hw : PlacedWF st) (k : Kind) (i c : ℕ) :
    ((slotB st k i c : ℕ) : ZMod 2) = slotEntryB st k i c := by
  cases k with
  | net n =>
    show ((spCnt _ i c + lkSide _ _ i c : ℕ) : ZMod 2) = sparseEntry _ i c + lookupEntry _ i c
    rw [Nat.cast_add, spCnt_cast _ (hw.netB n) i c, lkSide_cast _ _ (hw.lookup n) i c]
  | mask =>
    show ((if i = c then 1 else 0 : ℕ) : ZMod 2) = if i = c then 1 else 0
    split_ifs <;> rfl
  | comp => exact spCnt_cast _ hw.compB i c

/-! ## The matrices -/

theorem rangeCnt_cast (st : Stmt) (side : Kind → ℕ → ℕ → ℕ) (entry : Kind → ℕ → ℕ → ZMod 2)
    (h : ∀ k i c, ((side k i c : ℕ) : ZMod 2) = entry k i c) (i c : ℕ) :
    ((rangeCnt side st.c.ranges.toList i c : ℕ) : ZMod 2) = rangesEntry st entry i c := by
  unfold rangeCnt rangesEntry
  rw [lastCover_eq]
  cases FlockLevel3.lastCover st.c.ranges.toList c with
  | none => rfl
  | some r =>
    dsimp only
    split_ifs
    · exact h _ _ _
    · rfl

theorem dCnt_cast (d : Array (ℕ × ℕ)) (i c : ℕ) : ((dCnt d i c : ℕ) : ZMod 2) = deltaEntry d i c := by
  unfold dCnt deltaEntry
  rw [List.count_eq_length_filter]
  congr 2
  funext x
  simp

/-- **`stmtOf`'s `A₀` is `placedA`**, under `PlacedWF`. -/
theorem stmtOf_A₀ (st : Stmt) (h : StmtWF st) (hw : PlacedWF st)
    (regions : List (Model.Region st.c.kLog (st.m - st.c.kLog))) : (stmtOf st h regions).A₀ = placedA st := by
  funext r j
  show matA st r.val j.val = rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val
  unfold matA
  rw [Nat.cast_add, rangeCnt_cast st _ _ (slotA_cast st hw) r.val j.val, dCnt_cast]

/-- **`stmtOf`'s `B₀` is `placedB`**, under `PlacedWF`. -/
theorem stmtOf_B₀ (st : Stmt) (h : StmtWF st) (hw : PlacedWF st)
    (regions : List (Model.Region st.c.kLog (st.m - st.c.kLog))) : (stmtOf st h regions).B₀ = placedB st := by
  funext r j
  show matB st r.val j.val = rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val
  unfold matB
  rw [Nat.cast_add, rangeCnt_cast st _ _ (slotB_cast st hw) r.val j.val, dCnt_cast]

/-- **The model statement of `st` with the placed matrices is `stmtOf`**, under `PlacedWF`: the same `m`, `k_log`, pin
and regions, with `A₀ = placedA st` and `B₀ = placedB st`. -/
theorem stmtOf_eq_placed (st : Stmt) (h : StmtWF st) (hw : PlacedWF st)
    (regions : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    ({ m := st.m, kLog := st.c.kLog, hk := h.hk, hk6 := h.hk6, pin := ⟨st.pin, h.pin⟩, A₀ := placedA st,
      B₀ := placedB st, regions := regions } : Model.Statement) = stmtOf st h regions := by
  rw [← stmtOf_A₀ st h hw regions, ← stmtOf_B₀ st h hw regions]

end FlockSoundness.Refine
