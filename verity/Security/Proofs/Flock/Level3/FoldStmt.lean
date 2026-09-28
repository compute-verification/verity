/-
Level 3, L3-A: the lincheck fold `Stmt.fold` (PROTOCOL.md §10, §16.1). `fold_get`: when the fold succeeds, entry `c` of
the block is the base of the last range whose slots cover `c`, scaled by that slot's `eq(ρ_hi, q)`, plus Δ's entries at
`c`. `baseOf_get` and `slot_factor` then read a slot's base as `α·Aᵀe + Bᵀe` on the slot's rows.
-/
import Proofs.Flock.Level3.LookupFold

open Flock Lookup

set_option maxRecDepth 20000

namespace FlockLevel3

theorem set!_get (out : Array F128) (p c : ℕ) (v : F128) (hc : c < out.size) :
    (out.set! p v)[c]! = if p = c then v else out[c]! := by
  simp only [Array.set!]
  rw [getElem!_pos _ c (by simpa using hc), Array.getElem_setIfInBounds hc, getElem!_pos out c hc]

theorem placeSlot_size (s : F128) (base : Array F128) (at_ : ℕ) : ∀ (fuel i : ℕ) (out : Array F128),
    (Stmt.placeSlot s base at_ fuel i out).size = out.size := by
  intro fuel; induction fuel with
  | zero => intro i out; rfl
  | succ fuel ih => intro i out; simp only [Stmt.placeSlot]; rw [ih]; simp [Array.set!]

theorem placeSlot_get (s : F128) (base : Array F128) (at_ : ℕ) : ∀ (fuel i : ℕ) (out : Array F128) (c : ℕ),
    c < out.size → (Stmt.placeSlot s base at_ fuel i out)[c]! =
      if at_ + i ≤ c ∧ c < at_ + i + fuel then s * base[c - at_]! else out[c]! := by
  intro fuel
  induction fuel with
  | zero => intro i out c _; simp only [Stmt.placeSlot]; rw [if_neg (by omega)]
  | succ fuel ih =>
    intro i out c hc
    simp only [Stmt.placeSlot]
    rw [ih (i + 1) _ c (by simp [Array.set!, hc]), set!_get out _ c _ hc]
    by_cases h1 : at_ + (i + 1) ≤ c ∧ c < at_ + (i + 1) + fuel
    · rw [if_pos h1, if_pos (by omega)]
    · rw [if_neg h1]
      by_cases h2 : at_ + i = c
      · rw [if_pos h2, if_pos (by omega), show c - at_ = i by omega]
      · rw [if_neg h2, if_neg (by omega)]

theorem placeRange_size (r : Range) (eHi base : Array F128) : ∀ (fuel k : ℕ) (out : Array F128),
    (Stmt.placeRange r eHi base fuel k out).size = out.size := by
  intro fuel; induction fuel with
  | zero => intro k out; rfl
  | succ fuel ih => intro k out; simp only [Stmt.placeRange]; rw [ih, placeSlot_size]

theorem placeRange_get (r : Range) (eHi base : Array F128) : ∀ (fuel k : ℕ) (out : Array F128) (c : ℕ),
    c < out.size → (Stmt.placeRange r eHi base fuel k out)[c]! =
      if r.pos0 + k ≤ c / 2 ^ r.slotLog ∧ c / 2 ^ r.slotLog < r.pos0 + k + fuel then
        eHi[c / 2 ^ r.slotLog]! * base[c % 2 ^ r.slotLog]! else out[c]! := by
  have hpos : 0 < 2 ^ r.slotLog := Nat.two_pow_pos _
  intro fuel
  induction fuel with
  | zero => intro k out c _; simp only [Stmt.placeRange]; rw [if_neg (by omega)]
  | succ fuel ih =>
    intro k out c hc
    simp only [Stmt.placeRange]
    rw [ih (k + 1) _ c (by rw [placeSlot_size]; exact hc), placeSlot_get _ _ _ _ 0 out c hc]
    by_cases h1 : r.pos0 + (k + 1) ≤ c / 2 ^ r.slotLog ∧ c / 2 ^ r.slotLog < r.pos0 + (k + 1) + fuel
    · rw [if_pos h1, if_pos (by omega)]
    · rw [if_neg h1]
      have hslot : ((r.pos0 + k) * 2 ^ r.slotLog + 0 ≤ c ∧ c < (r.pos0 + k) * 2 ^ r.slotLog + 0 + 2 ^ r.slotLog) ↔
          c / 2 ^ r.slotLog = r.pos0 + k := by
        constructor
        · intro h; exact Nat.div_eq_of_lt_le (by linarith [h.1]) (by linarith [h.2])
        · intro h
          have h1 := Nat.div_mul_le_self c (2 ^ r.slotLog)
          have h2 := Nat.lt_div_mul_add (a := c) hpos
          rw [h] at h1 h2
          constructor <;> linarith
      by_cases h2 : c / 2 ^ r.slotLog = r.pos0 + k
      · rw [if_pos (hslot.mpr h2), if_pos (by omega), h2]
        rw [Nat.mod_eq_sub_mul_div, h2, mul_comm]
      · rw [if_neg (fun h => h2 (hslot.mp h)), if_neg (by omega)]

/-! ### The ranges, with the slot-type cache -/

/-- The base the fold uses for range `r` (the slot type's `slotBase`; junk if it fails, which the fold then does). -/
def baseFor (st : Stmt) (alpha : F128) (lw rho : Array F128) (r : Range) : Array F128 :=
  match st.slotBase alpha lw rho (r.kind, r.slotLog) with
  | .ok b => b
  | .error _ => #[]

def eHiFor (rho : Array F128) (r : Range) : Array F128 := eqTable (rho.extract (r.slotLog - 6) rho.size)

/-- `c` lies in one of `r`'s slots. -/
def covers (r : Range) (c : ℕ) : Prop := r.pos0 ≤ c / 2 ^ r.slotLog ∧ c / 2 ^ r.slotLog < r.pos0 + r.count

instance instDecidableCovers (r : Range) (c : ℕ) : Decidable (covers r c) := inferInstanceAs (Decidable (_ ∧ _))

/-- What the ranges leave at `c`: the last covering range's scaled base, else `v`. -/
def rangesVal (st : Stmt) (alpha : F128) (lw rho : Array F128) : List Range → ℕ → F128 → F128
  | [], _, v => v
  | r :: rs, c, v => rangesVal st alpha lw rho rs c
      (if covers r c then (eHiFor rho r)[c / 2 ^ r.slotLog]! * (baseFor st alpha lw rho r)[c % 2 ^ r.slotLog]! else v)

theorem foldRanges_get (st : Stmt) (alpha : F128) (lw rho : Array F128) : ∀ (rs : List Range)
    (bases : Array ((Kind × Nat) × Array F128)) (out res : Array F128),
    st.foldRanges alpha lw rho rs bases out = .ok res →
    (∀ kb ∈ bases, st.slotBase alpha lw rho kb.1 = .ok kb.2) →
    res.size = out.size ∧ ∀ c < out.size, res[c]! = rangesVal st alpha lw rho rs c out[c]! := by
  intro rs
  induction rs with
  | nil => intro bases out res h _; simp only [Stmt.foldRanges, pure, Except.pure] at h; cases h; exact ⟨rfl, fun _ _ => rfl⟩
  | cons r rs ih =>
    intro bases out res h hinv
    simp only [Stmt.foldRanges] at h
    cases hf : bases.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
    | some kb =>
      rw [hf] at h
      simp only [bind, Except.bind, pure, Except.pure] at h
      have hkb := Array.find?_some hf
      have hmem := Array.mem_of_find?_eq_some hf
      have hkey : kb.1 = (r.kind, r.slotLog) := of_decide_eq_true hkb
      have hbase : baseFor st alpha lw rho r = kb.2 := by
        rw [baseFor, ← hkey, hinv kb hmem]
      obtain ⟨hsz, hget⟩ := ih bases _ res h hinv
      refine ⟨by rw [hsz, placeRange_size], fun c hc => ?_⟩
      rw [hget c (by rw [placeRange_size]; exact hc), placeRange_get _ _ _ _ 0 out c hc, rangesVal, hbase]
      congr 1
    | none =>
      rw [hf] at h
      cases hb : st.slotBase alpha lw rho (r.kind, r.slotLog) with
      | error e => rw [hb] at h; simp [bind, Except.bind] at h
      | ok b =>
        rw [hb] at h
        simp only [bind, Except.bind, pure, Except.pure] at h
        have hbase : baseFor st alpha lw rho r = b := by rw [baseFor, hb]
        have hinv' : ∀ kb ∈ bases.push ((r.kind, r.slotLog), b), st.slotBase alpha lw rho kb.1 = .ok kb.2 := by
          intro kb hkb
          rcases Array.mem_push.mp hkb with h1 | h1
          · exact hinv kb h1
          · rw [h1]; exact hb
        obtain ⟨hsz, hget⟩ := ih _ _ res h hinv'
        refine ⟨by rw [hsz, placeRange_size], fun c hc => ?_⟩
        rw [hget c (by rw [placeRange_size]; exact hc), placeRange_get _ _ _ _ 0 out c hc, rangesVal, hbase]
        congr 1

/-! ### Δ, and the fold -/

theorem addDelta_get (w : ℕ → F128) : ∀ (pairs : List (ℕ × ℕ)) (out : Array F128) (c : ℕ), c < out.size →
    (Stmt.addDelta w pairs out).size = out.size ∧
    GF128.of (Stmt.addDelta w pairs out)[c]! =
      GF128.of out[c]! + (pairs.map fun ic => if ic.2 = c then GF128.of (w ic.1) else 0).sum := by
  intro pairs
  induction pairs with
  | nil => intro out c _; simp [Stmt.addDelta]
  | cons ic pairs ih =>
    intro out c hc
    have h := ih (out.modify ic.2 (· + w ic.1)) c (by simpa using hc)
    simp only [Stmt.addDelta, List.foldl_cons] at h ⊢
    refine ⟨by rw [h.1]; simp, ?_⟩
    rw [h.2, modify_get! out _ c _ hc, List.map_cons, List.sum_cons]
    split_ifs <;> simp [of_add] <;> ring

/-- `e(i) = lw[i mod 64] · eq(ρ, ⌊i/64⌋)`, the in-block tensor. -/
def eIn (lw rho : Array F128) (i : ℕ) : F128 := lw[i % 64]! * (eqTable rho)[i / 64]!

/-- **The fold.** When `Stmt.fold` succeeds, entry `c < 2^k_log` is what the ranges leave there (the last covering
range's base, scaled by `eq(ρ_hi, q)`) plus `α·e(i)` for each `(i, c) ∈ Δ_A` and `e(i)` for each `(i, c) ∈ Δ_B`. -/
theorem fold_get (st : Stmt) (alpha : F128) (lw rho out : Array F128) (h : st.fold alpha lw rho = .ok out)
    (c : ℕ) (hc : c < 2 ^ st.c.kLog) :
    GF128.of out[c]! = GF128.of (rangesVal st alpha lw rho st.c.ranges.toList c 0) +
      (st.da.toList.map fun ic => if ic.2 = c then GF128.of alpha * GF128.of (eIn lw rho ic.1) else 0).sum +
      (st.db.toList.map fun ic => if ic.2 = c then GF128.of (eIn lw rho ic.1) else 0).sum := by
  simp only [Stmt.fold] at h
  split_ifs at h with hck
  · simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  cases hr : st.foldRanges alpha lw rho st.c.ranges.toList #[] (Array.replicate (2 ^ st.c.kLog) 0) with
  | error e => rw [hr] at h; simp [bind, Except.bind] at h
  | ok res =>
    rw [hr] at h
    simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨hsz, hget⟩ := foldRanges_get st alpha lw rho _ #[] _ res hr (by simp)
    have hcr : c < res.size := by rw [hsz]; simpa using hc
    obtain ⟨hsz1, h1⟩ := addDelta_get (fun i => alpha * eIn lw rho i) st.da.toList res c hcr
    obtain ⟨_, h2⟩ := addDelta_get (eIn lw rho) st.db.toList _ c (by rw [hsz1]; exact hcr)
    change GF128.of (Stmt.addDelta (eIn lw rho) st.db.toList
      (Stmt.addDelta (fun i => alpha * eIn lw rho i) st.da.toList res))[c]! = _
    rw [h2, h1, hget c (by simpa using hc)]
    rw [getElem!_pos _ c (by simpa using hc), Array.getElem_replicate]
    simp only [of_mul]

/-! ### A slot's base is `α·Aᵀe_lo + Bᵀe_lo`, and `eq(ρ_hi, q)·e_lo(i) = e(q·2^sl + i)` -/

/-- The lookup table's share of the B side at `i'` (zero for a slot without a table). -/
noncomputable def lookupTerm (eLo : Array F128) : Option LookupTable → ℕ → GF128
  | none, _ => 0
  | some t, i' => ∑ h ∈ Finset.range (2 ^ (t.n - t.lo)), ∑ l ∈ Finset.range (2 ^ t.lo),
      if t.loTop + l = i' then wordSum (ej t (2 ^ (t.n - t.lo)) eLo h) (word t (2 ^ t.lo) (2 ^ t.bits - 1) h l) else 0

theorem foldRows_size (m : Sparse) (e : Array F128) : ∀ (fuel i : ℕ) (acc : Array F128),
    (m.foldRows e fuel i acc).size = acc.size := by
  intro fuel; induction fuel with
  | zero => intro i acc; rfl
  | succ fuel ih => intro i acc; simp only [Sparse.foldRows]; rw [ih]; split_ifs <;> simp [foldRow_size]

theorem foldT_size (m : Sparse) (e : Array F128) : (m.foldT e).size = m.width := by
  simp [Sparse.foldT, foldRows_size]

/-- `(Xᵀe)[i']` as the double sum `foldT_get` gives. -/
noncomputable def sparseT (m : Sparse) (e : Array F128) (i' : ℕ) : GF128 :=
  ∑ i ∈ Finset.range (min m.rows e.size), ∑ t ∈ Finset.range m.len[i]!, if m.col i t = i' then GF128.of e[i]! else 0

theorem baseOf_get (alpha : F128) (eLo : Array F128) (a b : Sparse) (lk : Option LookupTable) (sl i' : ℕ)
    (hi : i' < 2 ^ sl) (ha : i' < a.width) (hb : i' < b.width) :
    GF128.of (Stmt.baseOf alpha eLo a b lk sl)[i']! =
      GF128.of alpha * sparseT a eLo i' + sparseT b eLo i' + lookupTerm eLo lk i' := by
  unfold Stmt.baseOf
  rw [getElem!_pos _ i' (by simpa using hi)]
  simp only [Array.getElem_map, Array.getElem_range, of_add, of_mul, sparseT]
  rw [foldT_get a eLo i' ha]
  cases lk with
  | none => simp only [lookupTerm, add_zero]; rw [foldT_get b eLo i' hb]
  | some t =>
    simp only [lookupTerm]
    rw [foldB_get t eLo _ i' (by rw [foldT_size]; exact hb), foldT_get b eLo i' hb]
    ring

theorem slot_scale (lw rho : Array F128) (hlw : lw.size = 64) (sl : ℕ) (hsl : 6 ≤ sl) (hslr : sl - 6 ≤ rho.size)
    (q i : ℕ) (hq : q < 2 ^ (rho.size - (sl - 6))) (hi : i < 2 ^ sl) :
    GF128.of (eqTable (rho.extract (sl - 6) rho.size))[q]! * GF128.of (Stmt.tensor lw (rho.extract 0 (sl - 6)))[i]! =
      GF128.of (eIn lw rho (q * 2 ^ sl + i)) := by
  rw [eIn, slot_factor lw rho hlw sl hsl hslr q i hq hi, mul_comm]

/-- **A unit or compression slot's entry.** If the range's base is `baseOf` of the slot's matrices `A`, `B` (and table),
then entry `q·2^sl + i'` of the fold's slot part is `α·(Aᵀe)[i'] + (Bᵀe)[i']` (+ the table's B side) on the rows
`q·2^sl + i` of the slot, scaled into the block: the local tensor at `i` times `eq(ρ_hi, q)` is `e(q·2^sl + i)`. -/
theorem slot_term (st : Stmt) (alpha : F128) (lw rho : Array F128) (r : Range) (a b : Sparse) (lk : Option LookupTable)
    (hbase : baseFor st alpha lw rho r =
      Stmt.baseOf alpha (Stmt.tensor lw (rho.extract 0 (r.slotLog - 6))) a b lk r.slotLog)
    (c : ℕ) (hc : c % 2 ^ r.slotLog < a.width ∧ c % 2 ^ r.slotLog < b.width) :
    GF128.of ((eHiFor rho r)[c / 2 ^ r.slotLog]! * (baseFor st alpha lw rho r)[c % 2 ^ r.slotLog]!) =
      GF128.of (eHiFor rho r)[c / 2 ^ r.slotLog]! *
        (GF128.of alpha * sparseT a (Stmt.tensor lw (rho.extract 0 (r.slotLog - 6))) (c % 2 ^ r.slotLog) +
          sparseT b (Stmt.tensor lw (rho.extract 0 (r.slotLog - 6))) (c % 2 ^ r.slotLog) +
          lookupTerm (Stmt.tensor lw (rho.extract 0 (r.slotLog - 6))) lk (c % 2 ^ r.slotLog)) := by
  rw [of_mul, hbase, baseOf_get _ _ _ _ _ _ _ (Nat.mod_lt _ (Nat.two_pow_pos _)) hc.1 hc.2]

/-- **A mask slot's entry**: `(α + 1)·e(c)`. -/
theorem mask_term (lw rho : Array F128) (hlw : lw.size = 64) (sl : ℕ) (hsl : 6 ≤ sl) (hslr : sl - 6 ≤ rho.size)
    (alpha : F128) (c : ℕ) (hq : c / 2 ^ sl < 2 ^ (rho.size - (sl - 6))) :
    GF128.of ((eqTable (rho.extract (sl - 6) rho.size))[c / 2 ^ sl]! *
      ((Stmt.tensor lw (rho.extract 0 (sl - 6))).map fun x => (alpha + 1) * x)[c % 2 ^ sl]!) =
      (GF128.of alpha + 1) * GF128.of (eIn lw rho c) := by
  have hi : c % 2 ^ sl < 2 ^ sl := Nat.mod_lt _ (Nat.two_pow_pos _)
  have hsz : (Stmt.tensor lw (rho.extract 0 (sl - 6))).size = 2 ^ sl := by
    simp [Stmt.tensor, eqTable_size, hlw]
    rw [show sl = sl - 6 + 6 by omega, pow_add]; simp [min_eq_left hslr]; ring_nf
  rw [getElem!_pos _ (c % 2 ^ sl) (by simpa [hsz] using hi), Array.getElem_map, of_mul, of_mul, of_add, ← GF128.one_eq,
    ← getElem!_pos _ (c % 2 ^ sl) (by rw [hsz]; exact hi)]
  have := slot_scale lw rho hlw sl hsl hslr (c / 2 ^ sl) (c % 2 ^ sl) hq hi
  rw [Nat.div_add_mod' c (2 ^ sl)] at this
  rw [← this]; ring

theorem tensor_size (lw rho : Array F128) (hlw : lw.size = 64) (sl : ℕ) (hsl : 6 ≤ sl) (hslr : sl - 6 ≤ rho.size) :
    (Stmt.tensor lw (rho.extract 0 (sl - 6))).size = 2 ^ sl := by
  simp [Stmt.tensor, eqTable_size, hlw]
  rw [show sl = sl - 6 + 6 by omega, pow_add]; simp [min_eq_left hslr]; ring_nf

/-- **The slot's rows.** `eq(ρ_hi, q)·(Xᵀe_lo)[i'] = Σ_{i : col(i, t) = i'} e(q·2^sl + i)`: the sparse fold of the slot's
matrix on the block's tensor, restricted to the slot's rows. -/
theorem sparseT_scale (lw rho : Array F128) (hlw : lw.size = 64) (sl : ℕ) (hsl : 6 ≤ sl) (hslr : sl - 6 ≤ rho.size)
    (q : ℕ) (hq : q < 2 ^ (rho.size - (sl - 6))) (m : Sparse) (i' : ℕ) :
    GF128.of (eqTable (rho.extract (sl - 6) rho.size))[q]! * sparseT m (Stmt.tensor lw (rho.extract 0 (sl - 6))) i' =
      ∑ i ∈ Finset.range (min m.rows (2 ^ sl)), ∑ t ∈ Finset.range m.len[i]!,
        if m.col i t = i' then GF128.of (eIn lw rho (q * 2 ^ sl + i)) else 0 := by
  rw [sparseT, tensor_size lw rho hlw sl hsl hslr]
  refine (Finset.mul_sum _ _ _).trans (Finset.sum_congr rfl fun i hi => ?_)
  refine (Finset.mul_sum _ _ _).trans (Finset.sum_congr rfl fun t _ => ?_)
  have hi' : i < 2 ^ sl := lt_of_lt_of_le (Finset.mem_range.mp hi) (min_le_right _ _)
  split_ifs
  · exact slot_scale lw rho hlw sl hsl hslr q i hq hi'
  · exact mul_zero _

end FlockLevel3
