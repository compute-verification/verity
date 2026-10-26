import Proofs.Flock.Soundness.Lowering
import Flock.Net
-- level3 already realizes the lemmas this file's `simp` calls need (`Std.Legacy.Range.mk.congr_simp`), so they are
-- reused rather than declared a second time, which the audit's kernel replay refuses
import Proofs.Flock.Level3.LookupRows
import Proofs.Flock.Soundness.SparseRows
import Properties.Flock.Circuit.Rows

/-!
# The rows the verifier parses, as the audit's `Rows`

The executable's parser (`Flock.Net.parse`) holds every expanded circuit's rows to #144's order
(`Net.checkOrder`, PR #147): a computed row, `inCols ≤ r < constPos`, reads in `A` and in `B` only
the constant or an earlier column that is not input padding (`RowOrder`). `checkOrder_ok` turns an
accepted check into that property, and `Rows.ofOrdered` builds the audit's `Rows` from rows that
have it, with `topo` proved: what #144 named, the parser now checks.
-/

namespace FlockSoundness

open Flock

section ofOrdered

variable {inCols constPos : ℕ} {ra rb : Array (Array ℕ)} {h : RowOrder inCols constPos ra rb}

theorem Rows.ofOrdered_nIn (hle : inCols ≤ constPos) : (Rows.ofOrdered inCols constPos ra rb h).nIn = inCols := by
  show min inCols constPos = inCols
  exact Nat.min_eq_left hle

theorem Rows.ofOrdered_a (i : Fin (Rows.ofOrdered inCols constPos ra rb h).nComp) :
    ((Rows.ofOrdered inCols constPos ra rb h).a i).map Fin.val =
      (ra.getD ((Rows.ofOrdered inCols constPos ra rb h).nIn + i.val) #[]).toList := by
  simp [Rows.ofOrdered, List.map_pmap]

theorem Rows.ofOrdered_b (i : Fin (Rows.ofOrdered inCols constPos ra rb h).nComp) :
    ((Rows.ofOrdered inCols constPos ra rb h).b i).map Fin.val =
      (rb.getD ((Rows.ofOrdered inCols constPos ra rb h).nIn + i.val) #[]).toList := by
  simp [Rows.ofOrdered, List.map_pmap]

theorem Rows.ofOrdered_comp (i : Fin (Rows.ofOrdered inCols constPos ra rb h).nComp) :
    ((Rows.ofOrdered inCols constPos ra rb h).comp i).val =
      (Rows.ofOrdered inCols constPos ra rb h).nIn + i.val := rfl

theorem Rows.ofOrdered_one : (Rows.ofOrdered inCols constPos ra rb h).one.val = constPos := by
  show min inCols constPos + (constPos - min inCols constPos) = constPos
  omega

end ofOrdered

/-! ## What `Net.parse` accepted -/

theorem except_bind_ok {α β ε : Type} {x : Except ε α} {f : α → Except ε β} {v : β}
    (h : (x >>= f) = .ok v) : ∃ a, x = .ok a ∧ f a = .ok v := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem throw_bind_ne {α β : Type} {e : String} {f : α → Except String β} {v : β} :
    (throw e >>= f : Except String β) ≠ .ok v := by
  intro h; cases h

theorem ite_throw_ok {α β : Type} {c : Prop} [Decidable c] {e : String} {f : α → Except String β}
    {g : Except String β} {v : β} (h : (if c then (throw e >>= f) else g) = .ok v) : g = .ok v := by
  by_cases hc : c
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hc)] at h; cases h
  · rwa [ite_eq_right_of_eq_false _ _ (eq_false hc)] at h

theorem parse_tail {α : Type} {x y : Except String Sparse} {mk : Sparse → Sparse → α} {v : α}
    (h : (do let a ← x; let b ← y; pure (mk a b)) = .ok v) : ∃ a b, x = .ok a ∧ y = .ok b ∧ v = mk a b := by
  obtain ⟨a, ha, h⟩ := except_bind_ok h
  obtain ⟨b, hb, h⟩ := except_bind_ok h
  cases h
  exact ⟨a, b, ha, hb, rfl⟩

set_option maxHeartbeats 1000000 in
/-- **`Net.parse` builds its net with `Net.ofRows`**, after reading its rows and trailing lines. -/
theorem parse_built {lines : Array String} {net : Net} (h : Net.parse lines = .ok net) : Built net := by
  unfold Net.parse at h
  simp only at h
  replace h := ite_throw_ok h
  obtain ⟨useful, -, h⟩ := except_bind_ok h
  obtain ⟨constPos, -, h⟩ := except_bind_ok h
  obtain ⟨ig, -, h⟩ := except_bind_ok h
  obtain ⟨og, -, h⟩ := except_bind_ok h
  replace h := ite_throw_ok (ite_throw_ok (ite_throw_ok (ite_throw_ok (ite_throw_ok h))))
  obtain ⟨s, -, h⟩ := except_bind_ok h
  refine ⟨(tokens (lines.getD 0 "")).getD 1 "", useful, constPos, ig.1, og.1, s.1, s.2, ?_⟩
  by_cases hL : (lines.getD (1 + useful) "").startsWith "LEAVES " = true
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hL)] at h
    rcases hj : Lean.Json.parse ((lines.getD (1 + useful) "").drop 7).toString with e | j
    · rw [hj] at h; exact absurd h throw_bind_ne
    · rw [hj] at h
      exact ite_throw_ok (ite_throw_ok h)
  · rw [ite_eq_right_of_eq_false _ _ (eq_false hL)] at h
    exact ite_throw_ok (ite_throw_ok h)

/-- **The parser orders every net it accepts**: the rows it stored as `A` and `B` have `RowOrder`,
so #144's `Rows.topo` for a parsed net is an executable check, not a hypothesis. -/
theorem parse_rowOrder {lines : Array String} {net : Net} (h : Net.parse lines = .ok net) :
    ∃ ra rb, RowOrder (net.inWords * WORD) net.constPos ra rb ∧
      Sparse.ofRows ra (2 ^ net.unitLog) = .ok net.a ∧ Sparse.ofRows rb (2 ^ net.unitLog) = .ok net.b :=
  built_rowOrder (parse_built h)

/-! ## `Sparse`: the bytes `ofRows` writes, and `row` reads them back (`SparseRows`) -/

export SparseRows (le32 pushLe32_data getElem!_of_some testBit_mod256 le32_or rd32_le32 encRow le32_length
  encRow_length rowStarts forIn_cols_ok forIn_rows_ok ofRows_spec rowStarts_getElem? rowStarts_length
  flatMap_le32_getElem? flatMap_encRow_length ofRows_row)

/-! ## The audit's `Rows` for a built net -/

section ofNet

variable {net : Net} (h : Built net)

theorem Rows.ofNet_nIn : (Rows.ofNet h).nIn = min (net.inWords * WORD) net.constPos := rfl

theorem Rows.ofNet_one : (Rows.ofNet h).one.val = net.constPos := Rows.ofOrdered_one

theorem Rows.ofNet_comp (i : Fin (Rows.ofNet h).nComp) :
    ((Rows.ofNet h).comp i).val = (Rows.ofNet h).nIn + i.val := rfl

/-- Computed row `i` reads, in `A`, exactly the columns of the net's row `nIn + i`. -/
theorem Rows.ofNet_a (hu : net.unitLog ≤ 32) (i : Fin (Rows.ofNet h).nComp) :
    ((Rows.ofNet h).a i).map Fin.val = (net.a.row ((Rows.ofNet h).nIn + i.val)).toList := by
  have hs := Classical.choose_spec (Classical.choose_spec (built_rowOrder h))
  refine (Rows.ofOrdered_a i).trans ?_
  rw [ofRows_row hs.2.1 (Nat.pow_le_pow_right (by norm_num) hu)]
  rfl

/-- Computed row `i` reads, in `B`, exactly the columns of the net's row `nIn + i`. -/
theorem Rows.ofNet_b (hu : net.unitLog ≤ 32) (i : Fin (Rows.ofNet h).nComp) :
    ((Rows.ofNet h).b i).map Fin.val = (net.b.row ((Rows.ofNet h).nIn + i.val)).toList := by
  have hs := Classical.choose_spec (Classical.choose_spec (built_rowOrder h))
  refine (Rows.ofOrdered_b i).trans ?_
  rw [ofRows_row hs.2.2 (Nat.pow_le_pow_right (by norm_num) hu)]
  rfl

end ofNet

/-! ## An instance placed in a statement -/

section Place

variable (S : Model.Statement) (R : Rows) (k : ℕ) (base : ℕ → ℕ)
  (hb : ∀ u < k, base u + (R.nIn + R.nComp) < 2 ^ S.kLog)

/-- Copy `u`'s column `c` in the block: `base u + c`. -/
def posIn {u : ℕ} (hu : u < k) (c : Fin R.w) : Fin (2 ^ S.kLog) :=
  ⟨base u + c.val, by have := hb u hu; have := c.isLt; simp only [Rows.w] at this; omega⟩

/-- **Where the stacked columns sit in the block**: copy `u`'s inputs and computed rows at
`base u + c`, its constant row at its slot's constant `base u + (nIn + nComp)`, and the shared
constant at the pin. -/
def stackPlace (x : Fin (R.stackW k)) : Fin (2 ^ S.kLog) :=
  if h1 : x.val < k * R.nIn then
    ⟨base (x.val / R.nIn) + x.val % R.nIn, by
      have hn : 0 < R.nIn := Nat.pos_of_ne_zero fun h => by simp [h] at h1
      have hu : x.val / R.nIn < k := (Nat.div_lt_iff_lt_mul hn).2 h1
      have := hb _ hu; have := Nat.mod_lt x.val hn; omega⟩
  else if h2 : x.val < k * R.nIn + k * (R.nComp + 1) then
    ⟨base ((x.val - k * R.nIn) / (R.nComp + 1)) +
        (if (x.val - k * R.nIn) % (R.nComp + 1) = 0 then R.nIn + R.nComp
          else R.nIn + ((x.val - k * R.nIn) % (R.nComp + 1) - 1)), by
      have hu : (x.val - k * R.nIn) / (R.nComp + 1) < k := (Nat.div_lt_iff_lt_mul (by omega)).2 (by omega)
      have := hb _ hu
      have := Nat.mod_lt (x.val - k * R.nIn) (show 0 < R.nComp + 1 by omega)
      split <;> omega⟩
  else S.pin

theorem div_mod_copy {u n c : ℕ} (hc : c < n) : (u * n + c) / n = u ∧ (u * n + c) % n = c := by
  have hn : 0 < n := by omega
  refine ⟨?_, ?_⟩
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hc, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc]

/-- A copy's column, stacked and then placed, is `base u + c`. -/
theorem stackPlace_stackCol {u : ℕ} (hu : u < k) (c : Fin R.w) :
    stackPlace S R k base hb (R.stackCol k hu c) = posIn S R k base hb hu c := by
  have hc := c.isLt
  simp only [Rows.w] at hc
  unfold Rows.stackCol
  split
  · rename_i h
    have hlt := Rows.mul_add_lt hu h
    obtain ⟨hd, hm⟩ := div_mod_copy (u := u) h
    unfold stackPlace
    rw [dite_eq_left (show u * R.nIn + c.val < k * R.nIn from hlt)]
    simp only [posIn, hd, hm]
  · split
    · rename_i h h'
      have hlt := Rows.mul_add_lt hu (show 1 + (c.val - R.nIn) < R.nComp + 1 by omega)
      obtain ⟨hd, hm⟩ := div_mod_copy (u := u) (show 1 + (c.val - R.nIn) < R.nComp + 1 by omega)
      unfold stackPlace
      simp only
      rw [dite_eq_right (by omega), dite_eq_left (by omega)]
      simp only [posIn, Nat.add_sub_cancel_left, hd, hm, show 1 + (c.val - R.nIn) ≠ 0 by omega, ite_false]
      congr 2
      omega
    · rename_i h h'
      have hlt := Rows.mul_add_lt hu (show 0 < R.nComp + 1 by omega)
      obtain ⟨hd, hm⟩ := div_mod_copy (u := u) (show 0 < R.nComp + 1 by omega)
      unfold stackPlace
      simp only
      rw [dite_eq_right (by omega), dite_eq_left (by omega)]
      simp only [posIn, Nat.add_sub_cancel_left, hd, hm, ite_true]
      congr 2
      omega

theorem stackPlace_last (h : k * R.nIn + k * (R.nComp + 1) < R.stackW k) :
    stackPlace S R k base hb ⟨k * R.nIn + k * (R.nComp + 1), h⟩ = S.pin := by
  unfold stackPlace
  rw [dite_eq_right (by simp), dite_eq_right (by simp)]

theorem stackPlace_one : stackPlace S R k base hb (R.stack k).one = S.pin :=
  stackPlace_last S R k base hb _

/-- Stacked computed row `i` of copy `u = i / (nComp + 1)` sits at the copy's constant when it is the
copy's constant row, and at the copy's computed row otherwise. -/
theorem stackPlace_comp (i : Fin (R.stack k).nComp) :
    stackPlace S R k base hb ((R.stack k).comp i) =
      if hj : i.val % (R.nComp + 1) = 0 then posIn S R k base hb (R.stack_copy_lt k i.isLt) R.one
      else posIn S R k base hb (R.stack_copy_lt k i.isLt)
        (R.comp ⟨i.val % (R.nComp + 1) - 1, by
          have := Nat.mod_lt i.val (show 0 < R.nComp + 1 by omega); omega⟩) := by
  have hi : i.val < k * (R.nComp + 1) := i.isLt
  have hc : ((R.stack k).comp i).val = k * R.nIn + i.val := rfl
  have hn : (R.stack k).nIn = k * R.nIn := rfl
  unfold stackPlace
  rw [dite_eq_right (by omega), dite_eq_left (by omega)]
  apply Fin.ext
  by_cases hj : i.val % (R.nComp + 1) = 0
  · rw [dite_eq_left hj]
    simp only [Rows.comp, hn, Nat.add_sub_cancel_left, hj, ite_true, posIn, Rows.one]
  · rw [dite_eq_right hj]
    simp only [Rows.comp, hn, Nat.add_sub_cancel_left, hj, ite_false, posIn]

/-- **An instance's rows placed in a statement.** If, at every copy `u`'s computed rows, the level-0
matrices read the template's rows shifted to the copy's slot (`hrowA`, `hrowB`), and every copy's
constant row and the pin's own row read the pin (`hconstA`, …), then the stacked rows are placed at
`stackPlace`. -/
theorem placement_stack
    (hrowA : ∀ u (hu : u < k) (i : Fin R.nComp) (j : Fin (2 ^ S.kLog)),
      S.A₀ (posIn S R k base hb hu (R.comp i)) j = (((R.a i).map (posIn S R k base hb hu)).count j : ZMod 2))
    (hrowB : ∀ u (hu : u < k) (i : Fin R.nComp) (j : Fin (2 ^ S.kLog)),
      S.B₀ (posIn S R k base hb hu (R.comp i)) j = (((R.b i).map (posIn S R k base hb hu)).count j : ZMod 2))
    (hconstA : ∀ u (hu : u < k) (j : Fin (2 ^ S.kLog)),
      S.A₀ (posIn S R k base hb hu R.one) j = if j = S.pin then 1 else 0)
    (hconstB : ∀ u (hu : u < k) (j : Fin (2 ^ S.kLog)),
      S.B₀ (posIn S R k base hb hu R.one) j = if j = S.pin then 1 else 0)
    (hpinA : ∀ j, S.A₀ S.pin j = if j = S.pin then 1 else 0)
    (hpinB : ∀ j, S.B₀ S.pin j = if j = S.pin then 1 else 0) :
    Placement S (R.stack k) (stackPlace S R k base hb) := by
  have key : ∀ (M : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)) (f : Fin R.nComp → List (Fin R.w))
      (hrow : ∀ u (hu : u < k) (i : Fin R.nComp) (j : Fin (2 ^ S.kLog)),
        M (posIn S R k base hb hu (R.comp i)) j = (((f i).map (posIn S R k base hb hu)).count j : ZMod 2))
      (hconst : ∀ u (hu : u < k) (j : Fin (2 ^ S.kLog)),
        M (posIn S R k base hb hu R.one) j = if j = S.pin then 1 else 0)
      (i : Fin (R.stack k).nComp) (j : Fin (2 ^ S.kLog)),
      M (stackPlace S R k base hb ((R.stack k).comp i)) j =
        (((R.stackRow k f i).map (stackPlace S R k base hb)).count j : ZMod 2) := by
    intro M f hrow hconst i j
    rw [stackPlace_comp]
    unfold Rows.stackRow
    split
    · rename_i hj
      rw [hconst]
      simp only [List.map_cons, List.map_nil, stackPlace_last, List.count_singleton]
      by_cases h : j = S.pin
      · simp [h]
      · simp [h, Ne.symm h]
    · rename_i hj
      rw [hrow, List.map_map]
      congr 3
      funext c
      exact (stackPlace_stackCol S R k base hb _ c).symm
  exact {
    hA := key S.A₀ R.a hrowA hconstA
    hB := key S.B₀ R.b hrowB hconstB
    hA₁ := fun j => by rw [stackPlace_one]; exact hpinA j
    hB₁ := fun j => by rw [stackPlace_one]; exact hpinB j }

end Place

end FlockSoundness