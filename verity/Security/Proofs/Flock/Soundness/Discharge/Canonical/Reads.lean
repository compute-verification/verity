import Proofs.Flock.Soundness.Discharge.Canonical.Format
import Proofs.Flock.Soundness.Discharge.Layout.KeysRow

set_option Elab.async false

/-!
# Every unit input a computed row reads is a row bit or the zero, at any number of input rows and groups

`Layout.keysRow_of_parse` gives `KeysRow` at the row widths `16 · words` under `inGroups.size = 1` (the unit reads its
leaf group only) and `PortReads`. A canonical circuit may read any number of input rows, through its leaves (group 0) or
its leaf cuts (another group's port read straight from a row word, `HmRow.leafCuts`), and its rows are bit rows of
`ℓ_q` bits. `keysRowC_of_parse` is `KeysRow` at the canonical widths `rowW c q = ℓ_q`, from `formatOk`.

Δ's key of a read input is `Aliased.srcAt` at its port and bit (`keysRow_of_parse`'s first half, unchanged). `srcAt`
gives `.zero`, `.msg q (2 (x - start_q) + t / 8) (t % 8)` exactly when `16 (x - start_q) + t < ℓ_q` (a leaf's bit, or a
leaf cut's with `t < 16`), `.wire` or `.lit`; `formatOk`'s `readsRows` excludes the last two. The message key's bit
`8 · byte + bit = 16 (x - start_q) + t` is below `ℓ_q`, so it is a row bit at the canonical width. Nothing about the
key depends on how many groups the unit has.

`HiddenExec.copy_siteH` (a message-keyed input holds its row's bit at `Typed.msgCol`) already holds for any rows.

Every declaration here is proved: `keysRow_of_parse`'s proof with `Aliased.srcAt_leaf` replaced by a case on the key,
the width bound read from `srcAt`'s own `if` (`srcAt_msg`), and `q < ports.size` because a port past the end has no bits.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3

/-- **A message key is a bit below its row's length**: `Aliased.srcAt` gives `.msg q byte bit` only at a row bit,
`16 (x - start_q) + t = 8 · byte + bit < ℓ_q`, a leaf's or a leaf cut's. -/
theorem srcAt_msg {c : Circuit} {k p t q byte bit : ℕ} (h : Aliased.srcAt c k p t = .msg q byte bit) :
    8 * byte + bit < c.ports[q]!.bits ∧ bit < 8 := by
  unfold Aliased.srcAt at h
  split at h
  · split at h
    · cases h
    · simp only at h
      split at h
      · rename_i hc
        injection h with h1 h2 h3
        subst h1 h2 h3
        exact ⟨by omega, Nat.mod_lt _ (by decide)⟩
      · cases h
  · split at h
    · simp only at h
      split at h
      · rename_i hc
        injection h with h1 h2 h3
        subst h1 h2 h3
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
        exact ⟨by omega, Nat.mod_lt _ (by decide)⟩
      · cases h
    · split at h <;> cases h

/-- **`KeysRow` at the canonical widths, from the parse**: in a canonical circuit `HmRow.parse` accepts, every input a
computed row reads has the zero or a row bit below its row's `ℓ_q` bits as its key, for any number of input rows and
groups.

*Route*: the module docstring's. -/
theorem keysRowC_of_parse {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hU : Built c.unit) (_h : HmRow.parse tags bytes tables = .ok c) (hfmt : formatOk c = true) :
    Layout.KeysRow c hU (rowW c) := by
  obtain ⟨-, -, hrd, hpr, -⟩ := formatOk_spec hU hfmt
  intro i x hx hxn
  set R := Rows.ofNet hU with hR
  have hxn' : x.val < c.upv * R.nIn := hxn
  have hx' : x ∈ R.stackRow c.upv R.a i ++ R.stackRow c.upv R.b i := hx
  obtain ⟨j, y, hy, hyn, -, hxy⟩ : ∃ (j : Fin R.nComp) (y : Fin R.w), y ∈ R.a j ++ R.b j ∧ y.val < R.nIn ∧
      i.val / (R.nComp + 1) < c.upv ∧ x.val = i.val / (R.nComp + 1) * R.nIn + y.val := by
    rcases List.mem_append.1 hx' with hx' | hx'
    · obtain ⟨j, y, hy, h2, h3, h4⟩ := Layout.stackRow_input hx' hxn'
      exact ⟨j, y, List.mem_append_left _ hy, h2, h3, h4⟩
    · obtain ⟨j, y, hy, h2, h3, h4⟩ := Layout.stackRow_input hx' hxn'
      exact ⟨j, y, List.mem_append_right _ hy, h2, h3, h4⟩
  have hm := hpr j y hy hyn
  set u := i.val / (R.nComp + 1)
  have hW : R.nIn = c.unit.inWords * WORD := Layout.ofNet_nIn_eq
  rw [hW] at hyn hxy hxn'
  have hmod : x.val % (c.unit.inWords * WORD) = y.val := by
    rw [hxy, Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hyn]
  have hdiv : x.val / (c.unit.inWords * WORD) = u := by
    rw [hxy, Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hyn, Nat.add_zero]
  obtain ⟨p, t, hp, ht, hpt⟩ := Aliased.mem_portBits hm
  have hnd := Aliased.portBits_nodup hU
  have hkey : Aliased.srcKey c x.val = Aliased.srcAt c u p t := by
    unfold Aliased.srcKey
    rw [hmod, ite_eq_left hm, hdiv, hpt, FlockSoundness.portOf_eq hnd hp ht, Nat.add_sub_cancel_left]
  obtain ⟨hnw, hnl⟩ := hrd x.val hxn'
  unfold Layout.RowKey
  rcases hs : Aliased.srcKey c x.val with _ | ⟨q, byte, bit⟩ | ⟨net, slot, col⟩ | ⟨col⟩ | ⟨k, off⟩
  · exact Or.inl rfl
  · refine Or.inr ⟨q, byte, bit, rfl, ?_⟩
    rw [hkey] at hs
    obtain ⟨hb, hbit⟩ := srcAt_msg hs
    have hq : q < c.ports.size := by
      by_contra hq
      rw [getElem!_neg c.ports q hq] at hb
      exact absurd hb (Nat.not_lt_zero _)
    rw [getElem!_pos c.ports q hq] at hb
    exact ⟨hq, hb, hbit⟩
  · exact absurd hs (hnw _ _ _)
  · exact absurd hs (hnl _)
  · rw [hkey] at hs
    exact absurd hs (Aliased.srcAt_ne_pad _ _ _ _ _ _)

end FlockSoundness.Discharge.Canonical
