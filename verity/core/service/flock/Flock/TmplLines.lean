/-
A templated statement's template read off the file's line view (`ByteLines.View`), which its parse cuts once:
`HmOut.tmplOfView` is `HmOut.tmplOf` (`tmplOfView_eq`), without its second scan for `'\n'`s, its UTF-8 check and its
copy of the template's text.
-/
module

public import Flock.ByteLines
public import Flock.TmplSection

@[expose] public section

namespace Flock

theorem findByte_eq_find (b : ByteArray) (x : UInt8) (i : Nat) :
    findByte b x i = (List.range' i (b.size - i)).find? fun j => b.get! j == x := by
  have hl : ∀ (l : List Nat), (forIn (m := Id) l ((none : Option (Option Nat)), ()) fun j _ =>
      if (b.get! j == x) = true then pure (ForInStep.done (some (some j), ())) else pure (ForInStep.yield (none, ()))) =
      ((l.find? fun j => b.get! j == x).map some, ()) := by
    intro l
    induction l with
    | nil => rfl
    | cons a l ih =>
      simp only [List.forIn_cons, List.find?_cons]
      by_cases hx : (b.get! a == x) = true
      · simp only [hx, ite_true]; rfl
      · simp only [hx, Bool.false_eq_true, ite_false]
        exact ih
  unfold findByte
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.add_sub_cancel, Nat.div_one]
  rw [hl]
  cases (List.range' i (b.size - i)).find? fun j => b.get! j == x <;> rfl

/-- `findByte` from `i`: byte `i` when it is `x`, else `findByte` from `i + 1`. -/
theorem findByte_step (b : ByteArray) (x : UInt8) (i : Nat) :
    findByte b x i = if h : i < b.size then (if b[i] == x then some i else findByte b x (i + 1)) else none := by
  rw [findByte_eq_find, findByte_eq_find]
  by_cases h : i < b.size
  · rw [dite_eq_left_of_eq_true (eq_true h), show b.size - i = (b.size - (i + 1)) + 1 by omega, List.range'_succ,
      List.find?_cons]
    have hg : b.get! i = b[i] := by
      have : b.get! i = b.data[i]! := by cases b; rfl
      rw [this, getElem!_pos b.data i h]; rfl
    rw [hg]
    cases b[i] == x <;> rfl
  · rw [dite_eq_right_of_eq_false (eq_false h), show b.size - i = 0 by omega]
    rfl

namespace ByteLines

/-- `cutsN` keeps what `acc` holds and adds to it. -/
theorem cutsN_prefix (b : ByteArray) : ∀ (fuel i : Nat) (acc cs : Array Nat), cutsN b fuel i acc = some cs →
    acc.size ≤ cs.size ∧ ∀ j < acc.size, cs[j]? = acc[j]? := by
  intro fuel
  induction fuel with
  | zero =>
    intro i acc cs h
    simp only [cutsN, Option.some.injEq] at h
    subst h
    exact ⟨Nat.le_refl _, fun _ _ => rfl⟩
  | succ fuel ih =>
    intro i acc cs h
    unfold cutsN at h
    by_cases hi : i < b.size
    · rw [dite_eq_left_of_eq_true (eq_true hi)] at h
      split at h
      · by_cases hn : b[i] = 10
        · simp only [hn, beq_self_eq_true, ite_true] at h
          obtain ⟨hs, hp⟩ := ih (i + 1) (acc.push i) cs h
          refine ⟨by simp at hs; omega, fun j hj => ?_⟩
          rw [hp j (by simp; omega), Array.getElem?_push_lt hj, Array.getElem?_eq_getElem hj]
        · have hb : (b[i] == 10) = false := by simpa using hn
          simp only [hb, Bool.false_eq_true, ite_false] at h
          exact ih (i + 1) acc cs h
      · exact absurd h (by simp)
    · rw [dite_eq_right_of_eq_false (eq_false hi)] at h
      simp only [Option.some.injEq] at h
      subst h
      exact ⟨Nat.le_refl _, fun _ _ => rfl⟩

/-- Each cut `cutsN` adds is the first `'\n'` `findByte` finds from the byte after the one before it (from `i`, the
first one it adds). -/
theorem cutsN_findByte (b : ByteArray) : ∀ (fuel i : Nat) (acc cs : Array Nat), b.size ≤ i + fuel →
    cutsN b fuel i acc = some cs → ∀ k, acc.size ≤ k → k ≤ cs.size →
    findByte b 10 (if k = acc.size then i else cs[k - 1]! + 1) = cs[k]? := by
  intro fuel
  induction fuel with
  | zero =>
    intro i acc cs hf h k hk hkc
    simp only [cutsN, Option.some.injEq] at h
    subst h
    have hk' : k = acc.size := by omega
    subst hk'
    rw [if_pos rfl, findByte_step, dite_eq_right_of_eq_false (eq_false (by omega))]
    simp
  | succ fuel ih =>
    intro i acc cs hf h k hk hkc
    unfold cutsN at h
    by_cases hi : i < b.size
    · rw [dite_eq_left_of_eq_true (eq_true hi)] at h
      split at h
      · by_cases hn : b[i] = 10
        · simp only [hn, beq_self_eq_true, ite_true] at h
          obtain ⟨hs, hp⟩ := cutsN_prefix b fuel (i + 1) (acc.push i) cs h
          have hci : cs[acc.size]? = some i := by
            rw [hp acc.size (by simp), Array.getElem?_push_eq]
          by_cases hka : k = acc.size
          · subst hka
            rw [if_pos rfl, findByte_step, dite_eq_left_of_eq_true (eq_true hi), hn, hci]
            rfl
          · have := ih (i + 1) (acc.push i) cs (by omega) h k (by simp; omega) hkc
            rw [if_neg hka]
            by_cases hk1 : k = acc.size + 1
            · subst hk1
              rw [if_pos (by simp)] at this
              have hc1 : cs[acc.size + 1 - 1]! = i := by
                rw [Nat.add_sub_cancel, getElem!_def, hci]
              rw [hc1]
              exact this
            · rw [if_neg (by simp; omega)] at this
              exact this
        · have hb : (b[i] == 10) = false := by simpa using hn
          simp only [hb, Bool.false_eq_true, ite_false] at h
          have := ih (i + 1) acc cs (by omega) h k hk hkc
          by_cases hka : k = acc.size
          · rw [if_pos hka] at this ⊢
            rw [findByte_step, dite_eq_left_of_eq_true (eq_true hi), hb]
            exact this
          · rw [if_neg hka] at this ⊢
            exact this
      · exact absurd h (by simp)
    · rw [dite_eq_right_of_eq_false (eq_false hi)] at h
      simp only [Option.some.injEq] at h
      subst h
      have hk' : k = acc.size := by omega
      subst hk'
      rw [if_pos rfl, findByte_step, dite_eq_right_of_eq_false (eq_false hi)]
      simp

theorem View.of?_cuts {b : ByteArray} {v : View} (h : View.of? b = some v) : cutsN b b.size 0 #[] = some v.cuts := by
  unfold View.of? at h
  split at h
  · rename_i hb
    split at h
    · rename_i cs hc
      simp only [Option.some.injEq] at h
      subst h
      have := cutsU_eq b _ (toUSize_toNat hb) b.size 0 #[] (by rw [toUSize_toNat hb]; simp)
      rw [USize.toNat_zero] at this
      rw [← this]
      exact hc
    · simp at h
  · simp at h

/-- The first `'\n'` from line `k`'s start is its cut. -/
theorem View.findByte_start {b : ByteArray} {v : View} (h : View.of? b = some v) {k : Nat} (hk : k ≤ v.cuts.size) :
    findByte b 10 (v.start k) = v.cuts[k]? := by
  have := cutsN_findByte b b.size 0 #[] v.cuts (by omega) (View.of?_cuts h) k (Nat.zero_le k) hk
  simp only [Array.size_empty] at this
  unfold View.start
  exact this

theorem View.ascii_of? {b : ByteArray} {v : View} (h : View.of? b = some v) : ∀ j (hj : j < b.size), b[j] < 128 :=
  fun j hj => cutsN_ascii b b.size 0 #[] v.cuts (by omega) (View.of?_cuts h) j hj (Nat.zero_le j)

/-- An ASCII file's bytes `[a, c)` decode to its view's string between them. -/
theorem View.fromUTF8?_extract {b : ByteArray} {v : View} (h : View.of? b = some v) {a c : Nat} (hac : a ≤ c)
    (hc : c ≤ b.size) : String.fromUTF8? (b.extract a c) = some (String.Pos.Raw.extract v.s ⟨a⟩ ⟨c⟩) := by
  obtain ⟨hs, hok⟩ := View.of?_spec h
  have ha := View.ascii_of? h
  have hesz : (b.extract a c).size = c - a := by simp [ByteArray.size_extract]; omega
  have hae : ∀ j (hj : j < (b.extract a c).size), (b.extract a c)[j] < 128 := by
    intro j hj
    have : (b.extract a c)[j] = b[a + j]'(by omega) := by simp [ByteArray.getElem_extract]
    rw [this]; exact ha _ _
  have hv := valid_ascii hae
  have hvb := valid_ascii ha
  have hsv : v.s = String.fromUTF8 b hvb := by
    unfold String.fromUTF8? at hs
    rw [dite_eq_left_of_eq_true (eq_true hvb)] at hs
    exact (Option.some.inj hs).symm
  have hsz : v.s.utf8ByteSize = b.size := by rw [hsv, ← String.size_toByteArray]; rfl
  unfold String.fromUTF8?
  rw [dite_eq_left_of_eq_true (eq_true hv), (extract_ascii hok.ascii hac (by omega)).2]
  congr 1
  apply String.toList_inj.mp
  rw [toList_ascii hae hv, String.toList_ofList, hsv, toList_ascii ha hvb]
  simp only [chars, ByteArray.data_extract, Array.toList_extract, List.extract, List.map_take, List.map_drop]

end ByteLines

end Flock
