/-
A templated statement's template read off the file's line view (`ByteLines.View`), which its parse cuts once:
`HmOut.tmplOfView` is `HmOut.tmplOf` (`tmplOfView_eq`), without its second scan for `'\n'`s, its UTF-8 check and its
copy of the template's text. The template's parse reads its lines through the view (`View.tmplLine`), and each row's
numbers off the file's bytes (`View.tmplNums`), so a row makes no `String` (`View.parseAt_tmpl`).
-/
module

public import Flock.ByteLines
public import Flock.Tmpl
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

/-- What `findByte` finds from `i`: the first byte `x` at or past `i`. -/
theorem findByte_some {b : ByteArray} {x : UInt8} : ∀ (n i c : Nat), b.size - i ≤ n → findByte b x i = some c →
    ∃ hc : c < b.size, i ≤ c ∧ b[c] = x ∧ ∀ j (hj : j < c), i ≤ j → b[j]'(by omega) ≠ x := by
  intro n
  induction n with
  | zero =>
    intro i c hn h
    rw [findByte_step, dite_eq_right_of_eq_false (eq_false (by omega))] at h
    exact absurd h (by simp)
  | succ n ih =>
    intro i c hn h
    rw [findByte_step] at h
    by_cases hi : i < b.size
    · rw [dite_eq_left_of_eq_true (eq_true hi)] at h
      by_cases hx : b[i] = x
      · rw [ite_eq_left (by simp [hx])] at h
        cases h
        exact ⟨hi, Nat.le_refl _, hx, fun j hj hij => absurd hij (by omega)⟩
      · rw [ite_eq_right (by simp [hx])] at h
        obtain ⟨hc, hic, hcx, hlt⟩ := ih (i + 1) c (by omega) h
        refine ⟨hc, by omega, hcx, fun j hj hij => ?_⟩
        rcases Nat.eq_or_lt_of_le hij with rfl | hlt'
        · exact hx
        · exact hlt j hj (by omega)
    · rw [dite_eq_right_of_eq_false (eq_false hi)] at h
      exact absurd h (by simp)

namespace ByteLines

theorem getElem?_chars (b : ByteArray) (i : Nat) : (chars b)[i]? = b.data[i]?.map Char.ofUInt8 := by
  simp [chars]

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
    rw [ite_eq_left rfl, findByte_step, dite_eq_right_of_eq_false (eq_false (by omega))]
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
            rw [hp acc.size (by simp), Array.getElem?_push_size]
          by_cases hka : k = acc.size
          · subst hka
            rw [ite_eq_left rfl, findByte_step, dite_eq_left_of_eq_true (eq_true hi), hn, hci]
            rfl
          · have := ih (i + 1) (acc.push i) cs (by omega) h k (by simp; omega) hkc
            rw [ite_eq_right hka]
            by_cases hk1 : k = acc.size + 1
            · subst hk1
              rw [ite_eq_left (by simp)] at this
              have hc1 : cs[acc.size + 1 - 1]! = i := by
                rw [Nat.add_sub_cancel, getElem!_def, hci]
              rw [hc1]
              exact this
            · rw [ite_eq_right (by simp; omega)] at this
              exact this
        · have hb : (b[i] == 10) = false := by simpa using hn
          simp only [hb, Bool.false_eq_true, ite_false] at h
          have := ih (i + 1) acc cs (by omega) h k hk hkc
          by_cases hka : k = acc.size
          · rw [ite_eq_left hka] at this ⊢
            rw [findByte_step, dite_eq_left_of_eq_true (eq_true hi), hb]
            exact this
          · rw [ite_eq_right hka] at this ⊢
            exact this
      · exact absurd h (by simp)
    · rw [dite_eq_right_of_eq_false (eq_false hi)] at h
      simp only [Option.some.injEq] at h
      subst h
      have hk' : k = acc.size := by omega
      subst hk'
      rw [ite_eq_left rfl, findByte_step, dite_eq_right_of_eq_false (eq_false hi)]
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

theorem View.toList_of? {b : ByteArray} {v : View} (h : View.of? b = some v) :
    v.s.toList = chars b ∧ v.s.utf8ByteSize = b.size := by
  obtain ⟨hs, _⟩ := View.of?_spec h
  have ha := View.ascii_of? h
  have hvb := valid_ascii ha
  have hsv : v.s = String.fromUTF8 b hvb := by
    unfold String.fromUTF8? at hs
    rw [dite_eq_left_of_eq_true (eq_true hvb)] at hs
    exact (Option.some.inj hs).symm
  exact ⟨by rw [hsv, toList_ascii ha hvb], by rw [hsv, ← String.size_toByteArray]; rfl⟩

/-- Line `j`, below the number of cuts: the file from its start to its cut, a `'\n'`, with no `'\n'` before it. -/
theorem View.str_cut {b : ByteArray} {v : View} (h : View.of? b = some v) {j : Nat} (hj : j < v.cuts.size) :
    v.start j ≤ v.cuts[j]! ∧ v.cuts[j]! < v.s.utf8ByteSize ∧ v.s.toList[v.cuts[j]!]? = some '\n' ∧
      String.Pos.Raw.extract v.s ⟨v.start j⟩ ⟨v.cuts[j]!⟩ = v.str j ∧
      (v.str j).toList = (v.s.toList.drop (v.start j)).take (v.cuts[j]! - v.start j) ∧ '\n' ∉ (v.str j).toList := by
  obtain ⟨_, hok⟩ := View.of?_spec h
  obtain ⟨hF, hsz⟩ := View.toList_of? h
  have hf := View.findByte_start h (Nat.le_of_lt hj)
  rw [Array.getElem?_eq_getElem hj] at hf
  obtain ⟨hc, hle, hnl, hnot⟩ := findByte_some (b.size - v.start j) (v.start j) v.cuts[j] (Nat.le_refl _) hf
  have hj! : v.cuts[j]! = v.cuts[j] := getElem!_pos v.cuts j hj
  have hcnt : j < v.count := by have := hok.count; split at this <;> omega
  have hstr : String.Pos.Raw.extract v.s ⟨v.start j⟩ ⟨v.cuts[j]!⟩ = v.str j := by
    simp [View.str, hcnt, View.stop, Array.getElem?_eq_getElem hj, hj!]
  have hex := (extract_ascii hok.ascii (hj! ▸ hle) (by rw [hj!]; omega)).2
  have hl : (v.str j).toList = (v.s.toList.drop (v.start j)).take (v.cuts[j]! - v.start j) := by
    rw [← hstr, hex, String.toList_ofList]
  have hnl' : v.s.toList[v.cuts[j]!]? = some '\n' := by
    rw [hF, hj!, getElem?_chars, Array.getElem?_eq_getElem (by simpa using hc)]
    simp only [Option.map_some, Option.some.injEq]
    have : b.data[v.cuts[j]]'(by simpa using hc) = 10 := hnl
    rw [this]; rfl
  refine ⟨hj! ▸ hle, by rw [hj!, hsz]; exact hc, hnl', hstr, hl, ?_⟩
  rw [hl, hF]
  intro hm
  obtain ⟨i, hi⟩ := List.mem_iff_getElem?.mp hm
  rw [List.getElem?_take] at hi
  split at hi
  · rename_i hlt
    rw [List.getElem?_drop, getElem?_chars, Array.getElem?_eq_getElem (by simp; omega)] at hi
    simp only [Option.map_some, Option.some.injEq] at hi
    have hb := nl_iff (b.data[v.start j + i]'(by simp; omega))
    rw [hi] at hb
    simp only [beq_self_eq_true] at hb
    have hb' : b.data[v.start j + i]'(by simp; omega) = 10 := by simpa using hb.symm
    exact hnot (v.start j + i) (by omega) (by omega) hb'
  · exact absurd hi (by simp)

theorem View.start_succ (v : View) (j : Nat) : v.start (j + 1) = v.cuts[j]! + 1 := by
  simp [View.start]

theorem View.cut_mono {b : ByteArray} {v : View} (h : View.of? b = some v) :
    ∀ (m j : Nat), j + m < v.cuts.size → v.cuts[j]! + m ≤ v.cuts[j + m]! := by
  intro m
  induction m with
  | zero => intro j _; simp
  | succ m ih =>
    intro j hjm
    have h1 := ih j (by omega)
    obtain ⟨hle, _⟩ := View.str_cut h (j := j + m + 1) (by omega)
    rw [View.start_succ] at hle
    rw [show j + (m + 1) = j + m + 1 by omega]
    omega

/-- The file from line `j`'s start to cut `j + m`, split at its `'\n'`s: lines `j` to `j + m`. -/
theorem View.splitOn_cuts {b : ByteArray} {v : View} (h : View.of? b = some v) : ∀ (m j : Nat), j + m < v.cuts.size →
    (String.Pos.Raw.extract v.s ⟨v.start j⟩ ⟨v.cuts[j + m]!⟩).toList.splitOn '\n' =
      (List.range (m + 1)).map fun i => (v.str (j + i)).toList := by
  obtain ⟨_, hok⟩ := View.of?_spec h
  intro m
  induction m with
  | zero =>
    intro j hjm
    obtain ⟨_, _, _, hstr, _, hnl⟩ := View.str_cut h (j := j) (by omega)
    rw [Nat.add_zero, hstr, List.splitOn_eq_singleton hnl]
    simp
  | succ m ih =>
    intro j hjm
    obtain ⟨hle, hlt, hnl, _, hl, hno⟩ := View.str_cut h (j := j) (by omega)
    have hmono := View.cut_mono h (m + 1) j hjm
    have hend : v.cuts[j + (m + 1)]! < v.s.utf8ByteSize := (View.str_cut h (j := j + (m + 1)) hjm).2.1
    have hs1 := View.start_succ v j
    have hsplit : (String.Pos.Raw.extract v.s ⟨v.start j⟩ ⟨v.cuts[j + (m + 1)]!⟩).toList =
        (v.str j).toList ++ '\n' ::
          (String.Pos.Raw.extract v.s ⟨v.start (j + 1)⟩ ⟨v.cuts[j + 1 + m]!⟩).toList := by
      rw [show j + 1 + m = j + (m + 1) by omega, hs1,
        (extract_ascii hok.ascii (by omega) (by omega)).2, (extract_ascii hok.ascii (by omega) (by omega)).2,
        String.toList_ofList, String.toList_ofList, hl,
        show v.cuts[j + (m + 1)]! - v.start j = (v.cuts[j]! - v.start j) + (v.cuts[j + (m + 1)]! - v.cuts[j]!) by omega,
        List.take_add, List.drop_drop, show v.start j + (v.cuts[j]! - v.start j) = v.cuts[j]! by omega]
      congr 1
      have hc : v.cuts[j]! < v.s.toList.length := by
        have := Scan.size_ascii hok.ascii; omega
      rw [List.drop_eq_getElem_cons hc, show v.cuts[j + (m + 1)]! - v.cuts[j]! =
        (v.cuts[j + (m + 1)]! - (v.cuts[j]! + 1)) + 1 by omega, List.take_succ_cons]
      have : v.s.toList[v.cuts[j]!] = '\n' := by
        rw [List.getElem?_eq_getElem hc] at hnl; exact Option.some.inj hnl
      rw [this]
    rw [hsplit, List.splitOn_append_cons_self_of_not_mem hno, ih (j + 1) (by omega)]
    conv => rhs; rw [List.range_succ_eq_map]
    simp only [List.map_cons, List.map_map, Nat.add_zero]
    congr 1
    apply List.map_congr_left
    intro i _
    simp only [Function.comp_apply]
    rw [show j + 1 + i = j + (i + 1) by omega]

/-- Line `i` of the template's `k` on the view, from the file's line 3 on; `""` past the last. -/
def View.tmplLine (v : View) (k i : Nat) : String := if i < k then v.str (3 + i) else ""

/-- `View.tmplLine`'s numbers, read off the file's bytes (`View.nums`). -/
def View.tmplNums (v : View) (k i : Nat) : Option (Array Nat) := if i < k then v.nums (3 + i) else Net.lineNums ""

/-- The template's parse through the view is its parse of the template's lines. -/
theorem View.parseAt_tmpl {b : ByteArray} {v : View} (h : View.of? b = some v) (k : Nat) :
    Tmpl.parseAt k (v.tmplLine k) (v.tmplNums k) = Tmpl.parse ((Array.range k).map fun i => v.str (3 + i)) := by
  have hg : ∀ i, ((Array.range k).map fun i => v.str (3 + i)).getD i "" = v.tmplLine k i := fun i => by
    unfold View.tmplLine
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range]
    by_cases hi : i < k <;> simp [hi]
  rw [Tmpl.parse_eq_fast]
  unfold Tmpl.parseFast
  congr 1
  · simp
  · funext i; exact (hg i).symm
  · funext i
    rw [hg]
    unfold View.tmplLine View.tmplNums
    by_cases hi : i < k
    · simp only [hi, ↓reduceIte]; exact (of?_spec h).2.nums (3 + i)
    · simp only [hi, ↓reduceIte]

/-- `HmOut.tmplOf`'s loop from cut `j`: `l.length` more `'\n'`s, so cut `j + l.length`, or `msg` past the last cut. -/
theorem View.forIn_cuts {b : ByteArray} {v : View} (h : View.of? b = some v) (msg : String)
    (f : Nat → Nat → V (ForInStep Nat))
    (hf : ∀ x e, f x e = match findByte b 10 (e + 1) with
      | some e' => pure (ForInStep.yield e')
      | none => throw msg) :
    ∀ (l : List Nat) (j : Nat), j < v.cuts.size →
      forIn l v.cuts[j]! f = if j + l.length < v.cuts.size then pure v.cuts[j + l.length]! else throw msg := by
  intro l
  induction l with
  | nil =>
    intro j hj
    rw [List.length_nil, Nat.add_zero, ite_eq_left hj]
    rfl
  | cons x l ih =>
    intro j hj
    rw [List.forIn_cons, hf, ← View.start_succ, View.findByte_start h (by omega)]
    by_cases hj1 : j + 1 < v.cuts.size
    · rw [Array.getElem?_eq_getElem hj1]
      simp only [pure_bind]
      rw [← getElem!_pos v.cuts (j + 1) hj1, ih (j + 1) hj1, List.length_cons,
        show j + 1 + l.length = j + (l.length + 1) by omega]
    · rw [Array.getElem?_eq_none (by omega), ite_eq_right (by simp; omega)]
      rfl

end ByteLines

/-- `HmOut.tmplOf` off the file's line view `v` (`tmplOfView_eq`): its lines 3 to `2 + k` under the header
`TEMPLATE k` on line 2, the last of them ended by a `'\n'`. -/
def HmOut.tmplOfView (v : ByteLines.View) : V (Array String) :=
  let bad := "circuit: no template section (TEMPLATE n, then the template's n lines) after META"
  if v.cuts.size < 3 then throw bad
  else
    let h := tokens (v.str 2)
    match (if h.size == 2 && h[0]! == "TEMPLATE" then h[1]!.toNat? else none) with
    | none => throw bad
    | some k =>
      if k == 0 then pure #[]
      else if v.cuts.size < 3 + k then throw "circuit: the template section is truncated"
      else pure ((Array.range k).map fun i => v.str (3 + i))

theorem HmOut.tmplOfView_eq {b : ByteArray} {v : ByteLines.View} (h : ByteLines.View.of? b = some v) :
    HmOut.tmplOfView v = HmOut.tmplOf b := by
  open ByteLines in
  have hsome : ∀ j, j < v.cuts.size → v.cuts[j]? = some v.cuts[j]! := fun j hj => by
    rw [getElem!_pos v.cuts j hj, Array.getElem?_eq_getElem hj]
  have hnext : ∀ j, j < v.cuts.size → findByte b 10 (v.cuts[j]! + 1) = v.cuts[j + 1]? := fun j hj => by
    rw [← ByteLines.View.start_succ]; exact ByteLines.View.findByte_start h (by omega)
  have h0 : findByte b 10 = v.cuts[0]? := by
    have := ByteLines.View.findByte_start h (k := 0) (Nat.zero_le _)
    simpa [ByteLines.View.start] using this
  obtain ⟨hF, hsz⟩ := ByteLines.View.toList_of? h
  obtain ⟨_, hok⟩ := ByteLines.View.of?_spec h
  unfold HmOut.tmplOf HmOut.tmplOfView
  simp only [h0]
  rcases Nat.lt_or_ge v.cuts.size 3 with h3 | h3
  · rw [ite_eq_left h3]
    rcases (show v.cuts.size = 0 ∨ v.cuts.size = 1 ∨ v.cuts.size = 2 by omega) with hs | hs | hs
    · rw [Array.getElem?_eq_none (by omega)]
    · rw [hsome 0 (by omega)]
      simp only []
      rw [hnext 0 (by omega), Array.getElem?_eq_none (by omega)]
    · rw [hsome 0 (by omega)]
      simp only []
      rw [hnext 0 (by omega), hsome 1 (by omega)]
      simp only []
      rw [hnext 1 (by omega), Array.getElem?_eq_none (by omega)]
  · rw [ite_eq_right (by omega), hsome 0 (by omega)]
    simp only []
    rw [hnext 0 (by omega), hsome 1 (by omega)]
    simp only []
    rw [hnext 1 (by omega), hsome 2 (by omega)]
    simp only []
    obtain ⟨hle2, hlt2, _, hstr2, _⟩ := ByteLines.View.str_cut h (j := 2) (by omega)
    rw [ByteLines.View.start_succ] at hle2 hstr2
    rw [ByteLines.View.fromUTF8?_extract h hle2 (by omega), hstr2]
    simp only []
    split
    · rename_i hko
      rw [hko]
    · rename_i k hko
      rw [hko]
      simp only []
      by_cases hk0 : k = 0
      · subst hk0; rfl
      · have hk0' : (k == 0) = false := by simpa using hk0
        simp only [hk0', Bool.false_eq_true, ite_false]
        simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel,
          Nat.div_one]
        erw [ByteLines.View.forIn_cuts h "circuit: the template section is truncated" _ (fun x e => by
            (try dsimp only); cases findByte b 10 (e + 1) <;> rfl) (List.range' 0 k) 2 (by omega)]
        rw [List.length_range']
        by_cases hk : v.cuts.size < 3 + k
        · rw [ite_eq_left hk, ite_eq_right (by omega)]
          rfl
        · rw [ite_eq_right hk, ite_eq_left (by omega)]
          simp only [pure_bind]
          obtain ⟨_, hltk, _⟩ := ByteLines.View.str_cut h (j := 2 + k) (by omega)
          have hmono := ByteLines.View.cut_mono h k 2 (by omega)
          rw [ByteLines.View.fromUTF8?_extract h (by omega) (by omega)]
          simp only []
          have hsp := ByteLines.View.splitOn_cuts h (k - 1) 3 (by omega)
          rw [ByteLines.View.start_succ, show 3 + (k - 1) = 2 + k by omega,
            show k - 1 + 1 = k by omega] at hsp
          rw [SplitNL.splitOn_nl, hsp]
          congr 1
          apply Array.toList_inj.mp
          simp [Function.comp_def]

/-- How many lines the template on the view `v` has, with `tmplOfView`'s errors (`tmplOfView_count`): the parse reads
them off the view (`View.tmplLine`), never as one array. -/
def HmOut.tmplCountView (v : ByteLines.View) : V Nat :=
  let bad := "circuit: no template section (TEMPLATE n, then the template's n lines) after META"
  if v.cuts.size < 3 then throw bad
  else
    let h := tokens (v.str 2)
    match (if h.size == 2 && h[0]! == "TEMPLATE" then h[1]!.toNat? else none) with
    | none => throw bad
    | some k =>
      if k == 0 then pure 0
      else if v.cuts.size < 3 + k then throw "circuit: the template section is truncated"
      else pure k

theorem HmOut.tmplOfView_count (v : ByteLines.View) :
    HmOut.tmplOfView v = (fun k => (Array.range k).map fun i => v.str (3 + i)) <$> HmOut.tmplCountView v := by
  unfold tmplOfView tmplCountView
  by_cases h3 : v.cuts.size < 3
  · simp only [h3, ↓reduceIte]; rfl
  · simp only [h3, ↓reduceIte]
    split
    · rfl
    · rename_i k _
      by_cases hk0 : k = 0
      · subst hk0
        simp only [beq_self_eq_true, ↓reduceIte, map_pure, Array.range_eq_empty_iff.mpr rfl, Array.map_empty]
      · have hk0' : (k == 0) = false := by simpa using hk0
        simp only [hk0', Bool.false_eq_true, ↓reduceIte]
        by_cases hk : v.cuts.size < 3 + k
        · simp only [hk, ↓reduceIte]; rfl
        · simp only [hk, ↓reduceIte]; rfl

end Flock
