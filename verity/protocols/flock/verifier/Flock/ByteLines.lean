/-
An ASCII circuit file's lines, read off its bytes: one pass finds every `'\n'` and checks every byte is ASCII (`cutsU`),
and a `View` holds the file's `String` and those positions. A line's `String` is cut from the file's only when read
(`View.str`), and a row's numbers are read off the file's bytes (`View.nums`), so a row makes no `String`. `View.of?_spec`:
the view's lines are those `HmRow.parseWith` splits off the file (`String.splitOn "\n"`, a last empty line dropped), and
each line's numbers are `Net.lineNums`'.
-/
module

public import Flock.SplitNL

@[expose] public section

namespace Flock.ByteLines

open SplitNL

/-- The positions of `b`'s bytes `10` from `i` on, pushed onto `acc`; `none` when a byte from `i` on is not ASCII. -/
def cutsN (b : ByteArray) : Nat → Nat → Array Nat → Option (Array Nat)
  | 0, _, acc => some acc
  | fuel + 1, i, acc =>
    if h : i < b.size then
      if b[i] < 128 then cutsN b fuel (i + 1) (if b[i] == 10 then acc.push i else acc) else none
    else some acc

/-- `cutsN` with the index a `USize` below `n`, `b`'s size. -/
def cutsU (b : ByteArray) (n : USize) (hn : n.toNat = b.size) (i : USize) (acc : Array Nat) : Option (Array Nat) :=
  if h : i < n then
    let x := b.uget i (by have := USize.lt_iff_toNat_lt.mp h; omega)
    if x < 128 then cutsU b n hn (i + 1) (if x == 10 then acc.push i.toNat else acc) else none
  else some acc
termination_by n.toNat - i.toNat
decreasing_by
  have := USize.lt_iff_toNat_lt.mp h
  rw [Scan.toNat_succ h]
  omega

theorem cutsU_eq (b : ByteArray) (n : USize) (hn : n.toNat = b.size) : ∀ (fuel : Nat) (i : USize) (acc : Array Nat),
    n.toNat - i.toNat ≤ fuel → cutsU b n hn i acc = cutsN b fuel i.toNat acc := by
  intro fuel
  induction fuel with
  | zero =>
    intro i acc hf
    have h : ¬ i < n := fun h => by have := USize.lt_iff_toNat_lt.mp h; omega
    rw [cutsU, dite_eq_right_of_eq_false (eq_false h), cutsN]
  | succ fuel ih =>
    intro i acc hf
    rw [cutsU, cutsN]
    by_cases h : i < n
    · have hi := USize.lt_iff_toNat_lt.mp h
      have hi' : i.toNat < b.size := by omega
      rw [dite_eq_left_of_eq_true (eq_true h), dite_eq_left_of_eq_true (eq_true hi')]
      have hg : b.uget i (by omega) = b[i.toNat] := rfl
      simp only [hg]
      split
      · rw [ih _ _ (by rw [Scan.toNat_succ h]; omega), Scan.toNat_succ h]
      · rfl
    · have hi : ¬ i.toNat < b.size := by have := mt USize.lt_iff_toNat_lt.mpr h; omega
      rw [dite_eq_right_of_eq_false (eq_false h), dite_eq_right_of_eq_false (eq_false hi)]

theorem cutsN_ascii (b : ByteArray) : ∀ (fuel i : Nat) (acc cs : Array Nat), b.size ≤ i + fuel →
    cutsN b fuel i acc = some cs → ∀ j (hj : j < b.size), i ≤ j → b[j] < 128 := by
  intro fuel
  induction fuel with
  | zero => intro i acc cs hf _ j hj hij; omega
  | succ fuel ih =>
    intro i acc cs hf h j hj hij
    unfold cutsN at h
    by_cases hi : i < b.size
    · rw [dite_eq_left_of_eq_true (eq_true hi)] at h
      split at h
      · rename_i hlt
        rcases Nat.eq_or_lt_of_le hij with rfl | hlt'
        · exact hlt
        · exact ih (i + 1) _ cs (by omega) h j hj (by omega)
      · exact absurd h (by simp)
    · omega

theorem ofUInt8_utf8Size {x : UInt8} (h : x < 128) : (Char.ofUInt8 x).utf8Size = 1 := by
  apply Char.utf8Size_eq_one_iff.mpr
  have hv : (Char.ofUInt8 x).val.toNat = x.toNat := UInt8.toNat_toUInt32 x
  have : x.toNat < 128 := UInt8.lt_iff_toNat_lt.mp h
  rw [UInt32.le_iff_toNat_le, hv]
  exact Nat.le_of_lt_succ this

theorem ofUInt8_toUInt8 (x : UInt8) : (Char.ofUInt8 x).val.toUInt8 = x := by
  apply UInt8.toNat_inj.mp
  rw [UInt32.toNat_toUInt8, show (Char.ofUInt8 x).val.toNat = x.toNat from UInt8.toNat_toUInt32 x]
  exact Nat.mod_eq_of_lt x.toNat_lt

/-- An ASCII byte array's chars, one a byte. -/
def chars (b : ByteArray) : List Char := b.data.toList.map Char.ofUInt8

theorem chars_ascii {b : ByteArray} (h : ∀ j (hj : j < b.size), b[j] < 128) : ∀ c ∈ chars b, c.utf8Size = 1 := by
  intro c hc
  obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hc
  obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
  exact ofUInt8_utf8Size (h j (by simpa using hj))

theorem encode_chars {b : ByteArray} (h : ∀ j (hj : j < b.size), b[j] < 128) : b = (chars b).utf8Encode := by
  rw [List.utf8Encode, Scan.flatMap_ascii _ (chars_ascii h), chars, List.map_map]
  have : ((fun c : Char => c.val.toUInt8) ∘ Char.ofUInt8) = id := funext ofUInt8_toUInt8
  rw [this, List.map_id]
  apply ByteArray.ext
  simp [List.data_toByteArray]

theorem valid_ascii {b : ByteArray} (h : ∀ j (hj : j < b.size), b[j] < 128) : b.IsValidUTF8 :=
  ⟨chars b, encode_chars h⟩

theorem toList_ascii {b : ByteArray} (h : ∀ j (hj : j < b.size), b[j] < 128) (hv : b.IsValidUTF8) :
    (String.fromUTF8 b hv).toList = chars b := by
  have : String.fromUTF8 b hv = String.ofList (chars b) := by
    apply String.toByteArray_inj.mp
    rw [String.toByteArray_ofList, ← encode_chars h]
    rfl
  rw [this, String.toList_ofList]

theorem nl_iff (x : UInt8) : (Char.ofUInt8 x == '\n') = (x == 10) := by
  have hv : (Char.ofUInt8 x).val.toNat = x.toNat := UInt8.toNat_toUInt32 x
  apply Bool.eq_iff_iff.mpr
  simp only [beq_iff_eq]
  constructor
  · intro h
    rw [h] at hv
    exact UInt8.toNat_inj.mp (hv.symm.trans rfl)
  · intro h
    subst h
    rfl

/-- `linesGo` from byte `i` cuts its lines where `cutsN` finds the bytes `10`: the pieces between them (`chunks`). -/
theorem cutsN_lines (s : String) : ∀ (fuel st i : Nat) (acc : Array String) (cacc cs : Array Nat),
    s.utf8ByteSize ≤ i + fuel → cutsN s.toByteArray fuel i cacc = some cs →
    ∃ ext : List Nat, cs = cacc ++ ext.toArray ∧ (∀ c ∈ ext, c < s.utf8ByteSize) ∧
      linesGo s fuel st i acc = acc ++ (chunks s st ext).toArray := by
  intro fuel
  induction fuel with
  | zero =>
    intro st i acc cacc cs hf h
    simp only [cutsN, Option.some.injEq] at h
    subst h
    exact ⟨[], by simp, by simp, by simp [linesGo, chunks]⟩
  | succ fuel ih =>
    intro st i acc cacc cs hf h
    unfold cutsN at h
    unfold linesGo
    by_cases hi : i < s.utf8ByteSize
    · have hi' : i < s.toByteArray.size := by simpa [String.size_toByteArray] using hi
      rw [dite_eq_left_of_eq_true (eq_true hi')] at h
      rw [dite_eq_left_of_eq_true (eq_true hi), String.getUTF8Byte_eq_getElem, nl_iff]
      split at h
      · by_cases hn : s.toByteArray[i] = 10
        · simp only [hn, beq_self_eq_true, ite_true] at h ⊢
          obtain ⟨ext, he, hc, hl⟩ := ih (i + 1) (i + 1) (acc.push (String.Pos.Raw.extract s ⟨st⟩ ⟨i⟩)) _ cs
            (by omega) h
          refine ⟨i :: ext, by simp [he], ?_, by simp [hl, chunks]⟩
          intro c hc'
          simp only [List.mem_cons] at hc'
          rcases hc' with rfl | hc'
          · exact hi
          · exact hc c hc'
        · have hb : (s.toByteArray[i] == 10) = false := by simpa using hn
          simp only [hb, Bool.false_eq_true, ite_false] at h ⊢
          exact ih st (i + 1) acc cacc cs (by omega) h
      · exact absurd h (by simp)
    · have hi' : ¬ i < s.toByteArray.size := by simpa [String.size_toByteArray] using hi
      rw [dite_eq_right_of_eq_false (eq_false hi')] at h
      rw [dite_eq_right_of_eq_false (eq_false hi)]
      simp only [Option.some.injEq] at h
      subst h
      exact ⟨[], by simp, by simp, by simp [chunks]⟩

theorem ranges_getElem? (n : Nat) : ∀ (qs : List Nat) (a k : Nat), (ranges n a qs)[k]? =
    if k ≤ qs.length then some (if k = 0 then a else qs[k - 1]! + 1, qs[k]?.getD n) else none := by
  intro qs
  induction qs with
  | nil => intro a k; cases k <;> simp [ranges]
  | cons q qs ih =>
    intro a k
    cases k with
    | zero => simp [ranges]
    | succ k =>
      simp only [ranges, List.getElem?_cons_succ, ih, List.length_cons]
      cases k <;> simp

theorem toUSize_toNat {n : Nat} (h : n < USize.size) : n.toUSize.toNat = n := by
  rw [Nat.toUSize, USize.toNat_ofNat', ← USize.size_eq_two_pow]
  exact Nat.mod_eq_of_lt h

/-- Where the line after the last `'\n'` of `cuts` starts. -/
def lastStart (cuts : Array Nat) : Nat :=
  match cuts.back? with
  | some c => c + 1
  | none => 0

/-- A file's `String` and the positions of its `'\n'` bytes, in order: its lines, `count` of them, are those
`String.splitOn "\n"` gives without a last empty one (`View.of?_spec`). -/
structure View where
  s : String
  cuts : Array Nat
  count : Nat

namespace View

/-- Where line `k` starts: after the `'\n'` before it. -/
def start (v : View) (k : Nat) : Nat := if k = 0 then 0 else v.cuts[k - 1]! + 1

/-- Where line `k` stops: at its `'\n'`, or at the end of the file. -/
def stop (v : View) (k : Nat) : Nat := v.cuts[k]?.getD v.s.utf8ByteSize

/-- Line `k`, `""` past the last. -/
def str (v : View) (k : Nat) : String :=
  if k < v.count then String.Pos.Raw.extract v.s ⟨v.start k⟩ ⟨v.stop k⟩ else ""

/-- The lines. -/
def lines (v : View) : Array String := (Array.range v.count).map v.str

/-- Line `k`'s numbers (`Net.lineNums`), read off the file's bytes (`Scan.numsU`). -/
def nums (v : View) (k : Nat) : Option (Array Nat) :=
  if h : k < v.count ∧ v.stop k ≤ v.s.utf8ByteSize ∧ v.s.utf8ByteSize < USize.size then
    match Scan.numsU v.s (v.stop k).toUSize (by rw [toUSize_toNat (by omega)]; exact h.2.1)
      (v.start k).toUSize 0 0 #[] with
    | some w => some w
    | none => Net.lineNums (v.str k)
  else Net.lineNums (v.str k)

/-- The `NET` block of lines `[a, b)` (`net_eq`). -/
def net (v : View) (a b : Nat) : V Net :=
  Net.parseOn (fun j => if a + j < b then v.str (a + j) else "") (min b v.count - a)
    fun j => if a + j < b then v.nums (a + j) else Net.lineNums ""

/-- `bytes` as a view when they are ASCII. -/
def of? (b : ByteArray) : Option View :=
  if hb : b.size < USize.size then
    match h : cutsU b b.size.toUSize (toUSize_toNat hb) 0 #[] with
    | some cs =>
      have ha := cutsN_ascii b b.size 0 #[] cs (by omega)
        (by have := cutsU_eq b _ (toUSize_toNat hb) b.size 0 #[] (by rw [toUSize_toNat hb]; simp); rw [USize.toNat_zero] at this; rw [← this]; exact h)
      some ⟨String.fromUTF8 b (valid_ascii fun j hj => ha j hj (Nat.zero_le j)),
        cs, if lastStart cs = b.size then cs.size else cs.size + 1⟩
    | none => none
  else none

theorem size_lines (v : View) : v.lines.size = v.count := by simp [lines]

theorem getElem?_lines (v : View) (k : Nat) : v.lines[k]? = if k < v.count then some (v.str k) else none := by
  unfold lines
  rw [Array.getElem?_map, Array.getElem?_range]
  split <;> rfl

theorem str_past (v : View) {k : Nat} (hk : ¬ k < v.count) : v.str k = "" := by simp [str, hk]

theorem getElem!_lines (v : View) (k : Nat) : v.lines[k]! = v.str k := by
  rw [getElem!_def, getElem?_lines]
  by_cases hk : k < v.count
  · rw [ite_eq_left hk]
  · rw [ite_eq_right hk, str_past v hk]; rfl

theorem getD_lines (v : View) (k : Nat) : v.lines.getD k "" = v.str k := by
  rw [Array.getD_eq_getD_getElem?, getElem?_lines]
  split
  · rfl
  · rename_i hk; rw [str_past v hk]; rfl

theorem getD_extract (v : View) (a b j : Nat) :
    (v.lines.extract a b).getD j "" = if a + j < b then v.str (a + j) else "" := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_extract, size_lines]
  by_cases h : j < min b v.count - a
  · rw [ite_eq_left h, ite_eq_left (by omega), getElem?_lines, ite_eq_left (by omega)]
    rfl
  · rw [ite_eq_right h]
    split
    · rw [str_past v (by omega)]; rfl
    · rfl

theorem net_eq (v : View) (hv : ∀ k, v.nums k = Net.lineNums (v.str k)) (a b : Nat) :
    v.net a b = Net.parse (v.lines.extract a b) := by
  apply Net.parseOn_eq
  · intro k; rw [getD_extract]
  · simp [size_lines]
  · intro k; split
    · exact hv _
    · rfl

/-- What `of?` gives: an ASCII file below `USize.size` bytes, `cuts` its `'\n'`s, the pieces between them its
`String.splitOn "\n"`, and `count` of them without a last empty one. -/
structure Ok (v : View) : Prop where
  ascii : ∀ c ∈ v.s.toList, c.utf8Size = 1
  small : v.s.utf8ByteSize < USize.size
  lt : ∀ c ∈ v.cuts.toList, c < v.s.utf8ByteSize
  split : (v.s.splitOn "\n").toArray = (chunks v.s 0 v.cuts.toList).toArray
  count : v.count = if lastStart v.cuts = v.s.utf8ByteSize then v.cuts.size else v.cuts.size + 1

theorem getElem!_nat (a : Array Nat) (i : Nat) : a[i]! = a[i]?.getD 0 := by
  rw [getElem!_def]
  cases a[i]? <;> rfl

theorem getElem?_chunks (v : View) (k : Nat) : (chunks v.s 0 v.cuts.toList)[k]? =
    if k ≤ v.cuts.size then some (String.Pos.Raw.extract v.s ⟨v.start k⟩ ⟨v.stop k⟩) else none := by
  rw [chunks_eq_ranges, List.getElem?_map, ranges_getElem?, Array.length_toList]
  split
  · simp [start, stop, getElem!_nat]
  · rfl

theorem length_chunks (s : String) : ∀ (qs : List Nat) (a : Nat), (chunks s a qs).length = qs.length + 1
  | [], _ => rfl
  | _ :: qs, _ => by simp [chunks, length_chunks s qs]

theorem Ok.start_le {v : View} (h : v.Ok) {k : Nat} (hk : k ≤ v.cuts.size) : v.start k ≤ v.s.utf8ByteSize := by
  unfold start
  split
  · omega
  · have hk' : k - 1 < v.cuts.size := by omega
    have := h.lt v.cuts[k - 1] (by simp)
    rw [getElem!_pos v.cuts (k - 1) hk']
    omega

theorem Ok.stop_le {v : View} (h : v.Ok) (k : Nat) : v.stop k ≤ v.s.utf8ByteSize := by
  unfold stop
  by_cases hk : k < v.cuts.size
  · rw [getElem?_pos v.cuts k hk]
    exact Nat.le_of_lt (h.lt v.cuts[k] (by simp))
  · rw [getElem?_neg v.cuts k hk]
    simp

theorem Ok.lastStart_le {v : View} (h : v.Ok) : lastStart v.cuts ≤ v.s.utf8ByteSize := by
  unfold lastStart
  split
  · rename_i c hc
    have : c ∈ v.cuts.toList := by
      rw [Array.back?_eq_getElem?] at hc
      exact Array.mem_toList_iff.mpr (Array.mem_of_getElem? hc)
    have := h.lt c this
    omega
  · omega

theorem start_size (v : View) : v.start v.cuts.size = lastStart v.cuts := by
  unfold start lastStart
  rw [Array.back?_eq_getElem?]
  by_cases h0 : v.cuts.size = 0
  · rw [ite_eq_left h0, getElem?_neg v.cuts (v.cuts.size - 1) (by omega)]
  · rw [ite_eq_right h0, getElem?_pos v.cuts (v.cuts.size - 1) (by omega), getElem!_pos v.cuts (v.cuts.size - 1) (by omega)]

theorem stop_size (v : View) : v.stop v.cuts.size = v.s.utf8ByteSize := by
  simp [stop]

/-- `extract` of an ASCII string's bytes `[a, b)`: its chars between them. -/
theorem extract_ascii {s : String} (ha : ∀ c ∈ s.toList, c.utf8Size = 1) {a b : Nat} (hab : a ≤ b)
    (hb : b ≤ s.utf8ByteSize) :
    s.toList = s.toList.take a ++ (s.toList.drop a).take (b - a) ++ s.toList.drop b ∧
      String.Pos.Raw.extract s ⟨a⟩ ⟨b⟩ = String.ofList ((s.toList.drop a).take (b - a)) := by
  have hsz := Scan.size_ascii ha
  have hs : s.toList = s.toList.take a ++ (s.toList.drop a).take (b - a) ++ s.toList.drop b := by
    have h1 : s.toList.drop b = (s.toList.drop a).drop (b - a) := by
      rw [List.drop_drop]; congr 1; omega
    rw [h1, List.append_assoc, List.take_append_drop, List.take_append_drop]
  refine ⟨hs, ?_⟩
  have := extract_eq _ _ _ s hs
  have hlen : ∀ X : List Char, (∀ c ∈ X, c ∈ s.toList) → len X = X.length := fun X hX =>
    len_ascii X (fun c hc => ha c (hX c hc))
  rw [hlen _ (fun c hc => List.mem_of_mem_take hc),
    hlen _ (fun c hc => List.mem_of_mem_drop (List.mem_of_mem_take hc))] at this
  simp only [List.length_take, List.length_drop] at this
  rw [show min a s.toList.length = a by omega, show min (b - a) (s.toList.length - a) = b - a by omega,
    show a + (b - a) = b by omega] at this
  exact this

theorem extract_empty (s : String) {a b : Nat} (h : b ≤ a) : String.Pos.Raw.extract s ⟨a⟩ ⟨b⟩ = "" := by
  simp [String.Pos.Raw.extract, h]

theorem lineNums_empty : Net.lineNums "" = some #[] := by
  simp [Net.lineNums, Net.tokens_eq]

theorem Ok.lines {v : View} (h : v.Ok) :
    (if (v.s.splitOn "\n").toArray.back? == some "" then (v.s.splitOn "\n").toArray.pop
      else (v.s.splitOn "\n").toArray) = v.lines := by
  rw [h.split]
  have hsz : (chunks v.s 0 v.cuts.toList).toArray.size = v.cuts.size + 1 := by
    simp [length_chunks]
  have hlast : (chunks v.s 0 v.cuts.toList).toArray.back? =
      some (String.Pos.Raw.extract v.s ⟨lastStart v.cuts⟩ ⟨v.s.utf8ByteSize⟩) := by
    rw [Array.back?_eq_getElem?, hsz, Nat.add_sub_cancel, List.getElem?_toArray, getElem?_chunks,
      ite_eq_left (Nat.le_refl _), start_size, stop_size]
  rw [hlast]
  apply Array.ext_getElem?
  intro k
  rw [getElem?_lines, h.count]
  by_cases hl : lastStart v.cuts = v.s.utf8ByteSize
  · have he : (some (String.Pos.Raw.extract v.s ⟨lastStart v.cuts⟩ ⟨v.s.utf8ByteSize⟩) == some "") = true := by
      rw [hl, extract_empty v.s (Nat.le_refl _)]; rfl
    rw [ite_eq_left he, ite_eq_left hl, Array.getElem?_pop, hsz, Nat.add_sub_cancel]
    by_cases hk : k < v.cuts.size
    · rw [ite_eq_left hk, ite_eq_left hk, List.getElem?_toArray, getElem?_chunks, ite_eq_left (Nat.le_of_lt hk)]
      simp [str, h.count, hl, hk]
    · rw [ite_eq_right hk, ite_eq_right hk]
  · have hlt : lastStart v.cuts < v.s.utf8ByteSize := Nat.lt_of_le_of_ne h.lastStart_le hl
    have hne : String.Pos.Raw.extract v.s ⟨lastStart v.cuts⟩ ⟨v.s.utf8ByteSize⟩ ≠ "" := by
      rw [(extract_ascii h.ascii (Nat.le_of_lt hlt) (Nat.le_refl _)).2]
      intro he
      have := congrArg List.length (String.ofList_eq_empty_iff.mp he)
      have hsz := Scan.size_ascii h.ascii
      simp at this
      omega
    have he : (some (String.Pos.Raw.extract v.s ⟨lastStart v.cuts⟩ ⟨v.s.utf8ByteSize⟩) == some "") = false := by
      simpa using hne
    rw [ite_eq_right (by simp [he]), ite_eq_right hl, List.getElem?_toArray, getElem?_chunks]
    by_cases hk : k < v.cuts.size + 1
    · rw [ite_eq_left (by omega), ite_eq_left hk]
      simp [str, h.count, hl, hk]
    · rw [ite_eq_right (by omega), ite_eq_right hk]

theorem Ok.nums {v : View} (h : v.Ok) (k : Nat) : v.nums k = Net.lineNums (v.str k) := by
  unfold View.nums
  split
  · rename_i hk
    obtain ⟨hkc, hstop, hsmall⟩ := hk
    have hkcs : k ≤ v.cuts.size := by have := h.count; split at this <;> omega
    have hstart := h.start_le hkcs
    have hstr : v.str k = String.Pos.Raw.extract v.s ⟨v.start k⟩ ⟨v.stop k⟩ := by simp [str, hkc]
    split
    · rename_i w hw
      rw [hstr]
      have hsU : (v.start k).toUSize.toNat = v.start k := toUSize_toNat (by omega)
      have heU : (v.stop k).toUSize.toNat = v.stop k := toUSize_toNat (by omega)
      have hsz := Scan.size_ascii h.ascii
      by_cases hlt : v.start k < v.stop k
      · obtain ⟨hs, he⟩ := extract_ascii h.ascii (Nat.le_of_lt hlt) hstop
        rw [he]
        have hu := Scan.numsU_some v.s _ _ (v.s.utf8ByteSize + 1) (v.start k).toUSize 0 0 #[] w
          (by rw [heU, hsU]; omega) (by simp) (by simp) hw
        simp only [bne_self_eq_false, UInt64.toNat_zero] at hu
        rw [hsU] at hu
        have hn := Scan.nums_eq h.ascii (v.stop k).toUSize.toNat (by rw [heU]; exact hstop) ((v.s.toList.drop (v.start k)).take (v.stop k - v.start k))
          (v.s.toList.take (v.start k)) (v.s.toList.drop (v.stop k)) (v.s.utf8ByteSize + 1) false 0 #[] hs
          (by rw [heU]; simp only [List.length_take, List.length_drop]; omega)
          (by simp only [List.length_take, List.length_drop]; omega)
        have hP : (v.s.toList.take (v.start k)).length = v.start k := by simp; omega
        rw [hP] at hn
        rw [hn] at hu
        unfold Net.lineNums
        rw [Net.tokens_eq, String.toList_ofList]
        exact (Scan.numsL_tokens _ #[] "" false 0 #[] w (by simp) ⟨rfl, rfl⟩ hu).symm
      · have hw' : w = #[] := by
          unfold Scan.numsU at hw
          rw [dite_eq_right_of_eq_false (eq_false (by
            intro hh; have := USize.lt_iff_toNat_lt.mp hh; rw [heU, hsU] at this; omega))] at hw
          simpa using hw.symm
        subst hw'
        rw [extract_empty v.s (by omega), lineNums_empty]
    · rfl
  · rfl

theorem of?_spec {b : ByteArray} {v : View} (h : of? b = some v) : String.fromUTF8? b = some v.s ∧ v.Ok := by
  unfold of? at h
  split at h
  · rename_i hb
    split at h
    · rename_i cs hc
      simp only [Option.some.injEq] at h
      subst h
      have hcN : cutsN b b.size 0 #[] = some cs := by
        have := cutsU_eq b _ (toUSize_toNat hb) b.size 0 #[] (by rw [toUSize_toNat hb]; simp)
        rw [USize.toNat_zero] at this
        rw [← this]
        exact hc
      have ha : ∀ j (hj : j < b.size), b[j] < 128 := fun j hj =>
        cutsN_ascii b b.size 0 #[] cs (by omega) hcN j hj (Nat.zero_le j)
      have hv := valid_ascii ha
      have hchars := toList_ascii ha hv
      have hasc : ∀ c ∈ (String.fromUTF8 b hv).toList, c.utf8Size = 1 := by
        rw [hchars]; exact chars_ascii ha
      refine ⟨?_, ?_⟩
      · unfold String.fromUTF8?
        rw [dite_eq_left_of_eq_true (eq_true hv)]
      · obtain ⟨ext, he, hlt, hl⟩ := cutsN_lines (String.fromUTF8 b hv) b.size 0 0 #[] #[] cs
          (by rw [Nat.zero_add]; exact Nat.le_refl _) hcN
        rw [Array.empty_append] at he
        subst he
        have hsplit : ((String.fromUTF8 b hv).splitOn "\n").toArray = linesGo (String.fromUTF8 b hv) b.size 0 0 #[] := by
          have := linesGo_eq hasc (String.fromUTF8 b hv).toList [] [] b.size #[] (by simp) (by simp)
            (by rw [← Scan.size_ascii hasc]; show b.size < b.size + 1; omega)
          simp only [List.length_nil, Nat.add_zero, List.nil_append, Array.empty_append] at this
          rw [this, splitOn_nl]
        exact { ascii := hasc, small := hb, lt := by simpa using hlt,
                split := by rw [hsplit, hl, Array.empty_append, List.toList_toArray], count := rfl }
    · simp at h
  · simp at h

end View

end Flock.ByteLines
