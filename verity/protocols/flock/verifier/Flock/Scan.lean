/-
`Flock.Scan`: a row's numbers read off its bytes (`nums`, and `numsU` in machine words), as `Flock.tokens` and
`String.toNat?` read its chars. A line of ASCII decimal digits and whitespace gives the numbers
`(tokens s).mapM String.toNat?` gives (`numsU_some`, `nums_eq`, `numsL_tokens`); `Net.lineNumsFast` falls back to that
expression on any other line, and on a number past 19 digits, so the two agree on every line.
-/
module

import all Init.Data.String.Search
import all Init.Data.String.Slice

@[expose] public section

namespace Flock.Scan

/-- The value of the decimal digits `ds`. -/
def val (ds : List Char) : Nat := ds.foldl (fun n c => n * 10 + (c.toNat - 48)) 0

theorem val_append (ds : List Char) (c : Char) : val (ds ++ [c]) = val ds * 10 + (c.toNat - 48) := by
  simp [val, List.foldl_append]

theorem foldl_congr {α β : Type} (f g : α → β → α) :
    ∀ (l : List β) (a : α), (∀ x ∈ l, ∀ a, f a x = g a x) → l.foldl f a = l.foldl g a
  | [], _, _ => rfl
  | x :: xs, a, h => by
    simp only [List.foldl_cons, h x (by simp) a]
    exact foldl_congr f g xs _ (fun y hy => h y (by simp [hy]))

theorem forIn_digits {σ : Type} (l : List Char) (f : Char → σ → Id (ForInStep σ)) (z : σ)
    (hf : ∀ c ∈ l, ∀ x, f c x = pure (ForInStep.yield z)) (x : σ) :
    forIn l x f = pure (if l = [] then x else z) := by
  induction l generalizing x with
  | nil => rfl
  | cons c cs ih =>
    rw [List.forIn_cons, hf c (by simp) x]
    simp only [pure, bind]
    rw [ih (fun c hc => hf c (by simp [hc]))]
    cases cs <;> simp <;> rfl

/-- `String.toNat?` of a nonempty run of decimal digits is their value. -/
theorem toNat?_digits (ds : List Char) (hne : ds ≠ []) (hd : ∀ c ∈ ds, c.isDigit = true) :
    (String.ofList ds).toNat? = some (val ds) := by
  unfold String.toNat? String.Slice.toNat? String.Slice.isNat
  simp only [Id.run, String.Slice.forIn_eq_forIn_toList, String.Slice.foldl_eq_foldl_toList, String.copy_toSlice,
    String.toList_ofList]
  rw [forIn_digits ds _ (none, true) (fun c hc x => by
    have h1 := hd c hc
    have h2 : c ≠ '_' := by rintro rfl; simp at h1
    simp [h1, h2])]
  simp only [hne, ite_false, pure, bind, ite_true]
  unfold val
  congr 1
  apply foldl_congr
  intro c hc n
  have h2 : c ≠ '_' := by rintro rfl; have := hd _ hc; simp at this
  simp [h2]

/-- `Flock.tokens`' loop body: the tokens so far and the current one. -/
def tokStep (st : Array String × String) (c : Char) : Array String × String :=
  if c.isWhitespace then (if st.2 != "" then (st.1.push st.2, "") else st) else (st.1, st.2.push c)

theorem forIn_tok (l : List Char) (st : Array String × String) :
    forIn (m := Id) l st (fun c (__s : Array String × String) =>
      if c.isWhitespace = true then
        if (__s.snd != "") = true then pure (ForInStep.yield (__s.fst.push __s.snd, ""))
        else pure (ForInStep.yield (__s.fst, __s.snd))
      else pure (ForInStep.yield (__s.fst, __s.snd.push c))) = pure (l.foldl tokStep st) := by
  induction l generalizing st with
  | nil => rfl
  | cons c cs ih =>
    rw [List.forIn_cons, List.foldl_cons, ← ih]
    unfold tokStep
    by_cases h1 : c.isWhitespace = true
    · by_cases h2 : (st.snd != "") = true <;> simp [h1, h2]
    · simp [h1]

/-- The numbers of `cs` when it is decimal digits and whitespace (`Char.isWhitespace`) only, the current one `acc` while
`inTok`; `none` at any other char. -/
def numsL : List Char → Bool → Nat → Array Nat → Option (Array Nat)
  | [], inTok, acc, out => some (if inTok then out.push acc else out)
  | c :: cs, inTok, acc, out =>
    if c.isDigit then numsL cs true (acc * 10 + (c.toNat - 48)) out
    else if c.isWhitespace then numsL cs false 0 (if inTok then out.push acc else out)
    else none

/-- `numsL`'s state against `tokStep`'s: inside a token, its chars so far are digits worth `acc`; between tokens,
none. -/
def Inv (cur : String) (inTok : Bool) (acc : Nat) : Prop :=
  if inTok then ∃ ds, cur = String.ofList ds ∧ ds ≠ [] ∧ (∀ c ∈ ds, c.isDigit = true) ∧ acc = val ds
  else cur = "" ∧ acc = 0

theorem digit_not_ws {c : Char} (h : c.isDigit = true) : c.isWhitespace = false := by
  have h1 : c ≠ ' ' := by rintro rfl; exact absurd h (by decide)
  have h2 : c ≠ '\t' := by rintro rfl; exact absurd h (by decide)
  have h3 : c ≠ '\r' := by rintro rfl; exact absurd h (by decide)
  have h4 : c ≠ '\n' := by rintro rfl; exact absurd h (by decide)
  simp [Char.isWhitespace, h1, h2, h3, h4]

theorem mapM_push_opt {α β : Type} (f : α → Option β) (xs : Array α) (x : α) (ys : Array β) (y : β)
    (h1 : xs.mapM f = some ys) (h2 : f x = some y) : (xs.push x).mapM f = some (ys.push y) := by
  have : (xs.push x).mapM f = (xs ++ #[x]).mapM f := rfl
  rw [this, Array.mapM_append, h1]
  simp [Array.mapM_eq_mapM_toList, h2]

/-- **What `numsL` reads, `tokens` and `String.toNat?` read**: when `numsL` reads numbers off the rest of the line, the
tokens `tokStep` reads off it, finished as `Flock.tokens` finishes them, are those numbers. -/
theorem numsL_tokens : ∀ (cs : List Char) (out : Array String) (cur : String) (inTok : Bool) (acc : Nat)
    (outN v : Array Nat), out.mapM String.toNat? = some outN → Inv cur inTok acc →
    numsL cs inTok acc outN = some v →
    (let st := cs.foldl tokStep (out, cur); if st.2 != "" then st.1.push st.2 else st.1).mapM String.toNat? = some v
  | [], out, cur, inTok, acc, outN, v, hout, hinv, h => by
    simp only [numsL, Option.some.injEq] at h
    subst h
    simp only [List.foldl_nil]
    cases inTok with
    | false =>
      obtain ⟨rfl, rfl⟩ := hinv
      simpa using hout
    | true =>
      obtain ⟨ds, rfl, hne, hd, rfl⟩ := hinv
      have : (String.ofList ds != "") = true := by simpa [String.ofList_eq_empty_iff] using hne
      simp only [this, ite_true]
      exact mapM_push_opt _ _ _ _ _ hout (toNat?_digits ds hne hd)
  | c :: cs, out, cur, inTok, acc, outN, v, hout, hinv, h => by
    simp only [List.foldl_cons]
    by_cases hc : c.isDigit = true
    · simp only [numsL, hc, ite_true] at h
      have hw := digit_not_ws hc
      simp only [tokStep, hw, Bool.false_eq_true, ite_false]
      refine numsL_tokens cs out (cur.push c) true _ outN v hout ?_ h
      cases inTok with
      | false =>
        obtain ⟨rfl, rfl⟩ := hinv
        exact ⟨[c], by simp, by simp, by simp [hc], by simp [val]⟩
      | true =>
        obtain ⟨ds, rfl, hne, hd, rfl⟩ := hinv
        refine ⟨ds ++ [c], ?_, by simp, ?_, (val_append ds c).symm⟩
        · simp [String.ofList_append]
        · intro x hx
          simp only [List.mem_append, List.mem_singleton] at hx
          rcases hx with hx | rfl
          · exact hd x hx
          · exact hc
    · by_cases hw : c.isWhitespace = true
      · simp only [numsL, hc, hw, ite_true, Bool.false_eq_true, ite_false] at h
        simp only [tokStep, hw, ite_true]
        cases inTok with
        | false =>
          obtain ⟨rfl, rfl⟩ := hinv
          exact numsL_tokens cs out "" false 0 outN v hout ⟨rfl, rfl⟩ (by simpa using h)
        | true =>
          obtain ⟨ds, rfl, hne, hd, rfl⟩ := hinv
          have : (String.ofList ds != "") = true := by simpa [String.ofList_eq_empty_iff] using hne
          simp only [this, ite_true]
          exact numsL_tokens cs _ "" false 0 (outN.push (val ds)) v
            (mapM_push_opt _ _ _ _ _ hout (toNat?_digits ds hne hd)) ⟨rfl, rfl⟩ (by simpa using h)
      · simp [numsL, hc, hw] at h

theorem ofUInt8_val {c : Char} (h : c.utf8Size = 1) : Char.ofUInt8 c.val.toUInt8 = c := by
  have h' : c.val.toNat ≤ 127 := by simpa [UInt32.le_iff_toNat_le] using Char.utf8Size_eq_one_iff.mp h
  apply Char.ext
  apply UInt32.toNat_inj.mp
  show (c.val.toUInt8.toUInt32).toNat = c.val.toNat
  rw [UInt8.toNat_toUInt32, UInt32.toNat_toUInt8]
  omega

theorem sum_ascii (l : List Char) (h : ∀ c ∈ l, c.utf8Size = 1) : (l.map Char.utf8Size).sum = l.length := by
  induction l with
  | nil => rfl
  | cons c cs ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    rw [h c (by simp), ih (fun x hx => h x (by simp [hx]))]
    omega

theorem ascii_all (l : List Char) (h : l.length = (l.map Char.utf8Size).sum) : ∀ c ∈ l, c.utf8Size = 1 := by
  induction l with
  | nil => simp
  | cons c cs ih =>
    have hc := Char.utf8Size_pos c
    have hs : cs.length ≤ (cs.map Char.utf8Size).sum := by
      clear ih h
      induction cs with
      | nil => simp
      | cons d ds ih => have := Char.utf8Size_pos d; simp; omega
    simp only [List.length_cons, List.map_cons, List.sum_cons] at h
    intro x hx
    simp only [List.mem_cons] at hx
    rcases hx with rfl | hx
    · omega
    · exact ih (by omega) x hx

theorem flatMap_ascii (l : List Char) (h : ∀ c ∈ l, c.utf8Size = 1) :
    l.flatMap String.utf8EncodeChar = l.map (·.val.toUInt8) := by
  induction l with
  | nil => rfl
  | cons c cs ih =>
    simp only [List.flatMap_cons, List.map_cons]
    rw [String.utf8EncodeChar_eq_singleton (h c (by simp)), ih (fun x hx => h x (by simp [hx]))]
    rfl

theorem size_eq_sum (s : String) : s.utf8ByteSize = (s.toList.map Char.utf8Size).sum := by
  rw [← String.size_toByteArray, ← String.utf8Encode_toList]
  simp [List.utf8Encode, List.length_flatMap]

/-- Every char of `s` is one byte. `String.length` and `String.utf8ByteSize` are read in constant time. -/
def ascii (s : String) : Bool := s.length == s.utf8ByteSize

theorem ascii_chars {s : String} (h : ascii s = true) : ∀ c ∈ s.toList, c.utf8Size = 1 := by
  apply ascii_all
  rw [String.length_toList, ← size_eq_sum]
  simpa [ascii] using h

theorem size_ascii {s : String} (h : ∀ c ∈ s.toList, c.utf8Size = 1) : s.utf8ByteSize = s.toList.length := by
  rw [size_eq_sum, sum_ascii _ h]

/-- An ASCII string's byte at a char's index is that char. -/
theorem byte_ascii {s : String} (h : ∀ c ∈ s.toList, c.utf8Size = 1) (P R : List Char) (c : Char)
    (hs : s.toList = P ++ c :: R) (hi : P.length < s.toByteArray.size) : s.toByteArray[P.length] = c.val.toUInt8 := by
  have hb : s.toByteArray = (s.toList.map (·.val.toUInt8)).toByteArray := by
    rw [← String.utf8Encode_toList, List.utf8Encode, flatMap_ascii _ h]
  simp only [hb, List.getElem_toByteArray, List.getElem_map]
  simp [hs]

/-- The numbers of `s`'s bytes from `i` to `e`, read as `numsL` reads chars: decimal digits and whitespace only, else
`none`. -/
def nums (s : String) (e : Nat) (he : e ≤ s.utf8ByteSize) : Nat → Nat → Bool → Nat → Array Nat → Option (Array Nat)
  | 0, _, _, _, _ => none
  | fuel + 1, i, inTok, acc, out =>
    if h : i < e then
      let c := Char.ofUInt8 (s.getUTF8Byte ⟨i⟩ (String.Pos.Raw.lt_iff.mpr (by simp only [String.byteIdx_rawEndPos]; omega)))
      if c.isDigit then nums s e he fuel (i + 1) true (acc * 10 + (c.toNat - 48)) out
      else if c.isWhitespace then nums s e he fuel (i + 1) false 0 (if inTok then out.push acc else out)
      else none
    else some (if inTok then out.push acc else out)

/-- On an ASCII string, `nums` from a char's index to `e` reads what `numsL` reads off the chars between them. -/
theorem nums_eq {s : String} (ha : ∀ c ∈ s.toList, c.utf8Size = 1) (e : Nat) (he : e ≤ s.utf8ByteSize) :
    ∀ (R P T : List Char) (fuel : Nat) (inTok : Bool) (acc : Nat) (out : Array Nat), s.toList = P ++ R ++ T →
    P.length + R.length = e → R.length < fuel →
    nums s e he fuel P.length inTok acc out = numsL R inTok acc out := by
  have hsz := size_ascii ha
  intro R
  induction R with
  | nil =>
    intro P T fuel inTok acc out _ he' hf
    obtain ⟨fuel, rfl⟩ : ∃ k, fuel = k + 1 := ⟨fuel - 1, by omega⟩
    have : ¬ P.length < e := by simp at he'; omega
    simp [nums, this, numsL]
  | cons c R ih =>
    intro P T fuel inTok acc out hs he' hf
    obtain ⟨fuel, rfl⟩ : ∃ k, fuel = k + 1 := ⟨fuel - 1, by omega⟩
    have hle : P.length < e := by simp at he'; omega
    have hlt : P.length < s.utf8ByteSize := by omega
    have hs' : s.toList = P ++ c :: (R ++ T) := by simp [hs]
    have hb : Char.ofUInt8 (s.getUTF8Byte ⟨P.length⟩ (String.Pos.Raw.lt_iff.mpr (by simpa using hlt))) = c := by
      rw [String.getUTF8Byte_eq_getElem]
      rw [byte_ascii ha P (R ++ T) c hs' (by simpa [String.size_toByteArray] using hlt)]
      exact ofUInt8_val (ha c (by simp [hs']))
    have ih' := fun inTok acc out => ih (P ++ [c]) T fuel inTok acc out (by simp [hs]) (by simp at he' ⊢; omega)
      (by simp at hf; omega)
    simp only [List.length_append, List.length_singleton] at ih'
    simp only [nums, hle, dite_true, hb, numsL]
    split
    · exact ih' _ _ _
    · split
      · exact ih' _ _ _
      · rfl

theorem toNat_succ {i n : USize} (h : i < n) : (i + 1).toNat = i.toNat + 1 := by
  have h1 := USize.lt_iff_toNat_lt.mp h
  have h2 := USize.toNat_lt_size n
  rw [USize.toNat_add, USize.toNat_one]
  exact Nat.mod_eq_of_lt (by have : USize.size = 2 ^ System.Platform.numBits := rfl; omega)

/-- `nums` in machine words: the byte index a `USize` below `n`, the range's end, and a number's value a `UInt64` with
its digit count `k`, `none` past 19 digits. -/
def numsU (s : String) (n : USize) (hn : n.toNat ≤ s.utf8ByteSize) (i k : USize) (acc : UInt64) (out : Array Nat) :
    Option (Array Nat) :=
  if h : i < n then
    let b := s.getUTF8Byte ⟨i.toNat⟩ (String.Pos.Raw.lt_iff.mpr (by
      have := USize.lt_iff_toNat_lt.mp h; simp only [String.byteIdx_rawEndPos]; omega))
    let c := Char.ofUInt8 b
    if c.isDigit then
      if k < 19 then numsU s n hn (i + 1) (k + 1) (acc * 10 + (b.toUInt64 &&& 15)) out else none
    else if c.isWhitespace then numsU s n hn (i + 1) 0 0 (if k != 0 then out.push acc.toNat else out)
    else none
  else some (if k != 0 then out.push acc.toNat else out)
termination_by n.toNat - i.toNat
decreasing_by
  all_goals
    have := USize.lt_iff_toNat_lt.mp h
    rw [toNat_succ h]
    omega

theorem digit_val (b : UInt8) (h : (Char.ofUInt8 b).isDigit = true) :
    (b.toUInt64 &&& 15).toNat = (Char.ofUInt8 b).toNat - 48 ∧ 48 ≤ b.toNat ∧ b.toNat ≤ 57 := by
  have hv : (Char.ofUInt8 b).val.toNat = b.toNat := UInt8.toNat_toUInt32 b
  simp only [Char.isDigit, Bool.and_eq_true, decide_eq_true_eq] at h
  have h1 : 48 ≤ b.toNat := by
    have := UInt32.le_iff_toNat_le.mp h.1
    rw [hv] at this
    exact this
  have h2 : b.toNat ≤ 57 := by
    have := UInt32.le_iff_toNat_le.mp h.2
    rw [hv] at this
    exact this
  refine ⟨?_, h1, h2⟩
  rw [UInt64.toNat_and, UInt8.toNat_toUInt64, show (Char.ofUInt8 b).toNat = b.toNat from hv]
  have : (15 : UInt64).toNat = 2 ^ 4 - 1 := rfl
  rw [this, Nat.and_two_pow_sub_one_eq_mod]
  omega

theorem usize_lit {m : Nat} (h : m < 2 ^ 32) : (OfNat.ofNat m : USize).toNat = m := by
  rw [USize.toNat_ofNat]
  exact Nat.mod_eq_of_lt (by have := USize.le_size; have : USize.size = 2 ^ System.Platform.numBits := rfl; omega)

theorem numsU_some (s : String) (n : USize) (hn : n.toNat ≤ s.utf8ByteSize) : ∀ (fuel : Nat) (i k : USize)
    (acc : UInt64) (out : Array Nat) (v : Array Nat), n.toNat - i.toNat < fuel → k.toNat ≤ 19 →
    acc.toNat < 10 ^ k.toNat → numsU s n hn i k acc out = some v →
    nums s n.toNat hn fuel i.toNat (k != 0) acc.toNat out = some v := by
  intro fuel
  induction fuel with
  | zero => intro i k acc out v hf; omega
  | succ fuel ih =>
    intro i k acc out v hf hk hacc hv
    rw [numsU] at hv
    unfold nums
    split at hv
    · rename_i h
      have hi := USize.lt_iff_toNat_lt.mp h
      have hi' : i.toNat < s.utf8ByteSize := by omega
      rw [dite_eq_left_of_eq_true (eq_true hi)]
      simp only at hv ⊢
      have hp : (⟨i.toNat⟩ : String.Pos.Raw) < s.rawEndPos := String.Pos.Raw.lt_iff.mpr (by simpa using hi')
      generalize s.getUTF8Byte ⟨i.toNat⟩ hp = b at hv ⊢
      split at hv
      · rename_i hd
        rw [ite_eq_left_of_eq_true _ _ (eq_true hd)]
        split at hv
        · rename_i hk19
          have hk' := USize.lt_iff_toNat_lt.mp hk19
          have h19 : (19 : USize).toNat = 19 := usize_lit (by decide)
          have hsucc := toNat_succ h
          have hks : (k + 1).toNat = k.toNat + 1 := by
            have := USize.le_size
            have : USize.size = 2 ^ System.Platform.numBits := rfl
            rw [USize.toNat_add, USize.toNat_one]
            exact Nat.mod_eq_of_lt (by omega)
          obtain ⟨hdv, _, _⟩ := digit_val _ hd
          have hc : (Char.ofUInt8 b).toNat = b.toNat := UInt8.toNat_toUInt32 b
          have hlt : acc.toNat * 10 + ((Char.ofUInt8 b).toNat - 48) < 10 ^ (k.toNat + 1) := by
            rw [Nat.pow_succ]; omega
          have hpow : 10 ^ (k.toNat + 1) ≤ 10 ^ 19 := Nat.pow_le_pow_right (by omega) (by omega)
          have hacc' : (acc * 10 + (b.toUInt64 &&& 15)).toNat = acc.toNat * 10 + ((Char.ofUInt8 b).toNat - 48) := by
            rw [UInt64.toNat_add, UInt64.toNat_mul, hdv]
            have : (10 : UInt64).toNat = 10 := rfl
            rw [this]
            have h1 : acc.toNat * 10 < 2 ^ 64 := by omega
            rw [Nat.mod_eq_of_lt h1]
            exact Nat.mod_eq_of_lt (by omega)
          have := ih (i + 1) (k + 1) _ out v (by omega) (by omega) (by rw [hks, hacc']; exact hlt) hv
          rw [hsucc, hacc'] at this
          have hne : (k + 1 != 0) = true := by
            simp only [bne_iff_ne, ne_eq]
            intro h0; rw [h0] at hks; simp at hks
          rw [hne] at this
          exact this
        · exact absurd hv (by simp)
      · rename_i hd
        rw [ite_eq_right_of_eq_false _ _ (eq_false hd)]
        split at hv
        · rename_i hw
          rw [ite_eq_left_of_eq_true _ _ (eq_true hw)]
          have := ih (i + 1) 0 0 _ v (by rw [toNat_succ h]; omega) (by simp) (by simp) hv
          rw [toNat_succ h] at this
          simpa using this
        · exact absurd hv (by simp)
    · rename_i h
      have hi : ¬ i.toNat < n.toNat := mt USize.lt_iff_toNat_lt.mpr h
      rw [dite_eq_right_of_eq_false (eq_false hi)]
      exact hv

end Flock.Scan
