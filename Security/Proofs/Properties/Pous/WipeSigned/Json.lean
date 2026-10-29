import MemoryChallenger.StreamedWipe

/-!
# What `pyJsonStr` writes is read back

`pyJsonStr` writes a string as `json.dumps` does: each character as it is, or escaped (`enc`), between quotes. Every
escape starts with a backslash and a plain character is neither a quote nor a backslash, so each character is read back
from what it writes (`decC_enc`), and the string, with what follows its closing quote (`pyJsonStr_split`): the written
strings are a prefix code, whatever characters they hold. A number is its decimal digits (`natStr_split`, `intStr_inj`),
and the wipe's tiers, a list of `[name, bytes]` pairs, are a prefix code too (`tiers_split`).

`pyJsonStr`'s `hex4` and the wipe's `tiersJson` are private to their modules, so the statements here are about copies,
word for word (`jsonStr`, `tiersJson`), which are those definitions by `rfl` (`pyJsonStr_eq`).
-/

namespace MemoryChallenger.Properties.SecurityProofs.WipeJson

/-- `Records.lean`'s `hex4`, word for word. -/
def hex4 (n : Nat) : String :=
  let d := fun (k : Nat) => let x := (n >>> (4 * k)) % 16; if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)
  String.ofList [d 3, d 2, d 1, d 0]

/-- `pyJsonStr`, word for word, with the `hex4` above. -/
def jsonStr (s : String) : String := Id.run do
  let mut out := "\""
  for c in s.toList do
    let n := c.toNat
    out := match c with
      | '"' => out ++ "\\\""
      | '\\' => out ++ "\\\\"
      | '\n' => out ++ "\\n"
      | '\r' => out ++ "\\r"
      | '\t' => out ++ "\\t"
      | c => if n == 8 then out ++ "\\b" else if n == 12 then out ++ "\\f"
             else if 0x20 ≤ n && n ≤ 0x7e then out.push c
             else if n < 0x10000 then out ++ "\\u" ++ hex4 n
             else let m := n - 0x10000; out ++ "\\u" ++ hex4 (0xd800 + (m >>> 10)) ++ "\\u" ++ hex4 (0xdc00 + (m &&& 0x3ff))
  return out ++ "\""

theorem pyJsonStr_eq (s : String) : pyJsonStr s = jsonStr s := rfl

/-- `StreamedWipe.lean`'s `tiersJson`, word for word. -/
def tiersJson (tiers : Array (String × Nat)) : String :=
  "[" ++ ",".intercalate (tiers.toList.map fun (t, b) => "[" ++ pyJsonStr t ++ "," ++ toString b ++ "]") ++ "]"

/-- A hex digit as `hex4` writes it. -/
def hexDigit (x : Nat) : Char := if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)

/-- `hex4 n`'s characters. -/
def hex4L (n : Nat) : List Char :=
  [hexDigit (n / 4096 % 16), hexDigit (n / 256 % 16), hexDigit (n / 16 % 16), hexDigit (n % 16)]

theorem hex4_toList (n : Nat) : (hex4 n).toList = hex4L n := by
  simp [hex4, hex4L, hexDigit, Nat.shiftRight_eq_div_pow]

/-- A hex digit's value. -/
def hexVal (c : Char) : Nat := if c.toNat < 58 then c.toNat - 48 else c.toNat - 87

theorem hexVal_digit : ∀ x < 16, hexVal (hexDigit x) = x := by decide

/-- Four hex digits' value. -/
def hex4V (a b c d : Char) : Nat := hexVal a * 4096 + hexVal b * 256 + hexVal c * 16 + hexVal d

theorem hex4L_val {n : Nat} (h : n < 65536) :
    hex4V (hexDigit (n / 4096 % 16)) (hexDigit (n / 256 % 16)) (hexDigit (n / 16 % 16)) (hexDigit (n % 16)) = n := by
  unfold hex4V
  rw [hexVal_digit _ (Nat.mod_lt _ (by decide)), hexVal_digit _ (Nat.mod_lt _ (by decide)),
    hexVal_digit _ (Nat.mod_lt _ (by decide)), hexVal_digit _ (Nat.mod_lt _ (by decide))]
  omega

/-- What `pyJsonStr` writes for one character. -/
def enc (c : Char) : List Char :=
  if c = '"' then ['\\', '"'] else if c = '\\' then ['\\', '\\'] else if c = '\n' then ['\\', 'n']
  else if c = '\r' then ['\\', 'r'] else if c = '\t' then ['\\', 't'] else if c.toNat = 8 then ['\\', 'b']
  else if c.toNat = 12 then ['\\', 'f'] else if 0x20 ≤ c.toNat ∧ c.toNat ≤ 0x7e then [c]
  else if c.toNat < 0x10000 then '\\' :: 'u' :: hex4L c.toNat
  else '\\' :: 'u' :: (hex4L (0xd800 + (c.toNat - 0x10000) / 1024) ++
    '\\' :: 'u' :: hex4L (0xdc00 + (c.toNat - 0x10000) % 1024))

theorem foldl_enc {f : String → Char → String} (hf : ∀ b c, (f b c).toList = b.toList ++ enc c) :
    ∀ (l : List Char) (out : String), (l.foldl f out).toList = out.toList ++ l.flatMap enc
  | [], out => by simp
  | c :: l, out => by rw [List.foldl_cons, foldl_enc hf l, hf, List.flatMap_cons, List.append_assoc]

theorem pyJsonStr_toList (s : String) : (pyJsonStr s).toList = '"' :: (s.toList.flatMap enc ++ ['"']) := by
  rw [pyJsonStr_eq]
  unfold jsonStr
  simp only [Id.run]
  rw [List.forIn_pure_yield_eq_foldl]
  show (String.toList (List.foldl _ "\"" s.toList ++ "\"")) = _
  rw [String.toList_append, foldl_enc]
  · simp
  intro b c
  split
  · rw [String.toList_append]; congr 1
  · rw [String.toList_append]; congr 1
  · rw [String.toList_append]; congr 1
  · rw [String.toList_append]; congr 1
  · rw [String.toList_append]; congr 1
  · rename_i h1 h2 h3 h4 h5
    have hv : c.toNat < 0xd800 ∨ (0xdfff < c.toNat ∧ c.toNat < 0x110000) := c.valid
    simp only [enc, ite_eq_right h1, ite_eq_right h2, ite_eq_right h3, ite_eq_right h4, ite_eq_right h5]
    by_cases k8 : c.toNat = 8
    · simp [k8, String.toList_append]
    by_cases k12 : c.toNat = 12
    · simp [k12, String.toList_append]
    by_cases kp : 0x20 ≤ c.toNat ∧ c.toNat ≤ 0x7e
    · simp [k8, k12, kp, String.toList_push]
    by_cases ku : c.toNat < 0x10000
    · simp [k8, k12, kp, ku, String.toList_append, hex4_toList]
    · have hand : ∀ m : Nat, m &&& 0x3ff = m % 1024 := fun m => Nat.and_two_pow_sub_one_eq_mod m 10
      simp [k8, k12, kp, ku, String.toList_append, hex4_toList, Nat.shiftRight_eq_div_pow, hand]

/-! ## Reading it back -/

/-- The low half of a surrogate pair after the high half `v`. -/
def decLo (v : Nat) : List Char → Option (Char × List Char)
  | x :: y :: a :: b :: c :: d :: r =>
    if x = '\\' ∧ y = 'u' then some (Char.ofNat (0x10000 + (v - 0xd800) * 1024 + (hex4V a b c d - 0xdc00)), r)
    else none
  | _ => none

/-- A `\u` escape's digits, and the low half of a pair after a high half. -/
def decU : List Char → Option (Char × List Char)
  | a :: b :: c :: d :: r =>
    if 0xd800 ≤ hex4V a b c d ∧ hex4V a b c d < 0xdc00 then decLo (hex4V a b c d) r
    else some (Char.ofNat (hex4V a b c d), r)
  | _ => none

/-- An escape after its backslash. -/
def decEsc : List Char → Option (Char × List Char)
  | [] => none
  | e :: r =>
    if e = '"' then some ('"', r) else if e = '\\' then some ('\\', r) else if e = 'n' then some ('\n', r)
    else if e = 'r' then some ('\r', r) else if e = 't' then some ('\t', r) else if e = 'b' then some (Char.ofNat 8, r)
    else if e = 'f' then some (Char.ofNat 12, r) else if e = 'u' then decU r else none

/-- One character read back from what `pyJsonStr` writes, and what follows it; none at the closing quote. -/
def decC : List Char → Option (Char × List Char)
  | [] => none
  | c :: r => if c = '"' then none else if c = '\\' then decEsc r else some (c, r)

theorem decC_esc (x : Char) (r : List Char) : decC ('\\' :: x :: r) = decEsc (x :: r) := rfl

theorem decC_u (l : List Char) : decC ('\\' :: 'u' :: l) = decU l := rfl

theorem decU_eq (a b c d : Char) (r : List Char) : decU (a :: b :: c :: d :: r) =
    if 0xd800 ≤ hex4V a b c d ∧ hex4V a b c d < 0xdc00 then decLo (hex4V a b c d) r
    else some (Char.ofNat (hex4V a b c d), r) := rfl

theorem decLo_eq (v : Nat) (a b c d : Char) (r : List Char) : decLo v ('\\' :: 'u' :: a :: b :: c :: d :: r) =
    some (Char.ofNat (0x10000 + (v - 0xd800) * 1024 + (hex4V a b c d - 0xdc00)), r) := rfl

theorem decC_plain {c : Char} (h1 : ¬ c = '"') (h2 : ¬ c = '\\') (r : List Char) : decC (c :: r) = some (c, r) := by
  simp [decC, h1, h2]

theorem decC_enc (c : Char) (r : List Char) : decC (enc c ++ r) = some (c, r) := by
  have hv : c.toNat < 0xd800 ∨ (0xdfff < c.toNat ∧ c.toNat < 0x110000) := c.valid
  unfold enc
  by_cases h1 : c = '"'
  · subst h1; rfl
  rw [ite_eq_right h1]
  by_cases h2 : c = '\\'
  · subst h2; rfl
  rw [ite_eq_right h2]
  by_cases h3 : c = '\n'
  · subst h3; rfl
  rw [ite_eq_right h3]
  by_cases h4 : c = '\r'
  · subst h4; rfl
  rw [ite_eq_right h4]
  by_cases h5 : c = '\t'
  · subst h5; rfl
  rw [ite_eq_right h5]
  by_cases k8 : c.toNat = 8
  · rw [ite_eq_left k8]
    show decC ('\\' :: 'b' :: r) = _
    rw [decC_esc]
    show some (Char.ofNat 8, r) = _
    rw [← k8, Char.ofNat_toNat]
  rw [ite_eq_right k8]
  by_cases k12 : c.toNat = 12
  · rw [ite_eq_left k12]
    show decC ('\\' :: 'f' :: r) = _
    rw [decC_esc]
    show some (Char.ofNat 12, r) = _
    rw [← k12, Char.ofNat_toNat]
  rw [ite_eq_right k12]
  by_cases kp : 0x20 ≤ c.toNat ∧ c.toNat ≤ 0x7e
  · rw [ite_eq_left kp]
    exact decC_plain h1 h2 r
  rw [ite_eq_right kp]
  by_cases ku : c.toNat < 0x10000
  · rw [ite_eq_left ku]
    simp only [hex4L, List.cons_append, List.nil_append]
    rw [decC_u, decU_eq, hex4L_val ku, ite_eq_right (by omega), Char.ofNat_toNat]
  rw [ite_eq_right ku]
  have hi : 0xd800 + (c.toNat - 0x10000) / 1024 < 65536 := by omega
  have hl : 0xdc00 + (c.toNat - 0x10000) % 1024 < 65536 := by omega
  simp only [hex4L, List.cons_append, List.nil_append]
  rw [decC_u, decU_eq, hex4L_val hi, ite_eq_left (by omega), decLo_eq, hex4L_val hl]
  have e : 0x10000 + (0xd800 + (c.toNat - 0x10000) / 1024 - 0xd800) * 1024 +
      (0xdc00 + (c.toNat - 0x10000) % 1024 - 0xdc00) = c.toNat := by omega
  rw [e, Char.ofNat_toNat]

theorem enc_cons (c : Char) : ∃ h t, enc c = h :: t ∧ h ≠ '"' := by
  match he : enc c with
  | [] =>
    have := decC_enc c []
    rw [he] at this
    simp [decC] at this
  | h :: t =>
    refine ⟨h, t, rfl, fun hh => ?_⟩
    have := decC_enc c []
    rw [he, hh] at this
    simp [decC] at this

/-- A string read back from what `pyJsonStr` writes after its opening quote, and what follows its closing quote. -/
def decS : Nat → List Char → Option (List Char × List Char)
  | 0, _ => none
  | _ + 1, [] => none
  | fuel + 1, c :: r =>
    if c = '"' then some ([], r)
    else match decC (c :: r) with
      | some (x, r') => (decS fuel r').map fun p => (x :: p.1, p.2)
      | none => none

theorem decS_enc : ∀ (s r : List Char) (fuel : Nat), s.length < fuel →
    decS fuel (s.flatMap enc ++ '"' :: r) = some (s, r)
  | [], r, fuel + 1, _ => by simp [decS]
  | c :: s, r, fuel + 1, h => by
    obtain ⟨x, t, he, hx⟩ := enc_cons c
    have hd := decC_enc c (s.flatMap enc ++ '"' :: r)
    rw [he, List.cons_append] at hd
    rw [List.flatMap_cons, List.append_assoc, he, List.cons_append]
    simp only [decS, ite_eq_right hx, hd, decS_enc s r fuel (by simp at h; omega), Option.map_some]

/-- **`pyJsonStr` is a prefix code**: what it writes determines the string, and what follows. -/
theorem pyJsonStr_split {s s' : String} {r r' : List Char}
    (h : (pyJsonStr s).toList ++ r = (pyJsonStr s').toList ++ r') : s = s' ∧ r = r' := by
  rw [pyJsonStr_toList, pyJsonStr_toList] at h
  simp only [List.cons_append, List.cons.injEq, true_and, List.append_assoc, List.nil_append] at h
  have e1 := decS_enc s.toList r (s.toList.length + s'.toList.length + 1) (by omega)
  have e2 := decS_enc s'.toList r' (s.toList.length + s'.toList.length + 1) (by omega)
  rw [h, e2] at e1
  simp only [Option.some.injEq, Prod.mk.injEq] at e1
  exact ⟨String.toList_inj.mp e1.1.symm, e1.2.symm⟩

theorem pyJsonStr_inj {s s' : String} (h : pyJsonStr s = pyJsonStr s') : s = s' :=
  (pyJsonStr_split (r := []) (r' := []) (by rw [h])).1

/-! ## Numbers -/

/-- Two lists free of a stopping character, each followed by one, split there. -/
theorem split_stop {P : Char → Prop} : ∀ {a a' : List Char} {c c' : Char} {r r' : List Char},
    (∀ x ∈ a, ¬ P x) → (∀ x ∈ a', ¬ P x) → P c → P c' → a ++ c :: r = a' ++ c' :: r' → a = a' ∧ c :: r = c' :: r'
  | [], [], _, _, _, _, _, _, _, _, h => ⟨rfl, by simpa using h⟩
  | [], y :: a', _, _, _, _, _, h', hc, _, h => by
    simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
    exact absurd (h.1 ▸ hc) (h' y List.mem_cons_self)
  | y :: a, [], _, _, _, _, h, _, _, hc', e => by
    simp only [List.nil_append, List.cons_append, List.cons.injEq] at e
    exact absurd (e.1 ▸ hc') (h y List.mem_cons_self)
  | y :: a, y' :: a', c, c', r, r', h, h', hc, hc', e => by
    simp only [List.cons_append, List.cons.injEq] at e
    obtain ⟨rfl, e⟩ := e
    obtain ⟨h1, h2⟩ := split_stop (fun x m => h x (List.mem_cons_of_mem _ m))
      (fun x m => h' x (List.mem_cons_of_mem _ m)) hc hc' e
    exact ⟨by rw [h1], h2⟩

theorem digit_of_mem {n : Nat} {c : Char} (h : c ∈ (toString n).toList) : c.isDigit = true := by
  rw [Nat.toString_eq_repr, Nat.toList_repr] at h
  exact Nat.isDigit_of_mem_toDigits (by decide) (by decide) h

theorem natStr_inj {n n' : Nat} (h : (toString n).toList = (toString n').toList) : n = n' := by
  rw [Nat.toString_eq_repr, Nat.toString_eq_repr, Nat.toList_repr, Nat.toList_repr] at h
  rw [← Nat.ofDigitChars_ten_toDigits (n := n), h, Nat.ofDigitChars_ten_toDigits]

theorem natStr_split {x : Char} (hx : x.isDigit = false) {n n' : Nat} {r r' : List Char}
    (h : (toString n).toList ++ x :: r = (toString n').toList ++ x :: r') : n = n' ∧ r = r' := by
  obtain ⟨h1, h2⟩ := split_stop (P := fun c => c.isDigit = false) (fun c m hc => by rw [digit_of_mem m] at hc; cases hc)
    (fun c m hc => by rw [digit_of_mem m] at hc; cases hc) hx hx h
  exact ⟨natStr_inj h1, (List.cons.inj h2).2⟩

theorem intStr_toList (x : Int) : (toString x).toList =
    match x with
    | .ofNat m => (toString m).toList
    | .negSucc m => '-' :: (toString (m + 1)).toList := by
  cases x <;> simp [toString, Int.repr]

theorem intStr_inj {x y : Int} (h : (toString x).toList = (toString y).toList) : x = y := by
  rw [intStr_toList, intStr_toList] at h
  cases x <;> cases y <;> simp only at h
  · rw [natStr_inj h]
  · have := digit_of_mem (n := _) (c := '-') (by rw [h]; exact List.mem_cons_self)
    cases this
  · have := digit_of_mem (n := _) (c := '-') (by rw [← h]; exact List.mem_cons_self)
    cases this
  · simp only [List.cons.injEq, true_and] at h
    have := natStr_inj h
    rw [show _ = _ from Nat.succ.inj this]

theorem intStr_mem {x : Int} {c : Char} (h : c ∈ (toString x).toList) : c.isDigit = true ∨ c = '-' := by
  rw [intStr_toList] at h
  cases x with
  | ofNat m => exact Or.inl (digit_of_mem h)
  | negSucc m =>
    simp only [List.mem_cons] at h
    rcases h with h | h
    · exact Or.inr h
    · exact Or.inl (digit_of_mem h)

/-! ## The tiers -/

/-- One tier as `tiersJson` writes it. -/
def tierItem (x : String × Nat) : List Char := '[' :: ((pyJsonStr x.1).toList ++ ',' :: ((toString x.2).toList ++ [']']))

theorem item_split {x x' : String × Nat} {r r' : List Char} (h : tierItem x ++ r = tierItem x' ++ r') :
    x = x' ∧ r = r' := by
  simp only [tierItem, List.cons_append, List.cons.injEq, true_and, List.append_assoc] at h
  obtain ⟨h1, h⟩ := pyJsonStr_split h
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨h2, h3⟩ := natStr_split (by decide) h
  exact ⟨Prod.ext h1 h2, h3⟩

theorem rest_split : ∀ {l l' : List (String × Nat)} {r r' : List Char},
    (l.flatMap fun x => ',' :: tierItem x) ++ ']' :: r = (l'.flatMap fun x => ',' :: tierItem x) ++ ']' :: r' →
      l = l' ∧ r = r'
  | [], [], r, r', h => ⟨rfl, by simpa using h⟩
  | [], x :: l', r, r', h => by simp at h
  | x :: l, [], r, r', h => by simp at h
  | x :: l, x' :: l', r, r', h => by
    simp only [List.flatMap_cons, List.cons_append, List.cons.injEq, true_and, List.append_assoc] at h
    obtain ⟨hx, h⟩ := item_split h
    obtain ⟨hl, hr⟩ := rest_split h
    exact ⟨by rw [hx, hl], hr⟩

theorem intercalate_items : ∀ (l : List (String × Nat)) (y : String × Nat),
    [','].intercalate ((y :: l).map tierItem) = tierItem y ++ l.flatMap fun x => ',' :: tierItem x
  | [], y => by simp
  | x :: l, y => by
    rw [List.map_cons, List.map_cons, List.intercalate_cons_cons, ← List.map_cons, intercalate_items l x]
    simp

theorem tiersJson_toList (t : Array (String × Nat)) :
    (tiersJson t).toList = '[' :: ([','].intercalate (t.toList.map tierItem) ++ [']']) := by
  unfold tiersJson
  simp only [String.toList_append, String.toList_intercalate, List.map_map]
  have : (String.toList ∘ fun (x : String × Nat) => match x with
      | (t, b) => "[" ++ pyJsonStr t ++ "," ++ toString b ++ "]") = tierItem := by
    funext x
    obtain ⟨t, b⟩ := x
    simp [tierItem, String.toList_append]
  rw [this]
  simp

/-- **The tiers are a prefix code.** -/
theorem tiers_split {t t' : Array (String × Nat)} {r r' : List Char}
    (h : (tiersJson t).toList ++ r = (tiersJson t').toList ++ r') : t = t' ∧ r = r' := by
  rw [tiersJson_toList, tiersJson_toList] at h
  simp only [List.cons_append, List.cons.injEq, true_and, List.append_assoc] at h
  suffices t.toList = t'.toList ∧ r = r' from ⟨Array.toList_inj.mp this.1, this.2⟩
  generalize t.toList = l at h ⊢
  generalize t'.toList = l' at h ⊢
  match l, l' with
  | [], [] => exact ⟨rfl, by simpa using h⟩
  | [], y' :: l' =>
    rw [intercalate_items] at h
    simp [tierItem] at h
  | y :: l, [] =>
    rw [intercalate_items] at h
    simp [tierItem] at h
  | y :: l, y' :: l' =>
    rw [intercalate_items, intercalate_items, List.append_assoc, List.append_assoc] at h
    obtain ⟨hy, h⟩ := item_split h
    obtain ⟨hl, hr⟩ := rest_split h
    exact ⟨by rw [hy, hl], hr⟩

end MemoryChallenger.Properties.SecurityProofs.WipeJson
