module

public import Flock.Firewall
import all Init.Data.Repr
import Std.Data.DTreeMap.Internal.WF.Lemmas

/-!
# A plain value's text parses back to it

`printJson_roundTrip`: the text `printJson` writes of a plain value (`plain`: its numbers naturals, its strings and keys
without `"`, `\` or control characters, each object's keys ascending) parses (`JsonParse.parse`) to a value with its
canonical form, the form the ZK server's pins compare. `parse_printJson` names the value: `norm j`, each object rebuilt
by inserting its members, in its tree's order, into the empty tree, as the parser builds it.

`reads` is the parse at every fuel, by induction on it: `value` at fuel `k` reads the text of a plain value shorter
than `k`, followed by whatever may follow a value (`Follow`), through `str` (`str_plain`), `num` (`num_chars`) and the
loops of `array` (`array_loop`) and `object` (`object_loop`). `canonForm_norm` is the other half: keys inserted in
ascending order into a tree whose keys are all below them go after its members, in order (`toListModel_foldl`).

`fractional_printJson`: that text has no fraction or exponent, so `step` doesn't refuse it as `fractional`.
`fractional_eq` reads `fractional`'s loop as `scan`, and every part of a plain value's text takes `scan` from outside a
string to outside a string, whatever follows (`Clean`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.FirewallJson

open Lean (Json JsonNumber)
open Flock Flock.Firewall Flock.JsonParse
open Std.DTreeMap.Internal (Impl)

/-- After a value's text: the end, or what continues or closes an array or object. -/
def Follow (rest : List Char) : Prop := ∀ c cs, rest = c :: cs → c = ',' ∨ c = ']' ∨ c = '}'

theorem follow_nil : Follow [] := fun _ _ h => nomatch h

theorem follow_cons {c : Char} (cs : List Char) (h : c = ',' ∨ c = ']' ∨ c = '}') : Follow (c :: cs) :=
  fun _ _ e => by cases e; exact h

theorem ws_cons {c : Char} (cs : List Char) (o : Nat) (h : ¬(c = '\t' ∨ c = '\n' ∨ c = '\r' ∨ c = ' ')) :
    ws (c :: cs) o = (c :: cs, o) := by
  simp only [ws, h, ite_false]

theorem ws_follow {rest : List Char} (h : Follow rest) (o : Nat) : ws rest o = (rest, o) := by
  cases rest with
  | nil => rfl
  | cons c cs => exact ws_cons cs o (by rcases h c cs rfl with rfl | rfl | rfl <;> decide)

/-! ## Strings -/

theorem val_le (c : Char) : c.val ≤ 0x10ffff := by
  have h := c.valid
  simp only [UInt32.isValidChar, Nat.isValidChar] at h
  rw [UInt32.le_iff_toNat_le, show (0x10ffff : UInt32).toNat = 1114111 from rfl]
  omega

theorem str_plain : ∀ (cs : List Char) (k : Nat) (acc : String) (rest : List Char) (o : Nat),
    (∀ c ∈ cs, (c != '"' && c != '\\' && 0x20 ≤ c.val) = true) → cs.length < k →
    ∃ o', JsonParse.str k acc (cs ++ '"' :: rest) o = .ok rest o' (acc ++ String.ofList cs)
  | [], k + 1, acc, rest, o, _, _ => ⟨o + 1, by simp [JsonParse.str]⟩
  | c :: cs, k + 1, acc, rest, o, h, hk => by
    have hc := h c List.mem_cons_self
    simp only [bne_iff_ne, ne_eq, Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨⟨h1, h2⟩, h3⟩ := hc
    obtain ⟨o', ho⟩ := str_plain cs k (acc.push c) rest (o + c.utf8Size)
      (fun c' hc' => h c' (List.mem_cons_of_mem _ hc')) (by simp at hk; omega)
    refine ⟨o', ?_⟩
    simp only [List.cons_append, JsonParse.str, h1, h2, ite_false, h3, val_le c, and_self, ite_true, ho]
    congr 1
    rw [← String.toList_inj]
    simp

/-! ## Numbers -/

theorem digitChar_digit : ∀ d, d < 10 → isDigit d.digitChar = true ∧ (d.digitChar.val - '0'.val).toNat = d := by
  decide

/-- One more decimal digit on `a`, as `digits` reads it. -/
def dval (a : Nat) (c : Char) : Nat := 10 * a + (c.val - '0'.val).toNat

theorem digits_append : ∀ (l rest : List Char) (o acc n : Nat), (∀ c ∈ l, isDigit c = true) →
    (∀ c cs, rest = c :: cs → isDigit c = false) →
    digits (l ++ rest) o acc n = (rest, o + l.length, l.foldl dval acc, n + l.length)
  | [], rest, o, acc, n, _, hr => by
    cases rest with
    | nil => rfl
    | cons c cs =>
      have h := hr c cs rfl
      simp only [isDigit, decide_eq_false_iff_not] at h
      simp only [List.nil_append, digits, h, ite_false, List.length_nil, Nat.add_zero, List.foldl_nil]
  | c :: l, rest, o, acc, n, hl, hr => by
    have h := hl c List.mem_cons_self
    simp only [isDigit, decide_eq_true_eq] at h
    simp only [List.cons_append, digits, h, and_self, ↓reduceIte, List.length_cons, List.foldl_cons]
    rw [digits_append l rest _ _ _ (fun c' hc' => hl c' (List.mem_cons_of_mem _ hc')) hr,
      show o + 1 + l.length = o + (l.length + 1) by omega, show n + 1 + l.length = n + (l.length + 1) by omega]
    rfl

theorem toDigits_digits (m : Nat) : ∀ c ∈ Nat.toDigits 10 m, isDigit c = true := by
  induction m using Nat.strongRecOn with
  | ind m ih =>
    rw [Nat.toDigits_eq_ite (by decide)]
    split
    · intro c hc
      simp only [List.mem_singleton] at hc
      exact hc ▸ (digitChar_digit m (by omega)).1
    · intro c hc
      simp only [List.mem_append, List.mem_singleton] at hc
      rcases hc with hc | hc
      · exact ih (m / 10) (by omega) c hc
      · exact hc ▸ (digitChar_digit (m % 10) (Nat.mod_lt _ (by decide))).1

theorem toDigits_val (m : Nat) : (Nat.toDigits 10 m).foldl dval 0 = m := by
  induction m using Nat.strongRecOn with
  | ind m ih =>
    rw [Nat.toDigits_eq_ite (by decide)]
    split
    · simp only [List.foldl_cons, List.foldl_nil, dval]
      rw [(digitChar_digit m (by omega)).2]
      omega
    · rw [List.foldl_append, ih (m / 10) (by omega)]
      simp only [List.foldl_cons, List.foldl_nil, dval, (digitChar_digit (m % 10) (Nat.mod_lt _ (by decide))).2]
      omega

theorem toDigits_head (m : Nat) (hm : 0 < m) : ∃ d cs, 0 < d ∧ d < 10 ∧ Nat.toDigits 10 m = d.digitChar :: cs := by
  induction m using Nat.strongRecOn with
  | ind m ih =>
    rw [Nat.toDigits_eq_ite (by decide)]
    split
    · exact ⟨m, [], hm, by omega, rfl⟩
    · obtain ⟨d, cs, h0, h1, he⟩ := ih (m / 10) (by omega) (by omega)
      exact ⟨d, cs ++ [(m % 10).digitChar], h0, h1, by rw [he]; rfl⟩

theorem follow_digit {rest : List Char} (hr : Follow rest) : ∀ c cs, rest = c :: cs → isDigit c = false := by
  intro c cs h
  rcases hr c cs h with rfl | rfl | rfl <;> decide

theorem digit_cases : ∀ d, d < 10 → 0 < d → d.digitChar = '1' ∨ d.digitChar = '2' ∨ d.digitChar = '3' ∨
    d.digitChar = '4' ∨ d.digitChar = '5' ∨ d.digitChar = '6' ∨ d.digitChar = '7' ∨ d.digitChar = '8' ∨
    d.digitChar = '9' := by
  decide

theorem num_chars (m : Nat) {rest : List Char} (hr : Follow rest) (o : Nat) :
    ∃ o', num (Nat.toDigits 10 m ++ rest) o = .ok rest o' (JsonNumber.fromInt m) := by
  rcases Nat.eq_zero_or_pos m with rfl | hm
  · rw [Nat.toDigits_zero]
    rcases rest with _ | ⟨c, cs⟩
    · exact ⟨_, rfl⟩
    · rcases hr c cs rfl with rfl | rfl | rfl <;> exact ⟨_, rfl⟩
  · obtain ⟨d, cs, hd0, hd, he⟩ := toDigits_head m hm
    have hall := toDigits_digits m
    have hval := toDigits_val m
    rw [he] at hall hval ⊢
    have hdig := digits_append (d.digitChar :: cs) rest o 0 0 hall (follow_digit hr)
    simp only [List.cons_append] at hdig
    have hc := digit_cases d hd hd0
    rcases rest with _ | ⟨c, cs'⟩
    · simp only [List.append_nil] at hdig ⊢
      rcases hc with h | h | h | h | h | h | h | h | h <;> rw [h] at hdig hval ⊢ <;>
        exact ⟨_, by simp [num, natAfter, hdig, exponent, hval] <;> rfl⟩
    · rcases hr c cs' rfl with rfl | rfl | rfl <;>
      rcases hc with h | h | h | h | h | h | h | h | h <;> rw [h] at hdig hval ⊢ <;>
        exact ⟨_, by simp [num, natAfter, hdig, exponent, hval] <;> rfl⟩

/-! ## What the parser builds -/

mutual
/-- The value `JsonParse` builds from a value's text: each object rebuilt by inserting its members, in its tree's order,
into the empty tree. -/
def norm : Json → Json
  | .null => .null
  | .bool b => .bool b
  | .num n => .num n
  | .str s => .str s
  | .arr ⟨a⟩ => .arr ⟨normList a⟩
  | .obj ⟨⟨t⟩⟩ => .obj (normObj t ∅)
def normList : List Json → List Json
  | [] => []
  | j :: js => norm j :: normList js
def normObj : Impl String (fun _ => Json) → Std.TreeMap.Raw String Json → Std.TreeMap.Raw String Json
  | .leaf, acc => acc
  | .inner _ k v l r, acc => normObj r ((normObj l acc).insert k (norm v))
end

theorem normList_eq : ∀ a : List Json, normList a = a.map norm
  | [] => rfl
  | j :: js => by rw [normList, normList_eq js]; rfl

theorem normObj_eq : ∀ (t : Impl String (fun _ => Json)) (acc : Std.TreeMap.Raw String Json),
    normObj t acc = t.toListModel.foldl (fun a e => a.insert e.1 (norm e.2)) acc
  | .leaf, _ => rfl
  | .inner _ k v l r, acc => by
    rw [normObj, normObj_eq r, normObj_eq l, Impl.toListModel_inner, List.foldl_append, List.foldl_cons]

/-- An object's member as the worker writes it: its key in quotes, a colon, its value's text. -/
def memberChars (e : (_ : String) × Json) : List Char := ('"' :: e.1.toList ++ ['"', ':']) ++ jsonChars e.2

theorem jsonCharsList_eq : ∀ a : List Json, jsonCharsList a = a.map jsonChars
  | [] => rfl
  | j :: js => by rw [jsonCharsList, jsonCharsList_eq js]; rfl

theorem jsonCharsObj_eq : ∀ t : Impl String (fun _ => Json), jsonCharsObj t = t.toListModel.map memberChars
  | .leaf => rfl
  | .inner _ k v l r => by
    rw [jsonCharsObj, jsonCharsObj_eq l, jsonCharsObj_eq r, Impl.toListModel_inner]
    simp [memberChars]

theorem plainList_iff : ∀ a : List Json, plainList a = true ↔ ∀ j ∈ a, plain j = true
  | [] => by simp [plainList]
  | j :: js => by simp [plainList, plainList_iff js]

theorem plainObj_iff : ∀ t : Impl String (fun _ => Json),
    plainObj t = true ↔ ∀ e ∈ t.toListModel, plainStr e.1 = true ∧ plain e.2 = true
  | .leaf => by simp [plainObj]
  | .inner _ k v l r => by
    rw [plainObj, Impl.toListModel_inner]
    simp only [Bool.and_eq_true, plainObj_iff l, plainObj_iff r, List.mem_append, List.mem_cons]
    constructor
    · rintro ⟨⟨⟨hl, hk⟩, hv⟩, hr⟩ e (he | rfl | he)
      · exact hl e he
      · exact ⟨hk, hv⟩
      · exact hr e he
    · intro h
      exact ⟨⟨⟨fun e he => h e (.inl he), (h _ (.inr (.inl rfl))).1⟩, (h _ (.inr (.inl rfl))).2⟩,
        fun e he => h e (.inr (.inr he))⟩

theorem plainStr_chars {s : String} (h : plainStr s = true) :
    ∀ c ∈ s.toList, (c != '"' && c != '\\' && 0x20 ≤ c.val) = true := by
  rw [plainStr, String.all_bool_eq, List.all_eq_true] at h
  exact h

/-! ## Where a value's text starts -/

/-- A character a value's text starts with. -/
def Lead (c : Char) : Prop := c = 'n' ∨ c = 't' ∨ c = 'f' ∨ c = '"' ∨ c = '[' ∨ c = '{' ∨ ∃ d, d < 10 ∧ c = d.digitChar

theorem digit_lead : ∀ d, d < 10 →
    ¬(d.digitChar = '\t' ∨ d.digitChar = '\n' ∨ d.digitChar = '\r' ∨ d.digitChar = ' ') ∧ d.digitChar ≠ ']' := by
  decide

theorem lead_ne {c : Char} (h : Lead c) : ¬(c = '\t' ∨ c = '\n' ∨ c = '\r' ∨ c = ' ') ∧ c ≠ ']' := by
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl | ⟨d, hd, rfl⟩
  all_goals first | decide | exact digit_lead d hd

theorem jsonChars_head (j : Json) : ∃ c cs, jsonChars j = c :: cs ∧ Lead c := by
  match j with
  | .null => exact ⟨_, _, rfl, .inl rfl⟩
  | .bool true => exact ⟨_, _, rfl, .inr (.inl rfl)⟩
  | .bool false => exact ⟨_, _, rfl, .inr (.inr (.inl rfl))⟩
  | .num ⟨m, _⟩ =>
    rcases Nat.eq_zero_or_pos m.toNat with h | h
    · exact ⟨'0', [], by rw [jsonChars, h, Nat.toDigits_zero],
        .inr (.inr (.inr (.inr (.inr (.inr ⟨0, by decide, rfl⟩)))))⟩
    · obtain ⟨d, cs, _, hd, he⟩ := toDigits_head _ h
      exact ⟨_, cs, by rw [jsonChars, he], .inr (.inr (.inr (.inr (.inr (.inr ⟨d, hd, rfl⟩)))))⟩
  | .str _ => exact ⟨_, _, by rw [jsonChars, List.cons_append], .inr (.inr (.inr (.inl rfl)))⟩
  | .arr ⟨_⟩ => exact ⟨_, _, by rw [jsonChars, List.cons_append], .inr (.inr (.inr (.inr (.inl rfl))))⟩
  | .obj ⟨⟨_⟩⟩ => exact ⟨_, _, by rw [jsonChars, List.cons_append], .inr (.inr (.inr (.inr (.inr (.inl rfl)))))⟩

theorem ws_chars (j : Json) (r : List Char) (o : Nat) : ws (jsonChars j ++ r) o = (jsonChars j ++ r, o) := by
  obtain ⟨c, cs, he, hl⟩ := jsonChars_head j
  rw [he, List.cons_append, ws_cons _ _ (lead_ne hl).1]

theorem ws_joined (j : Json) (l : List (List Char)) (r : List Char) (o : Nat) :
    ws ([','].intercalate (jsonChars j :: l) ++ r) o = ([','].intercalate (jsonChars j :: l) ++ r, o) := by
  obtain ⟨c, cs, he, hl⟩ := jsonChars_head j
  rw [he, List.intercalate_cons_cons_left, List.cons_append, ws_cons _ _ (lead_ne hl).1]

/-! ## `value` reads a plain value's text back -/

/-- `value` at fuel `k` reads each plain value whose text is shorter than `k`, followed by what may follow a value, to
what the parser builds of it. -/
def Reads (k : Nat) : Prop :=
  ∀ j, plain j = true → (jsonChars j).length < k → ∀ rest, Follow rest → ∀ o,
    ∃ o', value k (jsonChars j ++ rest) o = .ok rest o' (norm j)

theorem array_loop {K : Nat} (IH : ∀ m, m < K → Reads m) : ∀ (l : List Json) (k : Nat) (acc : Array Json)
    (rest : List Char) (o : Nat), k ≤ K → l ≠ [] → (∀ e ∈ l, plain e = true) →
    ([','].intercalate (l.map jsonChars) ++ [']']).length < k → Follow rest →
    ∃ o', array k acc ([','].intercalate (l.map jsonChars) ++ ']' :: rest) o = .ok rest o' (acc ++ (l.map norm).toArray)
  | [], _, _, _, _, _, h, _, _, _ => absurd rfl h
  | [e], k + 1, acc, rest, o, hk, _, hp, hl, hr => by
    simp only [List.map_cons, List.map_nil, List.intercalate_singleton, List.length_append, List.length_singleton]
      at hl ⊢
    obtain ⟨o1, h1⟩ := IH k (by omega) e (hp e List.mem_cons_self) (by omega) (']' :: rest)
      (follow_cons _ (by simp)) o
    refine ⟨o1 + 1, ?_⟩
    simp only [array, h1, ↓reduceIte, ws_follow hr]
    simp
  | e :: e' :: es, k + 1, acc, rest, o, hk, _, hp, hl, hr => by
    simp only [List.map_cons, List.intercalate_cons_cons, List.length_append, List.length_singleton] at hl
    have he : [','].intercalate (jsonChars e :: jsonChars e' :: es.map jsonChars) ++ ']' :: rest =
        jsonChars e ++ (',' :: ([','].intercalate (jsonChars e' :: es.map jsonChars) ++ ']' :: rest)) := by simp
    obtain ⟨o1, h1⟩ := IH k (by omega) e (hp e List.mem_cons_self) (by omega)
      (',' :: ([','].intercalate (jsonChars e' :: es.map jsonChars) ++ ']' :: rest)) (follow_cons _ (by simp)) o
    have hl' : ([','].intercalate ((e' :: es).map jsonChars) ++ [']']).length < k := by
      simp only [List.map_cons, List.length_append, List.length_singleton]
      omega
    obtain ⟨o2, h2⟩ := array_loop IH (e' :: es) k (acc.push (norm e)) rest (o1 + 1) (by omega) (by simp)
      (fun x hx => hp x (List.mem_cons_of_mem _ hx)) hl' hr
    refine ⟨o2, ?_⟩
    simp only [List.map_cons] at h2 ⊢
    rw [he]
    simp only [array, h1, show (',' = ']') = False by decide, ↓reduceIte, ws_joined, h2]
    simp

theorem ws_colon (cs : List Char) (o : Nat) : ws (':' :: cs) o = (':' :: cs, o) := ws_cons cs o (by decide)

theorem ws_members (e : (_ : String) × Json) (l : List (List Char)) (r : List Char) (o : Nat) :
    ws ([','].intercalate (memberChars e :: l) ++ r) o = ([','].intercalate (memberChars e :: l) ++ r, o) := by
  rw [show memberChars e = '"' :: (e.1.toList ++ ['"', ':'] ++ jsonChars e.2) by simp [memberChars],
    List.intercalate_cons_cons_left, List.cons_append, ws_cons _ _ (by decide)]

theorem key_empty (key : String) : "" ++ String.ofList key.toList = key := by
  rw [String.ofList_toList]; rfl

theorem object_loop {K : Nat} (IH : ∀ m, m < K → Reads m) : ∀ (l : List ((_ : String) × Json)) (k : Nat)
    (kvs : Std.TreeMap.Raw String Json) (rest : List Char) (o : Nat), k ≤ K → l ≠ [] →
    (∀ e ∈ l, plainStr e.1 = true ∧ plain e.2 = true) →
    ([','].intercalate (l.map memberChars) ++ ['}']).length < k → Follow rest →
    ∃ o', object k kvs ([','].intercalate (l.map memberChars) ++ '}' :: rest) o =
      .ok rest o' (l.foldl (fun a e => a.insert e.1 (norm e.2)) kvs)
  | [], _, _, _, _, _, h, _, _, _ => absurd rfl h
  | [e], k + 1, kvs, rest, o, hk, _, hp, hl, hr => by
    obtain ⟨hs, hv⟩ := hp e List.mem_cons_self
    simp only [List.map_cons, List.map_nil, List.intercalate_singleton] at hl ⊢
    have hlen := hl
    simp only [memberChars, List.length_append, List.length_cons, List.length_nil] at hlen
    obtain ⟨o1, h1⟩ := str_plain e.1.toList k "" (':' :: (jsonChars e.2 ++ '}' :: rest)) (o + 1) (plainStr_chars hs)
      (by omega)
    obtain ⟨o2, h2⟩ := IH k (by omega) e.2 hv (by omega) ('}' :: rest) (follow_cons _ (by simp)) (o1 + 1)
    refine ⟨o2 + 1, ?_⟩
    rw [show memberChars e ++ '}' :: rest = '"' :: (e.1.toList ++ '"' :: ':' :: (jsonChars e.2 ++ '}' :: rest)) by
      simp [memberChars]]
    simp only [object, h1, key_empty, ws_colon, ws_chars, h2, ↓reduceIte, ws_follow hr, List.foldl_cons,
      List.foldl_nil]
  | e :: e' :: es, k + 1, kvs, rest, o, hk, _, hp, hl, hr => by
    obtain ⟨hs, hv⟩ := hp e List.mem_cons_self
    simp only [List.map_cons, List.intercalate_cons_cons] at hl ⊢
    have hlen := hl
    simp only [memberChars, List.length_append, List.length_cons, List.length_nil] at hlen
    let more := [','].intercalate (memberChars e' :: es.map memberChars) ++ '}' :: rest
    obtain ⟨o1, h1⟩ := str_plain e.1.toList k "" (':' :: (jsonChars e.2 ++ ',' :: more)) (o + 1) (plainStr_chars hs)
      (by omega)
    obtain ⟨o2, h2⟩ := IH k (by omega) e.2 hv (by omega) (',' :: more) (follow_cons _ (by simp)) (o1 + 1)
    have hl' : ([','].intercalate ((e' :: es).map memberChars) ++ ['}']).length < k := by
      simp only [List.map_cons, List.length_append, List.length_singleton] at hl ⊢
      omega
    obtain ⟨o3, h3⟩ := object_loop IH (e' :: es) k (kvs.insert e.1 (norm e.2)) rest (o2 + 1) (by omega) (by simp)
      (fun x hx => hp x (List.mem_cons_of_mem _ hx)) hl' hr
    refine ⟨o3, ?_⟩
    rw [show memberChars e ++ [','] ++ [','].intercalate (memberChars e' :: es.map memberChars) ++ '}' :: rest =
      '"' :: (e.1.toList ++ '"' :: ':' :: (jsonChars e.2 ++ ',' :: more)) by simp [memberChars, more]]
    simp only [List.map_cons] at h3
    simp only [object, h1, key_empty, ws_colon, ws_chars, h2, show (',' = '}') = False by decide, ↓reduceIte,
      more, ws_members, h3, List.foldl_cons]

theorem digit_tests : ∀ d, d < 10 → ¬d.digitChar = '[' ∧ ¬d.digitChar = '{' ∧ ¬d.digitChar = '"' ∧
    ¬d.digitChar = 'f' ∧ ¬d.digitChar = 't' ∧ ¬d.digitChar = 'n' ∧
    (d.digitChar = '-' ∨ ('0' ≤ d.digitChar ∧ d.digitChar ≤ '9')) := by
  decide

theorem value_num (n k : Nat) {rest : List Char} (hr : Follow rest) (o : Nat) :
    ∃ o', value (k + 1) (Nat.toDigits 10 n ++ rest) o = .ok rest o' (.num (JsonNumber.fromInt n)) := by
  obtain ⟨o', ho⟩ := num_chars n hr o
  refine ⟨o', ?_⟩
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · rw [Nat.toDigits_zero] at ho ⊢
    simp only [List.cons_append, List.nil_append] at ho ⊢
    simp [value, ho, ws_follow hr]
  · obtain ⟨d, cs, _, hd, he⟩ := toDigits_head n hn
    rw [he] at ho ⊢
    simp only [List.cons_append] at ho ⊢
    obtain ⟨t1, t2, t3, t4, t5, t6, t7⟩ := digit_tests d hd
    simp only [value, t1, t2, t3, t4, t5, t6, t7, ite_false, ite_true, ho, ws_follow hr]

/-- `value` reads a plain value's text back at every fuel that exceeds its length. -/
theorem reads : ∀ k, Reads k := by
  intro k
  induction k using Nat.strongRecOn with
  | ind k IH =>
    intro j hp hl rest hr o
    cases k with
    | zero => exact absurd hl (Nat.not_lt_zero _)
    | succ k =>
    match j with
    | .null => exact ⟨_, by simp [jsonChars, value, word, lit, ws_follow hr, norm] <;> rfl⟩
    | .bool true => exact ⟨_, by simp [jsonChars, value, word, lit, ws_follow hr, norm] <;> rfl⟩
    | .bool false => exact ⟨_, by simp [jsonChars, value, word, lit, ws_follow hr, norm] <;> rfl⟩
    | .num ⟨m, e⟩ =>
      simp only [plain, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp
      obtain ⟨rfl, hm⟩ := hp
      obtain ⟨o', ho⟩ := value_num m.toNat k hr o
      refine ⟨o', ?_⟩
      rw [jsonChars, ho, norm, JsonNumber.fromInt, Int.toNat_of_nonneg hm]
    | .str s =>
      simp only [plain] at hp
      simp only [jsonChars, List.length_append, List.length_cons] at hl
      obtain ⟨o1, h1⟩ := str_plain s.toList k "" rest (o + 1) (plainStr_chars hp) (by omega)
      refine ⟨o1, ?_⟩
      simp only [jsonChars, List.cons_append, List.append_assoc]
      simp [value, h1, ws_follow hr, norm]
    | .arr ⟨a⟩ =>
      simp only [plain, plainList_iff] at hp
      simp only [jsonChars, jsonCharsList_eq] at hl ⊢
      cases a with
      | nil =>
        refine ⟨o + 1 + 1, ?_⟩
        simp [value, ws_cons _ _ (show ¬(']' = '\t' ∨ ']' = '\n' ∨ ']' = '\r' ∨ ']' = ' ') by decide),
          ws_follow hr, norm, normList]
      | cons e es =>
        have hl' : ([','].intercalate ((e :: es).map jsonChars) ++ [']']).length < k := by
          simp only [List.length_cons, List.length_append] at hl ⊢
          omega
        obtain ⟨o', ho⟩ := array_loop IH (e :: es) k #[] rest (o + 1) (by omega) (by simp) hp hl' hr
        refine ⟨o', ?_⟩
        obtain ⟨c, cs, hc, hlead⟩ := jsonChars_head e
        have hne := (lead_ne hlead).2
        simp only [List.map_cons, hc, List.intercalate_cons_cons_left] at ho ⊢
        simp only [List.cons_append, List.append_assoc, List.nil_append] at ho ⊢
        simp only [value, ↓reduceIte, ws_cons _ _ (lead_ne hlead).1]
        split
        · rename_i h; simp at h
        · rename_i h; simp only [Prod.mk.injEq, List.cons.injEq] at h; exact absurd h.1.1 hne
        · rename_i heq
          simp only [Prod.mk.injEq] at heq
          obtain ⟨rfl, rfl⟩ := heq
          simp [ho, norm, normList_eq]
    | .obj ⟨⟨t⟩⟩ =>
      simp only [plain, Bool.and_eq_true, plainObj_iff] at hp
      simp only [jsonChars, jsonCharsObj_eq] at hl ⊢
      rw [norm, normObj_eq]
      cases ht : t.toListModel with
      | nil =>
        refine ⟨o + 1 + 1, ?_⟩
        simp [value, ws_cons _ _ (show ¬('}' = '\t' ∨ '}' = '\n' ∨ '}' = '\r' ∨ '}' = ' ') by decide),
          ws_follow hr]
      | cons e es =>
        rw [ht] at hl hp
        have hl' : ([','].intercalate ((e :: es).map memberChars) ++ ['}']).length < k := by
          simp only [List.length_cons, List.length_append] at hl ⊢
          omega
        obtain ⟨o', ho⟩ := object_loop IH (e :: es) k ∅ rest (o + 1) (by omega) (by simp) hp.1 hl' hr
        refine ⟨o', ?_⟩
        simp only [List.map_cons, show memberChars e = '"' :: (e.1.toList ++ ['"', ':'] ++ jsonChars e.2) by
          simp [memberChars], List.intercalate_cons_cons_left] at ho ⊢
        simp only [List.cons_append, List.append_assoc, List.nil_append, List.foldl_cons] at ho ⊢
        simp [value, ws_cons _ _ (show ¬('"' = '\t' ∨ '"' = '\n' ∨ '"' = '\r' ∨ '"' = ' ') by decide), ho]

/-- **A plain value's text parses back** to what the parser builds of it (`norm`). -/
theorem parse_printJson {j : Json} (h : plain j = true) : JsonParse.parse (printJson j) = .ok (norm j) := by
  rw [parse_eq, printJson, String.toList_ofList]
  obtain ⟨o', ho⟩ := reads (2 * (jsonChars j).length + 2) j h (by omega) [] follow_nil 0
  rw [List.append_nil] at ho
  have hws : ws (jsonChars j) 0 = (jsonChars j, 0) := by simpa using ws_chars j [] 0
  simp only [parseList, hws, ho]

/-! ## What the parser builds has the canonical form of what was printed -/

theorem canonFormList_eq : ∀ a : List Json, canonFormList a = a.flatMap canonForm
  | [] => rfl
  | j :: js => by rw [canonFormList, canonFormList_eq js]; rfl

theorem canonFormObj_eq : ∀ t : Impl String (fun _ => Json),
    canonFormObj t = t.toListModel.flatMap fun e => .key e.1 :: canonForm e.2
  | .leaf => rfl
  | .inner _ k v l r => by
    rw [canonFormObj, canonFormObj_eq l, canonFormObj_eq r, Impl.toListModel_inner]
    simp

theorem flatMap_congr {α β : Type} {l : List α} {f g : α → List β} (h : ∀ x ∈ l, f x = g x) :
    l.flatMap f = l.flatMap g := by
  induction l with
  | nil => rfl
  | cons x l ih => simp [h x List.mem_cons_self, ih fun y hy => h y (List.mem_cons_of_mem _ hy)]

theorem canonForm_obj (m : Std.TreeMap.Raw String Json) :
    canonForm (.obj m) = .obj :: canonFormObj m.inner.inner ++ [.close] := rfl

/-- Members inserted in ascending key order, each key above every key of the tree, go after its members, in order. -/
theorem toListModel_foldl : ∀ (L : List ((_ : String) × Json)) (m : Std.TreeMap.Raw String Json), m.inner.inner.WF →
    (m.inner.inner.toListModel ++ L).Pairwise (fun a b => compare a.1 b.1 = .lt) →
    (L.foldl (fun a e => a.insert e.1 e.2) m).inner.inner.toListModel = m.inner.inner.toListModel ++ L
  | [], _, _, _ => by simp
  | e :: L, m, hw, hp => by
    rw [List.foldl_cons]
    have hw' : (m.insert e.1 e.2).inner.inner.WF := hw.insert!
    have hfresh : Std.Internal.List.containsKey e.1 m.inner.inner.toListModel = false := by
      rw [Std.Internal.List.containsKey_eq_false_iff]
      intro b hb
      have hlt := (List.pairwise_append.mp hp).2.2 b hb e List.mem_cons_self
      rw [beq_eq_false_iff_ne]
      intro he
      rw [he] at hlt
      exact absurd hlt (by simp)
    have hperm := Impl.toListModel_insert! (k := e.1) (v := e.2) hw.balanced hw.ordered
    rw [Std.Internal.List.insertEntry_of_containsKey_eq_false hfresh] at hperm
    have heq : (m.insert e.1 e.2).inner.inner.toListModel = m.inner.inner.toListModel ++ [e] := by
      refine List.Perm.eq_of_pairwise (le := fun a b => compare a.1 b.1 = .lt) ?_ (by exact hw'.ordered) ?_
        (hperm.trans (List.perm_append_singleton _ _).symm)
      · intro a b _ _ hab hba
        rw [Std.OrientedCmp.gt_iff_lt.symm] at hba
        rw [hab] at hba
        cases hba
      · have := (List.pairwise_append.mp hp)
        rw [List.pairwise_append]
        exact ⟨this.1, by simp, fun a ha b hb => by
          rw [List.mem_singleton] at hb; subst hb; exact this.2.2 a ha _ List.mem_cons_self⟩
    rw [toListModel_foldl L _ hw' (by rw [heq, List.append_assoc]; exact hp), heq, List.append_assoc]
    rfl

mutual
theorem canonForm_norm : ∀ j : Json, plain j = true → canonForm (norm j) = canonForm j
  | .null, _ => rfl
  | .bool _, _ => rfl
  | .num _, _ => rfl
  | .str _, _ => rfl
  | .arr ⟨a⟩, h => by
    simp only [plain] at h
    simp only [norm, canonForm, canonFormList_normList a h]
  | .obj ⟨⟨t⟩⟩, h => by
    simp only [plain, Bool.and_eq_true] at h
    have hasc : t.toListModel.Pairwise (fun a b => compare a.1 b.1 = .lt) := of_decide_eq_true h.2
    have hf : t.toListModel.foldl (fun a e => a.insert e.1 (norm e.2)) (∅ : Std.TreeMap.Raw String Json) =
        (t.toListModel.map fun e => (⟨e.1, norm e.2⟩ : (_ : String) × Json)).foldl (fun a e => a.insert e.1 e.2) ∅ := by
      rw [List.foldl_map]
    have h0 : (∅ : Std.TreeMap.Raw String Json).inner.inner.toListModel = [] := rfl
    rw [norm, canonForm_obj, normObj_eq, hf, canonFormObj_eq,
      toListModel_foldl _ _ Impl.WF.empty (by rw [h0, List.nil_append, List.pairwise_map]; exact hasc), h0]
    simp only [canonForm, canonFormObj_eq]
    have hm := canonForm_members t h.1
    simp only [List.nil_append, List.flatMap_map]
    congr 2
    exact flatMap_congr fun e he => by rw [hm e he]
theorem canonFormList_normList : ∀ a : List Json, plainList a = true → canonFormList (normList a) = canonFormList a
  | [], _ => rfl
  | j :: js, h => by
    simp only [plainList, Bool.and_eq_true] at h
    simp only [normList, canonFormList, canonForm_norm j h.1, canonFormList_normList js h.2]
theorem canonForm_members : ∀ t : Impl String (fun _ => Json), plainObj t = true →
    ∀ e ∈ t.toListModel, canonForm (norm e.2) = canonForm e.2
  | .leaf, _ => by simp [Impl.toListModel]
  | .inner _ k v l r, h => by
    simp only [plainObj, Bool.and_eq_true] at h
    intro e he
    rw [Impl.toListModel_inner, List.mem_append, List.mem_cons] at he
    rcases he with he | rfl | he
    · exact canonForm_members l h.1.1.1 e he
    · exact canonForm_norm v h.1.2
    · exact canonForm_members r h.2 e he
end

/-- **A plain value's text round-trips**: it parses to a value with its canonical form, the form the ZK server's pins
compare. -/
theorem printJson_roundTrip {j : Json} (h : plain j = true) :
    ∃ j', JsonParse.parse (printJson j) = .ok j' ∧ canonForm j' = canonForm j :=
  ⟨norm j, parse_printJson h, canonForm_norm j h⟩

/-! ## No fraction or exponent -/

/-- `fractional`'s loop on the text left, from its state (in a string, after a backslash in one, in a number): `none`
once it meets a fraction or an exponent, else its state at the end. -/
def scan : Bool × Bool × Bool → List Char → Option (Bool × Bool × Bool)
  | st, [] => some st
  | (inStr, esc, inNum), c :: cs =>
    if inStr then
      if esc then scan (inStr, false, inNum) cs
      else if c == '\\' then scan (inStr, true, inNum) cs
      else if c == '"' then scan (false, esc, inNum) cs
      else scan (inStr, esc, inNum) cs
    else if c == '"' then scan (true, esc, false) cs
    else if c.isDigit || c == '-' then scan (inStr, esc, true) cs
    else if inNum && (c == '.' || c == 'e' || c == 'E') then none
    else scan (inStr, esc, false) cs

theorem fractional_eq (t : String) : fractional t = (scan (false, false, false) t.toList).isNone := by
  unfold fractional
  simp only [Id.run]
  generalize t.toList = cs
  generalize hs : ((none, false, false, false) : Option Bool × Bool × Bool × Bool) = s
  have h1 : s.1 = none := by rw [← hs]
  have h2 : (false, false, false) = s.2 := by rw [← hs]
  rw [h2]
  clear hs h2
  induction cs generalizing s with
  | nil =>
    obtain ⟨r, a, b, d⟩ := s
    cases h1
    rfl
  | cons c cs ih =>
    obtain ⟨r, a, b, d⟩ := s
    cases h1
    simp only [List.forIn_cons, scan]
    generalize (c == '\\') = q1
    generalize (c == '"') = q2
    generalize (c.isDigit || c == '-') = q3
    generalize (c == '.' || c == 'e' || c == 'E') = q4
    cases a <;> cases b <;> cases d <;> cases q1 <;> cases q2 <;> cases q3 <;> cases q4 <;>
      simp only [↓reduceIte, Bool.false_eq_true, Bool.and_true, Bool.and_false] <;>
      first | rfl | exact ih _ rfl

/-- Text the scan crosses from outside a string to outside a string, whatever follows it. -/
def Clean (l : List Char) : Prop := ∀ b rest, ∃ b', scan (false, false, b) (l ++ rest) = scan (false, false, b') rest

theorem clean_nil : Clean [] := fun b _ => ⟨b, rfl⟩

theorem clean_append {l l' : List Char} (h : Clean l) (h' : Clean l') : Clean (l ++ l') := by
  intro b rest
  obtain ⟨b1, h1⟩ := h b (l' ++ rest)
  obtain ⟨b2, h2⟩ := h' b1 rest
  exact ⟨b2, by rw [List.append_assoc, h1, h2]⟩

/-- A character outside a string that is no quote, digit, sign, point or exponent. -/
theorem clean_char {c : Char}
    (h : (c == '"') = false ∧ (c.isDigit || c == '-') = false ∧ (c == '.' || c == 'e' || c == 'E') = false) :
    Clean [c] := fun b rest => ⟨false, by simp only [List.singleton_append, scan, h.1, h.2.1, h.2.2]; cases b <;> rfl⟩

theorem clean_word : Clean ['n', 'u', 'l', 'l'] ∧ Clean ['t', 'r', 'u', 'e'] ∧ Clean ['f', 'a', 'l', 's', 'e'] :=
  ⟨fun b _ => ⟨false, by cases b <;> rfl⟩, fun b _ => ⟨false, by cases b <;> rfl⟩,
    fun b _ => ⟨false, by cases b <;> rfl⟩⟩

theorem clean_digits : ∀ ds : List Char, (∀ c ∈ ds, c.isDigit = true) → Clean ds
  | [], _ => clean_nil
  | c :: ds, h => by
    intro b rest
    have hc := h c (List.mem_cons_self ..)
    have hq : (c == '"') = false := by
      cases hq : c == '"'
      · rfl
      · rw [beq_iff_eq] at hq
        subst hq
        exact absurd hc (by decide)
    obtain ⟨b', hb⟩ := clean_digits ds (fun c' hc' => h c' (List.mem_cons_of_mem _ hc')) true rest
    exact ⟨b', by simp only [List.cons_append, scan, hq, hc, Bool.true_or, ↓reduceIte, Bool.false_eq_true]; exact hb⟩

theorem toDigits_isDigit (m : Nat) : ∀ c ∈ Nat.toDigits 10 m, c.isDigit = true := by
  have hd : ∀ d, d < 10 → d.digitChar.isDigit = true := by decide
  induction m using Nat.strongRecOn with
  | ind m ih =>
    rw [Nat.toDigits_eq_ite (by decide)]
    split
    · intro c hc
      simp only [List.mem_singleton] at hc
      exact hc ▸ hd m (by omega)
    · intro c hc
      simp only [List.mem_append, List.mem_singleton] at hc
      rcases hc with hc | hc
      · exact ih (m / 10) (by omega) c hc
      · exact hc ▸ hd (m % 10) (Nat.mod_lt _ (by decide))

theorem scan_plain : ∀ cs : List Char, (∀ c ∈ cs, (c != '"' && c != '\\' && 0x20 ≤ c.val) = true) →
    ∀ d rest, scan (true, false, d) (cs ++ rest) = scan (true, false, d) rest
  | [], _, _, _ => rfl
  | c :: cs, h, d, rest => by
    have hc := h c (List.mem_cons_self ..)
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hc
    simp only [List.cons_append, scan, ↓reduceIte, Bool.false_eq_true, beq_eq_false_iff_ne.mpr hc.1.1,
      beq_eq_false_iff_ne.mpr hc.1.2]
    exact scan_plain cs (fun c' hc' => h c' (List.mem_cons_of_mem _ hc')) d rest

theorem clean_str {s : String} (h : plainStr s = true) : Clean ('"' :: s.toList ++ ['"']) := fun b rest => by
  refine ⟨false, ?_⟩
  simp only [List.cons_append, List.append_assoc, List.nil_append]
  rw [show scan (false, false, b) ('"' :: (s.toList ++ '"' :: rest)) =
    scan (true, false, false) (s.toList ++ '"' :: rest) from rfl, scan_plain _ (plainStr_chars h)]
  rfl

theorem clean_join : ∀ L : List (List Char), (∀ l ∈ L, Clean l) → Clean ([','].intercalate L)
  | [], _ => clean_nil
  | [l], h => by rw [List.intercalate_singleton]; exact h l (List.mem_singleton_self _)
  | l :: l' :: L, h => by
    rw [List.intercalate_cons_cons]
    exact clean_append (clean_append (h l (List.mem_cons_self ..)) (clean_char (by decide)))
      (clean_join (l' :: L) fun x hx => h x (List.mem_cons_of_mem _ hx))

theorem clean_wrap {o c : Char} {X : List Char}
    (ho : (o == '"') = false ∧ (o.isDigit || o == '-') = false ∧ (o == '.' || o == 'e' || o == 'E') = false)
    (hc : (c == '"') = false ∧ (c.isDigit || c == '-') = false ∧ (c == '.' || c == 'e' || c == 'E') = false)
    (hX : Clean X) : Clean (o :: X ++ [c]) :=
  clean_append (clean_append (clean_char ho) hX) (clean_char hc)

theorem clean_key {k : String} (h : plainStr k = true) : Clean ('"' :: k.toList ++ ['"', ':']) := by
  have e : '"' :: k.toList ++ ['"', ':'] = ('"' :: k.toList ++ ['"']) ++ [':'] := by simp
  rw [e]
  exact clean_append (clean_str h) (clean_char (by decide))

mutual
theorem clean_value : ∀ j : Json, plain j = true → Clean (jsonChars j)
  | .null, _ => clean_word.1
  | .bool true, _ => clean_word.2.1
  | .bool false, _ => clean_word.2.2
  | .num ⟨m, _⟩, _ => clean_digits _ (toDigits_isDigit m.toNat)
  | .str _, h => clean_str h
  | .arr ⟨a⟩, h => by
    simp only [plain] at h
    exact clean_wrap (by decide) (by decide) (clean_join _ (clean_list a h))
  | .obj ⟨⟨t⟩⟩, h => by
    simp only [plain, Bool.and_eq_true] at h
    exact clean_wrap (by decide) (by decide) (clean_join _ (clean_members t h.1))
theorem clean_list : ∀ a : List Json, plainList a = true → ∀ l ∈ jsonCharsList a, Clean l
  | [], _ => by simp [jsonCharsList]
  | j :: js, h => by
    simp only [plainList, Bool.and_eq_true] at h
    intro l hl
    simp only [jsonCharsList, List.mem_cons] at hl
    rcases hl with rfl | hl
    · exact clean_value j h.1
    · exact clean_list js h.2 l hl
theorem clean_members : ∀ t : Impl String (fun _ => Json), plainObj t = true → ∀ l ∈ jsonCharsObj t, Clean l
  | .leaf, _ => by simp [jsonCharsObj]
  | .inner _ k v l r, h => by
    simp only [plainObj, Bool.and_eq_true] at h
    intro x hx
    simp only [jsonCharsObj, List.mem_append, List.mem_singleton] at hx
    rcases hx with (hx | rfl) | hx
    · exact clean_members l h.1.1.1 x hx
    · exact clean_append (clean_key h.1.1.2) (clean_value v h.1.2)
    · exact clean_members r h.2 x hx
end

/-- **A plain value's text has no fraction or exponent**, so the ZK server's `Hello` doesn't refuse it as
`fractional`. -/
theorem fractional_printJson {j : Json} (h : plain j = true) : fractional (printJson j) = false := by
  rw [fractional_eq, printJson, String.toList_ofList]
  obtain ⟨b, hb⟩ := clean_value j h false []
  rw [List.append_nil] at hb
  rw [hb]
  rfl

end FlockSoundness.Discharge.Recursion.FirewallJson
