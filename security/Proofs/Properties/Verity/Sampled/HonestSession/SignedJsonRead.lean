module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.SignedJson
public import Proofs.Flock.Recursive.ZkServerJson
public import Proofs.Flock.Claims.Complete.Json.RoundTrip
import all Init.Data.Repr
import all Init.Data.Int.Repr
import Std.Data.DTreeMap.Internal.WF.Lemmas

/-!
# A value's text with signed numbers parses back to it

`parse_jsonCharsS`: the text `jsonCharsS` writes of a value whose numbers are integers and whose strings, keys and
objects are plain (`plainS`) parses (`JsonParse.parse`) to `ZkServerJson.norm` of it. It is
`ZkServerJson.parse_printJson` with each number's sign: a number is read by `ReadBack.num_int`, and the rest is
`ZkServerJson.reads`' induction on the fuel over `jsonCharsS`.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.SignedJson

open Lean (Json JsonNumber)
open Flock Flock.ZkServer Flock.JsonParse
open FlockSoundness.Discharge.Recursion.ZkServerJson
open Std.DTreeMap.Internal (Impl)

mutual
/-- A value whose `jsonCharsS` text parses back to it: `ZkServer.plain` with integer numbers of either sign. -/
def plainS : Json → Bool
  | .null => true
  | .bool _ => true
  | .num ⟨_, e⟩ => e == 0
  | .str s => plainStr s
  | .arr ⟨a⟩ => plainSList a
  | .obj ⟨⟨t⟩⟩ => plainSObj t && ascending t
def plainSList : List Json → Bool
  | [] => true
  | j :: js => plainS j && plainSList js
def plainSObj : Impl String (fun _ => Json) → Bool
  | .leaf => true
  | .inner _ k v l r => plainSObj l && plainStr k && plainS v && plainSObj r
end

/-- An object's member as `jsonCharsS` writes it: its key in quotes, a colon, its value's text. -/
def memberCharsS (e : (_ : String) × Json) : List Char := ('"' :: e.1.toList ++ ['"', ':']) ++ jsonCharsS e.2

theorem jsonCharsSList_eq : ∀ a : List Json, jsonCharsSList a = a.map jsonCharsS
  | [] => rfl
  | j :: js => by rw [jsonCharsSList, jsonCharsSList_eq js]; rfl

theorem jsonCharsSObj_eq : ∀ t : Impl String (fun _ => Json), jsonCharsSObj t = t.toListModel.map memberCharsS
  | .leaf => rfl
  | .inner _ k v l r => by
    rw [jsonCharsSObj, jsonCharsSObj_eq l, jsonCharsSObj_eq r, Impl.toListModel_inner]
    simp [memberCharsS]

theorem plainSList_iff : ∀ a : List Json, plainSList a = true ↔ ∀ j ∈ a, plainS j = true
  | [] => by simp [plainSList]
  | j :: js => by simp [plainSList, plainSList_iff js]

theorem plainSObj_iff : ∀ t : Impl String (fun _ => Json),
    plainSObj t = true ↔ ∀ e ∈ t.toListModel, plainStr e.1 = true ∧ plainS e.2 = true
  | .leaf => by simp [plainSObj]
  | .inner _ k v l r => by
    rw [plainSObj, Impl.toListModel_inner]
    simp only [Bool.and_eq_true, plainSObj_iff l, plainSObj_iff r, List.mem_append, List.mem_cons]
    constructor
    · rintro ⟨⟨⟨hl, hk⟩, hv⟩, hr⟩ e (he | rfl | he)
      · exact hl e he
      · exact ⟨hk, hv⟩
      · exact hr e he
    · intro h
      exact ⟨⟨⟨fun e he => h e (.inl he), (h _ (.inr (.inl rfl))).1⟩, (h _ (.inr (.inl rfl))).2⟩,
        fun e he => h e (.inr (.inr he))⟩

/-! ## Numbers -/

theorem follow_stop {rest : List Char} (h : Follow rest) : FlockComplete.JsonRT.ReadBack.Stop rest := by
  rcases rest with _ | ⟨c, cs⟩
  · exact .inl rfl
  · exact .inr ⟨c, cs, rfl, h c cs rfl⟩

theorem repr_ofNat (n : Nat) : (Int.repr (Int.ofNat n)).toList = Nat.toDigits 10 n := by
  simp [Int.repr, Nat.repr]

theorem repr_negSucc (k : Nat) : (Int.repr (Int.negSucc k)).toList = '-' :: Nat.toDigits 10 (k + 1) := by
  simp [Int.repr, Nat.repr]

/-- A number's text starts with its sign or a digit. -/
theorem repr_head (m : Int) :
    ∃ c cs, (Int.repr m).toList = c :: cs ∧ (c = '-' ∨ ∃ d, d < 10 ∧ c = d.digitChar) := by
  cases m with
  | ofNat n =>
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · exact ⟨'0', [], by rw [repr_ofNat, Nat.toDigits_zero], .inr ⟨0, by decide, rfl⟩⟩
    · obtain ⟨d, cs, _, hd, he⟩ := toDigits_head n hn
      exact ⟨_, cs, by rw [repr_ofNat, he], .inr ⟨d, hd, rfl⟩⟩
  | negSucc k => exact ⟨'-', _, repr_negSucc k, .inl rfl⟩

theorem num_tests (c : Char) (h : c = '-' ∨ ∃ d, d < 10 ∧ c = d.digitChar) :
    ¬c = '[' ∧ ¬c = '{' ∧ ¬c = '"' ∧ ¬c = 'f' ∧ ¬c = 't' ∧ ¬c = 'n' ∧ (c = '-' ∨ ('0' ≤ c ∧ c ≤ '9')) := by
  rcases h with rfl | ⟨d, hd, rfl⟩
  · decide
  · exact digit_tests d hd

theorem value_numS (m : Int) (k : Nat) {rest : List Char} (hr : Follow rest) (o : Nat) :
    ∃ o', value (k + 1) ((Int.repr m).toList ++ rest) o = .ok rest o' (.num ⟨m, 0⟩) := by
  obtain ⟨o', ho⟩ := FlockComplete.JsonRT.ReadBack.num_int m rest (follow_stop hr) o
  refine ⟨o', ?_⟩
  obtain ⟨c, cs, he, hc⟩ := repr_head m
  rw [he] at ho ⊢
  simp only [List.cons_append] at ho ⊢
  obtain ⟨t1, t2, t3, t4, t5, t6, t7⟩ := num_tests c hc
  simp only [value, t1, t2, t3, t4, t5, t6, t7, ite_false, ite_true, ho, ws_follow hr]

/-! ## Where a value's text starts -/

theorem leadS_ne {c : Char} (h : Lead c ∨ c = '-') : ¬(c = '\t' ∨ c = '\n' ∨ c = '\r' ∨ c = ' ') ∧ c ≠ ']' := by
  rcases h with h | rfl
  · exact lead_ne h
  · decide

theorem jsonCharsS_head (j : Json) : ∃ c cs, jsonCharsS j = c :: cs ∧ (Lead c ∨ c = '-') := by
  match j with
  | .null => exact ⟨_, _, rfl, .inl (.inl rfl)⟩
  | .bool true => exact ⟨_, _, rfl, .inl (.inr (.inl rfl))⟩
  | .bool false => exact ⟨_, _, rfl, .inl (.inr (.inr (.inl rfl)))⟩
  | .num ⟨m, _⟩ =>
    obtain ⟨c, cs, he, hc⟩ := repr_head m
    refine ⟨c, cs, by rw [jsonCharsS]; exact he, ?_⟩
    rcases hc with rfl | ⟨d, hd, rfl⟩
    · exact .inr rfl
    · exact .inl (.inr (.inr (.inr (.inr (.inr (.inr ⟨d, hd, rfl⟩))))))
  | .str _ => exact ⟨_, _, by rw [jsonCharsS, List.cons_append], .inl (.inr (.inr (.inr (.inl rfl))))⟩
  | .arr ⟨_⟩ => exact ⟨_, _, by rw [jsonCharsS, List.cons_append], .inl (.inr (.inr (.inr (.inr (.inl rfl)))))⟩
  | .obj ⟨⟨_⟩⟩ =>
    exact ⟨_, _, by rw [jsonCharsS, List.cons_append], .inl (.inr (.inr (.inr (.inr (.inr (.inl rfl))))))⟩

theorem ws_charsS (j : Json) (r : List Char) (o : Nat) : ws (jsonCharsS j ++ r) o = (jsonCharsS j ++ r, o) := by
  obtain ⟨c, cs, he, hl⟩ := jsonCharsS_head j
  rw [he, List.cons_append, ws_cons _ _ (leadS_ne hl).1]

theorem ws_joinedS (j : Json) (l : List (List Char)) (r : List Char) (o : Nat) :
    ws ([','].intercalate (jsonCharsS j :: l) ++ r) o = ([','].intercalate (jsonCharsS j :: l) ++ r, o) := by
  obtain ⟨c, cs, he, hl⟩ := jsonCharsS_head j
  rw [he, List.intercalate_cons_cons_left, List.cons_append, ws_cons _ _ (leadS_ne hl).1]

theorem ws_membersS (e : (_ : String) × Json) (l : List (List Char)) (r : List Char) (o : Nat) :
    ws ([','].intercalate (memberCharsS e :: l) ++ r) o = ([','].intercalate (memberCharsS e :: l) ++ r, o) := by
  rw [show memberCharsS e = '"' :: (e.1.toList ++ ['"', ':'] ++ jsonCharsS e.2) by simp [memberCharsS],
    List.intercalate_cons_cons_left, List.cons_append, ws_cons _ _ (by decide)]

/-! ## `value` reads the text back -/

/-- `value` at fuel `k` reads each `plainS` value whose text is shorter than `k`, followed by what may follow a value, to
what the parser builds of it. -/
def ReadsS (k : Nat) : Prop :=
  ∀ j, plainS j = true → (jsonCharsS j).length < k → ∀ rest, Follow rest → ∀ o,
    ∃ o', value k (jsonCharsS j ++ rest) o = .ok rest o' (norm j)

theorem array_loopS {K : Nat} (IH : ∀ m, m < K → ReadsS m) : ∀ (l : List Json) (k : Nat) (acc : Array Json)
    (rest : List Char) (o : Nat), k ≤ K → l ≠ [] → (∀ e ∈ l, plainS e = true) →
    ([','].intercalate (l.map jsonCharsS) ++ [']']).length < k → Follow rest →
    ∃ o', array k acc ([','].intercalate (l.map jsonCharsS) ++ ']' :: rest) o = .ok rest o' (acc ++ (l.map norm).toArray)
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
    have he : [','].intercalate (jsonCharsS e :: jsonCharsS e' :: es.map jsonCharsS) ++ ']' :: rest =
        jsonCharsS e ++ (',' :: ([','].intercalate (jsonCharsS e' :: es.map jsonCharsS) ++ ']' :: rest)) := by simp
    obtain ⟨o1, h1⟩ := IH k (by omega) e (hp e List.mem_cons_self) (by omega)
      (',' :: ([','].intercalate (jsonCharsS e' :: es.map jsonCharsS) ++ ']' :: rest)) (follow_cons _ (by simp)) o
    have hl' : ([','].intercalate ((e' :: es).map jsonCharsS) ++ [']']).length < k := by
      simp only [List.map_cons, List.length_append, List.length_singleton]
      omega
    obtain ⟨o2, h2⟩ := array_loopS IH (e' :: es) k (acc.push (norm e)) rest (o1 + 1) (by omega) (by simp)
      (fun x hx => hp x (List.mem_cons_of_mem _ hx)) hl' hr
    refine ⟨o2, ?_⟩
    simp only [List.map_cons] at h2 ⊢
    rw [he]
    simp only [array, h1, show (',' = ']') = False by decide, ↓reduceIte, ws_joinedS, h2]
    simp

theorem object_loopS {K : Nat} (IH : ∀ m, m < K → ReadsS m) : ∀ (l : List ((_ : String) × Json)) (k : Nat)
    (kvs : Std.TreeMap.Raw String Json) (rest : List Char) (o : Nat), k ≤ K → l ≠ [] →
    (∀ e ∈ l, plainStr e.1 = true ∧ plainS e.2 = true) →
    ([','].intercalate (l.map memberCharsS) ++ ['}']).length < k → Follow rest →
    ∃ o', object k kvs ([','].intercalate (l.map memberCharsS) ++ '}' :: rest) o =
      .ok rest o' (l.foldl (fun a e => a.insert e.1 (norm e.2)) kvs)
  | [], _, _, _, _, _, h, _, _, _ => absurd rfl h
  | [e], k + 1, kvs, rest, o, hk, _, hp, hl, hr => by
    obtain ⟨hs, hv⟩ := hp e List.mem_cons_self
    simp only [List.map_cons, List.map_nil, List.intercalate_singleton] at hl ⊢
    have hlen := hl
    simp only [memberCharsS, List.length_append, List.length_cons, List.length_nil] at hlen
    obtain ⟨o1, h1⟩ := str_plain e.1.toList k "" (':' :: (jsonCharsS e.2 ++ '}' :: rest)) (o + 1) (plainStr_chars hs)
      (by omega)
    obtain ⟨o2, h2⟩ := IH k (by omega) e.2 hv (by omega) ('}' :: rest) (follow_cons _ (by simp)) (o1 + 1)
    refine ⟨o2 + 1, ?_⟩
    rw [show memberCharsS e ++ '}' :: rest = '"' :: (e.1.toList ++ '"' :: ':' :: (jsonCharsS e.2 ++ '}' :: rest)) by
      simp [memberCharsS]]
    simp only [object, h1, key_empty, ws_colon, ws_charsS, h2, ↓reduceIte, ws_follow hr, List.foldl_cons,
      List.foldl_nil]
  | e :: e' :: es, k + 1, kvs, rest, o, hk, _, hp, hl, hr => by
    obtain ⟨hs, hv⟩ := hp e List.mem_cons_self
    simp only [List.map_cons, List.intercalate_cons_cons] at hl ⊢
    have hlen := hl
    simp only [memberCharsS, List.length_append, List.length_cons, List.length_nil] at hlen
    let more := [','].intercalate (memberCharsS e' :: es.map memberCharsS) ++ '}' :: rest
    obtain ⟨o1, h1⟩ := str_plain e.1.toList k "" (':' :: (jsonCharsS e.2 ++ ',' :: more)) (o + 1) (plainStr_chars hs)
      (by omega)
    obtain ⟨o2, h2⟩ := IH k (by omega) e.2 hv (by omega) (',' :: more) (follow_cons _ (by simp)) (o1 + 1)
    have hl' : ([','].intercalate ((e' :: es).map memberCharsS) ++ ['}']).length < k := by
      simp only [List.map_cons, List.length_append, List.length_singleton] at hl ⊢
      omega
    obtain ⟨o3, h3⟩ := object_loopS IH (e' :: es) k (kvs.insert e.1 (norm e.2)) rest (o2 + 1) (by omega) (by simp)
      (fun x hx => hp x (List.mem_cons_of_mem _ hx)) hl' hr
    refine ⟨o3, ?_⟩
    rw [show memberCharsS e ++ [','] ++ [','].intercalate (memberCharsS e' :: es.map memberCharsS) ++ '}' :: rest =
      '"' :: (e.1.toList ++ '"' :: ':' :: (jsonCharsS e.2 ++ ',' :: more)) by simp [memberCharsS, more]]
    simp only [List.map_cons] at h3
    simp only [object, h1, key_empty, ws_colon, ws_charsS, h2, show (',' = '}') = False by decide, ↓reduceIte,
      more, ws_membersS, h3, List.foldl_cons]

/-- `value` reads a `plainS` value's text back at every fuel that exceeds its length. -/
theorem readsS : ∀ k, ReadsS k := by
  intro k
  induction k using Nat.strongRecOn with
  | ind k IH =>
    intro j hp hl rest hr o
    cases k with
    | zero => exact absurd hl (Nat.not_lt_zero _)
    | succ k =>
    match j with
    | .null => exact ⟨_, by simp [jsonCharsS, value, word, lit, ws_follow hr, norm] <;> rfl⟩
    | .bool true => exact ⟨_, by simp [jsonCharsS, value, word, lit, ws_follow hr, norm] <;> rfl⟩
    | .bool false => exact ⟨_, by simp [jsonCharsS, value, word, lit, ws_follow hr, norm] <;> rfl⟩
    | .num ⟨m, e⟩ =>
      simp only [plainS, beq_iff_eq] at hp
      subst hp
      obtain ⟨o', ho⟩ := value_numS m k hr o
      exact ⟨o', by rw [jsonCharsS, ho, norm]⟩
    | .str s =>
      simp only [plainS] at hp
      simp only [jsonCharsS, List.length_append, List.length_cons] at hl
      obtain ⟨o1, h1⟩ := str_plain s.toList k "" rest (o + 1) (plainStr_chars hp) (by omega)
      refine ⟨o1, ?_⟩
      simp only [jsonCharsS, List.cons_append, List.append_assoc]
      simp [value, h1, ws_follow hr, norm]
    | .arr ⟨a⟩ =>
      simp only [plainS, plainSList_iff] at hp
      simp only [jsonCharsS, jsonCharsSList_eq] at hl ⊢
      cases a with
      | nil =>
        refine ⟨o + 1 + 1, ?_⟩
        simp [value, ws_cons _ _ (show ¬(']' = '\t' ∨ ']' = '\n' ∨ ']' = '\r' ∨ ']' = ' ') by decide),
          ws_follow hr, norm, normList]
      | cons e es =>
        have hl' : ([','].intercalate ((e :: es).map jsonCharsS) ++ [']']).length < k := by
          simp only [List.length_cons, List.length_append] at hl ⊢
          omega
        obtain ⟨o', ho⟩ := array_loopS IH (e :: es) k #[] rest (o + 1) (by omega) (by simp) hp hl' hr
        refine ⟨o', ?_⟩
        obtain ⟨c, cs, hc, hlead⟩ := jsonCharsS_head e
        have hne := (leadS_ne hlead).2
        simp only [List.map_cons, hc, List.intercalate_cons_cons_left] at ho ⊢
        simp only [List.cons_append, List.append_assoc, List.nil_append] at ho ⊢
        simp only [value, ↓reduceIte, ws_cons _ _ (leadS_ne hlead).1]
        split
        · rename_i h; simp at h
        · rename_i h; simp only [Prod.mk.injEq, List.cons.injEq] at h; exact absurd h.1.1 hne
        · rename_i heq
          simp only [Prod.mk.injEq] at heq
          obtain ⟨rfl, rfl⟩ := heq
          simp [ho, norm, normList_eq]
    | .obj ⟨⟨t⟩⟩ =>
      simp only [plainS, Bool.and_eq_true, plainSObj_iff] at hp
      simp only [jsonCharsS, jsonCharsSObj_eq] at hl ⊢
      rw [norm, normObj_eq]
      cases ht : t.toListModel with
      | nil =>
        refine ⟨o + 1 + 1, ?_⟩
        simp [value, ws_cons _ _ (show ¬('}' = '\t' ∨ '}' = '\n' ∨ '}' = '\r' ∨ '}' = ' ') by decide),
          ws_follow hr]
      | cons e es =>
        rw [ht] at hl hp
        have hl' : ([','].intercalate ((e :: es).map memberCharsS) ++ ['}']).length < k := by
          simp only [List.length_cons, List.length_append] at hl ⊢
          omega
        obtain ⟨o', ho⟩ := object_loopS IH (e :: es) k ∅ rest (o + 1) (by omega) (by simp) hp.1 hl' hr
        refine ⟨o', ?_⟩
        simp only [List.map_cons, show memberCharsS e = '"' :: (e.1.toList ++ ['"', ':'] ++ jsonCharsS e.2) by
          simp [memberCharsS], List.intercalate_cons_cons_left] at ho ⊢
        simp only [List.cons_append, List.append_assoc, List.nil_append, List.foldl_cons] at ho ⊢
        simp [value, ws_cons _ _ (show ¬('"' = '\t' ∨ '"' = '\n' ∨ '"' = '\r' ∨ '"' = ' ') by decide), ho]

/-- **A `plainS` value's text parses back** to what the parser builds of it (`norm`). -/
theorem parse_jsonCharsS {j : Json} (h : plainS j = true) :
    JsonParse.parse (String.ofList (jsonCharsS j)) = .ok (norm j) := by
  rw [parse_eq, String.toList_ofList]
  obtain ⟨o', ho⟩ := readsS (2 * (jsonCharsS j).length + 2) j h (by omega) [] follow_nil 0
  rw [List.append_nil] at ho
  have hws : ws (jsonCharsS j) 0 = (jsonCharsS j, 0) := by simpa using ws_charsS j [] 0
  simp only [parseList, hws, ho]

end Verity.Sampled.Instance.HonestSession.SignedJson
