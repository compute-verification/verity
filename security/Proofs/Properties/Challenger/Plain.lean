module

public import Proofs.Properties.Challenger.CanonLine
public import Spot.PyJson
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# Plain values: the two writers write them alike, and their text escapes nothing

A ledger line's reader (`Spot.Challenger.ledgerEntry?`) takes a line that is its parse's text under two writers,
`Spot.PyJson.canonical` (Python's `canonical_json`) and `Flock.canon`, and that escapes no surrogate. The two writers
differ only in what they escape and in the numbers that are no integers. A plain value has neither:

* **A plain character** (`PlainChar`) is printable ASCII but for the quote and the backslash, and a plain string
  (`PlainStr`) holds only those: each writer writes it between quotes as it is (`str_plain`, `canonStr_plain`).
* **A plain value** nested at most `d` deep (`PlainAt d`): its strings and its keys plain, its numbers integers, its
  objects well-formed trees. The two writers write it alike (`write_canon`, `canonical_canon`), and its text has no
  backslash (`noEscape_canon`).
* **A text with no backslash escapes no surrogate** (`escapesSurrogate_false`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private hex4 from Spot.PyJson
open private escapesSurrogate from Spot.ChallengerSession

namespace Properties.Challenger.Plain

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger

/-! ## Plain strings -/

/-- A character both writers write as itself. -/
def PlainChar (c : Char) : Prop := ' ' ≤ c ∧ c ≤ '~' ∧ c ≠ '"' ∧ c ≠ '\\'

/-- A string both writers write between quotes as it is. -/
def PlainStr (s : String) : Prop := ∀ c ∈ s.toList, PlainChar c

theorem plain_cases {c : Char} (h : PlainChar c) :
    c ≠ '"' ∧ c ≠ '\\' ∧ c ≠ '\n' ∧ c ≠ '\r' ∧ c ≠ '\t' ∧ c ≠ '\x08' ∧ c ≠ '\x0c' := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  refine ⟨h3, h4, ?_, ?_, ?_, ?_, ?_⟩ <;> rintro rfl <;> exact absurd h1 (by decide)

def pyEsc (c : Char) : String :=
  match c with
  | '"' => "\\\""
  | '\\' => "\\\\"
  | '\n' => "\\n"
  | '\r' => "\\r"
  | '\t' => "\\t"
  | '\x08' => "\\b"
  | '\x0c' => "\\f"
  | c =>
    if ' ' ≤ c && c ≤ '~' then String.singleton c
    else if c.val < 0x10000 then "\\u" ++ hex4 c.val.toNat
    else
      let v := c.val.toNat - 0x10000
      "\\u" ++ hex4 (0xD800 + v / 1024) ++ "\\u" ++ hex4 (0xDC00 + v % 1024)

theorem pyEsc_plain {c : Char} (h : PlainChar c) : pyEsc c = String.singleton c := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := plain_cases h
  obtain ⟨hlo, hhi, -, -⟩ := h
  unfold pyEsc
  split <;> first | contradiction | simp [hlo, hhi]

theorem str_fold (s : String) : PyJson.str s = s.toList.foldl (fun out c => out ++ pyEsc c) "\"" ++ "\"" := by
  unfold PyJson.str
  simp only [Id.run, List.forIn_pure_yield_eq_foldl, bind_pure_comp, map_pure, pyEsc]
  rfl

theorem fold_plain (l : List Char) (h : ∀ c ∈ l, PlainChar c) (out : String) :
    l.foldl (fun out c => out ++ pyEsc c) out = out ++ String.ofList l := by
  induction l generalizing out with
  | nil => simp
  | cons c l ih =>
    rw [List.foldl_cons, ih (fun d hd => h d (List.mem_cons_of_mem _ hd)), pyEsc_plain (h c List.mem_cons_self)]
    apply String.toList_inj.1
    simp

/-- The Python-form writer writes a plain string between quotes as it is. -/
theorem str_plain {s : String} (h : PlainStr s) : PyJson.str s = "\"" ++ s ++ "\"" := by
  rw [str_fold, fold_plain _ h, String.ofList_toList]

def canEsc (out : String) (c : Char) : String :=
  match c with
  | '"' => out ++ "\\\""
  | '\\' => out ++ "\\\\"
  | '\n' => out ++ "\\n"
  | '\r' => out ++ "\\r"
  | '\t' => out ++ "\\t"
  | c => if c.toNat == 8 then out ++ "\\b" else if c.toNat == 12 then out ++ "\\f"
         else if c.toNat < 0x20 || 0x7f ≤ c.toNat then out ++ Flock.escU c.toNat else out.push c

theorem plain_toNat {c : Char} (h : PlainChar c) : 32 ≤ c.toNat ∧ c.toNat ≤ 126 := by
  obtain ⟨hlo, hhi, -, -⟩ := h
  have h1 : (' ' : Char).val ≤ c.val := hlo
  have h2 : c.val ≤ ('~' : Char).val := hhi
  rw [UInt32.le_iff_toNat_le] at h1 h2
  exact ⟨h1, h2⟩

theorem canEsc_plain {c : Char} (h : PlainChar c) (out : String) : canEsc out c = out.push c := by
  obtain ⟨h1, h2, h3, h4, h5, -, -⟩ := plain_cases h
  obtain ⟨hlo, hhi⟩ := plain_toNat h
  have h8 : (c.toNat == 8) = false := by rw [beq_eq_false_iff_ne]; omega
  have h12 : (c.toNat == 12) = false := by rw [beq_eq_false_iff_ne]; omega
  have hr : (decide (c.toNat < 0x20) || decide (0x7f ≤ c.toNat)) = false := by
    rw [Bool.or_eq_false_iff, decide_eq_false_iff_not, decide_eq_false_iff_not]; omega
  unfold canEsc
  split <;> first | contradiction | simp only [h8, h12, hr, Bool.false_eq_true, ↓reduceIte]

theorem canonStr_fold (s : String) : Flock.canonStr s = s.toList.foldl canEsc "\"" ++ "\"" := by
  unfold Flock.canonStr
  simp only [Id.run, List.forIn_pure_yield_eq_foldl, bind_pure_comp, map_pure]
  rfl

theorem canFold_plain (l : List Char) (h : ∀ c ∈ l, PlainChar c) (out : String) :
    l.foldl canEsc out = out ++ String.ofList l := by
  induction l generalizing out with
  | nil => simp
  | cons c l ih =>
    rw [List.foldl_cons, ih (fun d hd => h d (List.mem_cons_of_mem _ hd)), canEsc_plain (h c List.mem_cons_self)]
    apply String.toList_inj.1
    simp

/-- `Flock.canon` writes a plain string between quotes as it is. -/
theorem canonStr_plain {s : String} (h : PlainStr s) : Flock.canonStr s = "\"" ++ s ++ "\"" := by
  rw [canonStr_fold, canFold_plain _ h, String.ofList_toList]

/-- The two writers write a plain string alike. -/
theorem str_canonStr {s : String} (h : PlainStr s) : PyJson.str s = Flock.canonStr s := by
  rw [str_plain h, canonStr_plain h]

/-! ## Plain values -/

/-- A value nested at most `d` deep whose strings and keys are plain, whose numbers are integers and whose objects are
well-formed trees: one the two writers write alike. -/
inductive PlainAt : Nat → Json → Prop
  | null (d : Nat) : PlainAt (d + 1) .null
  | bool (d : Nat) (b : Bool) : PlainAt (d + 1) (.bool b)
  | num (d : Nat) (m : Int) : PlainAt (d + 1) (.num ⟨m, 0⟩)
  | str (d : Nat) {s : String} (h : PlainStr s) : PlainAt (d + 1) (.str s)
  | arr (d : Nat) {a : Array Json} (h : ∀ x ∈ a, PlainAt d x) : PlainAt (d + 1) (.arr a)
  | obj (d : Nat) {m : Std.TreeMap.Raw String Json} (hm : m.WF) (hk : ∀ p ∈ m.toArray, PlainStr p.1)
      (h : ∀ p ∈ m.toArray, PlainAt d p.2) : PlainAt (d + 1) (.obj m)

theorem mapM_ext {α β : Type} {f g : α → Flock.V β} :
    ∀ {l : List α}, (∀ x ∈ l, f x = g x) → l.mapM f = l.mapM g
  | [], _ => rfl
  | x :: l, h => by
    rw [List.mapM_cons, List.mapM_cons, h x List.mem_cons_self,
      mapM_ext fun y hy => h y (List.mem_cons_of_mem _ hy)]

/-- A printer that is another's results under `h` prints a list as the other's results under `h`. -/
theorem mapM_map {α β γ : Type} {f : α → Flock.V β} {g : α → Flock.V γ} {h : β → γ} :
    ∀ {l : List α}, (∀ x ∈ l, g x = h <$> f x) → l.mapM g = List.map h <$> l.mapM f
  | [], _ => rfl
  | x :: l, hx => by
    rw [List.mapM_cons, List.mapM_cons, hx x List.mem_cons_self,
      mapM_map fun y hy => hx y (List.mem_cons_of_mem _ hy)]
    cases f x with
    | error e => rfl
    | ok b => cases l.mapM f <;> rfl

theorem mapM_keys {f : Json → Flock.V String} : ∀ {l : List (String × Json)} {vs : List (String × String)},
    l.mapM (fun p => do pure (p.1, ← f p.2)) = .ok vs → vs.map (·.1) = l.map (·.1)
  | [], vs, h => by cases h; rfl
  | p :: l, vs, h => by
    rw [List.mapM_cons] at h
    cases hp : f p.2 with
    | error e => rw [hp] at h; cases h
    | ok t =>
      rw [hp] at h
      cases hl : l.mapM (fun p => do pure (p.1, ← f p.2)) with
      | error e => rw [hl] at h; cases h
      | ok ws =>
        rw [hl] at h
        cases h
        rw [List.map_cons, List.map_cons, mapM_keys hl]

/-- **The two writers write a plain value alike.** -/
theorem write_canon {d : Nat} {j : Json} (h : PlainAt d j) : PyJson.write d j = Flock.canon j := by
  induction h with
  | null d => rw [PyJson.write, Flock.canon]
  | bool d b => rw [PyJson.write, Flock.canon]
  | num d m => rw [PyJson.write, Flock.canon, PyJson.number]; rfl
  | str d h => rw [PyJson.write, Flock.canon, str_canonStr h]
  | arr d h ih =>
    rename_i a
    rw [PyJson.write, CanonLine.canon_arr, Array.mapM_eq_mapM_toList,
      mapM_ext (f := PyJson.write d) (g := Flock.canon) fun x hx => ih x (Array.mem_toList_iff.1 hx)]
    cases a.toList.mapM Flock.canon <;> rfl
  | obj d hm hk h ih =>
    rename_i m
    have hw : m.toArray.toList.mapM (fun (k, v) => do return PyJson.str k ++ ":" ++ (← PyJson.write d v)) =
        List.map (fun kv : String × String => Flock.canonStr kv.1 ++ ":" ++ kv.2) <$>
          m.toArray.toList.mapM (fun p => do pure (p.1, ← Flock.canon p.2)) :=
      mapM_map fun p hp => by
        have hp' := Array.mem_toList_iff.1 hp
        obtain ⟨k, v⟩ := p
        show (do return PyJson.str k ++ ":" ++ (← PyJson.write d v)) = _
        rw [ih _ hp', str_canonStr (hk _ hp')]
        cases Flock.canon v <;> rfl
    rw [PyJson.write, CanonLine.canon_obj, hw, Array.mapM_eq_mapM_toList]
    cases hv : m.toArray.toList.mapM (fun p => do pure (p.1, ← Flock.canon p.2)) with
    | error e => rfl
    | ok vs =>
      have hs : (vs.toArray.toList.map (·.1)).Pairwise (· < ·) := by
        rw [mapM_keys hv, Std.TreeMap.Raw.toList_toArray, List.pairwise_map]
        exact (Std.TreeMap.Raw.ordered_keys_toList hm).imp fun hab => compareOfLessAndEq_eq_lt.1 hab
      have hq : vs.toArray.qsort (fun a b => a.1 < b.1) = vs.toArray := CanonRead.qsort_keys hs
      show _ = Except.ok ("{" ++ ",".intercalate
        ((vs.toArray.qsort (fun a b => a.1 < b.1)).toList.map fun kv => Flock.canonStr kv.1 ++ ":" ++ kv.2) ++ "}")
      rw [hq]
      rfl

/-- **`PyJson.canonical` writes a plain value as `Flock.canon` does.** -/
theorem canonical_canon {j : Json} (h : PlainAt PyJson.DEPTH j) : PyJson.canonical j = Flock.canon j :=
  write_canon h

theorem mem_intercalate {c : Char} {sep : List Char} : ∀ {ls : List (List Char)}, c ∈ sep.intercalate ls →
    c ∈ sep ∨ ∃ l ∈ ls, c ∈ l
  | [], h => by simp at h
  | [l], h => by rw [List.intercalate_singleton] at h; exact .inr ⟨l, List.mem_cons_self, h⟩
  | l :: l' :: ls, h => by
    rw [List.intercalate_cons_cons, List.mem_append, List.mem_append] at h
    rcases h with (h | h) | h
    · exact .inr ⟨l, List.mem_cons_self, h⟩
    · exact .inl h
    · rcases mem_intercalate h with h | ⟨x, hx, hc⟩
      · exact .inl h
      · exact .inr ⟨x, List.mem_cons_of_mem _ hx, hc⟩

theorem mapM_ok {α β : Type} {f : α → Flock.V β} : ∀ {l : List α} {ts : List β}, l.mapM f = .ok ts →
    ∀ t ∈ ts, ∃ x ∈ l, f x = .ok t
  | [], ts, h => by cases h; simp
  | x :: l, ts, h => by
    rw [List.mapM_cons] at h
    cases hx : f x with
    | error e => rw [hx] at h; cases h
    | ok b =>
      rw [hx] at h
      cases hl : l.mapM f with
      | error e => rw [hl] at h; cases h
      | ok bs =>
        rw [hl] at h
        cases h
        intro t ht
        rcases List.mem_cons.1 ht with rfl | ht
        · exact ⟨x, List.mem_cons_self, hx⟩
        · obtain ⟨y, hy, hf⟩ := mapM_ok hl t ht
          exact ⟨y, List.mem_cons_of_mem _ hy, hf⟩

/-! ## A plain value's text -/

/-- A text with no backslash. -/
def NoEscape (t : String) : Prop := ∀ c ∈ t.toList, c ≠ '\\'

theorem noEscape_append {a b : String} (ha : NoEscape a) (hb : NoEscape b) : NoEscape (a ++ b) := fun c hc => by
  rw [String.toList_append, List.mem_append] at hc
  exact hc.elim (ha c) (hb c)

theorem noEscape_intercalate {ts : List String} (h : ∀ t ∈ ts, NoEscape t) : NoEscape (",".intercalate ts) :=
  fun c hc => by
    rw [String.toList_intercalate] at hc
    rcases mem_intercalate hc with hc | ⟨l, hl, hc⟩
    · rintro rfl; simp at hc
    · obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hl
      exact h t ht c hc

theorem noEscape_str {s : String} (h : PlainStr s) : NoEscape (Flock.canonStr s) := by
  rw [canonStr_plain h]
  refine noEscape_append (noEscape_append (fun c hc => ?_) fun c hc => (h c hc).2.2.2) fun c hc => ?_ <;>
    · rintro rfl; simp at hc

/-- **A plain value's text has no backslash.** -/
theorem noEscape_canon {d : Nat} {j : Json} (h : PlainAt d j) :
    ∀ {t : String}, Flock.canon j = .ok t → NoEscape t := by
  induction h with
  | null d => intro t ht; rw [Flock.canon] at ht; cases ht; intro c hc; rintro rfl; simp at hc
  | bool d b => intro t ht; rw [Flock.canon] at ht; cases ht; cases b <;> (intro c hc; rintro rfl; simp at hc)
  | num d m => intro t ht; rw [Flock.canon] at ht; cases ht; exact CanonLine.int_plain m
  | str d h => intro t ht; rw [Flock.canon] at ht; cases ht; exact noEscape_str h
  | arr d h ih =>
    rename_i a
    intro t ht
    rw [CanonLine.canon_arr, Array.mapM_eq_mapM_toList] at ht
    cases hp : a.toList.mapM Flock.canon with
    | error e => rw [hp] at ht; cases ht
    | ok ts =>
      rw [hp] at ht
      cases ht
      refine noEscape_append (noEscape_append (fun c hc => by rintro rfl; simp at hc) (noEscape_intercalate ?_))
        fun c hc => by rintro rfl; simp at hc
      intro t ht
      obtain ⟨x, hx, hf⟩ := mapM_ok hp t ht
      exact ih x (Array.mem_toList_iff.1 hx) hf
  | obj d hm hk h ih =>
    rename_i m
    intro t ht
    rw [CanonLine.canon_obj, Array.mapM_eq_mapM_toList] at ht
    cases hv : m.toArray.toList.mapM (fun p => do pure (p.1, ← Flock.canon p.2)) with
    | error e => rw [hv] at ht; cases ht
    | ok vs =>
      rw [hv] at ht
      have hs : (vs.toArray.toList.map (·.1)).Pairwise (· < ·) := by
        rw [mapM_keys hv, Std.TreeMap.Raw.toList_toArray, List.pairwise_map]
        exact (Std.TreeMap.Raw.ordered_keys_toList hm).imp fun hab => compareOfLessAndEq_eq_lt.1 hab
      have hq : vs.toArray.qsort (fun a b => a.1 < b.1) = vs.toArray := CanonRead.qsort_keys hs
      have ht' : Except.ok ("{" ++ ",".intercalate
          ((vs.toArray.qsort (fun a b => a.1 < b.1)).toList.map fun kv => Flock.canonStr kv.1 ++ ":" ++ kv.2) ++ "}") =
          Except.ok (ε := String) t := ht
      rw [hq] at ht'
      cases ht'
      refine noEscape_append (noEscape_append (fun c hc => by rintro rfl; simp at hc) (noEscape_intercalate ?_))
        fun c hc => by rintro rfl; simp at hc
      intro t ht
      obtain ⟨kv, hkv, rfl⟩ := List.mem_map.1 ht
      obtain ⟨p, hp, hf⟩ := mapM_ok hv kv hkv
      have hp' := Array.mem_toList_iff.1 hp
      cases hc : Flock.canon p.2 with
      | error e => rw [hc] at hf; cases hf
      | ok u =>
        rw [hc] at hf
        cases hf
        exact noEscape_append (noEscape_append (noEscape_str (hk p hp')) fun c hc => by rintro rfl; simp at hc)
          (ih p hp' hc)

/-! ## Surrogates -/

theorem forIn_none {cs : Array Char} (h : ∀ c ∈ cs, c ≠ '\\') : ∀ (l : List Nat), (∀ i ∈ l, i < cs.size) →
    (forIn (m := Id) l ((none : Option Bool), ()) fun i __s =>
      if (cs[i]! == '\\' && cs.getD (i + 1) ' ' == 'u' && cs.getD (i + 2) ' ' == 'd' &&
          "89abcdef".contains (cs.getD (i + 3) ' ')) = true then
        pure (ForInStep.done (some true, ()))
      else pure (ForInStep.yield (none, ()))) = pure (none, ())
  | [], _ => rfl
  | i :: l, hl => by
    have hi : i < cs.size := hl i List.mem_cons_self
    have hc : (cs[i]! == '\\') = false := by
      rw [getElem!_pos cs i hi, beq_eq_false_iff_ne]
      exact h _ (Array.getElem_mem hi)
    rw [List.forIn_cons]
    simp only [hc, Bool.false_and, Bool.false_eq_true, ↓reduceIte, pure_bind]
    exact forIn_none h l fun j hj => hl j (List.mem_cons_of_mem _ hj)

/-- A text with no backslash escapes no surrogate. -/
theorem escapesSurrogate_false {text : String} (h : NoEscape text) : escapesSurrogate text = false := by
  unfold escapesSurrogate
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range']
  rw [forIn_none (cs := text.toList.toArray) (fun c hc => h c (by simpa using hc))]
  · rfl
  · intro i hi
    simp only [List.mem_range'_1, Std.Legacy.Range.size] at hi
    omega

end Properties.Challenger.Plain
