module

public import Flock.Net
public import Flock.HmNets
public import VStar.Unit
public import Mathlib
import all Init.Data.String.Legacy
import all Init.Data.String.Defs
import all Init.Data.Repr
import all Init.Data.Int.Repr
import Std.Data.String.ToNat

/-!
# Text: lines, tokens, and strings without a newline or whitespace

What the proofs about a circuit file's text read of `String`. The legacy `String.splitOn "\n"` is `List.splitOn '\n'`
on the characters (`splitOn_nl`, as at any one-byte separator: `splitOn_sep`), so a first line without a newline splits
off (`splitOn_line`). `Flock.tokens` is a fold of `tstep` over the characters (`tokens_eq`), which a word without
whitespace passes through whole (`foldl_word`). `NoNl` and `NoWs` say a string holds no `\n` and no whitespace:
appending, joining, folding and writing a number keep them (`noNl_append`, `noNl_intercalate`, `foldl_noNl`, `nat_noNl`,
`nat_noWs`), and a net's header line has no newline when its name has none (`header_noNl`).

This module imports the verifier's code (`Flock.Net`, `Flock.HmNets`, `VStar.Unit`) and Mathlib, and nothing of the
analysis: the modules that build the concrete statement's file from these (`Claims/Recursion/VStarNets.lean`,
`SessionLoop/ConcreteNet.lean`, `SessionLoop/Concrete/File.lean`) are then rebuilt only when the verifier's code or one
of them changes. `Family/Names.lean` holds what the unit's name needs beyond these.
-/

@[expose] public section

namespace FlockVBridge.Names

/-! ## `String.splitOn "\n"` -/

/-- The UTF-8 bytes of a character list. -/
def ulen (l : List Char) : ℕ := (l.map Char.utf8Size).sum

theorem ulen_append (a b : List Char) : ulen (a ++ b) = ulen a + ulen b := by
  simp [ulen]

theorem ulen_cons (c : Char) (l : List Char) : ulen (c :: l) = c.utf8Size + ulen l := by
  simp [ulen]

theorem utf8ByteSize_eq_ulen (s : String) : s.utf8ByteSize = ulen s.toList := by
  rw [← String.ofList_toList (s := s)]
  generalize s.toList = l
  induction l with
  | nil => rfl
  | cons c l ih =>
    rw [String.ofList_cons, String.utf8ByteSize_append, String.utf8ByteSize_singleton, ih]
    simp [ulen]

theorem getAux_at (xs ys : List Char) (c : Char) (i : String.Pos.Raw) :
    String.Pos.Raw.utf8GetAux (xs ++ c :: ys) i ⟨i.byteIdx + ulen xs⟩ = c := by
  induction xs generalizing i with
  | nil => simp only [List.nil_append, String.Pos.Raw.utf8GetAux, ulen, List.map_nil, List.sum_nil, Nat.add_zero,
      ↓reduceIte]
  | cons x xs ih =>
    rw [List.cons_append, String.Pos.Raw.utf8GetAux, if_neg]
    · have := ih (i + x)
      rwa [String.Pos.Raw.byteIdx_add_char, Nat.add_assoc, ← ulen_cons] at this
    · intro h
      have := congrArg String.Pos.Raw.byteIdx h
      simp only [ulen_cons] at this
      have := x.utf8Size_pos
      omega

theorem go₂_at (mid post : List Char) (j : ℕ) :
    String.Pos.Raw.extract.go₂ (mid ++ post) ⟨j⟩ ⟨j + ulen mid⟩ = mid := by
  induction mid generalizing j with
  | nil =>
    cases post with
    | nil => rfl
    | cons c cs => simp [String.Pos.Raw.extract.go₂, ulen]
  | cons x xs ih =>
    rw [List.cons_append, String.Pos.Raw.extract.go₂, if_neg]
    · have := ih (j + x.utf8Size)
      rw [show j + x.utf8Size + ulen xs = j + ulen (x :: xs) by rw [ulen_cons]; omega] at this
      rw [String.Pos.Raw.add_char_eq]
      exact congrArg _ this
    · intro h
      have := congrArg String.Pos.Raw.byteIdx h
      simp only [ulen_cons] at this
      have := x.utf8Size_pos
      omega

theorem go₁_at (pre rest : List Char) (k : ℕ) (e : String.Pos.Raw) :
    String.Pos.Raw.extract.go₁ (pre ++ rest) ⟨k⟩ ⟨k + ulen pre⟩ e = String.Pos.Raw.extract.go₂ rest ⟨k + ulen pre⟩ e := by
  induction pre generalizing k with
  | nil =>
    cases rest with
    | nil => rfl
    | cons c cs => simp [String.Pos.Raw.extract.go₁, ulen]
  | cons x xs ih =>
    rw [List.cons_append, String.Pos.Raw.extract.go₁, if_neg]
    · rw [String.Pos.Raw.add_char_eq]
      have := ih (k + x.utf8Size)
      rwa [show k + x.utf8Size + ulen xs = k + ulen (x :: xs) by rw [ulen_cons]; omega] at this
    · intro h
      have := congrArg String.Pos.Raw.byteIdx h
      simp only [ulen_cons] at this
      have := x.utf8Size_pos
      omega

theorem extract_at (s : String) (pre mid post : List Char) (hs : s.toList = pre ++ mid ++ post) :
    String.Pos.Raw.extract s ⟨ulen pre⟩ ⟨ulen (pre ++ mid)⟩ = String.ofList mid := by
  unfold String.Pos.Raw.extract
  split
  rename_i s' _ _ _
  split
  · rename_i h
    simp only [ulen_append] at h
    cases mid with
    | nil => rfl
    | cons c cs => simp only [ulen_cons] at h; have := c.utf8Size_pos; omega
  · rw [hs, List.append_assoc]
    have h1 := go₁_at pre (mid ++ post) 0 ⟨ulen (pre ++ mid)⟩
    simp only [Nat.zero_add] at h1
    rw [show (0 : String.Pos.Raw) = ⟨0⟩ from rfl, h1, ulen_append, go₂_at]

/-- The legacy `String.splitOnAux` at a one-byte separator `sep`, the character `ch`, inside a field `mid` that holds no
`ch`: the fields before it, then `List.splitOn ch` of the rest. -/
theorem splitOnAux_sep (sep : String) (ch : Char) (hg : String.Pos.Raw.get sep 0 = ch)
    (hn : String.Pos.Raw.next sep 0 = ⟨1⟩) (he : String.Pos.Raw.atEnd sep ⟨1⟩ = true) (hz : ch.utf8Size = 1)
    (s : String) (pre mid post : List Char) (hs : s.toList = pre ++ mid ++ post) (hmid : ch ∉ mid) (r : List String) :
    String.splitOnAux s sep ⟨ulen pre⟩ ⟨ulen (pre ++ mid)⟩ 0 r =
      r.reverse ++ ((mid ++ post).splitOn ch).map String.ofList := by
  induction post generalizing pre mid r with
  | nil =>
    rw [String.splitOnAux, if_pos]
    · rw [extract_at s pre mid [] hs, List.reverse_cons, List.append_nil, List.splitOn_eq_splitOnP,
        List.splitOnP_eq_singleton]
      · rfl
      · intro x hx
        simp only [beq_eq_false_iff_ne, ne_eq]
        rintro rfl
        exact hmid hx
    · simp only [String.Pos.Raw.atEnd, utf8ByteSize_eq_ulen, hs, List.append_nil, ge_iff_le, decide_eq_true_eq]
      exact Nat.le_refl _
  | cons c post ih =>
    have hget : String.Pos.Raw.get s ⟨ulen (pre ++ mid)⟩ = c := by
      unfold String.Pos.Raw.get
      rw [hs, List.append_assoc]
      have := getAux_at (pre ++ mid) post c 0
      rw [show (0 : String.Pos.Raw).byteIdx + ulen (pre ++ mid) = ulen (pre ++ mid) from Nat.zero_add _] at this
      rw [← List.append_assoc]
      exact this
    have hnot : ¬ (String.Pos.Raw.atEnd s ⟨ulen (pre ++ mid)⟩ = true) := by
      simp only [String.Pos.Raw.atEnd, utf8ByteSize_eq_ulen, hs, ge_iff_le, decide_eq_true_eq, ulen_append, ulen_cons]
      have := c.utf8Size_pos
      omega
    rw [String.splitOnAux, if_neg hnot]
    have hmid' : ∀ x ∈ mid, (x == ch) = false := by
      intro x hx
      simp only [beq_eq_false_iff_ne, ne_eq]
      rintro rfl
      exact hmid hx
    by_cases hc : c = ch
    · subst hc
      rw [if_pos (by rw [hget, hg]; exact beq_self_eq_true c)]
      dsimp only
      rw [hn, if_pos he]
      have hnext : String.Pos.Raw.next s ⟨ulen (pre ++ mid)⟩ = ⟨ulen (pre ++ mid ++ [c])⟩ := by
        unfold String.Pos.Raw.next
        rw [hget, String.Pos.Raw.add_char_eq, ulen_append (pre ++ mid), ulen_cons, hz]
        rfl
      have hun : String.Pos.Raw.unoffsetBy ⟨ulen (pre ++ mid ++ [c])⟩ ⟨1⟩ = ⟨ulen (pre ++ mid)⟩ := by
        show (⟨ulen (pre ++ mid ++ [c]) - 1⟩ : String.Pos.Raw) = _
        rw [ulen_append (pre ++ mid), ulen_cons, hz]
        rfl
      rw [hnext, hun, extract_at s pre mid _ hs]
      have := ih (pre ++ mid ++ [c]) [] (by rw [hs]; simp) (by simp) (String.ofList mid :: r)
      rw [List.append_nil] at this
      rw [this, List.nil_append, show (mid ++ c :: post).splitOn c = mid :: post.splitOn c by
        simp only [List.splitOn_eq_splitOnP]; exact List.splitOnP_append_cons_of_forall_mem hmid' c (by simp) post]
      simp
    · rw [if_neg (by rw [hget, hg, beq_iff_eq]; exact hc)]
      have hnext : (String.Pos.Raw.unoffsetBy ⟨ulen (pre ++ mid)⟩ 0).next s = ⟨ulen (pre ++ (mid ++ [c]))⟩ := by
        show String.Pos.Raw.next s ⟨ulen (pre ++ mid) - 0⟩ = _
        rw [Nat.sub_zero]
        unfold String.Pos.Raw.next
        rw [hget, String.Pos.Raw.add_char_eq, ← List.append_assoc, ulen_append (pre ++ mid), ulen_cons]
        simp [ulen]
      rw [hnext, ih pre (mid ++ [c]) (by rw [hs]; simp) (by
        simp only [List.mem_append, List.mem_singleton, not_or]; exact ⟨hmid, fun h => hc h.symm⟩) r]
      simp

/-- **The legacy `String.splitOn` at a one-byte separator is `List.splitOn` of its character on the characters.** -/
theorem splitOn_sep (sep : String) (ch : Char) (hg : String.Pos.Raw.get sep 0 = ch)
    (hn : String.Pos.Raw.next sep 0 = ⟨1⟩) (he : String.Pos.Raw.atEnd sep ⟨1⟩ = true) (hz : ch.utf8Size = 1)
    (hne : sep ≠ "") (s : String) : s.splitOn sep = (s.toList.splitOn ch).map String.ofList := by
  unfold String.splitOn
  rw [if_neg (by simpa using hne)]
  exact splitOnAux_sep sep ch hg hn he hz s [] [] s.toList (by simp) (by simp) []

/-- **The legacy `String.splitOn "\n"` is `List.splitOn '\n'` on the characters.** -/
theorem splitOn_nl (s : String) : s.splitOn "\n" = (s.toList.splitOn '\n').map String.ofList :=
  splitOn_sep "\n" '\n' rfl rfl rfl rfl (by decide) s

theorem fromUTF8?_toUTF8 (s : String) : String.fromUTF8? s.toUTF8 = some s := by
  unfold String.fromUTF8?
  rw [dif_pos (by rw [String.toUTF8_eq_toByteArray]; exact s.isValidUTF8)]
  rfl

/-! ## `Flock.tokens` -/

/-- `tokens`' loop step: a whitespace character ends the current word, any other extends it. -/
def tstep (p : Array String × String) (c : Char) : Array String × String :=
  if c.isWhitespace then (if p.2 != "" then (p.1.push p.2, "") else p) else (p.1, p.2.push c)

theorem tokens_eq (s : String) :
    Flock.tokens s = (let p := s.toList.foldl tstep (#[], ""); if p.2 != "" then p.1.push p.2 else p.1) := by
  have e : ∀ (l : List Char) (init : Array String × String),
      (forIn (m := Id) l init fun c p =>
        if c.isWhitespace = true then
          if (p.snd != "") = true then pure (ForInStep.yield (p.fst.push p.snd, ""))
          else pure (ForInStep.yield (p.fst, p.snd))
        else pure (ForInStep.yield (p.fst, p.snd.push c))) = l.foldl tstep init := by
    intro l
    induction l with
    | nil => intro; rfl
    | cons c l ih =>
      intro init
      rw [List.forIn_cons, List.foldl_cons, ← ih]
      unfold tstep
      split <;> (try split) <;> rfl
  unfold Flock.tokens
  simp only [Id.run]
  rw [e]
  rfl

/-- No character of `s` is whitespace. -/
def NoWs (s : String) : Prop := ∀ c ∈ s.toList, c.isWhitespace = false

theorem foldl_word (l : List Char) (hl : ∀ c ∈ l, c.isWhitespace = false) (o : Array String) (cur : String) :
    l.foldl tstep (o, cur) = (o, cur ++ String.ofList l) := by
  induction l generalizing cur with
  | nil => simp
  | cons c l ih =>
    rw [List.foldl_cons]
    have hc := hl c List.mem_cons_self
    rw [show tstep (o, cur) c = (o, cur.push c) by unfold tstep; rw [if_neg (by rw [hc]; decide)]]
    rw [ih (fun x hx => hl x (List.mem_cons_of_mem _ hx)), String.push_eq_append, String.ofList_cons,
      String.append_assoc]

open Flock

theorem getElem!_str (a : Array String) (i : ℕ) : a[i]! = a.getD i "" := by
  simp only [getElem!_def, Array.getD_eq_getD_getElem?]
  cases a[i]? <;> rfl

/-! ## Lines without a newline -/

/-- `s` holds no `\n`. -/
def NoNl (s : String) : Prop := '\n' ∉ s.toList

theorem noNl_append {a b : String} : NoNl (a ++ b) ↔ NoNl a ∧ NoNl b := by
  simp [NoNl, String.toList_append]

theorem noNl_intercalate {sep : String} (hsep : NoNl sep) :
    ∀ l : List String, (∀ x ∈ l, NoNl x) → NoNl (sep.intercalate l)
  | [], _ => by simp [NoNl]
  | [a], h => by simpa using h a (by simp)
  | a :: b :: l, h => by
    rw [String.intercalate_cons_cons]
    refine noNl_append.2 ⟨noNl_append.2 ⟨h a (by simp), hsep⟩, noNl_intercalate hsep (b :: l) fun x hx => h x ?_⟩
    exact List.mem_cons_of_mem _ hx

theorem nat_noNl (n : ℕ) : NoNl (toString n) := by
  intro h
  have : '\n'.isDigit = true := Nat.isDigit_of_mem_toDigits (b := 10) (n := n) (by decide) (by decide) (by
    rw [← Nat.toList_repr]; exact h)
  exact absurd this (by decide)

theorem foldl_noNl {β : Type} (g : String → β → String) (hg : ∀ o c, NoNl o → NoNl (g o c)) :
    ∀ (l : List β) o, NoNl o → NoNl (l.foldl g o)
  | [], _, ho => ho
  | c :: l, o, ho => foldl_noNl g hg l _ (hg o c ho)

/-! ## Words without whitespace -/

theorem digit_noWs {c : Char} (h : c.isDigit = true) : c.isWhitespace = false := by
  simp only [Char.isWhitespace, Bool.or_eq_false_iff, decide_eq_false_iff_not]
  refine ⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩ <;> (rintro rfl; exact absurd h (by decide))

theorem nat_noWs (n : ℕ) : NoWs (toString n) := fun _ hc =>
  digit_noWs (Nat.isDigit_of_mem_toDigits (b := 10) (n := n) (by decide) (by decide) (by
    rw [← Nat.toList_repr]; exact hc))

/-! ## A net's header line, and a first line -/

theorem header_noNl (name : String) (l : Flock.HmNets.Laid) (h : NoNl name) : NoNl (VStar.header name l) := by
  unfold VStar.header
  refine noNl_intercalate (by simp [NoNl]) _ fun x hx => ?_
  simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_append, List.mem_flatMap, List.mem_map] at hx
  rcases hx with rfl | rfl | rfl | rfl | rfl | ⟨g, -, n, -, rfl⟩ | rfl | ⟨g, -, n, -, rfl⟩
  · simp [NoNl, VStar.FORMAT]
  · exact h
  all_goals exact nat_noNl _

theorem splitOn_line (a rest : String) (ha : NoNl a) : (a ++ "\n" ++ rest).splitOn "\n" = a :: rest.splitOn "\n" := by
  rw [splitOn_nl, splitOn_nl, String.toList_append, String.toList_append, show "\n".toList = ['\n'] by decide,
    List.append_assoc, List.singleton_append, List.splitOn_eq_splitOnP,
    List.splitOnP_append_cons_of_forall_mem (fun x hx => by
      simp only [beq_eq_false_iff_ne, ne_eq]; rintro rfl; exact ha hx) '\n' rfl, List.map_cons,
    String.ofList_toList, List.splitOn_eq_splitOnP]

end FlockVBridge.Names
