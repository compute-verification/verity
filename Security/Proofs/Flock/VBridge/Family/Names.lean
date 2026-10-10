import VStar.Compose
import Proofs.Flock.Soundness.ExecCircuit

/-!
# The circuit file's unit name, emitted and parsed

`circuitFile` (`Family.lean`) writes `flock-circuit <unit>`, the `META` line, then `CIRCUIT unit <n>` and the unit's
net, whose header line is `flock-ir-unit/v2 <unit> …` (`VStar.emitNet`, `Unit.lean:94`). `HmRow.parse`
(`HmRow.lean:550`) splits the text at `\n`, takes the first `CIRCUIT unit` section's net (`Net.parse`, `Net.lean:200`),
and that net's name is the second token of its header line. This file shows the text side: the legacy
`String.splitOn "\n"` is `List.splitOn '\n'` on the characters, and `Flock.tokens` reads a line's second word.
-/

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

theorem splitOnAux_nl (s : String) (pre mid post : List Char) (hs : s.toList = pre ++ mid ++ post)
    (hmid : '\n' ∉ mid) (r : List String) :
    String.splitOnAux s "\n" ⟨ulen pre⟩ ⟨ulen (pre ++ mid)⟩ 0 r =
      r.reverse ++ ((mid ++ post).splitOn '\n').map String.ofList := by
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
    have hnl : String.Pos.Raw.get "\n" 0 = '\n' := rfl
    have hsz : '\n'.utf8Size = 1 := rfl
    have hmid' : ∀ x ∈ mid, (x == '\n') = false := by
      intro x hx
      simp only [beq_eq_false_iff_ne, ne_eq]
      rintro rfl
      exact hmid hx
    by_cases hc : c = '\n'
    · subst hc
      rw [if_pos (by rw [hget, hnl]; rfl)]
      dsimp only
      have hend : String.Pos.Raw.atEnd "\n" (String.Pos.Raw.next "\n" 0) = true := rfl
      rw [if_pos hend]
      have hnext : String.Pos.Raw.next s ⟨ulen (pre ++ mid)⟩ = ⟨ulen (pre ++ mid ++ ['\n'])⟩ := by
        unfold String.Pos.Raw.next
        rw [hget, String.Pos.Raw.add_char_eq, ulen_append (pre ++ mid), ulen_cons, hsz]
        rfl
      have hun : (String.Pos.Raw.unoffsetBy ⟨ulen (pre ++ mid ++ ['\n'])⟩ (String.Pos.Raw.next "\n" 0)) =
          ⟨ulen (pre ++ mid)⟩ := by
        have : String.Pos.Raw.next "\n" 0 = ⟨1⟩ := rfl
        rw [this]
        show (⟨ulen (pre ++ mid ++ ['\n']) - 1⟩ : String.Pos.Raw) = _
        rw [ulen_append (pre ++ mid), ulen_cons, hsz]
        rfl
      rw [hnext, hun, extract_at s pre mid _ hs]
      have := ih (pre ++ mid ++ ['\n']) [] (by rw [hs]; simp) (by simp) (String.ofList mid :: r)
      rw [List.append_nil] at this
      rw [this, List.nil_append, show (mid ++ '\n' :: post).splitOn '\n' = mid :: post.splitOn '\n' from
        List.splitOnP_append_cons_of_forall_mem hmid' '\n' rfl post]
      simp
    · rw [if_neg (by rw [hget, hnl, beq_iff_eq]; exact hc)]
      have hnext : (String.Pos.Raw.unoffsetBy ⟨ulen (pre ++ mid)⟩ 0).next s = ⟨ulen (pre ++ (mid ++ [c]))⟩ := by
        show String.Pos.Raw.next s ⟨ulen (pre ++ mid) - 0⟩ = _
        rw [Nat.sub_zero]
        unfold String.Pos.Raw.next
        rw [hget, String.Pos.Raw.add_char_eq, ← List.append_assoc, ulen_append (pre ++ mid), ulen_cons]
        simp [ulen]
      rw [hnext, ih pre (mid ++ [c]) (by rw [hs]; simp) (by
        simp only [List.mem_append, List.mem_singleton, not_or]; exact ⟨hmid, fun h => hc h.symm⟩) r]
      simp

/-- **The legacy `String.splitOn "\n"` is `List.splitOn '\n'` on the characters.** -/
theorem splitOn_nl (s : String) : s.splitOn "\n" = (s.toList.splitOn '\n').map String.ofList := by
  unfold String.splitOn
  rw [if_neg (by decide)]
  have := splitOnAux_nl s [] [] s.toList (by simp) (by simp) []
  exact this

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

theorem foldl_prefix (l : List Char) (p : Array String × String) : ∃ ys, (l.foldl tstep p).1 = p.1 ++ ys := by
  induction l generalizing p with
  | nil => exact ⟨#[], by simp⟩
  | cons c l ih =>
    rw [List.foldl_cons]
    obtain ⟨ys, hys⟩ := ih (tstep p c)
    by_cases hw : c.isWhitespace
    · by_cases hc : p.2 != ""
      · have e : tstep p c = (p.1.push p.2, "") := by simp only [tstep, hw, hc, ↓reduceIte]
        rw [e] at hys ⊢
        exact ⟨#[p.2] ++ ys, by rw [hys]; simp⟩
      · have e : tstep p c = p := by simp only [tstep, hw, hc, ↓reduceIte, Bool.false_eq_true]
        rw [e] at hys ⊢
        exact ⟨ys, hys⟩
    · have e : tstep p c = (p.1, p.2.push c) := by simp only [tstep, hw, ↓reduceIte, Bool.false_eq_true]
      rw [e] at hys ⊢
      exact ⟨ys, hys⟩

/-- **A line's second token**: `w₀ w₁ rest` with two nonempty words of no whitespace has `w₁` as its token 1. -/
theorem tokens_second (w₀ w₁ rest : String) (h₀ : w₀ ≠ "") (h₁ : w₁ ≠ "") (n₀ : NoWs w₀) (n₁ : NoWs w₁) :
    (Flock.tokens (w₀ ++ " " ++ w₁ ++ " " ++ rest)).getD 1 "" = w₁ := by
  have hF : ∃ ys, (List.foldl tstep (#[], "") (w₀ ++ " " ++ w₁ ++ " " ++ rest).toList).1 = #[w₀, w₁] ++ ys := by
    simp only [String.toList_append, List.foldl_append]
    rw [foldl_word _ n₀, String.empty_append, String.ofList_toList]
    rw [show " ".toList = [' '] from rfl]
    simp only [List.foldl_cons, List.foldl_nil]
    rw [show tstep (#[], w₀) ' ' = (#[w₀], "") by simp [tstep, h₀]]
    rw [foldl_word _ n₁, String.empty_append, String.ofList_toList]
    rw [show tstep (#[w₀], w₁) ' ' = (#[w₀, w₁], "") by simp [tstep, h₁]]
    exact foldl_prefix rest.toList (#[w₀, w₁], "")
  obtain ⟨ys, hys⟩ := hF
  rw [tokens_eq]
  generalize List.foldl tstep (#[], "") (w₀ ++ " " ++ w₁ ++ " " ++ rest).toList = F at hys ⊢
  dsimp only
  split <;> simp [hys, Array.getD_eq_getD_getElem?, Array.getElem?_append]

/-! ## `HmRow.parse`'s unit net -/

open Flock FlockSoundness

/-- The lines `HmRow.parse` reads (`HmRow.lean:556-557`): the text split at `\n`, without a last empty line. -/
def linesOf (text : String) : Array String :=
  let lines0 := (text.splitOn "\n").toArray
  if lines0.back? == some "" then lines0.pop else lines0

/-- A `while` loop in `Except` ends in `Q` when each step keeps `P`, a continuing step lowers a measure, and a stopping
step ends in `Q` (`loop_inv` with the final state's own predicate). -/
theorem loop_inv' {β : Type} (P Q : β → Prop) (μ : β → ℕ) {f : Unit → β → Except String (ForInStep β)}
    (hf : ∀ b, P b → ∀ s, f () b = .ok s →
      (∃ b', s = .done b' ∧ Q b') ∨ ∃ b', s = .yield b' ∧ P b' ∧ μ b' < μ b) :
    ∀ n b v, μ b ≤ n → P b → forIn Lean.Loop.mk b f = .ok v → Q v := by
  intro n
  induction n with
  | zero =>
    intro b v hμ hb h
    change Lean.Loop.forIn Lean.Loop.mk b f = .ok v at h
    rw [Lean.Loop.forIn_eq_of_monadTail] at h
    obtain ⟨s, hs, h⟩ := except_bind_ok h
    rcases hf b hb s hs with ⟨b', rfl, hb'⟩ | ⟨b', rfl, -, hlt⟩
    · cases h; exact hb'
    · omega
  | succ n ih =>
    intro b v hμ hb h
    change Lean.Loop.forIn Lean.Loop.mk b f = .ok v at h
    rw [Lean.Loop.forIn_eq_of_monadTail] at h
    obtain ⟨s, hs, h⟩ := except_bind_ok h
    rcases hf b hb s hs with ⟨b', rfl, hb'⟩ | ⟨b', rfl, hb', hlt⟩
    · cases h; exact hb'
    · exact ih b' v (by omega) hb' h

theorem getElem!_str (a : Array String) (i : ℕ) : a[i]! = a.getD i "" := by
  simp only [getElem!_def, Array.getD_eq_getD_getElem?]
  cases a[i]? <;> rfl

/-- The first net read is the `Net.parse` of the lines after line 2, under line 2's second token. -/
def FirstNet (lines : Array String) (nets : Array (String × Net)) : Prop :=
  ∃ n net, 3 + n ≤ lines.size ∧ Net.parse (lines.extract 3 (3 + n)) = .ok net ∧
    nets.toList.head? = some ((tokens (lines.getD 2 "")).getD 1 "", net)

set_option maxHeartbeats 4000000 in
/-- **`HmRow.parse`'s unit is the first section's net** when line 2 (`CIRCUIT unit <n>`) names it `unit`: the
`Net.parse` of the `n` lines after it (`HmRow.lean:580-590`, `615`). -/
theorem parse_unit {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit} {text : String}
    (h : HmRow.parse tags bytes tables = .ok c) (htext : String.fromUTF8? bytes = some text)
    (hu : (tokens ((linesOf text).getD 2 "")).getD 1 "" = "unit") :
    ∃ n, 3 + n ≤ (linesOf text).size ∧ Net.parse ((linesOf text).extract 3 (3 + n)) = .ok c.unit := by
  unfold HmRow.parse at h
  simp (config := { zeta := false }) only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte, Option.mapM_none] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) sha jp0 at h)
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  rename_i txt htxt
  rw [htext] at htxt
  cases htxt
  (have hh := except_bind_ok h; clear h; obtain ⟨text', hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp0] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) lines0 lines head jp1 at h)
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp1] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) name metaLine jp2 at h)
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp2] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at h)
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  (have hh := except_bind_ok h; clear h; obtain ⟨mj, hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp3] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨f1, hf1, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨f2, hf2, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h)
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp5 jp6 at h)
  obtain ⟨classSha, h⟩ : ∃ classSha, jp5 classSha = .ok c := by
    split at h
    · replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp6] at h
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) part rule jp7 at h)
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp7] at h
      obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
    · obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
  simp (config := { zeta := false }) only [jp5] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets0 at0 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨s1, hwhile, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets1 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨u1, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpT at h)
  simp (config := { zeta := false }) only [jpT] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpL at h)
  obtain ⟨lks, h⟩ : ∃ lks, jpL lks = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpL] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨s2, hlook, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets2 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨ports, hrp, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpK at h)
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpK] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨g, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨upv, -, h⟩ := hh)
  split at h
  swap
  · cases h
  rename_i unitNet hunit
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) netOf at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨ranges, hranges, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leavesIn, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leavesOut, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummy, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummyDigests, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummyOutputs, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨outPorts, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpR at h)
  obtain ⟨outRowPorts, h⟩ : ∃ r, jpR r = .ok c := by
    split at h
    · exact ⟨_, h⟩
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨r, -, h⟩ := except_bind_ok h
      exact ⟨r, ite_throw_ok h⟩
  simp (config := { zeta := false }) only [jpR] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) perVu0 jpP at h)
  obtain ⟨pv, h⟩ : ∃ pv, jpP () pv = .ok c := by
    split at h
    · obtain ⟨p', -, h⟩ := except_bind_ok h
      exact ⟨_, h⟩
    · exact ⟨_, h⟩
  simp (config := { zeta := false }) only [jpP] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨shaNet, hsha, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨hmNet, hhm, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨outNet, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpO at h)
  obtain ⟨onp, h⟩ : ∃ x, jpO x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpO] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h)
  obtain ⟨lc, h⟩ : ∃ x, jpC x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpC] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpW at h)
  obtain ⟨wires, h⟩ : ∃ x, jpW x = .ok c := by
    split at h
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
  simp (config := { zeta := false }) only [jpW] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) derived wiresT rowNet at h)
  obtain ⟨hwires, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only at h
  (have hh := except_bind_ok h; clear h; obtain ⟨ls, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leafId, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpF at h)
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpF] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨kLog, -, h⟩ := hh)
  simp (config := { zeta := false }) only [pure_bind] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) c0 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨w, hcheck, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨w', hrows, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨w'', hrec, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨w''', hres, h⟩ := hh)
  have hc : c0 = c := Except.ok.inj h
  subst hc
  have hL : lines = linesOf text := rfl
  rw [← hL] at hu ⊢
  have h2 : 2 < lines.size := by
    by_contra hn
    have e : lines.getD 2 "" = "" := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]; rfl
    rw [e] at hu
    simp [tokens_eq] at hu
  have hfirst : FirstNet lines s1.1 := by
    refine loop_inv' (fun b : Array (String × Net) × ℕ => FirstNet lines b.1 ∨ (b.1 = #[] ∧ b.2 = 2))
      (fun b => FirstNet lines b.1) (fun b => lines.size - b.2) ?_ _ (nets0, at0) s1 le_rfl (Or.inr ⟨rfl, rfl⟩) hwhile
    intro b hb s h
    ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets at_ at h)
    split at h
    · rename_i hlt
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) hh jp at h)
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp] at h
      split at h
      · rename_i n hn
        ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp' at h)
        obtain ⟨hnot, h⟩ := ite_throw_ok' h
        simp (config := { zeta := false }) only [jp'] at h
        ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp'' at h)
        split at h
        · rename_i v hv
          (have hh' := except_bind_ok h; clear h; obtain ⟨net, hnet, h⟩ := hh')
          cases hnet
          simp (config := { zeta := false }) only [jp''] at h
          ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets' at' at h)
          have hs := Except.ok.inj h
          subst hs
          refine Or.inr ⟨_, rfl, ?_, ?_⟩
          · left
            rcases hb with ⟨n0, net0, hle, hp, hl⟩ | ⟨hb1, hb2⟩
            · exact ⟨n0, net0, hle, hp, by simp [nets', nets, Array.toList_push, List.head?_append, hl]⟩
            · refine ⟨n, v, ?_, ?_, ?_⟩
              · simp only [at_, hb2] at hnot; omega
              · simpa [at_, hb2] using hv
              · simp [nets', nets, hb1, hh, at_, hb2, getElem!_str]
          · simp only [at', at_]; omega
        · (have hh' := except_bind_ok h; clear h; obtain ⟨net, hnet, h⟩ := hh')
          cases hnet
      · (have hh' := except_bind_ok h; clear h; obtain ⟨w, hw, h⟩ := hh')
        cases hw
    · rename_i hge
      have hge' : ¬ b.2 < lines.size := hge
      have hs := Except.ok.inj h
      subst hs
      refine Or.inl ⟨_, rfl, ?_⟩
      rcases hb with hb | ⟨-, hb2⟩
      · exact hb
      · rw [hb2] at hge'; omega
  have hfirst2 : FirstNet lines nets2 := by
    rw [← Array.forIn_toList] at hlook
    refine forIn_except_inv (fun nets : Array (String × Net) => FirstNet lines nets) _ ?_ nets1 s2 hfirst hlook
    intro a _ b hb s h
    obtain ⟨lname, spec⟩ := a
    ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets at h)
    split at h
    · ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) size at h)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at h)
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jpA] at h
      (have hh := except_bind_ok h; clear h; obtain ⟨n, -, h⟩ := hh)
      (have hh := except_bind_ok h; clear h; obtain ⟨lo, -, h⟩ := hh)
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at h)
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jpB] at h
      split at h
      · ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h)
        replace h := ite_throw_ok h
        simp (config := { zeta := false }) only [jpC] at h
        (have hh := except_bind_ok h; clear h; obtain ⟨net, hbuild, h⟩ := hh)
        ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets' at h)
        have hs := Except.ok.inj h
        subst hs
        refine ⟨_, Or.inl rfl, ?_⟩
        obtain ⟨n0, net0, hle, hp, hl⟩ := hb
        exact ⟨n0, net0, hle, hp, by simp [nets', nets, Array.toList_push, List.head?_append, hl]⟩
      · (have hh := except_bind_ok h; clear h; obtain ⟨w, hw, h⟩ := hh)
        cases hw
    · (have hh := except_bind_ok h; clear h; obtain ⟨w, hw, h⟩ := hh)
      cases hw
  obtain ⟨n, net, hle, hp, hl⟩ := hfirst2
  rw [List.head?_eq_getElem?, Array.getElem?_toList] at hl
  obtain ⟨hsz, hl0⟩ := Array.getElem?_eq_some_iff.1 hl
  have h0 : unitNet = 0 := by
    obtain ⟨hlt, -, hbefore⟩ := Array.findIdx?_eq_some_iff_getElem.1 hunit
    by_contra hne
    have := hbefore 0 (by omega)
    rw [hl0, hu] at this
    exact this rfl
  refine ⟨n, hle, ?_⟩
  have e : (nets2.getD 0 default).2 = net := by
    rw [Array.getD_eq_getD_getElem?, hl]; rfl
  show Net.parse _ = .ok (Circuit.unit c0)
  simp only [Circuit.unit, c0, h0, e]
  exact hp

end FlockVBridge.Names
