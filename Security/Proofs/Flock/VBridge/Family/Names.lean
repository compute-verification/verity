import Proofs.Flock.VBridge.Family.Structure
import Proofs.Flock.Soundness.ExecCircuit

/-!
# The circuit file's unit name, emitted and parsed

`circuitFile` (`Family.lean`) writes `flock-circuit <unit>`, the `META` line, then `CIRCUIT unit <n>` and the unit's
net, whose header line is `flock-ir-unit/v2 <unit> …` (`VStar.emitNet`, `Unit.lean:94`). `HmRow.parse`
(`HmRow.lean:550`) splits the text at `\n`, takes the first `CIRCUIT unit` section's net (`Net.parse`, `Net.lean:200`),
and that net's name is the second token of its header line. The legacy `String.splitOn "\n"` is `List.splitOn '\n'` on
the characters (`splitOn_nl`), `Flock.tokens` reads a line's second word (`tokens_second`), `pyDumps` writes one line
(`pyDumps_noNl`), and the unit names are one word (`recOpen_name`, `residuals_name`), so the parse names the circuit's
unit as the statement's (`file_unitName`): `HUnitName` and `HOpenName` at any history and registration.
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
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
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

/-! ## Lines without a newline -/

/-- `s` holds no `\n`. -/
def NoNl (s : String) : Prop := '\n' ∉ s.toList

theorem noNl_append {a b : String} : NoNl (a ++ b) ↔ NoNl a ∧ NoNl b := by
  simp [NoNl, String.toList_append]

theorem noNl_of_noWs {s : String} (h : NoWs s) : NoNl s := fun hm => absurd (h _ hm) (by decide)

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

theorem int_noNl (i : ℤ) : NoNl (toString i) := by
  cases i with
  | ofNat n => exact nat_noNl n
  | negSucc n =>
    show NoNl ("-" ++ toString (n + 1))
    exact noNl_append.2 ⟨by simp [NoNl], nat_noNl _⟩

theorem hex_ne : ∀ x, x < 16 → (if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)) ≠ '\n' := by
  decide

theorem hex4_noNl (n : ℕ) : NoNl (VStar.hex4 n) := by
  unfold VStar.hex4
  simp only [NoNl, String.toList_ofList, List.mem_cons, List.not_mem_nil, or_false, not_or]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> exact fun h => hex_ne _ (Nat.mod_lt _ (by decide)) h.symm

theorem foldl_noNl {β : Type} (g : String → β → String) (hg : ∀ o c, NoNl o → NoNl (g o c)) :
    ∀ (l : List β) o, NoNl o → NoNl (l.foldl g o)
  | [], _, ho => ho
  | c :: l, o, ho => foldl_noNl g hg l _ (hg o c ho)

set_option maxRecDepth 100000 in
/-- **`pyStr` escapes every newline** (`PyJson.lean:20-34`). -/
theorem pyStr_noNl (s : String) : NoNl (VStar.pyStr s) := by
  unfold VStar.pyStr
  simp only [Id.run]
  rw [List.forIn_pure_yield_eq_foldl]
  refine noNl_append.2 ⟨foldl_noNl _ ?_ _ _ (by simp [NoNl]), by simp [NoNl]⟩
  intro o c ho
  have hx := hex4_noNl
  simp only [NoNl] at hx ho ⊢
  split <;> (repeat' split) <;> simp_all [String.toList_append]
  all_goals exact fun h => by subst h; contradiction

/-- **`pyDumps` writes one line** (`PyJson.lean:37-51`): strings through `pyStr`, the rest digits and punctuation. -/
theorem pyDumpsAt_noNl : ∀ (k : ℕ) (j : Lean.Json) (d : String), VStar.pyDumpsAt k j = .ok d → NoNl d
  | 0, _, _, h => by cases h
  | k + 1, .null, d, h => by cases h; simp [NoNl]
  | k + 1, .bool b, d, h => by cases h; cases b <;> simp [NoNl]
  | k + 1, .num n, d, h => by
    unfold VStar.pyDumpsAt at h
    split at h
    · cases h; exact int_noNl _
    · cases h
  | k + 1, .str s, d, h => by cases h; exact pyStr_noNl s
  | k + 1, .arr a, d, h => by
    unfold VStar.pyDumpsAt at h
    obtain ⟨parts, hp, h⟩ := except_bind_ok h
    cases h
    refine noNl_append.2 ⟨noNl_append.2 ⟨by simp [NoNl], noNl_intercalate (by simp [NoNl]) _ fun x hx => ?_⟩,
      by simp [NoNl]⟩
    obtain ⟨y, -, hy⟩ := array_mapM_ok hp x hx
    exact pyDumpsAt_noNl k y x hy
  | k + 1, .obj kvs, d, h => by
    unfold VStar.pyDumpsAt at h
    obtain ⟨parts, hp, h⟩ := except_bind_ok h
    cases h
    refine noNl_append.2 ⟨noNl_append.2 ⟨by simp [NoNl], noNl_intercalate (by simp [NoNl]) _ fun x hx => ?_⟩,
      by simp [NoNl]⟩
    obtain ⟨⟨key, v⟩, -, hy⟩ := array_mapM_ok hp x hx
    obtain ⟨e, he, hy⟩ := except_bind_ok hy
    cases hy
    exact noNl_append.2 ⟨noNl_append.2 ⟨pyStr_noNl key, by simp [NoNl]⟩, pyDumpsAt_noNl k v e he⟩

theorem pyDumps_noNl {j : Lean.Json} {d : String} (h : VStar.pyDumps j = .ok d) : NoNl d :=
  pyDumpsAt_noNl 64 j d h

theorem noWs_append {a b : String} : NoWs (a ++ b) ↔ NoWs a ∧ NoWs b := by
  simp only [NoWs, String.toList_append, List.mem_append]
  exact ⟨fun h => ⟨fun c hc => h c (Or.inl hc), fun c hc => h c (Or.inr hc)⟩,
    fun h c hc => hc.elim (h.1 c) (h.2 c)⟩

theorem digit_noWs {c : Char} (h : c.isDigit = true) : c.isWhitespace = false := by
  simp only [Char.isWhitespace, Bool.or_eq_false_iff, decide_eq_false_iff_not]
  refine ⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩ <;> (rintro rfl; exact absurd h (by decide))

theorem nat_noWs (n : ℕ) : NoWs (toString n) := fun _ hc =>
  digit_noWs (Nat.isDigit_of_mem_toDigits (b := 10) (n := n) (by decide) (by decide) (by
    rw [← Nat.toList_repr]; exact hc))

theorem append_ne_empty {a b : String} (ha : a ≠ "") : a ++ b ≠ "" := by
  intro h
  have := congrArg String.toList h
  simp only [String.toList_append, String.toList_empty, List.append_eq_nil_iff] at this
  exact ha (String.toList_eq_nil_iff.1 this.1)

/-- **An opening statement's unit name is one word.** -/
theorem recOpen_name (L H : ℕ) : NoWs (Flock.RecOpen.unitName L H) ∧ Flock.RecOpen.unitName L H ≠ "" := by
  show NoWs ("rec-open/l" ++ toString L ++ "-h" ++ toString H) ∧ "rec-open/l" ++ toString L ++ "-h" ++ toString H ≠ ""
  refine ⟨noWs_append.2 ⟨noWs_append.2 ⟨noWs_append.2 ⟨by simp [NoWs], nat_noWs _⟩, by simp [NoWs]⟩, nat_noWs _⟩,
    append_ne_empty (append_ne_empty (append_ne_empty (by decide)))⟩

/-- **A part statement's unit name is one word.** -/
theorem residuals_name (m k c j : ℕ) :
    NoWs (Flock.RecResiduals.unitName m k c j) ∧ Flock.RecResiduals.unitName m k c j ≠ "" := by
  show NoWs ("rec-residuals/m" ++ toString m ++ "-k" ++ toString k ++ "-c" ++ toString c ++ "-p" ++ toString j) ∧
    "rec-residuals/m" ++ toString m ++ "-k" ++ toString k ++ "-c" ++ toString c ++ "-p" ++ toString j ≠ ""
  simp only [noWs_append, nat_noWs, and_true]
  refine ⟨⟨⟨⟨by simp [NoWs], by simp [NoWs]⟩, by simp [NoWs]⟩, by simp [NoWs]⟩,
    append_ne_empty (append_ne_empty (append_ne_empty (append_ne_empty (append_ne_empty (append_ne_empty
      (append_ne_empty (by decide)))))))⟩

/-! ## The emitted text -/

/-- A `StateM String` program that only appends to its state. -/
def Ext {α : Type} (p : StateM String α) : Prop := ∀ s, ∃ x, (p.run s).2 = s ++ x

/-- A `StateM String` program that appends `w` and then more. -/
def Pre {α : Type} (w : String) (p : StateM String α) : Prop := ∀ s, ∃ x, (p.run s).2 = s ++ w ++ x

theorem ext_put (x : String) : Ext (modify (fun s : String => s ++ x)) := fun _ => ⟨x, rfl⟩

theorem ext_pure {α : Type} (a : α) : Ext (pure a : StateM String α) := fun _ => ⟨"", by simp; rfl⟩

theorem ext_bind {α β : Type} {p : StateM String α} {f : α → StateM String β} (hp : Ext p) (hf : ∀ a, Ext (f a)) :
    Ext (p >>= f) := by
  intro s
  obtain ⟨x, hx⟩ := hp s
  obtain ⟨y, hy⟩ := hf (p.run s).1 (p.run s).2
  refine ⟨x ++ y, ?_⟩
  rw [StateT.run_bind]
  show (StateT.run (f (p.run s).1) (p.run s).2).2 = _
  rw [hy, hx, String.append_assoc]

theorem ext_forIn {β σ : Type} (l : List β) (f : β → σ → StateM String (ForInStep σ)) (hf : ∀ b c, Ext (f b c)) :
    ∀ init, Ext (forIn l init f) := by
  induction l with
  | nil => intro init; exact ext_pure _
  | cons b l ih =>
    intro init
    rw [List.forIn_cons]
    refine ext_bind (hf b init) fun r => ?_
    cases r with
    | done c => exact ext_pure _
    | yield c => exact ih c

theorem ext_emitNet (name : String) (l : Flock.HmNets.Laid) (lv : Option String) :
    Ext (VStar.emitNet (m := StateM String) (fun x => modify (· ++ x)) name l lv) := by
  unfold VStar.emitNet
  refine ext_bind (ext_put _) fun _ => ext_bind ?_ fun _ => ?_
  · rw [Std.Legacy.Range.forIn_eq_forIn_range']
    exact ext_forIn _ _ (fun _ _ => ext_bind (ext_put _) fun _ => ext_pure _) _
  · split
    · exact ext_put _
    · exact ext_pure _

theorem pre_put {β : Type} (x w : String) {f : Unit → StateM String β} (hf : ∀ a, Pre w (f a)) :
    Pre (x ++ w) ((modify (· ++ x) : StateM String Unit) >>= f) := by
  intro s
  obtain ⟨y, hy⟩ := hf () (s ++ x)
  exact ⟨y, by show (StateT.run (f ()) (s ++ x)).2 = _; rw [hy]; simp only [String.append_assoc]⟩

theorem pre_bind {α β : Type} (w : String) {p : StateM String α} {f : α → StateM String β} (hp : Pre w p)
    (hf : ∀ a, Ext (f a)) : Pre w (p >>= f) := by
  intro s
  obtain ⟨x, hx⟩ := hp s
  obtain ⟨y, hy⟩ := hf (p.run s).1 (p.run s).2
  refine ⟨x ++ y, ?_⟩
  rw [StateT.run_bind]
  show (StateT.run (f (p.run s).1) (p.run s).2).2 = _
  rw [hy, hx]; simp only [String.append_assoc]

theorem pre_ext {α : Type} {p : StateM String α} (hp : Ext p) : Pre "" p := fun s => by
  obtain ⟨x, hx⟩ := hp s
  exact ⟨x, by rw [hx, String.append_empty]⟩

/-- **The sections start with the unit's**: `CIRCUIT unit <lines>`, then the unit's header line
(`Compose.lean:233-237`, `Unit.lean:94-100`). -/
theorem sections_text (u : VStar.UnitC) (l : Flock.HmNets.Laid) :
    Pre (("CIRCUIT unit " ++ toString (VStar.netLines l) ++ "\n") ++ (VStar.header u.name l ++ "\n"))
      (VStar.emitSections (m := StateM String) (fun x => modify (· ++ x)) u l) := by
  unfold VStar.emitSections
  rw [List.forIn_cons]
  refine pre_bind _ (pre_bind _ ?_ fun r => ?_) fun _ => ext_pure _
  · dsimp only
    refine pre_put _ _ fun _ => pre_bind _ ?_ fun _ => ext_pure _
    rw [show VStar.header u.name l ++ "\n" = (VStar.header u.name l ++ "\n") ++ "" from (String.append_empty).symm]
    unfold VStar.emitNet
    refine pre_put _ _ fun _ => pre_ext (ext_bind ?_ fun _ => ?_)
    · rw [Std.Legacy.Range.forIn_eq_forIn_range']
      exact ext_forIn _ _ (fun _ _ => ext_bind (ext_put _) fun _ => ext_pure _) _
    · exact ext_pure _
  · cases r with
    | done c => exact ext_pure _
    | yield c =>
      refine ext_forIn _ _ (fun a _ => ?_) _
      obtain ⟨nm, net, lay⟩ := a
      exact ext_bind (ext_put _) fun _ => ext_bind (ext_emitNet _ _ _) fun _ => ext_pure _

/-- **The file's first two lines** (`Compose.lean:187-230`): `flock-circuit <unit>` and `META ` with `pyDumps`'s text. -/
theorem metaHead_ok {u : VStar.UnitC} {l : Flock.HmNets.Laid} {sub : VStar.Sub} {b : VStar.Bindings}
    {sha head : String} (h : VStar.metaHead u l sub b sha = .ok head) :
    ∃ j d, VStar.pyDumps j = .ok d ∧ head = s!"{VStar.FORMAT_CIRCUIT} {u.name}\nMETA " ++ d ++ "\n" := by
  unfold VStar.metaHead at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat' (first | split at h | (injection h with h; subst h))
  all_goals first | exact absurd h (by simp) | exact ⟨_, _, ‹_›, rfl⟩

theorem emitCircuit_ok {m : Type → Type} [Monad m] {put : String → m Unit} {u : VStar.UnitC}
    {l : Flock.HmNets.Laid} {sub : VStar.Sub} {b : VStar.Bindings} {e : m Unit}
    (h : VStar.emitCircuit put u l sub b = .ok e) :
    ∃ sha head, VStar.metaHead u l sub b sha = .ok head ∧ e = (do put head; VStar.emitSections put u l) := by
  unfold VStar.emitCircuit at h
  obtain ⟨_, -, h⟩ := except_bind_ok h
  obtain ⟨head, hh, h⟩ := except_bind_ok h
  cases h
  exact ⟨_, _, hh, rfl⟩

theorem header_eq (name : String) (l : Flock.HmNets.Laid) :
    ∃ rest, VStar.header name l = VStar.FORMAT ++ " " ++ name ++ " " ++ rest := by
  unfold VStar.header
  simp only [List.cons_append]
  rw [String.intercalate_cons_cons, String.intercalate_cons_cons]
  exact ⟨_, by simp only [String.append_assoc]; exact rfl⟩

theorem header_noNl (name : String) (l : Flock.HmNets.Laid) (h : NoNl name) : NoNl (VStar.header name l) := by
  unfold VStar.header
  refine noNl_intercalate (by simp [NoNl]) _ fun x hx => ?_
  simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_append, List.mem_flatMap, List.mem_map] at hx
  rcases hx with rfl | rfl | rfl | rfl | rfl | ⟨g, -, n, -, rfl⟩ | rfl | ⟨g, -, n, -, rfl⟩
  · simp [NoNl, VStar.FORMAT]
  · exact h
  all_goals exact nat_noNl _

theorem recOpenUnit_name {L H : ℕ} {u : VStar.UnitC} (h : VStar.recOpenUnit L H = .ok u) :
    u.name = Flock.RecOpen.unitName L H := by
  unfold VStar.recOpenUnit at h
  obtain ⟨_, -, h⟩ := except_bind_ok h
  cases h
  rfl

theorem innerUnit_name {S : VStar.RepS} {nm : String} {u : VStar.UnitC} (h : VStar.innerUnit S nm = .ok u) :
    u.name = nm := by
  unfold VStar.innerUnit at h
  obtain ⟨_, -, h⟩ := except_bind_ok h
  obtain ⟨_, -, h⟩ := except_bind_ok h
  cases h
  rfl

/-! ## The split lines -/

theorem splitOn_line (a rest : String) (ha : NoNl a) : (a ++ "\n" ++ rest).splitOn "\n" = a :: rest.splitOn "\n" := by
  rw [splitOn_nl, splitOn_nl, String.toList_append, String.toList_append, show "\n".toList = ['\n'] by decide,
    List.append_assoc, List.singleton_append, List.splitOn_eq_splitOnP,
    List.splitOnP_append_cons_of_forall_mem (fun x hx => by
      simp only [beq_eq_false_iff_ne, ne_eq]; rintro rfl; exact ha hx) '\n' rfl, List.map_cons,
    String.ofList_toList, List.splitOn_eq_splitOnP]

theorem splitOn_ne_nil (s : String) : s.splitOn "\n" ≠ [] := by
  rw [splitOn_nl]; simp [List.splitOn_ne_nil '\n' s.toList]

theorem linesOf_four {text a b c d : String} {r : List String} (h : text.splitOn "\n" = a :: b :: c :: d :: r)
    (hr : r ≠ []) : (linesOf text).getD 2 "" = c ∧ (linesOf text).getD 3 "" = d ∧ 4 ≤ (linesOf text).size := by
  unfold linesOf
  have hs : 5 ≤ (text.splitOn "\n").toArray.size := by
    simp only [h, List.size_toArray, List.length_cons]
    cases r with
    | nil => exact absurd rfl hr
    | cons _ _ => simp
  have h2 : (text.splitOn "\n").toArray.getD 2 "" = c := by simp [h]
  have h3 : (text.splitOn "\n").toArray.getD 3 "" = d := by simp [h]
  dsimp only
  split
  · refine ⟨?_, ?_, by rw [Array.size_pop]; omega⟩
    · rw [Array.getD_eq_getD_getElem?, Array.getElem?_pop, if_pos (by omega), ← Array.getD_eq_getD_getElem?, h2]
    · rw [Array.getD_eq_getD_getElem?, Array.getElem?_pop, if_pos (by omega), ← Array.getD_eq_getD_getElem?, h3]
  · exact ⟨h2, h3, by omega⟩

/-! ## The parsed unit's name -/

theorem ofRows_name {name : String} {useful constPos : ℕ} {ig og : Array Group} {ra rb : Array (Array ℕ)} {n : Net}
    (h : Net.ofRows name useful constPos ig og ra rb = .ok n) : n.name = name := by
  unfold Net.ofRows at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat' (first | split at h | (injection h with h; subst h))
  all_goals first | exact absurd h (by simp) | rfl

/-- **`Net.parse`'s name is its header line's second token** (`Net.lean:200-204`, `ofRows`). -/
theorem parse_name {lines : Array String} {net : Net} (h : Net.parse lines = .ok net) :
    (tokens (lines.getD 0 ""))[0]? = some "flock-ir-unit/v2" ∧ net.name = (tokens (lines.getD 0 "")).getD 1 "" := by
  unfold Net.parse at h
  simp only at h
  by_cases h0 : ((tokens (lines.getD 0 ""))[0]? != some "flock-ir-unit/v2") = true
  · rw [ite_eq_left_of_eq_true _ _ (eq_true h0)] at h; cases h
  rw [ite_eq_right_of_eq_false _ _ (eq_false h0)] at h
  refine ⟨by simpa using h0, ?_⟩
  obtain ⟨useful, -, h⟩ := except_bind_ok h
  obtain ⟨constPos, -, h⟩ := except_bind_ok h
  obtain ⟨ig, -, h⟩ := except_bind_ok h
  obtain ⟨og, -, h⟩ := except_bind_ok h
  replace h := ite_throw_ok (ite_throw_ok (ite_throw_ok (ite_throw_ok (ite_throw_ok h))))
  obtain ⟨s, -, h⟩ := except_bind_ok h
  by_cases hL : (lines.getD (1 + useful) "").startsWith "LEAVES " = true
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hL)] at h
    rcases hj : Lean.Json.parse ((lines.getD (1 + useful) "").drop 7).toString with e | j
    · rw [hj] at h; exact absurd h throw_bind_ne
    · rw [hj] at h
      exact ofRows_name (ite_throw_ok (ite_throw_ok h))
  · rw [ite_eq_right_of_eq_false _ _ (eq_false hL)] at h
    exact ofRows_name (ite_throw_ok (ite_throw_ok h))

/-- **The parse of a statement's circuit file names its unit as the statement's unit**: `circuitFile` writes
`flock-circuit <unit>`, the one-line `META`, `CIRCUIT unit <n>` and the unit's header `flock-ir-unit/v2 <unit> …`
(`Family.lean:51`, `Compose.lean:187-244`, `Unit.lean:80-100`), and `HmRow.parse` takes the first section, `unit`, as
the circuit's unit (`parse_unit`), named by its header's second token (`parse_name`). The name is one word. -/
theorem file_unitName {s : VStar.Stmt} {b : VStar.Bindings} {file : ByteArray}
    (hf : FlockVBridge.Family.circuitFile s b = .ok file) {tags : Tags} {tables : List (String × ByteArray)}
    {c : Circuit} (hp : HmRow.parse tags file tables = .ok c)
    (hw : ∀ u, s.unit = .ok u → NoWs u.name ∧ u.name ≠ "") : ∃ u, s.unit = .ok u ∧ c.unit.name = u.name := by
  unfold FlockVBridge.Family.circuitFile at hf
  obtain ⟨u, hu, hf⟩ := except_bind_ok hf
  obtain ⟨l, -, hf⟩ := except_bind_ok hf
  obtain ⟨e, he, hf⟩ := except_bind_ok hf
  cases hf
  refine ⟨u, hu, ?_⟩
  obtain ⟨hws, hne⟩ := hw u hu
  obtain ⟨sha, head, hh, he'⟩ := emitCircuit_ok he
  obtain ⟨j, d, hd, rfl⟩ := metaHead_ok hh
  have hpre : Pre ((s!"{VStar.FORMAT_CIRCUIT} {u.name}\nMETA " ++ d ++ "\n") ++
      (("CIRCUIT unit " ++ toString (VStar.netLines l) ++ "\n") ++ (VStar.header u.name l ++ "\n"))) e := by
    rw [he']
    exact pre_put _ _ (fun _ => sections_text u l)
  obtain ⟨tl, hX⟩ := hpre ""
  have hp' : HmRow.parse tags (StateT.run e "").2.toUTF8 tables = .ok c := hp
  rw [hX] at hp'
  have key : ∀ text : String,
      text = "" ++ ((s!"{VStar.FORMAT_CIRCUIT} {u.name}\nMETA " ++ d ++ "\n") ++
        (("CIRCUIT unit " ++ toString (VStar.netLines l) ++ "\n") ++ (VStar.header u.name l ++ "\n"))) ++ tl →
      HmRow.parse tags text.toUTF8 tables = .ok c → c.unit.name = u.name := by
    intro text ht hp
    have e1 : text = ("flock-circuit " ++ u.name) ++ "\n" ++ (("META " ++ d) ++ "\n" ++
        (("CIRCUIT unit " ++ toString (VStar.netLines l)) ++ "\n" ++ (VStar.header u.name l ++ "\n" ++ tl))) := by
      rw [ht, show s!"{VStar.FORMAT_CIRCUIT} {u.name}\nMETA " = "flock-circuit " ++ u.name ++ "\n" ++ "META " from by
        show "flock-circuit" ++ " " ++ u.name ++ "\nMETA " = _
        rw [show ("flock-circuit" : String) ++ " " = "flock-circuit " by decide,
          show ("\nMETA " : String) = "\n" ++ "META " by decide]
        simp only [String.append_assoc]]
      simp only [String.empty_append, String.append_assoc]
    have hsplit : text.splitOn "\n" = ("flock-circuit " ++ u.name) :: ("META " ++ d) ::
        ("CIRCUIT unit " ++ toString (VStar.netLines l)) :: VStar.header u.name l :: tl.splitOn "\n" := by
      rw [e1, splitOn_line _ _ (noNl_append.2 ⟨by simp [NoNl], noNl_of_noWs hws⟩),
        splitOn_line _ _ (noNl_append.2 ⟨by simp [NoNl], pyDumps_noNl hd⟩),
        splitOn_line _ _ (noNl_append.2 ⟨by simp [NoNl], nat_noNl _⟩),
        splitOn_line _ _ (header_noNl _ _ (noNl_of_noWs hws))]
    obtain ⟨h2, h3, -⟩ := linesOf_four hsplit (splitOn_ne_nil tl)
    have hu2 : (tokens ((linesOf text).getD 2 "")).getD 1 "" = "unit" := by
      rw [h2, show ("CIRCUIT unit " : String) = "CIRCUIT" ++ " " ++ "unit" ++ " " by decide]
      exact tokens_second _ _ _ (by decide) (by decide) (by simp [NoWs]) (by simp [NoWs])
    obtain ⟨n, hn, hpn⟩ := parse_unit hp (fromUTF8?_toUTF8 text) hu2
    obtain ⟨hfmt, hname⟩ := parse_name hpn
    rw [hname]
    cases n with
    | zero =>
      have : ((linesOf text).extract 3 (3 + 0)).getD 0 "" = "" := by
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_extract, if_neg (by omega)]; rfl
      rw [this, tokens_eq] at hfmt
      simp at hfmt
    | succ n =>
      have : ((linesOf text).extract 3 (3 + (n + 1))).getD 0 "" = (linesOf text).getD 3 "" := by
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_extract, if_pos (by omega), ← Array.getD_eq_getD_getElem?]
      rw [this, h3]
      obtain ⟨rest, hr⟩ := header_eq u.name l
      rw [hr]
      exact tokens_second _ _ _ (by decide) hne (by simp [NoWs, VStar.FORMAT]) hws
  exact key _ rfl hp'

end FlockVBridge.Names

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockVBridge.Names

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- **`HUnitName` from the circuit file's round trip** (`file_unitName`): a part statement `.inner p nm pt` has unit
`innerUnit pt.S nm` (`Stage.lean:44-47`), named `nm` (`InnerRepCheck.lean:38-52`), which is
`RecResiduals.unitName …`, one word (`statements_name`). At any history and registration the session parses its circuit
from that file (`parse_circuitFile`). -/
theorem hUnitName_of (hne : Nonempty (Reg × L.Ω × Reg₂)) : F.HUnitName := by
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  intro j p nm pt hj
  have hp := F.parse_circuitFile R ω [] R₂ j (F.hfile R ω [] R₂ j)
  have hnm := statements_name F.hstmts j hj
  obtain ⟨u, hu, hn⟩ := file_unitName (F.hfile R ω [] R₂ j) hp fun u hu => by
    rw [hj] at hu
    rw [innerUnit_name hu, hnm]
    exact residuals_name _ _ _ _
  rw [hj] at hu
  rw [hn, innerUnit_name hu]

/-- **`HOpenName` from the circuit file's round trip** (`file_unitName`): an opening statement `.recOpen ℓ LANES H m`
has unit `recOpenUnit LANES H` (`Stage.lean:44-46`), named `RecOpen.unitName LANES H` (`Unit.lean:36-46`), one word. -/
theorem hOpenName_of (hne : Nonempty (Reg × L.Ω × Reg₂)) : F.HOpenName := by
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  intro j ℓ LANES H m hj
  have hp := F.parse_circuitFile R ω [] R₂ j (F.hfile R ω [] R₂ j)
  obtain ⟨u, hu, hn⟩ := file_unitName (F.hfile R ω [] R₂ j) hp fun u hu => by
    rw [hj] at hu
    rw [recOpenUnit_name hu]
    exact recOpen_name _ _
  rw [hj] at hu
  rw [hn, recOpenUnit_name hu]

end VStarFamily

end family

end FlockVBridge.Family
