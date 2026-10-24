/-
`String.splitOn s "\n"` (the legacy split every circuit parse starts from) as `List.splitOn '\n'` on `s.toList`
(`splitOn_nl`), with the byte positions it walks: the model a parse that splits a file's lines elsewhere (in tasks, from
byte offsets) is held to. `splitTasks` splits them so, reading each row's numbers in the same task (`Net.Row`).
-/
import Flock.Net

namespace Flock.SplitNL

/-- The UTF-8 byte length of a char list, as `Pos.Raw + c` advances. -/
def len : List Char → Nat
  | [] => 0
  | c :: cs => c.utf8Size + len cs

theorem len_append (a b : List Char) : len (a ++ b) = len a + len b := by
  induction a with
  | nil => simp [len]
  | cons c cs ih => simp [len, ih]; omega

theorem getAux_append (A B : List Char) (i : Nat) :
    String.Pos.Raw.utf8GetAux (A ++ B) ⟨i⟩ ⟨i + len A⟩ = String.Pos.Raw.utf8GetAux B ⟨i + len A⟩ ⟨i + len A⟩ := by
  induction A generalizing i with
  | nil => simp [len]
  | cons c cs ih =>
    have hpos := Char.utf8Size_pos c
    simp only [List.cons_append, String.Pos.Raw.utf8GetAux]
    have hne : (⟨i⟩ : String.Pos.Raw) ≠ ⟨i + len (c :: cs)⟩ := by
      intro h; have := congrArg String.Pos.Raw.byteIdx h; simp only [len] at this; omega
    rw [if_neg hne]
    have := ih (i + c.utf8Size)
    simp only [len] at *
    rw [show i + (c.utf8Size + len cs) = i + c.utf8Size + len cs by omega]
    exact this

theorem get_at (P : List Char) (c : Char) (R : List Char) (s : String) (hs : s.toList = P ++ c :: R) :
    String.Pos.Raw.get s ⟨len P⟩ = c := by
  unfold String.Pos.Raw.get
  rw [hs]
  have := getAux_append P (c :: R) 0
  simp only [Nat.zero_add] at this
  rw [show (0 : String.Pos.Raw) = ⟨0⟩ from rfl, this]
  simp [String.Pos.Raw.utf8GetAux]

theorem size_ofList (l : List Char) : (String.ofList l).utf8ByteSize = len l := by
  induction l with
  | nil => simp [len]
  | cons c cs ih =>
    rw [show c :: cs = [c] ++ cs from rfl, String.ofList_append, String.utf8ByteSize_append, ih]
    simp [len, String.utf8ByteSize_singleton]

theorem size_eq (s : String) : s.utf8ByteSize = len s.toList := by
  conv => lhs; rw [← String.ofList_toList (s := s)]
  exact size_ofList _

theorem go₂_append (C R : List Char) (j : Nat) :
    String.Pos.Raw.extract.go₂ (C ++ R) ⟨j⟩ ⟨j + len C⟩ = C := by
  induction C generalizing j with
  | nil => cases R <;> simp [len, String.Pos.Raw.extract.go₂]
  | cons c cs ih =>
    have hpos := Char.utf8Size_pos c
    have hne : (⟨j⟩ : String.Pos.Raw) ≠ ⟨j + len (c :: cs)⟩ := by
      intro h; have := congrArg String.Pos.Raw.byteIdx h; simp only [len] at this; omega
    simp only [List.cons_append, String.Pos.Raw.extract.go₂, if_neg hne]
    have := ih (j + c.utf8Size)
    simp only [len] at *
    rw [show j + (c.utf8Size + len cs) = j + c.utf8Size + len cs by omega]
    exact congrArg _ this

theorem go₁_append (P X : List Char) (i : Nat) (e : String.Pos.Raw) :
    String.Pos.Raw.extract.go₁ (P ++ X) ⟨i⟩ ⟨i + len P⟩ e = String.Pos.Raw.extract.go₁ X ⟨i + len P⟩ ⟨i + len P⟩ e := by
  induction P generalizing i with
  | nil => simp [len]
  | cons c cs ih =>
    have hpos := Char.utf8Size_pos c
    have hne : (⟨i⟩ : String.Pos.Raw) ≠ ⟨i + len (c :: cs)⟩ := by
      intro h; have := congrArg String.Pos.Raw.byteIdx h; simp only [len] at this; omega
    simp only [List.cons_append, String.Pos.Raw.extract.go₁, if_neg hne]
    have := ih (i + c.utf8Size)
    simp only [len] at *
    rw [show i + (c.utf8Size + len cs) = i + c.utf8Size + len cs by omega]
    exact this

theorem extract_eq (P C R : List Char) (s : String) (hs : s.toList = P ++ C ++ R) :
    String.Pos.Raw.extract s ⟨len P⟩ ⟨len P + len C⟩ = String.ofList C := by
  unfold String.Pos.Raw.extract
  cases C with
  | nil => simp [len]
  | cons c cs =>
    have hpos := Char.utf8Size_pos c
    show (if len P ≥ len P + len (c :: cs) then "" else String.ofList (String.Pos.Raw.extract.go₁ s.toList 0 ⟨len P⟩ ⟨len P + len (c :: cs)⟩)) = _
    rw [if_neg (by simp only [len]; omega), hs, List.append_assoc]
    have := go₁_append P (c :: cs ++ R) 0 ⟨len P + len (c :: cs)⟩
    simp only [Nat.zero_add] at this
    rw [show (0 : String.Pos.Raw) = ⟨0⟩ from rfl, this]
    simp only [List.cons_append, String.Pos.Raw.extract.go₁]
    simp only [if_pos trivial]
    rw [show c :: (cs ++ R) = (c :: cs) ++ R from rfl, go₂_append]

theorem aux (s : String) (rest : List Char) : ∀ (P cur : List Char) (r : List String),
    s.toList = P ++ cur ++ rest → '\n' ∉ cur →
    String.splitOnAux s "\n" ⟨len P⟩ ⟨len P + len cur⟩ 0 r = r.reverse ++ ((cur ++ rest).splitOn '\n').map String.ofList := by
  induction rest with
  | nil =>
    intro P cur r hs hc
    rw [String.splitOnAux]
    have hend : (⟨len P + len cur⟩ : String.Pos.Raw).atEnd s = true := by
      simp only [String.Pos.Raw.atEnd, size_eq, hs, List.append_nil, len_append, decide_eq_true_eq]; omega
    rw [if_pos hend]
    simp only [List.reverse_cons, List.append_nil]
    rw [extract_eq P cur [] s (by simpa using hs), List.splitOn_eq_singleton hc]
    simp
  | cons c rest ih =>
    intro P cur r hs hc
    rw [String.splitOnAux]
    have hpos := Char.utf8Size_pos c
    have hnot : ¬ (⟨len P + len cur⟩ : String.Pos.Raw).atEnd s = true := by
      simp only [String.Pos.Raw.atEnd, size_eq, hs, len_append, len, decide_eq_true_eq]; omega
    rw [if_neg hnot]
    have hg : String.Pos.Raw.get s ⟨len P + len cur⟩ = c := by
      rw [← len_append]; exact get_at (P ++ cur) c rest s (by simpa using hs)
    have hsep : String.Pos.Raw.get "\n" 0 = '\n' := by decide
    simp only [hg, hsep]
    by_cases hcn : c = '\n'
    · subst hcn
      simp only [beq_self_eq_true, if_true]
      have hj : (String.Pos.Raw.next "\n" 0) = ⟨1⟩ := by decide
      have hje : (String.Pos.Raw.atEnd "\n" ⟨1⟩) = true := by decide
      have hi : String.Pos.Raw.next s ⟨len P + len cur⟩ = ⟨len (P ++ cur ++ ['\n']) + len []⟩ := by
        simp only [String.Pos.Raw.next, hg, len_append, len]
        rfl
      rw [hj, hje, if_pos rfl, hi]
      have hu : (⟨len (P ++ cur ++ ['\n']) + len []⟩ : String.Pos.Raw).unoffsetBy ⟨1⟩ = ⟨len P + len cur⟩ := by
        simp only [String.Pos.Raw.unoffsetBy, len_append, len]
        congr 1
        all_goals simp [Char.utf8Size]
      rw [hu, extract_eq P cur (['\n'] ++ rest) s (by simpa using hs)]
      have := ih (P ++ cur ++ ['\n']) [] (String.ofList cur :: r) (by simpa using hs) (by simp)
      simp only [len, Nat.add_zero, List.nil_append] at this ⊢
      rw [this, List.splitOn_append_cons_self_of_not_mem hc]
      simp
    · have hb : (c == '\n') = false := by simpa using hcn
      simp only [hb]
      have hi : String.Pos.Raw.next s ((⟨len P + len cur⟩ : String.Pos.Raw).unoffsetBy 0) = ⟨len P + len (cur ++ [c])⟩ := by
        simp only [String.Pos.Raw.unoffsetBy, String.Pos.Raw.byteIdx_zero, Nat.sub_zero]
        simp only [String.Pos.Raw.next, hg]
        show (⟨len P + len cur + c.utf8Size⟩ : String.Pos.Raw) = _
        simp only [len_append, len]
        congr 1; omega
      rw [hi, ih P (cur ++ [c]) r (by simpa using hs) (by simp [hc, Ne.symm hcn])]
      simp

theorem splitOn_nl (s : String) : s.splitOn "\n" = (s.toList.splitOn '\n').map String.ofList := by
  unfold String.splitOn
  rw [if_neg (by decide)]
  have := aux s s.toList [] [] [] (by simp) (by simp)
  simpa [len] using this

theorem getAux_found (R : List Char) (i q : Nat) (c : Char) (hc : c ≠ default)
    (h : String.Pos.Raw.utf8GetAux R ⟨i⟩ ⟨q⟩ = c) : ∃ C R', R = C ++ c :: R' ∧ i + len C = q := by
  induction R generalizing i with
  | nil => simp [String.Pos.Raw.utf8GetAux] at h; exact absurd h.symm hc
  | cons d ds ih =>
    simp only [String.Pos.Raw.utf8GetAux] at h
    by_cases hiq : (⟨i⟩ : String.Pos.Raw) = ⟨q⟩
    · rw [if_pos hiq] at h
      subst h
      refine ⟨[], ds, rfl, ?_⟩
      have := congrArg String.Pos.Raw.byteIdx hiq; simp [len] at this ⊢; omega
    · rw [if_neg hiq] at h
      obtain ⟨C, R', h1, h2⟩ := ih (i + d.utf8Size) h
      exact ⟨d :: C, R', by simp [h1], by simp [len] at h2 ⊢; omega⟩

theorem getAux_skip (P R : List Char) (i q : Nat) (hq : i + len P ≤ q) :
    String.Pos.Raw.utf8GetAux (P ++ R) ⟨i⟩ ⟨q⟩ = String.Pos.Raw.utf8GetAux R ⟨i + len P⟩ ⟨q⟩ := by
  induction P generalizing i with
  | nil => simp [len]
  | cons c cs ih =>
    have hpos := Char.utf8Size_pos c
    have hne : (⟨i⟩ : String.Pos.Raw) ≠ ⟨q⟩ := by
      intro h; have := congrArg String.Pos.Raw.byteIdx h; simp only [len] at hq this; omega
    simp only [List.cons_append, String.Pos.Raw.utf8GetAux, if_neg hne]
    have := ih (i + c.utf8Size) (by simp only [len] at hq; omega)
    show String.Pos.Raw.utf8GetAux (cs ++ R) ⟨i + c.utf8Size⟩ ⟨q⟩ = _
    rw [this]
    simp only [len]
    congr 2; omega

/-- The pieces of `text` between the cuts `qs` (byte positions of its `'\n'`s, the first at or past `a`). -/
def chunks (text : String) : Nat → List Nat → List String
  | a, [] => [String.Pos.Raw.extract text ⟨a⟩ ⟨text.utf8ByteSize⟩]
  | a, q :: qs => String.Pos.Raw.extract text ⟨a⟩ ⟨q⟩ :: chunks text (q + 1) qs

/-- Each cut is a `'\n'` at or past the previous one's next byte. -/
def cutsOk (text : String) : Nat → List Nat → Bool
  | _, [] => true
  | a, q :: qs => a ≤ q && String.Pos.Raw.get text ⟨q⟩ == '\n' && cutsOk text (q + 1) qs

theorem chunks_split (text : String) (qs : List Nat) : ∀ (P R : List Char) (a : Nat),
    text.toList = P ++ R → len P = a → cutsOk text a qs = true →
    ((chunks text a qs).map fun c => c.toList.splitOn '\n').flatten = R.splitOn '\n' := by
  induction qs with
  | nil =>
    intro P R a hs ha _
    simp only [chunks, List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil]
    have he := extract_eq P R [] text (by simpa using hs)
    rw [show text.utf8ByteSize = len P + len R by rw [size_eq, hs, len_append], ← ha, he]
    simp
  | cons q qs ih =>
    intro P R a hs ha hok
    simp only [cutsOk, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hok
    obtain ⟨⟨haq, hg⟩, hrest⟩ := hok
    have hg' : String.Pos.Raw.utf8GetAux R ⟨len P⟩ ⟨q⟩ = '\n' := by
      unfold String.Pos.Raw.get at hg
      rw [hs, show (0 : String.Pos.Raw) = ⟨0⟩ from rfl, getAux_skip P R 0 q (by omega)] at hg
      simpa using hg
    obtain ⟨C, R', hR, hq⟩ := getAux_found R (len P) q '\n' (by decide) hg'
    subst hR
    simp only [chunks, List.map_cons, List.flatten_cons]
    have he := extract_eq P C ('\n' :: R') text (by simpa using hs)
    subst ha hq
    rw [he, ih (P ++ C ++ ['\n']) R' (len P + len C + 1) (by simpa using hs)
      (by simp only [len_append, len, show '\n'.utf8Size = 1 by decide]) hrest, List.splitOn_append_cons_self]
    simp

/-- The byte ranges between the cuts `qs`, from `a` to `n`. -/
def ranges (n : Nat) : Nat → List Nat → List (Nat × Nat)
  | a, [] => [(a, n)]
  | a, q :: qs => (a, q) :: ranges n (q + 1) qs

theorem chunks_eq_ranges (text : String) (qs : List Nat) : ∀ a,
    chunks text a qs = (ranges text.utf8ByteSize a qs).map fun r => String.Pos.Raw.extract text ⟨r.1⟩ ⟨r.2⟩ := by
  induction qs with
  | nil => intro a; rfl
  | cons q qs ih => intro a; simp [chunks, ranges, ih]

/-- A piece's lines, each with its numbers when it is a row (`Net.Row.of`). -/
def piece (s : String) : Array String × Array Net.Row :=
  let ls := (s.splitOn "\n").toArray
  (ls, ls.map Net.Row.of)

/-- The pieces of `text` between the cuts `qs`, split in tasks, when `qs` are `'\n'`s in order (`cutsOk`); else the
whole text split in one piece, with no rows read. -/
def splitTasks (text : String) (qs : List Nat) : List (Task (Array String × Array Net.Row)) :=
  if cutsOk text 0 qs then
    (ranges text.utf8ByteSize 0 qs).map fun r => Task.spawn fun _ => piece (String.Pos.Raw.extract text ⟨r.1⟩ ⟨r.2⟩)
  else [Task.pure ((text.splitOn "\n").toArray, #[])]

/-- The lines of the pieces, in order: `(text.splitOn "\n").toArray` (`linesOf_splitTasks`). -/
def linesOf (ts : List (Task (Array String × Array Net.Row))) : Array String :=
  ts.foldl (fun acc t => acc ++ t.get.1) #[]

/-- The rows read beside them, line `k` of `linesOf ts` at `k` when every piece read its rows. Nothing rests on them:
`Net.parseWith` reads a row's numbers from them only when its line is the same string (`Net.parseWith_eq`). -/
def rowsOf (ts : List (Task (Array String × Array Net.Row))) : Array Net.Row :=
  ts.foldl (fun acc t => acc ++ t.get.2) #[]

theorem foldl_tasks {α β γ : Type} (l : List α) (g : α → β) (p : β → Array γ) : ∀ acc : Array γ,
    (l.map fun x => Task.spawn fun _ => g x).foldl (fun acc t => acc ++ p t.get) acc =
      acc ++ (l.flatMap fun x => (p (g x)).toList).toArray := by
  induction l with
  | nil => intro acc; simp
  | cons x xs ih =>
    intro acc; simp only [List.map_cons, List.foldl_cons, List.flatMap_cons, ih]; simp only [Task.spawn, Array.append_assoc]
    congr 1; apply Array.ext'; simp

theorem linesOf_splitTasks (text : String) (qs : List Nat) :
    linesOf (splitTasks text qs) = (text.splitOn "\n").toArray := by
  unfold linesOf splitTasks
  split
  · rename_i hok
    rw [foldl_tasks _ _ Prod.fst]
    have h := chunks_split text qs [] text.toList 0 (by simp) rfl hok
    rw [chunks_eq_ranges] at h
    simp only [piece, Array.empty_append, splitOn_nl text]
    congr 1
    rw [← h]
    simp only [List.map_map, List.flatMap, splitOn_nl, List.map_flatten]
    rfl
  · simp

/-- Cuts for `splitTasks`: about `pieces` `'\n'` positions in order, the first `'\n'` at or after each multiple of the step.
Nothing rests on them: `splitPar` checks them (`cutsOk`) and splits the whole text in one piece when they fail. -/
def cutsNear (text : String) (pieces : Nat) : List Nat := Id.run do
  let n := text.utf8ByteSize
  let step := n / (pieces + 1) + 1
  let mut out : Array Nat := #[]
  let mut p := step
  for _ in [0:pieces] do
    if p ≥ n then break
    let mut q := p
    for _ in [0:step] do
      if q ≥ n || String.Pos.Raw.get text ⟨q⟩ == '\n' then break
      q := q + 1
    if q < n && String.Pos.Raw.get text ⟨q⟩ == '\n' then
      out := out.push q
      p := max (q + 1) (p + step)
    else
      p := p + step
  return out.toList

end Flock.SplitNL
