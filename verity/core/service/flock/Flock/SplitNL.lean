/-
`String.splitOn s "\n"` (the legacy split every circuit parse starts from) as `List.splitOn '\n'` on `s.toList`
(`splitOn_nl`), with the byte positions it walks: the model a parse that splits a file's lines elsewhere (in tasks, from
byte offsets) is held to.
-/

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

end Flock.SplitNL
