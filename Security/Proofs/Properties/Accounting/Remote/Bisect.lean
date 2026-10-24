import RemoteAudit.Basic

/-!
# Running ends and `bisectRight`

A window's draws and a plan's blocks both locate a position among running ends: `ends[r]` is the sum of the first
`r + 1` lengths (`prefix_fold`), and `bisectRight ends u` the first `r` with `u < ends[r]` (`bisectRight_spec`, by the
loop's invariant: everything below `lo` is at most `u`, everything from `hi` on above it, and `hi − lo` at most the
iterations left). So a position `u` below the total falls in exactly one length, `r = bisectRight ends u`, at offset
`u − ends[r − 1]` (`locate`).
-/

open RemoteAudit

namespace Properties.RemoteAudit

/-- `bisectRight`'s loop body. -/
def bisectStep (ends : Array Nat) (x : Nat) (_ : Nat) (s : Nat × Nat) : Id (ForInStep (Nat × Nat)) :=
  if s.1 < s.2 then
    if x < ends[(s.1 + s.2) / 2]! then pure (.yield (s.1, (s.1 + s.2) / 2)) else pure (.yield ((s.1 + s.2) / 2 + 1, s.2))
  else pure (.yield (s.1, s.2))

theorem bisectRight_eq (ends : Array Nat) (x : Nat) :
    bisectRight ends x = (Id.run (forIn (List.range' 0 (ends.size + 1)) (0, ends.size) (bisectStep ends x))).1 := by
  unfold bisectRight
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range']
  have hsize : ([:ends.size + 1] : Std.Legacy.Range).size = ends.size + 1 := by
    show (ends.size + 1 - 0 + 1 - 1) / 1 = _
    omega
  rw [hsize]
  rfl

/-- `bisectRight`'s invariant: everything below `lo` is at most `x`, everything from `hi` on above it. -/
structure BisectInv (ends : Array Nat) (x lo hi : Nat) : Prop where
  le : lo ≤ hi
  hi_le : hi ≤ ends.size
  below : ∀ a, a < lo → ends[a]! ≤ x
  above : ∀ a, hi ≤ a → a < ends.size → x < ends[a]!

def Monotone' (ends : Array Nat) : Prop := ∀ a b, a ≤ b → b < ends.size → ends[a]! ≤ ends[b]!

theorem bisect_loop (ends : Array Nat) (x : Nat) (mono : Monotone' ends) :
    ∀ (l : List Nat) (lo hi : Nat), BisectInv ends x lo hi → hi - lo ≤ l.length →
      (Id.run (forIn l (lo, hi) (bisectStep ends x))).1 = (Id.run (forIn l (lo, hi) (bisectStep ends x))).2 ∧
        BisectInv ends x (Id.run (forIn l (lo, hi) (bisectStep ends x))).1
          (Id.run (forIn l (lo, hi) (bisectStep ends x))).2 := by
  intro l
  induction l with
  | nil =>
    intro lo hi h hl
    simp only [List.length_nil] at hl
    have := h.le
    exact ⟨by simp only [List.forIn_nil, Id.run, pure]; omega, by simpa only [List.forIn_nil, Id.run, pure] using h⟩
  | cons y l ih =>
    intro lo hi h hl
    simp only [List.length_cons] at hl
    have hle := h.le
    have hhi := h.hi_le
    rw [List.forIn_cons]
    by_cases hlt : lo < hi
    · by_cases hx : x < ends[(lo + hi) / 2]!
      · have hs : bisectStep ends x y (lo, hi) = pure (.yield (lo, (lo + hi) / 2)) := by
          simp only [bisectStep, hlt, hx, ite_true]
        rw [hs]
        exact ih _ _ ⟨by omega, by omega, h.below, fun a ha ha' => Nat.lt_of_lt_of_le hx (mono _ _ ha ha')⟩
          (by omega)
      · have hs : bisectStep ends x y (lo, hi) = pure (.yield ((lo + hi) / 2 + 1, hi)) := by
          simp only [bisectStep, hlt, hx, ite_true, ite_false]
        rw [hs]
        exact ih _ _ ⟨by omega, hhi, fun a ha => Nat.le_trans (mono a ((lo + hi) / 2) (by omega) (by omega)) (by omega), h.above⟩
          (by omega)
    · have hs : bisectStep ends x y (lo, hi) = pure (.yield (lo, hi)) := by
        simp only [bisectStep, hlt, ite_false]
      rw [hs]
      exact ih _ _ h (by omega)

/-- **`bisectRight` is the first index above `x`**, on nondecreasing ends. -/
theorem bisectRight_spec (ends : Array Nat) (x : Nat) (mono : Monotone' ends) :
    bisectRight ends x ≤ ends.size ∧ (∀ a, a < bisectRight ends x → ends[a]! ≤ x) ∧
      ∀ a, bisectRight ends x ≤ a → a < ends.size → x < ends[a]! := by
  rw [bisectRight_eq]
  obtain ⟨_, h⟩ := bisect_loop ends x mono (List.range' 0 (ends.size + 1)) 0 ends.size
    ⟨Nat.zero_le _, Nat.le_refl _, fun a ha => absurd ha (Nat.not_lt_zero a), fun a ha ha' => absurd ha' (by omega)⟩
    (by simp)
  have := h.hi_le
  exact ⟨by omega, h.below, fun a ha ha' => h.above a (by omega) ha'⟩

/-! ## Running ends -/

theorem prefix_fold {β : Type} (g : β → Nat) : ∀ (l : List β) (a : Array Nat) (s : Nat),
    (l.foldl (fun (acc : Array Nat × Nat) x => (acc.1.push (acc.2 + g x), acc.2 + g x)) (a, s)).1.size =
        a.size + l.length ∧
      (∀ i, i < a.size → (l.foldl (fun (acc : Array Nat × Nat) x => (acc.1.push (acc.2 + g x), acc.2 + g x)) (a, s)).1[i]! = a[i]!) ∧
      (∀ i, i < l.length →
        (l.foldl (fun (acc : Array Nat × Nat) x => (acc.1.push (acc.2 + g x), acc.2 + g x)) (a, s)).1[a.size + i]! =
          s + ((l.take (i + 1)).map g).sum) := by
  intro l
  induction l with
  | nil => intro a s; simp
  | cons y l ih =>
    intro a s
    obtain ⟨h1, h2, h3⟩ := ih (a.push (s + g y)) (s + g y)
    simp only [List.foldl_cons, Array.size_push] at h1 h2 h3 ⊢
    refine ⟨by rw [h1]; simp; omega, fun i hi => ?_, fun i hi => ?_⟩
    · rw [h2 i (by omega)]
      simp [Array.getElem_push, hi, Nat.lt_succ_of_lt hi]
    · cases i with
      | zero =>
        rw [Nat.add_zero, h2 a.size (by omega)]
        simp
      | succ i =>
        have := h3 i (by simp at hi; omega)
        rw [show a.size + (i + 1) = a.size + 1 + i by omega, this]
        simp [List.take_succ_cons, Nat.add_assoc]

/-- `ends` holds the running sums of `l`. -/
def PrefixSums (ends : Array Nat) (l : List Nat) : Prop :=
  ends.size = l.length ∧ ∀ i, i < l.length → ends[i]! = (l.take (i + 1)).sum

theorem prefixSums_fold {β : Type} (g : β → Nat) (xs : Array β) :
    PrefixSums (xs.foldl (fun (acc : Array Nat × Nat) x => (acc.1.push (acc.2 + g x), acc.2 + g x)) (#[], 0)).1
      (xs.toList.map g) := by
  rw [← Array.foldl_toList]
  obtain ⟨h1, -, h3⟩ := prefix_fold g xs.toList #[] 0
  refine ⟨by simpa using h1, fun i hi => ?_⟩
  simp only [List.length_map] at hi
  have := h3 i hi
  simp only [List.size_toArray, List.length_nil, Nat.zero_add] at this
  simpa [List.map_take] using this

theorem back_getD_of_prefix {ends : Array Nat} {l : List Nat} (h : PrefixSums ends l) : ends.back?.getD 0 = l.sum := by
  have hs := h.1
  rcases Nat.eq_zero_or_pos l.length with h0 | h0
  · rw [Array.back?_eq_none_iff.2 (Array.size_eq_zero_iff.1 (by omega)), List.length_eq_zero_iff.1 h0]
    rfl
  · rw [Array.back?_eq_getElem?, Array.getElem?_eq_getElem (by omega)]
    simp only [Option.getD_some]
    rw [← getElem!_pos ends (ends.size - 1) (by omega), h.2 (ends.size - 1) (by omega),
      show ends.size - 1 + 1 = l.length by omega, List.take_length]

theorem sum_take_mono (l : List Nat) (a b : Nat) (h : a ≤ b) : (l.take a).sum ≤ (l.take b).sum := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [List.take_add, List.sum_append]
  omega

theorem sum_take_succ' (l : List Nat) (i : Nat) (hi : i < l.length) : (l.take (i + 1)).sum = (l.take i).sum + l[i] := by
  rw [List.take_add_one, List.sum_append]
  simp [List.getElem?_eq_getElem hi]

theorem prefixSums_mono {ends : Array Nat} {l : List Nat} (h : PrefixSums ends l) : Monotone' ends := by
  intro a b hab hb
  rw [h.2 a (by rw [← h.1]; omega), h.2 b (by rw [← h.1]; omega)]
  exact sum_take_mono _ _ _ (by omega)

/-- **A position below the total falls in length `r = bisectRight ends u`, at offset `u − ends[r − 1]`.** -/
theorem locate {ends : Array Nat} {l : List Nat} (h : PrefixSums ends l) (u : Nat) (hu : u < l.sum) :
    bisectRight ends u < l.length ∧ (l.take (bisectRight ends u)).sum ≤ u ∧
      u < (l.take (bisectRight ends u)).sum + l[bisectRight ends u]! ∧
      (if bisectRight ends u = 0 then 0 else ends[bisectRight ends u - 1]!) = (l.take (bisectRight ends u)).sum := by
  obtain ⟨hle, hbelow, habove⟩ := bisectRight_spec ends u (prefixSums_mono h)
  have hsz := h.1
  generalize bisectRight ends u = r at hle hbelow habove ⊢
  have hlast : r < l.length := by
    rcases Nat.lt_or_ge r l.length with hr | hr
    · exact hr
    · exfalso
      have hpos : 0 < l.length := by
        rcases Nat.eq_zero_or_pos l.length with h0 | h0
        · rw [List.length_eq_zero_iff.1 h0] at hu; simp at hu
        · exact h0
      have := hbelow (l.length - 1) (by omega)
      rw [h.2 _ (by omega), show l.length - 1 + 1 = l.length by omega, List.take_length] at this
      omega
  have hoff : (if r = 0 then 0 else ends[r - 1]!) = (l.take r).sum := by
    split
    · rename_i h0; subst h0; simp
    · rw [h.2 _ (by omega), show r - 1 + 1 = r by omega]
  refine ⟨hlast, ?_, ?_, hoff⟩
  · rw [← hoff]
    split
    · exact Nat.zero_le _
    · exact hbelow _ (by omega)
  · have := habove r (Nat.le_refl _) (by omega)
    rw [h.2 r hlast, sum_take_succ' l r hlast] at this
    rw [getElem!_pos l r hlast]
    exact this

end Properties.RemoteAudit
