import Properties.RemoteAudit.Line
import Mathlib.Algebra.Order.BigOperators.Group.List
import Mathlib.Algebra.BigOperators.Ring.List

/-!
# The memory line's arithmetic link, proved (`Properties.RemoteAudit.Line`: `LineChecked`, `StretchBound`,
`LineRegistered`)

`line` runs `check`, then `stretches` at each form's charge. `check_ok` reads `check`'s refusals off its `if … then
throw`s (`ite_throw_ok`, `ite_throw_bind_ok`), and `line_spec` reads `line`'s result as its two forms. `stretches` is a
`for` loop that refuses or pushes one stretch a step: `forIn_push` reads such a loop, and `stretches_spec` gives each
stretch it pushes as `Spec`, the code's own expressions (`sA` … `capOf`) with its inner fill loop as `greedy`.

The rest compares those expressions with stretch `m`'s paced session (`stretchPace`): the code's arrays at a group of
the record are its groups' fields (`sA_get` … `gapA_get`); the paced session's group sizes, coins, `stop` and `longest`
are the code's (`stretch_size` … `stretch_longest`, the code's `L` reading one more group, `lo − 1`); its rates are the
code's or below (`stretch_rate`, the closing group's `L/n` at the paced session's `L`). The greedy fill highest rate
first is the fractional knapsack's optimum (`le_fillR`, over `fillR`, the greedy in the reals, `greedy_cast`), so the
model's `fill` is at most the code's (`stretch_fill`). `stretch_bound_of` puts the terms together: the code's `lost` is
`S + (2·L·R + fill)/span` with `S` the chips' blocks, `chip/b` rounded up, as `ChipCeiling` reads them.
-/

namespace Properties.RemoteAudit.Line

open _root_.RemoteAudit.Line Accounting.Remote.Held

/-- A loop in `Except` whose every step pushes one value `x` with `P a x` (or refuses): each value of its result was in
the initial array or pushed at some `a ∈ l`. -/
theorem forIn_push {ε α β : Type} (l : List α) (f : α → Array β → Except ε (ForInStep (Array β)))
    (P : α → β → Prop)
    (hf : ∀ a acc r, f a acc = .ok r → ∃ x, r = .yield (acc.push x) ∧ P a x) :
    ∀ (init out : Array β), forIn l init f = .ok out → ∀ x ∈ out, x ∈ init ∨ ∃ a ∈ l, P a x := by
  induction l with
  | nil =>
    intro init out h x hx
    simp only [List.forIn_nil] at h
    cases h
    exact Or.inl hx
  | cons a l ih =>
    intro init out h x hx
    rw [List.forIn_cons] at h
    cases hfa : f a init with
    | error e => rw [hfa] at h; cases h
    | ok r =>
      rw [hfa] at h
      obtain ⟨y, rfl, hy⟩ := hf a init r hfa
      have h' : forIn l (init.push y) f = .ok out := h
      rcases ih _ _ h' x hx with hx' | ⟨b, hb, hP⟩
      · rcases Array.mem_push.1 hx' with h1 | h1
        · exact Or.inl h1
        · exact Or.inr ⟨a, List.mem_cons_self .., h1 ▸ hy⟩
      · exact Or.inr ⟨b, List.mem_cons_of_mem _ hb, hP⟩

/-- The code's fill loop's state `(left, fill)` after the groups `l`. -/
def greedy (v cap : Nat → Rat) : List Nat → Rat × Rat → Rat × Rat
  | [], st => st
  | g :: l, st => if st.1 ≤ 0 then st else greedy v cap l (st.1 - min st.1 (cap g), st.2 + min st.1 (cap g) * v g)

/-- The code's fill loop is `greedy`. -/
theorem forIn_greedy (v cap : Nat → Rat) (l : List Nat) (st : Rat × Rat) :
    forIn (m := Except String) l st (fun g s => if s.1 ≤ 0 then pure (ForInStep.done (s.1, s.2))
      else pure (ForInStep.yield (s.1 - min s.1 (cap g), s.2 + min s.1 (cap g) * v g))) =
      pure (greedy v cap l st) := by
  induction l generalizing st with
  | nil => rfl
  | cons g l ih =>
    rw [List.forIn_cons]
    by_cases h : st.1 ≤ 0
    · simp [h, greedy]
    · simp [h, greedy, ih]

def sA (gs : Array Group) : Array Nat := gs.map (·.releasedNs)
def nA (gs : Array Group) : Array Nat := gs.map (·.passes)
def eA (gs : Array Group) : Array Nat :=
  (Array.range gs.size).map fun g => if g + 1 < gs.size then max gs[g]!.lastNs (sA gs)[g + 1]! else gs[g]!.lastNs
def LA (gs : Array Group) : Array Nat := (Array.range gs.size).map fun g => (eA gs)[g]! - (sA gs)[g]!
def gapA (gs : Array Group) : Array Nat :=
  (Array.range gs.size).map fun g => if g + 1 < gs.size then (sA gs)[g + 1]! - (sA gs)[g]! else (LA gs)[g]!
def startOf (gs : Array Group) (per m : Nat) : Nat := if m = 0 then (eA gs)[0]! else (sA gs)[m * per]!
def lMaxOf (gs : Array Group) (per m : Nat) : Nat :=
  (List.range' (m * per - 1) ((m + 1) * per + 1 - (m * per - 1))).foldl (fun a g => max a (LA gs)[g]!) 0
def vOf (gs : Array Group) (per m : Nat) (g : Nat) : Rat :=
  ((if g = (m + 1) * per then lMaxOf gs per m else (gapA gs)[g]!) : Rat) / ((nA gs)[g]! : Rat)
def orderOf (gs : Array Group) (per m : Nat) : List Nat :=
  (List.range' (m * per) ((m + 1) * per + 1 - m * per)).mergeSort fun g h => decide (vOf gs per m h ≤ vOf gs per m g)
def capOf (gs : Array Group) (N : Nat) (g : Nat) : Rat := (((nA gs)[g]! * N : Nat) : Rat)

/-- Stretch `x` as `stretches` pushes it at step `m`: the code's expressions for its span, `L`, charge back, fill, lost
blocks and weights. -/
structure Spec (gs : Array Group) (per N b chip : Nat) (W : Rat) (c : Charge) (m : Nat) (x : Stretch) : Prop where
  stretch : x.stretch = m
  nonempty : startOf gs per m < (sA gs)[(m + 1) * per]!
  span : x.spanNs = (sA gs)[(m + 1) * per]! - startOf gs per m
  lMax : x.lMaxNs = lMaxOf gs per m
  rebuild : x.rebuild = c.rate * ((x.spanNs : Rat) + 2 * (x.lMaxNs : Rat)) + c.burst
  fill : x.fill = (greedy (vOf gs per m) (capOf gs N) (orderOf gs per m) (W, 0)).2
  lost : x.time.lost =
    (((chip + b - 1) / b : Nat) : Rat) + (2 * (x.lMaxNs : Rat) * x.rebuild + x.fill) / (x.spanNs : Rat)
  weights : x.time.weights = max 0 (x.weights - x.time.lost) * (b : Rat)

/-- Every stretch `stretches` returns is `Spec` at a step `m < G/per − 1`. -/
theorem stretches_spec {gs : Array Group} {per N b chip : Nat} {W : Rat} {c : Charge} {out : Array Stretch}
    (h : stretches gs per N b chip W c = .ok out) :
    ∀ x ∈ out, ∃ m, m < gs.size / per - 1 ∧ Spec gs per N b chip W c m x := by
  unfold stretches at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', bind_pure] at h
  have key := forIn_push _ _ (fun m x => Spec gs per N b chip W c m x) ?hf _ _ h
  · intro x hx
    rcases key x hx with h0 | ⟨m, hm, hP⟩
    · simp at h0
    · exact ⟨m, by simpa using hm, hP⟩
  case hf =>
    intro m acc r hr
    by_cases hm0 : m = 0
    · subst hm0
      simp only [↓reduceIte] at hr
      split at hr
      · exact absurd hr (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
      · rename_i hne
        rw [forIn_greedy] at hr
        simp only [pure_bind] at hr
        cases hr
        exact ⟨_, rfl, ⟨rfl, Nat.lt_of_not_le hne, rfl, rfl, rfl, rfl, rfl, rfl⟩⟩
    · simp only [hm0, ↓reduceIte] at hr
      split at hr
      · exact absurd hr (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
      · rename_i hne
        rw [forIn_greedy] at hr
        simp only [pure_bind] at hr
        cases hr
        refine ⟨_, rfl, ⟨rfl, ?_, ?_, rfl, rfl, rfl, rfl, rfl⟩⟩
        · simp only [startOf, hm0, ↓reduceIte]; exact Nat.lt_of_not_le hne
        · simp only [startOf, hm0, ↓reduceIte]; rfl


/-- The greedy fill's value from budget `a`, in the reals. -/
noncomputable def fillR (v cap : ℕ → ℝ) : List ℕ → ℝ → ℝ
  | [], _ => 0
  | g :: l, a => if a ≤ 0 then 0 else min a (cap g) * v g + fillR v cap l (a - min a (cap g))

theorem greedy_cast (v cap : ℕ → ℚ) (l : List ℕ) (a f : ℚ) :
    ((greedy v cap l (a, f)).2 : ℝ) = f + fillR (fun g => v g) (fun g => cap g) l a := by
  induction l generalizing a f with
  | nil => simp [greedy, fillR]
  | cons g l ih =>
    by_cases h : a ≤ 0
    · have h' : (a : ℝ) ≤ 0 := by exact_mod_cast h
      simp [greedy, fillR, h, h']
    · have h' : ¬ (a : ℝ) ≤ 0 := by exact_mod_cast h
      simp only [greedy, fillR, h, h', ite_false]
      rw [ih]
      push_cast
      ring

variable {v cap : ℕ → ℝ}

theorem fillR_nonpos {l : List ℕ} {a : ℝ} (ha : a ≤ 0) : fillR v cap l a = 0 := by
  cases l with
  | nil => rfl
  | cons g l => simp [fillR, ha]

/-- At most `a · r` from budget `a ≥ 0` when every rate is at most `r`. -/
theorem fillR_le (hc : ∀ g, 0 ≤ cap g) {r : ℝ} (hr : 0 ≤ r) :
    ∀ (l : List ℕ), (∀ g ∈ l, v g ≤ r) → ∀ a, 0 ≤ a → fillR v cap l a ≤ a * r := by
  intro l
  induction l with
  | nil => intro _ a ha; simp [fillR]; positivity
  | cons g l ih =>
    intro hl a ha
    by_cases h : a ≤ 0
    · simp [fillR, h]; positivity
    · simp only [fillR, h, ite_false]
      have ht0 : 0 ≤ min a (cap g) := le_min ha (hc g)
      have hta : min a (cap g) ≤ a := min_le_left _ _
      have h1 := ih (fun h hh => hl h (List.mem_cons_of_mem _ hh)) (a - min a (cap g)) (by linarith)
      have h2 : min a (cap g) * v g ≤ min a (cap g) * r :=
        mul_le_mul_of_nonneg_left (hl g List.mem_cons_self) ht0
      nlinarith

/-- The greedy's value grows at most `r` per unit of budget when every rate is at most `r`. -/
theorem fillR_slope (hc : ∀ g, 0 ≤ cap g) {r : ℝ} (hr : 0 ≤ r) :
    ∀ (l : List ℕ), (∀ g ∈ l, v g ≤ r) → ∀ a d, 0 ≤ d → fillR v cap l (a + d) ≤ fillR v cap l a + d * r := by
  intro l
  induction l with
  | nil => intro _ a d hd; simp [fillR]; positivity
  | cons g l ih =>
    intro hl a d hd
    have hl' : ∀ h ∈ l, v h ≤ r := fun h hh => hl h (List.mem_cons_of_mem _ hh)
    have hg := hl g List.mem_cons_self
    by_cases h1 : a + d ≤ 0
    · rw [fillR_nonpos h1, fillR_nonpos (by linarith)]
      positivity
    by_cases h2 : a ≤ 0
    · rw [fillR_nonpos h2]
      simp only [fillR, h1, ite_false]
      have ht0 : 0 ≤ min (a + d) (cap g) := le_min (by linarith) (hc g)
      have hta : min (a + d) (cap g) ≤ a + d := min_le_left _ _
      have := fillR_le hc hr l hl' (a + d - min (a + d) (cap g)) (by linarith)
      have h3 : min (a + d) (cap g) * v g ≤ min (a + d) (cap g) * r := mul_le_mul_of_nonneg_left hg ht0
      nlinarith
    · simp only [fillR, h1, h2, ite_false]
      set t := min a (cap g)
      set t' := min (a + d) (cap g)
      have hδ0 : t ≤ t' := min_le_min_right _ (by linarith)
      have hδd : t' - t ≤ d := by
        simp only [t, t']
        rcases le_total a (cap g) with h | h
        · rw [min_eq_left h]
          have := min_le_left (a + d) (cap g)
          linarith
        · rw [min_eq_right h, min_eq_right (by linarith)]
          linarith
      have hih := ih hl' (a - t) (d - (t' - t)) (by linarith)
      have heq : a - t + (d - (t' - t)) = a + d - t' := by ring
      rw [heq] at hih
      have h3 : (t' - t) * v g ≤ (t' - t) * r := mul_le_mul_of_nonneg_left hg (by linarith)
      nlinarith

/-- The greedy over groups taken highest rate first attains the fractional knapsack's optimum: every point within the
caps and the budget is worth at most the greedy's value. -/
theorem le_fillR (hc : ∀ g, 0 ≤ cap g) (hv : ∀ g, 0 ≤ v g) :
    ∀ (l : List ℕ), l.Pairwise (fun g h => v h ≤ v g) → ∀ (x : ℕ → ℝ), (∀ g ∈ l, 0 ≤ x g ∧ x g ≤ cap g) →
      ∀ a, (l.map x).sum ≤ a → (l.map fun g => v g * x g).sum ≤ fillR v cap l a := by
  intro l
  induction l with
  | nil => intro _ x _ a _; simp [fillR]
  | cons g l ih =>
    intro hs x hx a ha
    rw [List.pairwise_cons] at hs
    have hx' : ∀ h ∈ l, 0 ≤ x h ∧ x h ≤ cap h := fun h hh => hx h (List.mem_cons_of_mem _ hh)
    have hxg := hx g List.mem_cons_self
    have hsum0 : 0 ≤ (l.map x).sum := List.sum_nonneg fun y hy => by
      obtain ⟨h, hh, rfl⟩ := List.mem_map.1 hy
      exact (hx' h hh).1
    simp only [List.map_cons, List.sum_cons] at ha ⊢
    -- the rest is worth at most `v g` per unit of its total
    have hrest : (l.map fun h => v h * x h).sum ≤ v g * (l.map x).sum := by
      rw [← List.sum_map_mul_left]
      refine List.sum_le_sum fun h hh => ?_
      exact mul_le_mul_of_nonneg_right (hs.1 h hh) (hx' h hh).1
    by_cases h : a ≤ 0
    · rw [fillR_nonpos h]
      have : x g + (l.map x).sum ≤ 0 := ha.trans h
      have hxg0 : x g = 0 := by linarith [hxg.1]
      have hl0 : (l.map x).sum = 0 := by linarith [hxg.1]
      rw [hl0, mul_zero] at hrest
      rw [hxg0, mul_zero]
      linarith
    · simp only [fillR, h, ite_false]
      set t := min a (cap g)
      have hxt : x g ≤ t := le_min (by linarith) hxg.2
      have hih := ih hs.2 x hx' (a - t + (t - x g)) (by linarith)
      have hsl := fillR_slope hc (hv g) l (fun h hh => hs.1 h hh) (a - t) (t - x g) (by linarith)
      nlinarith


theorem ex_throw_bind {ε α β : Type} (e : ε) (f : α → Except ε β) : (throw e >>= f : Except ε β) = throw e := rfl

theorem ite_throw_ok {ε α : Type} {c : Prop} [Decidable c] {e : ε} {r : Except ε α} {a : α}
    (h : (if c then throw e else r) = .ok a) : ¬ c ∧ r = .ok a := by
  by_cases hc : c
  · simp only [hc, ↓reduceIte] at h; cases h
  · simp only [hc, ↓reduceIte] at h; exact ⟨hc, h⟩

theorem ite_throw_bind_ok {ε α β : Type} {c : Prop} [Decidable c] {e : ε} {x : α} {k : α → Except ε β} {a : β}
    (h : ((if c then throw e else pure x) >>= k) = .ok a) : ¬ c ∧ k x = .ok a := by
  by_cases hc : c
  · simp only [hc, ↓reduceIte] at h; cases h
  · simp only [hc, ↓reduceIte] at h; exact ⟨hc, h⟩

/-- What `check` accepting establishes of the rates and the bound on the wrong receipts. -/
theorem check_ok {gs : Array Group} {per blocks blockBytes nodeBytes : ℕ} {wrong : ℚ} {rates : Rates} {u : Unit}
    (h : check gs per blocks blockBytes nodeBytes wrong rates = .ok u) :
    1 ≤ blockBytes ∧ 1 ≤ rates.encodeSquarings ∧ 0 ≤ rates.squaringsPerS ∧ 0 ≤ rates.diskBytesPerS ∧
      0 ≤ rates.linkBytesPerS ∧ 0 ≤ rates.rebuildBurstBytes ∧ 0 ≤ rates.diskBurstBytes ∧
      0 ≤ rates.linkBurstBytes ∧ 0 ≤ wrong := by
  unfold check at h
  simp only [Rates.named, List.forIn_cons, List.forIn_nil, ex_throw_bind, bind_assoc] at h
  obtain ⟨h1, h⟩ := ite_throw_ok h
  obtain ⟨-, h⟩ := ite_throw_ok h
  obtain ⟨-, h⟩ := ite_throw_ok h
  obtain ⟨h4, h⟩ := ite_throw_ok h
  obtain ⟨h5, h⟩ := ite_throw_bind_ok h
  simp only [bind_assoc] at h
  obtain ⟨h6, h⟩ := ite_throw_bind_ok h
  simp only [bind_assoc] at h
  obtain ⟨h7, h⟩ := ite_throw_bind_ok h
  simp only [bind_assoc] at h
  obtain ⟨h8, h⟩ := ite_throw_bind_ok h
  simp only [bind_assoc] at h
  obtain ⟨h9, h⟩ := ite_throw_bind_ok h
  simp only [bind_assoc] at h
  obtain ⟨h10, h⟩ := ite_throw_bind_ok h
  simp only [pure_bind] at h
  obtain ⟨h11, -⟩ := ite_throw_ok h
  exact ⟨Nat.not_lt.1 h1, Nat.not_lt.1 h4, not_lt.1 h5, not_lt.1 h6, not_lt.1 h7,
    not_lt.1 h8, not_lt.1 h9, not_lt.1 h10, not_lt.1 h11⟩

/-- The pooled bound is read only at a `δ` above 0 and below 1. -/
theorem pooled_delta {ws : Array (ℕ × ℕ)} {δ : ℚ} {b : _root_.RemoteAudit.Pooled.Bound}
    (h : _root_.RemoteAudit.Pooled.pooled ws δ = .ok b) : 0 < δ ∧ δ < 1 := by
  by_contra hc
  have hδ : δ ≤ 0 ∨ 1 ≤ δ := by
    rcases le_or_gt δ 0 with h0 | h0
    · exact Or.inl h0
    · exact Or.inr (not_lt.1 fun h1 => hc ⟨h0, h1⟩)
  simp only [_root_.RemoteAudit.Pooled.pooled, hδ, if_true] at h
  by_cases hs : ws.size = 0
  · simp only [hs, if_true] at h
    cases h
  · simp only [hs, if_false] at h
    cases h

/-- A line the code computes read its pooled bound at a `δ` above 0 and below 1 (`Pooled.pooled` refuses any other), so
a node whose `δ` is a nonnegative rational (`Node.delta`) leaves out no line the code computes. -/
theorem line_delta {gs : Array Group} {per blocks blockBytes chipBytes nodeBytes : ℕ} {ws : Array (ℕ × ℕ)} {δ : ℚ}
    {rates : Rates} {l : Line} (h : line gs per blocks blockBytes chipBytes nodeBytes ws δ rates = .ok l) :
    0 < δ ∧ δ < 1 := by
  unfold line at h
  cases hp : _root_.RemoteAudit.Pooled.pooled ws δ with
  | error e => rw [hp] at h; cases h
  | ok bd => exact pooled_delta hp

/-- A line `line` computes: `check` accepted the wrong receipts' bound it read (`l.wrong`), and its forms are
`stretches` at each form's charge and that bound. -/
theorem line_spec {gs : Array Group} {per blocks blockBytes chipBytes nodeBytes : ℕ} {ws : Array (ℕ × ℕ)} {δ : ℚ}
    {rates : Rates} {l : Line} (h : line gs per blocks blockBytes chipBytes nodeBytes ws δ rates = .ok l) :
    check gs per blocks blockBytes nodeBytes l.wrong rates = .ok () ∧
    ∃ ts ls, stretches gs per blocks blockBytes chipBytes l.wrong (charges rates blockBytes).1 = .ok ts ∧
      stretches gs per blocks blockBytes chipBytes l.wrong (charges rates blockBytes).2 = .ok ls ∧
      l.forms = #[⟨"time-held", (charges rates blockBytes).1, ts⟩,
        ⟨"link-charged", (charges rates blockBytes).2, ls⟩] := by
  unfold line at h
  cases hp : _root_.RemoteAudit.Pooled.pooled ws δ with
  | error e => rw [hp] at h; cases h
  | ok bd =>
    have hp' : _root_.RemoteAudit.Pooled.pooled ws δ = pure bd := hp
    rw [hp', pure_bind] at h
    dsimp only at h
    cases hc : check gs per blocks blockBytes nodeBytes bd.wrong rates with
    | error e => rw [hc] at h; cases h
    | ok u =>
      rw [hc] at h
      cases ht : stretches gs per blocks blockBytes chipBytes bd.wrong (charges rates blockBytes).1 with
      | error e => simp only [ht] at h; cases h
      | ok ts =>
        cases hl : stretches gs per blocks blockBytes chipBytes bd.wrong (charges rates blockBytes).2 with
        | error e => simp only [ht, hl] at h; cases h
        | ok ls =>
          simp only [ht, hl] at h
          cases h
          exact ⟨hc, ts, ls, ht, hl, rfl⟩

section Facts

variable {gs : Array Group}

theorem sA_get {j : ℕ} (hj : j < gs.size) : (sA gs)[j]! = gs[j]!.releasedNs := by
  simp [sA, hj]

theorem nA_get {j : ℕ} (hj : j < gs.size) : (nA gs)[j]! = gs[j]!.passes := by
  simp [nA, hj]

theorem eA_get {j : ℕ} (hj : j < gs.size) : (eA gs)[j]! = ended gs j := by
  by_cases h1 : j + 1 < gs.size
  · simp [eA, ended, hj, h1, sA_get h1]
  · simp [eA, ended, hj, h1]

theorem LA_get {j : ℕ} (hj : j < gs.size) : (LA gs)[j]! = ended gs j - gs[j]!.releasedNs := by
  simp [LA, hj, eA_get hj, sA_get hj]

theorem gapA_get {j : ℕ} (hj : j + 1 < gs.size) :
    (gapA gs)[j]! = gs[j + 1]!.releasedNs - gs[j]!.releasedNs := by
  have hj' : j < gs.size := by omega
  simp [gapA, hj', hj, sA_get hj, sA_get hj']

end Facts

section Pace

variable {gs : Array Group} {per m : ℕ}

theorem firstPass_succ (k : ℕ) : firstPass gs (k + 1) = firstPass gs k + gs[k]!.passes := by
  simp [firstPass, List.range_succ]

theorem stretch_size (g : ℕ) : (stretchPace (fun _ : Unit => gs) per m).size () g = gs[m * per + g]!.passes := by
  simp only [Pace.size, stretchPace, ← Nat.add_assoc, firstPass_succ]
  omega

theorem stretch_released (hhi : (m + 1) * per < gs.size) {g : ℕ} (hg : g ≤ per) :
    (stretchPace (fun _ : Unit => gs) per m).released () g = gs[m * per + g]!.releasedNs := by
  simp only [stretchPace, released]
  congr 2
  have : m * per + g ≤ (m + 1) * per := by rw [Nat.succ_mul]; omega
  omega

theorem stretch_stop (hhi : (m + 1) * per < gs.size) :
    (stretchPace (fun _ : Unit => gs) per m).stop () = spanStop gs per m := by
  show (stretchPace (fun _ : Unit => gs) per m).released () per = _
  rw [stretch_released hhi le_rfl, spanStop, Nat.succ_mul]

theorem le_foldl_max (f : ℕ → ℕ) : ∀ (l : List ℕ) (init : ℕ),
    init ≤ l.foldl (fun a g => max a (f g)) init ∧ ∀ x ∈ l, f x ≤ l.foldl (fun a g => max a (f g)) init := by
  intro l
  induction l with
  | nil => intro init; simp
  | cons y l ih =>
    intro init
    obtain ⟨h1, h2⟩ := ih (max init (f y))
    refine ⟨(le_max_left _ _).trans h1, ?_⟩
    intro x hx
    rcases List.mem_cons.1 hx with rfl | hx
    · exact (le_max_right _ _).trans h1
    · exact h2 x hx

theorem stretch_longest (hhi : (m + 1) * per < gs.size) :
    (stretchPace (fun _ : Unit => gs) per m).longest () ≤ lMaxOf gs per m := by
  simp only [Pace.longest]
  refine Finset.sup_le fun g hg => ?_
  have hg' : g ≤ per := Nat.lt_succ_iff.1 (Finset.mem_range.1 hg)
  have hlo : m * per + g ≤ (m + 1) * per := by rw [Nat.succ_mul]; omega
  rw [stretch_released hhi hg']
  have hL := LA_get (gs := gs) (j := m * per + g) (by omega)
  have := (le_foldl_max (fun g => (LA gs)[g]!) (List.range' (m * per - 1) ((m + 1) * per + 1 - (m * per - 1)))
    0).2 (m * per + g) (by
    rw [List.mem_range'_1]; omega)
  simp only [stretchPace]
  rw [← hL]
  exact this

theorem vOf_nonneg (g : ℕ) : (0 : ℝ) ≤ (vOf gs per m g : ℝ) := by
  simp only [vOf]
  push_cast
  positivity

theorem stretch_rate (hhi : (m + 1) * per < gs.size) {g : ℕ} (hg : g ≤ per) :
    (stretchPace (fun _ : Unit => gs) per m).rate () g ≤ (vOf gs per m (m * per + g) : ℝ) := by
  have hlo : m * per + g ≤ (m + 1) * per := by rw [Nat.succ_mul]; omega
  simp only [Pace.rate, stretch_size, vOf]
  by_cases hgp : g < per
  · have hne : m * per + g ≠ (m + 1) * per := by rw [Nat.succ_mul]; omega
    have hc : (stretchPace (fun _ : Unit => gs) per m).credited = per := rfl
    rw [ite_eq_left (hc ▸ hgp), ite_eq_right hne, stretch_released hhi hgp, stretch_released hhi (by omega),
      gapA_get (by omega), nA_get (by omega)]
    simp only [Nat.succ_eq_add_one, ← Nat.add_assoc]
    push_cast
    exact le_rfl
  · have hgp' : g = per := by omega
    have hc : (stretchPace (fun _ : Unit => gs) per m).credited = per := rfl
    have heq : m * per + g = (m + 1) * per := by rw [Nat.succ_mul]; omega
    rw [ite_eq_right (hc ▸ hgp), ite_eq_left heq, nA_get (by omega)]
    push_cast
    exact div_le_div_of_nonneg_right (by exact_mod_cast stretch_longest hhi) (by positivity)

end Pace

section Fill

variable {gs : Array Group} {per m : ℕ}

theorem sum_range_list (F : ℕ → ℝ) (n : ℕ) : ∑ g ∈ Finset.range n, F g = ((List.range n).map F).sum := by
  induction n with
  | zero => simp
  | succ n ih => rw [Finset.sum_range_succ, List.range_succ, List.map_append, List.sum_append, ih]; simp

theorem order_len : (m + 1) * per + 1 - m * per = per + 1 := by rw [Nat.succ_mul]; omega

theorem order_perm : (orderOf gs per m).Perm (List.range' (m * per) (per + 1)) := by
  simp only [orderOf, order_len]
  exact List.mergeSort_perm _ _

theorem mem_order {j : ℕ} : j ∈ orderOf gs per m ↔ m * per ≤ j ∧ j < m * per + (per + 1) := by
  rw [order_perm.mem_iff, List.mem_range'_1]

theorem order_sum (F : ℕ → ℝ) :
    ((orderOf gs per m).map F).sum = ∑ g ∈ Finset.range (per + 1), F (m * per + g) := by
  rw [(order_perm.map F).sum_eq, List.range'_eq_map_range, List.map_map, sum_range_list]
  rfl

theorem order_sorted : (orderOf gs per m).Pairwise (fun g h => (vOf gs per m h : ℝ) ≤ vOf gs per m g) := by
  have := List.pairwise_mergeSort (le := fun g h => decide (vOf gs per m h ≤ vOf gs per m g))
    (fun a b c hab hbc => by
      simp only [decide_eq_true_eq] at hab hbc ⊢
      exact hbc.trans hab)
    (fun a b => by
      simp only [Bool.or_eq_true, decide_eq_true_eq]
      exact le_total _ _)
    (List.range' (m * per) ((m + 1) * per + 1 - m * per))
  refine this.imp fun h => ?_
  simp only [decide_eq_true_eq] at h
  exact_mod_cast h

/-- The model's `fill` of stretch `m`'s paced session is at most the code's greedy fill. -/
theorem stretch_fill (hhi : (m + 1) * per < gs.size) {N : ℕ} {W : ℚ} (hW : 0 ≤ W) :
    (stretchPace (fun _ : Unit => gs) per m).fill () N W ≤
      fillR (fun g => (vOf gs per m g : ℝ)) (fun g => (capOf gs N g : ℝ)) (orderOf gs per m) W := by
  unfold Pace.fill
  have hc : (stretchPace (fun _ : Unit => gs) per m).credited = per := rfl
  refine csSup_le ⟨_, 0, ⟨fun g => ⟨le_rfl, by positivity⟩, by simp; exact_mod_cast hW⟩, rfl⟩ ?_
  rintro _ ⟨x, ⟨hx, hsum⟩, rfl⟩
  rw [hc] at hsum ⊢
  have hy : ∀ g, x (m * per + g - m * per) = x g := fun g => by simp
  calc ∑ g ∈ Finset.range (per + 1), (stretchPace (fun _ : Unit => gs) per m).rate () g * x g
      ≤ ∑ g ∈ Finset.range (per + 1), (vOf gs per m (m * per + g) : ℝ) * x g :=
        Finset.sum_le_sum fun g hg =>
          mul_le_mul_of_nonneg_right (stretch_rate hhi (Nat.lt_succ_iff.1 (Finset.mem_range.1 hg))) (hx g).1
    _ = ((orderOf gs per m).map fun j => (vOf gs per m j : ℝ) * x (j - m * per)).sum := by
        rw [order_sum]; simp only [hy]
    _ ≤ _ := by
        refine le_fillR (fun g => by simp only [capOf]; positivity) (fun g => vOf_nonneg g) _ order_sorted
          (fun j => x (j - m * per)) (fun j hj => ?_) W ?_
        · obtain ⟨h1, h2⟩ := mem_order.1 hj
          refine ⟨(hx _).1, (hx _).2.trans ?_⟩
          have hj' : j < gs.size := by
            have : m * per + per = (m + 1) * per := by rw [Nat.succ_mul]
            omega
          rw [stretch_size, show m * per + (j - m * per) = j by omega, capOf, nA_get hj']
          push_cast
          exact le_rfl
        · rw [order_sum]; simpa only [hy] using hsum

end Fill

/-- Each stretch `stretches` returns at charge `c`, a nonnegative rate and burst, meets `StretchBound`'s conjuncts. -/
theorem stretch_bound_of {gs : Array Group} {per N b chip : ℕ} {W : ℚ} {c : Charge} {out : Array Stretch}
    {x : Stretch} (hb : 1 ≤ b) (hW : 0 ≤ W) (hcr : 0 ≤ c.rate) (hcb : 0 ≤ c.burst)
    (h : stretches gs per N b chip W c = .ok out) (hx : x ∈ out) :
    (x.stretch + 1) * per < gs.size ∧
      spanStart gs per x.stretch < spanStop gs per x.stretch ∧
      x.spanNs = spanStop gs per x.stretch - spanStart gs per x.stretch ∧
      spanStop gs per x.stretch = (stretchPace (fun _ : Unit => gs) per x.stretch).stop () ∧
      (spanStart gs per x.stretch = (stretchPace (fun _ : Unit => gs) per x.stretch).ended () 0 ∨
        spanStart gs per x.stretch = (stretchPace (fun _ : Unit => gs) per x.stretch).released () 0) ∧
      (x.spanNs : ℝ) * ((N : ℝ) - x.time.lost) ≤
        (x.spanNs : ℝ) * ((N : ℝ) - ((chip + b - 1) / b : ℕ))
          - 2 * (stretchPace (fun _ : Unit => gs) per x.stretch).longest () *
              ((c.rate : ℝ) * ((x.spanNs : ℝ) + 2 * (stretchPace (fun _ : Unit => gs) per x.stretch).longest ()) +
                c.burst)
          - (stretchPace (fun _ : Unit => gs) per x.stretch).fill () N W ∧
      x.time.weights = max 0 (x.weights - x.time.lost) * (b : ℚ) := by
  obtain ⟨m, hm, sp⟩ := stretches_spec h x hx
  rw [sp.stretch]
  have hper : 1 ≤ per := by
    rcases Nat.eq_zero_or_pos per with h0 | h0
    · subst h0; simp at hm
    · exact h0
  have hhi : (m + 1) * per < gs.size := by
    have h2 : (m + 2) * per ≤ gs.size :=
      (Nat.mul_le_mul_right per (by omega : m + 2 ≤ gs.size / per)).trans (Nat.div_mul_le_self _ _)
    have : (m + 1) * per < (m + 2) * per := Nat.mul_lt_mul_of_pos_right (by omega) hper
    omega
  have hstart : startOf gs per m = spanStart gs per m := by
    by_cases hm0 : m = 0
    · simp only [startOf, spanStart, hm0, ↓reduceIte]
      exact eA_get (by omega)
    · simp only [startOf, spanStart, hm0, ↓reduceIte]
      exact sA_get (by have : m * per ≤ (m + 1) * per := Nat.mul_le_mul_right _ (by omega); omega)
  have hstop : (sA gs)[(m + 1) * per]! = spanStop gs per m := sA_get hhi
  have hspan : x.spanNs = spanStop gs per m - spanStart gs per m := by rw [sp.span, hstart, hstop]
  have hne : spanStart gs per m < spanStop gs per m := by have := sp.nonempty; rwa [hstart, hstop] at this
  refine ⟨hhi, hne, hspan, (stretch_stop hhi).symm, ?_, ?_, sp.weights⟩
  · by_cases hm0 : m = 0
    · left
      subst hm0
      simp [spanStart, stretchPace]
    · right
      rw [stretch_released hhi (Nat.zero_le _)]
      simp [spanStart, hm0]
  · set K := stretchPace (fun _ : Unit => gs) per m
    have hs0 : (0 : ℝ) < x.spanNs := by rw [hspan]; exact_mod_cast Nat.sub_pos_of_lt hne
    have hb0 : (0 : ℝ) < b := by exact_mod_cast hb
    have hL : (K.longest () : ℝ) ≤ x.lMaxNs := by rw [sp.lMax]; exact_mod_cast stretch_longest hhi
    have hR : (x.rebuild : ℝ) = (c.rate : ℝ) * ((x.spanNs : ℝ) + 2 * (x.lMaxNs : ℝ)) + c.burst := by
      rw [sp.rebuild]; push_cast; ring
    have hlost : (x.spanNs : ℝ) * (x.time.lost : ℝ) =
        x.spanNs * (((chip + b - 1) / b : ℕ) : ℝ) + (2 * (x.lMaxNs : ℝ) * x.rebuild + x.fill) := by
      rw [sp.lost]; push_cast; field_simp
    have hfill : K.fill () N W ≤ (x.fill : ℝ) := by
      rw [sp.fill, greedy_cast, Rat.cast_zero, zero_add]
      exact stretch_fill hhi hW
    have hcr' : (0 : ℝ) ≤ c.rate := by exact_mod_cast hcr
    have hcb' : (0 : ℝ) ≤ c.burst := by exact_mod_cast hcb
    have hL0 : (0 : ℝ) ≤ K.longest () := Nat.cast_nonneg _
    have hin : (c.rate : ℝ) * ((x.spanNs : ℝ) + 2 * K.longest ()) + c.burst ≤ x.rebuild := by
      rw [hR]
      have := mul_le_mul_of_nonneg_left (show (x.spanNs : ℝ) + 2 * K.longest () ≤ x.spanNs + 2 * x.lMaxNs by
        linarith) hcr'
      linarith
    have hin0 : (0 : ℝ) ≤ (c.rate : ℝ) * ((x.spanNs : ℝ) + 2 * K.longest ()) + c.burst := by positivity
    have h2 : 2 * (K.longest () : ℝ) * ((c.rate : ℝ) * ((x.spanNs : ℝ) + 2 * K.longest ()) + c.burst) ≤
        2 * (x.lMaxNs : ℝ) * x.rebuild :=
      mul_le_mul (by linarith) hin hin0 (by positivity)
    rw [mul_sub, hlost]
    linarith

/-- **LineChecked**, proved. -/
theorem lineChecked : LineChecked := by
  intro gs per blocks blockBytes chipBytes nodeBytes windows δ rates l h
  exact check_ok (line_spec h).1

theorem charges_nonneg {rates : Rates} {b : ℕ} (hb : 1 ≤ b) (hE : 1 ≤ rates.encodeSquarings)
    (hs : 0 ≤ rates.squaringsPerS) (hd : 0 ≤ rates.diskBytesPerS) (hl : 0 ≤ rates.linkBytesPerS)
    (hr : 0 ≤ rates.rebuildBurstBytes) (hbd : 0 ≤ rates.diskBurstBytes) (hbl : 0 ≤ rates.linkBurstBytes) :
    0 ≤ (charges rates b).1.rate ∧ 0 ≤ (charges rates b).1.burst ∧
      0 ≤ (charges rates b).2.rate ∧ 0 ≤ (charges rates b).2.burst := by
  simp only [charges, ns]
  refine ⟨by positivity, by positivity, by positivity, by positivity⟩

/-- **The arithmetic link** (`StretchBound`), proved. -/
theorem stretchBound : StretchBound := by
  intro gs per blocks blockBytes chipBytes nodeBytes windows δ rates l h i hi f hf x hx
  obtain ⟨hck, ts, ls, hts, hls, hforms⟩ := line_spec h
  rw [hforms] at hf
  obtain ⟨hb, hE, hs, hd, hl, hr, hbd, hbl, hW⟩ := check_ok hck
  obtain ⟨h1r, h1b, h2r, h2b⟩ := charges_nonneg hb hE hs hd hl hr hbd hbl (rates := rates) (b := blockBytes)
  rcases (by omega : i = 0 ∨ i = 1) with rfl | rfl
  · simp at hf
    subst hf
    obtain ⟨a1, a2, a3, a4, a5, a6, a7⟩ := stretch_bound_of hb hW h1r h1b hts hx
    exact ⟨a1, a2, a3, a4, a5, by simp, a6, a7⟩
  · simp at hf
    subst hf
    obtain ⟨a1, a2, a3, a4, a5, a6, a7⟩ := stretch_bound_of hb hW h2r h2b hls hx
    exact ⟨a1, a2, a3, a4, a5, by simp, a6, a7⟩

local instance lawfulBEqByteArrayLine : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-- **LineRegistered**, proved: `lineOfRegistered` refuses unless `unregistered` finds nothing, and then it is
`lineOfRecordOn`. -/
theorem lineRegistered : LineRegistered := by
  intro onRecord g reg digest sm samples seen per blockBytes chipBytes nodeBytes δ rates l h
  by_cases hd : digest = reg.digest
  · by_cases hb : sm.blocks = reg.blocks
    · refine ⟨hd, hb, ?_⟩
      simpa [lineOfRegistered, unregistered, hd, hb] using h
    · simp [lineOfRegistered, unregistered, hd, hb] at h
  · simp [lineOfRegistered, unregistered, hd] at h

end Properties.RemoteAudit.Line
