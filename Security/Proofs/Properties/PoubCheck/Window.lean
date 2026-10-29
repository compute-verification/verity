import Properties.PoubCheck

/-!
# Proofs of the window's properties

`Properties.PoubCheck`'s `AcceptChecksOpenings`, `AcceptOpensDraws` and `ChunkedIsWhole`.

`checkWindow` is a `for` loop with an early return, which `do` elaborates into `forIn` over the indices with state
`(Option Verdict × HashMap Nat Receipter)`. `forIn_good` reads such a loop: when every step either continues with
nothing returned, its index good, or returns a bad value, the loop's result returned nothing and every index is good, or
it returned a bad value. The receipters' map only ever holds a pass's own receipter (`Cached`), so each step checks the
opening under the receipter of its pass's seed. `checkOpening_none` reads checks 1–5 off `checkOpening`'s early returns.

`forIn_findSome` reads such a loop exactly: its result is the first value over the list. So `checkWindow` is its first
rejection in draw order (`checkWindow_eq`, `rejectAt`), a chunk `[a, b)` checked alone is the first rejection among the
window's draws there (`checkWindow_extract`), and `checkWindowChunked`'s chunks, in order, cover the draws
(`chunks_cover`): its first chunk that rejects holds the window's first rejection (`chunkedIsWhole`). A spawned task's
value is its function's (`Task.spawn`'s definition), so the threads change nothing.
-/

open PoubCheck

namespace Properties.PoubCheck

/-- `ByteArray`'s `==` is its arrays', which is lawful. -/
instance lawfulBEqByteArray : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

/-- A loop with an early return, whose every step from nothing returned either continues with nothing returned and its
index good, or returns a bad value: it returns nothing with every index good, or a bad value. -/
theorem forIn_good {α β σ : Type} (Inv : σ → Prop) (Good : α → Prop) (Bad : β → Prop)
    (f : α → Option β × σ → Id (ForInStep (Option β × σ)))
    (hf : ∀ a m, Inv m → (∃ m', Inv m' ∧ Good a ∧ f a (none, m) = pure (.yield (none, m'))) ∨
      (∃ b m', Bad b ∧ f a (none, m) = pure (.done (some b, m')))) :
    ∀ (l : List α) m, Inv m →
      ((forIn l (none, m) f).run.1 = none ∧ ∀ a ∈ l, Good a) ∨ (∃ b, (forIn l (none, m) f).run.1 = some b ∧ Bad b) := by
  intro l
  induction l with
  | nil => intro m _; left; exact ⟨rfl, by simp⟩
  | cons a as ih =>
    intro m hm
    rw [List.forIn_cons]
    rcases hf a m hm with ⟨m', hm', ha, hfa⟩ | ⟨b, m', hb, hfa⟩
    · rw [hfa]
      simp only [pure_bind]
      rcases ih m' hm' with ⟨h1, h2⟩ | h
      · left
        refine ⟨h1, fun x hx => ?_⟩
        rcases List.mem_cons.1 hx with rfl | hx
        · exact ha
        · exact h2 x hx
      · right; exact h
    · rw [hfa]
      right
      exact ⟨b, rfl, hb⟩

/-- The receipters' map holds only each pass's own receipter. -/
def Cached (reg : Registration) (w : WindowRecord) (m : Std.HashMap Nat Receipter) : Prop :=
  ∀ i r, m.get? i = some r → r = receipterOf reg w i

theorem checkWindow_accept {reg : Registration} {digest : ByteArray} {appends : Array Append} {w : WindowRecord}
    {drawn : Array (Nat × Nat)} {openings : Array Opening}
    (h : checkWindow reg digest appends w drawn openings = .accept) :
    openings.size = drawn.size ∧ ∀ k (hk : k < drawn.size),
      checkOpening reg digest appends w (receipterOf reg w drawn[k].1) drawn[k].1 drawn[k].2 openings[k]! = none := by
  unfold checkWindow at h
  simp only [Id.run] at h
  split at h
  · simp at h
  rename_i hsize
  refine ⟨by simpa using hsize, ?_⟩
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  generalize hL : (forIn (m := Id) (List.range' _ _ _) ((none, ∅) : Option Verdict × Std.HashMap Nat Receipter) _ :
    Id _) = L at h
  have key := hL ▸ forIn_good (Cached reg w)
    (fun k => ∀ hk : k < drawn.size,
      checkOpening reg digest appends w (receipterOf reg w drawn[k].1) drawn[k].1 drawn[k].2 openings[k]! = none)
    (fun v => v ≠ Verdict.accept) _ ?_ _ ∅ (by intro i r hr; simp at hr)
  · intro k m hm
    have hg : m.get? drawn[k]!.1 = none ∨ m.get? drawn[k]!.1 = some (receipterOf reg w drawn[k]!.1) := by
      cases hg : m.get? drawn[k]!.1 with
      | none => exact Or.inl rfl
      | some r => exact Or.inr (by rw [hm _ _ hg])
    have hstep : ∀ hk : k < drawn.size, drawn[k]! = drawn[k] := fun hk => getElem!_pos drawn k hk
    rcases hg with hg | hg <;> simp only [hg] <;> simp only [receipterOf] at hm ⊢ <;>
    cases hc : checkOpening reg digest appends w (receipter reg.receipt (w.seedOf drawn[k]!.1) reg.rows)
      drawn[k]!.1 drawn[k]!.2 openings[k]! <;>
    first
      | (right; exact ⟨_, _, by simp, rfl⟩)
      | (left
         refine ⟨_, ?_, fun hk => by rw [← hstep hk]; exact hc, rfl⟩
         intro i r hi
         rw [Std.HashMap.get?_insert] at hi
         split at hi
         · rename_i heq
           simp at heq hi
           subst heq; exact hi.symm
         · exact hm i r hi)
  · rcases key with ⟨h1, h2⟩ | ⟨b, h1, h2⟩
    · intro k hk
      refine h2 k ?_ hk
      simp [Std.Legacy.Range.size, List.mem_range'_1]
      omega
    · exfalso
      simp only [bind, Id.run] at h h1
      rw [h1] at h
      exact h2 h

theorem verifyWindow_accept {reg : Registration} {appends : Array Append} {w : WindowRecord} {coin : ByteArray}
    {samples : Nat} {openings : Array Opening} (h : verifyWindow reg appends w coin samples openings = some .accept) :
    ∃ drawn, draws reg.digest w.index coin samples (w.passes.map (·.plan.length)) = some drawn ∧
      checkWindow reg reg.digest appends w drawn openings = .accept := by
  unfold verifyWindow at h
  cases hd : draws reg.digest w.index coin samples (w.passes.map (·.plan.length)) with
  | none => simp [hd] at h
  | some drawn => simp [hd] at h; exact ⟨drawn, rfl, h⟩

theorem acceptChecksOpenings : AcceptChecksOpenings := by
  intro reg appends w coin samples openings h
  obtain ⟨drawn, hd, hc⟩ := verifyWindow_accept h
  exact ⟨drawn, hd, checkWindow_accept hc⟩

theorem checkOpening_none {reg : Registration} {appends : Array Append} {w : WindowRecord} {i p : Nat} {o : Opening}
    (h : checkOpening reg reg.digest appends w (receipterOf reg w i) i p o = none) :
    OpeningPasses reg appends w i p o := by
  unfold checkOpening at h
  simp only [Id.run] at h
  unfold OpeningPasses
  dsimp only
  generalize w.passes[i]!.plan.block p = j at h ⊢
  split at h
  · simp at h
  rename_i h1
  simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, ne_eq, Decidable.not_not, Array.any_eq_true, not_exists]
    at h1
  obtain ⟨⟨⟨⟨⟨⟨hpass, hpos⟩, hblock⟩, htile⟩, hls⟩, hps⟩, hpath⟩ := h1
  have hpaths : ∀ x ∈ o.receiptPath ++ o.payloadPath, x.size = DIGEST_BYTES := by
    intro x hx
    obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hx
    exact hpath k hk
  refine ⟨hpass, hpos, hblock, htile, hls, hps, fun x hx => hpaths x (Array.mem_append.2 (Or.inl hx)),
    fun x hx => hpaths x (Array.mem_append.2 (Or.inr hx)), ?_⟩
  by_cases hb : j < reg.blocks
  · simp only [hb, ite_true] at h ⊢
    split at h
    · simp at h
    rename_i h2
    refine ⟨Decidable.not_not.1 h2, ?_⟩
    split at h
    · rename_i y hy
      split at h
      · simp at h
      rename_i h3
      refine ⟨y, hy, by simpa using h3, ?_⟩
      split at h
      · rename_i payload hdec
        split at h
        · simp at h
        rename_i h4
        split at h
        · simp at h
        rename_i h5
        exact ⟨⟨payload, Or.inl ⟨trivial, hdec, by simpa using h4⟩⟩, by simpa using h5⟩
      · simp at h
    · simp at h
  · simp only [hb, ite_false] at h ⊢
    split at h
    · simp at h
    rename_i h2
    refine ⟨Decidable.not_not.1 h2, ?_⟩
    split at h
    · rename_i y hy
      split at h
      · simp at h
      rename_i h3
      refine ⟨y, hy, by simpa using h3, ?_⟩
      cases ha : appendOf appends j with
      | none => simp [ha] at h
      | some a =>
        simp only [ha] at h
        split at h
        · rename_i payload hdec
          split at h
          · simp at h
          rename_i h4
          split at h
          · simp at h
          rename_i h5
          exact ⟨⟨payload, Or.inr ⟨Nat.le_of_not_lt hb, a, rfl, hdec, by simpa using h4⟩⟩, by simpa using h5⟩
        · simp at h
    · simp at h

theorem acceptOpensDraws : AcceptOpensDraws := by
  intro reg appends w coin samples openings h
  obtain ⟨drawn, hd, hsize, hk⟩ := acceptChecksOpenings reg appends w coin samples openings h
  exact ⟨drawn, hd, hsize, fun k hk' => checkOpening_none (hk k hk')⟩

/-! ## The chunked check -/

/-- A loop with an early return whose every step from nothing returned either continues with nothing returned where
`g` is none, or returns `g`'s value: its result is the first of `g`'s values over the list. -/
theorem forIn_findSome {α β σ : Type} (Inv : σ → Prop) (g : α → Option β)
    (f : α → Option β × σ → Id (ForInStep (Option β × σ)))
    (hf : ∀ a m, Inv m → (g a = none ∧ ∃ m', Inv m' ∧ f a (none, m) = pure (.yield (none, m'))) ∨
      (∃ b m', g a = some b ∧ f a (none, m) = pure (.done (some b, m')))) :
    ∀ (l : List α) m, Inv m → (forIn l (none, m) f).run.1 = l.findSome? g := by
  intro l
  induction l with
  | nil => intro m _; rfl
  | cons a as ih =>
    intro m hm
    rw [List.forIn_cons, List.findSome?_cons]
    rcases hf a m hm with ⟨hg, m', hm', hfa⟩ | ⟨b, m', hg, hfa⟩
    · rw [hfa, hg]
      exact ih m' hm'
    · rw [hfa, hg]
      rfl

theorem findSome?_congr {α β : Type} {f g : α → Option β} :
    ∀ {l : List α}, (∀ x ∈ l, f x = g x) → l.findSome? f = l.findSome? g
  | [], _ => rfl
  | a :: as, h => by
    rw [List.findSome?_cons, List.findSome?_cons, h a (List.mem_cons_self ..),
      findSome?_congr fun x hx => h x (List.mem_cons_of_mem _ hx)]

theorem findSome?_flatMap {α β γ : Type} (F : α → List β) (g : β → Option γ) :
    ∀ l : List α, (l.flatMap F).findSome? g = l.findSome? fun a => (F a).findSome? g
  | [] => rfl
  | a :: as => by
    rw [List.flatMap_cons, List.findSome?_append, List.findSome?_cons, findSome?_flatMap F g as]
    cases (F a).findSome? g <;> rfl

/-- Draw `k`'s rejection as `checkWindow` writes it, or none when its opening checks. -/
def rejectAt (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord)
    (drawn : Array (Nat × Nat)) (openings : Array Opening) (k : Nat) : Option Verdict :=
  (checkOpening reg digest appends w (receipterOf reg w drawn[k]!.1) drawn[k]!.1 drawn[k]!.2 openings[k]!).map
    fun f => .reject f (some (drawn[k]!.1, drawn[k]!.2))

theorem rejectAt_reject {reg : Registration} {digest : ByteArray} {appends : Array Append} {w : WindowRecord}
    {drawn : Array (Nat × Nat)} {openings : Array Opening} {k : Nat} {v : Verdict}
    (h : rejectAt reg digest appends w drawn openings k = some v) : ∃ f at?, v = .reject f at? := by
  unfold rejectAt at h
  cases hc : checkOpening reg digest appends w (receipterOf reg w drawn[k]!.1) drawn[k]!.1 drawn[k]!.2 openings[k]! <;>
    simp [hc] at h
  exact ⟨_, _, h.symm⟩

/-- `checkWindow` is its first rejection in draw order (`rejectAt`), or accept. -/
theorem checkWindow_eq (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord)
    (drawn : Array (Nat × Nat)) (openings : Array Opening) :
    checkWindow reg digest appends w drawn openings =
      if openings.size ≠ drawn.size then .reject .malformed none
      else ((List.range drawn.size).findSome? (rejectAt reg digest appends w drawn openings)).getD .accept := by
  unfold checkWindow
  simp only [Id.run]
  split
  · rfl
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  generalize hL : (forIn (m := Id) (List.range' _ _ _) ((none, ∅) : Option Verdict × Std.HashMap Nat Receipter) _ :
    Id _) = L
  have key := hL ▸ forIn_findSome (Cached reg w) (rejectAt reg digest appends w drawn openings) _ ?_ _ ∅
    (by intro i r hr; simp at hr)
  · intro k m hm
    have hg : m.get? drawn[k]!.1 = none ∨ m.get? drawn[k]!.1 = some (receipterOf reg w drawn[k]!.1) := by
      cases hg : m.get? drawn[k]!.1 with
      | none => exact Or.inl rfl
      | some r => exact Or.inr (by rw [hm _ _ hg])
    have hc : Cached reg w (m.insert drawn[k]!.1 (receipterOf reg w drawn[k]!.1)) := by
      intro i r hi
      rw [Std.HashMap.get?_insert] at hi
      split at hi
      · rename_i heq
        simp at heq hi
        subst heq; exact hi.symm
      · exact hm i r hi
    unfold rejectAt
    rcases hg with hg | hg <;> simp only [hg] <;> simp only [receipterOf] at hc ⊢ <;>
    cases hco : checkOpening reg digest appends w (receipter reg.receipt (w.seedOf drawn[k]!.1) reg.rows)
      drawn[k]!.1 drawn[k]!.2 openings[k]! <;>
    first
      | (right; exact ⟨_, _, by simp, rfl⟩)
      | (left; exact ⟨by simp, _, hc, rfl⟩)
  · have hr : List.range' [:drawn.size].start [:drawn.size].size [:drawn.size].step = List.range drawn.size := by
      simp [Std.Legacy.Range.size, List.range_eq_range']
    rw [hr] at key
    rcases L with ⟨o, m⟩
    simp only [Id.run] at key
    subst key
    cases List.findSome? (rejectAt reg digest appends w drawn openings) (List.range drawn.size) <;> rfl

/-- A window's draws `[a, b)` and their openings, checked alone, give the first rejection among those draws in the
whole window's order. -/
theorem checkWindow_extract (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord)
    (drawn : Array (Nat × Nat)) (openings : Array Opening) (hsize : openings.size = drawn.size) (a b : Nat) :
    checkWindow reg digest appends w (drawn.extract a b) (openings.extract a b) =
      ((List.range' a (min b drawn.size - a)).findSome? (rejectAt reg digest appends w drawn openings)).getD
        .accept := by
  rw [checkWindow_eq]
  simp only [Array.size_extract, hsize, ne_eq, not_true_eq_false, ↓reduceIte]
  rw [List.range'_eq_map_range, List.findSome?_map]
  refine congrArg (Option.getD · Verdict.accept) (findSome?_congr fun k hk => ?_)
  have hk : k < min b drawn.size - a := List.mem_range.1 hk
  have hd : (drawn.extract a b)[k]! = drawn[a + k]! := by
    rw [getElem!_pos _ _ (by rw [Array.size_extract]; omega), getElem!_pos _ _ (by omega), Array.getElem_extract]
  have ho : (openings.extract a b)[k]! = openings[a + k]! := by
    rw [getElem!_pos _ _ (by rw [Array.size_extract]; omega), getElem!_pos _ _ (by omega), Array.getElem_extract]
  simp only [rejectAt, Function.comp, hd, ho]

/-- The first `m` chunks of `s` draws each, of `n`, cover draws `[0, min (m·s) n)` in order. -/
theorem chunks_cover (n s : Nat) :
    ∀ m, (List.range m).flatMap (fun c => List.range' (c * s) (min ((c + 1) * s) n - c * s)) =
      List.range (min (m * s) n)
  | 0 => by simp
  | m + 1 => by
    rw [List.range_succ, List.flatMap_append, chunks_cover n s m, List.flatMap_singleton]
    by_cases h : m * s ≤ n
    · have happ := List.range'_append_1 (s := 0) (m := m * s) (n := min ((m + 1) * s) n - m * s)
      rw [Nat.zero_add] at happ
      rw [Nat.min_eq_left h, List.range_eq_range', List.range_eq_range', happ]
      congr 1
      rw [Nat.add_mul, Nat.one_mul]
      omega
    · rw [show min ((m + 1) * s) n - m * s = 0 by rw [Nat.add_mul, Nat.one_mul]; omega, List.range'_zero,
        List.append_nil]
      congr 1
      rw [Nat.add_mul, Nat.one_mul]
      omega

theorem le_ceil_mul (n s : Nat) (hs : 0 < s) : n ≤ (n + s - 1) / s * s := by
  have h1 := Nat.div_add_mod (n + s - 1) s
  have h2 := Nat.mod_lt (n + s - 1) hs
  rw [Nat.mul_comm] at h1
  omega

theorem chunkedIsWhole : ChunkedIsWhole := by
  intro size reg digest appends w drawn openings
  rw [checkWindow_eq]
  unfold checkWindowChunked
  simp only [Id.run]
  split
  · rfl
  rename_i hsize
  have hsize : openings.size = drawn.size := by simpa using hsize
  generalize hs : max 1 size = s
  have hs0 : 0 < s := by omega
  generalize hL : (forIn (m := Id) (List.map _ _) ((none, ()) : Option Verdict × Unit) _ : Id _) = L
  have key := hL ▸ forIn_findSome (fun _ => True)
    (fun t : Task Verdict => match t.get with | .reject f at? => some (Verdict.reject f at?) | .accept => none)
    _ ?_ _ () trivial
  · intro t m _
    cases h : t.get with
    | accept => left; simp
    | reject f at? => right; simp
  · rw [List.findSome?_map] at key
    have hc : ∀ c, ((fun t : Task Verdict => match t.get with
          | .reject f at? => some (Verdict.reject f at?) | .accept => none) ∘ fun c =>
        Task.spawn (fun _ => checkWindow reg digest appends w (drawn.extract (c * s) ((c + 1) * s))
          (openings.extract (c * s) ((c + 1) * s))) Task.Priority.dedicated) c =
        (List.range' (c * s) (min ((c + 1) * s) drawn.size - c * s)).findSome?
          (rejectAt reg digest appends w drawn openings) := by
      intro c
      simp only [Function.comp, Task.spawn]
      rw [checkWindow_extract reg digest appends w drawn openings hsize]
      cases h : (List.range' (c * s) (min ((c + 1) * s) drawn.size - c * s)).findSome?
          (rejectAt reg digest appends w drawn openings) with
      | none => rfl
      | some v =>
        obtain ⟨k, _, hk⟩ := List.exists_of_findSome?_eq_some h
        obtain ⟨f, at?, rfl⟩ := rejectAt_reject hk
        rfl
    rw [findSome?_congr (fun c _ => hc c), ← findSome?_flatMap, chunks_cover,
      Nat.min_eq_right (le_ceil_mul _ _ hs0)] at key
    rcases L with ⟨o, u⟩
    simp only [Id.run] at key
    subst key
    cases List.findSome? (rejectAt reg digest appends w drawn openings) (List.range drawn.size) <;> rfl

end Properties.PoubCheck
