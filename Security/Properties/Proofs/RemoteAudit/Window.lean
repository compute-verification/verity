import Properties.RemoteAudit

/-!
# Proofs of the window's properties (`Properties.RemoteAudit`: `AcceptChecksOpenings`, `AcceptOpensDraws`)

`checkWindow` is a `for` loop with an early return, which `do` elaborates into `forIn` over the indices with state
`(Option Verdict × HashMap Nat Receipter)`. `forIn_good` reads such a loop: when every step either continues with
nothing returned, its index good, or returns a bad value, the loop's result returned nothing and every index is good, or
it returned a bad value. The receipters' map only ever holds a pass's own receipter (`Cached`), so each step checks the
opening under the receipter of its pass's seed. `checkOpening_none` reads checks 1–5 off `checkOpening`'s early returns.
-/

open RemoteAudit

namespace Properties.RemoteAudit

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

end Properties.RemoteAudit
