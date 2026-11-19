module

public import Properties.Poub.Remote
import Std.Data.HashMap.Lemmas

/-!
# An accepted window's openings

`checkWindow` is a `for` loop with an early return, which `do` elaborates into `forIn` over the indices with state
`(Option Verdict × HashMap Nat Receipter)`. `forIn_good` reads such a loop: when every step either continues with
nothing returned, its index good, or returns a bad value, the loop's result returned nothing and every index is good, or
it returned a bad value. The receipters' map only ever holds a pass's own receipter (`Cached`), so each step checks the
opening under the receipter of its pass's seed (`checkWindow_accept`).
-/

@[expose] public section

open Poub

namespace Properties.Poub

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

@[simp] theorem reject_ne_accept (f : Failure) (at? : Option (Nat × Nat)) : Verdict.reject f at? ≠ Verdict.accept := by
  simp [Verdict.reject, Verdict.accept]

/-- The receipters' map holds only each pass's own receipter. -/
def Cached (reg : Registration) (w : WindowRecord) (m : Std.HashMap Nat Receipter) : Prop :=
  ∀ i r, m.get? i = some r → r = receipterOf reg w i

theorem checkWindow_accept {reg : Registration} {digest : ByteArray} {appends : Array Append} {w : WindowRecord}
    {sampled : Array (Nat × Nat)} {openings : Array Opening}
    (h : checkWindow reg digest appends w sampled openings = .accept) :
    openings.size = sampled.size ∧ ∀ k (hk : k < sampled.size),
      checkOpening reg digest appends w (receipterOf reg w sampled[k].1) sampled[k].1 sampled[k].2 openings[k]! = none := by
  unfold checkWindow at h
  simp only [Id.run] at h
  split at h
  · exact absurd h (reject_ne_accept _ _)
  rename_i hsize
  refine ⟨by simpa using hsize, ?_⟩
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  generalize hL : (forIn (m := Id) (List.range' _ _ _) ((none, ∅) : Option Verdict × Std.HashMap Nat Receipter) _ :
    Id _) = L at h
  have key := hL ▸ forIn_good (Cached reg w)
    (fun k => ∀ hk : k < sampled.size,
      checkOpening reg digest appends w (receipterOf reg w sampled[k].1) sampled[k].1 sampled[k].2 openings[k]! = none)
    (fun v => v ≠ Verdict.accept) _ ?_ _ ∅ (by intro i r hr; simp at hr)
  · intro k m hm
    have hg : m.get? sampled[k]!.1 = none ∨ m.get? sampled[k]!.1 = some (receipterOf reg w sampled[k]!.1) := by
      cases hg : m.get? sampled[k]!.1 with
      | none => exact Or.inl rfl
      | some r => exact Or.inr (by rw [hm _ _ hg])
    have hstep : ∀ hk : k < sampled.size, sampled[k]! = sampled[k] := fun hk => getElem!_pos sampled k hk
    rcases hg with hg | hg <;> simp only [hg] <;> simp only [receipterOf] at hm ⊢ <;>
    cases hc : checkOpening reg digest appends w (receipter reg.receipt (w.seedOf sampled[k]!.1) reg.rows)
      sampled[k]!.1 sampled[k]!.2 openings[k]! <;>
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
    ∃ sampled, receiptSample samples coin (w.passes.map (·.plan.length)) = some sampled ∧
      checkWindow reg reg.digest appends w sampled openings = .accept := by
  unfold verifyWindow at h
  cases hd : receiptSample samples coin (w.passes.map (·.plan.length)) with
  | none => simp [hd] at h
  | some sampled => simp [hd] at h; exact ⟨sampled, rfl, h⟩

end Properties.Poub
