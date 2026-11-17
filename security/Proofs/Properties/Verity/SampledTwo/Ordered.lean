module

public import Proofs.Properties.Verity.SampledTwo.Steps

/-! Where `stagesOrdered` passed on two windows, the stages' sessions are in order on one ledger. -/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)
open Spot

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem stagesOrdered_order {p : Params} {e : Evidence} (h : stagesOrdered p e = .ok ())
    {k : Spot.Challenger.ChallengerKey} {label w₁ w₂ : String} (hk : p.audit.base.challenger = some (k, label))
    (h1 : p.boundary.verifier.windowRoot = some w₁) (h2 : p.audit.base.verifier.windowRoot = some w₂)
    (hne : w₁ ≠ w₂) :
    ∃ first second, Spot.getD e.coins "ledger" = .arr first ∧
      Spot.getD e.interior.base.coins "ledger" = .arr second ∧ Spot.Challenger.extendsHead first second = true ∧
      ∃ wins ms, Spot.Challenger.checked (.arr second) k = .ok (wins, ms) ∧
        ∃ en r, (wins.find? (·.root == w₁)).bind (·.ended) = some en ∧
          ((wins.find? (·.root == w₂)).bind fun w => (second.filterMap Spot.Challenger.ledgerEntry?).find? fun l =>
            l.kind == "reserve" && l.session == w.opened.session) = some r ∧ en.seq < r.seq := by
  have hb : (w₁ == w₂) = false := by simpa using hne
  unfold stagesOrdered at h
  rw [hk, h1, h2] at h
  simp only [hb, Bool.false_eq_true, ↓reduceIte] at h
  generalize hf : Spot.getD e.coins "ledger" = f at h
  generalize hs : Spot.getD e.interior.base.coins "ledger" = s at h
  cases f <;> cases s <;> first | (cases h; done) | skip
  case arr.arr first second =>
    first
      | dsimp only at h
      | simp only at h
      | skip
    cases hx : Spot.Challenger.extendsHead first second
    · (simp only [hx, Bool.not_false, ↓reduceIte] at h) <;> cases h
    · simp only [hx, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
      obtain ⟨⟨wins, ms⟩, hc, h⟩ := bind_ok h
      try dsimp only at h
      split at h
      · cases h
      · rename_i heq
        obtain ⟨en, r, he, hr, hlt⟩ := orderReason_none heq
        first
          | exact ⟨first, second, rfl, rfl, hx, wins, ms, hc, en, r, he, hr, hlt⟩
          | exact ⟨first, second, hf, hs, hx, wins, ms, hc, en, r, he, hr, hlt⟩

end VStar.Recursive.SampledTwo
