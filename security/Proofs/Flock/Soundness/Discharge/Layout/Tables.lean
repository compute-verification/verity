module

public import Flock.HmRow
public import Mathlib.Data.Nat.Notation
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk

@[expose] public section

set_option Elab.async false

/-! Under a tag without shared-row files every loaded population has no tables (`loadPublic_tables`); under
`verity/flock-circuit@967b8d06`, which has them, the headline takes `pub.tables = none` as its condition `hpt`. -/

namespace FlockSoundness.Discharge.Layout

open Flock Refine

theorem sharedRows_none {tags : Tags} {c : Circuit} {header : Lean.Json} {r : Option (Array ℕ)}
    (hs : tags.sharedRows = false) (h : HmRow.sharedRows tags c header = .ok r) : r = none := by
  unfold HmRow.sharedRows at h
  dsimp only at h
  simp only [hs, Bool.not_false, ↓reduceIte, throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    exact h.symm

set_option maxHeartbeats 2000000 in
theorem loadPublic_tables {tags : Tags} {c : Circuit} {file : ByteArray} {pop : Public}
    (h : HmRow.loadPublic tags c file = .ok pop) (hs : tags.sharedRows = false) : pop.tables = none := by
  unfold HmRow.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [pure, Except.pure, Except.ok.injEq] at *
    subst_vars
  all_goals first
    | rfl
    | exact absurd (sharedRows_none hs ‹HmRow.sharedRows _ _ _ = .ok (some _)›) (Option.some_ne_none _)

theorem circuitEb90718f_sharedRows : Tags.circuitEb90718f.sharedRows = false := rfl

end FlockSoundness.Discharge.Layout
