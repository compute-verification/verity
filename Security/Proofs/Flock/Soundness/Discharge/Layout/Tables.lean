import Proofs.Flock.Soundness.Discharge.Layout.Drawn

set_option Elab.async false

/-!
# A population without shared rows, from the statement's tag

`bc_drawn` and the headline's sites read the registered population's `b ‖ c` as `n · np` consecutive pieces of its
public file, which holds for a file without shared rows (`pop.tables = none`; `BcSite` and `bc_drawnS` take any
population, `Layout.SharedRows`). `HmRow.sharedRows` refuses a
header's `shared_rows` unless the statement's tag has shared-row files, so under any tag without them every population
`HmRow.loadPublic` accepts has no tables (`loadPublic_tables`; `@eb90718f` by `rfl`, `circuitEb90718f_sharedRows`).
`verity/flock-circuit@967b8d06`, the tag the circuit prover's sessions on OS coins verify under, has them, so the
headline takes `pub.tables = none` as a condition on the loaded file (`hpt`), which the circuit prover's per-instance
public files meet.
-/

namespace FlockSoundness.Discharge.Layout

open Flock Refine

/-- Under a tag without shared-row files, `sharedRows` returns none whenever it returns: a header with `shared_rows`
is refused. -/
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
/-- **A population loaded under a tag without shared-row files has no tables**: `HmRow.loadPublic` reads the header's
`shared_rows` through `sharedRows`, which refuses them under such a tag, and builds tables only from them. -/
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
