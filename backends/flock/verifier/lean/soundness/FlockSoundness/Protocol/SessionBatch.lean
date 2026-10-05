import FlockSoundness.Protocol.CompiledSound
import FlockSoundness.Protocol.Extract
import FlockSoundness.Protocol.Game.Batch
import FlockSoundness.Protocol.Game.Prob

/-!
# Several tables in one batched session (PROTOCOL.md §5)

The protocol batches tables. `Commit` sends every table's level-0 cap, the link points are one
draw for all tables, and then every table's reps and openings run concurrently (`sessionB`). The
session accepts when every table accepts (`acceptedB`).

For each table `j`, everything after `Commit` is a batch with table `j` at position `j` and the
other tables around it (`sessAfterB_gamesB`), so `session_knowledge_sound` applies to it. Per
table, this gives the form the audit's randomized-extractor analysis consumes:

* `extractProbB`: the probability, over `K` reruns of everything after `Commit`, that the table
  the extractor fills for table `j` has a given property; `failProbB` is its failure (no
  satisfying witness near it), and the link event is any other property;
* `joint_le_B`: `Pr[the session accepts] · failProbB ≤ ksBoundB`, with `ksBoundB` table `j`'s
  `ε_c⁻ + K·adv₀ + N₀/(eK)`, computed in the session;
* `one_le_fail_add_B`: if commitment implies the event, every extracted table is uncommitted or has
  it, so the two probabilities add up to at least `1` (the audit's `cover`).

For a unit read from table `j`, these are the unit's `ksFail` (`failProbB`), its `linkFail`
(`extractProbB` of the link event), its `cover` (`one_le_fail_add_B'`) and its knowledge bound
(`joint_le_B'`), in `ℝ≥0∞` as `Audit.ExtractionAnalysis` takes them.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-- A table of a batched session: a statement, its `fast100` schedule by levels, and its link
layout for the session's `m_pts`. -/
structure TabSpec (mPts : ℕ) where
  S : Statement
  m : ℕ
  k0 : ℕ
  l₀ : Level
  l₁ : Level
  rest : List Level
  ext : ℕ
  hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
  hm : 13 ≤ S.m
  ptLocal : ℕ
  hlay : S.LinkLayout ptLocal mPts

variable {mPts : ℕ}

namespace TabSpec

/-- The table's schedule. -/
abbrev sch (T : TabSpec mPts) : Schedule := ⟨T.m, T.k0, T.l₀ :: T.l₁ :: T.rest, T.ext⟩

end TabSpec

/-! ## The session -/

/-! ## Per table: the extractor's probabilities and the knowledge bound -/

end FlockSoundness
