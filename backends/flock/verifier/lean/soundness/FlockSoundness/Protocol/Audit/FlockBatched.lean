import FlockSoundness.Protocol.Audit.OneStage
import FlockSoundness.Protocol.Instance
import FlockSoundness.Protocol.SessionBatch
import FlockSoundness.Protocol.Lowering

/-!
# The Flock audit at the compiled layer, several tables per session

The protocol's session (PROTOCOL.md §5, `SessionBatch.lean`): the verifier plans the tables from the
draw and the registration (`plan`), `Commit` sends every table's level-0 cap, one link-point draw
serves all of them, and then they run concurrently; the session accepts when every table accepts
(`batchedSession`). Each drawn unit is read from one table of the plan (`tab S R u`), with its own
decoder (`DecoderB`): two drawn units that share a wire hold it in their own columns.

**The analysis** (`analysisB`) reads the extractor of the unit's table, over `Kr` reruns of
everything after `Commit`:

* `ks`: `Pr[the session accepts] · Pr[the unit's table extracts to no satisfying witness]`
  (`failProbB`), the joint probability, since the fresh run and the reruns are independent
  (`expect_mul_expectN`);
* `link`: likewise with the link event on the extracted table (`LinkEvB`): it is committed, but
  every close message packing a satisfying witness decodes, on the unit's committed wires, to values
  other than the committed transcript's.

`cover` is `one_le_fail_add_B'`: by **the lowering** (`LoweringSoundB`, named, per table), a
committed table gives the link event for a wrong unit. `loweringSoundB_of_place` discharges it from
each drawn unit's place in its table (`UnitPlace`), with `C` the rows the verifier parses.

**The discharge** (`flock_batched_knowledgeSound`): per draw, the unit's table's
`ε_c⁻ + K·Adv₀ + N₀/(eK)`, computed in the session (`joint_le_B'`, from `session_knowledge_sound`),
bounded by the largest over the plan's tables, and averaged over the draw (`ksAvgB`), with no
named assumption; `δ_link` stays named (`LinkSound`).

`flock_batched_count`: `Pr[accept ∧ at least K wrong units] ≤ miss K + ε_ks(σ) + δ_link(σ)`.
-/

namespace FlockSoundness.Audit

open Game Model Accounting
open scoped ENNReal

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- The table of the plan that carries drawn unit `u`. -/
abbrev tabOf (S : Finset (Fin n)) (R : Reg) (u : Fin n) : TabSpec mPts := (plan S R).get (tab S R u)

/-- A decoder: for each drawn unit, the values a level-0 message of its table packs for it. -/
abbrev DecoderB := (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) →
  Message 𝔽 (tabOf plan tab S R u).sch.k0 (tabOf plan tab S R u).sch.level₀.logCols → (Fin C.N → Bool)

variable {L : Law n}

end Partition

end FlockSoundness.Audit
