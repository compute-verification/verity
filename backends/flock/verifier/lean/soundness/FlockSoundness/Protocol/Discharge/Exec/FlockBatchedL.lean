import FlockSoundness.Protocol.Discharge.Exec.SessionBatchL

/-!
# The Flock audit at the compiled layer, several tables per session

The protocol's session (PROTOCOL.md §5, `SessionBatch.lean`): the verifier plans the tables from the
draw and the registration (`plan`), `Commit` sends every table's level-0 cap, one link-point draw
serves all of them, and then they run concurrently; the session accepts when every table accepts
(`batchedSessionL`). Each drawn unit is read from one table of the plan (`tab S R u`), with its own
decoder (`DecoderB`): two drawn units that share a wire hold it in their own columns.

**The analysis** (`analysisBL`) reads the extractor of the unit's table, over `Kr` reruns of
everything after `Commit`:

* `ks`: `Pr[the session accepts] · Pr[the unit's table extracts to no satisfying witness]`
  (`failProbBL`), the joint probability, since the fresh run and the reruns are independent
  (`expect_mul_expectN`);
* `link`: likewise with the link event on the extracted table (`LinkEvB`): it is committed, but
  every close message packing a satisfying witness decodes, on the unit's committed wires, to values
  other than the committed transcript's.

`cover` is `one_le_fail_add_BL'`: by **the lowering** (`LoweringSoundB`, named, per table), a
committed table gives the link event for a wrong unit. `loweringSoundB_of_place` discharges it from
each drawn unit's place in its table (`UnitPlace`), with `C` the rows the verifier parses.

**The discharge** (`flock_batched_knowledgeSoundL`): per draw, the unit's table's
`ε_c⁻ + K·Adv₀ + N₀/(eK)`, computed in the session (`joint_le_BL'`, from `session_knowledge_soundL`),
bounded by the largest over the plan's tables, and averaged over the draw (`ksAvgBL`), with no
named assumption; `δ_link` stays named (`LinkSound`).

`flock_batched_countL`: `Pr[accept ∧ at least K wrong units] ≤ miss K + ε_ks(σ) + δ_link(σ)`.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128


variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- The strategy in the protocol's session inside a verdict strategy. -/
noncomputable def batchStratL {S : Finset (Fin n)} {R : Reg} (τ : Strategy (batchedSessionL H E Lf plan S R)) :
    Strategy (sessionBL execArith H E Lf (plan S R)) :=
  Strategy.ofMap _ _ τ

variable {L : Law n}

end FlockSoundness.Discharge.Exec
