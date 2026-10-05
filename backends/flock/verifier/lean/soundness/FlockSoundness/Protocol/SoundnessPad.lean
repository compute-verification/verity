import FlockSoundness.Protocol.LigeritoPadLevel0
import FlockSoundness.Protocol.LinkSound

/-!
# Soundness of a table with M1's padded level 0

**`table_sound_pad`.** For every prover strategy, the probability that the verifier accepts a table
with the padded level 0 (both reps, each folding its own `e` extra lanes) while no family of witness
lanes within the padded radius of the committed table packs a witness satisfying the statement
(`CommittedPad`), is at most `padTableError sch shape t e`.

It is `table_sound` with the padded level 0, for any padding `t` and extra lanes per rep `e`:
* the phases before Ligerito pay their union bounds over the padded code's witnesses (`padWits`,
  at most `padListBound` of them), through the phases' list forms
  (`zerocheck_phase_of_list` and its three siblings);
* Ligerito is `ligeritoPad_sound` for each rep's list, which the rep's families' witness lanes
  send into `padWits` (`padWit_mem_padWits`).
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **The padded table's witnesses**: the messages of the families of witness lanes within the
padded radius of the committed witness lanes. -/
def padWits (A : Arith F K) (sch : Schedule) (t : ℕ) (c₀ : F) (C : Oracle F) :
    Set (Message F sch.k0 sch.level₀.logCols) :=
  {P | ∃ R ∈ padList A sch.level₀ t sch.k0 0 0 C, padWit c₀ R = P}

/-- **The committed values satisfy the statement, padded level 0**: some family of witness lanes
within the padded radius of the committed table packs a witness that satisfies `S`. -/
def CommittedPad (A : Arith F K) (S : Statement) (sch : Schedule) (t : ℕ) (c₀ : F)
    (C : Oracle F) : Prop :=
  ∃ P ∈ padWits A sch t c₀ C, S.Satisfies (wit A S sch P)

/-! ## One rep -/

/-! ## One table -/

/-! ## At M1's values -/

end FlockSoundness
