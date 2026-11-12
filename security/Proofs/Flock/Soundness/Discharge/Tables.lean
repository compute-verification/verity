module

public import Proofs.Flock.Soundness.Discharge.Tables.Steps
public import Proofs.Flock.Soundness.Discharge.Tables.Sim
public import Proofs.Flock.Soundness.Discharge.Tables.Session
public import Proofs.Flock.Soundness.Discharge.Tables.Headline
public import Proofs.Flock.Soundness.Discharge.Tables.Accepts
public import Proofs.Flock.Soundness.Discharge.Tables.Statements

/-!
# Sessions of `J` tables

`--session-tables J`: one `Commit` of `J` roots, one link-point sample every table reads, the `2J` rep streams in any
order the prover picks, then every proof file.
* `Steps`: the model's moves inside a batch of interleaved reps (`Steps`, `Ctx`).
* `Sim`: the live session of `J` tables (`liveSession`), the model's (`modelSession`), and the simulation between them
  (`simSession`, `live_le_session`), given each table's decoder (`DecodesTable`).
* `Session`: the salted session as `modelSession`, and the refinement for any plan (`execRefines_tables`, given
  `DecodesOnJ`).
* `Headline`: the headline at the executable for any plan, at the live stratified and subset samples.
* `Accepts`: `Stmt.setupTables`'s tables at `count > 1` (`setupTables_parts`).
* `Statements`: the statements, as `Prop`s, and their theorems.
-/

@[expose] public section
