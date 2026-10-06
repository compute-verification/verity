import Definitions.Flock.Accounting.Bound
import Definitions.Flock.Accounting.Fast100
import Definitions.Flock.Accounting.Schedule
import Definitions.Flock.Audit.Circuit
import Definitions.Flock.Audit.Closure
import Definitions.Flock.Audit.Harm
import Definitions.Flock.Audit.Influence
import Definitions.Flock.Audit.Law
import Definitions.Flock.Audit.OneStage
import Definitions.Flock.Audit.Partitioning.Model
import Definitions.Flock.Audit.Stratified
import Definitions.Flock.Audit.StratifiedMiss
import Definitions.Flock.Audit.Window
import Definitions.Flock.Audit.Work
import Definitions.Flock.Defs
import Definitions.Flock.Game.Basic
import Definitions.Flock.Game.Expect
import Definitions.Flock.Game.Interleave
import Definitions.Flock.Game.Prob
import Definitions.Flock.Model.Basic
import Definitions.Flock.Model.Ligerito
import Definitions.Flock.Model.Piop
import Definitions.Flock.Model.Session
import Definitions.Flock.Model.Statement

/-!
# C-Flock's definitions (trusted)

The vocabulary C-Flock's guarantees and named assumptions are stated in: the soundness model's games, the protocol
model and its statements, the accounting schedules and bounds, and the audit's circuits, partitions and draw laws, each
under the name it had in `Proofs.Flock.Soundness` before it moved (`FlockSoundness.X`). A module that moved whole keeps
its path under `Definitions.Flock`; one that moved in part keeps its lemmas in `Proofs.Flock.Soundness` at the same path,
which imports it. They build on Mathlib alone: a module here imports Mathlib, core's definitions and other modules of
`Definitions.Flock`, never an assumption, a guarantee, a proof or ArkLib (`layers` in `lean-audit.json`). The few
lemmas here are the ones a definition's own elaboration needs, or that a guarantee's statement names.

This tree, `Specs.Flock.Assumptions` and `Specs.Flock.Guarantees` are C-Flock's trusted text. `Proofs.Flock` holds the
proofs and every other lemma.
-/
