import Specs.Flock.Guarantees.Audit
import Proofs.Flock.Soundness.Audit.Closure
import Proofs.Flock.Soundness.Audit.Influence
import Proofs.Flock.Soundness.Audit.StratifiedMiss
import Proofs.Flock.Soundness.Audit.Window
import Proofs.Flock.Soundness.Audit.Work

/-!
# Proofs of the audit's guarantees

One theorem `Flock.SecurityProofs.Audit.G : Flock.Guarantees.Audit.G` per guarantee, which is the theorem in
`Proofs.Flock.Soundness.Audit` that proves the same statement under its old name.
-/

namespace Flock.SecurityProofs.Audit

theorem ClosureEscape : Flock.Guarantees.Audit.ClosureEscape := @FlockSoundness.Audit.Law.closure_escape
theorem HarmLeUnsoundWork : Flock.Guarantees.Audit.HarmLeUnsoundWork := @FlockSoundness.Audit.Law.harm_le_unsoundWork
theorem CoversWindow : Flock.Guarantees.Audit.CoversWindow := @FlockSoundness.Audit.Law.covers_window
theorem RecordSizing : Flock.Guarantees.Audit.RecordSizing := FlockSoundness.Audit.Law.record_sizing
theorem RecordSizingOneFewer : Flock.Guarantees.Audit.RecordSizingOneFewer :=
  FlockSoundness.Audit.Law.record_sizing_one_fewer
theorem StratifiedMissEqGreedy : Flock.Guarantees.Audit.StratifiedMissEqGreedy :=
  @FlockSoundness.Audit.Law.stratified_miss_eq_greedy
theorem WorkEscapeLe : Flock.Guarantees.Audit.WorkEscapeLe := @FlockSoundness.Audit.Law.work_escape_le
theorem WorkWholeStratum : Flock.Guarantees.Audit.WorkWholeStratum := @FlockSoundness.Audit.audit_work_whole_stratum
theorem WorkFloor : Flock.Guarantees.Audit.WorkFloor := @FlockSoundness.Audit.audit_work_floor
theorem WindowSplitOfRecord : Flock.Guarantees.Audit.WindowSplitOfRecord :=
  @FlockSoundness.Audit.audit_window_split_of_record
theorem CardAdmissibleLe : Flock.Guarantees.Audit.CardAdmissibleLe := @FlockSoundness.Audit.Partition.card_admissible_le
theorem Exfiltration : Flock.Guarantees.Audit.Exfiltration := @FlockSoundness.Audit.Partition.audit_exfiltration

end Flock.SecurityProofs.Audit
