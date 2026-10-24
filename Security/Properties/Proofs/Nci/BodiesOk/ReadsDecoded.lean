import Properties.Proofs.Nci.BodiesOk.ForReads
import Properties.Proofs.Nci.ReadsExact

/-!
# `ReadsExact.Decoded`, proved

`ReadsExact.PartNonneg` is `BodiesOk.PartNonneg` (`partNonneg_iff`), so `ForReads.decoded` is `ReadsExact.Decoded` of
every decoded Program (`decoded_reads`).
-/

namespace NciCheck.Properties.SecurityProofs.BodiesOk
open Flock.Program

theorem partNonneg_iff (q : Part) : ReadsExact.PartNonneg q ↔ PartNonneg q := by
  cases q <;> exact Iff.rfl

/-- **The decoder's facts the core reads hold of every decoded Program.** -/
theorem decoded_reads (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P) :
    ReadsExact.Decoded P := by
  intro d hd
  obtain ⟨h1, h2, h3⟩ := decoded bytes P hdec d hd
  refine ⟨h1, h2, fun k hk i hi => ?_⟩
  obtain ⟨a, b, c⟩ := h3 k hk i hi
  exact ⟨a, fun q hq => (partNonneg_iff q).2 (b q hq), c⟩

end NciCheck.Properties.SecurityProofs.BodiesOk
