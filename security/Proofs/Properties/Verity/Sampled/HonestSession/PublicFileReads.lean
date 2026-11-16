module
public import Proofs.Properties.Verity.Sampled.HonestSession.PublicFileProof
import Flock.HmRow
@[expose] public section
open Flock
namespace Verity.Sampled.Instance.HonestSession

/-! # The setup's reads of the drawn public value `pubD`

The setup-reads proof (`HonestSession/SetupReads.lean`) proves `setupHidden_ok` and `shapeOk_ok` given, among others,
three reads of the drawn public value `pubD` that only the public-file proof (which owns `pubD`'s equation to the
explicit `Wdrawn`) can discharge, plus the program-side `pubI = Vout` the links check needs. The statements match
`SetupReads.setupFacts_of`'s hypotheses exactly, so the integration piece applies `setupFacts_of` directly. -/

/-- `pubD`'s population is one. -/
theorem pubD_n (hC : CircuitFileFacts) : pubD.n = 1 := by
  rw [pubD_eq hC]; kernel_rfl'

/-- `pubD`'s one instance's one-port row digest is 128 bytes: the drawn digest is the population's row, the concrete
`rowBc`'s first `BC_BYTES` bytes, whose size is `128` (no hash forced -- `rowBc` is the concrete zero row). -/
theorem pubD_digest_size (hC : CircuitFileFacts) : (pubD.digest 0 0 1).size = 128 := by
  rw [pubD_eq hC, show Wdrawn.digest 0 0 1 = rowBc.extract 0 128 from by
    simp [Flock.Public.digest, Wdrawn, drawnPF, pubPF, Draw.UnitDraw.proved, HmRow.BC_BYTES]]
  kernel_rfl'

/-- The program-side links check reads `pubI`; it is the loaded public value `Vout` (restated from the public-file
proof for citation by the program piece, which takes it as a named hypothesis). -/
theorem pubI_val (hC : CircuitFileFacts) : pubI = Vout := pubI_eq hC

end Verity.Sampled.Instance.HonestSession
