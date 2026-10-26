import Definitions.Core.Blocks
import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe on the device model (trusted)

The wipe that opens a server-time block (`Verity.Wipe`, `Definitions.Core.Blocks`) when it is PoUS's streamed-label
wipe (`Pous.StreamedWipe`; `verity/core/protocols/pous/PROTOCOL.md`, "The streamed-label wipe"), as `verity/wipe/v0`
runs it: the memory challenger streams `n` labels of 512 bits into the device's store, then reveals `k` spot reads one
at a time and reads each label back under a deadline.

* **The seed** (`Labels n`) is what the challenger streams: the table of labels itself, drawn uniform, the ideal labels.
  The deployed labels are SHA-512's PRF of a key the challenger keeps (`prf/sha-512`, C3), so nothing of the device's
  reads the key but through the labels, and the step from the ideal labels to the PRF's is the PRF's advantage, which
  `verity/wipe/v0`'s published `eps` leaves out.
* **The spot reads** (`Reads n k`): `k` indices into the labels, uniform and independent.
* **What answers the reads** (`Answers w answer`): on every run the challenger accepts, `answer` gives each spot read
  its label from the device's state when the fill ends, the egress, the reads revealed before it, in order, and its own
  index: the answers of `Pous.StreamedWipe.Accepts`, read off the device.
-/

namespace Accounting.Wipe

open Verity

/-- The streamed wipe's seed: the table of `n` labels of 512 bits (64 bytes), drawn uniform. -/
abbrev Labels (n : ℕ) : Type := Fin n → Pous.Bits 512

/-- The streamed wipe's spot reads: `k` indices into the `n` labels. -/
abbrev Reads (n k : ℕ) : Type := Fin k → Fin n

variable {ι : Type} {d : Accounting.Device ι} {C : Circuit} {B : ℕ} {s : Blocks C B} {b : Fin B} {n k : ℕ}

/-- **`answer` answers the wipe's accepted reads**: on every run the memory challenger accepts, from the device's state
`σ` at the boundary, the labels `L` and the spot reads `c`, read `i`'s answer, `answer` of the device's state when the
fill ends, the egress, the reads revealed before read `i` in order and read `i`'s index, is the label at that index. -/
def Answers (w : Verity.Wipe d s b (Labels n) (Reads n k))
    (answer : Verity.Device.State d → (Fin w.egress → Bool) → List (Fin n) → Fin n → Pous.Bits 512) : Prop :=
  ∀ σ L c, w.accepts σ L c → ∀ i : Fin k,
    answer (w.filled σ L).1 (w.filled σ L).2 (Pous.StreamedWipe.revealed c i) (c i) = L (c i)

end Accounting.Wipe
