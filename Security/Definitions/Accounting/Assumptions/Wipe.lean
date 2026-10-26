import Definitions.Accounting.Wipe

/-!
# What an accepted streamed wipe assumes of the device (trusted)

The premise of `MemoryChallenger.Properties.WipeSurvivorBound`, the bridge from an accepted `verity/wipe/v0` session to
the wipe's survivor bound (`Verity.Wipe.SurvivorBound`), a named `Prop` on a wipe of the device model:

* `StreamedReads w`: something of the device's state when the fill ends, the egress and the reads revealed so far
  answers every spot read of every accepted run (`Accounting.Wipe.Answers`). It is the streamed game's shape
  (`Pous.StreamedWipe.Prover`) on the device.
-/

namespace Accounting.Assumptions

open Verity

/-- `certifier-coverage` and `trusted-host`, at the streamed wipe (C5 and the sequential reveal of
`verity/core/protocols/pous/PROTOCOL.md`'s streamed-label wipe). **Streamed reads**: some `answer` answers every spot
read of every run the memory challenger accepts (`Accounting.Wipe.Answers w answer`), from the device's state when the
fill ends, the wipe's egress `w.egress`, the reads revealed before it in order and its own index.

* C5, no side channel: during the reads nothing reaches the device but the reads and what the outside sends back, and
  the outside holds only the egress, every bit that left the device during the wipe, which the network certifier on
  every boundary link counts (`certifier-coverage`), so what the device and the outside answer is a function of these.
* The sequential reveal: read `i`'s index is sent after read `i - 1`'s answer arrived (`StreamedWipe.verdict` refuses
  a batch size above 1), and the memory challenger, the auditor's device on a host the prover doesn't control
  (`trusted-host`), compares each answer with its label and accepts only when every one matches.
* No unaccounted tier: the device's state (`Verity.Device.State`, its chip and its store) is all it holds when the fill
  ends, as the commitment's tiers declare.

C6, the egress's count, is not here: `w.egress` is what the certifier counts, and the property bounds it by the count
the session publishes. A prover breaks this premise only through a channel no certifier sees, a store outside the
device's state, or a challenger that accepts a wrong or early answer. -/
def StreamedReads {ι : Type} {d : Accounting.Device ι} {C : Circuit} {B : ℕ} {s : Blocks C B} {b : Fin B} {n k : ℕ}
    (w : Verity.Wipe d s b (Accounting.Wipe.Labels n) (Accounting.Wipe.Reads n k)) : Prop :=
  ∃ answer, Accounting.Wipe.Answers w answer

end Accounting.Assumptions
