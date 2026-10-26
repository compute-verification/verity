import MemoryChallenger.WipeProtocol
import Definitions.Accounting.Assumptions.Wipe

/-!
# An accepted wipe meets the survivor bound it publishes

For a session it accepts and certifies, `MemoryChallenger.WipeProtocol.check` (`verity/wipe/v0`, whose verdict is
`StreamedWipe.verdict`) publishes `foreign_bits = ⌈M/19⌉ + E` at `eps = 2^-128` (`certify`). `WipeSurvivorBound` says
that pair is the survivor bound (`Verity.Wipe.SurvivorBound foreign_bits eps`) of every wipe of the device model
(`Accounting.Device`) that the session is a run of:

* its seed the session's labels (`Accounting.Wipe.Labels`) and its spot reads the session's `k` reads
  (`Accounting.Wipe.Reads`);
* its fill the published `memory_bits` (`M`, the fill's `labels · 512` bits), and its egress, every bit that left the
  device during the wipe as the network certifier counts it (C6), at most the published `egress_mib` whole MiB;
* its reads answered from the device's state when the fill ends, the egress and the reads before them
  (`Accounting.Assumptions.StreamedReads`, the named premise: C5, the sequential reveal and the trusted challenger).

The state the fill leaves out (`Verity.Wipe.unfilled`: the on-chip state and the store the fill doesn't reach) is
counted as `SurvivorBound` counts it, beside `foreign_bits`: in the streamed game it is egress, so nothing of the
device's state goes uncounted. `SurvivorBound`'s `ε`, a function of the adversary's moves, is read as the constant
`eps`, which reads the labels as the ideal ones, uniform: the published `eps` leaves out the PRF's advantage
(`prf/sha-512`), as `WipeProtocol.EXPERIMENTAL` says, and the network certifier's `δ_net`.
Proved from `Pous.Guarantees.StreamedWipeMeetsFamily` (`Proofs.Properties.Pous.Wipe`).
-/

namespace MemoryChallenger.Properties

open Verity
open scoped ENNReal

/-- **An accepted, certified wipe meets the survivor bound at the pair it publishes.** When `check` accepts the signed
session `e` of the commitment `c` and publishes it certified, with `egress_mib`, `eps`, `foreign_bits` and
`memory_bits`, every wipe `w` of a device whose seed is `c`'s labels and whose spot reads are its reads, whose fill is
`memory_bits`, whose egress is at most `egress_mib` MiB, and whose reads are streamed (`StreamedReads`) meets
`w.SurvivorBound foreign_bits` at the constant `eps`. -/
def WipeSurvivorBound : Prop :=
  ∀ (schemes : Schemes) (p : WipeProtocol.Params) (c : StreamedWipe.Commitment) (e : WipeProtocol.Evidence)
    (egressMib : Int) (eps : Core.Ratio) (foreignBits memoryBits : Nat) (profile : StreamedWipe.Params × Int),
    WipeProtocol.check schemes p c e = .accept ⟨true, egressMib, eps, foreignBits, memoryBits⟩ profile →
    ∀ {ι : Type} (d : Accounting.Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B) (b : Fin B)
      [NeZero c.params.labels]
      (w : Verity.Wipe d s b (Accounting.Wipe.Labels c.params.labels)
        (Accounting.Wipe.Reads c.params.labels c.params.k)),
      w.fill = memoryBits → w.egress ≤ egressMib.toNat * WipeProtocol.UNIT_BITS →
      Accounting.Assumptions.StreamedReads w →
      w.SurvivorBound foreignBits (fun _ _ _ => (eps.num : ℝ≥0∞) / eps.den)

end MemoryChallenger.Properties
