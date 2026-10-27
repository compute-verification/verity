import Definitions.Accounting.Guarantees.JointBridge.BandwidthOfRemoteLineLive
import Proofs.Core.Game.Prob

/-!
# Proof that PoUB's per-period guarantee over the causal game is the remote bandwidth audit's

On every run of every strategy, PoUB accepting while the bandwidth account read fewer than `Q` bits is the session
accepting while its line credits more bits read than the node read: the verdict is the session's (`accepts_iff`), and
`Q ≤ credit` on an accepted run (`cert_le`) while `readBits ≤ read` (`read_le`). The probability is monotone in the
event (`Verity.Game.prob_mono`).
-/

namespace Accounting.SecurityProofs.Bridge

open Accounting Verity.Game

/-- **PoUB's per-period guarantee over the causal game is the remote bandwidth audit's**
(`Accounting.Guarantees.Bridge.BandwidthOfRemoteLineLive`). -/
theorem BandwidthOfRemoteLineLive : Accounting.Guarantees.Bridge.BandwidthOfRemoteLineLive := by
  intro ι α _ d C O G r c acc credit readBits err real hQ s
  refine le_trans (prob_mono (fun o ⟨hacc, hshort⟩ => ?_) G s) (hQ s)
  have hacc' := (real.accepts_iff o).1 hacc
  refine ⟨hacc', lt_of_not_ge fun hle => hshort ?_⟩
  exact (real.cert_le o hacc').trans (hle.trans (real.read_le s o))

end Accounting.SecurityProofs.Bridge
