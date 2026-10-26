import Definitions.Accounting.Guarantees.JointBridge.BandwidthOfRemoteLinePeriod
import Proofs.Pouw.Prob

/-!
# Proof that PoUB's per-period guarantee is the remote bandwidth audit's

PoUB's coordinate of the coins is the session's coins (`coin`), so the joint event, PoUB accepting while the bandwidth
account read fewer than `Q` bits, maps into the session's: the verdict is the session's (`accepts_iff`), and
`Q ≤ credit ≤ readBits ≤ read` on an accepted period (`cert_le`, `read_le`) unless the line credits more bits than the
node read. Uniform coins carried by an equivalence keep their probabilities (`Pouw.pr_equiv`).
-/

namespace Accounting.SecurityProofs.Bridge

open Accounting

/-- **PoUB's per-period guarantee is the remote bandwidth audit's**
(`Accounting.Guarantees.Bridge.BandwidthOfRemoteLinePeriod`). -/
theorem BandwidthOfRemoteLinePeriod : Accounting.Guarantees.Bridge.BandwidthOfRemoteLinePeriod := by
  intro ι α I _ _ Ω _ d C r c p X _ acc credit readBits δ real hQ ξ
  refine le_trans (Pouw.pr_mono ?_)
    ((Pouw.pr_equiv real.coin (fun x => acc ξ x ∧ readBits ξ x < credit ξ x)).le.trans (hQ ξ))
  rintro y ⟨hacc, hshort⟩
  have hacc' := (real.accepts_iff ξ y).1 hacc
  refine ⟨hacc', lt_of_not_ge fun hle => hshort ?_⟩
  exact (real.cert_le ξ y hacc').trans (hle.trans (real.read_le ξ y))

end Accounting.SecurityProofs.Bridge
