import Properties.Onsite
import Proofs.Properties.Composed

/-! # Proof of onsite's sampled proofs properties (`Properties.Onsite`)

Onsite is a `registeredB` with one-stage v2 among its components (`v2_mem`), so `Properties.Composed.registeredAccepted`
and `registeredRegistered` give both.
-/

namespace Properties.Onsite

theorem v2_mem : SampledProofs.OneStageZk.protocol ∈ Core.Onsite.components := by
  unfold Core.Onsite.components
  repeat (first | exact List.Mem.head _ | apply List.Mem.tail)

theorem sampledProofsAccepts : SampledProofsAccepts := fun a b e blobs h =>
  Properties.Composed.registeredAccepted "verity/onsite/v0" NciCheck.Protocol.protocol SampledProofs.OneStageZk.protocol.id
    Core.Onsite.components v2_mem a b e blobs h

theorem sampledProofsRegistered : SampledProofsRegistered := fun a b e blobs h =>
  Properties.Composed.registeredRegistered (sampledProofsAccepts a b e blobs h)

end Properties.Onsite
