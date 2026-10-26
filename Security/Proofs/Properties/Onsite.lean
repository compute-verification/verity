import Properties.Onsite
import Proofs.Properties.Composed

/-! # Proof of onsite's sampled proofs properties (`Properties.Onsite`)

Onsite is a `productB` with one-stage v2 among its components (`v2_mem`), so `Properties.Composed.accepted` and
`registered` give both.
-/

namespace Properties.Onsite

theorem v2_mem : SampledProofs.OneStageZk.protocol ∈ Core.Onsite.components := by
  unfold Core.Onsite.components
  repeat (first | exact List.Mem.head _ | apply List.Mem.tail)

theorem sampledProofsAccepts : SampledProofsAccepts := fun a b e blobs h =>
  Properties.Composed.accepted "verity/onsite/v0" Core.Onsite.components v2_mem a b e blobs h

theorem sampledProofsRegistered : SampledProofsRegistered := fun a b e blobs h =>
  Properties.Composed.registered (sampledProofsAccepts a b e blobs h)

end Properties.Onsite
