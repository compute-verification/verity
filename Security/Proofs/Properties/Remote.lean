import Properties.Remote
import Proofs.Properties.Composed

/-! # Proof of remote's sampled proofs properties (`Properties.Remote`)

Remote is a `productB` with one-stage v2 among its components (`v2_mem`), so `Properties.Composed.accepted` and
`registered` give both.
-/

namespace Properties.Remote

theorem v2_mem : SampledProofs.OneStageZk.protocol ∈ Core.Remote.components := by
  unfold Core.Remote.components
  repeat (first | exact List.Mem.head _ | apply List.Mem.tail)

theorem sampledProofsAccepts : SampledProofsAccepts := fun a b e blobs h =>
  Properties.Composed.accepted "verity/remote/v0" Core.Remote.components v2_mem a b e blobs h

theorem sampledProofsRegistered : SampledProofsRegistered := fun a b e blobs h =>
  Properties.Composed.registered (sampledProofsAccepts a b e blobs h)

end Properties.Remote
