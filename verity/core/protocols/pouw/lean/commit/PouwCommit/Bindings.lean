import PouwCommit.Tagged

/-!
# The protocol's bindings (`verity.core.protocols.pouw.audit`)

What each of the protocol's trees is bound to, and the one root over the weights' roots, every one a `tagged` digest:

    weight_binding(w)            = tagged("verity/pouw/weight-rows/v1", w)
    activation_binding(salt, i)  = tagged("verity/pouw/activation-rows/v1", salt, i)
    weights_root(roots)          = tagged("verity/pouw/weights/v1", w₀, root₀, w₁, root₁, …)   in increasing w
    transcript_binding(epoch)    = tagged("verity/pouw/transcript/v1", epoch.salt, epoch.weights_root)
-/

namespace PouwCommit

def weightBinding (w : Nat) : ByteArray := tagged "verity/pouw/weight-rows/v1" [.int w]

def activationBinding (salt : ByteArray) (index : Nat) : ByteArray :=
  tagged "verity/pouw/activation-rows/v1" [.bytes salt, .int index]

/-- `weights_root` of the map from each weight to its root; it refuses a weight listed twice, which a map can't hold. -/
def weightsRoot (roots : List (Nat × ByteArray)) : Except String ByteArray := do
  let sorted := roots.mergeSort fun a b => a.1 ≤ b.1
  if (sorted.zip sorted.tail).any fun (a, b) => a.1 == b.1 then throw "a weight is listed twice"
  pure (tagged "verity/pouw/weights/v1" (sorted.flatMap fun (w, r) => [.int w, .bytes r]))

def transcriptBinding (salt weightsRoot : ByteArray) : ByteArray :=
  tagged "verity/pouw/transcript/v1" [.bytes salt, .bytes weightsRoot]

end PouwCommit
