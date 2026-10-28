import PouwCommit.Tagged
import Identity.Domains

/-!
# The protocol's bindings (`verity.core.protocols.pouw.audit`)

What each of the protocol's trees is bound to, and the one root over the weights' roots, every one a `tagged` digest:

    weight_binding(w)            = tagged("verity/pouw/weight-rows/v1", w)
    activation_binding(salt, i)  = tagged("verity/pouw/activation-rows/v1", salt, i)
    fold_roots(tag, roots)       = tagged(tag, w₀, root₀, w₁, root₁, …)                       in increasing w
    weights_root(roots)          = fold_roots("verity/pouw/weights/v1", roots)
    transcript_binding(epoch)    = tagged("verity/pouw/transcript/v1", epoch.salt, epoch.weights_root)
-/

namespace PouwCommit

def weightBinding (w : Nat) : ByteArray := tagged Domains.POUW_WEIGHT_ROWS [.int w]

def activationBinding (salt : ByteArray) (index : Nat) : ByteArray :=
  tagged Domains.POUW_ACTIVATION_ROWS [.bytes salt, .int index]

/-- `protocol.fold_roots`: `tagged` of `tag` and each `(id, root)` by ascending id, the one fold of registered roots
every PoUW weights' root is at its own tag (`weightsRoot`, `PouwCheck.weightsRootAt`, `PouwCheck.Credit.weightsRoot`). -/
def foldRoots (tag : String) (roots : List (Nat × ByteArray)) : ByteArray :=
  let sorted := roots.mergeSort fun a b => a.1 ≤ b.1
  tagged tag (sorted.flatMap fun (w, r) => [.int w, .bytes r])

/-- `weights_root` of the map from each weight to its root; it refuses a weight listed twice, which a map can't hold. -/
def weightsRoot (roots : List (Nat × ByteArray)) : Except String ByteArray := do
  let sorted := roots.mergeSort fun a b => a.1 ≤ b.1
  if (sorted.zip sorted.tail).any fun (a, b) => a.1 == b.1 then throw "a weight is listed twice"
  pure (foldRoots Domains.POUW_WEIGHTS_ROOT roots)

def transcriptBinding (salt weightsRoot : ByteArray) : ByteArray :=
  tagged Domains.POUW_TRANSCRIPT [.bytes salt, .bytes weightsRoot]

end PouwCommit
