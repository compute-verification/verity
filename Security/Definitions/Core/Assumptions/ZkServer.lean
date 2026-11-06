module

public import Definitions.Core.ValueSalts
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-!
# The ZK server's key is its own

`ZkServerOwnUser`: what the ZK server's value commitments' salts rest on of the deployment
(`Definitions.Core.ValueSalts`). The value service and the session loop hold the server's salt key, and the tags, the
record tags and the record's session count mean what they say only while nothing the serving stack runs reads the key.
-/

@[expose] public section

namespace Verity.Assumptions

open Verity.ValueSalts

/-- **`zk-server-own-user`**: the ZK server's value service and its session loop run under the ZK server's own OS user,
which no worker and no serving stack shares, and the salt key's file is that user's alone, so nothing the stack runs
reads the key: as a function of the key (`Verity.ValueSalts.Stack`), the stack `st` is the same under every key. -/
def ZkServerOwnUser {K V S O T R : Type} {m : ℕ} (st : Stack K V S O T R m) : Prop :=
  ∀ k k', st k = st k'

end Verity.Assumptions
