module

public import Definitions.Core.ValueSalts

/-!
# The ZK server's key is its own

`ZkServerOwnUser`: what the ZK server's value commitments' salts rest on of the deployment
(`Definitions.Core.ValueSalts`). The value service and the session loop hold the server's salt key, and the tags, the
record tags and the record's session count mean what they say only while nothing the serving stack runs reads the key.
-/

@[expose] public section

namespace Verity.Assumptions

/-- **`zk-server-own-user`**: the ZK server's value service and its session loop run under the ZK server's own OS user,
which no worker and no serving stack shares, and the salt key's file is that user's alone, so nothing the stack runs
reads the key: as a function of the key (`Verity.ValueSalts.Stack` over one registration, `Verity.ValueSalts.StreamStack`
over a stream), the stack `st` is the same under every key. -/
def ZkServerOwnUser {K X : Type} (st : K → X) : Prop :=
  ∀ k k', st k = st k'

end Verity.Assumptions
