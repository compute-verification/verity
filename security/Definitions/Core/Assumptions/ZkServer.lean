module

public import Definitions.Core.ValueSalts

/-! `ZkServerOwnUser`: nothing the serving stack runs reads the ZK server's salt key. -/

@[expose] public section

namespace Verity.Assumptions

/-- **`zk-server-own-user`**: the ZK server's value service and its session loop run under the ZK server's own OS user,
which no worker and no serving stack shares, and the salt key's file is that user's alone, so nothing the stack runs
reads the key: as a function of the key (`Verity.ValueSalts.Stack` over one registration, `Verity.ValueSalts.StreamStack`
over a stream), the stack `st` is the same under every key. -/
def ZkServerOwnUser {K X : Type} (st : K → X) : Prop :=
  ∀ k k', st k = st k'

end Verity.Assumptions
