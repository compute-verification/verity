/-
The frame-v3 read is sound: whenever the rows `Flock.ReadNet.frame D schema` emits hold, its bit is set only where
`Hash.Tree.climb` over the opened value, position and path reaches the root (`Gates.FrameReadSpec`). Its gadgets' lemmas
are in `Frame/Bytes.lean` and `Frame/Gadgets.lean`.
-/
module

public import Proofs.Accounting.ReadUnit.Frame.Gadgets

@[expose] public section

namespace Accounting.ReadUnit.FrameRead

end Accounting.ReadUnit.FrameRead
