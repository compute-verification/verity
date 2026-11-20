module

public import Flock.ProvedScope
public import Flock.Zk
meta import Flock.ProvedScope
meta import Flock.Zk

/-!
`ProvedScope.sharedRowsOk`: a shared-row public file passes, with or without `--zk`, only for a statement with hidden
outputs and `hm96-sha512/v1` leaves whose verifier holds its own registered reads. Without them, at public outputs and on
plain SHA-512 leaves (a recursive session's inner statement) it is refused.
-/

@[expose] public section

namespace Flock.Test.PlainShared

open ProvedScope

/-- The served statement with public outputs, the form only a superseded version has. -/
def publicOutputs : Tags := { Tags.circuit with hiddenOutputs := false }

#guard sharedRowsOk Tags.circuit false true
#guard sharedRowsOk Tags.circuit true true
#guard sharedRowsOk (Zk.tags Tags.circuit 0 2) true true

#guard !sharedRowsOk Tags.circuit false false
#guard !sharedRowsOk Tags.circuit true false
#guard !sharedRowsOk Tags.circuitPlainLeaves false true
#guard !sharedRowsOk Tags.circuitPlainLeaves true true
#guard !sharedRowsOk publicOutputs false true
#guard !sharedRowsOk publicOutputs true true

end Flock.Test.PlainShared
