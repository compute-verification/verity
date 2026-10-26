/-
`ProvedScope.sharedRowsOk`: a shared-row public file passes, with or without `--zk`, only for a statement with hidden
outputs and `hm96-sha512/v1` leaves whose verifier holds its own registered reads. Without them, at public outputs and on
plain SHA-512 leaves (a recursive session's inner statement) it is refused.
-/
import Flock.ProvedScope
import Flock.Zk

namespace Flock.Test.PlainShared

open ProvedScope

#guard sharedRowsOk Tags.circuit false true
#guard sharedRowsOk Tags.circuit true true
#guard sharedRowsOk (Zk.tags Tags.circuit 0 2) true true

#guard !sharedRowsOk Tags.circuit false false
#guard !sharedRowsOk Tags.circuit true false
#guard !sharedRowsOk Tags.circuitPlainLeaves false true
#guard !sharedRowsOk Tags.circuitPlainLeaves true true
#guard !sharedRowsOk Tags.circuit967b8d06 false true
#guard !sharedRowsOk Tags.circuit967b8d06 true true

end Flock.Test.PlainShared
