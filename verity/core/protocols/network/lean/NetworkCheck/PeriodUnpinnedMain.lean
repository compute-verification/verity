import NetworkCheck.Period

/-! The `network-period-unpinned` executable, a test's: `Core.Protocol.main` of `NetworkCheck.Period.unpinned`, the
period check with no scan pinned, audited under `NetworkCheck`. -/

def main (args : List String) : IO UInt32 := NetworkCheck.Period.unpinned.main args
