module

public import NetworkCheck.RowsAudit

/-! The `network-rows` executable: `Core.Protocol.main` of `NetworkCheck.RowsAudit.protocol`, audited under `NetworkCheck`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkCheck.RowsAudit.protocol.main args
