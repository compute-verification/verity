import NetworkCheck.RowsAudit

/-! The `network-rows` executable: `Core.Protocol.main` of `NetworkCheck.RowsAudit.protocol`, audited under `NetworkCheck`. -/

def main (args : List String) : IO UInt32 := NetworkCheck.RowsAudit.protocol.main args
