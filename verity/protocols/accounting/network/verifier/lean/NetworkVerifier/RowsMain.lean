module

public import NetworkVerifier.RowsAudit

/-! The `network-rows` executable: `Core.Protocol.main` of `NetworkVerifier.RowsAudit.protocol`, audited under `NetworkVerifier`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkVerifier.RowsAudit.protocol.main args
