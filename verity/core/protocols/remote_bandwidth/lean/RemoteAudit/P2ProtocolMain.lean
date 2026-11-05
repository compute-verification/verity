module

public import RemoteAudit.P2Protocol

@[expose] public section

def main (args : List String) : IO UInt32 := RemoteAudit.P2Protocol.protocol.main args
