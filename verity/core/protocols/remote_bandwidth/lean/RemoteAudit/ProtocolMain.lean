module

public import RemoteAudit.Protocol

@[expose] public section

def main (args : List String) : IO UInt32 := RemoteAudit.Protocol.protocol.main args
