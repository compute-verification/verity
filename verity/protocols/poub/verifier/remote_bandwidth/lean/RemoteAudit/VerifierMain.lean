module

public import RemoteAudit.Verifier

@[expose] public section

def main : IO UInt32 := RemoteAudit.Verifier.serve
