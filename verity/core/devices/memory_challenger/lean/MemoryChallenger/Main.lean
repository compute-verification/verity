module

public import MemoryChallenger.Protocol

@[expose] public section

def main (args : List String) : IO UInt32 := MemoryChallenger.protocol.main args
