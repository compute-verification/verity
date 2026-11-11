module

public import MemoryChallenger.WipeProtocol

@[expose] public section

def main (args : List String) : IO UInt32 := MemoryChallenger.WipeProtocol.testProtocol.main args
