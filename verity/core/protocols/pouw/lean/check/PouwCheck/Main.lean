module

public import PouwCheck.Protocol

@[expose] public section

def main (args : List String) : IO UInt32 := PouwCheck.protocol.main args
