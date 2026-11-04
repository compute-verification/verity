module

public import PouwCheck.Period

@[expose] public section

def main (args : List String) : IO UInt32 := PouwCheck.Period.protocol.main args
