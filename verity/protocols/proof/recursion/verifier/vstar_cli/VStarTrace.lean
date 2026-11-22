module

public import VStarTraceCli

@[expose] public section

def main (args : List String) : IO UInt32 := VStar.TraceCli.cmd args
