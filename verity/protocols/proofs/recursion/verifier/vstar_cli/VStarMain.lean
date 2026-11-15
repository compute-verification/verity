module

public import VStar.Cli

@[expose] public section

def main (args : List String) : IO UInt32 := VStar.Cli.cmd args
