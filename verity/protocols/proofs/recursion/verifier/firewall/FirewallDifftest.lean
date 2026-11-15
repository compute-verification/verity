module

public import Difftest
public import Firewall.Json

@[expose] public section

def main : IO Unit := Difftest.serve (Difftest.table [("op", Core.Firewall.opFn), ("outer", Core.Firewall.outerFn)])
