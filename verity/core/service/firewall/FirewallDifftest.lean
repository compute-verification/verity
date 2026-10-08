import Difftest
import Firewall.Json

def main : IO Unit := Difftest.serve (Difftest.table [("op", Core.Firewall.opFn), ("outer", Core.Firewall.outerFn)])
