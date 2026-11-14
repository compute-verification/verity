module

public import Difftest
public import ZkServer.Json

@[expose] public section

def main : IO Unit := Difftest.serve (Difftest.table [("op", Core.ZkServer.opFn), ("outer", Core.ZkServer.outerFn)])
