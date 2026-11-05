module

public import Difftest
public import SampledProofs.SiteDifftest

@[expose] public section

def main : IO Unit := Difftest.serve (Difftest.table SampledProofs.SiteDifftest.table)
