import Difftest
import SampledProofs.SiteDifftest

def main : IO Unit := Difftest.serve (Difftest.table SampledProofs.SiteDifftest.table)
