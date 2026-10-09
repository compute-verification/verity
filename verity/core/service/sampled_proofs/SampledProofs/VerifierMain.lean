import Difftest
import SampledProofs.Pieces

def main : IO Unit := Difftest.serve (Difftest.table SampledProofs.Pieces.verifier)
