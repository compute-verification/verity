import CompositionDifftest
import CoreRemote

def main : IO Unit := Difftest.serve (Difftest.table [("answer", Core.CompositionDifftest.answer Core.Remote.protocol)])
