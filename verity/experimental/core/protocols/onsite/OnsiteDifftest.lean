import CompositionDifftest
import CoreOnsite

def main : IO Unit := Difftest.serve (Difftest.table [("answer", Core.CompositionDifftest.answer Core.Onsite.protocol)])
