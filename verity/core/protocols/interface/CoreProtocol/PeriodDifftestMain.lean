import CoreProtocol.PeriodDifftest

def main : IO Unit := Difftest.serve Core.PeriodDifftest.table
