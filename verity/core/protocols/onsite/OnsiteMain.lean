import CoreOnsite
import CorePeriodFiles

def main (args : List String) : IO UInt32 := Core.PeriodFiles.main Core.Onsite.protocolWith args
