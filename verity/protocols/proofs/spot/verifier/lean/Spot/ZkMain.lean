module

public import Spot.MembersRefusal

@[expose] public section

def main (args : List String) : IO UInt32 := Spot.OneStageZk.refusalMain args
