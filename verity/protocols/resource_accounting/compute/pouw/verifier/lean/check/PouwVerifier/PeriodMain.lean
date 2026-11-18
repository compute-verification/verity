module

public import PouwVerifier.Period
public import Spot.ChallengerPin

/-!
# `pouw-period`: PoUW's credit-period protocol as an executable

The executable's entry point, which serves `PouwVerifier.Period.protocol` through core's command line, under the
challenger's pin when given (`pouw-period [--challenger-key-sha512 H [--challenger-retired FILE]] ...`,
`Spot.ChallengerPin.periodMain`).
-/

@[expose] public section

def main (args : List String) : IO UInt32 := Spot.ChallengerPin.periodMain PouwVerifier.Period.protocol args
