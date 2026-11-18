module

public import PouwVerifier.Period

/-!
# `pouw-period`: PoUW's credit-period protocol as an executable

The executable's entry point, which serves `PouwVerifier.Period.protocol` through core's command line, under the
challenger's pin when given (`pouw-period [--challenger-key-sha512 H [--challenger-retired FILE]] ...`,
`PouwVerifier.Period.pinned`).
-/

@[expose] public section

def main (args : List String) : IO UInt32 :=
  Spot.ChallengerPin.pinMain (fun pin => (PouwVerifier.Period.pinned pin).main) PouwVerifier.Period.protocol.main args
