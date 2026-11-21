module

public import Flock.Test.Compose

/-! The `flock-compose-check` executable, a test's: `Flock.Test.Compose.roundTrips` compiled. -/

@[expose] public section

def main : IO UInt32 := match Flock.Test.Compose.roundTrips with
  | .ok () => do IO.println "flock-compose-check: every block's file reads back as the block"; pure 0
  | .error e => do IO.eprintln s!"flock-compose-check: {e}"; pure 1
