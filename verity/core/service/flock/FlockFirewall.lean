/- The `flock-firewall` executable: `Flock.Firewall.cmd`. -/
import Flock.Firewall

def main (args : List String) : IO UInt32 := Flock.Firewall.cmd args
