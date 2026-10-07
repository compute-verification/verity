/- The `flock-firewall` executable: `Flock.Firewall.cmd`, and `flock-firewall contract [--public FILE] FILE…`,
`Flock.Firewall.Contract.cmd`. -/
import Flock.Firewall
import Flock.Firewall.Contract

def main : List String → IO UInt32
  | "contract" :: files => Flock.Firewall.Contract.cmd files
  | args => Flock.Firewall.cmd args
