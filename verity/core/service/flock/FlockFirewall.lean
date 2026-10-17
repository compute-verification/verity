/- The `flock-firewall` executable: `Flock.Firewall.cmd`, `flock-firewall contract [--public FILE] FILE…`,
`Flock.Firewall.Contract.cmd`, and `flock-firewall frame`, the outer firewall's release, `Flock.Firewall.Frame.cmd`. -/
import Flock.Firewall
import Flock.Firewall.Contract
import Flock.Firewall.Frame

def main : List String → IO UInt32
  | "contract" :: files => Flock.Firewall.Contract.cmd files
  | "frame" :: args => Flock.Firewall.Frame.cmd args
  | args => Flock.Firewall.cmd args
