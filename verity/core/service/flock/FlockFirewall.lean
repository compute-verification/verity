/- The `flock-firewall` executable: `Flock.Firewall.cmd`, `flock-firewall contract [--public FILE] FILE…`,
`Flock.Firewall.Contract.cmd`, `flock-firewall frame`, the outer firewall's release, `Flock.Firewall.Frame.cmd`,
`flock-firewall tap`, the OS bytes of a firewalled prover's other draws, `Flock.Firewall.Frame.tapCmd`,
`flock-firewall coins`, the coin tree's opening checks a firewalled prover's coins pass, `Flock.Firewall.Coins.cmd`,
`flock-firewall link`, the level-0 recheck whose roots a firewalled table's `Commit` carries, `Flock.Firewall.Link.cmd`,
`flock-firewall hello`, a firewalled session's `Hello` at a nonce it draws, `Flock.Firewall.Hello.cmd`, and
`flock-firewall finish`, a firewalled rep's proof file computed by the firewall, `Flock.Firewall.Finish.cmd`. -/
import Flock.Firewall
import Flock.Firewall.Contract
import Flock.Firewall.Finish
import Flock.Firewall.Frame
import Flock.Firewall.Hello
import Flock.Firewall.Link

def main : List String → IO UInt32
  | "contract" :: files => Flock.Firewall.Contract.cmd files
  | "frame" :: args => Flock.Firewall.Frame.cmd args
  | "tap" :: args => Flock.Firewall.Frame.tapCmd args
  | "coins" :: args => Flock.Firewall.Coins.cmd args
  | "link" :: args => Flock.Firewall.Link.cmd args
  | "hello" :: args => Flock.Firewall.Hello.cmd args
  | "finish" :: args => Flock.Firewall.Finish.cmd args
  | args => Flock.Firewall.cmd args
