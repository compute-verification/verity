/- The `flock-firewall` executable: `Flock.Firewall.cmd`, `flock-firewall contract [--public FILE] FILE…`,
`Flock.Firewall.Contract.cmd`, `flock-firewall frame`, the outer ZK server's release, `Flock.Firewall.Frame.cmd`,
`flock-firewall tap`, the OS bytes of a firewalled prover's other samples, `Flock.Firewall.Frame.tapCmd`,
`flock-firewall coins`, the coin tree's opening checks a firewalled prover's coins pass, `Flock.Firewall.Coins.cmd`,
`flock-firewall hello`, a ZK server session's `Hello` at a nonce it samples, `Flock.Firewall.Hello.cmd`,
`flock-firewall session`, the ZK server's session loop, which owns the auditor's connection, `Flock.Firewall.Session.cmd`,
`flock-firewall finish`, a firewalled rep's proof file computed by the ZK server, `Flock.Firewall.Finish.cmd`,
`flock-firewall values`, the ZK server's value service, which derives every value commitment's salts from its own key and
registers none without the ZK server's record (`core-firewall values` registers), `Flock.Firewall.Values.cmd`, and `flock-firewall live`, the inner phase's send check on one live session,
`Flock.Firewall.liveCmd`. -/
module

public import Flock.Firewall
public import Flock.Firewall.Contract
public import Flock.Firewall.Finish
public import Flock.Firewall.Frame
public import Flock.Firewall.Hello
public import Flock.Firewall.Session
public import Flock.Firewall.Values

@[expose] public section

def main : List String → IO UInt32
  | "contract" :: files => Flock.Firewall.Contract.cmd files
  | "frame" :: args => Flock.Firewall.Frame.cmd args
  | "tap" :: args => Flock.Firewall.Frame.tapCmd args
  | "coins" :: args => Flock.Firewall.Coins.cmd args
  | "hello" :: args => Flock.Firewall.Hello.cmd args
  | "session" :: args => Flock.Firewall.Session.cmd args
  | "finish" :: args => Flock.Firewall.Finish.cmd args
  | "values" :: args => Flock.Firewall.Values.cmd args
  | "live" :: args => Flock.Firewall.liveCmd args
  | args => Flock.Firewall.cmd args
