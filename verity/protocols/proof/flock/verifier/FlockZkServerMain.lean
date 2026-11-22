module

public import Flock.ZkServer
public import Flock.ZkServer.Contract
public import Flock.ZkServer.Finish
public import Flock.ZkServer.Frame
public import Flock.ZkServer.Hello
public import Flock.ZkServer.Session
public import Flock.ZkServer.Values

/-! The `flock-zk-server` executable: `Flock.ZkServer.cmd`, `flock-zk-server contract [--public FILE] FILE…`,
`Flock.ZkServer.Contract.cmd`, `flock-zk-server frame`, the outer ZK server's release, `Flock.ZkServer.Frame.cmd`,
`flock-zk-server tap`, the OS bytes of a zk_served prover's other samples, `Flock.ZkServer.Frame.tapCmd`,
`flock-zk-server coins`, the coin tree's opening checks a zk_served prover's coins pass, `Flock.ZkServer.Coins.cmd`,
`flock-zk-server hello`, a ZK server session's `Hello` at a nonce it samples, `Flock.ZkServer.Hello.cmd`,
`flock-zk-server session`, the ZK server's session loop, which owns the auditor's connection, `Flock.ZkServer.Session.cmd`,
`flock-zk-server finish`, a zk_served rep's proof file computed by the ZK server, `Flock.ZkServer.Finish.cmd`,
`flock-zk-server values`, the ZK server's value service, which derives every value commitment's salts from its own key and
registers none without the ZK server's record (`core-zk-server values` registers), `Flock.ZkServer.Values.cmd`, and `flock-zk-server live`, the inner phase's send check on one live session,
`Flock.ZkServer.liveCmd`. -/

@[expose] public section

def main : List String → IO UInt32
  | "contract" :: files => Flock.ZkServer.Contract.cmd files
  | "frame" :: args => Flock.ZkServer.Frame.cmd args
  | "tap" :: args => Flock.ZkServer.Frame.tapCmd args
  | "coins" :: args => Flock.ZkServer.Coins.cmd args
  | "hello" :: args => Flock.ZkServer.Hello.cmd args
  | "session" :: args => Flock.ZkServer.Session.cmd args
  | "finish" :: args => Flock.ZkServer.Finish.cmd args
  | "values" :: args => Flock.ZkServer.Values.cmd args
  | "live" :: args => Flock.ZkServer.liveCmd args
  | args => Flock.ZkServer.cmd args
