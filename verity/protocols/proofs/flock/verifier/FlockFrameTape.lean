module

public import Flock.ZkServer.Frame

/-! The `flock-frame-tape` executable, a test's: `Flock.ZkServer.Frame.cmdWith` with the rep's samples read from the file
`TAPE` in place of the operating system's bytes, so a test knows the pads the frame masks with. `flock-zk-server frame` and
the session loop's frames sample from the operating system and send only the pads' root. -/

@[expose] public section

def main : List String → IO UInt32
  | [tape] => do
    let b ← IO.FS.readBinFile tape
    Flock.ZkServer.Frame.cmdWith (fun _ => pure b) []
  | _ => do
    IO.eprintln "usage: flock-frame-tape TAPE"
    return 2
