/- The `flock-frame-tape` executable, a test's: `Flock.Firewall.Frame.cmdWith` with the rep's draws read from the file
`TAPE` in place of the operating system's bytes, so a test knows the pads the frame masks with. `flock-firewall frame` and
the session loop's frames draw from the operating system and send only the pads' root. -/
import Flock.Firewall.Frame

def main : List String → IO UInt32
  | [tape] => do
    let b ← IO.FS.readBinFile tape
    Flock.Firewall.Frame.cmdWith (fun _ => pure b) []
  | _ => do
    IO.eprintln "usage: flock-frame-tape TAPE"
    return 2
