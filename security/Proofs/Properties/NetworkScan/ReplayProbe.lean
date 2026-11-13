import Lean
import Properties.NetworkScan.Assumptions
import Properties.NetworkScan
import Proofs.Properties.NetworkScan.Causal
import Proofs.Properties.NetworkScan.Fits
import Proofs.Properties.NetworkScan.Walk
import Proofs.Properties.NetworkScan

/-!
Scratch, never lands: the kernel replay (`tools/verity/lean/Replay.lean`) of the modules below, in the tree this file
is elaborated in. Its imports make that tree build them; it then runs the toolchain's `lean --run` on the copy of the
tool beside this file, as the Lean check runs it, and fails where the replay fails.

The modules are the changed ones and every module that imports one, the set the tool needs.
-/

#eval show IO Unit from do
  let replayed : List String :=
    ["Properties.NetworkScan.Assumptions", "Properties.NetworkScan", "Proofs.Properties.NetworkScan.Causal",
     "Proofs.Properties.NetworkScan.Fits", "Proofs.Properties.NetworkScan.Walk", "Proofs.Properties.NetworkScan"]
  let lean := (← IO.appPath).toString
  let tool := "Properties/NetworkScan/ReplayTool.lean"
  let out : System.FilePath := ".lake" / "replay-probe.json"
  if ← out.pathExists then IO.FS.removeFile out
  let r ← IO.Process.output { cmd := lean, args := #["--run", tool, out.toString] ++ replayed.toArray }
  IO.println s!"replay-probe: lean --run {tool} on {replayed}: exit {r.exitCode}"
  unless r.stdout.isEmpty do IO.println r.stdout
  unless r.stderr.isEmpty do IO.println r.stderr
  if ← out.pathExists then
    IO.println s!"replay-probe: {← IO.FS.readFile out}"
    IO.FS.removeFile out
  unless r.exitCode == 0 do throw <| IO.userError s!"replay-probe: the replay exited {r.exitCode}"
