module

public import Proofs.Scratch.LiftWordCheck
meta import Proofs.Scratch.LiftWordCheck

/-! Scratch, not for merge: `LiftWordCheck` on the one-linear benchmark's C (`GemmC.json`, sha256 `84be6566…`, kept
untracked beside this file). -/

namespace Scratch.GemmCheck

open Scratch.LiftWordCheck

#eval show IO Unit from do
  let paths := ["Scratch/GemmC.json", "security/Proofs/Scratch/GemmC.json", "../Scratch/GemmC.json",
    "../../Scratch/GemmC.json"]
  let mut text : Option String := none
  for p in paths do
    if text.isNone then
      if ← System.FilePath.pathExists p then text := some (← IO.FS.readFile p)
  match text with
  | none => IO.println s!"GemmC.json not found from {← IO.currentDir}"
  | some t =>
    let t0 ← IO.monoMsNow
    for l in checkAll t [0] [] [1] do IO.println l
    IO.println s!"took {(← IO.monoMsNow) - t0} ms"

end Scratch.GemmCheck
