import Definitions.Hash
import PouwCommit.Rows
import Lean.Data.Json

/-! hash-redteam scratch: how long the model's pieces take in the interpreter, logged to `timing.txt` as they finish. -/

open Hash Lean

def logT (s : String) : IO Unit := do
  let h ← IO.FS.Handle.mk "timing.txt" .append
  h.putStrLn s!"{← IO.monoMsNow} {s}"
  h.flush

def tsha (m : Msg) : Digest := fun i => (Flock.Sha512.hash ⟨m.toArray⟩).get! i.val

#eval show IO Unit from do
  logT "start"
  let key := Flock.Hm96.Default512.key.data.toList
  let y : Msg := (List.range 192).map fun i => (i * 37 + 11).toUInt8
  let b := Hm96.maskBit key y 5
  logT s!"one maskBit: {b}"
  let m := Hm96.mask key y
  logT s!"one mask: {m.length}"
  let row : Row := (List.range 65536).map fun i => i % 3 == 0
  let p := RowSeg.packBits row
  logT s!"packBits 65536: {p.length}"
  let d := RowSeg.digest tsha row
  logT s!"row-seg digest 65536: {(d 0)}"
  let v ← IO.FS.readFile "../../verity/ml/commitments/frame_v3/vectors_sha512.json"
  logT s!"read vectors: {v.length}"
  let j ← IO.ofExcept (Json.parse v)
  logT s!"parsed vectors: {(j.getObjVal? "scheme").toOption.isSome}"
