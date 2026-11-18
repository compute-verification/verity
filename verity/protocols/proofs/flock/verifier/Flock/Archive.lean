module

public import Flock

/-! The verifier's archive: its own copies of objects under `ARCHIVE/sha512/<hex>`, each checked to hash to its key. -/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open Flock Flock.SelfTest

def archiveMismatch (hex : String) : String := s!"archive: the object held under {hex} does not hash to it"

/-- Whether `b`'s input hashes to the digest `hex`, by the SHA-512 `b` keeps, in a task. Not inlined: the compiler sinks a
`Task.spawn` bound in a body down to the branch that reads it. -/
@[noinline] def hashesToTask (b : KeptAt Sha512.hash) (hex : String) : Task Bool :=
  Task.spawn fun _ => hexOf b.value.get == hex

/-- `ARCHIVE/sha512/<hex>`: the verifier's own copy of an object, by content, with its SHA-512 as `hash` gives it
(`KeptAt`: the object is its `input`), its key's bytes, and whether it hashes to its key, still hashing (`archiveGetSha`
waits for it). Once it does, the key's bytes are the object's SHA-512 (`Hex.unhex_eq_of_hexOf`). -/
def archiveGetTask (archive hex : String) (hash : ByteArray → KeptAt Sha512.hash := KeptAt.of Sha512.hash) :
    IO (Except String (KeptAt Sha512.hash × ByteArray × Task Bool)) := do
  if hex.length != 128 || !hex.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) then
    return .error s!"archive: {hex} is not a SHA-512 digest"
  let p : System.FilePath := archive ++ s!"/sha512/{hex}"
  if !(← p.pathExists) then return .error s!"archive: {hex} is not held"
  let b := hash (← IO.FS.readBinFile p)
  let .ok key := unhex hex | return .error s!"archive: {hex} is not a SHA-512 digest"
  return .ok (b, key, hashesToTask b hex)

/-- `ARCHIVE/sha512/<hex>`: the verifier's own copy of an object, by content, and its SHA-512. The file must hash to its
key. -/
def archiveGetSha (archive hex : String) : IO (Except String (ByteArray × ByteArray)) := do
  match ← archiveGetTask archive hex with
  | .ok (b, key, ok) => return if ok.get then .ok (b.input, key) else .error (archiveMismatch hex)
  | .error e => return .error e

/-- `ARCHIVE/sha512/<hex>`: the verifier's own copy of an object, by content (`archiveGetSha`). -/
def archiveGet (archive hex : String) : IO (Except String ByteArray) :=
  return (← archiveGetSha archive hex).map (·.1)
