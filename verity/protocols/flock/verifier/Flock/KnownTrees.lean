/-
`flock-verify known-trees`: the verifier's list of a statement's known trees (`Flock.RegKnown`), from its own files,
written for its challenger.
-/
module

public import Flock.Comp
public import Flock.JsonKeys
public import Flock.RegKnown

@[expose] public section

namespace Flock.KnownTrees

/-- **The known trees of the statement whose public file is `pub`**, under the verifier's own registered reads, the
file `registered` (`Registered.knownTreesOf` on the reads and the public file's header): each port the reads name at
known content, held to its content's root, with the tree the statement publishes for it. A file that names a key
twice in one object is refused, as `flock-verify`'s `registeredFile` refuses it. -/
def listed (registered pub : ByteArray) : V (Array (String × ByteArray)) := do
  let some text := String.fromUTF8? registered | throw "registered reads: not UTF-8"
  if let some k := JsonKeys.dupKey text then
    throw s!"registered reads: an object of the file names the key {k} twice, which a JSON parser would read as once"
  let some nl := findByte pub 10 | throw "public file: no header line"
  let some head := String.fromUTF8? (pub.extract 0 nl) | throw "public file: its header is not UTF-8"
  Registered.knownTreesOf (← Flock.JsonParse.parse text) (← Flock.JsonParse.parse head)

/-- `flock-verify known-trees --registered F --public P --out LIST`, given the three paths: **the verifier's list of
a statement's known trees** (`listed`, `Registered.knownList`), from the verifier's own registered reads `F` and the
statement's public file `P`. The verifier's challenger sends `LIST` after the draw in its answer to the session's
`Register` (`flock-circuit serve --known-trees LIST`), where the ZK server's release side holds the statement's trees
to it (`Firewall.Salts.readsKnown`). One `KNOWN-TREES` line names each listed port with its entry, in hex. -/
def cmd (registered pub out : Option String) : IO UInt32 := do
  let (some rf, some pf, some out) := (registered, pub, out)
    | IO.eprintln "usage: flock-verify known-trees --registered F --public P --out LIST"; return 2
  let registered ← IO.FS.readBinFile rf
  let pub ← IO.FS.readBinFile pf
  match listed registered pub with
  | .error why => IO.println s!"KNOWN-TREES {(Lean.Json.mkObj [("error", Lean.Json.str why)]).compress}"; return 1
  | .ok ts =>
    IO.FS.writeBinFile out (Registered.knownList (ts.map (·.2)))
    IO.println s!"KNOWN-TREES {(Lean.Json.mkObj (ts.toList.map fun (name, e) => (name, Lean.Json.str (hexOf e)))).compress}"
    return 0

end Flock.KnownTrees
