import VStar.Compose
import VStar.Json

/-!
# The `vstar` command: write a V* statement's circuit file

    vstar rec-open LANES H BINDINGS OUT     RecOpen{LANES, H}'s circuit.txt
    vstar inner S NAME BINDINGS OUT         InnerRepCheck{S}'s, S a JSON file ({"gf", "ports"}), its unit named NAME

`BINDINGS` is a JSON file `{"partition", "program", "program_digests"}`: the statement's `verity/partition/v1` digest, its
Program's SHA-512 and its `program_digests`, which core's descriptor codec gives. The file is written line by line; its
SHA-512 and size go to stdout as JSON.
-/

namespace VStar.Cli

open Lean (Json)

def readJson (path : String) : IO Json := do
  match Json.parse (← IO.FS.readFile path) with
  | .ok j => pure j
  | .error e => throw (IO.userError s!"{path}: not JSON: {e}")

def orFail {α : Type} (x : Except String α) : IO α :=
  match x with
  | .ok a => pure a
  | .error e => throw (IO.userError e)

def bindings (path : String) : IO Bindings := do
  let o ← orFail (jObj (← readJson path) ["partition", "program", "program_digests"] "bindings")
  pure ⟨← orFail (jStr (o "partition") "partition"), ← orFail (jStr (o "program") "program"), o "program_digests"⟩

def write (u : UnitC) (sub : Sub) (b : Bindings) (out : String) : IO Unit := do
  let l ← orFail u.laid
  let h ← IO.FS.Handle.mk out .write
  let put : String → StateT Digests IO Unit := fun s => do
    h.putStr s
    modify (·.update s)
  let e ← orFail (emitCircuit put u l sub b)
  let ((), d) ← e.run {}
  h.flush
  IO.println (Json.mkObj [("sha512", Flock.hexOf d.sha512.finish), ("bytes", (d.bytes : Json)),
    ("unit_rows", (l.useful : Json))]).compress

def parseNat (s : String) : IO Nat :=
  match s.toNat? with
  | some n => pure n
  | none => throw (IO.userError s!"{s}: not a natural number")

def cmd (args : List String) : IO UInt32 := do
  match args with
  | ["rec-open", L, H, b, out] =>
    let L ← parseNat L
    let H ← parseNat H
    write (← orFail (recOpenUnit L H))
      { id := s!"rec-open/l{L}-h{H}", template := "rec-open", inputs := ["row", "aux", "acc", "coef", "dirs"],
        outputs := ["out", "acc_out"] } (← bindings b) out
    pure 0
  | ["inner", s, name, b, out] =>
    let S ← orFail (RepS.ofJson (← readJson s))
    write (← orFail (innerUnit S name))
      { id := name, template := "rec-residuals", inputs := S.ports.map (·.1) ++ ["v"], outputs := ["out"] } (← bindings b) out
    pure 0
  | _ =>
    IO.eprintln "usage: vstar rec-open LANES H BINDINGS OUT | vstar inner S NAME BINDINGS OUT"
    pure 2

end VStar.Cli
