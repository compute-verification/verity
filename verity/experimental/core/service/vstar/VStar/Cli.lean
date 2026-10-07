import VStar.Compose
import VStar.Json
import VStar.Stage

/-!
# The `vstar` command: write V*'s circuit files

    vstar rec-open LANES H BINDINGS OUT     RecOpen{LANES, H}'s circuit.txt
    vstar inner S NAME BINDINGS OUT         InnerRepCheck{S}'s, S a JSON file ({"gf", "ports"}), its unit named NAME
    vstar statements M K_LOG CLAIMS         V*'s statements over an inner session of that shape, as JSON (stdout)
    vstar stage M K_LOG CLAIMS BINDINGS OUT every statement's circuit, OUT/<dir>/circuit.txt

A statement's bindings are `{"partition", "program", "program_digests"}`: its `verity/partition/v1` digest, its Program's
SHA-512 and its `program_digests`, which core's descriptor codec gives. `BINDINGS` is a JSON file of one statement's, or for
`stage` a list of them in `statements`' order. Each file is written line by line; its SHA-512, size and unit rows go to
stdout as JSON.
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

def bindingsOf (j : Json) : Except String Bindings := do
  let o ← jObj j ["partition", "program", "program_digests"] "bindings"
  pure ⟨← jStr (o "partition") "partition", ← jStr (o "program") "program", o "program_digests"⟩

def write (u : UnitC) (sub : Sub) (b : Bindings) (out : String) : IO Json := do
  let l ← orFail u.laid
  let h ← IO.FS.Handle.mk out .write
  let put : String → StateT Digests IO Unit := fun s => do
    h.putStr s
    modify (·.update s)
  let e ← orFail (emitCircuit put u l sub b)
  let ((), d) ← e.run {}
  h.flush
  pure (Json.mkObj [("sha512", Flock.hexOf d.sha512.finish), ("bytes", (d.bytes : Json)), ("unit_rows", (l.useful : Json))])

def parseNat (s : String) : IO Nat :=
  match s.toNat? with
  | some n => pure n
  | none => throw (IO.userError s!"{s}: not a natural number")

def shape (m k c : String) : IO Algebra.Shape := do
  pure { m := ← parseNat m, kLog := ← parseNat k, claims := ← parseNat c }

def stmtJson : Stmt → Json
  | s@(.recOpen level L H n) => Json.mkObj [("name", s.name), ("template", "rec-open"), ("dir", s.dir),
      ("instances", (n : Json)), ("level", (level : Json)), ("LANES", (L : Json)), ("H", (H : Json))]
  | s@(.inner j _ p) => Json.mkObj [("name", s.name), ("template", "rec-residuals"), ("dir", s.dir),
      ("instances", (s.instances : Json)), ("part", (j : Json)), ("S", p.S.toJson),
      ("res", Json.arr (p.res.toArray.map fun (i : Nat) => (i : Json))),
      ("v", Json.arr (p.v.toArray.map fun (i : Nat) => (i : Json)))]

def cmd (args : List String) : IO UInt32 := do
  match args with
  | ["rec-open", L, H, b, out] =>
    let L ← parseNat L
    let H ← parseNat H
    IO.println (← write (← orFail (recOpenUnit L H)) (recOpenSub L H) (← orFail (bindingsOf (← readJson b))) out).compress
    pure 0
  | ["inner", s, name, b, out] =>
    let S ← orFail (RepS.ofJson (← readJson s))
    IO.println (← write (← orFail (innerUnit S name)) (innerSub S name) (← orFail (bindingsOf (← readJson b))) out).compress
    pure 0
  | ["statements", m, k, c] =>
    let st ← orFail (statements (← shape m k c))
    IO.println (Json.arr (st.map stmtJson)).compress
    pure 0
  | ["stage", m, k, c, b, out] =>
    let st ← orFail (statements (← shape m k c))
    let bj ← readJson b
    let bs ← orFail (do (← jArr bj "bindings").mapM bindingsOf)
    unless bs.size == st.size do throw (IO.userError s!"{bs.size} bindings for {st.size} statements")
    let mut res := #[]
    for (s, bb) in st.zip bs do
      let dir := s!"{out}/{s.dir}"
      IO.FS.createDirAll dir
      let j ← write (← orFail s.unit) s.sub bb s!"{dir}/circuit.txt"
      res := res.push (j.mergeObj (Json.mkObj [("name", s.name), ("dir", s.dir)]))
    IO.println (Json.arr res).compress
    pure 0
  | _ =>
    IO.eprintln ("usage: vstar rec-open LANES H BINDINGS OUT | vstar inner S NAME BINDINGS OUT | "
      ++ "vstar statements M K_LOG CLAIMS | vstar stage M K_LOG CLAIMS BINDINGS OUT")
    pure 2

end VStar.Cli
