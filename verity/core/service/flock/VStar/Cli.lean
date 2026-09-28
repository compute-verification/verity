import VStar.Compose
import VStar.Json
import VStar.Menu.Pinned
import VStar.Stage

/-!
# The `vstar` command: write V*'s circuit files

    vstar rec-open LANES H BINDINGS OUT     RecOpen{LANES, H}'s circuit.txt
    vstar inner S NAME BINDINGS OUT         InnerRepCheck{S}'s, S a JSON file ({"gf", "ports"}), its unit named NAME
    vstar statements M K_LOG CLAIMS         V*'s statements over an inner session of that shape, as JSON (stdout)
    vstar stage M K_LOG CLAIMS BINDINGS OUT every statement's circuit, OUT/<dir>/circuit.txt
    vstar menu [M K_LOG CLAIMS ...]         V* over every pinned menu's inner session (`VStar.Menu.sessions`), or over
                                            the shapes given; one JSON line each, exit 1 if V* refuses any

A statement's bindings are `{"partition", "program", "program_digests"}`: its `verity/partition/v1` digest, its Program's
SHA-512 and its `program_digests` (`circuit.program_digests`: the program digest keyed by the descriptor id core's codec
names the Definition by, which for `InnerRepCheck_v2{S}` holds S), as core's descriptor codec gives them. `BINDINGS` is a
JSON file of one statement's, or for `stage` a list of them in `statements`' order. Each file is written line by line; its
SHA-512, size and unit rows go to stdout as JSON.
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

/-- The pinned menus' inner sessions, each once, with the models whose menus hold it. -/
def menuSessions : Array (Algebra.Shape × Array String) := Id.run do
  let mut out : Array (Algebra.Shape × Array String) := #[]
  for (w, es) in Menu.pinned do
    for sh in Menu.sessions es do
      match out.findIdx? fun (o, _) => (o.m, o.kLog, o.claims) == (sh.m, sh.kLog, sh.claims) with
      | some i => out := out.modify i fun (o, ws) => (o, if ws.contains w then ws else ws.push w)
      | none => out := out.push (sh, #[w])
  pure out

def shapes : List String → IO (Array (Algebra.Shape × Array String))
  | [] => pure #[]
  | m :: k :: c :: rest => do pure (#[(← shape m k c, #[])] ++ (← shapes rest))
  | _ => throw (IO.userError "menu: shapes are M K_LOG CLAIMS triples")

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
  | "menu" :: given =>
    let mut refused := false
    let todo ← if given.isEmpty then pure menuSessions else shapes given
    let runs := todo.map fun (sh, ws) => (sh, ws, Task.spawn fun _ => statements sh)
    for (sh, ws, t) in runs do
      let base := [("m", (sh.m : Json)), ("k_log", (sh.kLog : Json)), ("claims", (sh.claims : Json)),
        ("models", Json.arr (ws.map Json.str))]
      match t.get with
      | .ok st => IO.println (Json.mkObj (base ++ [("statements", (st.size : Json))])).compress
      | .error e =>
        refused := true
        IO.println (Json.mkObj (base ++ [("refused", (e : Json))])).compress
    pure (if refused then 1 else 0)
  | _ =>
    IO.eprintln ("usage: vstar rec-open LANES H BINDINGS OUT | vstar inner S NAME BINDINGS OUT | "
      ++ "vstar statements M K_LOG CLAIMS | vstar stage M K_LOG CLAIMS BINDINGS OUT | vstar menu [M K_LOG CLAIMS ...]")
    pure 2

end VStar.Cli
