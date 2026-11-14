import Flock

/-!
`flock-served CIRCUIT PUBLIC STATEMENT PARTITION PROGRAM`: `verity-verify-proof verify --zk --partition --program`'s work on the
statement before any session, a step at a time, as `verifyCmd` runs it at one table, one `PHASE {json}` line per step on
stdout: the circuit file's read, its SHA-512 (what `--archive`'s `archiveGetSha` holds it to), `Zk.maskLogOf`, the parse
(`ProvedScope.circuitOf` at that SHA-512, as `circuitOfKnowing` reads it: `HmOut.parseCommitWith`, a templated
statement's template or a flat statement's netlist), the scope
check (`ProvedScope.checkOf`), the port roots (`PortRoots.checkOf`), the knowledge-soundness form (`KsScope.checkOf`, without a
sample), and the population's setup (`Stmt.setupTablesOf` without a sample, the tables a sampled session's setup derives its
own from). Each check's verdict is on its line. `flock_served.sh` builds it against the checkout's verifier and runs it on a
flat or a templated statement of one served kind; it is a measurement, not a verifier, and runs nothing `verify` doesn't.
-/

open Flock Lean

def statusKb (field : String) : IO Nat := do
  let s ← IO.FS.readFile "/proc/self/status"
  for l in s.splitOn "\n" do
    if l.startsWith field then
      return ((String.ofList ((l.toList.drop field.length).filter Char.isDigit)).toNat?).getD 0
  return 0

def report (phase : String) (t0 : Nat) (extra : List (String × Json) := []) : IO Unit := do
  let t1 ← IO.monoNanosNow
  let j := Json.mkObj ([("phase", Json.str phase), ("ms", toJson ((t1 - t0) / 1000000)),
    ("rss_mb", toJson ((← statusKb "VmRSS:") / 1024)), ("hwm_mb", toJson ((← statusKb "VmHWM:") / 1024))] ++ extra)
  IO.println s!"PHASE {j.compress}"
  (← IO.getStdout).flush

def verdict (r : Except String Unit) : List (String × Json) :=
  match r with
  | .ok () => [("ok", Json.bool true)]
  | .error e => [("ok", Json.bool false), ("why", Json.str e)]

def main (args : List String) : IO UInt32 := do
  let circuit :: pubPath :: stmt :: partPath :: progPath :: _ := args
    | IO.eprintln "usage: flock-served CIRCUIT PUBLIC STATEMENT PARTITION PROGRAM"; return 2
  let some tags0 := Tags.byName? stmt | IO.eprintln s!"unknown statement {stmt}"; return 2
  let pubB ← IO.FS.readBinFile pubPath
  let program ← IO.FS.readBinFile progPath
  let .ok partition := (String.fromUTF8? (← IO.FS.readBinFile partPath)).elim (.error "not UTF-8") JsonParse.parse
    | IO.eprintln "partition: not JSON"; return 2
  let t0 ← IO.monoNanosNow
  let bytes ← IO.FS.readBinFile circuit
  report "read" t0 [("bytes", toJson bytes.size)]
  let t0 ← IO.monoNanosNow
  let pin := Sha512.hash bytes
  report "sha512" t0 [("pin", Json.str ((hexOf pin).take 16).toString)]
  let count := 1
  let coins := Coins.tree (Zk.coinSpec (tags0.tableNames count))
  let t0 ← IO.monoNanosNow
  let ml := Zk.maskLogOf bytes
  report "maskLogOf" t0 [("mask_log", toJson ml)]
  let tags := Zk.tags tags0 ml count
  let t0 ← IO.monoNanosNow
  let hs : Hashes V := Hashes.ofSha512 fun b => if b == bytes then pin else Sha512.hash b
  let parsed := if tags.hiddenOutputs then HmOut.parseCommitWith hs tags bytes [] else HmRow.parseWith hs tags bytes []
  match parsed with
  | .error e => report "circuitOf" t0 [("error", Json.str e)]; return 1
  | .ok c => report "circuitOf" t0 [("templated", toJson c.tmpl.isSome), ("k_log", toJson c.kLog),
      ("nets", toJson (c.nets.map fun (nm, n) => Json.mkObj [("name", Json.str nm), ("useful", toJson n.useful)]))]
  let t0 ← IO.monoNanosNow
  let scope := ProvedScope.checkOf parsed tags true pubB count false false
  report "ProvedScope.checkOf" t0 (verdict scope)
  let t0 ← IO.monoNanosNow
  let roots := PortRoots.checkOf parsed tags pubB
  report "PortRoots.checkOf" t0 (verdict roots)
  let t0 ← IO.monoNanosNow
  let ks := KsScope.checkOf parsed tags pubB count none
  report "KsScope.checkOf" t0 (verdict ks)
  let t0 ← IO.monoNanosNow
  match Stmt.setupTablesOf parsed tags bytes pubB coins count [] none (some partition) (some program) with
  | .error e => report "setupTablesOf" t0 [("error", Json.str e)]; return 1
  | .ok (_, sts) => report "setupTablesOf" t0 [("tables", toJson sts.size), ("m", toJson (sts.map (·.m)))]
  return 0
