module

public import Flock.Registered
public import Flock.ProvedScope

/-!
# The verifier's program, read once

A setup reads the verifier's own program (`--program`) through `TemplateQuery.Desc.parse` and `Program.decode`, each of
which reads the descriptor's bytes as JSON (`Program.readJson`) and hashes its canonical form. Each function below named `…By` is the function it is
named after with those two readings as its parameters `readDesc` and `readProg`, and at the readings themselves it is that
function (`Stmt.setupHiddenOf_eq_by`, `Stmt.setupTablesOf_eq_by`). `ProgramRead` holds a program with its JSON and
both readings of it, each made when first asked for and kept, the two readings from the one JSON; its `jsonAt`, `descAt` and
`progAt` are `Program.readJson` and the two readings on every input (`ProgramRead.jsonAt_of`, `descAt_of`, `progAt_of`), so
a run that hands them to its setups parses its program once. `TemplateQuery.Desc.parseOf`, `Program.decodeOf` and
`ProvedScope.programDigestOf` are the two readings and the program's digest from that JSON.
-/

@[expose] public section

namespace Flock

open Lean (Json)

namespace TemplateQuery

/-- `Desc.parseWith` from the descriptor's bytes as `Program.readJson` read them (`Desc.parseWith_eq_of`). -/
def Desc.parseOfWith {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (sha512 : ByteArray → M ByteArray)
    (read : Option (Except String Json)) : M Desc := do
  let some parsed := read | throw "program: not UTF-8"
  let j ← match parsed with
    | .ok j => pure j
    | .error e => throw s!"program: not JSON: {e}"
  if (j.getObjValAs? String "schema").toOption != some "verity-ir/descriptor/v1" then
    throw "program: not a verity-ir/descriptor/v1 descriptor"
  let bare := match j with
    | .obj kvs => Json.mkObj (kvs.toArray.filter (·.1 != "annotations")).toList
    | x => x
  let callees ← match j.getObjVal? "callees" with
    | .ok (.arr a) => a.mapM fun c => match c.getStr? with | .ok s => pure s | .error _ => throw "program: a callee is not a string"
    | _ => throw "program: callees"
  let defs ← match j.getObjVal? "definitions" with
    | .ok (.obj kvs) => kvs.toArray.mapM fun (id, dj) => do
      let form := (dj.getObjValAs? String "form").toOption.getD ""
      if form == "primitive" then pure (id, ({ prim := true, nodes := #[] } : DDef))
      else if form == "composite" then
        let nodes ← match (dj.getObjVal? "body").toOption.bind (fun b => (b.getObjVal? "nodes").toOption) with
          | some (.arr ns) => ns.mapM fun nj => do
            let form := (nj.getObjValAs? String "form").toOption.getD ""
            let some fn := (nj.getObjValAs? Nat "fn").toOption | throw s!"program: {id}: a node without a callee"
            if fn ≥ callees.size then throw s!"program: {id}: a node's callee is outside callees"
            pure ({ form, fn, n := (nj.getObjValAs? Nat "n").toOption.getD 1 } : DNode)
          | _ => throw s!"program: {id}: a composite without body nodes"
        pure (id, ({ prim := false, nodes } : DDef))
      else throw s!"program: {id}: an unknown definition form {form}"
    | _ => throw "program: definitions"
  let some root := (j.getObjValAs? String "root").toOption | throw "program: root"
  if (defs.find? (·.1 == root)).isNone then throw "program: the root is not a definition"
  return { callees, defs, root, sha512 := ← sha512 (← canon bare).toUTF8 }

theorem Desc.parseWith_eq_of {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]
    (sha512 : ByteArray → M ByteArray) (bytes : ByteArray) :
    Desc.parseWith sha512 bytes = Desc.parseOfWith sha512 (Program.readJson bytes) := by
  unfold Desc.parseWith Desc.parseOfWith Program.readJson
  generalize String.fromUTF8? bytes = o
  cases o <;> rfl

/-- `Desc.parse` from the descriptor's bytes as `Program.readJson` read them (`Desc.parse_eq_of`). -/
def Desc.parseOf (read : Option (Except String Json)) : V Desc := Desc.parseOfWith (fun b => pure (Sha512.hash b)) read

theorem Desc.parse_eq_of (bytes : ByteArray) : Desc.parse bytes = Desc.parseOf (Program.readJson bytes) :=
  Desc.parseWith_eq_of _ bytes

end TemplateQuery

namespace ProvedScope

/-- `programDigest` from the program's bytes as `Program.readJson` read them (`programDigest_eq_of`). -/
def programDigestOf (read : Option (Except String Lean.Json)) : Option String := do
  let j ← (← read).toOption
  let bare := match j with
    | .obj kvs => Lean.Json.mkObj (kvs.toArray.filter (·.1 != "annotations")).toList
    | x => x
  return hexOf (Sha512.hash (← (canon bare).toOption).toUTF8)

theorem programDigest_eq_of (bytes : ByteArray) : programDigest bytes = programDigestOf (Program.readJson bytes) := by
  unfold programDigest programDigestOf Program.readJson
  generalize String.fromUTF8? bytes = o
  cases o <;> rfl

end ProvedScope

/-- A program, its bytes read as JSON (`Program.readJson`) and its two readings, `TemplateQuery.Desc.parse` and
`Program.decode` of `bytes`. -/
structure ProgramRead where
  bytes : ByteArray
  json : Thunk (Option (Except String Json))
  desc : Thunk (V TemplateQuery.Desc)
  prog : Thunk (V Program.Prog)

/-- `program` with its JSON and its readings, none made before it is asked for, both readings from the one JSON. -/
def ProgramRead.of (program : ByteArray) : ProgramRead :=
  let json := Thunk.mk fun _ => Program.readJson program
  ⟨program, json, Thunk.mk fun _ => TemplateQuery.Desc.parseOf json.get, Thunk.mk fun _ => Program.decodeOf json.get⟩

/-- `Program.readJson`: on `r`'s own bytes the JSON `r` keeps, elsewhere read anew. -/
def ProgramRead.jsonAt (r : ProgramRead) (b : ByteArray) : Option (Except String Json) :=
  match withPtrEqDecEq b r.bytes (fun _ => inferInstance) with
  | isTrue _ => r.json.get
  | isFalse _ => Program.readJson b

/-- `TemplateQuery.Desc.parse`: on `r`'s own bytes the reading `r` keeps, elsewhere made anew. -/
def ProgramRead.descAt (r : ProgramRead) (b : ByteArray) : V TemplateQuery.Desc :=
  match withPtrEqDecEq b r.bytes (fun _ => inferInstance) with
  | isTrue _ => r.desc.get
  | isFalse _ => TemplateQuery.Desc.parse b

/-- `Program.decode`: on `r`'s own bytes the reading `r` keeps, elsewhere made anew. -/
def ProgramRead.progAt (r : ProgramRead) (b : ByteArray) : V Program.Prog :=
  match withPtrEqDecEq b r.bytes (fun _ => inferInstance) with
  | isTrue _ => r.prog.get
  | isFalse _ => Program.decode b

theorem ProgramRead.jsonAt_of (program : ByteArray) : (ProgramRead.of program).jsonAt = Program.readJson := by
  funext b
  unfold ProgramRead.jsonAt
  split
  · rename_i h _
    exact congrArg Program.readJson (Eq.symm h)
  · rfl

theorem ProgramRead.descAt_of (program : ByteArray) : (ProgramRead.of program).descAt = TemplateQuery.Desc.parse := by
  funext b
  unfold ProgramRead.descAt
  split
  · rename_i h _
    exact (TemplateQuery.Desc.parse_eq_of program).symm.trans (congrArg TemplateQuery.Desc.parse (Eq.symm h))
  · rfl

theorem ProgramRead.progAt_of (program : ByteArray) : (ProgramRead.of program).progAt = Program.decode := by
  funext b
  unfold ProgramRead.progAt
  split
  · rename_i h _
    exact (Program.decode_eq_of program).symm.trans (congrArg Program.decode (Eq.symm h))
  · rfl

/-- `ProvedScope.programDigestOf` at the JSON a `ProgramRead` keeps is `ProvedScope.programDigest`. -/
theorem ProvedScope.programDigestOf_read (read bytes : ByteArray) :
    ProvedScope.programDigestOf ((ProgramRead.of read).jsonAt bytes) = ProvedScope.programDigest bytes := by
  rw [ProgramRead.jsonAt_of, ProvedScope.programDigest_eq_of]

namespace HmRow

/-- `deriveTemplateUnits`, the program read by `readDesc`. -/
def deriveTemplateUnitsBy (readDesc : ByteArray → V TemplateQuery.Desc)
    (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) : V String := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let name := (q.getObjValAs? String "name").toOption.getD ""
  let version := (q.getObjValAs? Nat "version").toOption.getD 0
  let some ts := TemplateQuery.templatesOf name version ((q.getObjVal? "params").toOption.getD Json.null)
    | throw s!"partition: {name} v{version} is not a template query"
  let ts ← ts
  let d ← readDesc program
  if (obj.getObjValAs? String "program").toOption != some (hexOf d.sha512) then
    throw "partition: the verifier's program is not the one the partition object names"
  let rs ← match TemplateQuery.evaluate d ts with
    | .ok rs => pure rs
    | .error e => throw s!"partition: {e}"
  let some (t, cls) := queryTemplate c | throw "partition: META program_digests does not name the circuit's one template"
  if !ts.contains t then throw s!"partition: the circuit's template {t} is not one the query lists"
  let pop := TemplateQuery.population rs
  for i in indices do
    match TemplateQuery.locate rs i with
    | none => throw s!"partition: unit {i} is outside the partition's {pop} units"
    | some (_, _, ti) => if ti != t then throw s!"partition: unit {i} is an instance of {ti}, not the circuit's template {t}"
  let of := if cls then s!" (META definition of the unit class {(circuitTemplate c).getD ""}; that the class circuit lowers it is not checked)" else ""
  return s!"units derived: {name} v{version} over the verifier's program ({pop} units); the statement's {indices.size} units are instances of {t}{of}"

/-- `deriveQwordUnits`, the program read by `readProg`. -/
def deriveQwordUnitsBy (readProg : ByteArray → V Program.Prog)
    (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) : V String := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let version := (q.getObjValAs? Nat "version").toOption.getD 0
  let ps := (q.getObjVal? "params").toOption.getD Json.null
  let X := (ps.getObjValAs? Nat "X").toOption.getD 0
  let W := (ps.getObjValAs? Nat "W").toOption.getD 0
  let P ← readProg program
  if (obj.getObjValAs? String "program").toOption != some (hexOf P.sha512) then
    throw "partition: the verifier's program is not the one the partition object names"
  let calls ← match Extract.evaluate P X with
    | .ok cs => pure cs
    | .error e => throw s!"partition: {e}"
  let some t := circuitTemplate c | throw "partition: META program_digests does not name the circuit's one template"
  let checked ← Extract.checkCalls P calls X W (report := version == 2)
  if let some (k, r) := (Extract.cutFailures calls checked)[0]? then
    throw s!"partition: cut: Call {k}'s cut fails {r.codes}{if r.wide.isEmpty then "" else s!" (too wide, (unit, bits, values): {(r.wide.extract 0 4).toList})"}"
  let across := if version == 2 then
      s!"; values computed in two units, reported, not refused: {checked.foldl (fun n (_, r) => n + r.across) 0} pair(s) over its Definitions"
    else ""
  match Extract.claimedUnits P calls t indices with
  | .ok note => return s!"units derived: Q_word v{version} {"{"}X: {X}, W: {W}{"}"} over the verifier's program ({note}); every Call's cut holds the partition invariant and the width rule{across}"
  | .error e => throw s!"partition: {e}"

/-- `deriveQcallUnits`, the program read by `readProg`. -/
def deriveQcallUnitsBy (readProg : ByteArray → V Program.Prog)
    (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) : V String := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let X := ((q.getObjVal? "params").toOption.getD Json.null |>.getObjValAs? Nat "X").toOption.getD 0
  let P ← readProg program
  if (obj.getObjValAs? String "program").toOption != some (hexOf P.sha512) then
    throw "partition: the verifier's program is not the one the partition object names"
  let calls ← match Qcall.evaluate P X with
    | .ok cs => pure cs
    | .error e => throw s!"partition: {e}"
  let some t := circuitTemplate c | throw "partition: META program_digests does not name the circuit's one template"
  let mut across := 0
  let mut seen : Array Nat := #[]
  for x in calls do
    let r := Qcall.check x.c X
    if !r.codes.isEmpty then
      throw s!"partition: cut: Call {x.k}'s cut fails {r.codes}{if r.wide.isEmpty then "" else s!" (too wide, (unit, bits): {(r.wide.extract 0 4).toList})"}"
    if !seen.contains x.d then
      seen := seen.push x.d
      across := across + r.across.size
  match Qcall.claimedUnits P calls t indices with
  | .ok note => return s!"units derived: Q_call v1 {"{"}X: {X}{"}"} over the verifier's program ({note}); every Call's cut holds the partition invariant and the width rule; values computed in two units, reported, not refused: {across} pair(s) over its Definitions"
  | .error e => throw s!"partition: {e}"

/-- `qwordFinding`, the program read by `readProg`. -/
def qwordFindingBy (readProg : ByteArray → V Program.Prog)
    (c : Circuit) (pub : Public) (obj : Json) (program : ByteArray) (f : PartitionFinding) :
    V (Option PartitionFinding) := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  if EVALUABLE.contains ((q.getObjValAs? String "name").toOption.getD "", (q.getObjValAs? Nat "version").toOption.getD 0) then
    let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
    pure (some { f with derived := true, note := ← deriveQwordUnitsBy readProg c obj program indices })
  else if ((q.getObjValAs? String "name").toOption.getD "", (q.getObjValAs? Nat "version").toOption.getD 0) == Q_CALL then
    let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
    pure (some { f with derived := true, note := ← deriveQcallUnitsBy readProg c obj program indices })
  else pure (some f)

/-- `templateStrata`, the program read by `readDesc`. -/
def templateStrataBy (readDesc : ByteArray → V TemplateQuery.Desc)
    (law : String) (obj : Json) (program : ByteArray) :
    V (Array (String × Array (Nat × Nat)) × Nat) := do
  let (name, version) ← checkPartitionObject obj
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let some ts := TemplateQuery.templatesOf name version ((q.getObjVal? "params").toOption.getD Json.null)
    | throw s!"{law}: {name} v{version} defines no strata (Q_template_instance v0 and Q_template_instances v0 do)"
  let ts ← ts
  let d ← readDesc program
  if (obj.getObjValAs? String "program").toOption != some (hexOf d.sha512) then
    throw "partition: the verifier's program is not the one the partition object names"
  let rs ← match TemplateQuery.evaluate d ts with
    | .ok rs => pure rs
    | .error e => throw s!"partition: {e}"
  return (Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)), TemplateQuery.population rs)

/-- `stratifiedLaw`, the program read by `readDesc`. -/
def stratifiedDesignBy (readDesc : ByteArray → V TemplateQuery.Desc)
    (obj : Json) (program : ByteArray) (k : Nat) : V (Draw.Law × Nat) := do
  let (strata, population) ← templateStrataBy readDesc "stratified" obj program
  match Draw.derive k strata with
  | .ok law => return (law, population)
  | .error e => throw s!"stratified: {e}"

/-- `workLaw`, the program read by `readDesc`. -/
def workDesignBy (readDesc : ByteArray → V TemplateQuery.Desc)
    (obj : Json) (program : ByteArray) (k : Nat) (table : Array (String × Nat × Nat)) : V (Draw.Law × Nat) := do
  let (strata, population) ← templateStrataBy readDesc "work" obj program
  match Draw.deriveWork k strata table with
  | .ok law => return (law, population)
  | .error e => throw s!"work: {e}"

end HmRow

namespace HmLinks

open HmRow

/-- `check`, the program read by `readProg`. -/
def checkBy (readProg : ByteArray → V Program.Prog)
    (c : Circuit) (pop : Public) (readers : Option (Array Nat)) (program : ByteArray) : V Nat := do
  let links ← parse c
  let P ← readProg program
  let ns := nodes P
  checkAt c pop P links (unitAt ns) (ownerAt ns) (count ns) readers

end HmLinks

/-- `Stmt.setupHiddenOf`, the program read by `readDesc` and `readProg` (`Stmt.setupHiddenOf_eq_by`). -/
def Stmt.setupHiddenBy (readDesc : ByteArray → V TemplateQuery.Desc) (readProg : ByteArray → V Program.Prog)
    (parsed : V Circuit) (tags : Tags) (publicFile : ByteArray) (coins : Coins)
    (draw : Option Json := none) (partition : Option Json := none) (program : Option ByteArray := none)
    (publicInputs : Option ByteArray := none) : V Stmt := do
  let c ← parsed
  let pi ← HmIn.parse c
  let finding0 ← if tags.bindings then (some <$> HmRow.checkPartition c partition)
    else if partition.isSome then throw s!"{tags.name}: the statement binds no partition" else pure none
  let some merkle := MerkleScheme.find? c.leafScheme | throw s!"leaf-commitment scheme {c.leafScheme}: not one this verifier implements"
  let pop ← HmIn.loadPublic tags c pi publicFile
  HmIn.checkOwn pi pop publicInputs
  let pub ← match draw with
    | some d => HmIn.drawn c pi pop d
    | none => pure pop
  if let some d := draw then
    if let (Except.ok (Draw.Law.stratified k strata), some obj, some prog) :=
        ((Draw.UnitDraw.ofJson d).map (·.law), partition, program) then
      let (law, _) ← HmRow.stratifiedDesignBy readDesc obj prog k
      if law != Draw.Law.stratified k strata then throw "U2: the draw's strata are not the verifier's own (its program and query)"
  HmOut.checkPublic c pub
  let linked := J.has c.metaJ "links"
  let finding ← match finding0, partition, program with
    | some f, some obj, some prog =>
      let q := (obj.getObjVal? "query").toOption.getD Json.null
      if (TemplateQuery.templatesOf ((q.getObjValAs? String "name").toOption.getD "") ((q.getObjValAs? Nat "version").toOption.getD 0) Json.null).isSome then
        let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
        let note ← HmRow.deriveTemplateUnitsBy readDesc c obj prog indices
        let wires ← HmLinks.checkBy readProg c pop (HmLinks.readersOf draw) prog
        let linking := if wires == 0 && !linked then "; no cut wire between the statement's units"
          else s!"; Linking checked: {wires} cut wire(s) into the {pub.n} proved units of the population's {pop.n}, each reader's link row the writer's committed output row"
        pure (some { f with derived := true, note := note ++ linking })
      else if linked then throw "META links: Linking is checked under a template query (Q_template_instance v0, Q_template_instances v0) only"
      else HmRow.qwordFindingBy readProg c pub obj prog f
    | f, _, _ =>
      if linked then throw "META links: Linking needs the verifier's own program and partition (--program, --partition)"
      pure f
  let n := pub.n
  let blocksReal := (n + c.g - 1) / c.g
  let nbl := max 3 (log2ceil (max blocksReal 1))
  let m := c.kLog + nbl
  if c.kLog > 27 || m > 35 then throw s!"m = {m}: no fast100 schedule (or k_log > 27)"
  let pin ← HmRow.pin c
  let (da, db) := HmIn.delta c pi pin
  let regions := (← HmOut.regions c pub) ++ (← HmIn.regions c pi pub)
  checkInRange c.kLog regions.size (PT_LOCAL + nbl)
  checkRegionBytes nbl regions
  let identity := if J.has c.metaJ "row_schemas" then Tags.rowV2Identity tags.identity else tags.identity
  let identity := if c.ports.any (·.segs.isSome) then Tags.rowSegIdentity identity else identity
  let v2Run := (List.range c.outRowPorts.size).any fun j =>
    (c.ports.getD (c.ports.size - c.outRowPorts.size + j) default).segs.isNone && c.outRowPorts[j]! > 1
  let identity := if !J.has c.metaJ "out_row_ports" then identity
    else if v2Run then Tags.outRowsV2Identity identity else Tags.outRowsIdentity identity
  let identity := if HmRow.pinnedCount c.metaJ > 0 then Tags.pinnedIdentity identity else identity
  let mut h := tags.statement.toUTF8 ++ c.sha ++ (← identityBytes identity)
  for x in [n, blocksReal, nbl, m, c.kLog, pin, regions.size, c.g] do h := pushLe64 h x
  for d in [da, db] do
    h := pushLe64 h d.size
    for (i, cc) in d do h := pushLe32 (pushLe32 h i) cc
  let digest := Sha512.hash h
  let sigma := Sha512.hash (tags.sigmaTag.toUTF8 ++ c.sha ++ digest ++ pub.sha)
  let hello ← helloOf tags sigma (PT_LOCAL + nbl) coins
  return { tags, leaf := default, c, pub, n, blocksReal, nbl, m, mPts := PT_LOCAL + nbl, pin, regions, da, db, digest, sigma,
           hello, coinTree := coins.tree?, merkle, partition := (finding.map (·.toJson)).getD Json.null }

/-- `Stmt.setupTablesOf`, the program read by `readDesc` and `readProg` where a one-table session with hidden outputs sets up
(`Stmt.setupTablesOf_eq_by`). -/
def Stmt.setupTablesBy (readDesc : ByteArray → V TemplateQuery.Desc) (readProg : ByteArray → V Program.Prog)
    (parsed : V Circuit) (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : Nat)
    (tables : List (String × ByteArray) := []) (draw : Option Json := none) (partition : Option Json := none)
    (program : Option ByteArray := none) (work : Option (Nat × Array (String × Nat × Nat)) := none)
    (stratified : Option Nat := none) (closure : Option (Std.HashMap Nat (Array Nat)) := none)
    (publicInputs : Option ByteArray := none) : V (SessionSpec × Array Stmt) := do
  -- a closure sample proves more units than it samples, so what it adds must be the verifier's own closure of the sample
  if let some d := draw then
    if let .ok u := Draw.UnitDraw.ofJson d then
      if let .work .. := u.law then
        match closure with
        | some cm =>
          if u.closure != Draw.closureOf cm u.units then
            throw "U3: the draw's closure is not the verifier's own (its closure map)"
        | none =>
          if !u.closure.isEmpty then throw "U3: a closure draw needs the verifier's own closure map (--closure)"
  -- a stratified sample's K and strata set every k_s, so its sampling design must be the verifier's own derivation, never taken as stated
  if let some d := draw then
    if let .ok (Draw.Law.stratified k strata) := (Draw.UnitDraw.ofJson d).map (·.law) then
      let (some obj, some prog, some K) := (partition, program, stratified)
        | throw "U2: a stratified draw needs the verifier's own K, program and partition (--stratified, --program, --partition)"
      if k != K then throw s!"U2: the draw's K is {k}, not the verifier's {K}"
      let (law, _) ← HmRow.stratifiedDesignBy readDesc obj prog K
      if law != Draw.Law.stratified k strata then
        throw "U2: the draw's strata are not the verifier's own (its K, program and query)"
  -- a work sample's K and stated work set every k_s, so its sampling design must be the verifier's own derivation, never taken as stated
  if let some d := draw then
    if let .ok (Draw.Law.work k strata) := (Draw.UnitDraw.ofJson d).map (·.law) then
      let (some obj, some prog, some (K, table)) := (partition, program, work)
        | throw "U2: a work draw needs the verifier's own K, work table, program and partition (--work, --work-table, --program, --partition)"
      if k != K then throw s!"U2: the draw's K is {k}, not the verifier's {K}"
      let (law, _) ← HmRow.workDesignBy readDesc obj prog K table
      if law != Draw.Law.work k strata then
        throw "U2: the draw's work law is not the verifier's own (its K, program, query and work table)"
  if publicInputs.isSome && !tags.hiddenOutputs then
    throw s!"--public-inputs: a {tags.name} statement has no public input ports (only a statement with hidden outputs has them)"
  let setup := fun (t : Tags) (d : Option Json) => do
    if tags.hiddenOutputs then
      let st ← Stmt.setupHidden t circuitFile publicFile coins tables d partition program publicInputs
      if HmRow.pinnedCount st.c.metaJ != 0 then throw s!"{tags.name}: refused: {HmRow.pinnedWhy}"
      pure st
    else
      let st ← Stmt.setupH t circuitFile publicFile coins tables d partition program
      if J.has st.c.metaJ "public_inputs" || J.has st.c.metaJ "shared_reads" then
        throw s!"META: public_inputs, shared_reads: a {tags.name} statement has no public input ports or shared reads (only a statement with hidden outputs has them)"
      pure st
  let setupOne := fun (d : Option Json) => do
    if tags.hiddenOutputs then
      let st ← Stmt.setupHiddenBy readDesc readProg parsed tags publicFile coins d partition program publicInputs
      if HmRow.pinnedCount st.c.metaJ != 0 then throw s!"{tags.name}: refused: {HmRow.pinnedWhy}"
      pure st
    else
      let st ← Stmt.setupH tags circuitFile publicFile coins tables d partition program
      if J.has st.c.metaJ "public_inputs" || J.has st.c.metaJ "shared_reads" then
        throw s!"META: public_inputs, shared_reads: a {tags.name} statement has no public input ports or shared reads (only a statement with hidden outputs has them)"
      pure st
  if count ≤ 1 then
    let st ← setupOne draw
    return (st.spec, #[st])
  let some tt := tags.manyTables | throw s!"{tags.name}: one table per session"
  if count > Tags.MAX_TABLES then throw s!"tables: at most {Tags.MAX_TABLES} per session, not {count}"
  let splits := fun (n : Nat) => if n == 0 || n % count != 0 then throw s!"tables: {n} units do not split into {count} equal tables"
    else (pure () : V Unit)
  -- the sample's law and count are refused before the circuit is read
  let u? ← draw.mapM Draw.UnitDraw.ofJson
  if let some u := u? then
    match u.law with
    | .subset _ | .stratified .. => splits u.units.size
    | _ => throw Draw.TABLES_LAW
  -- without a sample, the population's units (`setupH` checks a sample's population against the file)
  let (population, units) ← match u? with
    | some u => pure (u.population, u.units)
    | none => do
      let n ← if tags.hiddenOutputs then do
          let c ← HmOut.parse tags circuitFile tables
          pure (← HmIn.loadPublic tags c (← HmIn.parse c) publicFile).n
        else do pure (← HmRow.loadPublic tags (← HmRow.parse tags circuitFile tables) publicFile).n
      pure (n, Array.range n)
  splits units.size
  let k := units.size / count
  let sts ← (Array.range count).mapM fun j =>
    let part := Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson population), ("k", Lean.toJson k),
      ("units", Json.arr ((units.extract (j * k) (j * k + k)).map Lean.toJson))]
    setup { tags with identity := tags.tableIdentity count j } (some part)
  let mPts := sts[0]!.mPts
  if sts.any (·.mPts != mPts) then throw "tables: the tables' link points have different lengths"
  let sigma := Sha512.hash (sts.foldl (fun h st => h ++ st.sigma) (pushLe64 tt.sigmaTag.toUTF8 count))
  let names := tags.tableNames count
  let hello ← helloOf tags sigma mPts coins names
  return ({ tables := names, domain := tags.tableDomain count, sigma, hello, mPts, retainRounds := tags.retainRounds,
            unitDraw := draw, instances := units.size, sha512Digests := tags.hm96Rows, coinTree := coins.tree? }, sts)

set_option maxRecDepth 20000 in
theorem Stmt.setupHiddenOf_eq_by (parsed : V Circuit) (tags : Tags) (publicFile : ByteArray) (coins : Coins)
    (draw partition : Option Json) (program publicInputs : Option ByteArray) :
    Stmt.setupHiddenOf parsed tags publicFile coins draw partition program publicInputs
      = Stmt.setupHiddenBy TemplateQuery.Desc.parse Program.decode parsed tags publicFile coins draw partition program
          publicInputs :=
  rfl

set_option maxRecDepth 20000 in
theorem Stmt.setupTablesOf_eq_by (parsed : V Circuit) (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (count : Nat) (tables : List (String × ByteArray)) (draw partition : Option Json) (program : Option ByteArray)
    (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) :
    Stmt.setupTablesOf parsed tags circuitFile publicFile coins count tables draw partition program work stratified closure
        publicInputs
      = Stmt.setupTablesBy TemplateQuery.Desc.parse Program.decode parsed tags circuitFile publicFile coins count tables
          draw partition program work stratified closure publicInputs :=
  rfl

namespace Registered

/-- `checkProgram`, the program read by `readDesc`. -/
def checkProgramBy (readDesc : ByteArray → V TemplateQuery.Desc)
    (program : Option ByteArray) (registered : Bool) : V Unit :=
  match program, registered with
  | none, _ => .ok ()
  | some _, true => .ok ()
  | some b, false =>
    match readDesc b with
    | .error e => .error e
    | .ok d =>
      match (programInputs d)[0]? with
      | none => .ok ()
      | some id =>
        .error s!"registered reads: the verifier's program reads a registered value ({id}), and the verifier holds no registered roots (--registered): its rows would be committed afresh, whatever the statement's public file names"

theorem checkProgram_eq_by (program : Option ByteArray) (registered : Bool) :
    checkProgram program registered = checkProgramBy TemplateQuery.Desc.parse program registered :=
  rfl

/-- `checkCover`, the program read by `readProg`. -/
def checkCoverBy (readProg : ByteArray → V Program.Prog)
    (c : Circuit) (program : Option ByteArray) (partition : Option Json) (own : Option (Array Port))
    (params : Array String) : V Unit :=
  match program, own with
  | some b, some _ =>
    match readProg b with
    | .error e => .error e
    | .ok P =>
      if !regRoots P then .ok ()
      else match templatePartition partition, HmRow.queryTemplate c with
        | false, _ => .error "registered reads (cover): the verifier's program reads a registered value, and its partition is not a template query: a registered value is read by a template query's instances, whose parameters the verifier holds to its registered reads"
        | true, none => .error "registered reads (cover): META program_digests does not name the circuit's one template, whose parameters the verifier holds to its registered reads"
        | true, some (t, _) =>
          match coverMiss P t params with
          | none => .ok ()
          | some (name, _) => .error s!"registered reads (cover): parameter {name} of {t} reads a registered input of the verifier's program, and no port of the verifier's registered reads reads it: its rows would be committed afresh"
  | _, _ => .ok ()

theorem checkCover_eq_by (c : Circuit) (program : Option ByteArray) (partition : Option Json) (own : Option (Array Port))
    (params : Array String) :
    checkCover c program partition own params = checkCoverBy Program.decode c program partition own params :=
  rfl

/-- `checkProgramBy` at the reading a `ProgramRead` keeps is `checkProgram`. -/
theorem checkProgramBy_read (read : ByteArray) (program : Option ByteArray) (registered : Bool) :
    checkProgramBy (ProgramRead.of read).descAt program registered = checkProgram program registered := by
  rw [ProgramRead.descAt_of, checkProgram_eq_by]

/-- `checkCoverBy` at the reading a `ProgramRead` keeps is `checkCover`. -/
theorem checkCoverBy_read (read : ByteArray) (c : Circuit) (program : Option ByteArray) (partition : Option Json)
    (own : Option (Array Port)) (params : Array String) :
    checkCoverBy (ProgramRead.of read).progAt c program partition own params = checkCover c program partition own params := by
  rw [ProgramRead.progAt_of, checkCover_eq_by]

end Registered

end Flock
