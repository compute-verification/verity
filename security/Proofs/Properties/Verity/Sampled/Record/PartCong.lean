module

public import Flock.HmRow
public import Flock.RecSetup

/-!
# The session's setup reads its partition object only through what the partition's readers make of it

`Stmt.setupTables`, `Stmt.setupH` and `Stmt.setupHidden` read the verifier's partition object `obj` in a few places
only: whether there is one, the binding check (`HmRow.checkPartition`), a stratified or a work sample's design
(`HmRow.stratifiedLaw`, `HmRow.workLaw`), the template test on its query, and the units derived from the program
(`HmRow.deriveTemplateUnits`, `HmRow.qwordFinding`). `PartView` holds those readers at one object (`view obj`).

* `setupHV`, `setupHiddenV`, `setupTablesV`: the three setups' text with each read of `obj` made through a `PartView`.
  They are this file's own and are named in no statement another file reads: each is the verifier's function by `rfl`
  (`setupH_view`, `setupHidden_view`, `setupTables_view`), so a change to the verifier breaks this file's build, and
  nothing else.
* `setupTables_congr`: two partitions with the same view set the session up alike.
* `checks_none`: `RecSetup.checks` without an inner shape reads no partition (`RecVStar.checkStmt` at `inner = none`
  reads only its statement's unit name).
-/

@[expose] public section

namespace Flock.PartCong

open Lean (Json)

/-- What the session's setup reads of a partition object: its binding check at a circuit, a stratified and a work
sample's design at a program, whether its query is a template query, and the units derived at a circuit and program. -/
structure PartView where
  check : Circuit → V HmRow.PartitionFinding
  strat : ByteArray → Nat → V (Draw.Law × Nat)
  work : ByteArray → Nat → Array (String × Nat × Nat) → V (Draw.Law × Nat)
  tmpl : Bool
  derive : Circuit → ByteArray → Array Nat → V String
  qword : Circuit → Public → ByteArray → HmRow.PartitionFinding → V (Option HmRow.PartitionFinding)

/-- The readers at the object `obj`, as the setups call them. -/
def view (obj : Json) : PartView where
  check c := HmRow.checkPartition c (some obj)
  strat := HmRow.stratifiedLaw obj
  work := HmRow.workLaw obj
  tmpl :=
    let q := (obj.getObjVal? "query").toOption.getD Json.null
    (TemplateQuery.templatesOf ((q.getObjValAs? String "name").toOption.getD "")
      ((q.getObjValAs? Nat "version").toOption.getD 0) Json.null).isSome
  derive c := HmRow.deriveTemplateUnits c obj
  qword c pub := HmRow.qwordFinding c pub obj

/-- The binding check at a view, `HmRow.checkPartition`'s without an object. -/
def checkV (c : Circuit) : Option PartView → V HmRow.PartitionFinding
  | none => HmRow.checkPartition c none
  | some v => v.check c

/-- `Stmt.setupH`'s text, reading the partition object through its view (`setupH_view`). -/
def setupHV (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (draw : Option Json) (pv : Option PartView) (program : Option ByteArray) : V Stmt := do
  let c ← if tags.typed then HmRow.parseTyped tags circuitFile tables else HmRow.parse tags circuitFile tables
  let finding0 ← if tags.bindings then (some <$> checkV c pv)
    else if pv.isSome then throw s!"{tags.name}: the statement binds no partition" else pure none
  let some merkle := MerkleScheme.find? c.leafScheme | throw s!"leaf-commitment scheme {c.leafScheme}: not one this verifier implements"
  let pop ← HmRow.loadPublic tags c publicFile
  let pub ← match draw with
    | some d => HmRow.drawn c pop d
    | none => pure pop
  -- a stratified sample's strata are the verifier's own, from its program and query, when it holds both
  if let some d := draw then
    if let (Except.ok (Draw.Law.stratified k strata), some v, some prog) :=
        ((Draw.UnitDraw.ofJson d).map (·.law), pv, program) then
      let (law, _) ← v.strat prog k
      if law != Draw.Law.stratified k strata then throw "U2: the draw's strata are not the verifier's own (its program and query)"
  HmRow.checkPublic c pub
  -- a template query's units, derived from the verifier's own program: the statement's units must be instances of its template
  let finding ← match finding0, pv, program with
    | some f, some v, some prog =>
      if v.tmpl then
        let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
        pure (some { f with derived := true, note := ← v.derive c prog indices })
      else v.qword c pub prog f
    | f, _, _ => pure f
  let n := pub.n
  let blocksReal := (n + c.g - 1) / c.g
  let nbl := max 3 (log2ceil (max blocksReal 1))
  let m := c.kLog + nbl
  if c.kLog > 27 || m > 35 then throw s!"m = {m}: no fast100 schedule (or k_log > 27)"
  let pin ← HmRow.pin c
  let (da, db) := HmRow.delta c pin
  let regions ← HmRow.regions c pub
  HmRow.checkRegionWords c da db regions
  checkInRange c.kLog regions.size (PT_LOCAL + nbl)
  checkRegionBytes nbl regions
  let identity := if J.has c.metaJ "row_schemas" then Tags.rowV2Identity tags.identity else tags.identity
  let identity := if c.ports.any (·.segs.isSome) then Tags.rowSegIdentity identity else identity
  let v2Run := (List.range c.outRowPorts.size).any fun j =>
    (c.ports.getD (c.ports.size - c.outRowPorts.size + j) default).segs.isNone && c.outRowPorts[j]! > 1
  let identity := if !J.has c.metaJ "out_row_ports" then identity
    else if v2Run then Tags.outRowsV2Identity identity else Tags.outRowsIdentity identity
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

/-- `Stmt.setupHidden`'s text, reading the partition object through its view (`setupHidden_view`). -/
def setupHiddenV (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (draw : Option Json) (pv : Option PartView) (program : Option ByteArray) (publicInputs : Option ByteArray) : V Stmt := do
  let c ← HmOut.parseCommit tags circuitFile tables
  let pi ← HmIn.parse c
  let finding0 ← if tags.bindings then (some <$> checkV c pv)
    else if pv.isSome then throw s!"{tags.name}: the statement binds no partition" else pure none
  let some merkle := MerkleScheme.find? c.leafScheme | throw s!"leaf-commitment scheme {c.leafScheme}: not one this verifier implements"
  let pop ← HmIn.loadPublic tags c pi publicFile
  HmIn.checkOwn pi pop publicInputs
  let pub ← match draw with
    | some d => HmIn.drawn c pi pop d
    | none => pure pop
  if let some d := draw then
    if let (Except.ok (Draw.Law.stratified k strata), some v, some prog) :=
        ((Draw.UnitDraw.ofJson d).map (·.law), pv, program) then
      let (law, _) ← v.strat prog k
      if law != Draw.Law.stratified k strata then throw "U2: the draw's strata are not the verifier's own (its program and query)"
  HmOut.checkPublic c pub
  let linked := J.has c.metaJ "links"
  let finding ← match finding0, pv, program with
    | some f, some v, some prog =>
      if v.tmpl then
        let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
        let note ← v.derive c prog indices
        let wires ← HmLinks.check c pop (HmLinks.readersOf draw) prog
        let linking := if wires == 0 && !linked then "; no cut wire between the statement's units"
          else s!"; Linking checked: {wires} cut wire(s) into the {pub.n} proved units of the population's {pop.n}, each reader's link row the writer's committed output row"
        pure (some { f with derived := true, note := note ++ linking })
      else if linked then throw "META links: Linking is checked under a template query (Q_template_instance v0, Q_template_instances v0) only"
      else v.qword c pub prog f
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

/-- `Stmt.setupTables`'s text, reading the partition object through its view (`setupTables_view`). -/
def setupTablesV (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : Nat)
    (tables : List (String × ByteArray)) (draw : Option Json) (pv : Option PartView) (program : Option ByteArray)
    (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) : V (SessionSpec × Array Stmt) := do
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
      let (some v, some prog, some K) := (pv, program, stratified)
        | throw "U2: a stratified draw needs the verifier's own K, program and partition (--stratified, --program, --partition)"
      if k != K then throw s!"U2: the draw's K is {k}, not the verifier's {K}"
      let (law, _) ← v.strat prog K
      if law != Draw.Law.stratified k strata then
        throw "U2: the draw's strata are not the verifier's own (its K, program and query)"
  -- a work sample's K and stated work set every k_s, so its sampling design must be the verifier's own derivation, never taken as stated
  if let some d := draw then
    if let .ok (Draw.Law.work k strata) := (Draw.UnitDraw.ofJson d).map (·.law) then
      let (some v, some prog, some (K, table)) := (pv, program, work)
        | throw "U2: a work draw needs the verifier's own K, work table, program and partition (--work, --work-table, --program, --partition)"
      if k != K then throw s!"U2: the draw's K is {k}, not the verifier's {K}"
      let (law, _) ← v.work prog K table
      if law != Draw.Law.work k strata then
        throw "U2: the draw's work law is not the verifier's own (its K, program, query and work table)"
  if publicInputs.isSome && !tags.hiddenOutputs then
    throw s!"--public-inputs: a {tags.name} statement has no public input ports (only a statement with hidden outputs has them)"
  let setup := fun (t : Tags) (d : Option Json) => do
    if tags.hiddenOutputs then
      let st ← setupHiddenV t circuitFile publicFile coins tables d pv program publicInputs
      if HmRow.pinnedCount st.c.metaJ != 0 then throw s!"{tags.name}: refused: {HmRow.pinnedWhy}"
      pure st
    else
      let st ← setupHV t circuitFile publicFile coins tables d pv program
      if J.has st.c.metaJ "public_inputs" || J.has st.c.metaJ "shared_reads" then
        throw s!"META: public_inputs, shared_reads: a {tags.name} statement has no public input ports or shared reads (only a statement with hidden outputs has them)"
      pure st
  if count ≤ 1 then
    let st ← setup tags draw
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

set_option smartUnfolding false in
theorem setupH_view (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (tables : List (String × ByteArray)) (draw partition : Option Json) (program : Option ByteArray) :
    Stmt.setupH tags circuitFile publicFile coins tables draw partition program =
      setupHV tags circuitFile publicFile coins tables draw (partition.map view) program := by
  cases partition <;> rfl

set_option smartUnfolding false in
theorem setupHidden_view (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (tables : List (String × ByteArray)) (draw partition : Option Json) (program publicInputs : Option ByteArray) :
    Stmt.setupHidden tags circuitFile publicFile coins tables draw partition program publicInputs =
      setupHiddenV tags circuitFile publicFile coins tables draw (partition.map view) program publicInputs := by
  cases partition <;> rfl

set_option smartUnfolding false in
theorem setupTables_view (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : Nat)
    (tables : List (String × ByteArray)) (draw partition : Option Json) (program : Option ByteArray)
    (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) :
    Stmt.setupTables tags circuitFile publicFile coins count tables draw partition program work stratified closure
        publicInputs =
      setupTablesV tags circuitFile publicFile coins count tables draw (partition.map view) program work stratified
        closure publicInputs := by
  cases partition <;> rfl

/-- **Two partitions with the same view set the session up alike.** -/
theorem setupTables_congr (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : Nat)
    (tables : List (String × ByteArray)) (draw : Option Json) {pa pb : Option Json} (h : pa.map view = pb.map view)
    (program : Option ByteArray) (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) :
    Stmt.setupTables tags circuitFile publicFile coins count tables draw pa program work stratified closure
        publicInputs =
      Stmt.setupTables tags circuitFile publicFile coins count tables draw pb program work stratified closure
        publicInputs := by
  rw [setupTables_view, setupTables_view, h]

/-- **Without an inner shape, `RecSetup.checks` reads no partition.** -/
theorem checks_none (tags : Tags) (tables : List (String × ByteArray)) (sts : Array Stmt) (circuitB : ByteArray)
    (pa pb : Option Json) (program : Option ByteArray) (registered : Option (Array Registered.Port)) :
    RecSetup.checks none none tags tables sts circuitB pa program registered none none none =
      RecSetup.checks none none tags tables sts circuitB pb program registered none none none := by
  have h : ∀ st, RecVStar.checkStmt none tags tables st circuitB pa program registered none none none =
      RecVStar.checkStmt none tags tables st circuitB pb program registered none none none := fun _ => rfl
  simp only [RecSetup.checks, h]

end Flock.PartCong
