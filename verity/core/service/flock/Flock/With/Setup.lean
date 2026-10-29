/-
§16.10–§16.13's setup over the hash oracle (`Flock.With.Hashes`): `Stmt.setupTablesWith` is `Stmt.setupTables` with every
digest of what the session brings taken from the oracle, in a verdict monad `M`. Run in `V` with `Hashes.ofSha512 Sha512.hash`
it is `Stmt.setupTables` (`Flock.With.SetupEq`). The copies keep the originals' text; what hashes nothing is the compiled
verifier's own, lifted.

Through the oracle: the circuit file's digest and its class, the type and layout content digests META names, the partition
digest, the public file's digest, the program's digest (`Program.decodeWith`, beside `Program.decode` for its decoder state),
the frame-v3-sha512 roots of the public `b ‖ c` and outputs with their hm96 tree leaves, each table's statement digest and Σ,
and the session's Σ.

Real SHA-512, as in `Flock.With.Hashes`' hm96 key: the checks against constants the statement pins, whose hash input is
itself a constant or a table the verifier holds and checks only against its pinned digest. They are the MUFU and library
table pins (`Pinned.tableSha`), `row_key_sha512` (`Pinned.rowKey`), the zero row's and the padding instance's pinned dummy
`b ‖ c` (`HmRow.zeroRowBc`, `HmOut.rowBcZeroSalt`, functions of a port's shape), the rows' prefix midstates and the
`Sha512.compress` constants of the nets the verifier generates (`HmRow.midstateOf`, `HmRow.segMidstate`, `HmNets`,
`RecOpen`), and the Merkle registry's records (`MerkleScheme.find?`), whose closures a verdict never calls (`verifyWith` hashes
a registered scheme through the oracle, `MerkleScheme.opsWith`). No digest of these depends on the session: the flock lock's
`verifier.stop` names each.
-/
import Flock.With.Hashes
import Flock.HmRow

namespace Flock

open Lean (Json)

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

namespace Pinned

/-- A held table's name: its words' SHA-512 as lowercase hex, which a table pin (MUFU, library) compares with the digest
the statement pins. -/
def tableSha (bytes : ByteArray) : String := hexOf (Sha512.hash bytes)

/-- `row_key_sha512`: the SHA-512 of `hm96-sha512/v1`'s default key, as lowercase hex. -/
def rowKey : String := hexOf (Sha512.hash Hm96.Default512.key)

end Pinned

/-- `Hm96.Default512.treeLeaf` through the oracle. -/
def Hm96.Default512.treeLeafWith (hs : Hashes M) (bc : ByteArray) : M ByteArray :=
  hs.sha512 (Hm96.Default512.leafPrefix ++ bc)

namespace FV3

/-- `FV3.hash` with the oracle's SHA-512. -/
def hashWith (hs : Hashes M) (tag : String) (parts : List ByteArray) : M ByteArray :=
  let t := tag.toUTF8
  hs.sha512 (parts.foldl (fun acc p => acc ++ be64 p.size.toUInt64 ++ p) (FRAME ++ be32 t.size.toUInt32 ++ t))

/-- `FV3.leaf` with the oracle's SHA-512. -/
def leafWith (hs : Hashes M) (dom : ByteArray) (i : Nat) (schema : String) (payload : ByteArray) : M ByteArray :=
  hashWith hs "leaf" [dom, uint i, uint i, schema.toUTF8, payload]

/-- `FV3.root` with the oracle's SHA-512. -/
def rootWith (hs : Hashes M) (dom : ByteArray) (leaves : Array ByteArray) : M ByteArray := do
  let n := leaves.size
  let width := nextPow2 n
  let mut level := leaves ++ (← (Array.range (width - n)).mapM fun k => hashWith hs "pad" [dom, uint (n + k)])
  for depth in [0:log2ceil width] do
    level ← (Array.range (level.size / 2)).mapM fun i =>
      hashWith hs "node" [dom, uint depth, uint i, level[2 * i]!, level[2 * i + 1]!]
  return level.getD 0 ByteArray.empty

end FV3

/-- `CircuitType.digest` through the oracle. -/
def CircuitType.digestWith (hs : Hashes M) (t : CircuitType) : M String := do
  let s ← canon t.toJson
  return hexOf (← hs.sha512 (CircuitType.FORMAT.toUTF8 ++ ByteArray.mk #[0] ++ s.toUTF8))

/-- `Layout.digest` through the oracle. -/
def Layout.digestWith (hs : Hashes M) (l : Layout) : M String := do
  let s ← canon l.record
  return hexOf (← hs.sha512 (Layout.FORMAT.toUTF8 ++ ByteArray.mk #[0] ++ s.toUTF8))

namespace Typed

private def ctx : String := "typed statement META"

/-- `hold`, its pins by `Pinned.tableSha`. -/
def holdWith (types : Array (String × CircuitType)) (tables : List (String × ByteArray)) : V (Array Held) := do
  let mut out : Array Held := #[]
  for (_, t) in types do
    for (sha, _, _) in t.tables do
      if out.any (·.sha512 == sha) then continue
      let some lt := libraryTable? sha
        | throw s!"META: table {sha.take 16}…: not a library table, the only tables this verifier holds"
      let some (_, bytes) := tables.find? (·.1 == lt.name) | throw s!"table {lt.name}: not given (the verifier's copy, --tables)"
      if bytes.size != 4 * 2 ^ lt.indexBits || Pinned.tableSha bytes != sha then
        throw s!"table {lt.name}: its words are not the table its SHA-512 names"
      let words := (Array.range (bytes.size / 4)).map fun i => rd32 bytes (4 * i)
      out := out.push ⟨sha, bytes, words, Layout.tableInfo words⟩
  return out

/-- `classOf` through the oracle. -/
def classOfWith (hs : Hashes M) (mj : Json) (tables : List (String × ByteArray)) : M (Class × CircuitType) := do
  let kvs ← match mj.getObjVal? "types" with
    | .ok (Json.obj kvs) => pure kvs.toArray
    | _ => throw "META: types is not an object"
  let mut types : Array (String × CircuitType) := #[]
  for (d, c) in kvs do
    let t ← CircuitType.ofJson c
    if (← t.digestWith hs) != d then throw s!"META: type {d.take 16}… is not its content's digest"
    types := types.push (d, t)
  for (d, t) in types do
    if let .error e := CircuitType.check (typeIn types) t then throw s!"META: type {d.take 16}… breaks the rule '{e}'"
  let held ← holdWith types tables
  let ls ← (← J.arr mj "layouts" ctx).mapM fun r => do
    let l ← Layout.ofJson r
    if (← J.str r "sha512" ctx) != (← l.digestWith hs) then throw s!"{ctx}: a layout's sha512 is not its record's digest"
    pure l
  let unit ← J.nat mj "unit_layout" ctx
  if let .error e := Layout.check (typeIn types) ls unit (infoIn held) then throw s!"META: the layouts break the rule '{e}'"
  let some ul := ls[unit]? | throw "META: unit_layout names no layout"
  let some ut := ul.type | throw "META: the unit's layout names no type"
  if (← J.str mj "unit_type" ctx) != ut then throw "META: unit_type is not the unit's layout's type"
  if ul.own != .derive then throw "META: the unit's layout is not derived"
  if (← J.str mj "policy" ctx) != POLICY then throw s!"META: policy is not {POLICY}"
  if !(← J.arr mj "unit_in" ctx).isEmpty then
    throw "META: unit_in binds the unit's inputs to constants, which this verifier does not derive yet"
  let some t := typeIn types ut | throw "META: unit_type names no type"
  match h : DeriveAll.deriveChecked (typeIn types) ls (infoIn held) (wordsIn held) unit with
  | .ok done => return (⟨types, ls, held, unit, done, h⟩, t)
  | .error e => throw s!"META: the unit's rows: {e}"

/-- `templateOf` through the oracle. -/
def templateOfWith (hs : Hashes M) (name : String) (done : Array DeriveAll.D) (ls : Array Layout) (unit : Nat)
    (t : CircuitType) (held : Array Held) : M Template := do
  let u := done.getD unit default
  if u.reads.any (·.prod0 < u.size) then
    throw "META: the instance type reads a table in its own region, which this verifier's template reading does not fold yet"
  if u.parts.any (·.1 ≥ ls.size) then throw "META: a part places a layout the file does not have"
  let root ← netOfD (rootOf u) name
  let digests ← ls.toList.mapM (·.digestWith hs)
  let placed := (List.range ls.size).filter fun j => u.parts.any (·.1 == j)
  let layouts ← (placed.mapM (placedEntry digests ls done held) : V _)
  return { root, layouts := layouts.toArray, parts := u.parts.map fun (j, b) => (placed.idxOf j, b), own := u.size,
           ins := t.ins, outs := t.outs, inCols := u.inCols, bind := u.delta.filterMap fun (r, src) => src.map (r, ·.toArray),
           cross := crossOf u }

/-- `read` through the oracle. -/
def readWith (hs : Hashes M) (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) :
    M (Class × (Net ⊕ Template)) := do
  let (_, name, mj) ← split tags bytes
  let (k, t) ← classOfWith hs mj tables
  let u := k.done.getD k.unit default
  if isFlat u then return (k, .inl (← flatNet k.done u name))
  return (k, .inr (← templateOfWith hs name k.done k.ls k.unit t k.held))

end Typed

/-- `TemplateQuery.Desc.parse` through the oracle. -/
def TemplateQuery.Desc.parseWith (hs : Hashes M) (bytes : ByteArray) : M TemplateQuery.Desc := do
  let some text := String.fromUTF8? bytes | throw "program: not UTF-8"
  let j ← match Json.parse text with
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
      if form == "primitive" then pure (id, ({ prim := true, nodes := #[] } : TemplateQuery.DDef))
      else if form == "composite" then
        let nodes ← match (dj.getObjVal? "body").toOption.bind (fun b => (b.getObjVal? "nodes").toOption) with
          | some (.arr ns) => ns.mapM fun nj => do
            let form := (nj.getObjValAs? String "form").toOption.getD ""
            let some fn := (nj.getObjValAs? Nat "fn").toOption | throw s!"program: {id}: a node without a callee"
            if fn ≥ callees.size then throw s!"program: {id}: a node's callee is outside callees"
            pure ({ form, fn, n := (nj.getObjValAs? Nat "n").toOption.getD 1 } : TemplateQuery.DNode)
          | _ => throw s!"program: {id}: a composite without body nodes"
        pure (id, ({ prim := false, nodes } : TemplateQuery.DDef))
      else throw s!"program: {id}: an unknown definition form {form}"
    | _ => throw "program: definitions"
  let some root := (j.getObjValAs? String "root").toOption | throw "program: root"
  if (defs.find? (·.1 == root)).isNone then throw "program: the root is not a definition"
  return { callees, defs, root, sha512 := ← hs.sha512 (← canon bare).toUTF8 }

namespace HmRow

private def ctx := "circuit META"

private def intLists (j : Json) (k : String) : V (Array (Array Int)) := do
  (← J.arr j k ctx).mapM fun u => do (← J.asArr u s!"{ctx}.{k}").mapM fun x => J.asInt x s!"{ctx}.{k}"

/-- `classOf` through the oracle. -/
def classOfWith (hs : Hashes M) (bytes : ByteArray) (head : String) (mj : Json) : M ByteArray := do
  let some n1 := findByte bytes 10 | throw "circuit: no META line"
  let some n2 := findByte bytes 10 (n1 + 1) | throw "circuit: no section after META"
  let m := match mj with
    | .obj kvs => Json.mkObj (kvs.toArray.filter (fun kv => !BINDING_KEYS.contains kv.1)).toList
    | j => j
  hs.sha512 (head.toUTF8 ++ "\nMETA ".toUTF8 ++ (← canon m).toUTF8 ++ "\n".toUTF8 ++ bytes.extract (n2 + 1) bytes.size)

/-- `parse`'s sections after META from the state `s`, the nets so far and the line it reads next: each block line's net. -/
def sectionsLoop (tags : Tags) (lines : Array String) (s : Array (String × Net) × Nat) : V (Array (String × Net) × Nat) :=
  forIn Lean.Loop.mk s fun _ __s => do
    let nets := __s.1
    let at_ := __s.2
    if at_ < lines.size then
      let h := tokens lines[at_]!
      if h.size != 3 || h[0]! != tags.blockKeyword then throw s!"circuit: expected a {tags.blockKeyword} line"
      let some n := h[2]!.toNat? | throw s!"circuit: {tags.blockKeyword} line count"
      if at_ + 1 + n > lines.size then throw s!"circuit: {h[1]!} is truncated"
      let net ← match Net.parse (lines.extract (at_ + 1) (at_ + 1 + n)) with
        | .ok v => pure v
        | .error e => throw s!"{h[1]!}: {e}"
      return .yield (nets.push (h[1]!, net), at_ + 1 + n)
    else return .done (nets, at_)

/-- `parse` after the class digest, which takes nothing from the oracle: the sections after META, the lookups, rows,
ranges, wires and leaf scheme, from the circuit's digest `sha`, its lines, its name, META and its class digest. -/
def parseRest (tags : Tags) (tables : List (String × ByteArray)) (tmpl : Option Typed.Template) (pre : Array (String × Net))
    (sha : ByteArray) (lines : Array String) (name : String) (mj : Json) (classSha : ByteArray) : V Circuit := do
  let __s ← sectionsLoop tags lines ((match tmpl with
    | some t => #[("root", t.root)] ++ t.layouts.map fun (d, n, _) => (d, n)
    | none => pre), 2)
  let mut nets := __s.1
  for k in Circuit.FORBIDDEN do
    if J.has mj k then throw s!"META: {k} (a verifier-evaluated part of the computation); the verifier only checks proofs"
  if tmpl.isSome then
    for k in TEMPLATE_DERIVED do
      if J.has mj k then throw s!"META: {k}: a typed template's block wiring is derived from its types, never stated"
  let lks ← match mj.getObjVal? "lookups" with
    | .ok (Json.obj kvs) => pure (kvs.toArray.qsort (fun a b => a.1 < b.1))
    | .ok Json.null => pure #[]
    | .ok _ => throw "META: lookups is not an object"
    | .error _ => pure #[]
  for (lname, spec) in lks do
    let some pinned := MUFU_TABLES.find? (·.name == lname) | throw "META: a lookup of a table the statement does not pin"
    let size := pinned.words
    if (← J.str spec "sha256" ctx) != pinned.sha256 || (← J.str spec "table" ctx) != lname || (← J.nat spec "out_bits" ctx) != 31 then
      throw s!"META: lookup {lname} is not the pinned table"
    let n ← J.nat spec "in_bits" ctx
    let lo ← J.nat spec "lo_bits" ctx
    if n != Nat.log2 size || 2 ^ n != size || !(2 ≤ lo && lo < n) then throw s!"META: lookup {lname}'s index is not its table's"
    let some (_, words) := tables.find? (·.1 == lname) | throw s!"the circuit has a lookup of {lname}: its table is not given"
    if words.size != 4 * size || Pinned.tableSha words != pinned.sha512 then throw s!"MUFU table {lname}: not the pinned table"
    nets := nets.push (s!"lookup:{lname}", ← Lookup.build s!"lookup:{lname}" words n lo 31)
  let ports ← rowPorts mj tags.typed
  if (← J.str mj "row_key_sha512" ctx) != Pinned.rowKey then
    throw "META: the rows' hm96 key is not hm96-sha512/v1's default key"
  let g ← J.nat mj "vus_per_block" ctx
  -- a typed template's unit is its instance's root, one per VU
  let upv ← if tmpl.isSome then pure 1 else J.nat mj "units_per_vu" ctx
  let some unitNet := nets.findIdx? (·.1 == if tmpl.isSome then "root" else "unit") | throw "circuit: no unit circuit"
  let netOf := fun (nm : String) => match nets.findIdx? (·.1 == nm) with
    | some i => (pure i : V Nat)
    | none => throw s!"META: unknown circuit {nm}"
  let ranges ← (← J.arr mj "ranges" ctx).mapM fun r => do
    let kind ← match ← J.str r "type" ctx with
      | "mask" => pure Kind.mask
      | "unit" => pure (Kind.net (← netOf (← J.str r "net" ctx)))
      | _ => throw "a range of an unknown type"
    let packed := match r.getObjVal? "packed" with
      | .ok (Json.bool b) => b
      | _ => false
    pure ({ kind, slotLog := ← J.nat r "slot_log" ctx, pos0 := ← J.nat r "pos0" ctx, count := ← J.nat r "count" ctx,
            role := .none, packed } : Range)
  let leavesIn ← if tmpl.isSome then pure #[#[]] else intLists mj "leaves_in"
  let leavesOut ← if tmpl.isSome then pure #[#[]] else (← intLists mj "leaves_out").mapM fun u => u.mapM fun x =>
    if x < 0 then (throw "META: a negative output leaf" : V Nat) else pure x.toNat
  let dummy ← J.get mj "dummy" ctx
  let dummyDigests ← (← J.arr dummy "bc" ctx).mapM fun d => do
    let b ← unhex (← J.asStr d "dummy bc")
    if b.size != BC_BYTES then throw "b || c length"
    pure b
  let dummyOutputs ← (← J.arr dummy "outputs" ctx).mapM fun x => do
    let v ← J.asNat x "dummy output"
    if v ≥ 65536 then throw "META: a dummy output is not a u16"
    pure v
  let outPorts ← (← J.arr mj "out_ports" ctx).mapM fun p => do
    let a ← J.asArr p "out port"
    if a.size != 2 then throw "META: an out port is not [name, words]"
    pure (← J.asStr a[0]! "out port name", ← J.asNat a[1]! "out port words")
  let outRowPorts ← match mj.getObjVal? "out_row_ports" with
    | .error _ => pure (Array.replicate outPorts.size 1)
    | .ok v => do
      let bad := "META: out_row_ports is not each output row's run of output ports, in order, covering every output port once"
      let r ← (← J.asArr v "out_row_ports").mapM fun x => do
        let k ← J.asNat x "out_row_ports"
        if k < 1 || k > outPorts.size then throw bad
        pure k
      if r.foldl (· + ·) 0 != outPorts.size then throw bad
      pure r
  let mut perVu := (Array.replicate nets.size 0).set! unitNet upv
  match mj.getObjVal? "per_vu" with
  | .ok (Json.obj kvs) =>
    for (k, v) in kvs.toArray do
      perVu := perVu.set! (← netOf k) (← J.asNat v "per_vu")
  | _ => pure ()
  let shaNet ← netOf SHA_NET
  let hmNet ← netOf HM_NET
  let outNet ← if tmpl.isSome then pure unitNet else netOf ((mj.getObjValAs? String "out_net").toOption.getD "unit")
  let onp ← match mj.getObjVal? "out_net_ports" with
    | .ok (.arr a) => outNetEntries a
    | _ => pure (#[], none)
  let leafCuts ← match mj.getObjVal? "leaf_cuts" with
    | .ok (.arr a) => a.mapM fun w => do
      let x ← J.asArr w "a leaf cut"
      if x.size != 4 then throw "META: a leaf cut is not [circuit, unit, port, leaf]"
      if (x[0]!.getStr?).toOption != some "unit" then throw "META: a leaf cut into anything but a unit"
      pure (← J.asNat x[1]! "a leaf cut", ← J.asNat x[2]! "a leaf cut", ← J.asNat x[3]! "a leaf cut")
    | _ => pure #[]
  let wires ← match mj.getObjVal? "wires" with
    | .ok w => (← J.asArr w "wires").mapM fun x => do
      let a ← J.asArr x "wire"
      if a.size != 6 then throw "META: a wire is not six fields"
      pure (⟨(← netOf (← J.asStr a[0]! "wire"), ← J.asNat a[1]! "wire", ← J.asNat a[2]! "wire"),
              (← netOf (← J.asStr a[3]! "wire"), ← J.asNat a[4]! "wire", ← J.asNat a[5]! "wire")⟩ : Wire)
    | .error _ => pure #[]
  -- the rows' wires come first, exactly as the ports imply; a typed template's are just those
  let derived := rowWires ports shaNet hmNet
  let wires := if tmpl.isSome then derived else wires
  let rowNet := fun (n : Nat) => n == shaNet || n == hmNet
  if wires.size < derived.size || (List.range derived.size).any (fun i => wires[i]!.dst != derived[i]!.dst || wires[i]!.src != derived[i]!.src)
      || (wires.extract derived.size wires.size).any (fun w => rowNet w.dst.1 || rowNet w.src.1) then
    throw "META: the rows' wires are not the chains and hm96 inputs the ports imply"
  let leafScheme ← J.get mj "leaf_scheme" ctx
  let leafId ← J.str leafScheme "id" ctx
  if leafId != tags.merkleLeaf then throw s!"leaf-commitment scheme {leafId} is not one this verifier implements"
  let c : Circuit := { name, sha, nets, ranges, ports, outPorts, outRowPorts, g, kLog := ← kLogOf mj, upv, unitNet,
                       leavesIn, leavesOut, key := ByteArray.empty, dummyDigests, dummyOutputs, leafScheme := leafId, perVu, wires,
                       shaNet, hmNet, classSha, metaJ := mj, outNet, outNetPorts := onp.1, outNetSlots := onp.2, leafCuts,
                       typed := ← tmpl.mapM fun t => pure (blockOf t) }
  check c tags leafScheme
  HmNets.check c
  RecOpen.check c
  RecResiduals.check c
  RecSparse.check c
  RecOpening.check c
  return c

/-- `parse` through the oracle. -/
def parseWith (hs : Hashes M) (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := [])
    (tmpl : Option Typed.Template := none) (pre : Array (String × Net) := #[]) : M Circuit := do
  let sha ← hs.sha512 bytes
  let text ← match String.fromUTF8? bytes with
    | some s => pure s
    | none => throw "circuit: not UTF-8"
  let lines0 := (text.splitOn "\n").toArray
  let lines := if lines0.back? == some "" then lines0.pop else lines0
  let head := (lines.getD 0 "").trimAsciiEnd.toString
  if !head.startsWith (tags.fileFormat ++ " ") then throw s!"circuit: not a {tags.fileFormat} file"
  let name := (head.drop (tags.fileFormat.length + 1)).toString
  let metaLine := (lines.getD 1 "").trimAsciiEnd.toString
  if !metaLine.startsWith "META " then throw "circuit: no META line"
  let mj ← match Json.parse (metaLine.drop 5).toString with
    | .ok j => pure j
    | .error e => throw s!"META: {e}"
  if (← J.str mj "format" ctx) != tags.fileFormat || (← J.str mj "statement" ctx) != tags.statement then
    throw s!"META: not {tags.statement}"
  let classSha ← if tags.bindings then do
      if !isSha512Hex ((mj.getObjValAs? String "program_sha512").toOption.getD "") then
        throw "META: program_sha512 (the Program the partition is over) is not a SHA-512 digest"
      let part := (mj.getObjVal? "partition").toOption.getD Json.null
      let rule := (part.getObjValAs? String "rule").toOption.getD ""
      if !(rule == PARTITION_V1 || rule == PARTITION_PROVISIONAL) || !isSha512Hex ((part.getObjValAs? String "digest").toOption.getD "") then
        throw s!"META: the partition is not a {PARTITION_V1} (or the provisional {PARTITION_PROVISIONAL}) SHA-512 digest"
      classOfWith hs bytes head mj
    else pure ByteArray.empty
  parseRest tags tables tmpl pre sha lines name mj classSha

/-- `parseTyped` through the oracle. -/
def parseTypedWith (hs : Hashes M) (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) :
    M Circuit := do
  let (k, r) ← Typed.readWith hs tags bytes tables
  let c ← match r with
    | .inr t => parseWith hs tags bytes tables (tmpl := some t)
    | .inl unit => parseWith hs tags bytes tables (pre := #[("unit", unit)])
  return { c with cls := some k }

/-- `partitionDigest` through the oracle. -/
def partitionDigestWith (hs : Hashes M) (obj : Json) : M ByteArray := do
  hs.sha512 (PARTITION_V1.toUTF8.push 0 ++ (← canon obj).toUTF8)

/-- `deriveTemplateUnits` through the oracle. -/
def deriveTemplateUnitsWith (hs : Hashes M) (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    M String := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let name := (q.getObjValAs? String "name").toOption.getD ""
  let version := (q.getObjValAs? Nat "version").toOption.getD 0
  let some ts := TemplateQuery.templatesOf name version ((q.getObjVal? "params").toOption.getD Json.null)
    | throw s!"partition: {name} v{version} is not a template query"
  let ts ← ts
  let d ← TemplateQuery.Desc.parseWith hs program
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

/-- `deriveQwordUnits` through the oracle. -/
def deriveQwordUnitsWith (hs : Hashes M) (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    M String := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let version := (q.getObjValAs? Nat "version").toOption.getD 0
  let ps := (q.getObjVal? "params").toOption.getD Json.null
  let X := (ps.getObjValAs? Nat "X").toOption.getD 0
  let W := (ps.getObjValAs? Nat "W").toOption.getD 0
  let P ← Program.decodeWith hs.sha512 program
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

/-- `deriveQcallUnits` through the oracle. -/
def deriveQcallUnitsWith (hs : Hashes M) (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    M String := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let X := ((q.getObjVal? "params").toOption.getD Json.null |>.getObjValAs? Nat "X").toOption.getD 0
  let P ← Program.decodeWith hs.sha512 program
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

/-- `qwordFinding` through the oracle. -/
def qwordFindingWith (hs : Hashes M) (c : Circuit) (pub : Public) (obj : Json) (program : ByteArray)
    (f : PartitionFinding) : M (Option PartitionFinding) := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  if EVALUABLE.contains ((q.getObjValAs? String "name").toOption.getD "", (q.getObjValAs? Nat "version").toOption.getD 0) then
    let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
    pure (some { f with derived := true, note := ← deriveQwordUnitsWith hs c obj program indices })
  else if ((q.getObjValAs? String "name").toOption.getD "", (q.getObjValAs? Nat "version").toOption.getD 0) == Q_CALL then
    let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
    pure (some { f with derived := true, note := ← deriveQcallUnitsWith hs c obj program indices })
  else pure (some f)

/-- `checkPartition` through the oracle. -/
def checkPartitionWith (hs : Hashes M) (c : Circuit) (obj : Option Json) : M PartitionFinding := do
  let part := (c.metaJ.getObjVal? "partition").toOption.getD Json.null
  let rule := (part.getObjValAs? String "rule").toOption.getD ""
  let digest := (part.getObjValAs? String "digest").toOption.getD ""
  match rule, obj with
  | "verity/partition/v1", none => throw "partition: the circuit binds a verity/partition/v1 digest; the verifier's partition object is required"
  | "verity/partition/v1", some o =>
    let (name, version) ← checkPartitionObject o
    if hexOf (← partitionDigestWith hs o) != digest then
      throw "partition: the circuit's partition digest is not the verifier's partition object's"
    if (o.getObjValAs? String "program").toOption != (c.metaJ.getObjValAs? String "program_sha512").toOption then
      throw "partition: the partition object is over another program than META's program_sha512"
    let note := if (TemplateQuery.templatesOf name version Json.null).isSome then
        s!"units taken as stated, not derived: the verifier's copy of the program was not given (--program) to evaluate {name} v{version}"
      else if EVALUABLE.contains (name, version) || (name, version) == Q_CALL then
        s!"units taken as stated, not derived: the verifier's copy of the program was not given (--program) to evaluate {name} v{version}"
      else s!"units taken as stated, not derived: the verifier does not evaluate {name} v{version}"
    return { rule, query := s!"{name} v{version}", digest, derived := false, note }
  | _, some _ => throw "partition: the circuit binds the provisional unit cover, not a verity/partition/v1 object"
  | _, none => return { rule, digest, derived := false, note := "units taken as stated, not derived: the provisional unit cover names no query" }

/-- `publicDigest` through the oracle. -/
def publicDigestWith (hs : Hashes M) (header : Json) (pubs : ByteArray) : M ByteArray := do
  hs.sha512 ((← canon header).toUTF8 ++ pubs)

/-- `loadPublic` through the oracle. -/
def loadPublicWith (hs : Hashes M) (tags : Tags) (c : Circuit) (file : ByteArray) : M Public := do
  let some nl := findByte file 10 | throw "public file: no header line"
  let text ← match String.fromUTF8? (file.extract 0 nl) with
    | some s => pure s
    | none => throw "public file: header is not UTF-8"
  let header ← match Json.parse text with
    | .ok j => pure (Flock.norm j)
    | .error e => throw s!"public file: header: {e}"
  let pctx := "public file"
  if (← J.str header "format" pctx) != tags.publicFormat || (← J.str header "statement" pctx) != tags.statement then
    throw s!"public file: not a {tags.publicFormat} file"
  if (← J.str header tags.publicCircuitKey pctx) != hexOf c.sha then throw "public file: staged for another circuit"
  let n ← J.nat header "instances" pctx
  if (← J.bool header "rows" pctx) then throw "public file: it holds rows (the verifier's public file has none)"
  let np := c.ports.size
  let nOut := c.outWords
  let counts ← sharedRows tags c header
  let nBcs := (counts.map (·.foldl (· + ·) 0)).getD (n * np)
  let refsLen := if counts.isSome then 4 * n * np else 0
  let body := file.extract (nl + 1) file.size
  if body.size != BC_BYTES * nBcs + refsLen + 2 * n * nOut then throw "public file: length"
  let bcs := (Array.range nBcs).map fun i => body.extract (BC_BYTES * i) (BC_BYTES * i + BC_BYTES)
  let o := BC_BYTES * nBcs + refsLen
  let outputs := (Array.range (n * nOut)).map fun i => (body.get! (o + 2 * i)).toNat + 256 * (body.get! (o + 2 * i + 1)).toNat
  let tables ← match counts with
    | none => pure none
    | some counts => do
      let refs := (Array.range (n * np)).map fun k => (u32OfLE body (BC_BYTES * nBcs + 4 * k)).toNat
      if (Array.range refs.size).any (fun k => refs[k]! ≥ counts[k % np]!) then
        throw "shared rows: an instance's ref is past its table"
      let starts := (counts.foldl (fun (acc : Array Nat × Nat) r => (acc.1.push acc.2, acc.2 + r)) (#[], 0)).1
      pure (some ({ counts, starts, refs } : Tables))
  let units ← J.get header "units" pctx
  let indices ← (← J.arr units "indices" pctx).mapM (J.asNat · "a unit index")
  if !c.classSha.isEmpty then
    let classes ← (← J.arr units "classes" pctx).mapM (J.asStr · "a class")
    if classes.size != n || classes.any (· != hexOf c.classSha) then
      throw s!"header: an instance's class is not this circuit's class {(hexOf c.classSha).take 16}…"
  if indices.size != n || (List.range indices.size).any (fun i => i > 0 && indices[i - 1]! ≥ indices[i]!)
      || (← canon units) != (← canon (unitsEntry c indices)) then
    throw "header: the instances' unit indices are not ascending, one per instance, blocks of vus_per_block"
  if J.has header "unit_draw" then throw "header: a staged file is a registered population; a draw is made in the session"
  return { header, n, digests := bcs, outputs, sha := ← publicDigestWith hs header body, tables }

/-- `drawn` through the oracle. -/
def drawnWith (hs : Hashes M) (c : Circuit) (pub : Public) (draw : Json) : M Public := do
  let d ← Draw.UnitDraw.ofJson draw
  let proved := d.proved
  if d.population != pub.n || pub.population.isSome then throw "U: the draw's population is not this registered population"
  let np := c.ports.size
  let nOut := c.outWords
  let pctx := "public file"
  let popIdx ← (← J.arr (← J.get pub.header "units" pctx) "indices" pctx).mapM (J.asNat · "a unit index")
  let indices := proved.map fun u => popIdx.getD u 0
  let header := ((pub.header.setObjVal! "unit_draw" draw).setObjVal! "instances" (Lean.toJson proved.size)
    |>.setObjVal! "population" (Lean.toJson pub.n)).setObjVal! "units" (unitsEntry c indices)
  let mut pubs := pub.digests.foldl (· ++ ·) ByteArray.empty
  if let some t := pub.tables then
    for r in t.refs do pubs := pubs ++ le32 r.toUInt32
  for w in pub.outputs do pubs := pubs.push (w % 256).toUInt8 |>.push (w / 256).toUInt8
  return { header, n := proved.size,
           digests := if pub.tables.isSome then pub.digests
             else proved.foldl (fun acc u => acc ++ pub.digests.extract (u * np) (u * np + np)) #[],
           outputs := proved.foldl (fun acc u => acc ++ pub.outputs.extract (u * nOut) (u * nOut + nOut)) #[],
           sha := ← publicDigestWith hs header pubs, population := some (pub.n, pub.digests, pub.outputs),
           tables := pub.tables.map fun t => { t with refs := proved.foldl (fun acc u => acc ++ t.refs.extract (u * np) (u * np + np)) #[] } }

/-- `stratifiedLaw`'s and `workLaw`'s strata through the oracle. -/
def templateStrataWith (hs : Hashes M) (law : String) (obj : Json) (program : ByteArray) :
    M (Array (String × Array (Nat × Nat)) × Nat) := do
  let (name, version) ← checkPartitionObject obj
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let some ts := TemplateQuery.templatesOf name version ((q.getObjVal? "params").toOption.getD Json.null)
    | throw s!"{law}: {name} v{version} defines no strata (Q_template_instance v0 and Q_template_instances v0 do)"
  let ts ← ts
  let d ← TemplateQuery.Desc.parseWith hs program
  if (obj.getObjValAs? String "program").toOption != some (hexOf d.sha512) then
    throw "partition: the verifier's program is not the one the partition object names"
  let rs ← match TemplateQuery.evaluate d ts with
    | .ok rs => pure rs
    | .error e => throw s!"partition: {e}"
  return (Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)), TemplateQuery.population rs)

/-- `stratifiedLaw` through the oracle. -/
def stratifiedLawWith (hs : Hashes M) (obj : Json) (program : ByteArray) (k : Nat) : M (Draw.Law × Nat) := do
  let (strata, population) ← templateStrataWith hs "stratified" obj program
  match Draw.derive k strata with
  | .ok law => return (law, population)
  | .error e => throw s!"stratified: {e}"

/-- `workLaw` through the oracle. -/
def workLawWith (hs : Hashes M) (obj : Json) (program : ByteArray) (k : Nat) (table : Array (String × Nat × Nat)) :
    M (Draw.Law × Nat) := do
  let (strata, population) ← templateStrataWith hs "work" obj program
  match Draw.deriveWork k strata table with
  | .ok law => return (law, population)
  | .error e => throw s!"work: {e}"

/-- `checkPublic` through the oracle. -/
def checkPublicWith (hs : Hashes M) (c : Circuit) (pub : Public) : M Unit := do
  if J.has c.metaJ "links" then throw "META links: a statement with public outputs has no output row to link (links are §16.13's)"
  let fv ← pub.frameV3
  if (← J.str fv "frame" "public file") != "frame-v3-sha512"
      || (← J.str fv "row_key_sha512" "public file") != Pinned.rowKey then
    throw "the instance file's rows are not frame-v3-sha512 hm96 leaves under the default key"
  let np := c.ports.size
  let (n, bcs, outputs) := pub.population.getD (pub.n, pub.digests, pub.outputs)
  if let .ok d := pub.header.getObjVal? "unit_draw" then
    let u ← Draw.UnitDraw.ofJson d
    if (← J.nat pub.header "instances" "public file") != u.proved.size || pub.n != u.proved.size then
      throw "U3: the statement's instances are not the drawn units and their closure"
  for p in [0:np] do
    let port := c.ports[p]!
    let dom ← pub.fv3Bytes "domain_ids" port.name
    let schema := if port.segs.isSome then ROW_SCHEMA_SEG else if port.v2 then ROW_SCHEMA_V2 else ROW_SCHEMA
    if (← J.str (← J.get fv "schemas" "public file") port.name "public file") != schema then
      throw s!"input port {port.name}: schema is not {schema}"
    let (count, row) : Nat × (Nat → Nat) := match pub.tables with
      | some t => (t.counts[p]!, fun r => t.starts[p]! + r)
      | none => (n, fun i => i * np + p)
    let leaves ← (Array.range count).mapM fun r => do
      FV3.leafWith hs dom r schema (← Hm96.Default512.treeLeafWith hs (bcs.getD (row r) ByteArray.empty))
    if (← FV3.rootWith hs dom leaves) != (← pub.fv3Bytes "roots" port.name) then
      throw s!"input port {port.name}: the rows' b || c do not hash to the header's frame-v3-sha512 root"
    if zeroRowBc port != c.dummyDigests.getD p ByteArray.empty then
      throw s!"the pinned dummy b || c of port {port.name} is not the zero row's"
  let nOut := c.outWords
  let mut at_ := 0
  for (oname, w) in c.outPorts do
    let dom ← pub.fv3Bytes "domain_ids" oname
    let words := (Array.range n).foldl (fun acc i => acc ++ (Array.range w).map fun k => outputs.getD (i * nOut + at_ + k) 0) #[]
    let leaves ← (Array.range words.size).mapM fun r =>
      FV3.leafWith hs dom r "u16" (ByteArray.mk #[(words[r]! / 256).toUInt8, (words[r]! % 256).toUInt8])
    if (← FV3.rootWith hs dom leaves) != (← pub.fv3Bytes "roots" oname) then
      throw s!"output port {oname}: the output words do not hash to the header's frame-v3-sha512 root"
    at_ := at_ + w

end HmRow

/-- `Stmt.setupH` through the oracle. -/
def Stmt.setupHWith (hs : Hashes M) (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (tables : List (String × ByteArray) := []) (draw : Option Json := none) (partition : Option Json := none)
    (program : Option ByteArray := none) : M Stmt := do
  let c ← if tags.typed then HmRow.parseTypedWith hs tags circuitFile tables else HmRow.parseWith hs tags circuitFile tables
  let finding0 ← if tags.bindings then (some <$> HmRow.checkPartitionWith hs c partition)
    else if partition.isSome then throw s!"{tags.name}: the statement binds no partition" else pure none
  let some merkle := MerkleScheme.find? c.leafScheme | throw s!"leaf-commitment scheme {c.leafScheme}: not one this verifier implements"
  let pop ← HmRow.loadPublicWith hs tags c publicFile
  let pub ← match draw with
    | some d => HmRow.drawnWith hs c pop d
    | none => pure pop
  -- a stratified draw's strata are the verifier's own, from its program and query, when it holds both
  if let some d := draw then
    if let (Except.ok (Draw.Law.stratified k strata), some obj, some prog) :=
        ((Draw.UnitDraw.ofJson d).map (·.law), partition, program) then
      let (law, _) ← HmRow.stratifiedLawWith hs obj prog k
      if law != Draw.Law.stratified k strata then throw "U2: the draw's strata are not the verifier's own (its program and query)"
  HmRow.checkPublicWith hs c pub
  -- a template query's units, derived from the verifier's own program: the statement's units must be instances of its template
  let finding ← match finding0, partition, program with
    | some f, some obj, some prog =>
      let q := (obj.getObjVal? "query").toOption.getD Json.null
      if (TemplateQuery.templatesOf ((q.getObjValAs? String "name").toOption.getD "") ((q.getObjValAs? Nat "version").toOption.getD 0) Json.null).isSome then
        let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
        pure (some { f with derived := true, note := ← HmRow.deriveTemplateUnitsWith hs c obj prog indices })
      else HmRow.qwordFindingWith hs c pub obj prog f
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
  let mut h := tags.statement.toUTF8 ++ c.sha ++ (← canon identity).toUTF8
  for x in [n, blocksReal, nbl, m, c.kLog, pin, regions.size, c.g] do h := pushLe64 h x
  for d in [da, db] do
    h := pushLe64 h d.size
    for (i, cc) in d do h := pushLe32 (pushLe32 h i) cc
  let digest ← hs.sha512 h
  let sigma ← hs.sha512 (tags.sigmaTag.toUTF8 ++ c.sha ++ digest ++ pub.sha)
  let hello ← helloOf tags sigma (PT_LOCAL + nbl) coins
  return { tags, leaf := default, c, pub, n, blocksReal, nbl, m, mPts := PT_LOCAL + nbl, pin, regions, da, db, digest, sigma,
           hello, coinTree := coins.tree?, merkle, partition := (finding.map (·.toJson)).getD Json.null }

namespace HmOut

open HmRow

/-- `HmOut.parse` through the oracle. -/
def parseWith (hs : Hashes M) (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) :
    M Circuit := do
  let c ← if !tags.typed then HmRow.parseWith hs tags bytes tables else do
    let (k, r) ← Typed.readWith hs tags bytes tables
    match r with
    | .inr t =>
      let c ← HmRow.parseWith hs tags bytes tables (tmpl := some { t with ins := t.ins ++ t.outs.map fun (w, b) => ((w + 63) / 64 * 64, b) })
      pure { c with typed := some (blockOf t), cls := some k }
    | .inl unit => pure { (← HmRow.parseWith hs tags bytes tables (pre := #[("unit", unit)])) with cls := some k }
  checkRows c
  return c

/-- `HmOut.checkPublic` through the oracle. -/
def checkPublicWith (hs : Hashes M) (c : Circuit) (pub : Public) : M Unit := do
  let fv ← pub.frameV3
  if (← J.str fv "frame" "public file") != "frame-v3-sha512"
      || (← J.str fv "row_key_sha512" "public file") != Pinned.rowKey then
    throw "the instance file's rows are not frame-v3-sha512 hm96 leaves under the default key"
  let np := c.ports.size
  let (n, bcs, _) := pub.population.getD (pub.n, pub.digests, pub.outputs)
  if let .ok d := pub.header.getObjVal? "unit_draw" then
    let u ← Draw.UnitDraw.ofJson d
    if (← J.nat pub.header "instances" "public file") != u.proved.size || pub.n != u.proved.size then
      throw "U3: the statement's instances are not the drawn units and their closure"
  for p in [0:np] do
    let port := c.ports[p]!
    let dom ← pub.fv3Bytes "domain_ids" port.name
    let schema := if port.segs.isSome then ROW_SCHEMA_SEG else if port.v2 then ROW_SCHEMA_V2 else ROW_SCHEMA
    if (← J.str (← J.get fv "schemas" "public file") port.name "public file") != schema then
      throw s!"row {port.name}: schema is not {schema}"
    let (count, row) : Nat × (Nat → Nat) := match pub.tables with
      | some t => (t.counts[p]!, fun r => t.starts[p]! + r)
      | none => (n, fun i => i * np + p)
    let leaves ← (Array.range count).mapM fun r => do
      FV3.leafWith hs dom r schema (← Hm96.Default512.treeLeafWith hs (bcs.getD (row r) ByteArray.empty))
    if (← FV3.rootWith hs dom leaves) != (← pub.fv3Bytes "roots" port.name) then
      throw s!"row {port.name}: the rows' b || c do not hash to the header's frame-v3-sha512 root"
    if rowBcZeroSalt port (dummyRow c p) != c.dummyDigests.getD p ByteArray.empty then
      throw s!"the pinned dummy b || c of row {port.name} is not the padding instance's row under a zero salt"

end HmOut

namespace HmIn

open HmRow

/-- `HmIn.loadPublic` through the oracle. -/
def loadPublicWith (hs : Hashes M) (tags : Tags) (c : Circuit) (pi : Ports) (file : ByteArray) : M Public := do
  let pub ← HmRow.loadPublicWith hs tags { c with outPorts := if pi.ports.isEmpty then #[] else #[("public inputs", pi.words)] } file
  match pub.header.getObjVal? "public_inputs", pi.ports.isEmpty with
  | .error _, true => pure ()
  | .ok h, false => if (← canon h) != (← canon pi.json) then throw "public file: header public_inputs is not the circuit's"
  | _, _ => throw "public file: header public_inputs is not the circuit's"
  if let some t := pub.tables then
    let (np, k) := (c.ports.size, HmOut.nIn c)
    if (t.counts.extract k np).any (· != pub.n) then throw "header: an output row's table is not one row per instance"
    if (List.range t.refs.size).any (fun x => x % np ≥ k && t.refs[x]! != x / np) then
      throw "shared rows: an instance's output row is not its own"
  (HmIn.checkReads c pi pub.n pub.tables : V Unit)
  return pub

/-- `HmIn.drawn` through the oracle. -/
def drawnWith (hs : Hashes M) (c : Circuit) (pi : Ports) (pub : Public) (draw : Json) : M Public :=
  HmRow.drawnWith hs { c with outPorts := if pi.ports.isEmpty then #[] else #[("public inputs", pi.words)] } pub draw

end HmIn

namespace HmLinks

open HmRow

/-- `HmLinks.check` through the oracle. -/
def checkWith (hs : Hashes M) (c : Circuit) (pop : Public) (program : ByteArray) : M Nat := do
  let links ← parse c
  let P ← Program.decodeWith hs.sha512 program
  let us := units P
  let np := c.ports.size
  let popIdx ← (← J.arr (← J.get pop.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
  let inst : Std.HashMap Nat Nat := (Array.range popIdx.size).foldl (fun m i => m.insert popIdx[i]! i) {}
  let mut outs : Std.HashMap Nat (Array Nat) := {}
  let mut wires := 0
  for i in [0:popIdx.size] do
    let u := popIdx[i]!
    let some (k, m, _, _) := us[u]? | throw s!"links: unit {u} is outside the program's {us.size} units"
    let ops ← Program.operands P P.rootScope ((P.defs[P.root]!).body.nodes[k]!) m
    for t in [0:ops.size] do
      if let some v := owner us ops[t]! then
        if !links.any (fun (_, lo, hi) => lo ≤ t && t < hi) then
          throw s!"links: cut wire: unit {u}'s parameter leaf {t} reads unit {v}'s output (gate {ops[t]!}), and no link row (META links) carries it"
    for (p, lo, hi) in links do
      let name := (c.ports.getD p default).name
      if hi > ops.size then throw s!"links: {name}'s leaves [{lo}, {hi}) are past unit {u}'s {ops.size} parameter leaves"
      let rd := ops.extract lo hi
      match owner us rd[0]! with
      | none =>
        if rd.any (owner us · |>.isSome) then
          throw s!"links: unit {u}'s leaves [{lo}, {hi}) (row {name}) read both circuit inputs and a unit's outputs"
      | some v =>
        let og ← match outs[v]? with
          | some og => pure og
          | none => do
            let og ← outGates P us v
            outs := outs.insert v og
            pure og
        if og != rd then
          throw s!"links: unit {u}'s leaves [{lo}, {hi}) (row {name}) are not exactly unit {v}'s {og.size} outputs in order"
        let some w := inst[v]?
          | throw s!"links: unit {u}'s row {name} reads unit {v}, which is not a unit of this public file (a link across statements is not checked)"
        if pop.digest i p np != pop.digest w (np - 1) np then
          throw s!"links: cut wire [{lo}, {hi}) from unit {v} to unit {u}: the reader's row {name} is not the writer's committed output row (their b ‖ c differ)"
        wires := wires + 1
  return wires

end HmLinks

/-- `Stmt.setupHidden` through the oracle. -/
def Stmt.setupHiddenWith (hs : Hashes M) (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (tables : List (String × ByteArray) := []) (draw : Option Json := none) (partition : Option Json := none)
    (program : Option ByteArray := none) (publicInputs : Option ByteArray := none) : M Stmt := do
  let c ← HmOut.parseWith hs tags circuitFile tables
  let pi ← HmIn.parse c
  let finding0 ← if tags.bindings then (some <$> HmRow.checkPartitionWith hs c partition)
    else if partition.isSome then throw s!"{tags.name}: the statement binds no partition" else pure none
  let some merkle := MerkleScheme.find? c.leafScheme | throw s!"leaf-commitment scheme {c.leafScheme}: not one this verifier implements"
  let pop ← HmIn.loadPublicWith hs tags c pi publicFile
  HmIn.checkOwn pi pop publicInputs
  let pub ← match draw with
    | some d => HmIn.drawnWith hs c pi pop d
    | none => pure pop
  if let some d := draw then
    if let (Except.ok (Draw.Law.stratified k strata), some obj, some prog) :=
        ((Draw.UnitDraw.ofJson d).map (·.law), partition, program) then
      let (law, _) ← HmRow.stratifiedLawWith hs obj prog k
      if law != Draw.Law.stratified k strata then throw "U2: the draw's strata are not the verifier's own (its program and query)"
  HmOut.checkPublicWith hs c pub
  let linked := J.has c.metaJ "links"
  let finding ← match finding0, partition, program with
    | some f, some obj, some prog =>
      let q := (obj.getObjVal? "query").toOption.getD Json.null
      if (TemplateQuery.templatesOf ((q.getObjValAs? String "name").toOption.getD "") ((q.getObjValAs? Nat "version").toOption.getD 0) Json.null).isSome then
        let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
        let note ← HmRow.deriveTemplateUnitsWith hs c obj prog indices
        let wires ← HmLinks.checkWith hs c pop prog
        let linking := if wires == 0 && !linked then "; no cut wire between the statement's units"
          else s!"; Linking checked: {wires} cut wire(s) over the {pop.n} units of the population, each reader's link row the writer's committed output row"
        pure (some { f with derived := true, note := note ++ linking })
      else if linked then throw "META links: Linking is checked under a template query (Q_template_instance v0, Q_template_instances v0) only"
      else HmRow.qwordFindingWith hs c pub obj prog f
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
  let mut h := tags.statement.toUTF8 ++ c.sha ++ (← canon identity).toUTF8
  for x in [n, blocksReal, nbl, m, c.kLog, pin, regions.size, c.g] do h := pushLe64 h x
  for d in [da, db] do
    h := pushLe64 h d.size
    for (i, cc) in d do h := pushLe32 (pushLe32 h i) cc
  let digest ← hs.sha512 h
  let sigma ← hs.sha512 (tags.sigmaTag.toUTF8 ++ c.sha ++ digest ++ pub.sha)
  let hello ← helloOf tags sigma (PT_LOCAL + nbl) coins
  return { tags, leaf := default, c, pub, n, blocksReal, nbl, m, mPts := PT_LOCAL + nbl, pin, regions, da, db, digest, sigma,
           hello, coinTree := coins.tree?, merkle, partition := (finding.map (·.toJson)).getD Json.null }

/-- `Stmt.setupTables` through the oracle. -/
def Stmt.setupTablesWith (hs : Hashes M) (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : Nat)
    (tables : List (String × ByteArray) := []) (draw : Option Json := none) (partition : Option Json := none)
    (program : Option ByteArray := none) (work : Option (Nat × Array (String × Nat × Nat)) := none)
    (stratified : Option Nat := none) (closure : Option (Std.HashMap Nat (Array Nat)) := none)
    (publicInputs : Option ByteArray := none) : M (SessionSpec × Array Stmt) := do
  -- a closure draw proves more units than it draws, so what it adds must be the verifier's own closure of the draw
  if let some d := draw then
    if let .ok u := Draw.UnitDraw.ofJson d then
      if let .work .. := u.law then
        match closure with
        | some cm =>
          if u.closure != Draw.closureOf cm u.units then
            throw "U3: the draw's closure is not the verifier's own (its closure map)"
        | none =>
          if !u.closure.isEmpty then throw "U3: a closure draw needs the verifier's own closure map (--closure)"
  -- a stratified draw's K and strata set every k_s, so its law must be the verifier's own derivation, never taken as stated
  if let some d := draw then
    if let .ok (Draw.Law.stratified k strata) := (Draw.UnitDraw.ofJson d).map (·.law) then
      let (some obj, some prog, some K) := (partition, program, stratified)
        | throw "U2: a stratified draw needs the verifier's own K, program and partition (--stratified, --program, --partition)"
      if k != K then throw s!"U2: the draw's K is {k}, not the verifier's {K}"
      let (law, _) ← HmRow.stratifiedLawWith hs obj prog K
      if law != Draw.Law.stratified k strata then
        throw "U2: the draw's strata are not the verifier's own (its K, program and query)"
  -- a work draw's K and stated work set every k_s, so its law must be the verifier's own derivation, never taken as stated
  if let some d := draw then
    if let .ok (Draw.Law.work k strata) := (Draw.UnitDraw.ofJson d).map (·.law) then
      let (some obj, some prog, some (K, table)) := (partition, program, work)
        | throw "U2: a work draw needs the verifier's own K, work table, program and partition (--work, --work-table, --program, --partition)"
      if k != K then throw s!"U2: the draw's K is {k}, not the verifier's {K}"
      let (law, _) ← HmRow.workLawWith hs obj prog K table
      if law != Draw.Law.work k strata then
        throw "U2: the draw's work law is not the verifier's own (its K, program, query and work table)"
  if publicInputs.isSome && !tags.hiddenOutputs then
    throw s!"--public-inputs: a {tags.name} statement has no public input ports (only a statement with hidden outputs has them)"
  let setup := fun (t : Tags) (d : Option Json) => do
    if tags.hiddenOutputs then Stmt.setupHiddenWith hs t circuitFile publicFile coins tables d partition program publicInputs
    else
      let st ← Stmt.setupHWith hs t circuitFile publicFile coins tables d partition program
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
  -- the draw's law and count are refused before the circuit is read
  let u? ← (draw.mapM Draw.UnitDraw.ofJson : V _)
  if let some u := u? then
    match u.law with
    | .subset _ | .stratified .. => splits u.units.size
    | _ => throw Draw.TABLES_LAW
  -- without a draw, the population's units (`setupH` checks a draw's population against the file)
  let (population, units) ← match u? with
    | some u => pure (u.population, u.units)
    | none => do
      let n ← if tags.hiddenOutputs then do
          let c ← HmOut.parseWith hs tags circuitFile tables
          pure (← HmIn.loadPublicWith hs tags c (← HmIn.parse c) publicFile).n
        else do pure (← HmRow.loadPublicWith hs tags (← HmRow.parseWith hs tags circuitFile tables) publicFile).n
      pure (n, Array.range n)
  splits units.size
  let k := units.size / count
  let sts ← (Array.range count).mapM fun j =>
    let part := Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson population), ("k", Lean.toJson k),
      ("units", Json.arr ((units.extract (j * k) (j * k + k)).map Lean.toJson))]
    setup { tags with identity := tags.tableIdentity count j } (some part)
  let mPts := sts[0]!.mPts
  if sts.any (·.mPts != mPts) then throw "tables: the tables' link points have different lengths"
  let sigma ← hs.sha512 (sts.foldl (fun h st => h ++ st.sigma) (pushLe64 tt.sigmaTag.toUTF8 count))
  let names := tags.tableNames count
  let hello ← helloOf tags sigma mPts coins names
  return ({ tables := names, domain := tags.tableDomain count, sigma, hello, mPts, retainRounds := tags.retainRounds,
            unitDraw := draw, instances := units.size, sha512Digests := tags.hm96Rows, coinTree := coins.tree? }, sts)

end Flock
