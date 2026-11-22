/-
`Flock.Program.Prog.encode`: a Program's descriptor as the codec writes it (`codec.descriptor_chunks`,
`verity/circuits/PROTOCOL.md` §4), the canonical bytes without `annotations` that the program digest hashes. That decoding
the encoding gives the Program back is tested, on the decoder's corpus (`tests/tools/difftest/test_program_decode_port.py`)
and by the `#guard`s below, and not proved. A builder owes no such theorem: the Program it registers is the decode of the
bytes it wrote.
-/
module

public import Circuits.Descriptor
meta import Circuits.Descriptor

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Flock.Program

open Lean (Json)

deriving instance Hashable for Ref, Lane
deriving instance BEq, Hashable for Part

private def spaceJ (p : Bool) : Json := .str (if p then "p" else "n")

/-- A part as §3 spells it (`refs.*.to_json`). -/
def Part.toJson : Part → Json
  | .affine p i b c => Json.mkObj [("affine", .arr #[spaceJ p, Lean.toJson i, Lean.toJson b, Lean.toJson c])]
  | .strided p i b inner os is c =>
    Json.mkObj [("strided", .arr #[spaceJ p, Lean.toJson i, Lean.toJson b, Lean.toJson inner, Lean.toJson os, Lean.toJson is,
      Lean.toJson c])]
  | .explicit it => Json.mkObj [("explicit", .arr (it.map Ref.toJson))]
  | .grid run lanes => Json.mkObj [("grid", .arr #[Lean.toJson run, .arr (lanes.map fun l =>
      .arr #[spaceJ l.p, Lean.toJson l.i, Lean.toJson l.b, Lean.toJson l.e,
        .arr (l.lv.toArray.map fun (r, d) => .arr #[Lean.toJson r, Lean.toJson d])])])]

namespace Encode

/-- How many trailing parts an alias may leave written out (`codec.ALIAS_WINDOW`, §4.5). -/
def ALIAS_WINDOW : Nat := 8

/-- A part as the encoder writes it: an empty part is `{"explicit": []}`, the format's one form, though the decoder also reads
an affine or strided part of count 0. -/
def written : Part → Part
  | .affine _ _ _ 0 | .strided _ _ _ _ _ _ 0 => .explicit #[]
  | q => q

/-- §4.6's interning tables as a writer fills them: each entry once, in first-seen order, types compared by their JSON. -/
structure Tables where
  callees : Array String := #[]
  calleeAt : Std.HashMap String Nat := {}
  types : Array Json := #[]
  typeAt : Std.HashMap String Nat := {}

def Tables.callee (T : Tables) (key : String) : Nat × Tables :=
  match T.calleeAt[key]? with
  | some i => (i, T)
  | none => (T.callees.size, { T with callees := T.callees.push key, calleeAt := T.calleeAt.insert key T.callees.size })

def Tables.type (T : Tables) (t : Ty) : Nat × Tables :=
  let j := t.toJson
  let k := j.compress
  match T.typeAt[k]? with
  | some i => (i, T)
  | none => (T.types.size, { T with types := T.types.push j, typeAt := T.typeAt.insert k T.types.size })

/-- §4.5's writer over one body (`codec._BodyEncoder`): an id for each part list met, equal ids exactly for equal lists
(the empty list's 0), and the first node argument recorded under each full list of two parts or more. -/
structure Aliases where
  trie : Std.HashMap (Nat × Part) Nat := {}
  seqs : Std.HashMap Nat (Nat × Nat) := {}

/-- The ids of the prefixes of `ps`, the `m`th that of its first `m` parts. -/
def Aliases.ids (A : Aliases) (ps : Array Part) : Array Nat × Aliases :=
  ps.foldl (fun (ids, A) q =>
    let key := (ids.back!, q)
    match A.trie[key]? with
    | some i => (ids.push i, A)
    | none => (ids.push (A.trie.size + 1), { A with trie := A.trie.insert key (A.trie.size + 1) })) (#[0], A)

/-- A body's reference sequence as §4.5's writer writes it: as argument `(k, a)` of node `k`, or as the body's `ret` (none). -/
def Aliases.refs (A : Aliases) (s : Seq) (at? : Option (Nat × Nat)) : Json × Aliases :=
  let ps := s.parts.map written
  let n := ps.size
  let (ids, A) := A.ids ps
  let ms := (List.range (ALIAS_WINDOW + 1)).filterMap fun d => if d + 2 ≤ n then some (n - d) else none
  let hit := ms.findSome? fun m => match A.seqs[ids[m]!]? with
    | some (k, a) => if at?.all (k < ·.1) then some (m, k, a) else none
    | none => none
  let out : Array Json := match hit with
    | some (m, k, a) => #[Json.mkObj [("alias", .arr #[Lean.toJson k, Lean.toJson a])]] ++ (ps.extract m n).map Part.toJson
    | none => ps.map Part.toJson
  let A := match at? with
    | some ka => if n ≥ 2 && !A.seqs.contains ids[n]! then { A with seqs := A.seqs.insert ids[n]! ka } else A
    | none => A
  (if h : out.size = 1 then out[0] else Json.mkObj [("concat", .arr out)], A)

private def paramsJ (ps : Array (String × Ty)) : Json := .arr (ps.map fun (k, t) => .arr #[.str k, t.toJson])

/-- A composite's `bindings`, read back from its key `id{k=<canonical JSON>,...}` (§4.3), which the decoder holds to them. -/
def bindingsOf (d : Defn) : V Json := do
  if d.key == d.id then return Json.mkObj []
  let rec_ := (d.key.drop d.id.length).toString
  if !(d.key.startsWith d.id && rec_.startsWith "{") then throw s!"definition '{d.key}': a key is its id and binding record"
  let kvs ← (recordPairs rec_).mapM fun (k, v) => match JsonParse.parse v with
    | .ok j => pure (k, j)
    | .error e => throw s!"definition '{d.key}': binding {k}: {e}"
  return Json.mkObj kvs.toList

/-- Definition `d`'s entry (§4.2), adding to `T` what its body's nodes intern (§4.6). -/
def entry (P : Prog) (T : Tables) (d : Defn) : V (Json × Tables) := do
  if d.prim then
    return (Json.mkObj [("form", .str "primitive"), ("id", .str d.id), ("params", paramsJ d.params), ("ret", d.ret.toJson)], T)
  let mut T := T
  let mut A : Aliases := {}
  let mut nodes := #[]
  for h : k in [0:d.body.nodes.size] do
    let x := d.body.nodes[k]
    let (c, T1) := T.callee (P.defs[x.fn]!).key
    T := T1
    let mut args := #[]
    for h : i in [0:x.args.size] do
      let a := x.args[i]
      let (t, T2) := T.type a.ty
      T := T2
      let (r, A2) := A.refs a.seq (some (k, i))
      A := A2
      args := args.push (Json.mkObj [("refs", r), ("type", Lean.toJson t)])
    let (o, T3) := T.type x.out
    T := T3
    let counts : List (String × Json) := match x.form with
      | .batch => [("axes", .arr (x.axes.map fun | none => .null | some ax => Lean.toJson ax)), ("n", Lean.toJson x.n)]
      | .scan => [("n", Lean.toJson x.n), ("n_xs", Lean.toJson x.nxs)]
      | _ => []
    nodes := nodes.push (Json.mkObj ([("args", .arr args), ("fn", Lean.toJson c), ("form", .str x.form.name),
      ("out", Lean.toJson o)] ++ counts))
  let (ret, _) := A.refs d.body.ret none
  let body := Json.mkObj [("nodes", .arr nodes), ("params", paramsJ d.body.params), ("ret", ret),
    ("ret_type", d.body.retType.toJson)]
  return (Json.mkObj [("bindings", ← bindingsOf d), ("body", body), ("form", .str "composite"), ("id", .str d.id)], T)

/-- The Definitions Definition `d` reaches first: its nodes' callees in node order, then the functions its binding record
names, in record order (§4.6). -/
def reaches (P : Prog) (d : Defn) : Array Nat :=
  if d.prim then #[] else d.body.nodes.map (·.fn) ++ (recordFunctionIds d.key).filterMap (P.index[·]?)

/-- §4.6's post-order walk from Definition `i`, each Definition once (`codec._walk`), through at most `fuel` nested
Definitions. -/
def walk (P : Prog) : Nat → Nat → Std.HashSet Nat × Array Nat → V (Std.HashSet Nat × Array Nat)
  | 0, _, _ => throw "definitions nest deeper than this encoder follows"
  | fuel + 1, i, (seen, order) => do
    if seen.contains i then return (seen, order)
    let (seen, order) ← (reaches P P.defs[i]!).foldlM (fun acc c => walk P fuel c acc) (seen.insert i, order)
    return (seen, order.push i)

end Encode

open Encode in
/-- **The encoder** writes no ports: the writer of record for no description that has ports (§12), since `Prog` keeps no
trace of them and so can't tell a Program decoded from such a description. Otherwise the canonical bytes (§1) of `P`'s
descriptor without `annotations` (§4.1), as `codec.descriptor_chunks` writes them: the Definitions the root, `program_fn`'s
record and the Boolean map's bodies reach, the tables in §4.6's order, each body's sequences by §4.5's writer, and an empty
part as `{"explicit": []}`. -/
def Prog.encode (P : Prog) : V ByteArray := do
  let fuel := P.defs.size + 1
  let mut acc ← walk P fuel P.root ({}, #[])
  for f in recordFunctionIds P.programFn do
    if let some i := P.index[f]? then acc ← walk P fuel i acc
  let bm := P.boolean.toArray.qsort (·.1 < ·.1)
  for (w, b) in bm do
    let some i := P.index[b]? | throw s!"boolean: {w}'s body '{b}' is no Definition of the Program"
    acc ← walk P fuel i acc
  let mut T : Tables := {}
  let mut defs := #[]
  for i in acc.2 do
    let d := P.defs[i]!
    let (e, T') ← entry P T d
    T := T'
    defs := defs.push (d.key, e)
  let top := Json.mkObj [("boolean", Json.mkObj (bm.toList.map fun (k, v) => (k, Json.str v))),
    ("callees", .arr (T.callees.map Json.str)), ("definitions", Json.mkObj defs.toList), ("program_fn", .str P.programFn),
    ("root", .str (P.defs[P.root]!).key), ("schema", .str SCHEMA), ("types", .arr T.types)]
  return (← canon top).toUTF8

/-- A small descriptor with its interning tables out of the writer's order and an alias inside a concatenation of one part,
which the decoder accepts. -/
private def HAND : String :=
  "{\"boolean\":{},\"callees\":[\"Input1_v1\",\"G_v1{K=2}\",\"Xor2_v1\"],\"definitions\":{\"G_v1{K=2}\":{\"bindings\":{\"K\":2},\
  \"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"explicit\":[[\"p\",0,0],[\"p\",0,3]]},\"type\":2}],\"fn\":2,\"form\":\"primitive\",\
  \"out\":1}],\"params\":[[\"x\",[\"Array\",4,[\"Value\",1]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",\
  1]},\"form\":\"composite\",\"id\":\"G_v1\"},\"Input1_v1\":{\"form\":\"primitive\",\"id\":\"Input1_v1\",\"params\":[],\
  \"ret\":[\"Value\",1]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\
  \"form\":\"batch\",\"n\":4,\"out\":0},{\"args\":[{\"refs\":{\"concat\":[{\"affine\":[\"n\",0,2,2]},{\"affine\":[\"n\",\
  0,0,2]}]},\"type\":0}],\"fn\":1,\"form\":\"call\",\"out\":1},{\"args\":[{\"refs\":{\"concat\":[{\"alias\":[1,\
  0]}]},\"type\":0}],\"fn\":1,\"form\":\"call\",\"out\":1}],\"params\":[],\"ret\":{\"concat\":[{\"affine\":[\"n\",\
  1,0,1]},{\"affine\":[\"n\",2,0,1]}]},\"ret_type\":[\"Array\",2,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"},\
  \"Xor2_v1\":{\"form\":\"primitive\",\"id\":\"Xor2_v1\",\"params\":[[\"x\",[\"Array\",2,[\"Value\",1]]]],\"ret\":[\"Value\",\
  1]}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Array\",\
  4,[\"Value\",1]],[\"Value\",1],[\"Array\",2,[\"Value\",1]]]}"

/-- `HAND` as the codec writes it: a binding record, a batch, explicit and affine parts, a concatenation and an alias. -/
private def WRITTEN : String :=
  "{\"boolean\":{},\"callees\":[\"Xor2_v1\",\"Input1_v1\",\"G_v1{K=2}\"],\"definitions\":{\"G_v1{K=2}\":{\"bindings\":{\"K\":2},\
  \"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"explicit\":[[\"p\",0,0],[\"p\",0,3]]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\
  \"out\":1}],\"params\":[[\"x\",[\"Array\",4,[\"Value\",1]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",\
  1]},\"form\":\"composite\",\"id\":\"G_v1\"},\"Input1_v1\":{\"form\":\"primitive\",\"id\":\"Input1_v1\",\"params\":[],\
  \"ret\":[\"Value\",1]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":1,\
  \"form\":\"batch\",\"n\":4,\"out\":2},{\"args\":[{\"refs\":{\"concat\":[{\"affine\":[\"n\",0,2,2]},{\"affine\":[\"n\",\
  0,0,2]}]},\"type\":2}],\"fn\":2,\"form\":\"call\",\"out\":1},{\"args\":[{\"refs\":{\"alias\":[1,0]},\"type\":2}],\
  \"fn\":2,\"form\":\"call\",\"out\":1}],\"params\":[],\"ret\":{\"concat\":[{\"affine\":[\"n\",1,0,1]},{\"affine\":[\"n\",\
  2,0,1]}]},\"ret_type\":[\"Array\",2,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"},\"Xor2_v1\":{\"form\":\"primitive\",\
  \"id\":\"Xor2_v1\",\"params\":[[\"x\",[\"Array\",2,[\"Value\",1]]]],\"ret\":[\"Value\",1]}},\"program_fn\":\"P_v1\",\
  \"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Array\",2,[\"Value\",1]],[\"Value\",\
  1],[\"Array\",4,[\"Value\",1]]]}"

/-- A root that reads an input node at a width dividing its own (§4.4), with an unused entry in `types`. -/
private def RECUT : String :=
  "{\"boolean\":{},\"callees\":[\"Input16_v1\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\
  \"params\":[],\"ret\":[\"Value\",16]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\
  \"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",0,0,4]},\"ret_type\":[\"Array\",\
  4,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\
  \"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",16],[\"Array\",2,[\"Value\",16]],[\"Array\",4,[\"Value\",\
  8]]]}"

/-- `RECUT` as the codec writes it. -/
private def RECUT_WRITTEN : String :=
  "{\"boolean\":{},\"callees\":[\"Input16_v1\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\
  \"params\":[],\"ret\":[\"Value\",16]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\
  \"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":0}],\"params\":[],\"ret\":{\"affine\":[\"n\",0,0,4]},\"ret_type\":[\"Array\",\
  4,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\
  \"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Array\",2,[\"Value\",16]]]}"

/-- A root whose node reads, and which returns, empty affine parts at a position, and a callee returning an empty
strided part, which the decoder reads. -/
private def EMPTY : String :=
  "{\"boolean\":{},\"callees\":[\"Input1_v1\",\"E_v1\"],\"definitions\":{\"E_v1\":{\"bindings\":{},\"body\":{\"nodes\":[]\
  ,\"params\":[[\"x\",[\"Array\",0,[\"Value\",1]]]],\"ret\":{\"strided\":[\"p\",0,0,1,1,1,0]},\"ret_type\":[\"Array\",0,\
  [\"Value\",1]]},\"form\":\"composite\",\"id\":\"E_v1\"},\"Input1_v1\":{\"form\":\"primitive\",\"id\":\"Input1_v1\",\"p\
  arams\":[],\"ret\":[\"Value\",1]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"f\
  n\":0,\"form\":\"batch\",\"n\":4,\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,3,0]},\"type\":1}],\"fn\":1,\"fo\
  rm\":\"call\",\"out\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",1,0,0]},\"ret_type\":[\"Array\",0,[\"Value\",1]]},\"\
  form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity\
  -ir/descriptor/v2\",\"types\":[[\"Array\",4,[\"Value\",1]],[\"Array\",0,[\"Value\",1]]]}"

/-- `EMPTY` as the encoder writes it: each empty part `{"explicit": []}`. -/
private def EMPTY_WRITTEN : String :=
  "{\"boolean\":{},\"callees\":[\"Input1_v1\",\"E_v1\"],\"definitions\":{\"E_v1\":{\"bindings\":{},\"body\":{\"nodes\":[]\
  ,\"params\":[[\"x\",[\"Array\",0,[\"Value\",1]]]],\"ret\":{\"explicit\":[]},\"ret_type\":[\"Array\",0,[\"Value\",1]]},\
  \"form\":\"composite\",\"id\":\"E_v1\"},\"Input1_v1\":{\"form\":\"primitive\",\"id\":\"Input1_v1\",\"params\":[],\"ret\
  \":[\"Value\",1]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"\
  batch\",\"n\":4,\"out\":0},{\"args\":[{\"refs\":{\"explicit\":[]},\"type\":1}],\"fn\":1,\"form\":\"call\",\"out\":1}],\
  \"params\":[],\"ret\":{\"explicit\":[]},\"ret_type\":[\"Array\",0,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"Root\
  [P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"\
  Array\",4,[\"Value\",1]],[\"Array\",0,[\"Value\",1]]]}"

/-- `encode` of what `decode` reads from `text`, as text; none where either refuses. -/
private def reencode (text : String) : Option String :=
  match decode text.toUTF8 >>= Prog.encode with
  | .ok b => String.fromUTF8? b
  | .error _ => none

#guard reencode WRITTEN == some WRITTEN
#guard reencode HAND == some WRITTEN
#guard reencode RECUT_WRITTEN == some RECUT_WRITTEN
#guard reencode RECUT == some RECUT_WRITTEN
#guard reencode EMPTY_WRITTEN == some EMPTY_WRITTEN
#guard match decode (RECUT_WRITTEN.replace "\"id\":\"Root[P_v1]_v1\""
    "\"id\":\"Root[P_v1]_v1\",\"ports\":{\"ret\":[[0,4,{\"equals\":[\"seed\",0,4]}]]}").toUTF8 with
  | .error e => e.startsWith "E_PORT_DECL: Root[P_v1]_v1 return: a port is a non-empty object with keys from"
  | .ok _ => false
#guard reencode EMPTY == some EMPTY_WRITTEN
#guard HAND != WRITTEN && RECUT != RECUT_WRITTEN && EMPTY != EMPTY_WRITTEN

end Flock.Program
