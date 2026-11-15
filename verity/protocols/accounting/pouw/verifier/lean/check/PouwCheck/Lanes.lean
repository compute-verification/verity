module

public import PouwCheck.Credit
public import PouwWindow.KnownRows
public import Spot.Produced

/-!
# A sampled leaf past the cap verified as its lanes and the combine (`credit.py`'s lanes form)

`PouwCheck.Verified`'s steps for the lanes form (`leaf_lanes`), each `credit.py`'s of its name, string for string: at a K
the period-open commits as `lane<K>` and `combine<K>` (`lanesFiles`, `formOf`), each sampled leaf's statements are its
(R/4)(C/8) lanes and the combine (`leafAsks`), each verified against its own one-call leaf Program (the lanes evidence's
descriptor, `leafOf` the window's template at K and hashing to `LEAF_PROGRAMS`' digest), a population of one
(`leafPartitionOf`), and the `--registered` file credit builds from the descriptor alone (`leafRegisteredFile`: the
window's tables at the leaf's served place, its partials `<prefix>partials@<u>` at the produced record's root under the
window's sample root, `lanesRecord`, its known rows the verifier's own under `knownSalt`). Each session is judged by
`judged`; once all have accepted, every produced value must have been opened at its record's root and every opened row
of a value the verifier holds openings of be the one opening (`outcomes`, `Spot.Produced.opened` and `shared`),
and each sampled leaf is then one accepted outcome under the window's program, its registered finding its statements' reads
of the window's registered values.

Only the form `LeafLanesSound` covers is verified (`lanesShape`): a 32 × 32 tile, at a K the lane tile check takes, at
least `LEAST_K`, of one chunk and not J10-R's; any other sampled lanes-form leaf is refused (`lanes-shape`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace PouwCheck.Lanes

open Lean (Json toJson)
open PouwCheck.Credit

/-- `credit.LEAF_PROGRAMS`: the committed circuit files' field of each lanes-form file's leaf Program's digest. -/
def LEAF_PROGRAMS : String := "leaf_programs"
def LANE : String := "lane"
def COMBINE : String := "combine"
def PARTIALS : String := "partials"
/-- `credit.LANES_TILES` and its Python repr. -/
def LANES_TILES : List (Nat × Nat) := [(32, 32)]
def LANES_TILES_REPR : String := "((32, 32),)"
/-- `h1zy.LEAST_K`: the least K the combine is proved at. -/
def LEAST_K : Nat := 384
def LANES_KEYS : List String := ["leaf_programs", "produced", "statements"]
def LEAF_STATEMENT_KEYS : List String := ["circuit", "public", "sessions", "statement", "unit"]
def LANES_FORM : String :=
  "the window's lanes evidence is not {leaf_programs, produced, statements}: each lanes-form K's lane and " ++
  "combine leaf Program's descriptor in lowercase hex by lane<K> and combine<K>, the produced record " ++
  "(verity/registration/produced/v1) and the statements, each a call's {call, circuit, public, sessions} " ++
  "or a drawn leaf's {unit, statement, circuit, public, sessions}"
/-- `credit.LEAF_FN`: a sampled leaf's statements' wrappers. -/
def leafFn (kind : String) : String := if kind == LANE then "PouwH1WindowLane_v1" else "PouwH1WindowCombine_v1"
/-- `credit.LEAF_UNIT`: their units over the window's template's TILE. -/
def leafUnit (kind : String) : String := if kind == LANE then "PouwH1LaneZy_v1" else "PouwH1Combine_v1"
def ZY_UNIT : String := "PouwH1TileZy_v1"
/-- `leaf_lanes.KNOWN_SALT_TAG`. -/
def KNOWN_SALT_TAG : ByteArray := "verity/pouw/h1/leaf-known/v1".toUTF8.push 0
/-- `leaf_lanes.READ_IN_FULL` (Flock's `KnownRead.READ_KEY`). -/
def READ_IN_FULL : String := "read_in_full"

private def hex2 (n : Nat) : String :=
  let s := String.ofList (Nat.toDigits 16 n)
  "".pushn '0' (2 - s.length) ++ s

/-- Python's `repr` of a string: single-quoted unless it holds a `'` and no `"`, its backslashes, its quote and its
control characters escaped. -/
def pyRepr (s : String) : String :=
  let q := if s.contains '\'' && !s.contains '"' then '"' else '\''
  let body := s.toList.foldl (fun acc c => acc ++ match c with
    | '\\' => "\\\\"
    | '\n' => "\\n"
    | '\r' => "\\r"
    | '\t' => "\\t"
    | c => if c == q then "\\" ++ String.singleton c
      else if c.val < 32 || c.val == 127 then "\\x" ++ hex2 c.val.toNat
      else String.singleton c) ""
  String.singleton q ++ body ++ String.singleton q

/-- Python's `repr` of a list of strings. -/
def pyList (xs : List String) : String := "[" ++ ", ".intercalate (xs.map pyRepr) ++ "]"

/-- A sampled leaf's statement (`leaf_lanes.statements`' entry, `LEAF_STATEMENT_KEYS`' `statement`): its kind, then its
lane index or null. -/
abbrev St := String × List (Option Nat)

/-- Python's `repr` of the statement as a list, `['lane', 4]` or `['combine', None]`. -/
def St.repr (s : St) : String :=
  "[" ++ ", ".intercalate (pyRepr s.1 :: s.2.map fun | some n => toString n | none => "None") ++ "]"

/-- `leaf_lanes.statements(r, c)` of a leaf of `lanes` lanes: each lane with its index, then the combine. -/
def statementsOf (lanes : Nat) : List St := (List.range lanes).map (fun l => (LANE, [some l])) ++ [(COMBINE, [none])]

/-- `credit._LEAF_FILE`: a lanes-form circuit file's kind and K, `lane<K>` or `combine<K>` (no leading zero). -/
def leafFile? (x : String) : Option (String × Nat) :=
  let num := fun (d : List Char) => match d with
    | c :: _ => if c != '0' && d.all Char.isDigit then (String.ofList d).toNat? else none
    | [] => none
  if x.startsWith LANE then (LANE, ·) <$> num (x.toList.drop LANE.length)
  else if x.startsWith COMBINE then (COMBINE, ·) <$> num (x.toList.drop COMBINE.length)
  else none

/-- `credit.lanes_files`: why the committed circuit files `files` (an object) don't hold the lanes form, or none: each K
in one form, `k<K>` or `lane<K>` and `combine<K>`, and `LEAF_PROGRAMS` present (not null) exactly when some K is in the
lanes form, `{lane<K>: SHA-512, combine<K>: SHA-512}` over exactly those files. -/
def lanesFiles (files : Json) : Option String := Id.run do
  let mut leaf : List (Nat × List String) := []
  for (x, _) in fields files do
    if let some (kind, k) := leafFile? x then
      leaf := match leaf.lookup k with
        | some ks => leaf.map fun (k', ks') => (k', if k' == k && !ks.contains kind then ks' ++ [kind] else ks')
        | none => leaf ++ [(k, [kind])]
  let byK := leaf.mergeSort fun a b => a.1 ≤ b.1
  for (k, kinds) in byK do
    let kinds := kinds.mergeSort fun a b => a ≤ b
    if kinds != [COMBINE, LANE] then
      return some (s!"K = {k} commits {pyList kinds} of its lanes form's two files: a K proved as lanes commits " ++
        s!"lane{k} and combine{k}")
    if (PouwWindow.field? files s!"k{k}").isSome then
      return some (s!"K = {k} commits both k{k} and lane{k} + combine{k}: each K is in one form, one tile statement a " ++
        "call or each drawn leaf's lanes and combine")
  let names := (byK.flatMap fun (k, _) => [s!"{LANE}{k}", s!"{COMBINE}{k}"]).mergeSort fun a b => a ≤ b
  let progs := match PouwWindow.field? files LEAF_PROGRAMS with
    | some .null | none => none
    | some p => some p
  if progs.isNone && names.isEmpty then return none
  let ok := match progs with
    | some p => !names.isEmpty && isObj p && (fields p).map (·.1) == names
      && (fields p).all fun (_, v) => (digest? v).isSome
    | none => false
  if !ok then
    return some (s!"{LEAF_PROGRAMS} is not \{lane<K>: SHA-512, combine<K>: SHA-512} over exactly the lanes form's " ++
      s!"files {pyList names}: commit each lanes-form K's lane and combine leaf Program's descriptor digest " ++
      "(h1window.leaf_program) beside its circuit files")
  return none

/-- `credit.form_of`: K's form in the committed circuit files, `tile` (`k<K>`), `lanes` (`lane<K>`) or none. -/
def formOf (files : Json) (k : Nat) : Option String :=
  if (PouwWindow.field? files s!"k{k}").isSome then some "tile"
  else if (PouwWindow.field? files s!"{LANE}{k}").isSome then some "lanes" else none

/-- `credit.lanes_shape`: why a sampled leaf at K = `k` of a window of tile `r` × `c`, whose template at K is `tile`, is
outside what `LeafLanesSound` covers (`lanes-shape`), or none. -/
def lanesShape (r c : Nat) (tile : Credit.Tile) (k : Nat) : Option String :=
  if !LANES_TILES.contains (r, c) then
    some (s!"the window's {r} × {c} tile is not one whose lanes LeafLanesSound covers ({LANES_TILES_REPR}: it is " ++
      s!"stated over Fin 32 lanes and the 32 × 32 cells): its {r / 4 * (c / 8)} lanes wait on proofs' " ++
      "LeafLanesSound restated over L = (R/4)(C/8) lanes")
  else if k % 32 != 0 || k < 64 || 2 ^ 17 ≤ k then
    some (s!"K = {k} fails PouwWindow.Lanes.Tile.check (32 | K, 64 <= K < 2^17): pad the window's linears to a K " ++
      "it takes")
  else if k < LEAST_K then
    some s!"K = {k} is below LEAST_K = {LEAST_K}, the least K the combine is proved at, so the leaf has no combine"
  else if (chunks? k).getD 1 > 1 then
    some (s!"K = {k}'s lanes are {(chunks? k).getD 1} chunk lanes each (weights.chunks, K > {Chunks.CHUNK_MAX}), " ++
      "and LeafLanesSound is over whole lanes: they wait on proofs' chunked LeafLanesSound")
  else if tile.j10r then
    some (s!"K = {k}'s template is J10-R's (PouwH1Tile_v5: v2 lanes, chunks and oks), whose lanes wait on proofs' " ++
      "chunked LeafLanesSound")
  else none

/-- `credit._zy_tile`: the TILE binding (its canonical JSON, as a specialization id spells it) of the zy unit of the
window's template `tile` (`TILE_FN`'s UNIT, `PouwH1TileZy_v1{TILE=…}`), or none for another unit. -/
def zyTile (tile : Credit.Tile) : Option String := do
  let rec_ := (Flock.Program.recordPairs tile.fn).toList
  let unit ← loads? (((rec_.reverse.find? (·.1 == "UNIT")).map (·.2)).getD "")
  guard (keys unit == some ["fn"])
  let .str fn := get unit "fn" | none
  guard (fn.startsWith (ZY_UNIT ++ "{"))
  let inner := Flock.Program.recordPairs fn
  guard (inner.toList.map (·.1) == ["TILE"])
  pure inner[0]!.2

/-- `credit.leaf_unit_id`: the descriptor id of the unit sampled leaf statement `kind` proves over the TILE of the
window's own template `tile`, or none when that template's unit is not the zy unit. -/
def leafUnitId (tile : Credit.Tile) (kind : String) : Option String :=
  (zyTile tile).map fun t => s!"{leafUnit kind}\{TILE={t}}"

/-- `credit._parsed`: a descriptor's bytes read as `layoutOf` reads them, canonical JSON of `SCHEMA` with its callees,
definitions, program function and root; none for any other. -/
def parsed? (descriptor : ByteArray) : Option Json := do
  let d ← (String.fromUTF8? descriptor).bind loads?
  let (.str schema, .arr cs, .str _, .str _) := (get d "schema", get d "callees", get d "program_fn", get d "root")
    | none
  guard (schema == SCHEMA && isObj (get d "definitions") && (cs.toList.mapM str?).isSome)
  pure d

/-- `credit.leaf_of`: whether `leaf` (descriptor bytes) is `h1window.leaf_program`'s at K = `k` for `kind`: its program
function `LEAF_FN[kind]{UNIT={"fn": unit},K=k}`, its definition of `unit` the composite `LEAF_UNIT[kind]` bound to the
TILE its id spells, and its root's nodes other than inputs one call of `unit`. -/
def leafOf (leaf : ByteArray) (unit : String) (k : Nat) (kind : String) : Bool := Id.run do
  let some d := parsed? leaf | return false
  let some u := (Spot.PyJson.canonical (Json.mkObj [("fn", Json.str unit)])).toOption | return false
  unless get d "program_fn" == Json.str s!"{leafFn kind}\{UNIT={u},K={k}}" do return false
  let defs := get d "definitions"
  let dj := get defs unit
  let root := match get d "root" with
    | .str r => get defs r
    | _ => .null
  let b := get dj "bindings"
  let tileOk := match (Spot.PyJson.canonical (get b "TILE")).toOption with
    | some t => s!"{leafUnit kind}\{TILE={t}}" == unit
    | none => false
  unless isObj b && get dj "form" == Json.str "composite" && get dj "id" == Json.str (leafUnit kind)
      && keys b == some ["TILE"] && tileOk do
    return false
  let callees : Array String := match get d "callees" with
    | .arr cs => cs.filterMap str?
    | _ => #[]
  let body := if isObj root && get root "form" == Json.str "composite" then get root "body" else .null
  let .arr nodes := get body "nodes" | return false
  let some fns := nodes.toList.mapM (fun x => if isObj x then (nat? (get x "fn")).filter (· < callees.size) else none)
    | return false
  let calls := (nodes.toList.zip fns).filterMap fun (x, f) =>
    if Flock.Program.isInputId callees[f]! then none else some (callees[f]!, get x "form")
  return match calls with
    | [(c, f)] => c == unit && f == Json.str "call"
    | _ => false

/-- `credit.leaf_partition_of`: the partition object a sampled leaf's statement is verified under, its leaf Program's
(descriptor SHA-512 `program`) instances of its one unit `unit`, a population of one. -/
def leafPartitionOf (program unit : String) : Json :=
  Json.mkObj [("format", Json.str PARTITION_FORMAT), ("program", Json.str program),
    ("query", Json.mkObj [("name", Json.str QUERY_NAME), ("version", toJson QUERY_VERSION),
      ("params", Json.mkObj [("templates", Json.arr #[Json.str unit])])])]

/-- A layout's tile R × C (`layout["tile"]`). -/
def tileRC (layout : Json) : Nat × Nat :=
  match get layout "tile" with
  | .arr a => ((nat? (a.getD 0 .null)).getD 0, (nat? (a.getD 1 .null)).getD 0)
  | _ => (0, 0)

/-- The layout's linears `(name, K, n)`. -/
def linearsOf (layout : Json) : List (String × Nat × Nat) :=
  match get layout "linears" with
  | .arr xs => xs.toList.filterMap fun x => match x with
    | .arr #[.str s, k, n] => do pure (s, ← nat? k, ← nat? n)
    | _ => none
  | _ => []

/-- `credit.leaf_call`: unit `u`'s call, its tile in it and its K (its call's linear's, Python's `{name: k}`: a linear
named twice reads as its last). -/
def leafCall (layout : Json) (u : Nat) : Nat × Nat × Nat :=
  let sizes := (callWork layout).map (·.1)
  let starts := (sizes.foldl (fun (xs, at_) n => (xs.push at_, at_ + n)) ((#[] : Array Nat), 0)).1.toList
  let call := (starts.filter (· ≤ u)).length - 1
  let lin : String := match get layout "calls" with
    | .arr cs => match cs[call]? with
      | some (.arr r) => ((r[0]?).bind str?).getD ""
      | _ => ""
    | _ => ""
  let k := (((linearsOf layout).reverse.find? (·.1 == lin)).map (·.2.1)).getD 0
  (call, u - starts.getD call 0, k)

/-- `leaf_lanes.served`: sampled leaf `u`'s own `value`, `<prefix><value>@<u>`. -/
def served (pre value : String) (u : Nat) : String := s!"{pre}{value}@{u}"

/-- `credit.leaf_place`: each value a sampled leaf's statements read (`a`, `b`, `zy`, `receipt`, `partials`), its table
and the leaf's first row in it. -/
def leafPlace (layout : Json) (shape : Shape) (u steps : Nat) : List (String × String × Nat) :=
  let (r, c) := tileRC layout
  let (call, t, k) := leafCall layout u
  let lin : String := match get layout "calls" with
    | .arr cs => match cs[call]? with
      | some (.arr row) => ((row[0]?).bind str?).getD ""
      | _ => ""
    | _ => ""
  let lins := linearsOf layout
  let n := ((lins.reverse.find? (·.1 == lin)).map (·.2.2)).getD 0
  let b0 := ((lins.takeWhile (·.1 != lin)).filter (·.2.1 == k)).foldl (fun s x => s + x.2.2) 0
  let per := n / c
  let (ti, tj) := (t / per, t % per)
  let x := (chunks? k).getD 1
  let (s, _, at_) := stepRow layout steps call
  [("a", (s!"{shape.pre}a/{k}@{s}", x * (at_ + r * ti))), ("b", (s!"{shape.pre}b/{k}", x * (b0 + c * tj))),
    ("zy", (s!"{shape.pre}zy@{call}", t)), ("receipt", (s!"{shape.pre}receipt@{call}", t)),
    (PARTIALS, (served shape.pre PARTIALS u, 0))]

/-- `leaf_lanes.words`: a partial's u32 words at K, 2 (KS + m). -/
def words (k : Nat) : Nat := (PouwWindow.Lanes.Tile.mk 32 32 k).nw

/-- `leaf_lanes.reads` of a v1 leaf (one chunk, not J10-R): each port's value and row, lane `l`'s A rows by row slot,
B rows by column slot, `zy` and its partials row, the combine's partials rows and `receipt`. -/
def reads (r c : Nat) : St → List (String × String × Nat)
  | (_, [some l]) =>
    let T : PouwWindow.Lanes.Tile := ⟨r, c, 0⟩
    (List.range 4).map (fun s => (s!"a{s}", "a", T.rowOf l s)) ++
      (List.range 8).map (fun t => (s!"b{t}", "b", T.colOf l t)) ++ [("zy", "zy", 0), (PARTIALS, PARTIALS, l)]
  | _ => (List.range (r / 4 * (c / 8))).map (fun x => (s!"p{x}", PARTIALS, x)) ++ [("receipt", "receipt", 0)]

/-- `leaf_lanes.known_salt`: the salt a served leaf's known row `value` (its served name) is committed under, SHA-512 of
`KNOWN_SALT_TAG`, i (u32 little-endian), the window's sample root `registration` and the name, for i = 0, 1, 2. -/
def knownSalt (registration value : String) : ByteArray :=
  let head := (hex? (.str registration)).getD ByteArray.empty ++ value.toUTF8
  (List.range 3).foldl (fun acc i =>
    acc ++ Flock.Sha512.hash (KNOWN_SALT_TAG ++ ⟨#[i.toUInt8, 0, 0, 0]⟩ ++ head)) ByteArray.empty

/-- The roots `{value: [root, ...]}` gives a value. -/
def rootsAt (roots : Json) (v : String) : List Json :=
  match get roots v with
  | .arr xs => xs.toList
  | _ => []

private def regErr {α : Type} (x : Flock.V α) : R α := x.mapError fun e => ("registration", e)

/-- A `--registered` port (`produced.port`) of `value` at row `row`, a `row/v2` value of 16-bit leaves. -/
def portJson (value root : String) (leaves words : Nat) (domain : ByteArray) (row : Nat) : Json :=
  Json.mkObj [("value", Json.str value), ("root", Json.str root), ("leaves", toJson leaves), ("words", toJson words),
    ("word_bits", toJson (16 : Nat)), ("domain", Json.str (Flock.hexOf domain)),
    (SCHEMA_KEY, Json.str Flock.HmRow.ROW_SCHEMA_V2), ("reads", Json.arr #[toJson row])]

/-- `credit.leaf_registered_file`: the verifier's `--registered` file of sampled leaf `u`'s statement `st` and its known
openings (`{name: [(0, b ‖ c)]}`), or the refusal (`registration`): each `reads` port at its value's table and row past
the leaf's place (`leafPlace`), each table's entry the verifier's own at the one root `roots` gives it (its domain
binding the window's open root `window` but for the weights' `b/<K>`), each partials port at the produced record's root
`produced` under the window's sample root `registration`; its known ports (`pub` of the window's Program salt `salt`,
position u and the call's roots, `admit` every row, `o0` the row 1, each its own served value under `knownSalt`). An A,
B or partials port is read in full. -/
def leafRegisteredFile (program : String) (layout : Json) (shape : Shape) (u : Nat) (st : St) (roots : Json)
    (window : String) (steps : Nat) (salt registration : String) (produced : List (String × String)) :
    R (Json × Array (String × Array (Nat × String))) := do
  let (r, c) := tileRC layout
  let m := (nat? (get layout "m")).getD 0
  let at_ := leafPlace layout shape u steps
  let place := fun (v : String) => (at_.lookup v).getD ("", 0)
  let (call, _, k) := leafCall layout u
  let tile : Credit.Tile := (shape.tiles.lookup k).getD { fn := "", a := 0, b := 0, zy := 0 }
  let x := (chunks? k).getD 1
  let sizes := (callWork layout).map (·.1)
  let (a, b, zy, rc) := ((place "a").1, (place "b").1, (place "zy").1, (place "receipt").1)
  let rowsOf : List (String × Nat × Nat) :=
    [(a, x * m * (stepRow layout steps call).2.1, tile.a), (zy, sizes.getD call 0, tile.zy),
      (b, x * ((linearsOf layout).filter (·.2.1 == k)).foldl (fun s l => s + l.2.2) 0, tile.b),
      (rc, sizes.getD call 0, tile.receipt)]
  let rcs := rootsAt roots rc
  if rcs.length != 1 then
    throw ("registration", s!"drawn leaf {u}'s combine reads its call's receipts {rc}, which the window registers " ++
      s!"{if rcs.isEmpty then "not" else "twice"}: a lanes-form call registers its receipt table once, the receipts " ++
      "the combine checks the lanes' XOR against")
  let lanes := r / 4 * (c / 8)
  let mut ports : List (String × Json) := []
  for (port, v, row) in reads r c st do
    let (table, base) := place v
    let (root, leaves, words_, domain) ← if v == PARTIALS then do
        let ws := 2 * words k
        pure (Json.str ((produced.lookup table).getD ""), lanes, ws,
          ← regErr (Spot.Produced.domain program table ws lanes 16 registration))
      else do
        let got := rootsAt roots table
        if got.length != 1 then
          throw ("registration", s!"drawn leaf {u}'s statement {st.repr} reads {table}, which the window " ++
            s!"registers {if got.isEmpty then "not" else "twice"}: register each of a call's tables and its weights once")
        let (leaves, ws) := (rowsOf.lookup table).getD (0, 0)
        pure (got[0]!, leaves, ws,
          ← regErr (Spot.Registered.domainAt program table ws leaves 16 (if table == b then none else some window)))
    let some _ := digest? root
      | throw ("registration", s!"drawn leaf {u}'s statement {st.repr} reads {table}, whose root the verifier does " ++
          "not hold as a SHA-512 digest")
    let j := portJson table ((str? root).getD "") leaves words_ domain (base + row)
    let full := v == "a" || v == "b" || v == PARTIALS
    ports := ports ++ [(port, match full, j with
      | true, .obj kvs => .obj (kvs.insert READ_IN_FULL (.bool true))
      | _, j => j)]
  let (some ra, some rb) := (digest? ((rootsAt roots a).headD .null), digest? ((rootsAt roots b).headD .null))
    | throw ("registration", s!"drawn leaf {u}'s pub row reads the roots of {a} and {b}, which are not both " ++
        "SHA-512 digests (128 lowercase hex): register each table at its tree's root")
  let rows : List (String × String × Array Nat) := match st with
    | (_, [some l]) =>
      let sb := (hex? (.str salt)).getD ByteArray.empty
      [("o0", s!"ok{l}", PouwWindow.Lanes.okRow),
        ("pub", s!"pub{l}", PouwWindow.Lanes.pubRow (PouwWindow.Lanes.halvesNat sb) (PouwWindow.Lanes.positionNat u)
          (PouwWindow.Lanes.rootsNat ra rb) l),
        ("admit", "admit", PouwWindow.Lanes.admitRow (List.replicate r true))]
    | _ => [("o0", s!"ok{lanes}", PouwWindow.Lanes.okRow)]
  let mut openings : Array (String × Array (Nat × String)) := #[]
  for (port, value, row) in rows do
    let name := served shape.pre value u
    let (root, bc) ← regErr (Spot.Produced.known program name row (knownSalt registration name) registration 16)
    let d ← regErr (Spot.Produced.domain program name row.size 1 16 registration)
    ports := ports ++ [(port, portJson name (Flock.hexOf root) 1 row.size d 0)]
    openings := openings.push (name, #[(0, Flock.hexOf bc)])
  pure (Json.mkObj [("format", Json.str Flock.Registered.FORMAT_V0), ("ports", Json.mkObj ports)], openings)

/-- `credit._lanes_record`: the produced record's root of each produced value of the window's sampled lanes-form leaves
`units`, when the lanes evidence's record `rec` holds (`Spot.Produced.check` against the sample root
`registration`, the sampling coin's value `draw` and the verifier's own entries, each leaf u's `<prefix>partials@<u>`);
else the refusal (`registration`, its P- codes). -/
def lanesRecord (rec layout : Json) (shape : Shape) (program registration draw : String) (units : List Nat) :
    R (List (String × String)) := do
  let (r, c) := tileRC layout
  let mut mine : Array Spot.Produced.Mine := #[]
  for u in units do
    let name := served shape.pre PARTIALS u
    if mine.any (·.value == name) then continue
    let ws := 2 * words (leafCall layout u).2.2
    let d ← regErr (Spot.Produced.domain program name ws (r / 4 * (c / 8)) 16 registration)
    mine := mine.push { value := name, leaves := r / 4 * (c / 8), words := ws, wordBits := 16, domain := Flock.hexOf d }
  let codes := Spot.Produced.check rec registration draw (mine.qsort (·.value < ·.value))
  unless codes.isEmpty do
    throw ("registration", s!"the window's produced record ({Spot.Produced.FORMAT}) fails " ++
      s!"{", ".intercalate codes.toList}: it holds each drawn lanes-form leaf's partials@<u> under the window's " ++
      "draw root, fixed after the draw coin, and every row's opening once")
  pure <| match get rec "produced" with
    | .arr es => es.toList.filterMap fun e => (str? (get e "value")).map fun v => (v, (str? (get e "root")).getD "")
    | _ => []

/-- One verifier run of a sampled leaf's statement (`credit._leaf_asks`'): the statement's index, the verifier's inputs,
the leaf Program's file name `leaf`, the unit, the known openings and the ports of window values. -/
structure LeafAsk where
  statement : Nat
  circuit : String
  registered : Json
  partition : Json
  leaf : String
  unit : Nat
  known : Array (String × Array (Nat × String))
  windowPorts : List String

/-- A leaf run's input as the caller reads it: `{statement, circuit_sha512, registered, partition, leaf, unit, known,
window_ports}`. -/
def LeafAsk.toJson (a : LeafAsk) : Json :=
  Json.mkObj [("statement", Lean.toJson a.statement), ("circuit_sha512", Json.str a.circuit),
    ("registered", a.registered), ("partition", a.partition), ("leaf", Json.str a.leaf), ("unit", Lean.toJson a.unit),
    ("known", Json.mkObj (a.known.toList.map fun (v, ops) =>
      (v, Json.arr (ops.map fun (p, bc) => Json.arr #[Lean.toJson p, Json.str bc])))),
    ("window_ports", Json.arr (a.windowPorts.map Json.str).toArray)]

/-- `credit._asks`' `leaf_form`: a sampled leaf's statement (`LEAF_STATEMENT_KEYS`, its proof files, a unit of the
window's `population` and a statement of one to three items, a string then naturals or nulls). -/
def leafStatement? (s : Json) (population : Nat) : Option (Nat × St) := do
  guard (keys s == some LEAF_STATEMENT_KEYS)
  let (.str _, .str _, .arr ss) := (get s "circuit", get s "public", get s "sessions") | none
  guard (!ss.isEmpty && ss.all (str? · |>.isSome))
  let u ← nat? (get s "unit")
  guard (u < population)
  let .arr xs := get s "statement" | none
  guard (1 ≤ xs.size && xs.size ≤ 3)
  let .str kind := xs[0]! | none
  let rest ← xs.toList.tail.mapM fun (x : Json) => match x with
    | .null => some none
    | x => (some ·) <$> nat? x
  pure (u, (kind, rest))

/-- `credit._leaf_asks`: the verifier runs of the window's sampled lanes-form leaves (the units among `units` whose K
`files` commits as lanes, `formOf`), from its sampled leaves' statements `leafs` (each its index among the window's
statements), and the refusal of the first check before the verifier that refuses. In order: each such leaf's K within
`lanesShape` (`lanes-shape`); the lanes evidence `lanes` present (`missing`); each lanes-form K's leaf Programs (the
evidence's descriptors, hashing to `LEAF_PROGRAMS`' digests and `leafOf` the window's template at K, `circuit`); the
window's sample `draw` (`registration`); the produced record (`lanesRecord`); then each statement, of a sampled lanes-form
leaf, in `statementsOf`, and once (`outcome`), its `--registered` file (`leafRegisteredFile`); and last each sampled
lanes-form leaf with every one of its statements (`missing`). -/
def leafAsks (lanes : Option Json) (leafs : Array (Nat × Nat × St)) (files : Json) (program : String) (layout : Json)
    (shape : Shape) (roots : Json) (window : String) (units : List Nat) (steps : Nat) (salt : String) (draw : Json) :
    R (Array LeafAsk × Option (String × String)) := do
  let (r, c) := tileRC layout
  let kOf := fun (u : Nat) => (leafCall layout u).2.2
  let tileAt := fun (k : Nat) => (shape.tiles.lookup k).getD { fn := "", a := 0, b := 0, zy := 0 }
  let drawn := units.filter fun u => formOf files (kOf u) == some "lanes"
  for u in drawn do
    if let some why := lanesShape r c (tileAt (kOf u)) (kOf u) then return (#[], some ("lanes-shape", s!"drawn leaf {u}: {why}"))
  if drawn.isEmpty && leafs.isEmpty then return (#[], none)
  let some ev := lanes
    | return (#[], some ("missing", s!"drawn leaf {drawn.headD 0} is at a lanes-form K and the window carries no " ++
        s!"lanes evidence ({", ".intercalate LANES_KEYS}): prove every drawn lanes-form leaf as its lanes and combine"))
  let mut progs : List (String × String × String) := []
  for k in (drawn.map kOf).eraseDups.mergeSort (· ≤ ·) do
    for kind in [LANE, COMBINE] do
      let name := s!"{kind}{k}"
      let unit := leafUnitId (tileAt k) kind
      let leaf : Option ByteArray := match get (get ev LEAF_PROGRAMS) name with
        | .str s => if isHexBytes s then hex? (.str s) else none
        | _ => none
      let want := get (get files LEAF_PROGRAMS) name
      let some leaf := leaf.filter fun l => Json.str (Flock.hexOf (Flock.Sha512.hash l)) == want
        | return (#[], some ("circuit", s!"the window's {name} leaf Program is not the one the period-open committed " ++
            s!"({LEAF_PROGRAMS} {name}): verify each lanes-form statement against it"))
      let some unit := unit.filter (leafOf leaf · k kind)
        | return (#[], some ("circuit", s!"the window's {name} leaf Program is not h1window.leaf_program's at K = " ++
            s!"{k}: the one-call Program of {leafFn kind} over the TILE of the window's own template at K"))
      progs := progs ++ [(name, Flock.hexOf (Flock.Sha512.hash leaf), unit)]
  let some registration := (if isObj draw then (digest? (get draw "root")).map fun _ => (str? (get draw "root")).getD ""
      else none)
    | return (#[], some ("registration", "the window's draw root, the produced record's registration, is not a " ++
        "SHA-512 digest"))
  let coin ← match Spot.Challenger.LiveOpening.ofJson (get draw "coin") with
    | .ok o => pure (Flock.hexOf o.coin)
    | .error e => return (#[], some ("registration", s!"the window's draw coin, the produced record's draw, is not a " ++
        s!"live opening: {e}"))
  let produced ← match lanesRecord (get ev "produced") layout shape program registration coin drawn with
    | .ok p => pure p
    | .error why => return (#[], some why)
  let mut out : Array LeafAsk := #[]
  let mut seen : List (Nat × St) := []
  for (i, u, st) in leafs do
    unless drawn.contains u do
      return (out, some ("outcome", s!"unit {u} has a leaf statement and is no drawn lanes-form leaf: only the " ++
        "draw's leaves at a K committed as lanes are proved as lanes"))
    unless (statementsOf (r / 4 * (c / 8))).contains st do
      return (out, some ("outcome", s!"drawn leaf {u}'s statement {st.repr} is none of leaf_lanes.statements({r}, " ++
        s!"{c}): a lane with its index or the combine"))
    if seen.contains (u, st) then
      return (out, some ("outcome", s!"drawn leaf {u}'s statement {st.repr} is proved twice: each statement is " ++
        "proved and verified once, and a retry never reproves it"))
    seen := seen ++ [(u, st)]
    let name := s!"{st.1}{kOf u}"
    let (reg, known) ← match leafRegisteredFile program layout shape u st roots window steps salt registration produced with
      | .ok x => pure x
      | .error why => return (out, some why)
    let (sha, unit) := (progs.lookup name).getD ("", "")
    let knownNames := known.map (·.1)
    let windowPorts := ((fields (get reg "ports")).filterMap fun (p, e) =>
      let v := (str? (get e "value")).getD ""
      if (produced.lookup v).isSome || knownNames.contains v then none else some p)
    out := out.push {
      statement := i, circuit := (str? (get files name)).getD "", registered := reg,
      partition := leafPartitionOf sha unit, leaf := name, unit := u, known := known, windowPorts := windowPorts }
  for u in drawn do
    if let some left := (statementsOf (r / 4 * (c / 8))).find? fun st => !seen.contains (u, st) then
      return (out, some ("missing", s!"drawn leaf {u} has no statement {left.repr}: every drawn leaf's lanes and " ++
        "combine are proved and verified, and a leaf with one missing refuses the window (no partial credit)"))
  return (out, none)

/-- What the judged leaf sessions hold (`credit._leaf_judged`'s `held`): each leaf's window-value ports, the verdicts
`Spot.Produced.shared` reads, the produced values' opened roots (a value opened at two none) and the known
openings. -/
structure Held where
  ports : Array (Nat × Array (String × Json)) := #[]
  verdicts : Array (String × Json) := #[]
  opened : Array (String × Option String) := #[]
  known : Array (String × Array (Nat × String)) := #[]

/-- Python's `d[k] = v`: `k`'s entry replaced in place, or appended. -/
def upsert {α : Type} (xs : Array (String × α)) (k : String) (v : α) : Array (String × α) :=
  match xs.findIdx? (·.1 == k) with
  | some i => xs.set! i (k, v)
  | none => xs.push (k, v)

/-- `credit._leaf_judged`'s step for one session of sampled leaf statement `a`: the refusal of a rejected session
(`failed`), of one that proved other than the one unit of its population of one (`proof`, `checkFull`), or of one whose
registered finding is not exactly the file's ports at their values and roots (`registration`); else `held` with it. -/
def judged (a : LeafAsk) (session : String) (accepted : Bool) (why : String) (program registered : Json)
    (units : List Nat) (openings : Json) (held : Held) : R Held := do
  unless accepted do
    throw ("failed", s!"drawn leaf {a.unit}'s {a.leaf} session {session} was rejected by the verifier of record " ++
      s!"({why}): a failed statement refuses the whole window (no partial credit, and no retry)")
  unless units == [0] && program == get a.partition "program" do
    throw ("proof", s!"drawn leaf {a.unit}'s {a.leaf} session {session} proved units {listStr units} of another " ++
      "program than its leaf Program's one unit: a leaf statement's draw is every unit of its population of one " ++
      "(checkFull)")
  let want := (fields (get a.registered "ports")).map fun (p, e) =>
    (p, (str? (get e "value")).getD "", (str? (get e "root")).getD "")
  unless finding? registered == some want do
    throw ("registration", s!"drawn leaf {a.unit}'s {a.leaf} session {session} opened its ports at other values or " ++
      "roots than the verifier's own registered file (or a pub, admit or o0 row other than the known one)")
  let tag := s!"s{a.statement}"
  let mine := ((held.ports.find? (·.1 == a.unit)).map (·.2)).getD #[]
  let mine := a.windowPorts.foldl (fun acc p => upsert acc s!"{tag}/{p}" (get registered p)) mine
  let ports := match held.ports.findIdx? (·.1 == a.unit) with
    | some i => held.ports.set! i (a.unit, mine)
    | none => held.ports.push (a.unit, mine)
  let verdict := Json.mkObj [("of_record", .bool true), ("accepted", .bool true), ("registered", registered),
    ("openings", openings)]
  let opened := (fields registered).foldl (fun acc (_, e) =>
    let v := (str? (get e "value")).getD ""
    let root := (str? (get e "root")).getD ""
    match acc.find? (·.1 == v) with
    | some (_, got) => upsert acc v (if got == some root then some root else none)
    | none => acc.push (v, some root)) held.opened
  pure {
    ports, verdicts := upsert held.verdicts s!"{tag}/{session}" verdict, opened,
    known := a.known.foldl (fun acc (v, ops) => upsert acc v ops) held.known }

/-- `credit._leaf_outcomes`: after every sampled leaf's statements accepted, the produced record `rec`'s values each opened
at its root (`P-opened`) and every row the sessions opened of a value the verifier holds openings of the one opening
(`P-shared`), else the refusal (`registration`); then one outcome a sampled leaf, by unit, accepted under the window's
program `program`, its registered finding its statements' ports of the window's registered values. -/
def outcomes (rec : Json) (held : Held) (program : String) : R (Array Json) := do
  let opened := held.opened.map fun (v, r) => (v, match r with | some r => Json.str r | none => Json.null)
  unless (Spot.Produced.opened rec (some opened)).isEmpty do
    throw ("registration", "a produced value of the window's record was not opened, or opened at another root than " ++
      "the record's (P-opened): every leaf's lanes and combine read its partials at the record's root")
  unless (Spot.Produced.shared (Spot.Produced.wanted rec held.known) held.verdicts).isEmpty do
    throw ("registration", "a session opened a row of a produced value, a registered row the record keeps or a " ++
      "known row other than the one opening the verifier holds, or reported none (P-shared)")
  pure <| (held.ports.qsort (·.1 < ·.1)).map fun (u, ps) =>
    Json.mkObj [("unit", toJson u), ("verdict", Json.str "accept"), ("program", Json.str program),
      ("registered", Json.mkObj ps.toList)]

end PouwCheck.Lanes
