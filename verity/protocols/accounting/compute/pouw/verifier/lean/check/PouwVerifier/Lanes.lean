module

public import PouwVerifier.Credit
public import PouwWindow.KnownRows
public import PouwWindow.RegFile
public import Spot.Produced

/-!
# A sampled leaf past the cap verified as its lanes and the combine (`credit.py`'s lanes form)

`PouwVerifier.Verified`'s steps for the lanes form (`leaf_lanes`), each `credit.py`'s of its name, string for string: at a K
the period-open commits as `lane<K>` and `combine<K>` (`lanesFiles`, `formOf`), each sampled leaf's statements are its
(R/4)(C/8) lanes and the combine (`leafAsks`), each checked against its own one-call leaf Program (the lanes evidence's
descriptor, `leafOf` the window's template at K and hashing to `LEAF_PROGRAMS`' digest) and its partials
`<prefix>partials@<u>` against the produced record under the window's sample root (`lanesRecord`), then refused at the
`--registered` file credit builds from the descriptor alone (`leafRegisteredFile`: every leaf statement reads an A, B or
partials row in full, which the verifier's file cannot mark). No leaf statement is verified, and none credits.

Only one lanes variant is verified (`lanesShape`): a 32 × 32 tile, at a K the lane tile check takes, at
least `LEAST_K`, of one chunk and not J10-R's; any other sampled lanes-form leaf is refused (`lanes-shape`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace PouwVerifier.Lanes

open Lean (Json toJson)
open PouwVerifier.Credit

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
outside the lanes variant the verifier takes (`lanes-shape`), or none. -/
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

/-- A `--registered` port (`registered_value.port` before its reads) of `value`, a `row/v2` value of 16-bit leaves: the
keys of a `Flock.Registered.FORMAT` port and no position, which C and the partition give (`Flock.OneCircuit.pos`). -/
def portJson (value root : String) (leaves words : Nat) (domain : ByteArray) : Json :=
  Json.mkObj [("value", Json.str value), ("root", Json.str root), ("leaves", toJson leaves), ("words", toJson words),
    ("word_bits", toJson (16 : Nat)), ("domain", Json.str (Flock.hexOf domain)),
    (SCHEMA_KEY, Json.str Flock.HmRow.ROW_SCHEMA_V2)]

/-- `credit.leaf_registered_file`: the verifier's `--registered` file of sampled leaf `u`'s statement `st` and its known
openings (`{name: [(0, b ‖ c)]}`), or the refusal (`registration`): a `Flock.Registered.FORMAT` file, each `reads`
port at its value's table at the leaf's place (`leafPlace`) and no position, each table's entry the verifier's own at
the one root `roots` gives it (its domain binding the window's open root `window` but for the weights' `b/<K>`), each
partials port at the produced record's root
`produced` under the window's sample root `registration`; its known ports (`pub` of the window's Program salt `salt`,
position u and the call's roots, `admit` every row, `o0` the row 1, each its own served value under `knownSalt`). An A,
B or partials port is read in full, which the format cannot mark: a statement with one is refused by name
(`PouwWindow.Lanes.fullHole`), and every leaf statement has one. -/
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
  let full := ((reads r c st).filter fun (_, v, _) => v == "a" || v == "b" || v == PARTIALS).map (·.1)
  if !full.isEmpty then throw ("registration", PouwWindow.Lanes.fullHole full.toArray)
  let mut ports : List (String × Json) := []
  for (port, v, _) in reads r c st do
    let (table, _) := place v
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
    ports := ports ++ [(port, portJson table ((str? root).getD "") leaves words_ domain)]
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
    ports := ports ++ [(port, portJson name (Flock.hexOf root) 1 row.size d)]
    openings := openings.push (name, #[(0, Flock.hexOf bc)])
  pure (Json.mkObj [("format", Json.str Flock.Registered.FORMAT), ("ports", Json.mkObj ports)], openings)

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

/-- `credit._leaf_asks`: the checks of the window's sampled lanes-form leaves (the units among `units` whose K `files`
commits as lanes, `formOf`), from its sampled leaves' statements `leafs` (each its index among the window's
statements): the refusal of the first that refuses, or none. In order: each such leaf's K within
`lanesShape` (`lanes-shape`); the lanes evidence `lanes` present (`missing`); each lanes-form K's leaf Programs (the
evidence's descriptors, hashing to `LEAF_PROGRAMS`' digests and `leafOf` the window's template at K, `circuit`); the
window's sample `draw` (`registration`); the produced record (`lanesRecord`); then each statement, of a sampled lanes-form
leaf, in `statementsOf`, and once (`outcome`), its `--registered` file (`leafRegisteredFile`, which refuses every leaf
statement); and last each sampled lanes-form leaf with every one of its statements (`missing`). It names no verifier
run: a leaf whose file were built would have no outcome, which `Credit.executionPhase` refuses. -/
def leafAsks (lanes : Option Json) (leafs : Array (Nat × Nat × St)) (files : Json) (program : String) (layout : Json)
    (shape : Shape) (roots : Json) (window : String) (units : List Nat) (steps : Nat) (salt : String) (draw : Json) :
    R (Option (String × String)) := do
  let (r, c) := tileRC layout
  let kOf := fun (u : Nat) => (leafCall layout u).2.2
  let tileAt := fun (k : Nat) => (shape.tiles.lookup k).getD { fn := "", a := 0, b := 0, zy := 0 }
  let drawn := units.filter fun u => formOf files (kOf u) == some "lanes"
  for u in drawn do
    if let some why := lanesShape r c (tileAt (kOf u)) (kOf u) then return some ("lanes-shape", s!"drawn leaf {u}: {why}")
  if drawn.isEmpty && leafs.isEmpty then return none
  let some ev := lanes
    | return some ("missing", s!"drawn leaf {drawn.headD 0} is at a lanes-form K and the window carries no " ++
        s!"lanes evidence ({", ".intercalate LANES_KEYS}): prove every drawn lanes-form leaf as its lanes and combine")
  for k in (drawn.map kOf).eraseDups.mergeSort (· ≤ ·) do
    for kind in [LANE, COMBINE] do
      let name := s!"{kind}{k}"
      let unit := leafUnitId (tileAt k) kind
      let leaf : Option ByteArray := match get (get ev LEAF_PROGRAMS) name with
        | .str s => if isHexBytes s then hex? (.str s) else none
        | _ => none
      let want := get (get files LEAF_PROGRAMS) name
      let some leaf := leaf.filter fun l => Json.str (Flock.hexOf (Flock.Sha512.hash l)) == want
        | return some ("circuit", s!"the window's {name} leaf Program is not the one the period-open committed " ++
            s!"({LEAF_PROGRAMS} {name}): verify each lanes-form statement against it")
      let some _ := unit.filter (leafOf leaf · k kind)
        | return some ("circuit", s!"the window's {name} leaf Program is not h1window.leaf_program's at K = " ++
            s!"{k}: the one-call Program of {leafFn kind} over the TILE of the window's own template at K")
  let some registration := (if isObj draw then (digest? (get draw "root")).map fun _ => (str? (get draw "root")).getD ""
      else none)
    | return some ("registration", "the window's draw root, the produced record's registration, is not a " ++
        "SHA-512 digest")
  let coin ← match Spot.Challenger.LiveOpening.ofJson (get draw "coin") with
    | .ok o => pure (Flock.hexOf o.coin)
    | .error e => return some ("registration", s!"the window's draw coin, the produced record's draw, is not a " ++
        s!"live opening: {e}")
  let produced ← match lanesRecord (get ev "produced") layout shape program registration coin drawn with
    | .ok p => pure p
    | .error why => return some why
  let mut seen : List (Nat × St) := []
  for (_, u, st) in leafs do
    unless drawn.contains u do
      return some ("outcome", s!"unit {u} has a leaf statement and is no drawn lanes-form leaf: only the " ++
        "draw's leaves at a K committed as lanes are proved as lanes")
    unless (statementsOf (r / 4 * (c / 8))).contains st do
      return some ("outcome", s!"drawn leaf {u}'s statement {st.repr} is none of leaf_lanes.statements({r}, " ++
        s!"{c}): a lane with its index or the combine")
    if seen.contains (u, st) then
      return some ("outcome", s!"drawn leaf {u}'s statement {st.repr} is proved twice: each statement is " ++
        "proved and verified once, and a retry never reproves it")
    seen := seen ++ [(u, st)]
    if let .error why := leafRegisteredFile program layout shape u st roots window steps salt registration produced then
      return some why
  for u in drawn do
    if let some left := (statementsOf (r / 4 * (c / 8))).find? fun st => !seen.contains (u, st) then
      return some ("missing", s!"drawn leaf {u} has no statement {left.repr}: every drawn leaf's lanes and " ++
        "combine are proved and verified, and a leaf with one missing refuses the window (no partial credit)")
  return none

end PouwVerifier.Lanes
