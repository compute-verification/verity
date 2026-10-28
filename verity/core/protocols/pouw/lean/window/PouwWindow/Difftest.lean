import Difftest
import PouwWindow.Json
import PouwWindow.Length
import PouwWindow.Statement
import PouwWindow.Inputs
import PouwWindow.Rounds
import PouwWindow.Lanes
import PouwWindow.KnownRows
import PouwWindow.RegFile
import PouwWindow.FullDraw

/-!
# The difftest's functions: the window's Lean behind `Difftest.serve`

One function per ported function of `window.py`, each on a JSON object of its arguments, for the shared harness
(`verity/core/difftest`): an output, or `.error` where Python raises `ValueError` (a reject). The port's
suite (`tests/test_difftest.py`, and `tests/test_lanes.py` for a drawn leaf's statements) draws the inputs and holds each
to Python.

* `on_grid`, `floor_point`, `pad`, `level`, `bucket` (`bits`, `n`); `point`, `work_at` (`bits`, `ell`); `levels` (`bits`,
  `lo`, `hi`);
* `k_prime` (`eps`, `delta` as `[num, den]`, `credited`, `n`, `w_max`); `eps_at` (`k`, `delta`, `credited`, `n`,
  `w_max`: ε as `str(Fraction)`); `sampling` (`eps`, `delta`, `work`, `tile_work`,
  `bits`); `work_to_json` (`work`, `tile_work`);
* `constants`: `PUBLIC`, `REGISTERED`, `LEGACY_REGISTERED`, `KINDS`, `GRID_BITS` and `STOP_FIXED`;
* the fixed length: `length_item` (`max_tokens`); `length_problem` (`item`, `max_tokens`);
* the entry row: `entry_row`, `entry_words` (`entry`, as `Entry.to_json`); `entry_from_row` (`row` hex),
  `entry_from_words` (`words`); `layout_problem`, `pass_tiles`, `tiles`, `entry_digest` (`row`); `entry_bc` (`row`,
  `salt`); `entry_leaf` (`domain` `{binding, count}`, `e`, `bc`); `tile_in` (`row`, `t`); `tile_at` (`rows`, `t`);
  `tile_read` (`row`, `i`, `j`);
* a call's record: `record_digest` (`record`); `record_problem` (`row`, `record`);
* the side trees: `side_row` (`words`, `index`); `side_index` (`row`); `side_domain` (`window`, `side`); `pad_node`
  (`domain`, `level`, `index`); `side_levels` (`domain`, `leaves`); `side_path` (`domain`, `leaves`, `index`);
* the window statement: `window_problem` (`rows`, `work`, `bound`, `root_h3`, `bits`, and `records`, `indices` or null);
* the public inputs: `session_cap` (`k_log`, `per_block`); `sessions` (`k`, `cap`); `instance_ts` (`registration`,
  `drawn`, `sigma`, `cap`); `public_inputs` (those and `domain_h3`); `layout_constants`.
* a live window: `live_epoch_reason` (`commitment`, the challenger's live coin commitment); `round_key` (`commitment`,
  `coin`, `round`, `root`).
* a drawn leaf's statements (`PouwWindow.Lanes`, held to `h1zy.statement_ports` and to the lowered circuits' port
  records): `n_digits` (`ks`); `lane_ports` (`r`, `c`, `k`, `lane`); `combine_ports` (`r`, `c`, `k`); a J10-R leaf's
  `j10r_lane_ports` (`r`, `c`, `k`, `n` the lane's chunks, `lane`, `chunk`) and `j10r_combine_ports` (`r`, `c`, `k`,
  `n`); `field_of` (those, `lane` null for the combine, `n` null or absent for v1's, and `path`); `wires` (those);
  `wire_of` (those and `wire`).
* a drawn leaf's known rows (`PouwWindow.KnownRows`, held to `h1_lanes`' `known_rows` by `tests/test_known_rows.py`):
  `known_rows` (`admit` 0/1 list, `salt` and the two `roots` 64-byte hex, `position` a u64, `lane` or null for the
  combine): each known port's rows' words.
* a drawn leaf's registered files (`PouwWindow.RegFile`, held to `leaf_lanes.registered_file` at
  `leaf_lanes.verifier_entries` by `tests/test_reg_file.py`): `registered_file` (`registration` and `produced` records,
  `r`, `c`, `k`, `lane` or null, `known` an object of ports); `leaf_own` (those records, the tile and
  statement, and the `file`): `true` where `pouw-leaf-own` accepts the file, a reject where it refuses; `full_draw`
  (`record`, a session record's text): `true` where `PouwWindow.Lanes.checkFull` accepts its draw, a reject where it
  refuses.

A domain `{binding, count}` is `CommitmentDomain(binding, OWNER, RangeIndexedDomain(0, count), hash="sha512")`; bytes are
hex, read as `bytes.fromhex` reads them.
-/

namespace PouwWindow.Difftest

open Lean (Json ToJson toJson)

def need {α : Type} (what : String) : Option α → Except String α
  | some v => .ok v
  | none => .error s!"the input's {what} is malformed"

def get (j : Json) (k : String) : Except String Json := need k (field? j k)
def int (j : Json) (k : String) : Except String Int := do need k (intOf? (← get j k))
def nat (j : Json) (k : String) : Except String Nat := do need k (natOf? (← get j k))

/-- A rational `[num, den]`, `den > 0`. -/
def rat (j : Json) (k : String) : Except String Rat := do
  let .arr #[n, d] ← get j k | throw s!"the input's {k} is not [num, den]"
  let n ← need k (intOf? n)
  let d ← need k (natOf? d)
  if d = 0 then throw s!"the input's {k} has den 0"
  pure (mkRat n d)

def work (j : Json) : Except String Work := do
  let w ← get j "work"
  pure ⟨← nat w "level", ← nat w "credited", ← nat w "n"⟩

def arr (j : Json) (k : String) : Except String (Array Json) := do
  let .arr a ← get j k | throw s!"the input's {k} is not a list"
  pure a

def str (j : Json) (k : String) : Except String String := do
  let .str s ← get j k | throw s!"the input's {k} is not a string"
  pure s

def hex (j : Json) (k : String) : Except String ByteArray := do PouwCommit.pyFromHex (← str j k)

def hexJ (b : ByteArray) : Json := Json.str (Flock.hexOf b)

def entryJson (e : Entry) : Json := Json.mkObj [
  ("kind", e.kind.name), ("lo", e.lo), ("hi", e.hi), ("n_a", e.nA), ("n_b", e.nB), ("root_a", hexJ e.rootA),
  ("root_b", hexJ e.rootB), ("root_h3", hexJ e.rootH3), ("tile0", e.tile0), ("cols", e.cols), ("m", e.m),
  ("record", hexJ e.record)]

/-- `Entry.from_json(x)`, held to what `row()` encodes. -/
def entryOf (x : Json) : Except String Entry := do
  let u (k : String) : Except String Nat := do
    let i ← int x k
    if i < 0 then throw "an entry's integers are u64s"
    pure i.toNat
  let kindName ← str x "kind"
  let (lo, hi, nA, nB) := (← u "lo", ← u "hi", ← u "n_a", ← u "n_b")
  let (rootA, rootB, rootH3, record) := (← hex x "root_a", ← hex x "root_b", ← hex x "root_h3", ← hex x "record")
  let (tile0, cols, m) := (← u "tile0", ← u "cols", ← u "m")
  let some kind := Kind.ofName? kindName | throw s!"an entry's kind is one of ('call', 'filler', 'none'), not {kindName}"
  let e : Entry := { kind, lo, hi, nA, nB, rootA, rootB, rootH3, record, tile0, cols, m }
  e.check
  pure e

def rowEntry (j : Json) : Except String Entry := do Entry.fromRow? (← hex j "row")

def domain (j : Json) : Except String PouwCommit.Domain := do
  let d ← get j "domain"
  PouwCommit.Domain.mk? .sha512 (← hex d "binding") owner (← nat d "count")

def natList (j : Json) (k : String) : Except String (List Nat) := do need k (nats? (← get j k))

def ints (j : Json) (k : String) : Except String (List Int) := do
  (← arr j k).toList.mapM fun w => need k (intOf? w)

def hexes (j : Json) (k : String) : Except String (Array ByteArray) := do
  (← arr j k).mapM fun x => match x with
    | .str s => PouwCommit.pyFromHex s
    | _ => throw s!"the input's {k} is not a list of hex"

def optList (j : Json) (k : String) : Except String (Option (List Json)) :=
  match field? j k with
  | none | some .null => pure none
  | some (.arr a) => pure (some a.toList)
  | some _ => throw s!"the input's {k} is not a list or null"

def levelsOf (j : Json) : Except String (PouwCommit.Domain × Array (Array ByteArray)) := do
  let d ← domain j
  pure (d, ← sideLevels? d (← hexes j "leaves"))

/-- A function of `(bits, n)`. -/
def ofN {α : Type} [ToJson α] (f : Nat → Int → Except String α) : _root_.Difftest.Fn := fun j => do
  pure (toJson (← f (← nat j "bits") (← int j "n")))

def leafTile (j : Json) : Except String Lanes.Tile := do pure ⟨← nat j "r", ← nat j "c", ← nat j "k"⟩

/-- A J10-R leaf's chunk of a lane (`lane` its index, `chunk` its chunk of `n`), or its combine (`lane` null). -/
def j10rPorts (j : Json) : Except String (List Lanes.Port) := do
  match field? j "lane" with
  | none | some .null => Lanes.combine2? (← leafTile j) (← nat j "n")
  | some _ => Lanes.chunk? (← leafTile j) (← nat j "n") (← nat j "lane") (← nat j "chunk")

/-- A lane's ports (`lane` its index), or the combine's (`lane` null); a J10-R leaf's (`j10rPorts`) where `n` is given. -/
def leafPorts (j : Json) : Except String (List Lanes.Port) := do
  match field? j "n" with
  | some .null | none =>
    match field? j "lane" with
    | none | some .null => Lanes.combine? (← leafTile j)
    | some _ => Lanes.lane? (← leafTile j) (← nat j "lane")
  | some _ => j10rPorts j

def sourceJson : Lanes.Source → Json
  | .read v r => Json.mkObj [("read", toJson v), ("row", toJson r)]
  | .known v => Json.mkObj [("known", toJson v)]

def leafFieldJson (f : Lanes.Field) : List (String × Json) :=
  [("bit", toJson f.bit), ("bits", toJson f.bits), ("word", toJson f.word), ("words", toJson f.words)]

def leafPortJson (p : Lanes.Port) : Json := Json.mkObj [("name", toJson p.name), ("source", sourceJson p.source),
  ("output", toJson p.output), ("bits", toJson p.bits), ("words", toJson p.words),
  ("fields", Json.arr (p.fields.map fun f => Json.mkObj (("path", toJson f.path) :: leafFieldJson f)).toArray)]

/-- `known_rows`: the statement's known rows' words by port, from its public inputs' bytes as `seeds.leaves` reads them. -/
def knownRowsJson (j : Json) : Except String Json := do
  let pos ← nat j "position"
  if 2 ^ 64 ≤ pos then throw s!"a position or a counter is a u64, not {pos}"
  let salt ← hex j "salt"
  let #[r0, r1] ← hexes j "roots" | throw "the input's roots are not two digests"
  if salt.size != 64 || r0.size != 64 || r1.size != 64 then throw "a digest is 64 bytes"
  let admit := (← natList j "admit").map (· != 0)
  let lane ← match field? j "lane" with
    | none | some .null => pure none
    | some _ => some <$> nat j "lane"
  let ks := Lanes.knownRows admit (Lanes.halvesNat salt) (Lanes.positionNat pos) (Lanes.rootsNat r0 r1)
    (fun _ => .empty) lane
  pure (Json.mkObj (ks.toList.map fun (port, rows) => (port, toJson (rows.map (·.1)))))

/-- The statement: lane `lane`, or the combine where `lane` is null. -/
def leafStmt (j : Json) : Except String (Option Nat) :=
  match field? j "lane" with
  | none | some .null => pure none
  | some _ => some <$> nat j "lane"

/-- The unit's placement: `place`'s `{value: [table, base]}`, each value it doesn't name its own table from row 0
(`Lanes.selfPlace`), every value so where `place` is absent or null. -/
def leafPlace (j : Json) : Except String (String → String × Nat) := do
  match field? j "place" with
  | none | some .null => pure Lanes.selfPlace
  | some (.obj kvs) =>
    let ps ← kvs.toArray.toList.mapM fun (v, e) => match e with
      | .arr #[.str t, b] => do
        let some b := (b.getNat?).toOption | throw s!"value {v}'s base row is not a natural number"
        pure (v, t, b)
      | _ => throw s!"value {v}'s placement is not [table, base]"
    pure fun v => (ps.lookup v).getD (Lanes.selfPlace v)
  | some _ => throw "the input's place is not an object"

/-- `registered_file`: statement `lane`'s `--registered` file at the roots of `registration` and `produced` and the
placement `place` (`leafPlace`), every value's rows `hm96-sha512/row/v2`, with the known ports `known`. -/
def registeredFileJson (j : Json) : Except String Json := do
  let T ← leafTile j
  let _ ← leafPorts j
  let rs ← Lanes.roots? (← get j "registration") (← get j "produced") T.lanes (2 * T.nw) Flock.HmRow.ROW_SCHEMA_V2
  let .obj known ← get j "known" | throw "the input's known is not an object"
  Lanes.fileOf? rs (← leafPlace j) T (← leafStmt j) known.toArray.toList

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("on_grid", fun j => do pure (toJson (onGridInt (← nat j "bits") (← int j "n")))),
  ("floor_point", ofN floorPoint?),
  ("pad", ofN pad?),
  ("level", ofN level?),
  ("bucket", fun j => do pure ((← bucket? (← nat j "bits") (← int j "n")).toJson)),
  ("point", fun j => do pure (toJson (← point? (← nat j "bits") (← int j "ell")))),
  ("work_at", fun j => do pure ((← Work.at? (← nat j "bits") (← int j "ell")).toJson)),
  ("levels", fun j => do pure (toJson (← levels? (← nat j "bits") (← int j "lo") (← int j "hi")))),
  ("k_prime", fun j => do
    pure (toJson (← kPrime (← rat j "eps") (← rat j "delta") (← int j "credited") (← int j "n") (← int j "w_max")))),
  ("eps_at", fun j => do
    pure (toJson (toString (← epsAt (← int j "k") (← rat j "delta") (← int j "credited") (← int j "n")
      (← int j "w_max"))))),
  ("sampling", fun j => do
    pure (← sampling (← rat j "eps") (← rat j "delta") (← work j) (← int j "tile_work") (← nat j "bits")).toJson),
  ("work_to_json", fun j => do pure ((← work j).itemJson (← int j "tile_work"))),
  ("constants", fun _ => pure <| Json.mkObj [("PUBLIC", Json.arr (publicItems.map PublicItem.toJson).toArray),
    ("REGISTERED", toJson registered), ("LEGACY_REGISTERED", toJson legacyRegistered), ("KINDS", toJson kinds),
    ("GRID_BITS", toJson gridBits), ("STOP_FIXED", toJson stopFixed)]),
  ("length_item", fun j => do pure (lengthItem (← natList j "max_tokens"))),
  ("length_problem", fun j => do pure (toJson (lengthProblem (← get j "item") (← natList j "max_tokens")))),
  ("entry_row", fun j => do pure (hexJ (← entryOf (← get j "entry")).row)),
  ("entry_words", fun j => do pure (toJson (← entryOf (← get j "entry")).words)),
  ("entry_from_row", fun j => do pure (entryJson (← rowEntry j))),
  ("entry_from_words", fun j => do pure (entryJson (← Entry.fromWords? (← ints j "words")))),
  ("layout_problem", fun j => do pure (toJson (← rowEntry j).layoutProblem)),
  ("pass_tiles", fun j => do pure (toJson (← rowEntry j).passTiles)),
  ("tiles", fun j => do pure (toJson (← rowEntry j).tiles)),
  ("entry_digest", fun j => do pure (hexJ (← rowEntry j).digest)),
  ("entry_bc", fun j => do pure (hexJ (← (← rowEntry j).bc? (← hex j "salt")))),
  ("entry_leaf", fun j => do pure (hexJ (← entryLeaf? (← domain j) (← nat j "e") (← hex j "bc")))),
  ("tile_in", fun j => do
    let (i, b) ← (← rowEntry j).tileIn? (← int j "t")
    pure (toJson [i, b])),
  ("tile_at", fun j => do
    let rows ← hexes j "rows"
    let (e, i, b) ← tileAt? (← rows.toList.mapM Entry.fromRow?) (← int j "t")
    pure (toJson [e, i, b])),
  ("tile_read", fun j => do
    let (p, w) ← (← rowEntry j).tileRead? (← int j "i") (← int j "j")
    pure (toJson [(p : Int), w])),
  ("record_digest", fun j => do pure (hexJ (← recordDigest (← get j "record")))),
  ("record_problem", fun j => do pure (toJson (← recordProblem (← rowEntry j) (← get j "record")))),
  ("side_row", fun j => do pure (toJson (← sideRow? (← ints j "words") (← int j "index")))),
  ("side_index", fun j => do pure (toJson (← sideIndex? (← ints j "row")))),
  ("side_domain", fun j => do pure (hexJ (← sideDomain? (← hex j "window") (← str j "side")).id)),
  ("pad_node", fun j => do pure (hexJ (padNode (← domain j) (← nat j "level") (← nat j "index")))),
  ("side_levels", fun j => do
    let (_, lv) ← levelsOf j
    pure (Json.arr (lv.map fun l => Json.arr (l.map hexJ)))),
  ("side_path", fun j => do
    let (d, lv) ← levelsOf j
    pure (Json.arr ((sidePath d lv (← nat j "index")).map hexJ).toArray)),
  ("window_problem", fun j => do
    let rows ← hexes j "rows"
    pure (toJson (← windowProblem rows.toList (← work j) (← int j "bound") (← hex j "root_h3") (← nat j "bits")
      (← optList j "records") (← optList j "indices")))),
  ("session_cap", fun j => do pure (toJson (← sessionCap? (← int j "k_log") (← int j "per_block")))),
  ("sessions", fun j => do pure (toJson (← sessions? (← int j "k") (← int j "cap")))),
  ("instance_ts", fun j => do
    pure (toJson (← instanceTs? (← get j "registration") (← arr j "drawn").toList (← get j "sigma") (← int j "cap")))),
  ("public_inputs", fun j => do
    pure (hexJ (← publicInputs? (← get j "registration") (← arr j "drawn").toList (← get j "sigma") (← int j "cap")
      (← hex j "domain_h3")))),
  ("live_epoch_reason", fun j => do
    pure (toJson (liveEpochReason (← SampledProofs.Challenger.LiveCommitment.ofJson (← get j "commitment"))))),
  ("round_key", fun j => do
    let c ← SampledProofs.Challenger.LiveCommitment.ofJson (← get j "commitment")
    pure (hexJ (← roundKey c (← hex j "coin") (← nat j "round") (← hex j "root")))),
  ("layout_constants", fun _ => pure <| Json.mkObj [("T_BITS", toJson tBits),
    ("DEPTHS", Json.mkObj (sides.map fun s => (s.name, toJson s.depth))), ("PORT", toJson port),
    ("ENTRY_SCHEMA", toJson entrySchema), ("ENTRY_BYTES", toJson entryBytes), ("TILE", toJson tile),
    ("ROW_STEP", toJson rowStep), ("SIDE_DOMAIN", toJson SIDE_DOMAIN), ("OWNER", toJson owner),
    ("SIDE_INDEX_WORDS", toJson sideIndexWords), ("PUBLIC_INPUTS_FORMAT", toJson PUBLIC_INPUTS_FORMAT),
    ("M_MAX", toJson mMax), ("PUBLIC_PORTS", Json.arr (publicPorts.map fun (p, w, sd, d) => Json.mkObj
      [("port", toJson p), ("words", toJson w), ("seed", toJson sd), ("derivation", toJson d)]).toArray)]),
  ("n_digits", fun j => do pure (toJson (Lanes.nDigits (← nat j "ks")))),
  ("known_rows", knownRowsJson),
  ("registered_file", registeredFileJson),
  ("leaf_own", fun j => do
    Lanes.leafOwn? (← get j "registration") (← get j "produced") (← get j "file") (← leafPlace j) (← leafTile j)
      (← leafStmt j) Flock.HmRow.ROW_SCHEMA_V2
    pure (Json.bool true)),
  ("full_draw", fun j => do
    Lanes.checkFull (← str j "record")
    pure (Json.bool true)),
  ("lane_ports", fun j => do
    pure (Json.arr ((← Lanes.lane? (← leafTile j) (← nat j "lane")).map leafPortJson).toArray)),
  ("combine_ports", fun j => do pure (Json.arr ((← Lanes.combine? (← leafTile j)).map leafPortJson).toArray)),
  ("j10r_lane_ports", fun j => do
    pure (Json.arr ((← Lanes.chunk? (← leafTile j) (← nat j "n") (← nat j "lane") (← nat j "chunk")).map
      leafPortJson).toArray)),
  ("j10r_combine_ports", fun j => do
    pure (Json.arr ((← Lanes.combine2? (← leafTile j) (← nat j "n")).map leafPortJson).toArray)),
  ("field_of", fun j => do
    let path ← str j "path"
    pure <| match Lanes.fieldOf? (← leafPorts j) path with
      | none => Json.null
      | some (p, f) => Json.mkObj ([("port", toJson p.name), ("source", sourceJson p.source)] ++ leafFieldJson f)),
  ("wires", fun j => do
    pure (Json.arr ((← Lanes.wires? (← leafPorts j)).map fun (pw : Lanes.Port × Nat) => Json.mkObj
      [("port", toJson pw.1.name), ("source", sourceJson pw.1.source), ("wire", toJson pw.2),
       ("wires", toJson pw.1.bits)]).toArray)),
  ("wire_of", fun j => do
    let ps ← leafPorts j
    let _ ← Lanes.wires? ps
    let w ← nat j "wire"
    pure <| match Lanes.wireOf? ps w with
      | none => Json.null
      | some (p, b) => Json.mkObj ([("port", toJson p.name), ("source", sourceJson p.source), ("bit", toJson b)] ++
          match p.fieldAt? b with
          | none => []
          | some f => [("field", toJson f.path), ("field_bit", toJson (b - f.bit))]))]

end PouwWindow.Difftest
