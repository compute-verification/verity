import Difftest
import PouwWindow.Json
import PouwWindow.Statement
import PouwWindow.Inputs
import PouwWindow.Rounds

/-!
# The difftest's functions: the window's Lean behind `Difftest.serve`

One function per ported function of `window.py`, each on a JSON object of its arguments, for the shared harness
(`verity/core/difftest`): an output, or `.error` where Python raises `ValueError` (a reject). The port's
suite (`tests/test_difftest.py`) draws the inputs and holds each to Python.

* `on_grid`, `floor_point`, `pad`, `level` (`bits`, `n`); `point` (`bits`, `ell`); `work_at` (`bits`, `ell`, `n`);
  `credit` (`bits`, `real`, `n`); `levels` (`bits`, `lo`, `hi`);
* `k_prime` (`eps`, `delta` as `[num, den]`, `credited`, `n`, `w_max`); `sampling` (`eps`, `delta`, `work`, `tile_work`,
  `bits`); `work_to_json` (`work`, `tile_work`);
* `constants`: `PUBLIC`, `REGISTERED`, `KINDS` and `GRID_BITS`;
* the entry row: `entry_row`, `entry_words` (`entry`, as `Entry.to_json`); `entry_from_row` (`row` hex),
  `entry_from_words` (`words`); `layout_problem`, `pass_tiles`, `entry_digest` (`row`); `entry_bc` (`row`, `salt`);
  `entry_leaf` (`domain` `{binding, count}`, `e`, `bc`); `tile_read` (`row`, `i`, `j`);
* the shape: `shape_of`, `shape_tiles` (`shape`); `shape_problem`, `shape_entries` (`shape`, `bound`); `tile_at`
  (`shape`, `t`);
* the fixed length: `length_of`, `fixed_length` (`length`); `length_problem` (`length`, `shape`);
* a call's record: `record_digest` (`record`); `record_problem` (`row`, `record`);
* the side trees: `side_row` (`words`, `index`, `ok`, `admit`); `side_field` (`row`); `side_domain` (`window`, `side`);
  `pad_node` (`domain`, `level`, `index`); `side_levels` (`domain`, `leaves`); `side_path` (`domain`, `leaves`, `index`);
* the window statement: `window_problem` (`rows`, `work`, `bound`, `root_h3`, `shape`, `fields`, `bits`, and `records`
  or null);
* the public inputs: `session_cap` (`k_log`, `per_block`); `sessions` (`k`, `cap`); `instance_ts` (`registration`,
  `drawn`, `sigma`, `cap`); `public_inputs` (those and `domain_h3`); `layout_constants`.
* a live window: `live_epoch_reason` (`commitment`, the challenger's live coin commitment); `round_key` (`commitment`,
  `coin`, `round`, `root`).

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

def shapeArg (j : Json) : Except String (List Call) := do
  match shapeOf? (← get j "shape") with
  | some s => pure s
  | none => throw "a window's shape is a non-empty list of calls {m, n, tile0} of natural numbers"

def lengthArg (j : Json) : Except String Length := do
  match lengthOf? (← get j "length") with
  | some l => pure l
  | none => throw "a window's length is {ignore_eos, max_tokens}: a boolean and a non-empty list of natural numbers"

def levelsOf (j : Json) : Except String (PouwCommit.Domain × Array (Array ByteArray)) := do
  let d ← domain j
  pure (d, ← sideLevels? d (← hexes j "leaves"))

/-- A function of `(bits, n)`. -/
def ofN {α : Type} [ToJson α] (f : Nat → Int → Except String α) : _root_.Difftest.Fn := fun j => do
  pure (toJson (← f (← nat j "bits") (← int j "n")))

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("on_grid", fun j => do pure (toJson (onGridInt (← nat j "bits") (← int j "n")))),
  ("floor_point", ofN floorPoint?),
  ("pad", ofN pad?),
  ("level", ofN level?),
  ("credit", fun j => do pure ((← credit? (← nat j "bits") (← int j "real") (← int j "n")).toJson)),
  ("point", fun j => do pure (toJson (← point? (← nat j "bits") (← int j "ell")))),
  ("work_at", fun j => do pure ((← Work.at? (← nat j "bits") (← int j "ell") (← nat j "n")).toJson)),
  ("levels", fun j => do pure (toJson (← levels? (← nat j "bits") (← int j "lo") (← int j "hi")))),
  ("k_prime", fun j => do
    pure (toJson (← kPrime (← rat j "eps") (← rat j "delta") (← int j "credited") (← int j "n") (← int j "w_max")))),
  ("sampling", fun j => do
    pure (← sampling (← rat j "eps") (← rat j "delta") (← work j) (← int j "tile_work") (← nat j "bits")).toJson),
  ("work_to_json", fun j => do pure ((← work j).itemJson (← int j "tile_work"))),
  ("constants", fun _ => pure <| Json.mkObj [("PUBLIC", Json.arr (publicItems.map PublicItem.toJson).toArray),
    ("REGISTERED", toJson registered), ("KINDS", toJson kinds), ("GRID_BITS", toJson gridBits)]),
  ("entry_row", fun j => do pure (hexJ (← entryOf (← get j "entry")).row)),
  ("entry_words", fun j => do pure (toJson (← entryOf (← get j "entry")).words)),
  ("entry_from_row", fun j => do pure (entryJson (← rowEntry j))),
  ("entry_from_words", fun j => do pure (entryJson (← Entry.fromWords? (← ints j "words")))),
  ("layout_problem", fun j => do pure (toJson (← rowEntry j).layoutProblem)),
  ("pass_tiles", fun j => do pure (toJson (← rowEntry j).passTiles)),
  ("entry_digest", fun j => do pure (hexJ (← rowEntry j).digest)),
  ("entry_bc", fun j => do pure (hexJ (← (← rowEntry j).bc? (← hex j "salt")))),
  ("entry_leaf", fun j => do pure (hexJ (← entryLeaf? (← domain j) (← nat j "e") (← hex j "bc")))),
  ("shape_of", fun j => do pure (shapeJson (← shapeArg j))),
  ("shape_tiles", fun j => do pure (toJson (shapeTiles (← shapeArg j)))),
  ("shape_problem", fun j => do pure (toJson (shapeProblem (← shapeArg j) (← int j "bound")))),
  ("shape_entries", fun j => do pure (Json.arr ((shapeEntries (← shapeArg j) (← nat j "bound")).map entryJson).toArray)),
  ("tile_at", fun j => do
    let (e, i, b) ← tileAt? (← shapeArg j) (← int j "t")
    pure (toJson [e, i, b])),
  ("length_of", fun j => do pure (← lengthArg j).toJson),
  ("fixed_length", fun j => do pure (toJson (fixedLength (← get j "length")))),
  ("length_problem", fun j => do pure (toJson (lengthProblem (← lengthArg j) (← shapeArg j)))),
  ("tile_read", fun j => do
    let (p, w) ← (← rowEntry j).tileRead? (← int j "i") (← int j "j")
    pure (toJson [(p : Int), w])),
  ("record_digest", fun j => do pure (hexJ (← recordDigest (← get j "record")))),
  ("record_problem", fun j => do pure (toJson (← recordProblem (← rowEntry j) (← get j "record")))),
  ("side_row", fun j => do
    pure (toJson (← sideRow? (← ints j "words") (← int j "index") (← int j "ok") (← int j "admit")))),
  ("side_field", fun j => do
    let f ← sideField? (← ints j "row")
    pure (toJson [f.index, f.ok, f.admit, f.blank])),
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
    pure (toJson (← windowProblem rows.toList (← work j) (← int j "bound") (← hex j "root_h3") (← shapeArg j)
      (← get j "fields") (← nat j "bits") (← optList j "records")))),
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
      [("port", toJson p), ("words", toJson w), ("seed", toJson sd), ("derivation", toJson d)]).toArray)])]

end PouwWindow.Difftest
