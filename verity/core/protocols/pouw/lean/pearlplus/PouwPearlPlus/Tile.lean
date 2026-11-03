import PouwPearlPlus.Rule
import PouwPearlPlus.Run
import PouwCommit.Rows

/-!
# One drawn tile (`clear_audit.Verifier.check_tile`, `clear_work.check_opened`, `verify_run.check_y`)

`checkTile p wn tid o` is the verdict on tile `tid` of pass `p` from the prover's openings `o`, in the Python's order:

1. the openings cover the tile (A's rows and B's columns at the tile's positions, in order);
2. each A row opens against the call's activation root (`activation opening`), each B row against its weight's root
   (`weight opening`), and the transcript leaf at the tile's id against the pass's transcript root (`transcript opening`),
   all `hm96-sha512` trees (`PouwCommit.Rows`);
3. the tile's words from the opened values (`Words.encode`, `Words.checked`; a non-finite value is `inadmissible inputs`),
   and their leaf is the opened leaf (`checked values`);
4. the tile rule by the record (`Rule.checkOpened`);
5. on a call that commits y, the tile's real rows open against the call's y root at their positions (`y opening`), and
   each word is the unscale of the tile's U (`Rule.yReason`).

`wn` is the fixed operand's noise (`Words.weightNoise`), once per pass and weight; its failure is the tile's
`inadmissible inputs`, where Python's `encode` meets it.
-/

namespace PouwCheck.PearlPlus

open PouwCommit Words

/-- One opened position of a tree: the value (its preimage and 192-byte hm96 salt) and the sibling path, leaf first. -/
structure Opening where
  position : Nat
  value : ByteArray
  path : List ByteArray
  deriving Inhabited

/-- A tile's openings (`clear_audit.TileOpening`, and `Opened.open_y`'s y rows: the tile's rows below the call's real rows,
empty on a call without y). -/
structure TileOpening where
  a : List Opening
  b : List Opening
  leaf : Opening
  y : List Opening := []
  deriving Inhabited

def HM96 : String := "hm96-sha512"
def Y_ROWS_TAG : String := "verity/pouw/pearl-plus/y-rows/v1"

/-- `served_y.y_binding(salt, index)`. -/
def yBinding (salt : ByteArray) (index : Nat) : ByteArray := tagged Y_ROWS_TAG [.bytes salt, .int index]

/-- `verify_opening`, a raise (`ProtocolError`) refusing. -/
def opens (d : Except String RowsDomain) (root : ByteArray) (schema : String) (o : Opening) : Bool :=
  match d with
  | .ok d => match d.verify root o.position schema o.value o.path with
    | .ok b => b
    | .error _ => false
  | .error _ => false

/-- `opened_value`: an hm96 value less its salt. -/
def openedValue (v : ByteArray) : ByteArray := v.extract 0 (v.size - HM96_SALT)

def words32 (b : ByteArray) : Array Nat :=
  (Array.range (b.size / 4)).map fun i =>
    b[4 * i]!.toNat ||| (b[4 * i + 1]!.toNat <<< 8) ||| (b[4 * i + 2]!.toNat <<< 16) ||| (b[4 * i + 3]!.toNat <<< 24)

/-- The verdict on tile `tid` of pass `p`: `none` (it passes) or the reason it fails. -/
def checkTile (p : Pass) (wn : Except String WeightNoise) (tid : Nat) (o : TileOpening) : Option String := Id.run do
  let some (pos, rows, cols) := p.tile? tid | return some "tile does not match the verifier's own enumeration"
  let r := p.records[pos]!
  let u := r.matmul
  let rootA := p.rootsA[pos]!
  let rootB := (p.weightRoots.lookup u.weight).getD ByteArray.empty
  let da := rowsDomain (activationBinding p.salt u.index) u.index u.m HM96 false
  let db := rowsDomain (weightBinding u.weight) (-2) u.n HM96 false
  if o.a.map (·.position) != rows || o.b.map (·.position) != cols then return some "openings do not cover the tile"
  if !o.a.all (opens da rootA ROW_SCHEMA) then return some "activation opening"
  if !o.b.all (opens db rootB ROW_SCHEMA) then return some "weight opening"
  if o.leaf.position != tid || !opens (transcriptDomain p.salt p.weightsRoot p.count HM96) p.transcriptRoot TILE_SCHEMA o.leaf
  then return some "transcript opening"
  let work : Reject (Ops × Checked) := do
    let wn ← wn
    let ops ← encode p.salt rootB wn u.index u.k rows (o.a.map (openedValue ·.value)) cols (o.b.map (openedValue ·.value))
    pure (ops, ← checked ops)
  let .ok (ops, ch) := work | return some "inadmissible inputs"
  if leaf ch != openedValue o.leaf.value then return some "checked values"
  if let some why := Rule.checkOpened p.scheme r rows cols ops then return some why
  let some yRoot := p.yRoots[pos]! | return none
  let real := rows.filter (· < r.realRows)
  let dy := rowsDomain (yBinding p.salt u.index) u.index r.realRows HM96 false
  if o.y.map (·.position) != real || !o.y.all (opens dy yRoot ROW_SCHEMA) then return some "y opening"
  Rule.yReason rows cols ops ch (o.y.map fun y => words32 (openedValue y.value))

end PouwCheck.PearlPlus
