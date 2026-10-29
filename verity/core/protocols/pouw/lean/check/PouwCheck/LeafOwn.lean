import PouwCheck.Credit
import PouwWindow.RegFile

/-!
# A window unit's leaf statements: their tile and the placement of the values they read

A drawn unit of a window Program (tile `t` of call `c`, its K's template) is proved as its leaf's 33 statements
(`PouwWindow.Lanes.lanePorts`, `combinePorts`), each reading the leaf's rows of the values `a`, `b`, `zy` and
`receipt` and the partials (`PouwWindow.Lanes.regReads`). The window registers them as tables: the call's round's A
table of K, the weights' B of K, the call's z table and its receipt table. `placeOf` places each value in its table
from the unit's base row, from the window's committed descriptor and its round log alone, as `Credit.registeredFile`
places the tile statement's reads (`h1window.table_reads`):

* `a`: round s's A table of K `<prefix>a/<K>@<s>` from row `M·j + R·ti`, j the call's place among its round's calls of
  K and ti the tile's row block (the lane reads its A row at `rowOf l s` from there);
* `b`: the weights' `<prefix>b/<K>` from row `b0 + C·tj`, b0 the call's linear's first row in it and tj the tile's
  column block (`colOf l t` from there);
* at a chunked K (`Credit.chunks?` n > 1) `a`'s and `b`'s bases are leaf indices, n times those rows (`h1window.place`,
  the chunk lanes' base);
* `zy` and `receipt`: the call's `<prefix>zy@<c>` and `<prefix>receipt@<c>` from row `t`, one row a tile;
* the partials: their own table from row 0 (`PouwWindow.Lanes.selfPlace`), the leaf's produced value.

The tile's rounds are the round log's (`steps`, its calls the rounds' equal blocks), and its prefix and layout the
descriptor's Program (`Credit.programOf`, `Credit.layoutOf`, `Credit.prefixOf`). `leafOwn?` is
`PouwWindow.Lanes.leafOwn?` at that tile and placement, which `pouw-leaf-own-window` runs (`leafOwnCmd`). Its cost is
the descriptor's bytes and the window's calls.
-/

namespace PouwCheck.LeafOwn

open Lean (Json)

/-- **Unit `u`'s call and tile** in the window `layout` (`Credit.callWork`'s units, call after call): `(c, t)`. -/
def unitAt (layout : Json) (u : Nat) : Except String (Nat × Nat) := do
  let sizes := (Credit.callWork layout).map (·.1)
  let mut u0 := 0
  for (c, size) in (List.range sizes.length).zip sizes do
    if u < u0 + size then return (c, u - u0)
    u0 := u0 + size
  throw s!"the window has {u0} units, not a unit {u}"

/-- **Call `call`'s tile and unit `t`'s placement** in the window `layout` of prefix `pre` over `steps` rounds: the
call's K's leaf tile (`⟨R, C, K⟩`) and each value's table and base row (the module's list). -/
def placeOf (pre : String) (layout : Json) (steps call t : Nat) :
    Except String (PouwWindow.Lanes.Tile × (String → String × Nat)) := do
  let num := fun j => (nat? j).getD 0
  let (r, c) := (match get layout "tile" with
    | .arr a => (num (a.getD 0 .null), num (a.getD 1 .null))
    | _ => (0, 0))
  if r == 0 || c == 0 then throw "the window's tile is not [R, C] of positive R and C"
  let m := num (get layout "m")
  let rows := fun (j : Json) => (match j with
    | .arr a => a.toList.filterMap fun (x : Json) => (match x with
      | .arr b => some b.toList
      | _ => none)
    | _ => ([] : List (List Json)))
  let lins : List (String × Nat × Nat) := (rows (get layout "linears")).filterMap fun (x : List Json) => (match x with
    | [.str s, k, n] => some (s, num k, num n)
    | _ => none)
  let calls : List String := (rows (get layout "calls")).filterMap fun (x : List Json) => (match x with
    | [.str lin, _] => some lin
    | _ => none)
  let kOf := fun (lin : String) => (lins.lookup lin).map (·.1)
  let some lin := calls[call]? | throw s!"the window has {calls.length} calls, not a call {call}"
  let some (k, n) := lins.lookup lin | throw s!"call {call}'s linear {lin} is not one of the window's linears"
  let per := n / c
  unless t < (m / r) * per do throw s!"call {call} has {(m / r) * per} tiles, not a tile {t}"
  unless steps != 0 && calls.length % steps == 0 do
    throw s!"the window's {calls.length} calls are not its {steps} rounds' equal blocks, so no call has a round"
  let each := calls.length / steps
  let s := call / each
  let j := ((List.range' (s * each) (call - s * each)).filter fun c' => kOf (calls.getD c' "") == some k).length
  let b0 := ((lins.takeWhile (·.1 != lin)).filter (·.2.1 == k)).foldl (fun a x => a + x.2.2) 0
  let (ti, tj) := (t / per, t % per)
  let x := (Credit.chunks? k).getD 1
  let place := fun (v : String) =>
    if v == "a" then (s!"{pre}a/{k}@{s}", x * (m * j + r * ti))
    else if v == "b" then (s!"{pre}b/{k}", x * (b0 + c * tj))
    else if v == "zy" then (s!"{pre}zy@{call}", t)
    else if v == "receipt" then (s!"{pre}receipt@{call}", t)
    else PouwWindow.Lanes.selfPlace v
  pure (PouwWindow.Lanes.Tile.mk r c k, place)

/-- **The leaf verifier's check of unit `u`'s statement `k`'s file** in the window whose committed descriptor is
`descriptor` and whose round log is `rounds`: `PouwWindow.Lanes.leafOwn?` at the unit's tile and placement
(`placeOf`), against the window's accepted registration and the leaf's produced record, every value's rows
`hm96-sha512/row/v2`. -/
def leafOwn? (descriptor : ByteArray) (rounds registration produced file : Json) (u : Nat) (k : Option Nat) :
    Except String Unit := do
  let P ← (Credit.programOf descriptor).mapError (·.2)
  let layout ← (Credit.layoutOf P).mapError (·.2)
  let some pre := str? (Credit.prefixOf P) | throw "the window's descriptor names no PREFIX"
  let .arr log := rounds | throw "the round log is not a list"
  let (call, t) ← unitAt layout u
  let (T, place) ← placeOf pre layout log.size call t
  PouwWindow.Lanes.leafOwn? registration produced file place T k Flock.HmRow.ROW_SCHEMA_V2

/-- **`pouw-leaf-own-window DESCRIPTOR ROUNDS REGISTRATION PRODUCED REGISTERED UNIT (LANE | combine) SESSION...`**:
`leafOwn?` on the window's descriptor (its bytes), its round log, its registration, the leaf's produced record and the
statement's `--registered` file, at the window's unit `UNIT`, then `checkFull` on each session directory's record
(`SESSION/session.json`, at least one). It prints `OWN {"accepted", "why"?}` and exits 0 where it accepts, 1 where it
refuses and 2 on a usage error. -/
def leafOwnCmd (args : List String) : IO UInt32 := do
  let desc :: rounds :: reg :: prod :: file :: u :: stmt :: sess@(_ :: _) := args
    | PouwWindow.Lanes.usage
        "usage: pouw-leaf-own-window DESCRIPTOR ROUNDS REGISTRATION PRODUCED REGISTERED UNIT (LANE | combine) SESSION..."
  let some u := u.toNat? | PouwWindow.Lanes.usage "UNIT is a natural number"
  let some lane := PouwWindow.Lanes.stmtArg? stmt
    | PouwWindow.Lanes.usage s!"the statement is a lane's index or combine, not {stmt}"
  let d ← IO.FS.readBinFile desc
  let rj := PouwWindow.Lanes.readJson
  let (rounds, reg, prod, file) := (← rj rounds, ← rj reg, ← rj prod, ← rj file)
  let records ← PouwWindow.Lanes.sessionRecords sess
  PouwWindow.Lanes.printOwn do
    leafOwn? d rounds reg prod file u lane
    records.forM PouwWindow.Lanes.checkFull

end PouwCheck.LeafOwn
