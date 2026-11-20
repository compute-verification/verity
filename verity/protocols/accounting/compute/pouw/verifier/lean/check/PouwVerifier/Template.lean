module

public import Circuits.DefinitionDigest
public import Flock.Program
public import PouwVerifier.Chunks
public import PouwVerifier.Json

/-!
# The H1 window's template (`credit._tile`, `credit.TILE_PINS`)

`tileOf` reads K's template of an H1 window's descriptor: `h1window.template`'s form, and its closure
(`Flock.DefinitionDigest.closureDigest`, `codec.closure_digest`: its body and every Definition under it) the digest
`TILE_PINS` pins for its key, so the template's name over another body is no template. The pins are the
`h1_tiles.json` `credit.py` reads, which `test_h1_tile_pins.py` writes from the registry. `Credit.layoutOf` reads it.
-/

@[expose] public section

namespace PouwVerifier.Credit

open Lean (Json)

def TILE_FN : String := "PouwH1WindowTile_v1"
def SHARED : List String := ["zy", "salt", "position", "roots"]
/-- `credit.SHARED_J10R`: a J10-R template's, its call's first position and its unit's blocks after `SHARED`. -/
def J10R : List String := ["call", "blocks"]

/-- `credit.TILE_PINS`: each pinned instance of `h1window.template` (its key, `TILE_FN{UNIT=…,K=k}`) with its closure
digest. -/
def TILE_PINS : List (String × String) :=
  match Json.parse (include_str "../h1_tiles.json") with
  | .ok (.obj kvs) => kvs.toArray.toList.filterMap fun (key, d) => (str? d).map (key, ·)
  | _ => []

/-- `credit._pinned`: the definition keyed `key` in the descriptor `j` has its key's pin as its closure digest. -/
def pinned (j : Json) (key : String) : Bool :=
  match TILE_PINS.lookup key, Flock.DefinitionDigest.closureDigest j key with
  | some pin, .ok h => h == pin
  | _, _ => false

/-- How deep `bitsOf` reads a type: deeper than Python's recursion limit reads. -/
def TYPE_DEPTH : Nat := 1000

/-- `credit._bits`: a type's bits, its leaves' widths: `Value<w>` w, an Array n of its element's, a Tuple or Record its
fields'; zero past `fuel` levels. -/
def bitsOf : Nat → Flock.Program.Ty → Nat
  | 0, _ => 0
  | _ + 1, .value w => w
  | fuel + 1, .array n t => n * bitsOf fuel t
  | fuel + 1, .tuple fs => fs.foldl (fun s t => s + bitsOf fuel t) 0
  | fuel + 1, .record fs => fs.foldl (fun s t => s + bitsOf fuel t.2) 0

/-- The definition the decoded Program `P` keys by `id`, if it has one. -/
def defn? (P : Flock.Program.Prog) (id : String) : Option Flock.Program.Defn :=
  P.index[id]?.bind (P.defs[·]?)

/-- `credit._tile`: the R × C of K = `k`'s template `d` (its decoded definition in the descriptor `j`) and whether it is
J10-R's (its parameters end in `SHARED` then `J10R`), or none when it is not `h1window.template`'s: a composite
`TILE_FN` keyed `TILE_FN{UNIT=…,K=k}` (its binding record, which the decoder holds to its bindings), whose A and B rows
are n·R and n·C chunk rows of 8·K/n + 32 bits each, n = `Chunks.chunks` K, its closure in `j` pinned (`pinned`). -/
def tileOf (j : Json) (d : Flock.Program.Defn) (k : Nat) : Option (Nat × Nat × Bool) := do
  let x ← (Chunks.chunks k).toOption
  guard (!d.prim && d.id == TILE_FN)
  guard (pinned j d.key)
  let [("UNIT", _), ("K", kb)] := (Flock.Program.recordPairs d.key).toList | none
  guard (kb == toString k)
  let names := d.params.toList.map (·.1)
  let today := [s!"a_k{k}", s!"b_k{k}"] ++ SHARED
  let j10r := names == today ++ J10R
  guard (names == today || j10r)
  let rows ← (d.params.toList.take 2).mapM fun (_, t) => (match t with
    | .array n e => if 1 ≤ n && n % x == 0 && bitsOf TYPE_DEPTH e == 8 * (k / x) + 32 then some n else none
    | _ => none)
  match rows with
  | [r, c] => some (r / x, c / x, j10r)
  | _ => none

end PouwVerifier.Credit
