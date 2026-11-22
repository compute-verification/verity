module

public import Flock.Hm96
public import Flock.HmRow
public import PouwCommit.Frame
public import PouwCommit.Bindings
public import Identity.Domains.Pouw

/-!
# The protocol's row and tile trees (`audit.rows_domain`, `Hm96Domain`, `commit_rows`, `transcript_domain`)

A scheme's trees are frame-v3 trees under its `commitment_hash`: `hm96-sha512` (h3's, `audit`'s default), which is
`frame-v3-sha512` with hm96-sha512 leaves, `sha512`, or `sha256` (h0's, and NCP's). Under `hm96-sha512` a committed value is its preimage followed by a 192-byte salt
`y`; `x` is the preimage's `sha512/row-seg/v1` digest (a row) or, in a tile tree (`digests`), the 64-byte preimage itself;
and the frame leaf is the one over `hm96.SHA512.tree_leaf(b ‖ c)` under the schema `hm96-sha512/row-seg/v1`. Elsewhere a
leaf is the frame's plain leaf over the value under the tree's schema (`verity/pouw/row/v1`, `verity/pouw/tile/v1`).

`commit_rows` takes the hm96 salts as data, one 192-byte salt a row in rank order, each fresh operating-system bytes
that no key derives. It builds a plain tree only for a caller that says its rows are public.
-/

@[expose] public section

namespace PouwCommit

open Flock

def ROW_SCHEMA : String := Domains.POUW_AUDIT_ROW
def TILE_SCHEMA : String := Domains.POUW_AUDIT_TILE
def HM96_SALT : Nat := 192

/-- A tree's frame and leaves: a plain frame-v3 tree, or `hm96-sha512`'s, of rows or (`digests`) of tile digests. -/
structure RowsDomain where
  frame : Domain
  hm96 : Bool
  digests : Bool

/-- `rows_domain(binding, owner, count, hash, digests)`, `hash` one of `hm96-sha512`, `sha512`, `sha256`; the Lean takes it
named, with no default. -/
def rowsDomain (binding : ByteArray) (owner : Int) (count : Nat) (hash : String) (digests : Bool) :
    Except String RowsDomain := do
  match hash with
  | "hm96-sha512" => pure ⟨← Domain.mk? .sha512 binding owner count, true, digests⟩
  | "sha256" => pure ⟨← Domain.mk? .sha256 binding owner count, false, digests⟩
  | "sha512" => pure ⟨← Domain.mk? .sha512 binding owner count, false, digests⟩
  | h => throw s!"the frame hash {h} is not one of sha256, sha512, hm96-sha512"

namespace RowsDomain

/-- The frame leaf over `value` at `rank` (`Hm96Domain.leaf` under `hm96-sha512`, which does not read `schema`). -/
def leaf (d : RowsDomain) (rank : Nat) (schema : String) (value : ByteArray) : Except String ByteArray := do
  if !d.hm96 then return d.frame.leaf rank schema value
  if value.size < HM96_SALT || (d.digests && value.size != 64 + HM96_SALT) then
    throw "an hm96-sha512 value is its preimage and a 192-byte salt"
  let pre := value.extract 0 (value.size - HM96_SALT)
  let y := value.extract (value.size - HM96_SALT) value.size
  let x := if d.digests then pre else HmRow.segDigest (8 * pre.size) pre
  return d.frame.leaf rank HmRow.ROW_SCHEMA_SEG (Hm96.Default512.leaf x y)

/-- The tree's root over its values in rank order, under `schema`. -/
def root (d : RowsDomain) (schema : String) (values : Array ByteArray) : Except String ByteArray := do
  let leaves ← (Array.range values.size).mapM fun i => d.leaf i schema values[i]!
  match d.frame.root leaves with
  | some r => pure r
  | none => throw s!"expected values for {d.frame.count} positions, got {values.size}"

/-- `verify_opening` of `value` at `rank` against `root`, under `schema`. -/
def verify (d : RowsDomain) (root : ByteArray) (rank : Nat) (schema : String) (value : ByteArray)
    (path : List ByteArray) : Except String Bool :=
  d.frame.verify root rank value path (d.leaf rank schema)

end RowsDomain

/-- The values of an `hm96-sha512` tree: row `i` followed by `salts[i]`, one 192-byte salt a row in rank order. -/
def salted (rows salts : Array ByteArray) : Except String (Array ByteArray) := do
  if salts.size != rows.size then throw s!"{rows.size} rows take {rows.size} salts, not {salts.size}"
  if salts.any (·.size != HM96_SALT) then throw "a salt is 192 bytes"
  pure ((Array.range rows.size).map fun i => rows[i]! ++ salts[i]!)

/-- `commit_rows(binding, owner, rows, hash, salts, digests, public).commitment.root`: under `hm96-sha512` row `i` is
followed by `salts[i]`. A plain `sha256` or `sha512` tree takes no salts, is not hiding, and is refused unless `isPublic`
(Python's `public`). -/
def commitRows (binding : ByteArray) (owner : Int) (rows : Array ByteArray) (hash : String)
    (salts : Option (Array ByteArray)) (digests : Bool) (isPublic : Bool) : Except String ByteArray := do
  let values ← if hash == "hm96-sha512" then
      match salts with
      | none => throw "an hm96-sha512 tree takes a salt a row"
      | some ss => salted rows ss
    else if salts.isSome then throw s!"a plain {hash} tree takes no salts"
    else if !isPublic then throw s!"a plain {hash} tree is not hiding"
    else pure rows
  (← rowsDomain binding owner rows.size hash digests).root ROW_SCHEMA values

/-- `transcript_domain(epoch, count, hash)`: the tile digests' tree, bound to the execution phase's salt and weights' root. -/
def transcriptDomain (salt weightsRoot : ByteArray) (count : Nat) (hash : String) : Except String RowsDomain :=
  rowsDomain (transcriptBinding salt weightsRoot) (-1) count hash true

/-- The transcript's root over the tiles' digests in order (`Prover.prove`): under `hm96-sha512` digest `i` is followed by
`salts[i]`; a plain tree takes no salts. -/
def transcriptRoot (salt weightsRoot : ByteArray) (digests : Array ByteArray) (hash : String)
    (salts : Option (Array ByteArray)) : Except String ByteArray := do
  let values ← if hash == "hm96-sha512" then
      match salts with
      | none => throw "an hm96-sha512 tree takes a salt a row"
      | some ss => salted digests ss
    else if salts.isSome then throw s!"a plain {hash} tree takes no salts"
    else pure digests
  (← transcriptDomain salt weightsRoot digests.size hash).root TILE_SCHEMA values

end PouwCommit
