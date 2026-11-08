/-
The link table's V* part (`RecLink{UB, PB}`, `VStar.Stmt.link`): `VStar.linkParams` reads `rec-link/u<UB>-p<PB>` back
only as `linkName` writes it; `RecVStar.linkStatements` lists the part over the table's rows only where the inner shape
names a table, at a shape the link unit takes and a row count that is some table's `grid(L)`; the inner shape's `links`
reads back from its JSON; an unlisted link part names the missing table; `RecLinkPart.checkShape` holds the statement's
table (`--rec-links`) to the inner shape's; and `RecLinkPart.check` passes every other unit, refuses a `rec-link/` name
`linkParams` does not read, and refuses a link part without the table, the registered reads or the inner statement's
files, or whose `trow` is not the table at its root, rows, domain and fixed rows. `RecLinkPart.checkUnit` refuses a unit
that is not V*'s net (`checkUnit_refuses`, over every circuit), and `RecLinkPart.outsOk` passes V*'s link layout and
refuses it laid out any other way. That V*'s own file passes `checkParsed`, `checkUnit` and `outsOk` is checked by
`flock-verify` on a link part (`verity/ml/flock/tests/test_lean_rec_links_verify.py`): the interpreter can't build that
file, or V*'s net, in a module's time.
-/
module

public import Flock.RecLinkPart
public import Flock.Sizes
public import Std.Data.HashMap
public import Std.Data.HashSet
public import Std.Data.TreeMap.Raw.Lemmas
meta import Flock.RecLinkPart
meta import Flock.Sizes
meta import Std.Data.HashMap
meta import Std.Data.HashSet
meta import Std.Data.TreeMap.Raw.Lemmas

@[expose] public section

namespace Flock.Test.RecLinkPart

open Lean Flock Flock.RecVStar

#guard VStar.linkName 4 3 == "rec-link/u4-p3"
#guard VStar.linkParams "rec-link/u4-p3" == some (4, 3)
#guard VStar.linkParams "rec-link/u0-p0" == some (0, 0)
#guard VStar.linkParams "rec-link/u04-p3" == none
#guard VStar.linkParams "rec-link/u4-p3-p2" == none
#guard VStar.linkParams "rec-link/u4" == none
#guard VStar.linkParams "rec-cap/n128" == none
#guard isVStar "rec-link/u4-p3"

#guard (VStar.Stmt.link 4 3 18).name == "rec-link/u4-p3"
#guard (VStar.Stmt.link 4 3 18).instances == 18 && (VStar.Stmt.link 4 3 18).dir == "links"
#guard (VStar.Stmt.link 4 3 18).sub.inputs == ["trow", "xyr", "raux", "xyw", "waux"]
#guard VStar.linkRows 4 3 == [64, 128, VStar.auxWords 7, 128, VStar.auxWords 7]
#guard (VStar.linkRows 4 3).all (fun w => 2 * w % 128 == 0)
#guard (VStar.Stmt.link 4 3 18).sub.outputs == ["rin", "rout"]
#guard ownPorts (.link 4 3 18) == [] && rowPorts (.link 4 3 18) == [] && accPorts (.link 4 3 18) == []

def inner : Inner := { m := 25, kLog := 22, claims := 8, regions := 3 }

def linked (rows : Nat) (UB PB : Nat := 6) : Inner := { inner with links := some { UB, PB, rows } }

def listed (i : Inner) : Option (List (String × Nat)) :=
  ((linkStatements i).map fun ss => ss.toList.map fun s => (s.name, s.instances)).toOption

def refused (i : Inner) : Option String :=
  match linkStatements i with
  | .error e => some e
  | .ok _ => none

-- No table, no part; a table of `grid(L)` rows, one part over them; 17 rows is no table's grid (`grid 17 = 18`).
#guard listed inner == some []
#guard listed (linked 18) == some [("rec-link/u6-p6", 18)]
#guard listed (linked 1) == some [("rec-link/u6-p6", 1)]
#guard refused (linked 17) == some "V*'s list: the link table's 17 rows are no table's grid(L)"
#guard refused (linked 0) == some "V*'s list: the link table's 0 rows are no table's grid(L)"
#guard ((refused (linked 18 (UB := 31))).map (·.startsWith "RecLink's UB is 0 to 30")) == some true

#guard (Inner.ofJson (linked 18).toJson).toOption == some (linked 18)
-- the fewest bits that number n units or rows (`linkShapeOf`)
#guard [0, 1, 2, 3, 4, 5, 8, 9, 1024, 1025].map bitsFor == [0, 0, 1, 2, 2, 3, 3, 4, 10, 11]
#guard (Inner.ofJson inner.toJson).toOption == some inner
#guard (match Inner.ofJson (Json.setObjVal! inner.toJson LINKS_KEY (Json.mkObj [("UB", toJson (6 : Nat))])) with
  | .error e => e == "rec-inner: links is not the link table's shape (UB, PB, rows)"
  | .ok _ => false)

#guard unlisted inner "rec-link/u6-p6" #[] ==
  "rec-link/u6-p6: the link table's part, and the verifier's inner shape names no link table (links)"

def T : ByteArray := ⟨#[9]⟩

def shape (i : Option Inner) (l : Option (ByteArray × Nat)) : Option String :=
  match RecLinkPart.checkShape i l with
  | .error e => some e
  | .ok () => none

#guard shape none none == none
#guard shape (some inner) none == none
#guard shape (some (linked 18)) (some (T, 18)) == none
#guard shape (some (linked 18)) (some (T, 32)) ==
  some "rec-links: the statement's link table has 32 rows (--rec-links), and the verifier's inner shape's 18"
#guard shape (some (linked 18)) none ==
  some "rec-links: the verifier's inner shape names a link table of 18 rows, and the statement carries none (--rec-links ROOT:N)"
#guard shape (some inner) (some (T, 18)) ==
  some "rec-links: the statement carries a link table of 18 rows (--rec-links), and the verifier holds no inner shape naming one (--rec-inner F with links)"
#guard shape none (some (T, 18)) == shape (some inner) (some (T, 18))

def circuitOf (name : String) : Circuit :=
  { (default : Circuit) with
    nets := #[(name, { (default : Net) with name := name })]
    metaJ := Json.mkObj [("program_sha512", Json.str "p")] }

/-- A public file whose units header labels its instances `indices`. -/
def pubOf (indices : List Nat) : Public :=
  { (default : Public) with
    n := indices.length, header := Json.mkObj [("units", Json.mkObj [("indices", toJson indices)])] }

def files : RecCoins.Source :=
  .files { circuit := ByteArray.empty, pub := ByteArray.empty, partition := Json.null, record := "" }

def checked (name : String) (own : Option (Array Registered.Port)) (src : Option RecCoins.Source)
    (links : Option (ByteArray × Nat)) (pub : Public := pubOf (List.range 18)) : Option String :=
  match RecLinkPart.check { (default : Stmt) with c := circuitOf name, pub } own src links with
  | .error e => some e
  | .ok () => none

def dom : ByteArray := match Registered.domain "p" RecLinkPart.LINKS_VALUE VStar.LINK_ROW_WORDS 18 16 with
  | .ok d => d
  | .error _ => ByteArray.empty

def all : Array (Option Nat) := (Array.range 18).map some

def trow (value : String := RecLinkPart.LINKS_VALUE) (root : ByteArray := T) (leaves : Nat := 18)
    (domain : ByteArray := dom) (reads : Array (Option Nat) := all) : Registered.Port :=
  { name := "trow", value, root, leaves, words := some VStar.LINK_ROW_WORDS, wordBits := 16, domain, readsRel := reads }

def link : String := "rec-link/u6-p6"

-- Every other unit passes, whatever the verifier holds; a `rec-link/` name `linkParams` does not read is refused.
#guard checked "rec-cap/n128" none none none == none
#guard checked "rec-link/u06-p6" none none none ==
  some "rec-links: rec-link/u06-p6: a rec-link/ unit not named rec-link/u<UB>-p<PB>"
#guard checked link (some #[trow]) (some files) none ==
  some "rec-links: rec-link/u6-p6: the statement carries no link table (--rec-links ROOT:N), whose rows this part checks"
#guard checked link none (some files) (some (T, 18)) ==
  some "rec-links: rec-link/u6-p6: the verifier holds no registered reads (--registered): the table's rows would be the prover's"
#guard checked link (some #[trow]) none (some (T, 18)) ==
  some ("rec-links: rec-link/u6-p6: the verifier holds no inner statement (--rec-inner-circuit, --rec-inner-public, " ++
    "--rec-inner-partition, --rec-inner-session), whose rows the table links")

-- `trow` reads the statement's table: its value, root, rows, domain at the part's program, and row `i` in instance `i`.
#guard checked link (some #[]) (some files) (some (T, 18)) ==
  some "rec-links: rec-link/u6-p6: the verifier's registered reads hold no port trow: the table's rows would be the prover's"
#guard checked link (some #[trow (value := "rec-acc")]) (some files) (some (T, 18)) ==
  some "rec-links: rec-link/u6-p6: port trow reads rec-acc, not the link table rec-links"
#guard checked link (some #[trow (root := ⟨#[2]⟩)]) (some files) (some (T, 18)) ==
  some "rec-links: rec-link/u6-p6: rec-links: registered at root 02, not the statement's table 09 (--rec-links)"
#guard checked link (some #[trow (leaves := 32)]) (some files) (some (T, 18)) ==
  some ("rec-links: rec-link/u6-p6: rec-links: 32 hm96-sha512/row/v1 rows of 64 words registered, the table's 18 " ++
    "hm96-sha512/row/v1 rows of 64")
#guard checked link (some #[{ trow (root := T) with schema := "hm96-sha512/row/v2" }]) (some files) (some (T, 18)) ==
  some ("rec-links: rec-link/u6-p6: rec-links: 18 hm96-sha512/row/v2 rows of 64 words registered, the table's 18 " ++
    "hm96-sha512/row/v1 rows of 64")
#guard checked link (some #[trow (domain := ⟨#[7]⟩)]) (some files) (some (T, 18)) ==
  some "rec-links: rec-link/u6-p6: rec-links: the registered domain is not the table's at the part's program p"
#guard checked link (some #[trow (reads := all.set! 2 (some 3))]) (some files) (some (T, 18)) ==
  some "rec-links: rec-link/u6-p6: rec-links: the reads file does not read row i of the table in instance i"
#guard checked link (some #[trow]) (some files) (some (T, 18)) (pub := pubOf [1, 0]) ==
  some ("rec-links: rec-link/u6-p6: its instances are labelled #[1, 0], not 0 to 2 - 1 in order, so instance i " ++
    "would not read row i")
-- Past the table, the inner statement's rows: here its files are empty, so its setup refuses.
#guard ((checked link (some #[trow]) (some files) (some (T, 18))).map
  (·.startsWith "rec-coins: rec-link/u6-p6: the inner statement: ")) == some true

/-- **A unit that is not V\*'s net is refused**, naming its first difference; `RecLinkPart.checkUnit_ok` is the other
way. Stated over every circuit, as no #guard can afford V\*'s net (SHA-512 compressions at any `UB`, `PB`). -/
theorem checkUnit_refuses {c : Circuit} {UB PB : Nat} {g : Net} (hg : RecLinkPart.unitNet UB PB = .ok g)
    (h : c.unit ≠ g) :
    RecLinkPart.checkUnit c UB PB =
      .error s!"rec-links: {c.unit.name}: not the verifier's unit ({HmNets.firstDiff c.unit g})" := by
  unfold RecLinkPart.checkUnit
  rw [hg]
  dsimp only
  rw [ite_eq_right h]
  rfl

/-- V\*'s link layout at `UB = PB = 0`, from the statement's port names and `VStar.linkRows` rather than V\*'s file (which
the interpreter can't afford): its input rows, then `rin` and `rout`, one output row each, one copy returning words
`0..127` in order. -/
def laid : Circuit :=
  let s := VStar.Stmt.link 0 0 1
  let outs := s.sub.outputs.map (·, VStar.LINK_OUT_WORDS)
  { circuitOf "rec-link/u0-p0" with
    ports := (s.sub.inputs.zip (VStar.linkRows 0 0) ++ outs).toArray.map
      fun (name, words) => { name, words, chunks := 2 * words / 128, nb := 0 }
    outPorts := outs.toArray, outRowPorts := #[1, 1], leavesOut := #[Array.range 128] }

-- Its outputs are V*'s layout (`outsOk`): that layout has it, and one whose output ports are swapped, whose output rows
-- start a row early, or whose copy returns its words out of order does not.
#guard RecLinkPart.outsOk laid
#guard !RecLinkPart.outsOk { laid with outPorts := #[("rout", 64), ("rin", 64)] }
#guard !RecLinkPart.outsOk { laid with outRowPorts := laid.outRowPorts.push 1 }
#guard !RecLinkPart.outsOk { laid with leavesOut := #[(Array.range 128).reverse] }

end Flock.Test.RecLinkPart
