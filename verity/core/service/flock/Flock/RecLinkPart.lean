import Flock.RecCoins
import Flock.RecLinkRef

/-!
# The link table's V* part (`verity_flock.rec_link`)

A recursive statement that carries the inner population's link table (`--rec-links ROOT:N`, `RecLinkTree.linksArg`) has
one more V* part: `RecLink{UB, PB}` over the table's `N` rows (`VStar.Stmt.link`), which V*'s list holds just before the
level-0 cap when the verifier's inner shape names the table (`RecVStar.Inner.links`). `check` holds that part's statement
to the table and to the inner statement:

* its `trow` reads the table: the value `rec-links`, `N` `hm96-sha512/row/v1` rows of 64 words under its registered
  domain at the part's program (`Registered.domain`), whose root is the statement's `ROOT`, instance `i` reading row `i`
  (`checkTable`), its instances labelled `0` to `N − 1` in order (`Registered.unitIndices`);
* its `rin` and `rout` are rows the verifier registers itself (`RecCoins.checkValue`): every instance's `rin` is the
  input rows' tree's root and its `rout` the output rows' (`RecLinkRef.expected` of `RecLinkTree.rowsRoot`), each tree
  over the inner statement's own commit strings (`innerRows`).

An instance's outputs equal to those roots name a reader row and a writer row of the inner statement that commit one
digest, or a SHA-512 collision (`Proofs.Flock.Verifier.RecLinks.link_binding`). Every other unit passes. The trees are
built once per part, one hash a row a level: linear in the inner statement's public file, which setup already reads.
`checkShape` holds the statement's table to the verifier's inner shape, so the list has the part exactly when the
statement carries the table.

`verify` and `statement` still refuse `--rec-links` before reading anything (`RecLinks.refuseLinks`) until the
recursion's guarantee covers a statement carrying the table; these checks run behind that refusal.
-/

namespace Flock
namespace RecLinkPart

def LINKS_VALUE : String := "rec-links"
def IN_VALUE : String := "rec-link-in"
def OUT_VALUE : String := "rec-link-out"

/-- **The statement's table is the one the verifier's inner shape names**: a link table (`--rec-links ROOT:N`) only
with an inner shape whose `links` names one of `N` rows, and an inner shape naming one only with the table. -/
def checkShape (inner : Option RecVStar.Inner) (links : Option (ByteArray × Nat)) : V Unit :=
  match inner.bind (·.links), links with
  | none, none => pure ()
  | some l, some (_, N) =>
    if l.rows == N then pure ()
    else throw s!"rec-links: the statement's link table has {N} rows (--rec-links), and the verifier's inner shape's {l.rows}"
  | some l, none =>
    throw s!"rec-links: the verifier's inner shape names a link table of {l.rows} rows, and the statement carries none (--rec-links ROOT:N)"
  | none, some (_, N) =>
    throw (s!"rec-links: the statement carries a link table of {N} rows (--rec-links), and the verifier holds no inner " ++
      "shape naming one (--rec-inner F with links)")

/-- The inner statement's rows, each `((unit, row), b ‖ c)` in increasing order: unit `u`'s input row `p` is its commit
string at port `p` (`p < HmOut.nIn`), its output row `o` at port `HmOut.nIn + o`. -/
def innerRows (st : Stmt) : List ((Nat × Nat) × ByteArray) × List ((Nat × Nat) × ByteArray) :=
  let np := st.c.ports.size
  let nIn := HmOut.nIn st.c
  let units := List.range st.pub.n
  (units.flatMap fun u => (List.range nIn).map fun p => ((u, p), st.pub.digest u p np),
    units.flatMap fun u => (List.range st.c.outRowPorts.size).map fun o => ((u, o), st.pub.digest u (nIn + o) np))

/-- **The part's `trow` reads the statement's table** (root `T`, `N` rows): the verifier's reads file names it under
`LINKS_VALUE`, at `T`, `N` `hm96-sha512/row/v1` rows of `VStar.LINK_ROW_WORDS` words, the table's registered domain at
the part's program, instance `i` reading row `i`, and the part's table labels its instances `0` to `n − 1` in order. -/
def checkTable (c : Circuit) (pub : Public) (own : Array Registered.Port) (T : ByteArray) (N : Nat) : V Unit := do
  let unit := c.unit.name
  let some reg := own.find? (·.name == "trow")
    | throw s!"rec-links: {unit}: the verifier's registered reads hold no port trow: the table's rows would be the prover's"
  if reg.value != LINKS_VALUE then
    throw s!"rec-links: {unit}: port trow reads {reg.value}, not the link table {LINKS_VALUE}"
  if reg.root != T then
    throw s!"rec-links: {unit}: {LINKS_VALUE}: registered at root {hexOf reg.root}, not the statement's table {hexOf T} (--rec-links)"
  if reg.leaves != N || reg.words != some VStar.LINK_ROW_WORDS || reg.wordBits != 16 || reg.schema != HmRow.ROW_SCHEMA then
    throw (s!"rec-links: {unit}: {LINKS_VALUE}: {reg.leaves} {reg.schema} rows of {reg.words.getD 0} words registered, " ++
      s!"the table's {N} {HmRow.ROW_SCHEMA} rows of {VStar.LINK_ROW_WORDS}")
  let .ok prog := c.metaJ.getObjValAs? String "program_sha512" | throw s!"rec-links: {unit}: META names no program_sha512"
  if reg.domain != (← Registered.domain prog LINKS_VALUE VStar.LINK_ROW_WORDS N 16) then
    throw s!"rec-links: {unit}: {LINKS_VALUE}: the registered domain is not the table's at the part's program {prog}"
  if reg.reads != (Array.range N).map some then
    throw s!"rec-links: {unit}: {LINKS_VALUE}: the reads file does not read row i of the table in instance i"
  let idx ← Registered.unitIndices pub
  if idx != Array.range pub.n then
    throw (s!"rec-links: {unit}: its instances are labelled {idx}, not 0 to {pub.n} - 1 in order, so instance i " ++
      "would not read row i")

/-- **The link part against the table and the inner statement**: a `rec-link/u<UB>-p<PB>` unit (`VStar.linkParams`;
another `rec-link/` name is refused) is refused unless the statement carries the table (`links`, `--rec-links`), the
verifier holds its registered reads and the inner statement's files, its `trow` reads the table (`checkTable`), and its
`rin` and `rout` are the roots of the inner statement's input and output rows' trees (`innerRows`,
`RecLinkTree.rowsRoot`, which refuses a row past `UB` or `PB` bits), every instance's. Any other unit passes. -/
def check (st : Stmt) (own : Option (Array Registered.Port)) (files : Option RecCoins.Source)
    (links : Option (ByteArray × Nat)) : V Unit := do
  let unit := st.c.unit.name
  let some (UB, PB) := VStar.linkParams unit
    | if unit.startsWith "rec-link/" then throw s!"rec-links: {unit}: a rec-link/ unit not named rec-link/u<UB>-p<PB>"
      else return
  let some (T, N) := links
    | throw s!"rec-links: {unit}: the statement carries no link table (--rec-links ROOT:N), whose rows this part checks"
  let some own := own
    | throw s!"rec-links: {unit}: the verifier holds no registered reads (--registered): the table's rows would be the prover's"
  let some (.files f) := files
    | throw (s!"rec-links: {unit}: the verifier holds no inner statement (--rec-inner-circuit, --rec-inner-public, " ++
        "--rec-inner-partition, --rec-inner-session), whose rows the table links")
  checkTable st.c st.pub own T N
  let (ins, outs) := innerRows (← f.stmt unit).2
  let R ← (RecLinkTree.rowsRoot UB PB ins).mapError (s!"rec-links: {unit}: the input rows' tree: " ++ ·)
  let W ← (RecLinkTree.rowsRoot UB PB outs).mapError (s!"rec-links: {unit}: the output rows' tree: " ++ ·)
  RecCoins.checkValue st.c st.pub own "rin" IN_VALUE (Array.replicate N (RecLinkRef.expected R).toArray)
    fun i => s!"row {i}'s input rows' root"
  RecCoins.checkValue st.c st.pub own "rout" OUT_VALUE (Array.replicate N (RecLinkRef.expected W).toArray)
    fun i => s!"row {i}'s output rows' root"

end RecLinkPart
end Flock
