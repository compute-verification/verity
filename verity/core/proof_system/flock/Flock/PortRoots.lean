/-
The circuit and public files `verify --zk` holds to the form the sampled audit's knowledge soundness covers
(`Flock.Guarantees.SampledKSLiveV3`): every port a `sha512/row/v1`, `sha512/row/v2` or `sha512/row-seg/v1` row as the
scope check reads them (`rowsOk`), each of its own name, and no two ports whose registered frame-v3 roots over the public
file's population are one root under different domains (`collides`).

* `rootOf c pub nl q`: port `q`'s registered root, `FV3.root` over its domain (its name, `domOf`) and its `nl` rows'
  frame leaves at its leaf schema (`schemaOf`, `Registered.portSchema`) of the public file's `b ‖ c` (`leafOf`).
* `collides c pub nl`: some ports `q' < q` with one root and different domains. Each root is SHA-512 of a frame whose
  first part is its domain, so such a pair is a SHA-512 collision, which a file exhibits only by holding one.
* `checkOf parsed tags publicFile`: on the circuit as `HmOut.parse` reads it, `rowsOk` and no port name twice
  (`Registered.dupName`), then on the public file as `HmOut.loadPublic` reads it, no `collides` at its population
  (`pub.n`). `verify --zk` refuses every session whose files fail it, after every other check (`Main.verifyCmd`).
-/
import Flock.ProvedScope
import Flock.Registered

namespace Flock.PortRoots

/-- Port `q`'s frame-v3 domain: its name in the circuit file. -/
def domOf (c : Circuit) (q : Nat) : ByteArray := ((c.ports[q]?.map (·.name)).getD "").toUTF8

/-- Port `q`'s leaf schema: `Registered.portSchema`'s, as the verifier frames its rows. -/
def schemaOf (c : Circuit) (q : Nat) : String :=
  (c.ports[q]?.map Registered.portSchema).getD HmRow.ROW_SCHEMA_V2

/-- Row `u` of port `q`'s registered tree leaf: the hm96 tree leaf of the public file's `b ‖ c` there. -/
def leafOf (c : Circuit) (pub : Public) (q u : Nat) : ByteArray :=
  Hm96.Default512.treeLeaf (pub.digest u q c.ports.size)

/-- Port `q`'s registered root over `nl` rows, its tree's nodes and pads hashed by `H`. -/
def rootWith (H : ByteArray → ByteArray) (c : Circuit) (pub : Public) (nl q : Nat) : ByteArray :=
  FV3.root (domOf c q) ((Array.range nl).map fun u =>
    Registered.leafAs (schemaOf c q) (domOf c q) u (leafOf c pub q u)) H

/-- Port `q`'s registered root over `nl` rows, at SHA-512. -/
def rootOf (c : Circuit) (pub : Public) (nl q : Nat) : ByteArray := rootWith Sha512.hash c pub nl q

/-- **Every port is a row the scope check reads**: all `sha512/row/v1` rows (`ProvedScope.v1Ports`), or each a
`sha512/row/v2` or `sha512/row-seg/v1` row (`ProvedScope.bitRows`). -/
def rowsOk (c : Circuit) : Bool := ProvedScope.v1Ports c || ProvedScope.bitRows c

/-- Some ports `q' < q` with one root (`roots`, by port) and different domains. -/
def collidesIn (c : Circuit) (roots : Array ByteArray) : Bool :=
  (List.range c.ports.size).any fun q => (List.range q).any fun q' =>
    decide (roots.getD q' ByteArray.empty = roots.getD q ByteArray.empty) && decide (domOf c q' ≠ domOf c q)

/-- `collidesIn` at the roots `H` makes, each computed once. -/
def collidesWith (H : ByteArray → ByteArray) (c : Circuit) (pub : Public) (nl : Nat) : Bool :=
  collidesIn c ((Array.range c.ports.size).map (rootWith H c pub nl))

/-- **Two ports' registered roots over `nl` rows are one root under different domains.** -/
def collides (c : Circuit) (pub : Public) (nl : Nat) : Bool := collidesWith Sha512.hash c pub nl

def ROWS_WHY : String :=
  "a port that is not a sha512/row/v1, sha512/row/v2 or sha512/row-seg/v1 row the scope check reads"

def NAMES_WHY : String :=
  "two ports of one name (a registered root's domain is its port's name)"

def ROOTS_WHY : String :=
  "two ports whose registered roots are one root under different domains (a SHA-512 collision)"

/-- `checkOf` at the roots `H` makes. -/
def checkWith (H : ByteArray → ByteArray) (parsed : V Circuit) (tags : Tags) (publicFile : ByteArray) :
    Except String Unit :=
  match parsed with
  | .error e => .error (ProvedScope.refusal s!"a circuit that does not parse ({e})")
  | .ok c =>
    if !rowsOk c then .error (ProvedScope.refusal ROWS_WHY)
    else if (Registered.dupName c).isSome then .error (ProvedScope.refusal NAMES_WHY)
    else match HmOut.loadPublic tags c publicFile with
      | .error e => .error (ProvedScope.refusal s!"a public file that does not load ({e})")
      | .ok pub => if collidesWith H c pub pub.n then .error (ProvedScope.refusal ROOTS_WHY) else .ok ()

/-- **The files' rows, names and roots are the form the sampled audit's knowledge soundness covers**: the circuit
`parsed` (`HmOut.parse`'s) parses, every port is a `rowsOk` row and no two ports share a name (`Registered.dupName`),
the public file loads (`HmOut.loadPublic`), and no two ports' roots over its population `collides`. -/
def checkOf (parsed : V Circuit) (tags : Tags) (publicFile : ByteArray) : Except String Unit :=
  checkWith Sha512.hash parsed tags publicFile

/-- `checkOf` on the circuit file as `HmOut.parse` reads it. -/
def check (tags : Tags) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray)) :
    Except String Unit :=
  checkOf (HmOut.parse tags circuitFile tables) tags publicFile

/-- **What `check` passing says of the files**: at the circuit `HmOut.parse` reads and the public file
`HmOut.loadPublic` loads, every port is a `rowsOk` row, no two ports share a name, and no two ports' roots over the
population `collides`. -/
theorem check_ok {tags : Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit} {pub : Public}
    (h : check tags cf pf tables = .ok ()) (hc : HmOut.parse tags cf tables = .ok c)
    (hpub : HmOut.loadPublic tags c pf = .ok pub) :
    rowsOk c = true ∧ Registered.dupName c = none ∧ collides c pub pub.n = false := by
  unfold check checkOf checkWith at h
  rw [hc] at h
  cases hr : rowsOk c with
  | false => simp [hr] at h
  | true =>
    cases hd : Registered.dupName c with
    | some _ => simp [hr, hd] at h
    | none =>
      cases hx : collidesWith Sha512.hash c pub pub.n with
      | false => exact ⟨rfl, rfl, hx⟩
      | true => simp [hr, hd, hpub, hx] at h

/-- **No collision says each pair of ports with one root has one domain.** -/
theorem domOf_eq_of_collides_false {c : Circuit} {pub : Public} {nl : Nat} (h : collides c pub nl = false)
    {q q' : Nat} (hlt : q' < q) (hq : q < c.ports.size) (he : rootOf c pub nl q' = rootOf c pub nl q) :
    domOf c q' = domOf c q := by
  have hq' : q' < c.ports.size := Nat.lt_trans hlt hq
  have get : ∀ i, i < c.ports.size →
      ((Array.range c.ports.size).map (rootWith Sha512.hash c pub nl)).getD i ByteArray.empty = rootOf c pub nl i := by
    intro i hi
    have hs : i < ((Array.range c.ports.size).map (rootWith Sha512.hash c pub nl)).size := by simpa using hi
    first
      | (unfold Array.getD; rw [dif_pos hs]; simp [rootOf])
      | simp [Array.getD_eq_getD_getElem?, hi, rootOf]
      | simp [hi, rootOf]
  refine Classical.byContradiction fun hd => ?_
  have hx : collides c pub nl = true := by
    unfold collides collidesWith collidesIn
    rw [List.any_eq_true]
    refine ⟨q, List.mem_range.2 hq, ?_⟩
    rw [List.any_eq_true]
    refine ⟨q', List.mem_range.2 hlt, ?_⟩
    rw [get q' hq', get q hq, he]
    simp [hd]
  rw [hx] at h
  cases h

end Flock.PortRoots
