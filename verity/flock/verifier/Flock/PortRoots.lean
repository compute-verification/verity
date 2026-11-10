/-
The circuit and public files `verify --zk` holds to the form the sampled audit's knowledge soundness covers
(`Flock.Guarantees.SampledKSLiveV3`): every port a `sha512/row/v1`, `sha512/row/v2` or `sha512/row-seg/v1` row as the
scope check reads them (`rowsOk`), each of its own name, no two ports whose registered frame-v3 roots over the public
file's population are one root under different domains (`collides`), every row under `2^64` bits (`wordsOk`), and the
published tree's domains each 64 bytes and of their own port (`idsSized`, `dupId`), with no two ports' published roots
one root under different domains (`collidesPub`).

* `rootOf c pub nl q`: port `q`'s registered root, `FV3.root` over its domain (its name, `domOf`) and its `nl` rows'
  frame leaves at its leaf schema (`schemaOf`, `Registered.portSchema`) of the public file's `b ‖ c` (`leafOf`).
* `pubRoot c pub q`: port `q`'s published root, `FV3.root` over the public file's `frame_v3.domain_ids` entry for it
  (`idOf`) and its rows' frame leaves at its schema: its table's rows at a shared-row file, else one per instance
  (`countOf`, `digOf`), the tree `HmOut.checkPublic` holds to the header's root.
* `collides c pub nl`, `collidesPub c pub`: some ports `q' < q` with one root and different domains
  (`collidesBy`). Each root is SHA-512 of a frame whose first part is its domain, so such a pair is a SHA-512
  collision, which a file exhibits only by holding one. With no two published domains one (`dupId`) and no
  `collidesPub`, no two ports' published roots are one (`pubRoot_ne`).
* `checkOf parsed tags publicFile`: on the circuit `parsed` it is handed, `rowsOk` and no port name twice
  (`Registered.dupName`), then on the public file as `HmOut.loadPublic` reads it over that circuit the refusals of
  `loadedWhy`: `collides` at its population (`pub.n`), `wordsOk`, `idsSized`, `dupId` and `collidesPub`. At hidden
  outputs `verify --zk` hands it the circuit `ProvedScope.circuitOf` reads (`HmOut.parseCommit`), which for a templated
  statement is the template's parse (`HmOut.parseTmpl`) and for a flat one the circuit `HmOut.parse` reads, as `check`
  does. It refuses every session whose files fail it, after every other check (`Main.verifyCmd`).
-/
module

public import Flock.ProvedScope
public import Flock.Registered

@[expose] public section

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

/-- Port `q`'s published frame-v3 domain: the public file's `frame_v3.domain_ids` entry at its name, as bytes (empty
where the entry is missing or not hex). -/
def idOf (c : Circuit) (pub : Public) (q : Nat) : ByteArray :=
  (pub.fv3Bytes "domain_ids" ((c.ports[q]?.map (·.name)).getD "")).toOption.getD ByteArray.empty

/-- Port `q`'s published row count: its table's `counts[q]` at a shared-row file, else one row per instance. -/
def countOf (pub : Public) (q : Nat) : Nat :=
  match pub.tables with
  | some t => t.counts.getD q 0
  | none => pub.n

/-- Row `r` of port `q`'s published tree: the public file's `b ‖ c` at its table's `starts[q] + r` (a shared-row file),
else instance `r`'s. -/
def digOf (c : Circuit) (pub : Public) (q r : Nat) : ByteArray :=
  match pub.tables with
  | some t => pub.digests.getD (t.starts.getD q 0 + r) ByteArray.empty
  | none => pub.digest r q c.ports.size

/-- Port `q`'s published root, its tree's nodes and pads hashed by `H`: over its published domain (`idOf`) and its
`countOf` rows' frame leaves at its schema. -/
def pubRootWith (H : ByteArray → ByteArray) (c : Circuit) (pub : Public) (q : Nat) : ByteArray :=
  FV3.root (idOf c pub q) ((Array.range (countOf pub q)).map fun r =>
    Registered.leafAs (schemaOf c q) (idOf c pub q) r (Hm96.Default512.treeLeaf (digOf c pub q r))) H

/-- Port `q`'s published root, at SHA-512. -/
def pubRoot (c : Circuit) (pub : Public) (q : Nat) : ByteArray := pubRootWith Sha512.hash c pub q

/-- **Every port is a row the scope check reads**: all `sha512/row/v1` rows (`ProvedScope.v1Ports`), or each a
`sha512/row/v2` or `sha512/row-seg/v1` row (`ProvedScope.bitRows`). -/
def rowsOk (c : Circuit) : Bool := ProvedScope.v1Ports c || ProvedScope.bitRows c

/-- **Every port's row is under `2^64` bits**: its `16 · words` bits, which a row leaf's u64 length field holds. -/
def wordsOk (c : Circuit) : Bool := c.ports.all fun p => decide (16 * p.words < 2 ^ 64)

/-- Some ports `q' < q < np` with one root (`roots`, by port) and different domains (`dom`). -/
def collidesBy (np : Nat) (dom : Nat → ByteArray) (roots : Array ByteArray) : Bool :=
  (List.range np).any fun q => (List.range q).any fun q' =>
    decide (roots.getD q' ByteArray.empty = roots.getD q ByteArray.empty) && decide (dom q' ≠ dom q)

/-- Some ports `q' < q` with one root (`roots`, by port) and different names (`domOf`). -/
def collidesIn (c : Circuit) (roots : Array ByteArray) : Bool := collidesBy c.ports.size (domOf c) roots

/-- `collidesIn` at the roots `H` makes, each computed once. -/
def collidesWith (H : ByteArray → ByteArray) (c : Circuit) (pub : Public) (nl : Nat) : Bool :=
  collidesIn c ((Array.range c.ports.size).map (rootWith H c pub nl))

/-- **Two ports' registered roots over `nl` rows are one root under different domains.** -/
def collides (c : Circuit) (pub : Public) (nl : Nat) : Bool := collidesWith Sha512.hash c pub nl

/-- **Every port's published domain is 64 bytes**, a SHA-512 digest. -/
def idsSized (c : Circuit) (pub : Public) : Bool :=
  (List.range c.ports.size).all fun q => (idOf c pub q).size == 64

/-- **Two ports of one published domain.** -/
def dupId (c : Circuit) (pub : Public) : Bool :=
  (List.range c.ports.size).any fun q => (List.range q).any fun q' => decide (idOf c pub q' = idOf c pub q)

/-- `collidesPub` at the roots `H` makes, each computed once. -/
def collidesPubWith (H : ByteArray → ByteArray) (c : Circuit) (pub : Public) : Bool :=
  collidesBy c.ports.size (idOf c pub) ((Array.range c.ports.size).map (pubRootWith H c pub))

/-- **Two ports' published roots are one root under different published domains.** -/
def collidesPub (c : Circuit) (pub : Public) : Bool := collidesPubWith Sha512.hash c pub

def ROWS_WHY : String :=
  "a port that is not a sha512/row/v1, sha512/row/v2 or sha512/row-seg/v1 row the scope check reads"

def NAMES_WHY : String :=
  "two ports of one name (a registered root's domain is its port's name)"

def ROOTS_WHY : String :=
  "two ports whose registered roots are one root under different domains (a SHA-512 collision)"

def WORDS_WHY : String :=
  "a port whose rows are 2^64 bits or more (16 * words, past a row leaf's u64 length)"

def IDS_WHY : String :=
  "a port whose published frame-v3 domain (frame_v3.domain_ids) is missing or not 64 bytes"

def DUP_IDS_WHY : String :=
  "two ports of one published frame-v3 domain (frame_v3.domain_ids)"

def PUB_ROOTS_WHY : String :=
  "two ports whose published frame-v3 roots are one root under different domains (a SHA-512 collision)"

/-- `checkWith`'s refusals on the loaded public file, at the roots `H` makes: `collidesWith` at its population,
`wordsOk`, `idsSized`, `dupId` and `collidesPubWith`, in that order. -/
def loadedWhy (H : ByteArray → ByteArray) (c : Circuit) (pub : Public) : Option String :=
  if collidesWith H c pub pub.n then some ROOTS_WHY
  else if !wordsOk c then some WORDS_WHY
  else if !idsSized c pub then some IDS_WHY
  else if dupId c pub then some DUP_IDS_WHY
  else if collidesPubWith H c pub then some PUB_ROOTS_WHY
  else none

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
      | .ok pub => match loadedWhy H c pub with
        | some why => .error (ProvedScope.refusal why)
        | none => .ok ()

/-- **The files' rows, names, widths, domains and roots are the form the sampled audit's knowledge soundness covers**:
the circuit `parsed` (`ProvedScope.circuitOf`'s under `verify --zk`, `HmOut.parse`'s under `check`) parses, every port
is a `rowsOk` row and no two ports share a name
(`Registered.dupName`), the public file loads (`HmOut.loadPublic`), and `loadedWhy` refuses nothing: no two ports'
roots over its population `collides`, every row is under `2^64` bits, and the published domains are 64 bytes, pairwise
distinct, and fixed by their roots. -/
def checkOf (parsed : V Circuit) (tags : Tags) (publicFile : ByteArray) : Except String Unit :=
  checkWith Sha512.hash parsed tags publicFile

/-- `checkOf` on the circuit file as `HmOut.parse` reads it. -/
def check (tags : Tags) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray)) :
    Except String Unit :=
  checkOf (HmOut.parse tags circuitFile tables) tags publicFile

theorem loadedWhy_none {H : ByteArray → ByteArray} {c : Circuit} {pub : Public} (h : loadedWhy H c pub = none) :
    collidesWith H c pub pub.n = false ∧ wordsOk c = true ∧ idsSized c pub = true ∧ dupId c pub = false ∧
      collidesPubWith H c pub = false := by
  unfold loadedWhy at h
  cases h1 : collidesWith H c pub pub.n
  · cases h2 : wordsOk c
    · simp [h1, h2] at h
    · cases h3 : idsSized c pub
      · simp [h1, h2, h3] at h
      · cases h4 : dupId c pub
        · cases h5 : collidesPubWith H c pub
          · exact ⟨rfl, rfl, rfl, rfl, rfl⟩
          · simp [h1, h2, h3, h4, h5] at h
        · simp [h1, h2, h3, h4] at h
  · simp [h1] at h

/-- **`checkOf` passing at a circuit loads the public file** (`HmOut.loadPublic`). -/
theorem checkOf_load {tags : Tags} {c : Circuit} {pf : ByteArray} (h : checkOf (.ok c) tags pf = .ok ()) :
    ∃ pub, HmOut.loadPublic tags c pf = .ok pub := by
  unfold checkOf checkWith at h
  cases hl : HmOut.loadPublic tags c pf with
  | ok pub => exact ⟨pub, rfl⟩
  | error e =>
    simp only [hl] at h
    split at h
    · cases h
    · split at h <;> cases h

/-- **`checkOf` passing at a circuit, at the public file it loaded**: every port is a `rowsOk` row, no two ports share a
name, and `loadedWhy` refuses nothing at the public file `HmOut.loadPublic` loads. -/
theorem checkOf_loaded {tags : Tags} {c : Circuit} {pf : ByteArray} {pub : Public} (h : checkOf (.ok c) tags pf = .ok ())
    (hpub : HmOut.loadPublic tags c pf = .ok pub) :
    rowsOk c = true ∧ Registered.dupName c = none ∧ loadedWhy Sha512.hash c pub = none := by
  unfold checkOf checkWith at h
  cases hr : rowsOk c with
  | false => simp [hr] at h
  | true =>
    cases hd : Registered.dupName c with
    | some _ => simp [hr, hd] at h
    | none =>
      cases hw : loadedWhy Sha512.hash c pub with
      | none => exact ⟨rfl, rfl, rfl⟩
      | some _ => simp [hr, hd, hpub, hw] at h

/-- **What `checkOf` passing at a circuit says of the files**: at the public file `HmOut.loadPublic` loads, every port is
a `rowsOk` row, no two ports share a name, and no two ports' roots over the population `collides`. -/
theorem checkOf_ok {tags : Tags} {c : Circuit} {pf : ByteArray} {pub : Public} (h : checkOf (.ok c) tags pf = .ok ())
    (hpub : HmOut.loadPublic tags c pf = .ok pub) :
    rowsOk c = true ∧ Registered.dupName c = none ∧ collides c pub pub.n = false :=
  have hl := checkOf_loaded h hpub
  ⟨hl.1, hl.2.1, (loadedWhy_none hl.2.2).1⟩

/-- **`check` passing, at the files it read**: every port is a `rowsOk` row, no two ports share a name, and
`loadedWhy` refuses nothing at the public file `HmOut.loadPublic` loads. -/
theorem check_loaded {tags : Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit} {pub : Public}
    (h : check tags cf pf tables = .ok ()) (hc : HmOut.parse tags cf tables = .ok c)
    (hpub : HmOut.loadPublic tags c pf = .ok pub) :
    rowsOk c = true ∧ Registered.dupName c = none ∧ loadedWhy Sha512.hash c pub = none := by
  unfold check at h
  rw [hc] at h
  exact checkOf_loaded h hpub

/-- **What `check` passing says of the files**: at the circuit `HmOut.parse` reads and the public file
`HmOut.loadPublic` loads, every port is a `rowsOk` row, no two ports share a name, and no two ports' roots over the
population `collides`. -/
theorem check_ok {tags : Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit} {pub : Public}
    (h : check tags cf pf tables = .ok ()) (hc : HmOut.parse tags cf tables = .ok c)
    (hpub : HmOut.loadPublic tags c pf = .ok pub) :
    rowsOk c = true ∧ Registered.dupName c = none ∧ collides c pub pub.n = false :=
  have hl := check_loaded h hc hpub
  ⟨hl.1, hl.2.1, (loadedWhy_none hl.2.2).1⟩

/-- **What `check` passing says of the published tree**: at the circuit `HmOut.parse` reads and the public file
`HmOut.loadPublic` loads, every row is under `2^64` bits, every port's published domain is 64 bytes, no two are one, and
no two ports' published roots `collidesPub`. -/
theorem check_ok_pub {tags : Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    {pub : Public} (h : check tags cf pf tables = .ok ()) (hc : HmOut.parse tags cf tables = .ok c)
    (hpub : HmOut.loadPublic tags c pf = .ok pub) :
    wordsOk c = true ∧ idsSized c pub = true ∧ dupId c pub = false ∧ collidesPub c pub = false :=
  (loadedWhy_none (check_loaded h hc hpub).2.2).2

theorem getD_range_map (f : Nat → ByteArray) {n i : Nat} (hi : i < n) :
    ((Array.range n).map f).getD i ByteArray.empty = f i := by
  simp [Array.getD_eq_getD_getElem?, hi]

/-- **No collision says each pair of ports with one root has one domain.** -/
theorem dom_eq_of_collidesBy_false {np : Nat} {dom : Nat → ByteArray} {roots : Array ByteArray}
    (h : collidesBy np dom roots = false) {q q' : Nat} (hlt : q' < q) (hq : q < np)
    (he : roots.getD q' ByteArray.empty = roots.getD q ByteArray.empty) : dom q' = dom q := by
  refine Classical.byContradiction fun hd => ?_
  have hx : collidesBy np dom roots = true := by
    unfold collidesBy
    rw [List.any_eq_true]
    refine ⟨q, List.mem_range.2 hq, ?_⟩
    rw [List.any_eq_true]
    refine ⟨q', List.mem_range.2 hlt, ?_⟩
    rw [he]
    simp [hd]
  rw [hx] at h
  cases h

/-- **No collision says each pair of ports with one root has one domain.** -/
theorem domOf_eq_of_collides_false {c : Circuit} {pub : Public} {nl : Nat} (h : collides c pub nl = false)
    {q q' : Nat} (hlt : q' < q) (hq : q < c.ports.size) (he : rootOf c pub nl q' = rootOf c pub nl q) :
    domOf c q' = domOf c q :=
  dom_eq_of_collidesBy_false (np := c.ports.size) (dom := domOf c)
    (roots := (Array.range c.ports.size).map (rootWith Sha512.hash c pub nl)) h hlt hq
    (by rw [getD_range_map _ (Nat.lt_trans hlt hq), getD_range_map _ hq]; exact he)

/-- **No published collision says each pair of ports with one published root has one published domain.** -/
theorem idOf_eq_of_collidesPub_false {c : Circuit} {pub : Public} (h : collidesPub c pub = false) {q q' : Nat}
    (hlt : q' < q) (hq : q < c.ports.size) (he : pubRoot c pub q' = pubRoot c pub q) : idOf c pub q' = idOf c pub q :=
  dom_eq_of_collidesBy_false (np := c.ports.size) (dom := idOf c pub)
    (roots := (Array.range c.ports.size).map (pubRootWith Sha512.hash c pub)) h hlt hq
    (by rw [getD_range_map _ (Nat.lt_trans hlt hq), getD_range_map _ hq]; exact he)

/-- **No two published domains are one** at `dupId = false`. -/
theorem idOf_ne_of_dupId_false {c : Circuit} {pub : Public} (h : dupId c pub = false) {q q' : Nat} (hlt : q' < q)
    (hq : q < c.ports.size) : idOf c pub q' ≠ idOf c pub q := by
  intro he
  have hx : dupId c pub = true := by
    unfold dupId
    rw [List.any_eq_true]
    exact ⟨q, List.mem_range.2 hq, List.any_eq_true.2 ⟨q', List.mem_range.2 hlt, by simp [he]⟩⟩
  rw [hx] at h
  cases h

/-- **No two published roots are one** at pairwise distinct published domains and no published collision. -/
theorem pubRoot_ne {c : Circuit} {pub : Public} (hd : dupId c pub = false) (hx : collidesPub c pub = false)
    {q q' : Nat} (hlt : q' < q) (hq : q < c.ports.size) : pubRoot c pub q' ≠ pubRoot c pub q :=
  fun he => idOf_ne_of_dupId_false hd hlt hq (idOf_eq_of_collidesPub_false hx hlt hq he)

/-- **Every published domain is 64 bytes** at `idsSized`. -/
theorem idOf_size_of_idsSized {c : Circuit} {pub : Public} (h : idsSized c pub = true) {q : Nat}
    (hq : q < c.ports.size) : (idOf c pub q).size = 64 := by
  unfold idsSized at h
  simpa using List.all_eq_true.mp h q (List.mem_range.2 hq)

/-- **Every row is under `2^64` bits** at `wordsOk`. -/
theorem words_lt_of_wordsOk {c : Circuit} (h : wordsOk c = true) {q : Nat} (hq : q < c.ports.size) :
    16 * c.ports[q].words < 2 ^ 64 :=
  of_decide_eq_true ((Array.all_eq_true.mp h) q hq)

end Flock.PortRoots
