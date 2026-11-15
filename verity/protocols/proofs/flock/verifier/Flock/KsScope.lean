/-
The forms `verify --zk` refuses so that the sampled audit's knowledge soundness (`Flock.Guarantees.SampledKSLiveV3`)
takes none of `Flock.Bridge.SampledKSLiveRoots`' file preconditions as a hypothesis: no form goes ahead of its proof.
Under `--zk` every session of a statement outside them is refused, beside `ProvedScope.check` and `PortRoots.check`:

* `tagsWhy tags`: a typed statement, one with public outputs, one without retained rounds, or one whose leaves are not
  `hm96-sha512/v1` (`SampledKSLiveRoots`' `I.tags.typed = false` and `TagsOkZHJ`).
* `filesWhy tags c pub publicFile`: a circuit outside `ProvedScope.scopeOkH` (the unit reads one input group and returns
  the outputs, `SampledKSLiveRoots`' `scopeOkH`; `ProvedScope.check` holds a hidden-output circuit only to the weaker
  `scopeOkI`), and a shared-row public file (header `shared_rows`) outside the form `Flock.Guarantees.SampledKSLiveV5` covers
  (`sharedWhy`): the circuit's public input ports parse (`HmIn.parse`), the file loads over them (`HmIn.loadPublic`),
  and no two ports' roots over its tables are one root under different domains (`collidesT`, V5's `RootFixesDomainL` at
  the file's leaf map). A file without shared rows is V3's form (`pub.tables = none`).
* `drawWhy count draw`: at more than one table, a session whose record holds no sample that parses, or whose subset sample's
  `k` the tables don't divide (`SampledKSLiveRoots`' `kd % nTab = 0`).

`checkOf parsed tags publicFile count draw` runs them on the circuit `parsed` it is handed and the public file as
`HmOut.loadPublic` loads it over that circuit. `check` runs `checkOf` on the circuit `HmOut.parse` reads, a flat file's,
and `check_ok` says what passing `check` establishes: `SampledKSLiveRoots`' file preconditions, or V5's at a shared-row
file. Their sixth, that the unit is a built net, follows from the parse of an untyped statement. A V* session
(`--rec-inner`) is the recursion's, whose guarantee is its own, so `verify` doesn't run the check on it.
`flock-verify ks-form` runs `checkOf` alone on a statement's files, and the Python mirror
(`one_stage.proofs.ks_form_why`, `filesWhy` on the files' heads) is held to it.

`verify --zk` hands `checkOf` the circuit `ProvedScope.circuitOf` reads: at a templated statement (`Tags.isTmpl`) the
template's parse (`HmOut.parseTmpl`), which carries its template, never `HmOut.parse`'s, which refuses its tags; at a
flat one, `HmOut.parse`'s. A templated circuit is held to the unit's form on the template's description (`unitWhy`:
`ProvedScope.scopeOkT` at `count` tables), never on the rows the template unrolls to, and its statement circuit, the
flat circuit `HmOut.unrolled` gives, is then in `scopeOkH`. `checkOf_ok` says what passing establishes at any circuit.
-/
module

public import Flock.PortRoots

@[expose] public section

namespace Flock.KsScope

open Lean (Json)

/-- What `--zk`'s tags must be for the sampled audit's knowledge soundness: untyped, hidden outputs, retained rounds, and
`hm96-sha512/v1` leaves. -/
def tagsWhy (tags : Tags) : Option String :=
  if tags.typed then some s!"{tags.name}, a typed statement"
  else if !tags.hiddenOutputs then some s!"{tags.name}, a statement with public outputs under --zk"
  else if !tags.retainRounds then some s!"{tags.name}, a statement without retained rounds"
  else if tags.merkleLeaf != "hm96-sha512/v1" then some s!"{tags.name}, a statement with {tags.merkleLeaf} leaves"
  else none

def SCOPE_WHY : String :=
  "a circuit outside the unit's form under --zk (scopeOkH: the unit reads its one input group and returns the outputs)"

def TMPL_WHY : String :=
  "a templated circuit outside the unit's form under --zk (scopeOkT, on its template: one table, and the unit reads its \
    one input group and returns the outputs)"

def TABLES_WHY : String := "a shared-row public file (header shared_rows) under --zk"

def ROOTS_WHY : String :=
  "a shared-row public file two of whose ports' roots over its tables are one root under different domains"

/-- **Port `q`'s registered root over a `shared_rows` file's table**: the frame-v3 root, at SHA-512, over its table's
`counts[q]` rows, row `r` the frame leaf at the port's schema of the tree leaf of digest `starts[q] + r`; the root
`RootFixesDomainL` reads at the file's leaf map (`LeafMap.ofTables`). -/
def rootT (c : Circuit) (pub : Public) (t : Tables) (q : Nat) : ByteArray :=
  FV3.root (PortRoots.domOf c q) ((Array.range (t.counts.getD q 0)).map fun r =>
    Registered.leafAs (PortRoots.schemaOf c q) (PortRoots.domOf c q) r
      (Hm96.Default512.treeLeaf (pub.digests.getD (t.starts.getD q 0 + r) ByteArray.empty))) Sha512.hash

/-- **Two ports' roots over a `shared_rows` file's tables are one root under different domains** (`PortRoots.collidesIn`
at `rootT`). -/
def collidesT (c : Circuit) (pub : Public) (t : Tables) : Bool :=
  PortRoots.collidesIn c ((Array.range c.ports.size).map (rootT c pub t))

/-- **A shared-row public file in the form `SampledKSLiveV5` covers**: the circuit's public input ports parse
(`HmIn.parse`), the file loads over them (`HmIn.loadPublic`) with its tables, and no two ports' roots over them are one
root under different domains (`collidesT`). -/
def sharedWhy (tags : Tags) (c : Circuit) (publicFile : ByteArray) : Option String :=
  match HmIn.parse c with
  | .error e => some s!"{TABLES_WHY} whose circuit's public input ports do not parse ({e})"
  | .ok pi => match HmIn.loadPublic tags c pi publicFile with
    | .error e => some s!"{TABLES_WHY} that does not load over the circuit's public input ports ({e})"
    | .ok pub => match pub.tables with
      | none => some TABLES_WHY
      | some t => if collidesT c pub t then some ROOTS_WHY else none

/-- **The unit's form**: a templated statement's circuit (`Tags.isTmpl`) in `ProvedScope.scopeOkT` under `--zk` at
`count` tables, on its template (`Circuit.tmpl`), which `scopeOkT` reads without the rows it unrolls to; any other
circuit in `ProvedScope.scopeOkH`. -/
def unitWhy (tags : Tags) (count : Nat) (c : Circuit) : Option String :=
  if tags.isTmpl then
    match c.tmpl with
    | some tu => if ProvedScope.scopeOkT tags.hiddenOutputs true count c tu then none else some TMPL_WHY
    | none => some TMPL_WHY
  else if ProvedScope.scopeOkH c then none else some SCOPE_WHY

/-- The circuit and public file at `count` tables: the unit's form (`unitWhy`), and no shared rows or shared rows in V5's
form (`sharedWhy`). -/
def filesWhy (tags : Tags) (count : Nat) (c : Circuit) (pub : Public) (publicFile : ByteArray) : Option String :=
  match unitWhy tags count c with
  | some why => some why
  | none => if pub.tables.isSome then sharedWhy tags c publicFile else none

/-- The session's sample (`draw`, the record's `unit_draw`) at `count` tables: none to check at one table; at more, a sample
that parses, and a subset sample's `k` a multiple of `count`, so every table proves `k / count` units. -/
def drawWhy (count : Nat) (draw : Option Json) : Option String :=
  if count ≤ 1 then none
  else match draw with
    | none => some s!"a session of {count} tables without a draw"
    | some j => match Draw.UnitDraw.ofJson j with
      | .error e => some s!"a draw that does not parse ({e})"
      | .ok d => match d.law with
        | .subset k => if k % count = 0 then none
          else some s!"a subset draw of {k} units over {count} tables, which do not divide it"
        | _ => none

/-- **`verify --zk`'s files are in the form the sampled audit's knowledge soundness covers**: the tags (`tagsWhy`), on the
circuit `parsed` (`ProvedScope.circuitOf`'s under `verify --zk`) and the public file as `HmOut.loadPublic` loads it the
unit's form and no shared rows (`filesWhy`, or V5's shared-row form), and the session's sample (`drawWhy`). -/
def checkOf (parsed : V Circuit) (tags : Tags) (publicFile : ByteArray) (count : Nat) (draw : Option Json) :
    Except String Unit :=
  match tagsWhy tags with
  | some why => .error (ProvedScope.refusal why)
  | none => match parsed with
    | .error e => .error (ProvedScope.refusal s!"a circuit that does not parse ({e})")
    | .ok c => match HmOut.loadPublic tags c publicFile with
      | .error e => .error (ProvedScope.refusal s!"a public file that does not load ({e})")
      | .ok pub => match filesWhy tags count c pub publicFile with
        | some why => .error (ProvedScope.refusal why)
        | none => match drawWhy count draw with
          | some why => .error (ProvedScope.refusal why)
          | none => .ok ()

/-- `checkOf` on the circuit file as `HmOut.parse` reads it. -/
def check (tags : Tags) (circuitFile publicFile : ByteArray) (tables : List (String × ByteArray)) (count : Nat)
    (draw : Option Json) : Except String Unit :=
  checkOf (HmOut.parse tags circuitFile tables) tags publicFile count draw

theorem tagsWhy_none {tags : Tags} (h : tagsWhy tags = none) :
    tags.typed = false ∧ tags.hiddenOutputs = true ∧ tags.retainRounds = true ∧ tags.merkleLeaf = "hm96-sha512/v1" := by
  unfold tagsWhy at h
  split at h
  next => cases h
  next h1 =>
    split at h
    next => cases h
    next h2 =>
      split at h
      next => cases h
      next h3 =>
        split at h
        next => cases h
        next h4 => exact ⟨by simpa using h1, by simpa using h2, by simpa using h3, by simpa using h4⟩

/-- **What V5's shared-row form says of the files**: the public input ports, the file loaded over them, its tables, and
no collision of their roots. -/
def SharedOk (tags : Tags) (c : Circuit) (publicFile : ByteArray) : Prop :=
  ∃ pi pub t, HmIn.parse c = .ok pi ∧ HmIn.loadPublic tags c pi publicFile = .ok pub ∧ pub.tables = some t ∧
    collidesT c pub t = false

theorem sharedWhy_none {tags : Tags} {c : Circuit} {publicFile : ByteArray} (h : sharedWhy tags c publicFile = none) :
    SharedOk tags c publicFile := by
  unfold sharedWhy at h
  cases hp : HmIn.parse c with
  | error e => simp [hp] at h
  | ok pi =>
    cases hl : HmIn.loadPublic tags c pi publicFile with
    | error e => simp [hp, hl] at h
    | ok pub =>
      cases ht : pub.tables with
      | none => simp [hp, hl, ht] at h
      | some t =>
        cases hc : collidesT c pub t with
        | true => simp [hp, hl, ht, hc] at h
        | false => exact ⟨pi, pub, t, hp, hl, ht, hc⟩

/-- **No collision over a file's tables says each pair of ports with one root has one domain.** -/
theorem domOf_eq_of_collidesT_false {c : Circuit} {pub : Public} {t : Tables} (h : collidesT c pub t = false)
    {q q' : Nat} (hlt : q' < q) (hq : q < c.ports.size) (he : rootT c pub t q' = rootT c pub t q) :
    PortRoots.domOf c q' = PortRoots.domOf c q :=
  PortRoots.dom_eq_of_collidesBy_false (np := c.ports.size) (dom := PortRoots.domOf c)
    (roots := (Array.range c.ports.size).map (rootT c pub t)) h hlt hq
    (by rw [PortRoots.getD_range_map _ (Nat.lt_trans hlt hq), PortRoots.getD_range_map _ hq]; exact he)

/-- **A circuit of any other statement in the unit's form is in `scopeOkH`.** -/
theorem unitWhy_flat {tags : Tags} {count : Nat} {c : Circuit} (h : unitWhy tags count c = none)
    (ht : tags.isTmpl = false) : ProvedScope.scopeOkH c = true := by
  unfold unitWhy at h
  rw [ite_eq_right (Bool.eq_false_iff.mp ht)] at h
  split at h
  next hs => exact hs
  next => cases h

/-- **A templated statement's circuit in the unit's form carries its template, in `scopeOkT`.** -/
theorem unitWhy_tmpl {tags : Tags} {count : Nat} {c : Circuit} (h : unitWhy tags count c = none)
    (ht : tags.isTmpl = true) :
    ∃ tu, c.tmpl = some tu ∧ ProvedScope.scopeOkT tags.hiddenOutputs true count c tu = true := by
  unfold unitWhy at h
  rw [ite_eq_left ht] at h
  split at h
  next tu hc =>
    split at h
    next hs => exact ⟨tu, hc, hs⟩
    next => cases h
  next => cases h

theorem filesWhy_none {tags : Tags} {count : Nat} {c : Circuit} {pub : Public} {publicFile : ByteArray}
    (h : filesWhy tags count c pub publicFile = none) :
    unitWhy tags count c = none ∧ (pub.tables = none ∨ SharedOk tags c publicFile) := by
  unfold filesWhy at h
  split at h
  next => cases h
  next hu =>
    split at h
    next => exact ⟨hu, Or.inr (sharedWhy_none h)⟩
    next h2 => exact ⟨hu, Or.inl (by simpa using h2)⟩

/-- **At more than one table, a passing sample parses, and its subset `k` is a multiple of the tables.** -/
theorem drawWhy_none {count : Nat} {draw : Option Json} (h : drawWhy count draw = none) (hc : ¬ count ≤ 1) :
    ∃ j d, draw = some j ∧ Draw.UnitDraw.ofJson j = .ok d ∧ ∀ k, d.law = .subset k → k % count = 0 := by
  unfold drawWhy at h
  rw [if_neg hc] at h
  cases draw with
  | none => simp at h
  | some j =>
    cases hj : Draw.UnitDraw.ofJson j with
    | error e => simp [hj] at h
    | ok d =>
      refine ⟨j, d, rfl, hj, fun k hk => ?_⟩
      simp only [hj, hk] at h
      split at h
      next hm => exact hm
      next => cases h

/-- **What `check` passing says of the files**: the tags pass `tagsWhy`, and at the circuit `HmOut.parse` reads and the
public file `HmOut.loadPublic` loads, the circuit is in `scopeOkH`, the public file has no shared rows (V3's form) or is
in V5's shared-row form (`SharedOk`), and the sample passes `drawWhy`. -/
theorem check_ok {tags : Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : Nat}
    {draw : Option Json} {c : Circuit} {pub : Public} (h : check tags cf pf tables count draw = .ok ())
    (hc : HmOut.parse tags cf tables = .ok c) (hpub : HmOut.loadPublic tags c pf = .ok pub) :
    tagsWhy tags = none ∧ ProvedScope.scopeOkH c = true ∧ (pub.tables = none ∨ SharedOk tags c pf) ∧
      drawWhy count draw = none := by
  unfold check checkOf at h
  rw [hc] at h
  cases ht : tagsWhy tags with
  | some w => simp [ht] at h
  | none =>
    simp only [ht, hpub] at h
    cases hf : filesWhy tags count c pub pf with
    | some w => simp [hf] at h
    | none =>
      simp only [hf] at h
      cases hd : drawWhy count draw with
      | some w => simp [hd] at h
      | none =>
        obtain ⟨hu, hp⟩ := filesWhy_none hf
        exact ⟨rfl, unitWhy_flat hu (HmOut.parse_isTmpl hc), hp, rfl⟩

/-- **What `checkOf` passing says of the files**, at any circuit `parsed` holds (a templated statement's carrying its
template, as `ProvedScope.circuitOf` reads it): the tags pass `tagsWhy`, the circuit is in the unit's form (`unitWhy`:
`unitWhy_flat`, `unitWhy_tmpl`), the public file `HmOut.loadPublic` loads has no shared rows or is in V5's shared-row
form (`SharedOk`), and the sample passes `drawWhy`. -/
theorem checkOf_ok {parsed : V Circuit} {tags : Tags} {pf : ByteArray} {count : Nat} {draw : Option Json}
    {c : Circuit} {pub : Public} (h : checkOf parsed tags pf count draw = .ok ()) (hc : parsed = .ok c)
    (hpub : HmOut.loadPublic tags c pf = .ok pub) :
    tagsWhy tags = none ∧ unitWhy tags count c = none ∧ (pub.tables = none ∨ SharedOk tags c pf) ∧
      drawWhy count draw = none := by
  subst hc
  unfold checkOf at h
  cases ht : tagsWhy tags with
  | some w => simp [ht] at h
  | none =>
    simp only [ht, hpub] at h
    cases hf : filesWhy tags count c pub pf with
    | some w => simp [hf] at h
    | none =>
      simp only [hf] at h
      cases hd : drawWhy count draw with
      | some w => simp [hd] at h
      | none =>
        obtain ⟨hu, hp⟩ := filesWhy_none hf
        exact ⟨rfl, hu, hp, rfl⟩

end Flock.KsScope
