module

public import Flock.KsScope

/-!
# What a statement's checks read of its circuit file, kept by the file

`verity-verify-proof verify --zk` makes three checks of a statement's files on the parsed circuit and the public file
(`Flock.ProvedScope.checkOf`, `Flock.PortRoots.checkOf` and `Flock.KsScope.filesOf`). Each has two parts:

* verdicts of the circuit alone, which walk its rows: the wires' order and the unit's scope (`rulesOf`,
  `Flock.ProvedScope.circuitRules` past its refusal of a shared-row public file), the ports' form
  (`Flock.PortRoots.rowsOk`, `Flock.Registered.dupName`), the unit's form (`Flock.KsScope.unitWhy`) and the public
  input ports (`Flock.HmIn.parse`);
* checks of the public file, which read of the circuit its digest, its class, its ports, its units a proof block and
  its META, and none of its rows (`publicOf_bare`, `loadPublic_bare`, `loadPublicRow_bare`, `loadedWhy_bare`,
  `loadPublicIn_bare`, `collidesT_bare`).

`Kept` is the first part beside the circuit without its rows (`bare`). It is a function of the parsed circuit and the
statement's tags, so statements that name one circuit file share one, and nothing of it holds the circuit's rows.
The three checks each load the public file as a statement with hidden outputs does (`Flock.HmOut.loadPublic`);
`loadOf` is that load, made once for the three. `checkOf_kept`, `portsOf_kept` and `filesOf_kept` say each of the
three checks on the parse is `checkOf`, `portsOf` and `filesOf` here on its `Kept` and that one load.
-/

@[expose] public section

namespace Spot.CircuitKept

open Flock

/-- **A parsed circuit without its rows**: its nets, its wires and its leaf maps emptied, every other field its own.
What the public file's checks read of a circuit is in the fields it keeps. -/
def bare (c : Circuit) : Circuit :=
  { c with nets := #[], wires := #[], leavesIn := #[], leavesOut := #[], leafCuts := #[] }

theorem publicOf_bare (tags : Tags) (c : Circuit) (file : ByteArray) :
    ProvedScope.publicOf tags (bare c) file = ProvedScope.publicOf tags c file :=
  rfl

theorem loadPublic_bare (tags : Tags) (c : Circuit) (file : ByteArray) :
    HmOut.loadPublic tags (bare c) file = HmOut.loadPublic tags c file :=
  rfl

theorem loadPublicRow_bare (tags : Tags) (c : Circuit) (file : ByteArray) :
    HmRow.loadPublic tags (bare c) file = HmRow.loadPublic tags c file :=
  rfl

theorem loadedWhy_bare (H : ByteArray → ByteArray) (c : Circuit) (pub : Public) :
    PortRoots.loadedWhy H (bare c) pub = PortRoots.loadedWhy H c pub :=
  rfl

theorem loadPublicIn_bare (tags : Tags) (c : Circuit) (pi : HmIn.Ports) (file : ByteArray) :
    HmIn.loadPublic tags (bare c) pi file = HmIn.loadPublic tags c pi file :=
  rfl

theorem collidesT_bare (c : Circuit) (pub : Public) (t : Tables) :
    KsScope.collidesT (bare c) pub t = KsScope.collidesT c pub t :=
  rfl

/-- **`Flock.ProvedScope.circuitRules` past its refusal of a shared-row public file**: what it requires of the circuit
alone, at the tags `tags`, `--zk` or not and `count` tables. -/
def rulesOf (tags : Tags) (zk : Bool) (count : Nat) (c : Circuit) : Except String Unit :=
  match c.tmpl with
  | some tu =>
    if ProvedScope.scopeOkT tags.hiddenOutputs zk count c tu then .ok ()
    else .error (ProvedScope.refusal (ProvedScope.scopeWhyT tags.hiddenOutputs zk count c tu))
  | none =>
    if ProvedScope.wiresFeed c && !ProvedScope.wiresAcyclic c then .error (ProvedScope.refusal ProvedScope.wiresWhy)
    else if (if tags.hiddenOutputs then ProvedScope.scopeOkI c else ProvedScope.scopeOk c) then .ok ()
    else .error (ProvedScope.refusal (ProvedScope.scopeWhy tags.hiddenOutputs c))

theorem circuitRules_rulesOf (tags : Tags) (zk : Bool) (count : Nat) (registered : Bool) (c : Circuit) (pub : Public) :
    ProvedScope.circuitRules tags zk count registered c pub =
      if pub.tables.isSome && !ProvedScope.sharedRowsOk tags zk registered then
        .error (ProvedScope.refusal "a shared-row public file (header shared_rows)")
      else rulesOf tags zk count c :=
  rfl

/-- **What a statement's checks read of its circuit, by the circuit alone**: the circuit without its rows (`c`),
`rulesOf`'s verdict (`rules`), the caller's own check of the circuit (`own`), whether every port is a row the scope
check reads (`rows`) and whether two ports share a name (`named`), the unit's form (`unit`), and the public input
ports (`inputs`). -/
structure Kept where
  c : Circuit
  rules : Except String Unit
  own : V Unit
  rows : Bool
  named : Bool
  unit : Option String
  inputs : V HmIn.Ports

/-- The parsed circuit `c`'s `Kept`, at the tags `tags`, `--zk` or not, `count` tables and the caller's own check
`own`. -/
def keptOf (tags : Tags) (zk : Bool) (count : Nat) (own : Circuit → V Unit) (c : Circuit) : Kept :=
  { c := bare c
    rules := rulesOf tags zk count c
    own := own c
    rows := PortRoots.rowsOk c
    named := (Registered.dupName c).isSome
    unit := KsScope.unitWhy tags count c
    inputs := HmIn.parse c }

/-- The units a proof block a parse names are its `Kept`'s. -/
theorem keptOf_g (parsed : V Circuit) (tags : Tags) (zk : Bool) (count : Nat) (own : Circuit → V Unit) :
    parsed.toOption.map (·.g) = (parsed.map (keptOf tags zk count own)).toOption.map (·.c.g) := by
  cases parsed <;> rfl

/-- **The public file as a statement with hidden outputs loads it** (`Flock.HmOut.loadPublic`), on the circuit
without its rows; the circuit's own refusal where it does not parse. A statement's three checks read this one load. -/
def loadOf (k : V Kept) (tags : Tags) (publicFile : ByteArray) : V Public :=
  match k with
  | .error e => .error e
  | .ok k => HmOut.loadPublic tags k.c publicFile

/-- `Flock.ProvedScope.checkOf` on a circuit's `Kept` and the public file's load `pub` (`checkOf_kept`), which it
reads where the statement's outputs are hidden; where they are not it loads the file as such a statement does. -/
def checkOf (k : V Kept) (pub : V Public) (tags : Tags) (zk : Bool) (publicFile : ByteArray) (count : Nat)
    (publicInputs registered : Bool) : Except String Unit :=
  match ProvedScope.firstFailure (ProvedScope.statementRules tags zk count publicInputs) with
  | some why => .error (ProvedScope.refusal why)
  | none => match k with
    | .error e => .error (ProvedScope.refusal s!"a circuit that does not parse ({e})")
    | .ok k => match (if tags.hiddenOutputs then pub else HmRow.loadPublic tags k.c publicFile) with
      | .error e => .error (ProvedScope.refusal s!"a public file that does not load ({e})")
      | .ok pub =>
        if pub.tables.isSome && !ProvedScope.sharedRowsOk tags zk registered then
          .error (ProvedScope.refusal "a shared-row public file (header shared_rows)")
        else k.rules

/-- **The scope check on the parse is the scope check on its `Kept` and the public file's one load.** -/
theorem checkOf_kept (parsed : V Circuit) (tags : Tags) (zk : Bool) (publicFile : ByteArray) (count : Nat)
    (publicInputs registered : Bool) (own : Circuit → V Unit) :
    ProvedScope.checkOf parsed tags zk publicFile count publicInputs registered =
      checkOf (parsed.map (keptOf tags zk count own)) (loadOf (parsed.map (keptOf tags zk count own)) tags publicFile)
        tags zk publicFile count publicInputs registered := by
  unfold ProvedScope.checkOf checkOf
  cases ProvedScope.firstFailure (ProvedScope.statementRules tags zk count publicInputs) with
  | some why => rfl
  | none =>
    cases parsed with
    | error e => rfl
    | ok c =>
      simp only [Except.map, keptOf, loadOf, ProvedScope.publicOf, loadPublic_bare, loadPublicRow_bare,
        circuitRules_rulesOf] <;> rfl

/-- `Flock.PortRoots.checkOf` on a circuit's `Kept` and the public file's load `pub` (`portsOf_kept`). -/
def portsOf (k : V Kept) (pub : V Public) : Except String Unit :=
  match k with
  | .error e => .error (ProvedScope.refusal s!"a circuit that does not parse ({e})")
  | .ok k =>
    if !k.rows then .error (ProvedScope.refusal PortRoots.ROWS_WHY)
    else if k.named then .error (ProvedScope.refusal PortRoots.NAMES_WHY)
    else match pub with
      | .error e => .error (ProvedScope.refusal s!"a public file that does not load ({e})")
      | .ok pub => match PortRoots.loadedWhy Sha512.hash k.c pub with
        | some why => .error (ProvedScope.refusal why)
        | none => .ok ()

/-- **The ports' and roots' check on the parse is that check on its `Kept` and the public file's one load.** -/
theorem portsOf_kept (parsed : V Circuit) (tags : Tags) (publicFile : ByteArray) (zk : Bool) (count : Nat)
    (own : Circuit → V Unit) :
    PortRoots.checkOf parsed tags publicFile =
      portsOf (parsed.map (keptOf tags zk count own))
        (loadOf (parsed.map (keptOf tags zk count own)) tags publicFile) := by
  unfold PortRoots.checkOf PortRoots.checkWith portsOf
  cases parsed with
  | error e => rfl
  | ok c =>
    simp only [Except.map, keptOf, loadOf, loadPublic_bare, loadedWhy_bare] <;> rfl

/-- `Flock.KsScope.sharedWhy` on a circuit's `Kept`: a shared-row public file's form, on the public input ports the
circuit parsed with. -/
def sharedWhy (k : Kept) (tags : Tags) (publicFile : ByteArray) : Option String :=
  match k.inputs with
  | .error e => some s!"{KsScope.TABLES_WHY} whose circuit's public input ports do not parse ({e})"
  | .ok pi => match HmIn.loadPublic tags k.c pi publicFile with
    | .error e => some s!"{KsScope.TABLES_WHY} that does not load over the circuit's public input ports ({e})"
    | .ok pub => match pub.tables with
      | none => some KsScope.TABLES_WHY
      | some t => if KsScope.collidesT k.c pub t then some KsScope.ROOTS_WHY else none

theorem sharedWhy_kept (tags : Tags) (zk : Bool) (count : Nat) (own : Circuit → V Unit) (c : Circuit)
    (publicFile : ByteArray) :
    KsScope.sharedWhy tags c publicFile = sharedWhy (keptOf tags zk count own c) tags publicFile := by
  unfold KsScope.sharedWhy sharedWhy
  simp only [keptOf, loadPublicIn_bare, collidesT_bare] <;> rfl

/-- `Flock.KsScope.filesOf` on a circuit's `Kept` and the public file's load `pub` (`filesOf_kept`). -/
def filesOf (k : V Kept) (pub : V Public) (tags : Tags) (publicFile : ByteArray) : Except String Unit :=
  match KsScope.tagsWhy tags with
  | some why => .error (ProvedScope.refusal why)
  | none => match k with
    | .error e => .error (ProvedScope.refusal s!"a circuit that does not parse ({e})")
    | .ok k => match pub with
      | .error e => .error (ProvedScope.refusal s!"a public file that does not load ({e})")
      | .ok pub =>
        match (match k.unit with
          | some why => some why
          | none => if pub.tables.isSome then sharedWhy k tags publicFile else none) with
        | some why => .error (ProvedScope.refusal why)
        | none => .ok ()

/-- **The knowledge soundness form's check on the parse is that check on its `Kept` and the public file's one
load**, at the `count` tables the `Kept` is at. -/
theorem filesOf_kept (parsed : V Circuit) (tags : Tags) (publicFile : ByteArray) (zk : Bool) (count : Nat)
    (own : Circuit → V Unit) :
    KsScope.filesOf parsed tags publicFile count =
      filesOf (parsed.map (keptOf tags zk count own)) (loadOf (parsed.map (keptOf tags zk count own)) tags publicFile)
        tags publicFile := by
  unfold KsScope.filesOf filesOf
  cases KsScope.tagsWhy tags with
  | some why => rfl
  | none =>
    cases parsed with
    | error e => rfl
    | ok c =>
      simp only [Except.map, loadOf, KsScope.filesWhy, sharedWhy_kept tags zk count own c publicFile]
      simp only [keptOf, loadPublic_bare] <;> rfl

end Spot.CircuitKept
