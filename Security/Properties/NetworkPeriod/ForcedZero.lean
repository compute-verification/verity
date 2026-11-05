import NetworkCheck.Period
import Flock.ProvedScope

/-!
# No pair of Δ fills a forced-zero column (`NetworkCheck.Period.forcedCols`)

`NetworkCheck.Period.forcedCols cf` lists the input columns of a circuit file's unit that no port holds and whose rows
are empty in A and in B (`forcedZero`). At a unit slot, the statement's level-0 row there is the unit's own row unless a
pair of Δ (`Flock.Stmt.da`, `Flock.Stmt.db`) has its row there; an empty row holds the column's value to `0 · 0`.
`ForcedZeroUnfilled` says that no statement the verifier sets up has such a pair: wherever the verifier's statement check
(`Flock.ProvedScope.check`, at no public input words of the verifier's own) accepts a session and `Flock.Stmt.setupTables`
sets it up, every table's circuit is the one `Flock.ProvedScope.circuitOf` parses, and no pair of its Δ_A or Δ_B has its
row at a forced-zero column of a unit slot (`c.unitRange.slot s + p`). Δ fills the slots' constants, the bits of the rows'
and the output rows' messages, which sit in other nets' slots, and the unit's input port bits, which its leaves, cuts and
wires copy into. A column no port holds is none of these. The statement check refuses a typed statement
(`ProvedScope.statementRules`) and the setup a pinned output (`HmRow.pinnedCount`), so neither's Δ is among these.

* `ForcedZeroUnfilledAt`: what it says of a session's tables.
* `ForcedZeroUnfilledOf check setup`: the statement at any statement check and session setup.
* `ForcedZeroUnfilled`: at the verifier's own (`ProvedScope.check`, `Stmt.setupTables`).
* `acceptAllScope`, `fillForced`: a statement check that accepts everything and a setup whose one table's Δ fills a
  forced-zero column (`forcedStmt`), where `ForcedZeroUnfilledOf` fails.

`Proofs.Properties.NetworkPeriod.ForcedZero` proves `ForcedZeroUnfilled` (`forcedZeroUnfilled`) and that
`ForcedZeroUnfilledOf` fails at `acceptAllScope` and `fillForced` (`forcedZeroUnfilled_nontrivial`).
-/

namespace Properties.NetworkPeriod

open Flock

/-- **A statement check**, with `Flock.ProvedScope.check`'s arguments: the tags, `--zk`, the circuit and public files,
the lookup tables, the table count, and whether the verifier holds public input words and registered reads of its own. -/
abbrev ScopeCheck : Type :=
  Tags → Bool → ByteArray → ByteArray → List (String × ByteArray) → Nat → Bool → Bool → Except String Unit

/-- **A session's setup**, with `Flock.Stmt.setupTables`'s arguments: the tags, the circuit and public files, the coins,
the table count, the lookup tables, the draw, the partition, the program, the verifier's own work table, stratified `K`
and closure map, and its public input words. -/
abbrev TablesSetup : Type :=
  Tags → ByteArray → ByteArray → Coins → Nat → List (String × ByteArray) → Option Lean.Json → Option Lean.Json →
    Option ByteArray → Option (Nat × Array (String × Nat × Nat)) → Option Nat → Option (Std.HashMap Nat (Array Nat)) →
    Option ByteArray → V (SessionSpec × Array Stmt)

/-- What `ForcedZeroUnfilled` says of a session set up from the tags `tags`, the circuit file `cf` and the lookup tables
`tables` as the tables `sts`: every table `st` has the circuit `Flock.ProvedScope.circuitOf` parses, and no pair of
`st`'s Δ_A or Δ_B has its row at column `p` of unit slot `s`, for `p` a forced-zero column of `st`'s circuit
(`NetworkCheck.Period.forcedCols`). -/
def ForcedZeroUnfilledAt (tags : Tags) (cf : ByteArray) (tables : List (String × ByteArray)) (sts : Array Stmt) :
    Prop :=
  ∀ st ∈ sts.toList, ProvedScope.circuitOf tags cf tables = .ok st.c ∧
    ∀ s < st.c.unitRange.count, ∀ p ∈ NetworkCheck.Period.forcedCols st.c,
      ∀ q ∈ st.da.toList ++ st.db.toList, q.1 ≠ st.c.unitRange.slot s + p

/-- `ForcedZeroUnfilled` of the statement check `check` and the setup `setup`: `ForcedZeroUnfilledAt` of every session
whose statement `check` accepts, holding no public input words of its own, and that `setup` sets up without them. -/
def ForcedZeroUnfilledOf (check : ScopeCheck) (setup : TablesSetup) : Prop :=
  ∀ (tags : Tags) (zk : Bool) (cf pf : ByteArray) (tables : List (String × ByteArray)) (count : Nat) (registered : Bool)
    (coins : Coins) (draw partition : Option Lean.Json) (program : Option ByteArray)
    (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (spec : SessionSpec) (sts : Array Stmt),
    check tags zk cf pf tables count false registered = .ok () →
    setup tags cf pf coins count tables draw partition program work stratified closure none = .ok (spec, sts) →
    ForcedZeroUnfilledAt tags cf tables sts

/-- **No pair of Δ fills a forced-zero column.** If the verifier's statement check (`Flock.ProvedScope.check`) accepts a
session's statement and its setup (`Flock.Stmt.setupTables`) sets the session up as the tables `sts`, every table `st`
has the circuit `Flock.ProvedScope.circuitOf` parses, and no pair of `st`'s Δ_A or Δ_B has its row at column `p` of unit
slot `s`, for `p` a forced-zero column of `st`'s circuit (`NetworkCheck.Period.forcedCols`): `ForcedZeroUnfilledOf` at
those two. Its two literals are the `--zk` verifier's call (`Flock.Witness.scope`, `Flock.Witness.setupOf`): `false` is
the check's `publicInputs`, the one value at which the check accepts (it refuses public input ports,
`ProvedScope.statementRules`), and `none` is the setup's public input words, which the verifier's setup leaves at their
default. The plain path's statement check (`Flock.ProvedScope.checkP`) is not covered. -/
def ForcedZeroUnfilled : Prop :=
  ∀ (tags : Tags) (zk : Bool) (cf pf : ByteArray) (tables : List (String × ByteArray)) (count : Nat) (registered : Bool)
    (coins : Coins) (draw partition : Option Lean.Json) (program : Option ByteArray)
    (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (spec : SessionSpec) (sts : Array Stmt),
    ProvedScope.check tags zk cf pf tables count false registered = .ok () →
    Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure none =
      .ok (spec, sts) →
    ForcedZeroUnfilledAt tags cf tables sts

/-- A statement check that accepts every statement: what `ForcedZeroUnfilledOf` must fail at, beside `fillForced`. -/
def acceptAllScope : ScopeCheck := fun _ _ _ _ _ _ _ _ => .ok ()

/-- A unit with one input word whose constant is at column 1, so column 0 is its one input column: read (A and B hold one
row), empty, and held by no port (it has no input groups). -/
def forcedNet : Net :=
  { (default : Net) with
    inWords := 1, constPos := 1, aT := Thunk.pure { (default : Sparse) with rows := 1 },
    bT := Thunk.pure { (default : Sparse) with rows := 1 } }

/-- A statement whose circuit's unit is `forcedNet`, in one slot at column 0, and whose Δ has the pair `(0, 0)`: its row
is the forced-zero column 0 of that slot. -/
def forcedStmt : Stmt :=
  { (default : Stmt) with
    c := { (default : Circuit) with
      nets := #[("unit", forcedNet)], unitNet := 0,
      ranges := #[{ kind := .net 0, slotLog := 0, pos0 := 0, count := 1, role := .none }] }
    da := #[(0, 0)] }

/-- A setup that sets every session up as the one table `forcedStmt`. -/
def fillForced : TablesSetup := fun _ _ _ _ _ _ _ _ _ _ _ _ _ => .ok (default, #[forcedStmt])

end Properties.NetworkPeriod
