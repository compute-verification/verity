module

public import Proofs.Flock.SpotFlock.SlotsAt
public import Proofs.Flock.SpotFlock.RowsOpen
public import Proofs.Flock.SpotFlock.NoAlways

/-!
# The row fact of the address premise: where round 2 puts a unit, and which row each of its ports reads

`Addresses h` (`SlotsAt`) asks of each gate's address, at every accepted run's checked unit that needs the gate, that
the unit's instance is one of its session's (`CheckedRun.Counted`) and that the slot's leaf under the gate's roots is
the position the session header gives the instance's row (`CheckedRun.ReadsRegistered`'s last clause). Both say where
round 2 puts a unit: at which instance of its template's session, and at which row of the tree each port reads. This
file states that round 2's checks put it there; at a proof-block partition side by side every needed gate takes the
registered branch (`CheckedRun.addressHolds_reads`), so the registered branch is the one stated.

* **The instance** (`InstanceFact`). A template's session statement lists every slot of its part as its population, in
  order (`second-population`, `SecondStage.populationOk`), and its sample is the second draw's slots in the part
  (`second-sample`, `SecondStage.ownDraw`). Flock sets the session up at the sample (`HmRow.drawn`), so instance `n` is
  the sample's `n`-th unit and is labelled with the population's label there, slot `lo + unitOf draw n` (`drawSlot`,
  `LabelledAt`). With no proof block always checked (`Held.noAlways`), a checked unit's instance (`rankAt`) is then
  counted, and the registration maps its slot to the unit (`RegTwo.blockAt`).
* **The row round 2 holds a slot to** (`placedRow`, `ReadFact`). At a port a link or a wire names, and at a template's
  own port of the interior record, `second-read` holds each slot's read to one row (`SecondStage.readOk`). At a port
  reading a value a registration registers (`ReadKind.registered`) it holds the read only to the value's root, domain
  and leaves, and `second-values` holds its rows to the reference statement's slot by slot; nothing holds them to a
  row fixed for the unit across runs, so `FallbackRows` is a hypothesis.
* **The header's position is the read's** (`PositionFact`): at any slot map that labels the session's instances
  (`LabelledAt σ`), the position the header gives instance `n`'s row is the registered read's at label `σ n`
  (`rows_open`). The slot map is a parameter: `drawSlot` at a staging of every slot of the part, another at a staging
  of the drawn slots alone. `ParsedFact`: the registered read Flock parses reads at a label what round 2's entry for its
  name reads there.
* **The slot's leaf** (`rowShape`, `LeafFact`): the slot's position, stride and units, fixed before the run from the
  auditor's copies (the unit's template, replay block and place in its cell, and the design's links, wires and rows)
  and the read's kind; under the roots round 2 reads the value under (`readRoots`), that slot's leaf is round 2's row
  at the unit's slot.
* `RowFact`: the four together, in the shape `CheckedRun.ReadsRegistered` asks: the unit's instance is counted, and
  where the address's slot has the read's shape its leaf is the header's position.

The read's kind is the run's (`kindAt`: it reads the interior record's names), as the entry the address reads a value
off is (`readKeyAt`). That it is the same at every accepted run is the entry fact's, not this file's.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

variable (h : Held)

/-! ## The read's kind and the row round 2 holds a slot to -/

/-- **How round 2 reads a registered read's value** at a template's session port (`SecondStage.readOk`'s cases): a
port a link names reads the boundary port the link names (`boundaryLink`) or the interior registration's value of the
link's name (`innerLink`); a port a wire names, its writer's output port of the interior record (`wired`); any other its
own port of the interior record (`ownPort`) or a value a registration registers (`registered`). -/
inductive ReadKind where
  | boundaryLink
  | innerLink
  | wired
  | ownPort
  | registered
  deriving DecidableEq, Repr

/-- **The kind of template `t`'s read of value `v` at port `port`**, as round 2 tells it with what the verifier holds of
the run (`own`). -/
def kindOf (own : Spot.SecondStage.Own) (t port v : String) : ReadKind :=
  match Spot.SecondStage.linkOf own t port, Spot.SecondStage.wireOf own t port with
  | some _, _ => if (Spot.SecondStage.registeredEntry own.interior v).isSome then .innerLink else .boundaryLink
  | none, some _ => .wired
  | none, none =>
    if ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome then .ownPort else .registered

/-- **The kind of template `t`'s read of value `v` at port `port`** in the run at statement `x`, round-1 message `m₁`,
round 0's coin and the second stage's (`kindOf` at `ownOf`, as `readRoots` reads the run). -/
def kindAt (x m₁ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (t : ℕ) (port v : String) : ReadKind :=
  match decode x, jsonMsg m₁, secondDraw h c₀ c₁ with
  | some st, some rec, some draw => kindOf (ownOf h st rec draw c₀) (h.names.getD t "") port v
  | _, _, _ => .registered

/-- **The row round 2 holds slot `u` of part `p` to**, at the part's template's session port `port`, reading value `v`
of `l` leaves (`SecondStage.readOk`): at an `innerLink` or a `wired` port, its unit's row of the value's `per` rows of
its cell's sampled block (`wiredRow`); at a `boundaryLink` port, its unit's row of its cell's replay block's `per`
rows of the boundary port (`linkedRow`); at an `ownPort`, its own first row, `l / (q·n)` rows a slot over the part's
`q·n` slots; none at a `registered` port, whose rows round 2 holds only to the reference statement's. -/
def placedRow (own : Spot.SecondStage.Own) (p : Spot.SecondStage.Part) (port v : String) (l u : ℕ) : Option ℕ :=
  match kindOf own p.template port v with
  | .innerLink | .wired => (Spot.SecondStage.rowsOf own p.template port).bind fun r => Spot.SecondStage.wiredRow p r u
  | .boundaryLink => (Spot.SecondStage.rowsOf own p.template port).bind fun r => Spot.SecondStage.linkedRow own p r u
  | .ownPort => some ((u - p.lo) * (l / (own.sample.size * p.n)))
  | .registered => none

/-- **The read of slot `u`** in registered-reads entry `e`, as round 2 reads it: slot `reads_from + k` reads `reads[k]`
(`reads_from` 0 where null); none where the entry gives the slot no position. -/
def readsAt (e : Json) (u : ℕ) : Option ℕ :=
  let first : Option ℕ := match Spot.getD e "reads_from" with | .null => some 0 | x => Spot.nat? x
  match Spot.getD e "reads", first with
  | .arr rs, some f => if f ≤ u then (rs[u - f]?).bind Spot.nat? else none
  | _, _ => none

/-- **Round 2's row is the read's** (`second-read`): where round 2 passes template `p.template`'s read `e` of value `v`
of `l` leaves at port `port` (`readOk`) and holds slot `u` to a row (`placedRow`), the read of slot `u` is that row. -/
def ReadFact : Prop :=
  ∀ (own : Spot.SecondStage.Own) (named : Bool) (p : Spot.SecondStage.Part) (port v : String) (e : Json) (l u row pos : ℕ),
    Spot.SecondStage.readOk own named p port e = true → Spot.getD e "value" = .str v →
      Spot.nat? (Spot.getD e "leaves") = some l → placedRow own p port v l u = some row → readsAt e u = some pos →
        pos = row

/-! ## The instance and its label -/

/-- **A slot map labels a session's instances**: the session header's unit index of each instance `n` of its first
table is `σ n`. -/
def LabelledAt (σ : ℕ → ℕ) (sts : Array _root_.Flock.Stmt) : Prop :=
  ∀ idx, _root_.Flock.Registered.unitIndices sts[0]!.pub = .ok idx → ∀ n < sts[0]!.pub.n, idx.getD n 0 = σ n

/-- **The header's position for an instance is the registered read's at its label**, at any slot map that labels the
session's instances: at a session statement that sets up as a masked Flock statement with registered ports `own`, in
Flock's scope with hidden outputs, and a Flock witness its relation accepts, each port `reg` of `own` reads at label
`σ n` the position the header gives instance `n`'s row (`rows_open`, `Registered.Port.readAt`). -/
def PositionFact : Prop :=
  ∀ {xS : Msg} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    {own : Array _root_.Flock.Registered.Port} {w : Claims.Flock.Wit} (σ : ℕ → ℕ),
    FlockDeploy.setupOf true xS = some (ins, spec, sts) → ins.registered = some own → ScopedHidden xS →
      Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS w = true → LabelledAt σ sts →
        ∀ reg ∈ own, ∀ q tb positions paths,
          (List.range sts[0]!.c.ports.size).find? (fun q => sts[0]!.c.ports[q]!.name == reg.name) = some q →
          sts[0]!.pub.tables = some tb → _root_.Flock.Registered.entry sts[0]!.pub reg = .ok (positions, paths) →
          ∀ n < sts[0]!.pub.n,
            reg.readAt (σ n) =
              some (some (positions.getD (_root_.Flock.Registered.ref tb sts[0]!.c.ports.size q n) 0))

/-- **Friday's slot map**: instance `n` of part `p`'s session is the slot of its sample's `n`-th unit, the session's
population being the part's slots in order. -/
def drawSlot (p : Spot.SecondStage.Part) (draw : Option Json) (n : ℕ) : ℕ := p.lo + unitOf draw n

/-- **The replay sample of a checked run**: the first stage's, on round 0's coin. -/
def sampleOf (r : CheckedRun h.stage) : Array ℕ := h.reg.sample (src r.c₀)

/-- **The checked unit's part**: its template's, in the run's replay sample (`partAt`). -/
def partOf (r : CheckedRun h.stage) : Spot.SecondStage.Part :=
  partAt h (sampleOf h r).size (templateOf h.ms r.u)

/-- **Round 2 puts each checked unit at a counted instance of a slot of its**: with no proof block always checked, at
an accepted run's checked unit, the unit's instance (`rankAt`) is one of its session's (`Counted`), the session's
instances are labelled by their slots (`LabelledAt` at `drawSlot`), and the instance's slot is one of its part's that
the registration maps to the unit (`RegTwo.blockAt`). -/
def InstanceFact : Prop :=
  (∀ v, h.reg.always v = false) → ∀ r : CheckedRun h.stage, r.Ok h →
    r.Counted h ∧ LabelledAt (drawSlot (partOf h r) r.ins.read.draw) r.sts ∧
      unitOf r.ins.read.draw (r.n h) < (sampleOf h r).size * (partOf h r).n ∧
      h.reg.blockAt (sampleOf h r) (drawSlot (partOf h r) r.ins.read.draw (r.n h)) = r.u

/-- **The registered read Flock parses reads what round 2's entry for its name reads**: at an accepted run's checked
unit, each registered read `reg` of its session reads at each label `u` the position that the entry of `reg`'s name
in the registered reads round 2 checks (`stmtJsonOf`) reads at slot `u` (`readsAt`), and that entry names `reg`'s value
and leaf count. -/
def ParsedFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ q reg, regNamed r.ins r.sts q = some reg →
    ∃ w S, FlockDeploy.decode (r.xS h) = some w ∧ stmtJsonOf h (templateOf h.ms r.u) w = some S ∧
      Spot.getD (Spot.getD S.registered "ports") reg.name ≠ .null ∧
      Spot.getD (Spot.getD (Spot.getD S.registered "ports") reg.name) "value" = .str reg.value ∧
      Spot.nat? (Spot.getD (Spot.getD (Spot.getD S.registered "ports") reg.name) "leaves") = some reg.leaves ∧
      ∀ u pos, reg.readAt u = some (some pos) →
        readsAt (Spot.getD (Spot.getD S.registered "ports") reg.name) u = some pos

/-! ## The slot's shape and its leaf -/

/-- **Where a slot's row is**, the part of a `Slot` its leaf reads: its position, stride and units. -/
structure RowShape where
  pos : ℕ
  stride : ℕ
  units : ℕ
  deriving DecidableEq, Repr

/-- **A slot has a shape**: its position, stride and units are the shape's, and it is not sized, as a registered read's
slot is not (`ReadsRegistered`). -/
def RowShape.Fits (sh : RowShape) (s : Slot) : Prop :=
  s.pos = sh.pos ∧ s.stride = sh.stride ∧ s.units = sh.units ∧ s.sized = false

/-- **What the auditor's copies say of the design**, as round 2 reads them (`SecondStage.Own`): its links, wires and the
rows their units read; nothing of a run. -/
def heldOwn : Spot.SecondStage.Own :=
  { sample := #[], parts := none, interior := .null, boundary := .null, drawn := #[], links := h.links,
    wires := h.wires, rows := h.rows }

/-- **Unit `u`'s place in its template's cell**: its index among its template's units, modulo the template's units a
replay block. -/
def cellPos (u : ℕ) : ℕ :=
  (u - Properties.Spot.unitBase h.ms (templateOf h.ms u)) % h.reg.ns.getD (templateOf h.ms u) 0

/-- **Unit `u`'s replay block**: its index among its template's units, divided by the template's units a replay
block. -/
def cellBlock (u : ℕ) : ℕ :=
  (u - Properties.Spot.unitBase h.ms (templateOf h.ms u)) / h.reg.ns.getD (templateOf h.ms u) 0

/-- **The shape of unit `u`'s slot at port `port`**, read of kind `k`, fixed before the run: from the unit's template
`t`, its replay block `ρ` and its place `j` in its cell, and the design's rows of the port (`per` a block, unit `j`
reading row `x`): `x` on a stride of `per` at an `innerLink` or a `wired` port (under the interior roots, row
`i·per + x` at the replay block's place `i`); `ρ·per + x` at a `boundaryLink` port (under the statement's roots); `j` on
a stride of `n_t` at an `ownPort` (unit `i·n_t + j`'s first row); and `R t port j` at a `registered` port, the row
`FallbackRows` names. -/
def rowShape (R : ℕ → String → ℕ → ℕ) (u : ℕ) (port : String) (k : ReadKind) : RowShape :=
  let t := templateOf h.ms u
  let j := cellPos h u
  let x := ((Spot.SecondStage.rowsOf (heldOwn h) (h.names.getD t "") port).bind fun r =>
    Spot.SecondStage.rowAt r.runs j).getD 0
  let per := ((Spot.SecondStage.rowsOf (heldOwn h) (h.names.getD t "") port).map (·.per)).getD 0
  match k with
  | .innerLink | .wired => ⟨x, per, 0⟩
  | .boundaryLink => ⟨cellBlock h u * per + x, 0, 0⟩
  | .ownPort => ⟨j, h.reg.ns.getD t 0, 0⟩
  | .registered => ⟨R t port j, 0, 0⟩

/-- **A read of a registered value reads a row fixed for the unit** (the row fact's one premise round 2 doesn't give):
at an accepted run's checked unit, each registered read of its session that reads a value a registration registers
(`ReadKind.registered`) reads at the instance's label the row `R t port j` of the unit's template `t`, the port and the
unit's place `j` in its cell, `R` fixed before the run. `second-values` holds those rows to the reference statement's
slot by slot, so they may differ from cell to cell and from run to run. -/
def FallbackRows (R : ℕ → String → ℕ → ℕ) : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ q reg, regNamed r.ins r.sts q = some reg →
    kindAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value = .registered →
      ∀ idx, _root_.Flock.Registered.unitIndices r.sts[0]!.pub = .ok idx →
        reg.readAt (idx.getD (r.n h) 0) = some (some (R (templateOf h.ms r.u) reg.name (cellPos h r.u)))

/-- **The slot's leaf is round 2's row** at the unit's slot: at an accepted run's checked unit with no proof block
always checked, for each registered read `reg` of its session of a kind other than `registered`, round 2 holds the
unit's slot to a row (`placedRow` at the run's `ownOf`), and a slot of the read's shape (`rowShape`), under the roots
round 2 reads the value under (`readRoots`), at an entry of the read's leaf count, opens that row. -/
def LeafFact (R : ℕ → String → ℕ → ℕ) : Prop :=
  (∀ v, h.reg.always v = false) → ∀ r : CheckedRun h.stage, r.Ok h → ∀ q reg, regNamed r.ins r.sts q = some reg →
    kindAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value ≠ .registered →
    ∃ st rec draw row, decode r.x = some st ∧ jsonMsg r.m₁ = some rec ∧ secondDraw h r.c₀ r.c₁ = some draw ∧
      placedRow (ownOf h st rec draw r.c₀) (partOf h r) reg.name reg.value reg.leaves
        (drawSlot (partOf h r) r.ins.read.draw (r.n h)) = some row ∧
      ∀ s : Slot, (rowShape h R r.u reg.name
          (kindAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value)).Fits s →
        ∀ e, entryAt (readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value) s = some e →
          e.leaves = reg.leaves →
          leafAt (readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value) s e = row

/-! ## The row fact -/

/-- **The row fact**, in the shape `CheckedRun.ReadsRegistered` asks: with no proof block always checked and the
registered values' rows fixed for each unit (`FallbackRows`), at every accepted run's checked unit, the unit's instance
is one of its session's (`Counted`), and for each address whose port reads registered read `reg` under the roots round 2
reads its value under (`readRoots`), at a slot of the read's shape (`rowShape`), the slot's leaf at an entry of the
read's leaf count is the position the session header gives the instance's row. -/
def RowFact (R : ℕ → String → ℕ → ℕ) : Prop :=
  (∀ v, h.reg.always v = false) → FallbackRows h R → ∀ r : CheckedRun h.stage, r.Ok h →
    r.Counted h ∧ ∀ (S : Address) (g : ℕ) reg, regNamed r.ins r.sts (S.port r.u) = some reg →
      r.roots h g = readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value →
      (rowShape h R r.u reg.name (kindAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value)).Fits S.slot →
      ∀ t positions paths, r.sts[0]!.pub.tables = some t →
        _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths) →
        ∀ e, entryAt (r.roots h g) S.slot = some e → e.leaves = reg.leaves →
          leafAt (r.roots h g) S.slot e =
            positions.getD (_root_.Flock.Registered.ref t r.sts[0]!.c.ports.size (S.port r.u) (r.n h)) 0

end AddressRow

end SpotFlock
