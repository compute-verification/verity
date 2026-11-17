module

public import Proofs.Flock.SpotFlock.SlotsAt
public import Proofs.Flock.SpotFlock.RowsOpen
public import Proofs.Flock.SpotFlock.NoAlways

/-! The row fact of the address premise: where round 2 puts a unit and which row each of its ports reads; `RefRows`,
that the reference's rows repeat from cell to cell, is a hypothesis no check enforces. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

variable (h : Held)

inductive ReadKind where
  | boundaryLink
  | innerLink
  | wired
  | ownPort
  | registered
  deriving DecidableEq, Repr

def kindOf (own : Spot.SecondStage.Own) (t port v : String) : ReadKind :=
  match Spot.SecondStage.linkOf own t port, Spot.SecondStage.wireOf own t port with
  | some _, _ => if (Spot.SecondStage.registeredEntry own.interior v).isSome then .innerLink else .boundaryLink
  | none, some _ => .wired
  | none, none =>
    if ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome then .ownPort else .registered

def kindAt (x m₁ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (t : ℕ) (port v : String) : ReadKind :=
  match decode x, jsonMsg m₁, secondDraw h c₀ c₁ with
  | some st, some rec, some draw => kindOf (ownOf h st rec draw c₀) (h.names.getD t "") port v
  | _, _, _ => .registered

def placedRow (own : Spot.SecondStage.Own) (p : Spot.SecondStage.Part) (port v : String) (l u : ℕ) : Option ℕ :=
  match kindOf own p.template port v with
  | .innerLink | .wired => (Spot.SecondStage.rowsOf own p.template port).bind fun r => Spot.SecondStage.wiredRow p r u
  | .boundaryLink => (Spot.SecondStage.rowsOf own p.template port).bind fun r => Spot.SecondStage.linkedRow own p r u
  | .ownPort => some ((u - p.lo) * (l / (own.sample.size * p.n)))
  | .registered => none

def readsAt (e : Json) (u : ℕ) : Option ℕ :=
  let first : Option ℕ := match Spot.getD e "reads_from" with | .null => some 0 | x => Spot.nat? x
  match Spot.getD e "reads", first with
  | .arr rs, some f => if f ≤ u then (rs[u - f]?).bind Spot.nat? else none
  | _, _ => none

def ReadFact : Prop :=
  ∀ (own : Spot.SecondStage.Own) (named : Bool) (p : Spot.SecondStage.Part) (port v : String) (e : Json) (l u row pos : ℕ),
    Spot.SecondStage.readOk own named p port e = true → Spot.getD e "value" = .str v →
      Spot.nat? (Spot.getD e "leaves") = some l → placedRow own p port v l u = some row → readsAt e u = some pos →
        pos = row

def LabelledAt (σ : ℕ → ℕ) (sts : Array _root_.Flock.Stmt) : Prop :=
  ∀ idx, _root_.Flock.Registered.unitIndices sts[0]!.pub = .ok idx → ∀ n < sts[0]!.pub.n, idx.getD n 0 = σ n

def PositionFact : Prop :=
  ∀ {xS : Msg} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    {own : Array _root_.Flock.Registered.Port} {w : Claims.Flock.Wit} (σ : ℕ → ℕ),
    FlockDeploy.setupOf true xS = some (ins, spec, sts) → ins.registered = some own → ScopedHidden xS →
      Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS w = true → 0 < sts.size →
        LabelledAt σ sts →
        ∀ reg ∈ own, ∀ q tb positions paths,
          (List.range sts[0]!.c.ports.size).find? (fun q => sts[0]!.c.ports[q]!.name == reg.name) = some q →
          sts[0]!.pub.tables = some tb → _root_.Flock.Registered.entry sts[0]!.pub reg = .ok (positions, paths) →
          ∀ n < sts[0]!.pub.n,
            reg.readAt (σ n) =
              some (some (positions.getD (_root_.Flock.Registered.ref tb sts[0]!.c.ports.size q n) 0))

def sampleSlot (p : Spot.SecondStage.Part) (draw : Option Json) (n : ℕ) : ℕ := p.lo + unitOf draw n

def sampleOf (r : CheckedRun h.stage) : Array ℕ := h.reg.sample (src r.c₀)

def partOf (r : CheckedRun h.stage) : Spot.SecondStage.Part :=
  partAt h (sampleOf h r).size (templateOf h.ms r.u)

def InstanceFact : Prop :=
  (∀ v, h.reg.always v = false) → ∀ r : CheckedRun h.stage, r.Ok h →
    r.Counted h ∧ LabelledAt (sampleSlot (partOf h r) r.ins.read.draw) r.sts ∧
      unitOf r.ins.read.draw (r.n h) < (sampleOf h r).size * (partOf h r).n ∧
      h.reg.blockAt (sampleOf h r) (sampleSlot (partOf h r) r.ins.read.draw (r.n h)) = r.u

def ParsedFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ q reg, regNamed r.ins r.sts q = some reg →
    ∃ w S, FlockDeploy.decode (r.xS h) = some w ∧ stmtJsonOf h (templateOf h.ms r.u) w = some S ∧
      Spot.getD (Spot.getD S.registered "ports") reg.name ≠ .null ∧
      Spot.getD (Spot.getD (Spot.getD S.registered "ports") reg.name) "value" = .str reg.value ∧
      Spot.nat? (Spot.getD (Spot.getD (Spot.getD S.registered "ports") reg.name) "leaves") = some reg.leaves ∧
      ∀ u pos, reg.readAt u = some (some pos) →
        readsAt (Spot.getD (Spot.getD S.registered "ports") reg.name) u = some pos

structure RowShape where
  pos : ℕ
  stride : ℕ
  units : ℕ
  deriving DecidableEq, Repr

def RowShape.Fits (sh : RowShape) (s : Slot) : Prop :=
  s.pos = sh.pos ∧ s.stride = sh.stride ∧ s.units = sh.units ∧ s.sized = false

def heldOwn : Spot.SecondStage.Own :=
  { sample := #[], parts := none, interior := .null, boundary := .null, drawn := #[], links := h.links,
    wires := h.wires, rows := h.rows }

def cellPos (u : ℕ) : ℕ :=
  (u - Properties.Spot.unitBase h.ms (templateOf h.ms u)) % h.reg.ns.getD (templateOf h.ms u) 0

def cellBlock (u : ℕ) : ℕ :=
  (u - Properties.Spot.unitBase h.ms (templateOf h.ms u)) / h.reg.ns.getD (templateOf h.ms u) 0

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

/-- A read of a registered value reads at the instance's label a row `R t port j` fixed before the run; it follows
from `RefRows` (`FallbackFact`). -/
def FallbackRows (R : ℕ → String → ℕ → ℕ) : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ q reg, regNamed r.ins r.sts q = some reg →
    kindAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value = .registered →
      ∀ idx, _root_.Flock.Registered.unitIndices r.sts[0]!.pub = .ok idx →
        reg.readAt (idx.getD (r.n h) 0) = some (some (R (templateOf h.ms r.u) reg.name (cellPos h r.u)))

def kindHeld (t : ℕ) (port v : String) : ReadKind :=
  let own := heldOwn h
  let name := h.names.getD t ""
  match Spot.SecondStage.linkOf own name port, Spot.SecondStage.wireOf own name port with
  | some _, _ => if h.interiorNames.values.contains v then .innerLink else .boundaryLink
  | none, some _ => .wired
  | none, none => if h.interiorNames.ports.contains v then .ownPort else .registered

def KindFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ t < h.ms.length, ∀ port v,
    kindAt h r.x r.m₁ r.c₀ r.c₁ t port v = kindHeld h t port v

def refEntry (t : ℕ) (port : String) : Json := Spot.getD (Spot.getD (h.refReads t) "ports") port

/-- The row fact's one premise round 2 doesn't give, a property of the auditor's copies alone: no check holds the
reference entry's rows to repeat from cell to cell, which a positional table's don't. -/
def RefRows (R : ℕ → String → ℕ → ℕ) : Prop :=
  ∀ t < h.ms.length, ∀ port v, Spot.getD (refEntry h t port) "value" = .str v → kindHeld h t port v = .registered →
    ∀ c u, (partAt h (h.reg.sample c).size t).lo ≤ u →
      u < (partAt h (h.reg.sample c).size t).lo + (h.reg.sample c).size * (partAt h (h.reg.sample c).size t).n →
        readsAt (refEntry h t port) u =
          some (R t port ((u - (partAt h (h.reg.sample c).size t).lo) % (partAt h (h.reg.sample c).size t).n))

def FallbackFact (R : ℕ → String → ℕ → ℕ) : Prop :=
  (∀ v, h.reg.always v = false) → RefRows h R → FallbackRows h R

def LeafFact (R : ℕ → String → ℕ → ℕ) : Prop :=
  (∀ v, h.reg.always v = false) → ∀ r : CheckedRun h.stage, r.Ok h → ∀ q reg, regNamed r.ins r.sts q = some reg →
    kindAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value ≠ .registered →
    ∃ st rec draw row, decode r.x = some st ∧ jsonMsg r.m₁ = some rec ∧ secondDraw h r.c₀ r.c₁ = some draw ∧
      placedRow (ownOf h st rec draw r.c₀) (partOf h r) reg.name reg.value reg.leaves
        (sampleSlot (partOf h r) r.ins.read.draw (r.n h)) = some row ∧
      ∀ s : Slot, (rowShape h R r.u reg.name
          (kindAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value)).Fits s →
        ∀ e, entryAt (readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value) s = some e →
          e.leaves = reg.leaves →
          leafAt (readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value) s e = row

def RowFact (R : ℕ → String → ℕ → ℕ) : Prop :=
  (∀ v, h.reg.always v = false) → RefRows h R → ∀ r : CheckedRun h.stage, r.Ok h →
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
