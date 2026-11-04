import Definitions.NetworkCertifier.SampledCharge
import Definitions.NetworkCertifier.SampledReg

/-!
# A rows-form phase as one Program: its scan's and its tied compute audits', and the ties between them

Definitions only. In the rows form a phase's scan reads its `prog` from compute audits' slot entries, and the state each
slot's Program reads is the scan's own, so the egress a phase renders is one computation over several sampled
statements: its scan's and its tied audits'. Counted statement by statement, a value one statement reads from another
is a registered input at a value fixed before the run, which it is not: it varies with what the server did. A phase is
counted instead as one Program (`Phase.prog`): its statements' Programs side by side, the scan's first, with the ties
among its wires (`Phase.layout`), so a tied gate is no free input (`freeInputs`).

**The ties** (`Ties`), in the joint Program's gate numbers, each wiring a gate to its writer:

* `state`: each tied audit's state read at row `a ≥ 1` to the scan's `o1` of unit `a − 1`, and the scan's state read
  at unit `t ≥ 1` to its `o1` of unit `t − 1`. A state read at row `0` (`start`) is no tie: it is a registered input,
  the empty state, a value fixed before the run;
* `prog`: the scan's `prog` read at lane `l`, leaf `b` to its tied audit's slot entry of chunk `l`, leaf `b`.

Beside them each statement keeps its own wires (`Phase.ownWires`), which its sampled proofs show right (`KSWired`). A
tie is right by binding instead: its reader and its writer are committed as one bit of one leaf under one root
(`Placed`, at the registrations' placement `place`, bit by bit), so two openings of them that differ make a collision
(`SharedRootsParsed`, the deployed parse's premise, which the sampled leg's bridge discharges).

**What an accepted period gives**, each a named `Prop` that a property of the period check discharges:

* `RowsStateHeld` and `RowsProgWired`: the state and the `prog` ties are placed together;
* `RowsStateEmpty`: the state reads at row `0` are registered inputs that the registration opens at the empty state;
* `RowsCircuitHeld`: each tied audit's Program is the class circuit's, one Program for the period;
* `RowsTiesOrdered`: the phase's layout fits its joint Program (`BoundaryGates.Fits`, over `DependsWired`): the wires
  acyclic, and no step's egress reaching a later step's ingress through reads and wires, which holds when each slot
  reads the state of its boundary's first row;
* `CircuitCounts`: the check's count of free inputs is at least the joint Program's, less the registered inputs.

That a tied audit's every state read is tied, not some, is no separate `Prop`: a state read left untied is a free input
of the joint Program, which `CircuitCounts` then charges.
-/

namespace NetTiming

open Claims Claims.Sampling

/-- Gate `x` laid out `o` gates later: every wire it reads moved up by `o`. -/
def shiftGate (o : ℕ) : BGate → BGate
  | .input => .input
  | .const b => .const b
  | .not i => .not (i + o)
  | .and ws => .and (ws.map (· + o))
  | .xor ws => .xor (ws.map (· + o))

/-- **Programs side by side**: the first's gates, then the rest's, each laid out past the gates before it, and the gates
each returns. -/
def sideBySide : List Prog → Prog
  | [] => ⟨[], []⟩
  | C :: Cs =>
    let R := sideBySide Cs
    ⟨C.gates ++ R.gates.map (shiftGate C.gates.length), C.outs ++ R.outs.map (· + C.gates.length)⟩

#guard (sideBySide [⟨[.input, .not 0], [1]⟩, ⟨[.input, .and [0, 0]], [1]⟩]).gates ==
  [.input, .not 0, .input, .and [2, 2]]
#guard (sideBySide [⟨[.input, .not 0], [1]⟩, ⟨[.input, .and [0, 0]], [1]⟩]).outs == [1, 3]

/-- **A rows-form phase's statements**: its scan's and its tied compute audits', in that order. -/
structure Phase where
  scan : AuditStatement
  tied : List AuditStatement

/-- The phase's statements in layout order, the scan's first. -/
def Phase.stmts (ph : Phase) : List AuditStatement := ph.scan :: ph.tied

/-- **The phase's joint Program**: its statements' Programs side by side. -/
def Phase.prog (ph : Phase) : Prog := sideBySide (ph.stmts.map (·.prog))

/-- The joint Program's number of statement `j`'s first gate. -/
def Phase.offset (ph : Phase) (j : ℕ) : ℕ := ((ph.stmts.take j).map (·.prog.gates.length)).sum

/-- The statement that holds the joint Program's gate `g`, and `g`'s number in it. -/
def Phase.locate (ph : Phase) (g : ℕ) : Option (ℕ × ℕ) :=
  (List.range ph.stmts.length).findSome? fun j =>
    if ph.offset j ≤ g ∧ g < ph.offset (j + 1) then some (j, g - ph.offset j) else none

/-- Each statement's own wires, its layout's (`BoundaryGates.wires`), in the joint Program's gate numbers. -/
def Phase.ownWires {L S : ℕ} (ph : Phase) (lay : AuditStatement → BoundaryGates L S) (g : ℕ) : Option ℕ :=
  (ph.locate g).bind fun (j, h) => (ph.stmts[j]?).bind fun s => ((lay s).wires h).map (· + ph.offset j)

open Classical in
/-- **The phase's registered inputs**, in the joint Program's gate numbers: each statement's own (`reg s`). -/
noncomputable def Phase.reg (ph : Phase) (reg : AuditStatement → Finset ℕ) : Finset ℕ :=
  (Finset.range ph.prog.gates.length).filter fun g =>
    ∃ j h s, ph.locate g = some (j, h) ∧ ph.stmts[j]? = some s ∧ h ∈ reg s

/-- **The registration's value** at the joint Program's gate `g`: that of the registration of the statement that holds
it (`regVal`, its openings `ops s`), `false` off the phase's gates. -/
def Phase.regVal (ph : Phase) (ops : AuditStatement → List (Opening Bool)) (g : ℕ) : Bool :=
  ((ph.locate g).bind fun (j, h) => (ph.stmts[j]?).map fun s => NetTiming.regVal (ops s) h).getD false

/-- **A phase's ties**, in the joint Program's gate numbers: each tied gate's writer, the state reads' (`state`) and the
scan's `prog` reads' (`prog`), and the state reads at row `0`, which no tie writes (`start`). -/
structure Ties where
  state : ℕ → Option ℕ
  prog : ℕ → Option ℕ
  start : Finset ℕ

/-- **The phase's layout**: the scan's egress and ingress gates (laid out first, the scan's gates keep their numbers),
and as its wires each statement's own and the ties. -/
def Phase.layout {L S : ℕ} (ph : Phase) (lay : AuditStatement → BoundaryGates L S) (t : Ties) : BoundaryGates L S where
  egress := (lay ph.scan).egress
  ingress := (lay ph.scan).ingress
  wires g := ph.ownWires lay g <|> t.state g <|> t.prog g

/-- **Where the registrations commit each statement's gates**, bit by bit: `place s g = some (r, l, b)` when statement
`s`'s registration commits gate `g`'s value as bit `b` of leaf `l` under the root `r`. Two gates at one leaf are at two
bits of it. The position is the bit's place, built from `at_` and the deployed placement under `SampledProofs.Bits`
(`RowPos`: row, leaf, bit), never from `at_` alone. -/
abbrev Placement := AuditStatement → ℕ → Option (Msg × ℕ × ℕ)

/-- **The wires `t` are placed together**: each joins a gate of the phase to a writer that the registrations commit as
the same bit of the same leaf under the same root. -/
def Placed (ph : Phase) (place : Placement) (t : ℕ → Option ℕ) : Prop :=
  ∀ g w, t g = some w → ∃ (j h j' h' : ℕ) (s s' : AuditStatement) (p : Msg × ℕ × ℕ),
    ph.locate g = some (j, h) ∧ ph.locate w = some (j', h') ∧ ph.stmts[j]? = some s ∧ ph.stmts[j']? = some s' ∧
      place s h = some p ∧ place s' h' = some p

/-- **SharedRootsParsed**: at the deployment `D`, two gates of statements it parses that the registrations commit as one
bit of one leaf under one root (`place`) open to one value, or the two openings' `H` calls collide. A tie is right by
binding only if the deployment's parse puts each statement's registered roots among its `roots`, each gate at the bit
the registration names, so that two passing openings of them are openings of one committed bit. It is a fact about the
deployed parse, which the sampled leg's bridge proves at the deployed record, and no assumption about the period check,
which checks the placements (`RowsStateHeld`, `RowsProgWired`). -/
def SharedRootsParsed {Coin Col T : Type} (H : Msg → Msg) (D : Deploy Coin Col T) (place : Placement) : Prop :=
  ∀ (x x' : Msg) (s s' : AuditStatement) (g g' : ℕ) (p : Msg × ℕ × ℕ), D.parse x = some s → D.parse x' = some s' →
    place s g = some p → place s' g' = some p →
      ∀ o o' : Opening Bool, o.pos = g → o'.pos = g' → o.val ≠ o'.val → D.commit.Opens H s.roots o = true →
        D.commit.Opens H s'.roots o' = true →
          LogColl (((D.commit.opens s.roots o).eval H).2 ++ ((D.commit.opens s'.roots o').eval H).2)

/-- **RowsStateHeld**: the phase's state ties are placed together. Each tied audit's state read at row `a ≥ 1` is
committed where the scan's `o1` of unit `a − 1` is, under the scan's state root, and so is the scan's own state read at
unit `a`. -/
def RowsStateHeld (ph : Phase) (place : Placement) (t : Ties) : Prop := Placed ph place t.state

/-- **RowsProgWired**: the phase's `prog` ties are placed together. The scan's `prog` read at lane `l`, leaf `b` is
committed where its tied audit's slot entry of chunk `l`, leaf `b` is, under that audit's entries root. -/
def RowsProgWired (ph : Phase) (place : Placement) (t : Ties) : Prop := Placed ph place t.prog

/-- **RowsStateEmpty**: the phase's state reads at row `0` (`start`) are registered inputs of their statements (`reg`),
which the registration (its openings `ops`) opens at the empty state, every bit `false`. -/
def RowsStateEmpty (ph : Phase) (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool))
    (t : Ties) : Prop :=
  ∀ g ∈ t.start, g ∈ ph.reg reg ∧ ph.regVal ops g = false

/-- **RowsCircuitHeld**: each tied audit's Program is the class circuit `C`, the one Program the workload's class
registers for the period, so the prover chooses no phase's circuit. -/
def RowsCircuitHeld (ph : Phase) (C : Prog) : Prop := ∀ s ∈ ph.tied, s.prog = C

/-- **RowsTiesOrdered**: the phase's layout fits its joint Program over the horizon `k` (`BoundaryGates.Fits`): its
wires, each statement's own and the ties, are acyclic, and no step's egress depends through reads and wires on a later
step's ingress. -/
def RowsTiesOrdered {L S : ℕ} (ph : Phase) (lay : AuditStatement → BoundaryGates L S) (t : Ties) (k : ℕ) : Prop :=
  (ph.layout lay t).Fits ph.prog k

/-- **CircuitCounts**: `free`, the period check's count of the phase's free inputs, is at least the joint Program's free
inputs over the horizon `k` less the registered ones `reg` (`freeInputs`, which leaves out the ingress and every wired
gate, the tied ones among them). -/
def CircuitCounts {L S : ℕ} (ph : Phase) (lay : AuditStatement → BoundaryGates L S) (t : Ties) (k : ℕ)
    (reg : Finset ℕ) (free : ℕ) : Prop :=
  (freeInputs ph.prog (ph.layout lay t) k \ reg).card ≤ free

end NetTiming
