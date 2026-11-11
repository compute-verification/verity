module

public import Proofs.Flock.SpotFlock.SlotsAt
public import Proofs.Flock.Claims.Sampling.Rewire

/-!
# Where each gate's value sits, at the wired program

`Addresses h` (`SlotsAt.lean`) reads the members' programs side by side, where it fails at every registration with
an accepted run (`AddressesAt.lean`): every gate a proof block needs is committed there, so the program opens it
under the statement's roots, while round 2 reads a session's own ports, and a wired read, off the interior record.

`AddressesWired h ω keep` is the same property read at the members' programs wired by `ω`, returning `keep`
(`Prog.wired`), each proof block and replay block cut to the gates `ω` leaves alone (`cutBlock`). A run and its
checked unit are the verifier's own, side by side (`CheckedRun.Ok`): the verifier's checks don't change. What changes
is which gates a unit needs and which roots each is opened under.

* **The gates.** A cut block needs gate `g` at the wired program only where the block needs, side by side, a gate
  `i` with `ω i = g` (`Prog.needs_wired`): `g` itself where the unit writes it or reads it as a block input, and the
  unit's own source gate `i` where the unit reads `g` through a wire. The unit's session holds `i`, so the address's
  row is read at `i` (`RowAt` at `i`) and the address is `g`'s.
* **The roots** (`CheckedRun.rootsWired`): the replay block's, where `g` is a gate of the replay block, cut, that the
  wired program doesn't commit as that block sees it; else the statement's. A written bit that only units of its own
  replay block read through wires, and that the wired program doesn't return, is such a gate; an input no wire moves
  and a returned output are not.
* **The slot.** One slot a gate serves every unit that needs it, the unit that writes it and each unit that reads
  it, so an address's slot is held to the entry round 2 reads, not to the way a session names it: under the gate's
  roots the slot names the entry the read's key names (`entryAt … = keyAt …`), with the read's leaf count
  (`leavesAt`) and leaf (`leafAt`). The writer's session reads its output port as template `t`'s port `p` of the
  interior record, and a wired reader's session reads the same entry as a registered read of the port named
  `portOf t p`; `Addresses` asked the slot's key and `sized` to be both.

At the wiring that moves nothing and returns everything, the property is `Addresses`'s with the slot held as above.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

namespace CheckedRun

variable (r : CheckedRun)

/-- **The roots the wired program opens gate `g` of the unit under**: its replay block's, where `g` is a gate of the
replay block, cut, that the wired program doesn't commit as that block sees it; the statement's otherwise
(`Reveal.rootsAt`, at the wired program and the cut replay block). -/
def rootsWired (g : ℕ) : List Msg :=
  if g ∈ cutBlock ω r.rb ∧ (r.s.prog.wired ω keep).Boundary (cutBlock ω r.rb) g = false then
    iroots r.x r.seen r.ρ
  else r.s.roots

/-- **The address's port reads a registered value, at the wired program's gate `g`**: a registered read `reg` is
named like the unit's port, it is the first port of that name, the roots the wired program opens `g` under are those
round 2 reads `reg`'s value under, and under those roots the address's slot names the entry round 2 reads the value
off, at `reg`'s schema; at an entry of `reg`'s leaf count the slot's leaf count is `reg`'s and its leaf (`leafAt`) is
the position the session header gives the instance's row. -/
def ReadsRegisteredWired (S : Address) (g : ℕ) : Prop :=
  ∃ reg, regNamed r.ins r.sts (S.port r.u) = some reg ∧
    (List.range r.sts[0]!.c.ports.size).find? (fun q => r.sts[0]!.c.ports[q]!.name == reg.name) = some (S.port r.u) ∧
    r.rootsWired ω keep g = readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value ∧
    entryAt (r.rootsWired ω keep g) S.slot =
      keyAt (r.rootsWired ω keep g) (readKeyAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value) ∧
    S.slot.schema = reg.schema ∧
    ∀ e, entryAt (r.rootsWired ω keep g) S.slot = some e → e.leaves = reg.leaves →
      leavesAt (r.rootsWired ω keep g) S.slot e = reg.leaves ∧
      ∀ t positions paths, r.sts[0]!.pub.tables = some t →
        _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths) →
        leafAt (r.rootsWired ω keep g) S.slot e =
          positions.getD (_root_.Flock.Registered.ref t r.sts[0]!.c.ports.size (S.port r.u) (r.n h)) 0

/-- **The address's port reads the interior record, at the wired program's gate `g`**: no registered read is named
like the unit's port, the session has one table, `g` is a gate of the unit's replay block, cut, that the wired
program doesn't commit as that block sees it, and under the replay block's roots the address's slot names the
template's port of the interior record; over any population the session's public file loads, the slot has the
published tree's schema, its leaf count is the tree's and its leaf (`leafAt`) the drawn unit's row. -/
def ReadsInteriorWired (S : Address) (g : ℕ) : Prop :=
  regNamed r.ins r.sts (S.port r.u) = none ∧ r.ins.read.I.count ≤ 1 ∧ S.port r.u < r.sts[0]!.c.ports.size ∧
    (g ∈ cutBlock ω r.rb ∧ (r.s.prog.wired ω keep).Boundary (cutBlock ω r.rb) g = false) ∧
    entryAt (iroots r.x r.seen r.ρ) S.slot =
      keyAt (iroots r.x r.seen r.ρ)
        (.interior (h.names.getD (templateOf h.ms r.u) "") (r.sts[0]!.c.ports[S.port r.u]!).name) ∧
    ∀ pop, _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[0]!.c r.ins.read.I.publicFile = .ok pop →
      S.slot.schema = (pubPort r.sts[0]!.c pop (S.port r.u)).schema ∧
      ∀ e, entryAt (iroots r.x r.seen r.ρ) S.slot = some e →
        leavesAt (iroots r.x r.seen r.ρ) S.slot e = (pubPort r.sts[0]!.c pop (S.port r.u)).leaves ∧
        leafAt (iroots r.x r.seen r.ρ) S.slot e =
          rowAt r.sts[0]!.c pop (S.port r.u) (unitOf r.ins.read.draw (r.n h))

/-- **The wired program's gate `g` sits at address `S` in the run, at the unit's own gate `i`**: the unit's instance
is one of its session's (`Counted`), the address's place reads the row the session holds `i` in (`RowAt` at `i`), and
the address's port reads `g`'s value as a registered read or off the interior record, under the roots the wired
program opens `g` under. -/
def AddressHoldsWired (S : Address) (i g : ℕ) : Prop :=
  r.Counted h ∧ r.RowAt h S i ∧ (r.ReadsRegisteredWired h ω keep S g ∨ r.ReadsInteriorWired h ω keep S g)

end CheckedRun

/-- **Every gate's value has one address at the wired program**, fixed before the run: at every accepted run's
checked unit whose proof block, cut, needs gate `g` at the members' programs wired by `ω`, the address holds
(`CheckedRun.AddressHoldsWired`) at each gate `i` the unit's block needs side by side that `ω` sends to `g`. -/
def AddressesWired : Prop :=
  ∀ g, ∃ S : Address, ∀ r : CheckedRun, r.Ok h → (r.s.prog.wired ω keep).Needs (cutBlock ω r.b) g →
    ∀ i, r.s.prog.Needs r.b i → ω i = g → r.AddressHoldsWired h ω keep S i g

/-! ## The entry a writer and its readers name -/

/-- **A template's port of a record and the port of that name are one entry**: under any roots, a slot whose key is
template `t`'s port `p` names the entry the key of the port named `portOf t p`, at the roots' naming, names. So one
slot serves the session that writes a port and each session that reads it as a registered read. -/
theorem entryAt_interior_port (roots : List Msg) (S : Slot) (t p : String) (hk : S.key = .interior t p) :
    entryAt roots S = keyAt roots (.port (Spot.SecondStage.portOf (namedOf roots) t p)) := by
  rw [entryAt, hk]
  rfl

/-! ## A connected instance: the gates and their roots

Two members, each an input and its negation and each one proof block, in one replay block: the second's input wired
to the first's output, the first's output read only there and the second's returned. -/

/-- The two members side by side: gates `0, 1` the first's, `2, 3` the second's; both outputs returned. -/
def wiredPair : Prog := ⟨[.input, .not 0, .input, .not 2], [1, 3]⟩

/-- The wire: the second member's input to the first's output. -/
def pairWire (i : ℕ) : ℕ := if i = 2 then 1 else i

/-- What the wired program returns: the second member's output. -/
def pairKeep (g : ℕ) : Bool := g == 3

/-- **What the two cut blocks need at the wired program**: the first its input and its output, the second its output
and the first's output, which it reads through the wire. -/
theorem wiredPair_needs :
    (wiredPair.wired pairWire pairKeep).needsList (cutBlock pairWire [0, 1]) = [0, 1] ∧
      (wiredPair.wired pairWire pairKeep).needsList (cutBlock pairWire [2, 3]) = [3, 1] := by
  decide

/-- **Which of them the cut replay block commits at the wired program**: the first member's input and the second's
output; the wired gate, the first's output, it does not, so both units that need it open it under the replay block's
roots. -/
theorem wiredPair_boundary :
    cutBlock pairWire [0, 1, 2, 3] = [0, 1, 3] ∧
      (wiredPair.wired pairWire pairKeep).Boundary [0, 1, 3] 0 = true ∧
      (wiredPair.wired pairWire pairKeep).Boundary [0, 1, 3] 1 = false ∧
      (wiredPair.wired pairWire pairKeep).Boundary [0, 1, 3] 3 = true := by
  decide

/-- **Side by side the replay block commits every gate a block needs**, the second member's input among them, so the
program opens that input under the statement's roots, where no record of the first stage holds it. -/
theorem wiredPair_sideBySide :
    wiredPair.needsList [0, 1] = [0, 1] ∧ wiredPair.needsList [2, 3] = [2, 3] ∧
      ∀ g ∈ [0, 1, 2, 3], wiredPair.Boundary [0, 1, 2, 3] g = true := by
  decide

/-- **The roots at the instance**: at a run whose statement's program is the two members' and whose unit's replay
block holds both, the wired program opens the wired gate under the replay block's roots, and the first member's input
and the second's output under the statement's. -/
theorem rootsWired_wiredPair (r : CheckedRun) (hp : r.s.prog = wiredPair) (hrb : r.rb = [0, 1, 2, 3]) :
    r.rootsWired pairWire pairKeep 1 = iroots r.x r.seen r.ρ ∧ r.rootsWired pairWire pairKeep 0 = r.s.roots ∧
      r.rootsWired pairWire pairKeep 3 = r.s.roots := by
  obtain ⟨hc, h0, h1, h3⟩ := wiredPair_boundary
  unfold CheckedRun.rootsWired
  rw [hp, hrb, hc]
  refine ⟨ite_eq_left ⟨by decide, h1⟩, ite_eq_right fun hx => ?_, ite_eq_right fun hx => ?_⟩
  · rw [h0] at hx
    exact absurd hx.2 (by decide)
  · rw [h3] at hx
    exact absurd hx.2 (by decide)

end SpotFlock
