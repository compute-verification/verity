module

public import Proofs.Flock.SpotFlock.SlotsAt
public import Proofs.Flock.SpotFlock.WireMap
public import Proofs.Flock.Claims.Sampling.Rewire

/-!
# Where each gate's value sits, at the wired program

`Addresses h` (`SlotsAt.lean`) reads the members' programs side by side. Every gate a proof block needs is committed
there, so the program opens it under the statement's roots, while round 2 reads a session's own ports, and a wired
read, off the interior record: one accepted run with a checked unit all of whose session's reads round 2 reads off
the interior record refutes it (`AddressesAt.lean`).

`AddressesWired h ω keep` is the same property read at the members' programs wired by `ω`, returning `keep`
(`Prog.wired`), each proof block and replay block without the gates `ω` moves (`cutBlock`). A run and its checked
unit are the verifier's own, side by side (`CheckedRun.Ok`): the verifier's checks don't change. What changes is
which gates a unit needs and which roots each is opened under.

* **The gates.** A proof block needs gate `g` at the wired program only where it needs, side by side, a gate
  `i` with `ω i = g` (`Prog.needs_wired`): `g` itself where the unit writes it or reads it as a block input, and the
  unit's own source gate `i` where the unit reads `g` through a wire. The unit's session holds `i`, so an address is
  a side-by-side gate's: its row is read at `i` (`RowAt` at `i`), in the port that holds `i`, and its place and slot
  are those of `ω i`. Two gates of one unit that the wiring sends to one gate each have their own port and the one
  place and slot.
* **The roots** (`CheckedRun.rootsWired`): the replay block's, where `g` is a gate of the replay block that `ω`
  leaves alone and the wired program doesn't commit as that block sees it; else the statement's. A written bit that
  only units of its own replay block read through wires, and that the wired program doesn't return, is such a gate;
  an input no wire moves and a returned output are not.
* **The slot.** One slot a gate serves every unit that needs it, the unit that writes it and each unit that reads
  it, so an address's slot is held to the entry round 2 reads, not to the way a session names it: under the gate's
  roots the slot names the entry the read's key names (`entryAt … = keyAt …`), with the read's leaf count
  (`leavesAt`) and leaf (`leafAt`). The writer's session reads its output port as template `t`'s port `p` of the
  interior record, and a wired reader's session reads the same entry as a registered read of the port named
  `portOf t p`; `Addresses` asked the slot's key and `sized` to be both.

At the wiring that moves nothing and returns everything, `Addresses` gives the property (`addressesWired_id`): it asks
no more than `Addresses` of a design with no wire.

The property is a premise only where every source bit of a member is at a port a link or a wire names
(`sourcesNamed`, decided on what the verifier holds): round 2 reads any other port off the interior record, and the
wired program opens its gates under the statement's roots. `AddressesConnected h` is the premise in that form: the
property at the code's own wiring map (`wireMap h`, returning `returned h`), asked only of such a held code.
-/

@[expose] public section

namespace Claims.Sampling

/-- A gate rewired by the map that moves nothing is itself. -/
theorem BGate.rewire_id (x : BGate) : x.rewire id = x := by
  cases x <;> simp [BGate.rewire]

/-- **The wiring that moves nothing and returns everything leaves a program as it is.** -/
theorem Prog.wired_id (C : Prog) : C.wired id (fun _ => true) = C := by
  cases C with
  | mk gates outs =>
    have hg : gates.map (BGate.rewire id) = gates :=
      (List.map_congr_left fun x _ => BGate.rewire_id x).trans (List.map_id _)
    have ho : outs.filter (fun _ => true) = outs := List.filter_eq_self.2 fun _ _ => rfl
    show (⟨gates.map (BGate.rewire id), outs.filter fun _ => true⟩ : Prog) = ⟨gates, outs⟩
    rw [hg, ho]

/-- The map that moves nothing removes no gate of a set. -/
theorem cutBlock_id (u : List ℕ) : cutBlock id u = u := by
  unfold cutBlock
  exact List.filter_eq_self.2 fun g _ => decide_eq_true rfl

end Claims.Sampling

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

namespace CheckedRun

variable (r : CheckedRun h.stage)

variable {h} in
/-- **The roots the wired program opens gate `g` of the unit under**: its replay block's, where `g` is a gate of the
replay block that `ω` leaves alone and the wired program doesn't commit as that block sees it; the statement's
otherwise (`Reveal.rootsAt`, at the wired program and the replay block without the gates `ω` moves). -/
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
like the unit's port, the session has one table, `g` is a gate of the unit's replay block that `ω` leaves alone and
the wired program doesn't commit as that block sees it, and under the replay block's roots the address's slot names the
template's port of the interior record; over any population the session's public file loads, the slot has the
published tree's schema, its leaf count is the tree's and its leaf (`leafAt`) the sampled proof block's row. -/
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

/-- **Every gate's value has one address at the wired program**, fixed before the run. Each gate `i` of the members'
programs side by side has an address: the port that holds it in its unit's session, its place in that port's row,
and the frame's slot for the row. A gate the wiring moves has the place and the slot of the gate it is sent to: the
two are one committed bit. And at every accepted run's checked unit, at each gate `i` its proof block needs side by
side, the address holds (`CheckedRun.AddressHoldsWired`) under the roots the wired program opens `ω i` under.

A unit that reads one written gate at two of its own gates has an address for each, in its own port, at the one
place and slot; so has a unit that reads it and the unit that writes it. -/
def AddressesWired : Prop :=
  ∃ addr : ℕ → Address,
    (∀ i, (addr i).place = (addr (ω i)).place ∧ (addr i).slot = (addr (ω i)).slot) ∧
      ∀ r : CheckedRun h.stage, r.Ok h →
        ∀ i, r.s.prog.Needs r.b i → r.AddressHoldsWired h ω keep (addr i) i (ω i)

/-! ## The entry a writer and its readers name -/

/-- **A template's port of a record and the port of that name are one entry**: under any roots, a slot whose key is
template `t`'s port `p` names the entry the key of the port named `portOf t p`, at the roots' naming, names. So one
slot serves the session that writes a port and each session that reads it as a registered read. -/
theorem entryAt_interior_port (roots : List Msg) (S : Slot) (t p : String) (hk : S.key = .interior t p) :
    entryAt roots S = keyAt roots (.port (Spot.SecondStage.portOf (namedOf roots) t p)) := by
  rw [entryAt, hk]
  rfl

/-! ## The wiring that moves nothing -/

namespace CheckedRun

variable {h} {r : CheckedRun h.stage}

/-- **At the wiring that moves nothing and returns everything, the wired program opens a gate under the roots the
members' programs side by side open it under** (`Reveal.rootsAt_eq`). -/
theorem rootsWired_id (hr : r.Ok h) (g : ℕ) : r.rootsWired id (fun _ => true) g = r.roots h g := by
  obtain ⟨Pt, d, spec, -, hP, h2, -, -, -, -, -, hρ, hrb, -⟩ := hr
  have he : r.roots h g =
      if g ∈ r.rb ∧ r.s.prog.Boundary r.rb g = false then iroots r.x r.seen r.ρ else r.s.roots :=
    Reveal.rootsAt_eq (partition h) (two h) (iroots []) hP h2 hρ hrb
  rw [he]
  unfold rootsWired
  rw [cutBlock_id, Prog.wired_id]

/-- **An address that holds side by side holds at the wiring that moves nothing and returns everything**, at the
gate itself: a slot whose key is the read's names the read's entry, and its leaf count is the entry's where it is
not sized and its units' where it is. -/
theorem addressHoldsWired_id (hr : r.Ok h) {S : Address} {g : ℕ} (hH : r.AddressHolds h S g) :
    r.AddressHoldsWired h id (fun _ => true) S g g := by
  obtain ⟨hn, hR, hrd⟩ := hH
  have hroots := rootsWired_id hr g
  refine ⟨hn, hR, hrd.imp ?_ ?_⟩
  · rintro ⟨reg, hreg, hfirst, hrt, hkey, hsch, hsz, hleaf⟩
    refine ⟨reg, hreg, hfirst, hroots.trans hrt, congrArg (keyAt (r.rootsWired id (fun _ => true) g)) hkey, hsch,
      fun e he hl => ?_⟩
    rw [hroots] at he ⊢
    refine ⟨?_, fun t positions paths ht hent => hleaf t positions paths ht hent e he hl⟩
    unfold leavesAt
    rw [hsz]
    exact hl
  · rintro ⟨hnone, hc1, hq, hin, hkey, hsz, hpop⟩
    refine ⟨hnone, hc1, hq, ?_, congrArg (keyAt (iroots r.x r.seen r.ρ)) hkey, fun pop hp => ?_⟩
    · rw [cutBlock_id, Prog.wired_id]
      exact hin
    · obtain ⟨hs, hu, hl⟩ := hpop pop hp
      refine ⟨hs, fun e he => ⟨?_, hl e he⟩⟩
      unfold leavesAt
      rw [hsz]
      exact hu

end CheckedRun

/-- **Where `Addresses` holds, the property holds at the wiring that moves nothing and returns everything**, each
gate at the address `Addresses` gives it. So the property asks no more than `Addresses` of a design with no wire. -/
theorem addressesWired_id (hA : Addresses h) : AddressesWired h id (fun _ => true) :=
  ⟨fun g => (hA g).choose, fun _ => ⟨rfl, rfl⟩, fun r hr i hi =>
    CheckedRun.addressHoldsWired_id hr ((hA i).choose_spec r hr hi)⟩

/-! ## Where the property is a premise: every source bit at a port a link or a wire names -/

/-- **Every source bit is at a port a link or a wire names**, decided on what the verifier holds: each source gate
of a member that has a commitment position (`sourceBits`) is at a port the auditor's copies give a link or a wire
(`heldLink`, `heldWire`). Round 2 reads any other port at its own port of the interior record
(`SecondStage.readOk`), under the replay block's roots, while the wired program leaves its gates inputs, opened
under the statement's roots (`CheckedRun.rootsWired`). `AddressesWired` asks the two to be one, so it is a premise
only where this holds. -/
def sourcesNamed : Bool :=
  (sourceBits h.ms).all fun s =>
    (heldLink h (h.names.getD s.2.1 "") s.2.2.row).isSome ||
      (heldWire h (h.names.getD s.2.1 "") s.2.2.row).isSome

/-- **Only source bits at a port a link names stay inputs**: where every source bit is at a port a link or a wire
names and every wired source bit has a gate to go to (`wiresMove`), a source bit at a port no link names has one
(`wireTarget`), so the wiring map moves it. -/
theorem wireTarget_of_sourcesNamed (hn : sourcesNamed h = true) (hc : wiresMove h = true) {x t : ℕ}
    {r : Spot.Bits.RowPos} (hs : (x, t, r) ∈ sourceBits h.ms) (hl : heldLink h (h.names.getD t "") r.row = none) :
    ∃ y, wireTarget h x = some y := by
  have hall : ((heldLink h (h.names.getD t "") r.row).isSome ||
      (heldWire h (h.names.getD t "") r.row).isSome) = true := List.all_eq_true.1 hn (x, t, r) hs
  rw [hl] at hall
  cases hw : heldWire h (h.names.getD t "") r.row with
  | none =>
    rw [hw] at hall
    simp at hall
  | some w => exact wireTarget_of_wiresMove h hc hs hl hw

/-- **The premise at the code's own wiring**: where every source bit is at a port a link or a wire names
(`sourcesNamed`), the property holds at the wiring map the code's wires give (`wireMap`), returning the gates the
boundary record holds (`returned`). Of a held code with a source bit at a port neither names it asks nothing:
round 2 reads such a port off the interior record and the wired program opens its gate under the statement's roots
(`sourcesNamed`). So a guarantee that takes this premise holds the code to `sourcesNamed` itself. -/
def AddressesConnected : Prop :=
  sourcesNamed h = true → AddressesWired h (wireMap h) (returned h)

/-! ## A connected instance: the gates and their roots

Three members in one replay block, each one proof block: two writers, each an input and its negation, and a reader
of three inputs and their conjunction. The reader's first and third inputs are wired to the first writer's output
and its second to the second writer's; the writers' outputs are read only there and the reader's is returned. So one
written gate is read by one unit at two of its own gates, and the two writers' outputs are read by one reader. -/

/-- The three members side by side: gates `0, 1` the first writer's, `2, 3` the second's, `4` to `7` the reader's;
every output returned. -/
def fanProg : Prog := ⟨[.input, .not 0, .input, .not 2, .input, .input, .input, .and [4, 5, 6]], [1, 3, 7]⟩

/-- The wires: the reader's first and third inputs to the first writer's output, its second to the second's. -/
def fanWire (i : ℕ) : ℕ := if i = 4 then 1 else if i = 5 then 3 else if i = 6 then 1 else i

/-- What the wired program returns: the reader's output. -/
def fanKeep (g : ℕ) : Bool := g == 7

/-- **What the three proof blocks need at the wired program**: each writer its input and its output, the reader its
output and the two writers' outputs, the first's twice, once for each of its own gates wired to it. -/
theorem fanProg_needs :
    (fanProg.wired fanWire fanKeep).needsList (cutBlock fanWire [0, 1]) = [0, 1] ∧
      (fanProg.wired fanWire fanKeep).needsList (cutBlock fanWire [2, 3]) = [2, 3] ∧
      (fanProg.wired fanWire fanKeep).needsList (cutBlock fanWire [4, 5, 6, 7]) = [7, 1, 3, 1] := by
  decide

/-- **Which of them the replay block commits at the wired program**: the writers' inputs and the reader's
output; the two wired gates, the writers' outputs, it does not, so every unit that needs one opens it under the
replay block's roots. -/
theorem fanProg_boundary :
    cutBlock fanWire [0, 1, 2, 3, 4, 5, 6, 7] = [0, 1, 2, 3, 7] ∧
      (fanProg.wired fanWire fanKeep).Boundary [0, 1, 2, 3, 7] 0 = true ∧
      (fanProg.wired fanWire fanKeep).Boundary [0, 1, 2, 3, 7] 1 = false ∧
      (fanProg.wired fanWire fanKeep).Boundary [0, 1, 2, 3, 7] 2 = true ∧
      (fanProg.wired fanWire fanKeep).Boundary [0, 1, 2, 3, 7] 3 = false ∧
      (fanProg.wired fanWire fanKeep).Boundary [0, 1, 2, 3, 7] 7 = true := by
  decide

/-- **Side by side the replay block commits every gate a block needs**, the reader's three inputs among them, so the
program opens those inputs under the statement's roots, where no record of the first stage holds them. -/
theorem fanProg_sideBySide :
    fanProg.needsList [0, 1] = [0, 1] ∧ fanProg.needsList [2, 3] = [2, 3] ∧
      fanProg.needsList [4, 5, 6, 7] = [4, 5, 6, 7] ∧
      ∀ g ∈ [0, 1, 2, 3, 4, 5, 6, 7], fanProg.Boundary [0, 1, 2, 3, 4, 5, 6, 7] g = true := by
  decide

/-- **The roots at the instance**: at a run whose statement's program is the three members' and whose unit's replay
block holds them, the wired program opens the two wired gates under the replay block's roots, and the writers'
inputs and the reader's output under the statement's. -/
theorem rootsWired_fanProg (r : CheckedRun h.stage) (hp : r.s.prog = fanProg)
    (hrb : r.rb = [0, 1, 2, 3, 4, 5, 6, 7]) :
    r.rootsWired fanWire fanKeep 1 = iroots r.x r.seen r.ρ ∧ r.rootsWired fanWire fanKeep 3 = iroots r.x r.seen r.ρ ∧
      r.rootsWired fanWire fanKeep 0 = r.s.roots ∧ r.rootsWired fanWire fanKeep 2 = r.s.roots ∧
      r.rootsWired fanWire fanKeep 7 = r.s.roots := by
  obtain ⟨hc, h0, h1, h2, h3, h7⟩ := fanProg_boundary
  unfold CheckedRun.rootsWired
  rw [hp, hrb, hc]
  refine ⟨ite_eq_left ⟨by decide, h1⟩, ite_eq_left ⟨by decide, h3⟩, ite_eq_right fun hx => ?_,
    ite_eq_right fun hx => ?_, ite_eq_right fun hx => ?_⟩
  · rw [h0] at hx
    exact absurd hx.2 (by decide)
  · rw [h2] at hx
    exact absurd hx.2 (by decide)
  · rw [h7] at hx
    exact absurd hx.2 (by decide)

end SpotFlock
