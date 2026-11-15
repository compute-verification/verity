module

public import Proofs.Flock.SpotFlock.EntryRoots
public import Proofs.Flock.SpotFlock.PortsMember

/-!
# The width fact at the address map

`widthFact_of_wiring`: `WidthFact` at the address map (`addressOf`), from what the entry and one tree a gate ask of
the held code (`Wiring`) and one fact of the members' programs, `BitsInPorts`.

A gate a checked unit needs is a bit of the unit's member `t` at row position `rp` (`NeededBits`). The address map's
port is `t`'s first port of `rp`'s row, and its place is the place of the gate the wiring map sends the bit to, of
member `tw` at row position `rq`: `rq`'s bit in a row of `16 w` bits, `w` the words of `tw`'s port of `rq`'s row,
where the bit is inside that row, and a one-bit row where it isn't (`placeAt`). The session's first table has its
template's member circuit's ports (`PortsMember`). So the width holds where the gate the bit is sent to is inside its
port's row (`BitsInPorts`) and that port has as many words as the bit's own (`widthAt_of_target`), which the three
cases of `holds_of_wiring` give:

* a source bit at a port a link names, and a written bit: the wiring map leaves it alone, and the two ports are one;
* a source bit at a port a wire names: the wiring map sends it to the same bit of the wire's writer's output port,
  as many words as the reader's port where each wire's reader port is like its writer's (`wiresAlike`), a condition
  the second stage's design decides on the registration.

`widthFact_addressOf` states the same at one condition on the registration in place of both, `RowWidths`, for a
registration whose wiring map is known.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart memberProg)

variable (h : Held)

/-- **Each commitment position's bit is inside its port's row**: each source bit's and each written bit's bit is
less than the `16 w` bits of the first port of its row's name in its member's circuit. -/
def BitsInPorts : Prop :=
  (∀ s ∈ sourceBits h.ms,
      s.2.2.bit < 16 * ((memberCircuit h s.2.1).ports[portNamed h s.2.1 s.2.2.row]!).words) ∧
    ∀ y ∈ writtenBits h.ms,
      y.2.2.bit < 16 * ((memberCircuit h y.2.1).ports[portNamed h y.2.1 y.2.2.row]!).words

/-- **Every wired row has one width**: at each bit `i` of a member `t` at row position `rp`, the gate the wiring map
sends it to has a commitment position `(tw, rq)`, `rq`'s bit is inside the `16 w` bits of `tw`'s port of `rq`'s row,
and that port has as many words as `t`'s port of `rp`'s row. It reads only the registration. -/
def RowWidths : Prop :=
  ∀ i t k rp, BitOf h t k i rp → ∃ tw rq, bitAtGate h (wireMap h i) = some (tw, rq) ∧
    rq.bit < 16 * ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words ∧
    ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words =
      ((memberCircuit h t).ports[portNamed h t rp.row]!).words

variable {h}

namespace CheckedRun

variable {r : CheckedRun h.stage}

/-- **The width at a needed gate sent to a bit inside a port as wide as its own**: where the gate the wiring map
sends the unit's bit to is at row position `rq` of member `tw`, inside its port's row, and that port has as many
words as the bit's own port. -/
theorem widthAt_of_target (hP : PortsMember h) (hr : r.Ok h) (hω : ∀ j, wireMap h (wireMap h j) = wireMap h j)
    {k i tw : ℕ} {rp rq : Spot.Bits.RowPos} (hb : BitOf h (templateOf h.ms r.u) k i rp)
    (hw : bitAtGate h (wireMap h i) = some (tw, rq))
    (hbit : rq.bit < 16 * ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words)
    (hwords : ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words =
      ((memberCircuit h (templateOf h.ms r.u)).ports[portNamed h (templateOf h.ms r.u) rp.row]!).words) :
    r.WidthAt h (addressOf h i) := by
  show (addressOf h i).place.1 = 16 * (r.sts[0]!.c.ports[(addressOf h i).port r.u]!).words
  rw [addressOf_eq h hω, addressAt_of_bitOf hb hw]
  show (placeAt h tw rq).1 = 16 * (r.sts[0]!.c.ports[portNamed h (templateOf h.ms r.u) rp.row]!).words
  rw [placeAt, dite_eq_left hbit]
  show 16 * ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words = _
  rw [hwords, hP r hr]

/-- **Where a needed gate's bit is sent, at an accepted run** (`holds_of_wiring`'s cases): the wiring map is
idempotent, the gate is a bit `rp` of the unit, and the gate the wiring map sends it to is at row position `rq` of a
member `tw`, at `rp`'s own bit, inside its port's row, of a port as wide as `rp`'s. -/
theorem target_of_wiring (W : Wiring h) (hB : BitsInPorts h) (hr : r.Ok h) {i : ℕ}
    (hi : r.s.prog.Needs r.b i) :
    (∀ j, wireMap h (wireMap h j) = wireMap h j) ∧ ∃ rp tw rq,
      BitOf h (templateOf h.ms r.u) (r.u - Properties.Spot.unitBase h.ms (templateOf h.ms r.u)) i rp ∧
      bitAtGate h (wireMap h i) = some (tw, rq) ∧ rq.bit = rp.bit ∧
      rq.bit < 16 * ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words ∧
      ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words =
        ((memberCircuit h (templateOf h.ms r.u)).ports[portNamed h (templateOf h.ms r.u) rp.row]!).words := by
  obtain ⟨st, rec, draw, hR⟩ := reads_of_ok hr
  obtain ⟨d, h2⟩ := hR.design
  have htop := topological_of_two h2
  refine ⟨(rewires_of_two h2 W.first).idem, ?_⟩
  obtain ⟨rp, hb⟩ := W.needed r hr i hi
  rcases hb.2 with ⟨hs, -⟩ | ⟨hs, hw, -, -⟩
  · -- a source bit
    have hsm : (i, templateOf h.ms r.u, rp) ∈ sourceBits h.ms := List.mem_of_find?_eq_some hs
    cases hl : heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row with
    | some l =>
      have hfix : wireMap h i = i := wireMap_of_no_target h (wireTarget_of_linked hs hl)
      have hin := hB.1 (i, templateOf h.ms r.u, rp) hsm
      exact ⟨rp, templateOf h.ms r.u, rp, hb, by rw [hfix]; exact bitAtGate_of_bitOf hb, rfl, hin, rfl⟩
    | none =>
      obtain ⟨y, hxy⟩ := wireTarget_of_sourcesNamed h W.named W.move hsm hl
      obtain ⟨tw, wport, u', hwu, -, hym, -⟩ := wireTarget_at hs hxy
      obtain ⟨w, -, hwire, htw, hwp⟩ := wireUnit_wire hwu
      have htw' : heldWriter h w.writer = some tw := htw
      have hwp' : wport = w.wport := hwp
      rw [hwp'] at hym
      have hbit : bitAtGate h (wireMap h i) = some (tw, Spot.Bits.RowPos.mk w.wport u' rp.bit) := by
        rw [wireMap_of_target h hxy]
        simp only [bitAtGate, source_none_of_written htop hym, written_find W.once hym, Option.map_some]
      have hin := hB.2 (y, tw, Spot.Bits.RowPos.mk w.wport u' rp.bit) hym
      exact ⟨rp, tw, Spot.Bits.RowPos.mk w.wport u' rp.bit, hb, hbit, rfl, hin,
        (wiresAlike_spec W.alike hR.lt hwire htw').1.symm⟩
  · -- a written bit
    have hwm : (i, templateOf h.ms r.u, rp) ∈ writtenBits h.ms := List.mem_of_find?_eq_some hw
    have hin := hB.2 (i, templateOf h.ms r.u, rp) hwm
    exact ⟨rp, templateOf h.ms r.u, rp, hb, by rw [wireMap_of_not_source h hs]; exact bitAtGate_of_bitOf hb, rfl,
      hin, rfl⟩

/-- **The width at every gate a checked unit needs**, at the address map. -/
theorem widthAt_of_wiring (W : Wiring h) (hB : BitsInPorts h) (hr : r.Ok h) {i : ℕ}
    (hi : r.s.prog.Needs r.b i) : r.WidthAt h (addressOf h i) := by
  obtain ⟨hω, rp, tw, rq, hb, hw, -, hbit, hwords⟩ := target_of_wiring W hB hr hi
  exact widthAt_of_target W.ports hr hω hb hw hbit hwords

end CheckedRun

variable (h) in
/-- **The width, at the address map, where every wired row has one width** (`RowWidths`), the wiring map sending
each gate it moves to a gate it leaves alone. -/
theorem widthFact_addressOf (hω : ∀ j, wireMap h (wireMap h j) = wireMap h j) (hN : NeededBits h)
    (hR : RowWidths h) : WidthFact h (addressOf h) := by
  intro r hr i hi
  obtain ⟨rp, hb⟩ := hN r hr i hi
  obtain ⟨tw, rq, hw, hbit, hwords⟩ := hR i _ _ rp hb
  exact CheckedRun.widthAt_of_target (portsMember h) hr hω hb hw hbit hwords

/-- **The width, at the address map**: at every accepted run's checked unit, the place of each needed gate's address
is as wide as the row of its port in the unit's session. -/
theorem widthFact_of_wiring (W : Wiring h) (hB : BitsInPorts h) : WidthFact h (addressOf h) :=
  fun _ hr _ hi => CheckedRun.widthAt_of_wiring W hB hr hi

end SpotFlock
