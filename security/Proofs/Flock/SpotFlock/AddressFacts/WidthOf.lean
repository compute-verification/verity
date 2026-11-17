module

public import Proofs.Flock.SpotFlock.EntryRoots
public import Proofs.Flock.SpotFlock.PortsMember

/-! `WidthFact` at the address map (`widthFact_of_wiring`), from the wiring and `BitsInPorts`. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart memberProg)

variable (h : Held)

def BitsInPorts : Prop :=
  (∀ s ∈ sourceBits h.ms,
      s.2.2.bit < 16 * ((memberCircuit h s.2.1).ports[portNamed h s.2.1 s.2.2.row]!).words) ∧
    ∀ y ∈ writtenBits h.ms,
      y.2.2.bit < 16 * ((memberCircuit h y.2.1).ports[portNamed h y.2.1 y.2.2.row]!).words

def RowWidths : Prop :=
  ∀ i t k rp, BitOf h t k i rp → ∃ tw rq, bitAtGate h (wireMap h i) = some (tw, rq) ∧
    rq.bit < 16 * ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words ∧
    ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words =
      ((memberCircuit h t).ports[portNamed h t rp.row]!).words

variable {h}

namespace CheckedRun

variable {r : CheckedRun h.stage}

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
  ·
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
  ·
    have hwm : (i, templateOf h.ms r.u, rp) ∈ writtenBits h.ms := List.mem_of_find?_eq_some hw
    have hin := hB.2 (i, templateOf h.ms r.u, rp) hwm
    exact ⟨rp, templateOf h.ms r.u, rp, hb, by rw [wireMap_of_not_source h hs]; exact bitAtGate_of_bitOf hb, rfl,
      hin, rfl⟩

theorem widthAt_of_wiring (W : Wiring h) (hB : BitsInPorts h) (hr : r.Ok h) {i : ℕ}
    (hi : r.s.prog.Needs r.b i) : r.WidthAt h (addressOf h i) := by
  obtain ⟨hω, rp, tw, rq, hb, hw, -, hbit, hwords⟩ := target_of_wiring W hB hr hi
  exact widthAt_of_target W.ports hr hω hb hw hbit hwords

end CheckedRun

variable (h) in
theorem widthFact_addressOf (hω : ∀ j, wireMap h (wireMap h j) = wireMap h j) (hN : NeededBits h)
    (hR : RowWidths h) : WidthFact h (addressOf h) := by
  intro r hr i hi
  obtain ⟨rp, hb⟩ := hN r hr i hi
  obtain ⟨tw, rq, hw, hbit, hwords⟩ := hR i _ _ rp hb
  exact CheckedRun.widthAt_of_target (portsMember h) hr hω hb hw hbit hwords

theorem widthFact_of_wiring (W : Wiring h) (hB : BitsInPorts h) : WidthFact h (addressOf h) :=
  fun _ hr _ hi => CheckedRun.widthAt_of_wiring W hB hr hi

end SpotFlock
