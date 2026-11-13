module

public import Proofs.Flock.SpotFlock.AddressFacts.Width
public import Proofs.Flock.SpotFlock.NeededBits

/-!
# The width fact at the address map of a held code

`widthFact_addressOf`: `WidthFact` at `addressOf h`, from `NeededBits` and one condition on the registration alone,
`RowWidths`.

At a gate a checked unit needs, a bit of the unit's member `t` at row position `rp` (`NeededBits`), the address map's
port is `t`'s first port of `rp`'s row, and its place is the place of the gate the wiring map sends the bit to, of
member `tw` at row position `rq`: `rq`'s bit in a row of `16 w` bits, `w` the words of `tw`'s port of `rq`'s row
(`addressAt_of_bitOf`, `placeAt`). The session's first table has its template's member circuit's ports
(`CheckedRun.ports_member`, `CheckedRun.memberCircuit_eq`). So the width holds exactly where the bit the gate is sent
to is inside its port's row and that port has as many words as the bit's own port (`RowWidths`). For a gate the wiring
map leaves alone the two ports are one, and only the bit's range is asked; for a wired source bit it asks the reader's
port and its writer's output port to have one width, which no check of the verifier's decides today
(`SecondWires.wiresWide` asks only that the writer write every bit of the reader's port).

`widthFact_addressOf_wires` splits `RowWidths` in two (`rowWidths_of`): `wireWordsOk`, decided, that every wire's
reader port has its writer's output port's words; and `RowTargets`, that the gate a needed bit is sent to has a
position inside its port's row, at the bit's own row or at the output port of the writer of the wire naming it.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **Every wired row has one width**: at each bit `i` of a member `t` at row position `rp`, the gate the wiring map
sends it to has a commitment position `(tw, rq)`, `rq`'s bit is inside the `16 w` bits of `tw`'s port of `rq`'s row,
and that port has as many words as `t`'s port of `rp`'s row. It reads only the registration. -/
def RowWidths : Prop :=
  ∀ i t k rp, BitOf h t k i rp → ∃ tw rq, bitAtGate h (wireMap h i) = some (tw, rq) ∧
    rq.bit < 16 * ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words ∧
    ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words =
      ((memberCircuit h t).ports[portNamed h t rp.row]!).words

/-- **The words of member `t`'s first port named `port`.** -/
def memberWords (t : ℕ) (port : String) : ℕ := ((memberCircuit h t).ports[portNamed h t port]!).words

/-- **Every wire's reader port has its writer's output port's words**, decided: at each member `t`, each wire whose
reader is `t`'s name and whose writer names a template `tw` has `tw`'s output port as many words as `t`'s port. -/
def wireWordsOk : Bool :=
  (List.range h.ms.length).all fun t => (h.wires.getD #[]).all fun w =>
    !(w.template == h.names.getD t "") ||
      (heldWriter h w.writer).all fun tw => memberWords h tw w.wport == memberWords h t w.port

/-- **Where each needed bit's row goes under the wiring map**: the gate the wiring map sends a bit of member `t` at
`rp` to has a commitment position `(tw, rq)` whose bit is inside its port's row; and it is the bit's own row, or the
output port of the writer of the wire that names the bit's port. -/
def RowTargets : Prop :=
  ∀ i t k rp, BitOf h t k i rp → t < h.ms.length ∧ ∃ tw rq, bitAtGate h (wireMap h i) = some (tw, rq) ∧
    rq.bit < 16 * memberWords h tw rq.row ∧
    ((tw = t ∧ rq.row = rp.row) ∨
      ∃ w, heldWire h (h.names.getD t "") rp.row = some w ∧ heldWriter h w.writer = some tw ∧ rq.row = w.wport)

variable {h}

theorem memberWords_of_ok (hok : wireWordsOk h = true) {t : ℕ} {port : String} {w : Spot.SecondStage.Wire}
    {tw : ℕ} (ht : t < h.ms.length) (hw : heldWire h (h.names.getD t "") port = some w)
    (htw : heldWriter h w.writer = some tw) : memberWords h tw w.wport = memberWords h t port := by
  have hm : w ∈ h.wires.getD #[] := Array.mem_of_find?_eq_some hw
  have hp := Array.find?_some hw
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  obtain ⟨h1, h2⟩ := hp
  have hall := Array.all_eq_true_iff_forall_mem.1 (List.all_eq_true.1 hok t (List.mem_range.2 ht)) w hm
  simpa [h1, htw, h2] using hall

/-- **`RowWidths` from the wires' words and the rows' targets.** -/
theorem rowWidths_of (hok : wireWordsOk h = true) (hT : RowTargets h) : RowWidths h := by
  intro i t k rp hb
  obtain ⟨ht, tw, rq, hg, hbit, hrow⟩ := hT i t k rp hb
  refine ⟨tw, rq, hg, hbit, ?_⟩
  rcases hrow with ⟨rfl, hr⟩ | ⟨w, hw, htw, hr⟩
  · rw [hr]
  · show memberWords h tw rq.row = memberWords h t rp.row
    rw [hr]
    exact memberWords_of_ok hok ht hw htw

variable (h)

namespace CheckedRun

variable {h} {r : CheckedRun h.stage}

/-- **An accepted run's checked unit's template's member circuit is its reference statement's** (`memberCircuit`):
the reference statement's statement circuit is the member's (`templateOk_facts`). -/
theorem memberCircuit_eq (hr : r.Ok h) :
    ∃ m, h.ms[templateOf h.ms r.u]? = some m ∧ memberCircuit h (templateOf h.ms r.u) = m.1 := by
  obtain ⟨Pt, d, _, hx, hP, h2, hin, hok, hu, hck, hb, hρ, -, hset⟩ := hr
  have hT := fun t (ht : t < h.ms.length) => templateOk_facts h (templateOk_of_two h h2 ht) ht
  obtain ⟨t, ht, k, hut, -, -⟩ :=
    shape_of_checked h (fun t ht => ⟨(hT t ht).2.1, (hT t ht).2.2⟩) hx hP h2 hin hb hρ
  have htu : templateOf h.ms r.u = t := by
    rw [hut]
    exact templateOf_unitBase ht k.isLt
  rw [htu]
  refine ⟨h.ms[t], List.getElem?_eq_getElem ht, ?_⟩
  unfold memberCircuit stmtOf
  rw [(hT t ht).1]
  rfl

end CheckedRun

/-- **The width fact at the address map** (the module's docstring), where the wiring map sends each gate it moves to
a gate it leaves alone. -/
theorem widthFact_addressOf (hω : ∀ j, wireMap h (wireMap h j) = wireMap h j) (hN : NeededBits h)
    (hR : RowWidths h) : WidthFact h (addressOf h) := by
  intro r hr i hi
  obtain ⟨rp, hb⟩ := hN r hr i hi
  obtain ⟨tw, rq, hw, hbit, hwords⟩ := hR i _ _ rp hb
  obtain ⟨m, hm, hp⟩ := CheckedRun.ports_member hr
  obtain ⟨m', hm', hc⟩ := CheckedRun.memberCircuit_eq hr
  rw [Option.some.inj (hm'.symm.trans hm)] at hc
  show (addressOf h i).place.1 = 16 * (r.sts[0]!.c.ports[(addressOf h i).port r.u]!).words
  rw [addressOf_eq h hω, addressAt_of_bitOf hb hw]
  show (placeAt h tw rq).1 =
    16 * (r.sts[0]!.c.ports[portNamed h (templateOf h.ms r.u) rp.row]!).words
  rw [placeAt, dif_pos hbit]
  show 16 * ((memberCircuit h tw).ports[portNamed h tw rq.row]!).words = _
  rw [hwords, hp, ← hc]

/-- **The width fact at the address map from the wires' words** (`wireWordsOk`, decided) and where the rows go
(`RowTargets`). -/
theorem widthFact_addressOf_wires (hω : ∀ j, wireMap h (wireMap h j) = wireMap h j) (hN : NeededBits h)
    (hok : wireWordsOk h = true) (hT : RowTargets h) : WidthFact h (addressOf h) :=
  widthFact_addressOf h hω hN (rowWidths_of hok hT)

end SpotFlock
