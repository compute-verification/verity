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

end SpotFlock
