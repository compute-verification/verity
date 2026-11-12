module

public import Proofs.Flock.SpotFlock.SlotsAt

/-!
# The width fact of an address

`RowAt`'s first fact (`CheckedRun.RowAt`): an address's place is as wide as the row of its port in the unit's session's
first table, `16 w` bits for a port of `w` words. `Addresses` asks it at every accepted run's checked unit that needs
the gate, with one place for all of them (`WidthHolds`).

`CheckedRun.ports_member`: at a checked unit of an accepted run, the session's first table has its template's member
circuit's ports, since the pin holds the session's circuit to its template's reference statement's
(`pin_of_stmtOkAt`, `stmt_of_pin`), whose statement circuit is the member's (`templateOk_facts`). So a session port's
width is fixed before the run, by the template and the port alone (`portWords`), and the width fact is a property of
the registration and the address (`WidthRef`): every unit that needs the gate names, in its template's member circuit,
a port of the place's width. `widthHolds_of_ref` gives `WidthHolds` from it.

No check of the verifier's decides `WidthRef` today. It reads only the members and the address, so a check can.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **Port `q`'s words in template `t`'s member circuit**; `0` past the members or the ports. -/
def portWords (t q : ℕ) : ℕ := ((h.ms[t]?).map fun m => (m.1.ports[q]!).words).getD 0

/-- **The width fact at gate `g`'s address `S`**: at every accepted run's checked unit that needs `g`, the address's place
is as wide as its port's row in the unit's session's first table (`CheckedRun.RowAt`'s first fact). -/
def WidthHolds (S : Address) (g : ℕ) : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g →
    S.place.1 = 16 * (r.sts[0]!.c.ports[S.port r.u]!).words

/-- **The width fact as the registration states it**, over a set `U` of units: at each unit of `U`, the address's place is
as wide as its port in the unit's template's member circuit (`portWords`). -/
def WidthRef (S : Address) (U : ℕ → Prop) : Prop :=
  ∀ u, U u → S.place.1 = 16 * portWords h (templateOf h.ms u) (S.port u)

namespace CheckedRun

variable {h} {r : CheckedRun h.stage}

/-- **An accepted run's checked unit's session has its template's member circuit's ports** in its first table: the
table's circuit is the one the scope check parses (`accepted_check`), which unrolls, ports and all (`unrolled_ports`), to
its template's reference statement's statement circuit (`pin_of_stmtOkAt`, `stmt_of_pin`), the member's
(`templateOk_facts`). -/
theorem ports_member (hr : r.Ok h) :
    ∃ m, h.ms[templateOf h.ms r.u]? = some m ∧ r.sts[0]!.c.ports = m.1.ports := by
  have hj := sts_pos hr
  obtain ⟨Pt, d, _, hx, hP, h2, hin, hok, hu, hck, hb, hρ, -, hset⟩ := hr
  have hokAt := stmtOkAt_unit h hx hok hu
  have hT := fun t (ht : t < h.ms.length) => templateOk_facts h (templateOk_of_two h h2 ht) ht
  obtain ⟨t, ht, k, hut, -, -⟩ :=
    shape_of_checked h (fun t ht => ⟨(hT t ht).2.1, (hT t ht).2.2⟩) hx hP h2 hin hb hρ
  have htu : templateOf h.ms r.u = t := by
    rw [hut]
    exact templateOf_unitBase ht k.isLt
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h r.m₂ r.u) with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain ⟨p, hpb, -, hpp, hc, hh⟩ := parsed_of_stmtOkAt h hokAt hS
    obtain ⟨rfl, -⟩ := parsed_setupOf hset hpb
    obtain ⟨w, w₀, hw, hw₀, he⟩ := pin_of_stmtOkAt h hokAt
    rw [htu] at hw₀ ⊢
    have hst₀ := (hT t ht).1
    cases hr₀ : FlockDeploy.readOf true (h.xS₀ t) with
    | none => rw [hr₀] at hst₀; cases hst₀
    | some r₀ =>
      rw [hr₀] at hst₀
      have hm : r₀.stmt = h.ms[t].1 := by simpa using hst₀
      have hst := stmt_of_pin hw hw₀ he (readOf_of_parseBytes hpb) hr₀
      obtain ⟨T, a, hst0, -, hcirc, -⟩ := accepted_check hpp hc hh hj
      refine ⟨h.ms[t], List.getElem?_eq_getElem ht, ?_⟩
      calc r.sts[0]!.c.ports = a.st.c.ports := by rw [(getElem!_pos r.sts 0 hj).trans hst0.symm]
        _ = (_root_.Flock.HmOut.unrolled a.st.c).ports := (FlockSoundness.unrolled_ports a.st.c).symm
        _ = r₀.stmt.ports := by
          rw [← hcirc]
          exact congrArg _root_.Flock.Circuit.ports hst
        _ = h.ms[t].1.ports := by rw [hm]

/-- **A session port's words are its template's member circuit's** (`ports_member`). -/
theorem words_eq (hr : r.Ok h) (q : ℕ) :
    (r.sts[0]!.c.ports[q]!).words = portWords h (templateOf h.ms r.u) q := by
  obtain ⟨m, hm, hp⟩ := ports_member hr
  rw [portWords, hm, hp]
  rfl

end CheckedRun

/-- **The width fact from the registration's**: where every accepted run's checked unit that needs `g` is in `U`, and the
address's place is as wide as its port in each unit of `U`'s template's member circuit, it is as wide as its port's row
in every such unit's session (`CheckedRun.words_eq`). -/
theorem widthHolds_of_ref {S : Address} {g : ℕ} {U : ℕ → Prop}
    (hU : ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → U r.u) (hw : WidthRef h S U) :
    WidthHolds h S g := by
  intro r hr hg
  rw [hw r.u (hU r hr hg), CheckedRun.words_eq hr]

end SpotFlock
