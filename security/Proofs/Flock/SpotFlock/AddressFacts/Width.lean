module

public import Proofs.Flock.SpotFlock.AddressFacts

/-! The width fact of an address (`widthFact_of_ref`), from `WidthRef`, a condition on the registration that no check
of the verifier's decides. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

def portWords (t q : ℕ) : ℕ := ((h.ms[t]?).map fun m => (m.1.ports[q]!).words).getD 0

def WidthHolds (S : Address) (g : ℕ) : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g →
    S.place.1 = 16 * (r.sts[0]!.c.ports[S.port r.u]!).words

def WidthRef (S : Address) (U : ℕ → Prop) : Prop :=
  ∀ u, U u → S.place.1 = 16 * portWords h (templateOf h.ms u) (S.port u)

namespace CheckedRun

variable {h} {r : CheckedRun h.stage}

theorem stmt_member (hr : r.Ok h) :
    ∃ m, h.ms[templateOf h.ms r.u]? = some m ∧ r.p.toRead.stmt = m.1 := by
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
    obtain ⟨p, hpb, -, -, -, -⟩ := parsed_of_stmtOkAt h hokAt hS
    obtain ⟨rfl, -⟩ := parsed_setupOf hset hpb
    obtain ⟨w, w₀, hw, hw₀, he⟩ := pin_of_stmtOkAt h hokAt
    rw [htu] at hw₀ ⊢
    have hst₀ := (hT t ht).1
    cases hr₀ : FlockDeploy.readOf true (h.xS₀ t) with
    | none => rw [hr₀] at hst₀; cases hst₀
    | some r₀ =>
      rw [hr₀] at hst₀
      have hm : r₀.stmt = h.ms[t].1 := by simpa using hst₀
      exact ⟨h.ms[t], List.getElem?_eq_getElem ht, (stmt_of_pin hw hw₀ he (readOf_of_parseBytes hpb) hr₀).trans hm⟩

theorem ports_member (hr : r.Ok h) :
    ∃ m, h.ms[templateOf h.ms r.u]? = some m ∧ r.sts[0]!.c.ports = m.1.ports := by
  have hj := sts_pos hr
  obtain ⟨m, hm, hs⟩ := stmt_member hr
  obtain ⟨_, _, _, hx, -, -, -, hok, hu, hck, -, -, -, hset⟩ := hr
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h r.m₂ r.u) with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain ⟨p, hpb, -, hpp, hc, hh⟩ := parsed_of_stmtOkAt h (stmtOkAt_unit h hx hok hu) hS
    obtain ⟨rfl, -⟩ := parsed_setupOf hset hpb
    obtain ⟨T, a, hst0, -, hcirc, -⟩ := accepted_check hpp hc hh hj
    refine ⟨m, hm, ?_⟩
    calc r.sts[0]!.c.ports = a.st.c.ports := by rw [(getElem!_pos r.sts 0 hj).trans hst0.symm]
      _ = (_root_.Flock.HmOut.unrolled a.st.c).ports := (FlockSoundness.unrolled_ports a.st.c).symm
      _ = m.1.ports := by
        rw [← hcirc]
        exact congrArg _root_.Flock.Circuit.ports hs

theorem words_eq (hr : r.Ok h) (q : ℕ) :
    (r.sts[0]!.c.ports[q]!).words = portWords h (templateOf h.ms r.u) q := by
  obtain ⟨m, hm, hp⟩ := ports_member hr
  rw [portWords, hm, hp]
  rfl

end CheckedRun

theorem widthHolds_of_ref {S : Address} {g : ℕ} {U : ℕ → Prop}
    (hU : ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → U r.u) (hw : WidthRef h S U) :
    WidthHolds h S g := by
  intro r hr hg
  rw [hw r.u (hU r hr hg), CheckedRun.words_eq hr]

theorem widthFact_of_ref {addr : ℕ → Address} {U : ℕ → ℕ → Prop}
    (hU : ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → U i r.u)
    (hw : ∀ i, WidthRef h (addr i) (U i)) : WidthFact h addr :=
  fun r hr i hi => widthHolds_of_ref h (S := addr i) (g := i) (U := U i) (fun r hr hi => hU r hr i hi) (hw i) r hr hi

end SpotFlock
