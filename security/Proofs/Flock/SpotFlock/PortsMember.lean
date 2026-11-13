module

public import Proofs.Flock.SpotFlock.WiringOk

/-!
# A checked unit's session has its template's member circuit's ports

`portsMember : PortsMember h`, at every held code: the session's first table's circuit is the one the scope check
parses (`accepted_check`), which unrolls, ports and all (`unrolled_ports`), to its template's reference statement's
statement circuit (`pin_of_stmtOkAt`, `stmt_of_pin`), the member's circuit (`memberCircuit`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **A checked unit's session's first table has its template's member circuit's ports**, at every held code. -/
theorem portsMember (h : Held) : PortsMember h := by
  intro r hr
  have hj := CheckedRun.sts_pos hr
  obtain ⟨Pt, d, _, hx, hP, h2, hin, hok, hu, hck, hb, hρ, -, hset⟩ := hr
  have hokAt := stmtOkAt_unit h hx hok hu
  have ht := templateOf_lt_of_checked h hx hu
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h r.m₂ r.u) with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain ⟨p, hpb, -, hpp, hc, hh⟩ := parsed_of_stmtOkAt h hokAt hS
    obtain ⟨rfl, -⟩ := parsed_setupOf hset hpb
    obtain ⟨w, w₀, hw, hw₀, he⟩ := pin_of_stmtOkAt h hokAt
    have hok₀ := templateOk_of_two h h2 ht
    unfold templateOk at hok₀
    cases hr₀ : FlockDeploy.readOf true (h.xS₀ (templateOf h.ms r.u)) with
    | none => rw [hr₀] at hok₀; cases hok₀
    | some r₀ =>
      have hst := stmt_of_pin hw hw₀ he (readOf_of_parseBytes hpb) hr₀
      obtain ⟨T, a, hst0, -, hcirc, -⟩ := accepted_check hpp hc hh hj
      have hm : memberCircuit h (templateOf h.ms r.u) = r₀.stmt := by
        simp [memberCircuit, stmtOf, hr₀]
      rw [hm]
      calc r.sts[0]!.c.ports = a.st.c.ports := by rw [(getElem!_pos r.sts 0 hj).trans hst0.symm]
        _ = (_root_.Flock.HmOut.unrolled a.st.c).ports := (FlockSoundness.unrolled_ports a.st.c).symm
        _ = r₀.stmt.ports := by
          rw [← hcirc]
          exact congrArg _root_.Flock.Circuit.ports hst

end SpotFlock
