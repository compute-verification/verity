import Proofs.Accounting.RemoteHeld.P2Line

/-!
# PoUB-P2's line of record at the subprotocol, proved (`Properties.RemoteAudit.P2Subprotocol.LineHeldAtSubprotocol`)

Where the subprotocol accepts bytes that play the record (`Plays`), the verifier accepts the record's session
(`plays_verified`: `acceptIsVerified` gives the check's replay, which is the record's run, `run_of_replay`), so the
event is inside `P2Line.SecurityProofs.LineHeld`'s (`Pouw.pr_mono`), whose bound `P2Line.SecurityProofs.lineHeld`
gives.
-/

namespace Properties.RemoteAudit.P2Subprotocol.SecurityProofs

open Properties.RemoteAudit.P2Line

/-- Where the subprotocol accepts bytes that play the record on the node, the verifier accepts the record's session:
the check's replay ends accepted (`acceptIsVerified`), and it is the record's run on the node (`run_of_replay`), the
commitment holding the node's store and the log being the record's. -/
theorem plays_verified {node : Audited} {r : Replay} {a b e : ByteArray} (h : Plays node r a b e) :
    Verified node r := by
  obtain ⟨c, ev, -, -, hc, -, hl, hq, pub, prof, hacc⟩ := h
  obtain ⟨-, s, -, -, hrep, -, hout, -, -⟩ := acceptIsVerified _ _ _ _ _ hacc
  obtain ⟨params, hq', hrun⟩ := Properties.RemoteAudit.P2Line.SecurityProofs.run_of_replay hrep
  have e : RemoteAudit.P2Session.requestOf c.held r.request = .ok node.params := by rw [hc]; exact hq
  obtain rfl : params = node.params := Except.ok.inj (hq'.symm.trans e)
  rw [hc, hl] at hrun
  exact ⟨hq, s, hrun, hout⟩

/-- **PoUB-P2's line of record at the subprotocol** (`LineHeldAtSubprotocol` at the subprotocol's acceptance), proved:
its event is inside `P2Line.SecurityProofs.LineHeld`'s (`plays_verified`), on the same premises and bound. -/
theorem lineHeldAtSubprotocol : LineHeldAtSubprotocol Plays Verified Replay.summary := by
  intro node κ X Cm _ prf rest recs correct com T R δs δnet a b e r F O I hP hCv hSk hCov hIc hLive hRef hCnt
  exact le_trans (Pouw.pr_mono fun ω h => ⟨plays_verified h.1, h.2⟩)
    (Properties.RemoteAudit.P2Line.SecurityProofs.lineHeld node κ X Cm prf rest recs correct com T R δs δnet O I hP
      hCv hSk hCov hIc hLive hRef hCnt)

end Properties.RemoteAudit.P2Subprotocol.SecurityProofs
