import Proofs.Accounting.RemoteHeld.Line
import Proofs.Properties.RemoteAudit.Component

/-!
# The memory line of record at the bandwidth component, proved (`Properties.RemoteAudit.Component.LineHeldAtComponent`)

Where the component accepts bytes that play the record (`Plays`), the Lean verifier accepts the record's session
(`plays_verified`, from `acceptIsVerified`), so the event is inside `LineHeld`'s (`Pouw.pr_mono`), whose bound
`lineHeld` gives. Under `OpeningsPrf` and `CoinsHidden` every draw's PRF advantage and hiding term is nothing, so their
sums leave `LineHeld`'s bound (`dropped`), and what is left is `LineHeldAtComponent`'s.
-/

namespace Properties.RemoteAudit.Component.SecurityProofs

open Accounting.Remote.Draw Properties.RemoteAudit.Line

/-- Under `OpeningsPrf` and `CoinsHidden`, `LineHeld`'s PRF and hiding terms are nothing. -/
theorem dropped {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X Cm : Type} [Fintype X] (F : Committed n m s κ X Cm)
    (com : Coins n κ X → Cm) (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ) (hPrf : OpeningsPrf F com prf δ)
    (hHide : CoinsHidden F com prf δ) (x y : ℝ) :
    x + (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
        ((F.play (Committed.junk com t)).distinguisher prf δ t)).toReal +
      (∑ t, F.hide com prf δ t).toReal + y = x + y := by
  rw [Finset.sum_eq_zero fun t _ => hPrf t, Finset.sum_eq_zero fun t _ => hHide t]
  simp

theorem lineHeldAtComponent : LineHeldAtComponent := by
  intro node κ X Cm _ prf rest recs correct com T R δs a b e r F O hL hS hD hC hR hP hCv hSk hN hLive hRef hCnt
    hReg hPrf hHide
  exact le_trans (Pouw.pr_mono fun ω h => ⟨plays_verified h.1, h.2⟩)
    ((Properties.RemoteAudit.Line.SecurityProofs.lineHeld node κ X Cm prf rest recs correct com T R δs O hL hS hD hC
      hR hP hCv hSk hN hLive hRef hCnt hReg).trans_eq
      (congrArg (δs + ·) (dropped F com prf (node.delta : ℝ) hPrf hHide _ _)))

end Properties.RemoteAudit.Component.SecurityProofs
