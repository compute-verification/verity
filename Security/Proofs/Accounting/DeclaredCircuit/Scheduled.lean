import Proofs.Accounting.DeclaredCircuit.Requirement
import Properties.Accounting.DeclaredCircuit
import Properties.Proofs.RemoteAudit.Schedule

/-!
# A declared circuit's space and bandwidth at the committed steering cap, proved

`Properties.DeclaredCircuit.SecurityProofs.scheduledRequirement`, the theorem of
`Properties.DeclaredCircuit.ScheduledRequirement`: `Accounting.SecurityProofs.DeclaredCircuitRequirement` on summaries
of `a + (cap + 1)` bits, the devices' `a` bits then the schedule's `cap + 1`. The schedule's bits steer the layout
(`Properties.RemoteAudit.steered_of_checked`), so the layout's slot of a summary is the slot of its last `cap + 1` bits,
and the devices of a summary are the session of its first `a` (`SketchSummarySession.widen`), which runs on each pass's
summary as `A` runs on the devices' `a` bits (`widen_win`).
-/

namespace Accounting.RateSketch.SketchSummarySession

variable {a n P N m ρ h S L q ks G d : ℕ}

/-- `A` on summaries of `b` more bits: a summary's session is `A`'s at its first `a` bits. -/
def widen (A : SketchSummarySession a n P N m ρ h S L q ks G d) (b : ℕ) :
    SketchSummarySession (a + b) n P N m ρ h S L q ks G d :=
  ⟨fun s => A.session fun i => s (Fin.castAdd b i)⟩

theorem widen_session (A : SketchSummarySession a n P N m ρ h S L q ks G d) {b : ℕ} (s : Pous.Bits a)
    (t : Pous.Bits b) : (A.widen b).session (Fin.append s t) = A.session s := by
  simp only [widen, Fin.append_left]

theorem widen_win (A : SketchSummarySession a n P N m ρ h S L q ks G d) {b : ℕ} (w : Fin n)
    (ω : Fin n → SketchWindowCoins P N m ρ h ks G d) (s : Fin P → Pous.Bits a) (t : Fin P → Pous.Bits b) :
    (A.widen b).win w ω (fun j => Fin.append (s j) (t j)) = A.win w ω s := by
  simp only [win, widen_session]

end Accounting.RateSketch.SketchSummarySession

namespace Properties.DeclaredCircuit.SecurityProofs

open Accounting Accounting.RateSketch

/-- **A declared circuit's held and moved bits from an accepted stacked audit, at the committed steering cap**,
`Properties.DeclaredCircuit.ScheduledRequirement`. -/
theorem scheduledRequirement : ScheduledRequirement := by
  intro n P N ρ S q G Xr Lr qr τ k a L E B εnum εden T M ε δ ιW Wc caps Aud Ω Ca Ch _ _ enc choose layout steer
    joined A blocks cap idx pos appended sent V Kg Dv _ _ W pl ℓ grp bind mm plan addr hn hP hN hE0 hrank hKB hE hε hB
    hCC hgap hchecked hlife hres hprop hreads hx
  obtain ⟨sched, slot, hslot⟩ := Properties.RemoteAudit.steered_of_checked (idx := idx) hgap hchecked
  have hB' : ∀ s w η j pre, Assumptions.SketchHashBudget
      ((((A.widen (cap + 1)).session s).window w η).device j pre) (qr * τ) :=
    fun s w η j pre => hB (fun i => s (Fin.castAdd (cap + 1) i)) w η j pre
  have hsess : (fun c y => (A.widen (cap + 1)).session (Fin.append (steer c y) (sched c y))) =
      fun c y => A.session (steer c y) :=
    funext fun c => funext fun y => SketchSummarySession.widen_session A (steer c y) (sched c y)
  have hCC' : Assumptions.CircuitCarriers
      (fun c y => (A.widen (cap + 1)).session (Fin.append (steer c y) (sched c y)))
      (steeredStore enc choose layout) layout joined (Xr * τ) L B εnum εden Wc Aud := by
    rw [hsess]
    exact hCC
  have hslot' : Assumptions.SteeredLayout layout fun c y => Fin.append (steer c y) (sched c y) :=
    ⟨fun c s => slot c fun i => s (Fin.natAdd a i), fun c y => by
      simp only [Fin.append_right]
      exact hslot c y⟩
  have hG := Accounting.SecurityProofs.DeclaredCircuitRequirement n P N ρ S q G Xr Lr qr τ k (a + (cap + 1)) L E B
    εnum εden T M ε δ Wc caps Aud enc choose layout (fun c y => Fin.append (steer c y) (sched c y)) joined
    (A.widen (cap + 1)) W pl ℓ grp bind mm plan addr hn hP hN hE0 hrank hKB hE hε hB' hCC' hslot' hlife hres hprop
    hreads hx
  simpa only [SketchSummarySession.widen_win] using hG

end Properties.DeclaredCircuit.SecurityProofs
