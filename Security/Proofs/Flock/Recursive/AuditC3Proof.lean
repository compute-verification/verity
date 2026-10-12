import Proofs.Flock.Recursive.AuditC3
import Proofs.Flock.Recursive.FirewallZK
import Proofs.Flock.Recursive.Loads
import Proofs.Flock.C3.Fork.Sound

/-!
# The recursive audit at candidate 3: its proof

`Flock.SecurityProofs.RecursiveAuditC3 : Flock.Guarantees.RecursiveAuditC3`, `PublicAudit`'s proof at candidate 3's
table, and `RecursiveFirewallPrivate`.

In (i), as in `PublicAudit`'s:
* where some index has no session that types, the event is empty (`prob_sessionsR_untypable`);
* at a statement Lean's verifier doesn't build, or a V* list it doesn't build, the event is empty (its first conjunct);
* at one it builds, the class checks pass (`classOk_of_accepts`), and candidate 3's table at the executable's
  parameters is sound in fork form (piece (a), `FlockC3.Fork.holoForkSound_holoAt`);
* the hidden statement is R's registration (`xR` with the registrant's openings `_hreg`), a constant of the run, and the
  raw event is bounded by the typed sessions' at the transported prover (`prob_recT_le`, `verifies_zsOf`).

(ii) is `RecursiveFirewallPrivate`.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The recursive audit at candidate 3, as the verifier of record runs it.** -/
theorem RecursiveAuditC3 : Flock.Guarantees.RecursiveAuditC3 := by
  refine And.intro ?_ ?_
  swap
  · exact RecursiveFirewallPrivate
  intro nT r T CT _ _ n Ω _ _ draw Reg Cm Reg' Reg₂ obj prog st hp reg d xR hreg hstk ish Vs zr hCust bind rows hob hF
    dg N σ wmsg Sk cmt msg hA3 k Rw M hk hRw hM ρ hρ hr hbr
  by_cases hT : Typed zr
  swap
  · left
    simp only [Typed, not_forall, not_exists] at hT
    obtain ⟨j₀, hj₀⟩ := hT
    refine prob_recGame_zero _ σ fun ω t s => ?_
    rw [prob_outerVR]
    exact prob_sessionsR_untypable hob (zr σ.1 ω t s.1) (fun j => (hCust σ.1 ω t s.1 j).1) (hF σ.1 ω t s.1) j₀
      (hj₀ (σ.1, ω, t, s.1)) _ (fun b hb => hb.2.1) _
  by_cases hacc : Flock.Guarantees.AcceptsPublic obj prog st ∧ (RecVStar.statements ish).isOk = true
  swap
  · exact Or.inl (le_antisymm ((prob_mono_outcome _ σ fun o _ h => absurd h.1 hacc).trans (prob_false _ σ).le)
      (zero_le ..))
  refine Or.inr ⟨hT, ?_⟩
  have hin := FlockC3.Fork.holoForkSound_holoAt T CT hp (classOk_of_accepts hacc.1.acceptsStmt)
  have hr' : ∀ R ω t R₂ j S' R' i, rateZ (ZkHidden.planZHJ (Reg := Reg')
      (Flock.Guarantees.custodied zr hCust hT R ω t R₂ j).dj
      (Flock.Guarantees.custodied zr hCust hT R ω t R₂ j).y₀ S' R') i k ≤ ρ := by
    intro R ω t R₂ j S' R'
    dsimp only [Flock.Guarantees.custodied]
    by_cases hx' : (zr R ω t R₂ j).Typable
    · rw [zsOf_pos zr _ hT R ω t R₂ j hx']
      exact hr R ω t R₂ j _ _ hx'.y₀ S' R'
    · rw [zsOf_neg zr _ hT R ω t R₂ j hx']
      exact hr _ _ _ _ j _ _ _ S' R'
  rcases zk_sessions_recursive_fork_inner hA3 (Flock.Guarantees.custodied zr hCust hT) dg N hk hRw hM hρ hr'
    (Flock.Guarantees.transported zr hCust hT σ) (fun _ _ => Subtype.mk xR hreg) wmsg Sk cmt msg
    (hbr hT) Refine.H512 (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)) hin
    with hf | hf | hf | hf | hb
  · exact Or.inl hf
  · exact Or.inr (Or.inl hf)
  · exact Or.inr (Or.inr (Or.inl hf))
  · exact Or.inr (Or.inr (Or.inr (Or.inl hf)))
  · refine Or.inr (Or.inr (Or.inr (Or.inr (le_trans (prob_recT_le hob zr _ hT hF σ _
      (fun o => (Flock.Guarantees.AcceptsPublic obj prog st ∧ (RecVStar.statements ish).isOk = true) ∧
        Flock.Guarantees.VerifiesV (Flock.Guarantees.custodied zr hCust hT o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ¬ (holoInnerAt T CT st hp).holds (Subtype.mk xR hreg))
      (fun o ho => ho.2.1) (fun o hall ho =>
        ⟨ho.1, verifies_zsOf zr _ hT o.1 o.2.1 o.2.2.1 o.2.2.2 hall (acceptsVR_of_V ho.2.1.1) ho.2.1.2, ho.2.2⟩))
      (le_trans (prob_mono_outcome _ (Flock.Guarantees.transported zr hCust hT σ)
        fun o _ h => ⟨h.2.1.1, h.2.2⟩) hb)))))

end Flock.SecurityProofs
