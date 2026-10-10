import Proofs.Flock.Recursive.PublicAudit
import Proofs.Flock.Recursive.FirewallZK
import Proofs.Flock.Recursive.Loads

/-!
# The recursive audit at public units: its proof

`Flock.SecurityProofs.PublicAudit : Flock.Guarantees.PublicAudit`, `RecursiveAudit`'s proof
(`Proofs.Flock.Recursive.Audit`) at a public inner unit, and `RecursiveFirewallPrivate` at no registration rows.

In (i), from the statement and the premises, as in `RecursiveAudit`'s:
* where some index has no session that types, the event is empty (`prob_sessionsR_untypable`, through
  `typable_of_accepts` and so `loads_of_verifies`, open: gap 8);
* at a statement Lean's verifier doesn't build from its partition object and program, the event is empty
  (`AcceptsPublic` is its first conjunct);
* at one it builds, the statement is accepted (`AcceptsPublic.acceptsStmt`), so the class checks pass
  (`classOk_of_accepts`) and the compile premise holds at `execClassAt st` (`compiledForkSound_execClassAt`);
* the hidden statement is the class's one (`publicHidden st`), a constant of the run, so no description and no
  registered rows enter, and the raw event is bounded by the typed sessions' at the transported prover
  (`prob_recT_le`, `verifies_zsOf`).

(ii) is `RecursiveFirewallPrivate` at `nr = 0` (`noRows`) with the public circuit `P` as `p₀`: the registration's term
`2 n_r·2^-193` is zero.

Open in the proof, as in `RecursiveAudit`'s: that each accepted session loads V*'s statement (`loads_of_verifies`).
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The recursive audit at public units, as the verifier of record runs it.** -/
theorem PublicAudit : Flock.Guarantees.PublicAudit := by
  refine And.intro ?_ ?_
  swap
  · intro Out Prog Wit Rv Ωa Ωp Pub _ _ _ _ hKey s pub honest nh reqs msgs T V pubS stopped o P w hF hpub hC hH hw E
    have h := RecursiveFirewallPrivate hKey s 0 Flock.Guarantees.noRows pub honest nh reqs msgs T V pubS stopped P o P
      w hF hpub hC hH hw E
    simpa only [Nat.cast_zero, mul_zero, zero_mul, zero_add] using h
  intro nT r T CT _ _ n L Reg Cm Reg' Reg₂ obj prog st Vs zr hCust dg N σ wmsg Sk cmt msg hA3
    k Rw M hk hRw hM ρ hρ hr hbr
  by_cases hT : Typed zr
  swap
  · left
    simp only [Typed, not_forall, not_exists] at hT
    obtain ⟨j₀, hj₀⟩ := hT
    refine prob_recGame_zero _ σ fun ω t s => ?_
    rw [prob_outerVR]
    exact prob_sessionsR_untypable (zr σ.1 ω t s.1) (fun j => (hCust σ.1 ω t s.1 j).1) j₀ (hj₀ (σ.1, ω, t, s.1)) _
      (fun b hb => hb.2.1) _
  by_cases hacc : Flock.Guarantees.AcceptsPublic obj prog st
  swap
  · exact Or.inl (le_antisymm ((prob_mono_outcome _ σ fun o _ h => absurd h.1 hacc).trans (prob_false _ σ).le)
      (zero_le ..))
  refine Or.inr ⟨hT, ?_⟩
  have hin := compiledForkSound_execClassAt T CT (classOk_of_accepts hacc.acceptsStmt)
  have hr' : ∀ R ω t R₂ j S' R' i, rateZ (planZAt (Reg := Reg')
      (Flock.Guarantees.custodied zr hCust hT R ω t R₂ j).dj
      (Flock.Guarantees.custodied zr hCust hT R ω t R₂ j).y₀ S' R') i k ≤ ρ := by
    intro R ω t R₂ j S' R' i
    dsimp only [Flock.Guarantees.custodied]
    by_cases hx' : (zr R ω t R₂ j).Typable
    · rw [zsOf_pos zr _ hT R ω t R₂ j hx']
      exact hr R ω t R₂ j _ hx'.y₀ S' R' i
    · rw [zsOf_neg zr _ hT R ω t R₂ j hx']
      exact hr _ _ _ _ j _ _ S' R' i
  rcases zk_sessions_recursive_fork_inner hA3 (Flock.Guarantees.custodied zr hCust hT) dg N hk hRw hM hρ hr'
    (Flock.Guarantees.transported zr hCust hT σ) (fun _ _ => Flock.Guarantees.publicHidden st) wmsg Sk cmt msg
    (hbr hT) Refine.H512 (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leaf512 (execClassAt st).cls
      (execClassAt st).regs (execClassAt st).sch) hin with hf | hf | hf | hb
  · exact Or.inl hf
  · exact Or.inr (Or.inl hf)
  · exact Or.inr (Or.inr (Or.inl hf))
  · refine Or.inr (Or.inr (Or.inr (le_trans (prob_recT_le zr _ hT σ _
      (fun o => Flock.Guarantees.AcceptsPublic obj prog st ∧
        Flock.Guarantees.VerifiesV (Flock.Guarantees.custodied zr hCust hT o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ¬ (Flock.Guarantees.innerAt T CT st).holds (Flock.Guarantees.publicHidden st))
      (fun o ho => ho.2.1) (fun o hall ho =>
        ⟨ho.1, verifies_zsOf zr _ hT o.1 o.2.1 o.2.2.1 o.2.2.2 hall ho.2.1.1 ho.2.1.2, ho.2.2⟩))
      (le_trans (prob_mono_outcome _ (Flock.Guarantees.transported zr hCust hT σ)
        fun o _ h => ⟨h.2.1.1, h.2.2⟩) hb))))

end Flock.SecurityProofs
