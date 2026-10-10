import Proofs.Flock.Recursive.Property
import Proofs.Flock.Recursive.Private
import Proofs.Flock.Recursive.FirewallZK
import Proofs.Flock.Recursive.Loads

/-!
# The recursive audit's property: its proof and what it rests on

`Flock.SecurityProofs.RecursiveAudit : Flock.Guarantees.RecursiveAudit`, from `zk_sessions_recursive_fork_inner` at
the compiled table, the description's binding of the hidden statement to the registered rows
(`Recursion.ofRows_eq_or_collides`), and `RecursiveFirewallPrivate` (`Proofs.Flock.Recursive.FirewallZK`). Its
premises are `Proofs.Flock.Recursive.Premises`'.

In (i), from the statement and the premises (`Proofs.Flock.Recursive.ExecClass`, `Proofs.Flock.Recursive.Loads`):
* where some index has no session that types, the event is empty: a session that verifies types
  (`typable_of_accepts`, from `loads_of_verifies`, open: gap 8), so at that index none verifies
  (`prob_sessionsR_untypable`);
* at a statement Lean's verifier refuses, or whose description it refuses, the event is empty (`AcceptsStmt` and
  `AcceptsDescription` are its first conjuncts);
* at one whose description it accepts, the hidden statement at `st`'s rows (`hiddenAt`) is the statement at the
  registered strings `p`, or SHA-512 collides at a row (`Recursion.ofRows_eq_or_collides`, which reads only the rows
  below `rt.leaves`: `readsBelow_stmtAt`);
* at one it accepts, the class checks pass (`classOk_of_accepts`), and the compile premise holds at `execClassAt st`
  (`compiledForkSound_execClassAt`, from `flock_inner_sound_compiled`);
* the raw event bounded by the typed sessions' at the transported prover (`prob_recT_le`, `verifies_zsOf`).

Still taken by the property, each to become a lemma at the executable or a refusal of the verifier: the law's field
`L.nonempty`, and in (ii) the honest witnesses (`honest`) and the obligations on the firewall's code and V*'s staging
(`FirewallSends`, `SessionsPublic`, `SessionsCoined`, `SessionsHonest`, `OuterZK.Sessions`' `hz`, `hR`, `hS`).
Open in the proof: that each accepted session loads V*'s statement (`loads_of_verifies`), until the verifier checks it.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

/-- **V*'s staging reads only the inner statement's shape**: two statements of one shape (`shapeOf`: their regions'
bytes aside) give V* as many outer statements, over one class with as many regions (`execClassAt_shape`). -/
theorem outerStatementsAt_shape {st st' : Flock.Stmt} (h : shapeOf st = shapeOf st') :
    Flock.Guarantees.outerStatementsAt st = Flock.Guarantees.outerStatementsAt st' := by
  obtain ⟨hc, hl, -⟩ := execClassAt_shape h
  show Flock.Guarantees.outerParts (execClassAt st).cls.m (execClassAt st).cls.kLog
      (2 + 2 * (execClassAt st).regs.length) =
    Flock.Guarantees.outerParts (execClassAt st').cls.m (execClassAt st').cls.kLog
      (2 + 2 * (execClassAt st').regs.length)
  rw [hl, hc]

/-- **A statement of the registration's rows reads a list's first `rt.leaves` commit strings only.** -/
theorem readsBelow_stmtAt {Ω H : Type} (rt : Flock.Registered.Port) (stmt : (Fin rt.leaves → ByteArray) → Ω → H) :
    ReadsBelow rt.leaves (Flock.Guarantees.stmtAt rt stmt) :=
  fun _ _ h ω => congrArg (fun f : Fin rt.leaves → ByteArray => stmt f ω) (funext fun r => h r r.2)

/-- **The recursive audit, as the verifier of record runs it.** -/
theorem RecursiveAudit : Flock.Guarantees.RecursiveAudit := by
  refine And.intro ?_ ?_
  swap
  · exact RecursiveFirewallPrivate
  intro nT r T CT _ _ n L Reg Cm Reg' Reg₂ st Vs zr hCust dg N σ wmsg Sk cmt msg rt stmt hA3 p path₀ hreg
    k Rw M hk hRw hM ρ hρ hr hbr
  by_cases hT : Typed zr
  swap
  · left
    simp only [Typed, not_forall, not_exists] at hT
    obtain ⟨j₀, hj₀⟩ := hT
    refine prob_recGame_zero _ σ fun ω t s => ?_
    rw [prob_outerVR]
    exact prob_sessionsR_untypable (zr σ.1 ω t s.1) (fun j => (hCust σ.1 ω t s.1 j).1) j₀ (hj₀ (σ.1, ω, t, s.1)) _
      (fun b hb => hb.2.2.1) _
  by_cases hacc : AcceptsStmt st
  swap
  · exact Or.inl (le_antisymm ((prob_mono_outcome _ σ fun o _ h => absurd h.1 hacc).trans (prob_false _ σ).le)
      (zero_le ..))
  by_cases hdesc : AcceptsDescription rt st
  swap
  · exact Or.inl (le_antisymm ((prob_mono_outcome _ σ fun o _ h => absurd h.2.1 hdesc).trans (prob_false _ σ).le)
      (zero_le ..))
  rcases ofRows_eq_or_collides hdesc (readsBelow_stmtAt rt stmt) p path₀ hreg with heq | hcol
  swap
  · exact Or.inr (Or.inl hcol)
  refine Or.inr (Or.inr ⟨hT, ?_⟩)
  have hin := compiledForkSound_execClassAt T CT (classOk_of_accepts hacc)
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
    (Flock.Guarantees.transported zr hCust hT σ) (Flock.Guarantees.hiddenAt st rt stmt) wmsg Sk cmt msg (hbr hT)
    Refine.H512 (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leaf512 (execClassAt st).cls
      (execClassAt st).regs (execClassAt st).sch) hin with hf | hf | hf | hb
  · exact Or.inl hf
  · exact Or.inr (Or.inl hf)
  · exact Or.inr (Or.inr (Or.inl hf))
  · refine Or.inr (Or.inr (Or.inr (le_trans (prob_recT_le zr _ hT σ _
      (fun o => AcceptsStmt st ∧ AcceptsDescription rt st ∧
        Flock.Guarantees.VerifiesV (Flock.Guarantees.custodied zr hCust hT o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ¬ (Flock.Guarantees.innerAt T CT st).holds (Flock.Guarantees.stmtAt rt stmt p o.2.1))
      (fun o ho => ho.2.2.1) (fun o hall ho =>
        ⟨ho.1, ho.2.1, verifies_zsOf zr _ hT o.1 o.2.1 o.2.2.1 o.2.2.2 hall ho.2.2.1.1 ho.2.2.1.2, ho.2.2.2⟩))
      (le_trans (prob_mono_outcome _ (Flock.Guarantees.transported zr hCust hT σ)
        fun o _ h => ⟨h.2.2.1.1, fun hh => h.2.2.2
          (cast (congrArg (Flock.Guarantees.innerAt T CT st).holds (heq p o.2.1)) hh)⟩) hb))))

end Flock.SecurityProofs
