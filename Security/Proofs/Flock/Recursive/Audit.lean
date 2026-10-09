import Proofs.Flock.Recursive.Property
import Proofs.Flock.Recursive.Private

/-!
# The recursive audit's property: its proof and what it rests on

`Flock.SecurityProofs.RecursiveAudit : Flock.Guarantees.RecursiveAudit`, from `zk_sessions_recursive_fork_inner` at
the compiled table and the premise `CompiledForkSound`, the registration break of `RecursiveRanRegistered`'s proof at
the commit strings V* checks (`regBcCommit_binding`), and `RecursiveCircuitPrivateLeaves`. Its premises are
`Proofs.Flock.Recursive.Premises`'.

Open, a lemma whose proof is `sorry` (on tonight's path, owner proofs):
* `FlockSoundness.Discharge.Recursion.zk_sessions_recursive_fork_inner` (`Compile`): the recursive audit is sound with
  the inner protocol's soundness in fork form, the inner fork a third beside the round and session forks.

`FlockSoundness.Discharge.Recursion.flock_inner_sound_compiled` (`Compile`, the compile lemma, `sorry`) discharges the
premise `CompiledForkSound` at a class whose schedule is `fast100`'s and whose layout holds; the property takes the
premise until it is proved.

Still taken by the property, each to become a lemma at the executable (V*'s session row, the universal unit's class,
the firewall's schedule) or a refusal of the verifier: V*'s laws' strata (`_hS`), the read commit strings opening the
root (`_hop`) and the run's hidden statement being theirs (`_hx`), the fields of the class and law it ranges over, and
in (ii) V*'s sessions' zero knowledge (`_hOuter`, from `ZeroKnowledgeHidden`) and the honest witnesses (`honest`).
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

/-- **The registration binds each row's commit string**: two openings of one port at `i` to different commit strings
give a SHA-512 collision, of the two tree leaves' inputs when the leaves agree (the leaf prefix is the same, so the
inputs differ), else the registered path's (`Flock.Registered.opensLeaf_binding`). -/
theorem regBcCommit_binding (i : ℕ) : (Flock.Guarantees.regBcCommit i).Binding := by
  rintro reg bc bc' ⟨r1, b1, p1⟩ ⟨r2, b2, p2⟩ ⟨h1r, h1b, h1⟩ ⟨h2r, h2b, h2⟩ hne
  simp only at h1r h1b h1 h2r h2b h2
  subst h1r h1b h2b
  unfold Flock.Guarantees.regBcCommit
  simp only
  split_ifs with hl
  · refine collides_of_bytes (x := (Hm96.Default512.leafPrefix ++ b1, Hm96.Default512.leafPrefix ++ b2))
      ⟨fun e => hne ?_, hl⟩
    have e2 := congrArg (fun b : ByteArray => b.data.toList) e
    simp only [ByteArray.data_append, Array.toList_append] at e2
    exact (bytes_of_list b1).symm.trans
      ((congrArg (fun l : List UInt8 => (⟨l.toArray⟩ : ByteArray)) (List.append_cancel_left e2)).trans
        (bytes_of_list b2))
  · exact collides_of_bytes (Flock.Registered.opensLeaf_binding r1 i (Hm96.Default512.treeLeaf b1) p1
      (Hm96.Default512.treeLeaf b2) p2 (Sha512.hash_size _) (Sha512.hash_size _) hl h1 h2)

/-- **The recursive audit, as the verifier of record runs it.** -/
theorem RecursiveAudit : Flock.Guarantees.RecursiveAudit := by
  refine And.intro ?_ ?_
  swap
  · intro Out Prog Wit Rv Ωa Ωp Ωo Ωs V _ _ _ _ _ _ _ _ nr rows pub honest nh msgs view sim εo hOuter hKey p₀ o P w hw E
    exact RecursiveCircuitPrivateLeaves nr rows pub honest nh msgs view sim hOuter hKey p₀ o P w hw E
  intro nT r T CT _ _ n L Reg Cm Reg' Reg₂ cls regs sch Vs zs dg N σ x wmsg Sk cmt msg rt stmt reads row op hS _ hA3
    hin hop hx p op₀ hreg k Rw M hk hRw hM ρ hρ hr hbr
  rcases zk_sessions_recursive_fork_inner hS hA3 zs dg N hk hRw hM hρ hr σ x wmsg Sk cmt msg hbr Refine.H512
    (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leaf512 cls regs sch) hin with hf | hf | hf | hb
  · exact Or.inl hf
  · exact Or.inr (Or.inl hf)
  · exact Or.inr (Or.inr (Or.inl hf))
  · by_cases hc : ∃ o, FlockSoundness.Game.Outcome (recGame L Reg (Flock.Guarantees.innerExec T CT cls regs sch) Cm
        fun R ω t => outerV (zs R ω t) dg N) σ o ∧ Flock.Guarantees.VerifiesV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ∃ i, reads o i ∧ row o i ≠ p i ∧
          FlockSoundness.Binding.Collides ((Flock.Guarantees.regBcCommit i).ext (op o i) (op₀ i))
    · exact Or.inr (Or.inr (Or.inr (Or.inl hc)))
    · refine Or.inr (Or.inr (Or.inr (Or.inr (le_trans (prob_mono_outcome _ σ fun o hro h => ⟨h.1.1, fun hh => ?_⟩)
        hb))))
      have hagree : ∀ i, reads o i → p i = row o i := fun i hi => by
        by_contra hne
        exact hc ⟨o, hro, h.1, i, hi, fun he => hne he.symm,
          regBcCommit_binding i rt (row o i) (p i) (op o i) (op₀ i) (hop o h.1 i hi) (hreg i) fun he => hne he.symm⟩
      rw [hx o h.1 p hagree] at hh
      exact h.2 hh

end Flock.SecurityProofs
