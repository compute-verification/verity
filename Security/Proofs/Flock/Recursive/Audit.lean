import Proofs.Flock.Recursive.Property
import Proofs.Flock.Recursive.Private
import Proofs.Flock.Recursive.ReadsOpen
import Proofs.Flock.Recursive.Loads

/-!
# The recursive audit's property: its proof and what it rests on

`Flock.SecurityProofs.RecursiveAudit : Flock.Guarantees.RecursiveAudit`, from `zk_sessions_recursive_fork_inner` at
the compiled table, the registration break of `RecursiveRanRegistered`'s proof at the commit strings V* checks
(`regBcCommit_binding`), and `RecursiveCircuitPrivateLeaves`. Its premises are `Proofs.Flock.Recursive.Premises`'.

In (i), from the statement and the premises (`Proofs.Flock.Recursive.ExecClass`):
* at a statement Lean's verifier refuses, the event is empty (`AcceptsStmt` is its first conjunct);
* at one it accepts, the class checks pass (`classOk_of_accepts`), and the compile premise holds at `execClassAt st`
  (`compiledForkSound_execClassAt`, from `flock_inner_sound_compiled`);
* at an outcome σ reaches whose sessions all verify (else the event is empty), `_hS` at V*'s statements with their
  models read off their strata (`execStmts`): each session's draw file custody ties to the strata over the law's own
  population (`RecordCustody`'s `DrawFileZK`), which is V*'s units once the session loads V*'s statement
  (`VStmt.tiles_of_loads`, `VStmt.pop_eq_nv`, `VStmt.exec_execStrata`), and it does (`loads_of_verifies`, open: gap 8);
* the read commit strings opening the root at an outcome whose sessions all verify (`Recursion.zk_reads_open`, at the
  reads `readsZ`/`rowZ`/`opZ` read off the outcome);
* the fifth disjunct from `_hx`'s second case: two runs at one registration and draw that read different strings at a
  row, whose openings collide.

Still taken by the property, each to become a lemma at the executable or a refusal of the verifier: the run's hidden
statement being the read strings' unless two runs read one row differently (`_hx`), the law's field `L.nonempty`, and
in (ii) V*'s sessions' zero knowledge (`_hOuter`, from `ZeroKnowledgeHidden`) and the honest witnesses (`honest`).
Open in the proof: that each accepted session loads V*'s statement (`loads_of_verifies`), until the verifier checks it.
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

/-- **The recursive audit, as the verifier of record runs it.** -/
theorem RecursiveAudit : Flock.Guarantees.RecursiveAudit := by
  refine And.intro ?_ ?_
  swap
  · intro Out Prog Wit Rv Ωa Ωp Ωo Ωs V _ _ _ _ _ _ _ _ nr rows pub honest nh msgs view sim εo hOuter hKey p₀ o P w hw E
    exact RecursiveCircuitPrivateLeaves nr rows pub honest nh msgs view sim hOuter hKey p₀ o P w hw E
  intro nT r T CT _ _ n L Reg Cm Reg' Reg₂ st Vs zs₀ hCust dg N σ x wmsg Sk cmt msg rt stmt _ hA3 hx p op₀ hreg
    k Rw M hk hRw hM ρ hρ hr hbr
  by_cases hacc : AcceptsStmt st
  swap
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (((prob_mono_outcome _ σ fun o _ h => absurd h.1 hacc).trans
      (prob_false _ σ).le).trans (zero_le ..))))))
  have hin := compiledForkSound_execClassAt T CT (classOk_of_accepts hacc)
  by_cases hv : ∃ o, FlockSoundness.Game.Outcome (recGame L Reg (Flock.Guarantees.innerAt T CT st) Cm
      fun R ω t => outerV (Flock.Guarantees.custodied zs₀ hCust R ω t) dg N) σ o ∧
      Flock.Guarantees.VerifiesV (Flock.Guarantees.custodied zs₀ hCust o.1 o.2.1 o.2.2.1) o.2.2.2
  swap
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (((prob_mono_outcome _ σ fun o hro h =>
      absurd ⟨o, hro, h.2.1⟩ hv).trans (prob_false _ σ).le).trans (zero_le ..))))))
  obtain ⟨ov, -, hvv⟩ := hv
  have hS : ∀ j, Law.ExecStrata (execStmts Vs j).law.σs (execStmts Vs j).law.ks (execStmts Vs j).law.St := fun j =>
    (Vs j).exec_execStrata (VStmt.tiles_of_loads
      (Flock.Guarantees.custodied zs₀ hCust ov.1 ov.2.1 ov.2.2.1 ov.2.2.2.1 j)
      (hCust ov.1 ov.2.1 ov.2.2.1 ov.2.2.2.1 j).2 (hvv.2 j)
      (loads_of_verifies (Flock.Guarantees.custodied zs₀ hCust ov.1 ov.2.1 ov.2.2.1) ov.2.2.2 hvv.1 hvv.2 j))
  rcases zk_sessions_recursive_fork_inner hS hA3
    (Flock.Guarantees.custodied zs₀ hCust) dg N hk hRw hM hρ hr σ x wmsg Sk cmt msg hbr Refine.H512
    (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leaf512 (execClassAt st).cls (execClassAt st).regs
      (execClassAt st).sch) hin with hf | hf | hf | hb
  · exact Or.inl hf
  · exact Or.inr (Or.inl hf)
  · exact Or.inr (Or.inr (Or.inl hf))
  · by_cases hc : ∃ o, FlockSoundness.Game.Outcome (recGame L Reg (Flock.Guarantees.innerAt T CT st) Cm
        fun R ω t => outerV (Flock.Guarantees.custodied zs₀ hCust R ω t) dg N) σ o ∧
        Flock.Guarantees.VerifiesV (Flock.Guarantees.custodied zs₀ hCust o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ∃ i, readsZ (Flock.Guarantees.custodied zs₀ hCust) rt o i ∧
          rowZ (Flock.Guarantees.custodied zs₀ hCust) rt o i ≠ p i ∧
          FlockSoundness.Binding.Collides ((Flock.Guarantees.regBcCommit i).ext
            (opZ (Flock.Guarantees.custodied zs₀ hCust) rt o i) (op₀ i))
    · exact Or.inr (Or.inr (Or.inr (Or.inl hc)))
    · refine Or.inr (Or.inr (Or.inr (Or.inr (or_iff_not_imp_left.2 fun hc' =>
        le_trans (prob_mono_outcome _ σ fun o hro h => ⟨h.2.1.1, fun hh => ?_⟩) hb))))
      have hagree : ∀ i, readsZ (Flock.Guarantees.custodied zs₀ hCust) rt o i →
          p i = rowZ (Flock.Guarantees.custodied zs₀ hCust) rt o i := fun i hi => by
        by_contra hne
        exact hc ⟨o, hro, h.2.1, i, hi, fun he => hne he.symm,
          regBcCommit_binding i rt _ (p i) _ (op₀ i)
            (zk_reads_open (Flock.Guarantees.custodied zs₀ hCust) rt o h.2.1 i hi) (hreg i) fun he => hne he.symm⟩
      rcases hx o hro h.2.1 p hagree with he | ⟨o', hro', h1, h2, hv', i, hi, hi', hne, hcol⟩
      · rw [he] at hh
        exact h.2.2 hh
      · exact hc' ⟨o, o', hro, hro', h1, h2, h.2.1, hv', i, hi, hi', hne, hcol⟩

end Flock.SecurityProofs
