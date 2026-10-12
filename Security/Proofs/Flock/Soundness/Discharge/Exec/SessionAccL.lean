import Proofs.Flock.Soundness.Discharge.Exec.SessionSoundL
import Proofs.Flock.Soundness.SessionAcc

/-!
# Knowledge soundness of a salted table in the session, from accepted reruns

`SessionAcc.lean` over a leaf scheme `Lf` (`sessAfterL`). `session_knowledge_sound_accL`: for an event `Acc` of the
session's outputs that implies the table accepts, with probability `ε`, `ε · Pr[the table extracted from k ≥ 1 reruns
in which Acc holds is uncommitted] ≤ ε_c⁻ + N₀·ε/(ek) + √(k·adv₀)`, every term computed in the salted session.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type}

open Classical in
/-- **Knowledge soundness of a table in a batched session, from `k` accepted reruns.** For an event
`Acc` of the session's outputs that implies the table accepts, with probability `ε`: `ε` times the
probability that the table extracted from `k ≥ 1` reruns of everything after `Commit` in which `Acc`
holds is uncommitted, is at most `ε_c⁻ + N₀·ε/(ek) + √(k·adv₀L)`. -/
theorem session_knowledge_sound_accL (A : Arith F K)
    (hA : A.Correct) [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D)
    (Acc : List (OutCL F D Sl) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD pre.length dfltOutL).accepted = true)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀))
    (k : ℕ) (hk : 1 ≤ k) :
    expect (ind Acc)
        (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ *
      Extract.expectAcc (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post
          cap₀) σ Acc k
        (ind fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) ≤
      epsSL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post
          cap₀ σ +
        2 ^ l₀.logLen * expect (ind Acc)
          (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ /
          (Real.exp 1 * k) +
        Real.sqrt (k * adv₀SL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post
          cap₀ σ) := by
  -- the fixed-table fact, accepted runs only: an uncommitted extraction costs accepted off-table mass
  have hX : ∀ bs : List (List (OutCL F D Sl)),
      ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) →
      expect (ind Acc)
          (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ ≤
        epsSL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post
            cap₀ σ +
          expect (ind fun ω : List (OutCL F D Sl) => Acc ω ∧
              ∃ p, Rewinding.Off (fun outs : List (OutCL F D Sl) =>
                op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL))
                (Extract.firstTable (fun outs : List (OutCL F D Sl) =>
                  op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL))
                  (fun _ _ => 0) bs) ω p)
            (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := by
    intro bs hbs
    have h := session_sound_of_table_accL A hA H E Lf S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts
      hlay pre post cap₀ Acc hAcc
      (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) σ
    have e1 : (ind fun outs : List (OutCL F D Sl) => Acc outs ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) = ind Acc := by
      funext outs
      simp only [ind, hbs, not_false_eq_true, and_true]
    have e2 : (ind fun outs : List (OutCL F D Sl) => Acc outs ∧
        Off₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)
          (outs.getD pre.length dfltOutL)) =
        ind fun ω : List (OutCL F D Sl) => Acc ω ∧
          ∃ p, Rewinding.Off (fun outs : List (OutCL F D Sl) =>
            op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL))
            (Extract.firstTable (fun outs : List (OutCL F D Sl) =>
              op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL))
              (fun _ _ => 0) bs) ω p := by
      funext outs
      unfold ind Off₀L extractTableSL
      rw [rowT_tabOf]
    rw [e1, e2] at h
    exact h
  have hjoint := Extract.joint_acc_le (fun outs : List (OutCL F D Sl) =>
      op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOutL)) (fun _ _ => 0)
    (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ Acc
    (fun bs => Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (extractTableSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs))
    _ (epsS_nonnegL A H E Lf S m k0 l₀ l₁ rest ext _ hm ptLocal mPts pre post cap₀ σ) hX k hk
  rw [Fintype.card_fin] at hjoint
  convert hjoint using 4
  all_goals first
    | rfl
    | simp only [lvOf, List.getD_cons_zero, Nat.cast_pow, Nat.cast_ofNat]

end FlockSoundness.Discharge.Exec
