import Proofs.Flock.Soundness.Session
import Proofs.Flock.Soundness.ExtractAcc

/-!
# Knowledge soundness in the session, from accepted reruns (DESIGN.md §3)

The expected-time extractor fills a table from `k` accepted reruns of everything after `Commit`
(`Extract.expectAcc`), and the compiled bound's deviation needs only accepted runs
(`session_sound_of_table_acc`). So the positions an accepted fresh run opens off the extractor's
table cost a term relative to the acceptance probability `ε` (`session_knowledge_sound_acc`):
$$\varepsilon\cdot\Pr[\text{extraction fails}]\le\varepsilon_c^-+N_0\varepsilon/(ek)+\sqrt{k\,\mathrm{adv}_0},$$
which is `ε_c⁻ + ε/c + √(k·adv₀)` at `k = cN₀/e`. The conflict term's `1/ε` became `√(k·adv₀)`
because the product is also at most `ε`.

"Accepted" is any event of the session's outputs that implies the table accepts. The audit takes
the session's own acceptance (every table accepts), so that the relative term is relative to the
probability the audit accepts.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

open Classical in
/-- **Knowledge soundness of a table in a batched session, from `k` accepted reruns.** For an event
`Acc` of the session's outputs that implies the table accepts, with probability `ε`: `ε` times the
probability that the table extracted from `k ≥ 1` reruns of everything after `Commit` in which `Acc`
holds is uncommitted, is at most `ε_c⁻ + N₀·ε/(ek) + √(k·adv₀)`. -/
theorem session_knowledge_sound_acc (A : Arith F K)
    (hA : A.Correct) [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D)
    (Acc : List (OutC F D) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD pre.length dfltOut).accepted = true)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀))
    (k : ℕ) (hk : 1 ≤ k) :
    expect (ind Acc)
        (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ *
      Extract.expectAcc (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post
          cap₀) σ Acc k
        (ind fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) ≤
      epsS A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post
          cap₀ σ +
        2 ^ l₀.logLen * expect (ind Acc)
          (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ /
          (Real.exp 1 * k) +
        Real.sqrt (k * adv₀S A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post
          cap₀ σ) := by
  -- the fixed-table fact, accepted runs only: an uncommitted extraction costs accepted off-table mass
  have hX : ∀ bs : List (List (OutC F D)),
      ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) →
      expect (ind Acc)
          (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ ≤
        epsS A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post
            cap₀ σ +
          expect (ind fun ω : List (OutC F D) => Acc ω ∧
              ∃ p, Rewinding.Off (fun outs : List (OutC F D) =>
                op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut))
                (Extract.firstTable (fun outs : List (OutC F D) =>
                  op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut))
                  (fun _ _ => 0) bs) ω p)
            (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := by
    intro bs hbs
    have h := session_sound_of_table_acc A hA H E S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts
      hlay pre post cap₀ Acc hAcc
      (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) σ
    have e1 : (ind fun outs : List (OutC F D) => Acc outs ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) = ind Acc := by
      funext outs
      simp only [ind, hbs, not_false_eq_true, and_true]
    have e2 : (ind fun outs : List (OutC F D) => Acc outs ∧
        Off₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)
          (outs.getD pre.length dfltOut)) =
        ind fun ω : List (OutC F D) => Acc ω ∧
          ∃ p, Rewinding.Off (fun outs : List (OutC F D) =>
            op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut))
            (Extract.firstTable (fun outs : List (OutC F D) =>
              op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut))
              (fun _ _ => 0) bs) ω p := by
      funext outs
      unfold ind Off₀ extractTableS
      rw [rowT_tabOf]
    rw [e1, e2] at h
    exact h
  have hjoint := Extract.joint_acc_le (fun outs : List (OutC F D) =>
      op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut)) (fun _ _ => 0)
    (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ Acc
    (fun bs => Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs))
    _ (epsS_nonneg A H E S m k0 l₀ l₁ rest ext _ hm ptLocal mPts pre post cap₀ σ) hX k hk
  rw [Fintype.card_fin] at hjoint
  convert hjoint using 4
  all_goals first
    | rfl
    | simp only [lvOf, List.getD_cons_zero, Nat.cast_pow, Nat.cast_ofNat]

end FlockSoundness
