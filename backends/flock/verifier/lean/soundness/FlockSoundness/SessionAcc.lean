import FlockSoundness.Session
import FlockSoundness.ExtractAcc

/-!
# Knowledge soundness in the session, from accepted reruns (DESIGN.md §3)

The expected-time extractor fills a table from `k` accepted reruns of everything after `Commit`
(`Extract.expectAcc`), and the compiled bound's deviation needs only accepted runs
(`session_sound_of_table_acc`). So the positions an accepted fresh run opens off the extractor's
table cost a term relative to the acceptance probability `ε` (`session_knowledge_sound_acc`):
$$\varepsilon\cdot\Pr[\text{extraction fails}]\le\varepsilon_c^-+N_0\varepsilon/(ek)+\sqrt{k\,\mathrm{adv}_0},$$
which is `ε_c⁻ + ε/c + √(k·adv₀)` at `k = cN₀/e`. The conflict term's `1/ε` became `√(k·adv₀)`
because the product is also at most `ε`.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

open Classical in
/-- **Knowledge soundness of a table in a batched session, from `k` accepted reruns.** The
acceptance probability `ε` of a fresh run after `Commit`, times the probability that the table
extracted from `k ≥ 1` accepted reruns of everything after `Commit` is uncommitted, is at most
`ε_c⁻ + N₀·ε/(ek) + √(k·adv₀)`. -/
theorem session_knowledge_sound_acc (hMCA : Assumptions.BCHKS25Thm46) (A : Arith F K)
    (hA : A.Correct) [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀))
    (k : ℕ) (hk : 1 ≤ k) :
    expect (ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true)
        (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ *
      Extract.expectAcc (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post
          cap₀) σ (fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true) k
        (ind fun bs => ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) ≤
      epsS A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post
          cap₀ σ +
        2 ^ l₀.logLen * expect (ind fun outs : List (OutC F D) =>
            (outs.getD pre.length dfltOut).accepted = true)
          (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ /
          (Real.exp 1 * k) +
        Real.sqrt (k * adv₀S A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post
          cap₀ σ) := by
  -- the fixed-table fact, accepted runs only: an uncommitted extraction costs accepted off-table mass
  have hX : ∀ bs : List (List (OutC F D)),
      ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) →
      expect (ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true)
          (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ ≤
        epsS A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post
            cap₀ σ +
          expect (ind fun ω : List (OutC F D) => (ω.getD pre.length dfltOut).accepted = true ∧
              ∃ p, Rewinding.Off (fun outs : List (OutC F D) =>
                op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut))
                (Extract.firstTable (fun outs : List (OutC F D) =>
                  op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (outs.getD pre.length dfltOut))
                  (fun _ _ => 0) bs) ω p)
            (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := by
    intro bs hbs
    have h := session_sound_of_table_acc hMCA A hA H E S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts
      hlay pre post cap₀ (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs) σ
    have e1 : (ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)) =
        ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true := by
      funext outs
      simp only [ind, hbs, not_false_eq_true, and_true]
      split_ifs <;> rfl
    have e2 : (ind fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true ∧
        Off₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs)
          (outs.getD pre.length dfltOut)) =
        ind fun ω : List (OutC F D) => (ω.getD pre.length dfltOut).accepted = true ∧
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
    (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ
    (fun outs : List (OutC F D) => (outs.getD pre.length dfltOut).accepted = true)
    (fun bs => Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (extractTableS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length cap₀ bs))
    _ (epsS_nonneg A H E S m k0 l₀ l₁ rest ext _ hm ptLocal mPts pre post cap₀ σ) hX k hk
  rw [Fintype.card_fin] at hjoint
  convert hjoint using 4
  all_goals first
    | rfl
    | simp only [lvOf, List.getD_cons_zero, Nat.cast_pow, Nat.cast_ofNat]

end FlockSoundness
