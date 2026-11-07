module

public import Definitions.Flock.ZK.AnyValues
public import Proofs.Core.Game.Basic
public import Proofs.Flock.Soundness.ZK.RealLeaves

/-!
# Zero knowledge at any values: the generic forms

* `prCoin_snd`: an event of the second factor of a product space has its probability on that factor (`prCoin_fst`'s
  twin).
* `perOutcome_of_split`: released bytes that are each outcome's branch (`Splits`), each branch within its error of its
  simulator (`Within`), are simulated per outcome (`PerOutcome`).
* `gate_perOutcome`: check-then-release, two outcomes. Where the check passes the view is a derived view `d`
  (`AgreesWhere`) that a simulator matches; where it fails only the cut leaves, which another matches. The bytes
  (`gated`) are then simulated per outcome by the two (`gatedSim`). Nothing reads the values beyond these premises,
  which an instance proves at every witness.
* `PerOutcome.le_sum`, `PerOutcome.le_card`: summed over outcomes, the bound is `|St|` times the simulator that samples
  its outcome uniformly, plus `Σ_s ε s`: the `Pr_real ≤ a · Pr_sim + ε` form at `a = |St|`.
* `PerOutcome.le_leak`: the leakage form `Pr_real ≤ |St| · (Pr_sim + ε)`, `log₂ |St|` bits, for an `ε` above every
  `ε s`.
* `PerOutcome.chain`: two releases, the second adaptive. Its worker, bytes and outcome may read all of the first
  release's randomness, which the worker saw; its simulator reads only the first release's outcome and bytes. The pair
  is per outcome at the pair of outcomes, with the errors added, so the rule iterates. Summed by `le_card`, the stream
  costs `|St₁ × St₂|` times the uniform-outcome simulator plus `|St₂| Σ ε₁ + |St₁| Σ ε₂`.

Each is a lemma about one release whose simulators it is handed; a guarantee names its simulator before the worker.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

/-- An event of the second factor of a product randomness space has its probability on that factor. -/
theorem prCoin_snd {A B : Type} [Fintype A] [Fintype B] [Nonempty A] (P : B → Prop) :
    prCoin (fun p : A × B => P p.2) = prCoin P := by
  rw [prCoin_prod_split (fun (_ : A) b => P b), avg_const]

/-- **`|C|` times the average over `C` is the sum**, `C` empty or not. -/
theorem card_mul_avg {C : Type} [Fintype C] (g : C → ℝ≥0∞) : (Fintype.card C : ℝ≥0∞) * avg g = ∑ c, g c := by
  unfold avg
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun c _ => ?_
  have h : (Fintype.card C : ℝ≥0∞) ≠ 0 := Nat.cast_ne_zero.2 (Fintype.card_pos_iff.2 ⟨c⟩).ne'
  rw [← mul_assoc, ENNReal.mul_inv_cancel h (ENNReal.natCast_ne_top _), one_mul]

/-- The two averages of a function of two uniform coordinates commute. -/
theorem avg_swap {C D : Type} [Fintype C] [Fintype D] (f : C → D → ℝ≥0∞) :
    avg (fun c => avg fun d => f c d) = avg fun d => avg fun c => f c d := by
  unfold avg
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun d _ => Finset.sum_congr rfl fun c _ => mul_left_comm _ _ _

/-- **Released bytes that are each outcome's branch, each branch within its error of its simulator, are simulated per
outcome.** -/
theorem perOutcome_of_split {Ω St Ωs V : Type} [Fintype Ω] [Fintype Ωs] {rel : Ω → V} {out : Ω → St}
    {B : St → Ω → V} {S : St → Ωs → V} {ε : St → ℝ≥0∞} (hrel : Splits rel out B) (hB : ∀ s, Within (B s) (S s) (ε s)) :
    PerOutcome out rel S ε := fun s E =>
  (prCoin_mono fun ω h => by
    obtain ⟨hs, hE⟩ := h
    rw [hrel ω, hs] at hE
    exact hE).trans (hB s E)

/-- **Check-then-release**: where the cut is simulated by `simCut` within `ε false`, the derived view `d` by `simFull`
within `ε true`, and the view is the derived view wherever the check passes, the released bytes are simulated per
outcome by the two. -/
theorem gate_perOutcome {Ω Ωf Ωc V P : Type} [Fintype Ω] [Fintype Ωf] [Fintype Ωc] [Nonempty Ωf] [Nonempty Ωc]
    {view d : Ω → V} {cut : Ω → P} {go : Ω → Bool} {simFull : Ωf → V} {simCut : Ωc → P} {ε : Bool → ℝ≥0∞}
    (hcut : Within cut simCut (ε false)) (hd : Within d simFull (ε true)) (hgo : AgreesWhere go view d) :
    PerOutcome go (gated go view cut) (gatedSim simFull simCut) ε := by
  refine perOutcome_of_split (B := fun b => gated (fun _ => b) d cut) (fun ω => ?_) fun b E => ?_
  · show gated go view cut ω = gated (fun _ => go ω) d cut ω
    unfold gated
    cases hg : go ω
    · rfl
    · rw [hgo ω hg]
  · cases b
    · show prCoin (fun ω => E (.inr (cut ω))) ≤ prCoin (fun σ : Ωf × Ωc => E (.inr (simCut σ.2))) + ε false
      rw [prCoin_snd (A := Ωf) (fun c => E (.inr (simCut c)))]
      exact hcut fun v => E (.inr v)
    · show prCoin (fun ω => E (.inl (d ω))) ≤ prCoin (fun σ : Ωf × Ωc => E (.inl (simFull σ.1))) + ε true
      rw [prCoin_fst (E := Ωc) (fun c => E (.inl (simFull c)))]
      exact hd fun v => E (.inl v)

namespace PerOutcome

variable {Ω St Ωs V : Type} [Fintype Ω] [Fintype Ωs] {out : Ω → St} {rel : Ω → V} {S : St → Ωs → V} {ε : St → ℝ≥0∞}

/-- **Summed over outcomes** (the union bound). -/
theorem le_sum [Fintype St] (h : PerOutcome out rel S ε) (E : V → Prop) :
    prCoin (fun ω => E (rel ω)) ≤ ∑ s, (prCoin (fun σ => E (S s σ)) + ε s) :=
  (prCoin_mono fun ω h => ⟨out ω, Finset.mem_univ _, rfl, h⟩).trans
    ((prCoin_exists_le Finset.univ fun s ω => out ω = s ∧ E (rel ω)).trans (Finset.sum_le_sum fun s _ => h s E))

/-- **`|St|` times the uniform-outcome simulator**, plus the errors summed over outcomes: `Pr_real ≤ a · Pr_sim + ε`
at `a = |St|`. -/
theorem le_card [Fintype St] (h : PerOutcome out rel S ε) (E : V → Prop) :
    prCoin (fun ω => E (rel ω)) ≤ (Fintype.card St : ℝ≥0∞) * prCoin (fun p : St × Ωs => E (S p.1 p.2)) + ∑ s, ε s := by
  refine (h.le_sum E).trans (le_of_eq ?_)
  rw [Finset.sum_add_distrib, prCoin_prod_split (fun s σ => E (S s σ)), card_mul_avg]

/-- **The leakage form**: `Pr_real ≤ |St| · (Pr_sim + ε)` for an `ε` above every outcome's error, which is
`2^l · (Pr_sim + ε)` at `|St| = 2^l`. -/
theorem le_leak [Fintype St] (h : PerOutcome out rel S ε) {ε' : ℝ≥0∞} (hε : ∀ s, ε s ≤ ε') (E : V → Prop) :
    prCoin (fun ω => E (rel ω)) ≤ (Fintype.card St : ℝ≥0∞) * (prCoin (fun p : St × Ωs => E (S p.1 p.2)) + ε') := by
  refine (h.le_card E).trans ?_
  rw [mul_add]
  refine add_le_add le_rfl ?_
  calc ∑ s, ε s ≤ ∑ _s : St, ε' := Finset.sum_le_sum fun s _ => hε s
    _ = (Fintype.card St : ℝ≥0∞) * ε' := by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

end PerOutcome

/-- **The chain rule across releases.** If the first release is per outcome, and the second is, at every value of the
first release's randomness, with a simulator that reads only the first release's outcome and bytes, then the pair is per
outcome at the pair of outcomes, with the errors added. The proof conditions on the first release's randomness, swaps
the second release for its simulator, exchanges the order of averaging, and swaps the first release for its simulator
at each value of the second simulator's randomness. -/
theorem PerOutcome.chain {Ω₁ Ω₂ Ωs₁ Ωs₂ St₁ St₂ V₁ V₂ : Type} [Fintype Ω₁] [Fintype Ω₂] [Fintype Ωs₁]
    [Fintype Ωs₂] [Nonempty Ω₁] [Nonempty Ωs₂] {out₁ : Ω₁ → St₁} {rel₁ : Ω₁ → V₁} {S₁ : St₁ → Ωs₁ → V₁}
    {ε₁ : St₁ → ℝ≥0∞} {out₂ : Ω₁ → Ω₂ → St₂} {rel₂ : Ω₁ → Ω₂ → V₂} {S₂ : St₁ → V₁ → St₂ → Ωs₂ → V₂}
    {ε₂ : St₂ → ℝ≥0∞} (h₁ : PerOutcome out₁ rel₁ S₁ ε₁)
    (h₂ : ∀ ω₁, PerOutcome (out₂ ω₁) (rel₂ ω₁) (S₂ (out₁ ω₁) (rel₁ ω₁)) ε₂) :
    PerOutcome (fun p : Ω₁ × Ω₂ => (out₁ p.1, out₂ p.1 p.2)) (fun p => (rel₁ p.1, rel₂ p.1 p.2))
      (fun s (q : Ωs₁ × Ωs₂) => (S₁ s.1 q.1, S₂ s.1 (S₁ s.1 q.1) s.2 q.2)) (fun s => ε₁ s.1 + ε₂ s.2) := by
  rintro ⟨s₁, s₂⟩ E
  have step₂ : ∀ ω₁, prCoin (fun ω₂ => (out₁ ω₁, out₂ ω₁ ω₂) = (s₁, s₂) ∧ E (rel₁ ω₁, rel₂ ω₁ ω₂)) ≤
      prCoin (fun σ₂ => out₁ ω₁ = s₁ ∧ E (rel₁ ω₁, S₂ s₁ (rel₁ ω₁) s₂ σ₂)) + ε₂ s₂ := by
    intro ω₁
    by_cases h : out₁ ω₁ = s₁
    · calc prCoin (fun ω₂ => (out₁ ω₁, out₂ ω₁ ω₂) = (s₁, s₂) ∧ E (rel₁ ω₁, rel₂ ω₁ ω₂))
          ≤ prCoin (fun ω₂ => out₂ ω₁ ω₂ = s₂ ∧ (fun v => E (rel₁ ω₁, v)) (rel₂ ω₁ ω₂)) :=
            prCoin_mono fun ω₂ hω => ⟨(Prod.mk.inj hω.1).2, hω.2⟩
        _ ≤ prCoin (fun σ₂ => (fun v => E (rel₁ ω₁, v)) (S₂ (out₁ ω₁) (rel₁ ω₁) s₂ σ₂)) + ε₂ s₂ :=
            h₂ ω₁ s₂ fun v => E (rel₁ ω₁, v)
        _ = prCoin (fun σ₂ => out₁ ω₁ = s₁ ∧ E (rel₁ ω₁, S₂ s₁ (rel₁ ω₁) s₂ σ₂)) + ε₂ s₂ := by
            simp only [h, true_and]
    · calc prCoin (fun ω₂ => (out₁ ω₁, out₂ ω₁ ω₂) = (s₁, s₂) ∧ E (rel₁ ω₁, rel₂ ω₁ ω₂))
          ≤ prCoin (fun _ : Ω₂ => False) := prCoin_mono fun ω₂ hω => h (Prod.mk.inj hω.1).1
        _ = 0 := by simp [prCoin, avg]
        _ ≤ _ := zero_le
  have swap : ∀ {C D : Type} [Fintype C] [Fintype D] (P : C → D → Prop),
      (avg fun c => prCoin fun d => P c d) = avg fun d => prCoin fun c => P c d := fun P => by
    unfold prCoin
    exact avg_swap _
  calc prCoin (fun p : Ω₁ × Ω₂ => (out₁ p.1, out₂ p.1 p.2) = (s₁, s₂) ∧ E (rel₁ p.1, rel₂ p.1 p.2))
      = avg fun ω₁ => prCoin fun ω₂ => (out₁ ω₁, out₂ ω₁ ω₂) = (s₁, s₂) ∧ E (rel₁ ω₁, rel₂ ω₁ ω₂) :=
        prCoin_prod_split (fun ω₁ ω₂ => (out₁ ω₁, out₂ ω₁ ω₂) = (s₁, s₂) ∧ E (rel₁ ω₁, rel₂ ω₁ ω₂))
    _ ≤ avg fun ω₁ => prCoin (fun σ₂ => out₁ ω₁ = s₁ ∧ E (rel₁ ω₁, S₂ s₁ (rel₁ ω₁) s₂ σ₂)) + ε₂ s₂ :=
        avg_mono step₂
    _ = (avg fun σ₂ => prCoin fun ω₁ => out₁ ω₁ = s₁ ∧ E (rel₁ ω₁, S₂ s₁ (rel₁ ω₁) s₂ σ₂)) + ε₂ s₂ := by
        rw [avg_add, avg_const, swap fun ω₁ σ₂ => out₁ ω₁ = s₁ ∧ E (rel₁ ω₁, S₂ s₁ (rel₁ ω₁) s₂ σ₂)]
    _ ≤ (avg fun σ₂ => prCoin (fun σ₁ => E (S₁ s₁ σ₁, S₂ s₁ (S₁ s₁ σ₁) s₂ σ₂)) + ε₁ s₁) + ε₂ s₂ :=
        add_le_add (avg_mono fun σ₂ => h₁ s₁ fun v => E (v, S₂ s₁ v s₂ σ₂)) le_rfl
    _ = prCoin (fun q : Ωs₁ × Ωs₂ => E (S₁ s₁ q.1, S₂ s₁ (S₁ s₁ q.1) s₂ q.2)) + (ε₁ s₁ + ε₂ s₂) := by
        rw [avg_add, avg_const, add_assoc,
          prCoin_prod_split (fun σ₁ σ₂ => E (S₁ s₁ σ₁, S₂ s₁ (S₁ s₁ σ₁) s₂ σ₂)),
          swap fun σ₁ σ₂ => E (S₁ s₁ σ₁, S₂ s₁ (S₁ s₁ σ₁) s₂ σ₂)]

end FlockSoundness.ZK
