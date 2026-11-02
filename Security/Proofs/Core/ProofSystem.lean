import Definitions.Core.ProofSystem
import Definitions.Claims.Recursion
import Proofs.Flock.Claims.Prob

/-!
# Proof systems: the interface at the interaction model's forms

At the plain sample the interface is the interaction model's own forms: `PExtractor.rep` is `ERep`, `PrBadOn` is
`PrBad`, `RunFn.B` is `B`, so KS with `B`'s term alone is `Claims.KS` (`ProofSystem.ks_iff_claims`, `ks_iff_inner`).
Against aborting provers `PrBadOn` at `E^rep` is `Inner.PrBadAbort` and `B`'s term is `PrBAbort`'s, so KS there is
`Inner.KSAbort` (`prBadOn_abort`, `ksAbort_iff_inner`). Zero knowledge statement by statement is `ZKmalRx`
(`ZkSystem.zk_iff`).

KS gives KS against aborting provers when the extractor reruns the judged sample once and reads only its view, and only
at terms an abort doesn't shrink (`ProofSystem.ksAbort_of_viewOnly`). At `B`'s term it doesn't follow: an aborted run
makes no calls, so `B`'s term at an aborting prover can be `0` where its plain prover's is not, and a prover that aborts
exactly where `B` would find its collisions keeps its accepted runs.
-/

namespace Verity

open Claims
open scoped NNReal

private theorem pr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    Pr (fun a => p (e a)) = Pr p := by
  classical
  rw [Pr_eq_sum (fun a => p (e a)), Pr_eq_sum p, Fintype.card_congr e]
  congr 1
  exact e.sum_comp fun b => if p b then (1 : ℝ) else 0

section Plain

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Inhabited Coin] in
/-- `E^rep` over any sample space is `ERep` at the plain sample. -/
theorem PExtractor.rep_toExtractor {Op : Type} (V : Verifier Coin) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) :
    (PExtractor.rep V m k K opens assemble check).toExtractor = ERep V m k K opens assemble check :=
  rfl

/-- `PrBadOn` at a prover's plain runs is `PrBad`. -/
theorem prBadOn_plain (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : PExtractor Coin W V) (x : Msg)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBadOn H V R E x (RunFn.plain H V x P) = PrBad H V R E.toExtractor x P :=
  rfl

/-- `B` at a prover's plain runs is `B`. -/
theorem RunFn.B_plain (H : Msg → Msg) (V : Verifier Coin) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (hk : 0 < V.k x) : (RunFn.plain H V x P).B hk = Claims.B H V P x hk :=
  rfl

/-- `B`'s term at a prover's plain runs is `√(γ · Pr[B])`. -/
theorem KTerm.val_B_plain (H : Msg → Msg) (V : Verifier Coin) (γ : Msg → ℝ) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) :
    (KTerm.B γ : KTerm Coin V).val H x (RunFn.plain H V x P) = Real.sqrt (γ x * PrB H V P x) := by
  unfold KTerm.val KTerm.B PrB
  by_cases hk : 0 < V.k x
  · simp only [hk, ↓reduceDIte]
    rfl
  · simp [hk]

/-- KS against every prover, at the plain sample. -/
theorem ksOn_plain_iff (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : PExtractor Coin W V) (κ : Msg → ℝ)
    (terms : List (KTerm Coin V)) :
    KSOn H V R E κ terms (RunClass.plain H V) ↔
      ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
        PrBad H V R E.toExtractor x P ≤ κ x + (terms.map fun τ => τ.val H x (RunFn.plain H V x P)).sum := by
  constructor
  · intro h x Ω _ _ P
    exact h x (RunFn.plain H V x P) ⟨Ω, inferInstance, inferInstance, P, rfl⟩
  · rintro h x F ⟨Ω, hΩ, hne, P, rfl⟩
    exact @h x Ω hΩ hne P

/-- **A proof system's KS at `B`'s term alone is `Claims.KS`** at its extractor's plain form. -/
theorem ProofSystem.ks_iff_claims (H : Msg → Msg) (S : ProofSystem Coin W) (γ : Msg → ℝ)
    (h : S.terms = [KTerm.B γ]) : S.KS H ↔ Claims.KS H S.V S.R (S.E H).toExtractor S.κ γ := by
  unfold ProofSystem.KS Claims.KS
  rw [ksOn_plain_iff, h]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, KTerm.val_B_plain]

/-- KS at `E^rep` and `B`'s term is the inner protocol's own `KS`. -/
theorem ks_iff_inner (H : Msg → Msg) (I : Recursion.Inner Coin W) :
    KSOn H I.V I.R (PExtractor.rep I.V I.waits I.k I.cap I.opens I.assemble I.check) I.κ [KTerm.B I.γ]
      (RunClass.plain H I.V) ↔ I.KS H := by
  rw [ksOn_plain_iff]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, KTerm.val_B_plain]
  rfl

end Plain

section Abort

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`PrBadOn` at `E^rep` and an aborting prover's runs is `Inner.PrBadAbort`**, by unfolding. -/
theorem prBadOn_abort (H : Msg → Msg) (I : Recursion.Inner Coin W) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A] (keep : Ω → (Fin (I.V.k x) → Coin) → A → Bool) :
    PrBadOn H I.V I.R (PExtractor.rep I.V I.waits I.k I.cap I.opens I.assemble I.check) x
        (RunFn.abort H I.V x P A keep) = I.PrBadAbort H x P keep :=
  rfl

/-- `B`'s term at an aborting prover's runs is `√(γ · Pr[B])` at `BAbort`: the same draws, regrouped. -/
theorem KTerm.val_B_abort (H : Msg → Msg) (V : Verifier Coin) (γ : Msg → ℝ) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A] (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) :
    (KTerm.B γ : KTerm Coin V).val H x (RunFn.abort H V x P A keep) =
      Real.sqrt (γ x * Recursion.PrBAbort H V P x keep) := by
  unfold KTerm.val KTerm.B Recursion.PrBAbort
  by_cases hk : 0 < V.k x
  · simp only [hk, ↓reduceDIte]
    congr 2
    let e : BCoins Ω Coin (V.k x) × A × A ≃
        Fin (V.k x) × Ω × ((Fin (V.k x) → Coin) × A) × ((Fin (V.k x) → Coin) × A) :=
      { toFun := fun b => (b.1.1, b.1.2.1, (b.1.2.2.1, b.2.1), (b.1.2.2.2, b.2.2))
        invFun := fun s => ((s.1, s.2.1, s.2.2.1.1, s.2.2.2.1), s.2.2.1.2, s.2.2.2.2)
        left_inv := fun _ => rfl
        right_inv := fun _ => rfl }
    exact (pr_equiv e _).symm
  · simp [hk]

/-- KS against provers that may abort, at their runs. -/
theorem ksOn_aborting_iff (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : PExtractor Coin W V) (κ : Msg → ℝ)
    (terms : List (KTerm Coin V)) :
    KSOn H V R E κ terms (RunClass.aborting H V) ↔
      ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A]
        (keep : Ω → (Fin (V.k x) → Coin) → A → Bool),
        PrBadOn H V R E x (RunFn.abort H V x P A keep) ≤
          κ x + (terms.map fun τ => τ.val H x (RunFn.abort H V x P A keep)).sum := by
  constructor
  · intro h x Ω _ _ P A _ _ keep
    exact h x (RunFn.abort H V x P A keep)
      ⟨Ω, inferInstance, inferInstance, P, A, inferInstance, inferInstance, keep, rfl⟩
  · rintro h x F ⟨Ω, hΩ, hne, P, A, hA, hAne, keep, rfl⟩
    exact @h x Ω hΩ hne P A hA hAne keep

/-- **KS against aborting provers at `E^rep` and `B`'s term is `Inner.KSAbort`.** -/
theorem ksAbort_iff_inner (H : Msg → Msg) (I : Recursion.Inner Coin W) :
    KSOn H I.V I.R (PExtractor.rep I.V I.waits I.k I.cap I.opens I.assemble I.check) I.κ [KTerm.B I.γ]
      (RunClass.aborting H I.V) ↔ I.KSAbort H := by
  rw [ksOn_aborting_iff]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, KTerm.val_B_abort, prBadOn_abort]
  rfl

/-- **KS against aborting provers from KS**, for an extractor that reruns the judged sample once and reads only its view
(`g`), at terms whose value at a prover's runs is at most their value at its aborting runs: an aborted run is rejected,
so the aborting prover's bad event is its plain prover's, or nothing. `B`'s term is not one of these. -/
theorem ProofSystem.ksAbort_of_viewOnly (H : Msg → Msg) (S : ProofSystem Coin W) (g : Msg → Seen → Option W)
    (hE : ∀ x (D : SampleSpace Coin (S.V.k x)) d c, (S.E H).ext x D d c = .rerun d fun v => .ret (g x v))
    (hτ : ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A]
      (keep : Ω → (Fin (S.V.k x) → Coin) → A → Bool),
      (S.terms.map fun τ => τ.val H x (RunFn.plain H S.V x P)).sum ≤
        (S.terms.map fun τ => τ.val H x (RunFn.abort H S.V x P A keep)).sum)
    (hks : S.KS H) : S.KSAbort H := by
  rintro x F ⟨Ω, hΩ, hne, P, A, hA, hAne, keep, rfl⟩
  have hp := hks x (RunFn.plain H S.V x P) ⟨Ω, hΩ, hne, P, rfl⟩
  refine le_trans ?_ (hp.trans (add_le_add le_rfl (hτ x Ω P A keep)))
  let ev : Ω → (Fin (S.V.k x) → Coin) → Prop := fun ω r =>
    (run H P S.V x ω r).out = .acc ∧ ¬ S.R.ext x (g x (run H P S.V x ω r).seen)
  have hplain : PrBadOn H S.V S.R (S.E H) x (RunFn.plain H S.V x P) =
      Pr fun s : Ω × (Fin (S.V.k x) → Coin) => ev s.1 s.2 := by
    refine Eq.trans ?_ (Pr_drop (γ := (S.E H).C x (Fin (S.V.k x) → Coin)) ev)
    refine Pr_congr fun s => ?_
    show _ ∧ ¬ S.R.ext x (((S.E H).ext x _ s.2.1 s.2.2).eval H _).1 ↔ _
    rw [hE]
    rfl
  rw [hplain]
  calc PrBadOn H S.V S.R (S.E H) x (RunFn.abort H S.V x P A keep)
      ≤ Pr (fun s : Ω × ((Fin (S.V.k x) → Coin) × A) × (S.E H).C x ((Fin (S.V.k x) → Coin) × A) =>
          ev s.1 s.2.1.1) := by
        refine Pr_mono fun s hs => ?_
        change (if keep s.1 s.2.1.1 s.2.1.2 then run H P S.V x s.1 s.2.1.1 else ⟨[], [], [], .rej⟩).out = .acc ∧
          ¬ S.R.ext x (((S.E H).ext x (SampleSpace.withAbort Coin (S.V.k x) A) s.2.1 s.2.2).eval H
            fun t => if keep s.1 t.1 t.2 then run H P S.V x s.1 t.1 else ⟨[], [], [], .rej⟩).1 at hs
        rw [hE x (SampleSpace.withAbort Coin (S.V.k x) A) s.2.1 s.2.2] at hs
        by_cases hkeep : keep s.1 s.2.1.1 s.2.1.2 = true
        · simp only [RP.eval, hkeep, ↓reduceIte] at hs
          exact hs
        · simp [hkeep] at hs
    _ = Pr (fun s : Ω × ((Fin (S.V.k x) → Coin) × A) => ev s.1 s.2.1) :=
        Pr_drop (γ := (S.E H).C x ((Fin (S.V.k x) → Coin) × A))
          fun (ω : Ω) (t : (Fin (S.V.k x) → Coin) × A) => ev ω t.1
    _ = Pr (fun s : Ω × (Fin (S.V.k x) → Coin) => ev s.1 s.2) := Pr_drop (γ := A) ev

end Abort

/-- **Zero knowledge statement by statement is `ZKmalRx`**, the masked prover's tape indexed by the statement. -/
theorem ZkSystem.zk_iff {Coin W : Type} (H : Msg → Msg) (O : ZkSystem Coin W) :
    O.ZK H ↔ ∀ t, 0 < t → ∀ δ₁ : ℝ≥0, O.Hides H δ₁ →
      ZKmalRx O.Nonce O.Rest H O.kZK O.Pz O.R (O.Sim t) (O.εzk t δ₁) (O.γzk t) := by
  constructor
  · intro h t ht δ₁ hδ x w hR Θ Vs θ hk Ev
    exact h t ht δ₁ hδ x x w ⟨rfl, hR⟩ Θ Vs θ hk Ev
  · rintro h t ht δ₁ hδ x₀ x w ⟨rfl, hR⟩ Θ Vs θ hk Ev
    exact h t ht δ₁ hδ x w hR Θ Vs θ hk Ev

end Verity
