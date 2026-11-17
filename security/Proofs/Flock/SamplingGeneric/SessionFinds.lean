module

public import Proofs.Flock.SamplingGeneric.Session
public import Proofs.Flock.SamplingGeneric.OneWait
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob

/-! What a session's extraction finds, its one wait finds at half the rate. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims.Sampling (forkOn Ex_resample forkOn_empty forkOn_sub Pr_unit Ex_forkOn Pr_prod_eq_Ex)
open Claims.Recursion (serve viewing)
open Finset

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

def FoundWith (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (u : Fin (L.nPU x)) (r' : Fin (L.V.k x) → Coin) (c : A.EC (L.sh x u)) (G : Msg → A.W → Prop) : Prop :=
  ∃ xS vS, L.session x u (run' r').seen = some (xS, vS) ∧ ∃ h : A.shape xS = L.sh x u, ∃ w,
    ((serve (fun r'' => viewing (L.view x u) (L.place x u xS r' r''))
        (A.ext xS (L.coinsOf x u xS r') (cast (congrArg A.EC h.symm) c))).eval H run').1 = some w ∧ G xS w

theorem found_iff_foundWith (H : Msg → Msg) (L : Compiled Coin A) (x : Msg)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (u : Fin (L.nPU x)) (r' : Fin (L.V.k x) → Coin)
    (c : A.EC (L.sh x u)) : L.Found H x run' u r' c ↔ L.FoundWith H x run' u r' c A.R :=
  Iff.rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem forkOn_emb (L : Compiled Coin A) {x : Msg} {u : ℕ} {xS : Msg} (hcard : A.V.k xS = (L.rounds x u).card)
    (R0 : Fin (L.V.k x) → Coin) {σ σ' : Fin (L.V.k x) → Coin}
    (h : (fun i => σ (L.emb x u xS hcard i)) = fun i => σ' (L.emb x u xS hcard i)) :
    forkOn (L.rounds x u) R0 σ = forkOn (L.rounds x u) R0 σ' := by
  funext t
  unfold forkOn
  by_cases ht : t ∈ L.rounds x u
  · rw [ite_eq_left ht, ite_eq_left ht]
    have hr : t ∈ Set.range (L.emb x u xS hcard) := by
      unfold emb
      rw [Finset.range_orderEmbOfFin]
      exact ht
    obtain ⟨i, rfl⟩ := hr
    exact congrFun h i
  · rw [ite_eq_right ht, ite_eq_right ht]

theorem session_finds {H : Msg → Msg} (hF : A.OneWaitFinds H) (L : Compiled Coin A) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (R0 : Fin (L.V.k x) → Coin) (u : Fin (L.nPU x)) (xS : Msg) (P' : Prover Unit)
    (hshape : A.shape xS = L.sh x u) (hcard : A.V.k xS = (L.rounds x u).card)
    (hacc : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, r' t = R0 t) →
      (run H P L.V x ω r').out = .acc → (run H P' A.V xS () (L.coinsOf x u xS r')).out = .acc)
    (hview : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = R0 t) →
      L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen))
    (G : (Fin (L.V.k x) → Coin) → Msg → A.W → Prop) :
    Pr (fun s : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) =>
        (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
          L.FoundWith H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2
            (G (forkOn (L.rounds x u) R0 s.1))) ≤
      2 * Pr (fun s : (Fin (L.V.k x) → Coin) × A.oneWait.EC (L.sh x u) =>
        (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
          L.oneWait.FoundWith H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2
            (G (forkOn (L.rounds x u) R0 s.1))) +
        2 * A.κ xS + Real.sqrt (A.γ xS * PrB H A.V P' xS) := by
  have hag0 : ∀ σ : Fin (L.V.k x) → Coin, ∀ t ∉ L.rounds x u, forkOn (L.rounds x u) R0 σ t = R0 t :=
    fun σ t ht => by simp [forkOn, ht]
  let G' : (Fin (A.V.k xS) → Coin) → A.W → Prop := fun ρ w =>
    ∃ σ : Fin (L.V.k x) → Coin, (fun i => σ (L.emb x u xS hcard i)) = ρ ∧
      (run H P L.V x ω (forkOn (L.rounds x u) R0 σ)).out = .acc ∧ G (forkOn (L.rounds x u) R0 σ) xS w
  let Q : (Fin (A.V.k xS) → Coin) → A.EC (A.shape xS) → Prop := fun ρ c =>
    (run H P' A.V xS () ρ).out = .acc ∧ ∃ w, (A.E.eval H P' xS () ρ c).1 = some w ∧ G' ρ w
  let Q₁ : (Fin (A.V.k xS) → Coin) → A.oneWait.EC (A.shape xS) → Prop := fun ρ c =>
    (run H P' A.V xS () ρ).out = .acc ∧ ∃ w, (A.oneWait.E.eval H P' xS () ρ c).1 = some w ∧ G' ρ w
  have himp : ∀ s : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u),
      (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
          L.FoundWith H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2
            (G (forkOn (L.rounds x u) R0 s.1)) →
        Q (fun i => s.1 (L.emb x u xS hcard i)) (cast (congrArg A.EC hshape.symm) s.2) := by
    rintro s ⟨h1, xS', vS, hs', h, w, hw, hG⟩
    have hsv := hview _ fun t ht _ => hag0 s.1 t ht
    rw [hsv, Option.some.injEq, Prod.mk.injEq] at hs'
    obtain ⟨rfl, -⟩ := hs'
    have ha := hacc _ (hag0 s.1) h1
    rw [L.coinsOf_forkOn hcard] at ha
    rw [L.served_eval H P ω P' R0 _ hcard hview (hag0 s.1), L.coinsOf_forkOn hcard] at hw
    exact ⟨ha, w, hw, s.1, rfl, h1, hG⟩
  have himp₁ : ∀ s : (Fin (L.V.k x) → Coin) × A.oneWait.EC (L.sh x u),
      Q₁ (fun i => s.1 (L.emb x u xS hcard i)) (cast (congrArg A.oneWait.EC hshape.symm) s.2) →
        (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
          L.oneWait.FoundWith H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2
            (G (forkOn (L.rounds x u) R0 s.1)) := by
    rintro s ⟨-, w, hw, σ, hσ, h1, hG⟩
    rw [L.forkOn_emb hcard R0 hσ] at h1 hG
    refine ⟨h1, xS, _, hview _ fun t ht _ => hag0 s.1 t ht, hshape, w, ?_, hG⟩
    rw [L.oneWait.served_eval H P ω P' R0 _ hcard hview (hag0 s.1), L.oneWait.coinsOf_forkOn hcard]
    exact hw
  have hfin := hF xS Unit P' G'
  calc _ ≤ Pr (fun s : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) =>
          Q (fun i => s.1 (L.emb x u xS hcard i)) (cast (congrArg A.EC hshape.symm) s.2)) := Pr_mono himp
    _ = Pr (fun p : (Fin (A.V.k xS) → Coin) × A.EC (A.shape xS) => Q p.1 p.2) :=
        Pr_restrict_shape (L.emb x u xS hcard) (L.emb x u xS hcard).injective hshape Q
    _ = Pr (fun s : Unit × (Fin (A.V.k xS) → Coin) × A.E.C xS => (run H P' A.V xS s.1 s.2.1).out = .acc ∧
          ∃ w, (A.E.eval H P' xS s.1 s.2.1 s.2.2).1 = some w ∧ G' s.2.1 w) :=
        (Pr_unit (fun p : (Fin (A.V.k xS) → Coin) × A.EC (A.shape xS) => Q p.1 p.2)).symm
    _ ≤ 2 * Pr (fun s : Unit × (Fin (A.V.k xS) → Coin) × A.oneWait.E.C xS =>
          (run H P' A.V xS s.1 s.2.1).out = .acc ∧
            ∃ w, (A.oneWait.E.eval H P' xS s.1 s.2.1 s.2.2).1 = some w ∧ G' s.2.1 w) +
          2 * A.κ xS + Real.sqrt (A.γ xS * PrB H A.V P' xS) := hfin
    _ = 2 * Pr (fun p : (Fin (A.V.k xS) → Coin) × A.oneWait.EC (A.shape xS) => Q₁ p.1 p.2) +
          2 * A.κ xS + Real.sqrt (A.γ xS * PrB H A.V P' xS) :=
        congrArg (fun t : ℝ => 2 * t + 2 * A.κ xS + Real.sqrt (A.γ xS * PrB H A.V P' xS))
          (Pr_unit (fun p : (Fin (A.V.k xS) → Coin) × A.oneWait.EC (A.shape xS) => Q₁ p.1 p.2))
    _ = 2 * Pr (fun s : (Fin (L.V.k x) → Coin) × A.oneWait.EC (L.sh x u) =>
          Q₁ (fun i => s.1 (L.emb x u xS hcard i)) (cast (congrArg A.oneWait.EC hshape.symm) s.2)) +
          2 * A.κ xS + Real.sqrt (A.γ xS * PrB H A.V P' xS) := by
        rw [Pr_restrict_shape (A := A.oneWait) (L.emb x u xS hcard) (L.emb x u xS hcard).injective hshape Q₁]
    _ ≤ _ := by
        have := Pr_mono himp₁
        linarith

theorem fiber_finds {H : Msg → Msg} (hF : A.OneWaitFinds H) {L : Compiled Coin A} (hS : L.RunsFixed H) (x : Msg)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (R0 : Fin (L.V.k x) → Coin) (u : Fin (L.nPU x))
    (hu : u.val ∈ L.checked x R0) (G : (Fin (L.V.k x) → Coin) → Msg → A.W → Prop) :
    Pr (fun s : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) =>
        (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
          L.FoundWith H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2
            (G (forkOn (L.rounds x u) R0 s.1))) ≤
      2 * Pr (fun s : (Fin (L.V.k x) → Coin) × A.oneWait.EC (L.sh x u) =>
        (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
          L.oneWait.FoundWith H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2
            (G (forkOn (L.rounds x u) R0 s.1))) +
        (2 * L.κs x + Real.sqrt ((L.V.k x : ℝ) * L.γs x *
          Ex fun σ => forkCollAt H L.V P x ω (forkOn (L.rounds x u) R0 σ))) := by
  obtain ⟨xS, P', hshape, hκb, hγb, hcard, hacc, hview⟩ := hS.runs x Ω P ω R0 u hu
  have hT := L.session_finds hF x P ω R0 u xS P' hshape hcard hacc (fun r' h => (hview r' h).1) G
  have hκ := hκb.trans (L.κ_le_κs x u.isLt)
  have hγ := Real.sqrt_le_sqrt
    (L.session_gamma_le H x P ω R0 u xS P' hshape hγb hcard fun r' h => (hview r' h).2)
  linarith

noncomputable def selHit (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (sel : List ℕ → Option (Fin (L.nPU x)))
    (G : (Fin (L.V.k x) → Coin) → Fin (L.nPU x) → Msg → A.W → Prop) (R : Fin (L.V.k x) → Coin) : ℝ :=
  Pr fun c : L.SlotEC x => (run H P L.V x ω R).out = .acc ∧
    ∃ i u, L.selSlot x sel R = some (i, u) ∧ L.FoundWith H x (run H P L.V x ω) u R (c i u) (G R i)

theorem selHit_none (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (sel : List ℕ → Option (Fin (L.nPU x))) (G : (Fin (L.V.k x) → Coin) → Fin (L.nPU x) → Msg → A.W → Prop)
    {R : Fin (L.V.k x) → Coin} (h : L.selSlot x sel R = none) : L.selHit H P x ω sel G R = 0 := by
  unfold selHit
  rw [h]
  exact (Pr_congr fun _ => ⟨fun ⟨_, _, _, hu, _⟩ => Option.some_ne_none _ hu.symm, False.elim⟩).trans Pr_false

theorem selHit_forkOn {H : Msg → Msg} {L : Compiled Coin A} (hS : L.RunsFixed H) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (sel : List ℕ → Option (Fin (L.nPU x)))
    (G : (Fin (L.V.k x) → Coin) → Fin (L.nPU x) → Msg → A.W → Prop) {R0 : Fin (L.V.k x) → Coin}
    {i u : Fin (L.nPU x)} (h : L.selSlot x sel R0 = some (i, u)) (σ : Fin (L.V.k x) → Coin) :
    L.selHit H P x ω sel G (forkOn (L.rounds x u) R0 σ) =
      Pr fun c : A.EC (L.sh x u) => (run H P L.V x ω (forkOn (L.rounds x u) R0 σ)).out = .acc ∧
        L.FoundWith H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 σ) c
          (G (forkOn (L.rounds x u) R0 σ) i) := by
  let Q : A.EC (L.sh x u) → Prop := fun c => (run H P L.V x ω (forkOn (L.rounds x u) R0 σ)).out = .acc ∧
    L.FoundWith H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 σ) c (G (forkOn (L.rounds x u) R0 σ) i)
  unfold selHit
  rw [selSlot_forkOn hS h σ]
  refine (Pr_congr fun c => ⟨fun ⟨ha, i', u', hiu, hw⟩ => ?_, fun ⟨ha, hw⟩ => ⟨ha, i, u, rfl, hw⟩⟩).trans
    ((Pr_pi_eval_dep (E := fun _ : Fin (L.nPU x) => L.BlockEC x) i fun b : L.BlockEC x => Q (b u)).trans
      (Pr_pi_eval_dep (E := fun v : Fin (L.nPU x) => A.EC (L.sh x v)) u Q))
  rw [Option.some.injEq, Prod.mk.injEq] at hiu
  obtain ⟨rfl, rfl⟩ := hiu
  exact ⟨ha, hw⟩

theorem selHit_fiber {H : Msg → Msg} (hF : A.OneWaitFinds H) {L : Compiled Coin A} (hS : L.RunsFixed H)
    (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (sel : List ℕ → Option (Fin (L.nPU x)))
    (G : (Fin (L.V.k x) → Coin) → Fin (L.nPU x) → Msg → A.W → Prop) {R0 : Fin (L.V.k x) → Coin}
    {i u : Fin (L.nPU x)} (h : L.selSlot x sel R0 = some (i, u)) :
    (Ex fun σ => L.selHit H P x ω sel G (forkOn (L.rounds x u) R0 σ)) ≤
      2 * (Ex fun σ => L.oneWait.selHit H P x ω sel G (forkOn (L.rounds x u) R0 σ)) +
        (2 * L.κs x + Real.sqrt ((L.V.k x : ℝ) * L.γs x *
          Ex fun σ => forkCollAt H L.V P x ω (forkOn (L.rounds x u) R0 σ))) := by
  have hS₁ : L.oneWait.RunsFixed H := ⟨hS.fixed, hS.runs⟩
  have h₁ : L.oneWait.selSlot x sel R0 = some (i, u) := h
  simp only [selHit_forkOn hS x P ω sel G h]
  rw [show (fun σ => L.oneWait.selHit H P x ω sel G (forkOn (L.rounds x u) R0 σ)) = fun σ =>
      Pr fun c : A.oneWait.EC (L.sh x u) => (run H P L.V x ω (forkOn (L.rounds x u) R0 σ)).out = .acc ∧
        L.oneWait.FoundWith H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 σ) c
          (G (forkOn (L.rounds x u) R0 σ) i) from funext fun σ => selHit_forkOn hS₁ x P ω sel G h₁ σ]
  rw [← Pr_prod_eq_Ex (fun s : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) =>
      (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
        L.FoundWith H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2
          (G (forkOn (L.rounds x u) R0 s.1) i)),
    ← Pr_prod_eq_Ex (fun s : (Fin (L.V.k x) → Coin) × A.oneWait.EC (L.sh x u) =>
      (run H P L.V x ω (forkOn (L.rounds x u) R0 s.1)).out = .acc ∧
        L.oneWait.FoundWith H x (run H P L.V x ω) u (forkOn (L.rounds x u) R0 s.1) s.2
          (G (forkOn (L.rounds x u) R0 s.1) i))]
  exact fiber_finds hF hS x P ω R0 u (mem_of_selSlot h) fun R => G R i

theorem resample_finds {H : Msg → Msg} (hF : A.OneWaitFinds H) {L : Compiled Coin A} (hS : L.RunsFixed H)
    (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (sel : List ℕ → Option (Fin (L.nPU x)))
    (G : (Fin (L.V.k x) → Coin) → Fin (L.nPU x) → Msg → A.W → Prop) (r d : Fin (L.V.k x) → Coin) :
    (Ex fun σ => L.selHit H P x ω sel G
        (forkOn (L.secondStage x) r (forkOn (L.selRounds x sel (forkOn (L.secondStage x) r d)) d σ))) ≤
      2 * (Ex fun σ => L.oneWait.selHit H P x ω sel G
        (forkOn (L.secondStage x) r (forkOn (L.selRounds x sel (forkOn (L.secondStage x) r d)) d σ))) +
        (2 * L.κs x + Real.sqrt ((L.V.k x : ℝ) * L.γs x * Ex fun σ => forkCollAt H L.V P x ω
          (forkOn (L.secondStage x) r (forkOn (L.selRounds x sel (forkOn (L.secondStage x) r d)) d σ)))) := by
  cases h : L.selSlot x sel (forkOn (L.secondStage x) r d) with
  | none =>
    have h₁ : L.oneWait.selSlot x sel (forkOn (L.secondStage x) r d) = none := h
    rw [selRounds_none h]
    simp only [forkOn_empty]
    rw [selHit_none H L P x ω sel G h, selHit_none H L.oneWait P x ω sel G h₁,
      Ex_const (α := Fin (L.V.k x) → Coin) (0 : ℝ)]
    have := L.κs_nonneg x
    have := Real.sqrt_nonneg ((L.V.k x : ℝ) * L.γs x *
      Ex fun _ : Fin (L.V.k x) → Coin => forkCollAt H L.V P x ω (forkOn (L.secondStage x) r d))
    linarith
  | some iu =>
    obtain ⟨i, u⟩ := iu
    rw [selRounds_some h]
    simp only [forkOn_sub (hS.fixed x _ u (mem_of_selSlot h)).1]
    exact selHit_fiber hF hS x P ω sel G h

def SessionFinds (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    ∃ β : Ω → (Fin (L.V.k x) → Coin) → ℝ, (∀ ω r, 0 ≤ β ω r) ∧
      Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x ∧
      ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x)))
        (G : (Fin (L.V.k x) → Coin) → Fin (L.nPU x) → Msg → A.W → Prop),
        Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
          (run H P L.V x ω (forkOn (L.secondStage x) r p.1)).out = .acc ∧
            ∃ i, sel (L.checked x (forkOn (L.secondStage x) r p.1)) = some i ∧
              ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.secondStage x) r p.1))[i.val]? = some u.val ∧
                L.FoundWith H x (run H P L.V x ω) u (forkOn (L.secondStage x) r p.1) (p.2 i u)
                  (G (forkOn (L.secondStage x) r p.1) i)) ≤
          2 * Pr (fun p : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
            (run H P L.V x ω (forkOn (L.secondStage x) r p.1)).out = .acc ∧
              ∃ i, sel (L.checked x (forkOn (L.secondStage x) r p.1)) = some i ∧
                ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.secondStage x) r p.1))[i.val]? = some u.val ∧
                  L.oneWait.FoundWith H x (run H P L.V x ω) u (forkOn (L.secondStage x) r p.1) (p.2 i u)
                    (G (forkOn (L.secondStage x) r p.1) i)) +
            (2 * L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r))

theorem sessionFinds_of {H : Msg → Msg} (hF : A.OneWaitFinds H) {L : Compiled Coin A} (hS : L.RunsFixed H) :
    L.SessionFinds H := by
  intro x Ω _ _ P
  refine ⟨fun ω r => Ex fun d => forkCollAt H L.V P x ω (forkOn (L.secondStage x) r d),
    fun ω r => Ex_nonneg fun d => forkCollAt_nonneg H L.V P x ω _, le_of_eq ?_, fun ω r sel G => ?_⟩
  · rw [Ex_prod, ← Ex_forkCollAt H L.V P x, Ex_prod]
    congr 1
    funext ω
    exact (Ex_prod (fun p : (Fin (L.V.k x) → Coin) × (Fin (L.V.k x) → Coin) =>
      forkCollAt H L.V P x ω (forkOn (L.secondStage x) p.1 p.2))).symm.trans (Ex_forkOn _ _)
  have hres := Ex_resample (fun d => L.selRounds x sel (forkOn (L.secondStage x) r d)) (selRounds_resample hS sel r)
    fun d => forkCollAt H L.V P x ω (forkOn (L.secondStage x) r d)
  have hres' := Ex_resample (fun d => L.selRounds x sel (forkOn (L.secondStage x) r d)) (selRounds_resample hS sel r)
    fun d => L.selHit H P x ω sel G (forkOn (L.secondStage x) r d)
  have hres₁ := Ex_resample (fun d => L.selRounds x sel (forkOn (L.secondStage x) r d)) (selRounds_resample hS sel r)
    fun d => L.oneWait.selHit H P x ω sel G (forkOn (L.secondStage x) r d)
  beta_reduce at hres hres' hres₁
  have step1 : Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω (forkOn (L.secondStage x) r p.1)).out = .acc ∧
        ∃ i, sel (L.checked x (forkOn (L.secondStage x) r p.1)) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.secondStage x) r p.1))[i.val]? = some u.val ∧
            L.FoundWith H x (run H P L.V x ω) u (forkOn (L.secondStage x) r p.1) (p.2 i u)
              (G (forkOn (L.secondStage x) r p.1) i)) ≤
      Ex fun d => L.selHit H P x ω sel G (forkOn (L.secondStage x) r d) := by
    rw [Pr_prod_eq_Ex]
    exact Ex_mono fun d => Pr_mono fun c ⟨ha, i, hs, u, hm, hn⟩ => ⟨ha, i, u, selSlot_eq_some.2 ⟨hs, hm⟩, hn⟩
  have step2 : (Ex fun d => L.oneWait.selHit H P x ω sel G (forkOn (L.secondStage x) r d)) ≤
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
        (run H P L.V x ω (forkOn (L.secondStage x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.secondStage x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.secondStage x) r p.1))[i.val]? = some u.val ∧
              L.oneWait.FoundWith H x (run H P L.V x ω) u (forkOn (L.secondStage x) r p.1) (p.2 i u)
                (G (forkOn (L.secondStage x) r p.1) i)) := by
    rw [Pr_prod_eq_Ex]
    exact Ex_mono fun d => Pr_mono fun c ⟨ha, i, u, hiu, hn⟩ =>
      ⟨ha, i, (selSlot_eq_some.1 hiu).1, u, (selSlot_eq_some.1 hiu).2, hn⟩
  have hsq : (Ex fun d => Real.sqrt ((L.V.k x : ℝ) * L.γs x * Ex fun σ => forkCollAt H L.V P x ω
        (forkOn (L.secondStage x) r (forkOn (L.selRounds x sel (forkOn (L.secondStage x) r d)) d σ)))) ≤
      Real.sqrt ((L.V.k x : ℝ) * L.γs x * Ex fun d => forkCollAt H L.V P x ω (forkOn (L.secondStage x) r d)) := by
    refine (Ex_sqrt_le _ fun d => mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (L.γs_nonneg x))
      (Ex_nonneg fun σ => forkCollAt_nonneg H L.V P x ω _)).trans (le_of_eq ?_)
    rw [Ex_mul_left, hres]
  have hmid : (Ex fun d => L.selHit H P x ω sel G (forkOn (L.secondStage x) r d)) ≤
      2 * (Ex fun d => L.oneWait.selHit H P x ω sel G (forkOn (L.secondStage x) r d)) +
        (2 * L.κs x + Ex fun d => Real.sqrt ((L.V.k x : ℝ) * L.γs x * Ex fun σ => forkCollAt H L.V P x ω
          (forkOn (L.secondStage x) r (forkOn (L.selRounds x sel (forkOn (L.secondStage x) r d)) d σ)))) := by
    rw [← hres', ← hres₁]
    refine (Ex_mono fun d => resample_finds hF hS x P ω sel G r d).trans (le_of_eq ?_)
    rw [Ex_add, Ex_mul_left, Ex_add, Ex_const]
  linarith

end Compiled

end Claims.SamplingGeneric
