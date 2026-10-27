import Definitions.NetworkCertifier.Guarantees.Period
import Proofs.NetworkCertifier.SampledLeak
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.Tap
import Proofs.NetworkCertifier.TapRows
import Proofs.Core.UnionBound

/-!
# The leak through the network certifier over a monitoring period of sampled audits

A window that meets `WindowGood`, whose observations the network certifier registers at the transfer gates
(`TapOpens`), has them among the sequences `AccSeqsSampled` counts at its registration, with the registration's values
at `reg` (`mem_accSeqsSampled_of_windowGood`). The observations are the trusted network certifier's record of the cells
it sends, so the client's view reads back from them under its schedule, one its record format holds
(`tap_cells_read`): the views, and the secrets
they decode to, number at most `#Ss` times the size of the product of those sets (`encard_periodGood_le`). The record
holds the fail cells and the padding too, so the count needs no separate charge for them: the bound's
`∏_c (cap_c + 1)`, the fail choices in `EncardWardenViewsLe`, is room here. `periodLeak` adds the per-window bound at
the network certifier's rows (`sampledLeak_tap`).

`periodLeakUnion` takes that bound over the period's live coins (`periodLeakUnion_of`): the other windows' coins fix a
window's statement and prover, and its own coin gives its run's coins uniformly, so its failure over its own coin is the
sampled verifier's `PrBad` there; the union bound over the windows' coins (`Pouw.union_bound_update`) sums them.
-/

namespace NetTiming

open Claims Claims.Sampling Finset

theorem asReg_congr {L S : ℕ} (C : Prog) (lay : Layout L S) (k : ℕ) (R : Finset ℕ) {ρ ρ' : ℕ → Bool}
    (h : ∀ g ∈ R, ρ g = ρ' g) : AsReg C lay k R ρ = AsReg C lay k R ρ' := by
  ext a
  constructor <;> rintro ⟨h1, h2⟩ <;> refine ⟨fun g hg hf => ?_, h2⟩
  · rw [h1 g hg hf, h g hg]
  · rw [h1 g hg hf, ← h g hg]

section Deploy

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem mem_accSeqsSampled_of_windowGood {H : Msg → Msg} {D : Deploy Coin Col T} {L K S : ℕ}
    {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ} {reg : Stmt → Finset ℕ} {Dj : ℕ} {s₀ : Stmt} {P₀ : Partition}
    (hP₀ : D.partition s₀ = some P₀) {ρ : ℕ → Bool} {tops : Stmt → List (Opening Bool)}
    {op : ℕ → List (Obs L K S) → List Bool} {x : Msg} {w : Wit} {o : List (Obs L K S)}
    (htap : ∀ s, D.parse x = some s → TapOpens H D.commit (lay s) (k s) op s (tops s) o)
    (h : WindowGood H D L K S lay k δ reg Dj s₀ ρ tops x w o) :
    (lay s₀).Fits s₀.prog (k s₀) ∧
      o ∈ AccSeqsSampled s₀.prog P₀ s₀.design (lay s₀) (k s₀) (AsReg s₀.prog (lay s₀) (k s₀) (reg s₀) ρ) δ op Dj := by
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hl⟩, ⟨-, s₂, hs₂, hfits⟩, s, hs, ⟨hprog, hdes, hlay, hk, hreg, hpart⟩, hρ, hcov,
    hlen, hj⟩ := h
  rw [Option.some_injective _ (hs₁.symm.trans hs)] at hP₁ hl
  rw [Option.some_injective _ (hs₂.symm.trans hs)] at hfits
  rw [Option.some_injective _ (hP₁.symm.trans (hpart.trans hP₀))] at hl
  have hmem := hl op o (htap s hs) hcov hlen hj
  beta_reduce at hmem
  rw [asReg_congr _ _ _ _ hρ] at hmem
  rw [hprog, hdes, hlay, hk, hreg] at hmem
  rw [hprog, hlay, hk] at hfits
  exact ⟨hfits, hmem⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem encard_periodGood_le (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : Stmt → Layout L S)
    (k : Stmt → ℕ) (δ : ℝ) (reg : Stmt → Finset ℕ) (Dj : ℕ) {n : ℕ} (sw : Fin n → Stmt) (Pw : Fin n → Partition)
    (hPw : ∀ i, D.partition (sw i) = some (Pw i)) (ρ : Fin n → ℕ → Bool)
    (op : Fin n → ℕ → List (Obs L K S) → List Bool) (β : Fin n → ℕ)
    (hβ : ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : TapRecord Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
    (x : Z → Fin n → Msg) (wit : Z → Fin n → Wit) (tops : Stmt → List (Opening Bool))
    (htap : ∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → TapOpens H D.commit (lay s) (k s) (op i) s (tops s)
      ((run tap (schedOf z, srvOf z)).2 i)) :
    {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
        ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
      Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
        (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
  set G := {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
    ((run tap (schedOf z, srvOf z)).2 i)} with hG
  rcases G.eq_empty_or_nonempty with he | ⟨z₀, hz₀⟩
  · rw [he, Set.encard_empty]
    exact zero_le
  let A : ∀ i : Fin n, Set (List (Obs L K S)) := fun i =>
    AccSeqsSampled (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
      (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) δ (op i) Dj
  have hgood : ∀ z ∈ G, ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈ A i := fun z hz i =>
    (mem_accSeqsSampled_of_windowGood (hPw i) (htap z hz.1 i) (hz.2 i)).2
  have hfits : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)) := fun i =>
    (mem_accSeqsSampled_of_windowGood (hPw i) (htap z₀ hz₀.1 i) (hz₀.2 i)).1
  let V : Set (Conn → List (ℕ × CellView Cell)) :=
    (fun p : WardenSchedule Conn × (Fin n → List (Obs L K S)) => f.read p.1 p.2) '' (Ss ×ˢ Set.pi Set.univ A)
  have hsub : G ⊆ dec '' V := fun z hz =>
    ⟨(run tap (schedOf z, srvOf z)).1, ⟨(schedOf z, (run tap (schedOf z, srvOf z)).2),
      ⟨hs z hz.1, fun i _ => hgood z hz i⟩, (tap_cells_read f arr tap run htrust _ (hfit _ (hs z hz.1)) _).symm⟩,
      hdz z hz.1⟩
  have hA : (Set.pi Set.univ A).encard ≤ ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
      reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
    calc (Set.pi Set.univ A).encard = ∏ i, (A i).encard := Set.encard_pi_eq_prod_encard
      _ ≤ ∏ i, ((2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
          (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i) : ℕ) : ℕ∞) :=
        Finset.prod_le_prod fun i _ => encard_accSeqsSampled_le _ _ _ _ _ (hfits i) δ _ _
          (encard_asReg_le _ _ _ _ _) (β i) (hβ i) (op i) Dj
      _ = _ := (Nat.cast_prod _ _).symm
  have hcap1 : 0 < ∏ c, (cap c + 1) := Finset.prod_pos fun c _ => Nat.succ_pos _
  calc G.encard ≤ (dec '' V).encard := Set.encard_le_encard hsub
    _ ≤ V.encard := Set.encard_image_le _ _
    _ ≤ (Ss ×ˢ Set.pi Set.univ A).encard := Set.encard_image_le _ _
    _ = Ss.encard * (Set.pi Set.univ A).encard := Set.encard_prod
    _ ≤ _ := by
        gcongr
        refine hA.trans ?_
        exact_mod_cast Nat.le_mul_of_pos_left _ hcap1

/-- **The leak through the network certifier over a monitoring period**: `Guarantees.PeriodLeakAccepted`, from
`sampledLeak_tap` with the registered inputs pinned (`adviceIn_asReg`), and `encard_periodGood_le`. -/
theorem periodLeak : Guarantees.PeriodLeakAccepted := by
  intro Coin Col T _ _ _ _ _ H D hks L K S lay k δ decode hδ hdec reg Dj n sw Pw hPw ρ op β hβ Conn Cell Srv Dev Z _
    arr f tap run htrust cap Ss hcap hfit Zs schedOf srvOf hs dec hdz x wit tops hb
  exact ⟨fun m hm Ω _ _ P => TapProof.sampledLeak_tap H D hks hb L K S lay k δ decode hδ hdec _ Dj
    (fun _ _ _ _ => adviceIn_asReg K _ _ _ _ _ _ _ _) tops m hm Ω P,
    encard_periodGood_le H D L K S lay k δ reg Dj sw Pw hPw ρ op β hβ arr f tap run htrust cap Ss hfit Zs
      schedOf srvOf hs dec hdz x wit tops⟩

/-- **The union over a monitoring period's windows**, from a bound on one window at every statement in scope and every
prover (`hwin`, `PeriodLeakAccepted`'s first part): window `i`'s failure over its own coin, at every value of the other
coins, is the sampled verifier's `PrBad` at the statement and prover those coins fix (`hg`), at most
`δ + ε_ks,i + ε_bind,i + ε_tap,i`; the union bound over the windows' coins sums them. -/
theorem periodLeakUnion_of (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : Stmt → Layout L S)
    (k : Stmt → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (reg : Stmt → Finset ℕ) (Dj : ℕ)
    (tops : Stmt → List (Opening Bool))
    (hwin : ∀ m, ScopeAcc D lay k m → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      PrBad H D.V
          (RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops)
          (E D decode) m P ≤
        κ D δ m + Real.sqrt (γ D m * PrB H D.V P m) + (Bext H D.V (E D decode) Ω P m).Pr H +
          (Btap H D.V (E D decode) D.commit D.parse tops Ω P m).Pr H)
    {n : ℕ} (Γ : Fin n → Type) [∀ i, Fintype (Γ i)] [∀ i, Inhabited (Γ i)]
    (x : Fin n → ((j : Fin n) → Γ j) → Msg) (hx : ∀ i ζ, ScopeAcc D lay k (x i ζ))
    (Ω : Fin n → ((j : Fin n) → Γ j) → Type) [∀ i ζ, Fintype (Ω i ζ)] [∀ i ζ, Nonempty (Ω i ζ)]
    (P : (i : Fin n) → (ζ : (j : Fin n) → Γ j) → Prover (Ω i ζ))
    (g : (i : Fin n) → (ζ : (j : Fin n) → Γ j) → Γ i →
      Ω i ζ × (Fin (D.V.k (x i ζ)) → Coin) × (E D decode).C (x i ζ))
    (hg : ∀ i ζ (q : Ω i ζ × (Fin (D.V.k (x i ζ)) → Coin) × (E D decode).C (x i ζ) → Prop),
      Pr (fun y => q (g i ζ y)) = Pr q)
    (εks εbind εtap : Fin n → ℝ)
    (herr : ∀ i ζ, κ D δ (x i ζ) - δ + Real.sqrt (γ D (x i ζ) * PrB H D.V (P i ζ) (x i ζ)) ≤ εks i)
    (hbind : ∀ i ζ, (Bext H D.V (E D decode) (Ω i ζ) (P i ζ) (x i ζ)).Pr H ≤ εbind i)
    (htap : ∀ i ζ, (Btap H D.V (E D decode) D.commit D.parse tops (Ω i ζ) (P i ζ) (x i ζ)).Pr H ≤ εtap i) :
    Pr (fun ξ : (j : Fin n) → Γ j =>
        ∃ i, Guarantees.WindowLeakFails H D L K S lay k δ decode reg Dj tops x Ω P g i (Function.update ξ i default)
          (ξ i)) ≤
      ∑ i, (δ + εks i + εbind i + εtap i) := by
  have hone : ∀ i ζ, Pouw.pr (fun y => Guarantees.WindowLeakFails H D L K S lay k δ decode reg Dj tops x Ω P g i ζ y) ≤
      δ + εks i + εbind i + εtap i := fun i ζ => by
    have hpr := hg i ζ fun s =>
      (run H (P i ζ) D.V (x i ζ) s.1 s.2.1).out = .acc ∧
        ¬ (RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj
          tops).ext (x i ζ) ((E D decode).eval H (P i ζ) (x i ζ) s.1 s.2.1 s.2.2).1
    have h := hwin _ (hx i ζ) _ (P i ζ)
    have he := herr i ζ
    have hb := hbind i ζ
    have ht := htap i ζ
    calc Pouw.pr (fun y => Guarantees.WindowLeakFails H D L K S lay k δ decode reg Dj tops x Ω P g i ζ y)
        = PrBad H D.V
            (RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj
              tops) (E D decode) (x i ζ) (P i ζ) := hpr
      _ ≤ _ := h
      _ ≤ δ + εks i + εbind i + εtap i := by linarith
  refine Pouw.union_bound_update id (fun i ξ => Guarantees.WindowLeakFails H D L K S lay k δ decode reg Dj tops x Ω P g
    i (Function.update ξ i default) (ξ i)) _ fun i ξ => ?_
  simp only [id, Function.update_idem, Function.update_self]
  exact hone i _

/-- **The leak through the network certifier over a monitoring period's live coins**: `Guarantees.PeriodLeakUnion`,
`periodLeakUnion_of` at the per-window bound `periodLeak` has (`sampledLeak_tap`). -/
theorem periodLeakUnion : Guarantees.PeriodLeakUnion := by
  intro Coin Col T _ _ _ _ _ H D hks L K S lay k δ decode hδ hdec reg Dj tops hb n Γ _ _ x hx Ω _ _ P g hg εks εbind
    εtap herr hbind htap
  exact periodLeakUnion_of H D L K S lay k δ decode reg Dj tops
    (fun m hm Ω _ _ P => TapProof.sampledLeak_tap H D hks hb L K S lay k δ decode hδ hdec _ Dj
      (fun _ _ _ _ => adviceIn_asReg K _ _ _ _ _ _ _ _) tops m hm Ω P)
    Γ x hx Ω P g hg εks εbind εtap herr hbind htap

/-- **The union at one window**: at a period of one window whose statement `m` and prover `P` no coin fixes, its coin
giving the run's coins uniformly (`hg`), some window failing is the sampled verifier's `PrBad` at `m` and `P`. -/
theorem pr_windowLeakFails_one (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : Stmt → Layout L S)
    (k : Stmt → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (reg : Stmt → Finset ℕ) (Dj : ℕ)
    (tops : Stmt → List (Opening Bool)) {Γ₀ : Type} [Fintype Γ₀] [Inhabited Γ₀] (m : Msg) (Ω : Type) [Fintype Ω]
    (P : Prover Ω) (g : Γ₀ → Ω × (Fin (D.V.k m) → Coin) × (E D decode).C m)
    (hg : ∀ q : Ω × (Fin (D.V.k m) → Coin) × (E D decode).C m → Prop, Pr (fun y => q (g y)) = Pr q) :
    Pr (fun ξ : Fin 1 → Γ₀ => ∃ i, Guarantees.WindowLeakFails H D L K S lay k δ decode reg Dj tops (n := 1)
        (Γ := fun _ => Γ₀) (fun _ _ => m) (fun _ _ => Ω) (fun _ _ => P) (fun _ _ => g) i (Function.update ξ i default)
        (ξ i)) =
      PrBad H D.V
        (RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops)
        (E D decode) m P := by
  have hex : ∀ ξ : Fin 1 → Γ₀, (∃ i, Guarantees.WindowLeakFails H D L K S lay k δ decode reg Dj tops (n := 1)
      (Γ := fun _ => Γ₀) (fun _ _ => m) (fun _ _ => Ω) (fun _ _ => P) (fun _ _ => g) i (Function.update ξ i default)
      (ξ i)) ↔
      Guarantees.WindowLeakFails H D L K S lay k δ decode reg Dj tops (n := 1) (Γ := fun _ => Γ₀) (fun _ _ => m)
        (fun _ _ => Ω) (fun _ _ => P) (fun _ _ => g) 0 (default : Fin 1 → Γ₀) (Equiv.funUnique (Fin 1) Γ₀ ξ) :=
    fun ξ => by
      simp only [Fin.exists_fin_one, Equiv.funUnique_apply]
      rfl
  simp only [hex]
  rw [Pouw.Pr_eq_pr, Pouw.pr_equiv (Equiv.funUnique (Fin 1) Γ₀)]
  exact hg fun s => (run H P D.V m s.1 s.2.1).out = .acc ∧
    ¬ (RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj
      tops).ext m ((E D decode).eval H P m s.1 s.2.1 s.2.2).1

/-- At one window, with each error its prover's own (`ε_ks = κ − δ + √(γ·Pr[B])`, `ε_bind = Pr[B_ext]`,
`ε_tap = Pr[B_tap]`), `PeriodLeakUnion`'s bound is `PeriodLeakAccepted`'s per-window bound, so a statement and prover
where the per-window bound fails (an accepting verifier's) refute the union's too, and the union's hypotheses hold where
the per-window bound's do. -/
theorem windowLeakFails_one_le_iff (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : Stmt → Layout L S)
    (k : Stmt → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (reg : Stmt → Finset ℕ) (Dj : ℕ)
    (tops : Stmt → List (Opening Bool)) {Γ₀ : Type} [Fintype Γ₀] [Inhabited Γ₀] (m : Msg) (Ω : Type) [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (g : Γ₀ → Ω × (Fin (D.V.k m) → Coin) × (E D decode).C m)
    (hg : ∀ q : Ω × (Fin (D.V.k m) → Coin) × (E D decode).C m → Prop, Pr (fun y => q (g y)) = Pr q) :
    Pr (fun ξ : Fin 1 → Γ₀ => ∃ i, Guarantees.WindowLeakFails H D L K S lay k δ decode reg Dj tops (n := 1)
        (Γ := fun _ => Γ₀) (fun _ _ => m) (fun _ _ => Ω) (fun _ _ => P) (fun _ _ => g) i (Function.update ξ i default)
        (ξ i)) ≤
      ∑ _i : Fin 1, (δ + (κ D δ m - δ + Real.sqrt (γ D m * PrB H D.V P m)) + (Bext H D.V (E D decode) Ω P m).Pr H +
        (Btap H D.V (E D decode) D.commit D.parse tops Ω P m).Pr H) ↔
    PrBad H D.V
        (RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops)
        (E D decode) m P ≤
      κ D δ m + Real.sqrt (γ D m * PrB H D.V P m) + (Bext H D.V (E D decode) Ω P m).Pr H +
        (Btap H D.V (E D decode) D.commit D.parse tops Ω P m).Pr H := by
  rw [pr_windowLeakFails_one H D L K S lay k δ decode reg Dj tops m Ω P g hg, Fin.sum_univ_one]
  constructor <;> intro h <;> linarith

end Deploy

end NetTiming
