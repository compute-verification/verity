import Definitions.NetworkCertifier.Guarantees.Period
import Proofs.NetworkCertifier.SampledLeak
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.Tap

/-!
# The leak through the wardens over a period of sampled audits

A window that meets `WindowGood`, with its trace carrying its observations (`RowsCarried`), has them among the
sequences `AccSeqsSampled` counts at its registration, with the registration's values at `reg`
(`mem_accSeqsSampled_of_windowGood`). The observations are the trusted tap's record of the cells it sends, so the
client's view reads back from them under its schedule (`tap_cells_read`): the views, and the secrets they decode to,
number at most `#Ss` times the size of the product of those sets (`encard_periodGood_le`). The record holds the fail
cells and the padding too, so the count needs no separate charge for them: the bound's `∏_c (cap_c + 1)`, the warden's
fail choices in `EncardWardenViewsLe`, is room here. `periodLeak` adds the per-window bound of `sampledLeak_reg` at
`SampledAuditAccepted`.
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
    (hP₀ : D.partition s₀ = some P₀) {ρ : ℕ → Bool} {op : ℕ → List (Obs L K S) → List Bool} {x : Msg} {w : Wit}
    {o : List (Obs L K S)} (h : WindowGood H D L K S lay k δ reg Dj s₀ ρ x w o) (hc : RowsCarried H D lay k op x w o) :
    (lay s₀).Fits s₀.prog (k s₀) ∧
      o ∈ AccSeqsSampled s₀.prog P₀ s₀.design (lay s₀) (k s₀) (AsReg s₀.prog (lay s₀) (k s₀) (reg s₀) ρ) δ op Dj := by
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hl⟩, ⟨-, s₂, hs₂, hfits⟩, s, hs, ⟨hprog, hdes, hlay, hk, hreg, hpart⟩, hρ, hcov, hlen,
    hj⟩ := h
  have hc := hc s hs
  rw [Option.some_injective _ (hs₁.symm.trans hs)] at hP₁ hl
  rw [Option.some_injective _ (hs₂.symm.trans hs)] at hfits
  rw [Option.some_injective _ (hP₁.symm.trans (hpart.trans hP₀))] at hl
  have hmem := hl op o hc hcov hlen hj
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
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
    (x : Z → Fin n → Msg) (wit : Z → Fin n → Wit)
    (hcarry : ∀ z ∈ Zs, ∀ i, RowsCarried H D lay k (op i) (x z i) (wit z i) ((run tap (schedOf z, srvOf z)).2 i)) :
    {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) (x z i) (wit z i)
        ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
      Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
        (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
  set G := {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) (x z i) (wit z i)
    ((run tap (schedOf z, srvOf z)).2 i)} with hG
  rcases G.eq_empty_or_nonempty with he | ⟨z₀, hz₀⟩
  · rw [he, Set.encard_empty]
    exact zero_le
  let A : ∀ i : Fin n, Set (List (Obs L K S)) := fun i =>
    AccSeqsSampled (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
      (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) δ (op i) Dj
  have hgood : ∀ z ∈ G, ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈ A i := fun z hz i =>
    (mem_accSeqsSampled_of_windowGood (hPw i) (hz.2 i) (hcarry z hz.1 i)).2
  have hfits : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)) := fun i =>
    (mem_accSeqsSampled_of_windowGood (hPw i) (hz₀.2 i) (hcarry z₀ hz₀.1 i)).1
  let V : Set (Conn → List (ℕ × CellView Cell)) :=
    (fun p : WardenSchedule Conn × (Fin n → List (Obs L K S)) => f.read p.1 p.2) '' (Ss ×ˢ Set.pi Set.univ A)
  have hsub : G ⊆ dec '' V := fun z hz =>
    ⟨(run tap (schedOf z, srvOf z)).1, ⟨(schedOf z, (run tap (schedOf z, srvOf z)).2),
      ⟨hs z hz.1, fun i _ => hgood z hz i⟩, (tap_cells_read f arr tap run htrust _ _).symm⟩, hdz z hz.1⟩
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

/-- **The leak through the wardens over a period**: `Guarantees.PeriodLeakAccepted`, from `sampledLeak_reg` at
`SampledAuditAccepted` with each statement's largest faulty-bit count, and `encard_periodGood_le`. -/
theorem periodLeak : Guarantees.PeriodLeakAccepted := by
  intro Coin Col T _ _ _ _ _ H D hks L K S lay k δ decode hδ hdec reg Dj n sw Pw hPw ρ op β hβ Conn Cell Srv Dev Z _
    arr f tap run htrust cap Ss hcap Zs schedOf srvOf hs dec hdz x wit hcarry
  classical
  let βmax : Stmt → ℕ := fun s => match D.partition s with
    | some P => (faultySets s.design P δ).sup (faultyBits s.prog P)
    | none => 0
  have hβmax : ∀ s P, D.partition s = some P → ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ βmax s := by
    intro s P hP F hF
    simp only [βmax, hP]
    exact Finset.le_sup (f := faultyBits s.prog P) hF
  exact ⟨(sampledLeak_reg SecurityProofs.SampledAuditAccepted H D hks L K S lay k δ decode hδ hdec reg βmax Dj
    hβmax).1, encard_periodGood_le H D L K S lay k δ reg Dj sw Pw hPw ρ op β hβ arr f tap run htrust cap Ss hcap Zs
    schedOf srvOf hs dec hdz x wit hcarry⟩

end Deploy

end NetTiming
