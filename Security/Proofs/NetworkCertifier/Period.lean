import Definitions.NetworkCertifier.Guarantees.Period
import Proofs.NetworkCertifier.SampledLeak
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.Warden
import Proofs.NetworkCertifier.TapRows

/-!
# The leak through the wardens over a period of sampled audits

A window that meets `WindowGood`, whose observations the tap registers (`TapOpens`), has them among the sequences
`AccSeqsSampled` counts at its registration, with the registration's values at `reg`
(`mem_accSeqsSampled_of_windowGood`). The data through the wardens is then the cells framed from one of the products of
those sets (`WardenCertifierEgress`), and the views of the trusted wardens number at most
`#Ss · ∏_c (cap_c + 1)` times that product's size (`encard_wardenViews_le`), which bounds the secrets the views decode
to (`encard_periodGood_le`). `periodLeak` adds the per-window bound at the tap's rows (`sampledLeak_tap`).
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
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell) (w : Dev)
    (run : Dev → WardenSchedule Conn × Srv → Conn → List (ℕ × CellView Cell))
    (htrust : Verity.Assumptions.TrustedHost w run fun p => wardenCode p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run w (schedOf z, srvOf z)) = z)
    (obsOf : Srv → Fin n → List (Obs L K S)) (cells : (Fin n → List (Obs L K S)) → Conn → List Cell)
    (hegress : WardenCertifierEgress arr obsOf cells) (x : Z → Fin n → Msg) (wit : Z → Fin n → Wit)
    (tops : Stmt → List (Opening Bool))
    (htap : ∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → TapOpens H D.commit (lay s) (k s) (op i) s (tops s)
      (obsOf (srvOf z) i)) :
    {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
        (obsOf (srvOf z) i)}.encard ≤
      Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
        (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
  set G := {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
    (obsOf (srvOf z) i)} with hG
  rcases G.eq_empty_or_nonempty with he | ⟨z₀, hz₀⟩
  · rw [he, Set.encard_empty]
    exact zero_le
  let A : ∀ i : Fin n, Set (List (Obs L K S)) := fun i =>
    AccSeqsSampled (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
      (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) δ (op i) Dj
  have hgood : ∀ z ∈ G, ∀ i, obsOf (srvOf z) i ∈ A i := fun z hz i =>
    (mem_accSeqsSampled_of_windowGood (hPw i) (htap z hz.1 i) (hz.2 i)).2
  have hfits : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)) := fun i =>
    (mem_accSeqsSampled_of_windowGood (hPw i) (htap z₀ hz₀.1 i) (hz₀.2 i)).1
  let Ds : Set (Conn → List Cell) := cells '' Set.pi Set.univ A
  have hD : ∀ σ ∈ srvOf '' G, (fun c => (arr σ c).stream) ∈ Ds := by
    rintro _ ⟨z, hz, rfl⟩
    exact ⟨obsOf (srvOf z), fun i _ => hgood z hz i, (hegress (srvOf z)).symm⟩
  have hsub : G ⊆ dec '' {v | ∃ s ∈ Ss, ∃ σ ∈ srvOf '' G, run w (s, σ) = v} := fun z hz =>
    ⟨run w (schedOf z, srvOf z), ⟨schedOf z, hs z hz.1, srvOf z, ⟨z, hz, rfl⟩, rfl⟩, hdz z hz.1⟩
  have hDs : Ds.encard ≤ ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
      (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
    calc Ds.encard ≤ (Set.pi Set.univ A).encard := Set.encard_image_le _ _
      _ = ∏ i, (A i).encard := Set.encard_pi_eq_prod_encard
      _ ≤ ∏ i, ((2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
          (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i) : ℕ) : ℕ∞) :=
        Finset.prod_le_prod fun i _ => encard_accSeqsSampled_le _ _ _ _ _ (hfits i) δ _ _
          (encard_asReg_le _ _ _ _ _) (β i) (hβ i) (op i) Dj
      _ = _ := (Nat.cast_prod _ _).symm
  calc G.encard ≤ (dec '' {v | ∃ s ∈ Ss, ∃ σ ∈ srvOf '' G, run w (s, σ) = v}).encard := Set.encard_le_encard hsub
    _ ≤ {v | ∃ s ∈ Ss, ∃ σ ∈ srvOf '' G, run w (s, σ) = v}.encard := Set.encard_image_le _ _
    _ ≤ Ss.encard * ((∏ c, (cap c + 1) : ℕ) : ℕ∞) * Ds.encard :=
        encard_wardenViews_le arr w run htrust cap Ss hcap _ Ds hD
    _ ≤ Ss.encard * ((∏ c, (cap c + 1) : ℕ) : ℕ∞) * ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
          reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞) :=
        by gcongr
    _ = _ := by rw [mul_assoc, ← Nat.cast_mul]

/-- **The leak through the wardens over a period**: `Guarantees.PeriodLeakAccepted`, from `sampledLeak_tap` with the
registered inputs pinned (`adviceIn_asReg`), and `encard_periodGood_le`. -/
theorem periodLeak : Guarantees.PeriodLeakAccepted := by
  intro Coin Col T _ _ _ _ _ H D hks L K S lay k δ decode hδ hdec reg Dj n sw Pw hPw ρ op β hβ Conn Cell Srv Dev Z _
    arr w run htrust cap Ss hcap Zs schedOf srvOf hs dec hdz obsOf cells hegress x wit tops hb
  exact ⟨fun m hm Ω _ _ P => TapProof.sampledLeak_tap H D hks hb L K S lay k δ decode hδ hdec _ Dj
    (fun _ _ _ _ => adviceIn_asReg K _ _ _ _ _ _ _ _) tops m hm Ω P,
    encard_periodGood_le H D L K S lay k δ reg Dj sw Pw hPw ρ op β hβ arr w run htrust cap Ss hcap Zs schedOf
      srvOf hs dec hdz obsOf cells hegress x wit tops⟩

end Deploy

end NetTiming
