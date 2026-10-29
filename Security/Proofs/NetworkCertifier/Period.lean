import Definitions.NetworkCertifier.Guarantees.Period
import Proofs.NetworkCertifier.SampledLeak
import Proofs.NetworkCertifier.SampledLeakReg
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.NetworkCertifier.Tap
import Proofs.NetworkCertifier.TapRows

/-!
# The leak through the network certifier over a monitoring period of sampled audits

A window that meets `WindowGood`, whose observations the network certifier registers at the transfer gates
(`TapOpens`), has them among the sequences `AccSeqsSampled` counts at its registration, with the registration's values
at `reg`, where `WindowGood` pins them (`mem_accSeqsSampled_of_windowGood`). The observations are the trusted network certifier's record of the cells
it sends, so the client's view reads back from them under its schedule, one its record format holds
(`tap_cells_read`): the views, and the secrets
they decode to, number at most `#Ss` times the size of the product of those sets (`encard_periodGood_le`). The record
holds the fail cells and the padding too, so the count needs no separate charge for them: the bound's
`∏_c (cap_c + 1)`, one fail choice per connection, is room here. `periodLeak` adds the per-window bound at
the network certifier's rows and the registration's values (`sampledLeak_tap_registered`): an extracted trace that
opens a registered input to a value other than the registration's collides with the registration's opening (`B_reg`).
-/

namespace NetTiming

open Claims Claims.Sampling Finset

theorem asReg_congr {L S : ℕ} (C : Prog) (lay : BoundaryGates L S) (k : ℕ) (R : Finset ℕ) {ρ ρ' : ℕ → Bool}
    (h : ∀ g ∈ R, ρ g = ρ' g) : AsReg C lay k R ρ = AsReg C lay k R ρ' := by
  ext a
  constructor <;> rintro ⟨h1, h2⟩ <;> refine ⟨fun g hg hf => ?_, h2⟩
  · rw [h1 g hg hf, h g hg]
  · rw [h1 g hg hf, ← h g hg]

section Deploy

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem mem_accSeqsSampled_of_windowGood {H : Msg → Msg} {D : Deploy Coin Col T} {L K S : ℕ}
    {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ} {reg : AuditStatement → Finset ℕ} {Dj : ℕ} {s₀ : AuditStatement} {P₀ : Partition}
    (hP₀ : D.partition s₀ = some P₀) {ρ : ℕ → Bool} {tops : AuditStatement → List (Opening Bool)}
    {op : ℕ → List (Obs L K S) → List Bool} {x : Msg} {w : Wit} {o : List (Obs L K S)}
    (htap : ∀ s, D.parse x = some s → TapOpens H D.commit (lay s) (k s) op s (tops s) o)
    (h : WindowGood H D L K S lay k δ reg Dj s₀ ρ tops x w o) :
    (lay s₀).Fits s₀.prog (k s₀) ∧
      o ∈ AccSeqsSampled s₀.prog P₀ s₀.design (lay s₀) (k s₀) (AsReg s₀.prog (lay s₀) (k s₀) (reg s₀) ρ) δ op Dj := by
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hl⟩, ⟨-, s₂, hs₂, hfits⟩, s, hs, ⟨hprog, hdes, hlay, hk, hreg, hpart⟩, hcov,
    hlen, hj⟩ := h
  rw [Option.some_injective _ (hs₁.symm.trans hs)] at hP₁ hl
  rw [Option.some_injective _ (hs₂.symm.trans hs)] at hfits
  rw [Option.some_injective _ (hP₁.symm.trans (hpart.trans hP₀))] at hl
  have hmem := hl op o (htap s hs) hcov hlen hj
  beta_reduce at hmem
  rw [hprog, hdes, hlay, hk, hreg] at hmem
  rw [hprog, hlay, hk] at hfits
  exact ⟨hfits, hmem⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem encard_periodGood_le (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition)
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
    (x : Z → Fin n → Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool))
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

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **At the network certifier's rows and the registered values**: `TapProof.rLeakTap_of_r` with the registered inputs
pinned at `ρ`, at a trace that also agrees with `ρ` wherever it opens a registered input. A registered free input the
trace leaves unopened is in a faulty unit, held as advice, so its input value is never read (`valOver_congr_held`). -/
theorem rLeakTap_reg_of_r (H : Msg → Msg) (D : Deploy Coin Col T) {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) {δ : ℝ} (hδ : 0 ≤ δ) {Dj : ℕ} (tops : AuditStatement → List (Opening Bool))
    (reg : AuditStatement → Finset ℕ) (ρ : AuditStatement → ℕ → Bool) {x : Msg} (hx : ScopeAcc D lay k x) {w : Wit}
    (hR : R H D.commit D.parse D.partition δ x w)
    (hagree : ∀ s, D.parse x = some s → (∀ q ∈ tops s, D.commit.Opens H s.roots q = true) →
      ∀ g, (∃ q ∈ tops s, q.pos = g) → ∀ b, w.val H D.commit s g = some b → b = tapVal (tops s) g)
    (hreg : ∀ s, D.parse x = some s → ∀ g ∈ reg s, ∀ b, w.val H D.commit s g = some b → b = ρ s g) :
    RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (ρ s)) Dj
      tops x w := by
  obtain ⟨s, P, hs, hP, hM⟩ := hR
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hsc⟩, s₂, hs₂, hfit⟩ := hx
  rw [hs] at hs₁ hs₂
  obtain rfl := Option.some.inj hs₁
  obtain rfl := Option.some.inj hs₂
  rw [hP] at hP₁
  obtain rfl := Option.some.inj hP₁
  have hF := faulty_mem_faultySets hM (SampledAuditProof.disjoint_allSlots hδ hM)
  obtain ⟨ι, hι⟩ := SampledAuditProof.correct_of_not_faulty H D.commit s P w
  refine ⟨s, P, hs, hP, fun op o htap hcov hl hD => ?_⟩
  have hext : ∀ g b, w.val H D.commit s g = some b →
      TapProof.fill (w.val H D.commit s) (tapVal (tops s)) g = some b :=
    TapProof.fill_ext
  have hcar := TapProof.carries_fill hfit htap (hagree s hs htap.1)
  have hacc := SampledAuditProof.accepted_of hsc.1 hsc.2.1 hfit ι
    (fun v r hr hv u hu => TapProof.correct_mono (TapProof.ruRef_mono hext) (hι v r hr hv u hu)) hcar hcov
  set tv := traceVal s.prog (w.val H D.commit s) with htv
  set G := freeInputs s.prog (lay s) (k s) with hG
  let a' : ℕ → Bool := fun g => if g ∈ G then (if g ∈ reg s then ρ s g else tv g) else false
  refine ⟨hl, hD, _, hF, a', ⟨fun g hg hgf => ?_, fun g hg => ite_eq_right hg⟩,
    SampledAuditProof.heldAdv s.prog P (faulty H D.commit s P w) (TapProof.fill (w.val H D.commit s) (tapVal (tops s))),
    fun g b h => SampledAuditProof.faultyGate_of_heldAdv h, ?_⟩
  · show (if g ∈ G then (if g ∈ reg s then ρ s g else tv g) else false) = ρ s g
    rw [ite_eq_left hgf, ite_eq_left hg]
  · intro t ht
    obtain ⟨h, hr, hst⟩ := hacc t ht
    refine ⟨h, ?_, hst⟩
    have hv : valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t)
          (traceVal s.prog (TapProof.fill (w.val H D.commit s) (tapVal (tops s)))))
          (SampledAuditProof.heldAdv s.prog P (faulty H D.commit s P w)
            (TapProof.fill (w.val H D.commit s) (tapVal (tops s)))) =
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) a')
          (SampledAuditProof.heldAdv s.prog P (faulty H D.commit s P w)
            (TapProof.fill (w.val H D.commit s) (tapVal (tops s)))) :=
      funext (valOver_congr_held s.prog _ fun i hi hfi => by
        have hnf : ¬ FaultyGate s.prog P (faulty H D.commit s P w) i := fun hfg => by
          rw [SampledAuditProof.heldAdv_of hfg] at hfi
          cases hfi
        obtain ⟨b, hb⟩ := val_some_of_input hsc.2.1 hi hnf
        refine (inputOf_restrict s.prog (lay s) (k s) _ _ hi).trans (inputOf_congr_at (lay s) (k s) _ ?_)
        show (if i ∈ G then _ else false) = if i ∈ G then (if i ∈ reg s then ρ s i else tv i) else false
        by_cases hiG : i ∈ G
        · rw [ite_eq_left hiG, ite_eq_left hiG, traceVal_of_val hi (hext i b hb)]
          by_cases hir : i ∈ reg s
          · rw [ite_eq_left hir]
            exact hreg s hs i hir b hb
          · rw [ite_eq_right hir, htv, traceVal_of_val hi hb]
        · rw [ite_eq_right hiG, ite_eq_right hiG])
    rw [hr]
    simp only [circuitProgram, hv]

/-- **The leak bound at the network certifier's rows and the registration's values**: `TapProof.sampledLeak_tap` with
the registered inputs pinned at the registration's values (`AsReg` at `regVal`), on the statements in scope whose
registration opens their registered inputs (`RegOpens`): `Pr[acc ∧ ¬RLeakTap] ≤ δ + ε_ks + ε_bind + Pr[B_tap] +
Pr[B_reg]`, a trace that opens a registered input to another value colliding with the registration (`B_reg`). -/
theorem sampledLeak_tap_registered (H : Msg → Msg) (D : Deploy Coin Col T) (hks : KS H D) (hb : D.commit.Binds H)
    (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ)
    (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : AuditStatement → Finset ℕ)
    (rops : AuditStatement → List (Opening Bool)) (Dj : ℕ) (tops : AuditStatement → List (Opening Bool)) (x : Msg)
    (hx : ScopeAcc D lay k x) (hreg : ∀ st, D.parse x = some st → RegOpens H D.commit st (reg st) (rops st))
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBad H D.V
        (RLeakTap H D.commit D.parse D.partition L K S lay k δ
          (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (rops st))) Dj tops) (E D decode) x P ≤
      κ D δ x + Real.sqrt (γ D x * PrB H D.V P x) + (Bext H D.V (E D decode) Ω P x).Pr H +
        (Btap H D.V (E D decode) D.commit D.parse tops Ω P x).Pr H +
        (Breg H D.V (E D decode) D.commit D.parse rops Ω P x).Pr H := by
  have hks' := hks δ decode hδ hdec x hx.1 Ω P
  rw [one_mul] at hks'
  unfold PrBad at hks' ⊢
  calc Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ¬ (RLeakTap H D.commit D.parse D.partition L K S lay k δ
            (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (rops st))) Dj tops).ext x
            ((E D decode).eval H P x s.1 s.2.1 s.2.2).1)
      ≤ Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
          (((run H P D.V x s.1 s.2.1).out = .acc ∧
            ¬ (R H D.commit D.parse D.partition δ).ext x ((E D decode).eval H P x s.1 s.2.1 s.2.2).1) ∨
            TapProof.TapColl H D decode tops P x s) ∨ RegColl H D decode rops P x s) := by
        refine Pr_mono fun s ⟨hac, hR⟩ => ?_
        rcases hw : ((E D decode).eval H P x s.1 s.2.1 s.2.2).1 with _ | w
        · exact Or.inl (Or.inl ⟨hac, by simp [Rel.ext]⟩)
        · rw [hw] at hR
          by_cases hA : R H D.commit D.parse D.partition δ x w
          · by_contra hnc
            refine hR (rLeakTap_reg_of_r H D lay k hδ tops reg (fun st => regVal (rops st)) hx hA
              (fun st hst hpass g hq b hwb => ?_) (fun st hst g hg b hwb => ?_))
            · by_contra hne
              exact hnc (Or.inl (Or.inr
                (TapProof.tapColl_of_disagree H D hb decode tops P x s hst hpass hac hw hq hwb hne)))
            · by_contra hne
              exact hnc (Or.inr (regColl_of_disagree H D hb decode reg rops P x s hst (hreg st hst) hac hw hg hwb hne))
          · exact Or.inl (Or.inl ⟨hac, hA⟩)
    _ ≤ _ := Pr_union_le _ _
    _ ≤ _ := add_le_add (Pr_union_le _ _) le_rfl
    _ ≤ _ := add_le_add (add_le_add hks' (TapProof.Pr_tapColl_le_Btap H D decode tops P x))
        (Pr_regColl_le_Breg H D decode rops P x)

/-- **The leak through the network certifier over a monitoring period**: `Guarantees.PeriodLeakAccepted`, from
`sampledLeak_tap_registered` and `encard_periodGood_le`. -/
theorem periodLeak : Guarantees.PeriodLeakAccepted := by
  intro Coin Col T _ _ _ _ _ H D hks L K S lay k δ decode hδ hdec reg Dj n sw Pw hPw ρ op β hβ Conn Cell Srv Dev Z _
    arr f tap run htrust cap Ss hcap hfit Zs schedOf srvOf hs dec hdz x wit tops rops hb
  exact ⟨fun m hm hr Ω _ _ P =>
      sampledLeak_tap_registered H D hks hb L K S lay k δ decode hδ hdec reg rops Dj tops m hm hr Ω P,
    encard_periodGood_le H D L K S lay k δ reg Dj sw Pw hPw ρ op β hβ arr f tap run htrust cap Ss hfit Zs
      schedOf srvOf hs dec hdz x wit tops⟩

end Deploy

end NetTiming
