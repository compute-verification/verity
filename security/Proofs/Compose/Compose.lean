module

public import Proofs.Compose.Compose.ServiceCovers
public import Definitions.Compose.Guarantees.Compose
public import Proofs.Pouw.TS
public import Proofs.Compose.Compose.ServiceBudget
public import Proofs.Compose.Compose.SplitRateMerged
public import Proofs.Compose.Compose.BindingOfECR
public import Proofs.Compose.Compose.CircuitWork
public import Proofs.Compose.Compose.WholeLuckOfEntropy
public import Proofs.Compose.Compose.CircuitWorkSeeded
public import Proofs.Pouw.RelativeSecurity
public import Proofs.ResourceAccounting.Lemmas.Compose

/-! PoUW's credit on an arbitrary registered circuit: proofs, each guarantee here or in its own file under `Proofs/Compose/Compose/`. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Compose

open ResourceAccounting.Compose

/-- A gate of the circuit is one of its own unit's gates. -/
theorem mem_unitGates {ι : Type} (Cd : Declared ι) {i : ℕ} (h : i < Cd.dag.length) :
    i ∈ unitGates Cd (Cd.unit i) :=
  List.mem_filter.mpr ⟨List.mem_range.mpr h, by simp⟩

/-- Under `Declared.Wf` every unit's round is one of the circuit's. -/
theorem roundOf_lt {ι : Type} {Cd : Declared ι} (hw : Cd.Wf) (hr : 0 < Cd.rounds) (u : ℕ) :
    Cd.roundOf u < Cd.rounds := by
  unfold Declared.roundOf
  rcases hl : unitGates Cd u with _ | ⟨i, l⟩
  · exact hr
  · have hi : i ∈ unitGates Cd u := by rw [hl]; exact List.mem_cons_self
    exact hw.2.1 _ (hw.1 i (List.mem_range.mp (List.mem_filter.mp hi).1)).1

section Layer

variable {ι C Ξ V : Type} (Cd : Declared ι) {Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
  (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)

theorem layer_sample : (Cd.layer hr hK pr).draw = drawOf hK := rfl

theorem layer_src (ℓ : ℕ) : (Cd.layer hr hK pr).src ℓ =
    if h : ℓ < Cd.dag.length ∧ Cd.binding ℓ ≠ some .coin ∧ Cd.unit ℓ < Np then some (0, ⟨Cd.unit ℓ, h.2.2⟩)
    else none := rfl

/-- Every one of tile `t`'s IR slots is one of the layer's. -/
theorem layer_pos_of_mem (w : Fin 1) (t : Fin Np) (ℓ : ℕ) (h : ℓ ∈ Cd.slots t.val) :
    ∃ k, (Cd.layer hr hK pr).pos w t k = ℓ := by
  obtain ⟨k, hk⟩ := List.mem_iff_getElem?.mp h
  refine ⟨k, ?_⟩
  show (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length) = ℓ
  rw [hk, Option.getD_some]

/-- Every one of the layer's slots of tile `t` is one of its IR slots, or no position when it has none. -/
theorem layer_pos_mem (w : Fin 1) (t : Fin Np) (k : ℕ) :
    (Cd.layer hr hK pr).pos w t k ∈ Cd.slots t.val ∨
      (Cd.slots t.val = [] ∧ (Cd.layer hr hK pr).pos w t k = Cd.dag.length) := by
  show (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length) ∈ Cd.slots t.val ∨
    (Cd.slots t.val = [] ∧
      (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length) = Cd.dag.length)
  cases hl : Cd.slots t.val with
  | nil => exact Or.inr ⟨rfl, rfl⟩
  | cons a l =>
    refine Or.inl ?_
    cases hk : (a :: l)[k]? with
    | none => exact List.mem_cons_self
    | some b => exact List.mem_iff_getElem?.mpr ⟨k, hk⟩

theorem layer_stageCuts : StageCuts Cd (Cd.layer hr hK pr) := fun ℓ _ hc => by
  show min (max (Cd.cutOf ℓ) (Cd.commitRound ℓ)) Cd.rounds = min (Cd.cutOf ℓ) Cd.rounds
  rw [Nat.max_eq_left hc]

/-- The circuit cut at the IR commits by the IR's cuts. -/
theorem layerCommitted_atIR : LayerCommitted Cd.atIR := fun _ _ => Or.inl (Nat.zero_le _)

/-- The layer of the circuit committed at the sample cuts every position at the sample. -/
theorem layer_atSample_cut (ℓ : ℕ) : (((Cd.atSample.layer hr hK pr).cut ℓ).2 : ℕ) = Cd.rounds :=
  Nat.min_eq_right (Nat.le_max_right _ _)

theorem layer_stageCoins (hw : Cd.Wf) : StageCoins Cd (Cd.layer hr hK pr) :=
  fun t => Nat.min_eq_left (Nat.le_sub_one_of_lt (roundOf_lt hw hr t.val))

theorem layer_stageSlots : StageSlots Cd (Cd.layer hr hK pr) := by
  refine And.intro ?_ fun t k => layer_pos_mem Cd hr hK pr 0 t k
  intro ℓ u hu
  rw [layer_src] at hu
  split at hu
  · rename_i h
    obtain rfl := Option.some.inj hu
    exact And.intro (And.intro h.1 h.2.1) rfl
  · cases hu

theorem layer_writerReads : (Cd.layer hr hK pr).WriterReads := by
  intro ℓ u hu
  rw [layer_src] at hu
  split at hu
  · rename_i h
    obtain rfl := Option.some.inj hu
    have hm : ℓ ∈ Cd.slots (Cd.unit ℓ) := List.mem_append_left _ (mem_unitGates Cd h.1)
    exact layer_pos_of_mem Cd hr hK pr 0 ⟨Cd.unit ℓ, h.2.2⟩ ℓ hm
  · cases hu

theorem layer_stageComplete (hw : Cd.Wf) (hN : Cd.units ≤ Np) : StageComplete Cd (Cd.layer hr hK pr) := by
  refine And.intro ?_ fun t ℓ h => layer_pos_of_mem Cd hr hK pr 0 t ℓ h
  intro ℓ hℓ
  have hu : Cd.unit ℓ < Np := Nat.lt_of_lt_of_le (hw.1 ℓ hℓ.1).2 hN
  exact ⟨(0, ⟨Cd.unit ℓ, hu⟩),
    (layer_src Cd hr hK pr ℓ).trans (dite_eq_left (And.intro hℓ.1 (And.intro hℓ.2 hu))), rfl⟩

end Layer

/-- A split-stage window's service game is the joint game at its joint run. -/
theorem serviceGame_eq {C Ξ : Type} {N M : ℕ} (stg : Fin N → ℕ) (W : SplitStages C (Fin N) M) (w : Fin N → ℝ)
    (a b : ℝ) (K : ℕ) : serviceGame (Ξ := Ξ) stg W w a b K = jointGame (W.joint stg) w a b K := rfl

/-- A split-stage window's modes are its joint run's. -/
theorem tileMode_eq {ι C B : Type} {Np : ℕ} (Cd : Declared ι) (W : SplitStages C (Fin Np) Cd.rounds) :
    tileMode (B := B) Cd W = jointMode (W.joint (tileStage Cd Np)) := rfl

theorem SplitRateClosed : ResourceAccounting.Guarantees.Compose.SplitRateClosed := by
  exact Rates.splitRateClosed

theorem SplitRateExp : ResourceAccounting.Guarantees.Compose.SplitRateExp := by
  exact Rates.splitRateExp

theorem SplitRateMerged : ResourceAccounting.Guarantees.Compose.SplitRateMerged := by
  exact Rates.splitRateMerged

theorem ServiceBudget : ResourceAccounting.Guarantees.Compose.ServiceBudget := by
  exact Budget.serviceBudget

theorem ServiceCovers : ResourceAccounting.Guarantees.Compose.ServiceCovers := by
  exact CoversProof.serviceCovers

theorem BindingOfECR : ResourceAccounting.Guarantees.Compose.BindingOfECR := by
  exact Binding.bindingOfECR

theorem WholeLuckOfEntropy : ResourceAccounting.Guarantees.Compose.WholeLuckOfEntropy := by
  exact Whole.wholeLuckOfEntropy

theorem CircuitWork : ResourceAccounting.Guarantees.Compose.CircuitWork := by
  exact WorkProof.circuitWork

theorem CreditOfWork : ResourceAccounting.Guarantees.Compose.CreditOfWork := by
  intro ι K P T L U F X _ _ _ _ _ _ _ _ _ _ Sc Cd Sh Z A folded lam r hFits hMS hA hClock hDisp hTen hPipe hSmem
    hWords hSem hFold hFill hwork
  obtain ⟨h, hZ, hq, hqT, hb⟩ := hMS
  obtain ⟨-, hf, -, hB, -, -, -, -, -, -, -, -, hmu1, -⟩ := id hZ
  obtain ⟨-, hw0, hphi0, -⟩ := ResourceAccounting.SecurityProofs.TS.stars_nonneg Z (floorOf Sc Cd) lam hZ h
  obtain ⟨dF, d0, -, -, hfix, hfold, hlt⟩ := hwork
  rw [← hFits.2] at hfix hfold
  have hM : ResourceAccounting.Assumptions.TS.FacetCeilings Z.G A.run := by
    rw [hFits.1]
    exact ResourceAccounting.SecurityProofs.TS.facetCeilings_of_sheet Sh _
      (ResourceAccounting.SecurityProofs.TS.run_nonneg_of_split hA) hClock hDisp hTen hPipe hSmem
  have hΦ := (ResourceAccounting.SecurityProofs.TS.price_of_parts_folded Z (floorOf Sc Cd) A lam folded h hZ hA hWords.1
    (ResourceAccounting.SecurityProofs.TS.scalarRoute_half Z A hA hWords hq hqT hb) hSem.2 hSem.1 hFold hFill).trans
    (ResourceAccounting.SecurityProofs.TS.price_le_time Z.G lam A.run h hf hM)
  have hid := (div_eq_iff hB.ne').mp (ResourceAccounting.SecurityProofs.TS.identity Z (floorOf Sc Cd) lam 0 0 hZ)
  simp only [ResourceAccounting.TS.Model.epsLuck, ResourceAccounting.TS.Model.epsAdapt, zero_mul, sub_zero] at hid
  have h1 := mul_le_mul_of_nonneg_right hfix hw0
  have h2 := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hfold (sub_nonneg.mpr hmu1)) hphi0
  have hNW : Z.Nw * Z.wStar lam / Z.bnd (floorOf Sc Cd) * Z.bnd (floorOf Sc Cd) = Z.Nw * Z.wStar lam :=
    div_mul_cancel₀ _ hB.ne'
  have hc : (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd (floorOf Sc Cd) * Z.bnd (floorOf Sc Cd) =
      (1 - Z.muF) * Z.Nw * Z.phiStar lam := div_mul_cancel₀ _ hB.ne'
  have hab : (aOf Sc Cd Z lam * dF + bOf Sc Cd Z lam * d0) * Z.bnd (floorOf Sc Cd) =
      (dF + d0) * (Z.Nw * Z.wStar lam) + d0 * ((1 - Z.muF) * Z.Nw * Z.phiStar lam) := by
    show (Z.Nw * Z.wStar lam / Z.bnd (floorOf Sc Cd) * dF + (Z.Nw * Z.wStar lam / Z.bnd (floorOf Sc Cd) +
      (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd (floorOf Sc Cd)) * d0) * Z.bnd (floorOf Sc Cd) = _
    linear_combination (dF + d0) * hNW + d0 * hc
  have hrB := mul_lt_mul_of_pos_right hlt hB
  show (1 - (Z.epsSkip (floorOf Sc Cd) lam + Z.epsCert (floorOf Sc Cd) lam + Z.epsRoute (floorOf Sc Cd) lam +
    Z.epsMand (floorOf Sc Cd) lam + r)) * (Z.bnd (floorOf Sc Cd) / Z.G.fmax) ≤ A.run.time
  rw [← mul_div_assoc, div_le_iff₀ hf]
  linarith

/-- The circuit's costs `aOf` and `bOf` are `≥ 0` in the model's ranges. -/
theorem costs_nonneg {ι K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Sc : Scheme K) (Cd : Declared ι)
    (Z : ResourceAccounting.TS.Model K R T L U F X) (lam : R → ℝ) (h : ModelScope Sc Cd Z lam) :
    0 ≤ aOf Sc Cd Z lam ∧ 0 ≤ bOf Sc Cd Z lam := by
  have hst := ResourceAccounting.SecurityProofs.TS.stars_nonneg Z (floorOf Sc Cd) lam h.2.1 h.1
  obtain ⟨-, -, hNw, hbnd, -, -, -, -, -, -, -, -, hmu, -⟩ := h.2.1
  have ha : 0 ≤ aOf Sc Cd Z lam := by
    show 0 ≤ Z.Nw * Z.wStar lam / Z.bnd (floorOf Sc Cd)
    exact div_nonneg (mul_nonneg hNw.le hst.2.1) hbnd.le
  refine ⟨ha, ?_⟩
  show 0 ≤ aOf Sc Cd Z lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd (floorOf Sc Cd)
  exact add_nonneg ha (div_nonneg (mul_nonneg (mul_nonneg (sub_nonneg.mpr hmu) hNw.le) hst.2.2.1) hbnd.le)

theorem CircuitCredit : ResourceAccounting.Guarantees.Compose.CircuitCredit := by
  intro ι K P T L U F X C Ξ V _ _ _ _ _ _ _ _ _ _ _ _ _ Sc Cd Sh Z H Np Kd hN hr hK W Pl pr folded lam
    ρ p p0 wmax Λ δs δbind δks win hS hρ hρN hw hlk hΛ hSR hgame hOf hMoved hJS hPins hMu hRB hG hUF hFits hMS
    hSplit hClock hDisp hTen hPipe hSmem hWords hSem hFold hFill
  obtain ⟨ha, hb⟩ := costs_nonneg Sc Cd Z lam hMS
  unfold CreditCertificate.Holds
  refine le_trans (Pouw.RelativeSecurity.prob_mono ?_)
    (CircuitWork Sc Cd hN hr hK W Pl pr folded ρ p p0 _ _ wmax Λ δs δbind δks hS hρ hρN hw hlk ha hb hΛ hSR
      hgame hOf hMoved hJS hPins hMu hRB hG hUF)
  rintro x ⟨hacc, htime⟩
  exact ⟨hacc, fun hwork => (not_le.mpr htime) (CreditOfWork Sc Cd Sh Z (Pl.run x) (folded x) lam _ hFits hMS
    (hSplit x) (hClock x) (hDisp x) (hTen x) (hPipe x) (hSmem x) (hWords x) (hSem x) (hFold x) (hFill x) hwork)⟩

theorem CircuitWorkSeeded : ResourceAccounting.Guarantees.Compose.CircuitWorkSeeded := by
  exact Seeded.circuitWorkSeeded

theorem CircuitCreditSeeded : ResourceAccounting.Guarantees.Compose.CircuitCreditSeeded := by
  intro ι K P T L U F X B Ξ V _ _ _ _ _ _ _ _ _ _ _ _ _ Tq Sc Cd Sh Z H Np Kd hN hr hK R Pl pr folded lam sq εid
    T J ρ κ0 δs δg δb δbind δks win hS hδg hKd hρ hρN ha hκ hgame hOf hMoved hJS hPins hSE hSLF hSLW hUF hFits hMS
    hSplit hClock hDisp hTen hPipe hSmem hWords hSem hFold hFill
  have hb := (costs_nonneg Sc Cd Z lam hMS).2
  unfold CreditCertificate.Holds
  refine le_trans (Pouw.RelativeSecurity.prob_mono ?_)
    (CircuitWorkSeeded Sc Cd hN hr hK R Pl pr folded sq εid T J ρ _ _ κ0 δs δg δb δbind δks hS hδg hKd hρ hρN ha hb
      hκ hgame hOf hMoved hJS hPins hSE hSLF hSLW hUF)
  rintro x ⟨hacc, htime⟩
  exact ⟨hacc, fun hwork => (not_le.mpr htime) (CreditOfWork Sc Cd Sh Z (Pl.run x) (folded x) lam _ hFits hMS
    (hSplit x) (hClock x) (hDisp x) (hTen x) (hPipe x) (hSmem x) (hWords x) (hSem x) (hFold x) (hFill x) hwork)⟩

end ResourceAccounting.SecurityProofs.Compose
