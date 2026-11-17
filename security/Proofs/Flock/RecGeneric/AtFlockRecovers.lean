module

public import Proofs.Flock.RecGeneric.AtFlockProof
public import Proofs.Flock.RecGeneric.AtFlockServer
public import Proofs.Flock.RecGeneric.Coupling
public import Proofs.Flock.RecGeneric.StatementProof
public import Proofs.Flock.Claims.Recursion.OuterKS

/-! L1 at Flock gives `KSLinkOn` wherever the instance's contract holds, with the three term comparisons taken as hypotheses (`recoversLinkOn_of`). -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

theorem compAt_β_eq_βdev2 {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [DecidableEq Col'] (D : RecAt Coin) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T')
    (Cp : Caps Coin) (x : Msg) :
    (compAt D Bd decI DO decO).β (Proto.ofFlock Bd decI Cp) x =
      βdev2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x := by
  show (((compAt D Bd decI DO decO).NI (Proto.ofFlock Bd decI Cp) x : ℕ) +
      ∑ i : Fin 3, (![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).w ((compAt D Bd decI DO decO).xI x) *
        ((![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).d ((compAt D Bd decI DO decO).xI x) : ℝ)) *
      (Bd.DE.V.k ((compAt D Bd decI DO decO).xI x) : ℝ) = _
  rw [Fin.sum_univ_three]
  show (((NI (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x : ℕ) : ℝ) +
      (4 * ((Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x) : ℝ) * ((0 : ℕ) : ℝ) +
        4 * (Cp.width (xI (D.record Bd.IVStar) x) : ℝ) * ((2 : ℕ) : ℝ) +
        4 * (((Inner.ofFlock Bd.deploy decI).k (xI (D.record Bd.IVStar) x) + 1 : ℕ) : ℝ))) *
      ((Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x) : ℝ) =
    (((NI (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x : ℕ) : ℝ) +
      8 * (Cp.width (xI (D.record Bd.IVStar) x) : ℝ) +
      4 * (((Inner.ofFlock Bd.deploy decI).k (xI (D.record Bd.IVStar) x) : ℕ) + 1 : ℝ)) *
      ((Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x) : ℝ)
  push_cast
  ring

theorem compAt_γ_eq {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [DecidableEq Col'] (D : RecAt Coin) (Bd : Bound Coin Col T)
    (decI : List (ℕ × Col) → List T) (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T')
    (Cp : Caps Coin) (x : Msg) :
    (compAt D Bd decI DO decO).γ (Proto.ofFlock Bd decI Cp) x =
      Claims.Recursion.γ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x := by
  show ((NI (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x : ℕ) : ℝ) ^ 2 *
      (∑ s : Fin ((D.record Bd.IVStar).nS x),
        Real.sqrt (((D.record Bd.IVStar).nums x s).γ * (D.record Bd.IVStar).V.k x)) ^ 2 =
    (((NI (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x : ℕ) : ℝ) *
      ∑ s : Fin ((D.record Bd.IVStar).nS x),
        Real.sqrt (((D.record Bd.IVStar).nums x s).γ * (D.record Bd.IVStar).V.k x)) ^ 2
  rw [mul_pow]

theorem recoversLinkOn_of {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [DecidableEq Col'] [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin)
    (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop)
    (hForkPlansLeBinnerV : ForkPlansLeBinnerV H D Bd DO) (hCapPlanIsFCI : CapPlanIsFCI D Bd DO Cp)
    (hConflictPlanIsF0I : ConflictPlanIsF0I D Bd DO) : RecoversLinkOn H D Bd DO Cp G := by
  intro hL1 hKS hO hT hxI hdec hε hLC hB hdI hdO hC x hx Ω _ _ P M R
  obtain ⟨hnil, hCon⟩ := hC x hx
  have hCx := hCon Bd.deploy.decode hdI DO.decode hdO
  have hκO : 0 ≤ (compDep D Bd DO).κO x := by
    show (0 : ℝ) ≤ ∑ s : Fin ((D.record Bd.IVStar).nS x), ((D.record Bd.IVStar).nums x s).κ
    exact Finset.sum_nonneg fun s _ => Nums.κ_nonneg _ (hT x s)
  have hγO : 0 ≤ (compDep D Bd DO).γO x := by
    show (0 : ℝ) ≤ (∑ s : Fin ((D.record Bd.IVStar).nS x),
      Real.sqrt (((D.record Bd.IVStar).nums x s).γ * (D.record Bd.IVStar).V.k x)) ^ 2
    exact sq_nonneg _
  have hK := hL1 Coin _ H (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)
    (compAt D Bd Bd.deploy.decode DO DO.decode) (protoWFOfFlock Bd Bd.deploy.decode Cp hdec hε)
    (kSAbortLinOfFlockLin H Bd Cp hKS Bd.deploy.decode hdI) hLC (bindsOfBound Bd hB) x
    (outerKSOfSessions H D Bd.IVStar (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO hO DO.decode hdO x)
    hκO hγO hCx Ω P (![0, M, R] : Fin 3 → ℕ) M
  refine hK.trans ?_
  obtain ⟨hF1, hF2⟩ := hForkPlansLeBinnerV Bd.deploy.decode DO.decode x hCx.callsTrue hnil Ω P
  have hPO : (compAt D Bd Bd.deploy.decode DO DO.decode).PrOut H P x = 0 :=
    Comp.PrOut_eq_zero (K := compAt D Bd Bd.deploy.decode DO DO.decode) (H := H) (x := x) (fun _ => trivial) P
  show (compAt D Bd Bd.deploy.decode DO DO.decode).κ (Proto.ofFlock Bd Bd.deploy.decode Cp) x +
      Real.sqrt ((compAt D Bd Bd.deploy.decode DO DO.decode).γ (Proto.ofFlock Bd Bd.deploy.decode Cp) x *
        PrB H (compAt D Bd Bd.deploy.decode DO DO.decode).V P x) +
      ∑ i : Fin 3, (![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).w
          ((compAt D Bd Bd.deploy.decode DO DO.decode).xI x) *
        (((![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).F (![0, M, R] i)).PrAt H
            ((compAt D Bd Bd.deploy.decode DO DO.decode).innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H
              P x)
            ((compAt D Bd Bd.deploy.decode DO DO.decode).xI x) +
          (![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).tail ((compAt D Bd Bd.deploy.decode DO DO.decode).xI x)
            (![0, M, R] i)) +
      (compAt D Bd Bd.deploy.decode DO DO.decode).reads (Proto.ofFlock Bd Bd.deploy.decode Cp) x *
        (compAt D Bd Bd.deploy.decode DO DO.decode).PrOut H P x +
      (forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H
        ((compAt D Bd Bd.deploy.decode DO DO.decode).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compAt D Bd Bd.deploy.decode DO DO.decode).xI x) +
      (compAt D Bd Bd.deploy.decode DO DO.decode).β (Proto.ofFlock Bd Bd.deploy.decode Cp) x *
        ((linkPlan (Coin := Coin) Bd.DE.V.k M).PrAt H
            ((compAt D Bd Bd.deploy.decode DO DO.decode).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
            ((compAt D Bd Bd.deploy.decode DO DO.decode).xI x) +
          1 / ((M : ℝ) + 1) + (compAt D Bd Bd.deploy.decode DO DO.decode).PrOut H P x) ≤ _
  rw [Fin.sum_univ_three]
  show Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x +
      Real.sqrt ((compAt D Bd Bd.deploy.decode DO DO.decode).γ (Proto.ofFlock Bd Bd.deploy.decode Cp) x *
        PrB H (D.record Bd.IVStar).V P x) +
      (4 * ((Inner.ofFlock Bd.deploy Bd.deploy.decode).V.k (xI (D.record Bd.IVStar) x) : ℝ) *
          ((forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H
              ((compAt D Bd Bd.deploy.decode DO DO.decode).innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H
                P x)
              ((compAt D Bd Bd.deploy.decode DO DO.decode).xI x) + 0) +
        4 * (Cp.width (xI (D.record Bd.IVStar) x) : ℝ) *
          ((capPlan Cp M).PrAt H
              ((compAt D Bd Bd.deploy.decode DO DO.decode).innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H
                P x)
              ((compAt D Bd Bd.deploy.decode DO DO.decode).xI x) + 1 / ((M : ℝ) + 1)) +
        4 * ((conflictPlan (Coin := Coin) Bd.DE.k R).PrAt H
              ((compAt D Bd Bd.deploy.decode DO DO.decode).innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H
                P x)
              ((compAt D Bd Bd.deploy.decode DO DO.decode).xI x) +
            (((Inner.ofFlock Bd.deploy Bd.deploy.decode).k (xI (D.record Bd.IVStar) x) : ℝ) + 1) /
              ((R : ℝ) + 1))) +
      (compAt D Bd Bd.deploy.decode DO DO.decode).reads (Proto.ofFlock Bd Bd.deploy.decode Cp) x *
        (compAt D Bd Bd.deploy.decode DO DO.decode).PrOut H P x +
      (forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H
        ((compAt D Bd Bd.deploy.decode DO DO.decode).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compAt D Bd Bd.deploy.decode DO DO.decode).xI x) +
      (compAt D Bd Bd.deploy.decode DO DO.decode).β (Proto.ofFlock Bd Bd.deploy.decode Cp) x *
        ((linkPlan (Coin := Coin) Bd.DE.V.k M).PrAt H
            ((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
            ((compDep D Bd DO).xI x) +
          1 / ((M : ℝ) + 1) + (compAt D Bd Bd.deploy.decode DO DO.decode).PrOut H P x) ≤ _
  rw [hPO, compAt_γ_eq D Bd Bd.deploy.decode DO DO.decode Cp x,
    compAt_β_eq_βdev2 D Bd Bd.deploy.decode DO DO.decode Cp x,
    hCapPlanIsFCI hxI H Bd.deploy.decode DO.decode x Ω P M,
    hConflictPlanIsF0I hxI H Bd.deploy.decode DO.decode x Ω P R]
  have hk : (0 : ℝ) ≤ 4 * ((Inner.ofFlock Bd.deploy Bd.deploy.decode).V.k (xI (D.record Bd.IVStar) x) : ℝ) := by
    positivity
  have hF1' := mul_le_mul_of_nonneg_left hF1 hk
  linarith

theorem ksLinkOn_of {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [DecidableEq Col'] [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin)
    (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop)
    (hForkPlansLeBinnerV : ForkPlansLeBinnerV H D Bd DO) (hCapPlanIsFCI : CapPlanIsFCI D Bd DO Cp)
    (hConflictPlanIsF0I : ConflictPlanIsF0I D Bd DO) (hKS : KSFlockLin H Bd.DE Cp)
    (hO : OuterKS H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO)
    (hT : TableErrNonneg (D.record Bd.IVStar)) (hxI : ∀ x, D.xI x = Claims.Recursion.xI (D.record Bd.IVStar) x)
    (hdec : ∃ dec : Msg → Coin, ∀ c, dec (Bd.ans c) = c) (hε : ∀ x, 0 ≤ Bd.DE.εtable x)
    (hLC : (Proto.ofFlock Bd Bd.deploy.decode Cp).LastCoinFree H) (hB : Bd.Binds)
    (hdI : decOK Bd.DE Bd.deploy.decode) (hdO : decOK DO DO.decode) (hC : ContractOn H D Bd DO Cp G) :
    KSLinkOn H D Bd DO Cp G :=
  recoversLinkOn_of H D Bd DO Cp G hForkPlansLeBinnerV hCapPlanIsFCI hConflictPlanIsF0I
    compositionKS hKS hO hT hxI hdec hε hLC hB hdI hdO hC

end Claims.RecGeneric
