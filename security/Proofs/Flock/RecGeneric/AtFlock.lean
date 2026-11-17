module

public import Definitions.Claims.RecursionGeneric
public import Proofs.Flock.RecGeneric.Plans
public import Proofs.Flock.RecGeneric.Outer
public import Proofs.Flock.RecGeneric.ZK
public import Proofs.Flock.Recursive.KSStatement

/-! The generic recursion statements at Flock: hm96 at the bound key as the server, the element-level Flock as the inner protocol and `E_O` as the outer extraction; definitions, no proofs. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

def ForkPlanReads (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ k : Msg → ℕ, (forkPlan (Coin := Coin) k).ReadsLe fun _ => 0

def CapPlanReads (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ (Cp : Caps Coin) (M : ℕ), (capPlan Cp M).ReadsLe fun _ => 2

def ConflictPlanReads (Coin : Type) [Fintype Coin] [Nonempty Coin] : Prop :=
  ∀ (k : Msg → ℕ) (R : ℕ), (conflictPlan (Coin := Coin) k R).ReadsLe fun x => k x + 1

def ForkPlanIsBAbort (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] : Prop :=
  ∀ (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type)
    [Fintype A] [Nonempty A] (keep : Ω → (Fin (V.k x) → Coin) → A → Bool),
    (forkPlan V.k).PrAt H (runModel H V P x keep) x = PrBAbort H V P x keep

def CapPlanIsFC (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] : Prop :=
  ∀ (H : Msg → Msg) (V : Verifier Coin) (Cp : Caps Coin) (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A] (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (M : ℕ),
    (capPlan Cp M).PrAt H (runModel H V P x keep) x = PrFC H V Cp P x keep M

def ConflictPlanIsF0 (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] : Prop :=
  ∀ (H : Msg → Msg) (V : Verifier Coin) (k : Msg → ℕ) (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A] (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (R : ℕ),
    (conflictPlan k R).PrAt H (runModel H V P x keep) x = PrF0 H V P x keep (k x) R

def Server.Opens (S : Server) : Prop :=
  ∀ H : Msg → Msg, Out64 H → ∀ (row : Msg) (y : S.Salt),
    ((S.opens ((S.commit row y).eval H).1 row y).eval H).1 = true

section Instance

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **hm96 as the server**, at the bound key: a salt is `saltLen` bytes; a commit string is 128 bytes,
`b ‖ c` with `b = H(rowIn row) ⊕ Φy` and `c = H(p_salt ‖ y)`; it opens to a row under a salt as `Bound.opens` checks; a
round's rows join and the last message reads as the bound record's (`join`, `final`), at its schedule (`nAt`). -/
def Server.ofBound (Bd : Bound Coin Col T) : Server where
  Salt := Fin Bd.saltLen → UInt8
  width := 128
  nAt := Bd.nAt
  join := Bd.join
  final := Bd.final
  commit row y := .query (Bd.saltPrefix ++ List.ofFn y) fun c => .query (Bd.rowIn row) fun d =>
    .ret (xorMask d (Bd.mask y) ++ c)
  opens := Bd.opens

def IVStarOfBound (Bd : Bound Coin Col T) : Prop := (Server.ofBound Bd).IVStar Bd.DE.V Bd.ans = Bd.IVStar

def OpenViewOfBound (Bd : Bound Coin Col T) : Prop := (Server.ofBound Bd).openView = Bd.openView

def BindsOfBound (Bd : Bound Coin Col T) : Prop := Bd.Binds → (Server.ofBound Bd).Binds

def OpensOfBound (Bd : Bound Coin Col T) : Prop := (Server.ofBound Bd).Opens

/-- **`B`'s term** at `k` rounds: `forkPlan k` at every loop bound, weight `4·k(x)`, no tail, no run asked. -/
noncomputable def termB (k : Msg → ℕ) : Term Coin where
  F _ := forkPlan k
  w x := 4 * (k x : ℝ)
  tail _ _ := 0
  d _ := 0

/-- **`FC`'s term** at the caps `Cp`: `capPlan Cp M`, weight `4·W(x)`, tail `1/(M+1)`, two runs met. -/
noncomputable def termFC (Cp : Caps Coin) : Term Coin where
  F := capPlan Cp
  w x := 4 * (Cp.width x : ℝ)
  tail _ M := 1 / ((M : ℝ) + 1)
  d _ := 2

/-- **`F0`'s term** at `k(x)` waits: `conflictPlan k R`, weight `4`, tail `(k + 1)/(R+1)`, `k + 1` runs met. -/
noncomputable def termF0 (k : Msg → ℕ) : Term Coin where
  F := conflictPlan k
  w _ := 4
  tail x R := ((k x : ℝ) + 1) / ((R : ℝ) + 1)
  d x := k x + 1

/-- **The element-level Flock as the inner protocol**, at an inner decoder and the caps' reader: `Inner.ofFlock` at
`DE`, its rounds answered by their coins (`ans`), and `KSFlockLin`'s three terms. -/
noncomputable def Proto.ofFlock (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) :
    Proto Coin Claims.Flock.Wit :=
  { Inner.ofFlock Bd.DE decI with
    ans := Bd.ans
    nT := 3
    terms := ![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] }

def ProtoWFOfFlock (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (Cp : Caps Coin) : Prop :=
  (∃ dec : Msg → Coin, ∀ c, dec (Bd.ans c) = c) → (∀ x, 0 ≤ Bd.DE.εtable x) → (Proto.ofFlock Bd decI Cp).WF

def KSAbortLinOfFlockLin [Zero Col] (H : Msg → Msg) (Bd : Bound Coin Col T) (Cp : Caps Coin) : Prop :=
  KSFlockLin H Bd.DE Cp → ∀ decI, decOK Bd.DE decI → (Proto.ofFlock Bd decI Cp).KSAbortLin H

variable {Col' T' : Type} [DecidableEq Col']

/-- **Today's compiled record and `E_O` as a `Comp`**, at the inner verifier `IV`, the inner protocol `I` whose rounds
`E_O` skips, and the outer Flock at a decoder: the compiled verifier, its inner statement and form, `E_O`
(`Claims.Recursion.EO`) with the relation every witness list meets (its sessions' checks are inside `E_O`), the inner
messages the witnesses give (`RecAt.msgs`), the calls a finder searches at a rerun (`Deploy.callsAt`) and the compiled
verifier's own at a run's staged V\* statements (`setupCalls`, as `Claims.Recursion.sampleCalls` reads them), and the errors
`κ_O = Σ_s κ_s`, `γ_O = (Σ_s √(γ_s·k(x)))²`. -/
noncomputable def Comp.ofRec (D : RecAt Coin) (IV : Verifier Coin) {W : Type} (I : Claims.Recursion.Inner Coin W)
    (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') : Comp Coin where
  V := (D.record IV).V
  xI := Claims.Recursion.xI (D.record IV)
  parses x := ∃ s, (D.record IV).parse x = some s ∧ s.circ = none
  WO := List Claims.Flock.Wit
  RO _ _ := True
  EO := Claims.Recursion.EO (D.record IV) I DO decO
  msgs := D.msgs
  calls H x r ws := (D.record IV).callsAt H x r ws
  own H x ms := setupCalls H (D.record IV)
    ((D.record IV).vstar x (ms.take (I.V.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x)))
  κO x := ∑ s : Fin ((D.record IV).nS x), ((D.record IV).nums x s).κ
  γO x := (∑ s : Fin ((D.record IV).nS x), Real.sqrt (((D.record IV).nums x s).γ * (D.record IV).V.k x)) ^ 2

def OuterKSOfSessions [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (IV : Verifier Coin) {W : Type}
    (I : Claims.Recursion.Inner Coin W) (DO : Claims.Flock.Deploy Coin Col' T') : Prop :=
  OuterKS H (D.record IV) I DO → ∀ decO, decOK DO decO → ∀ x, (Comp.ofRec D IV I DO decO).OuterKSAt H x

/-- **The instance at the bound record**: `Comp.ofRec` at `IVStar`, with `E_O` skipping the bound record's inner
rounds. -/
noncomputable abbrev compAt (D : RecAt Coin) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') : Comp Coin :=
  Comp.ofRec D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO

/-- **The instance at the deployed decoders**: `compAt` at the inner and outer records' own decoders
(`Deploy.decode`, the candidate of the counted `Deploy.dec`), the ones the deployment runs. `Bound.deploy` keeps
`DE`'s `dec`, so the inner one is `DE`'s. -/
noncomputable abbrev compDep (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T') :
    Comp Coin :=
  compAt D Bd Bd.deploy.decode DO DO.decode

def ForkPlansLeBinnerV (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') : Prop :=
  ∀ decI decO (x : Msg), (compAt D Bd decI DO decO).CallsTrue H x →
    (∀ e ∈ (D.record Bd.IVStar).calls x [], H e.1 = e.2) →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      (forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H
          ((compAt D Bd decI DO decO).innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
          ((compAt D Bd decI DO decO).xI x) ≤
        PrBinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ∧
      (forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H
          ((compAt D Bd decI DO decO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
          ((compAt D Bd decI DO decO).xI x) ≤
        PrBinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x

def CapPlanIsFCI (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) :
    Prop :=
  (∀ x, D.xI x = Claims.Recursion.xI (D.record Bd.IVStar) x) →
  ∀ (H : Msg → Msg) decI decO (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M : ℕ),
    (capPlan Cp M).PrAt H
        ((compAt D Bd decI DO decO).innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compAt D Bd decI DO decO).xI x) =
      PrFCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M

def ConflictPlanIsF0I (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T') : Prop :=
  (∀ x, D.xI x = Claims.Recursion.xI (D.record Bd.IVStar) x) →
  ∀ (H : Msg → Msg) decI decO (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (R : ℕ),
    (conflictPlan Bd.DE.k R).PrAt H
        ((compAt D Bd decI DO decO).innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compAt D Bd decI DO decO).xI x) =
      PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x R

/-- **`KSLin2`'s bound with the gated link, at the composite**: at every statement `G` admits, the composite at today's
compiled record over hm96 and the element-level Flock, at the deployed decoders (`compDep`, `Comp.E`, which reads no
`H`, so its extractor is the deployment's counted decoding of the reruns' openings and no other) meets
`κ + √(γ·Pr[B]) + (4k_I + 1)·Pr[B_I] + 4·W_I·(Pr[FCI] + 1/(M+1)) + 4·(Pr[F0I] + (k_I + 1)/(R+1)) +
β₂·(Pr[link(M)] + 1/(M+1))`: `KSLin2`'s bound with its binding error the gated link's on the composite's reruns
(`linkPlan` at `verModel`, whose cost charges every rerun it names) in place of `FdevV`'s. -/
def KSLinkOn [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) : Prop :=
  ∀ (x : Msg), G x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M R : ℕ),
    PrBad H (compDep D Bd DO).V ((compDep D Bd DO).R (Proto.ofFlock Bd Bd.deploy.decode Cp))
        ((compDep D Bd DO).E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x P ≤
      Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x +
        Real.sqrt (Claims.Recursion.γ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x *
          PrB H (D.record Bd.IVStar).V P x) +
        (4 * (Inner.ofFlock Bd.deploy Bd.deploy.decode).V.k (Claims.Recursion.xI (D.record Bd.IVStar) x) + 1) *
          PrBinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode P x +
        4 * Cp.width (Claims.Recursion.xI (D.record Bd.IVStar) x) *
          (PrFCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode Cp P x M +
            1 / ((M : ℝ) + 1)) +
        4 * (PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode P x R +
          (((Inner.ofFlock Bd.deploy Bd.deploy.decode).k (Claims.Recursion.xI (D.record Bd.IVStar) x) : ℝ) + 1) /
            ((R : ℝ) + 1)) +
        βdev2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x *
          ((linkPlan (Coin := Coin) Bd.DE.V.k M).PrAt H
              ((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
              ((compDep D Bd DO).xI x) +
            1 / ((M : ℝ) + 1))

def ContractOn [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) : Prop :=
  ∀ x, G x → (∀ e ∈ (D.record Bd.IVStar).calls x [], H e.1 = e.2) ∧
    ∀ decI, decOK Bd.deploy decI → ∀ decO, decOK DO decO →
      (compAt D Bd decI DO decO).Contract H ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)
        (fun v => (Proto.ofFlock Bd decI Cp).opens ((Server.ofBound Bd).openView v)) x

def RecoversLinkOn [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) : Prop :=
  CompositionKS → KSFlockLin H Bd.DE Cp →
    OuterKS H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO →
    TableErrNonneg (D.record Bd.IVStar) → (∀ x, D.xI x = Claims.Recursion.xI (D.record Bd.IVStar) x) →
    (∃ dec : Msg → Coin, ∀ c, dec (Bd.ans c) = c) → (∀ x, 0 ≤ Bd.DE.εtable x) →
    (Proto.ofFlock Bd Bd.deploy.decode Cp).LastCoinFree H → Bd.Binds →
    decOK Bd.DE Bd.deploy.decode → decOK DO DO.decode → ContractOn H D Bd DO Cp G →
    KSLinkOn H D Bd DO Cp G

end Instance

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

/-- **Knowledge soundness of the recursive verifier at every statement, at a compiled record** `D`: the composite's
bound with the gated link (`Claims.RecGeneric.KSLinkOn`) at every statement, at SHA-512, the deployed relation's bound
record at the reader `rd`, the deployed caps and the deployed decoders. `Claims.RecGeneric.RecoversLinkOn` at
`G := fun _ => True` gives it from
the generic theorem and the instance's contract at every statement; V\*'s two facts are read only where some run
accepts, so at a record that accepts only dense statements they are the dense ones. -/
def RecursiveKSEveryAtRec (D : Claims.Recursion.RecAt Coin) (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ) : Prop :=
  Claims.RecGeneric.KSLinkOn FlockDeploy.sha512 D
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep fun _ => True

/-- **Knowledge soundness of the recursive verifier at every statement, at the deployed reader**: `RecursiveKS` with
no restriction to dense statements (`RecursiveKSEveryAtRec` at the deployment's record and values). It is the Flock
instance of the generic composition (`Claims.RecGeneric.CompositionKS`): `RecoversLinkOn` takes it. -/
def RecursiveKSEvery : Prop :=
  ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    RecursiveKSEveryAtRec
      (RecDeploy.recBoundAt _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR))
      RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wER RecDeploy.DeployedR.wR

end Flock.Guarantees

namespace Claims.RecGeneric

open FlockDeploy (Coin FinalP)

def ContractDenseAtRec (D : Claims.Recursion.RecAt Coin) (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ) : Prop :=
  ContractOn FlockDeploy.sha512 D
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt

def ContractDense : Prop :=
  ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    ContractDenseAtRec
      (RecDeploy.recBoundAt _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR))
      RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wER RecDeploy.DeployedR.wR

end Claims.RecGeneric
