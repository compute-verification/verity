module

public import Definitions.Pouw.Assumptions.CreditedWords
public import Mathlib.CategoryTheory.Types.Basic
public import Mathlib.RingTheory.WittVector.IsPoly
public import Mathlib.Tactic.Polynomial.Basic
public import Mathlib.Tactic.ReduceModChar
public import Proofs.Certifier.JointBridge.WorkOfH1CheckedCreditMonitoringPeriod
public import Proofs.Pouw.MonitoringPeriod.Honest
public import Proofs.Pouw.Prob
import Definitions.Pouw.TS
import Definitions.Claims
import Proofs.Pouw.PearlC.Records
import Base.Sizes
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import PouwCheck
import PouwWindow
import Proofs.Compose.Compose.CircuitWork
import Proofs.Compose.Compose.ServiceBudget
import Proofs.Compose.Compose.ServiceCovers
import Proofs.Compose.Compose.SplitRateClosed
import Proofs.Compose.Compose.SplitRateExp
import Proofs.Compose.ComposeMonitoringPeriod
import Proofs.Pouw.HMu
import Proofs.Pouw.Headline.Luck
import Proofs.Certifier.JointBridge.WorkOfCreditedWords
import Proofs.Pouw.TS
import Proofs.Pouw.TS.ReceiptDuoCP
import Proofs.Pouw.TS.ReceiptLi
import Proofs.Pouw.TS.Stages
import Proofs.Poub.Tag.Basic
import Proofs.Poub.Tag.LittlewoodOfford
import Proofs.Poub.Tag.Signs
import Proofs.Pouw.MonitoringPeriod
import Proofs.Pouw.MonitoringPeriod.Sort
import Proofs.Pouw.Receipt
import Proofs.Properties.Pouw.Check
import Proofs.Properties.Pouw.Sample
import Proofs.Properties.Pouw.Grid
import Proofs.Properties.Poub.Remote.Sort
import Spot.Record

/-! `WorkOfH1CheckedCreditMonitoringPeriodH200`'s witnesses: an honest period on the H200, and the verifier that accepts everything. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Bridge.HonestH200

open Properties.Pouw.MonitoringPeriod Properties.Pouw.MonitoringPeriod.SecurityProofs
open ResourceAccounting.TS ResourceAccounting.Compose ResourceAccounting.Assumptions.TS ResourceAccounting.Assumptions.Compose Pouw.RelativeSecurity
open ResourceAccounting.SecurityProofs.ComposeMonitoringPeriod

theorem foldl_min_const (k : ℕ) : ∀ l : List ℕ, (l.map fun _ => k).foldl min k = k
  | [] => rfl
  | _ :: l => by
    have hk : min k k = k := Nat.min_self k
    show (l.map fun _ => k).foldl min (min k k) = k
    rw [hk]
    exact foldl_min_const k l

/-- At `k` multiply-adds for every kind, the fewest a credited word stands for is `k`, on a circuit with a credited
word. -/
theorem minWordMacs_const {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (k : ℕ) (h : creditedWords Sc Cd ≠ []) :
    minWordMacs Sc Cd (fun _ => k) = k := by
  unfold minWordMacs
  obtain ⟨i, l, hl⟩ := List.exists_cons_of_ne_nil h
  rw [hl]
  show (l.map fun _ => k).foldl min k = k
  exact foldl_min_const k l

theorem creditedWords_ne_nil : creditedWords Honest.scheme HonestH1.circuit ≠ [] := by
  intro h
  have h128 := HonestH1.wordsOf_eq
  unfold wordsOf at h128
  rw [h] at h128
  exact absurd h128 (by decide)

/-- The honest circuit's words are priced at `PouwH1Tile_v1`'s `512` multiply-adds each. -/
theorem minWordMacs_honest :
    minWordMacs Honest.scheme HonestH1.circuit (fun _ => ResourceAccounting.receiptWordMacs 4 8) = 512 :=
  minWordMacs_const _ _ _ creditedWords_ne_nil

/-- An INT8 tensor-core multiply-add costs one dense E4M3 multiply-add on the H200. -/
theorem h200S8Price_tc (sha fetch : ℚ) : ResourceAccounting.h200S8Price sha fetch .tc = 1 := Nat.ceil_one

/-- A device of `512·128·n` tensor-core multiply-adds a tick and nothing else. -/
def device (n : ℕ) : ResourceAccounting.Device Unit := ⟨0, 0, fun o => if o = .tc then 65536 * n else 0, 0, fun _ => 0⟩

/-- The empty circuit: the integrity audit's units are not this witness's subject. -/
def emptyCircuit : Verity.Circuit where
  N := 0
  w := fun g => g.elim0
  isInput := fun g => g.elim0
  reads := fun g => g.elim0
  reads_lt := fun g => g.elim0
  input_reads := fun g => g.elim0
  op := fun g => g.elim0
  op_congr := fun g => g.elim0
  outputs := ∅

/-- Each tick, the compute account runs the device's whole rate. -/
def step (n : ℕ) : ResourceAccounting.Step Unit Unit :=
  ⟨fun _ o => (device n).rate o, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => 0⟩

theorem step_fits (n : ℕ) : (step n).Fits (device n) := by
  unfold ResourceAccounting.Step.Fits
  refine ⟨fun o => ?_, ?_, fun _ => ?_, ?_, ?_⟩ <;> simp [step, device]

/-- **The deployed check accepts the period**: the compiled PoUW check (`PouwCheck.protocol.check`) accepts every
window of H1's honest period on the coins `x`, each at its registration and the proof system's answer. -/
def checkedAccepts (n : ℕ) (x : Paths HonestH1.Ω1 n) : Prop :=
  ∀ w, Accepts HonestH1.prm ((HonestH1.executionPhases n).reg x w) ((HonestH1.executionPhases n).ev x w)

/-- H1's honest play, accepted where the deployed check accepts every window. -/
noncomputable def checkedPlay (n : ℕ) : Play HonestH1.Ω1 n Unit Unit Unit Unit :=
  { HonestH1.play n with accepts := checkedAccepts n }

noncomputable def honestRun (n : ℕ) :
    ResourceAccounting.PeriodRun (device n) Unit emptyCircuit (fun _ : Unit => Paths HonestH1.Ω1 n) where
  units := 0
  parts := ⟨fun g => g.elim0, fun g => g.elim0⟩
  own := fun _ => ()
  acct := fun _ => ()
  win := ⟨0, 1⟩
  schedule := fun _ _ => step n
  fits := fun _ _ => step_fits n
  trace := fun _ g => g.elim0
  commWithin := fun K _ => (0 : ℝ) ≤ K
  accepts := fun a ξ => match a with
    | .pouw => (checkedPlay n).accepts (ξ ())
    | _ => False

def acceptAll (n : ℕ) :
    ResourceAccounting.PeriodRun (device n) Unit emptyCircuit (fun _ : Unit => Paths HonestH1.Ω1 n) where
  units := 0
  parts := ⟨fun g => g.elim0, fun g => g.elim0⟩
  own := fun _ => ()
  acct := fun _ => ()
  win := ⟨0, 1⟩
  schedule := fun _ _ => step n
  fits := fun _ _ => step_fits n
  trace := fun _ g => g.elim0
  commWithin := fun _ _ => True
  accepts := fun _ _ => True

theorem work_step (n : ℕ) (p : ResourceAccounting.Price) (r : ResourceAccounting.PeriodRun (device n) Unit emptyCircuit
    (fun _ : Unit => Paths HonestH1.Ω1 n)) (hs : ∀ ξ, r.schedule ξ = fun _ => step n) (hw : r.win = ⟨0, 1⟩)
    (ξ : (i : Unit) → Paths HonestH1.Ω1 n) : r.work p ξ = p .tc * (65536 * n) := by
  unfold ResourceAccounting.PeriodRun.work ResourceAccounting.Schedule.work ResourceAccounting.Schedule.ops
  rw [hs, hw, Fintype.sum_eq_single ResourceAccounting.OpClass.tc]
  · simp [step, device, ResourceAccounting.Window.ticks]
  · intro o ho
    simp [step, device, ho]

theorem work_honestRun (n : ℕ) (p : ResourceAccounting.Price) (ξ : (i : Unit) → Paths HonestH1.Ω1 n) :
    (honestRun n).work p ξ = p .tc * (65536 * n) :=
  work_step n p _ (fun _ => rfl) rfl ξ

theorem work_acceptAll (n : ℕ) (p : ResourceAccounting.Price) (ξ : (i : Unit) → Paths HonestH1.Ω1 n) :
    (acceptAll n).work p ξ = p .tc * (65536 * n) :=
  work_step n p _ (fun _ => rfl) rfl ξ

/-- The certificates: the whole priced credit as `W`, and no bits through the network attesters. -/
noncomputable def cert (n : ℕ) (δs : ℝ) : ResourceAccounting.Certificates :=
  ⟨0, 512 * creditPrefix Honest.scheme HonestH1.circuit 144 n (128 * n) 1 1 (67 / 8) δs, 0, 0, 0⟩

/-- Its verdict is the play's at the run's coin by definition. -/
noncomputable def realizes (n : ℕ) (δs : ℝ) (sha fetch : ℚ) :
    ResourceAccounting.RealizesCreditedWords (honestRun n) (cert n δs) (ResourceAccounting.h200S8Price sha fetch)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => ResourceAccounting.receiptWordMacs 4 8))
      (ResourceAccounting.receiptUnitMacs 64 4 8) 1024 (Paths HonestH1.Ω1 n) (fun _ => (checkedPlay n).accepts)
      (fun _ x => creditPrefix Honest.scheme HonestH1.circuit 144 ((HonestH1.executionPhases n).prefixLen HonestH1.prm x)
        (checkedLive Honest.scheme HonestH1.circuit (HonestH1.period n) HonestH1.prm (HonestH1.executionPhases n) x) 1 1
        (67 / 8) δs) where
  coin := Equiv.refl _
  accepts_iff := fun _ _ => Iff.rfl
  cert_le := fun _ y _ => by
    simp only [Equiv.refl_apply, HonestH1.prefixLen_all, HonestH1.checkedLive_all, minWordMacs_honest,
      h200S8Price_tc, cert, ResourceAccounting.Certificates.helpUnits]
    norm_num

theorem wordCosts (n : ℕ) (δs : ℝ) (sha fetch : ℚ) :
    ResourceAccounting.Assumptions.WordCostsMacs (honestRun n) (ResourceAccounting.h200S8Price sha fetch)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => ResourceAccounting.receiptWordMacs 4 8))
      (ResourceAccounting.receiptUnitMacs 64 4 8) (realizes n δs sha fetch).coin (fun _ => (checkedPlay n).accepts)
      (fun _ x => n * ((checkedPlay n).run x).fixed) (fun _ _ => 0) := by
  intro ξ y _
  have hfix : ((checkedPlay n).run ((realizes n δs sha fetch).coin y)).fixed = 128 := rfl
  simp only [hfix, work_honestRun, minWordMacs_honest, h200S8Price_tc, mul_zero, add_zero]
  push_cast
  linarith

theorem helpWithin (n : ℕ) (δs : ℝ) (sha fetch : ℚ) :
    ResourceAccounting.Assumptions.HelpWithinCertifier (honestRun n) (cert n δs) 1024 (realizes n δs sha fetch).coin
      (fun _ => (checkedPlay n).accepts) (fun _ _ => 0) := by
  intro _ _ _
  simp [cert, ResourceAccounting.Certificates.helpUnits]

theorem meets (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) (sha fetch : ℚ) :
    Nonempty (ResourceAccounting.RealizesCreditedWords (honestRun n) (cert n δs) (ResourceAccounting.h200S8Price sha fetch)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => ResourceAccounting.receiptWordMacs 4 8))
      (ResourceAccounting.receiptUnitMacs 64 4 8) 1024 (Paths HonestH1.Ω1 n) (fun _ => (checkedPlay n).accepts)
      (fun _ x => creditPrefix Honest.scheme HonestH1.circuit 144 ((HonestH1.executionPhases n).prefixLen HonestH1.prm x)
        (checkedLive Honest.scheme HonestH1.circuit (HonestH1.period n) HonestH1.prm (HonestH1.executionPhases n) x) 1 1
        (67 / 8) δs)) ∧
    ResourceAccounting.Assumptions.WordCostsMacs (honestRun n) (ResourceAccounting.h200S8Price sha fetch)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => ResourceAccounting.receiptWordMacs 4 8))
      (ResourceAccounting.receiptUnitMacs 64 4 8) (realizes n δs sha fetch).coin (fun _ => (checkedPlay n).accepts)
      (fun _ x => n * ((checkedPlay n).run x).fixed) (fun _ _ => 0) ∧
    ResourceAccounting.Assumptions.HelpWithinCertifier (honestRun n) (cert n δs) 1024 (realizes n δs sha fetch).coin
      (fun _ => (checkedPlay n).accepts) (fun _ _ => 0) ∧
    Scope Honest.scheme HonestH1.circuit δs ∧
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * shareCone Honest.scheme HonestH1.circuit 144 t ≤ 1) ∧
    (∀ j, stageShare (tileStage HonestH1.circuit 144) (shareCone Honest.scheme HonestH1.circuit 144) j ≤ 1) ∧
    (∀ t : Fin 144, 0 < shareCone Honest.scheme HonestH1.circuit 144 t →
      tileLuck Honest.scheme HonestH1.circuit 144 t ≤ 7 / 8 ∧ tileLuck0 Honest.scheme HonestH1.circuit 144 t ≤ 1 / 2) ∧
    (checkedPlay n).game =
      monitoringPeriodGame (tileStage HonestH1.circuit 144) (liveCone Honest.scheme HonestH1.circuit 144) 1 1 24
        (HonestH1.period n) ∧
    (HonestH1.period n).EveryExecutionPhase (fun live _ => DeadConeDead Honest.scheme HonestH1.circuit 144 live) ∧
    PrefixOf (HonestH1.layer n) (checkedPlay n).game ∧
    (∀ w, prob (fun x => (HonestH1.layer n).MovedIn x w) ≤ 0) ∧
    RegisteredBeforeDraw (HonestH1.executionPhases n) ∧ SampleIsCoin HonestH1.hK24 HonestH1.prm (HonestH1.executionPhases n) ∧
    ServiceSound HonestH1.prm (HonestH1.executionPhases n) HonestH1.opened (stagedAt HonestH1.salt HonestH1.staged) 0 ∧
    SaltIsLiveCoin HonestH1.hr1 (HonestH1.salt (n := n)) HonestH1.enc ∧ StagedPins HonestH1.staged HonestH1.flat ∧
    (HonestH1.period n).EveryExecutionPhase
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k G _ => OperandHMu (Honest.scheme.mu k) G)) ∧
    (HonestH1.period n).EveryExecutionPhase
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k G _ => ReceiptLuckAt (Honest.scheme.rB k) G)) ∧
    (HonestH1.period n).EveryExecutionPhase
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k _ G0 => GuessLuckAt (Honest.scheme.guess k) G0)) ∧
    UnitFixed (checkedPlay n) (HonestH1.layer n) (monitoringPeriodMode HonestH1.circuit (HonestH1.period n) (HonestH1.run n))
      (fun _ t => (tileWords Honest.scheme HonestH1.circuit 144 t : ℝ)) (fun _ => 128) ∧
    ∀ x, checkedAccepts n x := by
  obtain ⟨-, -, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hStaged, hMu, hRB, hGs,
    hUF, hAcc, -, -⟩ := HonestH1.h1_meets n hn hδs
  exact ⟨⟨realizes n δs sha fetch⟩, wordCosts n δs sha fetch, helpWithin n δs sha fetch, hS, hρN, hwmax, hlk, hG,
    hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hStaged, hMu, hRB, hGs, fun hw x _ => hUF hw x trivial,
    fun x => hAcc x⟩

theorem h200_sound (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) (sha fetch : ℚ) :
    (honestRun n).Sound (cert n δs) (ResourceAccounting.h200S8Price sha fetch) .pouw (δs + n * (0 + 0)) := by
  obtain ⟨hN, hM, -⟩ := HonestH1.h1_meets n hn hδs
  obtain ⟨-, hWC, hHW, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hStaged, hMu, hRB, hGs,
    hUF, -⟩ := meets n hn hδs sha fetch
  exact WorkOfH1CheckedCreditMonitoringPeriodH200 sha fetch 64 4 8 1024 Unit Unit Unit (fun _ => Paths HonestH1.Ω1 n) (device n)
    emptyCircuit (honestRun n) (cert n δs) Honest.scheme HonestH1.circuit hn hN HonestH1.hr1 hM HonestH1.hK24
    HonestH1.prm HonestH1.enc HonestH1.staged HonestH1.flat δs 0 0 (fun _ => HonestH1.executionPhases n)
    (fun _ => HonestH1.opened) (fun _ => HonestH1.salt) (fun _ => HonestH1.period n) (fun _ => checkedPlay n)
    (fun _ _ => 128) (fun _ _ => 0) (realizes n δs sha fetch) hWC hHW
    hS hρN hwmax hlk (fun _ => hG) (fun _ => hDD) (fun _ => hPre) (fun _ => hMoved) (fun _ => hReg) (fun _ => hDraw)
    (fun _ => hServ) (fun _ => hSalt) hStaged (fun _ => hMu) (fun _ => hRB) (fun _ => hGs) (fun _ => hUF)

theorem credited_sound (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) (sha fetch : ℚ) :
    (honestRun n).Sound (cert n δs) (ResourceAccounting.h200S8Price sha fetch) .pouw (δs + n * (0 + 0)) := by
  refine WorkOfCreditedWords Unit Unit Unit (fun _ => Paths HonestH1.Ω1 n) (device n) emptyCircuit (honestRun n)
    (cert n δs) (ResourceAccounting.h200S8Price sha fetch) _ (ResourceAccounting.receiptUnitMacs 64 4 8) 1024 _ _
    (fun _ x => n * ((checkedPlay n).run x).fixed) _
    (fun _ _ => 0) (δs + n * (0 + 0)) (realizes n δs sha fetch) (wordCosts n δs sha fetch)
    (helpWithin n δs sha fetch) fun _ => ?_
  refine le_trans ?_ (HonestH1.h1_credit n hn hδs)
  exact pr_le_prob_of_equiv (Equiv.refl _) fun _ h => ⟨trivial, h.2⟩

end ResourceAccounting.SecurityProofs.Bridge.HonestH200

namespace ResourceAccounting.SecurityProofs.Bridge

open Properties.Pouw.MonitoringPeriod Properties.Pouw.MonitoringPeriod.SecurityProofs
open ResourceAccounting.TS ResourceAccounting.Compose ResourceAccounting.Assumptions.TS ResourceAccounting.Assumptions.Compose Pouw.RelativeSecurity
open ResourceAccounting.SecurityProofs.ComposeMonitoringPeriod
open HonestH200

/-- Its PoUW verdict is the compiled check's (`PouwCheck.protocol.check` accepting every window), at the H200's INT8 price,
which no SHA-512 or fetch price reads. -/
theorem WorkOfH1CheckedCreditMonitoringPeriodH200_honest :
    Nonempty (ResourceAccounting.RealizesCreditedWords (honestRun 1) (cert 1 (1 / 1000000)) (ResourceAccounting.h200S8Price 0 0)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => ResourceAccounting.receiptWordMacs 4 8))
      (ResourceAccounting.receiptUnitMacs 64 4 8) 1024 (Paths HonestH1.Ω1 1) (fun _ => (checkedPlay 1).accepts)
      (fun _ x => creditPrefix Honest.scheme HonestH1.circuit 144 ((HonestH1.executionPhases 1).prefixLen HonestH1.prm x)
        (checkedLive Honest.scheme HonestH1.circuit (HonestH1.period 1) HonestH1.prm (HonestH1.executionPhases 1) x) 1 1
        (67 / 8) (1 / 1000000))) ∧
    ResourceAccounting.Assumptions.WordCostsMacs (honestRun 1) (ResourceAccounting.h200S8Price 0 0)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => ResourceAccounting.receiptWordMacs 4 8))
      (ResourceAccounting.receiptUnitMacs 64 4 8) (realizes 1 (1 / 1000000) 0 0).coin (fun _ => (checkedPlay 1).accepts)
      (fun _ x => (1 : ℕ) * ((checkedPlay 1).run x).fixed) (fun _ _ => 0) ∧
    ResourceAccounting.Assumptions.HelpWithinCertifier (honestRun 1) (cert 1 (1 / 1000000)) 1024
      (realizes 1 (1 / 1000000) 0 0).coin (fun _ => (checkedPlay 1).accepts) (fun _ _ => 0) ∧
    Scope Honest.scheme HonestH1.circuit (1 / 1000000) ∧
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * shareCone Honest.scheme HonestH1.circuit 144 t ≤ 1) ∧
    (∀ j, stageShare (tileStage HonestH1.circuit 144) (shareCone Honest.scheme HonestH1.circuit 144) j ≤ 1) ∧
    (∀ t : Fin 144, 0 < shareCone Honest.scheme HonestH1.circuit 144 t →
      tileLuck Honest.scheme HonestH1.circuit 144 t ≤ 7 / 8 ∧ tileLuck0 Honest.scheme HonestH1.circuit 144 t ≤ 1 / 2) ∧
    (checkedPlay 1).game =
      monitoringPeriodGame (tileStage HonestH1.circuit 144) (liveCone Honest.scheme HonestH1.circuit 144) 1 1 24
        (HonestH1.period 1) ∧
    (HonestH1.period 1).EveryExecutionPhase (fun live _ => DeadConeDead Honest.scheme HonestH1.circuit 144 live) ∧
    PrefixOf (HonestH1.layer 1) (checkedPlay 1).game ∧
    (∀ w, prob (fun x => (HonestH1.layer 1).MovedIn x w) ≤ 0) ∧
    RegisteredBeforeDraw (HonestH1.executionPhases 1) ∧ SampleIsCoin HonestH1.hK24 HonestH1.prm (HonestH1.executionPhases 1) ∧
    ServiceSound HonestH1.prm (HonestH1.executionPhases 1) HonestH1.opened (stagedAt HonestH1.salt HonestH1.staged) 0 ∧
    SaltIsLiveCoin HonestH1.hr1 (HonestH1.salt (n := 1)) HonestH1.enc ∧ StagedPins HonestH1.staged HonestH1.flat ∧
    (HonestH1.period 1).EveryExecutionPhase
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k G _ => OperandHMu (Honest.scheme.mu k) G)) ∧
    (HonestH1.period 1).EveryExecutionPhase
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k G _ => ReceiptLuckAt (Honest.scheme.rB k) G)) ∧
    (HonestH1.period 1).EveryExecutionPhase
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k _ G0 => GuessLuckAt (Honest.scheme.guess k) G0)) ∧
    UnitFixed (checkedPlay 1) (HonestH1.layer 1) (monitoringPeriodMode HonestH1.circuit (HonestH1.period 1) (HonestH1.run 1))
      (fun _ t => (tileWords Honest.scheme HonestH1.circuit 144 t : ℝ)) (fun _ => 128) ∧
    ∀ x, checkedAccepts 1 x :=
  meets 1 Nat.one_pos (by norm_num) 0 0

theorem WorkOfH1CheckedCreditMonitoringPeriodH200_nontrivial :
    ¬ (acceptAll 0).Sound ⟨0, 1, 0, 0, 0⟩ (ResourceAccounting.h200S8Price 0 0) .pouw (1 / 2) := by
  intro h
  have h1 := h fun _ => ()
  have hev : ∀ x : Paths HonestH1.Ω1 0,
      (acceptAll 0).accepts .pouw (Function.update (fun _ => ()) ((acceptAll 0).own .pouw) x) ∧
        ¬ (acceptAll 0).Holds ⟨0, 1, 0, 0, 0⟩ (ResourceAccounting.h200S8Price 0 0) .pouw
          (Function.update (fun _ => ()) ((acceptAll 0).own .pouw) x) := fun x => by
    refine ⟨trivial, ?_⟩
    show ¬ (1 : ℝ) ≤ ((acceptAll 0).work _ _ : ℝ)
    rw [work_acceptAll]
    norm_num
  have hpr : Pouw.pr (fun x : Paths HonestH1.Ω1 0 =>
      (acceptAll 0).accepts .pouw (Function.update (fun _ => ()) ((acceptAll 0).own .pouw) x) ∧
        ¬ (acceptAll 0).Holds ⟨0, 1, 0, 0, 0⟩ (ResourceAccounting.h200S8Price 0 0) .pouw
          (Function.update (fun _ => ()) ((acceptAll 0).own .pouw) x)) = 1 :=
    haveI : Nonempty (Paths HonestH1.Ω1 0) := ⟨()⟩
    Pouw.pr_eq_one hev
  rw [hpr] at h1
  norm_num at h1

end ResourceAccounting.SecurityProofs.Bridge
