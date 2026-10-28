import Proofs.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod
import Proofs.Pouw.Period.Honest
import Proofs.Pouw.Prob

/-!
# `WorkOfH1CheckedCreditPeriodH200`'s witnesses: an honest period on the H200, and the verifier that accepts everything

The positive twin of `Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriodH200`, and of `WorkOfCreditedWords`
at its instance: `Properties.Pouw.Period.SecurityProofs.HonestH1`'s live-salt period of `n ≥ 1` windows, 128 words
live and fixed a window, run by a device that spends `512` INT8 tensor-core multiply-adds on each word and outsources
no unit, meets every hypothesis at the H200's INT8 price, at `PouwH1Tile_v1`'s `2048` multiply-adds a unit and
`b_unit = 1024`, with the work certificate the whole priced credit, `W = 512·credit` (`meets`, `h200_sound`,
`credited_sound`, and `WorkOfH1CheckedCreditPeriodH200_honest` at one window, `δ_s = 10⁻⁶`, the run's δ).

* The period (`honestRun`): one coin coordinate, PoUW's, which is H1's paths; one account; the empty circuit; a window
  of one tick, in which the compute account runs `512·128·n` tensor-core multiply-adds and nothing else, the device's
  whole rate (`device`). PoUW's verdict is the deployed check's (`checkedAccepts`): the compiled PoUW check,
  `PouwCheck.protocol.check` (`verity/core/protocols/pouw/lean/check`, held to the Python verifier by its difftest),
  accepts every window of the period, as `Properties.Pouw.Period.Accepts` reads it. The period's play
  (`checkedPlay`) is H1's honest play with that acceptance, so PoUW's verdict on the run is the play's at the run's
  coin by definition. No other subprotocol runs in the period, so none accepts. The network certifier's claim at
  `K` is that the bits it saw the node move, none in this period, are within `K`.
* The certificates (`cert`): `W = 512·creditPrefix(n, 128·n)`, the check's credit at the honest period's accepted
  prefix and live words (`HonestH1.prefixLen_all`, `HonestH1.checkedLive_all`) at `m = 512` and `p(tc) = 1`
  (`h200S8Price_tc`), and `K = 0`, so `help = 0` units meets `HelpWithinCertifier`.
* `m = 512`: the honest circuit's 128 words are all credited (`HonestH1.wordsOf_eq`), so the minimum is over them
  (`minWordMacs_const`), and `WordCostsMacs` is `512·1·(n·128) ≤ 512·128·n`, the period's work (`work_honestRun`).

The verifier that accepts everything (`acceptAll`) is the same period with every subprotocol accepting at every coin.
At no windows its device runs nothing, so a work certificate `W = 1` fails on every coin and PoUW's guarantee fails
there at any error below one (`WorkOfH1CheckedCreditPeriodH200_nontrivial`).
-/

namespace Accounting.SecurityProofs.Bridge.HonestH200

open Properties.Pouw.Period Properties.Pouw.Period.SecurityProofs
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposePeriod

/-! ## The word count and the price -/

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
    minWordMacs Honest.scheme HonestH1.circuit (fun _ => Accounting.receiptWordMacs 4 8) = 512 :=
  minWordMacs_const _ _ _ creditedWords_ne_nil

/-- An INT8 tensor-core multiply-add costs one dense E4M3 multiply-add on the H200. -/
theorem h200S8Price_tc (sha fetch : ℚ) : Accounting.h200S8Price sha fetch .tc = 1 := Nat.ceil_one

/-! ## The period -/

/-- A device of `512·128·n` tensor-core multiply-adds a tick and nothing else. -/
def device (n : ℕ) : Accounting.Device Unit := ⟨0, 0, fun o => if o = .tc then 65536 * n else 0, 0, fun _ => 0⟩

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
def step (n : ℕ) : Accounting.Step Unit Unit :=
  ⟨fun _ o => (device n).rate o, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => 0⟩

theorem step_fits (n : ℕ) : (step n).Fits (device n) := by
  unfold Accounting.Step.Fits
  refine ⟨fun o => ?_, ?_, fun _ => ?_, ?_, ?_⟩ <;> simp [step, device]

/-- **The deployed check accepts the period**: the compiled PoUW check (`PouwCheck.protocol.check`) accepts every
window of H1's honest period on the coins `x`, each at its registration and the proof service's answer. -/
def checkedAccepts (n : ℕ) (x : Paths HonestH1.Ω1 n) : Prop :=
  ∀ w, Accepts HonestH1.prm ((HonestH1.windows n).reg x w) ((HonestH1.windows n).ev x w)

/-- H1's honest play, accepted where the deployed check accepts every window. -/
noncomputable def checkedPlay (n : ℕ) : Play HonestH1.Ω1 n Unit Unit Unit Unit :=
  { HonestH1.play n with accepts := checkedAccepts n }

/-- **The honest period on PoUW's coins**, H1's paths: one tick of the device's whole rate, PoUW's verdict the
deployed check's on the period's windows at the run's coin, no other subprotocol accepting, and the network
certifier's claim at `K` that the bits it saw the node move, none, are within `K`. -/
noncomputable def honestRun (n : ℕ) :
    Accounting.PeriodRun (device n) Unit emptyCircuit (fun _ : Unit => Paths HonestH1.Ω1 n) where
  units := 0
  parts := ⟨fun g => g.elim0, fun g => g.elim0⟩
  own := fun _ => ()
  acct := fun _ => ()
  win := ⟨0, 1⟩
  schedule := fun _ _ => step n
  fits := fun _ _ => step_fits n
  transcript := fun _ g => g.elim0
  commWithin := fun K _ => (0 : ℝ) ≤ K
  accepts := fun a ξ => match a with
    | .pouw => (checkedPlay n).accepts (ξ ())
    | _ => False

/-- **The verifier that accepts everything**, on the same period: every subprotocol accepts at every coin, and the
network certifier's claim always holds. -/
def acceptAll (n : ℕ) :
    Accounting.PeriodRun (device n) Unit emptyCircuit (fun _ : Unit => Paths HonestH1.Ω1 n) where
  units := 0
  parts := ⟨fun g => g.elim0, fun g => g.elim0⟩
  own := fun _ => ()
  acct := fun _ => ()
  win := ⟨0, 1⟩
  schedule := fun _ _ => step n
  fits := fun _ _ => step_fits n
  transcript := fun _ g => g.elim0
  commWithin := fun _ _ => True
  accepts := fun _ _ => True

theorem work_step (n : ℕ) (p : Accounting.Price) (r : Accounting.PeriodRun (device n) Unit emptyCircuit
    (fun _ : Unit => Paths HonestH1.Ω1 n)) (hs : ∀ ξ, r.schedule ξ = fun _ => step n) (hw : r.win = ⟨0, 1⟩)
    (ξ : (i : Unit) → Paths HonestH1.Ω1 n) : r.work p ξ = p .tc * (65536 * n) := by
  unfold Accounting.PeriodRun.work Accounting.Schedule.work Accounting.Schedule.ops
  rw [hs, hw, Fintype.sum_eq_single Accounting.OpClass.tc]
  · simp [step, device, Accounting.Window.ticks]
  · intro o ho
    simp [step, device, ho]

theorem work_honestRun (n : ℕ) (p : Accounting.Price) (ξ : (i : Unit) → Paths HonestH1.Ω1 n) :
    (honestRun n).work p ξ = p .tc * (65536 * n) :=
  work_step n p _ (fun _ => rfl) rfl ξ

theorem work_acceptAll (n : ℕ) (p : Accounting.Price) (ξ : (i : Unit) → Paths HonestH1.Ω1 n) :
    (acceptAll n).work p ξ = p .tc * (65536 * n) :=
  work_step n p _ (fun _ => rfl) rfl ξ

/-- The certificates: the whole priced credit as `W`, and no bits through the network certifiers. -/
noncomputable def cert (n : ℕ) (δs : ℝ) : Accounting.Certificates :=
  ⟨0, 512 * creditPrefix Honest.scheme HonestH1.circuit 144 n (128 * n) 1 1 (67 / 8) δs, 0, 0, 0⟩

/-- PoUW's audit of the honest period credits H1's checked words at the H200's INT8 price, `512` multiply-adds a word,
`2048` a unit and `1024` bits a unit. Its verdict is the play's at the run's coin by definition. -/
noncomputable def realizes (n : ℕ) (δs : ℝ) (sha fetch : ℚ) :
    Accounting.RealizesCreditedWords (honestRun n) (cert n δs) (Accounting.h200S8Price sha fetch)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => Accounting.receiptWordMacs 4 8))
      (Accounting.receiptUnitMacs 64 4 8) 1024 (Paths HonestH1.Ω1 n) (fun _ => (checkedPlay n).accepts)
      (fun _ x => creditPrefix Honest.scheme HonestH1.circuit 144 ((HonestH1.windows n).prefixLen HonestH1.prm x)
        (checkedLive Honest.scheme HonestH1.circuit (HonestH1.period n) HonestH1.prm (HonestH1.windows n) x) 1 1
        (67 / 8) δs) where
  coin := Equiv.refl _
  accepts_iff := fun _ _ => Iff.rfl
  cert_le := fun _ y _ => by
    simp only [Equiv.refl_apply, HonestH1.prefixLen_all, HonestH1.checkedLive_all, minWordMacs_honest,
      h200S8Price_tc, cert, Accounting.Certificates.helpUnits]
    norm_num

theorem wordCosts (n : ℕ) (δs : ℝ) (sha fetch : ℚ) :
    Accounting.Assumptions.WordCostsMacs (honestRun n) (Accounting.h200S8Price sha fetch)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => Accounting.receiptWordMacs 4 8))
      (Accounting.receiptUnitMacs 64 4 8) (realizes n δs sha fetch).coin (fun _ => (checkedPlay n).accepts)
      (fun _ x => n * ((checkedPlay n).run x).fixed) (fun _ _ => 0) := by
  intro ξ y _
  have hfix : ((checkedPlay n).run ((realizes n δs sha fetch).coin y)).fixed = 128 := rfl
  simp only [hfix, work_honestRun, minWordMacs_honest, h200S8Price_tc, mul_zero, add_zero]
  push_cast
  linarith

theorem helpWithin (n : ℕ) (δs : ℝ) (sha fetch : ℚ) :
    Accounting.Assumptions.HelpWithinCertifier (honestRun n) (cert n δs) 1024 (realizes n δs sha fetch).coin
      (fun _ => (checkedPlay n).accepts) (fun _ _ => 0) := by
  intro _ _ _
  simp [cert, Accounting.Certificates.helpUnits]

/-! ## Every hypothesis -/

/-- **The honest period on the H200 meets every hypothesis of `WorkOfH1CheckedCreditPeriodH200`**, in its order, at
any SHA-512 and fetch price, `n ≥ 1` windows and `δ_s > 0`, with `δ_bind = δ_ks = 0`, and the deployed check accepts
the period on every path: the realization at the deployed check's verdict, the two bridge premises, and
`HonestH1.h1_meets`'s premises of `H1CheckedCreditPeriod` at the checked play. -/
theorem meets (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) (sha fetch : ℚ) :
    Nonempty (Accounting.RealizesCreditedWords (honestRun n) (cert n δs) (Accounting.h200S8Price sha fetch)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => Accounting.receiptWordMacs 4 8))
      (Accounting.receiptUnitMacs 64 4 8) 1024 (Paths HonestH1.Ω1 n) (fun _ => (checkedPlay n).accepts)
      (fun _ x => creditPrefix Honest.scheme HonestH1.circuit 144 ((HonestH1.windows n).prefixLen HonestH1.prm x)
        (checkedLive Honest.scheme HonestH1.circuit (HonestH1.period n) HonestH1.prm (HonestH1.windows n) x) 1 1
        (67 / 8) δs)) ∧
    Accounting.Assumptions.WordCostsMacs (honestRun n) (Accounting.h200S8Price sha fetch)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => Accounting.receiptWordMacs 4 8))
      (Accounting.receiptUnitMacs 64 4 8) (realizes n δs sha fetch).coin (fun _ => (checkedPlay n).accepts)
      (fun _ x => n * ((checkedPlay n).run x).fixed) (fun _ _ => 0) ∧
    Accounting.Assumptions.HelpWithinCertifier (honestRun n) (cert n δs) 1024 (realizes n δs sha fetch).coin
      (fun _ => (checkedPlay n).accepts) (fun _ _ => 0) ∧
    Scope Honest.scheme HonestH1.circuit δs ∧
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * shareCone Honest.scheme HonestH1.circuit 144 t ≤ 1) ∧
    (∀ j, stageShare (tileStage HonestH1.circuit 144) (shareCone Honest.scheme HonestH1.circuit 144) j ≤ 1) ∧
    (∀ t : Fin 144, 0 < shareCone Honest.scheme HonestH1.circuit 144 t →
      tileLuck Honest.scheme HonestH1.circuit 144 t ≤ 7 / 8 ∧ tileLuck0 Honest.scheme HonestH1.circuit 144 t ≤ 1 / 2) ∧
    (checkedPlay n).game =
      periodGame (tileStage HonestH1.circuit 144) (liveCone Honest.scheme HonestH1.circuit 144) 1 1 24
        (HonestH1.period n) ∧
    (HonestH1.period n).EveryWindow (fun live _ => DeadConeDead Honest.scheme HonestH1.circuit 144 live) ∧
    PrefixOf (HonestH1.layer n) (checkedPlay n).game ∧
    (∀ w, prob (fun x => (HonestH1.layer n).MovedIn x w) ≤ 0) ∧
    RegisteredBeforeDraw (HonestH1.windows n) ∧ DrawIsCoin HonestH1.hK24 HonestH1.prm (HonestH1.windows n) ∧
    ServiceSound HonestH1.prm (HonestH1.windows n) HonestH1.opened (stagedAt HonestH1.salt HonestH1.staged) 0 ∧
    SaltIsLiveCoin HonestH1.hr1 (HonestH1.salt (n := n)) HonestH1.enc ∧ StagedPins HonestH1.staged HonestH1.flat ∧
    (HonestH1.period n).EveryWindow
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k G _ => OperandHMu (Honest.scheme.mu k) G)) ∧
    (HonestH1.period n).EveryWindow
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k G _ => ReceiptLuckAt (Honest.scheme.rB k) G)) ∧
    (HonestH1.period n).EveryWindow
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k _ G0 => GuessLuckAt (Honest.scheme.guess k) G0)) ∧
    UnitFixed (checkedPlay n) (HonestH1.layer n) (periodMode HonestH1.circuit (HonestH1.period n) (HonestH1.run n))
      (fun _ t => (tileWords Honest.scheme HonestH1.circuit 144 t : ℝ)) (fun _ => 128) ∧
    ∀ x, checkedAccepts n x := by
  obtain ⟨-, -, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hStaged, hMu, hRB, hGs,
    hUF, hAcc, -, -⟩ := HonestH1.h1_meets n hn hδs
  exact ⟨⟨realizes n δs sha fetch⟩, wordCosts n δs sha fetch, helpWithin n δs sha fetch, hS, hρN, hwmax, hlk, hG,
    hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hStaged, hMu, hRB, hGs, fun hw x _ => hUF hw x trivial,
    fun x => hAcc x⟩

/-- **The honest period on the H200 meets `WorkOfH1CheckedCreditPeriodH200`'s hypotheses, so PoUW's per-period
guarantee holds on it**, at any SHA-512 and fetch price, `n ≥ 1` windows and `δ_s > 0`, with `δ_bind = δ_ks = 0`, at
the certificate `W = 512·credit`, the whole priced credit. -/
theorem h200_sound (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) (sha fetch : ℚ) :
    (honestRun n).Sound (cert n δs) (Accounting.h200S8Price sha fetch) .pouw (δs + n * (0 + 0)) := by
  obtain ⟨hN, hM, -⟩ := HonestH1.h1_meets n hn hδs
  obtain ⟨-, hWC, hHW, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hStaged, hMu, hRB, hGs,
    hUF, -⟩ := meets n hn hδs sha fetch
  exact WorkOfH1CheckedCreditPeriodH200 sha fetch 64 4 8 1024 Unit Unit Unit (fun _ => Paths HonestH1.Ω1 n) (device n)
    emptyCircuit (honestRun n) (cert n δs) Honest.scheme HonestH1.circuit hn hN HonestH1.hr1 hM HonestH1.hK24
    HonestH1.prm HonestH1.enc HonestH1.staged HonestH1.flat δs 0 0 (fun _ => HonestH1.windows n)
    (fun _ => HonestH1.opened) (fun _ => HonestH1.salt) (fun _ => HonestH1.period n) (fun _ => checkedPlay n)
    (fun _ _ => 128) (fun _ _ => 0) (realizes n δs sha fetch) hWC hHW
    hS hρN hwmax hlk (fun _ => hG) (fun _ => hDD) (fun _ => hPre) (fun _ => hMoved) (fun _ => hReg) (fun _ => hDraw)
    (fun _ => hServ) (fun _ => hSalt) hStaged (fun _ => hMu) (fun _ => hRB) (fun _ => hGs) (fun _ => hUF)

/-- **The honest period on the H200 meets every hypothesis of `WorkOfCreditedWords`**, with the source's bound
`HonestH1.h1_credit`: the same conclusion as `h200_sound`, through the generic bridge. -/
theorem credited_sound (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) (sha fetch : ℚ) :
    (honestRun n).Sound (cert n δs) (Accounting.h200S8Price sha fetch) .pouw (δs + n * (0 + 0)) := by
  refine WorkOfCreditedWords Unit Unit Unit (fun _ => Paths HonestH1.Ω1 n) (device n) emptyCircuit (honestRun n)
    (cert n δs) (Accounting.h200S8Price sha fetch) _ (Accounting.receiptUnitMacs 64 4 8) 1024 _ _
    (fun _ x => n * ((checkedPlay n).run x).fixed) _
    (fun _ _ => 0) (δs + n * (0 + 0)) (realizes n δs sha fetch) (wordCosts n δs sha fetch)
    (helpWithin n δs sha fetch) fun _ => ?_
  refine le_trans ?_ (HonestH1.h1_credit n hn hδs)
  exact pr_le_prob_of_equiv (Equiv.refl _) fun _ h => ⟨trivial, h.2⟩

end Accounting.SecurityProofs.Bridge.HonestH200

namespace Accounting.SecurityProofs.Bridge

open Properties.Pouw.Period Properties.Pouw.Period.SecurityProofs
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposePeriod
open HonestH200

/-- **`WorkOfH1CheckedCreditPeriodH200`'s honest witness**: its hypotheses hold at the honest period on the H200
(`HonestH200.honestRun`), one window of H1's honest period at the run's `δ = 10⁻⁶`, whose PoUW verdict is the compiled
check's (`PouwCheck.protocol.check` accepting every window, `HonestH200.checkedAccepts`), at the H200's INT8 price with
the SHA-512 and fetch prices, which no tensor-core price reads (`HonestH200.h200S8Price_tc`), at `0`. -/
theorem WorkOfH1CheckedCreditPeriodH200_honest :
    Nonempty (Accounting.RealizesCreditedWords (honestRun 1) (cert 1 (1 / 1000000)) (Accounting.h200S8Price 0 0)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => Accounting.receiptWordMacs 4 8))
      (Accounting.receiptUnitMacs 64 4 8) 1024 (Paths HonestH1.Ω1 1) (fun _ => (checkedPlay 1).accepts)
      (fun _ x => creditPrefix Honest.scheme HonestH1.circuit 144 ((HonestH1.windows 1).prefixLen HonestH1.prm x)
        (checkedLive Honest.scheme HonestH1.circuit (HonestH1.period 1) HonestH1.prm (HonestH1.windows 1) x) 1 1
        (67 / 8) (1 / 1000000))) ∧
    Accounting.Assumptions.WordCostsMacs (honestRun 1) (Accounting.h200S8Price 0 0)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => Accounting.receiptWordMacs 4 8))
      (Accounting.receiptUnitMacs 64 4 8) (realizes 1 (1 / 1000000) 0 0).coin (fun _ => (checkedPlay 1).accepts)
      (fun _ x => (1 : ℕ) * ((checkedPlay 1).run x).fixed) (fun _ _ => 0) ∧
    Accounting.Assumptions.HelpWithinCertifier (honestRun 1) (cert 1 (1 / 1000000)) 1024
      (realizes 1 (1 / 1000000) 0 0).coin (fun _ => (checkedPlay 1).accepts) (fun _ _ => 0) ∧
    Scope Honest.scheme HonestH1.circuit (1 / 1000000) ∧
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * shareCone Honest.scheme HonestH1.circuit 144 t ≤ 1) ∧
    (∀ j, stageShare (tileStage HonestH1.circuit 144) (shareCone Honest.scheme HonestH1.circuit 144) j ≤ 1) ∧
    (∀ t : Fin 144, 0 < shareCone Honest.scheme HonestH1.circuit 144 t →
      tileLuck Honest.scheme HonestH1.circuit 144 t ≤ 7 / 8 ∧ tileLuck0 Honest.scheme HonestH1.circuit 144 t ≤ 1 / 2) ∧
    (checkedPlay 1).game =
      periodGame (tileStage HonestH1.circuit 144) (liveCone Honest.scheme HonestH1.circuit 144) 1 1 24
        (HonestH1.period 1) ∧
    (HonestH1.period 1).EveryWindow (fun live _ => DeadConeDead Honest.scheme HonestH1.circuit 144 live) ∧
    PrefixOf (HonestH1.layer 1) (checkedPlay 1).game ∧
    (∀ w, prob (fun x => (HonestH1.layer 1).MovedIn x w) ≤ 0) ∧
    RegisteredBeforeDraw (HonestH1.windows 1) ∧ DrawIsCoin HonestH1.hK24 HonestH1.prm (HonestH1.windows 1) ∧
    ServiceSound HonestH1.prm (HonestH1.windows 1) HonestH1.opened (stagedAt HonestH1.salt HonestH1.staged) 0 ∧
    SaltIsLiveCoin HonestH1.hr1 (HonestH1.salt (n := 1)) HonestH1.enc ∧ StagedPins HonestH1.staged HonestH1.flat ∧
    (HonestH1.period 1).EveryWindow
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k G _ => OperandHMu (Honest.scheme.mu k) G)) ∧
    (HonestH1.period 1).EveryWindow
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k G _ => ReceiptLuckAt (Honest.scheme.rB k) G)) ∧
    (HonestH1.period 1).EveryWindow
      (fun live W => PerLiveTile Honest.scheme HonestH1.circuit live W
        (fun k _ G0 => GuessLuckAt (Honest.scheme.guess k) G0)) ∧
    UnitFixed (checkedPlay 1) (HonestH1.layer 1) (periodMode HonestH1.circuit (HonestH1.period 1) (HonestH1.run 1))
      (fun _ t => (tileWords Honest.scheme HonestH1.circuit 144 t : ℝ)) (fun _ => 128) ∧
    ∀ x, checkedAccepts 1 x :=
  meets 1 Nat.one_pos (by norm_num) 0 0

/-- **`WorkOfH1CheckedCreditPeriodH200`'s conclusion fails at the verifier that accepts everything**
(`HonestH200.acceptAll`): at no windows the device runs nothing, so at the work certificate `W = 1` the compute
account's work, `0`, is below `W` on every coin while PoUW accepts, and PoUW's guarantee fails at error `½`. -/
theorem WorkOfH1CheckedCreditPeriodH200_nontrivial :
    ¬ (acceptAll 0).Sound ⟨0, 1, 0, 0, 0⟩ (Accounting.h200S8Price 0 0) .pouw (1 / 2) := by
  intro h
  have h1 := h fun _ => ()
  have hev : ∀ x : Paths HonestH1.Ω1 0,
      (acceptAll 0).accepts .pouw (Function.update (fun _ => ()) ((acceptAll 0).own .pouw) x) ∧
        ¬ (acceptAll 0).Holds ⟨0, 1, 0, 0, 0⟩ (Accounting.h200S8Price 0 0) .pouw
          (Function.update (fun _ => ()) ((acceptAll 0).own .pouw) x) := fun x => by
    refine ⟨trivial, ?_⟩
    show ¬ (1 : ℝ) ≤ ((acceptAll 0).work _ _ : ℝ)
    rw [work_acceptAll]
    norm_num
  have hpr : Pouw.pr (fun x : Paths HonestH1.Ω1 0 =>
      (acceptAll 0).accepts .pouw (Function.update (fun _ => ()) ((acceptAll 0).own .pouw) x) ∧
        ¬ (acceptAll 0).Holds ⟨0, 1, 0, 0, 0⟩ (Accounting.h200S8Price 0 0) .pouw
          (Function.update (fun _ => ()) ((acceptAll 0).own .pouw) x)) = 1 :=
    haveI : Nonempty (Paths HonestH1.Ω1 0) := ⟨()⟩
    Pouw.pr_eq_one hev
  rw [hpr] at h1
  norm_num at h1

end Accounting.SecurityProofs.Bridge
