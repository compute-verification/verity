import Proofs.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod
import Proofs.Pouw.Period.Honest

/-!
# An honest period on the H200 meets every hypothesis of `WorkOfH1CheckedCreditPeriodH200`

The positive twin of `Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriodH200`, and of `WorkOfCreditedWords`
at its instance: `Properties.Pouw.Period.SecurityProofs.HonestH1`'s live-salt period of `n ≥ 1` windows, 128 words
live and fixed a window, run by a device that spends `512` INT8 tensor-core multiply-adds on each word and outsources
no unit, meets every hypothesis at the H200's INT8 price, at `PouwH1Tile_v1`'s `2048` multiply-adds a unit and
`b_unit = 1024`, with the work certificate the whole priced credit, `W = 512·credit` (`h200_sound`, `credited_sound`).

* The period (`jointRun`): one coin coordinate, PoUW's, which is H1's paths; one account; the empty circuit; a window
  of one tick, in which the compute account runs `512·128·n` tensor-core multiply-adds and nothing else, the device's
  whole rate (`device`). PoUW's audit accepts where the play does, everywhere.
* The certificates (`cert`): `W = 512·creditPrefix(n, 128·n)`, the check's credit at the honest period's accepted
  prefix and live words (`HonestH1.prefixLen_all`, `HonestH1.checkedLive_all`) at `m = 512` and `p(tc) = 1`
  (`h200S8Price_tc`), and `K = 0`, so `help = 0` units meets `HelpWithinCertifier`.
* `m = 512`: the honest circuit's 128 words are all credited (`HonestH1.wordsOf_eq`), so the minimum is over them
  (`minWordMacs_const`), and `WordCostsMacs` is `512·1·(n·128) ≤ 512·128·n`, the period's work (`work_jointRun`).
-/

namespace Accounting.SecurityProofs.Bridge.HonestH200

open Properties.Pouw.Period Properties.Pouw.Period.SecurityProofs
open Accounting.TS Accounting.Compose Pouw.RelativeSecurity
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
  unfold Verity.Step.Fits
  refine ⟨fun o => ?_, ?_, fun _ => ?_, ?_, ?_⟩ <;> simp [step, device]

/-- The honest period on PoUW's coins, H1's paths: one tick of the device's whole rate, every audit accepting. -/
def jointRun (n : ℕ) : Accounting.PeriodRun (device n) Unit emptyCircuit (fun _ : Unit => Paths HonestH1.Ω1 n) where
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

theorem work_jointRun (n : ℕ) (p : Accounting.Price) (ξ : (i : Unit) → Paths HonestH1.Ω1 n) :
    (jointRun n).work p ξ = p .tc * (65536 * n) := by
  unfold Accounting.PeriodRun.work Verity.Schedule.work Verity.Schedule.ops
  rw [Fintype.sum_eq_single Verity.OpClass.tc]
  · simp [jointRun, step, device, Verity.Window.ticks]
  · intro o ho
    simp [jointRun, step, device, ho]

/-- The certificates: the whole priced credit as `W`, and no bits through the network certifiers. -/
noncomputable def cert (n : ℕ) (δs : ℝ) : Accounting.Certificates :=
  ⟨0, 512 * creditPrefix Honest.scheme HonestH1.circuit 144 n (128 * n) 1 1 (67 / 8) δs, 0, 0, 0⟩

/-- PoUW's audit of the honest period credits H1's checked words at the H200's INT8 price, `512` multiply-adds a word,
`2048` a unit and `1024` bits a unit. -/
noncomputable def realizes (n : ℕ) (δs : ℝ) (sha fetch : ℚ) :
    Accounting.RealizesCreditedWords (jointRun n) (cert n δs) (Accounting.h200S8Price sha fetch)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => Accounting.receiptWordMacs 4 8))
      (Accounting.receiptUnitMacs 64 4 8) 1024 (Paths HonestH1.Ω1 n) (fun _ => (HonestH1.play n).accepts)
      (fun _ x => creditPrefix Honest.scheme HonestH1.circuit 144 ((HonestH1.windows n).prefixLen HonestH1.prm x)
        (checkedLive Honest.scheme HonestH1.circuit (HonestH1.period n) HonestH1.prm (HonestH1.windows n) x) 1 1
        (67 / 8) δs) where
  coin := Equiv.refl _
  accepts_iff := fun _ _ => ⟨fun _ => trivial, fun _ => trivial⟩
  cert_le := fun _ y _ => by
    simp only [Equiv.refl_apply, HonestH1.prefixLen_all, HonestH1.checkedLive_all, minWordMacs_honest,
      h200S8Price_tc, cert, Accounting.Certificates.helpUnits]
    norm_num

theorem wordCosts (n : ℕ) (δs : ℝ) (sha fetch : ℚ) :
    Accounting.Assumptions.WordCostsMacs (jointRun n) (Accounting.h200S8Price sha fetch)
      (minWordMacs Honest.scheme HonestH1.circuit (fun _ => Accounting.receiptWordMacs 4 8))
      (Accounting.receiptUnitMacs 64 4 8) (realizes n δs sha fetch).coin (fun _ => (HonestH1.play n).accepts)
      (fun _ x => n * ((HonestH1.play n).run x).fixed) (fun _ _ => 0) := by
  intro ξ y _
  have hfix : ((HonestH1.play n).run ((realizes n δs sha fetch).coin y)).fixed = 128 := rfl
  simp only [hfix, work_jointRun, minWordMacs_honest, h200S8Price_tc, mul_zero, add_zero]
  push_cast
  linarith

theorem helpWithin (n : ℕ) (δs : ℝ) (sha fetch : ℚ) :
    Accounting.Assumptions.HelpWithinCertifier (jointRun n) (cert n δs) 1024 (realizes n δs sha fetch).coin
      (fun _ => (HonestH1.play n).accepts) (fun _ _ => 0) := by
  intro _ _ _
  simp [cert, Accounting.Certificates.helpUnits]

/-! ## Every hypothesis -/

/-- **The honest period on the H200 meets every hypothesis of `WorkOfH1CheckedCreditPeriodH200`**, at any SHA-512
and fetch price, `n ≥ 1` windows and `δ_s > 0`, with `δ_bind = δ_ks = 0`: PoUW's per-period guarantee holds at the
certificate `W = 512·credit`, the whole priced credit. -/
theorem h200_sound (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) (sha fetch : ℚ) :
    (jointRun n).Sound (cert n δs) (Accounting.h200S8Price sha fetch) .pouw (δs + n * (0 + 0)) := by
  obtain ⟨hN, hM, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hStaged, hMu, hRB, hGs,
    hUF, -, -, -⟩ := HonestH1.h1_meets n hn hδs
  exact WorkOfH1CheckedCreditPeriodH200 sha fetch 64 4 8 1024 Unit Unit Unit (fun _ => Paths HonestH1.Ω1 n) (device n)
    emptyCircuit (jointRun n) (cert n δs) Honest.scheme HonestH1.circuit hn hN HonestH1.hr1 hM HonestH1.hK24
    HonestH1.prm HonestH1.enc HonestH1.staged HonestH1.flat δs 0 0 (fun _ => HonestH1.windows n)
    (fun _ => HonestH1.opened) (fun _ => HonestH1.salt) (fun _ => HonestH1.period n) (fun _ => HonestH1.play n)
    (fun _ _ => 128) (fun _ _ => 0) (realizes n δs sha fetch) (wordCosts n δs sha fetch) (helpWithin n δs sha fetch)
    hS hρN hwmax hlk (fun _ => hG) (fun _ => hDD) (fun _ => hPre) (fun _ => hMoved) (fun _ => hReg) (fun _ => hDraw)
    (fun _ => hServ) (fun _ => hSalt) hStaged (fun _ => hMu) (fun _ => hRB) (fun _ => hGs) (fun _ => hUF)

/-- **The honest period on the H200 meets every hypothesis of `WorkOfCreditedWords`**, with the source's bound
`HonestH1.h1_credit`: the same conclusion as `h200_sound`, through the generic bridge. -/
theorem credited_sound (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) (sha fetch : ℚ) :
    (jointRun n).Sound (cert n δs) (Accounting.h200S8Price sha fetch) .pouw (δs + n * (0 + 0)) := by
  refine WorkOfCreditedWords Unit Unit Unit (fun _ => Paths HonestH1.Ω1 n) (device n) emptyCircuit (jointRun n)
    (cert n δs) (Accounting.h200S8Price sha fetch) _ (Accounting.receiptUnitMacs 64 4 8) 1024 _ _
    (fun _ x => n * ((HonestH1.play n).run x).fixed) _
    (fun _ _ => 0) (δs + n * (0 + 0)) (realizes n δs sha fetch) (wordCosts n δs sha fetch)
    (helpWithin n δs sha fetch) fun _ => ?_
  refine le_trans ?_ (HonestH1.h1_credit n hn hδs)
  exact pr_le_prob_of_equiv (Equiv.refl _) fun _ h => h

end Accounting.SecurityProofs.Bridge.HonestH200
