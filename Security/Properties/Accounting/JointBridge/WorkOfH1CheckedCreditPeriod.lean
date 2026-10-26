import Definitions.Accounting.JointBridge.CreditedWords
import Definitions.Accounting.JointBridge.H200
import Definitions.Accounting.Assumptions.CreditedWords
import Properties.Pouw.Period

/-!
# The joint period's PoUW guarantee from H1's checked credit over the torch loop

`Accounting.Guarantees.JointPeriod` takes each audit's per-period guarantee (`Accounting.PeriodRun.Sound`) as a
hypothesis. `WorkOfH1CheckedCreditPeriodAt p macs b` gives PoUW's, at the work certificate `W` and price `p`, for a
period whose PoUW audit credits H1's checked words (`Accounting.RealizesCreditedWords` at the play's acceptance and the
check's credit over the accepted prefix, `m = Accounting.Compose.minWordMacs Sc Cd macs` multiply-adds a word and `b`
bits a word), from `Properties.Pouw.Period.H1CheckedCreditPeriod` and its premises, with its error
`δ_s + n·(δ_bind + δ_ks)`. It is `Properties.Accounting.JointBridge.WorkOfCreditedWords` at H1's instance.
`WorkOfH1CheckedCreditPeriod` is it at every price, word count and word width, and `WorkOfH1CheckedCreditPeriodH200`
at the H200's INT8 price (`Accounting.h200S8Price`), an `R × C` tile's `16·R·C` multiply-adds a receipt word
(`Accounting.receiptWordMacs R C`) and its `32` bits.

The other audits' coins fix the adversary the device is in H1's game, so the windows, the service's openings, the
salts, the period's live tiles, the play, its folded words and the right words it imported are functions of them
(`ws ξ`, …), and each premise of the source that reads them is taken at every value `ξ` of the coins. The scheme, the
declared circuit, the number of windows, the check's parameters, the salt's encoding, the staged and flat statements
and the errors are the protocol's and are fixed. The premises are the source's, unchanged; the gaps it names
(`RegisteredBeforeDraw` and the binding it reads on today's tags, `SaltIsLiveCoin`, `StagedPins`) stay premises here.
The bridge adds two (RT4-LOGIC-6): each right word the device didn't import cost `m` multiply-adds of priced work on it
(`Accounting.Assumptions.WordCostsMacs`, at `n` times the run's fixed words), and it imported at most `K / b` right
words (`Accounting.Assumptions.HelpWithinCertifier`).
-/

namespace Properties.Accounting.JointBridge

open Properties.Pouw.Period PouwCheck
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **PoUW's per-period guarantee in the joint period, from H1's checked credit on the torch loop, at price `p`,
`macs k` multiply-adds a word of kind `k` and `b` bits a word.** On any device and any period of a node's run on one
coin space whose PoUW audit credits H1's checked words (`Accounting.RealizesCreditedWords`: PoUW's coordinate of the
coins is the source game's, its verdict the play's acceptance, and the work certificate within the check's credit less
the `K / b` words the network certifier's bits hold, at `m = minWordMacs Sc Cd macs` multiply-adds a word at `p .tc`),
each right word the device didn't import costing `m` multiply-adds of priced work (`WordCostsMacs` at `n` times the
run's fixed words and `help` imported words), at most `K / b` right words imported (`HelpWithinCertifier`), and the
source's instance at each value `ξ` of the coins meeting `H1CheckedCreditPeriod`'s premises: PoUW's per-period
guarantee holds at certificates `c`, price `p` and error `δ_s + n·(δ_bind + δ_ks)`, i.e. at any value of the other
coins, the probability over PoUW's own coins that it accepts while the compute account did less than `W` of work at `p`
is at most that. -/
def WorkOfH1CheckedCreditPeriodAt (p : Accounting.Price) (macs : Accounting.Compose.Kind → ℕ) (b : ℕ) : Prop :=
  ∀ (ι α I : Type) [Fintype α] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    (d : Accounting.Device ι) (C : Verity.Circuit) (r : Accounting.PeriodRun d α C Ω) (c : Accounting.Certificates)
    {ι' K T L U Cn Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype Cn] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι') {n : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ 144)
    (hr : 0 < Cd.rounds) (_hM : Cd.rounds = 1) (hK : 24 ≤ 144) (prm : Params) (enc : Cn → ByteArray)
    (staged flat : ByteArray → Fin 144 → (ℕ → V) → Prop) (δs δbind δks : ℝ)
    (ws : ((i : I) → Ω i) → Windows (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n)
    (opened : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (salt : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ByteArray)
    (Wp : ((i : I) → Ω i) → Period Cn (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) (Fin 144) Cd.rounds n)
    (Pl : ((i : I) → Ω i) → Play (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n K T L U)
    (folded : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ)
    (help : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ)
    (real : Accounting.RealizesCreditedWords r c p (Accounting.Compose.minWordMacs Sc Cd macs) b
      (Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n) (fun ξ => (Pl ξ).accepts)
      (fun ξ x => creditPrefix Sc Cd 144 ((ws ξ).prefixLen prm x) (checkedLive Sc Cd (Wp ξ) prm (ws ξ) x) 1 1 (67 / 8)
        δs)),
    Accounting.Assumptions.WordCostsMacs r p (Accounting.Compose.minWordMacs Sc Cd macs) real.coin
      (fun ξ => (Pl ξ).accepts) (fun ξ x => n * ((Pl ξ).run x).fixed) help →
    Accounting.Assumptions.HelpWithinCertifier r c b real.coin (fun ξ => (Pl ξ).accepts) help →
    Scope Sc Cd δs →
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * shareCone Sc Cd 144 t ≤ 1) →
    (∀ j, stageShare (tileStage Cd 144) (shareCone Sc Cd 144) j ≤ 1) →
    (∀ t : Fin 144, 0 < shareCone Sc Cd 144 t → tileLuck Sc Cd 144 t ≤ 7 / 8 ∧ tileLuck0 Sc Cd 144 t ≤ 1 / 2) →
    (∀ ξ, (Pl ξ).game = periodGame (tileStage Cd 144) (liveCone Sc Cd 144) 1 1 24 (Wp ξ)) →
    (∀ ξ, (Wp ξ).EveryWindow (fun live _ => DeadConeDead Sc Cd 144 live)) →
    (∀ ξ, PrefixOf (checkedLayer Cd hr hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat))
      (Pl ξ).game) →
    (∀ ξ w, prob (fun x =>
      (checkedLayer Cd hr hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat)).MovedIn x w) ≤
        δbind) →
    (∀ ξ, RegisteredBeforeDraw (ws ξ)) → (∀ ξ, DrawIsCoin hK prm (ws ξ)) →
    (∀ ξ, ServiceSound prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) δks) →
    (∀ ξ, SaltIsLiveCoin hr (salt ξ) enc) → StagedPins staged flat →
    (∀ ξ, (Wp ξ).EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G))) →
    (∀ ξ, (Wp ξ).EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G))) →
    (∀ ξ, (Wp ξ).EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0))) →
    (∀ ξ, UnitFixed (Pl ξ)
      (checkedLayer Cd hr hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat))
      (periodMode Cd (Wp ξ) (checkedRun hK prm (ws ξ) (opened ξ) (stagedAt (salt ξ) staged) (flatAtCoin hr enc flat)))
      (fun _ t => (tileWords Sc Cd 144 t : ℝ)) (folded ξ)) →
    r.Sound c p .pouw (δs + n * (δbind + δks))

/-- **PoUW's per-period guarantee in the joint period from H1's checked credit**, at every price `p`, per-kind
multiply-add count `macs` and word width `b` (`WorkOfH1CheckedCreditPeriodAt`). -/
def WorkOfH1CheckedCreditPeriod : Prop :=
  ∀ (p : Accounting.Price) (macs : Accounting.Compose.Kind → ℕ) (b : ℕ), WorkOfH1CheckedCreditPeriodAt p macs b

/-- **PoUW's per-period guarantee in the joint period from H1's checked credit on the H200, at INT8, for `R × C`
tiles**: `WorkOfH1CheckedCreditPeriodAt` at `Accounting.h200S8Price sha fetch` (an INT8 tensor-core multiply-add `1`, a
CUDA-core op `32`, an oracle query `⌈sha⌉`, in dense E4M3 multiply-adds, W1's prices on `W1Device.s8H100`), every word
at an `R × C` tile's `receiptWordMacs R C = 16·R·C` multiply-adds at any `K`, and `32` bits a word (a receipt word is
the XOR of the lanes' 32-bit halves), for every SHA-512 and fetch price and every tile shape. `PouwH1Tile_v1`'s is
`R × C = 4 × 8`, `512` a word; `PouwH1Tile_v5` and `_v6` (v1 and v3 under J10-R's noise, with v1's receipt) at a warp,
`32 × 32`, `16384`. The certificate it admits on an accepted period is `W ≤ 16·R·C·(credit − K/32)` dense E4M3
multiply-adds, for a declared circuit with a credited word (`m` is `0` for one with none); a circuit of several tile
shapes takes the smallest. -/
def WorkOfH1CheckedCreditPeriodH200 : Prop :=
  ∀ (sha fetch : ℚ) (R C : ℕ),
    WorkOfH1CheckedCreditPeriodAt (Accounting.h200S8Price sha fetch) (fun _ => Accounting.receiptWordMacs R C) 32

end Properties.Accounting.JointBridge
