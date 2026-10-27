import Definitions.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod
import Properties.Pouw.Period

/-!
# The joint period's PoUW guarantee from H1's checked credit over the torch loop

`Accounting.Guarantees.JointPeriod` takes each audit's per-period guarantee (`Accounting.PeriodRun.Sound`) as a
hypothesis. `WorkOfH1CheckedCreditPeriod` gives PoUW's, at the work certificate `W` and price `p`, for a period that
realizes H1's checked credit (`Accounting.RealizesWorkOfH1CheckedCreditPeriod` at the play's acceptance, `n` times its
run's fixed words, and the check's credit over the accepted prefix), from
`Properties.Pouw.Period.H1CheckedCreditPeriod` and its premises, with its error `δ_s + n·(δ_bind + δ_ks)`.

The other audits' coins fix the adversary the device is in H1's game, so the windows, the service's openings, the
salts, the period's live tiles, the play and its folded words are functions of them (`ws ξ`, …), and each premise of
the source that reads them is taken at every value `ξ` of the coins. The scheme, the declared circuit, the number of
windows, the check's parameters, the salt's encoding, the staged and flat statements and the errors are the protocol's
and are fixed. The premises are the source's, unchanged; the gaps it names (`RegisteredBeforeDraw` and the binding it
reads on today's tags, `SaltIsLiveCoin`, `StagedPins`) stay premises here. The one gap this bridge adds is the
structure's `work_of_fixed` (RT4-LOGIC-6): the period's words fixed right are within the compute account's
device-model work at `p`.
-/

namespace Properties.Accounting.JointBridge

open Properties.Pouw.Period PouwCheck
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **PoUW's per-period guarantee in the joint period, from H1's checked credit on the torch loop.** On any device and
any period of a node's run on one coin space whose PoUW audit realizes H1's checked credit
(`Accounting.RealizesWorkOfH1CheckedCreditPeriod`: PoUW's coordinate of the coins is the source game's, its verdict
the play's acceptance, the work certificate within the check's credit on an accepted period, and the words fixed right
within the compute account's work at price `p`), with the source's instance at each value `ξ` of the coins meeting
`H1CheckedCreditPeriod`'s premises: PoUW's per-period guarantee holds at certificates `c`, price `p` and error
`δ_s + n·(δ_bind + δ_ks)`, i.e. at any value of the other coins, the probability over PoUW's own coins that it accepts
while the compute account did less than `W` of work at `p` is at most that. -/
def WorkOfH1CheckedCreditPeriod : Prop :=
  ∀ (ι α I : Type) [Fintype α] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    (d : Accounting.Device ι) (C : Verity.Circuit) (r : Accounting.PeriodRun d α C Ω) (c : Accounting.Certificates)
    (p : Accounting.Price)
    {ι' K T L U Cn Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype Cn] [Fintype Ξ]
    [Nonempty V] (Sc : Scheme K) (Cd : Declared ι') {n : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ 144)
    (hr : 0 < Cd.rounds) (_hM : Cd.rounds = 1) (hK : 24 ≤ 144) (prm : Params) (enc : Cn → ByteArray)
    (staged flat : ByteArray → Fin 144 → (ℕ → V) → Prop) (δs δbind δks : ℝ)
    (ws : ((i : I) → Ω i) → Windows (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n)
    (opened : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (salt : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → Fin n → ByteArray)
    (Wp : ((i : I) → Ω i) → Period Cn (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) (Fin 144) Cd.rounds n)
    (Pl : ((i : I) → Ω i) → Play (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n K T L U)
    (folded : ((i : I) → Ω i) → Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n → ℝ),
    Accounting.RealizesWorkOfH1CheckedCreditPeriod r c p (Paths (Paths Cn Cd.rounds × (Equiv.Perm (Fin 144) × Ξ)) n)
      (fun ξ => (Pl ξ).accepts) (fun ξ x => n * ((Pl ξ).run x).fixed)
      (fun ξ x => creditPrefix Sc Cd 144 ((ws ξ).prefixLen prm x) (checkedLive Sc Cd (Wp ξ) prm (ws ξ) x) 1 1 (67 / 8)
        δs) →
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

end Properties.Accounting.JointBridge
