import Definitions.Accounting.JointPeriod
import Definitions.NetworkCertifier.Period

/-!
# A period whose communication audit is the network certifier's period of sampled audits (trusted)

`Accounting.PeriodRun` leaves the network certifier's claim (`commWithin`) and its verdict (`accepts .network`) as
fields. `RealizesCommOfPeriodLeakAccepted` is a run whose communication audit is the period of `n` windows that
`NetTiming.Guarantees.PeriodLeakAccepted` is about: each window audited by the sampled verifier at a statement
registered as `sw i`, the network certifier recording each window's observations. The run's claim at `K` is that every
window meets `NetTiming.WindowGood`, the event whose secrets `PeriodLeakAccepted` counts, so `K` is what that count
allows the node to move; the audit's own coins, at each window, are a run of the sampled verifier against a prover
whose tape is the rest of the coins.
-/

namespace Accounting.JointBridge

open Claims Claims.Sampling NetTiming

/-- **A period whose communication audit is the network certifier's period of `n` sampled audits**, at the
deployment `D` of the sampled verifier, the public layouts `lay`, horizons `k`, registered inputs `reg` and jitter
bound `Dj`, at `δ` and the decoder `decode`, with the claim at `K`. Every field is a property of the deployed run:

* the period's windows, as `PeriodLeakAccepted` takes them: each window's registration (`sw`), its registered values
  (`ρ`) and its operator (`op`);
* at the coins, each window's statement (`stmt`), the audit's extracted witness there (`wit`) and the network
  certifier's record of the window (`obs`);
* the claim at `K` is that every window meets `WindowGood` (`comm_iff`): that is the event whose secrets
  `PeriodLeakAccepted`'s count bounds, so `K` is the bits that count allows;
* the statements are in the sampled audit's scope (`scope`): the registration refuses any other;
* at each window and every value of the other coins, the network audit's own coins are a run of the sampled verifier
  at the window's statement against a prover (`prover`, on a finite tape `Tape`, the rest of the coins), and the
  probability over the audit's own coins that the network certifier accepts while the window fails `WindowGood` is at
  most that run's `PrBad` at the leak relation `RLeak` (`window_le`). Past `RLeak` and the scope, `WindowGood` holds
  the trace's registered values and its transfer gates carrying the network certifier's rows (`Carries`, `Covered`):
  `window_le` holds them on acceptance, which is the gap `PeriodLeakAccepted` names (the registration's commitment
  binds the values, and the audit's registered reads of the certifier's rows, opened against its signed root, tie
  the transfer gates); no proof here gives either. -/
structure RealizesCommOfPeriodLeakAccepted {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin]
    [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (r : PeriodRun d α C Ω) (K : ℝ) (H : Msg → Msg) (D : Deploy Coin Col T) (L Kₒ S : ℕ)
    (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (reg : Stmt → Finset ℕ)
    (Dj n : ℕ) where
  /-- each window's registration -/
  sw : Fin n → Stmt
  /-- each window's registered values at its registered inputs -/
  ρ : Fin n → ℕ → Bool
  /-- each window's operator on the network certifier's observations -/
  op : Fin n → ℕ → List (Obs L Kₒ S) → List Bool
  /-- each window's statement at the coins -/
  stmt : ((i : I) → Ω i) → Fin n → Msg
  /-- the audit's extracted witness at each window, at the coins -/
  wit : ((i : I) → Ω i) → Fin n → Wit
  /-- the network certifier's record of each window, at the coins -/
  obs : ((i : I) → Ω i) → Fin n → List (Obs L Kₒ S)
  /-- the claim at `K`: every window meets its good event -/
  comm_iff : ∀ ξ, r.commWithin K ξ ↔
    ∀ i, WindowGood H D L Kₒ S lay k δ reg Dj (sw i) (ρ i) (op i) (stmt ξ i) (wit ξ i) (obs ξ i)
  /-- every window's statement is in the sampled audit's scope -/
  scope : ∀ ξ i, ScopeAcc D lay k (stmt ξ i)
  /-- the tape of window `i`'s prover at the other coins `ξ` -/
  Tape : ((i : I) → Ω i) → Fin n → Type
  [tapeFintype : ∀ ξ i, Fintype (Tape ξ i)]
  [tapeNonempty : ∀ ξ i, Nonempty (Tape ξ i)]
  /-- window `i`'s prover at the other coins `ξ` -/
  prover : (ξ : (i : I) → Ω i) → (i : Fin n) → Prover (Tape ξ i)
  /-- the network audit's acceptance with window `i` failing its good event, over the audit's own coins, is at most
  the sampled verifier's `PrBad` at the window's statement and prover -/
  window_le : ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      r.accepts .network (Function.update ξ (r.own .network) x) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (sw i) (ρ i) (op i) (stmt (Function.update ξ (r.own .network) x) i)
          (wit (Function.update ξ (r.own .network) x) i) (obs (Function.update ξ (r.own .network) x) i)) ≤
    PrBad H D.V (RLeak H D.commit D.parse D.partition L Kₒ S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj)
      (E D decode) (stmt ξ i) (prover ξ i)

attribute [instance] RealizesCommOfPeriodLeakAccepted.tapeFintype RealizesCommOfPeriodLeakAccepted.tapeNonempty

end Accounting.JointBridge
