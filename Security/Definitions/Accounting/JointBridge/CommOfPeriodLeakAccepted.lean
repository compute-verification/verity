import Definitions.Accounting.JointPeriod
import Definitions.NetworkCertifier.Period
import Definitions.NetworkCertifier.Tap
import Definitions.Core.Assumptions

/-!
# A period whose communication audit is the network certifier's period of sampled audits (trusted)

`Accounting.PeriodRun` leaves the network certifier's claim (`commWithin`) and its verdict (`accepts .network`) as
fields. `RealizesCommOfPeriodLeakAccepted` is a run whose communication audit is the monitoring period of `n` windows
that `NetTiming.Guarantees.PeriodLeakAccepted` is about: each window audited by the sampled verifier at a statement
registered as `sw i`, the network certifier recording each window's observations and registering its records at the
transfer gates (its openings `tops s`). The run's claim at `K` is that every window meets `NetTiming.WindowGood`, the
event whose secrets `PeriodLeakAccepted` counts, so `K` is the bits of that count; the audit's own coins, at each
window, are a run of the sampled verifier against a prover whose tape is the rest of the coins. The run carries the
deployed network certifier in `PeriodLeakAccepted`'s form, and its trust, `trusted-host`
(`Verity.Assumptions.TrustedHost`), is a field: the per-window bound (`window_le`) is what its records give.
-/

namespace Accounting.JointBridge

open Claims Claims.Sampling NetTiming

/-- **A period whose communication audit is the network certifier's period of `n` sampled audits**, at the
deployment `D` of the sampled verifier, the public layouts `lay`, horizons `k`, registered inputs `reg` and jitter
bound `Dj`, at `δ` and the decoder `decode`, with the claim at `K`. Every field is a property of the deployed run:

* the period's windows, as `PeriodLeakAccepted` takes them: each window's registration (`sw`) and its registered
  values (`ρ`), and the network certifier's openings at each statement (`tops`);
* at the coins, each window's statement (`stmt`), the audit's extracted witness there (`wit`) and the network
  certifier's record of the window (`obs`);
* the claim at `K` is that every window meets `WindowGood` (`comm_iff`): that is the event whose secrets
  `PeriodLeakAccepted`'s count bounds, so `K` is the bits of that count,
  `log2 (#Ss · ∏_c (cap_c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i)`. No field ties the
  number `K` to the count yet (the follow-up: `Properties.NetworkPeriod.rowsKBoundsCount`, that the rows form's `K` is
  at least `⌈log2⌉` of the count);
* the statements are in the sampled audit's scope (`scope`): the registration refuses any other;
* the deployed network certifier, in `PeriodLeakAccepted`'s form: one device (`device`) on the server's uplink, with
  its connections `Conn`, cells `Cell` and the server's state `Srv`, whose run (`run`) on a schedule and the server's
  state gives the cells the client receives and its records of the period's windows; its record format (`fmt`) and
  what reaches it from the server on each connection (`arr`);
* the network certifier runs its code on its host (`trusted`, `trusted-host`);
* at each window and every value of the other coins, the network audit's own coins are a run of the sampled verifier
  at the window's statement against a prover (`prover`, on a finite tape `Tape`, the rest of the coins), and the
  probability over the audit's own coins that the network certifier accepts while the window fails `WindowGood` is at
  most that run's `PrBad` at the leak relation at the network certifier's rows, `RLeakTap` (`window_le`). Past
  `RLeakTap` and the scope, `WindowGood` holds the trace's values at the registered inputs and the records' coverage
  (`Covered`, length and jitter): `window_le` holds them on acceptance. The registered values are the gap
  `PeriodLeakAccepted` names, which the registration's commitment binds; no proof here gives it. `window_le` is the
  per-window bound that the trusted network certifier's records give; deriving it from `PeriodLeakAccepted` at the
  deployed period and `trusted`, rather than reading it as a field, is the follow-up. -/
structure RealizesCommOfPeriodLeakAccepted {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin]
    [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (r : PeriodRun d α C Ω) (K : ℝ) (H : Msg → Msg) (D : Deploy Coin Col T) (L Kₒ S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (reg : AuditStatement → Finset ℕ)
    (Dj n : ℕ) where
  /-- each window's registration -/
  sw : Fin n → AuditStatement
  /-- each window's registered values at its registered inputs -/
  ρ : Fin n → ℕ → Bool
  /-- the network certifier's openings at each statement, under its roots -/
  tops : AuditStatement → List (Opening Bool)
  /-- each window's statement at the coins -/
  stmt : ((i : I) → Ω i) → Fin n → Msg
  /-- the audit's extracted witness at each window, at the coins -/
  wit : ((i : I) → Ω i) → Fin n → Wit
  /-- the network certifier's record of each window, at the coins -/
  obs : ((i : I) → Ω i) → Fin n → List (Obs L Kₒ S)
  /-- the claim at `K`: every window meets its good event -/
  comm_iff : ∀ ξ, r.commWithin K ξ ↔
    ∀ i, WindowGood H D L Kₒ S lay k δ reg Dj (sw i) (ρ i) tops (stmt ξ i) (wit ξ i) (obs ξ i)
  /-- every window's statement is in the sampled audit's scope -/
  scope : ∀ ξ i, ScopeAcc D lay k (stmt ξ i)
  /-- the tape of window `i`'s prover at the other coins `ξ` -/
  Tape : ((i : I) → Ω i) → Fin n → Type
  [tapeFintype : ∀ ξ i, Fintype (Tape ξ i)]
  [tapeNonempty : ∀ ξ i, Nonempty (Tape ξ i)]
  /-- window `i`'s prover at the other coins `ξ` -/
  prover : (ξ : (i : I) → Ω i) → (i : Fin n) → Prover (Tape ξ i)
  /-- the connections on the server's uplink that the network certifier carries -/
  Conn : Type
  [connFintype : Fintype Conn]
  /-- the cells it carries -/
  Cell : Type
  /-- the server's state, which fixes what reaches the network certifier -/
  Srv : Type
  /-- the network certifier's devices -/
  Dev : Type
  /-- what reaches the network certifier from the server on each connection, at the server's state -/
  arr : Srv → Conn → Arrivals Cell
  /-- the network certifier's record format over the period's windows -/
  fmt : TapRecord Conn Cell (Fin n → List (Obs L Kₒ S))
  /-- the deployed network certifier -/
  device : Dev
  /-- what the network certifier does on a schedule and the server's state: the cells the client receives and its
  records of the period's windows -/
  run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L Kₒ S))
  /-- the network certifier runs its code (`tapCode`) on its host: `trusted-host`, the trust the network audit rests
  on. Its records are what `window_le`'s per-window bound comes from; deriving `window_le` from `PeriodLeakAccepted`
  at this deployed period and this premise, not at the period of no windows, is the follow-up. -/
  trusted : Verity.Assumptions.TrustedHost device run fun p => tapCode fmt p.1 (arr p.2)
  /-- the network audit's acceptance with window `i` failing its good event, over the audit's own coins, is at most
  the sampled verifier's `PrBad` at the window's statement and prover: the per-window bound that the trusted network
  certifier's records (`trusted`) give. It is a field until it is derived from `PeriodLeakAccepted` at the deployed
  period (the follow-up). -/
  window_le : ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      r.accepts .network (Function.update ξ (r.own .network) x) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (sw i) (ρ i) tops (stmt (Function.update ξ (r.own .network) x) i)
          (wit (Function.update ξ (r.own .network) x) i) (obs (Function.update ξ (r.own .network) x) i)) ≤
    PrBad H D.V
      (RLeakTap H D.commit D.parse D.partition L Kₒ S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops)
      (E D decode) (stmt ξ i) (prover ξ i)

attribute [instance] RealizesCommOfPeriodLeakAccepted.tapeFintype RealizesCommOfPeriodLeakAccepted.tapeNonempty
  RealizesCommOfPeriodLeakAccepted.connFintype

end Accounting.JointBridge
