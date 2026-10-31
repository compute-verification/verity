import Definitions.Accounting.JointPeriod
import Definitions.NetworkCertifier.Period
import Definitions.NetworkCertifier.SampledReg
import Definitions.NetworkCertifier.Tap
import Definitions.Core.Assumptions

/-!
# A period whose communication audit is the network certifier's period of sampled audits (trusted)

`Accounting.PeriodRun` leaves the network certifier's claim (`commWithin`) and its verdict (`accepts .network`) as
fields. `RealizesCommOfPeriodLeakAccepted` is a run whose communication audit is the monitoring period of `n` windows
that `NetTiming.Guarantees.PeriodLeakAccepted` is about: each window audited by the sampled verifier at a statement
registered as `sw i`, the network certifier recording each window's observations and registering its records at the
transfer gates (its openings `tops s`). The run's claim at `K` is that every window meets `NetTiming.WindowGood`, the
event whose secrets `PeriodLeakAccepted` counts, and `K` is at least `log2` of that count at the period's schedules,
caps, partitions and faulty bits (`count_le`); the audit's own coins, at each window, are a run of the sampled verifier
against a prover whose tape is the rest of the coins. The run carries the deployed network certifier in
`PeriodLeakAccepted`'s form, its records the device's run on the period's schedule (`obs_run`), and its trust,
`trusted-host` (`Verity.Assumptions.TrustedHost`), is a field: the per-window bound (`window_le`) is what its records
give.
-/

namespace Accounting.JointBridge

open Claims Claims.Sampling NetTiming

/-- **A period whose communication audit is the network certifier's period of `n` sampled audits**, at the
deployment `D` of the sampled verifier, the public layouts `lay`, horizons `k`, registered inputs `reg` and jitter
bound `Dj`, at `δ` and the decoder `decode`, with the claim at `K`. Every field is a property of the deployed run:

* the period's windows, as `PeriodLeakAccepted` takes them: each window's registration (`sw`) and its registered
  values (`ρ`), and the network certifier's openings (`tops`) and the registration's (`rops`) at each statement;
* at the coins, each window's statement (`stmt`), the audit's extracted witness there (`wit`) and the network
  certifier's record of the window (`obs`);
* the claim at `K` is that every window meets `WindowGood` (`comm_iff`): that is the event whose secrets
  `PeriodLeakAccepted`'s count bounds;
* the statements are in the sampled audit's scope (`scope`): the registration refuses any other;
* each window's statement's registration opens its registered inputs under the statement's roots (`rops_opens`), to
  the window's registered values (`rops_ρ`);
* the deployed network certifier, in `PeriodLeakAccepted`'s form: one device (`device`) on the server's uplink, with
  its connections `Conn`, cells `Cell` and the server's state `Srv`, whose run (`run`) on a schedule and the server's
  state gives the cells the client receives and its records of the period's windows; its record format (`fmt`) and
  what reaches it from the server on each connection (`arr`);
* the network certifier runs its code on its host (`trusted`, `trusted-host`);
* each window's partition (`Pw`, its registration's) and a bound `β` on the committed bits of each of its possible
  faulty sets;
* the schedules the period admits (`Ss`), each giving connection `c` at most `cap c` cells and held by the record
  format; at the coins, the network certifier's schedule (`sched`), one of them (`sched_mem`), and the server's state
  (`srv`), on which the records are the device's run (`obs_run`);
* `K` is at least `log2` of `PeriodLeakAccepted`'s count at these (`count_le`):
  `#Ss · ∏_c (cap_c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i ≤ 2^K`, which the period
  check's `K` gives at its numbers (`Properties.NetworkPeriod.RowsKBoundsCount`, `ClockKBoundsCount`);
* at each window and every value of the other coins, the network audit's own coins are a run of the sampled verifier
  at the window's statement against a prover (`prover`, on a finite tape `Tape`, the rest of the coins), and the
  probability over the audit's own coins that the network certifier accepts while the window fails `WindowGood` is at
  most that run's `PrBad` at the leak relation at the network certifier's rows with the registered inputs pinned at the
  registration's values, `RLeakTap` at `regVal (rops s)` (`window_le`). By `rops_ρ` that relation is `WindowGood`'s
  at the window's statement; past it and the scope, `WindowGood` holds the records' coverage (`Covered`, length and
  jitter): `window_le` holds them on acceptance. A trace disagreeing with the registration opens its commitment two
  ways, a collision (`B_reg`) that the guarantee charges to the network component's finder like the other two.
  `window_le` is the per-window bound that the trusted network certifier's records give; deriving it from
  `PeriodLeakAccepted` at the deployed period and `trusted`, rather than reading it as a field, is the follow-up. -/
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
  /-- the registration's openings of the registered inputs at each statement, under its roots -/
  rops : AuditStatement → List (Opening Bool)
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
  /-- every window's statement's registration opens its registered inputs -/
  rops_opens : ∀ ξ i st, D.parse (stmt ξ i) = some st → RegOpens H D.commit st (reg st) (rops st)
  /-- to the window's registered values -/
  rops_ρ : ∀ ξ i st, D.parse (stmt ξ i) = some st → ∀ g ∈ reg st, regVal (rops st) g = ρ i g
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
  /-- each window's partition -/
  Pw : Fin n → Partition
  /-- the registration's partition -/
  hPw : ∀ i, D.partition (sw i) = some (Pw i)
  /-- a bound on the committed bits of each of window `i`'s possible faulty sets -/
  β : Fin n → ℕ
  hβ : ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i
  /-- the most cells the period admits on each connection -/
  cap : Conn → ℕ
  /-- the schedules the period admits -/
  Ss : Set (WardenSchedule Conn)
  hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c
  /-- the record format holds the client's view on every admitted schedule -/
  hfit : ∀ s ∈ Ss, fmt.fits s
  /-- the network certifier's schedule at the coins -/
  sched : ((i : I) → Ω i) → WardenSchedule Conn
  /-- the server's state at the coins -/
  srv : ((i : I) → Ω i) → Srv
  /-- the records are the deployed device's run on the schedule and the server's state -/
  obs_run : ∀ ξ, obs ξ = (run device (sched ξ, srv ξ)).2
  /-- the schedule is one the period admits -/
  sched_mem : ∀ ξ, sched ξ ∈ Ss
  /-- `K` is at least `log2` of `PeriodLeakAccepted`'s count at the period's schedules, caps, partitions and faulty
  bits -/
  count_le : ∃ N : ℕ, Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
      reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞) ≤ N ∧
    (N : ℝ) ≤ 2 ^ K
  /-- the network audit's acceptance with window `i` failing its good event, over the audit's own coins, is at most
  the sampled verifier's `PrBad` at the window's statement and prover, with the registered inputs pinned at the
  registration's values: the per-window bound that the trusted network certifier's records (`trusted`) give. It is a
  field until it is derived from `PeriodLeakAccepted` at the deployed period (the follow-up). -/
  window_le : ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      r.accepts .network (Function.update ξ (r.own .network) x) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (sw i) (ρ i) tops (stmt (Function.update ξ (r.own .network) x) i)
          (wit (Function.update ξ (r.own .network) x) i) (obs (Function.update ξ (r.own .network) x) i)) ≤
    PrBad H D.V
      (RLeakTap H D.commit D.parse D.partition L Kₒ S lay k δ
        (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (rops s))) Dj tops)
      (E D decode) (stmt ξ i) (prover ξ i)

attribute [instance] RealizesCommOfPeriodLeakAccepted.tapeFintype RealizesCommOfPeriodLeakAccepted.tapeNonempty
  RealizesCommOfPeriodLeakAccepted.connFintype

/-- **The finder `B` covers every window's collisions**: at every value of the coins and every window, `B` outputs a
collision of `H` at least as often as the window's extraction finder (`Claims.Sampling.Bext`), its finder of the
collisions against the network certifier's openings (`NetTiming.Btap`) and its finder of those against the
registration's openings (`NetTiming.Breg`), each at the window's prover and statement. In the causal game `B` is the
network component's finder (`Accounting.PeriodLive.finder .network s`). -/
def RealizesCommOfPeriodLeakAccepted.FinderCovers {ι : Type} {d : Device ι} {α : Type} [Fintype α]
    {C : Verity.Circuit} {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type}
    [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] {r : PeriodRun d α C Ω} {K : ℝ}
    {H : Msg → Msg} {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {δ : ℝ} {decode : List (ℕ × Col) → List T} {reg : AuditStatement → Finset ℕ} {Dj n : ℕ}
    (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n) (B : Claims.Finder) : Prop :=
  ∀ ξ i, (Bext H D.V (E D decode) (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ B.Pr H ∧
    (Btap H D.V (E D decode) D.commit D.parse R.tops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ B.Pr H ∧
    (Breg H D.V (E D decode) D.commit D.parse R.rops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ B.Pr H

end Accounting.JointBridge
