import Definitions.NetworkCertifier.Period
import Definitions.NetworkCertifier.Tap

/-!
# The network certifier's claim over a monitoring period of sampled audits

Definitions only. `Accounting.PeriodRun.commWithin K` is the network certifier's claim at `K`: what the network
certifiers saw the node move in the period is within `K` bits. `NetworkPeriod` is the monitoring period that
`NetTiming.Guarantees.PeriodLeakAccepted` is about, its windows' registrations and the deployed network certifier (one
device on the server's uplink, its record format and its schedules), with each window's statement, the audit's
extracted witness and the network certifier's record at the coins. `NetworkPeriod.commWithin` defines the claim from
it: every window meets its good event (`WindowGood`), and `K` bits name every secret whose every window does, which
`PeriodLeakAccepted` counts at most `NetworkPeriod.count` of: `count ≤ 2^b` for a whole number of bits `b ≤ K`.

A fleet is `m` network certifier devices, each with its own monitoring period. `NetworkPeriod.fleetCount` is the
product of their counts: a fleet's secret whose parts on the devices determine it, and whose every device's windows
meet their good events, is one of at most that many.
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **A monitoring period of `n` sampled audits through the network certifier**, at the deployment `D` of the sampled
verifier, the public layouts `lay`, horizons `k`, registered inputs `reg` and jitter bound `Dj`, at `δ`, over the
connections `Conn`, at the coins `X`, as `PeriodLeakAccepted` takes it:

* each window's registration (`sw`), its partition (`Pw`), the values it commits at its registered inputs (`ρ`), and
  `β` bounding the committed bits of each of its possible faulty sets;
* the deployed network certifier: one device (`device`) on the server's uplink, with its cells `Cell` and the
  server's state `Srv`, whose run (`run`) on a schedule and the server's state gives the cells the client receives and
  its records of the period's windows; what reaches it from the server on each connection (`arr`) and its record
  format (`fmt`);
* its schedules (`Ss`), each held by its record format and giving connection `c` at most `cap c` cells, and its
  openings at each statement (`tops`);
* at the coins, each window's statement (`stmt`), the audit's extracted witness there (`wit`) and the network
  certifier's record of the window (`obs`). -/
structure NetworkPeriod {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (D : Deploy Coin Col T) (L Kₒ S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ)
    (reg : Stmt → Finset ℕ) (Dj n : ℕ) (Conn X : Type) where
  /-- each window's registration -/
  sw : Fin n → Stmt
  /-- each window's partition -/
  Pw : Fin n → Claims.Sampling.Partition
  /-- each window's partition is its registration's -/
  hPw : ∀ i, D.partition (sw i) = some (Pw i)
  /-- each window's registered values at its registered inputs -/
  ρ : Fin n → ℕ → Bool
  /-- a bound on the committed bits of each of window `i`'s possible faulty sets -/
  β : Fin n → ℕ
  /-- `β` bounds them -/
  hβ : ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i
  /-- the cells the network certifier carries -/
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
  /-- each connection's cap on the cells a schedule gives it -/
  cap : Conn → ℕ
  /-- the network certifier's schedules -/
  Ss : Set (WardenSchedule Conn)
  /-- every schedule keeps to the caps -/
  hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c
  /-- the record format holds every schedule -/
  hfit : ∀ s ∈ Ss, fmt.fits s
  /-- the network certifier's openings at each statement, under its roots -/
  tops : Stmt → List (Opening Bool)
  /-- each window's statement at the coins -/
  stmt : X → Fin n → Msg
  /-- the audit's extracted witness at each window, at the coins -/
  wit : X → Fin n → Wit
  /-- the network certifier's record of each window, at the coins -/
  obs : X → Fin n → List (Obs L Kₒ S)

namespace NetworkPeriod

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
  {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ} {reg : Stmt → Finset ℕ}
  {Dj n : ℕ} {Conn X : Type} [Fintype Conn]

/-- The secrets of the period whose every window meets its good event number at most this, by `PeriodLeakAccepted`:
`#Ss · ∏_c (cap c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`. -/
noncomputable def count (N : NetworkPeriod D L Kₒ S lay k δ reg Dj n Conn X) : ℕ∞ :=
  N.Ss.encard * (((∏ c, (N.cap c + 1)) * ∏ i, (2 ^ (freeInputs (N.sw i).prog (lay (N.sw i)) (k (N.sw i)) \
    reg (N.sw i)).card * (faultySets (N.sw i).design (N.Pw i) δ).card * 2 ^ N.β i * Dj ^ k (N.sw i)) : ℕ) : ℕ∞)

/-- `K` bits name every secret the period's count bounds: `count ≤ 2^b` for a whole number of bits `b ≤ K`. -/
def Bits (N : NetworkPeriod D L Kₒ S lay k δ reg Dj n Conn X) (K : ℝ) : Prop :=
  ∃ b : ℕ, (b : ℝ) ≤ K ∧ N.count ≤ 2 ^ b

/-- **The network certifier's claim at `K` over the period, at the coins `x`**: `K` bits name every secret the
period's count bounds (`Bits`), and every window meets its good event at its statement, the audit's extracted witness
and the network certifier's record. -/
def commWithin (H : Msg → Msg) (N : NetworkPeriod D L Kₒ S lay k δ reg Dj n Conn X) (K : ℝ) (x : X) : Prop :=
  N.Bits K ∧ ∀ i, WindowGood H D L Kₒ S lay k δ reg Dj (N.sw i) (N.ρ i) N.tops (N.stmt x i) (N.wit x i) (N.obs x i)

/-- **A fleet's count**: over `m` network certifier devices, device `j`'s monitoring period `N j` of `n j` windows on
its connections `Conn j`, the product of the devices' counts. -/
noncomputable def fleetCount {m : ℕ} {n : Fin m → ℕ} {Conn : Fin m → Type} [∀ j, Fintype (Conn j)]
    (N : ∀ j, NetworkPeriod D L Kₒ S lay k δ reg Dj (n j) (Conn j) X) : ℕ∞ :=
  ∏ j, (N j).count

end NetworkPeriod

end NetTiming
