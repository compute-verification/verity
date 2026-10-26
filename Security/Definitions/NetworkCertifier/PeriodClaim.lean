import Definitions.NetworkCertifier.Period
import Definitions.NetworkCertifier.Warden

/-!
# The network certifier's claim over a monitoring period of sampled audits

Definitions only. `Accounting.PeriodRun.commWithin K` is the network certifier's claim at `K`: what the network
certifiers saw the node move in the period is within `K` bits. `NetworkPeriod` is the monitoring period that
`NetTiming.Guarantees.PeriodLeakAccepted` is about, its windows' registrations and the network certifier's schedules,
with each window's statement, the audit's extracted witness and the network certifier's record at the coins.
`NetworkPeriod.commWithin` defines the claim from it: every window meets its good event (`WindowGood`), and `K` bits
name every secret whose every window does, which `PeriodLeakAccepted` counts at most `NetworkPeriod.count` of:
`count ≤ 2^b` for a whole number of bits `b ≤ K`.
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **A monitoring period of `n` sampled audits through the network certifier**, at the deployment `D` of the sampled
verifier, the public layouts `lay`, horizons `k`, registered inputs `reg` and jitter bound `Dj`, at `δ`, over the
connections `Conn`, at the coins `X`, as `PeriodLeakAccepted` takes it:

* each window's registration (`sw`), its partition (`Pw`), the values it commits at its registered inputs (`ρ`), and
  `β` bounding the committed bits of each of its possible faulty sets;
* the network certifier's schedules (`Ss`), each giving connection `c` at most `cap c` cells, and its openings at each
  statement (`tops`);
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
  /-- each connection's cap on the cells a schedule gives it -/
  cap : Conn → ℕ
  /-- the network certifier's schedules -/
  Ss : Set (WardenSchedule Conn)
  /-- every schedule keeps to the caps -/
  hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c
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

end NetworkPeriod

end NetTiming
