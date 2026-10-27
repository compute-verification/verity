import Definitions.Accounting.Assumptions.Additive
import Definitions.Accounting.Assumptions.RateAudit
import Definitions.Accounting.Assumptions.Remote
import Definitions.Accounting.Physical
import Mathlib.Data.Fin.VecNotation

/-!
# Compute additivity in gates, on the actual circuit (trusted)

`Accounting.Guarantees.AccelAdditive` proves from the machine model 𝓜 (`Verity.Accel.Machine`) that instructions of
different subprotocols share one capacity per resource. Carrying that to the gates of the actual circuit `A`
(`Accounting.Placed`) takes two named assumptions that 𝓜 doesn't give:

* **The instructions pay for the gates** (`Accounting.Assumptions.PaidBy`). 𝓜 counts the resource units a device's
  instructions use and `A` counts gates, and nothing in 𝓜 relates the two: `PaidBy` stands in for that missing link.
* **No gate counts twice** (`Accounting.Assumptions.ShareNoGate`). That the works the subprotocols certify are done by
  disjoint gates of `A` is not a property of the hardware. It is a direct-sum statement, and for circuit size these fail
  in general, so it is a conjecture about the subprotocols' functions.

`GatesAdditive` states additivity in gates: works that share no gate of the node's region over `w`, `A[⋃X ∪ Y, w]`,
total at most `Σ_v Φ_v (T + β_v) + Φ_Y·T`, where device `v` is a machine `m v` at the leaves `X v` whose period meets
𝓜's rate at a backlog of `β_v` ticks (`Accounting.PeriodRate`), and `Y` is the node's other compute (its CPUs, which 𝓜
doesn't model) under `Accounting.Placed.ComputeRate`. A device's supply `Φ_v` is `Σ_r κ_r ρ_r` for any `κ` with which
its instructions pay for its gates. The pricing `π` is any, so the statement reads the same at a pricing that counts
gates (a gate's AND count, `Φ` in gates a tick) and at a priced one. `G1` (capacity of the stores and the code) doesn't
enter.

The named assumptions of the joint ledgers follow from `ShareNoGate` with this capacity:
* `RemoteComputeAdditiveOfGates`: the remote audit's `Accounting.Remote.Assumptions.ComputeAdditive`, the node's
  rebuild work beside PoUW's within `Φ_r·T`, once `Φ_r` covers the devices' supplies over their backlogs;
* `RateComputeAdditiveOfGates`: the rate audit's `Accounting.Assumptions.RateComputeAdditive`, the compute account's
  work beside the bandwidth game's within `F·T`, likewise;
* `AdditiveOfShareNoGate`: the confidence set's `Accounting.Additive`, `cert + req_μ(P) ≤ μ`, from `ShareNoGate`
  alone.

The proofs are in `Security/Proofs/Accounting/GatesAdditive.lean`.
-/

open scoped NNRat

namespace Accounting.Guarantees

open Finset Verity.Accel

/-- **GatesAdditive**: on a node whose devices `v` are machines `m v` at the leaves `X v`, each with a period's
instructions `iss v` that meet 𝓜's rate at a backlog of `β v` ticks and pay for its gates at `κ v` (`PaidBy`), and
whose other compute `Y` does at most `Φ_Y` work a tick: works `W j` that share no gate of the node's region over `w`
total at most `Σ_v Φ_v (T + β_v) + Φ_Y·T`, with `Φ_v = Σ_r κ_r ρ_r` (`supply`). -/
def GatesAdditive : Prop :=
  ∀ {L : Type*} [DecidableEq L] (A : Placed L) (π : Fin A.N → ℕ) {V ι : Type} [Fintype V] (m : V → Machine ι)
    (iss : ∀ v, List (List (m v).Instr)) (β : V → ℕ) (R : ∀ v, Finset (m v).Res) (κ : ∀ v, (m v).Res → ℚ≥0)
    (X : V → Finset L) (Y : Finset L) (ΦY : ℕ) (w : Window) {k : ℕ} (W : Fin k → ℕ),
    (∀ v, PeriodRate (iss v) (β v)) → (∀ v, Assumptions.PaidBy A (X v) π (R v) (κ v) (iss v) w) →
    A.ComputeRate {Y} π (fun _ => ΦY) → Assumptions.ShareNoGate A (univ.biUnion X ∪ Y) ↑w.ticks π W →
    ∑ j, (W j : ℚ≥0) ≤ ∑ v, supply (m v) (R v) (κ v) * ((w.len + β v : ℕ) : ℚ≥0) + ((ΦY * w.len : ℕ) : ℚ≥0)

/-- **RemoteComputeAdditiveOfGates**: the remote audit's W at the rebuild pricing (`ComputeAdditive`) holds when, on
every outcome `ω`, the node's actual circuit `A ω` and its devices' instructions `iss ω` are as in `GatesAdditive` over
a window of `T` ticks, the rebuild work and PoUW's work share no gate of the node's region, and `Φ_r` covers the
supplies: `Σ_v Φ_v (T + β_v) + Φ_Y·T ≤ Φ_r·T`. -/
def RemoteComputeAdditiveOfGates : Prop :=
  ∀ {τ : Type} {N : Remote.Node τ} {a : Remote.Audit} {Ω : Type} (r : Remote.Run N a Ω) {L : Type*} [DecidableEq L]
    (A : Ω → Placed L) (π : ∀ ω, Fin (A ω).N → ℕ) {V ι : Type} [Fintype V] (m : V → Machine ι)
    (iss : Ω → ∀ v, List (List (m v).Instr)) (β : V → ℕ) (R : ∀ v, Finset (m v).Res) (κ : ∀ v, (m v).Res → ℚ≥0)
    (X : V → Finset L) (Y : Finset L) (ΦY : ℕ) (w : Window),
    w.len = a.T → (∀ ω v, PeriodRate (iss ω v) (β v)) →
    (∀ ω v, Assumptions.PaidBy (A ω) (X v) (π ω) (R v) (κ v) (iss ω v) w) →
    (∀ ω, (A ω).ComputeRate {Y} (π ω) fun _ => ΦY) →
    (∀ ω, Assumptions.ShareNoGate (A ω) (univ.biUnion X ∪ Y) ↑w.ticks (π ω) ![r.rebuildWork ω, r.pouwWork ω]) →
    ∑ v, supply (m v) (R v) (κ v) * ((a.T + β v : ℕ) : ℚ≥0) + ((ΦY * a.T : ℕ) : ℚ≥0) ≤
      ((N.rebuild * a.T : ℕ) : ℚ≥0) →
    Remote.Assumptions.ComputeAdditive r

/-- **RateComputeAdditiveOfGates**: the rate audit's W (`RateComputeAdditive`) holds when, at every value `ω` of the
audits' coins, the device's actual circuit `A ω` and its instructions `iss ω` are as in `GatesAdditive` over the
compute window, the compute account's work and the bandwidth game's compute share no gate of the region, and the
device's priced work covers the supplies: `Σ_v Φ_v (T + β_v) + Φ_Y·T ≤ F·T`. -/
def RateComputeAdditiveOfGates : Prop :=
  ∀ (p : Price) (cTag : ℕ) (cost : ℕ → ℕ) {N b h ρ s : ℕ} {ι : Type} [Fintype ι] {d : Device ι} {α ΩW ΩS : Type}
    [Fintype α] {r : Run d α ΩW (RateCoins N b h ρ s) ΩS} (real : RealizesRatePartial r) {L : Type*}
    [DecidableEq L] (A : ΩW × RateCoins N b h ρ s × ΩS → Placed L) (π : ∀ ω, Fin (A ω).N → ℕ) {V ι' : Type}
    [Fintype V] (m : V → Machine ι') (iss : ΩW × RateCoins N b h ρ s × ΩS → ∀ v, List (List (m v).Instr))
    (β : V → ℕ) (R : ∀ v, Finset (m v).Res) (κ : ∀ v, (m v).Res → ℚ≥0) (X : V → Finset L) (Y : Finset L) (ΦY : ℕ),
    (∀ ω v, PeriodRate (iss ω v) (β v)) →
    (∀ ω v, Assumptions.PaidBy (A ω) (X v) (π ω) (R v) (κ v) (iss ω v) (r.win .compute)) →
    (∀ ω, (A ω).ComputeRate {Y} (π ω) fun _ => ΦY) →
    (∀ x y s', Assumptions.ShareNoGate (A (x, y, s')) (univ.biUnion X ∪ Y) ↑(r.win .compute).ticks (π (x, y, s'))
      ![r.work p (x, y, s'), (real.dev (x, s')).compute cTag cost y.1 y.2.1 y.2.2.1]) →
    ∑ v, supply (m v) (R v) (κ v) * (((r.win .compute).len + β v : ℕ) : ℚ≥0) +
        ((ΦY * (r.win .compute).len : ℕ) : ℚ≥0) ≤ ((d.priced p * (r.win .compute).len : ℕ) : ℚ≥0) →
    Assumptions.RateComputeAdditive p cTag cost real

/-- **AdditiveOfShareNoGate**: `μ` adds over `𝓒` for `P` above `cert` (`Additive`) when `μ` is at least the work of
`A[X, I]` on every run of `𝓒`, and on each that implements `P` the certified work `cert` and some work of at least
`req_μ(P)` share no gate of `A[X, I]`. -/
def AdditiveOfShareNoGate : Prop :=
  ∀ {L : Type*} (𝓒 : Set (Actual.Run L)) (μ : Actual.Run L → ℕ) (cert : ℕ) (r : ℕ∞) (impl : Actual.Run L → Prop)
    (X : Finset L) (I : Set ℕ) (π : ∀ R : Actual.Run L, Fin R.A.N → ℕ),
    (∀ R ∈ 𝓒, R.A.work (π R) (R.A.region X I) ≤ μ R) →
    (∀ R ∈ 𝓒, impl R → ∃ n : ℕ, r ≤ n ∧ Assumptions.ShareNoGate R.A X I (π R) ![cert, n]) →
    Additive 𝓒 μ cert r impl

end Accounting.Guarantees
