import Definitions.Accounting.JointBridge.CommOfPeriodLeakAccepted

/-!
# The network certifier's per-period guarantee, on a run that realizes its period of sampled audits (trusted)

`CommOfPeriodLeakAccepted` is the network audit's `PeriodRun.Sound`, the coins model's per-period guarantee,
for a run whose communication audit is the network certifier's period of sampled audits
(`Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted`), at the error of
`NetTiming.Guarantees.PeriodLeakAccepted`: `n · (δ + ε_ks + ε_bind + ε_tap)` over the period's `n` windows.
-/

namespace Accounting.Guarantees.Bridge

open Claims Claims.Sampling NetTiming Accounting.JointBridge

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The network certifier's per-period guarantee.** On a run that realizes the network certifier's period of `n`
sampled audits at the claim `c.K`, whose deployed network certifier runs its code on its host (`trusted-host`, the
realization's `trusted`), at a deployment `D` of the sampled verifier where `Claims.Sampling.KS` holds and the
trace's commitment binds (`Commitment.Binds`), at `δ ≥ 0` and a decoder that lists every trace within Flock's radius,
where each window's prover has `κ − δ + √(γ·Pr[B]) ≤ ε_ks`, `Pr[B_ext] ≤ ε_bind` and `Pr[B_tap] ≤ ε_tap` (`ε_ks` and
`ε_bind` as in `Claims.Sampling.KS`'s errors, `B_tap` the collisions against the network certifier's openings), the
network audit's `PeriodRun.Sound` holds at `n · (δ + ε_ks + ε_bind + ε_tap)`. -/
def CommOfPeriodLeakAccepted : Prop :=
  ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
    [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] {r : PeriodRun d α C Ω} (c : Certificates) (p : Price) {H : Msg → Msg}
    {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ}
    {decode : List (ℕ × Col) → List T} {reg : AuditStatement → Finset ℕ} {Dj n : ℕ}
    (R : RealizesCommOfPeriodLeakAccepted r c.K H D L Kₒ S lay k δ decode reg Dj n) (hks : KS H D)
    (hb : D.commit.Binds H) (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode)
    (εks εbind εtap : ℝ)
    (herr : ∀ ξ i, Claims.Sampling.κ D δ (R.stmt ξ i) - δ +
      Real.sqrt (Claims.Sampling.γ D (R.stmt ξ i) * PrB H D.V (R.prover ξ i) (R.stmt ξ i)) ≤ εks)
    (hbind : ∀ ξ i, (Bext H D.V (E D decode) (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ εbind)
    (htap : ∀ ξ i, (Btap H D.V (E D decode) D.commit D.parse R.tops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤
      εtap),
    r.Sound c p .network (n * (δ + εks + εbind + εtap))

end Accounting.Guarantees.Bridge
