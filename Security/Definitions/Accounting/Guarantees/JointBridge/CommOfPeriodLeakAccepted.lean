import Definitions.Accounting.JointBridge.CommOfPeriodLeakAccepted

/-!
# The network certifier's per-period guarantee, on a run that realizes its period of sampled audits (trusted)

`CommOfPeriodLeakAccepted` is the network audit's `PeriodRun.Sound`, the coins model's per-period guarantee,
for a run whose communication audit is the network certifier's period of sampled audits
(`Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted`), at the error of
`NetTiming.Guarantees.PeriodLeakAccepted` with its three collision terms charged to one finder: over the period's `n`
windows, `n · (δ + ε_ks + 3 · Pr[B finds a collision])`. `B` is the network component's fixed finder, built from the
strategy (in the causal game, `Accounting.PeriodLive.finder .network s`), and it exhibits every window's extraction
collision (`Claims.Sampling.Bext`), every collision against the network certifier's openings (`NetTiming.Btap`) and every
collision against the registration's (`NetTiming.Breg`), each with at least that finder's probability. So the binding failure is an event of the finder, not a number of its own:
`ecr/sha-512` is asked of `B` only, where the one theorem reads its error (`Accounting.PeriodLive.error`).

What the claim at `K` stands for is `PeriodLeakAccepted`'s count at the run's deployed period: a covert sender in the
server over the schedules the period admits, whose records the network certifier registers, has at most `2^K` secrets
whose every window meets `WindowGood`, so it moves at most `K` bits through the network certifier in the period.
-/

namespace Accounting.Guarantees.Bridge

open Claims Claims.Sampling NetTiming Accounting.JointBridge

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The network certifier's per-period guarantee.** On a run that realizes the network certifier's period of `n`
sampled audits at the claim `c.K`, whose deployed network certifier runs its code on its host (`trusted-host`, the
realization's `trusted`), at a deployment `D` of the sampled verifier where `Claims.Sampling.KS` holds and the
trace's commitment binds (`CommitmentScheme.Binds`), at `δ ≥ 0` and a decoder that lists every trace within Flock's radius,
where each window's prover has `κ − δ + √(γ·Pr[B]) ≤ ε_ks` (`Claims.Sampling.KS`'s knowledge error), and at a finder
`B` (the network component's: `PeriodLive.finder .network s` at the strategy `s`) that outputs a collision of `H` at
least as often as each window's extraction finder `B_ext` and its finders `B_tap` and `B_reg` of the collisions against
the network certifier's openings and the registration's:

* the network audit's `PeriodRun.Sound` holds at `n · (δ + ε_ks + 3 · Pr[B])`;
* a covert sender in the server over schedules the period admits (`R.Ss`), whose secrets a decoder recovers from the
  client's view, and at whose every window's statement the network certifier registers its observations at that
  window's operator (`TapOpens`), has at most `2^c.K` secrets whose every window meets `WindowGood`. -/
def CommOfPeriodLeakAccepted : Prop :=
  ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
    [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] {r : PeriodRun d α C Ω} (c : Certificates) (p : Price) {H : Msg → Msg}
    {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ}
    {decode : List (ℕ × Col) → List T} {reg : AuditStatement → Finset ℕ} {Dj n : ℕ}
    (R : RealizesCommOfPeriodLeakAccepted r c.K H D L Kₒ S lay k δ decode reg Dj n) (hks : KS H D)
    (hb : D.commit.Binds H) (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode)
    (εks : ℝ) (B : Claims.Finder)
    (herr : ∀ ξ i, Claims.Sampling.κ D δ (R.stmt ξ i) - δ +
      Real.sqrt (Claims.Sampling.γ D (R.stmt ξ i) * PrB H D.V (R.prover ξ i) (R.stmt ξ i)) ≤ εks)
    (hext : ∀ ξ i, (Bext H D.V (E D decode) (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ B.Pr H)
    (htap : ∀ ξ i, (Btap H D.V (E D decode) D.commit D.parse R.tops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤
      B.Pr H)
    (hreg : ∀ ξ i, (Breg H D.V (E D decode) D.commit D.parse R.rops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤
      B.Pr H),
    r.Sound c p .network (n * (δ + εks + 3 * B.Pr H)) ∧
      ∀ {Z : Type} (Zs : Set Z) (schedOf : Z → WardenSchedule R.Conn) (srvOf : Z → R.Srv)
        (hs : ∀ z ∈ Zs, schedOf z ∈ R.Ss) (dec : (R.Conn → List (ℕ × CellView R.Cell)) → Z)
        (hdz : ∀ z ∈ Zs, dec (R.run R.device (schedOf z, srvOf z)).1 = z)
        (op : Fin n → ℕ → List (Obs L Kₒ S) → List Bool) (x : Z → Fin n → Msg) (wit : Z → Fin n → Wit),
        (∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → TapOpens H D.commit (lay s) (k s) (op i) s (R.tops s)
            ((R.run R.device (schedOf z, srvOf z)).2 i)) →
          ∃ N : ℕ, {z | z ∈ Zs ∧ ∀ i, WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops (x z i) (wit z i)
              ((R.run R.device (schedOf z, srvOf z)).2 i)}.encard ≤ N ∧ (N : ℝ) ≤ 2 ^ c.K

end Accounting.Guarantees.Bridge
