module

public import Definitions.Accounting.PeriodBinding

/-!
# A period's binding per prover (trusted)

The binding premise of the period's credit (`SampledLayer.BindingAt` at `Declared.monitoringPeriodLayer`, which J10-R's
`tileLayer` is) bounded for each prover by its own fork finder, with no premise beyond `ecr/sha-512` at that finder.

A prover's period, at any circuit `Cd` of at least one round and any outcomes of its proofs `pr`, opens its positions by
frame-v3-sha512 paths (`HashedOpenings`). `L` is every position its tiles read (`Declared.monitoringPeriodPositions`), and `q` is
the SHA-512 calls of its fork finder at `L` (`HashedOpenings.finder`): its two runs from a position's cut, the
extractor's and the verifier's checks of their openings. Under `Claims.ExpectedCR` at that finder, binding fails in each
verification phase with probability at most `|L|·√(q/2^256 + κ)`, where `κ` is stated in full:
`2·Σ_{ℓ ∈ L} Pr[an accepted opening of ℓ isn't verified] + Pr[some ℓ ∈ L splits]`. The first term counts each accepted
opening of a position by a path the position's verifier doesn't accept at its root and rank, or whose row doesn't
decode to the value the proof opens (`HashedOpenings.Unverified`). The second counts a position whose verifier, root or
rank differs between a run and its continuation from the position's cut (`HashedOpenings.Split`). Neither is bounded
here: no premise yet ties an accepted proof's opening to a verified path of its position's leaf, or fixes a position's
leaf by its cut.

Each prover is charged its own finder's `q`, so the bound is per prover, as the floors per prover
(`Pouw.Guarantees.RelativeSecurity.StagedFloorKOneSidedPerProver`) are: `ecr/sha-512` is applied to that finder alone,
outside the theorem.

* `MonitoringPeriodBindingPerProver`: the bound.
* `MonitoringPeriodBindingPerProver_honest`: at `oneMonitoringPeriod`, a period whose proofs are all accepted and open every slot to
  `false` by a verified path, `κ = 0` and binding never fails.
* `MonitoringPeriodBindingPerProver_nontrivial`: at `oneMonitoringPeriod` with every proof accepted, opening every slot to the verification
  phase's coin bit by a verified path, `κ = 0`, and the bound at `q = 0`, which is 0, fails: the opened value moves with
  a coin sampled after the commitment.

Proved by `Pouw.SecurityProofs.MonitoringPeriodBindingPerProver`, `Pouw.SecurityProofs.MonitoringPeriodBindingPerProver_honest` and
`Pouw.SecurityProofs.MonitoringPeriodBindingPerProver_nontrivial`.
-/

@[expose] public section

namespace Pouw.Guarantees

open Accounting.Compose Accounting.Compose.PeriodBinding Accounting.TS Pouw.RelativeSecurity

/-- **A period's binding per prover.** For every circuit `Cd` of at least one round, outcomes `pr` of its period's
proofs and openings `hl` of their positions under the hash `H`, if the fork finder at the period's positions
`L = Cd.monitoringPeriodPositions n Np` and `q` SHA-512 calls meets `ecr/sha-512` (`Claims.ExpectedCR`), then in every
verification phase binding fails with probability at most `|L|·√(q/2^256 + κ)`,
`κ = 2·Σ_{ℓ ∈ L} Pr[Unverified at ℓ] + Pr[some ℓ ∈ L splits]`. -/
def MonitoringPeriodBindingPerProver : Prop :=
  ∀ {ι C Ξ V : Type} [Fintype C] [Fintype Ξ] [Nonempty V] (Cd : Declared ι) [Fact (0 < Cd.rounds)] {n Np Kd : ℕ}
    (hK : Kd ≤ Np) (pr : Outcomes C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np n V)
    [Nonempty (Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)]
    (hl : HashedOpenings C (Equiv.Perm (Fin Np) × Ξ) n Cd.rounds Kd ℕ (Fin n × ℕ) V) (H : Hash.Msg → Hash.Digest)
    (q : ℕ),
    Claims.ExpectedCR (fun u => (H u).bytes)
        (hl.finder (Cd.monitoringPeriodLayer Fact.out hK pr) H (Cd.monitoringPeriodPositions n Np) q) →
      (Cd.monitoringPeriodLayer Fact.out hK pr).BindingAt ((Cd.monitoringPeriodPositions n Np).card *
        √(q / 2 ^ (256 : ℝ) +
          (2 * ∑ ℓ ∈ Cd.monitoringPeriodPositions n Np, prob (fun x => hl.Unverified (Cd.monitoringPeriodLayer Fact.out hK pr) H x ℓ) +
            prob (fun p : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n ×
                Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n =>
              ∃ ℓ ∈ Cd.monitoringPeriodPositions n Np, hl.Split (Cd.monitoringPeriodLayer Fact.out hK pr) p.1 p.2 ℓ))))

/-- **`MonitoringPeriodBindingPerProver`'s honest witness**: at `oneMonitoringPeriod` with every proof accepted, opening every slot to
`false` by the verified path `oneOpenings` gives (`honestLayer`), the proofs are accepted on every coin, `κ` is 0 (no
accepted opening is unverified and no leaf splits), and binding never fails. -/
def MonitoringPeriodBindingPerProver_honest : Prop :=
  prob (fun x : OneCoins => ∀ i, honestLayer.accepted x 0 i) = 1 ∧
    2 * ∑ ℓ ∈ oneMonitoringPeriod.monitoringPeriodPositions 1 1,
        prob (fun x => (oneOpenings fun _ => false).Unverified honestLayer zeroHash x ℓ) +
      prob (fun p : OneCoins × OneCoins =>
        ∃ ℓ ∈ oneMonitoringPeriod.monitoringPeriodPositions 1 1, (oneOpenings fun _ => false).Split honestLayer p.1 p.2 ℓ) = 0 ∧
    honestLayer.BindingAt 0

/-- **`MonitoringPeriodBindingPerProver`'s nontrivial witness**: at `oneMonitoringPeriod` with every proof accepted, opening every slot to
the verification phase's coin bit by the verified path `oneOpenings` gives (`acceptAllLayer`),
`MonitoringPeriodBindingPerProver`'s bound at `q = 0` fails. There `κ` is 0, so the bound is 0, while the opened value moves
with a coin sampled after the commitment. -/
def MonitoringPeriodBindingPerProver_nontrivial : Prop :=
  ¬ acceptAllLayer.BindingAt ((oneMonitoringPeriod.monitoringPeriodPositions 1 1).card *
    √(((0 : ℕ) : ℝ) / 2 ^ (256 : ℝ) +
      (2 * ∑ ℓ ∈ oneMonitoringPeriod.monitoringPeriodPositions 1 1,
          prob (fun x => (oneOpenings coinBit).Unverified acceptAllLayer zeroHash x ℓ) +
        prob (fun p : OneCoins × OneCoins =>
          ∃ ℓ ∈ oneMonitoringPeriod.monitoringPeriodPositions 1 1, (oneOpenings coinBit).Split acceptAllLayer p.1 p.2 ℓ))))

end Pouw.Guarantees
