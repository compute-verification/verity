import Definitions.NetworkCertifier.SampledCharge

/-!
# The sampled audit's charge: the leak bound when step units are sampled

The leak bounds on `constantRate` (`Guarantees.AdviceString`) count runs that the Program accepts exactly. A sampled
audit gives that only at `δ ≥ miss(1)`, where no unit may be faulty. At a smaller `δ` it gives `Accepted` by the
circuit's Program (`circuitProgram`) with advice also held at the faulty units' committed gates
(`SampledAuditAccepted`, `Guarantees/SampledAudit`). `EncardAccSeqsSampledLe` counts those runs: the charge adds the
faulty units' committed bits and `log2` of the number of possible faulty sets.

**Composed with `SampledAuditAccepted`.** On the statements in its scope, which include that the layout fits the program
(`Layout.Fits`), an accepted audit meets `RAcc δ` except with probability `δ + ε_ks + ε_bind` over the selection. On
`RAcc δ`, the faulty set is one of `faultySets d P δ`. Every observation sequence the trace's transfer gates carry, with
every status complete, is then `Accepted` with advice held only at that set's committed gates. So, with its inputs other
than the ingress among `As`, the run's observations are among the sequences this guarantee counts. The leak is then at most
`c + log2 #faultySets + β + n·log2 D` bits, except with that probability. That the transfer gates carry the run's
observations is the certifier's trust (`trusted-host`), and the statuses are its published record. The theorem is
`NetTiming.sampledLeak` (`Proofs.NetworkCertifier.SampledLeak`), with `SampledAuditAccepted` a hypothesis. `RAcc` bounds
no input besides the ingress, so it takes that inputs among `As` suffice (`AdviceIn`); `sampledLeak_free` proves that for
every circuit at the inputs `false` off the free inputs, with `c` the number of free inputs.

**What `As` and `c` are.** `As` is what the circuit's inputs other than the ingress may be. Its count is a premise of the
run, not an assumption. Under the decided composition the registered inputs are fixed by the registration, and the
advice is advice v1. Its charge, the coded length counted by the in-circuit encoder, is at most the declared public `c`,
and there are at most `2^c` advice values charged `c` or less (Kraft, proved for advice v1's charge:
`Guarantees.EncardChargedLe`). Advice charged per window with `BitsCount` (#3533's form, which advice v1's charge meets)
gives `c = Σ_w b w`.
-/

namespace NetTiming.Guarantees

open Claims.Sampling

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak bound under a sampled audit.** Take the circuit's Program of `C` over a layout `lay` that fits it up to
the horizon `n`, a partition `P` and a design `d`. The observation sequences of length `n`, with jitter below `D`, that
are accepted with some possible faulty set at `δ` (`AccSeqsSampled`) number at most
`2^c · #faultySets · 2^β · D^n`. Here the inputs other than the ingress range over `As`, at most `2^c` values, and each
possible faulty set's units commit at most `β` bits (`faultyBits`). That is
`c + log2 #(faulty sets) + Σ_{v faulty} committed bits + n·log2 D` bits. At `δ ≥ miss(1)` no non-empty faulty set
is possible, and the bound is `c + n·log2 D`, as when the Program is accepted exactly. -/
def EncardAccSeqsSampledLe : Prop :=
  ∀ {L K S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : Layout L S) (n : ℕ) (hfits : lay.Fits C n)
    (δ : ℝ) (As : Set (ℕ → Bool)) (c : ℕ) (hAs : As.encard ≤ ((2 ^ c : ℕ) : ℕ∞)) (β : ℕ)
    (hβ : ∀ F ∈ faultySets d P δ, faultyBits C P F ≤ β) (op : ℕ → List (Obs L K S) → List Bool) (D : ℕ),
    (AccSeqsSampled C P d lay n As δ op D).encard ≤
      ((2 ^ c * (faultySets d P δ).card * 2 ^ β * D ^ n : ℕ) : ℕ∞)

end NetTiming.Guarantees
