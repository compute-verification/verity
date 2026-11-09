module

public import Definitions.NetworkCertifier.Carried

/-!
# The pool's roots: each run's in-circuit checks and the record rule chain the opened rows' digests

Definitions only. A run that carries the pool checks in-circuit that the row it opens at `prow` has its registered
opening root `pin` as its `sha512/row-seg/v1` digest, and that the row it leaves at `nrow` has its registered closing root
`pout` (`CheckedRoots`): its check holds both on every run whose faults avoid the proof blocks of classes sampled `all`,
which the pool's units are, proved every run. Under the record rule (`R7-pool-chain`) run `i`'s registered `pin` is run
`i - 1`'s registered `pout` where it carries the pool, and the empty pool's root otherwise: `PoolChained` over the runs'
registered roots. Together they give each run, at rows its and the previous run's leak relations accept, an opened row
with the digest of the row `poolOpen` names, with no probability: what `CarriedRoots` asks.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling

/-- **The check holds the pool's roots**: every run over the horizon `n` from inputs in `As` that holds one set of values,
only at committed gates of proof blocks of no class sampled `all`, and is `valid`, opens at `prow` a row whose
`sha512/row-seg/v1` digest is `pin` and leaves at `nrow` one whose digest is `pout`. -/
def CheckedRoots {L S : ℕ} (G : Hash.Msg → Hash.Digest) (C : Prog) (P : Partition) (d : Design)
    (lay : BoundaryGates L S) (n : ℕ) (As : Set (ℕ → Bool)) (valid : (ℕ → Bool) → Prop) (prow nrow : List ℕ)
    (pin pout : Hash.Digest) : Prop :=
  ∀ (ins : List (List Bool)), ∀ a ∈ As, ∀ free : ℕ → Option Bool, ins.length = n →
    (∀ g b, free g = some b → ∃ v ∉ allSlots d P, FaultyGate C P {v} g) →
    valid (valOver C (lay.inputOf n ins a) free) →
    Hash.RowSeg.digest G (prow.map (lay.inputOf n ins a)) = pin ∧
      Hash.RowSeg.digest G (nrow.map (valOver C (lay.inputOf n ins a) free)) = pout

end NetTiming
