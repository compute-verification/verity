module

public import Definitions.NetworkCertifier.PoolRoots

/-!
# The pool's roots, from each run's in-circuit checks and the record rule

Statements only. Over `m` runs that carry the pool, each run's check holds its registered roots (`CheckedRoots`), and
the registered roots chain under the record rule (`PoolChained` over the digests). `RootsOfChecked` then says: at rows
where every run's leak relation holds, each run opens a row with the digest of the row `poolOpen` names, with no
probability.
-/

@[expose] public section

namespace NetTiming.Guarantees

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

open Claims Claims.Sampling

/-- **The opened rows' digests chain where the runs' leak relations hold** (`RootsOfChecked`). Take `m` runs: run `i` is
registered as `sw i`, with values `ρ i` at its registered inputs. Its check holds its registered roots `pin i` and
`pout i` (`CheckedRoots`), and the registered roots chain under the record rule's restarts `chains`, from the empty
pool's root, the digest of `genesis i` (`PoolChained`). Then for any observations `o`, incorrect sets `W` and rows
`rin`, `rout` at which every run's leak relation holds (`LeakAtCarried`), each run opens a row with the digest of the row
`poolOpen` names. -/
def RootsOfChecked : Prop :=
  ∀ {L K S : ℕ} (G : Hash.Msg → Hash.Digest) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (prow nrow : AuditStatement → List ℕ) (Dj : ℕ) {m : ℕ} (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    (chains : Fin m → Bool) (genesis : Fin m → Hash.Row) (pin pout : Fin m → Hash.Digest)
    (hroots : ∀ i, CheckedRoots G (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
      (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) (valid (sw i)) (prow (sw i)) (nrow (sw i)) (pin i)
      (pout i))
    (hR7 : PoolChained chains (fun i => Hash.RowSeg.digest G (genesis i)) pin pout)
    (o : Fin m → List (Obs L K S)) (W : Fin m → Finset ℕ) (rin rout : Fin m → Hash.Row),
    (∀ i, LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (rin i) (rout i) (W i)
      (o i)) →
    ∀ i, Hash.RowSeg.digest G (rin i) = Hash.RowSeg.digest G (poolOpen chains genesis rout i)

end NetTiming.Guarantees
