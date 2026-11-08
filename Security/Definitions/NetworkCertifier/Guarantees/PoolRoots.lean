module

public import Definitions.NetworkCertifier.PoolRoots

/-!
# The pool's roots, from each run's in-circuit checks and the record rule

Statements only. Over `m` runs that carry the pool, each run's check holding its registered roots (`CheckedRoots`) and
the registered roots chaining under the record rule (`PoolChained` over the digests):
* `RootsOfChecked`: at rows where every run's leak relation holds, each run opens a row with the digest of the row
  `poolOpen` names. This holds pointwise, with no probability;
* `PeriodRootsOfChecked`: `PeriodRoots` holds at each run's extracted rows, at `eo i = e' i`, plus `e'` of the run
  before where run `i` carries the pool, when acceptance is prefix-closed, a period ending at its first failed run. Its
  own run's acceptance alone can't give it: after a failed run the record rule restarts the chain from the empty pool,
  while `poolOpen` names the failed run's row.
-/

@[expose] public section

namespace NetTiming.Guarantees

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

open Claims Claims.Sampling

/-- **The opened rows' digests chain where the runs' leak relations hold** (`RootsOfChecked`). Take `m` runs, run `i`
registered as `sw i` with values `ρ i` at its registered inputs, whose checks hold their registered roots `pin i` and
`pout i` (`CheckedRoots`), and whose registered roots chain under the record rule's restarts `chains`, from the empty
pool's root, the digest of `genesis i` (`PoolChained`). At rows `rin`, `rout` where every run's leak relation holds
(`LeakAtCarried`), each run opens a row with the digest of the row `poolOpen` names. -/
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
    (W : Fin m → Finset ℕ) (o : Fin m → List (Obs L K S)) (rin rout : Fin m → Hash.Row),
    (∀ i, LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (rin i) (rout i) (W i)
      (o i)) →
    ∀ i, Hash.RowSeg.digest G (rin i) = Hash.RowSeg.digest G (poolOpen chains genesis rout i)

/-- **`PeriodRoots` from each run's in-circuit checks and the record rule** (`PeriodRootsOfChecked`). With `m` runs as in
`RootsOfChecked`, a run's acceptance implying the run before it accepted (a period ends at its first failed run), and the
leak relation at each run's extracted rows failing on its accepted audit with probability at most `e' i`
(`PeriodLeaksCarried`), each run opens the root of the row `poolOpen` names (`PeriodRoots`) at `eo i = e' i`, plus
`e'` of the run before where run `i` carries the pool. -/
def PeriodRootsOfChecked : Prop :=
  ∀ {L K S : ℕ} (G : Hash.Msg → Hash.Digest) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (prow nrow : AuditStatement → List ℕ) (Dj : ℕ) {m : ℕ} (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    (chains : Fin m → Bool) (genesis : Fin m → Hash.Row) (pin pout : Fin m → Hash.Digest)
    (hroots : ∀ i, CheckedRoots G (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
      (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) (valid (sw i)) (prow (sw i)) (nrow (sw i)) (pin i)
      (pout i))
    (hR7 : PoolChained chains (fun i => Hash.RowSeg.digest G (genesis i)) pin pout)
    {Ω Z : Type} [Fintype Ω] (Zs : Set Z) (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop)
    (W : Z → (Fin m → Ω) → Fin m → Finset ℕ) (rin rout : Z → (Fin m → Ω) → Fin m → Hash.Row)
    (obs : Z → (Fin m → Ω) → Fin m → List (Obs L K S)) (e' : Fin m → ℝ)
    (hpre : ∀ z ω (i : Fin m), 0 < i.val → acc z i ω (ω i) → acc z (prevRun i) ω (ω (prevRun i)))
    (hleak : PeriodLeaksCarried lay k reg valid prow nrow Dj sw Pw ρ op Zs acc W rin rout obs e'),
    PeriodRoots G Zs acc chains genesis rin rout
      (fun i => e' i + if 0 < i.val ∧ chains i = true then e' (prevRun i) else 0)

end NetTiming.Guarantees
