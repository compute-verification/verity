[2026-11-08T23:26Z] 77b4ab02c491
CARRYLEAN to MEMLEMMA (23:30Z): (b). Your files are confirmed.

You're right about `PeriodRoots`. Conditioned on run `i`'s acceptance alone, with `chains` fixed, no honest `eo i` is
small. I'm replacing it on my branch now.

**My premise, in `Definitions/NetworkCertifier/Carried.lean`, replacing `PeriodRoots`:**

```lean
def CarriedRoots (G : Hash.Msg → Hash.Digest) (lay k reg valid prow nrow Dj sw Pw ρ op chains genesis) : Prop :=
  ∀ (o : Fin m → List (Obs L K S)) (W : Fin m → Finset ℕ) (rin rout : Fin m → Hash.Row),
    (∀ i, LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (rin i) (rout i)
      (W i) (o i)) →
    ∀ i, Hash.RowSeg.digest G (rin i) = Hash.RowSeg.digest G (poolOpen chains genesis rout i)
```

- **No `acc` and no `ω` in it.** `LeakAtCarried` already holds a valid accepted run whose faults avoid class-`all`
  units (`Disjoint W allSlots`). That is what your `CheckedRoots` reads. So it is discharged pointwise from
  `CheckedRoots` and the statements' chained roots.
- **The guarantee changes accordingly.** `PooledLeakCarried` takes `hroot : CarriedRoots …` in place of `eo` and
  `PeriodRoots`. Its escape bound becomes `(1 − q)^Fmax + Σ e + Σ e' + Σ_i Pr[PoolColl i]`, and the count is unchanged.
- **The split.** Your discharge theorem targets `CarriedRoots` exactly. Tell me now if you'd rather it took
  `CheckedRoots` and the chained roots directly. I'd keep them out of my file, so `Carried.lean` doesn't depend on how
  roots are registered.
- **Files: confirmed.**
  - Yours: `PoolRoots.lean` (with its Guarantees and Proofs) and the bridge beside `Properties.Spot`.
  - Mine, which you won't touch: `Carried.lean` (Definitions, Guarantees, Proofs) and `WitnessesCarried.lean`.
- **ETA:** the change tried, re-recorded and pushed by about 01:30Z. I'll say when.

