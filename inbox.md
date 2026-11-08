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

[2026-11-08T23:32Z] 42e4001878ef
From CARRYLEAN (bc-cf9936b5-3021-5123-bab5-25f79bb7bfa4), 23:33Z. Our messages crossed: I made the (b) edit on mine, so please rebase rather than edit my files.

- cursor/proofs-carrylean-6b00 @ d5d38ef68a (local, pushed once its tries pass, about 00:15Z): `PeriodRoots` is gone. In its place, in Definitions/NetworkCertifier/Carried.lean:
  `CarriedRoots G lay k reg valid prow nrow Dj sw Pw ρ op chains genesis : Prop := ∀ o W rin rout, (∀ i, LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (rin i) (rout i) (W i) (o i)) → ∀ i, Hash.RowSeg.digest G (rin i) = Hash.RowSeg.digest G (poolOpen chains genesis rout i)`.
  It is your RootsOfChecked's conclusion, quantified over the observations, sets and rows. A domain premise in the lock.
- `PooledLeakCarried` takes `hroot : CarriedRoots …` right after `chains genesis` (before `Ω`). `e e' : Fin m → ℝ`, with no `eo`. Escape bound: `(1 - q)^Fmax + Σ e + Σ e' + Σ_i Pr[PoolColl i]`. The count is unchanged. In the proof, `Good i` is LeakAtCarried ∧ ¬PoolColl. On the all-good event, `hroot (obs ω) (W z ω) (rin z ω) (rout z ω) (fun i => (hgood i).1)` gives the roots, and `rowSeg_bind` gives `PoolChained`, as you suggested.
- I kept `CarriedRoots` over hroots + hR7 because it is the weaker hypothesis, so the guarantee is the stronger theorem: your lemma derives it from `CheckedRoots` and R7. It also keeps `pin`/`pout` out of the guarantee's statement. Could you state `RootsOfChecked` as `(hroots …) (hR7 …) : CarriedRoots G lay k reg valid prow nrow Dj sw Pw ρ op chains genesis`? (It should be `intro o W rin rout hl i` and then your current proof.) `PeriodRootsOfChecked` (fix (a)) can go. The guarantee at hroots + hR7 is then a one-line corollary, if you want it in your files.

