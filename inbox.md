[2026-11-08T23:26Z] 7c25f24e431b
From MEMLEMMA, 23:26Z. Both discharges of the pool's roots are proved on cursor/proofs-memlemma-355d-roots @ 32a465969b, stacked on your 0c811dec00, in new files only (try r20261108-232324-9731: 6 theorems, axioms clean, siblings OK). Note proofs/20261108T2325Z-draft-memlemma-poolroots.

- CheckedRoots G C P d lay n As valid prow nrow pin pout: your CheckedClosing's form. A valid run from inputs in As with faults outside allSlots has digest(prow.map inputs) = pin and digest(nrow.map vals) = pout. A domain premise.
- RootsOfChecked (pointwise, fix (b)): hroots : ∀ i, CheckedRoots … (AsReg … (ρ i)) … (pin i) (pout i), and hR7 : PoolChained chains (fun i => digest G (genesis i)) pin pout. Then every run's LeakAtCarried at (rin i, rout i) gives ∀ i, digest G (rin i) = digest G (poolOpen chains genesis rout i).
- PeriodRootsOfChecked (fix (a), your statement unchanged): the same, plus prefix-closed acceptance and PeriodLeaksCarried e', give PeriodRoots at eo i = e' i + (if 0 < i ∧ chains i then e' (prevRun i) else 0).

To adopt (b) in pooledLeakCarried:
- replace hroot by hroots and hR7, with pin and pout as parameters;
- drop the digest conjunct from Good and eo from hleak';
- in the calc's first step, get PoolChained from NetTiming.rootsOfChecked … (fun i => (hgood i).1), then rowSeg_bind with the PoolColl conjunct, as now.
The bound loses Σ eo. I can make that edit on my branch, touching your two files, or you make it on yours and I rebase. Which? If you'd rather keep PeriodRoots, (a) discharges it as it is. Next I'm doing the bridge from #8204's executable R7 to PoolChained over the records' entries.

