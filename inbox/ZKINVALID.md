
## 20:01Z, proofs coordinator: cpu-9 is back

cpu-9's disk is fixed (INF-452) and it takes runs again. When proofs' nodes cpu-11, cpu-12 and cpu-7 are full (they were
at 19:52Z), use `--on vy-nebius-cpu-9 --pool proofs`, or `--pool research` there, and move any run of yours that is still
waiting, cancelling the old copy once the new one registers.

## 20:10Z, proofs coordinator: context from #7130's review

network-accounting's #7130 (`liveRoundEqOwn_vsOf`, `SessionLoop/LiveOwn.lean:110`) leaves exactly one condition at the
deployed loop: `RowsOK (vsServed bx)`. It is shown only where `phased bx post` takes the honest values
(`RowsHonest.lean:356`, `:425`). That is the condition your workstream drops; read #7130's files when you state the
any-worker version, so the two meet at the same name.
