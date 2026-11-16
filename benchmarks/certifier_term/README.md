# certifier_term

`verity bench certifier_term` prints the network attester's term for GLM-5.3: the bits its accepted transcripts can
carry over a monitoring period, at the load of a node, the census's reference datacenter and the world, in bits and
GB, beside a target and one copy of the weights. Each row counts all its scope serves by its check's own rule, as
the monitoring-period check charges it (`verity/protocols/resource_accounting/network/verifier/lean/NetworkCheck/`,
`rowsTerms` and `terms`). Its code is the calculator of K, `benchmarks/network_traces/leak_bounds.py`, whose docstring
has the rows, the terms, the tables and where the scale comes from; `run.py`'s has the result `--out` writes; and
`verity/protocols/resource_accounting/network/prover/stop_verdicts.py` and the verifier's `class_charge.charge` have how
the stop class's verdict records are read.

```text
verity bench certifier_term                              the inputs, the scopes, and the terms in GB and in bits
verity bench certifier_term --json                       the same as one document: value, unit, commit, run, inputs, rows
verity bench certifier_term --scope world --period year  only those rows (each flag repeats)
verity bench certifier_term --requests census            the census's request in place of the sampling plan's
verity bench certifier_term --out DIR                    also its result, for DIR: the amounts of the term of record's row
verity bench certifier_term --out DIR --registration F   with its capacity and verified, read from the registration log F
verity bench certifier_term --response-ids 1024 --out DIR   the command of record
verity bench certifier_term --scan-delta D               each execution phase's scan at δ = D, in place of the default
verity bench certifier_term --help                       every parameter and its default
```

It reads files and computes, so it needs no GPU and no run. Under `research run`, give the document its run and
commit: `--run "$RESEARCH_RUN_ID" --commit "$RESEARCH_SOURCE_SHA"`. Without `--commit` it names the checkout's HEAD.

The term of record is the scan's term over the world's year, charged at two release schedules (D_j = 2), in GB. The
term at D_j = 1 is what a network attester whose network card releases at a set time would be charged. No such card is
deployed, so a figure at D_j = 1, and every figure of a row whose release times are taken once, is never the term of
record and is printed only under the label "with a simulated set-time card".

## The scans' δ

A row's ε is its execution phases' scans' δ summed, a union bound, so a δ that suits one execution phase bounds
nothing over the world's year. With no `--scan-delta`, each execution phase's scan is at the δ at which the row
that pools the most execution phases, the largest scope's longest period, fails with at most 2^-20 over its
monitoring period: that error divided by the row's execution phases. The calculator finds that row and counts its
execution phases from its inputs each time, so neither the row nor the δ is typed in. It is one δ for every row.
That row's ε is then 2^-20 and every other row's is less, in proportion to its execution phases, so two rows' errors
compare. The δ is the input `scan_delta`, marked assumed. No term depends on it: only the errors move with it.

`--scan-delta D` gives another δ, as a fraction or a decimal. No error is then set (`eps_target` is null), a row's ε
is what `D` sums to, and a row whose ε is 1 or more has no bound at its period.

Tests: `uv run tools/check/suites.py benchmarks/certifier_term`; the calculator's are
`tests/benchmarks/network_traces/test_certifier_term.py`.
