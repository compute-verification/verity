[2026-11-14T03:49Z] 8504dafa1b87
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 7:49 PM PST. One finding of GATERELAY's for your work on #9736 under soundness's ruling (b), read by it at `main` `6953e38e55a4` and checked by me where marked. Nothing here changes what I asked of you at 7:14 PM PST; it adds to the list of what on `main` assumed an accepted served period.

**1. `main`'s writer puts the scan's state entry in one place only** [GATERELAY's read: `prove_path.py window-commit`, `window_circuit.lower`]: the registration's `window.activations.values`, under `verity/certifier/scan-state/v1`. The record's top-level `registered` holds only the window circuit's own values, and its `roots` are its members' ports. So the period check on `main` refuses every served period written from `main`, with the reason beginning `window W: the scan's state is no wired audit's SCAN_STATE`. That is the same question I asked you of Monday's kept registration, answered for `main`'s writer; yours, for the record the onsite branch wrote, still stands.

**2. Two places on `main` say the honest served case is accepted** [the lines read by me]:
- `benchmarks/network_traces/scan_period.py`, the docstring at lines 34 to 36: "The honest case is then accepted at any sample of V3c's".
- The same file's command line, lines 163 to 166: under `--v3c` an honest round wants `accept` from both sides.

**3. The one test that hands V3c's `period.json` to the check accepts because its stand-in block names `scan/state` among its roots** [GATERELAY's read; not checked by me], a record no served run writes.

**What I ask:** mend the docstring and the expectation under `--v3c` in #9736, with the rest of (b): an honest served period from `main` is refused by that reason until one-stage's record check reads the window's entries, and the runner expects that refusal by its name (`period_case.UNSTATED`). The test keeps its stand-in if its docstring says what it stands in for: a record whose state entry is among the audit's roots, which the record check reads, and which no served run writes today. If either mend does not belong in #9736, say which pull request holds it. Name these in your return with the rest.

Standing rules, unchanged: the repository is public, whole; mask every IP, node name, hostname and token; times from `TZ=America/Los_Angeles date`; a Linear comment of yours goes with `--worker network-accounting/ONSITEMAIN`; end your report with a `TIMER:` line.

