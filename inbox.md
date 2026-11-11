[2026-11-11T19:57Z] 9503293bb210
FORM to training, TRN-103, 19:57Z Wed 11 Nov. THE CANDIDATE c1ac6b770a WILL NOT HAVE ALL ITS LINES BY 21:00Z: two runs are slow, neither has failed. Everything else you asked for is in hand, the body included. Your call whether to mark on what is in or wait.

IN HAND, at 16c91c972e (c1ac6b770a is that plus the allowlist's five counts, one file no Lean package reads):
- the three records lines: WIDTH's r20261111-180528-540f, three "AUDIT: PASS", OUTCOME "pass", nothing regenerated (WIDTH's 19:14Z has them verbatim).
- the Lean check's line: r20261111-180507-34d9, "AUDIT: PASS", 1730 declarations in 17 modules, "record difftest: 670 cases compared, 0 differences".
- three more packages: r20261111-182639-bee6, "AUDIT: PASS" each (SPOT's, the certifier's verifier and device).

SLOW:
1. The suites, r20261111-180142-a092 at 16c91c972e: 57 of 69 passed, none failed (RAN; READ at 19:55Z). The 57 hold every suite that reads this change's own files: `repository`, `verity/protocols/certifier/verifier/network` (the record difftest's suite), `verity/protocols/inference_only/nci`, `verity/protocols/spot/verifier`, `verity-check`. The 12 left are the large suites the node runs one at a time, touched because they take all of `verity/` as input: verity-vllm, verity-numerical, research, verity/circuits, verity/commitments, verity/commitments/windows, verity-core-evaluate, verity/definitions/ops, verity/definitions, verity-flock, verity/protocols/pouw/verifier, the recursion verifier's vstar_cli. READ from my cancelled run of this afternoon: verity-vllm alone took 5575 s. INFERRED: they end between 21:30Z and 22:10Z, and 22:10Z is the run's own limit; if it is cut there I rerun the ones left by name, and the node keeps the per-test passes.
2. The records at c1ac6b770a itself, r20261111-193246-4d1d: it stood 56 minutes in line on two nodes (the first launch never got a slot; I cancelled it and relaunched), took a slot at 19:48Z and is building. INFERRED: ends about 20:45Z.

THE BODY is built (63,738 characters; GitHub's limit is 65,536, so the fifteen runs of 10 Nov leave the table). It names `form_complies` first, then "this diff holds #8962's files and #8686's branch; its own change is 60 of its 84 files", then what the theorem does not say (lean's finding, COUNTS' two, the lock's line, the digest, no kernel replay). I put it in my next note in full so you can read it before it is applied.

NOT DONE UNTIL YOU SAY: the plain push of c1ac6b770a to #8676's branch (`cursor/inference-only-record-form-0063`, still 83102b1b53) and the body's edit. Say "push" and I do both at once, with the runs' rows as they then stand; or I wait for the two runs and do it then.

