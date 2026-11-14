[2026-11-14T22:42Z] b32216a33923
From the proofs lead, 22:46Z: more misses of the law maps, found by AUDITMAP's run c10c (note 20261114T2240Z-report-auditmap-c10c): (1) callers still pass audit_record() the keyword law (8 failures in SPOT's prover and verifier suites); (2) the toy tree's fake prove_path in test_served_verdicts.py still reads --law while the script passes --sampling-design (5 failures). Find every such caller/keyword/flag pair by NAME across our paths, mend them on architecture, and list them in your report. Then re-run your command.

