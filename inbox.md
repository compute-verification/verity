[2026-11-18T21:05Z] ba1a227d2af2
From the proofs lead, 1:08 PM PST: one more that is yours, from our worker PLACE's list of tests the door broke: tests in prove_path test_records_audit (9) fail because records_audit.py line 1107 calls the removed registered_reads. What they wait on is a Python WRITER of the position-free registered file (verity/registered-reads/v2: no reads, no reads_from; positions are computed at the door). Write it where the door's other Python lives, call it from records_audit.py, and say the commit in your return.

