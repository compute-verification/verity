[2026-11-15T00:33Z] 7564e9f3cf40
CROSSTICK2 to network-accounting (lead): the network suite's 8 from ci's shard 3/3. Pushed; collect head d3fec0c4eb2.
(a) cured by my commits, 5: the span test's count_agrees_with_its_mirror, registered_hardware_refuses_the_parameters, every_change_reaches_its_verdict_in_the_mirror and registered_hardware_in_the_parameters (the old Site form, 7e4a21b2fcc4), and each_rejection_has_its_own_reason (the forced record, e5eca8eb4e33).
(b) ours, mended in c48fbfb9947 (test only), 2: rows_protocol_agrees_with_the_reference and rows_protocol_on_circuit_files[nci]. The inference-only rename 5751da8f37e on collect made the rows reasons "inference_only: ...", and the test still expected the verdict name "nci". The record's failure rule stays "nci".
(c) another lane's change: none.
(d) waits on the records run, 1: record_difftest's test_the_data_is_what_the_audit_pins. The inference-only lock's runs entry pins the old digests of cases.jsonl and python.jsonl. 67b0389dfda set N to 708 and left the digests to the records run (lake plus check.py --update). I did not edit the lock.
[ran] on the collect tip after merging: the whole network suite by file, 166 passed and 1 failed (the digest pin); test_wipe_count_ties, 8 passed; the composition span tests, 4 passed and 1 skipped (the opt-in nightly at 200 cases). I fetched the Flock prover fixtures locally first.
Merges: two plain merges of collect, with no move commits and nothing resolved by hand. The second crossed five "Rename by" commits; none touches my file and none of their maps names a word on my lines. py_compile is clean. Statements changed: none.
TIMER: none

