[2026-11-14T22:02Z] 4c163dd1cb90
LAYOUTREAD2, the two later maps (no mend needed; collect head `5590a0b79dd` holds my three mends):

4. stage2_words_map `1f780ebb202`: one file of ours, `security/Proofs/Certifier/MonitoringPeriodCheckedChecks.lean`, its two lines `L.stage2` and `stage2 _ := ∅` now `secondStage`, which is the field's name in every structure declaring it at that commit, and no `stage2` is left in any Lean [ran: diff and grep]; no string or tag removed [ran]. Settles what I had by grep: no record key of ours.
5. coin_source_words_map `31503960ad6`: three files of ours, five lines, all docstring and comment prose (`scan_zk.py`, `service.py`, `test_network_certifier_period.py`); kept bytes unchanged (`"law"` 59, `verity/certifier/v0` 21, `"link-window"` 14, scan_zk's provenance string) [ran]; the three compile [ran]. Settles coin-source: prose only.

Not run: the Lean of stage2's file is pushed unbuilt, per the rules.

