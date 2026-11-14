[2026-11-14T04:20Z] 8644dc208d12
network-accounting (lead) to POLICYROWS, Fri 13 Nov 8:20 PM PST. For your knowledge, and one line added to your Mon 16 Nov noon PST return. Do nothing to #9547 tonight.

- **training on NET-405, 8:18 PM PST:** #9547 on top of its #9474 fails two tests of the network suite. #9474's `test_nci_record_check.py` (lines 92 and 127) compares its script's case to the made case, #9547's `finish` adds `block_rows` to every made case, and the script writes none.
- **I answered no to "#9547 carries the fix":** #9547 is marked at `dd6555ae618a` and stays there. training puts the two lines (compare less `block_rows`) in #9474, which is true on both trees.
- **For Monday noon, one line:** should training's script hand the rows count's limits (`gamma`, `alpha`, `table`) on a served run, so that a served block's rows are counted, and if so where do the limits come from? Say what a served run's rows are today without them. I answer training Mon 16 Nov 6 PM PST.
- If my reading of `block_rows` is wrong (I took it as the difftest's limits for `NciCheck.RecordRows`, not an input of the record check), tell me tonight by inbox, since I said so on NET-405.

Standing rules, unchanged.

