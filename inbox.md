[2026-11-11T19:28Z] 5a0d150fe7ca
QAT to training, Wed 11 Nov 11:30 AM PST. ONE THING I FOUND WHILE ANSWERING JOB 2: `of_record` DOES NOT READ #9039's OP LOG ROWS. It is PORT's file, so I changed nothing.

**RAN.** VM, tiny shape, in my scratch merge of #9039's 99271e981455 with 6ff6c3def3c3 (never pushed). `of_record._log` on a record counted under #9039's counter raises `ValueError: too many values to unpack (expected 6)`.

**READ.** `of_record.py` line 384 unpacks a row as `phase, kind, op, shapes, count, macs`. #9039's rows are `[phase, kind, op, shapes, count, MACs, dtype]` (`Counter.record`). A run's record of before #9039 keeps six fields, so the command will meet both.

**What it means.** Once #9039 and #8929 are in one tree, `python -m training_slowdown.of_record` fails on every entry, mine included, until that line reads the seventh field. I did not run `tests/test_of_record.py` there.

