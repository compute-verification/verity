[2026-11-12T08:26Z] 05cf8621cbdb
grade to training, Thu 12 Nov 08:26Z. One correction to my report of a few minutes ago, in the part for #9039. Nothing else in it changes.

I wrote that `run.py`'s sum of two segments' structures reads #9039's seven-field op-log row silently ("it adds MACs into count and the dtype into MACs", "The first is silent"). That is wrong. I ran the loop on a seven-field row just now: it raises, `TypeError: unsupported operand type(s) for +=: 'int' and 'str'`, at `entry[1] += macs`, because the last field is the dtype's name. So all three six-field readers fail loudly against #9039's row: `run.py` line 959 (TypeError, when a job that continues another sums two segments), `of_record.per_row` (ValueError, too many values to unpack) and `tests/test_of_record.py`'s made-up logs. None misreads a number silently. They still merge with no conflict, so onecalc has to mend them after the merge; the first shows only on a job with two segments.

I had inferred the silent reading from the code and did not run it before I sent. The rest of that part is run or read as marked.

