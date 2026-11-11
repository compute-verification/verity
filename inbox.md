[2026-11-11T21:21Z] d977835efefd
training to resume, TRN-132, Wed 11 Nov 1:21 PM PST. YOUR 1:16 PM MESSAGE IS READ. THE BRANCH AND THE TEST AS YOU CHANGED IT STAND. RULING ON (3): DO NOT WAIT FOR A GPU WEEK, AND RELAUNCH NO NODE. ANSWER IT FROM THE RUNS' OWN FILES FIRST.

(3) IN THIS ORDER, STOP AT THE FIRST THAT ANSWERS:
a. THE DIRECT READING. For each of the nine runs of record, and each start they trained from: does a saved file hold the table itself (a buffer in a checkpoint's or a start's state dict)? If so, fetch by art id, compare each saved table with the row-wise table bit for bit, and report per run: equal, or the count and span of differing entries. That is a reading of the published runs, which no rerun on a like machine is. If the tables are not saved (a non-persistent buffer), say so in one line with the line of `model.py` that shows it.
b. ONLY IF (a) CANNOT ANSWER: 400 fresh processes at the shapes those nine runs' models use (their context and head sizes, from their configs), on your VM. Report it as what it is: an Intel machine with 4 threads, not the 192-thread kind of the runs. The 192-thread kind stays "not tested" in the note until infra has such a node up for another reason; I ask infra, you do not.
The finding note says which of (a) or (b) it rests on, in its first paragraph.

(2) AND THE PORT CHECK: as you propose (the scratch merge into #9003's head, never pushed, on nodes).

WHEN THE TEN AHEAD ARE ON MAIN (#8550 among them; ci's queue holds them in a candidate now): rebase the fix onto main, one commit, and tell me its head.

ONE WORD, FROM DANIEL TODAY: the training-cost ratio is "the slowdown" in everything written from now on, never "rho": notes, bodies, messages, new names. Existing names in the tree are renamed by another worker (TRN-133); change none yourself.

DATES UNCHANGED: (1), the note and the draft by Thu 12 Nov 10 AM PST; (2), (3) and the port check by Thu 12 Nov 6 PM PST.

