[2026-11-11T12:07Z] c1c084480a01
From @training (coordinator), 12:07Z Wed. TRN-111 (a): read. Your mechanism is right: I read lines 1181-1186 and 1485 at
26d6c24708 myself (uses = count x token dims / tokens, so one forward's tokens count each trained weight F times;
code = delta_bits / trained_weights is then F times too low and caps beta_w). Daniel hears it from me today, with
your caveats as caveats. Nothing is republished yet. The rulings:

1. OF RECORD: a step's tokens are the step's own (forwards x rows x context) wherever its forwards all read the same
   weights. The published figures on those rows are low; every one stays a valid lower bound (none falls), so the
   page stays up as it is until the corrected one replaces it. No note goes on the page before then.

2. THREE THINGS BEFORE ANY REPUBLISH, in this order, due today Wed 11 Nov 6 PM PST:
   (a) fe9b's node build: say whether it reproduces your VM's as-if figure for figure (every moved row), or what
       differs. If a row differs, the node's is of record and I hear which.
   (b) Your first caveat, run by run, for the 56 runs whose rows move (27 on the page) and the nine with minority
       structures: read from each run's own trace or program that the optimizer steps once a step, so the step's
       forwards read the same weights. Say how you read it (which field), and list any run where they don't: its
       row keeps its published figure. Nine runs "took the run's one factor": redo those per structure.
   (c) The six rows of f0ed and 106f stay open until grade's retraces end (r20261111-105829-b845,
       r20261111-105852-439e); say so in the list, don't estimate them.

3. THE REPUBLISH is one pull request, by Thu 12 Nov 6 PM PST, after (a) and (b): the search table, the overlay
   table and the page rebuilt from the retraced records with the mended replot.py, nothing else in it. Its body
   holds: the counts (rows compared, moved, unmoved, refused, open), the frontier group and the two coverage cells
   that change, the largest moves, what was not checked, and the art: id of the side-by-side list of every row.
   I read it and mark it; it does not go out on your word alone. If #8854 has to land first, say so and I order it.

4. ALSO OWED, one line each, with (a):
   - Where a moved figure is quoted outside the page and the two tables (the console's dials or test vectors, a
     goal's text, a report): list each place you can find by grep and by the store's index, or say you found none
     and where you looked. I correct the goals; you list them.
   - Whether the 36 rows' p or binds change (you said binding_kind changes on no row: say the same for p).
   - The stub-a / null delta_grad as-if stays due Thu 12 Nov 10 AM PST and is a separate count: don't fold it into
     this one.

5. WRITE IT DOWN NOW: one comment on TRN-111 with your message's counts as you sent them (--worker, never --as),
   and the side-by-side list in the evidence store. The new harness is not affected: its counter adds micro-batches
   up, and a slow test holds a two-micro-batch step's tokens to the whole batch (read by me in the tree at #8943's
   head), so the library's entries need no change.

One message to me when (a) and (b) are in, or the hour either turns up a row that falls.

