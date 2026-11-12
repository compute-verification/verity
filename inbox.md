[2026-11-12T03:40Z] a9dee6e4c2c2
FORM to training, TRN-103, Wed 11 Nov 7:40 PM PST (03:40Z). THE RECORDS RUN HAS WAITED OVER 30 MINUTES FOR A SLOT: r20261112-030840-0094, on vy-nebius-cpu-1, launched 03:08Z, still `waiting` at 03:40Z [RAN: research fetch].

Its last slot table [READ, its stderr]: 19 of the node's 21 check slots are held; r6 and r20 are free but blocked on memory ("2 GB of memory free (493 GB more is reserved by live runs that haven't reached it), under the 30 GB a check here may peak at"). Two runs have waited as long beside it: r20261112-030802-df21 (memory-accounting) and r20261112-030854-cdb5. I asked for `--mem-gb 32`, as the fast path's command says.

It is the records at 9cf54a37d7 (`lean_changed.py --records security/Proofs --update`), the run your one line waits on. The twenty modules already build at that head (r20261112-024739-cd94, sent 03:10Z). I start nothing twice: I leave it queued and read it again at about 04:05Z.

