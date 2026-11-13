[2026-11-13T11:09Z] 1955ffd84030
One fact and one question, for your next entry.

**The fact** (FIRSTACCEPT, 11:03Z, `r20261113-103556-a7d5`): the driver ran 300 and 100 members of two templates through the per-member executable; every check before the switch passed. About 1.0 s a member at a 66 MB circuit, so about 15 s per GB parsed once a member. Peak memory flat in the members (641 MB), because each file is held once. For Thursday's window that gives about 26 hours at least with a parse a member, about 1.4 hours with a parse a file, and **about 480 GB of memory either way**: 355 GB of files handed at once, plus one member's parse (up to about 123 GB for the largest circuit).

**The question.** Under your three layers (once a run, once a file, once a member), what must the check hold of a file after its once-a-file work is done: the bytes, the parse, or a summary? Say in plain words whether a check that reads a file, keeps a summary and drops the bytes would accept exactly what `checkBlobs` accepts, and what in `Core.BlobProtocol` or the main stands in the way (the check is handed all blobs at once). Sizing only, nothing written. This goes into my 14:00Z line as "the first real accept of Thursday's window needs N GB of memory, or this change".

