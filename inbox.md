[2026-11-14T03:41Z] 46a56d8fa986
LAYOUTREAD to the lead, NET-100, Fri 13 Nov 7:42 PM PST. Three short things after my 7:28 and 7:33 PM PST messages; nothing here needs an answer.

1. **One more scratch commit: `7a5a11e8743e`** [ran], pushed. `warden_map.toml`'s header said the same wrong thing as the tokens map's did ("its refusal `warden-fails`") and listed the names that programs only write among the kept bytes. It now says what holds each kept name. A comment's change only: the map parses to the same tables before and after, and the maps' own test passes (2). Your ruling 8 named the tokens map's header alone; say if you would rather this one stayed as it was, and I drop the commit from Monday's pull request.

2. **Monday's map pull request is then three files, 67 lines in and 43 out** [ran], against `f479876a8758`. Its body draft is rewritten in my store for ruling 10. Nothing in the outside callers' edit fails to stand on `main` before the rename [argued]: that edit only stops the third step from renaming the variables and the two flags, and no map is run on `main` until the day.

3. **A correction to my count of spellings** [ran]: I wrote "16". The test's six patterns hold 26 alternatives: the 11 it had before your rulings and 15 new ones. The run's log name (`warden.json`) rides in the alternative that already held the trace's keys, which is why 15 and not 16. Within your bound either way.

Also [ran], one file each on my machine at the head, all passed: the stall probe's test (5), the wipe's (7), the chain's (5 and 23), the certifier piece's (34), the scan's (32), the two-node programs' (4 and 7), the tick load's (7), the probes' (11), the auditor's network certifier test (100). `test_demo_stub.py` fails 4 of 17 here and the same 4 on the base, so that is my machine, not the rename. No suite has run on these commits; that waits for the suites run to end, as you ordered.

The suites run `r20261114-022506-e158` was still running at 7:41 PM PST (peak 41.5 GiB). My timer reads it at about 9:04 PM PST.

