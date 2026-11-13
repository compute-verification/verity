[2026-11-13T04:56Z] a721bb725506
WALKER, Thu 12 Nov, 8:58 PM PST. TRN-145.
**Both readers' word caches are the same, entry for entry and word for word, and that cache is what holds the memory. #9578's reader removes the frames, not the memory. At s(2,64,512) both stopped themselves at 48 GB, holding 365,244,066 words.**

Your two lines (RAN r20261113-042755-347b, rc 0, the same harness, `ST20_MEM_GB`=48):
- Recursive reader (102ad1d702's `work.py`): tiny (2,16,64) peaked at 0.88 GiB, with the cache peaking at 856 entries and 2,424,159 words. s(2,64,512) peaked at 43.5 GiB and stopped at 601 s, at node 2950 of TrainStep's 2951, with 3,862 entries holding 365,244,066 words.
- Iterative reader (#9578's, at 59b2182b10): tiny peaked at 0.86 GiB with 856 entries and 2,424,159 words. s(2,64,512) peaked at 43.8 GiB and stopped at 721 s. It finished that TrainStep summary at the same 3,862 entries and 365,244,066 words (39.2 GiB at 541 s), went on into the loop's `Iteration`, and was 377 of 2951 into a second TrainStep summary with 519 entries and 120,432,395 words.
- RAN, tiny under #9578's reader with the pass: claims 22, refused none, N 6,029,312. Origins 0, nonfinite nodes 8, NaN nodes 0. The reading before the pass took 17.4 s and the pass 1.3 s, over 26,682,288 words. The recursive copy died at `min_cut` with `No module named 'rho'`: the copy sat outside the tree, and `min_cut` finds rho from its own path. That death came after the summaries, so the cache and RSS figures stand.
- READ: the cache is `words_of[id(a)] = (a, list(a.refs))` in `Reader.summary`. It is on `main` already (7fd737067e), not added by #9558 or #9578. It is one list of every word of every argument of every node in the summary, kept until the summary returns.
- INFERRED: about 100 bytes a word (365M words in about 39 GiB). That makes 8901's 180 GiB stop about 1.8 billion cached words.

The relaunch, RAN r20261113-041530-8901 (s-t20, 200 GB, stop at 180 GiB): built in 925.5 s at 57.5 GiB. Reading: at 1,879 s it was at node 649 of TrainStep's 16,403, with 503 entries and 230,466,890 words at 35.5 GiB, moving about 0.76 nodes/s.
- INFERRED: at that rate the first TrainStep summary alone takes about 5.7 h, and s(2,64,512) went on to a second summary after the first. **The relaunch is unlikely to end by Fri 9 AM PST, and more likely stops on memory first.** I'm letting it run as you said, and will send its last heartbeat if it stops.
- An intermediate shape won't help: s(2,64,512) already passes 43 GiB in 10 min on the cache alone. A reading of s-t20 needs a change to the cache, i.e. the code change you said you'd rule on. One option: keep each argument's words as its `refs` (or a range of them) instead of a materialized list, and drop an argument's entry after its last reader node. I am not making it until you rule.

