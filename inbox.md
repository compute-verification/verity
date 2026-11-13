[2026-11-13T19:28Z] 0aeaec46e636
WALKER, Fri 13 Nov, 11:28 AM PST.

**s-t20 is in the pass, at 59.4 GiB of an 82.8 GiB stop. The reading before the pass added 43.3 GiB to the built Program at ctx 1,024, against 1.0 GiB at s(2,64,512).**

RAN r20261113-172658-0505 (30a0a4935c, ask 92 GB), untraced:

| | s-t20 | s(2,64,512) |
|---|---|---|
| build | 874.5 s, highest 57.52 GiB | 49.8 s, highest 2.96 GiB |
| resident after the build | 27.63 GiB | not read |
| resident after the build and one collection | 14.16 GiB | 1.25 GiB (the trace run) |
| reading before the pass | 5,551.0 s, highest 59.6 GiB | 105.2 s |
| resident at the pass's start | 57.5 GiB | 2.234 GiB (d990) |
| the reading's growth | 43.3 GiB | 1.0 GiB |

- RAN: the pass started 6,460 s into the run. At 6,662 s it holds 59.4 GiB, 61.1 GiB at its highest, with no product done yet.
- RAN: in those 200 s it built 851 indices of 16,384,000 entries in 93 s, 0.11 s each. They are the takes at an iteration's start, one a token position. INFERRED: 6,145 such builds an iteration, about 670 s, 1,350 s over the pass.
- INFERRED: ctx is 16 times and vocab 62.5 times s(2,64,512)'s, and the reading's growth is 43 times. One shape pair cannot split that between ctx and vocab.
- No stop so far. Next message when it ends or stops, and by 6 PM PST in any case.

