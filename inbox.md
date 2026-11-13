[2026-11-13T08:02Z] 300fb3ac25b9
WALKER, Fri 13 Nov, 12:05 AM PST. TRN-145.
**#9558 is done on its present base: head ea11a40e04 (`main` b843a10af4 merged in before your rule), suites `4 suites: 4 passed (0 cached), 0 failed` (RAN r20261113-055655-3021), body rewritten, draft. I won't merge `main` again until you say the chains are on.**

- #9676 (the cache change), head 3a94714cfb, is pushed; its suites passed too (RAN r20261113-061201-25db, 4 of 4). The body has the equality and memory table. One more finding is coming: s-t20 with the change, RAN r20261113-061108-54ec, is reading at 104.8 GiB at node 10760 of TrainStep's 16,403. It climbs about 9 GiB per layer of nodes with `words_of` gone, and tiny at 2 × 1024 tokens peaked at 33.49 GiB, so something else grows with the context. r20261113-080221-72e0 traces it on tiny(2,1024,64), and I'll send what it finds.
- I have received no timer or Slack event that isn't mine.

