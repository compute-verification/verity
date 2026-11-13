circuits' worker FP8READ, Fri 13 Nov 9:00 AM PST (17:00Z), fourth comment.

**The capture's launch stage, the part that calls the kernel on a card, has now run at #9567's merged head `70a27d97ec60`: 48,234,496 output words over 1,472 tokens compared, 0 differ. This closes the sentence in the pull request's body that the launch stage had not run at its head.** It is the kernel alone on random and planted rows, as on Thursday. Nothing was served.

- **Run:** `r20261113-163614-e29b`, exit 0, on node A's card 1 with cores 10-19, from exactly `70a27d97ec6086b5fcd45b3bf936efa1bd6702e1`.
- **Lease:** one, with its cores declared, held from 16:36:59Z to 16:37:11Z. The card read 0 MiB, no process and a free lease before the start and again at the run's end, 16:58:07Z.
- **Counts:** 30 cases, every launch check true. Against the reference function: 6,782,976 words, 0 differ. The other 55 pad heads against one pad head: 41,451,520 words, 0 differ. `FlashMlaV32Decode_v1`'s own evaluator: 73 tokens, 299,008 words, 0 differ.
- **Record:** ok, digest `14d0bd3a6419a5b399df5a9207adee13c8756e4a59f738810ffed19c19f594d2`, equal to the one I recomputed from the fetched record.
- **Against Thursday's capture `r20261112-181549-6fce`:** 246 of the 247 saved files are the same bytes (253,712,570 bytes). `launch.json` differs only in the two paths of its command line and in the `errors` key this head no longer writes. The 20 SASS listings are equal once unzipped.
