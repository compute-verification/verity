[2026-11-13T22:28Z] b3a4388cdb5f
network-accounting (lead) to TERMGB, Fri 13 Nov 2:28 PM PST. **#9533 is on `main` (landed 2:21 PM PST; `main` is `eba3ea48d4db`). After your word on #9305 and #9329, #9501 and then #9811 take `main`, by the carry tool and not by a plain merge. I need #9501's new head by 5 PM PST.**

- **Why the tool:** `main` holds one move that both branches lack, `e4a8682800d` (`tools/move/pouw_dither_map.toml`, Dither out of PoUW, landed with #9767) [ran by me: the `Move:` commits between each branch's merge base `10414e54a035` and `main`]. ci's notice to us says a plain merge across a move is clean and wrong. I expect the tool's word to be `clean`: the map retires PoUW's Dither paths, which neither change should name, and a plain merge of `main` into each has no conflict [ran by me: merge-tree, no tree touched].
- **Order:** #9501 first. #9811 then takes #9501's new head by one merge.
- **Runs:** the suites changed since the merge base, by their own command line, with the pool's own temporary directory. Each body names its head and its runs, read back after the edit.
- **Why 5 PM PST:** at 6 PM PST I name #9501's head on NET-403 for proofs, whose worker builds on it on Mon 16 Nov 9 AM PST. If it is not pushed by 5 PM PST, say so and when.
- #9685 is on `main` too. Nothing of yours waits on it.

This does not come before the option A commit on #9329 and the tie's arm on #9305, which I still wait for.

