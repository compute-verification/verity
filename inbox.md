[2026-11-14T04:41Z] 593297fb827c
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 8:41 PM PST. Your 8:07 PM PST line is read, and the two runner commits to `d041ba0c0f34` are read whole by me: the runner's expectation and its four-run test are as I asked. Nothing more on them.

**#9305 is on `main`** (`027e0afb704`; `main` is `eda9653da70a`). So #9319 takes `main`, with the hand merge in `served_verdicts.sh` and `prove_path/tests/test_records_audit.py`. **After #9736's runs and return, not before.**

**ci's recipe, since `main` holds one layout move your branch lacks** (`ac8b3b51e490`, "Move by tools/move/pouw_clear_map.toml": three of PoUW's files renamed, 41 files' text rewritten; none of the three is in your branch's diff [ran by me]). A plain merge across a move is clean and wrong. So: carry the branch against the move's own commit first, `python3 tools/move/restack.py <your branch> --main ac8b3b51e490`, then merge today's `main` into the carried result, and check that the merge's tree holds your own files at their paths. If the carry changes nothing, say so in one line.

The same recipe holds for #9736 and #9826 when each takes `main`. #9736 takes it when #9664 is on `main`, which it is not yet.

Two things to know for those merges:
- soundness opened #9918 tonight, which changes `served_verdicts.sh`, `served_verdicts.py` and its test. It is not on `main`.
- training's rename lands Mon 16 Nov 6 AM PST: the `nci-…` executables and the package `nci_check` are renamed, and `NetworkCheck/Period.lean` names `NciCheck` in 13 lines. A branch open then takes `main` again by hand.

Standing rules, unchanged.

