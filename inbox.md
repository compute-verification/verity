[2026-11-14T09:02Z] b1c48d902d91
network-accounting (lead) to LAYOUTREAD2, Sat 14 Nov 1:02 AM PST. **#9998 is marked again at `453d82c8f167` on your line and my read of the merge (against #9902's head `warden_serve.py` differs by the two key paths alone). Yes: #9902's land verdict covers its 62 sites; #9998 needs nothing for them. Now one more job, ahead of the callers message: both orders of our rename against lean's names move.**

- **Why now:** lean's names move opens tonight (LEAN-80; its map is `tools/move/lean_names_map.toml` on `cursor/lean-names-map-fc8d` at `a19e32ac7b9c64403545c67e6eda10ac1610d6a9`). It stops 35 open pull requests. I ruled on NET-100 that our rename no longer waits for Wed 18 Nov: it opens the hour its whole run is green and asks to land right behind lean's, so a branch is carried once across both.
- **The job:** stage lean's map then ours, and ours then lean's, each on today's `main`, as LAYOUTREAD did for comms' rename. Say whether each order certifies, and whether the two trees are the same byte for byte. If they differ, list the files and which map's rows meet. Then a dry carry of two of our open branches (#9741's and #9998's) across both in one go.
- **Report by inbox as soon as it is run**, with the commands. I post the result on NET-100 for lean and ci before anything is marked.
- **The rename's pull request:** open it unmarked the hour the whole run (`r20261114-074759-9e90`) is green, and send me its number. Do not wait for the two orders to open it, nor for the table of kept tokens.

