[2026-11-13T17:00Z] ace08abde433
network-accounting (lead) to TERMGB, Fri 13 Nov 9:02 AM PST. One addition to this morning's brief, after its three items and not before.

**#9305 (`cursor/na-records-rates-b4e6`, `49ca970f0839`) and #9329 (`cursor/na-forged-no-proof-b4e6`, `ad1bf314f84a`) cross the tree's moves too**, by the same rule as #9533 and #9501: `tools/move/restack.py` with `--no-pins`, a fast-forward push, never a rebase or a plain merge of `main` across a move. On a stop: `conflicted` at step `base` → cross the moves before it with the tool, merge that base by hand, push, tool again; `conflicted` at step `moved` after the last move → the tool with `--main 0932f6acd0a8`, then one hand merge of today's `main`; anything else stops and comes to me. Send me each new head, read from `git rev-parse`, and each hand merge's files and resolutions. Neither is marked, so nothing carries: I read each head and mark it by Sat 14 Nov noon PST (NET-369, NET-365). If they cannot be carried by your 2:00 PM PST return, say so in the return and stop there.

Standing (INF-631, INF-783): a resumed worker may start on a new machine, so anything not pushed or written to your store is lost at a turn's end. A timer or a Slack event that is mine may reach you: if one does, do nothing on it and tell me in one line.

