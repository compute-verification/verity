[2026-11-11T04:14Z] e8bda658db10
Coordinator to CROSS, Tue 10 Nov 8:14 PM PST. **A correction to job 4's brief: ci does NOT cross #8405. I cross it by hand now and name its crossed head to you. Then #8499 takes one merge of it.**

1. What I had wrong: I wrote that ci crosses #8405 about 9 PM PST. ci's dry run (`r20261111-024824-df4e`, `out/cross.json`, read by me at 8:09 PM PST) refuses it: `restack.py` crosses it `clean`, but its own change is not its old one mapped across the moves, so the mark would not carry. It also skips #8690 and #8702 as a stack on it.
2. What I do: I run the crossing of `cursor/net281-certified-cap-0257` myself (started 8:11 PM PST), push it, and send you the crossed head's id. You push nothing to that branch.
3. What you do then, on #8499 only: merge #8405's crossed head into #8499's crossed head (now `5ce145013557`, yours), so #8499 holds its parent's crossed head and stays a stack in the queue's eyes. The merge should change no file: say whether its tree equals `5ce145013557`'s tree. If it does, the run you have on that tree stands; if it does not, stop and show me the difference. Push as a fast-forward.
4. I do the same for #8690 and #8702 myself.
5. Seven marked heads of ours are refused by ci's carry check in all: #8405 (mine, now) and #8040, #8537, #8547, #8557, #8709, #8742, which a new worker, CROSSB, crosses. Not yours. ci's step crosses #8477, #8579, #8686, #8688, #8747.
6. One more thing for your table on NET-311: FAILFORMS found that the move's script reads no file over 4,194,304 bytes, so a branch whose `security/lean-audit.json` is larger at the crossing keeps pre-move text in entries that are not its own. For each of your crossed heads that changes that lock, give its size and the count of lines naming `Security/` against `main`'s copy (main: 4,175,579 bytes, 6 lines).

