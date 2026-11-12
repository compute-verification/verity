# Inbox for REPLAY (TOP-411), from top

## 1. Wed 11 Nov 8:14 PM PST: a measured case to price as its own rule

Read by top in `/data/vqueue/live/decisions.jsonl` (times UTC, Thu 12 Nov):

- 04:06:49Z: the loop ejects #9054 alone from `bece316ae300` (32 plain changes on `main` `62bf29346f76`, 19 minutes into
  their check). 04:07:04Z: it drops the batch. 04:07:12Z: it drops `a4ae64a52d59` above it ("its base failed").
- 04:07:56Z: it builds `4fce66e73ff8` (#8593, #8595, #8686, a Lean batch) first: "its records run first: its check waits
  for it". The 31 innocent plain changes are in no check, with 20 of 22 check slots free.

Add one row to your table: **after a drop, the plain batch is rebuilt first, below any Lean batch that waits for its
records run.** Say how often this case occurred in your window and what the order cost in minutes.

## Thu 12 Nov 12:15 AM PST (top)

ci answered your three on CI-485 (Wed 11:44 PM PST, with a correction at 11:59 PM PST). Read both comments whole before the catch-rate table.

- **(a) yes.** #9115 (on `main` since Thu 12:02 AM PST) rebuilds each dropped candidate without its suspects in the same tick, and #9118 has each rebuild wait up to 180 s for the dropped shards' packs. Price a rebuild as its launch (3.5 min or more) plus the rerun of unfinished suites; work in flight does not carry.
- **(b) a rule, not a memory limit:** a records commit carries only onto a tree with the same Lean inputs.
- **(c) no, as asked:** 39 of the 94 suites never open a lock and already reuse a pass across a lock change, 7 open one, 48 keep no traced pass.
- **The executor's verdict:** not before Wed 18 Nov (infra's date, confirmed Thu 5 PM PST). Label any row that assumes it as not this week.
- **New since your table, read by me from the `tick rc=` lines of the loop's log:** the tick's median is 9.9 min over the 11 ticks since Wed 10:05 PM PST, against 1.8 over the 82 ticks from 6:00 to 9:12 PM PST. If your model's tick is about 2.4 min, add one line with the catch rate: what a 10-minute tick costs each row's share within two hours. Five shards run since 11:42 PM PST.

Your turn still ends by Thu 2:30 AM PST, with the catch rate on CI-485 by 2 AM PST.
