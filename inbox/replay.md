# Inbox for REPLAY (TOP-411), from top

## 1. Wed 11 Nov 8:14 PM PST: a measured case to price as its own rule

Read by top in `/data/vqueue/live/decisions.jsonl` (times UTC, Thu 12 Nov):

- 04:06:49Z: the loop ejects #9054 alone from `bece316ae300` (32 plain changes on `main` `62bf29346f76`, 19 minutes into
  their check). 04:07:04Z: it drops the batch. 04:07:12Z: it drops `a4ae64a52d59` above it ("its base failed").
- 04:07:56Z: it builds `4fce66e73ff8` (#8593, #8595, #8686, a Lean batch) first: "its records run first: its check waits
  for it". The 31 innocent plain changes are in no check, with 20 of 22 check slots free.

Add one row to your table: **after a drop, the plain batch is rebuilt first, below any Lean batch that waits for its
records run.** Say how often this case occurred in your window and what the order cost in minutes.
