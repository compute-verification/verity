[2026-11-11T02:37Z] 7fb617ca4e32
Coordinator to TIMEPAIR, NET-296, Tue 10 Nov 6:37 PM PST. Read: the first burst case under the hold reads 0.018, the plan runs as fixed, the row comes with all four. Two things, neither changes a row:
- I asked infra on INF-664 whether cpu-10's host takes cores back from a guest that polls (your steal ticks) or the hold unit does anything on a timer, and asked it to read nothing on either node before 9:28 PM PST. It may offer to leave the hold on cpu-10 a little past 9:30 PM PST for one look. **Tell me by 8:55 PM PST, with block 1's row, whether you want that and what you would read in it** (only what your plan lets you add after the pre-registered rows are in; nothing that replaces one).
- For each remaining case, keep what you kept for this one: the pacer's shift by segment with its times, steal and `POLL` before and after. If the shift is already there at a case's start, say so in the row.

