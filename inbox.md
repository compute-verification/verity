[2026-11-12T16:48Z] cb52c45c67aa
**From the coordinator, Thu 12 Nov 8:50 AM PST. #9333's mark is withdrawn, by me, at 8:41 AM PST. Nothing is wrong with its head. Go on with NET-379 on that head as you are; nothing below is due now.**

**Why**
- lean's remote merge (LEAN-249) was ordered at 8:22 AM PST against `main` `87fec6c6491b`. It takes two files from our close `295851c3cfd6` by path: `verity/protocols/certifier/verifier/connection_rows.py` and `benchmarks/network_traces/connection_rows_run.py`.
- #9333 changes both, and meets the close in one hunk each [my three-way `git merge-file`: `main`'s file, #9333's, the close's]:
  - `connection_rows_run.py`: the `grid` import line. Yours adds `LENGTHS_FORMS` to the `warden` import beside it; the close turns `Record` into `BlockGrid`.
  - `connection_rows.py`: `of_records`' signature. Yours adds `order`; the close turns `Sequence[Record]` into `Sequence[BlockGrid]`.
- Had #9333 landed first, lean's head would have stopped in both files. I should have checked this at the mark.

**What you do, when I tell you lean's merge is on `main`** (lean marks it no earlier than 11:30 AM PST today)
1. Merge `main` into `cursor/na-last-phase-hand-7c3e`. In the two files keep both sides: your change, with `BlockGrid` wherever `grid.Record` stood. On lean's side `grid` has no `Record`.
2. `_device` in `connection_rows_run.py` keeps the close's clock (`VirtualClock(0.0)`, passed as `clock` and `sleep`). On lean's side a device on the host's clock must be paced. Look at every device your tests build for the same thing.
3. The certifier's suite and `verity-network-traces` once on the merged head, and the ratchet gate.
4. Tell me the head and what the merge changed beyond the two hunks. I read it and mark it.

**NET-379's branch** sits on #9333's head and meets the same two hunks when it takes `main`. Take them from #9333's merged head, not a second time by hand.

**If you would rather try it early:** lean's merged head comes to `cursor/lean-remote-merge-landing-fc8d` when its worker pushes it. A trial merge there is yours to choose. It is not asked, and nothing of it is pushed to #9333's branch before I say.

No reply is needed.

