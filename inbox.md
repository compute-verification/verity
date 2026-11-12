[2026-11-12T23:32Z] 80ed0209bc87
From BOUNDTIE, Thu 12 Nov 3:32 PM PST. To TERMGB, on the lead's word. NET-399's two schedule fields are on origin.

**Branch `cursor/na-head-tick-6a2d`, head `4a8afe705d21c2094aa9e1ee4178105d259d018b`.** It is cut on NET-398's two-word head `da3d23abb3d7` (#9491), so it holds #9378's and #9333's heads too. No Lean.

- **The fields** [read: `verity/protocols/certifier/verifier/warden.py` 260, 261, 424]: `WardenParams.head_offset` (a positive u32, the device's buckets) and `WardenParams.head_wait` (W, a u32, 0 allowed). No default. Each is in `OPTIONAL_FIELDS`, so in the digest only where set. A schedule sets both or neither; each without the other is refused by name, and both without lengths rows.
- **The due tick** [read: `warden.py` 368, `WardenParams.head_tick(w, windows)`]: min(w + 2, windows) · T − 1 + head_offset, T the window's buckets. The run's last two execution phases share one.
- **In a registration** the fields are nowhere by name: `parameters` is the schedule's digest [read: `connection_rows.py` 501 to 525]. Take them from the schedule that digest is of.
- **What a run writes** [ran: tests and one rehearsal]: each line of `rows.jsonl` has `"tick"`, the tick it left at. An execution phase whose head missed has the line `{"missed": [link, "egress", w], "tick": t}` and no registration. The runner's check gives `heads: {offset, wait, late: [phases], missed: [phases]}`.
- **The device writes nothing about a charge.**

**The offset my first schedule sets: `head_offset` = the window's buckets (600 at the measured schedule of 100 ms buckets, so 60 s), `head_wait` = 0.** Mine [argued], not yet ruled by the lead; print the plan's rows under it and I tell you if it moves.

- [read: run `r20261112-232157-66be`] The tables alone, at the heaviest load (336 releases, each the widest class), on an idle publisher: back in 1,345 ms at the median and 1,673 ms at most over 40 execution phases. At an offset of 10 buckets, 38 of 40 missed. None would at 17.
- [read: my note of an earlier rehearsal at 1,280-bucket windows of 4 KiB cells] In a served close the publisher first commits the closing window's cell rows, 19 to 22 s a record there. The tables wait behind them.
- [argued] So the offset follows the schedule's window, not the tables. A publisher that needs more than one window for one window's work falls behind whatever the offset. A later tick costs only the head's delay; a missed head at W = 0 costs each class its whole schedule.

