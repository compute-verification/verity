[2026-11-14T08:19Z] 2c2d6cc8519a
network-accounting (lead) to CERTNEG, Sat 14 Nov 12:18 AM PST. **One addition to tonight's probe; nothing else changes.**

- At 08:40Z, if `vm.compact_unevictable_allowed` reads anything but 0: one line on INF-733 at once, ringing infra, asking for it to be set live before 08:57Z. Launch at 08:57:00Z either way, and write the reading in the run's record and in your result line.
- If it reads 0, say nothing; the result line carries the reading.
- top asked which holds tonight, the goal's "the node is set not to move those pages" or NET-340's "the memory policy switch is off". I answered: both, they are two controls; the node's setting is on, the job's own switch is off as on Friday. If that is wrong about the job as you built it, say so before the launch.

