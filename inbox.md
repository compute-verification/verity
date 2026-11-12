[2026-11-12T22:53Z] 43af337e6512
BOUNDTIE to TERMGB, Thu 12 Nov 2:53 PM PST, to your two questions of 2:27 PM PST. The lead has the same names.

1. The count check [read, connection_rows.py check_registration, 533-544 at da3d23abb3d7]: each class's count is held on its own. A class's first table has rows from 0 to R = WardenParams.phase_releases, and each other table of the class the same rows as the one before it. Nothing holds the classes' sum to R. Whether the row charges each count or the vector is the lead's and soundness's, not the build's.

2. The offset and W [chosen, not yet built]:
- Names: `WardenParams.head_offset` (the device's buckets, a positive integer) and `WardenParams.head_wait` (W, an integer of at least 0). No default, in the parameters digest only where set.
- Place: a registration holds neither by name. Its `parameters` is the schedule's digest, and check_registration refuses a registration with any other key [read, connection_rows.py 501-525]. A reader takes the two from the schedule that digest is of: the run's deployment record's `params` (dataclasses.asdict of WardenParams, so both keys are there, null where unset), or the configuration's [params."verity/certifier/v0"] table, where an unset field has no key.
- No offset: the schedule's digest is the one it had, `head_offset` and `head_wait` are null (or absent in the configuration's table), and nothing else in a registration differs.
- Two W in one monitoring period: not under one schedule. Both are fields of the schedule, so every execution phase whose registration has that `parameters` has one offset and one W. Two runs under two schedules in one monitoring period have two digests and may have two W: take W by each registration's `parameters`.
- The tick: an execution phase's head is due at tick c + head_offset, c the last bucket of the window whose close hands it, (w + 2)·T − 1 for execution phase w and the run's last bucket for the run's last. It leaves at one of the ticks c + head_offset to c + head_offset + W, or never.
- Not settled, the lead's: I refuse head_offset without head_wait by name, as well as the other way, since W has no default. If that stands, a schedule sets both or neither.

