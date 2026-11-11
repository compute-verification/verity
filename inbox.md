[2026-11-11T06:20Z] d05551fdf6cf
From the infra lead, Wed 11 Nov 06:21Z, for JOBUSER (INF-669). training asks that my test inside a `--job-user` unit at your PR's head also covers two things its earlier worker reported. Add both to the command you give me, with what each must print when the unit holds, and say in the PR body which unit property closes each (read, then say whether your tests show it):
1. A route's process renames the parent of a hidden directory (so a later job finds a planted or missing final set at the registered path). In the unit the host is read-only but for the run's directory: the rename must fail.
2. A process the route leaves behind (for example `setsid sleep 600 &`) outlives it and can write the run's output. With `KillMode=control-group` nothing of the unit must be alive after it stops.

