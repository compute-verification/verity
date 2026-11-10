[2026-11-10T22:25Z] c599bc55115c
network-accounting to LENGTHSUNIT, Tue 10 Nov 2:25 PM PST. Your two runs are read (verity-flock `r20261110-220625-4178`, the seven `r20261110-220651-5a5d`, both on vy-nebius-cpu-4). **One amendment to "no rerun": a failure of the machine is not a failure of the change.**

- infra said this morning that cpu-4 and cpu-5 were short of inodes for runs over 30 minutes. I have not checked whether that still holds.
- If a suite ends on "No space left on device", an inode error, a killed worker or a lost slot, send me its last lines and relaunch that one run on another research pool (not cpu-4, cpu-5, cpu-8 or cpu-10), with the same arguments. Do not move a run that is going.
- A test that fails on its own assertion stays as I said: its name and last lines to me, no rerun, no push.
- The layout's move goes to `main` tonight, about 9 PM PST. Do not cross or merge #8690's branch yourself. If both runs have verdicts before then I mark it at e42861fca5 at once, so send each verdict line the minute it is in.

