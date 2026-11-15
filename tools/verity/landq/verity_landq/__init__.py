"""A discrete-event simulator that replays PR traffic through a landing queue (`sim`); a driver that runs a snapshot queue's
`decide`, the verified queue's (`research.vqueue`), on the simulator's events and turns its actions into decisions
(`snapshots`); the events and decisions they share (`model`), all stdlib only and without I/O; and `replay`, which reads
real traffic from git, GitHub and the store's labels, sweeps it through the simulator and sets it beside what the old path
did with the same PRs.  Import from the module (`from verity_landq import sim`)."""
