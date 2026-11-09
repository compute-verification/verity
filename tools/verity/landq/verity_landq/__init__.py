"""The landing queue as a pure reference model (`model`: `main` moves only to a candidate whose exact tree passed every gating
task) and a discrete-event simulator that replays PR traffic through it (`sim`), both stdlib only and without I/O; and
`replay`, which reads real traffic from git, GitHub and the store's labels and sweeps it through the simulator.  Import from
the module (`from verity_landq import model`)."""
