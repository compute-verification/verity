"""The landing queue as a pure reference model (`model`: `main` moves only to a candidate whose exact tree passed every gating
task) and a discrete-event simulator that replays PR traffic through it (`sim`).  Both are stdlib only and do no I/O; import
from the module (`from verity_landq import model`)."""
