DENSE, Sat 14 Nov 10:45 AM PST; rings nobody. **The join's 78-layer `Joined.unwired` count is 0, read at `40afce2` only. At the join's head, the 78-layer count is not read: its run, `r20261114-163856-e8ef`, still waits for a check slot.** [each run's record read on its node]

- **78 layers, eight ranks, at `40afce2`** (`r20261114-060847-8b48`, rc 0): `unwired` 0 and 35,616 peer ports wired. `free` was not empty: the prompt had 8 ports and `BatchedGemm_v1/w` 2,496 ports. Two later commits cover those rows: `24926ba` puts the prompt under ingress, and `f253fd8` reads the weights' `/bnk` entry by a registered read.
- **4 layers at `b66a450`** (`r20261114-073847-6e0a`, rc 0) and **1 layer at `b66a450`** (`r20261114-074340-e478`, its build and join): `free` and `unwired` are both empty.
- **So both are empty at 4 layers and at 1; at 78 layers only `unwired` is shown empty, at `40afce2`.** The head's 78-layer build, `e8ef` at `9ce331c`, has waited since 8:39 AM PST. I launch no other run.
- **The join is on `architecture`,** merged as it stands, with `ranks_of` and `free_rows` beside it (`node_window_join.py`).
