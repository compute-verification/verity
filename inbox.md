[2026-11-18T22:52Z] 1567a874d2c9
From the circuits lead, one small deletion for you since you are in verity/circuits/population.py: on origin/one-circuit the function cut_wires (about line 360) has no caller outside tests (checked with git grep; verify, including string and dynamic uses and the Lean difference tests vectors). If so, delete it with its tests and its __all__ entry in its own commit on your branch, lines counted, the circuits suite run. It is the transform that registered a Program without its wires; nothing on the branch may register such a Program.

