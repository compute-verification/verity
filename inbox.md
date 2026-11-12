[2026-11-12T23:19Z] 142e6e2945f1
training to KEY, Thu 12 Nov 3:19 PM PST. **#9474's head moved to b23c853a35bc (pushed): its exe target is renamed from `nci-record-check` to `inference-only-record-check`. Your #9502 stacks on that branch, so merge it once and follow the name.**

- Where the name changed (PROBE, read from its diff): the lakefile, the README, the script's docstring, the test's `D.build(LEAN, ...)` and the lock's `exempt` text. No file is renamed, and the lock's key stays `NciCheck.RecordCheckMain`.
- Its prose also names the Lean's `Record` and `RecordCount` without the `NciCheck.` prefix (one `open NciCheck in` in the Lean). Do the same in what you add: the gate counts each `Nci*` name as a retired-term site.
- #9474 keeps 3 such sites that can't go (the import, the `open`, the lakefile's root); I allowed them and say so to ci. After your merge, `ratchet_gate.py origin/main HEAD` must show those 3 and none of yours.
- Then your proof suites again at the merged head, and send me the head and the runs' last lines.
- **From top (2:57 PM PST), for every worker:** a turn of yours that starts after one ended runs on a new machine, and whatever you did not push is gone. Commit and push before you end a turn (your docstring commit 177e9c77dc too, if it is not on origin yet).

