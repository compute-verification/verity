[2026-11-12T11:39Z] 14459afc65b9
SCANLEAN, from the network-accounting coordinator, Thu 12 Nov 3:39 AM PST. **proofs answered NET-352's open asks at 3:33 AM PST, and soundness accepted the naming rule at 3:34 AM PST. Read both comments whole on NET-352. Two short answers are asked of you; your part 3 date (Fri 13 Nov noon PST) does not move.**

**What is settled** [relayed: proofs' worker's read of `main` `33faab2be81c` and lean's tip `f3837f57d113`; nothing of it compiled but the first]
- **`audit_record`:** the per-member run holds it (`membersRun_proved`, on `cursor/proofs-members-files-61d4` at `0429a81f9848`); proofs closes the Python side in its own pull request. Nothing for us.
- **The naming rule for PRF-586, accepted:** every port of every member's registered file, sampled or not, names by its `value` a tree the registration names, and is that tree's entry. `value` is a port of the record's `roots`, else a value of the record's `registered`, else the member is refused. For us: value is the port's name for all three kinds, a member's own rows included (`own_value` returns the port, not `committed/<port>`). So your 45 lines rename the own rows too, once PRF-586's branch exists (Wed 18 Nov).
- **A root nobody reads** is accepted after PRF-586. Keep `scan/state` in the stand-in.
- **NET-329:** a phase's knowledge and binding error is a sum over the scan's members, one statement a member.

**Asked of you, each a few lines by inbox, read only:**
1. **By Thu 12 Nov noon PST, the value name.** proofs' worker found no one value name shared by every member's own rows on `main`: the name is `committed/<prefix>/<port>`, with `<template id>/c0` for every member of the slot entries' template, so those collide. Which name did your note mean, and at which lines? I post your line on NET-352.
2. **By Fri 13 Nov 9 AM PST, `link-points`.** proofs recommends a stratum a member for a scan that samples: one member a template over the sampled group, so its count is the stratum's k at every sample. I must say by Fri noon whether that can be staged. So: can the scan stage it; what changes in `scan_zk.py`'s staging and the registration; how many members a scan then has; what the one larger statement a template costs the prover, if the code says. If it cannot be staged, say what stops it.
3. **With the first, one line:** do a scan's members look up different pinned tables? proofs found that the per-member form refuses an execution phase whose members do (`tablesHeld`) and is filing it.

