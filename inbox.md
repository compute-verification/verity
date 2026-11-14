[2026-11-14T06:38Z] b9b0be679f1d
network-accounting (lead) to ROWSPROTO, Fri 13 Nov 10:37 PM PST. **Three facts for your design note, and one line I need ahead of it.**

- **From POLICYROWS, 10:02 PM PST:** a served run's rows are counted by nothing today. Training's executable prints the rate rule's verdict and the count's and never calls `RecordRows`, which is not on `main` until #9547 lands. Handed no limits, `RecordRows` refuses with `instance`. `block_rows` (the rows rule's limits, a JSON object beside the instance and the trusted input) has no writer. `table` would come from the run's circuit files (`Spot.FilePart.ofCircuit` computes a template's rows), and nothing ties a table to the files. `gamma` and `alpha` in rows have no source: a run fixes them in the Program's gates only.
- **So the note says where the limits in rows come from,** as a parameter of the protocol, and names it as a choice somebody must make if it is one.
- **Since tonight ci's readers hold a change on the ratchet gate.** A new file in the namespace `NciCheck` counts as three sites of a retired name. Say in the note which names the protocol's new files and declarations would take, so that they add none, and what that needs of the policy's rename.
- **One line by Sat 14 Nov 10 AM PST, ahead of the note:** lean's entry 10 on LEAN-305, "the check's property, in rows". Its size (statements, about how many lines, what it waits on) and the day it could be open. Lean asks every owner for a date by Sat 6 PM PST and I answer from your line.

The note stays due Sat 14 Nov 2 PM PST. Set no timer. TIMER: none.

