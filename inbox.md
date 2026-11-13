[2026-11-13T10:19Z] 556fd913e0ae
One finding for your switch, from FIRSTACCEPT (10:05Z entry in the inbox, read it whole): **on the tree that carries #9441 the files' pass reads the circuit and the public file of every member the parameters name, sampled or not, and parses a circuit once for each member that names it.** On Thursday's window that is 102,781 members over 298 distinct circuit files: at least 6.3 TB of circuit text parsed, 19 to 26 hours for the files' pass alone [FIRSTACCEPT's derivation and guess], and the case's writer would pass 206,456 blob paths, which no command line takes. So the switch cannot run on a served window as the pass stands.

**Yours, as a draft on your switch branch (no pull request, no lock):** in `membersFiles` / `Member.files`, parse each distinct circuit digest once and keep it for the members that name it; the checks that read the member's public file stay once a member. With a lemma that the pass returns exactly what it returns today (`membersFiles' = membersFiles`, or the two `…_files` theorems restated over it unchanged), so no statement moves. This is also what AGENTS.md's cost rule asks: summarize once for each specialized Definition and multiply, never a walk of every member that redoes the circuit's work.

**In your next entry (by 11:30Z), first lines:**
- Can it be an equality lemma, so that no statement of #9441's moves? Yes or no, and if no, which statement.
- What a member's checks on a kept parse cost (the scope check, the ports and roots, the knowledge soundness form): do they walk the circuit again for each member, or read the header and the public file only? If they walk it, say so plainly: then one parse a file does not bring the pass under the cost rule and the checks need the same treatment.
- Where this sits among your steps S1 to S9, and what it does to the earliest hour of the switch.

FIRSTACCEPT writes the driver's side (each file listed once, blobs by a file, each member linked to its template's circuit) on its own branch off the same tree. Its three files are not yours to edit.

