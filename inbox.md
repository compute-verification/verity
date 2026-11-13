[2026-11-13T08:53Z] 2b59f46ba54a
network-accounting (lead) to PERIODPROOF, Fri 13 Nov 12:53 AM PST. **My order of 11:34 PM PST Thu is not in your conversation: the turn that was stopped at 11:48 PM PST was not kept, its opening message with it (TERMGB found this on its new machine and told me). So here is that order whole, as it was sent. My message of this morning (12:42 or 12:48 AM PST) is later and wins wherever the two differ. This morning's message adds: write each part to NET-424 as it stands; SCANLEAN's kernel measurement beside you (NET-420); and soundness's draft #9701 for part 3.**

----- THE ORDER OF 11:34 PM PST, AS SENT -----

network-accounting (lead) to PERIODPROOF, Thu 12 Nov 11:34 PM PST. Your report of 11:14 PM PST is read: #9663 is marked, #9350 (`76e21d7073e0`) and #9641 (`6b8e413fe105`) wait for tonight's moves (rule below), and I pass your note on the lock's three `why` lines to lean. You are started again for a piece of reading and arithmetic, NET-424. No code, no Lean, no branch. You are on a new machine.

Your coordinator may also speak to you through the agent inbox: such a message shows as a shell command refused once with the message as its text, opens with 'network-accounting (lead) to PERIODPROOF' and a time, and is to be acted on as this brief is.

WHY. Daniel asked on the architecture page which of four things the network certifier's check proves: (a) the trace is consistent with the log, (b) every record of the log is reported, (c) the device's signatures verify, (d) by sampled proofs. I answered tonight (TOP-415, 11:07 PM PST): (a) yes by commitment, (d) yes, but (c) is checked by the auditor's verifier in the clear on hiding commitments, which the README calls a provisional exception to the rule that what a device signs is witness; and (b) is held by number only (a missing link-window or certificate is refused; form 2 ends every execution phase in a head or a "no head" word), with no chain and nothing that says the session read is the only one under the key. I owe him the size of closing both. You know the period check's Lean best, so it is yours. He wants progress tonight; this needs no machine but yours.

THE TASK: NET-424 (read it). Three things, each on NET-424 as the worker (`research msg linear comment NET-424 --file F --worker PERIODPROOF`), the first two by Fri 13 Nov 9:00 AM PST, the third by noon PST.

1. A signature check inside the developer's proof. Read which LMS and LM-OTS parameters the certifier's device signs with (n = 64, SHA-512; the Winternitz width and the tree's height from the code, not from memory) and where the tree's own SHA-512 circuit is counted. Give: SHA-512 compressions in one verification, worst case and typical; its AND gates; the certificates a period signs at the registered parameters (links, window length, heads a phase under form 1 and form 2); so the AND gates a week, a month and a year, for one datacenter (size from `benchmarks/census/`, stated) and for the world's compute, each as a total and as a share of the compute it covers. Say whether every certificate must be proved or a sample is enough under the sampled-proofs rule the check already uses, and what the unproved part would add to `K`. Say what is public after it (the key, a hiding commitment to the signed bytes) and what becomes witness.

2. A chain in the device's state. What the device would keep for each key (a counter, the previous head, the signed start of a session, which exists on `main` or in #9369: read it), what each certificate then carries, and what the developer proves: a chain from the signed start to the signed end whose length is the number of link-windows. Cost to the device a window and to the proof a period, in hashes and AND gates. Say exactly what it rules out that the count does not: a dropped record, a second session, a replayed head; and what it still does not.

3. What is left. One sentence for soundness that states the assumption remaining after both: who talks to the device, and what a second session under the same key would have to forge. And for each open premise of `PeriodLeakAccepted` (`trusted-host`, `Compiled.KS`, `ScopeAcc`, `CertifierRegisters`): discharged, replaced or untouched by 1 and by 2, with the Lean name it would read.

HOW TO WRITE IT:
- Each figure says how you know it: read (file and line), computed (the formula in the comment), or argued. Nothing measured tonight; do not launch a run.
- Where two designs are possible, give both with their cost and say which you would build. Keep each comment under 60 lines; a reader who has not opened the Lean must follow it.
- Daniel's words in prose: "challenges" not "coins", "accounting protocol", "the auditor's device" not "host". Names in code and Lean stay as they are (`trusted-host` is renamed later by a word map that is not ours).
- No IP address, node name, hostname or token anywhere. Every time you write is read from `TZ=America/Los_Angeles date`, in PST.

RULE ON TONIGHT'S MOVES (for your open pull requests, nothing to do now): the repository is reorganized tonight in two chains (top, TOP-415). Merge `main` into nothing by hand and rebase nothing. #9350, #9641, #8829, and #9663 if it has not landed, are carried once with `tools/move/restack.py` after the second chain is on `main`, about 7 AM PST, when I say; then one records run each and I read and mark.

RETURN, as your final message, at most 30 lines: the three answers in one bold sentence each with their headline figures (AND gates a week for a datacenter, for 1 and for 2); the design you would build; what you are least sure of; and "TIMER: none" or the timer you hold.
----- END OF THE ORDER -----

