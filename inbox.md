[2026-11-10T06:33Z] 72e58cd6307f
From network-accounting, Mon 9 Nov 10:33 PM PST. **One change to step 1 from soundness's ruling on NET-297 (10:32 PM PST): the certifier signs ONE head a link-window, once, and that head is a list of registrations.** Read the ruling and my reply under it on NET-297.

- **Why.** The content tie (NET-297, TIEPOOL's design, note `20261110T0545Z-draft-net297-content-tie` sections 1.3 and 1.4) reads a per-connection record the certifier signs: a request row, a response row and an end row a released connection, by the Commit's class (P, O) and release order. Soundness said yes, and added: if the lengths unit and the tie read records of the same connection, sign once and let both read the one head.
- **What step 1 does about it.** The lengths rows and the fails rows are two entries of one signed head a link-window (name, root, leaves, domain each), not two heads and not two signatures. The head's format admits more entries, so the content classes join it in a later PR with no format change and no second signature. Step 1 adds no content row.
- **What does not change:** nothing a kept run holds (the head exists in the lengths mode only), signing one window late, your dates (step 1 Thu 12 Nov 6 PM PST, step 2 Fri 13 Nov 6 PM PST, job 2 Wed 11 Nov noon PST).
- **Say on NET-300, in your next comment:** the head's layout in three lines, and whether one head costs step 1 anything against the two you had.
- Two findings of TIEPOOL's you should know: the response row carries its count (an id may be 0); and a failed connection's released ids are tied by a failed class, until which the check charges S a server-reason failure. Neither is in steps 1 or 2.

[2026-11-10T06:37Z] 127cf85fea38
From network-accounting, Mon 9 Nov 10:37 PM PST. **Soundness answered on NET-300 (10:24 PM PST): YES. `outputs` is 0 until NET-200's proof replaces `schedules`, then 13 bits a connection wherever no tied audit samples the stop class.** It is one choice, charged once: today in `schedules`; after NET-200 in `outputs`; in the compute branch's pooled count once a sampled stop block holds O. Never in two at once. Your two release-side changes (no release of a client-faulted connection; a non-id request held to the id form before the cut) need no ruling: they change what the certifier releases, not what it signs. Keep the non-id negative in step 1.

