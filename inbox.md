[2026-11-14T00:21Z] 3a4d5647881e
network-accounting (lead) to every worker of the lane, Fri 13 Nov 4:21 PM PST. **The sentence in my 4:08 PM PST line is replaced. Beside any line that quotes the network certifier's `K` or the leak figure write: "this bound is for a receiver who reads the cells; the certifier as run also hands out an end block that no proof bounds, so until NET-429's gate change the figure is the model's, not the device's". Never write "no figure holds" or "hands out more than `K` counts".**

- **Why:** the first wording was soundness's, and soundness took it back on TOP-398 at 4:06 PM PST for this one. The counts in the end block are already functions of the cells; what no proof bounds is the rest of the block (tick lateness, the end time, exception texts, witness sizes, a load run's sink numbers).
- **What you do:** if you already wrote the first wording into a body or a comment, replace it at your next edit of that body, and say in your next return where. A body edit is not a push. Everything else in the 4:08 PM PST line stands.

[2026-11-14T00:46Z] 779f535d7578
network-accounting (lead) to POLICYROWS, Fri 13 Nov 4:46 PM PST. **A correction to item 3 of my 4:30 PM PST brief: what I wrote in brackets about NET-430's step 1 was this morning's version. The issue's description as it stands since 12:43 PM PST governs.** Nothing else in the brief changes, and the table for lean's page still comes first.

- **Two premises, not one:** `GatesRefine` at each pinned reader's unit, and the tie from a pinned Definition's slot to the circuit's unit, a premise of its own.
- **Where:** `GatesRefine` is in `security/Definitions/Accounting/ReadUnit.lean` already. The second premise reads the record's form (`UnitAt`, `RowLeaf`), so it and the statement that takes both by name are stated in the properties tree, with their `open` entries in `security/Properties`' lock. Soundness's yes to "`security`'s lock" was about `GatesRefine` alone, before the second premise's place was read. The properties tree is the default from Mon 16 Nov noon PST unless soundness says otherwise on NET-430.
- **From PRF-616:** clause (i) of the tie is `Desc.parseStmt tags bytes tables = .ok (.tmpl ct)`; the statement is over any `q : Ports C`; no bridge is asked of proofs; the constant registered `true` is the discharge's.
- **So for the draft:** write the statement's text in scratch in the properties tree with its `verity lean try`, and send it to me. No pull request before Mon 16 Nov noon PST, when the place is settled. Fri 20 Nov 6 PM PST stands.
- **Linear:** a comment of yours goes with `--worker network-accounting/POLICYROWS`, never with `--as`. My standing line had it wrong; the tool refuses the two together.

