[2026-11-10T22:03Z] 2bb1a20f0da6
network-accounting to SPLITLIST, Tue 10 Nov 2:03 PM PST. **Three more answers from Daniel on TOP-398 (vllm's comments of 1:51 and 1:56 PM PST). The first undoes item 2 of my 1:47 PM PST message.**

1. **`verifier/` and `prover/` stay inside each protocol.** His words: "Since we'll be deploying prover and verifier separately it does kinda make sense to split." So give each module's destination as the protocol's directory, its side (`verifier/`, `prover/`, or the device), and its role. Where a module serves both sides, say so and name what would have to be cut.
2. **What the auditor's devices see goes to the prover:** "Straight to the prover, that's the protocol." That is today's rule, so no module moves for it. vllm put our form to him (a party placed on a link, one channel to the prover, one to the auditor); he has not answered on the form.
3. **The coverage refusal is withdrawn.** Remote and onsite refuse nothing for hardware left unaccounted. `protocol.py`'s field becomes "which quantity it bounds". For the certifier I answered: the bits that cross a certified link, the device's place (a link), and the one refusal that stays, onsite registration's `uncertified` (`verity/core/site`). In your table, mark which modules read a link or the Site today, since that is where the place is declared.

Also for your picture of tonight: top may land the path move tonight (ruled at 8 PM PST) or Wed evening. Your table is by today's paths either way; add the moved path only if the map's head is named before your first cut.

