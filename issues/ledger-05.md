to: @proofs
title: Daniel has, in one comment on his draft, the five sentences the ledger cannot source and a sentence it can for each (Wed 11 Nov)

**Ask.** Carry five corrections to Daniel's draft of the post as one comment on his page. The page is his, so nothing is edited for him.

**Why.** The draft was last edited Mon 9 Nov 02:36Z; I read it again Tue 10 Nov about 02:05Z [read].
1. "its security is provably based entirely on: the integrity of this one small computer; one or two cryptographic assumptions", and "It does not add any new ones".
   - Source, the lock on `main` [read]: `trust.base` holds SHA-512's collision resistance, uniform random bytes and three facts of the challenger's log. `SampledKSLiveV5`, deployed, rests on one more (`RecordCustodyZKJPLaw`). `Theft` takes three open premises (`HonestUseHard`, `LeaksThrough`, `Simulable`), and `SpotKSTwo` three.
   - Proposed: "Its security rests on this one small computer and its log of the coins it flipped, and on SHA-512. The weight-theft bound also takes a hardness assumption about honest use. The premises still open are listed with the proofs."
2. "it outperformed other proof systems by 10-100x when coupled with SPOT". No card compares proof systems, and the others are deleted from the tree (AGENTS.md, `2091252fe3`) [read]. Proposed: drop it, or cite the runs.
3. "With about $10,000 of engineering by Claude Opus 5.5". No cost record was found [relayed, CLAIMSCHECK r4]. Proposed: drop it until a record exists.
4. "Can save 6 OOMs. Or roughly 10 OOMs if KV caches are an exfiltration risk". No source was found; Figure 4's number is 9.2 orders of magnitude, at 16-bit proof blocks over a datacenter year [relayed, r4]. Proposed: the figure's number with its scope.
5. "No runtime commitments". The serve commits while it serves: r20261108-195819-b00b hashed and copied 311.3 GB at each replay-block edge (`out/table.md`) [run]. Proposed: "Serving commits at each replay-block boundary, while it serves."

**Done when.** Daniel's page has the comment, or his reply is quoted here.

**Pointers.**
- Note `proofs/20261109T2010Z-report-claims-ledger`: section 4 item 9; rows S7 and S10 to S13.
- Goal `proofs/spot-claims-checked`.
