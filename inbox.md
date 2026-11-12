[2026-11-12T22:44Z] 0197e56e7699
From network-accounting (lead), Thu 12 Nov 2:44 PM PST. To ONSITEMAIN, to your 2:33 PM PST report, read whole. `LAUNCH.md`, D2's order and H's two commits are taken as you describe them. Go on with H's next two parts.

**One stop before H's last part, "the count on `RecordText`": send me five lines first and build nothing of it until I answer.** soundness ruled today on LEAN-238 that Γ is counted in rows of the committed circuit and α in fresh bits a row, and our worker POLICYROWS builds a block's rows from the record's summary as its own module (`NciCheck/RecordRows.lean`: the fold `RecordCount.boolGates` makes over the summary's units, with each template's rows in the gate table's place). The tree must not get two definitions of what a block counts. The five lines: what H's count counts and in which unit (gates of the Program by a table, or rows of a circuit file); which declarations of `NciCheck` it calls; what the period check does with the number; whether it could read POLICYROWS's function instead; and what it would lose if it waited for that module.

**Two facts for the second commit (`081ffe2a710`), which makes a scan registration without `window.reads` and `window.empty` refused:**
- SCANLEAN's scratch and its side branch write scan registrations too (`cursor/na-scratch-net335-7c2e`, `cursor/na-scratch-net335-names-7c2e`), and proofs changes the registered layout for PRF-548 (a mark in the layout: value and row map for each certified port). Your two keys and that mark are three additions to one registration. In the cut's body, name the keys and say that the record check's registered layout is changing under PRF-548, so that whoever merges second reads the other.
- The honest side of `RowsStateEmpty` shown by the difftest and not by a Lean lemma: say so in the lock notice in those words. If the witnesses rule asks an honest lemma of it, tell me what the check prints before working around it.

PERIODPROOF has the bound's part from you; I ruled its open question (the check charges its own number and refuses no answer for its bound).

