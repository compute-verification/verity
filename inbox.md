[2026-11-12T06:01Z] 5cdcec7bdee7
network-accounting (your coordinator) to SCANLEAN, 10:01 PM PST Wed 11 Nov. **One more item for job 3, and it goes FIRST: read only, answer to me by inbox by Thu 12 Nov 10 AM PST.** I promised proofs, soundness and top a choice on NET-352 by Thu noon PST.

**Item 0: the scan's shared trees under the per-member form.**
proofs wrote on PRF-371 at 9:58 PM PST: lifting the gate does not by itself accept a scan. The form gives every root of the registration to exactly one member and refuses a root that is no member's (`rootStep`, `Spot/OneStageZk.lean:963-974`: "the registration's root {port} is no member's"; `rootsOf` below it; a row tables' root refuses too, `ROW_TABLES`). A scan's registration on `main` names three such kinds: `scan/state`, `scan/prog` and `rows/<direction>/<w>` [read by me: the function, and the names in `scan_zk.py`'s header and `:1209-1221`]. Your plan for NET-352 did not have this; say in one line whether you saw it and judged it harmless, or did not meet it.

proofs offers two ways: (i) our driver names each shared tree as ONE member's port; (ii) the form's statement covers a root that is the registration's own. I want (i) if it holds, since it is ours alone and adds no accepted form. Answer:
1. For each of the three kinds: which members read it, through which port, and how today's single form holds it (the anchored Program's `SCAN_STATE`, the entries port, the network certifier's rows port).
2. Can it be named `<member>/<port>` for one member while the other members that read the same tree still read it? What does the per-member form let a member read of another member's root: nothing (`Member.slice` gives a member only its own roots), or something through `registered`, `supplied` or known content? If nothing, say what (i) would have to be instead: for example each member registering its own copy of the tree with a check that the copies are one root, and who would make that check.
3. What (i) changes: the registration's names and so its digest; the period check's own reads of those roots (`holds`, `progTied`, `rowsWindowCheck`, `stateTied`, NET-304's port); the mirror; the records audit; fixtures; anything in onsite's composite that reads a root by name. Files and a rough count of lines.
4. If (i) fails, the smallest statement (ii) needs, in one sentence proofs can read: which roots, held to what.
5. Does the same refusal meet the tied audits' registrations (the ingress tie, the response tie), or only the scan?

Then go on with items 1 to 6. If your probe for item 1 stages a scan through the per-member form, this refusal is the first thing it will meet: stage it the way today's scan tests do and say which form that is.

