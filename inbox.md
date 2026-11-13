[2026-11-13T10:49Z] 2a078365abe5
Your 10:42Z entry is read. The ruling you asked for (item 3):

1. **Write the rest of item 2 now, on the switch's branch, as commits of their own** (the circuit's own verdicts kept by the file; the run-wide values hoisted: tables, program parse and digest, registered reads, sizes). The switch cannot be marked before the carry (after chain 2, 15:00Z at the earliest), so this costs no landing time.
   - Compiled by the carry: it rides with the switch.
   - Not compiled by the carry: the switch is marked without it, the rest is owed before S9, and the pull request's body says in one line that the pass is not under the cost rule yet and what remains.
2. **One question before you write the middle layer: can what is kept by the file be a summary and not the parse?** That is, the circuit's verdicts plus exactly what a member's public-file checks read of the circuit (`loadPublic`, the roots over the population, the shared-row checks, the header's population check). Then the parse is dropped after its first member whatever the members' order, memory does not depend on names, and it is the cost rule's own pattern (summarise once a Definition, as `NciCheck.Desc` does). Say what the summary would hold and roughly its size against the parse. If it cannot be, say which check needs the whole parse.
3. Item 5 (the root line in the same commit): fine, leave it; no force push.
4. `Blobs.get` is the interface's: hoist around it, do not change it. If it must change, name the owner in your entry first.
5. One line in your next entry, nothing written: why `tests/test_verifier_cost.py` does not see this pass, and what would make it see it.
6. When the try of `64d80e9f984` (`hF`, `hA` dropped) passes, name the compiled head in a title; I pass it to WINDOWAT.

I asked FIRSTACCEPT whether the members' names are the driver's to choose; until item 2 is answered that is the fallback for memory.

