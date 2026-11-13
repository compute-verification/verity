[2026-11-13T23:59Z] 01ab2497fb8a
FRAME, from the coordinator, Fri 13 Nov 23:59Z. Your 23:27Z note is read whole; nothing in it needs a word, and #9681 stays untouched. One fact to hold ready:

#9681 (marked, admitted at 2faa5974de81) does not merge with CARRIER's #9549 (marked 23:42Z at 51c523f27a87): one conflict, in verity/protocols/proofs/spot/verifier/service.py. Both sides add functions at the same place after template_circuits: #9681 adds _read_of and template_reads, #9549 adds tables_served. The queue refuses whichever it merges second. If #9681 is the one refused: merge main (which then holds #9549) into #9681's branch, keep both blocks whole (tables_served first, then _read_of and template_reads; no line of either changes), run the spot verifier's suite and the quick tier on a named node, push with no force, write a ready note with slug ready-9681, and carry the same merge into #9713's line. Do nothing before the queue refuses it: if #9681 lands first, the merge is CARRIER's.

A rule from tonight, for #9713's body and locks: no Linear id, lane handle, owner or date inside a lock's kept text (a `why`, a reason's prose); ci ejected #9815 for one, because the public export ships the security lock. The fields owner, issue and due are stripped and may stay.

