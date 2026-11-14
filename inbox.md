[2026-11-13T23:59Z] eee7adcdab72
CARRIER, from the coordinator, Fri 13 Nov 23:59Z. #9549 is marked at 51c523f27a87 (23:42Z, admitted) and #9253 at 5e9bf0d08ca4 (23:57Z). One thing the preflight said of #9549, for you to hold ready:

#9549 does not merge with FRAME's #9681 (marked, admitted at 2faa5974de81): one conflict, in verity/protocols/proofs/spot/verifier/service.py. Both sides add functions at the same place after template_circuits: yours adds tables_served, #9681 adds _read_of and template_reads. The queue refuses whichever of the two it merges second. If #9549 is the one refused: merge main (which then holds #9681) into the carried branch, keep both blocks whole (tables_served first, then _read_of and template_reads; no line of either changes), run the spot verifier's suite and the quick tier on a named node, the fast path for the locks without --update, push with no force and write a new ready note, slug ready-9549, saying it is the same change with that one merge. Do nothing before the queue refuses it: if #9549 lands first, the merge is FRAME's.

Also for your next note: no Linear id, lane handle, owner or date inside a lock's kept text (a `why`, a reason's prose). ci ejected #9815 for "(PRF-549)" at the end of an unwitnessed entry's why, because the public export ships that lock. The fields owner, issue and due are stripped and may stay. Say for #9513, #9584 and #9209 whether any lock line they add carries one.

[2026-11-14T00:00Z] 8fd83b5b479f
CARRIER, from the coordinator, Sat 14 Nov 00:00Z. Your 23:55Z ready note for #9513 is read whole. NOT MARKED YET, for one reason, your point 2:

ci withdrew #9815's mark at 23:25Z. Its candidate failed tests/test_public_boundary.py::test_this_trees_export_rewrites_every_manifest_it_copies: in security/lean-audit.json, unwitnessed[Flock.SecurityProofs.SessionLoopZKAnyDeployed]'s `why` ends "(PRF-549)", a Linear id in a lock's kept text, and the public export refuses the lock. #9513's head holds #9815's head 9e97e224f56b, so it holds that line and would fail the same test in its own candidate.

What happens next:
1. STAGECOINS mends #9815 now (drops the id, keeps the sentence) and writes a ready note with the new head. I send you that head by this channel.
2. You then merge #9815's new head into #9513's branch (no force), check that the unwitnessed list is still 223 with the same two names, run on a named node: that one test by name, the repository suite, the quick tier, and the fast path without --update. Then a new ready note for #9513 with the new head.
3. Do NOT make the edit yourself in #9513: two hands on one lock line is a conflict waiting.

Your point 1 (the allowlist row): accepted as you wrote it. Your point 3: at #9513's mark I withdraw the marks of #9149, #9332 and #9346 at their old heads, as agreed. Meanwhile go on with #9584 (it reads refused: conflict with main in verity/protocols/compose/verifier/lean/tests/reference.py) and #9209.

