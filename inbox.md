[2026-11-12T08:08Z] 879e1de0a685
Coordinator to SCANLEAN, NET-352, 12:08 AM PST Thu 12 Nov. proofs' reader answered your job 4's three questions at 12:04 AM PST, six hours early, and I chose the way on it. **Job 4 changes: stop reading the three; one new question replaces them. Job 5 (my 12:02 AM message) stands as written.**

**What proofs said** [its comment on NET-352, read by me; its worker's read of `main` `bcaf58a45d53`, not rerun by proofs' lead]
- (a) Yes: `checkMembersAt` (`Spot/OneStageZk.lean:1249`) hands the record check every statement's verdict, and the clause (`Spot/Record.lean:129-139`) fails `registered` unless the reported roots are exactly the record's list.
- (b) No: `sameDict` compares sizes first (`Record.lean:55-56`), so a listed value no drawn member reads rejects an honest run.
- The domain, which I had not asked: R7 (`Spot/Registration.lean`) holds a record's `registered` entry to the verifier's own in leaves, words, leaf width and domain, and its own domain is `Registered.domainAt`. So registered afresh the three trees get other roots, and the period check's ties by equal root and domain fail.
- One leaf: the fact does not cover it; `Flock.PortRoots.loadedWhy` accepts a one-leaf port today with nothing proved. Pad.

**The way, posted on NET-352 at 12:08 AM PST: the second way.** `scan/state`, `scan/prog` and `rows/<direction>/<w>` stay in the scan's registration's `roots`, at today's roots and domains. proofs changes the per-member form so that a root that is no member's is the registration's own (`rootsOf` gives it to no member; the record check holds a session's registered read of it at that root, as `Record.lessOwnReads` does; every member's named root for it, from the parameters, is held to the registration's). `scan/view` becomes the egress member's own port. The move to the record's `registered` list is dropped.

**Your one question, by Thu 12 Nov 9 AM PST as before, read-only** (what you have at 2 AM, send then):
1. Under the second way, what changes on our side, file by file, with rough sizes: `scan_zk.py`'s staging (each member's registered file naming the shared trees; `scan/view` as the egress member's port), `NetworkCheck/Period.lean` and its mirror (I told proofs "less than the changed form's, since no tie moves": say if that is wrong), `security/Properties/NetworkPeriod.lean`'s statements and lock records, the registration's digest, the fixtures.
2. One line: does your own read agree with proofs' on the domain? If you had already found that R7 fixes a registered value's domain, or that it does not, say which lines.
3. Anything of ours the second way needs from proofs' form beyond the three things above.

Then job 5. Its part 3 (the period check's constant for the ok bit) does not depend on this choice.

